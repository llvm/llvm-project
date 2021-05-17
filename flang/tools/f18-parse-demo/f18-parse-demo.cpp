//===-- tools/f18/f18-parse-demo.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// F18 parsing demonstration.
//   f18-parse-demo [ -E | -fdump-parse-tree | -funparse-only ]
//     foo.{f,F,f77,F77,f90,F90,&c.}
//
// By default, runs the supplied source files through the F18 preprocessing and
// parsing phases, reconstitutes a Fortran program from the parse tree, and
// passes that Fortran program to a Fortran compiler identified by the $F18_FC
// environment variable (defaulting to gfortran).  The Fortran preprocessor is
// always run, whatever the case of the source file extension.  Unrecognized
// options are passed through to the underlying Fortran compiler.
//
// This program is actually a stripped-down variant of f18.cpp, a temporary
// scaffolding compiler driver that can test some semantic passes of the
// F18 compiler under development.

#include "flang/Common/Fortran-features.h"
#include "flang/Common/default-kinds.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/unparse.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <list>
#include <memory>
#include <optional>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>

static std::list<std::string> argList(int argc, char *const argv[]) {
  std::list<std::string> result;
  for (int j = 0; j < argc; ++j) {
    result.emplace_back(argv[j]);
  }
  return result;
}

std::vector<std::string> filesToDelete;

void CleanUpAtExit() {
  for (const auto &path : filesToDelete) {
    if (!path.empty()) {
      llvm::sys::fs::remove(path);
    }
  }
}

#if _POSIX_C_SOURCE >= 199309L && _POSIX_TIMERS > 0 && _POSIX_CPUTIME && \
    defined CLOCK_PROCESS_CPUTIME_ID
static constexpr bool canTime{true};
double CPUseconds() {
  struct timespec tspec;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tspec);
  return tspec.tv_nsec * 1.0e-9 + tspec.tv_sec;
}
#else
static constexpr bool canTime{false};
double CPUseconds() { return 0; }
#endif

struct DriverOptions {
  DriverOptions() {}
  bool verbose{false}; // -v
  bool compileOnly{false}; // -c
  std::string outputPath; // -o path
  std::vector<std::string> searchDirectories{"."s}; // -I dir
  bool forcedForm{false}; // -Mfixed or -Mfree appeared
  bool warnOnNonstandardUsage{false}; // -Mstandard
  bool warningsAreErrors{false}; // -Werror
  Fortran::parser::Encoding encoding{Fortran::parser::Encoding::LATIN_1};
  bool syntaxOnly{false};
  bool dumpProvenance{false};
  bool dumpCookedChars{false};
  bool dumpUnparse{false};
  bool dumpParseTree{false};
  bool timeParse{false};
  std::vector<std::string> fcArgs;
  const char *prefix{nullptr};
};

void Exec(std::vector<llvm::StringRef> &argv, bool verbose = false) {
  if (verbose) {
    for (size_t j{0}; j < argv.size(); ++j) {
      llvm::errs() << (j > 0 ? " " : "") << argv[j];
    }
    llvm::errs() << '\n';
  }
  std::string ErrMsg;
  llvm::ErrorOr<std::string> Program = llvm::sys::findProgramByName(argv[0]);
  if (!Program)
    ErrMsg = Program.getError().message();
  if (!Program ||
      llvm::sys::ExecuteAndWait(
          Program.get(), argv, llvm::None, {}, 0, 0, &ErrMsg)) {
    llvm::errs() << "execvp(" << argv[0] << ") failed: " << ErrMsg << '\n';
    exit(EXIT_FAILURE);
  }
}

void RunOtherCompiler(DriverOptions &driver, char *source, char *relo) {
  std::vector<llvm::StringRef> argv;
  for (size_t j{0}; j < driver.fcArgs.size(); ++j) {
    argv.push_back(driver.fcArgs[j]);
  }
  char dashC[3] = "-c", dashO[3] = "-o";
  argv.push_back(dashC);
  argv.push_back(dashO);
  argv.push_back(relo);
  argv.push_back(source);
  Exec(argv, driver.verbose);
}

std::string RelocatableName(const DriverOptions &driver, std::string path) {
  if (driver.compileOnly && !driver.outputPath.empty()) {
    return driver.outputPath;
  }
  std::string base{path};
  auto slash{base.rfind("/")};
  if (slash != std::string::npos) {
    base = base.substr(slash + 1);
  }
  std::string relo{base};
  auto dot{base.rfind(".")};
  if (dot != std::string::npos) {
    relo = base.substr(0, dot);
  }
  relo += ".o";
  return relo;
}

int exitStatus{EXIT_SUCCESS};

std::string CompileFortran(
    std::string path, Fortran::parser::Options options, DriverOptions &driver) {
  if (!driver.forcedForm) {
    auto dot{path.rfind(".")};
    if (dot != std::string::npos) {
      std::string suffix{path.substr(dot + 1)};
      options.isFixedForm = suffix == "f" || suffix == "F" || suffix == "ff";
    }
  }
  options.searchDirectories = driver.searchDirectories;
  Fortran::parser::AllSources allSources;
  Fortran::parser::AllCookedSources allCookedSources{allSources};
  Fortran::parser::Parsing parsing{allCookedSources};

  auto start{CPUseconds()};
  parsing.Prescan(path, options);
  if (!parsing.messages().empty() &&
      (driver.warningsAreErrors || parsing.messages().AnyFatalError())) {
    llvm::errs() << driver.prefix << "could not scan " << path << '\n';
    parsing.messages().Emit(llvm::errs(), parsing.allCooked());
    exitStatus = EXIT_FAILURE;
    return {};
  }
  if (driver.dumpProvenance) {
    parsing.DumpProvenance(llvm::outs());
    return {};
  }
  if (driver.dumpCookedChars) {
    parsing.DumpCookedChars(llvm::outs());
    return {};
  }
  parsing.Parse(llvm::outs());
  auto stop{CPUseconds()};
  if (driver.timeParse) {
    if (canTime) {
      llvm::outs() << "parse time for " << path << ": " << (stop - start)
                   << " CPU seconds\n";
    } else {
      llvm::outs() << "no timing information due to lack of clock_gettime()\n";
    }
  }

  parsing.ClearLog();
  parsing.messages().Emit(llvm::errs(), parsing.allCooked());
  if (!parsing.consumedWholeFile()) {
    parsing.EmitMessage(llvm::errs(), parsing.finalRestingPlace(),
        "parser FAIL (final position)");
    exitStatus = EXIT_FAILURE;
    return {};
  }
  if ((!parsing.messages().empty() &&
          (driver.warningsAreErrors || parsing.messages().AnyFatalError())) ||
      !parsing.parseTree()) {
    llvm::errs() << driver.prefix << "could not parse " << path << '\n';
    exitStatus = EXIT_FAILURE;
    return {};
  }
  auto &parseTree{*parsing.parseTree()};
  if (driver.dumpParseTree) {
    Fortran::parser::DumpTree(llvm::outs(), parseTree);
    return {};
  }
  if (driver.dumpUnparse) {
    Unparse(llvm::outs(), parseTree, driver.encoding, true /*capitalize*/,
        options.features.IsEnabled(
            Fortran::common::LanguageFeature::BackslashEscapes));
    return {};
  }
  if (driver.syntaxOnly) {
    return {};
  }

  std::string relo{RelocatableName(driver, path)};

  llvm::SmallString<32> tmpSourcePath;
  {
    int fd;
    std::error_code EC =
        llvm::sys::fs::createUniqueFile("f18-%%%%.f90", fd, tmpSourcePath);
    if (EC) {
      llvm::errs() << EC.message() << "\n";
      std::exit(EXIT_FAILURE);
    }
    llvm::raw_fd_ostream tmpSource(fd, /*shouldClose*/ true);
    Unparse(tmpSource, parseTree, driver.encoding, true /*capitalize*/,
        options.features.IsEnabled(
            Fortran::common::LanguageFeature::BackslashEscapes));
  }

  RunOtherCompiler(driver, tmpSourcePath.data(), relo.data());
  filesToDelete.emplace_back(tmpSourcePath);
  if (!driver.compileOnly && driver.outputPath.empty()) {
    filesToDelete.push_back(relo);
  }
  return relo;
}

std::string CompileOtherLanguage(std::string path, DriverOptions &driver) {
  std::string relo{RelocatableName(driver, path)};
  RunOtherCompiler(driver, path.data(), relo.data());
  if (!driver.compileOnly && driver.outputPath.empty()) {
    filesToDelete.push_back(relo);
  }
  return relo;
}

void Link(std::vector<std::string> &relocatables, DriverOptions &driver) {
  std::vector<llvm::StringRef> argv;
  for (size_t j{0}; j < driver.fcArgs.size(); ++j) {
    argv.push_back(driver.fcArgs[j].data());
  }
  for (auto &relo : relocatables) {
    argv.push_back(relo.data());
  }
  if (!driver.outputPath.empty()) {
    char dashO[3] = "-o";
    argv.push_back(dashO);
    argv.push_back(driver.outputPath.data());
  }
  Exec(argv, driver.verbose);
}

int main(int argc, char *const argv[]) {

  atexit(CleanUpAtExit);

  DriverOptions driver;
  const char *fc{getenv("F18_FC")};
  driver.fcArgs.push_back(fc ? fc : "gfortran");

  std::list<std::string> args{argList(argc, argv)};
  std::string prefix{args.front()};
  args.pop_front();
  prefix += ": ";
  driver.prefix = prefix.data();

  Fortran::parser::Options options;
  options.predefinitions.emplace_back("__F18", "1");
  options.predefinitions.emplace_back("__F18_MAJOR__", "1");
  options.predefinitions.emplace_back("__F18_MINOR__", "1");
  options.predefinitions.emplace_back("__F18_PATCHLEVEL__", "1");

  options.features.Enable(
      Fortran::common::LanguageFeature::BackslashEscapes, true);

  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;

  std::vector<std::string> fortranSources, otherSources, relocatables;
  bool anyFiles{false};

  while (!args.empty()) {
    std::string arg{std::move(args.front())};
    args.pop_front();
    if (arg.empty()) {
    } else if (arg.at(0) != '-') {
      anyFiles = true;
      auto dot{arg.rfind(".")};
      if (dot == std::string::npos) {
        driver.fcArgs.push_back(arg);
      } else {
        std::string suffix{arg.substr(dot + 1)};
        if (suffix == "f" || suffix == "F" || suffix == "ff" ||
            suffix == "f90" || suffix == "F90" || suffix == "ff90" ||
            suffix == "f95" || suffix == "F95" || suffix == "ff95" ||
            suffix == "cuf" || suffix == "CUF" || suffix == "f18" ||
            suffix == "F18" || suffix == "ff18") {
          fortranSources.push_back(arg);
        } else if (suffix == "o" || suffix == "a") {
          relocatables.push_back(arg);
        } else {
          otherSources.push_back(arg);
        }
      }
    } else if (arg == "-") {
      fortranSources.push_back("-");
    } else if (arg == "--") {
      while (!args.empty()) {
        fortranSources.emplace_back(std::move(args.front()));
        args.pop_front();
      }
      break;
    } else if (arg == "-Mfixed") {
      driver.forcedForm = true;
      options.isFixedForm = true;
    } else if (arg == "-Mfree") {
      driver.forcedForm = true;
      options.isFixedForm = false;
    } else if (arg == "-Mextend") {
      options.fixedFormColumns = 132;
    } else if (arg == "-Mbackslash") {
      options.features.Enable(
          Fortran::common::LanguageFeature::BackslashEscapes, false);
    } else if (arg == "-Mnobackslash") {
      options.features.Enable(
          Fortran::common::LanguageFeature::BackslashEscapes);
    } else if (arg == "-Mstandard") {
      driver.warnOnNonstandardUsage = true;
    } else if (arg == "-fopenmp") {
      options.features.Enable(Fortran::common::LanguageFeature::OpenMP);
      options.predefinitions.emplace_back("_OPENMP", "201511");
    } else if (arg == "-Werror") {
      driver.warningsAreErrors = true;
    } else if (arg == "-ed") {
      options.features.Enable(Fortran::common::LanguageFeature::OldDebugLines);
    } else if (arg == "-E" || arg == "-fpreprocess-only") {
      driver.dumpCookedChars = true;
    } else if (arg == "-fbackslash") {
      options.features.Enable(
          Fortran::common::LanguageFeature::BackslashEscapes);
    } else if (arg == "-fno-backslash") {
      options.features.Enable(
          Fortran::common::LanguageFeature::BackslashEscapes, false);
    } else if (arg == "-fdump-provenance") {
      driver.dumpProvenance = true;
    } else if (arg == "-fdump-parse-tree") {
      driver.dumpParseTree = true;
    } else if (arg == "-funparse") {
      driver.dumpUnparse = true;
    } else if (arg == "-ftime-parse") {
      driver.timeParse = true;
    } else if (arg == "-fparse-only" || arg == "-fsyntax-only") {
      driver.syntaxOnly = true;
    } else if (arg == "-c") {
      driver.compileOnly = true;
    } else if (arg == "-o") {
      driver.outputPath = args.front();
      args.pop_front();
    } else if (arg.substr(0, 2) == "-D") {
      auto eq{arg.find('=')};
      if (eq == std::string::npos) {
        options.predefinitions.emplace_back(arg.substr(2), "1");
      } else {
        options.predefinitions.emplace_back(
            arg.substr(2, eq - 2), arg.substr(eq + 1));
      }
    } else if (arg.substr(0, 2) == "-U") {
      options.predefinitions.emplace_back(
          arg.substr(2), std::optional<std::string>{});
    } else if (arg == "-r8" || arg == "-fdefault-real-8") {
      defaultKinds.set_defaultRealKind(8);
    } else if (arg == "-i8" || arg == "-fdefault-integer-8") {
      defaultKinds.set_defaultIntegerKind(8);
    } else if (arg == "-help" || arg == "--help" || arg == "-?") {
      llvm::errs()
          << "f18-parse-demo options:\n"
          << "  -Mfixed | -Mfree     force the source form\n"
          << "  -Mextend             132-column fixed form\n"
          << "  -f[no-]backslash     enable[disable] \\escapes in literals\n"
          << "  -M[no]backslash      disable[enable] \\escapes in literals\n"
          << "  -Mstandard           enable conformance warnings\n"
          << "  -r8 | -fdefault-real-8 | -i8 | -fdefault-integer-8  "
             "change default kinds of intrinsic types\n"
          << "  -Werror              treat warnings as errors\n"
          << "  -ed                  enable fixed form D lines\n"
          << "  -E                   prescan & preprocess only\n"
          << "  -ftime-parse         measure parsing time\n"
          << "  -fsyntax-only        parse only, no output except messages\n"
          << "  -funparse            parse & reformat only, no code "
             "generation\n"
          << "  -fdump-provenance    dump the provenance table (no code)\n"
          << "  -fdump-parse-tree    dump the parse tree (no code)\n"
          << "  -v -c -o -I -D -U    have their usual meanings\n"
          << "  -help                print this again\n"
          << "Other options are passed through to the $F18_FC compiler.\n";
      return exitStatus;
    } else if (arg == "-V") {
      llvm::errs() << "\nf18-parse-demo\n";
      return exitStatus;
    } else {
      driver.fcArgs.push_back(arg);
      if (arg == "-v") {
        driver.verbose = true;
      } else if (arg == "-I") {
        driver.fcArgs.push_back(args.front());
        driver.searchDirectories.push_back(args.front());
        args.pop_front();
      } else if (arg.substr(0, 2) == "-I") {
        driver.searchDirectories.push_back(arg.substr(2));
      }
    }
  }

  if (driver.warnOnNonstandardUsage) {
    options.features.WarnOnAllNonstandard();
  }
  if (!options.features.IsEnabled(
          Fortran::common::LanguageFeature::BackslashEscapes)) {
    driver.fcArgs.push_back("-fno-backslash"); // PGI "-Mbackslash"
  }

  if (!anyFiles) {
    driver.dumpUnparse = true;
    CompileFortran("-", options, driver);
    return exitStatus;
  }
  for (const auto &path : fortranSources) {
    std::string relo{CompileFortran(path, options, driver)};
    if (!driver.compileOnly && !relo.empty()) {
      relocatables.push_back(relo);
    }
  }
  for (const auto &path : otherSources) {
    std::string relo{CompileOtherLanguage(path, driver)};
    if (!driver.compileOnly && !relo.empty()) {
      relocatables.push_back(relo);
    }
  }
  if (!relocatables.empty()) {
    Link(relocatables, driver);
  }
  return exitStatus;
}
