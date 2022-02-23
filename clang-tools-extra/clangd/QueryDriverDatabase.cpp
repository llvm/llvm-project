//===--- QueryDriverDatabase.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some compiler drivers have implicit search mechanism for system headers.
// This compilation database implementation tries to extract that information by
// executing the driver in verbose mode. gcc-compatible drivers print something
// like:
// ....
// ....
// #include <...> search starts here:
//  /usr/lib/gcc/x86_64-linux-gnu/7/include
//  /usr/local/include
//  /usr/lib/gcc/x86_64-linux-gnu/7/include-fixed
//  /usr/include/x86_64-linux-gnu
//  /usr/include
// End of search list.
// ....
// ....
// This component parses that output and adds each path to command line args
// provided by Base, after prepending them with -isystem. Therefore current
// implementation would not work with a driver that is not gcc-compatible.
//
// First argument of the command line received from underlying compilation
// database is used as compiler driver path. Due to this arbitrary binary
// execution, this mechanism is not used by default and only executes binaries
// in the paths that are explicitly included by the user.

#include "GlobalCompilationDatabase.h"
#include "support/Logger.h"
#include "support/Path.h"
#include "support/Trace.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Driver/Types.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ScopedPrinter.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace clang {
namespace clangd {
namespace {

struct DriverInfo {
  std::vector<std::string> SystemIncludes;
  std::string Target;
};

bool isValidTarget(llvm::StringRef Triple) {
  std::shared_ptr<TargetOptions> TargetOpts(new TargetOptions);
  TargetOpts->Triple = Triple.str();
  DiagnosticsEngine Diags(new DiagnosticIDs, new DiagnosticOptions,
                          new IgnoringDiagConsumer);
  IntrusiveRefCntPtr<TargetInfo> Target =
      TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  return bool(Target);
}

llvm::Optional<DriverInfo> parseDriverOutput(llvm::StringRef Output) {
  DriverInfo Info;
  const char SIS[] = "#include <...> search starts here:";
  const char SIE[] = "End of search list.";
  const char TS[] = "Target: ";
  llvm::SmallVector<llvm::StringRef> Lines;
  Output.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  enum {
    Initial,            // Initial state: searching for target or includes list.
    IncludesExtracting, // Includes extracting.
    Done                // Includes and target extraction done.
  } State = Initial;
  bool SeenIncludes = false;
  bool SeenTarget = false;
  for (auto *It = Lines.begin(); State != Done && It != Lines.end(); ++It) {
    auto Line = *It;
    switch (State) {
    case Initial:
      if (!SeenIncludes && Line.trim() == SIS) {
        SeenIncludes = true;
        State = IncludesExtracting;
      } else if (!SeenTarget && Line.trim().startswith(TS)) {
        SeenTarget = true;
        llvm::StringRef TargetLine = Line.trim();
        TargetLine.consume_front(TS);
        // Only detect targets that clang understands
        if (!isValidTarget(TargetLine)) {
          elog("System include extraction: invalid target \"{0}\", ignoring",
               TargetLine);
        } else {
          Info.Target = TargetLine.str();
          vlog("System include extraction: target extracted: \"{0}\"",
               TargetLine);
        }
      }
      break;
    case IncludesExtracting:
      if (Line.trim() == SIE) {
        State = SeenTarget ? Done : Initial;
      } else {
        Info.SystemIncludes.push_back(Line.trim().str());
        vlog("System include extraction: adding {0}", Line);
      }
      break;
    default:
      llvm_unreachable("Impossible state of the driver output parser");
      break;
    }
  }
  if (!SeenIncludes) {
    elog("System include extraction: start marker not found: {0}", Output);
    return llvm::None;
  }
  if (State == IncludesExtracting) {
    elog("System include extraction: end marker missing: {0}", Output);
    return llvm::None;
  }
  return std::move(Info);
}

llvm::Optional<DriverInfo>
extractSystemIncludesAndTarget(llvm::SmallString<128> Driver,
                               llvm::StringRef Lang,
                               llvm::ArrayRef<std::string> CommandLine,
                               const llvm::Regex &QueryDriverRegex) {
  trace::Span Tracer("Extract system includes and target");

  if (!llvm::sys::path::is_absolute(Driver)) {
    assert(llvm::none_of(
        Driver, [](char C) { return llvm::sys::path::is_separator(C); }));
    auto DriverProgram = llvm::sys::findProgramByName(Driver);
    if (DriverProgram) {
      vlog("System include extraction: driver {0} expanded to {1}", Driver,
           *DriverProgram);
      Driver = *DriverProgram;
    } else {
      elog("System include extraction: driver {0} not found in PATH", Driver);
      return llvm::None;
    }
  }

  SPAN_ATTACH(Tracer, "driver", Driver);
  SPAN_ATTACH(Tracer, "lang", Lang);

  if (!QueryDriverRegex.match(Driver)) {
    vlog("System include extraction: not allowed driver {0}", Driver);
    return llvm::None;
  }

  llvm::SmallString<128> StdErrPath;
  if (auto EC = llvm::sys::fs::createTemporaryFile("system-includes", "clangd",
                                                   StdErrPath)) {
    elog("System include extraction: failed to create temporary file with "
         "error {0}",
         EC.message());
    return llvm::None;
  }
  auto CleanUp = llvm::make_scope_exit(
      [&StdErrPath]() { llvm::sys::fs::remove(StdErrPath); });

  llvm::Optional<llvm::StringRef> Redirects[] = {{""}, {""}, StdErrPath.str()};

  llvm::SmallVector<llvm::StringRef> Args = {Driver, "-E", "-x",
                                             Lang,   "-",  "-v"};

  // These flags will be preserved
  const llvm::StringRef FlagsToPreserve[] = {
      "-nostdinc", "--no-standard-includes", "-nostdinc++", "-nobuiltininc"};
  // Preserves these flags and their values, either as separate args or with an
  // equalsbetween them
  const llvm::StringRef ArgsToPreserve[] = {"--sysroot", "-isysroot"};

  for (size_t I = 0, E = CommandLine.size(); I < E; ++I) {
    llvm::StringRef Arg = CommandLine[I];
    if (llvm::any_of(FlagsToPreserve,
                     [&Arg](llvm::StringRef S) { return S == Arg; })) {
      Args.push_back(Arg);
    } else {
      const auto *Found =
          llvm::find_if(ArgsToPreserve, [&Arg](llvm::StringRef S) {
            return Arg.startswith(S);
          });
      if (Found == std::end(ArgsToPreserve))
        continue;
      Arg = Arg.drop_front(Found->size());
      if (Arg.empty() && I + 1 < E) {
        Args.push_back(CommandLine[I]);
        Args.push_back(CommandLine[++I]);
      } else if (Arg.startswith("=")) {
        Args.push_back(CommandLine[I]);
      }
    }
  }

  std::string ErrMsg;
  if (int RC = llvm::sys::ExecuteAndWait(Driver, Args, /*Env=*/llvm::None,
                                         Redirects, /*SecondsToWait=*/0,
                                         /*MemoryLimit=*/0, &ErrMsg)) {
    elog("System include extraction: driver execution failed with return code: "
         "{0} - '{1}'. Args: [{2}]",
         llvm::to_string(RC), ErrMsg, printArgv(Args));
    return llvm::None;
  }

  auto BufOrError = llvm::MemoryBuffer::getFile(StdErrPath);
  if (!BufOrError) {
    elog("System include extraction: failed to read {0} with error {1}",
         StdErrPath, BufOrError.getError().message());
    return llvm::None;
  }

  llvm::Optional<DriverInfo> Info =
      parseDriverOutput(BufOrError->get()->getBuffer());
  if (!Info)
    return llvm::None;
  log("System includes extractor: successfully executed {0}\n\tgot includes: "
      "\"{1}\"\n\tgot target: \"{2}\"",
      Driver, llvm::join(Info->SystemIncludes, ", "), Info->Target);
  return Info;
}

tooling::CompileCommand &
addSystemIncludes(tooling::CompileCommand &Cmd,
                  llvm::ArrayRef<std::string> SystemIncludes) {
  for (llvm::StringRef Include : SystemIncludes) {
    // FIXME(kadircet): This doesn't work when we have "--driver-mode=cl"
    Cmd.CommandLine.push_back("-isystem");
    Cmd.CommandLine.push_back(Include.str());
  }
  return Cmd;
}

tooling::CompileCommand &setTarget(tooling::CompileCommand &Cmd,
                                   const std::string &Target) {
  if (!Target.empty()) {
    // We do not want to override existing target with extracted one.
    for (llvm::StringRef Arg : Cmd.CommandLine) {
      if (Arg == "-target" || Arg.startswith("--target="))
        return Cmd;
    }
    Cmd.CommandLine.push_back("--target=" + Target);
  }
  return Cmd;
}

/// Converts a glob containing only ** or * into a regex.
std::string convertGlobToRegex(llvm::StringRef Glob) {
  std::string RegText;
  llvm::raw_string_ostream RegStream(RegText);
  RegStream << '^';
  for (size_t I = 0, E = Glob.size(); I < E; ++I) {
    if (Glob[I] == '*') {
      if (I + 1 < E && Glob[I + 1] == '*') {
        // Double star, accept any sequence.
        RegStream << ".*";
        // Also skip the second star.
        ++I;
      } else {
        // Single star, accept any sequence without a slash.
        RegStream << "[^/]*";
      }
    } else if (llvm::sys::path::is_separator(Glob[I]) &&
               llvm::sys::path::is_separator('/') &&
               llvm::sys::path::is_separator('\\')) {
      RegStream << R"([/\\])"; // Accept either slash on windows.
    } else {
      RegStream << llvm::Regex::escape(Glob.substr(I, 1));
    }
  }
  RegStream << '$';
  RegStream.flush();
  return RegText;
}

/// Converts a glob containing only ** or * into a regex.
llvm::Regex convertGlobsToRegex(llvm::ArrayRef<std::string> Globs) {
  assert(!Globs.empty() && "Globs cannot be empty!");
  std::vector<std::string> RegTexts;
  RegTexts.reserve(Globs.size());
  for (llvm::StringRef Glob : Globs)
    RegTexts.push_back(convertGlobToRegex(Glob));

  // Tempting to pass IgnoreCase, but we don't know the FS sensitivity.
  llvm::Regex Reg(llvm::join(RegTexts, "|"));
  assert(Reg.isValid(RegTexts.front()) &&
         "Created an invalid regex from globs");
  return Reg;
}

/// Extracts system includes from a trusted driver by parsing the output of
/// include search path and appends them to the commands coming from underlying
/// compilation database.
class QueryDriverDatabase : public DelegatingCDB {
public:
  QueryDriverDatabase(llvm::ArrayRef<std::string> QueryDriverGlobs,
                      std::unique_ptr<GlobalCompilationDatabase> Base)
      : DelegatingCDB(std::move(Base)),
        QueryDriverRegex(convertGlobsToRegex(QueryDriverGlobs)) {}

  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const override {
    auto Cmd = DelegatingCDB::getCompileCommand(File);
    if (!Cmd || Cmd->CommandLine.empty())
      return Cmd;

    llvm::StringRef Lang;
    for (size_t I = 0, E = Cmd->CommandLine.size(); I < E; ++I) {
      llvm::StringRef Arg = Cmd->CommandLine[I];
      if (Arg == "-x" && I + 1 < E)
        Lang = Cmd->CommandLine[I + 1];
      else if (Arg.startswith("-x"))
        Lang = Arg.drop_front(2).trim();
    }
    if (Lang.empty()) {
      llvm::StringRef Ext = llvm::sys::path::extension(File).trim('.');
      auto Type = driver::types::lookupTypeForExtension(Ext);
      if (Type == driver::types::TY_INVALID) {
        elog("System include extraction: invalid file type for {0}", Ext);
        return {};
      }
      Lang = driver::types::getTypeName(Type);
    }

    llvm::SmallString<128> Driver(Cmd->CommandLine.front());
    if (llvm::any_of(Driver,
                       [](char C) { return llvm::sys::path::is_separator(C); }))
      // Driver is a not a single executable name but instead a path (either
      // relative or absolute).
      llvm::sys::fs::make_absolute(Cmd->Directory, Driver);

    if (auto Info =
            QueriedDrivers.get(/*Key=*/(Driver + ":" + Lang).str(), [&] {
              return extractSystemIncludesAndTarget(
                  Driver, Lang, Cmd->CommandLine, QueryDriverRegex);
            })) {
      setTarget(addSystemIncludes(*Cmd, Info->SystemIncludes), Info->Target);
    }
    return Cmd;
  }

private:
  // Caches includes extracted from a driver. Key is driver:lang.
  Memoize<llvm::StringMap<llvm::Optional<DriverInfo>>> QueriedDrivers;
  llvm::Regex QueryDriverRegex;
};
} // namespace

std::unique_ptr<GlobalCompilationDatabase>
getQueryDriverDatabase(llvm::ArrayRef<std::string> QueryDriverGlobs,
                       std::unique_ptr<GlobalCompilationDatabase> Base) {
  assert(Base && "Null base to SystemIncludeExtractor");
  if (QueryDriverGlobs.empty())
    return Base;
  return std::make_unique<QueryDriverDatabase>(QueryDriverGlobs,
                                               std::move(Base));
}

} // namespace clangd
} // namespace clang
