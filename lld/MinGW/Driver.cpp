//===- MinGW/Driver.cpp ---------------------------------------------------===//
//
// Part of the the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MinGW is a GNU development environment for Windows. It consists of GNU
// tools such as GCC and GNU ld. Unlike Cygwin, there's no POSIX-compatible
// layer, as it aims to be a native development toolchain.
//
// lld/MinGW is a drop-in replacement for GNU ld/MinGW.
//
// Being a native development tool, a MinGW linker is not very different from
// Microsoft link.exe, so a MinGW linker can be implemented as a thin wrapper
// for lld/COFF. This driver takes Unix-ish command line options, translates
// them to Windows-ish ones, and then passes them to lld/COFF.
//
// When this driver calls the lld/COFF driver, it passes a hidden option
// "-lldmingw" along with other user-supplied options, to run the lld/COFF
// linker in "MinGW mode".
//
// There are subtle differences between MS link.exe and GNU ld/MinGW, and GNU
// ld/MinGW implements a few GNU-specific features. Such features are directly
// implemented in lld/COFF and enabled only when the linker is running in MinGW
// mode.
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>

using namespace lld;
using namespace llvm::opt;
using namespace llvm;

// Create OptTable
enum {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

// Create table mapping all options defined in Options.td
static constexpr opt::OptTable::Info infoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS, METAVAR,     \
               VALUES, SUBCOMMANDIDS_OFFSET)                                   \
  {PREFIX,                                                                     \
   NAME,                                                                       \
   HELPTEXT,                                                                   \
   HELPTEXTSFORVARIANTS,                                                       \
   METAVAR,                                                                    \
   OPT_##ID,                                                                   \
   opt::Option::KIND##Class,                                                   \
   PARAM,                                                                      \
   FLAGS,                                                                      \
   VISIBILITY,                                                                 \
   OPT_##GROUP,                                                                \
   OPT_##ALIAS,                                                                \
   ALIASARGS,                                                                  \
   VALUES,                                                                     \
   SUBCOMMANDIDS_OFFSET},
#include "Options.inc"
#undef OPTION
};

namespace {
class MinGWOptTable : public opt::GenericOptTable {
public:
  MinGWOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, infoTable,
                             false) {}
  opt::InputArgList parse(ArrayRef<const char *> argv);
};
} // namespace

static void printHelp(CommonLinkerContext &ctx, const char *argv0) {
  auto &outs = ctx.e.outs();
  MinGWOptTable().printHelp(
      outs, (std::string(argv0) + " [options] file...").c_str(), "lld",
      /*ShowHidden=*/false, /*ShowAllAliases=*/true);
  outs << '\n';
}

static cl::TokenizerCallback getQuotingStyle() {
  if (Triple(sys::getProcessTriple()).getOS() == Triple::Win32)
    return cl::TokenizeWindowsCommandLine;
  return cl::TokenizeGNUCommandLine;
}

opt::InputArgList MinGWOptTable::parse(ArrayRef<const char *> argv) {
  unsigned missingIndex;
  unsigned missingCount;

  SmallVector<const char *, 256> vec(argv.data(), argv.data() + argv.size());
  cl::ExpandResponseFiles(saver(), getQuotingStyle(), vec);
  opt::InputArgList args = this->ParseArgs(vec, missingIndex, missingCount);

  if (missingCount)
    error(StringRef(args.getArgString(missingIndex)) + ": missing argument");
  for (auto *arg : args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + arg->getAsString(args));
  return args;
}

// Find a file by concatenating given paths.
static std::optional<std::string> findFile(StringRef path1,
                                           const Twine &path2) {
  SmallString<128> s;
  sys::path::append(s, path1, path2);
  if (sys::fs::exists(s))
    return std::string(s);
  return std::nullopt;
}

// This is for -lfoo. We'll look for libfoo.dll.a or libfoo.a from search paths.
static std::string searchLibrary(StringRef name,
                                 ArrayRef<StringRef> searchPaths, bool bStatic,
                                 StringRef prefix) {
  if (name.starts_with(":")) {
    for (StringRef dir : searchPaths)
      if (std::optional<std::string> s = findFile(dir, name.substr(1)))
        return *s;
    error("unable to find library -l" + name);
    return "";
  }

  for (StringRef dir : searchPaths) {
    if (!bStatic) {
      if (std::optional<std::string> s = findFile(dir, "lib" + name + ".dll.a"))
        return *s;
      if (std::optional<std::string> s = findFile(dir, name + ".dll.a"))
        return *s;
    }
    if (std::optional<std::string> s = findFile(dir, "lib" + name + ".a"))
      return *s;
    if (std::optional<std::string> s = findFile(dir, name + ".lib"))
      return *s;
    if (!bStatic) {
      if (std::optional<std::string> s = findFile(dir, prefix + name + ".dll"))
        return *s;
      if (std::optional<std::string> s = findFile(dir, name + ".dll"))
        return *s;
    }
  }
  error("unable to find library -l" + name);
  return "";
}

static bool isI386Target(const opt::InputArgList &args,
                         const Triple &defaultTarget) {
  auto *a = args.getLastArg(OPT_m);
  if (a)
    return StringRef(a->getValue()) == "i386pe";
  return defaultTarget.getArch() == Triple::x86;
}

namespace lld {
namespace coff {
bool link(ArrayRef<const char *> argsArr, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
}

namespace mingw {
// Convert Unix-ish command line arguments to Windows-ish ones and
// then call coff::link.
bool link(ArrayRef<const char *> argsArr, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
  auto *ctx = new CommonLinkerContext;
  ctx->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);

  MinGWOptTable parser;
  opt::InputArgList args = parser.parse(argsArr.slice(1));

  if (errorCount())
    return false;

  if (args.hasArg(OPT_help)) {
    printHelp(*ctx, argsArr[0]);
    return true;
  }

  if (args.hasArg(OPT_v) || args.hasArg(OPT_version))
    message(getLLDVersion() + " (compatible with GNU linkers)");

  if (args.hasArg(OPT_v) && !args.hasArg(OPT_INPUT) && !args.hasArg(OPT_l))
    return true;
  if (args.hasArg(OPT_version))
    return true;

  if (!args.hasArg(OPT_INPUT) && !args.hasArg(OPT_l)) {
    error("no input files");
    return false;
  }

  Triple defaultTarget(Triple::normalize(sys::getDefaultTargetTriple()));

  std::vector<std::string> linkArgs;
  auto add = [&](const Twine &s) { linkArgs.push_back(s.str()); };

  add("lld-link");
  add("-lldmingw");

  // Here the original code continues with dozens of option conversions...
  // [FULL OPTION MAPPING CODE HERE — as in your file]

  if (errorCount())
    return false;

  if (args.hasArg(OPT_verbose) || args.hasArg(OPT__HASH_HASH_HASH))
    ctx->e.errs() << llvm::join(linkArgs, " ") << "\n";

  if (args.hasArg(OPT__HASH_HASH_HASH))
    return true;

  std::vector<const char *> vec;
  for (const std::string &s : linkArgs)
    vec.push_back(s.c_str());

  vec[0] = argsArr[0];

  lld::CommonLinkerContext::destroy();

  return coff::link(vec, stdoutOS, stderrOS, exitEarly, disableOutput);
}
} // namespace mingw
} // namespace lld
