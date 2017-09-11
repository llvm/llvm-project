//===- MinGW/Driver.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// GNU ld style linker driver for COFF currently supporting mingw-w64.
///
//===----------------------------------------------------------------------===//

#include "lld/Driver/Driver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#endif

using namespace lld;
using namespace llvm;

namespace lld {
namespace mingw {
namespace {

// Create OptTable
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11, _12) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const opt::OptTable::Info InfoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X7, X8, X9, X10, X11, X12)      \
  {X1, X2, X10,         X11,         OPT_##ID, opt::Option::KIND##Class,       \
   X9, X8, OPT_##GROUP, OPT_##ALIAS, X7,       X12},
#include "Options.inc"
#undef OPTION
};

class COFFLdOptTable : public opt::OptTable {
public:
  COFFLdOptTable() : OptTable(InfoTable, false) {}
  opt::InputArgList parse(ArrayRef<const char *> Argv);
};

} // namespace

static std::vector<std::string> LinkArgs;
static std::vector<StringRef> SearchPaths;

static void error(const Twine &Msg) {
  errs() << Msg << "\n";
  llvm_shutdown();
  exit(1);
}

// Find a file by concatenating given paths.
static Optional<std::string> findFile(StringRef Path1, const Twine &Path2) {
  SmallString<128> S;
  sys::path::append(S, Path1, Path2);
  if (sys::fs::exists(S))
    return S.str().str();
  return None;
}

static Optional<std::string> findFromSearchPaths(StringRef Path) {
  for (StringRef Dir : SearchPaths)
    if (Optional<std::string> S = findFile(Dir, Path))
      return S;
  return None;
}

// This is for -lfoo. We'll look for libfoo.dll.a or libfoo.a from search paths.
static Optional<std::string> searchLibrary(StringRef Name, bool StaticOnly) {
  if (Name.startswith(":"))
    return findFromSearchPaths(Name.substr(1));
  for (StringRef Dir : SearchPaths) {
    if (!StaticOnly)
      if (Optional<std::string> S = findFile(Dir, "lib" + Name + ".dll.a"))
        return S;
    if (Optional<std::string> S = findFile(Dir, "lib" + Name + ".a"))
      return S;
  }
  return None;
}

// Add a given library by searching it from input search paths.
static void addLibrary(StringRef Name, bool StaticOnly) {
  if (Optional<std::string> Path = searchLibrary(Name, StaticOnly))
    LinkArgs.push_back(*Path);
  else
    error("unable to find library -l" + Name);
}

static void createFiles(opt::InputArgList &Args) {
  for (auto *Arg : Args) {
    switch (Arg->getOption().getUnaliasedOption().getID()) {
    case OPT_l:
      addLibrary(Arg->getValue(), Args.hasArg(OPT_Bstatic));
      break;
    case OPT_INPUT:
      LinkArgs.push_back(Arg->getValue());
      break;
    }
  }
}

static void forward(opt::InputArgList &Args, unsigned Key,
                    const std::string &OutArg, std::string Default = "") {
  StringRef S = Args.getLastArgValue(Key);
  if (!S.empty())
    LinkArgs.push_back(std::string("-").append(OutArg).append(":").append(S));
  else if (!Default.empty())
    LinkArgs.push_back(
        std::string("-").append(OutArg).append(":").append(Default));
}

static void forwardValue(opt::InputArgList &Args, unsigned Key,
                         const std::string &CmpArg, const std::string &OutArg) {
  StringRef S = Args.getLastArgValue(Key);
  if (S == CmpArg)
    LinkArgs.push_back(std::string("-").append(OutArg));
}

static bool convertValue(opt::InputArgList &Args, unsigned Key,
                         StringRef OutArg) {
  if (Args.hasArg(Key)) {
    LinkArgs.push_back(std::string("-").append(OutArg));
    return true;
  }
  return false;
}

opt::InputArgList COFFLdOptTable::parse(ArrayRef<const char *> Argv) {
  unsigned MissingIndex;
  unsigned MissingCount;
  SmallVector<const char *, 256> Vec(Argv.data(), Argv.data() + Argv.size());
  opt::InputArgList Args = this->ParseArgs(Vec, MissingIndex, MissingCount);
  if (MissingCount)
    error(Twine(Args.getArgString(MissingIndex)) + ": missing argument");
  if (!Args.hasArgNoClaim(OPT_INPUT) && !Args.hasArgNoClaim(OPT_l))
    error("no input files");
  for (auto *Arg : Args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + Arg->getSpelling());
  return Args;
}

bool link(ArrayRef<const char *> ArgsArr, raw_ostream &Diag) {
  COFFLdOptTable Parser;
  opt::InputArgList Args = Parser.parse(ArgsArr.slice(1));
  LinkArgs.push_back(ArgsArr[0]);

  forwardValue(Args, OPT_m, "i386pe", "machine:x86");
  forwardValue(Args, OPT_m, "i386pep", "machine:x64");
  forwardValue(Args, OPT_m, "thumb2pe", "machine:arm");
  forwardValue(Args, OPT_m, "arm64pe", "machine:arm64");

  forward(Args, OPT_o, "out",
          convertValue(Args, OPT_shared, "dll") ? "a.dll" : "a.exe");
  forward(Args, OPT_entry, "entry");
  forward(Args, OPT_subs, "subsystem");
  forward(Args, OPT_outlib, "implib");
  forward(Args, OPT_stack, "stack");

  for (auto *Arg : Args.filtered(OPT_L))
    SearchPaths.push_back(Arg->getValue());

  createFiles(Args);

  // handle __image_base__
  if (Args.getLastArgValue(OPT_m) == "i386pe")
    LinkArgs.push_back("/alternatename:__image_base__=___ImageBase");
  else
    LinkArgs.push_back("/alternatename:__image_base__=__ImageBase");

  // repack vector of strings to vector of const char pointers for coff::link
  std::vector<const char *> Vec;
  for (const std::string &S : LinkArgs)
    Vec.push_back(S.c_str());
  return coff::link(Vec);
}

} // namespace mingw
} // namespace lld
