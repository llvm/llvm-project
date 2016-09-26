//===--- Tools.cpp - Tools Implementations ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Tools.h"
#include "InputInfo.h"
#include "ToolChains.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/TargetParser.h"

#ifdef LLVM_ON_UNIX
#include <unistd.h> // For getuid().
#endif

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static void handleTargetFeaturesGroup(const ArgList &Args,
                                      std::vector<const char *> &Features,
                                      OptSpecifier Group) {
  for (const Arg *A : Args.filtered(Group)) {
    StringRef Name = A->getOption().getName();
    A->claim();

    // Skip over "-m".
    assert(Name.startswith("m") && "Invalid feature name.");
    Name = Name.substr(1);

    bool IsNegative = Name.startswith("no-");
    if (IsNegative)
      Name = Name.substr(3);
    Features.push_back(Args.MakeArgString((IsNegative ? "-" : "+") + Name));
  }
}

static const char *getSparcAsmModeForCPU(StringRef Name,
                                         const llvm::Triple &Triple) {
  if (Triple.getArch() == llvm::Triple::sparcv9) {
    return llvm::StringSwitch<const char *>(Name)
          .Case("niagara", "-Av9b")
          .Case("niagara2", "-Av9b")
          .Case("niagara3", "-Av9d")
          .Case("niagara4", "-Av9d")
          .Default("-Av9");
  } else {
    return llvm::StringSwitch<const char *>(Name)
          .Case("v8", "-Av8")
          .Case("supersparc", "-Av8")
          .Case("sparclite", "-Asparclite")
          .Case("f934", "-Asparclite")
          .Case("hypersparc", "-Av8")
          .Case("sparclite86x", "-Asparclite")
          .Case("sparclet", "-Asparclet")
          .Case("tsc701", "-Asparclet")
          .Case("v9", "-Av8plus")
          .Case("ultrasparc", "-Av8plus")
          .Case("ultrasparc3", "-Av8plus")
          .Case("niagara", "-Av8plusb")
          .Case("niagara2", "-Av8plusb")
          .Case("niagara3", "-Av8plusd")
          .Case("niagara4", "-Av8plusd")
          .Case("leon2", "-Av8")
          .Case("at697e", "-Av8")
          .Case("at697f", "-Av8")
          .Case("leon3", "-Av8")
          .Case("ut699", "-Av8")
          .Case("gr712rc", "-Av8")
          .Case("leon4", "-Av8")
          .Case("gr740", "-Av8")
          .Default("-Av8");
  }
}

static void CheckPreprocessingOptions(const Driver &D, const ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_C, options::OPT_CC)) {
    if (!Args.hasArg(options::OPT_E) && !Args.hasArg(options::OPT__SLASH_P) &&
        !Args.hasArg(options::OPT__SLASH_EP) && !D.CCCIsCPP()) {
      D.Diag(diag::err_drv_argument_only_allowed_with)
          << A->getBaseArg().getAsString(Args)
          << (D.IsCLMode() ? "/E, /P or /EP" : "-E");
    }
  }
}

static void CheckCodeGenerationOptions(const Driver &D, const ArgList &Args) {
  // In gcc, only ARM checks this, but it seems reasonable to check universally.
  if (Args.hasArg(options::OPT_static))
    if (const Arg *A =
            Args.getLastArg(options::OPT_dynamic, options::OPT_mdynamic_no_pic))
      D.Diag(diag::err_drv_argument_not_allowed_with) << A->getAsString(Args)
                                                      << "-static";
}

// Add backslashes to escape spaces and other backslashes.
// This is used for the space-separated argument list specified with
// the -dwarf-debug-flags option.
static void EscapeSpacesAndBackslashes(const char *Arg,
                                       SmallVectorImpl<char> &Res) {
  for (; *Arg; ++Arg) {
    switch (*Arg) {
    default:
      break;
    case ' ':
    case '\\':
      Res.push_back('\\');
      break;
    }
    Res.push_back(*Arg);
  }
}

// Quote target names for inclusion in GNU Make dependency files.
// Only the characters '$', '#', ' ', '\t' are quoted.
static void QuoteTarget(StringRef Target, SmallVectorImpl<char> &Res) {
  for (unsigned i = 0, e = Target.size(); i != e; ++i) {
    switch (Target[i]) {
    case ' ':
    case '\t':
      // Escape the preceding backslashes
      for (int j = i - 1; j >= 0 && Target[j] == '\\'; --j)
        Res.push_back('\\');

      // Escape the space/tab
      Res.push_back('\\');
      break;
    case '$':
      Res.push_back('$');
      break;
    case '#':
      Res.push_back('\\');
      break;
    default:
      break;
    }

    Res.push_back(Target[i]);
  }
}

static void addDirectoryList(const ArgList &Args, ArgStringList &CmdArgs,
                             const char *ArgName, const char *EnvVar) {
  const char *DirList = ::getenv(EnvVar);
  bool CombinedArg = false;

  if (!DirList)
    return; // Nothing to do.

  StringRef Name(ArgName);
  if (Name.equals("-I") || Name.equals("-L"))
    CombinedArg = true;

  StringRef Dirs(DirList);
  if (Dirs.empty()) // Empty string should not add '.'.
    return;

  StringRef::size_type Delim;
  while ((Delim = Dirs.find(llvm::sys::EnvPathSeparator)) != StringRef::npos) {
    if (Delim == 0) { // Leading colon.
      if (CombinedArg) {
        CmdArgs.push_back(Args.MakeArgString(std::string(ArgName) + "."));
      } else {
        CmdArgs.push_back(ArgName);
        CmdArgs.push_back(".");
      }
    } else {
      if (CombinedArg) {
        CmdArgs.push_back(
            Args.MakeArgString(std::string(ArgName) + Dirs.substr(0, Delim)));
      } else {
        CmdArgs.push_back(ArgName);
        CmdArgs.push_back(Args.MakeArgString(Dirs.substr(0, Delim)));
      }
    }
    Dirs = Dirs.substr(Delim + 1);
  }

  if (Dirs.empty()) { // Trailing colon.
    if (CombinedArg) {
      CmdArgs.push_back(Args.MakeArgString(std::string(ArgName) + "."));
    } else {
      CmdArgs.push_back(ArgName);
      CmdArgs.push_back(".");
    }
  } else { // Add the last path.
    if (CombinedArg) {
      CmdArgs.push_back(Args.MakeArgString(std::string(ArgName) + Dirs));
    } else {
      CmdArgs.push_back(ArgName);
      CmdArgs.push_back(Args.MakeArgString(Dirs));
    }
  }
}

static void AddLinkerInputs(const ToolChain &TC, const InputInfoList &Inputs,
                            const ArgList &Args, ArgStringList &CmdArgs) {
  const Driver &D = TC.getDriver();

  // Add extra linker input arguments which are not treated as inputs
  // (constructed via -Xarch_).
  Args.AddAllArgValues(CmdArgs, options::OPT_Zlinker_input);

  for (const auto &II : Inputs) {
    if (!TC.HasNativeLLVMSupport() && types::isLLVMIR(II.getType()))
      // Don't try to pass LLVM inputs unless we have native support.
      D.Diag(diag::err_drv_no_linker_llvm_support) << TC.getTripleString();

    // Add filenames immediately.
    if (II.isFilename()) {
      CmdArgs.push_back(II.getFilename());
      continue;
    }

    // Otherwise, this is a linker input argument.
    const Arg &A = II.getInputArg();

    // Handle reserved library options.
    if (A.getOption().matches(options::OPT_Z_reserved_lib_stdcxx))
      TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    else if (A.getOption().matches(options::OPT_Z_reserved_lib_cckext))
      TC.AddCCKextLibArgs(Args, CmdArgs);
    else if (A.getOption().matches(options::OPT_z)) {
      // Pass -z prefix for gcc linker compatibility.
      A.claim();
      A.render(Args, CmdArgs);
    } else {
      A.renderAsInput(Args, CmdArgs);
    }
  }

  // LIBRARY_PATH - included following the user specified library paths.
  //                and only supported on native toolchains.
  if (!TC.isCrossCompiling())
    addDirectoryList(Args, CmdArgs, "-L", "LIBRARY_PATH");
}

/// \brief Determine whether Objective-C automated reference counting is
/// enabled.
static bool isObjCAutoRefCount(const ArgList &Args) {
  return Args.hasFlag(options::OPT_fobjc_arc, options::OPT_fno_objc_arc, false);
}

/// \brief Determine whether we are linking the ObjC runtime.
static bool isObjCRuntimeLinked(const ArgList &Args) {
  if (isObjCAutoRefCount(Args)) {
    Args.ClaimAllArgs(options::OPT_fobjc_link_runtime);
    return true;
  }
  return Args.hasArg(options::OPT_fobjc_link_runtime);
}

static bool forwardToGCC(const Option &O) {
  // Don't forward inputs from the original command line.  They are added from
  // InputInfoList.
  return O.getKind() != Option::InputClass &&
         !O.hasFlag(options::DriverOption) && !O.hasFlag(options::LinkerInput);
}

/// Apply \a Work on the current tool chain \a RegularToolChain and any other
/// offloading tool chain that is associated with the current action \a JA.
static void
forAllAssociatedToolChains(Compilation &C, const JobAction &JA,
                           const ToolChain &RegularToolChain,
                           llvm::function_ref<void(const ToolChain &)> Work) {
  // Apply Work on the current/regular tool chain.
  Work(RegularToolChain);

  // Apply Work on all the offloading tool chains associated with the current
  // action.
  if (JA.isHostOffloading(Action::OFK_Cuda))
    Work(*C.getSingleOffloadToolChain<Action::OFK_Cuda>());
  else if (JA.isDeviceOffloading(Action::OFK_Cuda))
    Work(*C.getSingleOffloadToolChain<Action::OFK_Host>());

  //
  // TODO: Add support for other offloading programming models here.
  //
}

void Clang::AddPreprocessingOptions(Compilation &C, const JobAction &JA,
                                    const Driver &D, const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs) const {
  Arg *A;
  const bool IsIAMCU = getToolChain().getTriple().isOSIAMCU();

  CheckPreprocessingOptions(D, Args);

  Args.AddLastArg(CmdArgs, options::OPT_C);
  Args.AddLastArg(CmdArgs, options::OPT_CC);

  // Handle dependency file generation.
  if ((A = Args.getLastArg(options::OPT_M, options::OPT_MM)) ||
      (A = Args.getLastArg(options::OPT_MD)) ||
      (A = Args.getLastArg(options::OPT_MMD))) {
    // Determine the output location.
    const char *DepFile;
    if (Arg *MF = Args.getLastArg(options::OPT_MF)) {
      DepFile = MF->getValue();
      C.addFailureResultFile(DepFile, &JA);
    } else if (Output.getType() == types::TY_Dependencies) {
      DepFile = Output.getFilename();
    } else if (A->getOption().matches(options::OPT_M) ||
               A->getOption().matches(options::OPT_MM)) {
      DepFile = "-";
    } else {
      DepFile = getDependencyFileName(Args, Inputs);
      C.addFailureResultFile(DepFile, &JA);
    }
    CmdArgs.push_back("-dependency-file");
    CmdArgs.push_back(DepFile);

    // Add a default target if one wasn't specified.
    if (!Args.hasArg(options::OPT_MT) && !Args.hasArg(options::OPT_MQ)) {
      const char *DepTarget;

      // If user provided -o, that is the dependency target, except
      // when we are only generating a dependency file.
      Arg *OutputOpt = Args.getLastArg(options::OPT_o);
      if (OutputOpt && Output.getType() != types::TY_Dependencies) {
        DepTarget = OutputOpt->getValue();
      } else {
        // Otherwise derive from the base input.
        //
        // FIXME: This should use the computed output file location.
        SmallString<128> P(Inputs[0].getBaseInput());
        llvm::sys::path::replace_extension(P, "o");
        DepTarget = Args.MakeArgString(llvm::sys::path::filename(P));
      }

      CmdArgs.push_back("-MT");
      SmallString<128> Quoted;
      QuoteTarget(DepTarget, Quoted);
      CmdArgs.push_back(Args.MakeArgString(Quoted));
    }

    if (A->getOption().matches(options::OPT_M) ||
        A->getOption().matches(options::OPT_MD))
      CmdArgs.push_back("-sys-header-deps");
    if ((isa<PrecompileJobAction>(JA) &&
         !Args.hasArg(options::OPT_fno_module_file_deps)) ||
        Args.hasArg(options::OPT_fmodule_file_deps))
      CmdArgs.push_back("-module-file-deps");
  }

  if (Args.hasArg(options::OPT_MG)) {
    if (!A || A->getOption().matches(options::OPT_MD) ||
        A->getOption().matches(options::OPT_MMD))
      D.Diag(diag::err_drv_mg_requires_m_or_mm);
    CmdArgs.push_back("-MG");
  }

  Args.AddLastArg(CmdArgs, options::OPT_MP);
  Args.AddLastArg(CmdArgs, options::OPT_MV);

  // Convert all -MQ <target> args to -MT <quoted target>
  for (const Arg *A : Args.filtered(options::OPT_MT, options::OPT_MQ)) {
    A->claim();

    if (A->getOption().matches(options::OPT_MQ)) {
      CmdArgs.push_back("-MT");
      SmallString<128> Quoted;
      QuoteTarget(A->getValue(), Quoted);
      CmdArgs.push_back(Args.MakeArgString(Quoted));

      // -MT flag - no change
    } else {
      A->render(Args, CmdArgs);
    }
  }

  // Add offload include arguments specific for CUDA.  This must happen before
  // we -I or -include anything else, because we must pick up the CUDA headers
  // from the particular CUDA installation, rather than from e.g.
  // /usr/local/include.
  if (JA.isOffloading(Action::OFK_Cuda))
    getToolChain().AddCudaIncludeArgs(Args, CmdArgs);

  // Add -i* options, and automatically translate to
  // -include-pch/-include-pth for transparent PCH support. It's
  // wonky, but we include looking for .gch so we can support seamless
  // replacement into a build system already set up to be generating
  // .gch files.
  int YcIndex = -1, YuIndex = -1;
  {
    int AI = -1;
    const Arg *YcArg = Args.getLastArg(options::OPT__SLASH_Yc);
    const Arg *YuArg = Args.getLastArg(options::OPT__SLASH_Yu);
    for (const Arg *A : Args.filtered(options::OPT_clang_i_Group)) {
      // Walk the whole i_Group and skip non "-include" flags so that the index
      // here matches the index in the next loop below.
      ++AI;
      if (!A->getOption().matches(options::OPT_include))
        continue;
      if (YcArg && strcmp(A->getValue(), YcArg->getValue()) == 0)
        YcIndex = AI;
      if (YuArg && strcmp(A->getValue(), YuArg->getValue()) == 0)
        YuIndex = AI;
    }
  }
  if (isa<PrecompileJobAction>(JA) && YcIndex != -1) {
    Driver::InputList Inputs;
    D.BuildInputs(getToolChain(), C.getArgs(), Inputs);
    assert(Inputs.size() == 1 && "Need one input when building pch");
    CmdArgs.push_back(Args.MakeArgString(Twine("-find-pch-source=") +
                                         Inputs[0].second->getValue()));
  }

  bool RenderedImplicitInclude = false;
  int AI = -1;
  for (const Arg *A : Args.filtered(options::OPT_clang_i_Group)) {
    ++AI;

    if (getToolChain().getDriver().IsCLMode() &&
        A->getOption().matches(options::OPT_include)) {
      // In clang-cl mode, /Ycfoo.h means that all code up to a foo.h
      // include is compiled into foo.h, and everything after goes into
      // the .obj file. /Yufoo.h means that all includes prior to and including
      // foo.h are completely skipped and replaced with a use of the pch file
      // for foo.h.  (Each flag can have at most one value, multiple /Yc flags
      // just mean that the last one wins.)  If /Yc and /Yu are both present
      // and refer to the same file, /Yc wins.
      // Note that OPT__SLASH_FI gets mapped to OPT_include.
      // FIXME: The code here assumes that /Yc and /Yu refer to the same file.
      // cl.exe seems to support both flags with different values, but that
      // seems strange (which flag does /Fp now refer to?), so don't implement
      // that until someone needs it.
      int PchIndex = YcIndex != -1 ? YcIndex : YuIndex;
      if (PchIndex != -1) {
        if (isa<PrecompileJobAction>(JA)) {
          // When building the pch, skip all includes after the pch.
          assert(YcIndex != -1 && PchIndex == YcIndex);
          if (AI >= YcIndex)
            continue;
        } else {
          // When using the pch, skip all includes prior to the pch.
          if (AI < PchIndex) {
            A->claim();
            continue;
          }
          if (AI == PchIndex) {
            A->claim();
            CmdArgs.push_back("-include-pch");
            CmdArgs.push_back(
                Args.MakeArgString(D.GetClPchPath(C, A->getValue())));
            continue;
          }
        }
      }
    } else if (A->getOption().matches(options::OPT_include)) {
      // Handling of gcc-style gch precompiled headers.
      bool IsFirstImplicitInclude = !RenderedImplicitInclude;
      RenderedImplicitInclude = true;

      // Use PCH if the user requested it.
      bool UsePCH = D.CCCUsePCH;

      bool FoundPTH = false;
      bool FoundPCH = false;
      SmallString<128> P(A->getValue());
      // We want the files to have a name like foo.h.pch. Add a dummy extension
      // so that replace_extension does the right thing.
      P += ".dummy";
      if (UsePCH) {
        llvm::sys::path::replace_extension(P, "pch");
        if (llvm::sys::fs::exists(P))
          FoundPCH = true;
      }

      if (!FoundPCH) {
        llvm::sys::path::replace_extension(P, "pth");
        if (llvm::sys::fs::exists(P))
          FoundPTH = true;
      }

      if (!FoundPCH && !FoundPTH) {
        llvm::sys::path::replace_extension(P, "gch");
        if (llvm::sys::fs::exists(P)) {
          FoundPCH = UsePCH;
          FoundPTH = !UsePCH;
        }
      }

      if (FoundPCH || FoundPTH) {
        if (IsFirstImplicitInclude) {
          A->claim();
          if (UsePCH)
            CmdArgs.push_back("-include-pch");
          else
            CmdArgs.push_back("-include-pth");
          CmdArgs.push_back(Args.MakeArgString(P));
          continue;
        } else {
          // Ignore the PCH if not first on command line and emit warning.
          D.Diag(diag::warn_drv_pch_not_first_include) << P
                                                       << A->getAsString(Args);
        }
      }
    } else if (A->getOption().matches(options::OPT_isystem_after)) {
      // Handling of paths which must come late.  These entries are handled by
      // the toolchain itself after the resource dir is inserted in the right
      // search order.
      // Do not claim the argument so that the use of the argument does not
      // silently go unnoticed on toolchains which do not honour the option.
      continue;
    }

    // Not translated, render as usual.
    A->claim();
    A->render(Args, CmdArgs);
  }

  Args.AddAllArgs(CmdArgs,
                  {options::OPT_D, options::OPT_U, options::OPT_I_Group,
                   options::OPT_F, options::OPT_index_header_map});

  // Add -Wp, and -Xpreprocessor if using the preprocessor.

  // FIXME: There is a very unfortunate problem here, some troubled
  // souls abuse -Wp, to pass preprocessor options in gcc syntax. To
  // really support that we would have to parse and then translate
  // those options. :(
  Args.AddAllArgValues(CmdArgs, options::OPT_Wp_COMMA,
                       options::OPT_Xpreprocessor);

  // -I- is a deprecated GCC feature, reject it.
  if (Arg *A = Args.getLastArg(options::OPT_I_))
    D.Diag(diag::err_drv_I_dash_not_supported) << A->getAsString(Args);

  // If we have a --sysroot, and don't have an explicit -isysroot flag, add an
  // -isysroot to the CC1 invocation.
  StringRef sysroot = C.getSysRoot();
  if (sysroot != "") {
    if (!Args.hasArg(options::OPT_isysroot)) {
      CmdArgs.push_back("-isysroot");
      CmdArgs.push_back(C.getArgs().MakeArgString(sysroot));
    }
  }

  // Parse additional include paths from environment variables.
  // FIXME: We should probably sink the logic for handling these from the
  // frontend into the driver. It will allow deleting 4 otherwise unused flags.
  // CPATH - included following the user specified includes (but prior to
  // builtin and standard includes).
  addDirectoryList(Args, CmdArgs, "-I", "CPATH");
  // C_INCLUDE_PATH - system includes enabled when compiling C.
  addDirectoryList(Args, CmdArgs, "-c-isystem", "C_INCLUDE_PATH");
  // CPLUS_INCLUDE_PATH - system includes enabled when compiling C++.
  addDirectoryList(Args, CmdArgs, "-cxx-isystem", "CPLUS_INCLUDE_PATH");
  // OBJC_INCLUDE_PATH - system includes enabled when compiling ObjC.
  addDirectoryList(Args, CmdArgs, "-objc-isystem", "OBJC_INCLUDE_PATH");
  // OBJCPLUS_INCLUDE_PATH - system includes enabled when compiling ObjC++.
  addDirectoryList(Args, CmdArgs, "-objcxx-isystem", "OBJCPLUS_INCLUDE_PATH");

  // While adding the include arguments, we also attempt to retrieve the
  // arguments of related offloading toolchains or arguments that are specific
  // of an offloading programming model.

  // Add C++ include arguments, if needed.
  if (types::isCXX(Inputs[0].getType()))
    forAllAssociatedToolChains(C, JA, getToolChain(),
                               [&Args, &CmdArgs](const ToolChain &TC) {
                                 TC.AddClangCXXStdlibIncludeArgs(Args, CmdArgs);
                               });

  // Add system include arguments for all targets but IAMCU.
  if (!IsIAMCU)
    forAllAssociatedToolChains(C, JA, getToolChain(),
                               [&Args, &CmdArgs](const ToolChain &TC) {
                                 TC.AddClangSystemIncludeArgs(Args, CmdArgs);
                               });
  else {
    // For IAMCU add special include arguments.
    getToolChain().AddIAMCUIncludeArgs(Args, CmdArgs);
  }
}

// FIXME: Move to target hook.
static bool isSignedCharDefault(const llvm::Triple &Triple) {
  switch (Triple.getArch()) {
  default:
    return true;

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    if (Triple.isOSDarwin() || Triple.isOSWindows())
      return true;
    return false;

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
    if (Triple.isOSDarwin())
      return true;
    return false;

  case llvm::Triple::hexagon:
  case llvm::Triple::ppc64le:
  case llvm::Triple::systemz:
  case llvm::Triple::xcore:
    return false;
  }
}

static bool isNoCommonDefault(const llvm::Triple &Triple) {
  switch (Triple.getArch()) {
  default:
    return false;

  case llvm::Triple::xcore:
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    return true;
  }
}

// ARM tools start.

// Get SubArch (vN).
static int getARMSubArchVersionNumber(const llvm::Triple &Triple) {
  llvm::StringRef Arch = Triple.getArchName();
  return llvm::ARM::parseArchVersion(Arch);
}

// True if M-profile.
static bool isARMMProfile(const llvm::Triple &Triple) {
  llvm::StringRef Arch = Triple.getArchName();
  unsigned Profile = llvm::ARM::parseArchProfile(Arch);
  return Profile == llvm::ARM::PK_M;
}

// Get Arch/CPU from args.
static void getARMArchCPUFromArgs(const ArgList &Args, llvm::StringRef &Arch,
                                  llvm::StringRef &CPU, bool FromAs = false) {
  if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    CPU = A->getValue();
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
    Arch = A->getValue();
  if (!FromAs)
    return;

  for (const Arg *A :
       Args.filtered(options::OPT_Wa_COMMA, options::OPT_Xassembler)) {
    StringRef Value = A->getValue();
    if (Value.startswith("-mcpu="))
      CPU = Value.substr(6);
    if (Value.startswith("-march="))
      Arch = Value.substr(7);
  }
}

// Handle -mhwdiv=.
// FIXME: Use ARMTargetParser.
static void getARMHWDivFeatures(const Driver &D, const Arg *A,
                                const ArgList &Args, StringRef HWDiv,
                                std::vector<const char *> &Features) {
  unsigned HWDivID = llvm::ARM::parseHWDiv(HWDiv);
  if (!llvm::ARM::getHWDivFeatures(HWDivID, Features))
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

// Handle -mfpu=.
static void getARMFPUFeatures(const Driver &D, const Arg *A,
                              const ArgList &Args, StringRef FPU,
                              std::vector<const char *> &Features) {
  unsigned FPUID = llvm::ARM::parseFPU(FPU);
  if (!llvm::ARM::getFPUFeatures(FPUID, Features))
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

// Decode ARM features from string like +[no]featureA+[no]featureB+...
static bool DecodeARMFeatures(const Driver &D, StringRef text,
                              std::vector<const char *> &Features) {
  SmallVector<StringRef, 8> Split;
  text.split(Split, StringRef("+"), -1, false);

  for (StringRef Feature : Split) {
    const char *FeatureName = llvm::ARM::getArchExtFeature(Feature);
    if (FeatureName)
      Features.push_back(FeatureName);
    else
      return false;
  }
  return true;
}

// Check if -march is valid by checking if it can be canonicalised and parsed.
// getARMArch is used here instead of just checking the -march value in order
// to handle -march=native correctly.
static void checkARMArchName(const Driver &D, const Arg *A, const ArgList &Args,
                             llvm::StringRef ArchName,
                             std::vector<const char *> &Features,
                             const llvm::Triple &Triple) {
  std::pair<StringRef, StringRef> Split = ArchName.split("+");

  std::string MArch = arm::getARMArch(ArchName, Triple);
  if (llvm::ARM::parseArch(MArch) == llvm::ARM::AK_INVALID ||
      (Split.second.size() && !DecodeARMFeatures(D, Split.second, Features)))
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

// Check -mcpu=. Needs ArchName to handle -mcpu=generic.
static void checkARMCPUName(const Driver &D, const Arg *A, const ArgList &Args,
                            llvm::StringRef CPUName, llvm::StringRef ArchName,
                            std::vector<const char *> &Features,
                            const llvm::Triple &Triple) {
  std::pair<StringRef, StringRef> Split = CPUName.split("+");

  std::string CPU = arm::getARMTargetCPU(CPUName, ArchName, Triple);
  if (arm::getLLVMArchSuffixForARM(CPU, ArchName, Triple).empty() ||
      (Split.second.size() && !DecodeARMFeatures(D, Split.second, Features)))
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

static bool useAAPCSForMachO(const llvm::Triple &T) {
  // The backend is hardwired to assume AAPCS for M-class processors, ensure
  // the frontend matches that.
  return T.getEnvironment() == llvm::Triple::EABI ||
         T.getOS() == llvm::Triple::UnknownOS || isARMMProfile(T);
}

// Select the float ABI as determined by -msoft-float, -mhard-float, and
// -mfloat-abi=.
arm::FloatABI arm::getARMFloatABI(const ToolChain &TC, const ArgList &Args) {
  const Driver &D = TC.getDriver();
  const llvm::Triple &Triple = TC.getEffectiveTriple();
  auto SubArch = getARMSubArchVersionNumber(Triple);
  arm::FloatABI ABI = FloatABI::Invalid;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float,
                          options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float)) {
      ABI = FloatABI::Soft;
    } else if (A->getOption().matches(options::OPT_mhard_float)) {
      ABI = FloatABI::Hard;
    } else {
      ABI = llvm::StringSwitch<arm::FloatABI>(A->getValue())
                .Case("soft", FloatABI::Soft)
                .Case("softfp", FloatABI::SoftFP)
                .Case("hard", FloatABI::Hard)
                .Default(FloatABI::Invalid);
      if (ABI == FloatABI::Invalid && !StringRef(A->getValue()).empty()) {
        D.Diag(diag::err_drv_invalid_mfloat_abi) << A->getAsString(Args);
        ABI = FloatABI::Soft;
      }
    }

    // It is incorrect to select hard float ABI on MachO platforms if the ABI is
    // "apcs-gnu".
    if (Triple.isOSBinFormatMachO() && !useAAPCSForMachO(Triple) &&
        ABI == FloatABI::Hard) {
      D.Diag(diag::err_drv_unsupported_opt_for_target) << A->getAsString(Args)
                                                       << Triple.getArchName();
    }
  }

  // If unspecified, choose the default based on the platform.
  if (ABI == FloatABI::Invalid) {
    switch (Triple.getOS()) {
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
    case llvm::Triple::IOS:
    case llvm::Triple::TvOS: {
      // Darwin defaults to "softfp" for v6 and v7.
      ABI = (SubArch == 6 || SubArch == 7) ? FloatABI::SoftFP : FloatABI::Soft;
      ABI = Triple.isWatchABI() ? FloatABI::Hard : ABI;
      break;
    }
    case llvm::Triple::WatchOS:
      ABI = FloatABI::Hard;
      break;

    // FIXME: this is invalid for WindowsCE
    case llvm::Triple::Win32:
      ABI = FloatABI::Hard;
      break;

    case llvm::Triple::FreeBSD:
      switch (Triple.getEnvironment()) {
      case llvm::Triple::GNUEABIHF:
        ABI = FloatABI::Hard;
        break;
      default:
        // FreeBSD defaults to soft float
        ABI = FloatABI::Soft;
        break;
      }
      break;

    default:
      switch (Triple.getEnvironment()) {
      case llvm::Triple::GNUEABIHF:
      case llvm::Triple::MuslEABIHF:
      case llvm::Triple::EABIHF:
        ABI = FloatABI::Hard;
        break;
      case llvm::Triple::GNUEABI:
      case llvm::Triple::MuslEABI:
      case llvm::Triple::EABI:
        // EABI is always AAPCS, and if it was not marked 'hard', it's softfp
        ABI = FloatABI::SoftFP;
        break;
      case llvm::Triple::Android:
        ABI = (SubArch == 7) ? FloatABI::SoftFP : FloatABI::Soft;
        break;
      default:
        // Assume "soft", but warn the user we are guessing.
        if (Triple.isOSBinFormatMachO() &&
            Triple.getSubArch() == llvm::Triple::ARMSubArch_v7em)
          ABI = FloatABI::Hard;
        else
          ABI = FloatABI::Soft;

        if (Triple.getOS() != llvm::Triple::UnknownOS ||
            !Triple.isOSBinFormatMachO())
          D.Diag(diag::warn_drv_assuming_mfloat_abi_is) << "soft";
        break;
      }
    }
  }

  assert(ABI != FloatABI::Invalid && "must select an ABI");
  return ABI;
}

static void getARMTargetFeatures(const ToolChain &TC,
                                 const llvm::Triple &Triple,
                                 const ArgList &Args,
                                 std::vector<const char *> &Features,
                                 bool ForAS) {
  const Driver &D = TC.getDriver();

  bool KernelOrKext =
      Args.hasArg(options::OPT_mkernel, options::OPT_fapple_kext);
  arm::FloatABI ABI = arm::getARMFloatABI(TC, Args);
  const Arg *WaCPU = nullptr, *WaFPU = nullptr;
  const Arg *WaHDiv = nullptr, *WaArch = nullptr;

  if (!ForAS) {
    // FIXME: Note, this is a hack, the LLVM backend doesn't actually use these
    // yet (it uses the -mfloat-abi and -msoft-float options), and it is
    // stripped out by the ARM target. We should probably pass this a new
    // -target-option, which is handled by the -cc1/-cc1as invocation.
    //
    // FIXME2:  For consistency, it would be ideal if we set up the target
    // machine state the same when using the frontend or the assembler. We don't
    // currently do that for the assembler, we pass the options directly to the
    // backend and never even instantiate the frontend TargetInfo. If we did,
    // and used its handleTargetFeatures hook, then we could ensure the
    // assembler and the frontend behave the same.

    // Use software floating point operations?
    if (ABI == arm::FloatABI::Soft)
      Features.push_back("+soft-float");

    // Use software floating point argument passing?
    if (ABI != arm::FloatABI::Hard)
      Features.push_back("+soft-float-abi");
  } else {
    // Here, we make sure that -Wa,-mfpu/cpu/arch/hwdiv will be passed down
    // to the assembler correctly.
    for (const Arg *A :
         Args.filtered(options::OPT_Wa_COMMA, options::OPT_Xassembler)) {
      StringRef Value = A->getValue();
      if (Value.startswith("-mfpu=")) {
        WaFPU = A;
      } else if (Value.startswith("-mcpu=")) {
        WaCPU = A;
      } else if (Value.startswith("-mhwdiv=")) {
        WaHDiv = A;
      } else if (Value.startswith("-march=")) {
        WaArch = A;
      }
    }
  }

  // Check -march. ClangAs gives preference to -Wa,-march=.
  const Arg *ArchArg = Args.getLastArg(options::OPT_march_EQ);
  StringRef ArchName;
  if (WaArch) {
    if (ArchArg)
      D.Diag(clang::diag::warn_drv_unused_argument)
          << ArchArg->getAsString(Args);
    ArchName = StringRef(WaArch->getValue()).substr(7);
    checkARMArchName(D, WaArch, Args, ArchName, Features, Triple);
    // FIXME: Set Arch.
    D.Diag(clang::diag::warn_drv_unused_argument) << WaArch->getAsString(Args);
  } else if (ArchArg) {
    ArchName = ArchArg->getValue();
    checkARMArchName(D, ArchArg, Args, ArchName, Features, Triple);
  }

  // Check -mcpu. ClangAs gives preference to -Wa,-mcpu=.
  const Arg *CPUArg = Args.getLastArg(options::OPT_mcpu_EQ);
  StringRef CPUName;
  if (WaCPU) {
    if (CPUArg)
      D.Diag(clang::diag::warn_drv_unused_argument)
          << CPUArg->getAsString(Args);
    CPUName = StringRef(WaCPU->getValue()).substr(6);
    checkARMCPUName(D, WaCPU, Args, CPUName, ArchName, Features, Triple);
  } else if (CPUArg) {
    CPUName = CPUArg->getValue();
    checkARMCPUName(D, CPUArg, Args, CPUName, ArchName, Features, Triple);
  }

  // Add CPU features for generic CPUs
  if (CPUName == "native") {
    llvm::StringMap<bool> HostFeatures;
    if (llvm::sys::getHostCPUFeatures(HostFeatures))
      for (auto &F : HostFeatures)
        Features.push_back(
            Args.MakeArgString((F.second ? "+" : "-") + F.first()));
  }

  // Honor -mfpu=. ClangAs gives preference to -Wa,-mfpu=.
  const Arg *FPUArg = Args.getLastArg(options::OPT_mfpu_EQ);
  if (WaFPU) {
    if (FPUArg)
      D.Diag(clang::diag::warn_drv_unused_argument)
          << FPUArg->getAsString(Args);
    getARMFPUFeatures(D, WaFPU, Args, StringRef(WaFPU->getValue()).substr(6),
                      Features);
  } else if (FPUArg) {
    getARMFPUFeatures(D, FPUArg, Args, FPUArg->getValue(), Features);
  }

  // Honor -mhwdiv=. ClangAs gives preference to -Wa,-mhwdiv=.
  const Arg *HDivArg = Args.getLastArg(options::OPT_mhwdiv_EQ);
  if (WaHDiv) {
    if (HDivArg)
      D.Diag(clang::diag::warn_drv_unused_argument)
          << HDivArg->getAsString(Args);
    getARMHWDivFeatures(D, WaHDiv, Args,
                        StringRef(WaHDiv->getValue()).substr(8), Features);
  } else if (HDivArg)
    getARMHWDivFeatures(D, HDivArg, Args, HDivArg->getValue(), Features);

  // Setting -msoft-float effectively disables NEON because of the GCC
  // implementation, although the same isn't true of VFP or VFP3.
  if (ABI == arm::FloatABI::Soft) {
    Features.push_back("-neon");
    // Also need to explicitly disable features which imply NEON.
    Features.push_back("-crypto");
  }

  // En/disable crc code generation.
  if (Arg *A = Args.getLastArg(options::OPT_mcrc, options::OPT_mnocrc)) {
    if (A->getOption().matches(options::OPT_mcrc))
      Features.push_back("+crc");
    else
      Features.push_back("-crc");
  }

  // Look for the last occurrence of -mlong-calls or -mno-long-calls. If
  // neither options are specified, see if we are compiling for kernel/kext and
  // decide whether to pass "+long-calls" based on the OS and its version.
  if (Arg *A = Args.getLastArg(options::OPT_mlong_calls,
                               options::OPT_mno_long_calls)) {
    if (A->getOption().matches(options::OPT_mlong_calls))
      Features.push_back("+long-calls");
  } else if (KernelOrKext && (!Triple.isiOS() || Triple.isOSVersionLT(6)) &&
             !Triple.isWatchOS()) {
      Features.push_back("+long-calls");
  }

  // Kernel code has more strict alignment requirements.
  if (KernelOrKext)
    Features.push_back("+strict-align");
  else if (Arg *A = Args.getLastArg(options::OPT_mno_unaligned_access,
                                    options::OPT_munaligned_access)) {
    if (A->getOption().matches(options::OPT_munaligned_access)) {
      // No v6M core supports unaligned memory access (v6M ARM ARM A3.2).
      if (Triple.getSubArch() == llvm::Triple::SubArchType::ARMSubArch_v6m)
        D.Diag(diag::err_target_unsupported_unaligned) << "v6m";
      // v8M Baseline follows on from v6M, so doesn't support unaligned memory
      // access either.
      else if (Triple.getSubArch() == llvm::Triple::SubArchType::ARMSubArch_v8m_baseline)
        D.Diag(diag::err_target_unsupported_unaligned) << "v8m.base";
    } else
      Features.push_back("+strict-align");
  } else {
    // Assume pre-ARMv6 doesn't support unaligned accesses.
    //
    // ARMv6 may or may not support unaligned accesses depending on the
    // SCTLR.U bit, which is architecture-specific. We assume ARMv6
    // Darwin and NetBSD targets support unaligned accesses, and others don't.
    //
    // ARMv7 always has SCTLR.U set to 1, but it has a new SCTLR.A bit
    // which raises an alignment fault on unaligned accesses. Linux
    // defaults this bit to 0 and handles it as a system-wide (not
    // per-process) setting. It is therefore safe to assume that ARMv7+
    // Linux targets support unaligned accesses. The same goes for NaCl.
    //
    // The above behavior is consistent with GCC.
    int VersionNum = getARMSubArchVersionNumber(Triple);
    if (Triple.isOSDarwin() || Triple.isOSNetBSD()) {
      if (VersionNum < 6 ||
          Triple.getSubArch() == llvm::Triple::SubArchType::ARMSubArch_v6m)
        Features.push_back("+strict-align");
    } else if (Triple.isOSLinux() || Triple.isOSNaCl()) {
      if (VersionNum < 7)
        Features.push_back("+strict-align");
    } else
      Features.push_back("+strict-align");
  }

  // llvm does not support reserving registers in general. There is support
  // for reserving r9 on ARM though (defined as a platform-specific register
  // in ARM EABI).
  if (Args.hasArg(options::OPT_ffixed_r9))
    Features.push_back("+reserve-r9");

  // The kext linker doesn't know how to deal with movw/movt.
  if (KernelOrKext || Args.hasArg(options::OPT_mno_movt))
    Features.push_back("+no-movt");
}

void Clang::AddARMTargetArgs(const llvm::Triple &Triple, const ArgList &Args,
                             ArgStringList &CmdArgs, bool KernelOrKext) const {
  // Select the ABI to use.
  // FIXME: Support -meabi.
  // FIXME: Parts of this are duplicated in the backend, unify this somehow.
  const char *ABIName = nullptr;
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ)) {
    ABIName = A->getValue();
  } else if (Triple.isOSBinFormatMachO()) {
    if (useAAPCSForMachO(Triple)) {
      ABIName = "aapcs";
    } else if (Triple.isWatchABI()) {
      ABIName = "aapcs16";
    } else {
      ABIName = "apcs-gnu";
    }
  } else if (Triple.isOSWindows()) {
    // FIXME: this is invalid for WindowsCE
    ABIName = "aapcs";
  } else {
    // Select the default based on the platform.
    switch (Triple.getEnvironment()) {
    case llvm::Triple::Android:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::GNUEABIHF:
    case llvm::Triple::MuslEABI:
    case llvm::Triple::MuslEABIHF:
      ABIName = "aapcs-linux";
      break;
    case llvm::Triple::EABIHF:
    case llvm::Triple::EABI:
      ABIName = "aapcs";
      break;
    default:
      if (Triple.getOS() == llvm::Triple::NetBSD)
        ABIName = "apcs-gnu";
      else
        ABIName = "aapcs";
      break;
    }
  }
  CmdArgs.push_back("-target-abi");
  CmdArgs.push_back(ABIName);

  // Determine floating point ABI from the options & target defaults.
  arm::FloatABI ABI = arm::getARMFloatABI(getToolChain(), Args);
  if (ABI == arm::FloatABI::Soft) {
    // Floating point operations and argument passing are soft.
    // FIXME: This changes CPP defines, we need -target-soft-float.
    CmdArgs.push_back("-msoft-float");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else if (ABI == arm::FloatABI::SoftFP) {
    // Floating point operations are hard, but argument passing is soft.
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else {
    // Floating point operations and argument passing are hard.
    assert(ABI == arm::FloatABI::Hard && "Invalid float abi!");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("hard");
  }

  // Forward the -mglobal-merge option for explicit control over the pass.
  if (Arg *A = Args.getLastArg(options::OPT_mglobal_merge,
                               options::OPT_mno_global_merge)) {
    CmdArgs.push_back("-backend-option");
    if (A->getOption().matches(options::OPT_mno_global_merge))
      CmdArgs.push_back("-arm-global-merge=false");
    else
      CmdArgs.push_back("-arm-global-merge=true");
  }

  if (!Args.hasFlag(options::OPT_mimplicit_float,
                    options::OPT_mno_implicit_float, true))
    CmdArgs.push_back("-no-implicit-float");
}
// ARM tools end.

/// getAArch64TargetCPU - Get the (LLVM) name of the AArch64 cpu we are
/// targeting. Set \p A to the Arg corresponding to the -mcpu or -mtune
/// arguments if they are provided, or to nullptr otherwise.
static std::string getAArch64TargetCPU(const ArgList &Args, Arg *&A) {
  std::string CPU;
  // If we have -mtune or -mcpu, use that.
  if ((A = Args.getLastArg(options::OPT_mtune_EQ))) {
    CPU = StringRef(A->getValue()).lower();
  } else if ((A = Args.getLastArg(options::OPT_mcpu_EQ))) {
    StringRef Mcpu = A->getValue();
    CPU = Mcpu.split("+").first.lower();
  }

  // Handle CPU name is 'native'.
  if (CPU == "native")
    return llvm::sys::getHostCPUName();
  else if (CPU.size())
    return CPU;

  // Make sure we pick "cyclone" if -arch is used.
  // FIXME: Should this be picked by checking the target triple instead?
  if (Args.getLastArg(options::OPT_arch))
    return "cyclone";

  return "generic";
}

void Clang::AddAArch64TargetArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();

  if (!Args.hasFlag(options::OPT_mred_zone, options::OPT_mno_red_zone, true) ||
      Args.hasArg(options::OPT_mkernel) ||
      Args.hasArg(options::OPT_fapple_kext))
    CmdArgs.push_back("-disable-red-zone");

  if (!Args.hasFlag(options::OPT_mimplicit_float,
                    options::OPT_mno_implicit_float, true))
    CmdArgs.push_back("-no-implicit-float");

  const char *ABIName = nullptr;
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    ABIName = A->getValue();
  else if (Triple.isOSDarwin())
    ABIName = "darwinpcs";
  else
    ABIName = "aapcs";

  CmdArgs.push_back("-target-abi");
  CmdArgs.push_back(ABIName);

  if (Arg *A = Args.getLastArg(options::OPT_mfix_cortex_a53_835769,
                               options::OPT_mno_fix_cortex_a53_835769)) {
    CmdArgs.push_back("-backend-option");
    if (A->getOption().matches(options::OPT_mfix_cortex_a53_835769))
      CmdArgs.push_back("-aarch64-fix-cortex-a53-835769=1");
    else
      CmdArgs.push_back("-aarch64-fix-cortex-a53-835769=0");
  } else if (Triple.isAndroid()) {
    // Enabled A53 errata (835769) workaround by default on android
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-aarch64-fix-cortex-a53-835769=1");
  }

  // Forward the -mglobal-merge option for explicit control over the pass.
  if (Arg *A = Args.getLastArg(options::OPT_mglobal_merge,
                               options::OPT_mno_global_merge)) {
    CmdArgs.push_back("-backend-option");
    if (A->getOption().matches(options::OPT_mno_global_merge))
      CmdArgs.push_back("-aarch64-global-merge=false");
    else
      CmdArgs.push_back("-aarch64-global-merge=true");
  }
}

// Get CPU and ABI names. They are not independent
// so we have to calculate them together.
void mips::getMipsCPUAndABI(const ArgList &Args, const llvm::Triple &Triple,
                            StringRef &CPUName, StringRef &ABIName) {
  const char *DefMips32CPU = "mips32r2";
  const char *DefMips64CPU = "mips64r2";

  // MIPS32r6 is the default for mips(el)?-img-linux-gnu and MIPS64r6 is the
  // default for mips64(el)?-img-linux-gnu.
  if (Triple.getVendor() == llvm::Triple::ImaginationTechnologies &&
      Triple.getEnvironment() == llvm::Triple::GNU) {
    DefMips32CPU = "mips32r6";
    DefMips64CPU = "mips64r6";
  }

  // MIPS64r6 is the default for Android MIPS64 (mips64el-linux-android).
  if (Triple.isAndroid()) {
    DefMips32CPU = "mips32";
    DefMips64CPU = "mips64r6";
  }

  // MIPS3 is the default for mips64*-unknown-openbsd.
  if (Triple.getOS() == llvm::Triple::OpenBSD)
    DefMips64CPU = "mips3";

  if (Arg *A = Args.getLastArg(options::OPT_march_EQ, options::OPT_mcpu_EQ))
    CPUName = A->getValue();

  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ)) {
    ABIName = A->getValue();
    // Convert a GNU style Mips ABI name to the name
    // accepted by LLVM Mips backend.
    ABIName = llvm::StringSwitch<llvm::StringRef>(ABIName)
                  .Case("32", "o32")
                  .Case("64", "n64")
                  .Default(ABIName);
  }

  // Setup default CPU and ABI names.
  if (CPUName.empty() && ABIName.empty()) {
    switch (Triple.getArch()) {
    default:
      llvm_unreachable("Unexpected triple arch name");
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
      CPUName = DefMips32CPU;
      break;
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
      CPUName = DefMips64CPU;
      break;
    }
  }

  if (ABIName.empty() &&
      (Triple.getVendor() == llvm::Triple::MipsTechnologies ||
       Triple.getVendor() == llvm::Triple::ImaginationTechnologies)) {
    ABIName = llvm::StringSwitch<const char *>(CPUName)
                  .Case("mips1", "o32")
                  .Case("mips2", "o32")
                  .Case("mips3", "n64")
                  .Case("mips4", "n64")
                  .Case("mips5", "n64")
                  .Case("mips32", "o32")
                  .Case("mips32r2", "o32")
                  .Case("mips32r3", "o32")
                  .Case("mips32r5", "o32")
                  .Case("mips32r6", "o32")
                  .Case("mips64", "n64")
                  .Case("mips64r2", "n64")
                  .Case("mips64r3", "n64")
                  .Case("mips64r5", "n64")
                  .Case("mips64r6", "n64")
                  .Case("octeon", "n64")
                  .Case("p5600", "o32")
                  .Default("");
  }

  if (ABIName.empty()) {
    // Deduce ABI name from the target triple.
    if (Triple.getArch() == llvm::Triple::mips ||
        Triple.getArch() == llvm::Triple::mipsel)
      ABIName = "o32";
    else
      ABIName = "n64";
  }

  if (CPUName.empty()) {
    // Deduce CPU name from ABI name.
    CPUName = llvm::StringSwitch<const char *>(ABIName)
                  .Case("o32", DefMips32CPU)
                  .Cases("n32", "n64", DefMips64CPU)
                  .Default("");
  }

  // FIXME: Warn on inconsistent use of -march and -mabi.
}

std::string mips::getMipsABILibSuffix(const ArgList &Args,
                                      const llvm::Triple &Triple) {
  StringRef CPUName, ABIName;
  tools::mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);
  return llvm::StringSwitch<std::string>(ABIName)
      .Case("o32", "")
      .Case("n32", "32")
      .Case("n64", "64");
}

// Convert ABI name to the GNU tools acceptable variant.
static StringRef getGnuCompatibleMipsABIName(StringRef ABI) {
  return llvm::StringSwitch<llvm::StringRef>(ABI)
      .Case("o32", "32")
      .Case("n64", "64")
      .Default(ABI);
}

// Select the MIPS float ABI as determined by -msoft-float, -mhard-float,
// and -mfloat-abi=.
static mips::FloatABI getMipsFloatABI(const Driver &D, const ArgList &Args) {
  mips::FloatABI ABI = mips::FloatABI::Invalid;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float,
                          options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float))
      ABI = mips::FloatABI::Soft;
    else if (A->getOption().matches(options::OPT_mhard_float))
      ABI = mips::FloatABI::Hard;
    else {
      ABI = llvm::StringSwitch<mips::FloatABI>(A->getValue())
                .Case("soft", mips::FloatABI::Soft)
                .Case("hard", mips::FloatABI::Hard)
                .Default(mips::FloatABI::Invalid);
      if (ABI == mips::FloatABI::Invalid && !StringRef(A->getValue()).empty()) {
        D.Diag(diag::err_drv_invalid_mfloat_abi) << A->getAsString(Args);
        ABI = mips::FloatABI::Hard;
      }
    }
  }

  // If unspecified, choose the default based on the platform.
  if (ABI == mips::FloatABI::Invalid) {
    // Assume "hard", because it's a default value used by gcc.
    // When we start to recognize specific target MIPS processors,
    // we will be able to select the default more correctly.
    ABI = mips::FloatABI::Hard;
  }

  assert(ABI != mips::FloatABI::Invalid && "must select an ABI");
  return ABI;
}

static void AddTargetFeature(const ArgList &Args,
                             std::vector<const char *> &Features,
                             OptSpecifier OnOpt, OptSpecifier OffOpt,
                             StringRef FeatureName) {
  if (Arg *A = Args.getLastArg(OnOpt, OffOpt)) {
    if (A->getOption().matches(OnOpt))
      Features.push_back(Args.MakeArgString("+" + FeatureName));
    else
      Features.push_back(Args.MakeArgString("-" + FeatureName));
  }
}

static void getMIPSTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                  const ArgList &Args,
                                  std::vector<const char *> &Features) {
  StringRef CPUName;
  StringRef ABIName;
  mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);
  ABIName = getGnuCompatibleMipsABIName(ABIName);

  AddTargetFeature(Args, Features, options::OPT_mno_abicalls,
                   options::OPT_mabicalls, "noabicalls");

  mips::FloatABI FloatABI = getMipsFloatABI(D, Args);
  if (FloatABI == mips::FloatABI::Soft) {
    // FIXME: Note, this is a hack. We need to pass the selected float
    // mode to the MipsTargetInfoBase to define appropriate macros there.
    // Now it is the only method.
    Features.push_back("+soft-float");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mnan_EQ)) {
    StringRef Val = StringRef(A->getValue());
    if (Val == "2008") {
      if (mips::getSupportedNanEncoding(CPUName) & mips::Nan2008)
        Features.push_back("+nan2008");
      else {
        Features.push_back("-nan2008");
        D.Diag(diag::warn_target_unsupported_nan2008) << CPUName;
      }
    } else if (Val == "legacy") {
      if (mips::getSupportedNanEncoding(CPUName) & mips::NanLegacy)
        Features.push_back("-nan2008");
      else {
        Features.push_back("+nan2008");
        D.Diag(diag::warn_target_unsupported_nanlegacy) << CPUName;
      }
    } else
      D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Val;
  }

  AddTargetFeature(Args, Features, options::OPT_msingle_float,
                   options::OPT_mdouble_float, "single-float");
  AddTargetFeature(Args, Features, options::OPT_mips16, options::OPT_mno_mips16,
                   "mips16");
  AddTargetFeature(Args, Features, options::OPT_mmicromips,
                   options::OPT_mno_micromips, "micromips");
  AddTargetFeature(Args, Features, options::OPT_mdsp, options::OPT_mno_dsp,
                   "dsp");
  AddTargetFeature(Args, Features, options::OPT_mdspr2, options::OPT_mno_dspr2,
                   "dspr2");
  AddTargetFeature(Args, Features, options::OPT_mmsa, options::OPT_mno_msa,
                   "msa");

  // Add the last -mfp32/-mfpxx/-mfp64, if none are given and the ABI is O32
  // pass -mfpxx, or if none are given and fp64a is default, pass fp64 and
  // nooddspreg.
  if (Arg *A = Args.getLastArg(options::OPT_mfp32, options::OPT_mfpxx,
                               options::OPT_mfp64)) {
    if (A->getOption().matches(options::OPT_mfp32))
      Features.push_back(Args.MakeArgString("-fp64"));
    else if (A->getOption().matches(options::OPT_mfpxx)) {
      Features.push_back(Args.MakeArgString("+fpxx"));
      Features.push_back(Args.MakeArgString("+nooddspreg"));
    } else
      Features.push_back(Args.MakeArgString("+fp64"));
  } else if (mips::shouldUseFPXX(Args, Triple, CPUName, ABIName, FloatABI)) {
    Features.push_back(Args.MakeArgString("+fpxx"));
    Features.push_back(Args.MakeArgString("+nooddspreg"));
  } else if (mips::isFP64ADefault(Triple, CPUName)) {
    Features.push_back(Args.MakeArgString("+fp64"));
    Features.push_back(Args.MakeArgString("+nooddspreg"));
  }

  AddTargetFeature(Args, Features, options::OPT_mno_odd_spreg,
                   options::OPT_modd_spreg, "nooddspreg");
}

void Clang::AddMIPSTargetArgs(const ArgList &Args,
                              ArgStringList &CmdArgs) const {
  const Driver &D = getToolChain().getDriver();
  StringRef CPUName;
  StringRef ABIName;
  const llvm::Triple &Triple = getToolChain().getTriple();
  mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);

  CmdArgs.push_back("-target-abi");
  CmdArgs.push_back(ABIName.data());

  mips::FloatABI ABI = getMipsFloatABI(D, Args);
  if (ABI == mips::FloatABI::Soft) {
    // Floating point operations and argument passing are soft.
    CmdArgs.push_back("-msoft-float");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else {
    // Floating point operations and argument passing are hard.
    assert(ABI == mips::FloatABI::Hard && "Invalid float abi!");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("hard");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mxgot, options::OPT_mno_xgot)) {
    if (A->getOption().matches(options::OPT_mxgot)) {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back("-mxgot");
    }
  }

  if (Arg *A = Args.getLastArg(options::OPT_mldc1_sdc1,
                               options::OPT_mno_ldc1_sdc1)) {
    if (A->getOption().matches(options::OPT_mno_ldc1_sdc1)) {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back("-mno-ldc1-sdc1");
    }
  }

  if (Arg *A = Args.getLastArg(options::OPT_mcheck_zero_division,
                               options::OPT_mno_check_zero_division)) {
    if (A->getOption().matches(options::OPT_mno_check_zero_division)) {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back("-mno-check-zero-division");
    }
  }

  if (Arg *A = Args.getLastArg(options::OPT_G)) {
    StringRef v = A->getValue();
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back(Args.MakeArgString("-mips-ssection-threshold=" + v));
    A->claim();
  }

  if (Arg *A = Args.getLastArg(options::OPT_mcompact_branches_EQ)) {
    StringRef Val = StringRef(A->getValue());
    if (mips::hasCompactBranches(CPUName)) {
      if (Val == "never" || Val == "always" || Val == "optimal") {
        CmdArgs.push_back("-mllvm");
        CmdArgs.push_back(Args.MakeArgString("-mips-compact-branches=" + Val));
      } else
        D.Diag(diag::err_drv_unsupported_option_argument)
            << A->getOption().getName() << Val;
    } else
      D.Diag(diag::warn_target_unsupported_compact_branches) << CPUName;
  }
}

/// getPPCTargetCPU - Get the (LLVM) name of the PowerPC cpu we are targeting.
static std::string getPPCTargetCPU(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    StringRef CPUName = A->getValue();

    if (CPUName == "native") {
      std::string CPU = llvm::sys::getHostCPUName();
      if (!CPU.empty() && CPU != "generic")
        return CPU;
      else
        return "";
    }

    return llvm::StringSwitch<const char *>(CPUName)
        .Case("common", "generic")
        .Case("440", "440")
        .Case("440fp", "440")
        .Case("450", "450")
        .Case("601", "601")
        .Case("602", "602")
        .Case("603", "603")
        .Case("603e", "603e")
        .Case("603ev", "603ev")
        .Case("604", "604")
        .Case("604e", "604e")
        .Case("620", "620")
        .Case("630", "pwr3")
        .Case("G3", "g3")
        .Case("7400", "7400")
        .Case("G4", "g4")
        .Case("7450", "7450")
        .Case("G4+", "g4+")
        .Case("750", "750")
        .Case("970", "970")
        .Case("G5", "g5")
        .Case("a2", "a2")
        .Case("a2q", "a2q")
        .Case("e500mc", "e500mc")
        .Case("e5500", "e5500")
        .Case("power3", "pwr3")
        .Case("power4", "pwr4")
        .Case("power5", "pwr5")
        .Case("power5x", "pwr5x")
        .Case("power6", "pwr6")
        .Case("power6x", "pwr6x")
        .Case("power7", "pwr7")
        .Case("power8", "pwr8")
        .Case("power9", "pwr9")
        .Case("pwr3", "pwr3")
        .Case("pwr4", "pwr4")
        .Case("pwr5", "pwr5")
        .Case("pwr5x", "pwr5x")
        .Case("pwr6", "pwr6")
        .Case("pwr6x", "pwr6x")
        .Case("pwr7", "pwr7")
        .Case("pwr8", "pwr8")
        .Case("pwr9", "pwr9")
        .Case("powerpc", "ppc")
        .Case("powerpc64", "ppc64")
        .Case("powerpc64le", "ppc64le")
        .Default("");
  }

  return "";
}

static void getPPCTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args,
                                 std::vector<const char *> &Features) {
  handleTargetFeaturesGroup(Args, Features, options::OPT_m_ppc_Features_Group);

  ppc::FloatABI FloatABI = ppc::getPPCFloatABI(D, Args);
  if (FloatABI == ppc::FloatABI::Soft &&
      !(Triple.getArch() == llvm::Triple::ppc64 ||
        Triple.getArch() == llvm::Triple::ppc64le))
    Features.push_back("+soft-float");
  else if (FloatABI == ppc::FloatABI::Soft &&
           (Triple.getArch() == llvm::Triple::ppc64 ||
            Triple.getArch() == llvm::Triple::ppc64le))
    D.Diag(diag::err_drv_invalid_mfloat_abi)
        << "soft float is not supported for ppc64";

  // Altivec is a bit weird, allow overriding of the Altivec feature here.
  AddTargetFeature(Args, Features, options::OPT_faltivec,
                   options::OPT_fno_altivec, "altivec");
}

ppc::FloatABI ppc::getPPCFloatABI(const Driver &D, const ArgList &Args) {
  ppc::FloatABI ABI = ppc::FloatABI::Invalid;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float,
                          options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float))
      ABI = ppc::FloatABI::Soft;
    else if (A->getOption().matches(options::OPT_mhard_float))
      ABI = ppc::FloatABI::Hard;
    else {
      ABI = llvm::StringSwitch<ppc::FloatABI>(A->getValue())
                .Case("soft", ppc::FloatABI::Soft)
                .Case("hard", ppc::FloatABI::Hard)
                .Default(ppc::FloatABI::Invalid);
      if (ABI == ppc::FloatABI::Invalid && !StringRef(A->getValue()).empty()) {
        D.Diag(diag::err_drv_invalid_mfloat_abi) << A->getAsString(Args);
        ABI = ppc::FloatABI::Hard;
      }
    }
  }

  // If unspecified, choose the default based on the platform.
  if (ABI == ppc::FloatABI::Invalid) {
    ABI = ppc::FloatABI::Hard;
  }

  return ABI;
}

void Clang::AddPPCTargetArgs(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  // Select the ABI to use.
  const char *ABIName = nullptr;
  if (getToolChain().getTriple().isOSLinux())
    switch (getToolChain().getArch()) {
    case llvm::Triple::ppc64: {
      // When targeting a processor that supports QPX, or if QPX is
      // specifically enabled, default to using the ABI that supports QPX (so
      // long as it is not specifically disabled).
      bool HasQPX = false;
      if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
        HasQPX = A->getValue() == StringRef("a2q");
      HasQPX = Args.hasFlag(options::OPT_mqpx, options::OPT_mno_qpx, HasQPX);
      if (HasQPX) {
        ABIName = "elfv1-qpx";
        break;
      }

      ABIName = "elfv1";
      break;
    }
    case llvm::Triple::ppc64le:
      ABIName = "elfv2";
      break;
    default:
      break;
    }

  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ))
    // The ppc64 linux abis are all "altivec" abis by default. Accept and ignore
    // the option if given as we don't have backend support for any targets
    // that don't use the altivec abi.
    if (StringRef(A->getValue()) != "altivec")
      ABIName = A->getValue();

  ppc::FloatABI FloatABI =
      ppc::getPPCFloatABI(getToolChain().getDriver(), Args);

  if (FloatABI == ppc::FloatABI::Soft) {
    // Floating point operations and argument passing are soft.
    CmdArgs.push_back("-msoft-float");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else {
    // Floating point operations and argument passing are hard.
    assert(FloatABI == ppc::FloatABI::Hard && "Invalid float abi!");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("hard");
  }

  if (ABIName) {
    CmdArgs.push_back("-target-abi");
    CmdArgs.push_back(ABIName);
  }
}

bool ppc::hasPPCAbiArg(const ArgList &Args, const char *Value) {
  Arg *A = Args.getLastArg(options::OPT_mabi_EQ);
  return A && (A->getValue() == StringRef(Value));
}

/// Get the (LLVM) name of the R600 gpu we are targeting.
static std::string getR600TargetGPU(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    const char *GPUName = A->getValue();
    return llvm::StringSwitch<const char *>(GPUName)
        .Cases("rv630", "rv635", "r600")
        .Cases("rv610", "rv620", "rs780", "rs880")
        .Case("rv740", "rv770")
        .Case("palm", "cedar")
        .Cases("sumo", "sumo2", "sumo")
        .Case("hemlock", "cypress")
        .Case("aruba", "cayman")
        .Default(GPUName);
  }
  return "";
}

static std::string getLanaiTargetCPU(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    return A->getValue();
  }
  return "";
}

sparc::FloatABI sparc::getSparcFloatABI(const Driver &D,
                                        const ArgList &Args) {
  sparc::FloatABI ABI = sparc::FloatABI::Invalid;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float,
                          options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float))
      ABI = sparc::FloatABI::Soft;
    else if (A->getOption().matches(options::OPT_mhard_float))
      ABI = sparc::FloatABI::Hard;
    else {
      ABI = llvm::StringSwitch<sparc::FloatABI>(A->getValue())
                .Case("soft", sparc::FloatABI::Soft)
                .Case("hard", sparc::FloatABI::Hard)
                .Default(sparc::FloatABI::Invalid);
      if (ABI == sparc::FloatABI::Invalid &&
          !StringRef(A->getValue()).empty()) {
        D.Diag(diag::err_drv_invalid_mfloat_abi) << A->getAsString(Args);
        ABI = sparc::FloatABI::Hard;
      }
    }
  }

  // If unspecified, choose the default based on the platform.
  // Only the hard-float ABI on Sparc is standardized, and it is the
  // default. GCC also supports a nonstandard soft-float ABI mode, also
  // implemented in LLVM. However as this is not standard we set the default
  // to be hard-float.
  if (ABI == sparc::FloatABI::Invalid) {
    ABI = sparc::FloatABI::Hard;
  }

  return ABI;
}

static void getSparcTargetFeatures(const Driver &D, const ArgList &Args,
                                 std::vector<const char *> &Features) {
  sparc::FloatABI FloatABI = sparc::getSparcFloatABI(D, Args);
  if (FloatABI == sparc::FloatABI::Soft)
    Features.push_back("+soft-float");
}

void Clang::AddSparcTargetArgs(const ArgList &Args,
                               ArgStringList &CmdArgs) const {
  sparc::FloatABI FloatABI =
      sparc::getSparcFloatABI(getToolChain().getDriver(), Args);

  if (FloatABI == sparc::FloatABI::Soft) {
    // Floating point operations and argument passing are soft.
    CmdArgs.push_back("-msoft-float");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else {
    // Floating point operations and argument passing are hard.
    assert(FloatABI == sparc::FloatABI::Hard && "Invalid float abi!");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("hard");
  }
}

void Clang::AddSystemZTargetArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  if (Args.hasFlag(options::OPT_mbackchain, options::OPT_mno_backchain, false))
    CmdArgs.push_back("-mbackchain");
}

static const char *getSystemZTargetCPU(const ArgList &Args) {
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
    return A->getValue();
  return "z10";
}

static void getSystemZTargetFeatures(const ArgList &Args,
                                     std::vector<const char *> &Features) {
  // -m(no-)htm overrides use of the transactional-execution facility.
  if (Arg *A = Args.getLastArg(options::OPT_mhtm, options::OPT_mno_htm)) {
    if (A->getOption().matches(options::OPT_mhtm))
      Features.push_back("+transactional-execution");
    else
      Features.push_back("-transactional-execution");
  }
  // -m(no-)vx overrides use of the vector facility.
  if (Arg *A = Args.getLastArg(options::OPT_mvx, options::OPT_mno_vx)) {
    if (A->getOption().matches(options::OPT_mvx))
      Features.push_back("+vector");
    else
      Features.push_back("-vector");
  }
}

static const char *getX86TargetCPU(const ArgList &Args,
                                   const llvm::Triple &Triple) {
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    if (StringRef(A->getValue()) != "native") {
      if (Triple.isOSDarwin() && Triple.getArchName() == "x86_64h")
        return "core-avx2";

      return A->getValue();
    }

    // FIXME: Reject attempts to use -march=native unless the target matches
    // the host.
    //
    // FIXME: We should also incorporate the detected target features for use
    // with -native.
    std::string CPU = llvm::sys::getHostCPUName();
    if (!CPU.empty() && CPU != "generic")
      return Args.MakeArgString(CPU);
  }

  if (const Arg *A = Args.getLastArg(options::OPT__SLASH_arch)) {
    // Mapping built by referring to X86TargetInfo::getDefaultFeatures().
    StringRef Arch = A->getValue();
    const char *CPU;
    if (Triple.getArch() == llvm::Triple::x86) {
      CPU = llvm::StringSwitch<const char *>(Arch)
                .Case("IA32", "i386")
                .Case("SSE", "pentium3")
                .Case("SSE2", "pentium4")
                .Case("AVX", "sandybridge")
                .Case("AVX2", "haswell")
                .Default(nullptr);
    } else {
      CPU = llvm::StringSwitch<const char *>(Arch)
                .Case("AVX", "sandybridge")
                .Case("AVX2", "haswell")
                .Default(nullptr);
    }
    if (CPU)
      return CPU;
  }

  // Select the default CPU if none was given (or detection failed).

  if (Triple.getArch() != llvm::Triple::x86_64 &&
      Triple.getArch() != llvm::Triple::x86)
    return nullptr; // This routine is only handling x86 targets.

  bool Is64Bit = Triple.getArch() == llvm::Triple::x86_64;

  // FIXME: Need target hooks.
  if (Triple.isOSDarwin()) {
    if (Triple.getArchName() == "x86_64h")
      return "core-avx2";
    return Is64Bit ? "core2" : "yonah";
  }

  // Set up default CPU name for PS4 compilers.
  if (Triple.isPS4CPU())
    return "btver2";

  // On Android use targets compatible with gcc
  if (Triple.isAndroid())
    return Is64Bit ? "x86-64" : "i686";

  // Everything else goes to x86-64 in 64-bit mode.
  if (Is64Bit)
    return "x86-64";

  switch (Triple.getOS()) {
  case llvm::Triple::FreeBSD:
  case llvm::Triple::NetBSD:
  case llvm::Triple::OpenBSD:
    return "i486";
  case llvm::Triple::Haiku:
    return "i586";
  case llvm::Triple::Bitrig:
    return "i686";
  default:
    // Fallback to p4.
    return "pentium4";
  }
}

/// Get the (LLVM) name of the WebAssembly cpu we are targeting.
static StringRef getWebAssemblyTargetCPU(const ArgList &Args) {
  // If we have -mcpu=, use that.
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    StringRef CPU = A->getValue();

#ifdef __wasm__
    // Handle "native" by examining the host. "native" isn't meaningful when
    // cross compiling, so only support this when the host is also WebAssembly.
    if (CPU == "native")
      return llvm::sys::getHostCPUName();
#endif

    return CPU;
  }

  return "generic";
}

static std::string getCPUName(const ArgList &Args, const llvm::Triple &T,
                              bool FromAs = false) {
  Arg *A;

  switch (T.getArch()) {
  default:
    return "";

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    return getAArch64TargetCPU(Args, A);

  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb: {
    StringRef MArch, MCPU;
    getARMArchCPUFromArgs(Args, MArch, MCPU, FromAs);
    return arm::getARMTargetCPU(MCPU, MArch, T);
  }
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, T, CPUName, ABIName);
    return CPUName;
  }

  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
      return A->getValue();
    return "";

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le: {
    std::string TargetCPUName = getPPCTargetCPU(Args);
    // LLVM may default to generating code for the native CPU,
    // but, like gcc, we default to a more generic option for
    // each architecture. (except on Darwin)
    if (TargetCPUName.empty() && !T.isOSDarwin()) {
      if (T.getArch() == llvm::Triple::ppc64)
        TargetCPUName = "ppc64";
      else if (T.getArch() == llvm::Triple::ppc64le)
        TargetCPUName = "ppc64le";
      else
        TargetCPUName = "ppc";
    }
    return TargetCPUName;
  }

  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::sparcv9:
    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      return A->getValue();
    return "";

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return getX86TargetCPU(Args, T);

  case llvm::Triple::hexagon:
    return "hexagon" +
           toolchains::HexagonToolChain::GetTargetCPUVersion(Args).str();

  case llvm::Triple::lanai:
    return getLanaiTargetCPU(Args);

  case llvm::Triple::systemz:
    return getSystemZTargetCPU(Args);

  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
    return getR600TargetGPU(Args);

  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    return getWebAssemblyTargetCPU(Args);
  }
}

static unsigned getLTOParallelism(const ArgList &Args, const Driver &D) {
  unsigned Parallelism = 0;
  Arg *LtoJobsArg = Args.getLastArg(options::OPT_flto_jobs_EQ);
  if (LtoJobsArg &&
      StringRef(LtoJobsArg->getValue()).getAsInteger(10, Parallelism))
    D.Diag(diag::err_drv_invalid_int_value) << LtoJobsArg->getAsString(Args)
                                            << LtoJobsArg->getValue();
  return Parallelism;
}

static void AddGoldPlugin(const ToolChain &ToolChain, const ArgList &Args,
                          ArgStringList &CmdArgs, bool IsThinLTO,
                          const Driver &D) {
  // Tell the linker to load the plugin. This has to come before AddLinkerInputs
  // as gold requires -plugin to come before any -plugin-opt that -Wl might
  // forward.
  CmdArgs.push_back("-plugin");
  std::string Plugin =
      ToolChain.getDriver().Dir + "/../lib" CLANG_LIBDIR_SUFFIX "/LLVMgold.so";
  CmdArgs.push_back(Args.MakeArgString(Plugin));

  // Try to pass driver level flags relevant to LTO code generation down to
  // the plugin.

  // Handle flags for selecting CPU variants.
  std::string CPU = getCPUName(Args, ToolChain.getTriple());
  if (!CPU.empty())
    CmdArgs.push_back(Args.MakeArgString(Twine("-plugin-opt=mcpu=") + CPU));

  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    StringRef OOpt;
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O))
      OOpt = A->getValue();
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    if (!OOpt.empty())
      CmdArgs.push_back(Args.MakeArgString(Twine("-plugin-opt=O") + OOpt));
  }

  if (IsThinLTO)
    CmdArgs.push_back("-plugin-opt=thinlto");

  if (unsigned Parallelism = getLTOParallelism(Args, D))
    CmdArgs.push_back(Args.MakeArgString(Twine("-plugin-opt=jobs=") +
                                         llvm::to_string(Parallelism)));

  // If an explicit debugger tuning argument appeared, pass it along.
  if (Arg *A = Args.getLastArg(options::OPT_gTune_Group,
                               options::OPT_ggdbN_Group)) {
    if (A->getOption().matches(options::OPT_glldb))
      CmdArgs.push_back("-plugin-opt=-debugger-tune=lldb");
    else if (A->getOption().matches(options::OPT_gsce))
      CmdArgs.push_back("-plugin-opt=-debugger-tune=sce");
    else
      CmdArgs.push_back("-plugin-opt=-debugger-tune=gdb");
  }
}

/// This is a helper function for validating the optional refinement step
/// parameter in reciprocal argument strings. Return false if there is an error
/// parsing the refinement step. Otherwise, return true and set the Position
/// of the refinement step in the input string.
static bool getRefinementStep(StringRef In, const Driver &D,
                              const Arg &A, size_t &Position) {
  const char RefinementStepToken = ':';
  Position = In.find(RefinementStepToken);
  if (Position != StringRef::npos) {
    StringRef Option = A.getOption().getName();
    StringRef RefStep = In.substr(Position + 1);
    // Allow exactly one numeric character for the additional refinement
    // step parameter. This is reasonable for all currently-supported
    // operations and architectures because we would expect that a larger value
    // of refinement steps would cause the estimate "optimization" to
    // under-perform the native operation. Also, if the estimate does not
    // converge quickly, it probably will not ever converge, so further
    // refinement steps will not produce a better answer.
    if (RefStep.size() != 1) {
      D.Diag(diag::err_drv_invalid_value) << Option << RefStep;
      return false;
    }
    char RefStepChar = RefStep[0];
    if (RefStepChar < '0' || RefStepChar > '9') {
      D.Diag(diag::err_drv_invalid_value) << Option << RefStep;
      return false;
    }
  }
  return true;
}

/// The -mrecip flag requires processing of many optional parameters.
static void ParseMRecip(const Driver &D, const ArgList &Args,
                        ArgStringList &OutStrings) {
  StringRef DisabledPrefixIn = "!";
  StringRef DisabledPrefixOut = "!";
  StringRef EnabledPrefixOut = "";
  StringRef Out = "-mrecip=";

  Arg *A = Args.getLastArg(options::OPT_mrecip, options::OPT_mrecip_EQ);
  if (!A)
    return;

  unsigned NumOptions = A->getNumValues();
  if (NumOptions == 0) {
    // No option is the same as "all".
    OutStrings.push_back(Args.MakeArgString(Out + "all"));
    return;
  }

  // Pass through "all", "none", or "default" with an optional refinement step.
  if (NumOptions == 1) {
    StringRef Val = A->getValue(0);
    size_t RefStepLoc;
    if (!getRefinementStep(Val, D, *A, RefStepLoc))
      return;
    StringRef ValBase = Val.slice(0, RefStepLoc);
    if (ValBase == "all" || ValBase == "none" || ValBase == "default") {
      OutStrings.push_back(Args.MakeArgString(Out + Val));
      return;
    }
  }

  // Each reciprocal type may be enabled or disabled individually.
  // Check each input value for validity, concatenate them all back together,
  // and pass through.

  llvm::StringMap<bool> OptionStrings;
  OptionStrings.insert(std::make_pair("divd", false));
  OptionStrings.insert(std::make_pair("divf", false));
  OptionStrings.insert(std::make_pair("vec-divd", false));
  OptionStrings.insert(std::make_pair("vec-divf", false));
  OptionStrings.insert(std::make_pair("sqrtd", false));
  OptionStrings.insert(std::make_pair("sqrtf", false));
  OptionStrings.insert(std::make_pair("vec-sqrtd", false));
  OptionStrings.insert(std::make_pair("vec-sqrtf", false));

  for (unsigned i = 0; i != NumOptions; ++i) {
    StringRef Val = A->getValue(i);

    bool IsDisabled = Val.startswith(DisabledPrefixIn);
    // Ignore the disablement token for string matching.
    if (IsDisabled)
      Val = Val.substr(1);

    size_t RefStep;
    if (!getRefinementStep(Val, D, *A, RefStep))
      return;

    StringRef ValBase = Val.slice(0, RefStep);
    llvm::StringMap<bool>::iterator OptionIter = OptionStrings.find(ValBase);
    if (OptionIter == OptionStrings.end()) {
      // Try again specifying float suffix.
      OptionIter = OptionStrings.find(ValBase.str() + 'f');
      if (OptionIter == OptionStrings.end()) {
        // The input name did not match any known option string.
        D.Diag(diag::err_drv_unknown_argument) << Val;
        return;
      }
      // The option was specified without a float or double suffix.
      // Make sure that the double entry was not already specified.
      // The float entry will be checked below.
      if (OptionStrings[ValBase.str() + 'd']) {
        D.Diag(diag::err_drv_invalid_value) << A->getOption().getName() << Val;
        return;
      }
    }

    if (OptionIter->second == true) {
      // Duplicate option specified.
      D.Diag(diag::err_drv_invalid_value) << A->getOption().getName() << Val;
      return;
    }

    // Mark the matched option as found. Do not allow duplicate specifiers.
    OptionIter->second = true;

    // If the precision was not specified, also mark the double entry as found.
    if (ValBase.back() != 'f' && ValBase.back() != 'd')
      OptionStrings[ValBase.str() + 'd'] = true;

    // Build the output string.
    StringRef Prefix = IsDisabled ? DisabledPrefixOut : EnabledPrefixOut;
    Out = Args.MakeArgString(Out + Prefix + Val);
    if (i != NumOptions - 1)
      Out = Args.MakeArgString(Out + ",");
  }

  OutStrings.push_back(Args.MakeArgString(Out));
}

static void getX86TargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args,
                                 std::vector<const char *> &Features) {
  // If -march=native, autodetect the feature list.
  if (const Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
    if (StringRef(A->getValue()) == "native") {
      llvm::StringMap<bool> HostFeatures;
      if (llvm::sys::getHostCPUFeatures(HostFeatures))
        for (auto &F : HostFeatures)
          Features.push_back(
              Args.MakeArgString((F.second ? "+" : "-") + F.first()));
    }
  }

  if (Triple.getArchName() == "x86_64h") {
    // x86_64h implies quite a few of the more modern subtarget features
    // for Haswell class CPUs, but not all of them. Opt-out of a few.
    Features.push_back("-rdrnd");
    Features.push_back("-aes");
    Features.push_back("-pclmul");
    Features.push_back("-rtm");
    Features.push_back("-hle");
    Features.push_back("-fsgsbase");
  }

  const llvm::Triple::ArchType ArchType = Triple.getArch();
  // Add features to be compatible with gcc for Android.
  if (Triple.isAndroid()) {
    if (ArchType == llvm::Triple::x86_64) {
      Features.push_back("+sse4.2");
      Features.push_back("+popcnt");
    } else
      Features.push_back("+ssse3");
  }

  // Set features according to the -arch flag on MSVC.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_arch)) {
    StringRef Arch = A->getValue();
    bool ArchUsed = false;
    // First, look for flags that are shared in x86 and x86-64.
    if (ArchType == llvm::Triple::x86_64 || ArchType == llvm::Triple::x86) {
      if (Arch == "AVX" || Arch == "AVX2") {
        ArchUsed = true;
        Features.push_back(Args.MakeArgString("+" + Arch.lower()));
      }
    }
    // Then, look for x86-specific flags.
    if (ArchType == llvm::Triple::x86) {
      if (Arch == "IA32") {
        ArchUsed = true;
      } else if (Arch == "SSE" || Arch == "SSE2") {
        ArchUsed = true;
        Features.push_back(Args.MakeArgString("+" + Arch.lower()));
      }
    }
    if (!ArchUsed)
      D.Diag(clang::diag::warn_drv_unused_argument) << A->getAsString(Args);
  }

  // Now add any that the user explicitly requested on the command line,
  // which may override the defaults.
  handleTargetFeaturesGroup(Args, Features, options::OPT_m_x86_Features_Group);
}

void Clang::AddX86TargetArgs(const ArgList &Args,
                             ArgStringList &CmdArgs) const {
  if (!Args.hasFlag(options::OPT_mred_zone, options::OPT_mno_red_zone, true) ||
      Args.hasArg(options::OPT_mkernel) ||
      Args.hasArg(options::OPT_fapple_kext))
    CmdArgs.push_back("-disable-red-zone");

  // Default to avoid implicit floating-point for kernel/kext code, but allow
  // that to be overridden with -mno-soft-float.
  bool NoImplicitFloat = (Args.hasArg(options::OPT_mkernel) ||
                          Args.hasArg(options::OPT_fapple_kext));
  if (Arg *A = Args.getLastArg(
          options::OPT_msoft_float, options::OPT_mno_soft_float,
          options::OPT_mimplicit_float, options::OPT_mno_implicit_float)) {
    const Option &O = A->getOption();
    NoImplicitFloat = (O.matches(options::OPT_mno_implicit_float) ||
                       O.matches(options::OPT_msoft_float));
  }
  if (NoImplicitFloat)
    CmdArgs.push_back("-no-implicit-float");

  if (Arg *A = Args.getLastArg(options::OPT_masm_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "intel" || Value == "att") {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back(Args.MakeArgString("-x86-asm-syntax=" + Value));
    } else {
      getToolChain().getDriver().Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Value;
    }
  }

  // Set flags to support MCU ABI.
  if (Args.hasFlag(options::OPT_miamcu, options::OPT_mno_iamcu, false)) {
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
    CmdArgs.push_back("-mstack-alignment=4");
  }
}

void Clang::AddHexagonTargetArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-mqdsp6-compat");
  CmdArgs.push_back("-Wreturn-type");

  if (auto G = toolchains::HexagonToolChain::getSmallDataThreshold(Args)) {
    std::string N = llvm::utostr(G.getValue());
    std::string Opt = std::string("-hexagon-small-data-threshold=") + N;
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back(Args.MakeArgString(Opt));
  }

  if (!Args.hasArg(options::OPT_fno_short_enums))
    CmdArgs.push_back("-fshort-enums");
  if (Args.getLastArg(options::OPT_mieee_rnd_near)) {
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back("-enable-hexagon-ieee-rnd-near");
  }
  CmdArgs.push_back("-mllvm");
  CmdArgs.push_back("-machine-sink-split=0");
}

void Clang::AddLanaiTargetArgs(const ArgList &Args,
                               ArgStringList &CmdArgs) const {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
    StringRef CPUName = A->getValue();

    CmdArgs.push_back("-target-cpu");
    CmdArgs.push_back(Args.MakeArgString(CPUName));
  }
  if (Arg *A = Args.getLastArg(options::OPT_mregparm_EQ)) {
    StringRef Value = A->getValue();
    // Only support mregparm=4 to support old usage. Report error for all other
    // cases.
    int Mregparm;
    if (Value.getAsInteger(10, Mregparm)) {
      if (Mregparm != 4) {
        getToolChain().getDriver().Diag(
            diag::err_drv_unsupported_option_argument)
            << A->getOption().getName() << Value;
      }
    }
  }
}

void Clang::AddWebAssemblyTargetArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
  // Default to "hidden" visibility.
  if (!Args.hasArg(options::OPT_fvisibility_EQ,
                   options::OPT_fvisibility_ms_compat)) {
    CmdArgs.push_back("-fvisibility");
    CmdArgs.push_back("hidden");
  }
}

// Decode AArch64 features from string like +[no]featureA+[no]featureB+...
static bool DecodeAArch64Features(const Driver &D, StringRef text,
                                  std::vector<const char *> &Features) {
  SmallVector<StringRef, 8> Split;
  text.split(Split, StringRef("+"), -1, false);

  for (StringRef Feature : Split) {
    if (const char *FeatureName = llvm::AArch64::getArchExtFeature(Feature))
      Features.push_back(FeatureName);
    else if (Feature == "neon" || Feature == "noneon")
      D.Diag(diag::err_drv_no_neon_modifier);
    else
      return false;
  }
  return true;
}

// Check if the CPU name and feature modifiers in -mcpu are legal. If yes,
// decode CPU and feature.
static bool DecodeAArch64Mcpu(const Driver &D, StringRef Mcpu, StringRef &CPU,
                              std::vector<const char *> &Features) {
  std::pair<StringRef, StringRef> Split = Mcpu.split("+");
  CPU = Split.first;

  if (CPU == "generic") {
    Features.push_back("+neon");
  } else {
    unsigned ArchKind = llvm::AArch64::parseCPUArch(CPU);
    unsigned Extersion = llvm::AArch64::getDefaultExtensions(CPU, ArchKind);

    if (!llvm::AArch64::getExtensionFeatures(Extersion, Features))
      return false;
   }

  if (Split.second.size() && !DecodeAArch64Features(D, Split.second, Features))
    return false;

  return true;
}

static bool
getAArch64ArchFeaturesFromMarch(const Driver &D, StringRef March,
                                const ArgList &Args,
                                std::vector<const char *> &Features) {
  std::string MarchLowerCase = March.lower();
  std::pair<StringRef, StringRef> Split = StringRef(MarchLowerCase).split("+");

  unsigned ArchKind = llvm::AArch64::parseArch(Split.first);
  if (ArchKind == static_cast<unsigned>(llvm::AArch64::ArchKind::AK_INVALID) ||
      !llvm::AArch64::getArchFeatures(ArchKind, Features) ||
      (Split.second.size() && !DecodeAArch64Features(D, Split.second, Features)))
    return false;

  return true;
}

static bool
getAArch64ArchFeaturesFromMcpu(const Driver &D, StringRef Mcpu,
                               const ArgList &Args,
                               std::vector<const char *> &Features) {
  StringRef CPU;
  std::string McpuLowerCase = Mcpu.lower();
  if (!DecodeAArch64Mcpu(D, McpuLowerCase, CPU, Features))
    return false;

  return true;
}

static bool
getAArch64MicroArchFeaturesFromMtune(const Driver &D, StringRef Mtune,
                                     const ArgList &Args,
                                     std::vector<const char *> &Features) {
  std::string MtuneLowerCase = Mtune.lower();
  // Handle CPU name is 'native'.
  if (MtuneLowerCase == "native")
    MtuneLowerCase = llvm::sys::getHostCPUName();
  if (MtuneLowerCase == "cyclone") {
    Features.push_back("+zcm");
    Features.push_back("+zcz");
  }
  return true;
}

static bool
getAArch64MicroArchFeaturesFromMcpu(const Driver &D, StringRef Mcpu,
                                    const ArgList &Args,
                                    std::vector<const char *> &Features) {
  StringRef CPU;
  std::vector<const char *> DecodedFeature;
  std::string McpuLowerCase = Mcpu.lower();
  if (!DecodeAArch64Mcpu(D, McpuLowerCase, CPU, DecodedFeature))
    return false;

  return getAArch64MicroArchFeaturesFromMtune(D, CPU, Args, Features);
}

static void getAArch64TargetFeatures(const Driver &D, const ArgList &Args,
                                     std::vector<const char *> &Features) {
  Arg *A;
  bool success = true;
  // Enable NEON by default.
  Features.push_back("+neon");
  if ((A = Args.getLastArg(options::OPT_march_EQ)))
    success = getAArch64ArchFeaturesFromMarch(D, A->getValue(), Args, Features);
  else if ((A = Args.getLastArg(options::OPT_mcpu_EQ)))
    success = getAArch64ArchFeaturesFromMcpu(D, A->getValue(), Args, Features);
  else if (Args.hasArg(options::OPT_arch))
    success = getAArch64ArchFeaturesFromMcpu(D, getAArch64TargetCPU(Args, A),
                                             Args, Features);

  if (success && (A = Args.getLastArg(options::OPT_mtune_EQ)))
    success =
        getAArch64MicroArchFeaturesFromMtune(D, A->getValue(), Args, Features);
  else if (success && (A = Args.getLastArg(options::OPT_mcpu_EQ)))
    success =
        getAArch64MicroArchFeaturesFromMcpu(D, A->getValue(), Args, Features);
  else if (success && Args.hasArg(options::OPT_arch))
    success = getAArch64MicroArchFeaturesFromMcpu(
        D, getAArch64TargetCPU(Args, A), Args, Features);

  if (!success)
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);

  if (Args.getLastArg(options::OPT_mgeneral_regs_only)) {
    Features.push_back("-fp-armv8");
    Features.push_back("-crypto");
    Features.push_back("-neon");
  }

  // En/disable crc
  if (Arg *A = Args.getLastArg(options::OPT_mcrc, options::OPT_mnocrc)) {
    if (A->getOption().matches(options::OPT_mcrc))
      Features.push_back("+crc");
    else
      Features.push_back("-crc");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mno_unaligned_access,
                               options::OPT_munaligned_access))
    if (A->getOption().matches(options::OPT_mno_unaligned_access))
      Features.push_back("+strict-align");

  if (Args.hasArg(options::OPT_ffixed_x18))
    Features.push_back("+reserve-x18");
}

static void getHexagonTargetFeatures(const ArgList &Args,
                                     std::vector<const char *> &Features) {
  handleTargetFeaturesGroup(Args, Features,
                            options::OPT_m_hexagon_Features_Group);

  bool UseLongCalls = false;
  if (Arg *A = Args.getLastArg(options::OPT_mlong_calls,
                               options::OPT_mno_long_calls)) {
    if (A->getOption().matches(options::OPT_mlong_calls))
      UseLongCalls = true;
  }

  Features.push_back(UseLongCalls ? "+long-calls" : "-long-calls");
}

static void getWebAssemblyTargetFeatures(const ArgList &Args,
                                         std::vector<const char *> &Features) {
  handleTargetFeaturesGroup(Args, Features, options::OPT_m_wasm_Features_Group);
}

static void getAMDGPUTargetFeatures(const Driver &D, const ArgList &Args,
                                    std::vector<const char *> &Features) {
  if (const Arg *dAbi = Args.getLastArg(options::OPT_mamdgpu_debugger_abi)) {
    StringRef value = dAbi->getValue();
    if (value == "1.0") {
      Features.push_back("+amdgpu-debugger-insert-nops");
      Features.push_back("+amdgpu-debugger-reserve-regs");
      Features.push_back("+amdgpu-debugger-emit-prologue");
    } else {
      D.Diag(diag::err_drv_clang_unsupported) << dAbi->getAsString(Args);
    }
  }

  handleTargetFeaturesGroup(
    Args, Features, options::OPT_m_amdgpu_Features_Group);
}

static void getTargetFeatures(const ToolChain &TC, const llvm::Triple &Triple,
                              const ArgList &Args, ArgStringList &CmdArgs,
                              bool ForAS) {
  const Driver &D = TC.getDriver();
  std::vector<const char *> Features;
  switch (Triple.getArch()) {
  default:
    break;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    getMIPSTargetFeatures(D, Triple, Args, Features);
    break;

  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    getARMTargetFeatures(TC, Triple, Args, Features, ForAS);
    break;

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    getPPCTargetFeatures(D, Triple, Args, Features);
    break;
  case llvm::Triple::systemz:
    getSystemZTargetFeatures(Args, Features);
    break;
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    getAArch64TargetFeatures(D, Args, Features);
    break;
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    getX86TargetFeatures(D, Triple, Args, Features);
    break;
  case llvm::Triple::hexagon:
    getHexagonTargetFeatures(Args, Features);
    break;
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    getWebAssemblyTargetFeatures(Args, Features);
    break;
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::sparcv9:
    getSparcTargetFeatures(D, Args, Features);
    break;
  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
    getAMDGPUTargetFeatures(D, Args, Features);
    break;
  }

  // Find the last of each feature.
  llvm::StringMap<unsigned> LastOpt;
  for (unsigned I = 0, N = Features.size(); I < N; ++I) {
    const char *Name = Features[I];
    assert(Name[0] == '-' || Name[0] == '+');
    LastOpt[Name + 1] = I;
  }

  for (unsigned I = 0, N = Features.size(); I < N; ++I) {
    // If this feature was overridden, ignore it.
    const char *Name = Features[I];
    llvm::StringMap<unsigned>::iterator LastI = LastOpt.find(Name + 1);
    assert(LastI != LastOpt.end());
    unsigned Last = LastI->second;
    if (Last != I)
      continue;

    CmdArgs.push_back("-target-feature");
    CmdArgs.push_back(Name);
  }
}

static bool
shouldUseExceptionTablesForObjCExceptions(const ObjCRuntime &runtime,
                                          const llvm::Triple &Triple) {
  // We use the zero-cost exception tables for Objective-C if the non-fragile
  // ABI is enabled or when compiling for x86_64 and ARM on Snow Leopard and
  // later.
  if (runtime.isNonFragile())
    return true;

  if (!Triple.isMacOSX())
    return false;

  return (!Triple.isMacOSXVersionLT(10, 5) &&
          (Triple.getArch() == llvm::Triple::x86_64 ||
           Triple.getArch() == llvm::Triple::arm));
}

/// Adds exception related arguments to the driver command arguments. There's a
/// master flag, -fexceptions and also language specific flags to enable/disable
/// C++ and Objective-C exceptions. This makes it possible to for example
/// disable C++ exceptions but enable Objective-C exceptions.
static void addExceptionArgs(const ArgList &Args, types::ID InputType,
                             const ToolChain &TC, bool KernelOrKext,
                             const ObjCRuntime &objcRuntime,
                             ArgStringList &CmdArgs) {
  const Driver &D = TC.getDriver();
  const llvm::Triple &Triple = TC.getTriple();

  if (KernelOrKext) {
    // -mkernel and -fapple-kext imply no exceptions, so claim exception related
    // arguments now to avoid warnings about unused arguments.
    Args.ClaimAllArgs(options::OPT_fexceptions);
    Args.ClaimAllArgs(options::OPT_fno_exceptions);
    Args.ClaimAllArgs(options::OPT_fobjc_exceptions);
    Args.ClaimAllArgs(options::OPT_fno_objc_exceptions);
    Args.ClaimAllArgs(options::OPT_fcxx_exceptions);
    Args.ClaimAllArgs(options::OPT_fno_cxx_exceptions);
    return;
  }

  // See if the user explicitly enabled exceptions.
  bool EH = Args.hasFlag(options::OPT_fexceptions, options::OPT_fno_exceptions,
                         false);

  // Obj-C exceptions are enabled by default, regardless of -fexceptions. This
  // is not necessarily sensible, but follows GCC.
  if (types::isObjC(InputType) &&
      Args.hasFlag(options::OPT_fobjc_exceptions,
                   options::OPT_fno_objc_exceptions, true)) {
    CmdArgs.push_back("-fobjc-exceptions");

    EH |= shouldUseExceptionTablesForObjCExceptions(objcRuntime, Triple);
  }

  if (types::isCXX(InputType)) {
    // Disable C++ EH by default on XCore and PS4.
    bool CXXExceptionsEnabled =
        Triple.getArch() != llvm::Triple::xcore && !Triple.isPS4CPU();
    Arg *ExceptionArg = Args.getLastArg(
        options::OPT_fcxx_exceptions, options::OPT_fno_cxx_exceptions,
        options::OPT_fexceptions, options::OPT_fno_exceptions);
    if (ExceptionArg)
      CXXExceptionsEnabled =
          ExceptionArg->getOption().matches(options::OPT_fcxx_exceptions) ||
          ExceptionArg->getOption().matches(options::OPT_fexceptions);

    if (CXXExceptionsEnabled) {
      if (Triple.isPS4CPU()) {
        ToolChain::RTTIMode RTTIMode = TC.getRTTIMode();
        assert(ExceptionArg &&
               "On the PS4 exceptions should only be enabled if passing "
               "an argument");
        if (RTTIMode == ToolChain::RM_DisabledExplicitly) {
          const Arg *RTTIArg = TC.getRTTIArg();
          assert(RTTIArg && "RTTI disabled explicitly but no RTTIArg!");
          D.Diag(diag::err_drv_argument_not_allowed_with)
              << RTTIArg->getAsString(Args) << ExceptionArg->getAsString(Args);
        } else if (RTTIMode == ToolChain::RM_EnabledImplicitly)
          D.Diag(diag::warn_drv_enabling_rtti_with_exceptions);
      } else
        assert(TC.getRTTIMode() != ToolChain::RM_DisabledImplicitly);

      CmdArgs.push_back("-fcxx-exceptions");

      EH = true;
    }
  }

  if (EH)
    CmdArgs.push_back("-fexceptions");
}

static bool ShouldDisableAutolink(const ArgList &Args, const ToolChain &TC) {
  bool Default = true;
  if (TC.getTriple().isOSDarwin()) {
    // The native darwin assembler doesn't support the linker_option directives,
    // so we disable them if we think the .s file will be passed to it.
    Default = TC.useIntegratedAs();
  }
  return !Args.hasFlag(options::OPT_fautolink, options::OPT_fno_autolink,
                       Default);
}

static bool ShouldDisableDwarfDirectory(const ArgList &Args,
                                        const ToolChain &TC) {
  bool UseDwarfDirectory =
      Args.hasFlag(options::OPT_fdwarf_directory_asm,
                   options::OPT_fno_dwarf_directory_asm, TC.useIntegratedAs());
  return !UseDwarfDirectory;
}

/// \brief Check whether the given input tree contains any compilation actions.
static bool ContainsCompileAction(const Action *A) {
  if (isa<CompileJobAction>(A) || isa<BackendJobAction>(A))
    return true;

  for (const auto &AI : A->inputs())
    if (ContainsCompileAction(AI))
      return true;

  return false;
}

/// \brief Check if -relax-all should be passed to the internal assembler.
/// This is done by default when compiling non-assembler source with -O0.
static bool UseRelaxAll(Compilation &C, const ArgList &Args) {
  bool RelaxDefault = true;

  if (Arg *A = Args.getLastArg(options::OPT_O_Group))
    RelaxDefault = A->getOption().matches(options::OPT_O0);

  if (RelaxDefault) {
    RelaxDefault = false;
    for (const auto &Act : C.getActions()) {
      if (ContainsCompileAction(Act)) {
        RelaxDefault = true;
        break;
      }
    }
  }

  return Args.hasFlag(options::OPT_mrelax_all, options::OPT_mno_relax_all,
                      RelaxDefault);
}

// Convert an arg of the form "-gN" or "-ggdbN" or one of their aliases
// to the corresponding DebugInfoKind.
static codegenoptions::DebugInfoKind DebugLevelToInfoKind(const Arg &A) {
  assert(A.getOption().matches(options::OPT_gN_Group) &&
         "Not a -g option that specifies a debug-info level");
  if (A.getOption().matches(options::OPT_g0) ||
      A.getOption().matches(options::OPT_ggdb0))
    return codegenoptions::NoDebugInfo;
  if (A.getOption().matches(options::OPT_gline_tables_only) ||
      A.getOption().matches(options::OPT_ggdb1))
    return codegenoptions::DebugLineTablesOnly;
  return codegenoptions::LimitedDebugInfo;
}

// Extract the integer N from a string spelled "-dwarf-N", returning 0
// on mismatch. The StringRef input (rather than an Arg) allows
// for use by the "-Xassembler" option parser.
static unsigned DwarfVersionNum(StringRef ArgValue) {
  return llvm::StringSwitch<unsigned>(ArgValue)
      .Case("-gdwarf-2", 2)
      .Case("-gdwarf-3", 3)
      .Case("-gdwarf-4", 4)
      .Case("-gdwarf-5", 5)
      .Default(0);
}

static void RenderDebugEnablingArgs(const ArgList &Args, ArgStringList &CmdArgs,
                                    codegenoptions::DebugInfoKind DebugInfoKind,
                                    unsigned DwarfVersion,
                                    llvm::DebuggerKind DebuggerTuning) {
  switch (DebugInfoKind) {
  case codegenoptions::DebugLineTablesOnly:
    CmdArgs.push_back("-debug-info-kind=line-tables-only");
    break;
  case codegenoptions::LimitedDebugInfo:
    CmdArgs.push_back("-debug-info-kind=limited");
    break;
  case codegenoptions::FullDebugInfo:
    CmdArgs.push_back("-debug-info-kind=standalone");
    break;
  default:
    break;
  }
  if (DwarfVersion > 0)
    CmdArgs.push_back(
        Args.MakeArgString("-dwarf-version=" + Twine(DwarfVersion)));
  switch (DebuggerTuning) {
  case llvm::DebuggerKind::GDB:
    CmdArgs.push_back("-debugger-tuning=gdb");
    break;
  case llvm::DebuggerKind::LLDB:
    CmdArgs.push_back("-debugger-tuning=lldb");
    break;
  case llvm::DebuggerKind::SCE:
    CmdArgs.push_back("-debugger-tuning=sce");
    break;
  default:
    break;
  }
}

static void CollectArgsForIntegratedAssembler(Compilation &C,
                                              const ArgList &Args,
                                              ArgStringList &CmdArgs,
                                              const Driver &D) {
  if (UseRelaxAll(C, Args))
    CmdArgs.push_back("-mrelax-all");

  // Only default to -mincremental-linker-compatible if we think we are
  // targeting the MSVC linker.
  bool DefaultIncrementalLinkerCompatible =
      C.getDefaultToolChain().getTriple().isWindowsMSVCEnvironment();
  if (Args.hasFlag(options::OPT_mincremental_linker_compatible,
                   options::OPT_mno_incremental_linker_compatible,
                   DefaultIncrementalLinkerCompatible))
    CmdArgs.push_back("-mincremental-linker-compatible");

  switch (C.getDefaultToolChain().getArch()) {
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    if (Arg *A = Args.getLastArg(options::OPT_mimplicit_it_EQ)) {
      StringRef Value = A->getValue();
      if (Value == "always" || Value == "never" || Value == "arm" ||
          Value == "thumb") {
        CmdArgs.push_back("-mllvm");
        CmdArgs.push_back(Args.MakeArgString("-arm-implicit-it=" + Value));
      } else {
        D.Diag(diag::err_drv_unsupported_option_argument)
            << A->getOption().getName() << Value;
      }
    }
    break;
  default:
    break;
  }

  // When passing -I arguments to the assembler we sometimes need to
  // unconditionally take the next argument.  For example, when parsing
  // '-Wa,-I -Wa,foo' we need to accept the -Wa,foo arg after seeing the
  // -Wa,-I arg and when parsing '-Wa,-I,foo' we need to accept the 'foo'
  // arg after parsing the '-I' arg.
  bool TakeNextArg = false;

  // When using an integrated assembler, translate -Wa, and -Xassembler
  // options.
  bool CompressDebugSections = false;

  bool UseRelaxRelocations = ENABLE_X86_RELAX_RELOCATIONS;
  const char *MipsTargetFeature = nullptr;
  for (const Arg *A :
       Args.filtered(options::OPT_Wa_COMMA, options::OPT_Xassembler)) {
    A->claim();

    for (StringRef Value : A->getValues()) {
      if (TakeNextArg) {
        CmdArgs.push_back(Value.data());
        TakeNextArg = false;
        continue;
      }

      switch (C.getDefaultToolChain().getArch()) {
      default:
        break;
      case llvm::Triple::mips:
      case llvm::Triple::mipsel:
      case llvm::Triple::mips64:
      case llvm::Triple::mips64el:
        if (Value == "--trap") {
          CmdArgs.push_back("-target-feature");
          CmdArgs.push_back("+use-tcc-in-div");
          continue;
        }
        if (Value == "--break") {
          CmdArgs.push_back("-target-feature");
          CmdArgs.push_back("-use-tcc-in-div");
          continue;
        }
        if (Value.startswith("-msoft-float")) {
          CmdArgs.push_back("-target-feature");
          CmdArgs.push_back("+soft-float");
          continue;
        }
        if (Value.startswith("-mhard-float")) {
          CmdArgs.push_back("-target-feature");
          CmdArgs.push_back("-soft-float");
          continue;
        }

        MipsTargetFeature = llvm::StringSwitch<const char *>(Value)
                                .Case("-mips1", "+mips1")
                                .Case("-mips2", "+mips2")
                                .Case("-mips3", "+mips3")
                                .Case("-mips4", "+mips4")
                                .Case("-mips5", "+mips5")
                                .Case("-mips32", "+mips32")
                                .Case("-mips32r2", "+mips32r2")
                                .Case("-mips32r3", "+mips32r3")
                                .Case("-mips32r5", "+mips32r5")
                                .Case("-mips32r6", "+mips32r6")
                                .Case("-mips64", "+mips64")
                                .Case("-mips64r2", "+mips64r2")
                                .Case("-mips64r3", "+mips64r3")
                                .Case("-mips64r5", "+mips64r5")
                                .Case("-mips64r6", "+mips64r6")
                                .Default(nullptr);
        if (MipsTargetFeature)
          continue;
      }

      if (Value == "-force_cpusubtype_ALL") {
        // Do nothing, this is the default and we don't support anything else.
      } else if (Value == "-L") {
        CmdArgs.push_back("-msave-temp-labels");
      } else if (Value == "--fatal-warnings") {
        CmdArgs.push_back("-massembler-fatal-warnings");
      } else if (Value == "--noexecstack") {
        CmdArgs.push_back("-mnoexecstack");
      } else if (Value == "-compress-debug-sections" ||
                 Value == "--compress-debug-sections") {
        CompressDebugSections = true;
      } else if (Value == "-nocompress-debug-sections" ||
                 Value == "--nocompress-debug-sections") {
        CompressDebugSections = false;
      } else if (Value == "-mrelax-relocations=yes" ||
                 Value == "--mrelax-relocations=yes") {
        UseRelaxRelocations = true;
      } else if (Value == "-mrelax-relocations=no" ||
                 Value == "--mrelax-relocations=no") {
        UseRelaxRelocations = false;
      } else if (Value.startswith("-I")) {
        CmdArgs.push_back(Value.data());
        // We need to consume the next argument if the current arg is a plain
        // -I. The next arg will be the include directory.
        if (Value == "-I")
          TakeNextArg = true;
      } else if (Value.startswith("-gdwarf-")) {
        // "-gdwarf-N" options are not cc1as options.
        unsigned DwarfVersion = DwarfVersionNum(Value);
        if (DwarfVersion == 0) { // Send it onward, and let cc1as complain.
          CmdArgs.push_back(Value.data());
        } else {
          RenderDebugEnablingArgs(Args, CmdArgs,
                                  codegenoptions::LimitedDebugInfo,
                                  DwarfVersion, llvm::DebuggerKind::Default);
        }
      } else if (Value.startswith("-mcpu") || Value.startswith("-mfpu") ||
                 Value.startswith("-mhwdiv") || Value.startswith("-march")) {
        // Do nothing, we'll validate it later.
      } else {
        D.Diag(diag::err_drv_unsupported_option_argument)
            << A->getOption().getName() << Value;
      }
    }
  }
  if (CompressDebugSections) {
    if (llvm::zlib::isAvailable())
      CmdArgs.push_back("-compress-debug-sections");
    else
      D.Diag(diag::warn_debug_compression_unavailable);
  }
  if (UseRelaxRelocations)
    CmdArgs.push_back("--mrelax-relocations");
  if (MipsTargetFeature != nullptr) {
    CmdArgs.push_back("-target-feature");
    CmdArgs.push_back(MipsTargetFeature);
  }
}

// This adds the static libclang_rt.builtins-arch.a directly to the command line
// FIXME: Make sure we can also emit shared objects if they're requested
// and available, check for possible errors, etc.
static void addClangRT(const ToolChain &TC, const ArgList &Args,
                       ArgStringList &CmdArgs) {
  CmdArgs.push_back(TC.getCompilerRTArgString(Args, "builtins"));
}

namespace {
enum OpenMPRuntimeKind {
  /// An unknown OpenMP runtime. We can't generate effective OpenMP code
  /// without knowing what runtime to target.
  OMPRT_Unknown,

  /// The LLVM OpenMP runtime. When completed and integrated, this will become
  /// the default for Clang.
  OMPRT_OMP,

  /// The GNU OpenMP runtime. Clang doesn't support generating OpenMP code for
  /// this runtime but can swallow the pragmas, and find and link against the
  /// runtime library itself.
  OMPRT_GOMP,

  /// The legacy name for the LLVM OpenMP runtime from when it was the Intel
  /// OpenMP runtime. We support this mode for users with existing dependencies
  /// on this runtime library name.
  OMPRT_IOMP5
};
}

/// Compute the desired OpenMP runtime from the flag provided.
static OpenMPRuntimeKind getOpenMPRuntime(const ToolChain &TC,
                                          const ArgList &Args) {
  StringRef RuntimeName(CLANG_DEFAULT_OPENMP_RUNTIME);

  const Arg *A = Args.getLastArg(options::OPT_fopenmp_EQ);
  if (A)
    RuntimeName = A->getValue();

  auto RT = llvm::StringSwitch<OpenMPRuntimeKind>(RuntimeName)
                .Case("libomp", OMPRT_OMP)
                .Case("libgomp", OMPRT_GOMP)
                .Case("libiomp5", OMPRT_IOMP5)
                .Default(OMPRT_Unknown);

  if (RT == OMPRT_Unknown) {
    if (A)
      TC.getDriver().Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << A->getValue();
    else
      // FIXME: We could use a nicer diagnostic here.
      TC.getDriver().Diag(diag::err_drv_unsupported_opt) << "-fopenmp";
  }

  return RT;
}

static void addOpenMPRuntime(ArgStringList &CmdArgs, const ToolChain &TC,
                              const ArgList &Args) {
  if (!Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                    options::OPT_fno_openmp, false))
    return;

  switch (getOpenMPRuntime(TC, Args)) {
  case OMPRT_OMP:
    CmdArgs.push_back("-lomp");
    break;
  case OMPRT_GOMP:
    CmdArgs.push_back("-lgomp");
    break;
  case OMPRT_IOMP5:
    CmdArgs.push_back("-liomp5");
    break;
  case OMPRT_Unknown:
    // Already diagnosed.
    break;
  }
}

static void addSanitizerRuntime(const ToolChain &TC, const ArgList &Args,
                                ArgStringList &CmdArgs, StringRef Sanitizer,
                                bool IsShared, bool IsWhole) {
  // Wrap any static runtimes that must be forced into executable in
  // whole-archive.
  if (IsWhole) CmdArgs.push_back("-whole-archive");
  CmdArgs.push_back(TC.getCompilerRTArgString(Args, Sanitizer, IsShared));
  if (IsWhole) CmdArgs.push_back("-no-whole-archive");
}

// Tries to use a file with the list of dynamic symbols that need to be exported
// from the runtime library. Returns true if the file was found.
static bool addSanitizerDynamicList(const ToolChain &TC, const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    StringRef Sanitizer) {
  SmallString<128> SanRT(TC.getCompilerRT(Args, Sanitizer));
  if (llvm::sys::fs::exists(SanRT + ".syms")) {
    CmdArgs.push_back(Args.MakeArgString("--dynamic-list=" + SanRT + ".syms"));
    return true;
  }
  return false;
}

static void linkSanitizerRuntimeDeps(const ToolChain &TC,
                                     ArgStringList &CmdArgs) {
  // Force linking against the system libraries sanitizers depends on
  // (see PR15823 why this is necessary).
  CmdArgs.push_back("--no-as-needed");
  // There's no libpthread or librt on RTEMS.
  if (TC.getTriple().getOS() != llvm::Triple::RTEMS) {
    CmdArgs.push_back("-lpthread");
    CmdArgs.push_back("-lrt");
  }
  CmdArgs.push_back("-lm");
  // There's no libdl on FreeBSD or RTEMS.
  if (TC.getTriple().getOS() != llvm::Triple::FreeBSD &&
      TC.getTriple().getOS() != llvm::Triple::RTEMS)
    CmdArgs.push_back("-ldl");
}

static void
collectSanitizerRuntimes(const ToolChain &TC, const ArgList &Args,
                         SmallVectorImpl<StringRef> &SharedRuntimes,
                         SmallVectorImpl<StringRef> &StaticRuntimes,
                         SmallVectorImpl<StringRef> &NonWholeStaticRuntimes,
                         SmallVectorImpl<StringRef> &HelperStaticRuntimes,
                         SmallVectorImpl<StringRef> &RequiredSymbols) {
  const SanitizerArgs &SanArgs = TC.getSanitizerArgs();
  // Collect shared runtimes.
  if (SanArgs.needsAsanRt() && SanArgs.needsSharedAsanRt()) {
    SharedRuntimes.push_back("asan");
  }
  // The stats_client library is also statically linked into DSOs.
  if (SanArgs.needsStatsRt())
    StaticRuntimes.push_back("stats_client");

  // Collect static runtimes.
  if (Args.hasArg(options::OPT_shared) || TC.getTriple().isAndroid()) {
    // Don't link static runtimes into DSOs or if compiling for Android.
    return;
  }
  if (SanArgs.needsAsanRt()) {
    if (SanArgs.needsSharedAsanRt()) {
      HelperStaticRuntimes.push_back("asan-preinit");
    } else {
      StaticRuntimes.push_back("asan");
      if (SanArgs.linkCXXRuntimes())
        StaticRuntimes.push_back("asan_cxx");
    }
  }
  if (SanArgs.needsDfsanRt())
    StaticRuntimes.push_back("dfsan");
  if (SanArgs.needsLsanRt())
    StaticRuntimes.push_back("lsan");
  if (SanArgs.needsMsanRt()) {
    StaticRuntimes.push_back("msan");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("msan_cxx");
  }
  if (SanArgs.needsTsanRt()) {
    StaticRuntimes.push_back("tsan");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("tsan_cxx");
  }
  if (SanArgs.needsUbsanRt()) {
    StaticRuntimes.push_back("ubsan_standalone");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("ubsan_standalone_cxx");
  }
  if (SanArgs.needsSafeStackRt())
    StaticRuntimes.push_back("safestack");
  if (SanArgs.needsCfiRt())
    StaticRuntimes.push_back("cfi");
  if (SanArgs.needsCfiDiagRt()) {
    StaticRuntimes.push_back("cfi_diag");
    if (SanArgs.linkCXXRuntimes())
      StaticRuntimes.push_back("ubsan_standalone_cxx");
  }
  if (SanArgs.needsStatsRt()) {
    NonWholeStaticRuntimes.push_back("stats");
    RequiredSymbols.push_back("__sanitizer_stats_register");
  }
  if (SanArgs.needsEsanRt())
    StaticRuntimes.push_back("esan");
}

// Should be called before we add system libraries (C++ ABI, libstdc++/libc++,
// C runtime, etc). Returns true if sanitizer system deps need to be linked in.
static bool addSanitizerRuntimes(const ToolChain &TC, const ArgList &Args,
                                 ArgStringList &CmdArgs) {
  SmallVector<StringRef, 4> SharedRuntimes, StaticRuntimes,
      NonWholeStaticRuntimes, HelperStaticRuntimes, RequiredSymbols;
  collectSanitizerRuntimes(TC, Args, SharedRuntimes, StaticRuntimes,
                           NonWholeStaticRuntimes, HelperStaticRuntimes,
                           RequiredSymbols);
  for (auto RT : SharedRuntimes)
    addSanitizerRuntime(TC, Args, CmdArgs, RT, true, false);
  for (auto RT : HelperStaticRuntimes)
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false, true);
  bool AddExportDynamic = false;
  for (auto RT : StaticRuntimes) {
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false, true);
    AddExportDynamic |= !addSanitizerDynamicList(TC, Args, CmdArgs, RT);
  }
  for (auto RT : NonWholeStaticRuntimes) {
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false, false);
    AddExportDynamic |= !addSanitizerDynamicList(TC, Args, CmdArgs, RT);
  }
  for (auto S : RequiredSymbols) {
    CmdArgs.push_back("-u");
    CmdArgs.push_back(Args.MakeArgString(S));
  }
  // If there is a static runtime with no dynamic list, force all the symbols
  // to be dynamic to be sure we export sanitizer interface functions.
  if (AddExportDynamic)
    CmdArgs.push_back("-export-dynamic");

  const SanitizerArgs &SanArgs = TC.getSanitizerArgs();
  if (SanArgs.hasCrossDsoCfi() && !AddExportDynamic)
    CmdArgs.push_back("-export-dynamic-symbol=__cfi_check");

  return !StaticRuntimes.empty();
}

static bool addXRayRuntime(const ToolChain &TC, const ArgList &Args,
                           ArgStringList &CmdArgs) {
  if (Args.hasFlag(options::OPT_fxray_instrument,
                   options::OPT_fnoxray_instrument, false)) {
    CmdArgs.push_back("-whole-archive");
    CmdArgs.push_back(TC.getCompilerRTArgString(Args, "xray", false));
    CmdArgs.push_back("-no-whole-archive");
    return true;
  }
  return false;
}

static void linkXRayRuntimeDeps(const ToolChain &TC, const ArgList &Args,
                                ArgStringList &CmdArgs) {
  CmdArgs.push_back("--no-as-needed");
  CmdArgs.push_back("-lpthread");
  CmdArgs.push_back("-lrt");
  CmdArgs.push_back("-lm");
  CmdArgs.push_back("-latomic");
  if (TC.GetCXXStdlibType(Args) == ToolChain::CST_Libcxx)
    CmdArgs.push_back("-lc++");
  else
    CmdArgs.push_back("-lstdc++");
  if (TC.getTriple().getOS() != llvm::Triple::FreeBSD)
    CmdArgs.push_back("-ldl");
}

static bool areOptimizationsEnabled(const ArgList &Args) {
  // Find the last -O arg and see if it is non-zero.
  if (Arg *A = Args.getLastArg(options::OPT_O_Group))
    return !A->getOption().matches(options::OPT_O0);
  // Defaults to -O0.
  return false;
}

static bool mustUseFramePointerForTarget(const llvm::Triple &Triple) {
  switch (Triple.getArch()){
  default:
    return false;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // ARM Darwin targets require a frame pointer to be always present to aid
    // offline debugging via backtraces.
    return Triple.isOSDarwin();
  }
}

static bool useFramePointerForTargetByDefault(const ArgList &Args,
                                              const llvm::Triple &Triple) {
  switch (Triple.getArch()) {
  case llvm::Triple::xcore:
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    // XCore never wants frame pointers, regardless of OS.
    // WebAssembly never wants frame pointers.
    return false;
  default:
    break;
  }

  if (Triple.isOSLinux() || Triple.getOS() == llvm::Triple::CloudABI) {
    switch (Triple.getArch()) {
    // Don't use a frame pointer on linux if optimizing for certain targets.
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::systemz:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      return !areOptimizationsEnabled(Args);
    default:
      return true;
    }
  }

  if (Triple.isOSWindows()) {
    switch (Triple.getArch()) {
    case llvm::Triple::x86:
      return !areOptimizationsEnabled(Args);
    case llvm::Triple::x86_64:
      return Triple.isOSBinFormatMachO();
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      // Windows on ARM builds with FPO disabled to aid fast stack walking
      return true;
    default:
      // All other supported Windows ISAs use xdata unwind information, so frame
      // pointers are not generally useful.
      return false;
    }
  }

  return true;
}

static bool shouldUseFramePointer(const ArgList &Args,
                                  const llvm::Triple &Triple) {
  if (Arg *A = Args.getLastArg(options::OPT_fno_omit_frame_pointer,
                               options::OPT_fomit_frame_pointer))
    return A->getOption().matches(options::OPT_fno_omit_frame_pointer) ||
           mustUseFramePointerForTarget(Triple);

  if (Args.hasArg(options::OPT_pg))
    return true;

  return useFramePointerForTargetByDefault(Args, Triple);
}

static bool shouldUseLeafFramePointer(const ArgList &Args,
                                      const llvm::Triple &Triple) {
  if (Arg *A = Args.getLastArg(options::OPT_mno_omit_leaf_frame_pointer,
                               options::OPT_momit_leaf_frame_pointer))
    return A->getOption().matches(options::OPT_mno_omit_leaf_frame_pointer) ||
           mustUseFramePointerForTarget(Triple);

  if (Args.hasArg(options::OPT_pg))
    return true;

  if (Triple.isPS4CPU())
    return false;

  return useFramePointerForTargetByDefault(Args, Triple);
}

/// Add a CC1 option to specify the debug compilation directory.
static void addDebugCompDirArg(const ArgList &Args, ArgStringList &CmdArgs) {
  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd)) {
    CmdArgs.push_back("-fdebug-compilation-dir");
    CmdArgs.push_back(Args.MakeArgString(cwd));
  }
}

static const char *SplitDebugName(const ArgList &Args, const InputInfo &Input) {
  Arg *FinalOutput = Args.getLastArg(options::OPT_o);
  if (FinalOutput && Args.hasArg(options::OPT_c)) {
    SmallString<128> T(FinalOutput->getValue());
    llvm::sys::path::replace_extension(T, "dwo");
    return Args.MakeArgString(T);
  } else {
    // Use the compilation dir.
    SmallString<128> T(
        Args.getLastArgValue(options::OPT_fdebug_compilation_dir));
    SmallString<128> F(llvm::sys::path::stem(Input.getBaseInput()));
    llvm::sys::path::replace_extension(F, "dwo");
    T += F;
    return Args.MakeArgString(F);
  }
}

static void SplitDebugInfo(const ToolChain &TC, Compilation &C, const Tool &T,
                           const JobAction &JA, const ArgList &Args,
                           const InputInfo &Output, const char *OutFile) {
  ArgStringList ExtractArgs;
  ExtractArgs.push_back("--extract-dwo");

  ArgStringList StripArgs;
  StripArgs.push_back("--strip-dwo");

  // Grabbing the output of the earlier compile step.
  StripArgs.push_back(Output.getFilename());
  ExtractArgs.push_back(Output.getFilename());
  ExtractArgs.push_back(OutFile);

  const char *Exec = Args.MakeArgString(TC.GetProgramPath("objcopy"));
  InputInfo II(types::TY_Object, Output.getFilename(), Output.getFilename());

  // First extract the dwo sections.
  C.addCommand(llvm::make_unique<Command>(JA, T, Exec, ExtractArgs, II));

  // Then remove them from the original .o file.
  C.addCommand(llvm::make_unique<Command>(JA, T, Exec, StripArgs, II));
}

/// \brief Vectorize at all optimization levels greater than 1 except for -Oz.
/// For -Oz the loop vectorizer is disable, while the slp vectorizer is enabled.
static bool shouldEnableVectorizerAtOLevel(const ArgList &Args, bool isSlpVec) {
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      return true;

    if (A->getOption().matches(options::OPT_O0))
      return false;

    assert(A->getOption().matches(options::OPT_O) && "Must have a -O flag");

    // Vectorize -Os.
    StringRef S(A->getValue());
    if (S == "s")
      return true;

    // Don't vectorize -Oz, unless it's the slp vectorizer.
    if (S == "z")
      return isSlpVec;

    unsigned OptLevel = 0;
    if (S.getAsInteger(10, OptLevel))
      return false;

    return OptLevel > 1;
  }

  return false;
}

/// Add -x lang to \p CmdArgs for \p Input.
static void addDashXForInput(const ArgList &Args, const InputInfo &Input,
                             ArgStringList &CmdArgs) {
  // When using -verify-pch, we don't want to provide the type
  // 'precompiled-header' if it was inferred from the file extension
  if (Args.hasArg(options::OPT_verify_pch) && Input.getType() == types::TY_PCH)
    return;

  CmdArgs.push_back("-x");
  if (Args.hasArg(options::OPT_rewrite_objc))
    CmdArgs.push_back(types::getTypeName(types::TY_PP_ObjCXX));
  else
    CmdArgs.push_back(types::getTypeName(Input.getType()));
}

static VersionTuple getMSCompatibilityVersion(unsigned Version) {
  if (Version < 100)
    return VersionTuple(Version);

  if (Version < 10000)
    return VersionTuple(Version / 100, Version % 100);

  unsigned Build = 0, Factor = 1;
  for (; Version > 10000; Version = Version / 10, Factor = Factor * 10)
    Build = Build + (Version % 10) * Factor;
  return VersionTuple(Version / 100, Version % 100, Build);
}

// Claim options we don't want to warn if they are unused. We do this for
// options that build systems might add but are unused when assembling or only
// running the preprocessor for example.
static void claimNoWarnArgs(const ArgList &Args) {
  // Don't warn about unused -f(no-)?lto.  This can happen when we're
  // preprocessing, precompiling or assembling.
  Args.ClaimAllArgs(options::OPT_flto_EQ);
  Args.ClaimAllArgs(options::OPT_flto);
  Args.ClaimAllArgs(options::OPT_fno_lto);
}

static void appendUserToPath(SmallVectorImpl<char> &Result) {
#ifdef LLVM_ON_UNIX
  const char *Username = getenv("LOGNAME");
#else
  const char *Username = getenv("USERNAME");
#endif
  if (Username) {
    // Validate that LoginName can be used in a path, and get its length.
    size_t Len = 0;
    for (const char *P = Username; *P; ++P, ++Len) {
      if (!isAlphanumeric(*P) && *P != '_') {
        Username = nullptr;
        break;
      }
    }

    if (Username && Len > 0) {
      Result.append(Username, Username + Len);
      return;
    }
  }

// Fallback to user id.
#ifdef LLVM_ON_UNIX
  std::string UID = llvm::utostr(getuid());
#else
  // FIXME: Windows seems to have an 'SID' that might work.
  std::string UID = "9999";
#endif
  Result.append(UID.begin(), UID.end());
}

VersionTuple visualstudio::getMSVCVersion(const Driver *D, const ToolChain &TC,
                                          const llvm::Triple &Triple,
                                          const llvm::opt::ArgList &Args,
                                          bool IsWindowsMSVC) {
  if (Args.hasFlag(options::OPT_fms_extensions, options::OPT_fno_ms_extensions,
                   IsWindowsMSVC) ||
      Args.hasArg(options::OPT_fmsc_version) ||
      Args.hasArg(options::OPT_fms_compatibility_version)) {
    const Arg *MSCVersion = Args.getLastArg(options::OPT_fmsc_version);
    const Arg *MSCompatibilityVersion =
        Args.getLastArg(options::OPT_fms_compatibility_version);

    if (MSCVersion && MSCompatibilityVersion) {
      if (D)
        D->Diag(diag::err_drv_argument_not_allowed_with)
            << MSCVersion->getAsString(Args)
            << MSCompatibilityVersion->getAsString(Args);
      return VersionTuple();
    }

    if (MSCompatibilityVersion) {
      VersionTuple MSVT;
      if (MSVT.tryParse(MSCompatibilityVersion->getValue()) && D)
        D->Diag(diag::err_drv_invalid_value)
            << MSCompatibilityVersion->getAsString(Args)
            << MSCompatibilityVersion->getValue();
      return MSVT;
    }

    if (MSCVersion) {
      unsigned Version = 0;
      if (StringRef(MSCVersion->getValue()).getAsInteger(10, Version) && D)
        D->Diag(diag::err_drv_invalid_value) << MSCVersion->getAsString(Args)
                                             << MSCVersion->getValue();
      return getMSCompatibilityVersion(Version);
    }

    unsigned Major, Minor, Micro;
    Triple.getEnvironmentVersion(Major, Minor, Micro);
    if (Major || Minor || Micro)
      return VersionTuple(Major, Minor, Micro);

    if (IsWindowsMSVC) {
      VersionTuple MSVT = TC.getMSVCVersionFromExe();
      if (!MSVT.empty())
        return MSVT;

      // FIXME: Consider bumping this to 19 (MSVC2015) soon.
      return VersionTuple(18);
    }
  }
  return VersionTuple();
}

static void addPGOAndCoverageFlags(Compilation &C, const Driver &D,
                                   const InputInfo &Output, const ArgList &Args,
                                   ArgStringList &CmdArgs) {

  auto *PGOGenerateArg = Args.getLastArg(options::OPT_fprofile_generate,
                                         options::OPT_fprofile_generate_EQ,
                                         options::OPT_fno_profile_generate);
  if (PGOGenerateArg &&
      PGOGenerateArg->getOption().matches(options::OPT_fno_profile_generate))
    PGOGenerateArg = nullptr;

  auto *ProfileGenerateArg = Args.getLastArg(
      options::OPT_fprofile_instr_generate,
      options::OPT_fprofile_instr_generate_EQ,
      options::OPT_fno_profile_instr_generate);
  if (ProfileGenerateArg &&
      ProfileGenerateArg->getOption().matches(
          options::OPT_fno_profile_instr_generate))
    ProfileGenerateArg = nullptr;

  if (PGOGenerateArg && ProfileGenerateArg)
    D.Diag(diag::err_drv_argument_not_allowed_with)
        << PGOGenerateArg->getSpelling() << ProfileGenerateArg->getSpelling();

  auto *ProfileUseArg = Args.getLastArg(
      options::OPT_fprofile_instr_use, options::OPT_fprofile_instr_use_EQ,
      options::OPT_fprofile_use, options::OPT_fprofile_use_EQ,
      options::OPT_fno_profile_instr_use);
  if (ProfileUseArg &&
      ProfileUseArg->getOption().matches(options::OPT_fno_profile_instr_use))
    ProfileUseArg = nullptr;

  if (PGOGenerateArg && ProfileUseArg)
    D.Diag(diag::err_drv_argument_not_allowed_with)
        << ProfileUseArg->getSpelling() << PGOGenerateArg->getSpelling();

  if (ProfileGenerateArg && ProfileUseArg)
    D.Diag(diag::err_drv_argument_not_allowed_with)
        << ProfileGenerateArg->getSpelling() << ProfileUseArg->getSpelling();

  if (ProfileGenerateArg) {
    if (ProfileGenerateArg->getOption().matches(
            options::OPT_fprofile_instr_generate_EQ))
      CmdArgs.push_back(Args.MakeArgString(Twine("-fprofile-instrument-path=") +
                                           ProfileGenerateArg->getValue()));
    // The default is to use Clang Instrumentation.
    CmdArgs.push_back("-fprofile-instrument=clang");
  }

  if (PGOGenerateArg) {
    CmdArgs.push_back("-fprofile-instrument=llvm");
    if (PGOGenerateArg->getOption().matches(
            options::OPT_fprofile_generate_EQ)) {
      SmallString<128> Path(PGOGenerateArg->getValue());
      llvm::sys::path::append(Path, "default_%m.profraw");
      CmdArgs.push_back(
          Args.MakeArgString(Twine("-fprofile-instrument-path=") + Path));
    }
  }

  if (ProfileUseArg) {
    if (ProfileUseArg->getOption().matches(options::OPT_fprofile_instr_use_EQ))
      CmdArgs.push_back(Args.MakeArgString(
          Twine("-fprofile-instrument-use-path=") + ProfileUseArg->getValue()));
    else if ((ProfileUseArg->getOption().matches(
                  options::OPT_fprofile_use_EQ) ||
              ProfileUseArg->getOption().matches(
                  options::OPT_fprofile_instr_use))) {
      SmallString<128> Path(
          ProfileUseArg->getNumValues() == 0 ? "" : ProfileUseArg->getValue());
      if (Path.empty() || llvm::sys::fs::is_directory(Path))
        llvm::sys::path::append(Path, "default.profdata");
      CmdArgs.push_back(
          Args.MakeArgString(Twine("-fprofile-instrument-use-path=") + Path));
    }
  }

  if (Args.hasArg(options::OPT_ftest_coverage) ||
      Args.hasArg(options::OPT_coverage))
    CmdArgs.push_back("-femit-coverage-notes");
  if (Args.hasFlag(options::OPT_fprofile_arcs, options::OPT_fno_profile_arcs,
                   false) ||
      Args.hasArg(options::OPT_coverage))
    CmdArgs.push_back("-femit-coverage-data");

  if (Args.hasFlag(options::OPT_fcoverage_mapping,
                   options::OPT_fno_coverage_mapping, false) &&
      !ProfileGenerateArg)
    D.Diag(diag::err_drv_argument_only_allowed_with)
        << "-fcoverage-mapping"
        << "-fprofile-instr-generate";

  if (Args.hasFlag(options::OPT_fcoverage_mapping,
                   options::OPT_fno_coverage_mapping, false))
    CmdArgs.push_back("-fcoverage-mapping");

  if (C.getArgs().hasArg(options::OPT_c) ||
      C.getArgs().hasArg(options::OPT_S)) {
    if (Output.isFilename()) {
      CmdArgs.push_back("-coverage-notes-file");
      SmallString<128> OutputFilename;
      if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o))
        OutputFilename = FinalOutput->getValue();
      else
        OutputFilename = llvm::sys::path::filename(Output.getBaseInput());
      SmallString<128> CoverageFilename = OutputFilename;
      if (llvm::sys::path::is_relative(CoverageFilename)) {
        SmallString<128> Pwd;
        if (!llvm::sys::fs::current_path(Pwd)) {
          llvm::sys::path::append(Pwd, CoverageFilename);
          CoverageFilename.swap(Pwd);
        }
      }
      llvm::sys::path::replace_extension(CoverageFilename, "gcno");
      CmdArgs.push_back(Args.MakeArgString(CoverageFilename));

      // Leave -fprofile-dir= an unused argument unless .gcda emission is
      // enabled. To be polite, with '-fprofile-arcs -fno-profile-arcs' consider
      // the flag used. There is no -fno-profile-dir, so the user has no
      // targeted way to suppress the warning.
      if (Args.hasArg(options::OPT_fprofile_arcs) ||
          Args.hasArg(options::OPT_coverage)) {
        CmdArgs.push_back("-coverage-data-file");
        if (Arg *FProfileDir = Args.getLastArg(options::OPT_fprofile_dir)) {
          CoverageFilename = FProfileDir->getValue();
          llvm::sys::path::append(CoverageFilename, OutputFilename);
        }
        llvm::sys::path::replace_extension(CoverageFilename, "gcda");
        CmdArgs.push_back(Args.MakeArgString(CoverageFilename));
      }
    }
  }
}

static void addPS4ProfileRTArgs(const ToolChain &TC, const ArgList &Args,
                                ArgStringList &CmdArgs) {
  if ((Args.hasFlag(options::OPT_fprofile_arcs, options::OPT_fno_profile_arcs,
                    false) ||
       Args.hasFlag(options::OPT_fprofile_generate,
                    options::OPT_fno_profile_instr_generate, false) ||
       Args.hasFlag(options::OPT_fprofile_generate_EQ,
                    options::OPT_fno_profile_instr_generate, false) ||
       Args.hasFlag(options::OPT_fprofile_instr_generate,
                    options::OPT_fno_profile_instr_generate, false) ||
       Args.hasFlag(options::OPT_fprofile_instr_generate_EQ,
                    options::OPT_fno_profile_instr_generate, false) ||
       Args.hasArg(options::OPT_fcreate_profile) ||
       Args.hasArg(options::OPT_coverage)))
    CmdArgs.push_back("--dependent-lib=libclang_rt.profile-x86_64.a");
}

/// Parses the various -fpic/-fPIC/-fpie/-fPIE arguments.  Then,
/// smooshes them together with platform defaults, to decide whether
/// this compile should be using PIC mode or not. Returns a tuple of
/// (RelocationModel, PICLevel, IsPIE).
static std::tuple<llvm::Reloc::Model, unsigned, bool>
ParsePICArgs(const ToolChain &ToolChain, const llvm::Triple &Triple,
             const ArgList &Args) {
  // FIXME: why does this code...and so much everywhere else, use both
  // ToolChain.getTriple() and Triple?
  bool PIE = ToolChain.isPIEDefault();
  bool PIC = PIE || ToolChain.isPICDefault();
  // The Darwin/MachO default to use PIC does not apply when using -static.
  if (ToolChain.getTriple().isOSBinFormatMachO() &&
      Args.hasArg(options::OPT_static))
    PIE = PIC = false;
  bool IsPICLevelTwo = PIC;

  bool KernelOrKext =
      Args.hasArg(options::OPT_mkernel, options::OPT_fapple_kext);

  // Android-specific defaults for PIC/PIE
  if (ToolChain.getTriple().isAndroid()) {
    switch (ToolChain.getArch()) {
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::aarch64:
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
      PIC = true; // "-fpic"
      break;

    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      PIC = true; // "-fPIC"
      IsPICLevelTwo = true;
      break;

    default:
      break;
    }
  }

  // OpenBSD-specific defaults for PIE
  if (ToolChain.getTriple().getOS() == llvm::Triple::OpenBSD) {
    switch (ToolChain.getArch()) {
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
    case llvm::Triple::sparcel:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      IsPICLevelTwo = false; // "-fpie"
      break;

    case llvm::Triple::ppc:
    case llvm::Triple::sparc:
    case llvm::Triple::sparcv9:
      IsPICLevelTwo = true; // "-fPIE"
      break;

    default:
      break;
    }
  }

  // The last argument relating to either PIC or PIE wins, and no
  // other argument is used. If the last argument is any flavor of the
  // '-fno-...' arguments, both PIC and PIE are disabled. Any PIE
  // option implicitly enables PIC at the same level.
  Arg *LastPICArg = Args.getLastArg(options::OPT_fPIC, options::OPT_fno_PIC,
                                    options::OPT_fpic, options::OPT_fno_pic,
                                    options::OPT_fPIE, options::OPT_fno_PIE,
                                    options::OPT_fpie, options::OPT_fno_pie);
  // Check whether the tool chain trumps the PIC-ness decision. If the PIC-ness
  // is forced, then neither PIC nor PIE flags will have no effect.
  if (!ToolChain.isPICDefaultForced()) {
    if (LastPICArg) {
      Option O = LastPICArg->getOption();
      if (O.matches(options::OPT_fPIC) || O.matches(options::OPT_fpic) ||
          O.matches(options::OPT_fPIE) || O.matches(options::OPT_fpie)) {
        PIE = O.matches(options::OPT_fPIE) || O.matches(options::OPT_fpie);
        PIC =
            PIE || O.matches(options::OPT_fPIC) || O.matches(options::OPT_fpic);
        IsPICLevelTwo =
            O.matches(options::OPT_fPIE) || O.matches(options::OPT_fPIC);
      } else {
        PIE = PIC = false;
        if (Triple.isPS4CPU()) {
          Arg *ModelArg = Args.getLastArg(options::OPT_mcmodel_EQ);
          StringRef Model = ModelArg ? ModelArg->getValue() : "";
          if (Model != "kernel") {
            PIC = true;
            ToolChain.getDriver().Diag(diag::warn_drv_ps4_force_pic)
                << LastPICArg->getSpelling();
          }
        }
      }
    }
  }

  // Introduce a Darwin and PS4-specific hack. If the default is PIC, but the
  // PIC level would've been set to level 1, force it back to level 2 PIC
  // instead.
  if (PIC && (ToolChain.getTriple().isOSDarwin() || Triple.isPS4CPU()))
    IsPICLevelTwo |= ToolChain.isPICDefault();

  // This kernel flags are a trump-card: they will disable PIC/PIE
  // generation, independent of the argument order.
  if (KernelOrKext && ((!Triple.isiOS() || Triple.isOSVersionLT(6)) &&
                       !Triple.isWatchOS()))
    PIC = PIE = false;

  if (Arg *A = Args.getLastArg(options::OPT_mdynamic_no_pic)) {
    // This is a very special mode. It trumps the other modes, almost no one
    // uses it, and it isn't even valid on any OS but Darwin.
    if (!ToolChain.getTriple().isOSDarwin())
      ToolChain.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << ToolChain.getTriple().str();

    // FIXME: Warn when this flag trumps some other PIC or PIE flag.

    // Only a forced PIC mode can cause the actual compile to have PIC defines
    // etc., no flags are sufficient. This behavior was selected to closely
    // match that of llvm-gcc and Apple GCC before that.
    PIC = ToolChain.isPICDefault() && ToolChain.isPICDefaultForced();

    return std::make_tuple(llvm::Reloc::DynamicNoPIC, PIC ? 2U : 0U, false);
  }

  bool EmbeddedPISupported;
  switch (ToolChain.getArch()) {
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      EmbeddedPISupported = true;
      break;
    default:
      EmbeddedPISupported = false;
      break;
  }

  bool ROPI = false, RWPI = false;
  Arg* LastROPIArg = Args.getLastArg(options::OPT_fropi, options::OPT_fno_ropi);
  if (LastROPIArg && LastROPIArg->getOption().matches(options::OPT_fropi)) {
    if (!EmbeddedPISupported)
      ToolChain.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
          << LastROPIArg->getSpelling() << ToolChain.getTriple().str();
    ROPI = true;
  }
  Arg *LastRWPIArg = Args.getLastArg(options::OPT_frwpi, options::OPT_fno_rwpi);
  if (LastRWPIArg && LastRWPIArg->getOption().matches(options::OPT_frwpi)) {
    if (!EmbeddedPISupported)
      ToolChain.getDriver().Diag(diag::err_drv_unsupported_opt_for_target)
          << LastRWPIArg->getSpelling() << ToolChain.getTriple().str();
    RWPI = true;
  }

  // ROPI and RWPI are not comaptible with PIC or PIE.
  if ((ROPI || RWPI) && (PIC || PIE)) {
    ToolChain.getDriver().Diag(diag::err_drv_ropi_rwpi_incompatible_with_pic);
  }

  if (PIC)
    return std::make_tuple(llvm::Reloc::PIC_, IsPICLevelTwo ? 2U : 1U, PIE);

  llvm::Reloc::Model RelocM = llvm::Reloc::Static;
  if (ROPI && RWPI)
    RelocM = llvm::Reloc::ROPI_RWPI;
  else if (ROPI)
    RelocM = llvm::Reloc::ROPI;
  else if (RWPI)
    RelocM = llvm::Reloc::RWPI;

  return std::make_tuple(RelocM, 0U, false);
}

static const char *RelocationModelName(llvm::Reloc::Model Model) {
  switch (Model) {
  case llvm::Reloc::Static:
    return "static";
  case llvm::Reloc::PIC_:
    return "pic";
  case llvm::Reloc::DynamicNoPIC:
    return "dynamic-no-pic";
  case llvm::Reloc::ROPI:
    return "ropi";
  case llvm::Reloc::RWPI:
    return "rwpi";
  case llvm::Reloc::ROPI_RWPI:
    return "ropi-rwpi";
  }
  llvm_unreachable("Unknown Reloc::Model kind");
}

static void AddAssemblerKPIC(const ToolChain &ToolChain, const ArgList &Args,
                             ArgStringList &CmdArgs) {
  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) =
      ParsePICArgs(ToolChain, ToolChain.getTriple(), Args);

  if (RelocationModel != llvm::Reloc::Static)
    CmdArgs.push_back("-KPIC");
}

void Clang::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, const char *LinkingOutput) const {
  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();
  const std::string &TripleStr = Triple.getTriple();

  bool KernelOrKext =
      Args.hasArg(options::OPT_mkernel, options::OPT_fapple_kext);
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  bool IsWindowsGNU = getToolChain().getTriple().isWindowsGNUEnvironment();
  bool IsWindowsCygnus =
      getToolChain().getTriple().isWindowsCygwinEnvironment();
  bool IsWindowsMSVC = getToolChain().getTriple().isWindowsMSVCEnvironment();
  bool IsPS4CPU = getToolChain().getTriple().isPS4CPU();
  bool IsIAMCU = getToolChain().getTriple().isOSIAMCU();

  // Check number of inputs for sanity. We need at least one input.
  assert(Inputs.size() >= 1 && "Must have at least one input.");
  const InputInfo &Input = Inputs[0];
  // CUDA compilation may have multiple inputs (source file + results of
  // device-side compilations). All other jobs are expected to have exactly one
  // input.
  bool IsCuda = JA.isOffloading(Action::OFK_Cuda);
  assert((IsCuda || Inputs.size() == 1) && "Unable to handle multiple inputs.");

  // C++ is not supported for IAMCU.
  if (IsIAMCU && types::isCXX(Input.getType()))
    D.Diag(diag::err_drv_clang_unsupported) << "C++ for IAMCU";

  // Invoke ourselves in -cc1 mode.
  //
  // FIXME: Implement custom jobs for internal actions.
  CmdArgs.push_back("-cc1");

  // Add the "effective" target triple.
  CmdArgs.push_back("-triple");
  CmdArgs.push_back(Args.MakeArgString(TripleStr));

  if (IsCuda) {
    // We have to pass the triple of the host if compiling for a CUDA device and
    // vice-versa.
    std::string NormalizedTriple;
    if (JA.isDeviceOffloading(Action::OFK_Cuda))
      NormalizedTriple = C.getSingleOffloadToolChain<Action::OFK_Host>()
                             ->getTriple()
                             .normalize();
    else
      NormalizedTriple = C.getSingleOffloadToolChain<Action::OFK_Cuda>()
                             ->getTriple()
                             .normalize();

    CmdArgs.push_back("-aux-triple");
    CmdArgs.push_back(Args.MakeArgString(NormalizedTriple));
  }

  if (Triple.isOSWindows() && (Triple.getArch() == llvm::Triple::arm ||
                               Triple.getArch() == llvm::Triple::thumb)) {
    unsigned Offset = Triple.getArch() == llvm::Triple::arm ? 4 : 6;
    unsigned Version;
    Triple.getArchName().substr(Offset).getAsInteger(10, Version);
    if (Version < 7)
      D.Diag(diag::err_target_unsupported_arch) << Triple.getArchName()
                                                << TripleStr;
  }

  // Push all default warning arguments that are specific to
  // the given target.  These come before user provided warning options
  // are provided.
  getToolChain().addClangWarningOptions(CmdArgs);

  // Select the appropriate action.
  RewriteKind rewriteKind = RK_None;

  if (isa<AnalyzeJobAction>(JA)) {
    assert(JA.getType() == types::TY_Plist && "Invalid output type.");
    CmdArgs.push_back("-analyze");
  } else if (isa<MigrateJobAction>(JA)) {
    CmdArgs.push_back("-migrate");
  } else if (isa<PreprocessJobAction>(JA)) {
    if (Output.getType() == types::TY_Dependencies)
      CmdArgs.push_back("-Eonly");
    else {
      CmdArgs.push_back("-E");
      if (Args.hasArg(options::OPT_rewrite_objc) &&
          !Args.hasArg(options::OPT_g_Group))
        CmdArgs.push_back("-P");
    }
  } else if (isa<AssembleJobAction>(JA)) {
    CmdArgs.push_back("-emit-obj");

    CollectArgsForIntegratedAssembler(C, Args, CmdArgs, D);

    // Also ignore explicit -force_cpusubtype_ALL option.
    (void)Args.hasArg(options::OPT_force__cpusubtype__ALL);
  } else if (isa<PrecompileJobAction>(JA)) {
    // Use PCH if the user requested it.
    bool UsePCH = D.CCCUsePCH;

    if (JA.getType() == types::TY_Nothing)
      CmdArgs.push_back("-fsyntax-only");
    else if (JA.getType() == types::TY_ModuleFile)
      CmdArgs.push_back("-emit-module-interface");
    else if (UsePCH)
      CmdArgs.push_back("-emit-pch");
    else
      CmdArgs.push_back("-emit-pth");
  } else if (isa<VerifyPCHJobAction>(JA)) {
    CmdArgs.push_back("-verify-pch");
  } else {
    assert((isa<CompileJobAction>(JA) || isa<BackendJobAction>(JA)) &&
           "Invalid action for clang tool.");
    if (JA.getType() == types::TY_Nothing) {
      CmdArgs.push_back("-fsyntax-only");
    } else if (JA.getType() == types::TY_LLVM_IR ||
               JA.getType() == types::TY_LTO_IR) {
      CmdArgs.push_back("-emit-llvm");
    } else if (JA.getType() == types::TY_LLVM_BC ||
               JA.getType() == types::TY_LTO_BC) {
      CmdArgs.push_back("-emit-llvm-bc");
    } else if (JA.getType() == types::TY_PP_Asm) {
      CmdArgs.push_back("-S");
    } else if (JA.getType() == types::TY_AST) {
      CmdArgs.push_back("-emit-pch");
    } else if (JA.getType() == types::TY_ModuleFile) {
      CmdArgs.push_back("-module-file-info");
    } else if (JA.getType() == types::TY_RewrittenObjC) {
      CmdArgs.push_back("-rewrite-objc");
      rewriteKind = RK_NonFragile;
    } else if (JA.getType() == types::TY_RewrittenLegacyObjC) {
      CmdArgs.push_back("-rewrite-objc");
      rewriteKind = RK_Fragile;
    } else {
      assert(JA.getType() == types::TY_PP_Asm && "Unexpected output type!");
    }

    // Preserve use-list order by default when emitting bitcode, so that
    // loading the bitcode up in 'opt' or 'llc' and running passes gives the
    // same result as running passes here.  For LTO, we don't need to preserve
    // the use-list order, since serialization to bitcode is part of the flow.
    if (JA.getType() == types::TY_LLVM_BC)
      CmdArgs.push_back("-emit-llvm-uselists");

    if (D.isUsingLTO())
      Args.AddLastArg(CmdArgs, options::OPT_flto, options::OPT_flto_EQ);
  }

  if (const Arg *A = Args.getLastArg(options::OPT_fthinlto_index_EQ)) {
    if (!types::isLLVMIR(Input.getType()))
      D.Diag(diag::err_drv_argument_only_allowed_with) << A->getAsString(Args)
                                                       << "-x ir";
    Args.AddLastArg(CmdArgs, options::OPT_fthinlto_index_EQ);
  }

  // Embed-bitcode option.
  if (C.getDriver().embedBitcodeEnabled() &&
      (isa<BackendJobAction>(JA) || isa<AssembleJobAction>(JA))) {
    // Add flags implied by -fembed-bitcode.
    Args.AddLastArg(CmdArgs, options::OPT_fembed_bitcode_EQ);
    // Disable all llvm IR level optimizations.
    CmdArgs.push_back("-disable-llvm-optzns");
  }
  if (C.getDriver().embedBitcodeMarkerOnly())
    CmdArgs.push_back("-fembed-bitcode=marker");

  // We normally speed up the clang process a bit by skipping destructors at
  // exit, but when we're generating diagnostics we can rely on some of the
  // cleanup.
  if (!C.isForDiagnostics())
    CmdArgs.push_back("-disable-free");

// Disable the verification pass in -asserts builds.
#ifdef NDEBUG
  CmdArgs.push_back("-disable-llvm-verifier");
  // Discard LLVM value names in -asserts builds.
  CmdArgs.push_back("-discard-value-names");
#endif

  // Set the main file name, so that debug info works even with
  // -save-temps.
  CmdArgs.push_back("-main-file-name");
  CmdArgs.push_back(getBaseInputName(Args, Input));

  // Some flags which affect the language (via preprocessor
  // defines).
  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-static-define");

  if (isa<AnalyzeJobAction>(JA)) {
    // Enable region store model by default.
    CmdArgs.push_back("-analyzer-store=region");

    // Treat blocks as analysis entry points.
    CmdArgs.push_back("-analyzer-opt-analyze-nested-blocks");

    CmdArgs.push_back("-analyzer-eagerly-assume");

    // Add default argument set.
    if (!Args.hasArg(options::OPT__analyzer_no_default_checks)) {
      CmdArgs.push_back("-analyzer-checker=core");

    if (!IsWindowsMSVC) {
      CmdArgs.push_back("-analyzer-checker=unix");
    } else {
      // Enable "unix" checkers that also work on Windows.
      CmdArgs.push_back("-analyzer-checker=unix.API");
      CmdArgs.push_back("-analyzer-checker=unix.Malloc");
      CmdArgs.push_back("-analyzer-checker=unix.MallocSizeof");
      CmdArgs.push_back("-analyzer-checker=unix.MismatchedDeallocator");
      CmdArgs.push_back("-analyzer-checker=unix.cstring.BadSizeArg");
      CmdArgs.push_back("-analyzer-checker=unix.cstring.NullArg");
    }

      // Disable some unix checkers for PS4.
      if (IsPS4CPU) {
        CmdArgs.push_back("-analyzer-disable-checker=unix.API");
        CmdArgs.push_back("-analyzer-disable-checker=unix.Vfork");
      }

      if (getToolChain().getTriple().getVendor() == llvm::Triple::Apple)
        CmdArgs.push_back("-analyzer-checker=osx");

      CmdArgs.push_back("-analyzer-checker=deadcode");

      if (types::isCXX(Input.getType()))
        CmdArgs.push_back("-analyzer-checker=cplusplus");

      if (!IsPS4CPU) {
        CmdArgs.push_back(
            "-analyzer-checker=security.insecureAPI.UncheckedReturn");
        CmdArgs.push_back("-analyzer-checker=security.insecureAPI.getpw");
        CmdArgs.push_back("-analyzer-checker=security.insecureAPI.gets");
        CmdArgs.push_back("-analyzer-checker=security.insecureAPI.mktemp");
        CmdArgs.push_back("-analyzer-checker=security.insecureAPI.mkstemp");
        CmdArgs.push_back("-analyzer-checker=security.insecureAPI.vfork");
      }

      // Default nullability checks.
      CmdArgs.push_back("-analyzer-checker=nullability.NullPassedToNonnull");
      CmdArgs.push_back(
          "-analyzer-checker=nullability.NullReturnedFromNonnull");
    }

    // Set the output format. The default is plist, for (lame) historical
    // reasons.
    CmdArgs.push_back("-analyzer-output");
    if (Arg *A = Args.getLastArg(options::OPT__analyzer_output))
      CmdArgs.push_back(A->getValue());
    else
      CmdArgs.push_back("plist");

    // Disable the presentation of standard compiler warnings when
    // using --analyze.  We only want to show static analyzer diagnostics
    // or frontend errors.
    CmdArgs.push_back("-w");

    // Add -Xanalyzer arguments when running as analyzer.
    Args.AddAllArgValues(CmdArgs, options::OPT_Xanalyzer);
  }

  CheckCodeGenerationOptions(D, Args);

  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) =
      ParsePICArgs(getToolChain(), Triple, Args);

  const char *RMName = RelocationModelName(RelocationModel);

  if ((RelocationModel == llvm::Reloc::ROPI ||
       RelocationModel == llvm::Reloc::ROPI_RWPI) &&
      types::isCXX(Input.getType()) &&
      !Args.hasArg(options::OPT_fallow_unsupported))
    D.Diag(diag::err_drv_ropi_incompatible_with_cxx);

  if (RMName) {
    CmdArgs.push_back("-mrelocation-model");
    CmdArgs.push_back(RMName);
  }
  if (PICLevel > 0) {
    CmdArgs.push_back("-pic-level");
    CmdArgs.push_back(PICLevel == 1 ? "1" : "2");
    if (IsPIE)
      CmdArgs.push_back("-pic-is-pie");
  }

  if (Arg *A = Args.getLastArg(options::OPT_meabi)) {
    CmdArgs.push_back("-meabi");
    CmdArgs.push_back(A->getValue());
  }

  CmdArgs.push_back("-mthread-model");
  if (Arg *A = Args.getLastArg(options::OPT_mthread_model))
    CmdArgs.push_back(A->getValue());
  else
    CmdArgs.push_back(Args.MakeArgString(getToolChain().getThreadModel()));

  Args.AddLastArg(CmdArgs, options::OPT_fveclib);

  if (!Args.hasFlag(options::OPT_fmerge_all_constants,
                    options::OPT_fno_merge_all_constants))
    CmdArgs.push_back("-fno-merge-all-constants");

  // LLVM Code Generator Options.

  if (Args.hasArg(options::OPT_frewrite_map_file) ||
      Args.hasArg(options::OPT_frewrite_map_file_EQ)) {
    for (const Arg *A : Args.filtered(options::OPT_frewrite_map_file,
                                      options::OPT_frewrite_map_file_EQ)) {
      StringRef Map = A->getValue();
      if (!llvm::sys::fs::exists(Map)) {
        D.Diag(diag::err_drv_no_such_file) << Map;
      } else {
        CmdArgs.push_back("-frewrite-map-file");
        CmdArgs.push_back(A->getValue());
        A->claim();
      }
    }
  }

  if (Arg *A = Args.getLastArg(options::OPT_Wframe_larger_than_EQ)) {
    StringRef v = A->getValue();
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back(Args.MakeArgString("-warn-stack-size=" + v));
    A->claim();
  }

  if (!Args.hasFlag(options::OPT_fjump_tables, options::OPT_fno_jump_tables,
                    true))
    CmdArgs.push_back("-fno-jump-tables");

  if (!Args.hasFlag(options::OPT_fpreserve_as_comments,
                    options::OPT_fno_preserve_as_comments, true))
    CmdArgs.push_back("-fno-preserve-as-comments");

  if (Arg *A = Args.getLastArg(options::OPT_mregparm_EQ)) {
    CmdArgs.push_back("-mregparm");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_fpcc_struct_return,
                               options::OPT_freg_struct_return)) {
    if (getToolChain().getArch() != llvm::Triple::x86) {
      D.Diag(diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << getToolChain().getTriple().str();
    } else if (A->getOption().matches(options::OPT_fpcc_struct_return)) {
      CmdArgs.push_back("-fpcc-struct-return");
    } else {
      assert(A->getOption().matches(options::OPT_freg_struct_return));
      CmdArgs.push_back("-freg-struct-return");
    }
  }

  if (Args.hasFlag(options::OPT_mrtd, options::OPT_mno_rtd, false))
    CmdArgs.push_back("-fdefault-calling-conv=stdcall");

  if (shouldUseFramePointer(Args, getToolChain().getTriple()))
    CmdArgs.push_back("-mdisable-fp-elim");
  if (!Args.hasFlag(options::OPT_fzero_initialized_in_bss,
                    options::OPT_fno_zero_initialized_in_bss))
    CmdArgs.push_back("-mno-zero-initialized-in-bss");

  bool OFastEnabled = isOptimizationLevelFast(Args);
  // If -Ofast is the optimization level, then -fstrict-aliasing should be
  // enabled.  This alias option is being used to simplify the hasFlag logic.
  OptSpecifier StrictAliasingAliasOption =
      OFastEnabled ? options::OPT_Ofast : options::OPT_fstrict_aliasing;
  // We turn strict aliasing off by default if we're in CL mode, since MSVC
  // doesn't do any TBAA.
  bool TBAAOnByDefault = !getToolChain().getDriver().IsCLMode();
  if (!Args.hasFlag(options::OPT_fstrict_aliasing, StrictAliasingAliasOption,
                    options::OPT_fno_strict_aliasing, TBAAOnByDefault))
    CmdArgs.push_back("-relaxed-aliasing");
  if (!Args.hasFlag(options::OPT_fstruct_path_tbaa,
                    options::OPT_fno_struct_path_tbaa))
    CmdArgs.push_back("-no-struct-path-tbaa");
  if (Args.hasFlag(options::OPT_fstrict_enums, options::OPT_fno_strict_enums,
                   false))
    CmdArgs.push_back("-fstrict-enums");
  if (Args.hasFlag(options::OPT_fstrict_vtable_pointers,
                   options::OPT_fno_strict_vtable_pointers,
                   false))
    CmdArgs.push_back("-fstrict-vtable-pointers");
  if (!Args.hasFlag(options::OPT_foptimize_sibling_calls,
                    options::OPT_fno_optimize_sibling_calls))
    CmdArgs.push_back("-mdisable-tail-calls");

  // Handle segmented stacks.
  if (Args.hasArg(options::OPT_fsplit_stack))
    CmdArgs.push_back("-split-stacks");

  // If -Ofast is the optimization level, then -ffast-math should be enabled.
  // This alias option is being used to simplify the getLastArg logic.
  OptSpecifier FastMathAliasOption =
      OFastEnabled ? options::OPT_Ofast : options::OPT_ffast_math;

  // Handle various floating point optimization flags, mapping them to the
  // appropriate LLVM code generation flags. The pattern for all of these is to
  // default off the codegen optimizations, and if any flag enables them and no
  // flag disables them after the flag enabling them, enable the codegen
  // optimization. This is complicated by several "umbrella" flags.
  if (Arg *A = Args.getLastArg(
          options::OPT_ffast_math, FastMathAliasOption,
          options::OPT_fno_fast_math, options::OPT_ffinite_math_only,
          options::OPT_fno_finite_math_only, options::OPT_fhonor_infinities,
          options::OPT_fno_honor_infinities))
    if (A->getOption().getID() != options::OPT_fno_fast_math &&
        A->getOption().getID() != options::OPT_fno_finite_math_only &&
        A->getOption().getID() != options::OPT_fhonor_infinities)
      CmdArgs.push_back("-menable-no-infs");
  if (Arg *A = Args.getLastArg(
          options::OPT_ffast_math, FastMathAliasOption,
          options::OPT_fno_fast_math, options::OPT_ffinite_math_only,
          options::OPT_fno_finite_math_only, options::OPT_fhonor_nans,
          options::OPT_fno_honor_nans))
    if (A->getOption().getID() != options::OPT_fno_fast_math &&
        A->getOption().getID() != options::OPT_fno_finite_math_only &&
        A->getOption().getID() != options::OPT_fhonor_nans)
      CmdArgs.push_back("-menable-no-nans");

  // -fmath-errno is the default on some platforms, e.g. BSD-derived OSes.
  bool MathErrno = getToolChain().IsMathErrnoDefault();
  if (Arg *A =
          Args.getLastArg(options::OPT_ffast_math, FastMathAliasOption,
                          options::OPT_fno_fast_math, options::OPT_fmath_errno,
                          options::OPT_fno_math_errno)) {
    // Turning on -ffast_math (with either flag) removes the need for MathErrno.
    // However, turning *off* -ffast_math merely restores the toolchain default
    // (which may be false).
    if (A->getOption().getID() == options::OPT_fno_math_errno ||
        A->getOption().getID() == options::OPT_ffast_math ||
        A->getOption().getID() == options::OPT_Ofast)
      MathErrno = false;
    else if (A->getOption().getID() == options::OPT_fmath_errno)
      MathErrno = true;
  }
  if (MathErrno)
    CmdArgs.push_back("-fmath-errno");

  // There are several flags which require disabling very specific
  // optimizations. Any of these being disabled forces us to turn off the
  // entire set of LLVM optimizations, so collect them through all the flag
  // madness.
  bool AssociativeMath = false;
  if (Arg *A = Args.getLastArg(
          options::OPT_ffast_math, FastMathAliasOption,
          options::OPT_fno_fast_math, options::OPT_funsafe_math_optimizations,
          options::OPT_fno_unsafe_math_optimizations,
          options::OPT_fassociative_math, options::OPT_fno_associative_math))
    if (A->getOption().getID() != options::OPT_fno_fast_math &&
        A->getOption().getID() != options::OPT_fno_unsafe_math_optimizations &&
        A->getOption().getID() != options::OPT_fno_associative_math)
      AssociativeMath = true;
  bool ReciprocalMath = false;
  if (Arg *A = Args.getLastArg(
          options::OPT_ffast_math, FastMathAliasOption,
          options::OPT_fno_fast_math, options::OPT_funsafe_math_optimizations,
          options::OPT_fno_unsafe_math_optimizations,
          options::OPT_freciprocal_math, options::OPT_fno_reciprocal_math))
    if (A->getOption().getID() != options::OPT_fno_fast_math &&
        A->getOption().getID() != options::OPT_fno_unsafe_math_optimizations &&
        A->getOption().getID() != options::OPT_fno_reciprocal_math)
      ReciprocalMath = true;
  bool SignedZeros = true;
  if (Arg *A = Args.getLastArg(
          options::OPT_ffast_math, FastMathAliasOption,
          options::OPT_fno_fast_math, options::OPT_funsafe_math_optimizations,
          options::OPT_fno_unsafe_math_optimizations,
          options::OPT_fsigned_zeros, options::OPT_fno_signed_zeros))
    if (A->getOption().getID() != options::OPT_fno_fast_math &&
        A->getOption().getID() != options::OPT_fno_unsafe_math_optimizations &&
        A->getOption().getID() != options::OPT_fsigned_zeros)
      SignedZeros = false;
  bool TrappingMath = true;
  if (Arg *A = Args.getLastArg(
          options::OPT_ffast_math, FastMathAliasOption,
          options::OPT_fno_fast_math, options::OPT_funsafe_math_optimizations,
          options::OPT_fno_unsafe_math_optimizations,
          options::OPT_ftrapping_math, options::OPT_fno_trapping_math))
    if (A->getOption().getID() != options::OPT_fno_fast_math &&
        A->getOption().getID() != options::OPT_fno_unsafe_math_optimizations &&
        A->getOption().getID() != options::OPT_ftrapping_math)
      TrappingMath = false;
  if (!MathErrno && AssociativeMath && ReciprocalMath && !SignedZeros &&
      !TrappingMath)
    CmdArgs.push_back("-menable-unsafe-fp-math");

  if (!SignedZeros)
    CmdArgs.push_back("-fno-signed-zeros");

  if (ReciprocalMath)
    CmdArgs.push_back("-freciprocal-math");

  if (!TrappingMath) 
    CmdArgs.push_back("-fno-trapping-math");

  if (Args.hasArg(options::OPT_fdenormal_fp_math_EQ))
    Args.AddLastArg(CmdArgs, options::OPT_fdenormal_fp_math_EQ);

  // Validate and pass through -fp-contract option.
  if (Arg *A = Args.getLastArg(options::OPT_ffast_math, FastMathAliasOption,
                               options::OPT_fno_fast_math,
                               options::OPT_ffp_contract)) {
    if (A->getOption().getID() == options::OPT_ffp_contract) {
      StringRef Val = A->getValue();
      if (Val == "fast" || Val == "on" || Val == "off") {
        CmdArgs.push_back(Args.MakeArgString("-ffp-contract=" + Val));
      } else {
        D.Diag(diag::err_drv_unsupported_option_argument)
            << A->getOption().getName() << Val;
      }
    } else if (A->getOption().matches(options::OPT_ffast_math) ||
               (OFastEnabled && A->getOption().matches(options::OPT_Ofast))) {
      // If fast-math is set then set the fp-contract mode to fast.
      CmdArgs.push_back(Args.MakeArgString("-ffp-contract=fast"));
    }
  }

  ParseMRecip(getToolChain().getDriver(), Args, CmdArgs);

  // We separately look for the '-ffast-math' and '-ffinite-math-only' flags,
  // and if we find them, tell the frontend to provide the appropriate
  // preprocessor macros. This is distinct from enabling any optimizations as
  // these options induce language changes which must survive serialization
  // and deserialization, etc.
  if (Arg *A = Args.getLastArg(options::OPT_ffast_math, FastMathAliasOption,
                               options::OPT_fno_fast_math))
    if (!A->getOption().matches(options::OPT_fno_fast_math))
      CmdArgs.push_back("-ffast-math");
  if (Arg *A = Args.getLastArg(options::OPT_ffinite_math_only,
                               options::OPT_fno_fast_math))
    if (A->getOption().matches(options::OPT_ffinite_math_only))
      CmdArgs.push_back("-ffinite-math-only");

  // Decide whether to use verbose asm. Verbose assembly is the default on
  // toolchains which have the integrated assembler on by default.
  bool IsIntegratedAssemblerDefault =
      getToolChain().IsIntegratedAssemblerDefault();
  if (Args.hasFlag(options::OPT_fverbose_asm, options::OPT_fno_verbose_asm,
                   IsIntegratedAssemblerDefault) ||
      Args.hasArg(options::OPT_dA))
    CmdArgs.push_back("-masm-verbose");

  if (!Args.hasFlag(options::OPT_fintegrated_as, options::OPT_fno_integrated_as,
                    IsIntegratedAssemblerDefault))
    CmdArgs.push_back("-no-integrated-as");

  if (Args.hasArg(options::OPT_fdebug_pass_structure)) {
    CmdArgs.push_back("-mdebug-pass");
    CmdArgs.push_back("Structure");
  }
  if (Args.hasArg(options::OPT_fdebug_pass_arguments)) {
    CmdArgs.push_back("-mdebug-pass");
    CmdArgs.push_back("Arguments");
  }

  // Enable -mconstructor-aliases except on darwin, where we have to work around
  // a linker bug (see <rdar://problem/7651567>), and CUDA device code, where
  // aliases aren't supported.
  if (!getToolChain().getTriple().isOSDarwin() &&
      !getToolChain().getTriple().isNVPTX())
    CmdArgs.push_back("-mconstructor-aliases");

  // Darwin's kernel doesn't support guard variables; just die if we
  // try to use them.
  if (KernelOrKext && getToolChain().getTriple().isOSDarwin())
    CmdArgs.push_back("-fforbid-guard-variables");

  if (Args.hasFlag(options::OPT_mms_bitfields, options::OPT_mno_ms_bitfields,
                   false)) {
    CmdArgs.push_back("-mms-bitfields");
  }

  // This is a coarse approximation of what llvm-gcc actually does, both
  // -fasynchronous-unwind-tables and -fnon-call-exceptions interact in more
  // complicated ways.
  bool AsynchronousUnwindTables =
      Args.hasFlag(options::OPT_fasynchronous_unwind_tables,
                   options::OPT_fno_asynchronous_unwind_tables,
                   (getToolChain().IsUnwindTablesDefault() ||
                    getToolChain().getSanitizerArgs().needsUnwindTables()) &&
                       !KernelOrKext);
  if (Args.hasFlag(options::OPT_funwind_tables, options::OPT_fno_unwind_tables,
                   AsynchronousUnwindTables))
    CmdArgs.push_back("-munwind-tables");

  getToolChain().addClangTargetOptions(Args, CmdArgs);

  if (Arg *A = Args.getLastArg(options::OPT_flimited_precision_EQ)) {
    CmdArgs.push_back("-mlimit-float-precision");
    CmdArgs.push_back(A->getValue());
  }

  // FIXME: Handle -mtune=.
  (void)Args.hasArg(options::OPT_mtune_EQ);

  if (Arg *A = Args.getLastArg(options::OPT_mcmodel_EQ)) {
    CmdArgs.push_back("-mcode-model");
    CmdArgs.push_back(A->getValue());
  }

  // Add the target cpu
  std::string CPU = getCPUName(Args, Triple, /*FromAs*/ false);
  if (!CPU.empty()) {
    CmdArgs.push_back("-target-cpu");
    CmdArgs.push_back(Args.MakeArgString(CPU));
  }

  if (const Arg *A = Args.getLastArg(options::OPT_mfpmath_EQ)) {
    CmdArgs.push_back("-mfpmath");
    CmdArgs.push_back(A->getValue());
  }

  // Add the target features
  getTargetFeatures(getToolChain(), Triple, Args, CmdArgs, false);

  // Add target specific flags.
  switch (getToolChain().getArch()) {
  default:
    break;

  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    // Use the effective triple, which takes into account the deployment target.
    AddARMTargetArgs(Triple, Args, CmdArgs, KernelOrKext);
    break;

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    AddAArch64TargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    AddMIPSTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    AddPPCTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::sparcv9:
    AddSparcTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::systemz:
    AddSystemZTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    AddX86TargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::lanai:
    AddLanaiTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::hexagon:
    AddHexagonTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    AddWebAssemblyTargetArgs(Args, CmdArgs);
    break;
  }

  // The 'g' groups options involve a somewhat intricate sequence of decisions
  // about what to pass from the driver to the frontend, but by the time they
  // reach cc1 they've been factored into three well-defined orthogonal choices:
  //  * what level of debug info to generate
  //  * what dwarf version to write
  //  * what debugger tuning to use
  // This avoids having to monkey around further in cc1 other than to disable
  // codeview if not running in a Windows environment. Perhaps even that
  // decision should be made in the driver as well though.
  unsigned DwarfVersion = 0;
  llvm::DebuggerKind DebuggerTuning = getToolChain().getDefaultDebuggerTuning();
  // These two are potentially updated by AddClangCLArgs.
  codegenoptions::DebugInfoKind DebugInfoKind = codegenoptions::NoDebugInfo;
  bool EmitCodeView = false;

  // Add clang-cl arguments.
  types::ID InputType = Input.getType();
  if (getToolChain().getDriver().IsCLMode())
    AddClangCLArgs(Args, InputType, CmdArgs, &DebugInfoKind, &EmitCodeView);

  // Pass the linker version in use.
  if (Arg *A = Args.getLastArg(options::OPT_mlinker_version_EQ)) {
    CmdArgs.push_back("-target-linker-version");
    CmdArgs.push_back(A->getValue());
  }

  if (!shouldUseLeafFramePointer(Args, getToolChain().getTriple()))
    CmdArgs.push_back("-momit-leaf-frame-pointer");

  // Explicitly error on some things we know we don't support and can't just
  // ignore.
  if (!Args.hasArg(options::OPT_fallow_unsupported)) {
    Arg *Unsupported;
    if (types::isCXX(InputType) && getToolChain().getTriple().isOSDarwin() &&
        getToolChain().getArch() == llvm::Triple::x86) {
      if ((Unsupported = Args.getLastArg(options::OPT_fapple_kext)) ||
          (Unsupported = Args.getLastArg(options::OPT_mkernel)))
        D.Diag(diag::err_drv_clang_unsupported_opt_cxx_darwin_i386)
            << Unsupported->getOption().getName();
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_v);
  Args.AddLastArg(CmdArgs, options::OPT_H);
  if (D.CCPrintHeaders && !D.CCGenDiagnostics) {
    CmdArgs.push_back("-header-include-file");
    CmdArgs.push_back(D.CCPrintHeadersFilename ? D.CCPrintHeadersFilename
                                               : "-");
  }
  Args.AddLastArg(CmdArgs, options::OPT_P);
  Args.AddLastArg(CmdArgs, options::OPT_print_ivar_layout);

  if (D.CCLogDiagnostics && !D.CCGenDiagnostics) {
    CmdArgs.push_back("-diagnostic-log-file");
    CmdArgs.push_back(D.CCLogDiagnosticsFilename ? D.CCLogDiagnosticsFilename
                                                 : "-");
  }

  bool splitDwarfInlining =
      Args.hasFlag(options::OPT_fsplit_dwarf_inlining,
                   options::OPT_fno_split_dwarf_inlining, true);

  Args.ClaimAllArgs(options::OPT_g_Group);
  Arg *SplitDwarfArg = Args.getLastArg(options::OPT_gsplit_dwarf);
  if (Arg *A = Args.getLastArg(options::OPT_g_Group)) {
    // If the last option explicitly specified a debug-info level, use it.
    if (A->getOption().matches(options::OPT_gN_Group)) {
      DebugInfoKind = DebugLevelToInfoKind(*A);
      // If you say "-gsplit-dwarf -gline-tables-only", -gsplit-dwarf loses.
      // But -gsplit-dwarf is not a g_group option, hence we have to check the
      // order explicitly. (If -gsplit-dwarf wins, we fix DebugInfoKind later.)
      // This gets a bit more complicated if you've disabled inline info in the
      // skeleton CUs (splitDwarfInlining) - then there's value in composing
      // split-dwarf and line-tables-only, so let those compose naturally in
      // that case.
      // And if you just turned off debug info, (-gsplit-dwarf -g0) - do that.
      if (SplitDwarfArg) {
        if (A->getIndex() > SplitDwarfArg->getIndex()) {
          if (DebugInfoKind == codegenoptions::NoDebugInfo ||
              (DebugInfoKind == codegenoptions::DebugLineTablesOnly &&
               splitDwarfInlining))
            SplitDwarfArg = nullptr;
        } else if (splitDwarfInlining)
          DebugInfoKind = codegenoptions::NoDebugInfo;
      }
    } else
      // For any other 'g' option, use Limited.
      DebugInfoKind = codegenoptions::LimitedDebugInfo;
  }

  // If a debugger tuning argument appeared, remember it.
  if (Arg *A = Args.getLastArg(options::OPT_gTune_Group,
                               options::OPT_ggdbN_Group)) {
    if (A->getOption().matches(options::OPT_glldb))
      DebuggerTuning = llvm::DebuggerKind::LLDB;
    else if (A->getOption().matches(options::OPT_gsce))
      DebuggerTuning = llvm::DebuggerKind::SCE;
    else
      DebuggerTuning = llvm::DebuggerKind::GDB;
  }

  // If a -gdwarf argument appeared, remember it.
  if (Arg *A = Args.getLastArg(options::OPT_gdwarf_2, options::OPT_gdwarf_3,
                               options::OPT_gdwarf_4, options::OPT_gdwarf_5))
    DwarfVersion = DwarfVersionNum(A->getSpelling());

  // Forward -gcodeview. EmitCodeView might have been set by CL-compatibility
  // argument parsing.
  if (Args.hasArg(options::OPT_gcodeview) || EmitCodeView) {
    // DwarfVersion remains at 0 if no explicit choice was made.
    CmdArgs.push_back("-gcodeview");
  } else if (DwarfVersion == 0 &&
             DebugInfoKind != codegenoptions::NoDebugInfo) {
    DwarfVersion = getToolChain().GetDefaultDwarfVersion();
  }

  // We ignore flags -gstrict-dwarf and -grecord-gcc-switches for now.
  Args.ClaimAllArgs(options::OPT_g_flags_Group);

  // Column info is included by default for everything except PS4 and CodeView.
  // Clang doesn't track end columns, just starting columns, which, in theory,
  // is fine for CodeView (and PDB).  In practice, however, the Microsoft
  // debuggers don't handle missing end columns well, so it's better not to
  // include any column info.
  if (Args.hasFlag(options::OPT_gcolumn_info, options::OPT_gno_column_info,
                   /*Default=*/ !IsPS4CPU && !(IsWindowsMSVC && EmitCodeView)))
    CmdArgs.push_back("-dwarf-column-info");

  // FIXME: Move backend command line options to the module.
  if (Args.hasArg(options::OPT_gmodules)) {
    DebugInfoKind = codegenoptions::LimitedDebugInfo;
    CmdArgs.push_back("-dwarf-ext-refs");
    CmdArgs.push_back("-fmodule-format=obj");
  }

  // -gsplit-dwarf should turn on -g and enable the backend dwarf
  // splitting and extraction.
  // FIXME: Currently only works on Linux.
  if (getToolChain().getTriple().isOSLinux() && SplitDwarfArg) {
    if (!splitDwarfInlining)
      CmdArgs.push_back("-fno-split-dwarf-inlining");
    if (DebugInfoKind == codegenoptions::NoDebugInfo)
      DebugInfoKind = codegenoptions::LimitedDebugInfo;
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-split-dwarf=Enable");
  }

  // After we've dealt with all combinations of things that could
  // make DebugInfoKind be other than None or DebugLineTablesOnly,
  // figure out if we need to "upgrade" it to standalone debug info.
  // We parse these two '-f' options whether or not they will be used,
  // to claim them even if you wrote "-fstandalone-debug -gline-tables-only"
  bool NeedFullDebug = Args.hasFlag(options::OPT_fstandalone_debug,
                                    options::OPT_fno_standalone_debug,
                                    getToolChain().GetDefaultStandaloneDebug());
  if (DebugInfoKind == codegenoptions::LimitedDebugInfo && NeedFullDebug)
    DebugInfoKind = codegenoptions::FullDebugInfo;
  RenderDebugEnablingArgs(Args, CmdArgs, DebugInfoKind, DwarfVersion,
                          DebuggerTuning);

  // -ggnu-pubnames turns on gnu style pubnames in the backend.
  if (Args.hasArg(options::OPT_ggnu_pubnames)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-generate-gnu-dwarf-pub-sections");
  }

  // -gdwarf-aranges turns on the emission of the aranges section in the
  // backend.
  // Always enabled on the PS4.
  if (Args.hasArg(options::OPT_gdwarf_aranges) || IsPS4CPU) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-generate-arange-section");
  }

  if (Args.hasFlag(options::OPT_fdebug_types_section,
                   options::OPT_fno_debug_types_section, false)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-generate-type-units");
  }

  // CloudABI and WebAssembly use -ffunction-sections and -fdata-sections by
  // default.
  bool UseSeparateSections = Triple.getOS() == llvm::Triple::CloudABI ||
                             Triple.getArch() == llvm::Triple::wasm32 ||
                             Triple.getArch() == llvm::Triple::wasm64;

  if (Args.hasFlag(options::OPT_ffunction_sections,
                   options::OPT_fno_function_sections, UseSeparateSections)) {
    CmdArgs.push_back("-ffunction-sections");
  }

  if (Args.hasFlag(options::OPT_fdata_sections, options::OPT_fno_data_sections,
                   UseSeparateSections)) {
    CmdArgs.push_back("-fdata-sections");
  }

  if (!Args.hasFlag(options::OPT_funique_section_names,
                    options::OPT_fno_unique_section_names, true))
    CmdArgs.push_back("-fno-unique-section-names");

  Args.AddAllArgs(CmdArgs, options::OPT_finstrument_functions);

  if (Args.hasFlag(options::OPT_fxray_instrument,
                   options::OPT_fnoxray_instrument, false)) {
    CmdArgs.push_back("-fxray-instrument");
    if (const Arg *A =
            Args.getLastArg(options::OPT_fxray_instruction_threshold_,
                            options::OPT_fxray_instruction_threshold_EQ)) {
      CmdArgs.push_back("-fxray-instruction-threshold");
      CmdArgs.push_back(A->getValue());
    }
  }

  addPGOAndCoverageFlags(C, D, Output, Args, CmdArgs);

  // Add runtime flag for PS4 when PGO or Coverage are enabled.
  if (getToolChain().getTriple().isPS4CPU())
    addPS4ProfileRTArgs(getToolChain(), Args, CmdArgs);

  // Pass options for controlling the default header search paths.
  if (Args.hasArg(options::OPT_nostdinc)) {
    CmdArgs.push_back("-nostdsysteminc");
    CmdArgs.push_back("-nobuiltininc");
  } else {
    if (Args.hasArg(options::OPT_nostdlibinc))
      CmdArgs.push_back("-nostdsysteminc");
    Args.AddLastArg(CmdArgs, options::OPT_nostdincxx);
    Args.AddLastArg(CmdArgs, options::OPT_nobuiltininc);
  }

  // Pass the path to compiler resource files.
  CmdArgs.push_back("-resource-dir");
  CmdArgs.push_back(D.ResourceDir.c_str());

  Args.AddLastArg(CmdArgs, options::OPT_working_directory);

  bool ARCMTEnabled = false;
  if (!Args.hasArg(options::OPT_fno_objc_arc, options::OPT_fobjc_arc)) {
    if (const Arg *A = Args.getLastArg(options::OPT_ccc_arcmt_check,
                                       options::OPT_ccc_arcmt_modify,
                                       options::OPT_ccc_arcmt_migrate)) {
      ARCMTEnabled = true;
      switch (A->getOption().getID()) {
      default:
        llvm_unreachable("missed a case");
      case options::OPT_ccc_arcmt_check:
        CmdArgs.push_back("-arcmt-check");
        break;
      case options::OPT_ccc_arcmt_modify:
        CmdArgs.push_back("-arcmt-modify");
        break;
      case options::OPT_ccc_arcmt_migrate:
        CmdArgs.push_back("-arcmt-migrate");
        CmdArgs.push_back("-mt-migrate-directory");
        CmdArgs.push_back(A->getValue());

        Args.AddLastArg(CmdArgs, options::OPT_arcmt_migrate_report_output);
        Args.AddLastArg(CmdArgs, options::OPT_arcmt_migrate_emit_arc_errors);
        break;
      }
    }
  } else {
    Args.ClaimAllArgs(options::OPT_ccc_arcmt_check);
    Args.ClaimAllArgs(options::OPT_ccc_arcmt_modify);
    Args.ClaimAllArgs(options::OPT_ccc_arcmt_migrate);
  }

  if (const Arg *A = Args.getLastArg(options::OPT_ccc_objcmt_migrate)) {
    if (ARCMTEnabled) {
      D.Diag(diag::err_drv_argument_not_allowed_with) << A->getAsString(Args)
                                                      << "-ccc-arcmt-migrate";
    }
    CmdArgs.push_back("-mt-migrate-directory");
    CmdArgs.push_back(A->getValue());

    if (!Args.hasArg(options::OPT_objcmt_migrate_literals,
                     options::OPT_objcmt_migrate_subscripting,
                     options::OPT_objcmt_migrate_property)) {
      // None specified, means enable them all.
      CmdArgs.push_back("-objcmt-migrate-literals");
      CmdArgs.push_back("-objcmt-migrate-subscripting");
      CmdArgs.push_back("-objcmt-migrate-property");
    } else {
      Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_literals);
      Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_subscripting);
      Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_property);
    }
  } else {
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_literals);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_subscripting);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_property);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_all);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_readonly_property);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_readwrite_property);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_property_dot_syntax);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_annotation);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_instancetype);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_nsmacros);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_protocol_conformance);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_atomic_property);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_returns_innerpointer_property);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_ns_nonatomic_iosonly);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_migrate_designated_init);
    Args.AddLastArg(CmdArgs, options::OPT_objcmt_whitelist_dir_path);
  }

  // Add preprocessing options like -I, -D, etc. if we are using the
  // preprocessor.
  //
  // FIXME: Support -fpreprocessed
  if (types::getPreprocessedType(InputType) != types::TY_INVALID)
    AddPreprocessingOptions(C, JA, D, Args, CmdArgs, Output, Inputs);

  // Don't warn about "clang -c -DPIC -fPIC test.i" because libtool.m4 assumes
  // that "The compiler can only warn and ignore the option if not recognized".
  // When building with ccache, it will pass -D options to clang even on
  // preprocessed inputs and configure concludes that -fPIC is not supported.
  Args.ClaimAllArgs(options::OPT_D);

  // Manually translate -O4 to -O3; let clang reject others.
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O4)) {
      CmdArgs.push_back("-O3");
      D.Diag(diag::warn_O4_is_O3);
    } else {
      A->render(Args, CmdArgs);
    }
  }

  // Warn about ignored options to clang.
  for (const Arg *A :
       Args.filtered(options::OPT_clang_ignored_gcc_optimization_f_Group)) {
    D.Diag(diag::warn_ignored_gcc_optimization) << A->getAsString(Args);
    A->claim();
  }

  claimNoWarnArgs(Args);

  Args.AddAllArgs(CmdArgs, options::OPT_R_Group);

  Args.AddAllArgs(CmdArgs, options::OPT_W_Group);
  if (Args.hasFlag(options::OPT_pedantic, options::OPT_no_pedantic, false))
    CmdArgs.push_back("-pedantic");
  Args.AddLastArg(CmdArgs, options::OPT_pedantic_errors);
  Args.AddLastArg(CmdArgs, options::OPT_w);

  // Handle -{std, ansi, trigraphs} -- take the last of -{std, ansi}
  // (-ansi is equivalent to -std=c89 or -std=c++98).
  //
  // If a std is supplied, only add -trigraphs if it follows the
  // option.
  bool ImplyVCPPCXXVer = false;
  if (Arg *Std = Args.getLastArg(options::OPT_std_EQ, options::OPT_ansi)) {
    if (Std->getOption().matches(options::OPT_ansi))
      if (types::isCXX(InputType))
        CmdArgs.push_back("-std=c++98");
      else
        CmdArgs.push_back("-std=c89");
    else
      Std->render(Args, CmdArgs);

    // If -f(no-)trigraphs appears after the language standard flag, honor it.
    if (Arg *A = Args.getLastArg(options::OPT_std_EQ, options::OPT_ansi,
                                 options::OPT_ftrigraphs,
                                 options::OPT_fno_trigraphs))
      if (A != Std)
        A->render(Args, CmdArgs);
  } else {
    // Honor -std-default.
    //
    // FIXME: Clang doesn't correctly handle -std= when the input language
    // doesn't match. For the time being just ignore this for C++ inputs;
    // eventually we want to do all the standard defaulting here instead of
    // splitting it between the driver and clang -cc1.
    if (!types::isCXX(InputType))
      Args.AddAllArgsTranslated(CmdArgs, options::OPT_std_default_EQ, "-std=",
                                /*Joined=*/true);
    else if (IsWindowsMSVC)
      ImplyVCPPCXXVer = true;

    Args.AddLastArg(CmdArgs, options::OPT_ftrigraphs,
                    options::OPT_fno_trigraphs);
  }

  // GCC's behavior for -Wwrite-strings is a bit strange:
  //  * In C, this "warning flag" changes the types of string literals from
  //    'char[N]' to 'const char[N]', and thus triggers an unrelated warning
  //    for the discarded qualifier.
  //  * In C++, this is just a normal warning flag.
  //
  // Implementing this warning correctly in C is hard, so we follow GCC's
  // behavior for now. FIXME: Directly diagnose uses of a string literal as
  // a non-const char* in C, rather than using this crude hack.
  if (!types::isCXX(InputType)) {
    // FIXME: This should behave just like a warning flag, and thus should also
    // respect -Weverything, -Wno-everything, -Werror=write-strings, and so on.
    Arg *WriteStrings =
        Args.getLastArg(options::OPT_Wwrite_strings,
                        options::OPT_Wno_write_strings, options::OPT_w);
    if (WriteStrings &&
        WriteStrings->getOption().matches(options::OPT_Wwrite_strings))
      CmdArgs.push_back("-fconst-strings");
  }

  // GCC provides a macro definition '__DEPRECATED' when -Wdeprecated is active
  // during C++ compilation, which it is by default. GCC keeps this define even
  // in the presence of '-w', match this behavior bug-for-bug.
  if (types::isCXX(InputType) &&
      Args.hasFlag(options::OPT_Wdeprecated, options::OPT_Wno_deprecated,
                   true)) {
    CmdArgs.push_back("-fdeprecated-macro");
  }

  // Translate GCC's misnamer '-fasm' arguments to '-fgnu-keywords'.
  if (Arg *Asm = Args.getLastArg(options::OPT_fasm, options::OPT_fno_asm)) {
    if (Asm->getOption().matches(options::OPT_fasm))
      CmdArgs.push_back("-fgnu-keywords");
    else
      CmdArgs.push_back("-fno-gnu-keywords");
  }

  if (ShouldDisableDwarfDirectory(Args, getToolChain()))
    CmdArgs.push_back("-fno-dwarf-directory-asm");

  if (ShouldDisableAutolink(Args, getToolChain()))
    CmdArgs.push_back("-fno-autolink");

  // Add in -fdebug-compilation-dir if necessary.
  addDebugCompDirArg(Args, CmdArgs);

  for (const Arg *A : Args.filtered(options::OPT_fdebug_prefix_map_EQ)) {
    StringRef Map = A->getValue();
    if (Map.find('=') == StringRef::npos)
      D.Diag(diag::err_drv_invalid_argument_to_fdebug_prefix_map) << Map;
    else
      CmdArgs.push_back(Args.MakeArgString("-fdebug-prefix-map=" + Map));
    A->claim();
  }

  if (Arg *A = Args.getLastArg(options::OPT_ftemplate_depth_,
                               options::OPT_ftemplate_depth_EQ)) {
    CmdArgs.push_back("-ftemplate-depth");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_foperator_arrow_depth_EQ)) {
    CmdArgs.push_back("-foperator-arrow-depth");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_fconstexpr_depth_EQ)) {
    CmdArgs.push_back("-fconstexpr-depth");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_fconstexpr_steps_EQ)) {
    CmdArgs.push_back("-fconstexpr-steps");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_fbracket_depth_EQ)) {
    CmdArgs.push_back("-fbracket-depth");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_Wlarge_by_value_copy_EQ,
                               options::OPT_Wlarge_by_value_copy_def)) {
    if (A->getNumValues()) {
      StringRef bytes = A->getValue();
      CmdArgs.push_back(Args.MakeArgString("-Wlarge-by-value-copy=" + bytes));
    } else
      CmdArgs.push_back("-Wlarge-by-value-copy=64"); // default value
  }

  if (Args.hasArg(options::OPT_relocatable_pch))
    CmdArgs.push_back("-relocatable-pch");

  if (Arg *A = Args.getLastArg(options::OPT_fconstant_string_class_EQ)) {
    CmdArgs.push_back("-fconstant-string-class");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_ftabstop_EQ)) {
    CmdArgs.push_back("-ftabstop");
    CmdArgs.push_back(A->getValue());
  }

  CmdArgs.push_back("-ferror-limit");
  if (Arg *A = Args.getLastArg(options::OPT_ferror_limit_EQ))
    CmdArgs.push_back(A->getValue());
  else
    CmdArgs.push_back("19");

  if (Arg *A = Args.getLastArg(options::OPT_fmacro_backtrace_limit_EQ)) {
    CmdArgs.push_back("-fmacro-backtrace-limit");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_ftemplate_backtrace_limit_EQ)) {
    CmdArgs.push_back("-ftemplate-backtrace-limit");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_fconstexpr_backtrace_limit_EQ)) {
    CmdArgs.push_back("-fconstexpr-backtrace-limit");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(options::OPT_fspell_checking_limit_EQ)) {
    CmdArgs.push_back("-fspell-checking-limit");
    CmdArgs.push_back(A->getValue());
  }

  // Pass -fmessage-length=.
  CmdArgs.push_back("-fmessage-length");
  if (Arg *A = Args.getLastArg(options::OPT_fmessage_length_EQ)) {
    CmdArgs.push_back(A->getValue());
  } else {
    // If -fmessage-length=N was not specified, determine whether this is a
    // terminal and, if so, implicitly define -fmessage-length appropriately.
    unsigned N = llvm::sys::Process::StandardErrColumns();
    CmdArgs.push_back(Args.MakeArgString(Twine(N)));
  }

  // -fvisibility= and -fvisibility-ms-compat are of a piece.
  if (const Arg *A = Args.getLastArg(options::OPT_fvisibility_EQ,
                                     options::OPT_fvisibility_ms_compat)) {
    if (A->getOption().matches(options::OPT_fvisibility_EQ)) {
      CmdArgs.push_back("-fvisibility");
      CmdArgs.push_back(A->getValue());
    } else {
      assert(A->getOption().matches(options::OPT_fvisibility_ms_compat));
      CmdArgs.push_back("-fvisibility");
      CmdArgs.push_back("hidden");
      CmdArgs.push_back("-ftype-visibility");
      CmdArgs.push_back("default");
    }
  }

  Args.AddLastArg(CmdArgs, options::OPT_fvisibility_inlines_hidden);

  Args.AddLastArg(CmdArgs, options::OPT_ftlsmodel_EQ);

  // -fhosted is default.
  if (Args.hasFlag(options::OPT_ffreestanding, options::OPT_fhosted, false) ||
      KernelOrKext)
    CmdArgs.push_back("-ffreestanding");

  // Forward -f (flag) options which we can pass directly.
  Args.AddLastArg(CmdArgs, options::OPT_femit_all_decls);
  Args.AddLastArg(CmdArgs, options::OPT_fheinous_gnu_extensions);
  Args.AddLastArg(CmdArgs, options::OPT_fno_operator_names);
  // Emulated TLS is enabled by default on Android, and can be enabled manually
  // with -femulated-tls.
  bool EmulatedTLSDefault = Triple.isAndroid() || Triple.isWindowsCygwinEnvironment();
  if (Args.hasFlag(options::OPT_femulated_tls, options::OPT_fno_emulated_tls,
                   EmulatedTLSDefault))
    CmdArgs.push_back("-femulated-tls");
  // AltiVec-like language extensions aren't relevant for assembling.
  if (!isa<PreprocessJobAction>(JA) || Output.getType() != types::TY_PP_Asm) {
    Args.AddLastArg(CmdArgs, options::OPT_faltivec);
    Args.AddLastArg(CmdArgs, options::OPT_fzvector);
  }
  Args.AddLastArg(CmdArgs, options::OPT_fdiagnostics_show_template_tree);
  Args.AddLastArg(CmdArgs, options::OPT_fno_elide_type);

  // Forward flags for OpenMP. We don't do this if the current action is an
  // device offloading action.
  //
  // TODO: Allow OpenMP offload actions when they become available.
  if (Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                   options::OPT_fno_openmp, false) &&
      JA.isDeviceOffloading(Action::OFK_None)) {
    switch (getOpenMPRuntime(getToolChain(), Args)) {
    case OMPRT_OMP:
    case OMPRT_IOMP5:
      // Clang can generate useful OpenMP code for these two runtime libraries.
      CmdArgs.push_back("-fopenmp");

      // If no option regarding the use of TLS in OpenMP codegeneration is
      // given, decide a default based on the target. Otherwise rely on the
      // options and pass the right information to the frontend.
      if (!Args.hasFlag(options::OPT_fopenmp_use_tls,
                        options::OPT_fnoopenmp_use_tls, /*Default=*/true))
        CmdArgs.push_back("-fnoopenmp-use-tls");
      Args.AddAllArgs(CmdArgs, options::OPT_fopenmp_version_EQ);
      break;
    default:
      // By default, if Clang doesn't know how to generate useful OpenMP code
      // for a specific runtime library, we just don't pass the '-fopenmp' flag
      // down to the actual compilation.
      // FIXME: It would be better to have a mode which *only* omits IR
      // generation based on the OpenMP support so that we get consistent
      // semantic analysis, etc.
      break;
    }
  }

  const SanitizerArgs &Sanitize = getToolChain().getSanitizerArgs();
  Sanitize.addArgs(getToolChain(), Args, CmdArgs, InputType);

  // Report an error for -faltivec on anything other than PowerPC.
  if (const Arg *A = Args.getLastArg(options::OPT_faltivec)) {
    const llvm::Triple::ArchType Arch = getToolChain().getArch();
    if (!(Arch == llvm::Triple::ppc || Arch == llvm::Triple::ppc64 ||
          Arch == llvm::Triple::ppc64le))
      D.Diag(diag::err_drv_argument_only_allowed_with) << A->getAsString(Args)
                                                       << "ppc/ppc64/ppc64le";
  }

  // -fzvector is incompatible with -faltivec.
  if (Arg *A = Args.getLastArg(options::OPT_fzvector))
    if (Args.hasArg(options::OPT_faltivec))
      D.Diag(diag::err_drv_argument_not_allowed_with) << A->getAsString(Args)
                                                      << "-faltivec";

  if (getToolChain().SupportsProfiling())
    Args.AddLastArg(CmdArgs, options::OPT_pg);

  // -flax-vector-conversions is default.
  if (!Args.hasFlag(options::OPT_flax_vector_conversions,
                    options::OPT_fno_lax_vector_conversions))
    CmdArgs.push_back("-fno-lax-vector-conversions");

  if (Args.getLastArg(options::OPT_fapple_kext) ||
      (Args.hasArg(options::OPT_mkernel) && types::isCXX(InputType)))
    CmdArgs.push_back("-fapple-kext");

  Args.AddLastArg(CmdArgs, options::OPT_fobjc_sender_dependent_dispatch);
  Args.AddLastArg(CmdArgs, options::OPT_fdiagnostics_print_source_range_info);
  Args.AddLastArg(CmdArgs, options::OPT_fdiagnostics_parseable_fixits);
  Args.AddLastArg(CmdArgs, options::OPT_ftime_report);
  Args.AddLastArg(CmdArgs, options::OPT_ftrapv);

  if (Arg *A = Args.getLastArg(options::OPT_ftrapv_handler_EQ)) {
    CmdArgs.push_back("-ftrapv-handler");
    CmdArgs.push_back(A->getValue());
  }

  Args.AddLastArg(CmdArgs, options::OPT_ftrap_function_EQ);

  // -fno-strict-overflow implies -fwrapv if it isn't disabled, but
  // -fstrict-overflow won't turn off an explicitly enabled -fwrapv.
  if (Arg *A = Args.getLastArg(options::OPT_fwrapv, options::OPT_fno_wrapv)) {
    if (A->getOption().matches(options::OPT_fwrapv))
      CmdArgs.push_back("-fwrapv");
  } else if (Arg *A = Args.getLastArg(options::OPT_fstrict_overflow,
                                      options::OPT_fno_strict_overflow)) {
    if (A->getOption().matches(options::OPT_fno_strict_overflow))
      CmdArgs.push_back("-fwrapv");
  }

  if (Arg *A = Args.getLastArg(options::OPT_freroll_loops,
                               options::OPT_fno_reroll_loops))
    if (A->getOption().matches(options::OPT_freroll_loops))
      CmdArgs.push_back("-freroll-loops");

  Args.AddLastArg(CmdArgs, options::OPT_fwritable_strings);
  Args.AddLastArg(CmdArgs, options::OPT_funroll_loops,
                  options::OPT_fno_unroll_loops);

  Args.AddLastArg(CmdArgs, options::OPT_pthread);

  // -stack-protector=0 is default.
  unsigned StackProtectorLevel = 0;
  if (Arg *A = Args.getLastArg(options::OPT_fno_stack_protector,
                               options::OPT_fstack_protector_all,
                               options::OPT_fstack_protector_strong,
                               options::OPT_fstack_protector)) {
    if (A->getOption().matches(options::OPT_fstack_protector)) {
      StackProtectorLevel = std::max<unsigned>(
          LangOptions::SSPOn,
          getToolChain().GetDefaultStackProtectorLevel(KernelOrKext));
    } else if (A->getOption().matches(options::OPT_fstack_protector_strong))
      StackProtectorLevel = LangOptions::SSPStrong;
    else if (A->getOption().matches(options::OPT_fstack_protector_all))
      StackProtectorLevel = LangOptions::SSPReq;
  } else {
    StackProtectorLevel =
        getToolChain().GetDefaultStackProtectorLevel(KernelOrKext);
  }
  if (StackProtectorLevel) {
    CmdArgs.push_back("-stack-protector");
    CmdArgs.push_back(Args.MakeArgString(Twine(StackProtectorLevel)));
  }

  // --param ssp-buffer-size=
  for (const Arg *A : Args.filtered(options::OPT__param)) {
    StringRef Str(A->getValue());
    if (Str.startswith("ssp-buffer-size=")) {
      if (StackProtectorLevel) {
        CmdArgs.push_back("-stack-protector-buffer-size");
        // FIXME: Verify the argument is a valid integer.
        CmdArgs.push_back(Args.MakeArgString(Str.drop_front(16)));
      }
      A->claim();
    }
  }

  // Translate -mstackrealign
  if (Args.hasFlag(options::OPT_mstackrealign, options::OPT_mno_stackrealign,
                   false))
    CmdArgs.push_back(Args.MakeArgString("-mstackrealign"));

  if (Args.hasArg(options::OPT_mstack_alignment)) {
    StringRef alignment = Args.getLastArgValue(options::OPT_mstack_alignment);
    CmdArgs.push_back(Args.MakeArgString("-mstack-alignment=" + alignment));
  }

  if (Args.hasArg(options::OPT_mstack_probe_size)) {
    StringRef Size = Args.getLastArgValue(options::OPT_mstack_probe_size);

    if (!Size.empty())
      CmdArgs.push_back(Args.MakeArgString("-mstack-probe-size=" + Size));
    else
      CmdArgs.push_back("-mstack-probe-size=0");
  }

  switch (getToolChain().getArch()) {
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    CmdArgs.push_back("-fallow-half-arguments-and-returns");
    break;

  default:
    break;
  }

  if (Arg *A = Args.getLastArg(options::OPT_mrestrict_it,
                               options::OPT_mno_restrict_it)) {
    if (A->getOption().matches(options::OPT_mrestrict_it)) {
      CmdArgs.push_back("-backend-option");
      CmdArgs.push_back("-arm-restrict-it");
    } else {
      CmdArgs.push_back("-backend-option");
      CmdArgs.push_back("-arm-no-restrict-it");
    }
  } else if (Triple.isOSWindows() &&
             (Triple.getArch() == llvm::Triple::arm ||
              Triple.getArch() == llvm::Triple::thumb)) {
    // Windows on ARM expects restricted IT blocks
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-arm-restrict-it");
  }

  // Forward -cl options to -cc1
  if (Args.getLastArg(options::OPT_cl_opt_disable)) {
    CmdArgs.push_back("-cl-opt-disable");
  }
  if (Args.getLastArg(options::OPT_cl_strict_aliasing)) {
    CmdArgs.push_back("-cl-strict-aliasing");
  }
  if (Args.getLastArg(options::OPT_cl_single_precision_constant)) {
    CmdArgs.push_back("-cl-single-precision-constant");
  }
  if (Args.getLastArg(options::OPT_cl_finite_math_only)) {
    CmdArgs.push_back("-cl-finite-math-only");
  }
  if (Args.getLastArg(options::OPT_cl_kernel_arg_info)) {
    CmdArgs.push_back("-cl-kernel-arg-info");
  }
  if (Args.getLastArg(options::OPT_cl_unsafe_math_optimizations)) {
    CmdArgs.push_back("-cl-unsafe-math-optimizations");
  }
  if (Args.getLastArg(options::OPT_cl_fast_relaxed_math)) {
    CmdArgs.push_back("-cl-fast-relaxed-math");
  }
  if (Args.getLastArg(options::OPT_cl_mad_enable)) {
    CmdArgs.push_back("-cl-mad-enable");
  }
  if (Args.getLastArg(options::OPT_cl_no_signed_zeros)) {
    CmdArgs.push_back("-cl-no-signed-zeros");
  }
  if (Arg *A = Args.getLastArg(options::OPT_cl_std_EQ)) {
    std::string CLStdStr = "-cl-std=";
    CLStdStr += A->getValue();
    CmdArgs.push_back(Args.MakeArgString(CLStdStr));
  }
  if (Args.getLastArg(options::OPT_cl_denorms_are_zero)) {
    CmdArgs.push_back("-cl-denorms-are-zero");
  }
  if (Args.getLastArg(options::OPT_cl_fp32_correctly_rounded_divide_sqrt)) {
    CmdArgs.push_back("-cl-fp32-correctly-rounded-divide-sqrt");
  }

  // Forward -f options with positive and negative forms; we translate
  // these by hand.
  if (Arg *A = Args.getLastArg(options::OPT_fprofile_sample_use_EQ)) {
    StringRef fname = A->getValue();
    if (!llvm::sys::fs::exists(fname))
      D.Diag(diag::err_drv_no_such_file) << fname;
    else
      A->render(Args, CmdArgs);
  }

  // -fbuiltin is default unless -mkernel is used.
  bool UseBuiltins =
      Args.hasFlag(options::OPT_fbuiltin, options::OPT_fno_builtin,
                   !Args.hasArg(options::OPT_mkernel));
  if (!UseBuiltins)
    CmdArgs.push_back("-fno-builtin");

  // -ffreestanding implies -fno-builtin.
  if (Args.hasArg(options::OPT_ffreestanding))
    UseBuiltins = false;

  // Process the -fno-builtin-* options.
  for (const auto &Arg : Args) {
    const Option &O = Arg->getOption();
    if (!O.matches(options::OPT_fno_builtin_))
      continue;

    Arg->claim();
    // If -fno-builtin is specified, then there's no need to pass the option to
    // the frontend.
    if (!UseBuiltins)
      continue;

    StringRef FuncName = Arg->getValue();
    CmdArgs.push_back(Args.MakeArgString("-fno-builtin-" + FuncName));
  }

  if (!Args.hasFlag(options::OPT_fassume_sane_operator_new,
                    options::OPT_fno_assume_sane_operator_new))
    CmdArgs.push_back("-fno-assume-sane-operator-new");

  if (Args.hasFlag(options::OPT_fapinotes, options::OPT_fno_apinotes,
                   false) ||
      Args.hasArg(options::OPT_iapinotes_modules)) {
    if (Args.hasFlag(options::OPT_fapinotes, options::OPT_fno_apinotes, false))
      CmdArgs.push_back("-fapinotes");

    SmallString<128> APINotesCachePath;
    if (Arg *A = Args.getLastArg(options::OPT_fapinotes_cache_path)) {
      APINotesCachePath = A->getValue();
    }

    if (C.isForDiagnostics()) {
      // When generating crash reports, we want to emit the API notes along with
      // the reproduction sources, so we ignore any provided API notes path.
      APINotesCachePath = Output.getFilename();
      llvm::sys::path::replace_extension(APINotesCachePath, ".cache");
      llvm::sys::path::append(APINotesCachePath, "apinotes");
    } else if (APINotesCachePath.empty()) {
      // No API notes path was provided: use the default.
      llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/false,
                                             APINotesCachePath);
      llvm::sys::path::append(APINotesCachePath, "org.llvm.clang");
      llvm::sys::path::append(APINotesCachePath, "APINotesCache");
    }
    const char Arg[] = "-fapinotes-cache-path=";
    APINotesCachePath.insert(APINotesCachePath.begin(), Arg, Arg + strlen(Arg));
    CmdArgs.push_back(Args.MakeArgString(APINotesCachePath));
  }

  // -fblocks=0 is default.
  if (Args.hasFlag(options::OPT_fblocks, options::OPT_fno_blocks,
                   getToolChain().IsBlocksDefault()) ||
      (Args.hasArg(options::OPT_fgnu_runtime) &&
       Args.hasArg(options::OPT_fobjc_nonfragile_abi) &&
       !Args.hasArg(options::OPT_fno_blocks))) {
    CmdArgs.push_back("-fblocks");

    if (!Args.hasArg(options::OPT_fgnu_runtime) &&
        !getToolChain().hasBlocksRuntime())
      CmdArgs.push_back("-fblocks-runtime-optional");
  }

  // -fmodules enables the use of precompiled modules (off by default).
  // Users can pass -fno-cxx-modules to turn off modules support for
  // C++/Objective-C++ programs.
  bool HaveClangModules = false;
  if (Args.hasFlag(options::OPT_fmodules, options::OPT_fno_modules, false)) {
    bool AllowedInCXX = Args.hasFlag(options::OPT_fcxx_modules,
                                     options::OPT_fno_cxx_modules, true);
    if (AllowedInCXX || !types::isCXX(InputType)) {
      CmdArgs.push_back("-fmodules");
      HaveClangModules = true;
    }
  }

  bool HaveAnyModules = HaveClangModules;
  if (Args.hasArg(options::OPT_fmodules_ts)) {
    CmdArgs.push_back("-fmodules-ts");
    HaveAnyModules = true;
  }

  // -fmodule-maps enables implicit reading of module map files. By default,
  // this is enabled if we are using Clang's flavor of precompiled modules.
  if (Args.hasFlag(options::OPT_fimplicit_module_maps,
                   options::OPT_fno_implicit_module_maps, HaveClangModules)) {
    CmdArgs.push_back("-fimplicit-module-maps");
  }

  // -fmodules-decluse checks that modules used are declared so (off by
  // default).
  if (Args.hasFlag(options::OPT_fmodules_decluse,
                   options::OPT_fno_modules_decluse, false)) {
    CmdArgs.push_back("-fmodules-decluse");
  }

  // -fmodules-strict-decluse is like -fmodule-decluse, but also checks that
  // all #included headers are part of modules.
  if (Args.hasFlag(options::OPT_fmodules_strict_decluse,
                   options::OPT_fno_modules_strict_decluse, false)) {
    CmdArgs.push_back("-fmodules-strict-decluse");
  }

  // -fno-implicit-modules turns off implicitly compiling modules on demand.
  if (!Args.hasFlag(options::OPT_fimplicit_modules,
                    options::OPT_fno_implicit_modules, HaveClangModules)) {
    if (HaveAnyModules)
      CmdArgs.push_back("-fno-implicit-modules");
  } else if (HaveAnyModules) {
    // -fmodule-cache-path specifies where our implicitly-built module files
    // should be written.
    SmallString<128> Path;
    if (Arg *A = Args.getLastArg(options::OPT_fmodules_cache_path))
      Path = A->getValue();
    if (C.isForDiagnostics()) {
      // When generating crash reports, we want to emit the modules along with
      // the reproduction sources, so we ignore any provided module path.
      Path = Output.getFilename();
      llvm::sys::path::replace_extension(Path, ".cache");
      llvm::sys::path::append(Path, "modules");
    } else if (Path.empty()) {
      // No module path was provided: use the default.
      llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/false, Path);
      llvm::sys::path::append(Path, "org.llvm.clang.");
      appendUserToPath(Path);
      llvm::sys::path::append(Path, "ModuleCache");
    }
    const char Arg[] = "-fmodules-cache-path=";
    Path.insert(Path.begin(), Arg, Arg + strlen(Arg));
    CmdArgs.push_back(Args.MakeArgString(Path));
  }

  if (HaveAnyModules) {
    // -fprebuilt-module-path specifies where to load the prebuilt module files.
    for (const Arg *A : Args.filtered(options::OPT_fprebuilt_module_path))
      CmdArgs.push_back(Args.MakeArgString(
          std::string("-fprebuilt-module-path=") + A->getValue()));
  }
      
  // -fmodule-name specifies the module that is currently being built (or
  // used for header checking by -fmodule-maps).
  Args.AddLastArg(CmdArgs, options::OPT_fmodule_name_EQ);

  // -fmodule-map-file can be used to specify files containing module
  // definitions.
  Args.AddAllArgs(CmdArgs, options::OPT_fmodule_map_file);

  // -fmodule-file can be used to specify files containing precompiled modules.
  if (HaveAnyModules)
    Args.AddAllArgs(CmdArgs, options::OPT_fmodule_file);
  else
    Args.ClaimAllArgs(options::OPT_fmodule_file);

  // When building modules and generating crashdumps, we need to dump a module
  // dependency VFS alongside the output.
  if (HaveClangModules && C.isForDiagnostics()) {
    SmallString<128> VFSDir(Output.getFilename());
    llvm::sys::path::replace_extension(VFSDir, ".cache");
    // Add the cache directory as a temp so the crash diagnostics pick it up.
    C.addTempFile(Args.MakeArgString(VFSDir));

    llvm::sys::path::append(VFSDir, "vfs");
    CmdArgs.push_back("-module-dependency-dir");
    CmdArgs.push_back(Args.MakeArgString(VFSDir));
  }

  if (HaveClangModules)
    Args.AddLastArg(CmdArgs, options::OPT_fmodules_user_build_path);

  // Pass through all -fmodules-ignore-macro arguments.
  Args.AddAllArgs(CmdArgs, options::OPT_fmodules_ignore_macro);
  Args.AddLastArg(CmdArgs, options::OPT_fmodules_prune_interval);
  Args.AddLastArg(CmdArgs, options::OPT_fmodules_prune_after);

  Args.AddLastArg(CmdArgs, options::OPT_fbuild_session_timestamp);

  if (Arg *A = Args.getLastArg(options::OPT_fbuild_session_file)) {
    if (Args.hasArg(options::OPT_fbuild_session_timestamp))
      D.Diag(diag::err_drv_argument_not_allowed_with)
          << A->getAsString(Args) << "-fbuild-session-timestamp";

    llvm::sys::fs::file_status Status;
    if (llvm::sys::fs::status(A->getValue(), Status))
      D.Diag(diag::err_drv_no_such_file) << A->getValue();
    CmdArgs.push_back(Args.MakeArgString(
        "-fbuild-session-timestamp=" +
        Twine((uint64_t)Status.getLastModificationTime().toEpochTime())));
  }

  if (Args.getLastArg(options::OPT_fmodules_validate_once_per_build_session)) {
    if (!Args.getLastArg(options::OPT_fbuild_session_timestamp,
                         options::OPT_fbuild_session_file))
      D.Diag(diag::err_drv_modules_validate_once_requires_timestamp);

    Args.AddLastArg(CmdArgs,
                    options::OPT_fmodules_validate_once_per_build_session);
  }

  Args.AddLastArg(CmdArgs, options::OPT_fmodules_validate_system_headers);
  Args.AddLastArg(CmdArgs, options::OPT_fmodules_disable_diagnostic_validation);

  // -faccess-control is default.
  if (Args.hasFlag(options::OPT_fno_access_control,
                   options::OPT_faccess_control, false))
    CmdArgs.push_back("-fno-access-control");

  // -felide-constructors is the default.
  if (Args.hasFlag(options::OPT_fno_elide_constructors,
                   options::OPT_felide_constructors, false))
    CmdArgs.push_back("-fno-elide-constructors");

  ToolChain::RTTIMode RTTIMode = getToolChain().getRTTIMode();

  if (KernelOrKext || (types::isCXX(InputType) &&
                       (RTTIMode == ToolChain::RM_DisabledExplicitly ||
                        RTTIMode == ToolChain::RM_DisabledImplicitly)))
    CmdArgs.push_back("-fno-rtti");

  // -fshort-enums=0 is default for all architectures except Hexagon.
  if (Args.hasFlag(options::OPT_fshort_enums, options::OPT_fno_short_enums,
                   getToolChain().getArch() == llvm::Triple::hexagon))
    CmdArgs.push_back("-fshort-enums");

  // -fsigned-char is default.
  if (Arg *A = Args.getLastArg(
          options::OPT_fsigned_char, options::OPT_fno_signed_char,
          options::OPT_funsigned_char, options::OPT_fno_unsigned_char)) {
    if (A->getOption().matches(options::OPT_funsigned_char) ||
        A->getOption().matches(options::OPT_fno_signed_char)) {
      CmdArgs.push_back("-fno-signed-char");
    }
  } else if (!isSignedCharDefault(getToolChain().getTriple())) {
    CmdArgs.push_back("-fno-signed-char");
  }

  // -fuse-cxa-atexit is default.
  if (!Args.hasFlag(
          options::OPT_fuse_cxa_atexit, options::OPT_fno_use_cxa_atexit,
          !IsWindowsCygnus && !IsWindowsGNU &&
              getToolChain().getTriple().getOS() != llvm::Triple::Solaris &&
              getToolChain().getArch() != llvm::Triple::hexagon &&
              getToolChain().getArch() != llvm::Triple::xcore &&
              ((getToolChain().getTriple().getVendor() !=
                llvm::Triple::MipsTechnologies) ||
               getToolChain().getTriple().hasEnvironment())) ||
      KernelOrKext)
    CmdArgs.push_back("-fno-use-cxa-atexit");

  // -fms-extensions=0 is default.
  if (Args.hasFlag(options::OPT_fms_extensions, options::OPT_fno_ms_extensions,
                   IsWindowsMSVC))
    CmdArgs.push_back("-fms-extensions");

  // -fno-use-line-directives is default.
  if (Args.hasFlag(options::OPT_fuse_line_directives,
                   options::OPT_fno_use_line_directives, false))
    CmdArgs.push_back("-fuse-line-directives");

  // -fms-compatibility=0 is default.
  if (Args.hasFlag(options::OPT_fms_compatibility,
                   options::OPT_fno_ms_compatibility,
                   (IsWindowsMSVC &&
                    Args.hasFlag(options::OPT_fms_extensions,
                                 options::OPT_fno_ms_extensions, true))))
    CmdArgs.push_back("-fms-compatibility");

  // -fms-compatibility-version=18.00 is default.
  VersionTuple MSVT = visualstudio::getMSVCVersion(
      &D, getToolChain(), getToolChain().getTriple(), Args, IsWindowsMSVC);
  if (!MSVT.empty())
    CmdArgs.push_back(
        Args.MakeArgString("-fms-compatibility-version=" + MSVT.getAsString()));

  bool IsMSVC2015Compatible = MSVT.getMajor() >= 19;
  if (ImplyVCPPCXXVer) {
    StringRef LanguageStandard;
    if (const Arg *StdArg = Args.getLastArg(options::OPT__SLASH_std)) {
      LanguageStandard = llvm::StringSwitch<StringRef>(StdArg->getValue())
                             .Case("c++14", "-std=c++14")
                             .Case("c++latest", "-std=c++1z")
                             .Default("");
      if (LanguageStandard.empty())
        D.Diag(clang::diag::warn_drv_unused_argument)
            << StdArg->getAsString(Args);
    }

    if (LanguageStandard.empty()) {
      if (IsMSVC2015Compatible)
        LanguageStandard = "-std=c++14";
      else
        LanguageStandard = "-std=c++11";
    }

    CmdArgs.push_back(LanguageStandard.data());
  }

  // -fno-borland-extensions is default.
  if (Args.hasFlag(options::OPT_fborland_extensions,
                   options::OPT_fno_borland_extensions, false))
    CmdArgs.push_back("-fborland-extensions");

  // -fno-declspec is default, except for PS4.
  if (Args.hasFlag(options::OPT_fdeclspec, options::OPT_fno_declspec,
                   getToolChain().getTriple().isPS4()))
    CmdArgs.push_back("-fdeclspec");
  else if (Args.hasArg(options::OPT_fno_declspec))
    CmdArgs.push_back("-fno-declspec"); // Explicitly disabling __declspec.

  // -fthreadsafe-static is default, except for MSVC compatibility versions less
  // than 19.
  if (!Args.hasFlag(options::OPT_fthreadsafe_statics,
                    options::OPT_fno_threadsafe_statics,
                    !IsWindowsMSVC || IsMSVC2015Compatible))
    CmdArgs.push_back("-fno-threadsafe-statics");

  // -fno-delayed-template-parsing is default, except for Windows where MSVC STL
  // needs it.
  if (Args.hasFlag(options::OPT_fdelayed_template_parsing,
                   options::OPT_fno_delayed_template_parsing, IsWindowsMSVC))
    CmdArgs.push_back("-fdelayed-template-parsing");

  // -fgnu-keywords default varies depending on language; only pass if
  // specified.
  if (Arg *A = Args.getLastArg(options::OPT_fgnu_keywords,
                               options::OPT_fno_gnu_keywords))
    A->render(Args, CmdArgs);

  if (Args.hasFlag(options::OPT_fgnu89_inline, options::OPT_fno_gnu89_inline,
                   false))
    CmdArgs.push_back("-fgnu89-inline");

  if (Args.hasArg(options::OPT_fno_inline))
    CmdArgs.push_back("-fno-inline");

  if (Arg* InlineArg = Args.getLastArg(options::OPT_finline_functions,
                                       options::OPT_finline_hint_functions,
                                       options::OPT_fno_inline_functions))
    InlineArg->render(Args, CmdArgs);

  ObjCRuntime objcRuntime = AddObjCRuntimeArgs(Args, CmdArgs, rewriteKind);

  // -fobjc-dispatch-method is only relevant with the nonfragile-abi, and
  // legacy is the default. Except for deployment taget of 10.5,
  // next runtime is always legacy dispatch and -fno-objc-legacy-dispatch
  // gets ignored silently.
  if (objcRuntime.isNonFragile()) {
    if (!Args.hasFlag(options::OPT_fobjc_legacy_dispatch,
                      options::OPT_fno_objc_legacy_dispatch,
                      objcRuntime.isLegacyDispatchDefaultForArch(
                          getToolChain().getArch()))) {
      if (getToolChain().UseObjCMixedDispatch())
        CmdArgs.push_back("-fobjc-dispatch-method=mixed");
      else
        CmdArgs.push_back("-fobjc-dispatch-method=non-legacy");
    }
  }

  // When ObjectiveC legacy runtime is in effect on MacOSX,
  // turn on the option to do Array/Dictionary subscripting
  // by default.
  if (getToolChain().getArch() == llvm::Triple::x86 &&
      getToolChain().getTriple().isMacOSX() &&
      !getToolChain().getTriple().isMacOSXVersionLT(10, 7) &&
      objcRuntime.getKind() == ObjCRuntime::FragileMacOSX &&
      objcRuntime.isNeXTFamily())
    CmdArgs.push_back("-fobjc-subscripting-legacy-runtime");

  // -fencode-extended-block-signature=1 is default.
  if (getToolChain().IsEncodeExtendedBlockSignatureDefault()) {
    CmdArgs.push_back("-fencode-extended-block-signature");
  }

  // Allow -fno-objc-arr to trump -fobjc-arr/-fobjc-arc.
  // NOTE: This logic is duplicated in ToolChains.cpp.
  bool ARC = isObjCAutoRefCount(Args);
  if (ARC) {
    getToolChain().CheckObjCARC();

    CmdArgs.push_back("-fobjc-arc");

    // FIXME: It seems like this entire block, and several around it should be
    // wrapped in isObjC, but for now we just use it here as this is where it
    // was being used previously.
    if (types::isCXX(InputType) && types::isObjC(InputType)) {
      if (getToolChain().GetCXXStdlibType(Args) == ToolChain::CST_Libcxx)
        CmdArgs.push_back("-fobjc-arc-cxxlib=libc++");
      else
        CmdArgs.push_back("-fobjc-arc-cxxlib=libstdc++");
    }

    // Allow the user to enable full exceptions code emission.
    // We define off for Objective-CC, on for Objective-C++.
    if (Args.hasFlag(options::OPT_fobjc_arc_exceptions,
                     options::OPT_fno_objc_arc_exceptions,
                     /*default*/ types::isCXX(InputType)))
      CmdArgs.push_back("-fobjc-arc-exceptions");

  }

  // -fobjc-infer-related-result-type is the default, except in the Objective-C
  // rewriter.
  if (rewriteKind != RK_None)
    CmdArgs.push_back("-fno-objc-infer-related-result-type");

  // Handle -fobjc-gc and -fobjc-gc-only. They are exclusive, and -fobjc-gc-only
  // takes precedence.
  const Arg *GCArg = Args.getLastArg(options::OPT_fobjc_gc_only);
  if (!GCArg)
    GCArg = Args.getLastArg(options::OPT_fobjc_gc);
  if (GCArg) {
    if (ARC) {
      D.Diag(diag::err_drv_objc_gc_arr) << GCArg->getAsString(Args);
    } else if (getToolChain().SupportsObjCGC()) {
      GCArg->render(Args, CmdArgs);
    } else {
      // FIXME: We should move this to a hard error.
      D.Diag(diag::warn_drv_objc_gc_unsupported) << GCArg->getAsString(Args);
    }
  }

  // Pass down -fobjc-weak or -fno-objc-weak if present.
  if (types::isObjC(InputType)) {
    auto WeakArg = Args.getLastArg(options::OPT_fobjc_weak,
                                   options::OPT_fno_objc_weak);
    if (!WeakArg) {
      // nothing to do
    } else if (GCArg) {
      if (WeakArg->getOption().matches(options::OPT_fobjc_weak))
        D.Diag(diag::err_objc_weak_with_gc);
    } else if (!objcRuntime.allowsWeak()) {
      if (WeakArg->getOption().matches(options::OPT_fobjc_weak))
        D.Diag(diag::err_objc_weak_unsupported);
    } else {
      WeakArg->render(Args, CmdArgs);
    }
  }

  if (Args.hasFlag(options::OPT_fapplication_extension,
                   options::OPT_fno_application_extension, false))
    CmdArgs.push_back("-fapplication-extension");

  // Handle GCC-style exception args.
  if (!C.getDriver().IsCLMode())
    addExceptionArgs(Args, InputType, getToolChain(), KernelOrKext, objcRuntime,
                     CmdArgs);

  if (Args.hasArg(options::OPT_fsjlj_exceptions) ||
      getToolChain().UseSjLjExceptions(Args))
    CmdArgs.push_back("-fsjlj-exceptions");

  // C++ "sane" operator new.
  if (!Args.hasFlag(options::OPT_fassume_sane_operator_new,
                    options::OPT_fno_assume_sane_operator_new))
    CmdArgs.push_back("-fno-assume-sane-operator-new");

  // -fsized-deallocation is off by default, as it is an ABI-breaking change for
  // most platforms.
  if (Args.hasFlag(options::OPT_fsized_deallocation,
                   options::OPT_fno_sized_deallocation, false))
    CmdArgs.push_back("-fsized-deallocation");

  // -fconstant-cfstrings is default, and may be subject to argument translation
  // on Darwin.
  if (!Args.hasFlag(options::OPT_fconstant_cfstrings,
                    options::OPT_fno_constant_cfstrings) ||
      !Args.hasFlag(options::OPT_mconstant_cfstrings,
                    options::OPT_mno_constant_cfstrings))
    CmdArgs.push_back("-fno-constant-cfstrings");

  // -fshort-wchar default varies depending on platform; only
  // pass if specified.
  if (Arg *A = Args.getLastArg(options::OPT_fshort_wchar,
                               options::OPT_fno_short_wchar))
    A->render(Args, CmdArgs);

  // -fno-pascal-strings is default, only pass non-default.
  if (Args.hasFlag(options::OPT_fpascal_strings,
                   options::OPT_fno_pascal_strings, false))
    CmdArgs.push_back("-fpascal-strings");

  // Honor -fpack-struct= and -fpack-struct, if given. Note that
  // -fno-pack-struct doesn't apply to -fpack-struct=.
  if (Arg *A = Args.getLastArg(options::OPT_fpack_struct_EQ)) {
    std::string PackStructStr = "-fpack-struct=";
    PackStructStr += A->getValue();
    CmdArgs.push_back(Args.MakeArgString(PackStructStr));
  } else if (Args.hasFlag(options::OPT_fpack_struct,
                          options::OPT_fno_pack_struct, false)) {
    CmdArgs.push_back("-fpack-struct=1");
  }

  // Handle -fmax-type-align=N and -fno-type-align
  bool SkipMaxTypeAlign = Args.hasArg(options::OPT_fno_max_type_align);
  if (Arg *A = Args.getLastArg(options::OPT_fmax_type_align_EQ)) {
    if (!SkipMaxTypeAlign) {
      std::string MaxTypeAlignStr = "-fmax-type-align=";
      MaxTypeAlignStr += A->getValue();
      CmdArgs.push_back(Args.MakeArgString(MaxTypeAlignStr));
    }
  } else if (getToolChain().getTriple().isOSDarwin()) {
    if (!SkipMaxTypeAlign) {
      std::string MaxTypeAlignStr = "-fmax-type-align=16";
      CmdArgs.push_back(Args.MakeArgString(MaxTypeAlignStr));
    }
  }

  // -fcommon is the default unless compiling kernel code or the target says so
  bool NoCommonDefault =
      KernelOrKext || isNoCommonDefault(getToolChain().getTriple());
  if (!Args.hasFlag(options::OPT_fcommon, options::OPT_fno_common,
                    !NoCommonDefault))
    CmdArgs.push_back("-fno-common");

  // -fsigned-bitfields is default, and clang doesn't yet support
  // -funsigned-bitfields.
  if (!Args.hasFlag(options::OPT_fsigned_bitfields,
                    options::OPT_funsigned_bitfields))
    D.Diag(diag::warn_drv_clang_unsupported)
        << Args.getLastArg(options::OPT_funsigned_bitfields)->getAsString(Args);

  // -fsigned-bitfields is default, and clang doesn't support -fno-for-scope.
  if (!Args.hasFlag(options::OPT_ffor_scope, options::OPT_fno_for_scope))
    D.Diag(diag::err_drv_clang_unsupported)
        << Args.getLastArg(options::OPT_fno_for_scope)->getAsString(Args);

  // -finput_charset=UTF-8 is default. Reject others
  if (Arg *inputCharset = Args.getLastArg(options::OPT_finput_charset_EQ)) {
    StringRef value = inputCharset->getValue();
    if (!value.equals_lower("utf-8"))
      D.Diag(diag::err_drv_invalid_value) << inputCharset->getAsString(Args)
                                          << value;
  }

  // -fexec_charset=UTF-8 is default. Reject others
  if (Arg *execCharset = Args.getLastArg(options::OPT_fexec_charset_EQ)) {
    StringRef value = execCharset->getValue();
    if (!value.equals_lower("utf-8"))
      D.Diag(diag::err_drv_invalid_value) << execCharset->getAsString(Args)
                                          << value;
  }

  // -fcaret-diagnostics is default.
  if (!Args.hasFlag(options::OPT_fcaret_diagnostics,
                    options::OPT_fno_caret_diagnostics, true))
    CmdArgs.push_back("-fno-caret-diagnostics");

  // -fdiagnostics-fixit-info is default, only pass non-default.
  if (!Args.hasFlag(options::OPT_fdiagnostics_fixit_info,
                    options::OPT_fno_diagnostics_fixit_info))
    CmdArgs.push_back("-fno-diagnostics-fixit-info");

  // Enable -fdiagnostics-show-option by default.
  if (Args.hasFlag(options::OPT_fdiagnostics_show_option,
                   options::OPT_fno_diagnostics_show_option))
    CmdArgs.push_back("-fdiagnostics-show-option");

  if (const Arg *A =
          Args.getLastArg(options::OPT_fdiagnostics_show_category_EQ)) {
    CmdArgs.push_back("-fdiagnostics-show-category");
    CmdArgs.push_back(A->getValue());
  }

  if (Args.hasFlag(options::OPT_fdiagnostics_show_hotness,
                   options::OPT_fno_diagnostics_show_hotness, false))
    CmdArgs.push_back("-fdiagnostics-show-hotness");

  if (const Arg *A = Args.getLastArg(options::OPT_fdiagnostics_format_EQ)) {
    CmdArgs.push_back("-fdiagnostics-format");
    CmdArgs.push_back(A->getValue());
  }

  if (Arg *A = Args.getLastArg(
          options::OPT_fdiagnostics_show_note_include_stack,
          options::OPT_fno_diagnostics_show_note_include_stack)) {
    if (A->getOption().matches(
            options::OPT_fdiagnostics_show_note_include_stack))
      CmdArgs.push_back("-fdiagnostics-show-note-include-stack");
    else
      CmdArgs.push_back("-fno-diagnostics-show-note-include-stack");
  }

  // Color diagnostics are parsed by the driver directly from argv
  // and later re-parsed to construct this job; claim any possible
  // color diagnostic here to avoid warn_drv_unused_argument and
  // diagnose bad OPT_fdiagnostics_color_EQ values.
  for (Arg *A : Args) {
    const Option &O = A->getOption();
    if (!O.matches(options::OPT_fcolor_diagnostics) &&
        !O.matches(options::OPT_fdiagnostics_color) &&
        !O.matches(options::OPT_fno_color_diagnostics) &&
        !O.matches(options::OPT_fno_diagnostics_color) &&
        !O.matches(options::OPT_fdiagnostics_color_EQ))
      continue;
    if (O.matches(options::OPT_fdiagnostics_color_EQ)) {
      StringRef Value(A->getValue());
      if (Value != "always" && Value != "never" && Value != "auto")
        getToolChain().getDriver().Diag(diag::err_drv_clang_unsupported)
              << ("-fdiagnostics-color=" + Value).str();
    }
    A->claim();
  }
  if (D.getDiags().getDiagnosticOptions().ShowColors)
    CmdArgs.push_back("-fcolor-diagnostics");

  if (Args.hasArg(options::OPT_fansi_escape_codes))
    CmdArgs.push_back("-fansi-escape-codes");

  if (!Args.hasFlag(options::OPT_fshow_source_location,
                    options::OPT_fno_show_source_location))
    CmdArgs.push_back("-fno-show-source-location");

  if (Args.hasArg(options::OPT_fdiagnostics_absolute_paths))
    CmdArgs.push_back("-fdiagnostics-absolute-paths");

  if (!Args.hasFlag(options::OPT_fshow_column, options::OPT_fno_show_column,
                    true))
    CmdArgs.push_back("-fno-show-column");

  if (!Args.hasFlag(options::OPT_fspell_checking,
                    options::OPT_fno_spell_checking))
    CmdArgs.push_back("-fno-spell-checking");

  // -fno-asm-blocks is default.
  if (Args.hasFlag(options::OPT_fasm_blocks, options::OPT_fno_asm_blocks,
                   false))
    CmdArgs.push_back("-fasm-blocks");

  // -fgnu-inline-asm is default.
  if (!Args.hasFlag(options::OPT_fgnu_inline_asm,
                    options::OPT_fno_gnu_inline_asm, true))
    CmdArgs.push_back("-fno-gnu-inline-asm");

  // Enable vectorization per default according to the optimization level
  // selected. For optimization levels that want vectorization we use the alias
  // option to simplify the hasFlag logic.
  bool EnableVec = shouldEnableVectorizerAtOLevel(Args, false);
  OptSpecifier VectorizeAliasOption =
      EnableVec ? options::OPT_O_Group : options::OPT_fvectorize;
  if (Args.hasFlag(options::OPT_fvectorize, VectorizeAliasOption,
                   options::OPT_fno_vectorize, EnableVec))
    CmdArgs.push_back("-vectorize-loops");

  // -fslp-vectorize is enabled based on the optimization level selected.
  bool EnableSLPVec = shouldEnableVectorizerAtOLevel(Args, true);
  OptSpecifier SLPVectAliasOption =
      EnableSLPVec ? options::OPT_O_Group : options::OPT_fslp_vectorize;
  if (Args.hasFlag(options::OPT_fslp_vectorize, SLPVectAliasOption,
                   options::OPT_fno_slp_vectorize, EnableSLPVec))
    CmdArgs.push_back("-vectorize-slp");

  // -fno-slp-vectorize-aggressive is default.
  if (Args.hasFlag(options::OPT_fslp_vectorize_aggressive,
                   options::OPT_fno_slp_vectorize_aggressive, false))
    CmdArgs.push_back("-vectorize-slp-aggressive");

  if (Arg *A = Args.getLastArg(options::OPT_fshow_overloads_EQ))
    A->render(Args, CmdArgs);

  if (Arg *A = Args.getLastArg(
          options::OPT_fsanitize_undefined_strip_path_components_EQ))
    A->render(Args, CmdArgs);

  // -fdollars-in-identifiers default varies depending on platform and
  // language; only pass if specified.
  if (Arg *A = Args.getLastArg(options::OPT_fdollars_in_identifiers,
                               options::OPT_fno_dollars_in_identifiers)) {
    if (A->getOption().matches(options::OPT_fdollars_in_identifiers))
      CmdArgs.push_back("-fdollars-in-identifiers");
    else
      CmdArgs.push_back("-fno-dollars-in-identifiers");
  }

  // -funit-at-a-time is default, and we don't support -fno-unit-at-a-time for
  // practical purposes.
  if (Arg *A = Args.getLastArg(options::OPT_funit_at_a_time,
                               options::OPT_fno_unit_at_a_time)) {
    if (A->getOption().matches(options::OPT_fno_unit_at_a_time))
      D.Diag(diag::warn_drv_clang_unsupported) << A->getAsString(Args);
  }

  if (Args.hasFlag(options::OPT_fapple_pragma_pack,
                   options::OPT_fno_apple_pragma_pack, false))
    CmdArgs.push_back("-fapple-pragma-pack");

  // le32-specific flags:
  //  -fno-math-builtin: clang should not convert math builtins to intrinsics
  //                     by default.
  if (getToolChain().getArch() == llvm::Triple::le32) {
    CmdArgs.push_back("-fno-math-builtin");
  }

// Default to -fno-builtin-str{cat,cpy} on Darwin for ARM.
//
// FIXME: Now that PR4941 has been fixed this can be enabled.
#if 0
  if (getToolChain().getTriple().isOSDarwin() &&
      (getToolChain().getArch() == llvm::Triple::arm ||
       getToolChain().getArch() == llvm::Triple::thumb)) {
    if (!Args.hasArg(options::OPT_fbuiltin_strcat))
      CmdArgs.push_back("-fno-builtin-strcat");
    if (!Args.hasArg(options::OPT_fbuiltin_strcpy))
      CmdArgs.push_back("-fno-builtin-strcpy");
  }
#endif

  // Enable rewrite includes if the user's asked for it or if we're generating
  // diagnostics.
  // TODO: Once -module-dependency-dir works with -frewrite-includes it'd be
  // nice to enable this when doing a crashdump for modules as well.
  if (Args.hasFlag(options::OPT_frewrite_includes,
                   options::OPT_fno_rewrite_includes, false) ||
      (C.isForDiagnostics() && !HaveAnyModules))
    CmdArgs.push_back("-frewrite-includes");

  // Only allow -traditional or -traditional-cpp outside in preprocessing modes.
  if (Arg *A = Args.getLastArg(options::OPT_traditional,
                               options::OPT_traditional_cpp)) {
    if (isa<PreprocessJobAction>(JA))
      CmdArgs.push_back("-traditional-cpp");
    else
      D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
  }

  Args.AddLastArg(CmdArgs, options::OPT_dM);
  Args.AddLastArg(CmdArgs, options::OPT_dD);

  // Handle serialized diagnostics.
  if (Arg *A = Args.getLastArg(options::OPT__serialize_diags)) {
    CmdArgs.push_back("-serialize-diagnostic-file");
    CmdArgs.push_back(Args.MakeArgString(A->getValue()));
  }

  if (Args.hasArg(options::OPT_fretain_comments_from_system_headers))
    CmdArgs.push_back("-fretain-comments-from-system-headers");

  // Forward -fcomment-block-commands to -cc1.
  Args.AddAllArgs(CmdArgs, options::OPT_fcomment_block_commands);
  // Forward -fparse-all-comments to -cc1.
  Args.AddAllArgs(CmdArgs, options::OPT_fparse_all_comments);

  // Turn -fplugin=name.so into -load name.so
  for (const Arg *A : Args.filtered(options::OPT_fplugin_EQ)) {
    CmdArgs.push_back("-load");
    CmdArgs.push_back(A->getValue());
    A->claim();
  }

  // Forward -Xclang arguments to -cc1, and -mllvm arguments to the LLVM option
  // parser.
  Args.AddAllArgValues(CmdArgs, options::OPT_Xclang);
  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    A->claim();

    // We translate this by hand to the -cc1 argument, since nightly test uses
    // it and developers have been trained to spell it with -mllvm.
    if (StringRef(A->getValue(0)) == "-disable-llvm-optzns") {
      CmdArgs.push_back("-disable-llvm-optzns");
    } else
      A->render(Args, CmdArgs);
  }

  // With -save-temps, we want to save the unoptimized bitcode output from the
  // CompileJobAction, use -disable-llvm-passes to get pristine IR generated
  // by the frontend.
  // When -fembed-bitcode is enabled, optimized bitcode is emitted because it
  // has slightly different breakdown between stages.
  // FIXME: -fembed-bitcode -save-temps will save optimized bitcode instead of
  // pristine IR generated by the frontend. Ideally, a new compile action should
  // be added so both IR can be captured.
  if (C.getDriver().isSaveTempsEnabled() &&
      !C.getDriver().embedBitcodeEnabled() && isa<CompileJobAction>(JA))
    CmdArgs.push_back("-disable-llvm-passes");

  if (Output.getType() == types::TY_Dependencies) {
    // Handled with other dependency code.
  } else if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  addDashXForInput(Args, Input, CmdArgs);

  if (Input.isFilename())
    CmdArgs.push_back(Input.getFilename());
  else
    Input.getInputArg().renderAsInput(Args, CmdArgs);

  Args.AddAllArgs(CmdArgs, options::OPT_undef);

  const char *Exec = getToolChain().getDriver().getClangProgramPath();

  // Optionally embed the -cc1 level arguments into the debug info, for build
  // analysis.
  if (getToolChain().UseDwarfDebugFlags()) {
    ArgStringList OriginalArgs;
    for (const auto &Arg : Args)
      Arg->render(Args, OriginalArgs);

    SmallString<256> Flags;
    Flags += Exec;
    for (const char *OriginalArg : OriginalArgs) {
      SmallString<128> EscapedArg;
      EscapeSpacesAndBackslashes(OriginalArg, EscapedArg);
      Flags += " ";
      Flags += EscapedArg;
    }
    CmdArgs.push_back("-dwarf-debug-flags");
    CmdArgs.push_back(Args.MakeArgString(Flags));
  }

  // Add the split debug info name to the command lines here so we
  // can propagate it to the backend.
  bool SplitDwarf = SplitDwarfArg && getToolChain().getTriple().isOSLinux() &&
                    (isa<AssembleJobAction>(JA) || isa<CompileJobAction>(JA) ||
                     isa<BackendJobAction>(JA));
  const char *SplitDwarfOut;
  if (SplitDwarf) {
    CmdArgs.push_back("-split-dwarf-file");
    SplitDwarfOut = SplitDebugName(Args, Input);
    CmdArgs.push_back(SplitDwarfOut);
  }

  // Host-side cuda compilation receives device-side outputs as Inputs[1...].
  // Include them with -fcuda-include-gpubinary.
  if (IsCuda && Inputs.size() > 1)
    for (auto I = std::next(Inputs.begin()), E = Inputs.end(); I != E; ++I) {
      CmdArgs.push_back("-fcuda-include-gpubinary");
      CmdArgs.push_back(I->getFilename());
    }

  bool WholeProgramVTables =
      Args.hasFlag(options::OPT_fwhole_program_vtables,
                   options::OPT_fno_whole_program_vtables, false);
  if (WholeProgramVTables) {
    if (!D.isUsingLTO())
      D.Diag(diag::err_drv_argument_only_allowed_with)
          << "-fwhole-program-vtables"
          << "-flto";
    CmdArgs.push_back("-fwhole-program-vtables");
  }

  // Finally add the compile command to the compilation.
  if (Args.hasArg(options::OPT__SLASH_fallback) &&
      Output.getType() == types::TY_Object &&
      (InputType == types::TY_C || InputType == types::TY_CXX)) {
    auto CLCommand =
        getCLFallback()->GetCommand(C, JA, Output, Inputs, Args, LinkingOutput);
    C.addCommand(llvm::make_unique<FallbackCommand>(
        JA, *this, Exec, CmdArgs, Inputs, std::move(CLCommand)));
  } else if (Args.hasArg(options::OPT__SLASH_fallback) &&
             isa<PrecompileJobAction>(JA)) {
    // In /fallback builds, run the main compilation even if the pch generation
    // fails, so that the main compilation's fallback to cl.exe runs.
    C.addCommand(llvm::make_unique<ForceSuccessCommand>(JA, *this, Exec,
                                                        CmdArgs, Inputs));
  } else {
    C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
  }

  // Handle the debug info splitting at object creation time if we're
  // creating an object.
  // TODO: Currently only works on linux with newer objcopy.
  if (SplitDwarf && Output.getType() == types::TY_Object)
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output, SplitDwarfOut);

  if (Arg *A = Args.getLastArg(options::OPT_pg))
    if (Args.hasArg(options::OPT_fomit_frame_pointer))
      D.Diag(diag::err_drv_argument_not_allowed_with) << "-fomit-frame-pointer"
                                                      << A->getAsString(Args);

  // Claim some arguments which clang supports automatically.

  // -fpch-preprocess is used with gcc to add a special marker in the output to
  // include the PCH file. Clang's PTH solution is completely transparent, so we
  // do not need to deal with it at all.
  Args.ClaimAllArgs(options::OPT_fpch_preprocess);

  // Claim some arguments which clang doesn't support, but we don't
  // care to warn the user about.
  Args.ClaimAllArgs(options::OPT_clang_ignored_f_Group);
  Args.ClaimAllArgs(options::OPT_clang_ignored_m_Group);

  // Disable warnings for clang -E -emit-llvm foo.c
  Args.ClaimAllArgs(options::OPT_emit_llvm);
}

/// Add options related to the Objective-C runtime/ABI.
///
/// Returns true if the runtime is non-fragile.
ObjCRuntime Clang::AddObjCRuntimeArgs(const ArgList &args,
                                      ArgStringList &cmdArgs,
                                      RewriteKind rewriteKind) const {
  // Look for the controlling runtime option.
  Arg *runtimeArg =
      args.getLastArg(options::OPT_fnext_runtime, options::OPT_fgnu_runtime,
                      options::OPT_fobjc_runtime_EQ);

  // Just forward -fobjc-runtime= to the frontend.  This supercedes
  // options about fragility.
  if (runtimeArg &&
      runtimeArg->getOption().matches(options::OPT_fobjc_runtime_EQ)) {
    ObjCRuntime runtime;
    StringRef value = runtimeArg->getValue();
    if (runtime.tryParse(value)) {
      getToolChain().getDriver().Diag(diag::err_drv_unknown_objc_runtime)
          << value;
    }

    runtimeArg->render(args, cmdArgs);
    return runtime;
  }

  // Otherwise, we'll need the ABI "version".  Version numbers are
  // slightly confusing for historical reasons:
  //   1 - Traditional "fragile" ABI
  //   2 - Non-fragile ABI, version 1
  //   3 - Non-fragile ABI, version 2
  unsigned objcABIVersion = 1;
  // If -fobjc-abi-version= is present, use that to set the version.
  if (Arg *abiArg = args.getLastArg(options::OPT_fobjc_abi_version_EQ)) {
    StringRef value = abiArg->getValue();
    if (value == "1")
      objcABIVersion = 1;
    else if (value == "2")
      objcABIVersion = 2;
    else if (value == "3")
      objcABIVersion = 3;
    else
      getToolChain().getDriver().Diag(diag::err_drv_clang_unsupported) << value;
  } else {
    // Otherwise, determine if we are using the non-fragile ABI.
    bool nonFragileABIIsDefault =
        (rewriteKind == RK_NonFragile ||
         (rewriteKind == RK_None &&
          getToolChain().IsObjCNonFragileABIDefault()));
    if (args.hasFlag(options::OPT_fobjc_nonfragile_abi,
                     options::OPT_fno_objc_nonfragile_abi,
                     nonFragileABIIsDefault)) {
// Determine the non-fragile ABI version to use.
#ifdef DISABLE_DEFAULT_NONFRAGILEABI_TWO
      unsigned nonFragileABIVersion = 1;
#else
      unsigned nonFragileABIVersion = 2;
#endif

      if (Arg *abiArg =
              args.getLastArg(options::OPT_fobjc_nonfragile_abi_version_EQ)) {
        StringRef value = abiArg->getValue();
        if (value == "1")
          nonFragileABIVersion = 1;
        else if (value == "2")
          nonFragileABIVersion = 2;
        else
          getToolChain().getDriver().Diag(diag::err_drv_clang_unsupported)
              << value;
      }

      objcABIVersion = 1 + nonFragileABIVersion;
    } else {
      objcABIVersion = 1;
    }
  }

  // We don't actually care about the ABI version other than whether
  // it's non-fragile.
  bool isNonFragile = objcABIVersion != 1;

  // If we have no runtime argument, ask the toolchain for its default runtime.
  // However, the rewriter only really supports the Mac runtime, so assume that.
  ObjCRuntime runtime;
  if (!runtimeArg) {
    switch (rewriteKind) {
    case RK_None:
      runtime = getToolChain().getDefaultObjCRuntime(isNonFragile);
      break;
    case RK_Fragile:
      runtime = ObjCRuntime(ObjCRuntime::FragileMacOSX, VersionTuple());
      break;
    case RK_NonFragile:
      runtime = ObjCRuntime(ObjCRuntime::MacOSX, VersionTuple());
      break;
    }

    // -fnext-runtime
  } else if (runtimeArg->getOption().matches(options::OPT_fnext_runtime)) {
    // On Darwin, make this use the default behavior for the toolchain.
    if (getToolChain().getTriple().isOSDarwin()) {
      runtime = getToolChain().getDefaultObjCRuntime(isNonFragile);

      // Otherwise, build for a generic macosx port.
    } else {
      runtime = ObjCRuntime(ObjCRuntime::MacOSX, VersionTuple());
    }

    // -fgnu-runtime
  } else {
    assert(runtimeArg->getOption().matches(options::OPT_fgnu_runtime));
    // Legacy behaviour is to target the gnustep runtime if we are in
    // non-fragile mode or the GCC runtime in fragile mode.
    if (isNonFragile)
      runtime = ObjCRuntime(ObjCRuntime::GNUstep, VersionTuple(1, 6));
    else
      runtime = ObjCRuntime(ObjCRuntime::GCC, VersionTuple());
  }

  cmdArgs.push_back(
      args.MakeArgString("-fobjc-runtime=" + runtime.getAsString()));
  return runtime;
}

static bool maybeConsumeDash(const std::string &EH, size_t &I) {
  bool HaveDash = (I + 1 < EH.size() && EH[I + 1] == '-');
  I += HaveDash;
  return !HaveDash;
}

namespace {
struct EHFlags {
  bool Synch = false;
  bool Asynch = false;
  bool NoUnwindC = false;
};
} // end anonymous namespace

/// /EH controls whether to run destructor cleanups when exceptions are
/// thrown.  There are three modifiers:
/// - s: Cleanup after "synchronous" exceptions, aka C++ exceptions.
/// - a: Cleanup after "asynchronous" exceptions, aka structured exceptions.
///      The 'a' modifier is unimplemented and fundamentally hard in LLVM IR.
/// - c: Assume that extern "C" functions are implicitly nounwind.
/// The default is /EHs-c-, meaning cleanups are disabled.
static EHFlags parseClangCLEHFlags(const Driver &D, const ArgList &Args) {
  EHFlags EH;

  std::vector<std::string> EHArgs =
      Args.getAllArgValues(options::OPT__SLASH_EH);
  for (auto EHVal : EHArgs) {
    for (size_t I = 0, E = EHVal.size(); I != E; ++I) {
      switch (EHVal[I]) {
      case 'a':
        EH.Asynch = maybeConsumeDash(EHVal, I);
        if (EH.Asynch)
          EH.Synch = false;
        continue;
      case 'c':
        EH.NoUnwindC = maybeConsumeDash(EHVal, I);
        continue;
      case 's':
        EH.Synch = maybeConsumeDash(EHVal, I);
        if (EH.Synch)
          EH.Asynch = false;
        continue;
      default:
        break;
      }
      D.Diag(clang::diag::err_drv_invalid_value) << "/EH" << EHVal;
      break;
    }
  }
  // The /GX, /GX- flags are only processed if there are not /EH flags.
  // The default is that /GX is not specified.
  if (EHArgs.empty() &&
      Args.hasFlag(options::OPT__SLASH_GX, options::OPT__SLASH_GX_,
                   /*default=*/false)) {
    EH.Synch = true;
    EH.NoUnwindC = true;
  }

  return EH;
}

void Clang::AddClangCLArgs(const ArgList &Args, types::ID InputType,
                           ArgStringList &CmdArgs,
                           codegenoptions::DebugInfoKind *DebugInfoKind,
                           bool *EmitCodeView) const {
  unsigned RTOptionID = options::OPT__SLASH_MT;

  if (Args.hasArg(options::OPT__SLASH_LDd))
    // The /LDd option implies /MTd. The dependent lib part can be overridden,
    // but defining _DEBUG is sticky.
    RTOptionID = options::OPT__SLASH_MTd;

  if (Arg *A = Args.getLastArg(options::OPT__SLASH_M_Group))
    RTOptionID = A->getOption().getID();

  StringRef FlagForCRT;
  switch (RTOptionID) {
  case options::OPT__SLASH_MD:
    if (Args.hasArg(options::OPT__SLASH_LDd))
      CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("-D_DLL");
    FlagForCRT = "--dependent-lib=msvcrt";
    break;
  case options::OPT__SLASH_MDd:
    CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("-D_DLL");
    FlagForCRT = "--dependent-lib=msvcrtd";
    break;
  case options::OPT__SLASH_MT:
    if (Args.hasArg(options::OPT__SLASH_LDd))
      CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("-flto-visibility-public-std");
    FlagForCRT = "--dependent-lib=libcmt";
    break;
  case options::OPT__SLASH_MTd:
    CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("-flto-visibility-public-std");
    FlagForCRT = "--dependent-lib=libcmtd";
    break;
  default:
    llvm_unreachable("Unexpected option ID.");
  }

  if (Args.hasArg(options::OPT__SLASH_Zl)) {
    CmdArgs.push_back("-D_VC_NODEFAULTLIB");
  } else {
    CmdArgs.push_back(FlagForCRT.data());

    // This provides POSIX compatibility (maps 'open' to '_open'), which most
    // users want.  The /Za flag to cl.exe turns this off, but it's not
    // implemented in clang.
    CmdArgs.push_back("--dependent-lib=oldnames");
  }

  // Both /showIncludes and /E (and /EP) write to stdout. Allowing both
  // would produce interleaved output, so ignore /showIncludes in such cases.
  if (!Args.hasArg(options::OPT_E) && !Args.hasArg(options::OPT__SLASH_EP))
    if (Arg *A = Args.getLastArg(options::OPT_show_includes))
      A->render(Args, CmdArgs);

  // This controls whether or not we emit RTTI data for polymorphic types.
  if (Args.hasFlag(options::OPT__SLASH_GR_, options::OPT__SLASH_GR,
                   /*default=*/false))
    CmdArgs.push_back("-fno-rtti-data");

  // This controls whether or not we emit stack-protector instrumentation.
  // In MSVC, Buffer Security Check (/GS) is on by default.
  if (Args.hasFlag(options::OPT__SLASH_GS, options::OPT__SLASH_GS_,
                   /*default=*/true)) {
    CmdArgs.push_back("-stack-protector");
    CmdArgs.push_back(Args.MakeArgString(Twine(LangOptions::SSPStrong)));
  }

  // Emit CodeView if -Z7, -Zd, or -gline-tables-only are present.
  if (Arg *DebugInfoArg =
          Args.getLastArg(options::OPT__SLASH_Z7, options::OPT__SLASH_Zd,
                          options::OPT_gline_tables_only)) {
    *EmitCodeView = true;
    if (DebugInfoArg->getOption().matches(options::OPT__SLASH_Z7))
      *DebugInfoKind = codegenoptions::LimitedDebugInfo;
    else
      *DebugInfoKind = codegenoptions::DebugLineTablesOnly;
    CmdArgs.push_back("-gcodeview");
  } else {
    *EmitCodeView = false;
  }

  const Driver &D = getToolChain().getDriver();
  EHFlags EH = parseClangCLEHFlags(D, Args);
  if (EH.Synch || EH.Asynch) {
    if (types::isCXX(InputType))
      CmdArgs.push_back("-fcxx-exceptions");
    CmdArgs.push_back("-fexceptions");
  }
  if (types::isCXX(InputType) && EH.Synch && EH.NoUnwindC)
    CmdArgs.push_back("-fexternc-nounwind");

  // /EP should expand to -E -P.
  if (Args.hasArg(options::OPT__SLASH_EP)) {
    CmdArgs.push_back("-E");
    CmdArgs.push_back("-P");
  }

  unsigned VolatileOptionID;
  if (getToolChain().getArch() == llvm::Triple::x86_64 ||
      getToolChain().getArch() == llvm::Triple::x86)
    VolatileOptionID = options::OPT__SLASH_volatile_ms;
  else
    VolatileOptionID = options::OPT__SLASH_volatile_iso;

  if (Arg *A = Args.getLastArg(options::OPT__SLASH_volatile_Group))
    VolatileOptionID = A->getOption().getID();

  if (VolatileOptionID == options::OPT__SLASH_volatile_ms)
    CmdArgs.push_back("-fms-volatile");

  Arg *MostGeneralArg = Args.getLastArg(options::OPT__SLASH_vmg);
  Arg *BestCaseArg = Args.getLastArg(options::OPT__SLASH_vmb);
  if (MostGeneralArg && BestCaseArg)
    D.Diag(clang::diag::err_drv_argument_not_allowed_with)
        << MostGeneralArg->getAsString(Args) << BestCaseArg->getAsString(Args);

  if (MostGeneralArg) {
    Arg *SingleArg = Args.getLastArg(options::OPT__SLASH_vms);
    Arg *MultipleArg = Args.getLastArg(options::OPT__SLASH_vmm);
    Arg *VirtualArg = Args.getLastArg(options::OPT__SLASH_vmv);

    Arg *FirstConflict = SingleArg ? SingleArg : MultipleArg;
    Arg *SecondConflict = VirtualArg ? VirtualArg : MultipleArg;
    if (FirstConflict && SecondConflict && FirstConflict != SecondConflict)
      D.Diag(clang::diag::err_drv_argument_not_allowed_with)
          << FirstConflict->getAsString(Args)
          << SecondConflict->getAsString(Args);

    if (SingleArg)
      CmdArgs.push_back("-fms-memptr-rep=single");
    else if (MultipleArg)
      CmdArgs.push_back("-fms-memptr-rep=multiple");
    else
      CmdArgs.push_back("-fms-memptr-rep=virtual");
  }

  if (Args.getLastArg(options::OPT__SLASH_Gd))
     CmdArgs.push_back("-fdefault-calling-conv=cdecl");
  else if (Args.getLastArg(options::OPT__SLASH_Gr))
     CmdArgs.push_back("-fdefault-calling-conv=fastcall");
  else if (Args.getLastArg(options::OPT__SLASH_Gz))
     CmdArgs.push_back("-fdefault-calling-conv=stdcall");
  else if (Args.getLastArg(options::OPT__SLASH_Gv))
     CmdArgs.push_back("-fdefault-calling-conv=vectorcall");

  if (Arg *A = Args.getLastArg(options::OPT_vtordisp_mode_EQ))
    A->render(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_fdiagnostics_format_EQ)) {
    CmdArgs.push_back("-fdiagnostics-format");
    if (Args.hasArg(options::OPT__SLASH_fallback))
      CmdArgs.push_back("msvc-fallback");
    else
      CmdArgs.push_back("msvc");
  }
}

visualstudio::Compiler *Clang::getCLFallback() const {
  if (!CLFallback)
    CLFallback.reset(new visualstudio::Compiler(getToolChain()));
  return CLFallback.get();
}

void ClangAs::AddMIPSTargetArgs(const ArgList &Args,
                                ArgStringList &CmdArgs) const {
  StringRef CPUName;
  StringRef ABIName;
  const llvm::Triple &Triple = getToolChain().getTriple();
  mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);

  CmdArgs.push_back("-target-abi");
  CmdArgs.push_back(ABIName.data());
}

void ClangAs::AddX86TargetArgs(const ArgList &Args,
                               ArgStringList &CmdArgs) const {
  if (Arg *A = Args.getLastArg(options::OPT_masm_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "intel" || Value == "att") {
      CmdArgs.push_back("-mllvm");
      CmdArgs.push_back(Args.MakeArgString("-x86-asm-syntax=" + Value));
    } else {
      getToolChain().getDriver().Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << Value;
    }
  }
}

void ClangAs::ConstructJob(Compilation &C, const JobAction &JA,
                           const InputInfo &Output, const InputInfoList &Inputs,
                           const ArgList &Args,
                           const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unexpected number of inputs.");
  const InputInfo &Input = Inputs[0];

  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();
  const std::string &TripleStr = Triple.getTriple();

  // Don't warn about "clang -w -c foo.s"
  Args.ClaimAllArgs(options::OPT_w);
  // and "clang -emit-llvm -c foo.s"
  Args.ClaimAllArgs(options::OPT_emit_llvm);

  claimNoWarnArgs(Args);

  // Invoke ourselves in -cc1as mode.
  //
  // FIXME: Implement custom jobs for internal actions.
  CmdArgs.push_back("-cc1as");

  // Add the "effective" target triple.
  CmdArgs.push_back("-triple");
  CmdArgs.push_back(Args.MakeArgString(TripleStr));

  // Set the output mode, we currently only expect to be used as a real
  // assembler.
  CmdArgs.push_back("-filetype");
  CmdArgs.push_back("obj");

  // Set the main file name, so that debug info works even with
  // -save-temps or preprocessed assembly.
  CmdArgs.push_back("-main-file-name");
  CmdArgs.push_back(Clang::getBaseInputName(Args, Input));

  // Add the target cpu
  std::string CPU = getCPUName(Args, Triple, /*FromAs*/ true);
  if (!CPU.empty()) {
    CmdArgs.push_back("-target-cpu");
    CmdArgs.push_back(Args.MakeArgString(CPU));
  }

  // Add the target features
  getTargetFeatures(getToolChain(), Triple, Args, CmdArgs, true);

  // Ignore explicit -force_cpusubtype_ALL option.
  (void)Args.hasArg(options::OPT_force__cpusubtype__ALL);

  // Pass along any -I options so we get proper .include search paths.
  Args.AddAllArgs(CmdArgs, options::OPT_I_Group);

  // Determine the original source input.
  const Action *SourceAction = &JA;
  while (SourceAction->getKind() != Action::InputClass) {
    assert(!SourceAction->getInputs().empty() && "unexpected root action!");
    SourceAction = SourceAction->getInputs()[0];
  }

  // Forward -g and handle debug info related flags, assuming we are dealing
  // with an actual assembly file.
  bool WantDebug = false;
  unsigned DwarfVersion = 0;
  Args.ClaimAllArgs(options::OPT_g_Group);
  if (Arg *A = Args.getLastArg(options::OPT_g_Group)) {
    WantDebug = !A->getOption().matches(options::OPT_g0) &&
                !A->getOption().matches(options::OPT_ggdb0);
    if (WantDebug)
      DwarfVersion = DwarfVersionNum(A->getSpelling());
  }
  if (DwarfVersion == 0)
    DwarfVersion = getToolChain().GetDefaultDwarfVersion();

  codegenoptions::DebugInfoKind DebugInfoKind = codegenoptions::NoDebugInfo;

  if (SourceAction->getType() == types::TY_Asm ||
      SourceAction->getType() == types::TY_PP_Asm) {
    // You might think that it would be ok to set DebugInfoKind outside of
    // the guard for source type, however there is a test which asserts
    // that some assembler invocation receives no -debug-info-kind,
    // and it's not clear whether that test is just overly restrictive.
    DebugInfoKind = (WantDebug ? codegenoptions::LimitedDebugInfo
                               : codegenoptions::NoDebugInfo);
    // Add the -fdebug-compilation-dir flag if needed.
    addDebugCompDirArg(Args, CmdArgs);

    // Set the AT_producer to the clang version when using the integrated
    // assembler on assembly source files.
    CmdArgs.push_back("-dwarf-debug-producer");
    CmdArgs.push_back(Args.MakeArgString(getClangFullVersion()));

    // And pass along -I options
    Args.AddAllArgs(CmdArgs, options::OPT_I);
  }
  RenderDebugEnablingArgs(Args, CmdArgs, DebugInfoKind, DwarfVersion,
                          llvm::DebuggerKind::Default);

  // Handle -fPIC et al -- the relocation-model affects the assembler
  // for some targets.
  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) =
      ParsePICArgs(getToolChain(), Triple, Args);

  const char *RMName = RelocationModelName(RelocationModel);
  if (RMName) {
    CmdArgs.push_back("-mrelocation-model");
    CmdArgs.push_back(RMName);
  }

  // Optionally embed the -cc1as level arguments into the debug info, for build
  // analysis.
  if (getToolChain().UseDwarfDebugFlags()) {
    ArgStringList OriginalArgs;
    for (const auto &Arg : Args)
      Arg->render(Args, OriginalArgs);

    SmallString<256> Flags;
    const char *Exec = getToolChain().getDriver().getClangProgramPath();
    Flags += Exec;
    for (const char *OriginalArg : OriginalArgs) {
      SmallString<128> EscapedArg;
      EscapeSpacesAndBackslashes(OriginalArg, EscapedArg);
      Flags += " ";
      Flags += EscapedArg;
    }
    CmdArgs.push_back("-dwarf-debug-flags");
    CmdArgs.push_back(Args.MakeArgString(Flags));
  }

  // FIXME: Add -static support, once we have it.

  // Add target specific flags.
  switch (getToolChain().getArch()) {
  default:
    break;

  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    AddMIPSTargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    AddX86TargetArgs(Args, CmdArgs);
    break;
  }

  // Consume all the warning flags. Usually this would be handled more
  // gracefully by -cc1 (warning about unknown warning flags, etc) but -cc1as
  // doesn't handle that so rather than warning about unused flags that are
  // actually used, we'll lie by omission instead.
  // FIXME: Stop lying and consume only the appropriate driver flags
  Args.ClaimAllArgs(options::OPT_W_Group);

  CollectArgsForIntegratedAssembler(C, Args, CmdArgs,
                                    getToolChain().getDriver());

  Args.AddAllArgs(CmdArgs, options::OPT_mllvm);

  assert(Output.isFilename() && "Unexpected lipo output.");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  assert(Input.isFilename() && "Invalid input.");
  CmdArgs.push_back(Input.getFilename());

  const char *Exec = getToolChain().getDriver().getClangProgramPath();
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));

  // Handle the debug info splitting at object creation time if we're
  // creating an object.
  // TODO: Currently only works on linux with newer objcopy.
  if (Args.hasArg(options::OPT_gsplit_dwarf) &&
      getToolChain().getTriple().isOSLinux())
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output,
                   SplitDebugName(Args, Input));
}

void GnuTool::anchor() {}

void gcc::Common::ConstructJob(Compilation &C, const JobAction &JA,
                               const InputInfo &Output,
                               const InputInfoList &Inputs, const ArgList &Args,
                               const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  for (const auto &A : Args) {
    if (forwardToGCC(A->getOption())) {
      // It is unfortunate that we have to claim here, as this means
      // we will basically never report anything interesting for
      // platforms using a generic gcc, even if we are just using gcc
      // to get to the assembler.
      A->claim();

      // Don't forward any -g arguments to assembly steps.
      if (isa<AssembleJobAction>(JA) &&
          A->getOption().matches(options::OPT_g_Group))
        continue;

      // Don't forward any -W arguments to assembly and link steps.
      if ((isa<AssembleJobAction>(JA) || isa<LinkJobAction>(JA)) &&
          A->getOption().matches(options::OPT_W_Group))
        continue;

      A->render(Args, CmdArgs);
    }
  }

  RenderExtraToolArgs(JA, CmdArgs);

  // If using a driver driver, force the arch.
  if (getToolChain().getTriple().isOSDarwin()) {
    CmdArgs.push_back("-arch");
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().getDefaultUniversalArchName()));
  }

  // Try to force gcc to match the tool chain we want, if we recognize
  // the arch.
  //
  // FIXME: The triple class should directly provide the information we want
  // here.
  switch (getToolChain().getArch()) {
  default:
    break;
  case llvm::Triple::x86:
  case llvm::Triple::ppc:
    CmdArgs.push_back("-m32");
    break;
  case llvm::Triple::x86_64:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    CmdArgs.push_back("-m64");
    break;
  case llvm::Triple::sparcel:
    CmdArgs.push_back("-EL");
    break;
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Unexpected output");
    CmdArgs.push_back("-fsyntax-only");
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  // Only pass -x if gcc will understand it; otherwise hope gcc
  // understands the suffix correctly. The main use case this would go
  // wrong in is for linker inputs if they happened to have an odd
  // suffix; really the only way to get this to happen is a command
  // like '-x foobar a.c' which will treat a.c like a linker input.
  //
  // FIXME: For the linker case specifically, can we safely convert
  // inputs into '-Wl,' options?
  for (const auto &II : Inputs) {
    // Don't try to pass LLVM or AST inputs to a generic gcc.
    if (types::isLLVMIR(II.getType()))
      D.Diag(diag::err_drv_no_linker_llvm_support)
          << getToolChain().getTripleString();
    else if (II.getType() == types::TY_AST)
      D.Diag(diag::err_drv_no_ast_support) << getToolChain().getTripleString();
    else if (II.getType() == types::TY_ModuleFile)
      D.Diag(diag::err_drv_no_module_support)
          << getToolChain().getTripleString();

    if (types::canTypeBeUserSpecified(II.getType())) {
      CmdArgs.push_back("-x");
      CmdArgs.push_back(types::getTypeName(II.getType()));
    }

    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else {
      const Arg &A = II.getInputArg();

      // Reverse translate some rewritten options.
      if (A.getOption().matches(options::OPT_Z_reserved_lib_stdcxx)) {
        CmdArgs.push_back("-lstdc++");
        continue;
      }

      // Don't render as input, we need gcc to do the translations.
      A.render(Args, CmdArgs);
    }
  }

  const std::string &customGCCName = D.getCCCGenericGCCName();
  const char *GCCName;
  if (!customGCCName.empty())
    GCCName = customGCCName.c_str();
  else if (D.CCCIsCXX()) {
    GCCName = "g++";
  } else
    GCCName = "gcc";

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath(GCCName));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void gcc::Preprocessor::RenderExtraToolArgs(const JobAction &JA,
                                            ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-E");
}

void gcc::Compiler::RenderExtraToolArgs(const JobAction &JA,
                                        ArgStringList &CmdArgs) const {
  const Driver &D = getToolChain().getDriver();

  switch (JA.getType()) {
  // If -flto, etc. are present then make sure not to force assembly output.
  case types::TY_LLVM_IR:
  case types::TY_LTO_IR:
  case types::TY_LLVM_BC:
  case types::TY_LTO_BC:
    CmdArgs.push_back("-c");
    break;
  // We assume we've got an "integrated" assembler in that gcc will produce an
  // object file itself.
  case types::TY_Object:
    CmdArgs.push_back("-c");
    break;
  case types::TY_PP_Asm:
    CmdArgs.push_back("-S");
    break;
  case types::TY_Nothing:
    CmdArgs.push_back("-fsyntax-only");
    break;
  default:
    D.Diag(diag::err_drv_invalid_gcc_output_type) << getTypeName(JA.getType());
  }
}

void gcc::Linker::RenderExtraToolArgs(const JobAction &JA,
                                      ArgStringList &CmdArgs) const {
  // The types are (hopefully) good enough.
}

// Hexagon tools start.
void hexagon::Assembler::RenderExtraToolArgs(const JobAction &JA,
                                             ArgStringList &CmdArgs) const {
}

void hexagon::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);

  auto &HTC = static_cast<const toolchains::HexagonToolChain&>(getToolChain());
  const Driver &D = HTC.getDriver();
  ArgStringList CmdArgs;

  std::string MArchString = "-march=hexagon";
  CmdArgs.push_back(Args.MakeArgString(MArchString));

  RenderExtraToolArgs(JA, CmdArgs);

  std::string AsName = "hexagon-llvm-mc";
  std::string MCpuString = "-mcpu=hexagon" +
        toolchains::HexagonToolChain::GetTargetCPUVersion(Args).str();
  CmdArgs.push_back("-filetype=obj");
  CmdArgs.push_back(Args.MakeArgString(MCpuString));

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Unexpected output");
    CmdArgs.push_back("-fsyntax-only");
  }

  if (auto G = toolchains::HexagonToolChain::getSmallDataThreshold(Args)) {
    std::string N = llvm::utostr(G.getValue());
    CmdArgs.push_back(Args.MakeArgString(std::string("-gpsize=") + N));
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  // Only pass -x if gcc will understand it; otherwise hope gcc
  // understands the suffix correctly. The main use case this would go
  // wrong in is for linker inputs if they happened to have an odd
  // suffix; really the only way to get this to happen is a command
  // like '-x foobar a.c' which will treat a.c like a linker input.
  //
  // FIXME: For the linker case specifically, can we safely convert
  // inputs into '-Wl,' options?
  for (const auto &II : Inputs) {
    // Don't try to pass LLVM or AST inputs to a generic gcc.
    if (types::isLLVMIR(II.getType()))
      D.Diag(clang::diag::err_drv_no_linker_llvm_support)
          << HTC.getTripleString();
    else if (II.getType() == types::TY_AST)
      D.Diag(clang::diag::err_drv_no_ast_support)
          << HTC.getTripleString();
    else if (II.getType() == types::TY_ModuleFile)
      D.Diag(diag::err_drv_no_module_support)
          << HTC.getTripleString();

    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      // Don't render as input, we need gcc to do the translations.
      // FIXME: What is this?
      II.getInputArg().render(Args, CmdArgs);
  }

  auto *Exec = Args.MakeArgString(HTC.GetProgramPath(AsName.c_str()));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void hexagon::Linker::RenderExtraToolArgs(const JobAction &JA,
                                          ArgStringList &CmdArgs) const {
}

static void
constructHexagonLinkArgs(Compilation &C, const JobAction &JA,
                         const toolchains::HexagonToolChain &HTC,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, ArgStringList &CmdArgs,
                         const char *LinkingOutput) {

  const Driver &D = HTC.getDriver();

  //----------------------------------------------------------------------------
  //
  //----------------------------------------------------------------------------
  bool IsStatic = Args.hasArg(options::OPT_static);
  bool IsShared = Args.hasArg(options::OPT_shared);
  bool IsPIE = Args.hasArg(options::OPT_pie);
  bool IncStdLib = !Args.hasArg(options::OPT_nostdlib);
  bool IncStartFiles = !Args.hasArg(options::OPT_nostartfiles);
  bool IncDefLibs = !Args.hasArg(options::OPT_nodefaultlibs);
  bool UseG0 = false;
  bool UseShared = IsShared && !IsStatic;

  //----------------------------------------------------------------------------
  // Silence warnings for various options
  //----------------------------------------------------------------------------
  Args.ClaimAllArgs(options::OPT_g_Group);
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  Args.ClaimAllArgs(options::OPT_w); // Other warning options are already
                                     // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_static_libgcc);

  //----------------------------------------------------------------------------
  //
  //----------------------------------------------------------------------------
  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  if (Args.hasArg(options::OPT_r))
    CmdArgs.push_back("-r");

  for (const auto &Opt : HTC.ExtraOpts)
    CmdArgs.push_back(Opt.c_str());

  CmdArgs.push_back("-march=hexagon");
  std::string CpuVer =
        toolchains::HexagonToolChain::GetTargetCPUVersion(Args).str();
  std::string MCpuString = "-mcpu=hexagon" + CpuVer;
  CmdArgs.push_back(Args.MakeArgString(MCpuString));

  if (IsShared) {
    CmdArgs.push_back("-shared");
    // The following should be the default, but doing as hexagon-gcc does.
    CmdArgs.push_back("-call_shared");
  }

  if (IsStatic)
    CmdArgs.push_back("-static");

  if (IsPIE && !IsShared)
    CmdArgs.push_back("-pie");

  if (auto G = toolchains::HexagonToolChain::getSmallDataThreshold(Args)) {
    std::string N = llvm::utostr(G.getValue());
    CmdArgs.push_back(Args.MakeArgString(std::string("-G") + N));
    UseG0 = G.getValue() == 0;
  }

  //----------------------------------------------------------------------------
  //
  //----------------------------------------------------------------------------
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  //----------------------------------------------------------------------------
  // moslib
  //----------------------------------------------------------------------------
  std::vector<std::string> OsLibs;
  bool HasStandalone = false;

  for (const Arg *A : Args.filtered(options::OPT_moslib_EQ)) {
    A->claim();
    OsLibs.emplace_back(A->getValue());
    HasStandalone = HasStandalone || (OsLibs.back() == "standalone");
  }
  if (OsLibs.empty()) {
    OsLibs.push_back("standalone");
    HasStandalone = true;
  }

  //----------------------------------------------------------------------------
  // Start Files
  //----------------------------------------------------------------------------
  const std::string MCpuSuffix = "/" + CpuVer;
  const std::string MCpuG0Suffix = MCpuSuffix + "/G0";
  const std::string RootDir =
      HTC.getHexagonTargetDir(D.InstalledDir, D.PrefixDirs) + "/";
  const std::string StartSubDir =
      "hexagon/lib" + (UseG0 ? MCpuG0Suffix : MCpuSuffix);

  auto Find = [&HTC] (const std::string &RootDir, const std::string &SubDir,
                      const char *Name) -> std::string {
    std::string RelName = SubDir + Name;
    std::string P = HTC.GetFilePath(RelName.c_str());
    if (llvm::sys::fs::exists(P))
      return P;
    return RootDir + RelName;
  };

  if (IncStdLib && IncStartFiles) {
    if (!IsShared) {
      if (HasStandalone) {
        std::string Crt0SA = Find(RootDir, StartSubDir, "/crt0_standalone.o");
        CmdArgs.push_back(Args.MakeArgString(Crt0SA));
      }
      std::string Crt0 = Find(RootDir, StartSubDir, "/crt0.o");
      CmdArgs.push_back(Args.MakeArgString(Crt0));
    }
    std::string Init = UseShared
          ? Find(RootDir, StartSubDir + "/pic", "/initS.o")
          : Find(RootDir, StartSubDir, "/init.o");
    CmdArgs.push_back(Args.MakeArgString(Init));
  }

  //----------------------------------------------------------------------------
  // Library Search Paths
  //----------------------------------------------------------------------------
  const ToolChain::path_list &LibPaths = HTC.getFilePaths();
  for (const auto &LibPath : LibPaths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + LibPath));

  //----------------------------------------------------------------------------
  //
  //----------------------------------------------------------------------------
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_T_Group, options::OPT_e, options::OPT_s,
                   options::OPT_t, options::OPT_u_Group});

  AddLinkerInputs(HTC, Inputs, Args, CmdArgs);

  //----------------------------------------------------------------------------
  // Libraries
  //----------------------------------------------------------------------------
  if (IncStdLib && IncDefLibs) {
    if (D.CCCIsCXX()) {
      HTC.AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }

    CmdArgs.push_back("--start-group");

    if (!IsShared) {
      for (const std::string &Lib : OsLibs)
        CmdArgs.push_back(Args.MakeArgString("-l" + Lib));
      CmdArgs.push_back("-lc");
    }
    CmdArgs.push_back("-lgcc");

    CmdArgs.push_back("--end-group");
  }

  //----------------------------------------------------------------------------
  // End files
  //----------------------------------------------------------------------------
  if (IncStdLib && IncStartFiles) {
    std::string Fini = UseShared
          ? Find(RootDir, StartSubDir + "/pic", "/finiS.o")
          : Find(RootDir, StartSubDir, "/fini.o");
    CmdArgs.push_back(Args.MakeArgString(Fini));
  }
}

void hexagon::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  auto &HTC = static_cast<const toolchains::HexagonToolChain&>(getToolChain());

  ArgStringList CmdArgs;
  constructHexagonLinkArgs(C, JA, HTC, Output, Inputs, Args, CmdArgs,
                           LinkingOutput);

  std::string Linker = HTC.GetProgramPath("hexagon-link");
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Linker),
                                          CmdArgs, Inputs));
}
// Hexagon tools end.

void amdgpu::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {

  std::string Linker = getToolChain().GetProgramPath(getShortName());
  ArgStringList CmdArgs;
  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);
  CmdArgs.push_back("-shared");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Linker),
                                          CmdArgs, Inputs));
}
// AMDGPU tools end.

wasm::Linker::Linker(const ToolChain &TC)
  : GnuTool("wasm::Linker", "lld", TC) {}

bool wasm::Linker::isLinkJob() const {
  return true;
}

bool wasm::Linker::hasIntegratedCPP() const {
  return false;
}

void wasm::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {

  const ToolChain &ToolChain = getToolChain();
  const Driver &D = ToolChain.getDriver();
  const char *Linker = Args.MakeArgString(ToolChain.GetLinkerPath());
  ArgStringList CmdArgs;
  CmdArgs.push_back("-flavor");
  CmdArgs.push_back("ld");

  // Enable garbage collection of unused input sections by default, since code
  // size is of particular importance. This is significantly facilitated by
  // the enabling of -ffunction-sections and -fdata-sections in
  // Clang::ConstructJob.
  if (areOptimizationsEnabled(Args))
    CmdArgs.push_back("--gc-sections");

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");
  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("--strip-all");
  if (Args.hasArg(options::OPT_shared))
    CmdArgs.push_back("-shared");
  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-Bstatic");

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("rcrt1.o")));
    else if (Args.hasArg(options::OPT_pie))
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("Scrt1.o")));
    else
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt1.o")));

    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crti.o")));
  }

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX())
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);

    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");

    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lcompiler_rt");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles))
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  C.addCommand(llvm::make_unique<Command>(JA, *this, Linker, CmdArgs, Inputs));
}

const std::string arm::getARMArch(StringRef Arch, const llvm::Triple &Triple) {
  std::string MArch;
  if (!Arch.empty())
    MArch = Arch;
  else
    MArch = Triple.getArchName();
  MArch = StringRef(MArch).split("+").first.lower();

  // Handle -march=native.
  if (MArch == "native") {
    std::string CPU = llvm::sys::getHostCPUName();
    if (CPU != "generic") {
      // Translate the native cpu into the architecture suffix for that CPU.
      StringRef Suffix = arm::getLLVMArchSuffixForARM(CPU, MArch, Triple);
      // If there is no valid architecture suffix for this CPU we don't know how
      // to handle it, so return no architecture.
      if (Suffix.empty())
        MArch = "";
      else
        MArch = std::string("arm") + Suffix.str();
    }
  }

  return MArch;
}

/// Get the (LLVM) name of the minimum ARM CPU for the arch we are targeting.
StringRef arm::getARMCPUForMArch(StringRef Arch, const llvm::Triple &Triple) {
  std::string MArch = getARMArch(Arch, Triple);
  // getARMCPUForArch defaults to the triple if MArch is empty, but empty MArch
  // here means an -march=native that we can't handle, so instead return no CPU.
  if (MArch.empty())
    return StringRef();

  // We need to return an empty string here on invalid MArch values as the
  // various places that call this function can't cope with a null result.
  return Triple.getARMCPUForArch(MArch);
}

/// getARMTargetCPU - Get the (LLVM) name of the ARM cpu we are targeting.
std::string arm::getARMTargetCPU(StringRef CPU, StringRef Arch,
                                 const llvm::Triple &Triple) {
  // FIXME: Warn on inconsistent use of -mcpu and -march.
  // If we have -mcpu=, use that.
  if (!CPU.empty()) {
    std::string MCPU = StringRef(CPU).split("+").first.lower();
    // Handle -mcpu=native.
    if (MCPU == "native")
      return llvm::sys::getHostCPUName();
    else
      return MCPU;
  }

  return getARMCPUForMArch(Arch, Triple);
}

/// getLLVMArchSuffixForARM - Get the LLVM arch name to use for a particular
/// CPU  (or Arch, if CPU is generic).
// FIXME: This is redundant with -mcpu, why does LLVM use this.
StringRef arm::getLLVMArchSuffixForARM(StringRef CPU, StringRef Arch,
                                       const llvm::Triple &Triple) {
  unsigned ArchKind;
  if (CPU == "generic") {
    std::string ARMArch = tools::arm::getARMArch(Arch, Triple);
    ArchKind = llvm::ARM::parseArch(ARMArch);
    if (ArchKind == llvm::ARM::AK_INVALID)
      // In case of generic Arch, i.e. "arm",
      // extract arch from default cpu of the Triple
      ArchKind = llvm::ARM::parseCPUArch(Triple.getARMCPUForArch(ARMArch));
  } else {
    // FIXME: horrible hack to get around the fact that Cortex-A7 is only an
    // armv7k triple if it's actually been specified via "-arch armv7k".
    ArchKind = (Arch == "armv7k" || Arch == "thumbv7k")
                          ? (unsigned)llvm::ARM::AK_ARMV7K
                          : llvm::ARM::parseCPUArch(CPU);
  }
  if (ArchKind == llvm::ARM::AK_INVALID)
    return "";
  return llvm::ARM::getSubArch(ArchKind);
}

void arm::appendEBLinkFlags(const ArgList &Args, ArgStringList &CmdArgs,
                            const llvm::Triple &Triple) {
  if (Args.hasArg(options::OPT_r))
    return;

  // ARMv7 (and later) and ARMv6-M do not support BE-32, so instruct the linker
  // to generate BE-8 executables.
  if (getARMSubArchVersionNumber(Triple) >= 7 || isARMMProfile(Triple))
    CmdArgs.push_back("--be8");
}

mips::NanEncoding mips::getSupportedNanEncoding(StringRef &CPU) {
  // Strictly speaking, mips32r2 and mips64r2 are NanLegacy-only since Nan2008
  // was first introduced in Release 3. However, other compilers have
  // traditionally allowed it for Release 2 so we should do the same.
  return (NanEncoding)llvm::StringSwitch<int>(CPU)
      .Case("mips1", NanLegacy)
      .Case("mips2", NanLegacy)
      .Case("mips3", NanLegacy)
      .Case("mips4", NanLegacy)
      .Case("mips5", NanLegacy)
      .Case("mips32", NanLegacy)
      .Case("mips32r2", NanLegacy | Nan2008)
      .Case("mips32r3", NanLegacy | Nan2008)
      .Case("mips32r5", NanLegacy | Nan2008)
      .Case("mips32r6", Nan2008)
      .Case("mips64", NanLegacy)
      .Case("mips64r2", NanLegacy | Nan2008)
      .Case("mips64r3", NanLegacy | Nan2008)
      .Case("mips64r5", NanLegacy | Nan2008)
      .Case("mips64r6", Nan2008)
      .Default(NanLegacy);
}

bool mips::hasCompactBranches(StringRef &CPU) {
  // mips32r6 and mips64r6 have compact branches.
  return llvm::StringSwitch<bool>(CPU)
      .Case("mips32r6", true)
      .Case("mips64r6", true)
      .Default(false);
}

bool mips::hasMipsAbiArg(const ArgList &Args, const char *Value) {
  Arg *A = Args.getLastArg(options::OPT_mabi_EQ);
  return A && (A->getValue() == StringRef(Value));
}

bool mips::isUCLibc(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_m_libc_Group);
  return A && A->getOption().matches(options::OPT_muclibc);
}

bool mips::isNaN2008(const ArgList &Args, const llvm::Triple &Triple) {
  if (Arg *NaNArg = Args.getLastArg(options::OPT_mnan_EQ))
    return llvm::StringSwitch<bool>(NaNArg->getValue())
        .Case("2008", true)
        .Case("legacy", false)
        .Default(false);

  // NaN2008 is the default for MIPS32r6/MIPS64r6.
  return llvm::StringSwitch<bool>(getCPUName(Args, Triple))
      .Cases("mips32r6", "mips64r6", true)
      .Default(false);

  return false;
}

bool mips::isFP64ADefault(const llvm::Triple &Triple, StringRef CPUName) {
  if (!Triple.isAndroid())
    return false;

  // Android MIPS32R6 defaults to FP64A.
  return llvm::StringSwitch<bool>(CPUName)
      .Case("mips32r6", true)
      .Default(false);
}

bool mips::isFPXXDefault(const llvm::Triple &Triple, StringRef CPUName,
                         StringRef ABIName, mips::FloatABI FloatABI) {
  if (Triple.getVendor() != llvm::Triple::ImaginationTechnologies &&
      Triple.getVendor() != llvm::Triple::MipsTechnologies &&
      !Triple.isAndroid())
    return false;

  if (ABIName != "32")
    return false;

  // FPXX shouldn't be used if either -msoft-float or -mfloat-abi=soft is
  // present.
  if (FloatABI == mips::FloatABI::Soft)
    return false;

  return llvm::StringSwitch<bool>(CPUName)
      .Cases("mips2", "mips3", "mips4", "mips5", true)
      .Cases("mips32", "mips32r2", "mips32r3", "mips32r5", true)
      .Cases("mips64", "mips64r2", "mips64r3", "mips64r5", true)
      .Default(false);
}

bool mips::shouldUseFPXX(const ArgList &Args, const llvm::Triple &Triple,
                         StringRef CPUName, StringRef ABIName,
                         mips::FloatABI FloatABI) {
  bool UseFPXX = isFPXXDefault(Triple, CPUName, ABIName, FloatABI);

  // FPXX shouldn't be used if -msingle-float is present.
  if (Arg *A = Args.getLastArg(options::OPT_msingle_float,
                               options::OPT_mdouble_float))
    if (A->getOption().matches(options::OPT_msingle_float))
      UseFPXX = false;

  return UseFPXX;
}

llvm::Triple::ArchType darwin::getArchTypeForMachOArchName(StringRef Str) {
  // See arch(3) and llvm-gcc's driver-driver.c. We don't implement support for
  // archs which Darwin doesn't use.

  // The matching this routine does is fairly pointless, since it is neither the
  // complete architecture list, nor a reasonable subset. The problem is that
  // historically the driver driver accepts this and also ties its -march=
  // handling to the architecture name, so we need to be careful before removing
  // support for it.

  // This code must be kept in sync with Clang's Darwin specific argument
  // translation.

  return llvm::StringSwitch<llvm::Triple::ArchType>(Str)
      .Cases("ppc", "ppc601", "ppc603", "ppc604", "ppc604e", llvm::Triple::ppc)
      .Cases("ppc750", "ppc7400", "ppc7450", "ppc970", llvm::Triple::ppc)
      .Case("ppc64", llvm::Triple::ppc64)
      .Cases("i386", "i486", "i486SX", "i586", "i686", llvm::Triple::x86)
      .Cases("pentium", "pentpro", "pentIIm3", "pentIIm5", "pentium4",
             llvm::Triple::x86)
      .Cases("x86_64", "x86_64h", llvm::Triple::x86_64)
      // This is derived from the driver driver.
      .Cases("arm", "armv4t", "armv5", "armv6", "armv6m", llvm::Triple::arm)
      .Cases("armv7", "armv7em", "armv7k", "armv7m", llvm::Triple::arm)
      .Cases("armv7s", "xscale", llvm::Triple::arm)
      .Case("arm64", llvm::Triple::aarch64)
      .Case("r600", llvm::Triple::r600)
      .Case("amdgcn", llvm::Triple::amdgcn)
      .Case("nvptx", llvm::Triple::nvptx)
      .Case("nvptx64", llvm::Triple::nvptx64)
      .Case("amdil", llvm::Triple::amdil)
      .Case("spir", llvm::Triple::spir)
      .Default(llvm::Triple::UnknownArch);
}

void darwin::setTripleTypeForMachOArchName(llvm::Triple &T, StringRef Str) {
  const llvm::Triple::ArchType Arch = getArchTypeForMachOArchName(Str);
  unsigned ArchKind = llvm::ARM::parseArch(Str);
  T.setArch(Arch);

  if (Str == "x86_64h")
    T.setArchName(Str);
  else if (ArchKind == llvm::ARM::AK_ARMV6M ||
           ArchKind == llvm::ARM::AK_ARMV7M ||
           ArchKind == llvm::ARM::AK_ARMV7EM) {
    T.setOS(llvm::Triple::UnknownOS);
    T.setObjectFormat(llvm::Triple::MachO);
  }
}

const char *Clang::getBaseInputName(const ArgList &Args,
                                    const InputInfo &Input) {
  return Args.MakeArgString(llvm::sys::path::filename(Input.getBaseInput()));
}

const char *Clang::getBaseInputStem(const ArgList &Args,
                                    const InputInfoList &Inputs) {
  const char *Str = getBaseInputName(Args, Inputs[0]);

  if (const char *End = strrchr(Str, '.'))
    return Args.MakeArgString(std::string(Str, End));

  return Str;
}

const char *Clang::getDependencyFileName(const ArgList &Args,
                                         const InputInfoList &Inputs) {
  // FIXME: Think about this more.
  std::string Res;

  if (Arg *OutputOpt = Args.getLastArg(options::OPT_o)) {
    std::string Str(OutputOpt->getValue());
    Res = Str.substr(0, Str.rfind('.'));
  } else {
    Res = getBaseInputStem(Args, Inputs);
  }
  return Args.MakeArgString(Res + ".d");
}

void cloudabi::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  const ToolChain &ToolChain = getToolChain();
  const Driver &D = ToolChain.getDriver();
  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  // CloudABI only supports static linkage.
  CmdArgs.push_back("-Bstatic");
  CmdArgs.push_back("--no-dynamic-linker");

  // Provide PIE linker flags in case PIE is default for the architecture.
  if (ToolChain.isPIEDefault()) {
    CmdArgs.push_back("-pie");
    CmdArgs.push_back("-zrelro");
  }

  CmdArgs.push_back("--eh-frame-hdr");
  CmdArgs.push_back("--gc-sections");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtbegin.o")));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_T_Group, options::OPT_e, options::OPT_s,
                   options::OPT_t, options::OPT_Z_Flag, options::OPT_r});

  if (D.isUsingLTO())
    AddGoldPlugin(ToolChain, Args, CmdArgs, D.getLTOMode() == LTOK_Thin, D);

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX())
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lcompiler_rt");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles))
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtend.o")));

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void darwin::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unexpected number of inputs.");
  const InputInfo &Input = Inputs[0];

  // Determine the original source input.
  const Action *SourceAction = &JA;
  while (SourceAction->getKind() != Action::InputClass) {
    assert(!SourceAction->getInputs().empty() && "unexpected root action!");
    SourceAction = SourceAction->getInputs()[0];
  }

  // If -fno-integrated-as is used add -Q to the darwin assember driver to make
  // sure it runs its system assembler not clang's integrated assembler.
  // Applicable to darwin11+ and Xcode 4+.  darwin<10 lacked integrated-as.
  // FIXME: at run-time detect assembler capabilities or rely on version
  // information forwarded by -target-assembler-version.
  if (Args.hasArg(options::OPT_fno_integrated_as)) {
    const llvm::Triple &T(getToolChain().getTriple());
    if (!(T.isMacOSX() && T.isMacOSXVersionLT(10, 7)))
      CmdArgs.push_back("-Q");
  }

  // Forward -g, assuming we are dealing with an actual assembly file.
  if (SourceAction->getType() == types::TY_Asm ||
      SourceAction->getType() == types::TY_PP_Asm) {
    if (Args.hasArg(options::OPT_gstabs))
      CmdArgs.push_back("--gstabs");
    else if (Args.hasArg(options::OPT_g_Group))
      CmdArgs.push_back("-g");
  }

  // Derived from asm spec.
  AddMachOArch(Args, CmdArgs);

  // Use -force_cpusubtype_ALL on x86 by default.
  if (getToolChain().getArch() == llvm::Triple::x86 ||
      getToolChain().getArch() == llvm::Triple::x86_64 ||
      Args.hasArg(options::OPT_force__cpusubtype__ALL))
    CmdArgs.push_back("-force_cpusubtype_ALL");

  if (getToolChain().getArch() != llvm::Triple::x86_64 &&
      (((Args.hasArg(options::OPT_mkernel) ||
         Args.hasArg(options::OPT_fapple_kext)) &&
        getMachOToolChain().isKernelStatic()) ||
       Args.hasArg(options::OPT_static)))
    CmdArgs.push_back("-static");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  assert(Output.isFilename() && "Unexpected lipo output.");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  assert(Input.isFilename() && "Invalid input.");
  CmdArgs.push_back(Input.getFilename());

  // asm_final spec is empty.

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void darwin::MachOTool::anchor() {}

void darwin::MachOTool::AddMachOArch(const ArgList &Args,
                                     ArgStringList &CmdArgs) const {
  StringRef ArchName = getMachOToolChain().getMachOArchName(Args);

  // Derived from darwin_arch spec.
  CmdArgs.push_back("-arch");
  CmdArgs.push_back(Args.MakeArgString(ArchName));

  // FIXME: Is this needed anymore?
  if (ArchName == "arm")
    CmdArgs.push_back("-force_cpusubtype_ALL");
}

bool darwin::Linker::NeedsTempPath(const InputInfoList &Inputs) const {
  // We only need to generate a temp path for LTO if we aren't compiling object
  // files. When compiling source files, we run 'dsymutil' after linking. We
  // don't run 'dsymutil' when compiling object files.
  for (const auto &Input : Inputs)
    if (Input.getType() != types::TY_Object)
      return true;

  return false;
}

void darwin::Linker::AddLinkArgs(Compilation &C, const ArgList &Args,
                                 ArgStringList &CmdArgs,
                                 const InputInfoList &Inputs) const {
  const Driver &D = getToolChain().getDriver();
  const toolchains::MachO &MachOTC = getMachOToolChain();

  unsigned Version[5] = {0, 0, 0, 0, 0};
  if (Arg *A = Args.getLastArg(options::OPT_mlinker_version_EQ)) {
    if (!Driver::GetReleaseVersion(A->getValue(), Version))
      D.Diag(diag::err_drv_invalid_version_number) << A->getAsString(Args);
  }

  // Newer linkers support -demangle. Pass it if supported and not disabled by
  // the user.
  if (Version[0] >= 100 && !Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("-demangle");

  if (Args.hasArg(options::OPT_rdynamic) && Version[0] >= 137)
    CmdArgs.push_back("-export_dynamic");

  // If we are using App Extension restrictions, pass a flag to the linker
  // telling it that the compiled code has been audited.
  if (Args.hasFlag(options::OPT_fapplication_extension,
                   options::OPT_fno_application_extension, false))
    CmdArgs.push_back("-application_extension");

  if (D.isUsingLTO()) {
    // If we are using LTO, then automatically create a temporary file path for
    // the linker to use, so that it's lifetime will extend past a possible
    // dsymutil step.
    if (Version[0] >= 116 && NeedsTempPath(Inputs)) {
      const char *TmpPath = C.getArgs().MakeArgString(
          D.GetTemporaryPath("cc", types::getTypeTempSuffix(types::TY_Object)));
      C.addTempFile(TmpPath);
      CmdArgs.push_back("-object_path_lto");
      CmdArgs.push_back(TmpPath);
    }

    // Use -lto_library option to specify the libLTO.dylib path. Try to find
    // it in clang installed libraries. If not found, the option is not used
    // and 'ld' will use its default mechanism to search for libLTO.dylib.
    if (Version[0] >= 133) {
      // Search for libLTO in <InstalledDir>/../lib/libLTO.dylib
      StringRef P = llvm::sys::path::parent_path(D.getInstalledDir());
      SmallString<128> LibLTOPath(P);
      llvm::sys::path::append(LibLTOPath, "lib");
      llvm::sys::path::append(LibLTOPath, "libLTO.dylib");
      if (llvm::sys::fs::exists(LibLTOPath)) {
        CmdArgs.push_back("-lto_library");
        CmdArgs.push_back(C.getArgs().MakeArgString(LibLTOPath));
      } else {
        D.Diag(diag::warn_drv_lto_libpath);
      }
    }
  }

  // Derived from the "link" spec.
  Args.AddAllArgs(CmdArgs, options::OPT_static);
  if (!Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-dynamic");
  if (Args.hasArg(options::OPT_fgnu_runtime)) {
    // FIXME: gcc replaces -lobjc in forward args with -lobjc-gnu
    // here. How do we wish to handle such things?
  }

  if (!Args.hasArg(options::OPT_dynamiclib)) {
    AddMachOArch(Args, CmdArgs);
    // FIXME: Why do this only on this path?
    Args.AddLastArg(CmdArgs, options::OPT_force__cpusubtype__ALL);

    Args.AddLastArg(CmdArgs, options::OPT_bundle);
    Args.AddAllArgs(CmdArgs, options::OPT_bundle__loader);
    Args.AddAllArgs(CmdArgs, options::OPT_client__name);

    Arg *A;
    if ((A = Args.getLastArg(options::OPT_compatibility__version)) ||
        (A = Args.getLastArg(options::OPT_current__version)) ||
        (A = Args.getLastArg(options::OPT_install__name)))
      D.Diag(diag::err_drv_argument_only_allowed_with) << A->getAsString(Args)
                                                       << "-dynamiclib";

    Args.AddLastArg(CmdArgs, options::OPT_force__flat__namespace);
    Args.AddLastArg(CmdArgs, options::OPT_keep__private__externs);
    Args.AddLastArg(CmdArgs, options::OPT_private__bundle);
  } else {
    CmdArgs.push_back("-dylib");

    Arg *A;
    if ((A = Args.getLastArg(options::OPT_bundle)) ||
        (A = Args.getLastArg(options::OPT_bundle__loader)) ||
        (A = Args.getLastArg(options::OPT_client__name)) ||
        (A = Args.getLastArg(options::OPT_force__flat__namespace)) ||
        (A = Args.getLastArg(options::OPT_keep__private__externs)) ||
        (A = Args.getLastArg(options::OPT_private__bundle)))
      D.Diag(diag::err_drv_argument_not_allowed_with) << A->getAsString(Args)
                                                      << "-dynamiclib";

    Args.AddAllArgsTranslated(CmdArgs, options::OPT_compatibility__version,
                              "-dylib_compatibility_version");
    Args.AddAllArgsTranslated(CmdArgs, options::OPT_current__version,
                              "-dylib_current_version");

    AddMachOArch(Args, CmdArgs);

    Args.AddAllArgsTranslated(CmdArgs, options::OPT_install__name,
                              "-dylib_install_name");
  }

  Args.AddLastArg(CmdArgs, options::OPT_all__load);
  Args.AddAllArgs(CmdArgs, options::OPT_allowable__client);
  Args.AddLastArg(CmdArgs, options::OPT_bind__at__load);
  if (MachOTC.isTargetIOSBased())
    Args.AddLastArg(CmdArgs, options::OPT_arch__errors__fatal);
  Args.AddLastArg(CmdArgs, options::OPT_dead__strip);
  Args.AddLastArg(CmdArgs, options::OPT_no__dead__strip__inits__and__terms);
  Args.AddAllArgs(CmdArgs, options::OPT_dylib__file);
  Args.AddLastArg(CmdArgs, options::OPT_dynamic);
  Args.AddAllArgs(CmdArgs, options::OPT_exported__symbols__list);
  Args.AddLastArg(CmdArgs, options::OPT_flat__namespace);
  Args.AddAllArgs(CmdArgs, options::OPT_force__load);
  Args.AddAllArgs(CmdArgs, options::OPT_headerpad__max__install__names);
  Args.AddAllArgs(CmdArgs, options::OPT_image__base);
  Args.AddAllArgs(CmdArgs, options::OPT_init);

  // Add the deployment target.
  MachOTC.addMinVersionArgs(Args, CmdArgs);

  Args.AddLastArg(CmdArgs, options::OPT_nomultidefs);
  Args.AddLastArg(CmdArgs, options::OPT_multi__module);
  Args.AddLastArg(CmdArgs, options::OPT_single__module);
  Args.AddAllArgs(CmdArgs, options::OPT_multiply__defined);
  Args.AddAllArgs(CmdArgs, options::OPT_multiply__defined__unused);

  if (const Arg *A =
          Args.getLastArg(options::OPT_fpie, options::OPT_fPIE,
                          options::OPT_fno_pie, options::OPT_fno_PIE)) {
    if (A->getOption().matches(options::OPT_fpie) ||
        A->getOption().matches(options::OPT_fPIE))
      CmdArgs.push_back("-pie");
    else
      CmdArgs.push_back("-no_pie");
  }
  // for embed-bitcode, use -bitcode_bundle in linker command
  if (C.getDriver().embedBitcodeEnabled() ||
      C.getDriver().embedBitcodeMarkerOnly()) {
    // Check if the toolchain supports bitcode build flow.
    if (MachOTC.SupportsEmbeddedBitcode())
      CmdArgs.push_back("-bitcode_bundle");
    else
      D.Diag(diag::err_drv_bitcode_unsupported_on_toolchain);
  }

  Args.AddLastArg(CmdArgs, options::OPT_prebind);
  Args.AddLastArg(CmdArgs, options::OPT_noprebind);
  Args.AddLastArg(CmdArgs, options::OPT_nofixprebinding);
  Args.AddLastArg(CmdArgs, options::OPT_prebind__all__twolevel__modules);
  Args.AddLastArg(CmdArgs, options::OPT_read__only__relocs);
  Args.AddAllArgs(CmdArgs, options::OPT_sectcreate);
  Args.AddAllArgs(CmdArgs, options::OPT_sectorder);
  Args.AddAllArgs(CmdArgs, options::OPT_seg1addr);
  Args.AddAllArgs(CmdArgs, options::OPT_segprot);
  Args.AddAllArgs(CmdArgs, options::OPT_segaddr);
  Args.AddAllArgs(CmdArgs, options::OPT_segs__read__only__addr);
  Args.AddAllArgs(CmdArgs, options::OPT_segs__read__write__addr);
  Args.AddAllArgs(CmdArgs, options::OPT_seg__addr__table);
  Args.AddAllArgs(CmdArgs, options::OPT_seg__addr__table__filename);
  Args.AddAllArgs(CmdArgs, options::OPT_sub__library);
  Args.AddAllArgs(CmdArgs, options::OPT_sub__umbrella);

  // Give --sysroot= preference, over the Apple specific behavior to also use
  // --isysroot as the syslibroot.
  StringRef sysroot = C.getSysRoot();
  if (sysroot != "") {
    CmdArgs.push_back("-syslibroot");
    CmdArgs.push_back(C.getArgs().MakeArgString(sysroot));
  } else if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
    CmdArgs.push_back("-syslibroot");
    CmdArgs.push_back(A->getValue());
  }

  Args.AddLastArg(CmdArgs, options::OPT_twolevel__namespace);
  Args.AddLastArg(CmdArgs, options::OPT_twolevel__namespace__hints);
  Args.AddAllArgs(CmdArgs, options::OPT_umbrella);
  Args.AddAllArgs(CmdArgs, options::OPT_undefined);
  Args.AddAllArgs(CmdArgs, options::OPT_unexported__symbols__list);
  Args.AddAllArgs(CmdArgs, options::OPT_weak__reference__mismatches);
  Args.AddLastArg(CmdArgs, options::OPT_X_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_y);
  Args.AddLastArg(CmdArgs, options::OPT_w);
  Args.AddAllArgs(CmdArgs, options::OPT_pagezero__size);
  Args.AddAllArgs(CmdArgs, options::OPT_segs__read__);
  Args.AddLastArg(CmdArgs, options::OPT_seglinkedit);
  Args.AddLastArg(CmdArgs, options::OPT_noseglinkedit);
  Args.AddAllArgs(CmdArgs, options::OPT_sectalign);
  Args.AddAllArgs(CmdArgs, options::OPT_sectobjectsymbols);
  Args.AddAllArgs(CmdArgs, options::OPT_segcreate);
  Args.AddLastArg(CmdArgs, options::OPT_whyload);
  Args.AddLastArg(CmdArgs, options::OPT_whatsloaded);
  Args.AddAllArgs(CmdArgs, options::OPT_dylinker__install__name);
  Args.AddLastArg(CmdArgs, options::OPT_dylinker);
  Args.AddLastArg(CmdArgs, options::OPT_Mach);
}

void darwin::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  assert(Output.getType() == types::TY_Image && "Invalid linker output type.");

  // If the number of arguments surpasses the system limits, we will encode the
  // input files in a separate file, shortening the command line. To this end,
  // build a list of input file names that can be passed via a file with the
  // -filelist linker option.
  llvm::opt::ArgStringList InputFileList;

  // The logic here is derived from gcc's behavior; most of which
  // comes from specs (starting with link_command). Consult gcc for
  // more information.
  ArgStringList CmdArgs;

  /// Hack(tm) to ignore linking errors when we are doing ARC migration.
  if (Args.hasArg(options::OPT_ccc_arcmt_check,
                  options::OPT_ccc_arcmt_migrate)) {
    for (const auto &Arg : Args)
      Arg->claim();
    const char *Exec =
        Args.MakeArgString(getToolChain().GetProgramPath("touch"));
    CmdArgs.push_back(Output.getFilename());
    C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, None));
    return;
  }

  // I'm not sure why this particular decomposition exists in gcc, but
  // we follow suite for ease of comparison.
  AddLinkArgs(C, Args, CmdArgs, Inputs);

  // It seems that the 'e' option is completely ignored for dynamic executables
  // (the default), and with static executables, the last one wins, as expected.
  Args.AddAllArgs(CmdArgs, {options::OPT_d_Flag, options::OPT_s, options::OPT_t,
                            options::OPT_Z_Flag, options::OPT_u_Group,
                            options::OPT_e, options::OPT_r});

  // Forward -ObjC when either -ObjC or -ObjC++ is used, to force loading
  // members of static archive libraries which implement Objective-C classes or
  // categories.
  if (Args.hasArg(options::OPT_ObjC) || Args.hasArg(options::OPT_ObjCXX))
    CmdArgs.push_back("-ObjC");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles))
    getMachOToolChain().addStartObjectFileArgs(Args, CmdArgs);

  // SafeStack requires its own runtime libraries
  // These libraries should be linked first, to make sure the
  // __safestack_init constructor executes before everything else
  if (getToolChain().getSanitizerArgs().needsSafeStackRt()) {
    getMachOToolChain().AddLinkRuntimeLib(Args, CmdArgs,
                                          "libclang_rt.safestack_osx.a",
                                          /*AlwaysLink=*/true);
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);
  // Build the input file for -filelist (list of linker input files) in case we
  // need it later
  for (const auto &II : Inputs) {
    if (!II.isFilename()) {
      // This is a linker input argument.
      // We cannot mix input arguments and file names in a -filelist input, thus
      // we prematurely stop our list (remaining files shall be passed as
      // arguments).
      if (InputFileList.size() > 0)
        break;

      continue;
    }

    InputFileList.push_back(II.getFilename());
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs))
    addOpenMPRuntime(CmdArgs, getToolChain(), Args);

  if (isObjCRuntimeLinked(Args) &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    // We use arclite library for both ARC and subscripting support.
    getMachOToolChain().AddLinkARCArgs(Args, CmdArgs);

    CmdArgs.push_back("-framework");
    CmdArgs.push_back("Foundation");
    // Link libobj.
    CmdArgs.push_back("-lobjc");
  }

  if (LinkingOutput) {
    CmdArgs.push_back("-arch_multiple");
    CmdArgs.push_back("-final_output");
    CmdArgs.push_back(LinkingOutput);
  }

  if (Args.hasArg(options::OPT_fnested_functions))
    CmdArgs.push_back("-allow_stack_execute");

  getMachOToolChain().addProfileRTLibs(Args, CmdArgs);

  if (unsigned Parallelism =
          getLTOParallelism(Args, getToolChain().getDriver())) {
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back(
        Args.MakeArgString(Twine("-threads=") + llvm::to_string(Parallelism)));
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (getToolChain().getDriver().CCCIsCXX())
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);

    // link_ssp spec is empty.

    // Let the tool chain choose which runtime library to link.
    getMachOToolChain().AddLinkRuntimeLibArgs(Args, CmdArgs);
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    // endfile_spec is empty.
  }

  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_F);

  // -iframework should be forwarded as -F.
  for (const Arg *A : Args.filtered(options::OPT_iframework))
    CmdArgs.push_back(Args.MakeArgString(std::string("-F") + A->getValue()));

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (Arg *A = Args.getLastArg(options::OPT_fveclib)) {
      if (A->getValue() == StringRef("Accelerate")) {
        CmdArgs.push_back("-framework");
        CmdArgs.push_back("Accelerate");
      }
    }
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  std::unique_ptr<Command> Cmd =
      llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs);
  Cmd->setInputFileList(std::move(InputFileList));
  C.addCommand(std::move(Cmd));
}

void darwin::Lipo::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  CmdArgs.push_back("-create");
  assert(Output.isFilename() && "Unexpected lipo output.");

  CmdArgs.push_back("-output");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs) {
    assert(II.isFilename() && "Unexpected lipo input.");
    CmdArgs.push_back(II.getFilename());
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("lipo"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void darwin::Dsymutil::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  assert(Inputs.size() == 1 && "Unable to handle multiple inputs.");
  const InputInfo &Input = Inputs[0];
  assert(Input.isFilename() && "Unexpected dsymutil input.");
  CmdArgs.push_back(Input.getFilename());

  const char *Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("dsymutil"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void darwin::VerifyDebug::ConstructJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  CmdArgs.push_back("--verify");
  CmdArgs.push_back("--debug-info");
  CmdArgs.push_back("--eh-frame");
  CmdArgs.push_back("--quiet");

  assert(Inputs.size() == 1 && "Unable to handle multiple inputs.");
  const InputInfo &Input = Inputs[0];
  assert(Input.isFilename() && "Unexpected verify input");

  // Grabbing the output of the earlier dsymutil run.
  CmdArgs.push_back(Input.getFilename());

  const char *Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("dwarfdump"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void solaris::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void solaris::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  // Demangle C++ names in errors
  CmdArgs.push_back("-C");

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_shared)) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("_start");
  }

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
    CmdArgs.push_back("-dn");
  } else {
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else {
      CmdArgs.push_back("--dynamic-linker");
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("ld.so.1")));
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crt1.o")));

    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("values-Xa.o")));
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
  }

  getToolChain().AddFilePathLibArgs(Args, CmdArgs);

  Args.AddAllArgs(CmdArgs, {options::OPT_L, options::OPT_T_Group,
                            options::OPT_e, options::OPT_r});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (getToolChain().getDriver().CCCIsCXX())
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
    CmdArgs.push_back("-lgcc_s");
    CmdArgs.push_back("-lc");
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-lgcc");
      CmdArgs.push_back("-lm");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
  }
  CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));

  getToolChain().addProfileRTLibs(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void openbsd::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  switch (getToolChain().getArch()) {
  case llvm::Triple::x86:
    // When building 32-bit code on OpenBSD/amd64, we have to explicitly
    // instruct as in the base system to assemble 32-bit code.
    CmdArgs.push_back("--32");
    break;

  case llvm::Triple::ppc:
    CmdArgs.push_back("-mppc");
    CmdArgs.push_back("-many");
    break;

  case llvm::Triple::sparc:
  case llvm::Triple::sparcel: {
    CmdArgs.push_back("-32");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  case llvm::Triple::sparcv9: {
    CmdArgs.push_back("-64");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, getToolChain().getTriple(), CPUName, ABIName);

    CmdArgs.push_back("-mabi");
    CmdArgs.push_back(getGnuCompatibleMipsABIName(ABIName).data());

    if (getToolChain().getArch() == llvm::Triple::mips64)
      CmdArgs.push_back("-EB");
    else
      CmdArgs.push_back("-EL");

    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  default:
    break;
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void openbsd::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (getToolChain().getArch() == llvm::Triple::mips64)
    CmdArgs.push_back("-EB");
  else if (getToolChain().getArch() == llvm::Triple::mips64el)
    CmdArgs.push_back("-EL");

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_shared)) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("__start");
  }

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    CmdArgs.push_back("--eh-frame-hdr");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/usr/libexec/ld.so");
    }
  }

  if (Args.hasArg(options::OPT_nopie))
    CmdArgs.push_back("-nopie");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("gcrt0.o")));
      else
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("crt0.o")));
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    } else {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbeginS.o")));
    }
  }

  std::string Triple = getToolChain().getTripleString();
  if (Triple.substr(0, 6) == "x86_64")
    Triple.replace(0, 6, "amd64");
  CmdArgs.push_back(
      Args.MakeArgString("-L/usr/lib/gcc-lib/" + Triple + "/4.2.1"));

  Args.AddAllArgs(CmdArgs, {options::OPT_L, options::OPT_T_Group,
                            options::OPT_e, options::OPT_s, options::OPT_t,
                            options::OPT_Z_Flag, options::OPT_r});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX()) {
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lm_p");
      else
        CmdArgs.push_back("-lm");
    }

    // FIXME: For some reason GCC passes -lgcc before adding
    // the default system libraries. Just mimic this for now.
    CmdArgs.push_back("-lgcc");

    if (Args.hasArg(options::OPT_pthread)) {
      if (!Args.hasArg(options::OPT_shared) && Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lpthread_p");
      else
        CmdArgs.push_back("-lpthread");
    }

    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lc_p");
      else
        CmdArgs.push_back("-lc");
    }

    CmdArgs.push_back("-lgcc");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void bitrig::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void bitrig::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_shared)) {
    CmdArgs.push_back("-e");
    CmdArgs.push_back("__start");
  }

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    CmdArgs.push_back("--eh-frame-hdr");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-shared");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/usr/libexec/ld.so");
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("gcrt0.o")));
      else
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("crt0.o")));
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    } else {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbeginS.o")));
    }
  }

  Args.AddAllArgs(CmdArgs,
                  {options::OPT_L, options::OPT_T_Group, options::OPT_e});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX()) {
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lm_p");
      else
        CmdArgs.push_back("-lm");
    }

    if (Args.hasArg(options::OPT_pthread)) {
      if (!Args.hasArg(options::OPT_shared) && Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lpthread_p");
      else
        CmdArgs.push_back("-lpthread");
    }

    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lc_p");
      else
        CmdArgs.push_back("-lc");
    }

    StringRef MyArch;
    switch (getToolChain().getArch()) {
    case llvm::Triple::arm:
      MyArch = "arm";
      break;
    case llvm::Triple::x86:
      MyArch = "i386";
      break;
    case llvm::Triple::x86_64:
      MyArch = "amd64";
      break;
    default:
      llvm_unreachable("Unsupported architecture");
    }
    CmdArgs.push_back(Args.MakeArgString("-lclang_rt." + MyArch));
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void freebsd::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  // When building 32-bit code on FreeBSD/amd64, we have to explicitly
  // instruct as in the base system to assemble 32-bit code.
  switch (getToolChain().getArch()) {
  default:
    break;
  case llvm::Triple::x86:
    CmdArgs.push_back("--32");
    break;
  case llvm::Triple::ppc:
    CmdArgs.push_back("-a32");
    break;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, getToolChain().getTriple(), CPUName, ABIName);

    CmdArgs.push_back("-march");
    CmdArgs.push_back(CPUName.data());

    CmdArgs.push_back("-mabi");
    CmdArgs.push_back(getGnuCompatibleMipsABIName(ABIName).data());

    if (getToolChain().getArch() == llvm::Triple::mips ||
        getToolChain().getArch() == llvm::Triple::mips64)
      CmdArgs.push_back("-EB");
    else
      CmdArgs.push_back("-EL");

    if (Arg *A = Args.getLastArg(options::OPT_G)) {
      StringRef v = A->getValue();
      CmdArgs.push_back(Args.MakeArgString("-G" + v));
      A->claim();
    }

    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb: {
    arm::FloatABI ABI = arm::getARMFloatABI(getToolChain(), Args);

    if (ABI == arm::FloatABI::Hard)
      CmdArgs.push_back("-mfpu=vfp");
    else
      CmdArgs.push_back("-mfpu=softvfp");

    switch (getToolChain().getTriple().getEnvironment()) {
    case llvm::Triple::GNUEABIHF:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::EABI:
      CmdArgs.push_back("-meabi=5");
      break;

    default:
      CmdArgs.push_back("-matpcs");
    }
    break;
  }
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::sparcv9: {
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void freebsd::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  const toolchains::FreeBSD &ToolChain =
      static_cast<const toolchains::FreeBSD &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  const llvm::Triple::ArchType Arch = ToolChain.getArch();
  const bool IsPIE =
      !Args.hasArg(options::OPT_shared) &&
      (Args.hasArg(options::OPT_pie) || ToolChain.isPIEDefault());
  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (IsPIE)
    CmdArgs.push_back("-pie");

  CmdArgs.push_back("--eh-frame-hdr");
  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-Bshareable");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/libexec/ld-elf.so.1");
    }
    if (ToolChain.getTriple().getOSMajorVersion() >= 9) {
      if (Arch == llvm::Triple::arm || Arch == llvm::Triple::sparc ||
          Arch == llvm::Triple::x86 || Arch == llvm::Triple::x86_64) {
        CmdArgs.push_back("--hash-style=both");
      }
    }
    CmdArgs.push_back("--enable-new-dtags");
  }

  // When building 32-bit code on FreeBSD/amd64, we have to explicitly
  // instruct ld in the base system to link 32-bit code.
  if (Arch == llvm::Triple::x86) {
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf_i386_fbsd");
  }

  if (Arch == llvm::Triple::ppc) {
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf32ppc_fbsd");
  }

  if (Arg *A = Args.getLastArg(options::OPT_G)) {
    if (ToolChain.getArch() == llvm::Triple::mips ||
      ToolChain.getArch() == llvm::Triple::mipsel ||
      ToolChain.getArch() == llvm::Triple::mips64 ||
      ToolChain.getArch() == llvm::Triple::mips64el) {
      StringRef v = A->getValue();
      CmdArgs.push_back(Args.MakeArgString("-G" + v));
      A->claim();
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    const char *crt1 = nullptr;
    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        crt1 = "gcrt1.o";
      else if (IsPIE)
        crt1 = "Scrt1.o";
      else
        crt1 = "crt1.o";
    }
    if (crt1)
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crt1)));

    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crti.o")));

    const char *crtbegin = nullptr;
    if (Args.hasArg(options::OPT_static))
      crtbegin = "crtbeginT.o";
    else if (Args.hasArg(options::OPT_shared) || IsPIE)
      crtbegin = "crtbeginS.o";
    else
      crtbegin = "crtbegin.o";

    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_Z_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  if (D.isUsingLTO())
    AddGoldPlugin(ToolChain, Args, CmdArgs, D.getLTOMode() == LTOK_Thin, D);

  bool NeedsSanitizerDeps = addSanitizerRuntimes(ToolChain, Args, CmdArgs);
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    addOpenMPRuntime(CmdArgs, ToolChain, Args);
    if (D.CCCIsCXX()) {
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lm_p");
      else
        CmdArgs.push_back("-lm");
    }
    if (NeedsSanitizerDeps)
      linkSanitizerRuntimeDeps(ToolChain, CmdArgs);
    // FIXME: For some reason GCC passes -lgcc and -lgcc_s before adding
    // the default system libraries. Just mimic this for now.
    if (Args.hasArg(options::OPT_pg))
      CmdArgs.push_back("-lgcc_p");
    else
      CmdArgs.push_back("-lgcc");
    if (Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-lgcc_eh");
    } else if (Args.hasArg(options::OPT_pg)) {
      CmdArgs.push_back("-lgcc_eh_p");
    } else {
      CmdArgs.push_back("--as-needed");
      CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("--no-as-needed");
    }

    if (Args.hasArg(options::OPT_pthread)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lpthread_p");
      else
        CmdArgs.push_back("-lpthread");
    }

    if (Args.hasArg(options::OPT_pg)) {
      if (Args.hasArg(options::OPT_shared))
        CmdArgs.push_back("-lc");
      else
        CmdArgs.push_back("-lc_p");
      CmdArgs.push_back("-lgcc_p");
    } else {
      CmdArgs.push_back("-lc");
      CmdArgs.push_back("-lgcc");
    }

    if (Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-lgcc_eh");
    } else if (Args.hasArg(options::OPT_pg)) {
      CmdArgs.push_back("-lgcc_eh_p");
    } else {
      CmdArgs.push_back("--as-needed");
      CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("--no-as-needed");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || IsPIE)
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtendS.o")));
    else
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtend.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));
  }

  ToolChain.addProfileRTLibs(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void netbsd::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  // GNU as needs different flags for creating the correct output format
  // on architectures with different ABIs or optional feature sets.
  switch (getToolChain().getArch()) {
  case llvm::Triple::x86:
    CmdArgs.push_back("--32");
    break;
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb: {
    StringRef MArch, MCPU;
    getARMArchCPUFromArgs(Args, MArch, MCPU, /*FromAs*/ true);
    std::string Arch =
        arm::getARMTargetCPU(MCPU, MArch, getToolChain().getTriple());
    CmdArgs.push_back(Args.MakeArgString("-mcpu=" + Arch));
    break;
  }

  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, getToolChain().getTriple(), CPUName, ABIName);

    CmdArgs.push_back("-march");
    CmdArgs.push_back(CPUName.data());

    CmdArgs.push_back("-mabi");
    CmdArgs.push_back(getGnuCompatibleMipsABIName(ABIName).data());

    if (getToolChain().getArch() == llvm::Triple::mips ||
        getToolChain().getArch() == llvm::Triple::mips64)
      CmdArgs.push_back("-EB");
    else
      CmdArgs.push_back("-EL");

    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  case llvm::Triple::sparc:
  case llvm::Triple::sparcel: {
    CmdArgs.push_back("-32");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  case llvm::Triple::sparcv9: {
    CmdArgs.push_back("-64");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }

  default:
    break;
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString((getToolChain().GetProgramPath("as")));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void netbsd::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  CmdArgs.push_back("--eh-frame-hdr");
  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-Bshareable");
    } else {
      Args.AddAllArgs(CmdArgs, options::OPT_pie);
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/libexec/ld.elf_so");
    }
  }

  // Many NetBSD architectures support more than one ABI.
  // Determine the correct emulation for ld.
  switch (getToolChain().getArch()) {
  case llvm::Triple::x86:
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf_i386");
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    CmdArgs.push_back("-m");
    switch (getToolChain().getTriple().getEnvironment()) {
    case llvm::Triple::EABI:
    case llvm::Triple::GNUEABI:
      CmdArgs.push_back("armelf_nbsd_eabi");
      break;
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABIHF:
      CmdArgs.push_back("armelf_nbsd_eabihf");
      break;
    default:
      CmdArgs.push_back("armelf_nbsd");
      break;
    }
    break;
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb:
    arm::appendEBLinkFlags(Args, CmdArgs, getToolChain().getEffectiveTriple());
    CmdArgs.push_back("-m");
    switch (getToolChain().getTriple().getEnvironment()) {
    case llvm::Triple::EABI:
    case llvm::Triple::GNUEABI:
      CmdArgs.push_back("armelfb_nbsd_eabi");
      break;
    case llvm::Triple::EABIHF:
    case llvm::Triple::GNUEABIHF:
      CmdArgs.push_back("armelfb_nbsd_eabihf");
      break;
    default:
      CmdArgs.push_back("armelfb_nbsd");
      break;
    }
    break;
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
    if (mips::hasMipsAbiArg(Args, "32")) {
      CmdArgs.push_back("-m");
      if (getToolChain().getArch() == llvm::Triple::mips64)
        CmdArgs.push_back("elf32btsmip");
      else
        CmdArgs.push_back("elf32ltsmip");
    } else if (mips::hasMipsAbiArg(Args, "64")) {
      CmdArgs.push_back("-m");
      if (getToolChain().getArch() == llvm::Triple::mips64)
        CmdArgs.push_back("elf64btsmip");
      else
        CmdArgs.push_back("elf64ltsmip");
    }
    break;
  case llvm::Triple::ppc:
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf32ppc_nbsd");
    break;

  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf64ppc");
    break;

  case llvm::Triple::sparc:
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf32_sparc");
    break;

  case llvm::Triple::sparcv9:
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf64_sparc");
    break;

  default:
    break;
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crt0.o")));
    }
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_pie)) {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbeginS.o")));
    } else {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_Z_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  unsigned Major, Minor, Micro;
  getToolChain().getTriple().getOSVersion(Major, Minor, Micro);
  bool useLibgcc = true;
  if (Major >= 7 || Major == 0) {
    switch (getToolChain().getArch()) {
    case llvm::Triple::aarch64:
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
    case llvm::Triple::ppc64le:
    case llvm::Triple::sparc:
    case llvm::Triple::sparcv9:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      useLibgcc = false;
      break;
    default:
      break;
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    addOpenMPRuntime(CmdArgs, getToolChain(), Args);
    if (D.CCCIsCXX()) {
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }
    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");
    CmdArgs.push_back("-lc");

    if (useLibgcc) {
      if (Args.hasArg(options::OPT_static)) {
        // libgcc_eh depends on libc, so resolve as much as possible,
        // pull in any new requirements from libc and then get the rest
        // of libgcc.
        CmdArgs.push_back("-lgcc_eh");
        CmdArgs.push_back("-lc");
        CmdArgs.push_back("-lgcc");
      } else {
        CmdArgs.push_back("-lgcc");
        CmdArgs.push_back("--as-needed");
        CmdArgs.push_back("-lgcc_s");
        CmdArgs.push_back("--no-as-needed");
      }
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_pie))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));
  }

  getToolChain().addProfileRTLibs(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void gnutools::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const char *LinkingOutput) const {
  claimNoWarnArgs(Args);

  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();

  ArgStringList CmdArgs;

  llvm::Reloc::Model RelocationModel;
  unsigned PICLevel;
  bool IsPIE;
  std::tie(RelocationModel, PICLevel, IsPIE) =
      ParsePICArgs(getToolChain(), Triple, Args);

  switch (getToolChain().getArch()) {
  default:
    break;
  // Add --32/--64 to make sure we get the format we want.
  // This is incomplete
  case llvm::Triple::x86:
    CmdArgs.push_back("--32");
    break;
  case llvm::Triple::x86_64:
    if (getToolChain().getTriple().getEnvironment() == llvm::Triple::GNUX32)
      CmdArgs.push_back("--x32");
    else
      CmdArgs.push_back("--64");
    break;
  case llvm::Triple::ppc:
    CmdArgs.push_back("-a32");
    CmdArgs.push_back("-mppc");
    CmdArgs.push_back("-many");
    break;
  case llvm::Triple::ppc64:
    CmdArgs.push_back("-a64");
    CmdArgs.push_back("-mppc64");
    CmdArgs.push_back("-many");
    break;
  case llvm::Triple::ppc64le:
    CmdArgs.push_back("-a64");
    CmdArgs.push_back("-mppc64");
    CmdArgs.push_back("-many");
    CmdArgs.push_back("-mlittle-endian");
    break;
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel: {
    CmdArgs.push_back("-32");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }
  case llvm::Triple::sparcv9: {
    CmdArgs.push_back("-64");
    std::string CPU = getCPUName(Args, getToolChain().getTriple());
    CmdArgs.push_back(getSparcAsmModeForCPU(CPU, getToolChain().getTriple()));
    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb: {
    const llvm::Triple &Triple2 = getToolChain().getTriple();
    switch (Triple2.getSubArch()) {
    case llvm::Triple::ARMSubArch_v7:
      CmdArgs.push_back("-mfpu=neon");
      break;
    case llvm::Triple::ARMSubArch_v8:
      CmdArgs.push_back("-mfpu=crypto-neon-fp-armv8");
      break;
    default:
      break;
    }

    switch (arm::getARMFloatABI(getToolChain(), Args)) {
    case arm::FloatABI::Invalid: llvm_unreachable("must have an ABI!");
    case arm::FloatABI::Soft:
      CmdArgs.push_back(Args.MakeArgString("-mfloat-abi=soft"));
      break;
    case arm::FloatABI::SoftFP:
      CmdArgs.push_back(Args.MakeArgString("-mfloat-abi=softfp"));
      break;
    case arm::FloatABI::Hard:
      CmdArgs.push_back(Args.MakeArgString("-mfloat-abi=hard"));
      break;
    }

    Args.AddLastArg(CmdArgs, options::OPT_march_EQ);

    // FIXME: remove krait check when GNU tools support krait cpu
    // for now replace it with -mcpu=cortex-a15 to avoid a lower
    // march from being picked in the absence of a cpu flag.
    Arg *A;
    if ((A = Args.getLastArg(options::OPT_mcpu_EQ)) &&
        StringRef(A->getValue()).equals_lower("krait"))
      CmdArgs.push_back("-mcpu=cortex-a15");
    else
      Args.AddLastArg(CmdArgs, options::OPT_mcpu_EQ);
    Args.AddLastArg(CmdArgs, options::OPT_mfpu_EQ);
    break;
  }
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, getToolChain().getTriple(), CPUName, ABIName);
    ABIName = getGnuCompatibleMipsABIName(ABIName);

    CmdArgs.push_back("-march");
    CmdArgs.push_back(CPUName.data());

    CmdArgs.push_back("-mabi");
    CmdArgs.push_back(ABIName.data());

    // -mno-shared should be emitted unless -fpic, -fpie, -fPIC, -fPIE,
    // or -mshared (not implemented) is in effect.
    if (RelocationModel == llvm::Reloc::Static)
      CmdArgs.push_back("-mno-shared");

    // LLVM doesn't support -mplt yet and acts as if it is always given.
    // However, -mplt has no effect with the N64 ABI.
    CmdArgs.push_back(ABIName == "64" ? "-KPIC" : "-call_nonpic");

    if (getToolChain().getArch() == llvm::Triple::mips ||
        getToolChain().getArch() == llvm::Triple::mips64)
      CmdArgs.push_back("-EB");
    else
      CmdArgs.push_back("-EL");

    if (Arg *A = Args.getLastArg(options::OPT_mnan_EQ)) {
      if (StringRef(A->getValue()) == "2008")
        CmdArgs.push_back(Args.MakeArgString("-mnan=2008"));
    }

    // Add the last -mfp32/-mfpxx/-mfp64 or -mfpxx if it is enabled by default.
    if (Arg *A = Args.getLastArg(options::OPT_mfp32, options::OPT_mfpxx,
                                 options::OPT_mfp64)) {
      A->claim();
      A->render(Args, CmdArgs);
    } else if (mips::shouldUseFPXX(
                   Args, getToolChain().getTriple(), CPUName, ABIName,
                   getMipsFloatABI(getToolChain().getDriver(), Args)))
      CmdArgs.push_back("-mfpxx");

    // Pass on -mmips16 or -mno-mips16. However, the assembler equivalent of
    // -mno-mips16 is actually -no-mips16.
    if (Arg *A =
            Args.getLastArg(options::OPT_mips16, options::OPT_mno_mips16)) {
      if (A->getOption().matches(options::OPT_mips16)) {
        A->claim();
        A->render(Args, CmdArgs);
      } else {
        A->claim();
        CmdArgs.push_back("-no-mips16");
      }
    }

    Args.AddLastArg(CmdArgs, options::OPT_mmicromips,
                    options::OPT_mno_micromips);
    Args.AddLastArg(CmdArgs, options::OPT_mdsp, options::OPT_mno_dsp);
    Args.AddLastArg(CmdArgs, options::OPT_mdspr2, options::OPT_mno_dspr2);

    if (Arg *A = Args.getLastArg(options::OPT_mmsa, options::OPT_mno_msa)) {
      // Do not use AddLastArg because not all versions of MIPS assembler
      // support -mmsa / -mno-msa options.
      if (A->getOption().matches(options::OPT_mmsa))
        CmdArgs.push_back(Args.MakeArgString("-mmsa"));
    }

    Args.AddLastArg(CmdArgs, options::OPT_mhard_float,
                    options::OPT_msoft_float);

    Args.AddLastArg(CmdArgs, options::OPT_mdouble_float,
                    options::OPT_msingle_float);

    Args.AddLastArg(CmdArgs, options::OPT_modd_spreg,
                    options::OPT_mno_odd_spreg);

    AddAssemblerKPIC(getToolChain(), Args, CmdArgs);
    break;
  }
  case llvm::Triple::systemz: {
    // Always pass an -march option, since our default of z10 is later
    // than the GNU assembler's default.
    StringRef CPUName = getSystemZTargetCPU(Args);
    CmdArgs.push_back(Args.MakeArgString("-march=" + CPUName));
    break;
  }
  }

  Args.AddAllArgs(CmdArgs, options::OPT_I);
  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));

  // Handle the debug info splitting at object creation time if we're
  // creating an object.
  // TODO: Currently only works on linux with newer objcopy.
  if (Args.hasArg(options::OPT_gsplit_dwarf) &&
      getToolChain().getTriple().isOSLinux())
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output,
                   SplitDebugName(Args, Inputs[0]));
}

static void AddLibgcc(const llvm::Triple &Triple, const Driver &D,
                      ArgStringList &CmdArgs, const ArgList &Args) {
  bool isAndroid = Triple.isAndroid();
  bool isCygMing = Triple.isOSCygMing();
  bool IsIAMCU = Triple.isOSIAMCU();
  bool StaticLibgcc = Args.hasArg(options::OPT_static_libgcc) ||
                      Args.hasArg(options::OPT_static);
  if (!D.CCCIsCXX())
    CmdArgs.push_back("-lgcc");

  if (StaticLibgcc || isAndroid) {
    if (D.CCCIsCXX())
      CmdArgs.push_back("-lgcc");
  } else {
    if (!D.CCCIsCXX() && !isCygMing)
      CmdArgs.push_back("--as-needed");
    CmdArgs.push_back("-lgcc_s");
    if (!D.CCCIsCXX() && !isCygMing)
      CmdArgs.push_back("--no-as-needed");
  }

  if (StaticLibgcc && !isAndroid && !IsIAMCU)
    CmdArgs.push_back("-lgcc_eh");
  else if (!Args.hasArg(options::OPT_shared) && D.CCCIsCXX())
    CmdArgs.push_back("-lgcc");

  // According to Android ABI, we have to link with libdl if we are
  // linking with non-static libgcc.
  //
  // NOTE: This fixes a link error on Android MIPS as well.  The non-static
  // libgcc for MIPS relies on _Unwind_Find_FDE and dl_iterate_phdr from libdl.
  if (isAndroid && !StaticLibgcc)
    CmdArgs.push_back("-ldl");
}

static void AddRunTimeLibs(const ToolChain &TC, const Driver &D,
                           ArgStringList &CmdArgs, const ArgList &Args) {
  // Make use of compiler-rt if --rtlib option is used
  ToolChain::RuntimeLibType RLT = TC.GetRuntimeLibType(Args);

  switch (RLT) {
  case ToolChain::RLT_CompilerRT:
    switch (TC.getTriple().getOS()) {
    default:
      llvm_unreachable("unsupported OS");
    case llvm::Triple::Win32:
    case llvm::Triple::Linux:
      addClangRT(TC, Args, CmdArgs);
      break;
    }
    break;
  case ToolChain::RLT_Libgcc:
    // Make sure libgcc is not used under MSVC environment by default
    if (TC.getTriple().isKnownWindowsMSVCEnvironment()) {
      // Issue error diagnostic if libgcc is explicitly specified
      // through command line as --rtlib option argument.
      if (Args.hasArg(options::OPT_rtlib_EQ)) {
        TC.getDriver().Diag(diag::err_drv_unsupported_rtlib_for_platform)
            << Args.getLastArg(options::OPT_rtlib_EQ)->getValue() << "MSVC";
      }
    } else
      AddLibgcc(TC.getTriple(), D, CmdArgs, Args);
    break;
  }
}

static const char *getLDMOption(const llvm::Triple &T, const ArgList &Args) {
  switch (T.getArch()) {
  case llvm::Triple::x86:
    if (T.isOSIAMCU())
      return "elf_iamcu";
    return "elf_i386";
  case llvm::Triple::aarch64:
    return "aarch64linux";
  case llvm::Triple::aarch64_be:
    return "aarch64_be_linux";
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    return "armelf_linux_eabi";
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb:
    return "armelfb_linux_eabi";
  case llvm::Triple::ppc:
    return "elf32ppclinux";
  case llvm::Triple::ppc64:
    return "elf64ppc";
  case llvm::Triple::ppc64le:
    return "elf64lppc";
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
    return "elf32_sparc";
  case llvm::Triple::sparcv9:
    return "elf64_sparc";
  case llvm::Triple::mips:
    return "elf32btsmip";
  case llvm::Triple::mipsel:
    return "elf32ltsmip";
  case llvm::Triple::mips64:
    if (mips::hasMipsAbiArg(Args, "n32"))
      return "elf32btsmipn32";
    return "elf64btsmip";
  case llvm::Triple::mips64el:
    if (mips::hasMipsAbiArg(Args, "n32"))
      return "elf32ltsmipn32";
    return "elf64ltsmip";
  case llvm::Triple::systemz:
    return "elf64_s390";
  case llvm::Triple::x86_64:
    if (T.getEnvironment() == llvm::Triple::GNUX32)
      return "elf32_x86_64";
    return "elf_x86_64";
  default:
    llvm_unreachable("Unexpected arch");
  }
}

void gnutools::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  const toolchains::Linux &ToolChain =
      static_cast<const toolchains::Linux &>(getToolChain());
  const Driver &D = ToolChain.getDriver();

  const llvm::Triple &Triple = getToolChain().getEffectiveTriple();

  const llvm::Triple::ArchType Arch = ToolChain.getArch();
  const bool isAndroid = ToolChain.getTriple().isAndroid();
  const bool IsIAMCU = ToolChain.getTriple().isOSIAMCU();
  const bool IsPIE =
      !Args.hasArg(options::OPT_shared) && !Args.hasArg(options::OPT_static) &&
      (Args.hasArg(options::OPT_pie) || ToolChain.isPIEDefault());
  const bool HasCRTBeginEndFiles =
      ToolChain.getTriple().hasEnvironment() ||
      (ToolChain.getTriple().getVendor() != llvm::Triple::MipsTechnologies);

  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  if (llvm::sys::path::filename(Exec) == "lld") {
    CmdArgs.push_back("-flavor");
    CmdArgs.push_back("old-gnu");
    CmdArgs.push_back("-target");
    CmdArgs.push_back(Args.MakeArgString(getToolChain().getTripleString()));
  }

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (IsPIE)
    CmdArgs.push_back("-pie");

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  if (Arch == llvm::Triple::armeb || Arch == llvm::Triple::thumbeb)
    arm::appendEBLinkFlags(Args, CmdArgs, Triple);

  for (const auto &Opt : ToolChain.ExtraOpts)
    CmdArgs.push_back(Opt.c_str());

  if (!Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("--eh-frame-hdr");
  }

  CmdArgs.push_back("-m");
  CmdArgs.push_back(getLDMOption(ToolChain.getTriple(), Args));

  if (Args.hasArg(options::OPT_static)) {
    if (Arch == llvm::Triple::arm || Arch == llvm::Triple::armeb ||
        Arch == llvm::Triple::thumb || Arch == llvm::Triple::thumbeb)
      CmdArgs.push_back("-Bstatic");
    else
      CmdArgs.push_back("-static");
  } else if (Args.hasArg(options::OPT_shared)) {
    CmdArgs.push_back("-shared");
  }

  if (!Args.hasArg(options::OPT_static)) {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");

    if (!Args.hasArg(options::OPT_shared)) {
      const std::string Loader =
          D.DyldPrefix + ToolChain.getDynamicLinker(Args);
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back(Args.MakeArgString(Loader));
    }
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!isAndroid && !IsIAMCU) {
      const char *crt1 = nullptr;
      if (!Args.hasArg(options::OPT_shared)) {
        if (Args.hasArg(options::OPT_pg))
          crt1 = "gcrt1.o";
        else if (IsPIE)
          crt1 = "Scrt1.o";
        else
          crt1 = "crt1.o";
      }
      if (crt1)
        CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crt1)));

      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crti.o")));
    }

    if (IsIAMCU)
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    else {
      const char *crtbegin;
      if (Args.hasArg(options::OPT_static))
        crtbegin = isAndroid ? "crtbegin_static.o" : "crtbeginT.o";
      else if (Args.hasArg(options::OPT_shared))
        crtbegin = isAndroid ? "crtbegin_so.o" : "crtbeginS.o";
      else if (IsPIE)
        crtbegin = isAndroid ? "crtbegin_dynamic.o" : "crtbeginS.o";
      else
        crtbegin = isAndroid ? "crtbegin_dynamic.o" : "crtbegin.o";

      if (HasCRTBeginEndFiles)
        CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
    }

    // Add crtfastmath.o if available and fast math is enabled.
    ToolChain.AddFastMathRuntimeIfAvailable(Args, CmdArgs);
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (D.isUsingLTO())
    AddGoldPlugin(ToolChain, Args, CmdArgs, D.getLTOMode() == LTOK_Thin, D);

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  bool NeedsSanitizerDeps = addSanitizerRuntimes(ToolChain, Args, CmdArgs);
  bool NeedsXRayDeps = addXRayRuntime(ToolChain, Args, CmdArgs);
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);
  // The profile runtime also needs access to system libraries.
  getToolChain().addProfileRTLibs(Args, CmdArgs);

  if (D.CCCIsCXX() &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                               !Args.hasArg(options::OPT_static);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
    CmdArgs.push_back("-lm");
  }
  // Silence warnings when linking C code with a C++ '-stdlib' argument.
  Args.ClaimAllArgs(options::OPT_stdlib_EQ);

  if (!Args.hasArg(options::OPT_nostdlib)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--start-group");

      if (NeedsSanitizerDeps)
        linkSanitizerRuntimeDeps(ToolChain, CmdArgs);

      if (NeedsXRayDeps)
        linkXRayRuntimeDeps(ToolChain, Args, CmdArgs);

      bool WantPthread = Args.hasArg(options::OPT_pthread) ||
                         Args.hasArg(options::OPT_pthreads);

      if (Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                       options::OPT_fno_openmp, false)) {
        // OpenMP runtimes implies pthreads when using the GNU toolchain.
        // FIXME: Does this really make sense for all GNU toolchains?
        WantPthread = true;

        // Also link the particular OpenMP runtimes.
        switch (getOpenMPRuntime(ToolChain, Args)) {
        case OMPRT_OMP:
          CmdArgs.push_back("-lomp");
          break;
        case OMPRT_GOMP:
          CmdArgs.push_back("-lgomp");

          // FIXME: Exclude this for platforms with libgomp that don't require
          // librt. Most modern Linux platforms require it, but some may not.
          CmdArgs.push_back("-lrt");
          break;
        case OMPRT_IOMP5:
          CmdArgs.push_back("-liomp5");
          break;
        case OMPRT_Unknown:
          // Already diagnosed.
          break;
        }
      }

      AddRunTimeLibs(ToolChain, D, CmdArgs, Args);

      if (WantPthread && !isAndroid)
        CmdArgs.push_back("-lpthread");

      if (Args.hasArg(options::OPT_fsplit_stack))
        CmdArgs.push_back("--wrap=pthread_create");

      CmdArgs.push_back("-lc");

      // Add IAMCU specific libs, if needed.
      if (IsIAMCU)
        CmdArgs.push_back("-lgloss");

      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--end-group");
      else
        AddRunTimeLibs(ToolChain, D, CmdArgs, Args);

      // Add IAMCU specific libs (outside the group), if needed.
      if (IsIAMCU) {
        CmdArgs.push_back("--as-needed");
        CmdArgs.push_back("-lsoftfp");
        CmdArgs.push_back("--no-as-needed");
      }
    }

    if (!Args.hasArg(options::OPT_nostartfiles) && !IsIAMCU) {
      const char *crtend;
      if (Args.hasArg(options::OPT_shared))
        crtend = isAndroid ? "crtend_so.o" : "crtendS.o";
      else if (IsPIE)
        crtend = isAndroid ? "crtend_android.o" : "crtendS.o";
      else
        crtend = isAndroid ? "crtend_android.o" : "crtend.o";

      if (HasCRTBeginEndFiles)
        CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtend)));
      if (!isAndroid)
        CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));
    }
  }

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

// NaCl ARM assembly (inline or standalone) can be written with a set of macros
// for the various SFI requirements like register masking. The assembly tool
// inserts the file containing the macros as an input into all the assembly
// jobs.
void nacltools::AssemblerARM::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  const toolchains::NaClToolChain &ToolChain =
      static_cast<const toolchains::NaClToolChain &>(getToolChain());
  InputInfo NaClMacros(types::TY_PP_Asm, ToolChain.GetNaClArmMacrosPath(),
                       "nacl-arm-macros.s");
  InputInfoList NewInputs;
  NewInputs.push_back(NaClMacros);
  NewInputs.append(Inputs.begin(), Inputs.end());
  gnutools::Assembler::ConstructJob(C, JA, Output, NewInputs, Args,
                                    LinkingOutput);
}

// This is quite similar to gnutools::Linker::ConstructJob with changes that
// we use static by default, do not yet support sanitizers or LTO, and a few
// others. Eventually we can support more of that and hopefully migrate back
// to gnutools::Linker.
void nacltools::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {

  const toolchains::NaClToolChain &ToolChain =
      static_cast<const toolchains::NaClToolChain &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  const llvm::Triple::ArchType Arch = ToolChain.getArch();
  const bool IsStatic =
      !Args.hasArg(options::OPT_dynamic) && !Args.hasArg(options::OPT_shared);

  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  // NaClToolChain doesn't have ExtraOpts like Linux; the only relevant flag
  // from there is --build-id, which we do want.
  CmdArgs.push_back("--build-id");

  if (!IsStatic)
    CmdArgs.push_back("--eh-frame-hdr");

  CmdArgs.push_back("-m");
  if (Arch == llvm::Triple::x86)
    CmdArgs.push_back("elf_i386_nacl");
  else if (Arch == llvm::Triple::arm)
    CmdArgs.push_back("armelf_nacl");
  else if (Arch == llvm::Triple::x86_64)
    CmdArgs.push_back("elf_x86_64_nacl");
  else if (Arch == llvm::Triple::mipsel)
    CmdArgs.push_back("mipselelf_nacl");
  else
    D.Diag(diag::err_target_unsupported_arch) << ToolChain.getArchName()
                                              << "Native Client";

  if (IsStatic)
    CmdArgs.push_back("-static");
  else if (Args.hasArg(options::OPT_shared))
    CmdArgs.push_back("-shared");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt1.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crti.o")));

    const char *crtbegin;
    if (IsStatic)
      crtbegin = "crtbeginT.o";
    else if (Args.hasArg(options::OPT_shared))
      crtbegin = "crtbeginS.o";
    else
      crtbegin = "crtbegin.o";
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);

  ToolChain.AddFilePathLibArgs(Args, CmdArgs);

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (D.CCCIsCXX() &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    bool OnlyLibstdcxxStatic =
        Args.hasArg(options::OPT_static_libstdcxx) && !IsStatic;
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
    CmdArgs.push_back("-lm");
  }

  if (!Args.hasArg(options::OPT_nostdlib)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      // Always use groups, since it has no effect on dynamic libraries.
      CmdArgs.push_back("--start-group");
      CmdArgs.push_back("-lc");
      // NaCl's libc++ currently requires libpthread, so just always include it
      // in the group for C++.
      if (Args.hasArg(options::OPT_pthread) ||
          Args.hasArg(options::OPT_pthreads) || D.CCCIsCXX()) {
        // Gold, used by Mips, handles nested groups differently than ld, and
        // without '-lnacl' it prefers symbols from libpthread.a over libnacl.a,
        // which is not a desired behaviour here.
        // See https://sourceware.org/ml/binutils/2015-03/msg00034.html
        if (getToolChain().getArch() == llvm::Triple::mipsel)
          CmdArgs.push_back("-lnacl");

        CmdArgs.push_back("-lpthread");
      }

      CmdArgs.push_back("-lgcc");
      CmdArgs.push_back("--as-needed");
      if (IsStatic)
        CmdArgs.push_back("-lgcc_eh");
      else
        CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("--no-as-needed");

      // Mips needs to create and use pnacl_legacy library that contains
      // definitions from bitcode/pnaclmm.c and definitions for
      // __nacl_tp_tls_offset() and __nacl_tp_tdb_offset().
      if (getToolChain().getArch() == llvm::Triple::mipsel)
        CmdArgs.push_back("-lpnacl_legacy");

      CmdArgs.push_back("--end-group");
    }

    if (!Args.hasArg(options::OPT_nostartfiles)) {
      const char *crtend;
      if (Args.hasArg(options::OPT_shared))
        crtend = "crtendS.o";
      else
        crtend = "crtend.o";

      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtend)));
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));
    }
  }

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void minix::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void minix::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crt1.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));
  }

  Args.AddAllArgs(CmdArgs,
                  {options::OPT_L, options::OPT_T_Group, options::OPT_e});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  getToolChain().addProfileRTLibs(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX()) {
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lCompilerRT-Generic");
    CmdArgs.push_back("-L/usr/pkg/compiler-rt/lib");
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

/// DragonFly Tools

// For now, DragonFly Assemble does just about the same as for
// FreeBSD, but this may change soon.
void dragonfly::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                        const InputInfo &Output,
                                        const InputInfoList &Inputs,
                                        const ArgList &Args,
                                        const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  // When building 32-bit code on DragonFly/pc64, we have to explicitly
  // instruct as in the base system to assemble 32-bit code.
  if (getToolChain().getArch() == llvm::Triple::x86)
    CmdArgs.push_back("--32");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void dragonfly::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  CmdArgs.push_back("--eh-frame-hdr");
  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    if (Args.hasArg(options::OPT_shared))
      CmdArgs.push_back("-Bshareable");
    else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/usr/libexec/ld-elf.so.2");
    }
    CmdArgs.push_back("--hash-style=gnu");
    CmdArgs.push_back("--enable-new-dtags");
  }

  // When building 32-bit code on DragonFly/pc64, we have to explicitly
  // instruct ld in the base system to link 32-bit code.
  if (getToolChain().getArch() == llvm::Triple::x86) {
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf_i386");
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back(
            Args.MakeArgString(getToolChain().GetFilePath("gcrt1.o")));
      else {
        if (Args.hasArg(options::OPT_pie))
          CmdArgs.push_back(
              Args.MakeArgString(getToolChain().GetFilePath("Scrt1.o")));
        else
          CmdArgs.push_back(
              Args.MakeArgString(getToolChain().GetFilePath("crt1.o")));
      }
    }
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_pie))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbeginS.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
  }

  Args.AddAllArgs(CmdArgs,
                  {options::OPT_L, options::OPT_T_Group, options::OPT_e});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    CmdArgs.push_back("-L/usr/lib/gcc50");

    if (!Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-rpath");
      CmdArgs.push_back("/usr/lib/gcc50");
    }

    if (D.CCCIsCXX()) {
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }

    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");

    if (!Args.hasArg(options::OPT_nolibc)) {
      CmdArgs.push_back("-lc");
    }

    if (Args.hasArg(options::OPT_static) ||
        Args.hasArg(options::OPT_static_libgcc)) {
        CmdArgs.push_back("-lgcc");
        CmdArgs.push_back("-lgcc_eh");
    } else {
      if (Args.hasArg(options::OPT_shared_libgcc)) {
          CmdArgs.push_back("-lgcc_pic");
          if (!Args.hasArg(options::OPT_shared))
            CmdArgs.push_back("-lgcc");
      } else {
          CmdArgs.push_back("-lgcc");
          CmdArgs.push_back("--as-needed");
          CmdArgs.push_back("-lgcc_pic");
          CmdArgs.push_back("--no-as-needed");
      }
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_pie))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));
  }

  getToolChain().addProfileRTLibs(Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

// Try to find Exe from a Visual Studio distribution.  This first tries to find
// an installed copy of Visual Studio and, failing that, looks in the PATH,
// making sure that whatever executable that's found is not a same-named exe
// from clang itself to prevent clang from falling back to itself.
static std::string FindVisualStudioExecutable(const ToolChain &TC,
                                              const char *Exe,
                                              const char *ClangProgramPath) {
  const auto &MSVC = static_cast<const toolchains::MSVCToolChain &>(TC);
  std::string visualStudioBinDir;
  if (MSVC.getVisualStudioBinariesFolder(ClangProgramPath,
                                         visualStudioBinDir)) {
    SmallString<128> FilePath(visualStudioBinDir);
    llvm::sys::path::append(FilePath, Exe);
    if (llvm::sys::fs::can_execute(FilePath.c_str()))
      return FilePath.str();
  }

  return Exe;
}

void visualstudio::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                        const InputInfo &Output,
                                        const InputInfoList &Inputs,
                                        const ArgList &Args,
                                        const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  const ToolChain &TC = getToolChain();

  assert((Output.isFilename() || Output.isNothing()) && "invalid output");
  if (Output.isFilename())
    CmdArgs.push_back(
        Args.MakeArgString(std::string("-out:") + Output.getFilename()));

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles) &&
      !C.getDriver().IsCLMode())
    CmdArgs.push_back("-defaultlib:libcmt");

  if (!llvm::sys::Process::GetEnv("LIB")) {
    // If the VC environment hasn't been configured (perhaps because the user
    // did not run vcvarsall), try to build a consistent link environment.  If
    // the environment variable is set however, assume the user knows what
    // they're doing.
    std::string VisualStudioDir;
    const auto &MSVC = static_cast<const toolchains::MSVCToolChain &>(TC);
    if (MSVC.getVisualStudioInstallDir(VisualStudioDir)) {
      SmallString<128> LibDir(VisualStudioDir);
      llvm::sys::path::append(LibDir, "VC", "lib");
      switch (MSVC.getArch()) {
      case llvm::Triple::x86:
        // x86 just puts the libraries directly in lib
        break;
      case llvm::Triple::x86_64:
        llvm::sys::path::append(LibDir, "amd64");
        break;
      case llvm::Triple::arm:
        llvm::sys::path::append(LibDir, "arm");
        break;
      default:
        break;
      }
      CmdArgs.push_back(
          Args.MakeArgString(std::string("-libpath:") + LibDir.c_str()));

      if (MSVC.useUniversalCRT(VisualStudioDir)) {
        std::string UniversalCRTLibPath;
        if (MSVC.getUniversalCRTLibraryPath(UniversalCRTLibPath))
          CmdArgs.push_back(Args.MakeArgString(std::string("-libpath:") +
                                               UniversalCRTLibPath.c_str()));
      }
    }

    std::string WindowsSdkLibPath;
    if (MSVC.getWindowsSDKLibraryPath(WindowsSdkLibPath))
      CmdArgs.push_back(Args.MakeArgString(std::string("-libpath:") +
                                           WindowsSdkLibPath.c_str()));
  }

  if (!C.getDriver().IsCLMode() && Args.hasArg(options::OPT_L))
    for (const auto &LibPath : Args.getAllArgValues(options::OPT_L))
      CmdArgs.push_back(Args.MakeArgString("-libpath:" + LibPath));

  CmdArgs.push_back("-nologo");

  if (Args.hasArg(options::OPT_g_Group, options::OPT__SLASH_Z7,
                  options::OPT__SLASH_Zd))
    CmdArgs.push_back("-debug");

  bool DLL = Args.hasArg(options::OPT__SLASH_LD, options::OPT__SLASH_LDd,
                         options::OPT_shared);
  if (DLL) {
    CmdArgs.push_back(Args.MakeArgString("-dll"));

    SmallString<128> ImplibName(Output.getFilename());
    llvm::sys::path::replace_extension(ImplibName, "lib");
    CmdArgs.push_back(Args.MakeArgString(std::string("-implib:") + ImplibName));
  }

  if (TC.getSanitizerArgs().needsAsanRt()) {
    CmdArgs.push_back(Args.MakeArgString("-debug"));
    CmdArgs.push_back(Args.MakeArgString("-incremental:no"));
    if (Args.hasArg(options::OPT__SLASH_MD, options::OPT__SLASH_MDd)) {
      for (const auto &Lib : {"asan_dynamic", "asan_dynamic_runtime_thunk"})
        CmdArgs.push_back(TC.getCompilerRTArgString(Args, Lib));
      // Make sure the dynamic runtime thunk is not optimized out at link time
      // to ensure proper SEH handling.
      CmdArgs.push_back(Args.MakeArgString(
          TC.getArch() == llvm::Triple::x86
              ? "-include:___asan_seh_interceptor"
              : "-include:__asan_seh_interceptor"));
    } else if (DLL) {
      CmdArgs.push_back(TC.getCompilerRTArgString(Args, "asan_dll_thunk"));
    } else {
      for (const auto &Lib : {"asan", "asan_cxx"})
        CmdArgs.push_back(TC.getCompilerRTArgString(Args, Lib));
    }
  }

  Args.AddAllArgValues(CmdArgs, options::OPT__SLASH_link);

  if (Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                   options::OPT_fno_openmp, false)) {
    CmdArgs.push_back("-nodefaultlib:vcomp.lib");
    CmdArgs.push_back("-nodefaultlib:vcompd.lib");
    CmdArgs.push_back(Args.MakeArgString(std::string("-libpath:") +
                                         TC.getDriver().Dir + "/../lib"));
    switch (getOpenMPRuntime(getToolChain(), Args)) {
    case OMPRT_OMP:
      CmdArgs.push_back("-defaultlib:libomp.lib");
      break;
    case OMPRT_IOMP5:
      CmdArgs.push_back("-defaultlib:libiomp5md.lib");
      break;
    case OMPRT_GOMP:
      break;
    case OMPRT_Unknown:
      // Already diagnosed.
      break;
    }
  }

  // Add compiler-rt lib in case if it was explicitly
  // specified as an argument for --rtlib option.
  if (!Args.hasArg(options::OPT_nostdlib)) {
    AddRunTimeLibs(TC, TC.getDriver(), CmdArgs, Args);
  }

  // Add filenames, libraries, and other linker inputs.
  for (const auto &Input : Inputs) {
    if (Input.isFilename()) {
      CmdArgs.push_back(Input.getFilename());
      continue;
    }

    const Arg &A = Input.getInputArg();

    // Render -l options differently for the MSVC linker.
    if (A.getOption().matches(options::OPT_l)) {
      StringRef Lib = A.getValue();
      const char *LinkLibArg;
      if (Lib.endswith(".lib"))
        LinkLibArg = Args.MakeArgString(Lib);
      else
        LinkLibArg = Args.MakeArgString(Lib + ".lib");
      CmdArgs.push_back(LinkLibArg);
      continue;
    }

    // Otherwise, this is some other kind of linker input option like -Wl, -z,
    // or -L. Render it, even if MSVC doesn't understand it.
    A.renderAsInput(Args, CmdArgs);
  }

  TC.addProfileRTLibs(Args, CmdArgs);

  // We need to special case some linker paths.  In the case of lld, we need to
  // translate 'lld' into 'lld-link', and in the case of the regular msvc
  // linker, we need to use a special search algorithm.
  llvm::SmallString<128> linkPath;
  StringRef Linker = Args.getLastArgValue(options::OPT_fuse_ld_EQ, "link");
  if (Linker.equals_lower("lld"))
    Linker = "lld-link";

  if (Linker.equals_lower("link")) {
    // If we're using the MSVC linker, it's not sufficient to just use link
    // from the program PATH, because other environments like GnuWin32 install
    // their own link.exe which may come first.
    linkPath = FindVisualStudioExecutable(TC, "link.exe",
                                          C.getDriver().getClangProgramPath());
  } else {
    linkPath = Linker;
    llvm::sys::path::replace_extension(linkPath, "exe");
    linkPath = TC.GetProgramPath(linkPath.c_str());
  }

  const char *Exec = Args.MakeArgString(linkPath);
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void visualstudio::Compiler::ConstructJob(Compilation &C, const JobAction &JA,
                                          const InputInfo &Output,
                                          const InputInfoList &Inputs,
                                          const ArgList &Args,
                                          const char *LinkingOutput) const {
  C.addCommand(GetCommand(C, JA, Output, Inputs, Args, LinkingOutput));
}

std::unique_ptr<Command> visualstudio::Compiler::GetCommand(
    Compilation &C, const JobAction &JA, const InputInfo &Output,
    const InputInfoList &Inputs, const ArgList &Args,
    const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  CmdArgs.push_back("/nologo");
  CmdArgs.push_back("/c");  // Compile only.
  CmdArgs.push_back("/W0"); // No warnings.

  // The goal is to be able to invoke this tool correctly based on
  // any flag accepted by clang-cl.

  // These are spelled the same way in clang and cl.exe,.
  Args.AddAllArgs(CmdArgs, {options::OPT_D, options::OPT_U, options::OPT_I});

  // Optimization level.
  if (Arg *A = Args.getLastArg(options::OPT_fbuiltin, options::OPT_fno_builtin))
    CmdArgs.push_back(A->getOption().getID() == options::OPT_fbuiltin ? "/Oi"
                                                                      : "/Oi-");
  if (Arg *A = Args.getLastArg(options::OPT_O, options::OPT_O0)) {
    if (A->getOption().getID() == options::OPT_O0) {
      CmdArgs.push_back("/Od");
    } else {
      CmdArgs.push_back("/Og");

      StringRef OptLevel = A->getValue();
      if (OptLevel == "s" || OptLevel == "z")
        CmdArgs.push_back("/Os");
      else
        CmdArgs.push_back("/Ot");

      CmdArgs.push_back("/Ob2");
    }
  }
  if (Arg *A = Args.getLastArg(options::OPT_fomit_frame_pointer,
                               options::OPT_fno_omit_frame_pointer))
    CmdArgs.push_back(A->getOption().getID() == options::OPT_fomit_frame_pointer
                          ? "/Oy"
                          : "/Oy-");
  if (!Args.hasArg(options::OPT_fwritable_strings))
    CmdArgs.push_back("/GF");

  // Flags for which clang-cl has an alias.
  // FIXME: How can we ensure this stays in sync with relevant clang-cl options?

  if (Args.hasFlag(options::OPT__SLASH_GR_, options::OPT__SLASH_GR,
                   /*default=*/false))
    CmdArgs.push_back("/GR-");

  if (Args.hasFlag(options::OPT__SLASH_GS_, options::OPT__SLASH_GS,
                   /*default=*/false))
    CmdArgs.push_back("/GS-");

  if (Arg *A = Args.getLastArg(options::OPT_ffunction_sections,
                               options::OPT_fno_function_sections))
    CmdArgs.push_back(A->getOption().getID() == options::OPT_ffunction_sections
                          ? "/Gy"
                          : "/Gy-");
  if (Arg *A = Args.getLastArg(options::OPT_fdata_sections,
                               options::OPT_fno_data_sections))
    CmdArgs.push_back(
        A->getOption().getID() == options::OPT_fdata_sections ? "/Gw" : "/Gw-");
  if (Args.hasArg(options::OPT_fsyntax_only))
    CmdArgs.push_back("/Zs");
  if (Args.hasArg(options::OPT_g_Flag, options::OPT_gline_tables_only,
                  options::OPT__SLASH_Z7))
    CmdArgs.push_back("/Z7");

  std::vector<std::string> Includes =
      Args.getAllArgValues(options::OPT_include);
  for (const auto &Include : Includes)
    CmdArgs.push_back(Args.MakeArgString(std::string("/FI") + Include));

  // Flags that can simply be passed through.
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_LD);
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_LDd);
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_GX);
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_GX_);
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_EH);
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_Zl);

  // The order of these flags is relevant, so pick the last one.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_MD, options::OPT__SLASH_MDd,
                               options::OPT__SLASH_MT, options::OPT__SLASH_MTd))
    A->render(Args, CmdArgs);

  // Pass through all unknown arguments so that the fallback command can see
  // them too.
  Args.AddAllArgs(CmdArgs, options::OPT_UNKNOWN);

  // Input filename.
  assert(Inputs.size() == 1);
  const InputInfo &II = Inputs[0];
  assert(II.getType() == types::TY_C || II.getType() == types::TY_CXX);
  CmdArgs.push_back(II.getType() == types::TY_C ? "/Tc" : "/Tp");
  if (II.isFilename())
    CmdArgs.push_back(II.getFilename());
  else
    II.getInputArg().renderAsInput(Args, CmdArgs);

  // Output filename.
  assert(Output.getType() == types::TY_Object);
  const char *Fo =
      Args.MakeArgString(std::string("/Fo") + Output.getFilename());
  CmdArgs.push_back(Fo);

  const Driver &D = getToolChain().getDriver();
  std::string Exec = FindVisualStudioExecutable(getToolChain(), "cl.exe",
                                                D.getClangProgramPath());
  return llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Exec),
                                    CmdArgs, Inputs);
}

/// MinGW Tools
void MinGW::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  if (getToolChain().getArch() == llvm::Triple::x86) {
    CmdArgs.push_back("--32");
  } else if (getToolChain().getArch() == llvm::Triple::x86_64) {
    CmdArgs.push_back("--64");
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));

  if (Args.hasArg(options::OPT_gsplit_dwarf))
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output,
                   SplitDebugName(Args, Inputs[0]));
}

void MinGW::Linker::AddLibGCC(const ArgList &Args,
                              ArgStringList &CmdArgs) const {
  if (Args.hasArg(options::OPT_mthreads))
    CmdArgs.push_back("-lmingwthrd");
  CmdArgs.push_back("-lmingw32");

  // Make use of compiler-rt if --rtlib option is used
  ToolChain::RuntimeLibType RLT = getToolChain().GetRuntimeLibType(Args);
  if (RLT == ToolChain::RLT_Libgcc) {
    bool Static = Args.hasArg(options::OPT_static_libgcc) ||
                  Args.hasArg(options::OPT_static);
    bool Shared = Args.hasArg(options::OPT_shared);
    bool CXX = getToolChain().getDriver().CCCIsCXX();

    if (Static || (!CXX && !Shared)) {
      CmdArgs.push_back("-lgcc");
      CmdArgs.push_back("-lgcc_eh");
    } else {
      CmdArgs.push_back("-lgcc_s");
      CmdArgs.push_back("-lgcc");
    }
  } else {
    AddRunTimeLibs(getToolChain(), getToolChain().getDriver(), CmdArgs, Args);
  }

  CmdArgs.push_back("-lmoldname");
  CmdArgs.push_back("-lmingwex");
  CmdArgs.push_back("-lmsvcrt");
}

void MinGW::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const ToolChain &TC = getToolChain();
  const Driver &D = TC.getDriver();
  // const SanitizerArgs &Sanitize = TC.getSanitizerArgs();

  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  StringRef LinkerName = Args.getLastArgValue(options::OPT_fuse_ld_EQ, "ld");
  if (LinkerName.equals_lower("lld")) {
    CmdArgs.push_back("-flavor");
    CmdArgs.push_back("gnu");
  } else if (!LinkerName.equals_lower("ld")) {
    D.Diag(diag::err_drv_unsupported_linker) << LinkerName;
  }

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  CmdArgs.push_back("-m");
  if (TC.getArch() == llvm::Triple::x86)
    CmdArgs.push_back("i386pe");
  if (TC.getArch() == llvm::Triple::x86_64)
    CmdArgs.push_back("i386pep");
  if (TC.getArch() == llvm::Triple::arm)
    CmdArgs.push_back("thumb2pe");

  if (Args.hasArg(options::OPT_mwindows)) {
    CmdArgs.push_back("--subsystem");
    CmdArgs.push_back("windows");
  } else if (Args.hasArg(options::OPT_mconsole)) {
    CmdArgs.push_back("--subsystem");
    CmdArgs.push_back("console");
  }

  if (Args.hasArg(options::OPT_static))
    CmdArgs.push_back("-Bstatic");
  else {
    if (Args.hasArg(options::OPT_mdll))
      CmdArgs.push_back("--dll");
    else if (Args.hasArg(options::OPT_shared))
      CmdArgs.push_back("--shared");
    CmdArgs.push_back("-Bdynamic");
    if (Args.hasArg(options::OPT_mdll) || Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-e");
      if (TC.getArch() == llvm::Triple::x86)
        CmdArgs.push_back("_DllMainCRTStartup@12");
      else
        CmdArgs.push_back("DllMainCRTStartup");
      CmdArgs.push_back("--enable-auto-image-base");
    }
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  Args.AddAllArgs(CmdArgs, options::OPT_e);
  // FIXME: add -N, -n flags
  Args.AddLastArg(CmdArgs, options::OPT_r);
  Args.AddLastArg(CmdArgs, options::OPT_s);
  Args.AddLastArg(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_u_Group);
  Args.AddLastArg(CmdArgs, options::OPT_Z_Flag);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_mdll)) {
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("dllcrt2.o")));
    } else {
      if (Args.hasArg(options::OPT_municode))
        CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crt2u.o")));
      else
        CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crt2.o")));
    }
    if (Args.hasArg(options::OPT_pg))
      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("gcrt2.o")));
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtbegin.o")));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  TC.AddFilePathLibArgs(Args, CmdArgs);
  AddLinkerInputs(TC, Inputs, Args, CmdArgs);

  // TODO: Add ASan stuff here

  // TODO: Add profile stuff here

  if (D.CCCIsCXX() &&
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                               !Args.hasArg(options::OPT_static);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
  }

  if (!Args.hasArg(options::OPT_nostdlib)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--start-group");

      if (Args.hasArg(options::OPT_fstack_protector) ||
          Args.hasArg(options::OPT_fstack_protector_strong) ||
          Args.hasArg(options::OPT_fstack_protector_all)) {
        CmdArgs.push_back("-lssp_nonshared");
        CmdArgs.push_back("-lssp");
      }
      if (Args.hasArg(options::OPT_fopenmp))
        CmdArgs.push_back("-lgomp");

      AddLibGCC(Args, CmdArgs);

      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lgmon");

      if (Args.hasArg(options::OPT_pthread))
        CmdArgs.push_back("-lpthread");

      // add system libraries
      if (Args.hasArg(options::OPT_mwindows)) {
        CmdArgs.push_back("-lgdi32");
        CmdArgs.push_back("-lcomdlg32");
      }
      CmdArgs.push_back("-ladvapi32");
      CmdArgs.push_back("-lshell32");
      CmdArgs.push_back("-luser32");
      CmdArgs.push_back("-lkernel32");

      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--end-group");
      else if (!LinkerName.equals_lower("lld"))
        AddLibGCC(Args, CmdArgs);
    }

    if (!Args.hasArg(options::OPT_nostartfiles)) {
      // Add crtfastmath.o if available and fast math is enabled.
      TC.AddFastMathRuntimeIfAvailable(Args, CmdArgs);

      CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtend.o")));
    }
  }
  const char *Exec = Args.MakeArgString(TC.GetProgramPath(LinkerName.data()));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

/// XCore Tools
// We pass assemble and link construction to the xcc tool.

void XCore::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  CmdArgs.push_back("-c");

  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-v");

  if (Arg *A = Args.getLastArg(options::OPT_g_Group))
    if (!A->getOption().matches(options::OPT_g0))
      CmdArgs.push_back("-g");

  if (Args.hasFlag(options::OPT_fverbose_asm, options::OPT_fno_verbose_asm,
                   false))
    CmdArgs.push_back("-fverbose-asm");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("xcc"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void XCore::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-v");

  // Pass -fexceptions through to the linker if it was present.
  if (Args.hasFlag(options::OPT_fexceptions, options::OPT_fno_exceptions,
                   false))
    CmdArgs.push_back("-fexceptions");

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("xcc"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void CrossWindows::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  const auto &TC =
      static_cast<const toolchains::CrossWindowsToolChain &>(getToolChain());
  ArgStringList CmdArgs;
  const char *Exec;

  switch (TC.getArch()) {
  default:
    llvm_unreachable("unsupported architecture");
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    break;
  case llvm::Triple::x86:
    CmdArgs.push_back("--32");
    break;
  case llvm::Triple::x86_64:
    CmdArgs.push_back("--64");
    break;
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &Input : Inputs)
    CmdArgs.push_back(Input.getFilename());

  const std::string Assembler = TC.GetProgramPath("as");
  Exec = Args.MakeArgString(Assembler);

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void CrossWindows::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                        const InputInfo &Output,
                                        const InputInfoList &Inputs,
                                        const ArgList &Args,
                                        const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::CrossWindowsToolChain &>(getToolChain());
  const llvm::Triple &T = TC.getTriple();
  const Driver &D = TC.getDriver();
  SmallString<128> EntryPoint;
  ArgStringList CmdArgs;
  const char *Exec;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_w);
  // Other warning options are already handled somewhere else.

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_pie))
    CmdArgs.push_back("-pie");
  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");
  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("--strip-all");

  CmdArgs.push_back("-m");
  switch (TC.getArch()) {
  default:
    llvm_unreachable("unsupported architecture");
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // FIXME: this is incorrect for WinCE
    CmdArgs.push_back("thumb2pe");
    break;
  case llvm::Triple::x86:
    CmdArgs.push_back("i386pe");
    EntryPoint.append("_");
    break;
  case llvm::Triple::x86_64:
    CmdArgs.push_back("i386pep");
    break;
  }

  if (Args.hasArg(options::OPT_shared)) {
    switch (T.getArch()) {
    default:
      llvm_unreachable("unsupported architecture");
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
    case llvm::Triple::x86_64:
      EntryPoint.append("_DllMainCRTStartup");
      break;
    case llvm::Triple::x86:
      EntryPoint.append("_DllMainCRTStartup@12");
      break;
    }

    CmdArgs.push_back("-shared");
    CmdArgs.push_back("-Bdynamic");

    CmdArgs.push_back("--enable-auto-image-base");

    CmdArgs.push_back("--entry");
    CmdArgs.push_back(Args.MakeArgString(EntryPoint));
  } else {
    EntryPoint.append("mainCRTStartup");

    CmdArgs.push_back(Args.hasArg(options::OPT_static) ? "-Bstatic"
                                                       : "-Bdynamic");

    if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
      CmdArgs.push_back("--entry");
      CmdArgs.push_back(Args.MakeArgString(EntryPoint));
    }

    // FIXME: handle subsystem
  }

  // NOTE: deal with multiple definitions on Windows (e.g. COMDAT)
  CmdArgs.push_back("--allow-multiple-definition");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_rdynamic)) {
    SmallString<261> ImpLib(Output.getFilename());
    llvm::sys::path::replace_extension(ImpLib, ".lib");

    CmdArgs.push_back("--out-implib");
    CmdArgs.push_back(Args.MakeArgString(ImpLib));
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    const std::string CRTPath(D.SysRoot + "/usr/lib/");
    const char *CRTBegin;

    CRTBegin =
        Args.hasArg(options::OPT_shared) ? "crtbeginS.obj" : "crtbegin.obj";
    CmdArgs.push_back(Args.MakeArgString(CRTPath + CRTBegin));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  TC.AddFilePathLibArgs(Args, CmdArgs);
  AddLinkerInputs(TC, Inputs, Args, CmdArgs);

  if (D.CCCIsCXX() && !Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    bool StaticCXX = Args.hasArg(options::OPT_static_libstdcxx) &&
                     !Args.hasArg(options::OPT_static);
    if (StaticCXX)
      CmdArgs.push_back("-Bstatic");
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (StaticCXX)
      CmdArgs.push_back("-Bdynamic");
  }

  if (!Args.hasArg(options::OPT_nostdlib)) {
    if (!Args.hasArg(options::OPT_nodefaultlibs)) {
      // TODO handle /MT[d] /MD[d]
      CmdArgs.push_back("-lmsvcrt");
      AddRunTimeLibs(TC, D, CmdArgs, Args);
    }
  }

  if (TC.getSanitizerArgs().needsAsanRt()) {
    // TODO handle /MT[d] /MD[d]
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(TC.getCompilerRTArgString(Args, "asan_dll_thunk"));
    } else {
      for (const auto &Lib : {"asan_dynamic", "asan_dynamic_runtime_thunk"})
        CmdArgs.push_back(TC.getCompilerRTArgString(Args, Lib));
      // Make sure the dynamic runtime thunk is not optimized out at link time
      // to ensure proper SEH handling.
      CmdArgs.push_back(Args.MakeArgString("--undefined"));
      CmdArgs.push_back(Args.MakeArgString(TC.getArch() == llvm::Triple::x86
                                               ? "___asan_seh_interceptor"
                                               : "__asan_seh_interceptor"));
    }
  }

  Exec = Args.MakeArgString(TC.GetLinkerPath());

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

void tools::SHAVE::Compiler::ConstructJob(Compilation &C, const JobAction &JA,
                                          const InputInfo &Output,
                                          const InputInfoList &Inputs,
                                          const ArgList &Args,
                                          const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  assert(Inputs.size() == 1);
  const InputInfo &II = Inputs[0];
  assert(II.getType() == types::TY_C || II.getType() == types::TY_CXX ||
         II.getType() == types::TY_PP_CXX);

  if (JA.getKind() == Action::PreprocessJobClass) {
    Args.ClaimAllArgs();
    CmdArgs.push_back("-E");
  } else {
    assert(Output.getType() == types::TY_PP_Asm); // Require preprocessed asm.
    CmdArgs.push_back("-S");
    CmdArgs.push_back("-fno-exceptions"); // Always do this even if unspecified.
  }
  CmdArgs.push_back("-DMYRIAD2");

  // Append all -I, -iquote, -isystem paths, defines/undefines,
  // 'f' flags, optimize flags, and warning options.
  // These are spelled the same way in clang and moviCompile.
  Args.AddAllArgs(CmdArgs, {options::OPT_I_Group, options::OPT_clang_i_Group,
                            options::OPT_std_EQ, options::OPT_D, options::OPT_U,
                            options::OPT_f_Group, options::OPT_f_clang_Group,
                            options::OPT_g_Group, options::OPT_M_Group,
                            options::OPT_O_Group, options::OPT_W_Group,
                            options::OPT_mcpu_EQ});

  // If we're producing a dependency file, and assembly is the final action,
  // then the name of the target in the dependency file should be the '.o'
  // file, not the '.s' file produced by this step. For example, instead of
  //  /tmp/mumble.s: mumble.c .../someheader.h
  // the filename on the lefthand side should be "mumble.o"
  if (Args.getLastArg(options::OPT_MF) && !Args.getLastArg(options::OPT_MT) &&
      C.getActions().size() == 1 &&
      C.getActions()[0]->getKind() == Action::AssembleJobClass) {
    Arg *A = Args.getLastArg(options::OPT_o);
    if (A) {
      CmdArgs.push_back("-MT");
      CmdArgs.push_back(Args.MakeArgString(A->getValue()));
    }
  }

  CmdArgs.push_back(II.getFilename());
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  std::string Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("moviCompile"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Exec),
                                          CmdArgs, Inputs));
}

void tools::SHAVE::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1);
  const InputInfo &II = Inputs[0];
  assert(II.getType() == types::TY_PP_Asm); // Require preprocessed asm input.
  assert(Output.getType() == types::TY_Object);

  CmdArgs.push_back("-no6thSlotCompression");
  const Arg *CPUArg = Args.getLastArg(options::OPT_mcpu_EQ);
  if (CPUArg)
    CmdArgs.push_back(
        Args.MakeArgString("-cv:" + StringRef(CPUArg->getValue())));
  CmdArgs.push_back("-noSPrefixing");
  CmdArgs.push_back("-a"); // Mystery option.
  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);
  for (const Arg *A : Args.filtered(options::OPT_I, options::OPT_isystem)) {
    A->claim();
    CmdArgs.push_back(
        Args.MakeArgString(std::string("-i:") + A->getValue(0)));
  }
  CmdArgs.push_back("-elf"); // Output format.
  CmdArgs.push_back(II.getFilename());
  CmdArgs.push_back(
      Args.MakeArgString(std::string("-o:") + Output.getFilename()));

  std::string Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("moviAsm"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Exec),
                                          CmdArgs, Inputs));
}

void tools::Myriad::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                         const InputInfo &Output,
                                         const InputInfoList &Inputs,
                                         const ArgList &Args,
                                         const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::MyriadToolChain &>(getToolChain());
  const llvm::Triple &T = TC.getTriple();
  ArgStringList CmdArgs;
  bool UseStartfiles =
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles);
  bool UseDefaultLibs =
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs);

  if (T.getArch() == llvm::Triple::sparc)
    CmdArgs.push_back("-EB");
  else // SHAVE assumes little-endian, and sparcel is expressly so.
    CmdArgs.push_back("-EL");

  // The remaining logic is mostly like gnutools::Linker::ConstructJob,
  // but we never pass through a --sysroot option and various other bits.
  // For example, there are no sanitizers (yet) nor gold linker.

  // Eat some arguments that may be present but have no effect.
  Args.ClaimAllArgs(options::OPT_g_Group);
  Args.ClaimAllArgs(options::OPT_w);
  Args.ClaimAllArgs(options::OPT_static_libgcc);

  if (Args.hasArg(options::OPT_s)) // Pass the 'strip' option.
    CmdArgs.push_back("-s");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (UseStartfiles) {
    // If you want startfiles, it means you want the builtin crti and crtbegin,
    // but not crt0. Myriad link commands provide their own crt0.o as needed.
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crti.o")));
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtbegin.o")));
  }

  Args.AddAllArgs(CmdArgs, {options::OPT_L, options::OPT_T_Group,
                            options::OPT_e, options::OPT_s, options::OPT_t,
                            options::OPT_Z_Flag, options::OPT_r});

  TC.AddFilePathLibArgs(Args, CmdArgs);

  bool NeedsSanitizerDeps = addSanitizerRuntimes(TC, Args, CmdArgs);
  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (UseDefaultLibs) {
    if (NeedsSanitizerDeps)
      linkSanitizerRuntimeDeps(TC, CmdArgs);
    if (C.getDriver().CCCIsCXX())
      CmdArgs.push_back("-lstdc++");
    if (T.getOS() == llvm::Triple::RTEMS) {
      CmdArgs.push_back("--start-group");
      CmdArgs.push_back("-lc");
      // You must provide your own "-L" option to enable finding these.
      CmdArgs.push_back("-lrtemscpu");
      CmdArgs.push_back("-lrtemsbsp");
      CmdArgs.push_back("--end-group");
    } else {
      CmdArgs.push_back("-lc");
    }
    CmdArgs.push_back("-lgcc");
  }
  if (UseStartfiles) {
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtend.o")));
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtn.o")));
  }

  std::string Exec =
      Args.MakeArgString(TC.GetProgramPath("sparc-myriad-elf-ld"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Exec),
                                          CmdArgs, Inputs));
}

void PS4cpu::Assemble::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  assert(Inputs.size() == 1 && "Unexpected number of inputs.");
  const InputInfo &Input = Inputs[0];
  assert(Input.isFilename() && "Invalid input.");
  CmdArgs.push_back(Input.getFilename());

  const char *Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("orbis-as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

static void AddPS4SanitizerArgs(const ToolChain &TC, ArgStringList &CmdArgs) {
  const SanitizerArgs &SanArgs = TC.getSanitizerArgs();
  if (SanArgs.needsUbsanRt()) {
    CmdArgs.push_back("-lSceDbgUBSanitizer_stub_weak");
  }
  if (SanArgs.needsAsanRt()) {
    CmdArgs.push_back("-lSceDbgAddressSanitizer_stub_weak");
  }
}

static void ConstructPS4LinkJob(const Tool &T, Compilation &C,
                                const JobAction &JA, const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) {
  const toolchains::FreeBSD &ToolChain =
      static_cast<const toolchains::FreeBSD &>(T.getToolChain());
  const Driver &D = ToolChain.getDriver();
  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_pie))
    CmdArgs.push_back("-pie");

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");
  if (Args.hasArg(options::OPT_shared))
    CmdArgs.push_back("--oformat=so");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  AddPS4SanitizerArgs(ToolChain, CmdArgs);

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (Args.hasArg(options::OPT_pthread)) {
    CmdArgs.push_back("-lpthread");
  }

  const char *Exec = Args.MakeArgString(ToolChain.GetProgramPath("orbis-ld"));

  C.addCommand(llvm::make_unique<Command>(JA, T, Exec, CmdArgs, Inputs));
}

static void ConstructGoldLinkJob(const Tool &T, Compilation &C,
                                 const JobAction &JA, const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) {
  const toolchains::FreeBSD &ToolChain =
      static_cast<const toolchains::FreeBSD &>(T.getToolChain());
  const Driver &D = ToolChain.getDriver();
  ArgStringList CmdArgs;

  // Silence warning for "clang -g foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_g_Group);
  // and "clang -emit-llvm foo.o -o foo"
  Args.ClaimAllArgs(options::OPT_emit_llvm);
  // and for "clang -w foo.o -o foo". Other warning options are already
  // handled somewhere else.
  Args.ClaimAllArgs(options::OPT_w);

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (Args.hasArg(options::OPT_pie))
    CmdArgs.push_back("-pie");

  if (Args.hasArg(options::OPT_static)) {
    CmdArgs.push_back("-Bstatic");
  } else {
    if (Args.hasArg(options::OPT_rdynamic))
      CmdArgs.push_back("-export-dynamic");
    CmdArgs.push_back("--eh-frame-hdr");
    if (Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-Bshareable");
    } else {
      CmdArgs.push_back("-dynamic-linker");
      CmdArgs.push_back("/libexec/ld-elf.so.1");
    }
    CmdArgs.push_back("--enable-new-dtags");
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  AddPS4SanitizerArgs(ToolChain, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    const char *crt1 = nullptr;
    if (!Args.hasArg(options::OPT_shared)) {
      if (Args.hasArg(options::OPT_pg))
        crt1 = "gcrt1.o";
      else if (Args.hasArg(options::OPT_pie))
        crt1 = "Scrt1.o";
      else
        crt1 = "crt1.o";
    }
    if (crt1)
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crt1)));

    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crti.o")));

    const char *crtbegin = nullptr;
    if (Args.hasArg(options::OPT_static))
      crtbegin = "crtbeginT.o";
    else if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_pie))
      crtbegin = "crtbeginS.o";
    else
      crtbegin = "crtbegin.o";

    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs)) {
    // For PS4, we always want to pass libm, libstdc++ and libkernel
    // libraries for both C and C++ compilations.
    CmdArgs.push_back("-lkernel");
    if (D.CCCIsCXX()) {
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lm_p");
      else
        CmdArgs.push_back("-lm");
    }
    // FIXME: For some reason GCC passes -lgcc and -lgcc_s before adding
    // the default system libraries. Just mimic this for now.
    if (Args.hasArg(options::OPT_pg))
      CmdArgs.push_back("-lgcc_p");
    else
      CmdArgs.push_back("-lcompiler_rt");
    if (Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-lstdc++");
    } else if (Args.hasArg(options::OPT_pg)) {
      CmdArgs.push_back("-lgcc_eh_p");
    } else {
      CmdArgs.push_back("--as-needed");
      CmdArgs.push_back("-lstdc++");
      CmdArgs.push_back("--no-as-needed");
    }

    if (Args.hasArg(options::OPT_pthread)) {
      if (Args.hasArg(options::OPT_pg))
        CmdArgs.push_back("-lpthread_p");
      else
        CmdArgs.push_back("-lpthread");
    }

    if (Args.hasArg(options::OPT_pg)) {
      if (Args.hasArg(options::OPT_shared))
        CmdArgs.push_back("-lc");
      else {
        if (Args.hasArg(options::OPT_static)) {
          CmdArgs.push_back("--start-group");
          CmdArgs.push_back("-lc_p");
          CmdArgs.push_back("-lpthread_p");
          CmdArgs.push_back("--end-group");
        } else {
          CmdArgs.push_back("-lc_p");
        }
      }
      CmdArgs.push_back("-lgcc_p");
    } else {
      if (Args.hasArg(options::OPT_static)) {
        CmdArgs.push_back("--start-group");
        CmdArgs.push_back("-lc");
        CmdArgs.push_back("-lpthread");
        CmdArgs.push_back("--end-group");
      } else {
        CmdArgs.push_back("-lc");
      }
      CmdArgs.push_back("-lcompiler_rt");
    }

    if (Args.hasArg(options::OPT_static)) {
      CmdArgs.push_back("-lstdc++");
    } else if (Args.hasArg(options::OPT_pg)) {
      CmdArgs.push_back("-lgcc_eh_p");
    } else {
      CmdArgs.push_back("--as-needed");
      CmdArgs.push_back("-lstdc++");
      CmdArgs.push_back("--no-as-needed");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_pie))
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtendS.o")));
    else
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtend.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));
  }

  const char *Exec =
#ifdef LLVM_ON_WIN32
      Args.MakeArgString(ToolChain.GetProgramPath("orbis-ld.gold"));
#else
      Args.MakeArgString(ToolChain.GetProgramPath("orbis-ld"));
#endif

  C.addCommand(llvm::make_unique<Command>(JA, T, Exec, CmdArgs, Inputs));
}

void PS4cpu::Link::ConstructJob(Compilation &C, const JobAction &JA,
                                const InputInfo &Output,
                                const InputInfoList &Inputs,
                                const ArgList &Args,
                                const char *LinkingOutput) const {
  const toolchains::FreeBSD &ToolChain =
      static_cast<const toolchains::FreeBSD &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  bool PS4Linker;
  StringRef LinkerOptName;
  if (const Arg *A = Args.getLastArg(options::OPT_fuse_ld_EQ)) {
    LinkerOptName = A->getValue();
    if (LinkerOptName != "ps4" && LinkerOptName != "gold")
      D.Diag(diag::err_drv_unsupported_linker) << LinkerOptName;
  }

  if (LinkerOptName == "gold")
    PS4Linker = false;
  else if (LinkerOptName == "ps4")
    PS4Linker = true;
  else
    PS4Linker = !Args.hasArg(options::OPT_shared);

  if (PS4Linker)
    ConstructPS4LinkJob(*this, C, JA, Output, Inputs, Args, LinkingOutput);
  else
    ConstructGoldLinkJob(*this, C, JA, Output, Inputs, Args, LinkingOutput);
}

void NVPTX::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs,
                                    const ArgList &Args,
                                    const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::CudaToolChain &>(getToolChain());
  assert(TC.getTriple().isNVPTX() && "Wrong platform");

  // Obtain architecture from the action.
  CudaArch gpu_arch = StringToCudaArch(JA.getOffloadingArch());
  assert(gpu_arch != CudaArch::UNKNOWN &&
         "Device action expected to have an architecture.");

  // Check that our installation's ptxas supports gpu_arch.
  if (!Args.hasArg(options::OPT_no_cuda_version_check)) {
    TC.cudaInstallation().CheckCudaVersionSupportsArch(gpu_arch);
  }

  ArgStringList CmdArgs;
  CmdArgs.push_back(TC.getTriple().isArch64Bit() ? "-m64" : "-m32");
  if (Args.hasFlag(options::OPT_cuda_noopt_device_debug,
                   options::OPT_no_cuda_noopt_device_debug, false)) {
    // ptxas does not accept -g option if optimization is enabled, so
    // we ignore the compiler's -O* options if we want debug info.
    CmdArgs.push_back("-g");
    CmdArgs.push_back("--dont-merge-basicblocks");
    CmdArgs.push_back("--return-at-end");
  } else if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    // Map the -O we received to -O{0,1,2,3}.
    //
    // TODO: Perhaps we should map host -O2 to ptxas -O3. -O3 is ptxas's
    // default, so it may correspond more closely to the spirit of clang -O2.

    // -O3 seems like the least-bad option when -Osomething is specified to
    // clang but it isn't handled below.
    StringRef OOpt = "3";
    if (A->getOption().matches(options::OPT_O4) ||
        A->getOption().matches(options::OPT_Ofast))
      OOpt = "3";
    else if (A->getOption().matches(options::OPT_O0))
      OOpt = "0";
    else if (A->getOption().matches(options::OPT_O)) {
      // -Os, -Oz, and -O(anything else) map to -O2, for lack of better options.
      OOpt = llvm::StringSwitch<const char *>(A->getValue())
                 .Case("1", "1")
                 .Case("2", "2")
                 .Case("3", "3")
                 .Case("s", "2")
                 .Case("z", "2")
                 .Default("2");
    }
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("-O") + OOpt));
  } else {
    // If no -O was passed, pass -O0 to ptxas -- no opt flag should correspond
    // to no optimizations, but ptxas's default is -O3.
    CmdArgs.push_back("-O0");
  }

  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(Args.MakeArgString(CudaArchToString(gpu_arch)));
  CmdArgs.push_back("--output-file");
  CmdArgs.push_back(Args.MakeArgString(Output.getFilename()));
  for (const auto& II : Inputs)
    CmdArgs.push_back(Args.MakeArgString(II.getFilename()));

  for (const auto& A : Args.getAllArgValues(options::OPT_Xcuda_ptxas))
    CmdArgs.push_back(Args.MakeArgString(A));

  const char *Exec = Args.MakeArgString(TC.GetProgramPath("ptxas"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}

// All inputs to this linker must be from CudaDeviceActions, as we need to look
// at the Inputs' Actions in order to figure out which GPU architecture they
// correspond to.
void NVPTX::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::CudaToolChain &>(getToolChain());
  assert(TC.getTriple().isNVPTX() && "Wrong platform");

  ArgStringList CmdArgs;
  CmdArgs.push_back("--cuda");
  CmdArgs.push_back(TC.getTriple().isArch64Bit() ? "-64" : "-32");
  CmdArgs.push_back(Args.MakeArgString("--create"));
  CmdArgs.push_back(Args.MakeArgString(Output.getFilename()));

  for (const auto& II : Inputs) {
    auto *A = II.getAction();
    assert(A->getInputs().size() == 1 &&
           "Device offload action is expected to have a single input");
    const char *gpu_arch_str = A->getOffloadingArch();
    assert(gpu_arch_str &&
           "Device action expected to have associated a GPU architecture!");
    CudaArch gpu_arch = StringToCudaArch(gpu_arch_str);

    // We need to pass an Arch of the form "sm_XX" for cubin files and
    // "compute_XX" for ptx.
    const char *Arch =
        (II.getType() == types::TY_PP_Asm)
            ? CudaVirtualArchToString(VirtualArchForCudaArch(gpu_arch))
            : gpu_arch_str;
    CmdArgs.push_back(Args.MakeArgString(llvm::Twine("--image=profile=") +
                                         Arch + ",file=" + II.getFilename()));
  }

  for (const auto& A : Args.getAllArgValues(options::OPT_Xcuda_fatbinary))
    CmdArgs.push_back(Args.MakeArgString(A));

  const char *Exec = Args.MakeArgString(TC.GetProgramPath("fatbinary"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs, Inputs));
}
