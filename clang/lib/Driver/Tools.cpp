//===--- Tools.cpp - Tools Implementations --------------------------------===//
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
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#ifdef LLVM_ON_UNIX
#include <unistd.h> // For getuid().
#endif

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm::opt;

static void addAssemblerKPIC(const ArgList &Args, ArgStringList &CmdArgs) {
  Arg *LastPICArg = Args.getLastArg(options::OPT_fPIC, options::OPT_fno_PIC,
                                    options::OPT_fpic, options::OPT_fno_pic,
                                    options::OPT_fPIE, options::OPT_fno_PIE,
                                    options::OPT_fpie, options::OPT_fno_pie);
  if (!LastPICArg)
    return;
  if (LastPICArg->getOption().matches(options::OPT_fPIC) ||
      LastPICArg->getOption().matches(options::OPT_fpic) ||
      LastPICArg->getOption().matches(options::OPT_fPIE) ||
      LastPICArg->getOption().matches(options::OPT_fpie)) {
    CmdArgs.push_back("-KPIC");
  }
}

/// CheckPreprocessingOptions - Perform some validation of preprocessing
/// arguments that is shared with gcc.
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

/// CheckCodeGenerationOptions - Perform some validation of code generation
/// arguments that is shared with gcc.
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
    if (!TC.HasNativeLLVMSupport()) {
      // Don't try to pass LLVM inputs unless we have native support.
      if (II.getType() == types::TY_LLVM_IR ||
          II.getType() == types::TY_LTO_IR ||
          II.getType() == types::TY_LLVM_BC || II.getType() == types::TY_LTO_BC)
        D.Diag(diag::err_drv_no_linker_llvm_support) << TC.getTripleString();
    }

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

void Clang::AddPreprocessingOptions(Compilation &C, const JobAction &JA,
                                    const Driver &D, const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    const InputInfo &Output,
                                    const InputInfoList &Inputs) const {
  Arg *A;

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

  // Add -i* options, and automatically translate to
  // -include-pch/-include-pth for transparent PCH support. It's
  // wonky, but we include looking for .gch so we can support seamless
  // replacement into a build system already set up to be generating
  // .gch files.
  bool RenderedImplicitInclude = false;
  for (const Arg *A : Args.filtered(options::OPT_clang_i_Group)) {
    if (A->getOption().matches(options::OPT_include)) {
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
    }

    // Not translated, render as usual.
    A->claim();
    A->render(Args, CmdArgs);
  }

  Args.AddAllArgs(CmdArgs, options::OPT_D, options::OPT_U);
  Args.AddAllArgs(CmdArgs, options::OPT_I_Group, options::OPT_F,
                  options::OPT_index_header_map);

  // Add -Wp, and -Xassembler if using the preprocessor.

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

  // Add C++ include arguments, if needed.
  if (types::isCXX(Inputs[0].getType()))
    getToolChain().AddClangCXXStdlibIncludeArgs(Args, CmdArgs);

  // Add system include arguments.
  getToolChain().AddClangSystemIncludeArgs(Args, CmdArgs);
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
    return true;
  }
}

// ARM tools start.

// Get SubArch (vN).
static int getARMSubArchVersionNumber(const llvm::Triple &Triple) {
  llvm::StringRef Arch = Triple.getArchName();
  return llvm::ARMTargetParser::parseArchVersion(Arch);
}

// True if M-profile.
static bool isARMMProfile(const llvm::Triple &Triple) {
  llvm::StringRef Arch = Triple.getArchName();
  unsigned Profile = llvm::ARMTargetParser::parseArchProfile(Arch);
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
  if (HWDiv == "arm") {
    Features.push_back("+hwdiv-arm");
    Features.push_back("-hwdiv");
  } else if (HWDiv == "thumb") {
    Features.push_back("-hwdiv-arm");
    Features.push_back("+hwdiv");
  } else if (HWDiv == "arm,thumb" || HWDiv == "thumb,arm") {
    Features.push_back("+hwdiv-arm");
    Features.push_back("+hwdiv");
  } else if (HWDiv == "none") {
    Features.push_back("-hwdiv-arm");
    Features.push_back("-hwdiv");
  } else
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

// Handle -mfpu=.
static void getARMFPUFeatures(const Driver &D, const Arg *A,
                              const ArgList &Args, StringRef FPU,
                              std::vector<const char *> &Features) {
  unsigned FPUID = llvm::ARMTargetParser::parseFPU(FPU);
  if (!llvm::ARMTargetParser::getFPUFeatures(FPUID, Features))
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

// Check if -march is valid by checking if it can be canonicalised and parsed.
// getARMArch is used here instead of just checking the -march value in order
// to handle -march=native correctly.
static void checkARMArchName(const Driver &D, const Arg *A, const ArgList &Args,
                             llvm::StringRef ArchName,
                             const llvm::Triple &Triple) {
  std::string MArch = arm::getARMArch(ArchName, Triple);
  if (llvm::ARMTargetParser::parseArch(MArch) == llvm::ARM::AK_INVALID)
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

// Check -mcpu=. Needs ArchName to handle -mcpu=generic.
static void checkARMCPUName(const Driver &D, const Arg *A, const ArgList &Args,
                            llvm::StringRef CPUName, llvm::StringRef ArchName,
                            const llvm::Triple &Triple) {
  std::string CPU = arm::getARMTargetCPU(CPUName, ArchName, Triple);
  std::string Arch = arm::getARMArch(ArchName, Triple);
  if (strcmp(arm::getLLVMArchSuffixForARM(CPU, Arch), "") == 0)
    D.Diag(diag::err_drv_clang_unsupported) << A->getAsString(Args);
}

// Select the float ABI as determined by -msoft-float, -mhard-float, and
// -mfloat-abi=.
StringRef tools::arm::getARMFloatABI(const Driver &D, const ArgList &Args,
                                     const llvm::Triple &Triple) {
  StringRef FloatABI;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float,
                          options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float))
      FloatABI = "soft";
    else if (A->getOption().matches(options::OPT_mhard_float))
      FloatABI = "hard";
    else {
      FloatABI = A->getValue();
      if (FloatABI != "soft" && FloatABI != "softfp" && FloatABI != "hard") {
        D.Diag(diag::err_drv_invalid_mfloat_abi) << A->getAsString(Args);
        FloatABI = "soft";
      }
    }
  }

  // If unspecified, choose the default based on the platform.
  if (FloatABI.empty()) {
    switch (Triple.getOS()) {
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
    case llvm::Triple::IOS: {
      // Darwin defaults to "softfp" for v6 and v7.
      //
      if (getARMSubArchVersionNumber(Triple) == 6 ||
          getARMSubArchVersionNumber(Triple) == 7)
        FloatABI = "softfp";
      else
        FloatABI = "soft";
      break;
    }

    // FIXME: this is invalid for WindowsCE
    case llvm::Triple::Win32:
      FloatABI = "hard";
      break;

    case llvm::Triple::FreeBSD:
      switch (Triple.getEnvironment()) {
      case llvm::Triple::GNUEABIHF:
        FloatABI = "hard";
        break;
      default:
        // FreeBSD defaults to soft float
        FloatABI = "soft";
        break;
      }
      break;

    default:
      switch (Triple.getEnvironment()) {
      case llvm::Triple::GNUEABIHF:
        FloatABI = "hard";
        break;
      case llvm::Triple::GNUEABI:
        FloatABI = "softfp";
        break;
      case llvm::Triple::EABIHF:
        FloatABI = "hard";
        break;
      case llvm::Triple::EABI:
        // EABI is always AAPCS, and if it was not marked 'hard', it's softfp
        FloatABI = "softfp";
        break;
      case llvm::Triple::Android: {
        if (getARMSubArchVersionNumber(Triple) == 7)
          FloatABI = "softfp";
        else
          FloatABI = "soft";
        break;
      }
      default:
        // Assume "soft", but warn the user we are guessing.
        FloatABI = "soft";
        if (Triple.getOS() != llvm::Triple::UnknownOS ||
            !Triple.isOSBinFormatMachO())
          D.Diag(diag::warn_drv_assuming_mfloat_abi_is) << "soft";
        break;
      }
    }
  }

  return FloatABI;
}

static void getARMTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args,
                                 std::vector<const char *> &Features,
                                 bool ForAS) {
  bool KernelOrKext =
      Args.hasArg(options::OPT_mkernel, options::OPT_fapple_kext);
  StringRef FloatABI = tools::arm::getARMFloatABI(D, Args, Triple);
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
    if (FloatABI == "soft")
      Features.push_back("+soft-float");

    // Use software floating point argument passing?
    if (FloatABI != "hard")
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

  // Check -march. ClangAs gives preference to -Wa,-march=.
  const Arg *ArchArg = Args.getLastArg(options::OPT_march_EQ);
  StringRef ArchName;
  if (WaArch) {
    if (ArchArg)
      D.Diag(clang::diag::warn_drv_unused_argument)
          << ArchArg->getAsString(Args);
    ArchName = StringRef(WaArch->getValue()).substr(7);
    checkARMArchName(D, WaArch, Args, ArchName, Triple);
    // FIXME: Set Arch.
    D.Diag(clang::diag::warn_drv_unused_argument) << WaArch->getAsString(Args);
  } else if (ArchArg) {
    ArchName = ArchArg->getValue();
    checkARMArchName(D, ArchArg, Args, ArchName, Triple);
  }

  // Check -mcpu. ClangAs gives preference to -Wa,-mcpu=.
  const Arg *CPUArg = Args.getLastArg(options::OPT_mcpu_EQ);
  StringRef CPUName;
  if (WaCPU) {
    if (CPUArg)
      D.Diag(clang::diag::warn_drv_unused_argument)
          << CPUArg->getAsString(Args);
    CPUName = StringRef(WaCPU->getValue()).substr(6);
    checkARMCPUName(D, WaCPU, Args, CPUName, ArchName, Triple);
  } else if (CPUArg) {
    CPUName = CPUArg->getValue();
    checkARMCPUName(D, CPUArg, Args, CPUName, ArchName, Triple);
  }

  // Setting -msoft-float effectively disables NEON because of the GCC
  // implementation, although the same isn't true of VFP or VFP3.
  if (FloatABI == "soft") {
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

  if (Triple.getSubArch() == llvm::Triple::SubArchType::ARMSubArch_v8_1a) {
    Features.insert(Features.begin(), "+v8.1a");
  }

  // Look for the last occurrence of -mlong-calls or -mno-long-calls. If
  // neither options are specified, see if we are compiling for kernel/kext and
  // decide whether to pass "+long-calls" based on the OS and its version.
  if (Arg *A = Args.getLastArg(options::OPT_mlong_calls,
                               options::OPT_mno_long_calls)) {
    if (A->getOption().matches(options::OPT_mlong_calls))
      Features.push_back("+long-calls");
  } else if (KernelOrKext && (!Triple.isiOS() || Triple.isOSVersionLT(6))) {
      Features.push_back("+long-calls");
  }
}

void Clang::AddARMTargetArgs(const ArgList &Args, ArgStringList &CmdArgs,
                             bool KernelOrKext) const {
  const Driver &D = getToolChain().getDriver();
  // Get the effective triple, which takes into account the deployment target.
  std::string TripleStr = getToolChain().ComputeEffectiveClangTriple(Args);
  llvm::Triple Triple(TripleStr);

  // Select the ABI to use.
  //
  // FIXME: Support -meabi.
  // FIXME: Parts of this are duplicated in the backend, unify this somehow.
  const char *ABIName = nullptr;
  if (Arg *A = Args.getLastArg(options::OPT_mabi_EQ)) {
    ABIName = A->getValue();
  } else if (Triple.isOSBinFormatMachO()) {
    // The backend is hardwired to assume AAPCS for M-class processors, ensure
    // the frontend matches that.
    if (Triple.getEnvironment() == llvm::Triple::EABI ||
        Triple.getOS() == llvm::Triple::UnknownOS || isARMMProfile(Triple)) {
      ABIName = "aapcs";
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
  StringRef FloatABI = tools::arm::getARMFloatABI(D, Args, Triple);
  if (FloatABI == "soft") {
    // Floating point operations and argument passing are soft.
    //
    // FIXME: This changes CPP defines, we need -target-soft-float.
    CmdArgs.push_back("-msoft-float");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else if (FloatABI == "softfp") {
    // Floating point operations are hard, but argument passing is soft.
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else {
    // Floating point operations and argument passing are hard.
    assert(FloatABI == "hard" && "Invalid float abi!");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("hard");
  }

  // Kernel code has more strict alignment requirements.
  if (KernelOrKext) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-arm-strict-align");

    // The kext linker doesn't know how to deal with movw/movt.
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-arm-use-movt=0");
  }

  // -mkernel implies -mstrict-align; don't add the redundant option.
  if (!KernelOrKext) {
    if (Arg *A = Args.getLastArg(options::OPT_mno_unaligned_access,
                                 options::OPT_munaligned_access)) {
      CmdArgs.push_back("-backend-option");
      if (A->getOption().matches(options::OPT_mno_unaligned_access))
        CmdArgs.push_back("-arm-strict-align");
      else {
        if (Triple.getSubArch() == llvm::Triple::SubArchType::ARMSubArch_v6m)
          D.Diag(diag::err_target_unsupported_unaligned) << "v6m";
        CmdArgs.push_back("-arm-no-strict-align");
      }
    }
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

  // llvm does not support reserving registers in general. There is support
  // for reserving r9 on ARM though (defined as a platform-specific register
  // in ARM EABI).
  if (Args.hasArg(options::OPT_ffixed_r9)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-arm-reserve-r9");
  }
}
// ARM tools end.

/// getAArch64TargetCPU - Get the (LLVM) name of the AArch64 cpu we are
/// targeting.
static std::string getAArch64TargetCPU(const ArgList &Args) {
  Arg *A;
  std::string CPU;
  // If we have -mtune or -mcpu, use that.
  if ((A = Args.getLastArg(options::OPT_mtune_EQ))) {
    CPU = A->getValue();
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
  std::string TripleStr = getToolChain().ComputeEffectiveClangTriple(Args);
  llvm::Triple Triple(TripleStr);

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

  if (Arg *A = Args.getLastArg(options::OPT_mno_unaligned_access,
                               options::OPT_munaligned_access)) {
    CmdArgs.push_back("-backend-option");
    if (A->getOption().matches(options::OPT_mno_unaligned_access))
      CmdArgs.push_back("-aarch64-strict-align");
    else
      CmdArgs.push_back("-aarch64-no-strict-align");
  }

  if (Arg *A = Args.getLastArg(options::OPT_mfix_cortex_a53_835769,
                               options::OPT_mno_fix_cortex_a53_835769)) {
    CmdArgs.push_back("-backend-option");
    if (A->getOption().matches(options::OPT_mfix_cortex_a53_835769))
      CmdArgs.push_back("-aarch64-fix-cortex-a53-835769=1");
    else
      CmdArgs.push_back("-aarch64-fix-cortex-a53-835769=0");
  } else if (Triple.getEnvironment() == llvm::Triple::Android) {
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

  if (Args.hasArg(options::OPT_ffixed_x18)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-aarch64-reserve-x18");
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
                  .Cases("o32", "eabi", DefMips32CPU)
                  .Cases("n32", "n64", DefMips64CPU)
                  .Default("");
  }

  // FIXME: Warn on inconsistent use of -march and -mabi.
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
static StringRef getMipsFloatABI(const Driver &D, const ArgList &Args) {
  StringRef FloatABI;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float,
                          options::OPT_mfloat_abi_EQ)) {
    if (A->getOption().matches(options::OPT_msoft_float))
      FloatABI = "soft";
    else if (A->getOption().matches(options::OPT_mhard_float))
      FloatABI = "hard";
    else {
      FloatABI = A->getValue();
      if (FloatABI != "soft" && FloatABI != "hard") {
        D.Diag(diag::err_drv_invalid_mfloat_abi) << A->getAsString(Args);
        FloatABI = "hard";
      }
    }
  }

  // If unspecified, choose the default based on the platform.
  if (FloatABI.empty()) {
    // Assume "hard", because it's a default value used by gcc.
    // When we start to recognize specific target MIPS processors,
    // we will be able to select the default more correctly.
    FloatABI = "hard";
  }

  return FloatABI;
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

  StringRef FloatABI = getMipsFloatABI(D, Args);
  if (FloatABI == "soft") {
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

  // Add the last -mfp32/-mfpxx/-mfp64 or if none are given and the ABI is O32
  // pass -mfpxx
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

  StringRef FloatABI = getMipsFloatABI(D, Args);

  if (FloatABI == "soft") {
    // Floating point operations and argument passing are soft.
    CmdArgs.push_back("-msoft-float");
    CmdArgs.push_back("-mfloat-abi");
    CmdArgs.push_back("soft");
  } else {
    // Floating point operations and argument passing are hard.
    assert(FloatABI == "hard" && "Invalid float abi!");
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
        .Case("pwr3", "pwr3")
        .Case("pwr4", "pwr4")
        .Case("pwr5", "pwr5")
        .Case("pwr5x", "pwr5x")
        .Case("pwr6", "pwr6")
        .Case("pwr6x", "pwr6x")
        .Case("pwr7", "pwr7")
        .Case("pwr8", "pwr8")
        .Case("powerpc", "ppc")
        .Case("powerpc64", "ppc64")
        .Case("powerpc64le", "ppc64le")
        .Default("");
  }

  return "";
}

static void getPPCTargetFeatures(const ArgList &Args,
                                 std::vector<const char *> &Features) {
  for (const Arg *A : Args.filtered(options::OPT_m_ppc_Features_Group)) {
    StringRef Name = A->getOption().getName();
    A->claim();

    // Skip over "-m".
    assert(Name.startswith("m") && "Invalid feature name.");
    Name = Name.substr(1);

    bool IsNegative = Name.startswith("no-");
    if (IsNegative)
      Name = Name.substr(3);

    // Note that gcc calls this mfcrf and LLVM calls this mfocrf so we
    // pass the correct option to the backend while calling the frontend
    // option the same.
    // TODO: Change the LLVM backend option maybe?
    if (Name == "mfcrf")
      Name = "mfocrf";

    Features.push_back(Args.MakeArgString((IsNegative ? "-" : "+") + Name));
  }

  // Altivec is a bit weird, allow overriding of the Altivec feature here.
  AddTargetFeature(Args, Features, options::OPT_faltivec,
                   options::OPT_fno_altivec, "altivec");
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

void Clang::AddSparcTargetArgs(const ArgList &Args,
                               ArgStringList &CmdArgs) const {
  const Driver &D = getToolChain().getDriver();
  std::string Triple = getToolChain().ComputeEffectiveClangTriple(Args);

  bool SoftFloatABI = false;
  if (Arg *A =
          Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float)) {
    if (A->getOption().matches(options::OPT_msoft_float))
      SoftFloatABI = true;
  }

  // Only the hard-float ABI on Sparc is standardized, and it is the
  // default. GCC also supports a nonstandard soft-float ABI mode, and
  // perhaps LLVM should implement that, too. However, since llvm
  // currently does not support Sparc soft-float, at all, display an
  // error if it's requested.
  if (SoftFloatABI) {
    D.Diag(diag::err_drv_unsupported_opt_for_target) << "-msoft-float"
                                                     << Triple;
  }
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
  if (Triple.getEnvironment() == llvm::Triple::Android)
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

static std::string getCPUName(const ArgList &Args, const llvm::Triple &T,
                              bool FromAs = false) {
  switch (T.getArch()) {
  default:
    return "";

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    return getAArch64TargetCPU(Args);

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
    return "hexagon" + toolchains::Hexagon_TC::GetTargetCPU(Args).str();

  case llvm::Triple::systemz:
    return getSystemZTargetCPU(Args);

  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
    return getR600TargetGPU(Args);
  }
}

static void AddGoldPlugin(const ToolChain &ToolChain, const ArgList &Args,
                          ArgStringList &CmdArgs) {
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
}

/// This is a helper function for validating the optional refinement step
/// parameter in reciprocal argument strings. Return false if there is an error
/// parsing the refinement step. Otherwise, return true and set the Position
/// of the refinement step in the input string.
static bool getRefinementStep(const StringRef &In, const Driver &D,
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
  if (Triple.getEnvironment() == llvm::Triple::Android) {
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
  for (const Arg *A : Args.filtered(options::OPT_m_x86_Features_Group)) {
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
}

void Clang::AddHexagonTargetArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-mqdsp6-compat");
  CmdArgs.push_back("-Wreturn-type");

  if (const char *v = toolchains::Hexagon_TC::GetSmallDataThreshold(Args)) {
    std::string SmallDataThreshold = "-hexagon-small-data-threshold=";
    SmallDataThreshold += v;
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back(Args.MakeArgString(SmallDataThreshold));
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

// Decode AArch64 features from string like +[no]featureA+[no]featureB+...
static bool DecodeAArch64Features(const Driver &D, StringRef text,
                                  std::vector<const char *> &Features) {
  SmallVector<StringRef, 8> Split;
  text.split(Split, StringRef("+"), -1, false);

  for (const StringRef Feature : Split) {
    const char *result = llvm::StringSwitch<const char *>(Feature)
                             .Case("fp", "+fp-armv8")
                             .Case("simd", "+neon")
                             .Case("crc", "+crc")
                             .Case("crypto", "+crypto")
                             .Case("nofp", "-fp-armv8")
                             .Case("nosimd", "-neon")
                             .Case("nocrc", "-crc")
                             .Case("nocrypto", "-crypto")
                             .Default(nullptr);
    if (result)
      Features.push_back(result);
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
  if (CPU == "cyclone" || CPU == "cortex-a53" || CPU == "cortex-a57" ||
      CPU == "cortex-a72") {
    Features.push_back("+neon");
    Features.push_back("+crc");
    Features.push_back("+crypto");
  } else if (CPU == "generic") {
    Features.push_back("+neon");
  } else {
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

  if (Split.first == "armv8-a" || Split.first == "armv8a") {
    // ok, no additional features.
  } else if (Split.first == "armv8.1-a" || Split.first == "armv8.1a") {
    Features.push_back("+v8.1a");
  } else {
    return false;
  }

  if (Split.second.size() && !DecodeAArch64Features(D, Split.second, Features))
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
  // Handle CPU name is 'native'.
  if (Mtune == "native")
    Mtune = llvm::sys::getHostCPUName();
  if (Mtune == "cyclone") {
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
    success = getAArch64ArchFeaturesFromMcpu(D, getAArch64TargetCPU(Args), Args,
                                             Features);

  if (success && (A = Args.getLastArg(options::OPT_mtune_EQ)))
    success =
        getAArch64MicroArchFeaturesFromMtune(D, A->getValue(), Args, Features);
  else if (success && (A = Args.getLastArg(options::OPT_mcpu_EQ)))
    success =
        getAArch64MicroArchFeaturesFromMcpu(D, A->getValue(), Args, Features);
  else if (Args.hasArg(options::OPT_arch))
    success = getAArch64MicroArchFeaturesFromMcpu(D, getAArch64TargetCPU(Args),
                                                  Args, Features);

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
}

static void getTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                              const ArgList &Args, ArgStringList &CmdArgs,
                              bool ForAS) {
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
    getARMTargetFeatures(D, Triple, Args, Features, ForAS);
    break;

  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    getPPCTargetFeatures(Args, Features);
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
    // Disable C++ EH by default on XCore, PS4, and MSVC.
    // FIXME: Remove MSVC from this list once things work.
    bool CXXExceptionsEnabled = Triple.getArch() != llvm::Triple::xcore &&
                                !Triple.isPS4CPU() &&
                                !Triple.isWindowsMSVCEnvironment();
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

  for (const auto &Act : *A)
    if (ContainsCompileAction(Act))
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

static void CollectArgsForIntegratedAssembler(Compilation &C,
                                              const ArgList &Args,
                                              ArgStringList &CmdArgs,
                                              const Driver &D) {
  if (UseRelaxAll(C, Args))
    CmdArgs.push_back("-mrelax-all");

  // When passing -I arguments to the assembler we sometimes need to
  // unconditionally take the next argument.  For example, when parsing
  // '-Wa,-I -Wa,foo' we need to accept the -Wa,foo arg after seeing the
  // -Wa,-I arg and when parsing '-Wa,-I,foo' we need to accept the 'foo'
  // arg after parsing the '-I' arg.
  bool TakeNextArg = false;

  // When using an integrated assembler, translate -Wa, and -Xassembler
  // options.
  bool CompressDebugSections = false;
  for (const Arg *A :
       Args.filtered(options::OPT_Wa_COMMA, options::OPT_Xassembler)) {
    A->claim();

    for (const StringRef Value : A->getValues()) {
      if (TakeNextArg) {
        CmdArgs.push_back(Value.data());
        TakeNextArg = false;
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
      } else if (Value.startswith("-I")) {
        CmdArgs.push_back(Value.data());
        // We need to consume the next argument if the current arg is a plain
        // -I. The next arg will be the include directory.
        if (Value == "-I")
          TakeNextArg = true;
      } else if (Value.startswith("-gdwarf-")) {
        CmdArgs.push_back(Value.data());
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
}

// Until ARM libraries are build separately, we have them all in one library
static StringRef getArchNameForCompilerRTLib(const ToolChain &TC) {
  if (TC.getTriple().isWindowsMSVCEnvironment() &&
      TC.getArch() == llvm::Triple::x86)
    return "i386";
  if (TC.getArch() == llvm::Triple::arm || TC.getArch() == llvm::Triple::armeb)
    return "arm";
  return TC.getArchName();
}

static SmallString<128> getCompilerRTLibDir(const ToolChain &TC) {
  // The runtimes are located in the OS-specific resource directory.
  SmallString<128> Res(TC.getDriver().ResourceDir);
  const llvm::Triple &Triple = TC.getTriple();
  // TC.getOS() yield "freebsd10.0" whereas "freebsd" is expected.
  StringRef OSLibName =
      (Triple.getOS() == llvm::Triple::FreeBSD) ? "freebsd" : TC.getOS();
  llvm::sys::path::append(Res, "lib", OSLibName);
  return Res;
}

SmallString<128> tools::getCompilerRT(const ToolChain &TC, StringRef Component,
                                      bool Shared) {
  const char *Env = TC.getTriple().getEnvironment() == llvm::Triple::Android
                        ? "-android"
                        : "";

  bool IsOSWindows = TC.getTriple().isOSWindows();
  bool IsITANMSVCWindows = TC.getTriple().isWindowsMSVCEnvironment() ||
                           TC.getTriple().isWindowsItaniumEnvironment();
  StringRef Arch = getArchNameForCompilerRTLib(TC);
  const char *Prefix = IsITANMSVCWindows ? "" : "lib";
  const char *Suffix =
      Shared ? (IsOSWindows ? ".dll" : ".so") : (IsITANMSVCWindows ? ".lib" : ".a");

  SmallString<128> Path = getCompilerRTLibDir(TC);
  llvm::sys::path::append(Path, Prefix + Twine("clang_rt.") + Component + "-" +
                                    Arch + Env + Suffix);

  return Path;
}

// This adds the static libclang_rt.builtins-arch.a directly to the command line
// FIXME: Make sure we can also emit shared objects if they're requested
// and available, check for possible errors, etc.
static void addClangRT(const ToolChain &TC, const ArgList &Args,
                       ArgStringList &CmdArgs) {
  CmdArgs.push_back(Args.MakeArgString(getCompilerRT(TC, "builtins")));

  if (!TC.getTriple().isOSWindows()) {
    // FIXME: why do we link against gcc when we are using compiler-rt?
    CmdArgs.push_back("-lgcc_s");
    if (TC.getDriver().CCCIsCXX())
      CmdArgs.push_back("-lgcc_eh");
  }
}

static void addProfileRT(const ToolChain &TC, const ArgList &Args,
                         ArgStringList &CmdArgs) {
  if (!(Args.hasFlag(options::OPT_fprofile_arcs, options::OPT_fno_profile_arcs,
                     false) ||
        Args.hasArg(options::OPT_fprofile_generate) ||
        Args.hasArg(options::OPT_fprofile_generate_EQ) ||
        Args.hasArg(options::OPT_fprofile_instr_generate) ||
        Args.hasArg(options::OPT_fprofile_instr_generate_EQ) ||
        Args.hasArg(options::OPT_fcreate_profile) ||
        Args.hasArg(options::OPT_coverage)))
    return;

  CmdArgs.push_back(Args.MakeArgString(getCompilerRT(TC, "profile")));
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
                                bool IsShared) {
  // Static runtimes must be forced into executable, so we wrap them in
  // whole-archive.
  if (!IsShared)
    CmdArgs.push_back("-whole-archive");
  CmdArgs.push_back(Args.MakeArgString(getCompilerRT(TC, Sanitizer, IsShared)));
  if (!IsShared)
    CmdArgs.push_back("-no-whole-archive");
}

// Tries to use a file with the list of dynamic symbols that need to be exported
// from the runtime library. Returns true if the file was found.
static bool addSanitizerDynamicList(const ToolChain &TC, const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    StringRef Sanitizer) {
  SmallString<128> SanRT = getCompilerRT(TC, Sanitizer);
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
  CmdArgs.push_back("-lpthread");
  CmdArgs.push_back("-lrt");
  CmdArgs.push_back("-lm");
  // There's no libdl on FreeBSD.
  if (TC.getTriple().getOS() != llvm::Triple::FreeBSD)
    CmdArgs.push_back("-ldl");
}

static void
collectSanitizerRuntimes(const ToolChain &TC, const ArgList &Args,
                         SmallVectorImpl<StringRef> &SharedRuntimes,
                         SmallVectorImpl<StringRef> &StaticRuntimes,
                         SmallVectorImpl<StringRef> &HelperStaticRuntimes) {
  const SanitizerArgs &SanArgs = TC.getSanitizerArgs();
  // Collect shared runtimes.
  if (SanArgs.needsAsanRt() && SanArgs.needsSharedAsanRt()) {
    SharedRuntimes.push_back("asan");
  }

  // Collect static runtimes.
  if (Args.hasArg(options::OPT_shared) ||
      (TC.getTriple().getEnvironment() == llvm::Triple::Android)) {
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
}

// Should be called before we add system libraries (C++ ABI, libstdc++/libc++,
// C runtime, etc). Returns true if sanitizer system deps need to be linked in.
static bool addSanitizerRuntimes(const ToolChain &TC, const ArgList &Args,
                                 ArgStringList &CmdArgs) {
  SmallVector<StringRef, 4> SharedRuntimes, StaticRuntimes,
      HelperStaticRuntimes;
  collectSanitizerRuntimes(TC, Args, SharedRuntimes, StaticRuntimes,
                           HelperStaticRuntimes);
  for (auto RT : SharedRuntimes)
    addSanitizerRuntime(TC, Args, CmdArgs, RT, true);
  for (auto RT : HelperStaticRuntimes)
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false);
  bool AddExportDynamic = false;
  for (auto RT : StaticRuntimes) {
    addSanitizerRuntime(TC, Args, CmdArgs, RT, false);
    AddExportDynamic |= !addSanitizerDynamicList(TC, Args, CmdArgs, RT);
  }
  // If there is a static runtime with no dynamic list, force all the symbols
  // to be dynamic to be sure we export sanitizer interface functions.
  if (AddExportDynamic)
    CmdArgs.push_back("-export-dynamic");
  return !StaticRuntimes.empty();
}

static bool areOptimizationsEnabled(const ArgList &Args) {
  // Find the last -O arg and see if it is non-zero.
  if (Arg *A = Args.getLastArg(options::OPT_O_Group))
    return !A->getOption().matches(options::OPT_O0);
  // Defaults to -O0.
  return false;
}

static bool shouldUseFramePointerForTarget(const ArgList &Args,
                                           const llvm::Triple &Triple) {
  // XCore never wants frame pointers, regardless of OS.
  if (Triple.getArch() == llvm::Triple::xcore) {
    return false;
  }

  if (Triple.isOSLinux()) {
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
    return A->getOption().matches(options::OPT_fno_omit_frame_pointer);

  return shouldUseFramePointerForTarget(Args, Triple);
}

static bool shouldUseLeafFramePointer(const ArgList &Args,
                                      const llvm::Triple &Triple) {
  if (Arg *A = Args.getLastArg(options::OPT_mno_omit_leaf_frame_pointer,
                               options::OPT_momit_leaf_frame_pointer))
    return A->getOption().matches(options::OPT_mno_omit_leaf_frame_pointer);

  if (Triple.isPS4CPU())
    return false;

  return shouldUseFramePointerForTarget(Args, Triple);
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

  // First extract the dwo sections.
  C.addCommand(llvm::make_unique<Command>(JA, T, Exec, ExtractArgs));

  // Then remove them from the original .o file.
  C.addCommand(llvm::make_unique<Command>(JA, T, Exec, StripArgs));
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

VersionTuple visualstudio::getMSVCVersion(const Driver *D,
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

    return VersionTuple(18);
  }
  return VersionTuple();
}

static void addPGOAndCoverageFlags(Compilation &C, const Driver &D,
                                   const InputInfo &Output, const ArgList &Args,
                                   ArgStringList &CmdArgs) {
  auto *ProfileGenerateArg = Args.getLastArg(
      options::OPT_fprofile_instr_generate,
      options::OPT_fprofile_instr_generate_EQ, options::OPT_fprofile_generate,
      options::OPT_fprofile_generate_EQ);

  auto *ProfileUseArg = Args.getLastArg(
      options::OPT_fprofile_instr_use, options::OPT_fprofile_instr_use_EQ,
      options::OPT_fprofile_use, options::OPT_fprofile_use_EQ);

  if (ProfileGenerateArg && ProfileUseArg)
    D.Diag(diag::err_drv_argument_not_allowed_with)
        << ProfileGenerateArg->getSpelling() << ProfileUseArg->getSpelling();

  if (ProfileGenerateArg &&
      ProfileGenerateArg->getOption().matches(
          options::OPT_fprofile_instr_generate_EQ))
    ProfileGenerateArg->render(Args, CmdArgs);
  else if (ProfileGenerateArg &&
           ProfileGenerateArg->getOption().matches(
               options::OPT_fprofile_generate_EQ)) {
    SmallString<128> Path(ProfileGenerateArg->getValue());
    llvm::sys::path::append(Path, "default.profraw");
    CmdArgs.push_back(
        Args.MakeArgString(Twine("-fprofile-instr-generate=") + Path));
  } else
    Args.AddAllArgs(CmdArgs, options::OPT_fprofile_instr_generate);

  if (ProfileUseArg &&
      ProfileUseArg->getOption().matches(options::OPT_fprofile_instr_use_EQ))
    ProfileUseArg->render(Args, CmdArgs);
  else if (ProfileUseArg &&
           (ProfileUseArg->getOption().matches(options::OPT_fprofile_use_EQ) ||
            ProfileUseArg->getOption().matches(
                options::OPT_fprofile_instr_use))) {
    SmallString<128> Path(
        ProfileUseArg->getNumValues() == 0 ? "" : ProfileUseArg->getValue());
    if (Path.empty() || llvm::sys::fs::is_directory(Path))
      llvm::sys::path::append(Path, "default.profdata");
    CmdArgs.push_back(Args.MakeArgString(Twine("-fprofile-instr-use=") + Path));
  }

  if (Args.hasArg(options::OPT_ftest_coverage) ||
      Args.hasArg(options::OPT_coverage))
    CmdArgs.push_back("-femit-coverage-notes");
  if (Args.hasFlag(options::OPT_fprofile_arcs, options::OPT_fno_profile_arcs,
                   false) ||
      Args.hasArg(options::OPT_coverage))
    CmdArgs.push_back("-femit-coverage-data");

  if (Args.hasArg(options::OPT_fcoverage_mapping) && !ProfileGenerateArg)
    D.Diag(diag::err_drv_argument_only_allowed_with)
        << "-fcoverage-mapping"
        << "-fprofile-instr-generate";

  if (Args.hasArg(options::OPT_fcoverage_mapping))
    CmdArgs.push_back("-fcoverage-mapping");

  if (C.getArgs().hasArg(options::OPT_c) ||
      C.getArgs().hasArg(options::OPT_S)) {
    if (Output.isFilename()) {
      CmdArgs.push_back("-coverage-file");
      SmallString<128> CoverageFilename;
      if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o)) {
        CoverageFilename = FinalOutput->getValue();
      } else {
        CoverageFilename = llvm::sys::path::filename(Output.getBaseInput());
      }
      if (llvm::sys::path::is_relative(CoverageFilename)) {
        SmallString<128> Pwd;
        if (!llvm::sys::fs::current_path(Pwd)) {
          llvm::sys::path::append(Pwd, CoverageFilename);
          CoverageFilename.swap(Pwd);
        }
      }
      CmdArgs.push_back(Args.MakeArgString(CoverageFilename));
    }
  }
}

void Clang::ConstructJob(Compilation &C, const JobAction &JA,
                         const InputInfo &Output, const InputInfoList &Inputs,
                         const ArgList &Args, const char *LinkingOutput) const {
  bool KernelOrKext =
      Args.hasArg(options::OPT_mkernel, options::OPT_fapple_kext);
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  bool IsWindowsGNU = getToolChain().getTriple().isWindowsGNUEnvironment();
  bool IsWindowsCygnus =
      getToolChain().getTriple().isWindowsCygwinEnvironment();
  bool IsWindowsMSVC = getToolChain().getTriple().isWindowsMSVCEnvironment();

  // Check number of inputs for sanity. We need at least one input.
  assert(Inputs.size() >= 1 && "Must have at least one input.");
  const InputInfo &Input = Inputs[0];
  // CUDA compilation may have multiple inputs (source file + results of
  // device-side compilations). All other jobs are expected to have exactly one
  // input.
  bool IsCuda = types::isCuda(Input.getType());
  assert((IsCuda || Inputs.size() == 1) && "Unable to handle multiple inputs.");

  // Invoke ourselves in -cc1 mode.
  //
  // FIXME: Implement custom jobs for internal actions.
  CmdArgs.push_back("-cc1");

  // Add the "effective" target triple.
  CmdArgs.push_back("-triple");
  std::string TripleStr = getToolChain().ComputeEffectiveClangTriple(Args);
  CmdArgs.push_back(Args.MakeArgString(TripleStr));

  const llvm::Triple TT(TripleStr);
  if (TT.isOSWindows() && (TT.getArch() == llvm::Triple::arm ||
                           TT.getArch() == llvm::Triple::thumb)) {
    unsigned Offset = TT.getArch() == llvm::Triple::arm ? 4 : 6;
    unsigned Version;
    TT.getArchName().substr(Offset).getAsInteger(10, Version);
    if (Version < 7)
      D.Diag(diag::err_target_unsupported_arch) << TT.getArchName()
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
    else if (UsePCH)
      CmdArgs.push_back("-emit-pch");
    else
      CmdArgs.push_back("-emit-pth");
  } else if (isa<VerifyPCHJobAction>(JA)) {
    CmdArgs.push_back("-verify-pch");
  } else {
    assert((isa<CompileJobAction>(JA) || isa<BackendJobAction>(JA)) &&
           "Invalid action for clang tool.");

    if (JA.getType() == types::TY_LTO_IR || JA.getType() == types::TY_LTO_BC) {
      CmdArgs.push_back("-flto");
    }
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
  }

  // We normally speed up the clang process a bit by skipping destructors at
  // exit, but when we're generating diagnostics we can rely on some of the
  // cleanup.
  if (!C.isForDiagnostics())
    CmdArgs.push_back("-disable-free");

// Disable the verification pass in -asserts builds.
#ifdef NDEBUG
  CmdArgs.push_back("-disable-llvm-verifier");
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

      if (!IsWindowsMSVC)
        CmdArgs.push_back("-analyzer-checker=unix");

      if (getToolChain().getTriple().getVendor() == llvm::Triple::Apple)
        CmdArgs.push_back("-analyzer-checker=osx");

      CmdArgs.push_back("-analyzer-checker=deadcode");

      if (types::isCXX(Input.getType()))
        CmdArgs.push_back("-analyzer-checker=cplusplus");

      // Enable the following experimental checkers for testing.
      CmdArgs.push_back(
          "-analyzer-checker=security.insecureAPI.UncheckedReturn");
      CmdArgs.push_back("-analyzer-checker=security.insecureAPI.getpw");
      CmdArgs.push_back("-analyzer-checker=security.insecureAPI.gets");
      CmdArgs.push_back("-analyzer-checker=security.insecureAPI.mktemp");
      CmdArgs.push_back("-analyzer-checker=security.insecureAPI.mkstemp");
      CmdArgs.push_back("-analyzer-checker=security.insecureAPI.vfork");
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

  bool PIE = getToolChain().isPIEDefault();
  bool PIC = PIE || getToolChain().isPICDefault();
  bool IsPICLevelTwo = PIC;

  // Android-specific defaults for PIC/PIE
  if (getToolChain().getTriple().getEnvironment() == llvm::Triple::Android) {
    switch (getToolChain().getArch()) {
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
  if (getToolChain().getTriple().getOS() == llvm::Triple::OpenBSD) {
    switch (getToolChain().getArch()) {
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

  // For the PIC and PIE flag options, this logic is different from the
  // legacy logic in very old versions of GCC, as that logic was just
  // a bug no one had ever fixed. This logic is both more rational and
  // consistent with GCC's new logic now that the bugs are fixed. The last
  // argument relating to either PIC or PIE wins, and no other argument is
  // used. If the last argument is any flavor of the '-fno-...' arguments,
  // both PIC and PIE are disabled. Any PIE option implicitly enables PIC
  // at the same level.
  Arg *LastPICArg = Args.getLastArg(options::OPT_fPIC, options::OPT_fno_PIC,
                                    options::OPT_fpic, options::OPT_fno_pic,
                                    options::OPT_fPIE, options::OPT_fno_PIE,
                                    options::OPT_fpie, options::OPT_fno_pie);
  // Check whether the tool chain trumps the PIC-ness decision. If the PIC-ness
  // is forced, then neither PIC nor PIE flags will have no effect.
  if (!getToolChain().isPICDefaultForced()) {
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
      }
    }
  }

  // Introduce a Darwin-specific hack. If the default is PIC but the flags
  // specified while enabling PIC enabled level 1 PIC, just force it back to
  // level 2 PIC instead. This matches the behavior of Darwin GCC (based on my
  // informal testing).
  if (PIC && getToolChain().getTriple().isOSDarwin())
    IsPICLevelTwo |= getToolChain().isPICDefault();

  // Note that these flags are trump-cards. Regardless of the order w.r.t. the
  // PIC or PIE options above, if these show up, PIC is disabled.
  llvm::Triple Triple(TripleStr);
  if (KernelOrKext && (!Triple.isiOS() || Triple.isOSVersionLT(6)))
    PIC = PIE = false;
  if (Args.hasArg(options::OPT_static))
    PIC = PIE = false;

  if (Arg *A = Args.getLastArg(options::OPT_mdynamic_no_pic)) {
    // This is a very special mode. It trumps the other modes, almost no one
    // uses it, and it isn't even valid on any OS but Darwin.
    if (!getToolChain().getTriple().isOSDarwin())
      D.Diag(diag::err_drv_unsupported_opt_for_target)
          << A->getSpelling() << getToolChain().getTriple().str();

    // FIXME: Warn when this flag trumps some other PIC or PIE flag.

    CmdArgs.push_back("-mrelocation-model");
    CmdArgs.push_back("dynamic-no-pic");

    // Only a forced PIC mode can cause the actual compile to have PIC defines
    // etc., no flags are sufficient. This behavior was selected to closely
    // match that of llvm-gcc and Apple GCC before that.
    if (getToolChain().isPICDefault() && getToolChain().isPICDefaultForced()) {
      CmdArgs.push_back("-pic-level");
      CmdArgs.push_back("2");
    }
  } else {
    // Currently, LLVM only knows about PIC vs. static; the PIE differences are
    // handled in Clang's IRGen by the -pie-level flag.
    CmdArgs.push_back("-mrelocation-model");
    CmdArgs.push_back(PIC ? "pic" : "static");

    if (PIC) {
      CmdArgs.push_back("-pic-level");
      CmdArgs.push_back(IsPICLevelTwo ? "2" : "1");
      if (PIE) {
        CmdArgs.push_back("-pie-level");
        CmdArgs.push_back(IsPICLevelTwo ? "2" : "1");
      }
    }
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
      CmdArgs.push_back("-frewrite-map-file");
      CmdArgs.push_back(A->getValue());
      A->claim();
    }
  }

  if (Arg *A = Args.getLastArg(options::OPT_Wframe_larger_than_EQ)) {
    StringRef v = A->getValue();
    CmdArgs.push_back("-mllvm");
    CmdArgs.push_back(Args.MakeArgString("-warn-stack-size=" + v));
    A->claim();
  }

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
    CmdArgs.push_back("-mrtd");

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

  // Enable -mconstructor-aliases except on darwin, where we have to
  // work around a linker bug;  see <rdar://problem/7651567>.
  if (!getToolChain().getTriple().isOSDarwin())
    CmdArgs.push_back("-mconstructor-aliases");

  // Darwin's kernel doesn't support guard variables; just die if we
  // try to use them.
  if (KernelOrKext && getToolChain().getTriple().isOSDarwin())
    CmdArgs.push_back("-fforbid-guard-variables");

  if (Args.hasArg(options::OPT_mms_bitfields)) {
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
  getTargetFeatures(D, Triple, Args, CmdArgs, false);

  // Add target specific flags.
  switch (getToolChain().getArch()) {
  default:
    break;

  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    AddARMTargetArgs(Args, CmdArgs, KernelOrKext);
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

  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    AddX86TargetArgs(Args, CmdArgs);
    break;

  case llvm::Triple::hexagon:
    AddHexagonTargetArgs(Args, CmdArgs);
    break;
  }

  // Add clang-cl arguments.
  if (getToolChain().getDriver().IsCLMode())
    AddClangCLArgs(Args, CmdArgs);

  // Pass the linker version in use.
  if (Arg *A = Args.getLastArg(options::OPT_mlinker_version_EQ)) {
    CmdArgs.push_back("-target-linker-version");
    CmdArgs.push_back(A->getValue());
  }

  if (!shouldUseLeafFramePointer(Args, getToolChain().getTriple()))
    CmdArgs.push_back("-momit-leaf-frame-pointer");

  // Explicitly error on some things we know we don't support and can't just
  // ignore.
  types::ID InputType = Input.getType();
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

  // Use the last option from "-g" group. "-gline-tables-only" and "-gdwarf-x"
  // are preserved, all other debug options are substituted with "-g".
  Args.ClaimAllArgs(options::OPT_g_Group);
  if (Arg *A = Args.getLastArg(options::OPT_g_Group)) {
    if (A->getOption().matches(options::OPT_gline_tables_only) ||
        A->getOption().matches(options::OPT_g1)) {
      // FIXME: we should support specifying dwarf version with
      // -gline-tables-only.
      CmdArgs.push_back("-gline-tables-only");
      // Default is dwarf-2 for Darwin, OpenBSD, FreeBSD and Solaris.
      const llvm::Triple &Triple = getToolChain().getTriple();
      if (Triple.isOSDarwin() || Triple.getOS() == llvm::Triple::OpenBSD ||
          Triple.getOS() == llvm::Triple::FreeBSD ||
          Triple.getOS() == llvm::Triple::Solaris)
        CmdArgs.push_back("-gdwarf-2");
    } else if (A->getOption().matches(options::OPT_gdwarf_2))
      CmdArgs.push_back("-gdwarf-2");
    else if (A->getOption().matches(options::OPT_gdwarf_3))
      CmdArgs.push_back("-gdwarf-3");
    else if (A->getOption().matches(options::OPT_gdwarf_4))
      CmdArgs.push_back("-gdwarf-4");
    else if (!A->getOption().matches(options::OPT_g0) &&
             !A->getOption().matches(options::OPT_ggdb0)) {
      // Default is dwarf-2 for Darwin, OpenBSD, FreeBSD and Solaris.
      const llvm::Triple &Triple = getToolChain().getTriple();
      if (Triple.isOSDarwin() || Triple.getOS() == llvm::Triple::OpenBSD ||
          Triple.getOS() == llvm::Triple::FreeBSD ||
          Triple.getOS() == llvm::Triple::Solaris)
        CmdArgs.push_back("-gdwarf-2");
      else
        CmdArgs.push_back("-g");
    }
  }

  // We ignore flags -gstrict-dwarf and -grecord-gcc-switches for now.
  Args.ClaimAllArgs(options::OPT_g_flags_Group);
  if (Args.hasFlag(options::OPT_gcolumn_info, options::OPT_gno_column_info,
                   /*Default*/ true))
    CmdArgs.push_back("-dwarf-column-info");

  // FIXME: Move backend command line options to the module.
  // -gsplit-dwarf should turn on -g and enable the backend dwarf
  // splitting and extraction.
  // FIXME: Currently only works on Linux.
  if (getToolChain().getTriple().isOSLinux() &&
      Args.hasArg(options::OPT_gsplit_dwarf)) {
    CmdArgs.push_back("-g");
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-split-dwarf=Enable");
  }

  // -ggnu-pubnames turns on gnu style pubnames in the backend.
  if (Args.hasArg(options::OPT_ggnu_pubnames)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-generate-gnu-dwarf-pub-sections");
  }

  // -gdwarf-aranges turns on the emission of the aranges section in the
  // backend.
  if (Args.hasArg(options::OPT_gdwarf_aranges)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-generate-arange-section");
  }

  if (Args.hasFlag(options::OPT_fdebug_types_section,
                   options::OPT_fno_debug_types_section, false)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-generate-type-units");
  }

  // CloudABI uses -ffunction-sections and -fdata-sections by default.
  bool UseSeparateSections = Triple.getOS() == llvm::Triple::CloudABI;

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

  addPGOAndCoverageFlags(C, D, Output, Args, CmdArgs);

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
  Args.AddLastArg(CmdArgs, options::OPT_fstandalone_debug);
  Args.AddLastArg(CmdArgs, options::OPT_fno_standalone_debug);
  Args.AddLastArg(CmdArgs, options::OPT_fno_operator_names);
  // AltiVec-like language extensions aren't relevant for assembling.
  if (!isa<PreprocessJobAction>(JA) || Output.getType() != types::TY_PP_Asm) {
    Args.AddLastArg(CmdArgs, options::OPT_faltivec);
    Args.AddLastArg(CmdArgs, options::OPT_fzvector);
  }
  Args.AddLastArg(CmdArgs, options::OPT_fdiagnostics_show_template_tree);
  Args.AddLastArg(CmdArgs, options::OPT_fno_elide_type);

  // Forward flags for OpenMP
  if (Args.hasFlag(options::OPT_fopenmp, options::OPT_fopenmp_EQ,
                   options::OPT_fno_openmp, false))
    switch (getOpenMPRuntime(getToolChain(), Args)) {
    case OMPRT_OMP:
    case OMPRT_IOMP5:
      // Clang can generate useful OpenMP code for these two runtime libraries.
      CmdArgs.push_back("-fopenmp");

      // If no option regarding the use of TLS in OpenMP codegeneration is
      // given, decide a default based on the target. Otherwise rely on the
      // options and pass the right information to the frontend.
      if (!Args.hasFlag(options::OPT_fopenmp_use_tls,
                        options::OPT_fnoopenmp_use_tls,
                        getToolChain().getArch() == llvm::Triple::ppc ||
                            getToolChain().getArch() == llvm::Triple::ppc64 ||
                            getToolChain().getArch() == llvm::Triple::ppc64le))
        CmdArgs.push_back("-fnoopenmp-use-tls");
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

  if (Args.getLastArg(options::OPT_fapple_kext))
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
  if (getToolChain().getSanitizerArgs().needsSafeStackRt()) {
    Args.ClaimAllArgs(options::OPT_fno_stack_protector);
    Args.ClaimAllArgs(options::OPT_fstack_protector_all);
    Args.ClaimAllArgs(options::OPT_fstack_protector_strong);
    Args.ClaimAllArgs(options::OPT_fstack_protector);
  } else if (Arg *A = Args.getLastArg(options::OPT_fno_stack_protector,
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
                   false)) {
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-force-align-stack");
  }
  if (!Args.hasFlag(options::OPT_mno_stackrealign, options::OPT_mstackrealign,
                    false)) {
    CmdArgs.push_back(Args.MakeArgString("-mstackrealign"));
  }

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

  if (getToolChain().getArch() == llvm::Triple::aarch64 ||
      getToolChain().getArch() == llvm::Triple::aarch64_be)
    CmdArgs.push_back("-fallow-half-arguments-and-returns");

  if (Arg *A = Args.getLastArg(options::OPT_mrestrict_it,
                               options::OPT_mno_restrict_it)) {
    if (A->getOption().matches(options::OPT_mrestrict_it)) {
      CmdArgs.push_back("-backend-option");
      CmdArgs.push_back("-arm-restrict-it");
    } else {
      CmdArgs.push_back("-backend-option");
      CmdArgs.push_back("-arm-no-restrict-it");
    }
  } else if (TT.isOSWindows() && (TT.getArch() == llvm::Triple::arm ||
                                  TT.getArch() == llvm::Triple::thumb)) {
    // Windows on ARM expects restricted IT blocks
    CmdArgs.push_back("-backend-option");
    CmdArgs.push_back("-arm-restrict-it");
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

  if (Args.hasArg(options::OPT_mkernel)) {
    if (!Args.hasArg(options::OPT_fapple_kext) && types::isCXX(InputType))
      CmdArgs.push_back("-fapple-kext");
    if (!Args.hasArg(options::OPT_fbuiltin))
      CmdArgs.push_back("-fno-builtin");
    Args.ClaimAllArgs(options::OPT_fno_builtin);
  }
  // -fbuiltin is default.
  else if (!Args.hasFlag(options::OPT_fbuiltin, options::OPT_fno_builtin))
    CmdArgs.push_back("-fno-builtin");

  if (!Args.hasFlag(options::OPT_fassume_sane_operator_new,
                    options::OPT_fno_assume_sane_operator_new))
    CmdArgs.push_back("-fno-assume-sane-operator-new");

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
  bool HaveModules = false;
  if (Args.hasFlag(options::OPT_fmodules, options::OPT_fno_modules, false)) {
    bool AllowedInCXX = Args.hasFlag(options::OPT_fcxx_modules,
                                     options::OPT_fno_cxx_modules, true);
    if (AllowedInCXX || !types::isCXX(InputType)) {
      CmdArgs.push_back("-fmodules");
      HaveModules = true;
    }
  }

  // -fmodule-maps enables implicit reading of module map files. By default,
  // this is enabled if we are using precompiled modules.
  if (Args.hasFlag(options::OPT_fimplicit_module_maps,
                   options::OPT_fno_implicit_module_maps, HaveModules)) {
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
                    options::OPT_fno_implicit_modules)) {
    CmdArgs.push_back("-fno-implicit-modules");
  }

  // -fmodule-name specifies the module that is currently being built (or
  // used for header checking by -fmodule-maps).
  Args.AddLastArg(CmdArgs, options::OPT_fmodule_name);

  // -fmodule-map-file can be used to specify files containing module
  // definitions.
  Args.AddAllArgs(CmdArgs, options::OPT_fmodule_map_file);

  // -fmodule-file can be used to specify files containing precompiled modules.
  Args.AddAllArgs(CmdArgs, options::OPT_fmodule_file);

  // -fmodule-cache-path specifies where our implicitly-built module files
  // should be written.
  SmallString<128> Path;
  if (Arg *A = Args.getLastArg(options::OPT_fmodules_cache_path))
    Path = A->getValue();
  if (HaveModules) {
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

  // When building modules and generating crashdumps, we need to dump a module
  // dependency VFS alongside the output.
  if (HaveModules && C.isForDiagnostics()) {
    SmallString<128> VFSDir(Output.getFilename());
    llvm::sys::path::replace_extension(VFSDir, ".cache");
    // Add the cache directory as a temp so the crash diagnostics pick it up.
    C.addTempFile(Args.MakeArgString(VFSDir));

    llvm::sys::path::append(VFSDir, "vfs");
    CmdArgs.push_back("-module-dependency-dir");
    CmdArgs.push_back(Args.MakeArgString(VFSDir));
  }

  if (HaveModules)
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
  if (!Args.hasFlag(options::OPT_fuse_cxa_atexit,
                    options::OPT_fno_use_cxa_atexit,
                    !IsWindowsCygnus && !IsWindowsGNU &&
                        getToolChain().getArch() != llvm::Triple::hexagon &&
                        getToolChain().getArch() != llvm::Triple::xcore) ||
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
      &D, getToolChain().getTriple(), Args, IsWindowsMSVC);
  if (!MSVT.empty())
    CmdArgs.push_back(
        Args.MakeArgString("-fms-compatibility-version=" + MSVT.getAsString()));

  bool IsMSVC2015Compatible = MSVT.getMajor() >= 19;
  if (ImplyVCPPCXXVer) {
    if (IsMSVC2015Compatible)
      CmdArgs.push_back("-std=c++14");
    else
      CmdArgs.push_back("-std=c++11");
  }

  // -fno-borland-extensions is default.
  if (Args.hasFlag(options::OPT_fborland_extensions,
                   options::OPT_fno_borland_extensions, false))
    CmdArgs.push_back("-fborland-extensions");

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

  if (Args.hasArg(options::OPT_fno_inline_functions))
    CmdArgs.push_back("-fno-inline-functions");

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

  if (Args.hasFlag(options::OPT_fapplication_extension,
                   options::OPT_fno_application_extension, false))
    CmdArgs.push_back("-fapplication-extension");

  // Handle GCC-style exception args.
  if (!C.getDriver().IsCLMode())
    addExceptionArgs(Args, InputType, getToolChain(), KernelOrKext, objcRuntime,
                     CmdArgs);

  if (getToolChain().UseSjLjExceptions())
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

  if (KernelOrKext || isNoCommonDefault(getToolChain().getTriple())) {
    if (!Args.hasArg(options::OPT_fcommon))
      CmdArgs.push_back("-fno-common");
    Args.ClaimAllArgs(options::OPT_fno_common);
  }

  // -fcommon is default, only pass non-default.
  else if (!Args.hasFlag(options::OPT_fcommon, options::OPT_fno_common))
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
    if (value != "UTF-8")
      D.Diag(diag::err_drv_invalid_value) << inputCharset->getAsString(Args)
                                          << value;
  }

  // -fexec_charset=UTF-8 is default. Reject others
  if (Arg *execCharset = Args.getLastArg(options::OPT_fexec_charset_EQ)) {
    StringRef value = execCharset->getValue();
    if (value != "UTF-8")
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

  // Color diagnostics are the default, unless the terminal doesn't support
  // them.
  // Support both clang's -f[no-]color-diagnostics and gcc's
  // -f[no-]diagnostics-colors[=never|always|auto].
  enum { Colors_On, Colors_Off, Colors_Auto } ShowColors = Colors_Auto;
  for (const auto &Arg : Args) {
    const Option &O = Arg->getOption();
    if (!O.matches(options::OPT_fcolor_diagnostics) &&
        !O.matches(options::OPT_fdiagnostics_color) &&
        !O.matches(options::OPT_fno_color_diagnostics) &&
        !O.matches(options::OPT_fno_diagnostics_color) &&
        !O.matches(options::OPT_fdiagnostics_color_EQ))
      continue;

    Arg->claim();
    if (O.matches(options::OPT_fcolor_diagnostics) ||
        O.matches(options::OPT_fdiagnostics_color)) {
      ShowColors = Colors_On;
    } else if (O.matches(options::OPT_fno_color_diagnostics) ||
               O.matches(options::OPT_fno_diagnostics_color)) {
      ShowColors = Colors_Off;
    } else {
      assert(O.matches(options::OPT_fdiagnostics_color_EQ));
      StringRef value(Arg->getValue());
      if (value == "always")
        ShowColors = Colors_On;
      else if (value == "never")
        ShowColors = Colors_Off;
      else if (value == "auto")
        ShowColors = Colors_Auto;
      else
        getToolChain().getDriver().Diag(diag::err_drv_clang_unsupported)
            << ("-fdiagnostics-color=" + value).str();
    }
  }
  if (ShowColors == Colors_On ||
      (ShowColors == Colors_Auto && llvm::sys::Process::StandardErrHasColors()))
    CmdArgs.push_back("-fcolor-diagnostics");

  if (Args.hasArg(options::OPT_fansi_escape_codes))
    CmdArgs.push_back("-fansi-escape-codes");

  if (!Args.hasFlag(options::OPT_fshow_source_location,
                    options::OPT_fno_show_source_location))
    CmdArgs.push_back("-fno-show-source-location");

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
// FIXME: This is disabled until clang -cc1 supports -fno-builtin-foo. PR4941.
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
      (C.isForDiagnostics() && !HaveModules))
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

  // Forward -Xclang arguments to -cc1, and -mllvm arguments to the LLVM option
  // parser.
  Args.AddAllArgValues(CmdArgs, options::OPT_Xclang);
  bool OptDisabled = false;
  for (const Arg *A : Args.filtered(options::OPT_mllvm)) {
    A->claim();

    // We translate this by hand to the -cc1 argument, since nightly test uses
    // it and developers have been trained to spell it with -mllvm.
    if (StringRef(A->getValue(0)) == "-disable-llvm-optzns") {
      CmdArgs.push_back("-disable-llvm-optzns");
      OptDisabled = true;
    } else
      A->render(Args, CmdArgs);
  }

  // With -save-temps, we want to save the unoptimized bitcode output from the
  // CompileJobAction, so disable optimizations if they are not already
  // disabled.
  if (C.getDriver().isSaveTempsEnabled() && !OptDisabled &&
      isa<CompileJobAction>(JA))
    CmdArgs.push_back("-disable-llvm-optzns");

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
  bool SplitDwarf = Args.hasArg(options::OPT_gsplit_dwarf) &&
                    getToolChain().getTriple().isOSLinux() &&
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
    for (InputInfoList::const_iterator it = std::next(Inputs.begin()),
                                       ie = Inputs.end();
         it != ie; ++it) {
      CmdArgs.push_back("-fcuda-include-gpubinary");
      CmdArgs.push_back(it->getFilename());
    }

  // Finally add the compile command to the compilation.
  if (Args.hasArg(options::OPT__SLASH_fallback) &&
      Output.getType() == types::TY_Object &&
      (InputType == types::TY_C || InputType == types::TY_CXX)) {
    auto CLCommand =
        getCLFallback()->GetCommand(C, JA, Output, Inputs, Args, LinkingOutput);
    C.addCommand(llvm::make_unique<FallbackCommand>(JA, *this, Exec, CmdArgs,
                                                    std::move(CLCommand)));
  } else {
    C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
  }

  // Handle the debug info splitting at object creation time if we're
  // creating an object.
  // TODO: Currently only works on linux with newer objcopy.
  if (SplitDwarf && !isa<CompileJobAction>(JA) && !isa<BackendJobAction>(JA))
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
    // Legacy behaviour is to target the gnustep runtime if we are i
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

struct EHFlags {
  EHFlags() : Synch(false), Asynch(false), NoExceptC(false) {}
  bool Synch;
  bool Asynch;
  bool NoExceptC;
};

/// /EH controls whether to run destructor cleanups when exceptions are
/// thrown.  There are three modifiers:
/// - s: Cleanup after "synchronous" exceptions, aka C++ exceptions.
/// - a: Cleanup after "asynchronous" exceptions, aka structured exceptions.
///      The 'a' modifier is unimplemented and fundamentally hard in LLVM IR.
/// - c: Assume that extern "C" functions are implicitly noexcept.  This
///      modifier is an optimization, so we ignore it for now.
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
        continue;
      case 'c':
        EH.NoExceptC = maybeConsumeDash(EHVal, I);
        continue;
      case 's':
        EH.Synch = maybeConsumeDash(EHVal, I);
        continue;
      default:
        break;
      }
      D.Diag(clang::diag::err_drv_invalid_value) << "/EH" << EHVal;
      break;
    }
  }

  // FIXME: Disable C++ EH completely, until it becomes more reliable. Users
  // can use -Xclang to manually enable C++ EH until then.
  EH = EHFlags();

  return EH;
}

void Clang::AddClangCLArgs(const ArgList &Args, ArgStringList &CmdArgs) const {
  unsigned RTOptionID = options::OPT__SLASH_MT;

  if (Args.hasArg(options::OPT__SLASH_LDd))
    // The /LDd option implies /MTd. The dependent lib part can be overridden,
    // but defining _DEBUG is sticky.
    RTOptionID = options::OPT__SLASH_MTd;

  if (Arg *A = Args.getLastArg(options::OPT__SLASH_M_Group))
    RTOptionID = A->getOption().getID();

  switch (RTOptionID) {
  case options::OPT__SLASH_MD:
    if (Args.hasArg(options::OPT__SLASH_LDd))
      CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("-D_DLL");
    CmdArgs.push_back("--dependent-lib=msvcrt");
    break;
  case options::OPT__SLASH_MDd:
    CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("-D_DLL");
    CmdArgs.push_back("--dependent-lib=msvcrtd");
    break;
  case options::OPT__SLASH_MT:
    if (Args.hasArg(options::OPT__SLASH_LDd))
      CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("--dependent-lib=libcmt");
    break;
  case options::OPT__SLASH_MTd:
    CmdArgs.push_back("-D_DEBUG");
    CmdArgs.push_back("-D_MT");
    CmdArgs.push_back("--dependent-lib=libcmtd");
    break;
  default:
    llvm_unreachable("Unexpected option ID.");
  }

  // This provides POSIX compatibility (maps 'open' to '_open'), which most
  // users want.  The /Za flag to cl.exe turns this off, but it's not
  // implemented in clang.
  CmdArgs.push_back("--dependent-lib=oldnames");

  // Both /showIncludes and /E (and /EP) write to stdout. Allowing both
  // would produce interleaved output, so ignore /showIncludes in such cases.
  if (!Args.hasArg(options::OPT_E) && !Args.hasArg(options::OPT__SLASH_EP))
    if (Arg *A = Args.getLastArg(options::OPT_show_includes))
      A->render(Args, CmdArgs);

  // This controls whether or not we emit RTTI data for polymorphic types.
  if (Args.hasFlag(options::OPT__SLASH_GR_, options::OPT__SLASH_GR,
                   /*default=*/false))
    CmdArgs.push_back("-fno-rtti-data");

  const Driver &D = getToolChain().getDriver();
  EHFlags EH = parseClangCLEHFlags(D, Args);
  // FIXME: Do something with NoExceptC.
  if (EH.Synch || EH.Asynch) {
    CmdArgs.push_back("-fcxx-exceptions");
    CmdArgs.push_back("-fexceptions");
  }

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

void ClangAs::ConstructJob(Compilation &C, const JobAction &JA,
                           const InputInfo &Output, const InputInfoList &Inputs,
                           const ArgList &Args,
                           const char *LinkingOutput) const {
  ArgStringList CmdArgs;

  assert(Inputs.size() == 1 && "Unexpected number of inputs.");
  const InputInfo &Input = Inputs[0];

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
  std::string TripleStr =
      getToolChain().ComputeEffectiveClangTriple(Args, Input.getType());
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
  const llvm::Triple Triple(TripleStr);
  std::string CPU = getCPUName(Args, Triple, /*FromAs*/ true);
  if (!CPU.empty()) {
    CmdArgs.push_back("-target-cpu");
    CmdArgs.push_back(Args.MakeArgString(CPU));
  }

  // Add the target features
  const Driver &D = getToolChain().getDriver();
  getTargetFeatures(D, Triple, Args, CmdArgs, true);

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
  if (SourceAction->getType() == types::TY_Asm ||
      SourceAction->getType() == types::TY_PP_Asm) {
    Args.ClaimAllArgs(options::OPT_g_Group);
    if (Arg *A = Args.getLastArg(options::OPT_g_Group))
      if (!A->getOption().matches(options::OPT_g0))
        CmdArgs.push_back("-g");

    if (Args.hasArg(options::OPT_gdwarf_2))
      CmdArgs.push_back("-gdwarf-2");
    if (Args.hasArg(options::OPT_gdwarf_3))
      CmdArgs.push_back("-gdwarf-3");
    if (Args.hasArg(options::OPT_gdwarf_4))
      CmdArgs.push_back("-gdwarf-4");

    // Add the -fdebug-compilation-dir flag if needed.
    addDebugCompDirArg(Args, CmdArgs);

    // Set the AT_producer to the clang version when using the integrated
    // assembler on assembly source files.
    CmdArgs.push_back("-dwarf-debug-producer");
    CmdArgs.push_back(Args.MakeArgString(getClangFullVersion()));
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
  }

  // Consume all the warning flags. Usually this would be handled more
  // gracefully by -cc1 (warning about unknown warning flags, etc) but -cc1as
  // doesn't handle that so rather than warning about unused flags that are
  // actually used, we'll lie by omission instead.
  // FIXME: Stop lying and consume only the appropriate driver flags
  for (const Arg *A : Args.filtered(options::OPT_W_Group))
    A->claim();

  CollectArgsForIntegratedAssembler(C, Args, CmdArgs,
                                    getToolChain().getDriver());

  Args.AddAllArgs(CmdArgs, options::OPT_mllvm);

  assert(Output.isFilename() && "Unexpected lipo output.");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  assert(Input.isFilename() && "Invalid input.");
  CmdArgs.push_back(Input.getFilename());

  const char *Exec = getToolChain().getDriver().getClangProgramPath();
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));

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
      // Don't forward any -g arguments to assembly steps.
      if (isa<AssembleJobAction>(JA) &&
          A->getOption().matches(options::OPT_g_Group))
        continue;

      // Don't forward any -W arguments to assembly and link steps.
      if ((isa<AssembleJobAction>(JA) || isa<LinkJobAction>(JA)) &&
          A->getOption().matches(options::OPT_W_Group))
        continue;

      // It is unfortunate that we have to claim here, as this means
      // we will basically never report anything interesting for
      // platforms using a generic gcc, even if we are just using gcc
      // to get to the assembler.
      A->claim();
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
  const llvm::Triple::ArchType Arch = getToolChain().getArch();
  if (Arch == llvm::Triple::x86 || Arch == llvm::Triple::ppc)
    CmdArgs.push_back("-m32");
  else if (Arch == llvm::Triple::x86_64 || Arch == llvm::Triple::ppc64 ||
           Arch == llvm::Triple::ppc64le)
    CmdArgs.push_back("-m64");

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
    if (II.getType() == types::TY_LLVM_IR || II.getType() == types::TY_LTO_IR ||
        II.getType() == types::TY_LLVM_BC || II.getType() == types::TY_LTO_BC)
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

  const std::string customGCCName = D.getCCCGenericGCCName();
  const char *GCCName;
  if (!customGCCName.empty())
    GCCName = customGCCName.c_str();
  else if (D.CCCIsCXX()) {
    GCCName = "g++";
  } else
    GCCName = "gcc";

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath(GCCName));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
                                             ArgStringList &CmdArgs) const {}
void hexagon::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);

  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  std::string MarchString = "-march=";
  MarchString += toolchains::Hexagon_TC::GetTargetCPU(Args);
  CmdArgs.push_back(Args.MakeArgString(MarchString));

  RenderExtraToolArgs(JA, CmdArgs);

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Unexpected output");
    CmdArgs.push_back("-fsyntax-only");
  }

  if (const char *v = toolchains::Hexagon_TC::GetSmallDataThreshold(Args))
    CmdArgs.push_back(Args.MakeArgString(std::string("-G") + v));

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
    if (II.getType() == types::TY_LLVM_IR || II.getType() == types::TY_LTO_IR ||
        II.getType() == types::TY_LLVM_BC || II.getType() == types::TY_LTO_BC)
      D.Diag(clang::diag::err_drv_no_linker_llvm_support)
          << getToolChain().getTripleString();
    else if (II.getType() == types::TY_AST)
      D.Diag(clang::diag::err_drv_no_ast_support)
          << getToolChain().getTripleString();
    else if (II.getType() == types::TY_ModuleFile)
      D.Diag(diag::err_drv_no_module_support)
          << getToolChain().getTripleString();

    if (II.isFilename())
      CmdArgs.push_back(II.getFilename());
    else
      // Don't render as input, we need gcc to do the translations.
      // FIXME: Pranav: What is this ?
      II.getInputArg().render(Args, CmdArgs);
  }

  const char *GCCName = "hexagon-as";
  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath(GCCName));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
}

void hexagon::Linker::RenderExtraToolArgs(const JobAction &JA,
                                          ArgStringList &CmdArgs) const {
  // The types are (hopefully) good enough.
}

static void constructHexagonLinkArgs(Compilation &C, const JobAction &JA,
                                     const toolchains::Hexagon_TC &ToolChain,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     ArgStringList &CmdArgs,
                                     const char *LinkingOutput) {

  const Driver &D = ToolChain.getDriver();

  //----------------------------------------------------------------------------
  //
  //----------------------------------------------------------------------------
  bool hasStaticArg = Args.hasArg(options::OPT_static);
  bool buildingLib = Args.hasArg(options::OPT_shared);
  bool buildPIE = Args.hasArg(options::OPT_pie);
  bool incStdLib = !Args.hasArg(options::OPT_nostdlib);
  bool incStartFiles = !Args.hasArg(options::OPT_nostartfiles);
  bool incDefLibs = !Args.hasArg(options::OPT_nodefaultlibs);
  bool useG0 = false;
  bool useShared = buildingLib && !hasStaticArg;

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
  for (const auto &Opt : ToolChain.ExtraOpts)
    CmdArgs.push_back(Opt.c_str());

  std::string MarchString = toolchains::Hexagon_TC::GetTargetCPU(Args);
  CmdArgs.push_back(Args.MakeArgString("-m" + MarchString));

  if (buildingLib) {
    CmdArgs.push_back("-shared");
    CmdArgs.push_back("-call_shared"); // should be the default, but doing as
                                       // hexagon-gcc does
  }

  if (hasStaticArg)
    CmdArgs.push_back("-static");

  if (buildPIE && !buildingLib)
    CmdArgs.push_back("-pie");

  if (const char *v = toolchains::Hexagon_TC::GetSmallDataThreshold(Args)) {
    CmdArgs.push_back(Args.MakeArgString(std::string("-G") + v));
    useG0 = toolchains::Hexagon_TC::UsesG0(v);
  }

  //----------------------------------------------------------------------------
  //
  //----------------------------------------------------------------------------
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  const std::string MarchSuffix = "/" + MarchString;
  const std::string G0Suffix = "/G0";
  const std::string MarchG0Suffix = MarchSuffix + G0Suffix;
  const std::string RootDir =
      toolchains::Hexagon_TC::GetGnuDir(D.InstalledDir, Args) + "/";
  const std::string StartFilesDir =
      RootDir + "hexagon/lib" + (useG0 ? MarchG0Suffix : MarchSuffix);

  //----------------------------------------------------------------------------
  // moslib
  //----------------------------------------------------------------------------
  std::vector<std::string> oslibs;
  bool hasStandalone = false;

  for (const Arg *A : Args.filtered(options::OPT_moslib_EQ)) {
    A->claim();
    oslibs.emplace_back(A->getValue());
    hasStandalone = hasStandalone || (oslibs.back() == "standalone");
  }
  if (oslibs.empty()) {
    oslibs.push_back("standalone");
    hasStandalone = true;
  }

  //----------------------------------------------------------------------------
  // Start Files
  //----------------------------------------------------------------------------
  if (incStdLib && incStartFiles) {

    if (!buildingLib) {
      if (hasStandalone) {
        CmdArgs.push_back(
            Args.MakeArgString(StartFilesDir + "/crt0_standalone.o"));
      }
      CmdArgs.push_back(Args.MakeArgString(StartFilesDir + "/crt0.o"));
    }
    std::string initObj = useShared ? "/initS.o" : "/init.o";
    CmdArgs.push_back(Args.MakeArgString(StartFilesDir + initObj));
  }

  //----------------------------------------------------------------------------
  // Library Search Paths
  //----------------------------------------------------------------------------
  const ToolChain::path_list &LibPaths = ToolChain.getFilePaths();
  for (const auto &LibPath : LibPaths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + LibPath));

  //----------------------------------------------------------------------------
  //
  //----------------------------------------------------------------------------
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_u_Group);

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  //----------------------------------------------------------------------------
  // Libraries
  //----------------------------------------------------------------------------
  if (incStdLib && incDefLibs) {
    if (D.CCCIsCXX()) {
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }

    CmdArgs.push_back("--start-group");

    if (!buildingLib) {
      for (const std::string &Lib : oslibs)
        CmdArgs.push_back(Args.MakeArgString("-l" + Lib));
      CmdArgs.push_back("-lc");
    }
    CmdArgs.push_back("-lgcc");

    CmdArgs.push_back("--end-group");
  }

  //----------------------------------------------------------------------------
  // End files
  //----------------------------------------------------------------------------
  if (incStdLib && incStartFiles) {
    std::string finiObj = useShared ? "/finiS.o" : "/fini.o";
    CmdArgs.push_back(Args.MakeArgString(StartFilesDir + finiObj));
  }
}

void hexagon::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {

  const toolchains::Hexagon_TC &ToolChain =
      static_cast<const toolchains::Hexagon_TC &>(getToolChain());

  ArgStringList CmdArgs;
  constructHexagonLinkArgs(C, JA, ToolChain, Output, Inputs, Args, CmdArgs,
                           LinkingOutput);

  std::string Linker = ToolChain.GetProgramPath("hexagon-ld");
  C.addCommand(llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Linker),
                                          CmdArgs));
}
// Hexagon tools end.

const std::string arm::getARMArch(StringRef Arch, const llvm::Triple &Triple) {
  std::string MArch;
  if (!Arch.empty())
    MArch = Arch;
  else
    MArch = Triple.getArchName();
  MArch = StringRef(MArch).lower();

  // Handle -march=native.
  if (MArch == "native") {
    std::string CPU = llvm::sys::getHostCPUName();
    if (CPU != "generic") {
      // Translate the native cpu into the architecture suffix for that CPU.
      const char *Suffix = arm::getLLVMArchSuffixForARM(CPU, MArch);
      // If there is no valid architecture suffix for this CPU we don't know how
      // to handle it, so return no architecture.
      if (strcmp(Suffix, "") == 0)
        MArch = "";
      else
        MArch = std::string("arm") + Suffix;
    }
  }

  return MArch;
}
/// Get the (LLVM) name of the minimum ARM CPU for the arch we are targeting.
const char *arm::getARMCPUForMArch(StringRef Arch, const llvm::Triple &Triple) {
  std::string MArch = getARMArch(Arch, Triple);
  // getARMCPUForArch defaults to the triple if MArch is empty, but empty MArch
  // here means an -march=native that we can't handle, so instead return no CPU.
  if (MArch.empty())
    return "";

  // We need to return an empty string here on invalid MArch values as the
  // various places that call this function can't cope with a null result.
  const char *result = Triple.getARMCPUForArch(MArch);
  if (result)
    return result;
  else
    return "";
}

/// getARMTargetCPU - Get the (LLVM) name of the ARM cpu we are targeting.
std::string arm::getARMTargetCPU(StringRef CPU, StringRef Arch,
                                 const llvm::Triple &Triple) {
  // FIXME: Warn on inconsistent use of -mcpu and -march.
  // If we have -mcpu=, use that.
  if (!CPU.empty()) {
    std::string MCPU = StringRef(CPU).lower();
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
const char *arm::getLLVMArchSuffixForARM(StringRef CPU, StringRef Arch) {
  if (CPU == "generic" &&
      llvm::ARMTargetParser::parseArch(Arch) == llvm::ARM::AK_ARMV8_1A)
    return "v8.1a";

  unsigned ArchKind = llvm::ARMTargetParser::parseCPUArch(CPU);
  if (ArchKind == llvm::ARM::AK_INVALID)
    return "";
  return llvm::ARMTargetParser::getSubArch(ArchKind);
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
  return (NanEncoding)llvm::StringSwitch<int>(CPU)
      .Case("mips1", NanLegacy)
      .Case("mips2", NanLegacy)
      .Case("mips3", NanLegacy)
      .Case("mips4", NanLegacy)
      .Case("mips5", NanLegacy)
      .Case("mips32", NanLegacy)
      .Case("mips32r2", NanLegacy)
      .Case("mips32r3", NanLegacy | Nan2008)
      .Case("mips32r5", NanLegacy | Nan2008)
      .Case("mips32r6", Nan2008)
      .Case("mips64", NanLegacy)
      .Case("mips64r2", NanLegacy)
      .Case("mips64r3", NanLegacy | Nan2008)
      .Case("mips64r5", NanLegacy | Nan2008)
      .Case("mips64r6", Nan2008)
      .Default(NanLegacy);
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

bool mips::isFPXXDefault(const llvm::Triple &Triple, StringRef CPUName,
                         StringRef ABIName, StringRef FloatABI) {
  if (Triple.getVendor() != llvm::Triple::ImaginationTechnologies &&
      Triple.getVendor() != llvm::Triple::MipsTechnologies)
    return false;

  if (ABIName != "32")
    return false;

  // FPXX shouldn't be used if either -msoft-float or -mfloat-abi=soft is
  // present.
  if (FloatABI == "soft")
    return false;

  return llvm::StringSwitch<bool>(CPUName)
      .Cases("mips2", "mips3", "mips4", "mips5", true)
      .Cases("mips32", "mips32r2", "mips32r3", "mips32r5", true)
      .Cases("mips64", "mips64r2", "mips64r3", "mips64r5", true)
      .Default(false);
}

bool mips::shouldUseFPXX(const ArgList &Args, const llvm::Triple &Triple,
                         StringRef CPUName, StringRef ABIName,
                         StringRef FloatABI) {
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
  T.setArch(Arch);

  if (Str == "x86_64h")
    T.setArchName(Str);
  else if (Str == "armv6m" || Str == "armv7m" || Str == "armv7em") {
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
  CmdArgs.push_back("--eh-frame-hdr");
  CmdArgs.push_back("--gc-sections");

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtbegin.o")));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  const ToolChain::path_list &Paths = ToolChain.getFilePaths();
  for (const auto &Path : Paths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + Path));
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_Z_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  if (D.IsUsingLTO(Args))
    AddGoldPlugin(ToolChain, Args, CmdArgs);

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX())
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lcompiler_rt");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles))
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtend.o")));

  const char *Exec = Args.MakeArgString(ToolChain.GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

  // If -fno_integrated_as is used add -Q to the darwin assember driver to make
  // sure it runs its system assembler not clang's integrated assembler.
  // Applicable to darwin11+ and Xcode 4+.  darwin<10 lacked integrated-as.
  // FIXME: at run-time detect assembler capabilities or rely on version
  // information forwarded by -target-assembler-version (future)
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

  unsigned Version[3] = {0, 0, 0};
  if (Arg *A = Args.getLastArg(options::OPT_mlinker_version_EQ)) {
    bool HadExtra;
    if (!Driver::GetReleaseVersion(A->getValue(), Version[0], Version[1],
                                   Version[2], HadExtra) ||
        HadExtra)
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

  // If we are using LTO, then automatically create a temporary file path for
  // the linker to use, so that it's lifetime will extend past a possible
  // dsymutil step.
  if (Version[0] >= 116 && D.IsUsingLTO(Args) && NeedsTempPath(Inputs)) {
    const char *TmpPath = C.getArgs().MakeArgString(
        D.GetTemporaryPath("cc", types::getTypeTempSuffix(types::TY_Object)));
    C.addTempFile(TmpPath);
    CmdArgs.push_back("-object_path_lto");
    CmdArgs.push_back(TmpPath);
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
    C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
    return;
  }

  // I'm not sure why this particular decomposition exists in gcc, but
  // we follow suite for ease of comparison.
  AddLinkArgs(C, Args, CmdArgs, Inputs);

  Args.AddAllArgs(CmdArgs, options::OPT_d_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_Z_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_u_Group);
  Args.AddLastArg(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  // Forward -ObjC when either -ObjC or -ObjC++ is used, to force loading
  // members of static archive libraries which implement Objective-C classes or
  // categories.
  if (Args.hasArg(options::OPT_ObjC) || Args.hasArg(options::OPT_ObjCXX))
    CmdArgs.push_back("-ObjC");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles))
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs))
    addOpenMPRuntime(CmdArgs, getToolChain(), Args);

  if (isObjCRuntimeLinked(Args) && !Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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

  // TODO: It would be nice to use addProfileRT() here, but darwin's compiler-rt
  // paths are different enough from other toolchains that this needs a fair
  // amount of refactoring done first.
  getMachOToolChain().addProfileRTLibs(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (getToolChain().getDriver().CCCIsCXX())
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);

    // link_ssp spec is empty.

    // Let the tool chain choose which runtime library to link.
    getMachOToolChain().AddLinkRuntimeLibArgs(Args, CmdArgs);
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    // endfile_spec is empty.
  }

  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_F);

  // -iframework should be forwarded as -F.
  for (const Arg *A : Args.filtered(options::OPT_iframework))
    CmdArgs.push_back(Args.MakeArgString(std::string("-F") + A->getValue()));

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (Arg *A = Args.getLastArg(options::OPT_fveclib)) {
      if (A->getValue() == StringRef("Accelerate")) {
        CmdArgs.push_back("-framework");
        CmdArgs.push_back("Accelerate");
      }
    }
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  std::unique_ptr<Command> Cmd =
      llvm::make_unique<Command>(JA, *this, Exec, CmdArgs);
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
}

void solaris::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                   const InputInfo &Output,
                                   const InputInfoList &Inputs,
                                   const ArgList &Args,
                                   const char *LinkingOutput) const {
  // FIXME: Find a real GCC, don't hard-code versions here
  std::string GCCLibPath = "/usr/gcc/4.5/lib/gcc/";
  const llvm::Triple &T = getToolChain().getTriple();
  std::string LibPath = "/usr/lib/";
  const llvm::Triple::ArchType Arch = T.getArch();
  switch (Arch) {
  case llvm::Triple::x86:
    GCCLibPath +=
        ("i386-" + T.getVendorName() + "-" + T.getOSName()).str() + "/4.5.2/";
    break;
  case llvm::Triple::x86_64:
    GCCLibPath += ("i386-" + T.getVendorName() + "-" + T.getOSName()).str();
    GCCLibPath += "/4.5.2/amd64/";
    LibPath += "amd64/";
    break;
  default:
    llvm_unreachable("Unsupported architecture");
  }

  ArgStringList CmdArgs;

  // Demangle C++ names in errors
  CmdArgs.push_back("-C");

  if ((!Args.hasArg(options::OPT_nostdlib)) &&
      (!Args.hasArg(options::OPT_shared))) {
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
      CmdArgs.push_back(Args.MakeArgString(LibPath + "ld.so.1"));
    }
  }

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(Args.MakeArgString(LibPath + "crt1.o"));
      CmdArgs.push_back(Args.MakeArgString(LibPath + "crti.o"));
      CmdArgs.push_back(Args.MakeArgString(LibPath + "values-Xa.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "crtbegin.o"));
    } else {
      CmdArgs.push_back(Args.MakeArgString(LibPath + "crti.o"));
      CmdArgs.push_back(Args.MakeArgString(LibPath + "values-Xa.o"));
      CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "crtbegin.o"));
    }
    if (getToolChain().getDriver().CCCIsCXX())
      CmdArgs.push_back(Args.MakeArgString(LibPath + "cxa_finalize.o"));
  }

  CmdArgs.push_back(Args.MakeArgString("-L" + GCCLibPath));

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (getToolChain().getDriver().CCCIsCXX())
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
    CmdArgs.push_back("-lgcc_s");
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back("-lgcc");
      CmdArgs.push_back("-lc");
      CmdArgs.push_back("-lm");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    CmdArgs.push_back(Args.MakeArgString(GCCLibPath + "crtend.o"));
  }
  CmdArgs.push_back(Args.MakeArgString(LibPath + "crtn.o"));

  addProfileRT(getToolChain(), Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
}

void openbsd::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                      const InputInfo &Output,
                                      const InputInfoList &Inputs,
                                      const ArgList &Args,
                                      const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;
  bool NeedsKPIC = false;

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
  case llvm::Triple::sparcel:
    CmdArgs.push_back("-32");
    NeedsKPIC = true;
    break;

  case llvm::Triple::sparcv9:
    CmdArgs.push_back("-64");
    CmdArgs.push_back("-Av9a");
    NeedsKPIC = true;
    break;

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

    NeedsKPIC = true;
    break;
  }

  default:
    break;
  }

  if (NeedsKPIC)
    addAssemblerKPIC(Args, CmdArgs);

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

  if ((!Args.hasArg(options::OPT_nostdlib)) &&
      (!Args.hasArg(options::OPT_shared))) {
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
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

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_Z_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
}

void bitrig::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  if ((!Args.hasArg(options::OPT_nostdlib)) &&
      (!Args.hasArg(options::OPT_shared))) {
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
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

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  if (getToolChain().getArch() == llvm::Triple::x86)
    CmdArgs.push_back("--32");
  else if (getToolChain().getArch() == llvm::Triple::ppc)
    CmdArgs.push_back("-a32");
  else if (getToolChain().getArch() == llvm::Triple::mips ||
           getToolChain().getArch() == llvm::Triple::mipsel ||
           getToolChain().getArch() == llvm::Triple::mips64 ||
           getToolChain().getArch() == llvm::Triple::mips64el) {
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

    addAssemblerKPIC(Args, CmdArgs);
  } else if (getToolChain().getArch() == llvm::Triple::arm ||
             getToolChain().getArch() == llvm::Triple::armeb ||
             getToolChain().getArch() == llvm::Triple::thumb ||
             getToolChain().getArch() == llvm::Triple::thumbeb) {
    const Driver &D = getToolChain().getDriver();
    const llvm::Triple &Triple = getToolChain().getTriple();
    StringRef FloatABI = arm::getARMFloatABI(D, Args, Triple);

    if (FloatABI == "hard") {
      CmdArgs.push_back("-mfpu=vfp");
    } else {
      CmdArgs.push_back("-mfpu=softvfp");
    }

    switch (getToolChain().getTriple().getEnvironment()) {
    case llvm::Triple::GNUEABIHF:
    case llvm::Triple::GNUEABI:
    case llvm::Triple::EABI:
      CmdArgs.push_back("-meabi=5");
      break;

    default:
      CmdArgs.push_back("-matpcs");
    }
  } else if (getToolChain().getArch() == llvm::Triple::sparc ||
             getToolChain().getArch() == llvm::Triple::sparcel ||
             getToolChain().getArch() == llvm::Triple::sparcv9) {
    if (getToolChain().getArch() == llvm::Triple::sparc)
      CmdArgs.push_back("-Av8plusa");
    else
      CmdArgs.push_back("-Av9a");

    addAssemblerKPIC(Args, CmdArgs);
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
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
  const ToolChain::path_list &Paths = ToolChain.getFilePaths();
  for (const auto &Path : Paths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + Path));
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);
  Args.AddAllArgs(CmdArgs, options::OPT_s);
  Args.AddAllArgs(CmdArgs, options::OPT_t);
  Args.AddAllArgs(CmdArgs, options::OPT_Z_Flag);
  Args.AddAllArgs(CmdArgs, options::OPT_r);

  if (D.IsUsingLTO(Args))
    AddGoldPlugin(ToolChain, Args, CmdArgs);

  bool NeedsSanitizerDeps = addSanitizerRuntimes(ToolChain, Args, CmdArgs);
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || IsPIE)
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtendS.o")));
    else
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtend.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));
  }

  addProfileRT(ToolChain, Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

    addAssemblerKPIC(Args, CmdArgs);
    break;
  }

  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
    CmdArgs.push_back("-32");
    addAssemblerKPIC(Args, CmdArgs);
    break;

  case llvm::Triple::sparcv9:
    CmdArgs.push_back("-64");
    CmdArgs.push_back("-Av9");
    addAssemblerKPIC(Args, CmdArgs);
    break;

  default:
    break;
  }

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString((getToolChain().GetProgramPath("as")));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
    arm::appendEBLinkFlags(
        Args, CmdArgs,
        llvm::Triple(getToolChain().ComputeEffectiveClangTriple(Args)));
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared)) {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crt0.o")));
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    } else {
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtbeginS.o")));
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
  if (Major >= 7 || (Major == 6 && Minor == 99 && Micro >= 49) || Major == 0) {
    switch (getToolChain().getArch()) {
    case llvm::Triple::aarch64:
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
    case llvm::Triple::ppc64le:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      useLibgcc = false;
      break;
    default:
      break;
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!Args.hasArg(options::OPT_shared))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));
  }

  addProfileRT(getToolChain(), Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
}

void gnutools::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const InputInfoList &Inputs,
                                       const ArgList &Args,
                                       const char *LinkingOutput) const {
  claimNoWarnArgs(Args);

  ArgStringList CmdArgs;
  bool NeedsKPIC = false;

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
  case llvm::Triple::sparcel:
    CmdArgs.push_back("-32");
    CmdArgs.push_back("-Av8plusa");
    NeedsKPIC = true;
    break;
  case llvm::Triple::sparcv9:
    CmdArgs.push_back("-64");
    CmdArgs.push_back("-Av9a");
    NeedsKPIC = true;
    break;
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb: {
    const llvm::Triple &Triple = getToolChain().getTriple();
    switch (Triple.getSubArch()) {
    case llvm::Triple::ARMSubArch_v7:
      CmdArgs.push_back("-mfpu=neon");
      break;
    case llvm::Triple::ARMSubArch_v8:
      CmdArgs.push_back("-mfpu=crypto-neon-fp-armv8");
      break;
    default:
      break;
    }

    StringRef ARMFloatABI = tools::arm::getARMFloatABI(
        getToolChain().getDriver(), Args,
        llvm::Triple(getToolChain().ComputeEffectiveClangTriple(Args)));
    CmdArgs.push_back(Args.MakeArgString("-mfloat-abi=" + ARMFloatABI));

    Args.AddLastArg(CmdArgs, options::OPT_march_EQ);

    // FIXME: remove krait check when GNU tools support krait cpu
    // for now replace it with -march=armv7-a  to avoid a lower
    // march from being picked in the absence of a cpu flag.
    Arg *A;
    if ((A = Args.getLastArg(options::OPT_mcpu_EQ)) &&
        StringRef(A->getValue()).lower() == "krait")
      CmdArgs.push_back("-march=armv7-a");
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
    bool IsPicOrPie = false;
    if (Arg *A = Args.getLastArg(options::OPT_fPIC, options::OPT_fno_PIC,
                                 options::OPT_fpic, options::OPT_fno_pic,
                                 options::OPT_fPIE, options::OPT_fno_PIE,
                                 options::OPT_fpie, options::OPT_fno_pie)) {
      if (A->getOption().matches(options::OPT_fPIC) ||
          A->getOption().matches(options::OPT_fpic) ||
          A->getOption().matches(options::OPT_fPIE) ||
          A->getOption().matches(options::OPT_fpie))
        IsPicOrPie = true;
    }
    if (!IsPicOrPie)
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
    StringRef MIPSFloatABI = getMipsFloatABI(getToolChain().getDriver(), Args);
    if (Arg *A = Args.getLastArg(options::OPT_mfp32, options::OPT_mfpxx,
                                 options::OPT_mfp64)) {
      A->claim();
      A->render(Args, CmdArgs);
    } else if (mips::shouldUseFPXX(Args, getToolChain().getTriple(), CPUName,
                                   ABIName, MIPSFloatABI))
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

    NeedsKPIC = true;
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

  if (NeedsKPIC)
    addAssemblerKPIC(Args, CmdArgs);

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));

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
  bool isAndroid = Triple.getEnvironment() == llvm::Triple::Android;
  bool isCygMing = Triple.isOSCygMing();
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

  if (StaticLibgcc && !isAndroid)
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

static std::string getLinuxDynamicLinker(const ArgList &Args,
                                         const toolchains::Linux &ToolChain) {
  const llvm::Triple::ArchType Arch = ToolChain.getArch();

  if (ToolChain.getTriple().getEnvironment() == llvm::Triple::Android) {
    if (ToolChain.getTriple().isArch64Bit())
      return "/system/bin/linker64";
    else
      return "/system/bin/linker";
  } else if (Arch == llvm::Triple::x86 || Arch == llvm::Triple::sparc ||
             Arch == llvm::Triple::sparcel)
    return "/lib/ld-linux.so.2";
  else if (Arch == llvm::Triple::aarch64)
    return "/lib/ld-linux-aarch64.so.1";
  else if (Arch == llvm::Triple::aarch64_be)
    return "/lib/ld-linux-aarch64_be.so.1";
  else if (Arch == llvm::Triple::arm || Arch == llvm::Triple::thumb) {
    if (ToolChain.getTriple().getEnvironment() == llvm::Triple::GNUEABIHF ||
        tools::arm::getARMFloatABI(ToolChain.getDriver(), Args, ToolChain.getTriple()) == "hard")
      return "/lib/ld-linux-armhf.so.3";
    else
      return "/lib/ld-linux.so.3";
  } else if (Arch == llvm::Triple::armeb || Arch == llvm::Triple::thumbeb) {
    // TODO: check which dynamic linker name.
    if (ToolChain.getTriple().getEnvironment() == llvm::Triple::GNUEABIHF ||
        tools::arm::getARMFloatABI(ToolChain.getDriver(), Args, ToolChain.getTriple()) == "hard")
      return "/lib/ld-linux-armhf.so.3";
    else
      return "/lib/ld-linux.so.3";
  } else if (Arch == llvm::Triple::mips || Arch == llvm::Triple::mipsel ||
             Arch == llvm::Triple::mips64 || Arch == llvm::Triple::mips64el) {
    StringRef CPUName;
    StringRef ABIName;
    mips::getMipsCPUAndABI(Args, ToolChain.getTriple(), CPUName, ABIName);
    bool IsNaN2008 = mips::isNaN2008(Args, ToolChain.getTriple());

    StringRef LibDir = llvm::StringSwitch<llvm::StringRef>(ABIName)
                           .Case("o32", "/lib")
                           .Case("n32", "/lib32")
                           .Case("n64", "/lib64")
                           .Default("/lib");
    StringRef LibName;
    if (mips::isUCLibc(Args))
      LibName = IsNaN2008 ? "ld-uClibc-mipsn8.so.0" : "ld-uClibc.so.0";
    else
      LibName = IsNaN2008 ? "ld-linux-mipsn8.so.1" : "ld.so.1";

    return (LibDir + "/" + LibName).str();
  } else if (Arch == llvm::Triple::ppc)
    return "/lib/ld.so.1";
  else if (Arch == llvm::Triple::ppc64) {
    if (ppc::hasPPCAbiArg(Args, "elfv2"))
      return "/lib64/ld64.so.2";
    return "/lib64/ld64.so.1";
  } else if (Arch == llvm::Triple::ppc64le) {
    if (ppc::hasPPCAbiArg(Args, "elfv1"))
      return "/lib64/ld64.so.1";
    return "/lib64/ld64.so.2";
  } else if (Arch == llvm::Triple::systemz)
    return "/lib64/ld64.so.1";
  else if (Arch == llvm::Triple::sparcv9)
    return "/lib64/ld-linux.so.2";
  else if (Arch == llvm::Triple::x86_64 &&
           ToolChain.getTriple().getEnvironment() == llvm::Triple::GNUX32)
    return "/libx32/ld-linux-x32.so.2";
  else
    return "/lib64/ld-linux-x86-64.so.2";
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
    AddLibgcc(TC.getTriple(), D, CmdArgs, Args);
    break;
  }
}

static const char *getLDMOption(const llvm::Triple &T, const ArgList &Args) {
  switch (T.getArch()) {
  case llvm::Triple::x86:
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
    return "armebelf_linux_eabi"; /* TODO: check which NAME.  */
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
  const llvm::Triple::ArchType Arch = ToolChain.getArch();
  const bool isAndroid =
      ToolChain.getTriple().getEnvironment() == llvm::Triple::Android;
  const bool IsPIE =
      !Args.hasArg(options::OPT_shared) && !Args.hasArg(options::OPT_static) &&
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

  if (Args.hasArg(options::OPT_rdynamic))
    CmdArgs.push_back("-export-dynamic");

  if (Args.hasArg(options::OPT_s))
    CmdArgs.push_back("-s");

  if (Arch == llvm::Triple::armeb || Arch == llvm::Triple::thumbeb)
    arm::appendEBLinkFlags(
        Args, CmdArgs,
        llvm::Triple(getToolChain().ComputeEffectiveClangTriple(Args)));

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

  if (Arch == llvm::Triple::arm || Arch == llvm::Triple::armeb ||
      Arch == llvm::Triple::thumb || Arch == llvm::Triple::thumbeb ||
      (!Args.hasArg(options::OPT_static) &&
       !Args.hasArg(options::OPT_shared))) {
    CmdArgs.push_back("-dynamic-linker");
    CmdArgs.push_back(Args.MakeArgString(
        D.DyldPrefix + getLinuxDynamicLinker(Args, ToolChain)));
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (!isAndroid) {
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

    const char *crtbegin;
    if (Args.hasArg(options::OPT_static))
      crtbegin = isAndroid ? "crtbegin_static.o" : "crtbeginT.o";
    else if (Args.hasArg(options::OPT_shared))
      crtbegin = isAndroid ? "crtbegin_so.o" : "crtbeginS.o";
    else if (IsPIE)
      crtbegin = isAndroid ? "crtbegin_dynamic.o" : "crtbeginS.o";
    else
      crtbegin = isAndroid ? "crtbegin_dynamic.o" : "crtbegin.o";
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));

    // Add crtfastmath.o if available and fast math is enabled.
    ToolChain.AddFastMathRuntimeIfAvailable(Args, CmdArgs);
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);

  const ToolChain::path_list &Paths = ToolChain.getFilePaths();

  for (const auto &Path : Paths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + Path));

  if (D.IsUsingLTO(Args))
    AddGoldPlugin(ToolChain, Args, CmdArgs);

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  bool NeedsSanitizerDeps = addSanitizerRuntimes(ToolChain, Args, CmdArgs);
  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);
  // The profile runtime also needs access to system libraries.
  addProfileRT(getToolChain(), Args, CmdArgs);

  if (D.CCCIsCXX() && !Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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

      CmdArgs.push_back("-lc");

      if (Args.hasArg(options::OPT_static))
        CmdArgs.push_back("--end-group");
      else
        AddRunTimeLibs(ToolChain, D, CmdArgs, Args);
    }

    if (!Args.hasArg(options::OPT_nostartfiles)) {
      const char *crtend;
      if (Args.hasArg(options::OPT_shared))
        crtend = isAndroid ? "crtend_so.o" : "crtendS.o";
      else if (IsPIE)
        crtend = isAndroid ? "crtend_android.o" : "crtendS.o";
      else
        crtend = isAndroid ? "crtend_android.o" : "crtend.o";

      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtend)));
      if (!isAndroid)
        CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crtn.o")));
    }
  }

  C.addCommand(
      llvm::make_unique<Command>(JA, *this, ToolChain.Linker.c_str(), CmdArgs));
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
  const toolchains::NaCl_TC &ToolChain =
      static_cast<const toolchains::NaCl_TC &>(getToolChain());
  InputInfo NaClMacros(ToolChain.GetNaClArmMacrosPath(), types::TY_PP_Asm,
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

  const toolchains::NaCl_TC &ToolChain =
      static_cast<const toolchains::NaCl_TC &>(getToolChain());
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

  // NaCl_TC doesn't have ExtraOpts like Linux; the only relevant flag from
  // there is --build-id, which we do want.
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
  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
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

  const ToolChain::path_list &Paths = ToolChain.getFilePaths();

  for (const auto &Path : Paths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + Path));

  if (Args.hasArg(options::OPT_Z_Xlinker__no_demangle))
    CmdArgs.push_back("--no-demangle");

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs);

  if (D.CCCIsCXX() && !Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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

  C.addCommand(
      llvm::make_unique<Command>(JA, *this, ToolChain.Linker.c_str(), CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crt1.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  addProfileRT(getToolChain(), Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (D.CCCIsCXX()) {
      getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lCompilerRT-Generic");
    CmdArgs.push_back("-L/usr/pkg/compiler-rt/lib");
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
}

void dragonfly::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;
  bool UseGCC47 = llvm::sys::fs::exists("/usr/lib/gcc47");

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
    CmdArgs.push_back("--hash-style=both");
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
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

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_T_Group);
  Args.AddAllArgs(CmdArgs, options::OPT_e);

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    // FIXME: GCC passes on -lgcc, -lgcc_pic and a whole lot of
    //         rpaths
    if (UseGCC47)
      CmdArgs.push_back("-L/usr/lib/gcc47");
    else
      CmdArgs.push_back("-L/usr/lib/gcc44");

    if (!Args.hasArg(options::OPT_static)) {
      if (UseGCC47) {
        CmdArgs.push_back("-rpath");
        CmdArgs.push_back("/usr/lib/gcc47");
      } else {
        CmdArgs.push_back("-rpath");
        CmdArgs.push_back("/usr/lib/gcc44");
      }
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

    if (UseGCC47) {
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
    } else {
      if (Args.hasArg(options::OPT_shared)) {
        CmdArgs.push_back("-lgcc_pic");
      } else {
        CmdArgs.push_back("-lgcc");
      }
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    if (Args.hasArg(options::OPT_shared) || Args.hasArg(options::OPT_pie))
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtendS.o")));
    else
      CmdArgs.push_back(
          Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));
  }

  addProfileRT(getToolChain(), Args, CmdArgs);

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles) && !C.getDriver().IsCLMode())
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
    }

    std::string WindowsSdkLibPath;
    if (MSVC.getWindowsSDKLibraryPath(WindowsSdkLibPath))
      CmdArgs.push_back(Args.MakeArgString(std::string("-libpath:") +
                                           WindowsSdkLibPath.c_str()));
  }

  CmdArgs.push_back("-nologo");

  if (Args.hasArg(options::OPT_g_Group))
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
      static const char *CompilerRTComponents[] = {
          "asan_dynamic", "asan_dynamic_runtime_thunk",
      };
      for (const auto &Component : CompilerRTComponents)
        CmdArgs.push_back(Args.MakeArgString(getCompilerRT(TC, Component)));
      // Make sure the dynamic runtime thunk is not optimized out at link time
      // to ensure proper SEH handling.
      CmdArgs.push_back(Args.MakeArgString("-include:___asan_seh_interceptor"));
    } else if (DLL) {
      CmdArgs.push_back(
          Args.MakeArgString(getCompilerRT(TC, "asan_dll_thunk")));
    } else {
      static const char *CompilerRTComponents[] = {
          "asan", "asan_cxx",
      };
      for (const auto &Component : CompilerRTComponents)
        CmdArgs.push_back(Args.MakeArgString(getCompilerRT(TC, Component)));
    }
  }

  Args.AddAllArgValues(CmdArgs, options::OPT__SLASH_link);

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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  Args.AddAllArgs(CmdArgs, options::OPT_D, options::OPT_U);
  Args.AddAllArgs(CmdArgs, options::OPT_I);

  // Optimization level.
  if (Arg *A = Args.getLastArg(options::OPT_O, options::OPT_O0)) {
    if (A->getOption().getID() == options::OPT_O0) {
      CmdArgs.push_back("/Od");
    } else {
      StringRef OptLevel = A->getValue();
      if (OptLevel == "1" || OptLevel == "2" || OptLevel == "s")
        A->render(Args, CmdArgs);
      else if (OptLevel == "3")
        CmdArgs.push_back("/Ox");
    }
  }

  // Flags for which clang-cl has an alias.
  // FIXME: How can we ensure this stays in sync with relevant clang-cl options?

  if (Args.hasFlag(options::OPT__SLASH_GR_, options::OPT__SLASH_GR,
                   /*default=*/false))
    CmdArgs.push_back("/GR-");
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
  if (Args.hasArg(options::OPT_g_Flag, options::OPT_gline_tables_only))
    CmdArgs.push_back("/Z7");

  std::vector<std::string> Includes =
      Args.getAllArgValues(options::OPT_include);
  for (const auto &Include : Includes)
    CmdArgs.push_back(Args.MakeArgString(std::string("/FI") + Include));

  // Flags that can simply be passed through.
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_LD);
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_LDd);
  Args.AddAllArgs(CmdArgs, options::OPT__SLASH_EH);

  // The order of these flags is relevant, so pick the last one.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_MD, options::OPT__SLASH_MDd,
                               options::OPT__SLASH_MT, options::OPT__SLASH_MTd))
    A->render(Args, CmdArgs);

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
                                    CmdArgs);
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));

  if (Args.hasArg(options::OPT_gsplit_dwarf))
    SplitDebugInfo(getToolChain(), C, *this, JA, Args, Output,
                   SplitDebugName(Args, Inputs[0]));
}

void MinGW::Linker::AddLibGCC(const ArgList &Args,
                              ArgStringList &CmdArgs) const {
  if (Args.hasArg(options::OPT_mthreads))
    CmdArgs.push_back("-lmingwthrd");
  CmdArgs.push_back("-lmingw32");

  // Add libgcc or compiler-rt.
  AddRunTimeLibs(getToolChain(), getToolChain().getDriver(), CmdArgs, Args);

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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
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
  const ToolChain::path_list Paths = TC.getFilePaths();
  for (const auto &Path : Paths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + Path));

  AddLinkerInputs(TC, Inputs, Args, CmdArgs);

  // TODO: Add ASan stuff here

  // TODO: Add profile stuff here

  if (D.CCCIsCXX() && !Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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
  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
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

    if (!Args.hasArg(options::OPT_nostdlib) &&
        !Args.hasArg(options::OPT_nostartfiles)) {
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

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nostartfiles)) {
    const std::string CRTPath(D.SysRoot + "/usr/lib/");
    const char *CRTBegin;

    CRTBegin =
        Args.hasArg(options::OPT_shared) ? "crtbeginS.obj" : "crtbegin.obj";
    CmdArgs.push_back(Args.MakeArgString(CRTPath + CRTBegin));
  }

  Args.AddAllArgs(CmdArgs, options::OPT_L);

  const auto &Paths = TC.getFilePaths();
  for (const auto &Path : Paths)
    CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + Path));

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

  const std::string Linker = TC.GetProgramPath("ld");
  Exec = Args.MakeArgString(Linker);

  C.addCommand(llvm::make_unique<Command>(JA, *this, Exec, CmdArgs));
}

void tools::SHAVE::Compiler::ConstructJob(Compilation &C, const JobAction &JA,
                                          const InputInfo &Output,
                                          const InputInfoList &Inputs,
                                          const ArgList &Args,
                                          const char *LinkingOutput) const {

  ArgStringList CmdArgs;

  assert(Inputs.size() == 1);
  const InputInfo &II = Inputs[0];
  assert(II.getType() == types::TY_C || II.getType() == types::TY_CXX);
  assert(Output.getType() == types::TY_PP_Asm); // Require preprocessed asm.

  // Append all -I, -iquote, -isystem paths.
  Args.AddAllArgs(CmdArgs, options::OPT_clang_i_Group);
  // These are spelled the same way in clang and moviCompile.
  Args.AddAllArgs(CmdArgs, options::OPT_D, options::OPT_U);

  CmdArgs.push_back("-DMYRIAD2");
  CmdArgs.push_back("-mcpu=myriad2");
  CmdArgs.push_back("-S");

  // Any -O option passes through without translation. What about -Ofast ?
  if (Arg *A = Args.getLastArg(options::OPT_O_Group))
    A->render(Args, CmdArgs);

  if (Args.hasFlag(options::OPT_ffunction_sections,
                   options::OPT_fno_function_sections)) {
    CmdArgs.push_back("-ffunction-sections");
  }
  if (Args.hasArg(options::OPT_fno_inline_functions))
    CmdArgs.push_back("-fno-inline-functions");

  CmdArgs.push_back("-fno-exceptions"); // Always do this even if unspecified.

  CmdArgs.push_back(II.getFilename());
  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  std::string Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("moviCompile"));
  C.addCommand(
      llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Exec), CmdArgs));
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
  CmdArgs.push_back("-cv:myriad2"); // Chip Version ?
  CmdArgs.push_back("-noSPrefixing");
  CmdArgs.push_back("-a"); // Mystery option.
  for (auto Arg : Args.filtered(options::OPT_I)) {
    Arg->claim();
    CmdArgs.push_back(
        Args.MakeArgString(std::string("-i:") + Arg->getValue(0)));
  }
  CmdArgs.push_back("-elf"); // Output format.
  CmdArgs.push_back(II.getFilename());
  CmdArgs.push_back(
      Args.MakeArgString(std::string("-o:") + Output.getFilename()));

  std::string Exec =
      Args.MakeArgString(getToolChain().GetProgramPath("moviAsm"));
  C.addCommand(
      llvm::make_unique<Command>(JA, *this, Args.MakeArgString(Exec), CmdArgs));
}
