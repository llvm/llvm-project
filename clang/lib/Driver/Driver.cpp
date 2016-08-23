//===--- Driver.cpp - Clang GCC Compatible Driver -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"
#include "InputInfo.h"
#include "ToolChains.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Config/config.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptSpecifier.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <memory>
#include <utility>

using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

Driver::Driver(StringRef ClangExecutable, StringRef DefaultTargetTriple,
               DiagnosticsEngine &Diags,
               IntrusiveRefCntPtr<vfs::FileSystem> VFS)
    : Opts(createDriverOptTable()), Diags(Diags), VFS(std::move(VFS)),
      Mode(GCCMode), SaveTemps(SaveTempsNone), BitcodeEmbed(EmbedNone),
      LTOMode(LTOK_None), ClangExecutable(ClangExecutable),
      SysRoot(DEFAULT_SYSROOT), UseStdLib(true),
      DriverTitle("clang LLVM compiler"), CCPrintOptionsFilename(nullptr),
      CCPrintHeadersFilename(nullptr), CCLogDiagnosticsFilename(nullptr),
      CCCPrintBindings(false), CCPrintHeaders(false), CCLogDiagnostics(false),
      CCGenDiagnostics(false), DefaultTargetTriple(DefaultTargetTriple),
      CCCGenericGCCName(""), CheckInputsExist(true), CCCUsePCH(true),
      SuppressMissingInputWarning(false) {

  // Provide a sane fallback if no VFS is specified.
  if (!this->VFS)
    this->VFS = vfs::getRealFileSystem();

  Name = llvm::sys::path::filename(ClangExecutable);
  Dir = llvm::sys::path::parent_path(ClangExecutable);
  InstalledDir = Dir; // Provide a sensible default installed dir.

  // Compute the path to the resource directory.
  StringRef ClangResourceDir(CLANG_RESOURCE_DIR);
  SmallString<128> P(Dir);
  if (ClangResourceDir != "") {
    llvm::sys::path::append(P, ClangResourceDir);
  } else {
    StringRef ClangLibdirSuffix(CLANG_LIBDIR_SUFFIX);
    llvm::sys::path::append(P, "..", Twine("lib") + ClangLibdirSuffix, "clang",
                            CLANG_VERSION_STRING);
  }
  ResourceDir = P.str();
}

Driver::~Driver() {
  delete Opts;

  llvm::DeleteContainerSeconds(ToolChains);
}

void Driver::ParseDriverMode(StringRef ProgramName,
                             ArrayRef<const char *> Args) {
  auto Default = ToolChain::getTargetAndModeFromProgramName(ProgramName);
  StringRef DefaultMode(Default.second);
  setDriverModeFromOption(DefaultMode);

  for (const char *ArgPtr : Args) {
    // Ingore nullptrs, they are response file's EOL markers
    if (ArgPtr == nullptr)
      continue;
    const StringRef Arg = ArgPtr;
    setDriverModeFromOption(Arg);
  }
}

void Driver::setDriverModeFromOption(StringRef Opt) {
  const std::string OptName =
      getOpts().getOption(options::OPT_driver_mode).getPrefixedName();
  if (!Opt.startswith(OptName))
    return;
  StringRef Value = Opt.drop_front(OptName.size());

  const unsigned M = llvm::StringSwitch<unsigned>(Value)
                         .Case("gcc", GCCMode)
                         .Case("g++", GXXMode)
                         .Case("cpp", CPPMode)
                         .Case("cl", CLMode)
                         .Default(~0U);

  if (M != ~0U)
    Mode = static_cast<DriverMode>(M);
  else
    Diag(diag::err_drv_unsupported_option_argument) << OptName << Value;
}

InputArgList Driver::ParseArgStrings(ArrayRef<const char *> ArgStrings) {
  llvm::PrettyStackTraceString CrashInfo("Command line argument parsing");

  unsigned IncludedFlagsBitmask;
  unsigned ExcludedFlagsBitmask;
  std::tie(IncludedFlagsBitmask, ExcludedFlagsBitmask) =
      getIncludeExcludeOptionFlagMasks();

  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args =
      getOpts().ParseArgs(ArgStrings, MissingArgIndex, MissingArgCount,
                          IncludedFlagsBitmask, ExcludedFlagsBitmask);

  // Check for missing argument error.
  if (MissingArgCount)
    Diag(clang::diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;

  // Check for unsupported options.
  for (const Arg *A : Args) {
    if (A->getOption().hasFlag(options::Unsupported)) {
      Diag(clang::diag::err_drv_unsupported_opt) << A->getAsString(Args);
      continue;
    }

    // Warn about -mcpu= without an argument.
    if (A->getOption().matches(options::OPT_mcpu_EQ) && A->containsValue("")) {
      Diag(clang::diag::warn_drv_empty_joined_argument) << A->getAsString(Args);
    }
  }

  for (const Arg *A : Args.filtered(options::OPT_UNKNOWN))
    Diags.Report(IsCLMode() ? diag::warn_drv_unknown_argument_clang_cl :
                              diag::err_drv_unknown_argument)
      << A->getAsString(Args);

  return Args;
}

// Determine which compilation mode we are in. We look for options which
// affect the phase, starting with the earliest phases, and record which
// option we used to determine the final phase.
phases::ID Driver::getFinalPhase(const DerivedArgList &DAL,
                                 Arg **FinalPhaseArg) const {
  Arg *PhaseArg = nullptr;
  phases::ID FinalPhase;

  // -{E,EP,P,M,MM} only run the preprocessor.
  if (CCCIsCPP() || (PhaseArg = DAL.getLastArg(options::OPT_E)) ||
      (PhaseArg = DAL.getLastArg(options::OPT__SLASH_EP)) ||
      (PhaseArg = DAL.getLastArg(options::OPT_M, options::OPT_MM)) ||
      (PhaseArg = DAL.getLastArg(options::OPT__SLASH_P))) {
    FinalPhase = phases::Preprocess;

    // -{fsyntax-only,-analyze,emit-ast} only run up to the compiler.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_fsyntax_only)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_module_file_info)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_verify_pch)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_rewrite_objc)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_rewrite_legacy_objc)) ||
             (PhaseArg = DAL.getLastArg(options::OPT__migrate)) ||
             (PhaseArg = DAL.getLastArg(options::OPT__analyze,
                                        options::OPT__analyze_auto)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_emit_ast))) {
    FinalPhase = phases::Compile;

    // -S only runs up to the backend.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_S))) {
    FinalPhase = phases::Backend;

    // -c compilation only runs up to the assembler.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_c))) {
    FinalPhase = phases::Assemble;

    // Otherwise do everything.
  } else
    FinalPhase = phases::Link;

  if (FinalPhaseArg)
    *FinalPhaseArg = PhaseArg;

  return FinalPhase;
}

static Arg *MakeInputArg(DerivedArgList &Args, OptTable *Opts,
                         StringRef Value) {
  Arg *A = new Arg(Opts->getOption(options::OPT_INPUT), Value,
                   Args.getBaseArgs().MakeIndex(Value), Value.data());
  Args.AddSynthesizedArg(A);
  A->claim();
  return A;
}

DerivedArgList *Driver::TranslateInputArgs(const InputArgList &Args) const {
  DerivedArgList *DAL = new DerivedArgList(Args);

  bool HasNostdlib = Args.hasArg(options::OPT_nostdlib);
  bool HasNodefaultlib = Args.hasArg(options::OPT_nodefaultlibs);
  for (Arg *A : Args) {
    // Unfortunately, we have to parse some forwarding options (-Xassembler,
    // -Xlinker, -Xpreprocessor) because we either integrate their functionality
    // (assembler and preprocessor), or bypass a previous driver ('collect2').

    // Rewrite linker options, to replace --no-demangle with a custom internal
    // option.
    if ((A->getOption().matches(options::OPT_Wl_COMMA) ||
         A->getOption().matches(options::OPT_Xlinker)) &&
        A->containsValue("--no-demangle")) {
      // Add the rewritten no-demangle argument.
      DAL->AddFlagArg(A, Opts->getOption(options::OPT_Z_Xlinker__no_demangle));

      // Add the remaining values as Xlinker arguments.
      for (StringRef Val : A->getValues())
        if (Val != "--no-demangle")
          DAL->AddSeparateArg(A, Opts->getOption(options::OPT_Xlinker), Val);

      continue;
    }

    // Rewrite preprocessor options, to replace -Wp,-MD,FOO which is used by
    // some build systems. We don't try to be complete here because we don't
    // care to encourage this usage model.
    if (A->getOption().matches(options::OPT_Wp_COMMA) &&
        (A->getValue(0) == StringRef("-MD") ||
         A->getValue(0) == StringRef("-MMD"))) {
      // Rewrite to -MD/-MMD along with -MF.
      if (A->getValue(0) == StringRef("-MD"))
        DAL->AddFlagArg(A, Opts->getOption(options::OPT_MD));
      else
        DAL->AddFlagArg(A, Opts->getOption(options::OPT_MMD));
      if (A->getNumValues() == 2)
        DAL->AddSeparateArg(A, Opts->getOption(options::OPT_MF),
                            A->getValue(1));
      continue;
    }

    // Rewrite reserved library names.
    if (A->getOption().matches(options::OPT_l)) {
      StringRef Value = A->getValue();

      // Rewrite unless -nostdlib is present.
      if (!HasNostdlib && !HasNodefaultlib && Value == "stdc++") {
        DAL->AddFlagArg(A, Opts->getOption(options::OPT_Z_reserved_lib_stdcxx));
        continue;
      }

      // Rewrite unconditionally.
      if (Value == "cc_kext") {
        DAL->AddFlagArg(A, Opts->getOption(options::OPT_Z_reserved_lib_cckext));
        continue;
      }
    }

    // Pick up inputs via the -- option.
    if (A->getOption().matches(options::OPT__DASH_DASH)) {
      A->claim();
      for (StringRef Val : A->getValues())
        DAL->append(MakeInputArg(*DAL, Opts, Val));
      continue;
    }

    DAL->append(A);
  }

  // Enforce -static if -miamcu is present.
  if (Args.hasFlag(options::OPT_miamcu, options::OPT_mno_iamcu, false))
    DAL->AddFlagArg(0, Opts->getOption(options::OPT_static));

// Add a default value of -mlinker-version=, if one was given and the user
// didn't specify one.
#if defined(HOST_LINK_VERSION)
  if (!Args.hasArg(options::OPT_mlinker_version_EQ) &&
      strlen(HOST_LINK_VERSION) > 0) {
    DAL->AddJoinedArg(0, Opts->getOption(options::OPT_mlinker_version_EQ),
                      HOST_LINK_VERSION);
    DAL->getLastArg(options::OPT_mlinker_version_EQ)->claim();
  }
#endif

  return DAL;
}

/// \brief Compute target triple from args.
///
/// This routine provides the logic to compute a target triple from various
/// args passed to the driver and the default triple string.
static llvm::Triple computeTargetTriple(const Driver &D,
                                        StringRef DefaultTargetTriple,
                                        const ArgList &Args,
                                        StringRef DarwinArchName = "") {
  // FIXME: Already done in Compilation *Driver::BuildCompilation
  if (const Arg *A = Args.getLastArg(options::OPT_target))
    DefaultTargetTriple = A->getValue();

  llvm::Triple Target(llvm::Triple::normalize(DefaultTargetTriple));

  // Handle Apple-specific options available here.
  if (Target.isOSBinFormatMachO()) {
    // If an explict Darwin arch name is given, that trumps all.
    if (!DarwinArchName.empty()) {
      tools::darwin::setTripleTypeForMachOArchName(Target, DarwinArchName);
      return Target;
    }

    // Handle the Darwin '-arch' flag.
    if (Arg *A = Args.getLastArg(options::OPT_arch)) {
      StringRef ArchName = A->getValue();
      tools::darwin::setTripleTypeForMachOArchName(Target, ArchName);
    }
  }

  // Handle pseudo-target flags '-mlittle-endian'/'-EL' and
  // '-mbig-endian'/'-EB'.
  if (Arg *A = Args.getLastArg(options::OPT_mlittle_endian,
                               options::OPT_mbig_endian)) {
    if (A->getOption().matches(options::OPT_mlittle_endian)) {
      llvm::Triple LE = Target.getLittleEndianArchVariant();
      if (LE.getArch() != llvm::Triple::UnknownArch)
        Target = std::move(LE);
    } else {
      llvm::Triple BE = Target.getBigEndianArchVariant();
      if (BE.getArch() != llvm::Triple::UnknownArch)
        Target = std::move(BE);
    }
  }

  // Skip further flag support on OSes which don't support '-m32' or '-m64'.
  if (Target.getArch() == llvm::Triple::tce ||
      Target.getOS() == llvm::Triple::Minix)
    return Target;

  // Handle pseudo-target flags '-m64', '-mx32', '-m32' and '-m16'.
  Arg *A = Args.getLastArg(options::OPT_m64, options::OPT_mx32,
                           options::OPT_m32, options::OPT_m16);
  if (A) {
    llvm::Triple::ArchType AT = llvm::Triple::UnknownArch;

    if (A->getOption().matches(options::OPT_m64)) {
      AT = Target.get64BitArchVariant().getArch();
      if (Target.getEnvironment() == llvm::Triple::GNUX32)
        Target.setEnvironment(llvm::Triple::GNU);
    } else if (A->getOption().matches(options::OPT_mx32) &&
               Target.get64BitArchVariant().getArch() == llvm::Triple::x86_64) {
      AT = llvm::Triple::x86_64;
      Target.setEnvironment(llvm::Triple::GNUX32);
    } else if (A->getOption().matches(options::OPT_m32)) {
      AT = Target.get32BitArchVariant().getArch();
      if (Target.getEnvironment() == llvm::Triple::GNUX32)
        Target.setEnvironment(llvm::Triple::GNU);
    } else if (A->getOption().matches(options::OPT_m16) &&
               Target.get32BitArchVariant().getArch() == llvm::Triple::x86) {
      AT = llvm::Triple::x86;
      Target.setEnvironment(llvm::Triple::CODE16);
    }

    if (AT != llvm::Triple::UnknownArch && AT != Target.getArch())
      Target.setArch(AT);
  }

  // Handle -miamcu flag.
  if (Args.hasFlag(options::OPT_miamcu, options::OPT_mno_iamcu, false)) {
    if (Target.get32BitArchVariant().getArch() != llvm::Triple::x86)
      D.Diag(diag::err_drv_unsupported_opt_for_target) << "-miamcu"
                                                       << Target.str();

    if (A && !A->getOption().matches(options::OPT_m32))
      D.Diag(diag::err_drv_argument_not_allowed_with)
          << "-miamcu" << A->getBaseArg().getAsString(Args);

    Target.setArch(llvm::Triple::x86);
    Target.setArchName("i586");
    Target.setEnvironment(llvm::Triple::UnknownEnvironment);
    Target.setEnvironmentName("");
    Target.setOS(llvm::Triple::ELFIAMCU);
    Target.setVendor(llvm::Triple::UnknownVendor);
    Target.setVendorName("intel");
  }

  return Target;
}

// \brief Parse the LTO options and record the type of LTO compilation
// based on which -f(no-)?lto(=.*)? option occurs last.
void Driver::setLTOMode(const llvm::opt::ArgList &Args) {
  LTOMode = LTOK_None;
  if (!Args.hasFlag(options::OPT_flto, options::OPT_flto_EQ,
                    options::OPT_fno_lto, false))
    return;

  StringRef LTOName("full");

  const Arg *A = Args.getLastArg(options::OPT_flto_EQ);
  if (A)
    LTOName = A->getValue();

  LTOMode = llvm::StringSwitch<LTOKind>(LTOName)
                .Case("full", LTOK_Full)
                .Case("thin", LTOK_Thin)
                .Default(LTOK_Unknown);

  if (LTOMode == LTOK_Unknown) {
    assert(A);
    Diag(diag::err_drv_unsupported_option_argument) << A->getOption().getName()
                                                    << A->getValue();
  }
}

void Driver::CreateOffloadingDeviceToolChains(Compilation &C,
                                              InputList &Inputs) {

  //
  // CUDA
  //
  // We need to generate a CUDA toolchain if any of the inputs has a CUDA type.
  if (llvm::any_of(Inputs, [](std::pair<types::ID, const llvm::opt::Arg *> &I) {
        return types::isCuda(I.first);
      })) {
    const ToolChain &TC = getToolChain(
        C.getInputArgs(),
        llvm::Triple(C.getSingleOffloadToolChain<Action::OFK_Host>()
                             ->getTriple()
                             .isArch64Bit()
                         ? "nvptx64-nvidia-cuda"
                         : "nvptx-nvidia-cuda"));
    C.addOffloadDeviceToolChain(&TC, Action::OFK_Cuda);
  }

  //
  // TODO: Add support for other offloading programming models here.
  //

  return;
}

Compilation *Driver::BuildCompilation(ArrayRef<const char *> ArgList) {
  llvm::PrettyStackTraceString CrashInfo("Compilation construction");

  // FIXME: Handle environment options which affect driver behavior, somewhere
  // (client?). GCC_EXEC_PREFIX, LPATH, CC_PRINT_OPTIONS.

  if (Optional<std::string> CompilerPathValue =
          llvm::sys::Process::GetEnv("COMPILER_PATH")) {
    StringRef CompilerPath = *CompilerPathValue;
    while (!CompilerPath.empty()) {
      std::pair<StringRef, StringRef> Split =
          CompilerPath.split(llvm::sys::EnvPathSeparator);
      PrefixDirs.push_back(Split.first);
      CompilerPath = Split.second;
    }
  }

  // We look for the driver mode option early, because the mode can affect
  // how other options are parsed.
  ParseDriverMode(ClangExecutable, ArgList.slice(1));

  // FIXME: What are we going to do with -V and -b?

  // FIXME: This stuff needs to go into the Compilation, not the driver.
  bool CCCPrintPhases;

  InputArgList Args = ParseArgStrings(ArgList.slice(1));

  // Silence driver warnings if requested
  Diags.setIgnoreAllWarnings(Args.hasArg(options::OPT_w));

  // -no-canonical-prefixes is used very early in main.
  Args.ClaimAllArgs(options::OPT_no_canonical_prefixes);

  // Ignore -pipe.
  Args.ClaimAllArgs(options::OPT_pipe);

  // Extract -ccc args.
  //
  // FIXME: We need to figure out where this behavior should live. Most of it
  // should be outside in the client; the parts that aren't should have proper
  // options, either by introducing new ones or by overloading gcc ones like -V
  // or -b.
  CCCPrintPhases = Args.hasArg(options::OPT_ccc_print_phases);
  CCCPrintBindings = Args.hasArg(options::OPT_ccc_print_bindings);
  if (const Arg *A = Args.getLastArg(options::OPT_ccc_gcc_name))
    CCCGenericGCCName = A->getValue();
  CCCUsePCH =
      Args.hasFlag(options::OPT_ccc_pch_is_pch, options::OPT_ccc_pch_is_pth);
  // FIXME: DefaultTargetTriple is used by the target-prefixed calls to as/ld
  // and getToolChain is const.
  if (IsCLMode()) {
    // clang-cl targets MSVC-style Win32.
    llvm::Triple T(DefaultTargetTriple);
    T.setOS(llvm::Triple::Win32);
    T.setVendor(llvm::Triple::PC);
    T.setEnvironment(llvm::Triple::MSVC);
    DefaultTargetTriple = T.str();
  }
  if (const Arg *A = Args.getLastArg(options::OPT_target))
    DefaultTargetTriple = A->getValue();
  if (const Arg *A = Args.getLastArg(options::OPT_ccc_install_dir))
    Dir = InstalledDir = A->getValue();
  for (const Arg *A : Args.filtered(options::OPT_B)) {
    A->claim();
    PrefixDirs.push_back(A->getValue(0));
  }
  if (const Arg *A = Args.getLastArg(options::OPT__sysroot_EQ))
    SysRoot = A->getValue();
  if (const Arg *A = Args.getLastArg(options::OPT__dyld_prefix_EQ))
    DyldPrefix = A->getValue();
  if (Args.hasArg(options::OPT_nostdlib))
    UseStdLib = false;

  if (const Arg *A = Args.getLastArg(options::OPT_resource_dir))
    ResourceDir = A->getValue();

  if (const Arg *A = Args.getLastArg(options::OPT_save_temps_EQ)) {
    SaveTemps = llvm::StringSwitch<SaveTempsMode>(A->getValue())
                    .Case("cwd", SaveTempsCwd)
                    .Case("obj", SaveTempsObj)
                    .Default(SaveTempsCwd);
  }

  setLTOMode(Args);

  // Ignore -fembed-bitcode options with LTO
  // since the output will be bitcode anyway.
  if (getLTOMode() == LTOK_None) {
    if (Arg *A = Args.getLastArg(options::OPT_fembed_bitcode_EQ)) {
      StringRef Name = A->getValue();
      unsigned Model = llvm::StringSwitch<unsigned>(Name)
          .Case("off", EmbedNone)
          .Case("all", EmbedBitcode)
          .Case("bitcode", EmbedBitcode)
          .Case("marker", EmbedMarker)
          .Default(~0U);
      if (Model == ~0U) {
        Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                  << Name;
      } else
        BitcodeEmbed = static_cast<BitcodeEmbedMode>(Model);
    }
  } else {
    // claim the bitcode option under LTO so no warning is issued.
    Args.ClaimAllArgs(options::OPT_fembed_bitcode_EQ);
  }

  std::unique_ptr<llvm::opt::InputArgList> UArgs =
      llvm::make_unique<InputArgList>(std::move(Args));

  // Perform the default argument translations.
  DerivedArgList *TranslatedArgs = TranslateInputArgs(*UArgs);

  // Owned by the host.
  const ToolChain &TC = getToolChain(
      *UArgs, computeTargetTriple(*this, DefaultTargetTriple, *UArgs));

  // The compilation takes ownership of Args.
  Compilation *C = new Compilation(*this, TC, UArgs.release(), TranslatedArgs);

  if (!HandleImmediateArgs(*C))
    return C;

  // Construct the list of inputs.
  InputList Inputs;
  BuildInputs(C->getDefaultToolChain(), *TranslatedArgs, Inputs);

  // Populate the tool chains for the offloading devices, if any.
  CreateOffloadingDeviceToolChains(*C, Inputs);

  // Construct the list of abstract actions to perform for this compilation. On
  // MachO targets this uses the driver-driver and universal actions.
  if (TC.getTriple().isOSBinFormatMachO())
    BuildUniversalActions(*C, C->getDefaultToolChain(), Inputs);
  else
    BuildActions(*C, C->getArgs(), Inputs, C->getActions());

  if (CCCPrintPhases) {
    PrintActions(*C);
    return C;
  }

  BuildJobs(*C);

  return C;
}

static void printArgList(raw_ostream &OS, const llvm::opt::ArgList &Args) {
  llvm::opt::ArgStringList ASL;
  for (const auto *A : Args)
    A->render(Args, ASL);

  for (auto I = ASL.begin(), E = ASL.end(); I != E; ++I) {
    if (I != ASL.begin())
      OS << ' ';
    Command::printArg(OS, *I, true);
  }
  OS << '\n';
}

// When clang crashes, produce diagnostic information including the fully
// preprocessed source file(s).  Request that the developer attach the
// diagnostic information to a bug report.
void Driver::generateCompilationDiagnostics(Compilation &C,
                                            const Command &FailingCommand) {
  if (C.getArgs().hasArg(options::OPT_fno_crash_diagnostics))
    return;

  // Don't try to generate diagnostics for link or dsymutil jobs.
  if (FailingCommand.getCreator().isLinkJob() ||
      FailingCommand.getCreator().isDsymutilJob())
    return;

  // Print the version of the compiler.
  PrintVersion(C, llvm::errs());

  Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "PLEASE submit a bug report to " BUG_REPORT_URL " and include the "
         "crash backtrace, preprocessed source, and associated run script.";

  // Suppress driver output and emit preprocessor output to temp file.
  Mode = CPPMode;
  CCGenDiagnostics = true;

  // Save the original job command(s).
  Command Cmd = FailingCommand;

  // Keep track of whether we produce any errors while trying to produce
  // preprocessed sources.
  DiagnosticErrorTrap Trap(Diags);

  // Suppress tool output.
  C.initCompilationForDiagnostics();

  // Construct the list of inputs.
  InputList Inputs;
  BuildInputs(C.getDefaultToolChain(), C.getArgs(), Inputs);

  for (InputList::iterator it = Inputs.begin(), ie = Inputs.end(); it != ie;) {
    bool IgnoreInput = false;

    // Ignore input from stdin or any inputs that cannot be preprocessed.
    // Check type first as not all linker inputs have a value.
    if (types::getPreprocessedType(it->first) == types::TY_INVALID) {
      IgnoreInput = true;
    } else if (!strcmp(it->second->getValue(), "-")) {
      Diag(clang::diag::note_drv_command_failed_diag_msg)
          << "Error generating preprocessed source(s) - "
             "ignoring input from stdin.";
      IgnoreInput = true;
    }

    if (IgnoreInput) {
      it = Inputs.erase(it);
      ie = Inputs.end();
    } else {
      ++it;
    }
  }

  if (Inputs.empty()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s) - "
           "no preprocessable inputs.";
    return;
  }

  // Don't attempt to generate preprocessed files if multiple -arch options are
  // used, unless they're all duplicates.
  llvm::StringSet<> ArchNames;
  for (const Arg *A : C.getArgs()) {
    if (A->getOption().matches(options::OPT_arch)) {
      StringRef ArchName = A->getValue();
      ArchNames.insert(ArchName);
    }
  }
  if (ArchNames.size() > 1) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s) - cannot generate "
           "preprocessed source with multiple -arch options.";
    return;
  }

  // Construct the list of abstract actions to perform for this compilation. On
  // Darwin OSes this uses the driver-driver and builds universal actions.
  const ToolChain &TC = C.getDefaultToolChain();
  if (TC.getTriple().isOSBinFormatMachO())
    BuildUniversalActions(C, TC, Inputs);
  else
    BuildActions(C, C.getArgs(), Inputs, C.getActions());

  BuildJobs(C);

  // If there were errors building the compilation, quit now.
  if (Trap.hasErrorOccurred()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s).";
    return;
  }

  // Generate preprocessed output.
  SmallVector<std::pair<int, const Command *>, 4> FailingCommands;
  C.ExecuteJobs(C.getJobs(), FailingCommands);

  // If any of the preprocessing commands failed, clean up and exit.
  if (!FailingCommands.empty()) {
    if (!isSaveTempsEnabled())
      C.CleanupFileList(C.getTempFiles(), true);

    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s).";
    return;
  }

  const ArgStringList &TempFiles = C.getTempFiles();
  if (TempFiles.empty()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s).";
    return;
  }

  Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "\n********************\n\n"
         "PLEASE ATTACH THE FOLLOWING FILES TO THE BUG REPORT:\n"
         "Preprocessed source(s) and associated run script(s) are located at:";

  SmallString<128> VFS;
  for (const char *TempFile : TempFiles) {
    Diag(clang::diag::note_drv_command_failed_diag_msg) << TempFile;
    if (StringRef(TempFile).endswith(".cache")) {
      // In some cases (modules) we'll dump extra data to help with reproducing
      // the crash into a directory next to the output.
      VFS = llvm::sys::path::filename(TempFile);
      llvm::sys::path::append(VFS, "vfs", "vfs.yaml");
    }
  }

  // Assume associated files are based off of the first temporary file.
  CrashReportInfo CrashInfo(TempFiles[0], VFS);

  std::string Script = CrashInfo.Filename.rsplit('.').first.str() + ".sh";
  std::error_code EC;
  llvm::raw_fd_ostream ScriptOS(Script, EC, llvm::sys::fs::F_Excl);
  if (EC) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating run script: " + Script + " " + EC.message();
  } else {
    ScriptOS << "# Crash reproducer for " << getClangFullVersion() << "\n"
             << "# Driver args: ";
    printArgList(ScriptOS, C.getInputArgs());
    ScriptOS << "# Original command: ";
    Cmd.Print(ScriptOS, "\n", /*Quote=*/true);
    Cmd.Print(ScriptOS, "\n", /*Quote=*/true, &CrashInfo);
    Diag(clang::diag::note_drv_command_failed_diag_msg) << Script;
  }

  for (const auto &A : C.getArgs().filtered(options::OPT_frewrite_map_file,
                                            options::OPT_frewrite_map_file_EQ))
    Diag(clang::diag::note_drv_command_failed_diag_msg) << A->getValue();

  Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "\n\n********************";
}

void Driver::setUpResponseFiles(Compilation &C, Command &Cmd) {
  // Since commandLineFitsWithinSystemLimits() may underestimate system's capacity
  // if the tool does not support response files, there is a chance/ that things
  // will just work without a response file, so we silently just skip it.
  if (Cmd.getCreator().getResponseFilesSupport() == Tool::RF_None ||
      llvm::sys::commandLineFitsWithinSystemLimits(Cmd.getExecutable(), Cmd.getArguments()))
    return;

  std::string TmpName = GetTemporaryPath("response", "txt");
  Cmd.setResponseFile(
      C.addTempFile(C.getArgs().MakeArgString(TmpName.c_str())));
}

int Driver::ExecuteCompilation(
    Compilation &C,
    SmallVectorImpl<std::pair<int, const Command *>> &FailingCommands) {
  // Just print if -### was present.
  if (C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    C.getJobs().Print(llvm::errs(), "\n", true);
    return 0;
  }

  // If there were errors building the compilation, quit now.
  if (Diags.hasErrorOccurred())
    return 1;

  // Set up response file names for each command, if necessary
  for (auto &Job : C.getJobs())
    setUpResponseFiles(C, Job);

  C.ExecuteJobs(C.getJobs(), FailingCommands);

  // Remove temp files.
  C.CleanupFileList(C.getTempFiles());

  // If the command succeeded, we are done.
  if (FailingCommands.empty())
    return 0;

  // Otherwise, remove result files and print extra information about abnormal
  // failures.
  for (const auto &CmdPair : FailingCommands) {
    int Res = CmdPair.first;
    const Command *FailingCommand = CmdPair.second;

    // Remove result files if we're not saving temps.
    if (!isSaveTempsEnabled()) {
      const JobAction *JA = cast<JobAction>(&FailingCommand->getSource());
      C.CleanupFileMap(C.getResultFiles(), JA, true);

      // Failure result files are valid unless we crashed.
      if (Res < 0)
        C.CleanupFileMap(C.getFailureResultFiles(), JA, true);
    }

    // Print extra information about abnormal failures, if possible.
    //
    // This is ad-hoc, but we don't want to be excessively noisy. If the result
    // status was 1, assume the command failed normally. In particular, if it
    // was the compiler then assume it gave a reasonable error code. Failures
    // in other tools are less common, and they generally have worse
    // diagnostics, so always print the diagnostic there.
    const Tool &FailingTool = FailingCommand->getCreator();

    if (!FailingCommand->getCreator().hasGoodDiagnostics() || Res != 1) {
      // FIXME: See FIXME above regarding result code interpretation.
      if (Res < 0)
        Diag(clang::diag::err_drv_command_signalled)
            << FailingTool.getShortName();
      else
        Diag(clang::diag::err_drv_command_failed) << FailingTool.getShortName()
                                                  << Res;
    }
  }
  return 0;
}

void Driver::PrintHelp(bool ShowHidden) const {
  unsigned IncludedFlagsBitmask;
  unsigned ExcludedFlagsBitmask;
  std::tie(IncludedFlagsBitmask, ExcludedFlagsBitmask) =
      getIncludeExcludeOptionFlagMasks();

  ExcludedFlagsBitmask |= options::NoDriverOption;
  if (!ShowHidden)
    ExcludedFlagsBitmask |= HelpHidden;

  getOpts().PrintHelp(llvm::outs(), Name.c_str(), DriverTitle.c_str(),
                      IncludedFlagsBitmask, ExcludedFlagsBitmask);
}

void Driver::PrintVersion(const Compilation &C, raw_ostream &OS) const {
  // FIXME: The following handlers should use a callback mechanism, we don't
  // know what the client would like to do.
  OS << getClangFullVersion() << '\n';
  const ToolChain &TC = C.getDefaultToolChain();
  OS << "Target: " << TC.getTripleString() << '\n';

  // Print the threading model.
  if (Arg *A = C.getArgs().getLastArg(options::OPT_mthread_model)) {
    // Don't print if the ToolChain would have barfed on it already
    if (TC.isThreadModelSupported(A->getValue()))
      OS << "Thread model: " << A->getValue();
  } else
    OS << "Thread model: " << TC.getThreadModel();
  OS << '\n';

  // Print out the install directory.
  OS << "InstalledDir: " << InstalledDir << '\n';
}

/// PrintDiagnosticCategories - Implement the --print-diagnostic-categories
/// option.
static void PrintDiagnosticCategories(raw_ostream &OS) {
  // Skip the empty category.
  for (unsigned i = 1, max = DiagnosticIDs::getNumberOfCategories(); i != max;
       ++i)
    OS << i << ',' << DiagnosticIDs::getCategoryNameFromID(i) << '\n';
}

bool Driver::HandleImmediateArgs(const Compilation &C) {
  // The order these options are handled in gcc is all over the place, but we
  // don't expect inconsistencies w.r.t. that to matter in practice.

  if (C.getArgs().hasArg(options::OPT_dumpmachine)) {
    llvm::outs() << C.getDefaultToolChain().getTripleString() << '\n';
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_dumpversion)) {
    // Since -dumpversion is only implemented for pedantic GCC compatibility, we
    // return an answer which matches our definition of __VERSION__.
    //
    // If we want to return a more correct answer some day, then we should
    // introduce a non-pedantically GCC compatible mode to Clang in which we
    // provide sensible definitions for -dumpversion, __VERSION__, etc.
    llvm::outs() << "4.2.1\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT__print_diagnostic_categories)) {
    PrintDiagnosticCategories(llvm::outs());
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_help) ||
      C.getArgs().hasArg(options::OPT__help_hidden)) {
    PrintHelp(C.getArgs().hasArg(options::OPT__help_hidden));
    return false;
  }

  if (C.getArgs().hasArg(options::OPT__version)) {
    // Follow gcc behavior and use stdout for --version and stderr for -v.
    PrintVersion(C, llvm::outs());
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_v) ||
      C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    PrintVersion(C, llvm::errs());
    SuppressMissingInputWarning = true;
  }

  const ToolChain &TC = C.getDefaultToolChain();

  if (C.getArgs().hasArg(options::OPT_v))
    TC.printVerboseInfo(llvm::errs());

  if (C.getArgs().hasArg(options::OPT_print_search_dirs)) {
    llvm::outs() << "programs: =";
    bool separator = false;
    for (const std::string &Path : TC.getProgramPaths()) {
      if (separator)
        llvm::outs() << ':';
      llvm::outs() << Path;
      separator = true;
    }
    llvm::outs() << "\n";
    llvm::outs() << "libraries: =" << ResourceDir;

    StringRef sysroot = C.getSysRoot();

    for (const std::string &Path : TC.getFilePaths()) {
      // Always print a separator. ResourceDir was the first item shown.
      llvm::outs() << ':';
      // Interpretation of leading '=' is needed only for NetBSD.
      if (Path[0] == '=')
        llvm::outs() << sysroot << Path.substr(1);
      else
        llvm::outs() << Path;
    }
    llvm::outs() << "\n";
    return false;
  }

  // FIXME: The following handlers should use a callback mechanism, we don't
  // know what the client would like to do.
  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_file_name_EQ)) {
    llvm::outs() << GetFilePath(A->getValue(), TC) << "\n";
    return false;
  }

  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_prog_name_EQ)) {
    llvm::outs() << GetProgramPath(A->getValue(), TC) << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_libgcc_file_name)) {
    llvm::outs() << GetFilePath("libgcc.a", TC) << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_multi_lib)) {
    for (const Multilib &Multilib : TC.getMultilibs())
      llvm::outs() << Multilib << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_multi_directory)) {
    for (const Multilib &Multilib : TC.getMultilibs()) {
      if (Multilib.gccSuffix().empty())
        llvm::outs() << ".\n";
      else {
        StringRef Suffix(Multilib.gccSuffix());
        assert(Suffix.front() == '/');
        llvm::outs() << Suffix.substr(1) << "\n";
      }
    }
    return false;
  }
  return true;
}

// Display an action graph human-readably.  Action A is the "sink" node
// and latest-occuring action. Traversal is in pre-order, visiting the
// inputs to each action before printing the action itself.
static unsigned PrintActions1(const Compilation &C, Action *A,
                              std::map<Action *, unsigned> &Ids) {
  if (Ids.count(A)) // A was already visited.
    return Ids[A];

  std::string str;
  llvm::raw_string_ostream os(str);

  os << Action::getClassName(A->getKind()) << ", ";
  if (InputAction *IA = dyn_cast<InputAction>(A)) {
    os << "\"" << IA->getInputArg().getValue() << "\"";
  } else if (BindArchAction *BIA = dyn_cast<BindArchAction>(A)) {
    os << '"' << BIA->getArchName() << '"' << ", {"
       << PrintActions1(C, *BIA->input_begin(), Ids) << "}";
  } else if (OffloadAction *OA = dyn_cast<OffloadAction>(A)) {
    bool IsFirst = true;
    OA->doOnEachDependence(
        [&](Action *A, const ToolChain *TC, const char *BoundArch) {
          // E.g. for two CUDA device dependences whose bound arch is sm_20 and
          // sm_35 this will generate:
          // "cuda-device" (nvptx64-nvidia-cuda:sm_20) {#ID}, "cuda-device"
          // (nvptx64-nvidia-cuda:sm_35) {#ID}
          if (!IsFirst)
            os << ", ";
          os << '"';
          if (TC)
            os << A->getOffloadingKindPrefix();
          else
            os << "host";
          os << " (";
          os << TC->getTriple().normalize();

          if (BoundArch)
            os << ":" << BoundArch;
          os << ")";
          os << '"';
          os << " {" << PrintActions1(C, A, Ids) << "}";
          IsFirst = false;
        });
  } else {
    const ActionList *AL = &A->getInputs();

    if (AL->size()) {
      const char *Prefix = "{";
      for (Action *PreRequisite : *AL) {
        os << Prefix << PrintActions1(C, PreRequisite, Ids);
        Prefix = ", ";
      }
      os << "}";
    } else
      os << "{}";
  }

  // Append offload info for all options other than the offloading action
  // itself (e.g. (cuda-device, sm_20) or (cuda-host)).
  std::string offload_str;
  llvm::raw_string_ostream offload_os(offload_str);
  if (!isa<OffloadAction>(A)) {
    auto S = A->getOffloadingKindPrefix();
    if (!S.empty()) {
      offload_os << ", (" << S;
      if (A->getOffloadingArch())
        offload_os << ", " << A->getOffloadingArch();
      offload_os << ")";
    }
  }

  unsigned Id = Ids.size();
  Ids[A] = Id;
  llvm::errs() << Id << ": " << os.str() << ", "
               << types::getTypeName(A->getType()) << offload_os.str() << "\n";

  return Id;
}

// Print the action graphs in a compilation C.
// For example "clang -c file1.c file2.c" is composed of two subgraphs.
void Driver::PrintActions(const Compilation &C) const {
  std::map<Action *, unsigned> Ids;
  for (Action *A : C.getActions())
    PrintActions1(C, A, Ids);
}

/// \brief Check whether the given input tree contains any compilation or
/// assembly actions.
static bool ContainsCompileOrAssembleAction(const Action *A) {
  if (isa<CompileJobAction>(A) || isa<BackendJobAction>(A) ||
      isa<AssembleJobAction>(A))
    return true;

  for (const Action *Input : A->inputs())
    if (ContainsCompileOrAssembleAction(Input))
      return true;

  return false;
}

void Driver::BuildUniversalActions(Compilation &C, const ToolChain &TC,
                                   const InputList &BAInputs) const {
  DerivedArgList &Args = C.getArgs();
  ActionList &Actions = C.getActions();
  llvm::PrettyStackTraceString CrashInfo("Building universal build actions");
  // Collect the list of architectures. Duplicates are allowed, but should only
  // be handled once (in the order seen).
  llvm::StringSet<> ArchNames;
  SmallVector<const char *, 4> Archs;
  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT_arch)) {
      // Validate the option here; we don't save the type here because its
      // particular spelling may participate in other driver choices.
      llvm::Triple::ArchType Arch =
          tools::darwin::getArchTypeForMachOArchName(A->getValue());
      if (Arch == llvm::Triple::UnknownArch) {
        Diag(clang::diag::err_drv_invalid_arch_name) << A->getAsString(Args);
        continue;
      }

      A->claim();
      if (ArchNames.insert(A->getValue()).second)
        Archs.push_back(A->getValue());
    }
  }

  // When there is no explicit arch for this platform, make sure we still bind
  // the architecture (to the default) so that -Xarch_ is handled correctly.
  if (!Archs.size())
    Archs.push_back(Args.MakeArgString(TC.getDefaultUniversalArchName()));

  ActionList SingleActions;
  BuildActions(C, Args, BAInputs, SingleActions);

  // Add in arch bindings for every top level action, as well as lipo and
  // dsymutil steps if needed.
  for (Action* Act : SingleActions) {
    // Make sure we can lipo this kind of output. If not (and it is an actual
    // output) then we disallow, since we can't create an output file with the
    // right name without overwriting it. We could remove this oddity by just
    // changing the output names to include the arch, which would also fix
    // -save-temps. Compatibility wins for now.

    if (Archs.size() > 1 && !types::canLipoType(Act->getType()))
      Diag(clang::diag::err_drv_invalid_output_with_multiple_archs)
          << types::getTypeName(Act->getType());

    ActionList Inputs;
    for (unsigned i = 0, e = Archs.size(); i != e; ++i)
      Inputs.push_back(C.MakeAction<BindArchAction>(Act, Archs[i]));

    // Lipo if necessary, we do it this way because we need to set the arch flag
    // so that -Xarch_ gets overwritten.
    if (Inputs.size() == 1 || Act->getType() == types::TY_Nothing)
      Actions.append(Inputs.begin(), Inputs.end());
    else
      Actions.push_back(C.MakeAction<LipoJobAction>(Inputs, Act->getType()));

    // Handle debug info queries.
    Arg *A = Args.getLastArg(options::OPT_g_Group);
    if (A && !A->getOption().matches(options::OPT_g0) &&
        !A->getOption().matches(options::OPT_gstabs) &&
        ContainsCompileOrAssembleAction(Actions.back())) {

      // Add a 'dsymutil' step if necessary, when debug info is enabled and we
      // have a compile input. We need to run 'dsymutil' ourselves in such cases
      // because the debug info will refer to a temporary object file which
      // will be removed at the end of the compilation process.
      if (Act->getType() == types::TY_Image) {
        ActionList Inputs;
        Inputs.push_back(Actions.back());
        Actions.pop_back();
        Actions.push_back(
            C.MakeAction<DsymutilJobAction>(Inputs, types::TY_dSYM));
      }

      // Verify the debug info output.
      if (Args.hasArg(options::OPT_verify_debug_info)) {
        Action* LastAction = Actions.back();
        Actions.pop_back();
        Actions.push_back(C.MakeAction<VerifyDebugInfoJobAction>(
            LastAction, types::TY_Nothing));
      }
    }
  }
}

/// \brief Check that the file referenced by Value exists. If it doesn't,
/// issue a diagnostic and return false.
static bool DiagnoseInputExistence(const Driver &D, const DerivedArgList &Args,
                                   StringRef Value, types::ID Ty) {
  if (!D.getCheckInputsExist())
    return true;

  // stdin always exists.
  if (Value == "-")
    return true;

  SmallString<64> Path(Value);
  if (Arg *WorkDir = Args.getLastArg(options::OPT_working_directory)) {
    if (!llvm::sys::path::is_absolute(Path)) {
      SmallString<64> Directory(WorkDir->getValue());
      llvm::sys::path::append(Directory, Value);
      Path.assign(Directory);
    }
  }

  if (llvm::sys::fs::exists(Twine(Path)))
    return true;

  if (D.IsCLMode()) {
    if (!llvm::sys::path::is_absolute(Twine(Path)) &&
        llvm::sys::Process::FindInEnvPath("LIB", Value))
      return true;

    if (Args.hasArg(options::OPT__SLASH_link) && Ty == types::TY_Object) {
      // Arguments to the /link flag might cause the linker to search for object
      // and library files in paths we don't know about. Don't error in such
      // cases.
      return true;
    }
  }

  D.Diag(clang::diag::err_drv_no_such_file) << Path;
  return false;
}

// Construct a the list of inputs and their types.
void Driver::BuildInputs(const ToolChain &TC, DerivedArgList &Args,
                         InputList &Inputs) const {
  // Track the current user specified (-x) input. We also explicitly track the
  // argument used to set the type; we only want to claim the type when we
  // actually use it, so we warn about unused -x arguments.
  types::ID InputType = types::TY_Nothing;
  Arg *InputTypeArg = nullptr;

  // The last /TC or /TP option sets the input type to C or C++ globally.
  if (Arg *TCTP = Args.getLastArgNoClaim(options::OPT__SLASH_TC,
                                         options::OPT__SLASH_TP)) {
    InputTypeArg = TCTP;
    InputType = TCTP->getOption().matches(options::OPT__SLASH_TC)
                    ? types::TY_C
                    : types::TY_CXX;

    arg_iterator it =
        Args.filtered_begin(options::OPT__SLASH_TC, options::OPT__SLASH_TP);
    const arg_iterator ie = Args.filtered_end();
    Arg *Previous = *it++;
    bool ShowNote = false;
    while (it != ie) {
      Diag(clang::diag::warn_drv_overriding_flag_option)
          << Previous->getSpelling() << (*it)->getSpelling();
      Previous = *it++;
      ShowNote = true;
    }
    if (ShowNote)
      Diag(clang::diag::note_drv_t_option_is_global);

    // No driver mode exposes -x and /TC or /TP; we don't support mixing them.
    assert(!Args.hasArg(options::OPT_x) && "-x and /TC or /TP is not allowed");
  }

  for (Arg *A : Args) {
    if (A->getOption().getKind() == Option::InputClass) {
      const char *Value = A->getValue();
      types::ID Ty = types::TY_INVALID;

      // Infer the input type if necessary.
      if (InputType == types::TY_Nothing) {
        // If there was an explicit arg for this, claim it.
        if (InputTypeArg)
          InputTypeArg->claim();

        // stdin must be handled specially.
        if (memcmp(Value, "-", 2) == 0) {
          // If running with -E, treat as a C input (this changes the builtin
          // macros, for example). This may be overridden by -ObjC below.
          //
          // Otherwise emit an error but still use a valid type to avoid
          // spurious errors (e.g., no inputs).
          if (!Args.hasArgNoClaim(options::OPT_E) && !CCCIsCPP())
            Diag(IsCLMode() ? clang::diag::err_drv_unknown_stdin_type_clang_cl
                            : clang::diag::err_drv_unknown_stdin_type);
          Ty = types::TY_C;
        } else {
          // Otherwise lookup by extension.
          // Fallback is C if invoked as C preprocessor or Object otherwise.
          // We use a host hook here because Darwin at least has its own
          // idea of what .s is.
          if (const char *Ext = strrchr(Value, '.'))
            Ty = TC.LookupTypeForExtension(Ext + 1);

          if (Ty == types::TY_INVALID) {
            if (CCCIsCPP())
              Ty = types::TY_C;
            else
              Ty = types::TY_Object;
          }

          // If the driver is invoked as C++ compiler (like clang++ or c++) it
          // should autodetect some input files as C++ for g++ compatibility.
          if (CCCIsCXX()) {
            types::ID OldTy = Ty;
            Ty = types::lookupCXXTypeForCType(Ty);

            if (Ty != OldTy)
              Diag(clang::diag::warn_drv_treating_input_as_cxx)
                  << getTypeName(OldTy) << getTypeName(Ty);
          }
        }

        // -ObjC and -ObjC++ override the default language, but only for "source
        // files". We just treat everything that isn't a linker input as a
        // source file.
        //
        // FIXME: Clean this up if we move the phase sequence into the type.
        if (Ty != types::TY_Object) {
          if (Args.hasArg(options::OPT_ObjC))
            Ty = types::TY_ObjC;
          else if (Args.hasArg(options::OPT_ObjCXX))
            Ty = types::TY_ObjCXX;
        }
      } else {
        assert(InputTypeArg && "InputType set w/o InputTypeArg");
        if (!InputTypeArg->getOption().matches(options::OPT_x)) {
          // If emulating cl.exe, make sure that /TC and /TP don't affect input
          // object files.
          const char *Ext = strrchr(Value, '.');
          if (Ext && TC.LookupTypeForExtension(Ext + 1) == types::TY_Object)
            Ty = types::TY_Object;
        }
        if (Ty == types::TY_INVALID) {
          Ty = InputType;
          InputTypeArg->claim();
        }
      }

      if (DiagnoseInputExistence(*this, Args, Value, Ty))
        Inputs.push_back(std::make_pair(Ty, A));

    } else if (A->getOption().matches(options::OPT__SLASH_Tc)) {
      StringRef Value = A->getValue();
      if (DiagnoseInputExistence(*this, Args, Value, types::TY_C)) {
        Arg *InputArg = MakeInputArg(Args, Opts, A->getValue());
        Inputs.push_back(std::make_pair(types::TY_C, InputArg));
      }
      A->claim();
    } else if (A->getOption().matches(options::OPT__SLASH_Tp)) {
      StringRef Value = A->getValue();
      if (DiagnoseInputExistence(*this, Args, Value, types::TY_CXX)) {
        Arg *InputArg = MakeInputArg(Args, Opts, A->getValue());
        Inputs.push_back(std::make_pair(types::TY_CXX, InputArg));
      }
      A->claim();
    } else if (A->getOption().hasFlag(options::LinkerInput)) {
      // Just treat as object type, we could make a special type for this if
      // necessary.
      Inputs.push_back(std::make_pair(types::TY_Object, A));

    } else if (A->getOption().matches(options::OPT_x)) {
      InputTypeArg = A;
      InputType = types::lookupTypeForTypeSpecifier(A->getValue());
      A->claim();

      // Follow gcc behavior and treat as linker input for invalid -x
      // options. Its not clear why we shouldn't just revert to unknown; but
      // this isn't very important, we might as well be bug compatible.
      if (!InputType) {
        Diag(clang::diag::err_drv_unknown_language) << A->getValue();
        InputType = types::TY_Object;
      }
    }
  }
  if (CCCIsCPP() && Inputs.empty()) {
    // If called as standalone preprocessor, stdin is processed
    // if no other input is present.
    Arg *A = MakeInputArg(Args, Opts, "-");
    Inputs.push_back(std::make_pair(types::TY_C, A));
  }
}

// For each unique --cuda-gpu-arch= argument creates a TY_CUDA_DEVICE
// input action and then wraps each in CudaDeviceAction paired with
// appropriate GPU arch name. In case of partial (i.e preprocessing
// only) or device-only compilation, each device action is added to /p
// Actions and /p Current is released. Otherwise the function creates
// and returns a new CudaHostAction which wraps /p Current and device
// side actions.
static Action *buildCudaActions(Compilation &C, DerivedArgList &Args,
                                const Arg *InputArg, Action *HostAction,
                                ActionList &Actions) {
  Arg *PartialCompilationArg = Args.getLastArg(
      options::OPT_cuda_host_only, options::OPT_cuda_device_only,
      options::OPT_cuda_compile_host_device);
  bool CompileHostOnly =
      PartialCompilationArg &&
      PartialCompilationArg->getOption().matches(options::OPT_cuda_host_only);
  bool CompileDeviceOnly =
      PartialCompilationArg &&
      PartialCompilationArg->getOption().matches(options::OPT_cuda_device_only);
  const ToolChain *HostTC = C.getSingleOffloadToolChain<Action::OFK_Host>();
  assert(HostTC && "No toolchain for host compilation.");
  if (HostTC->getTriple().isNVPTX()) {
    // We do not support targeting NVPTX for host compilation. Throw
    // an error and abort pipeline construction early so we don't trip
    // asserts that assume device-side compilation.
    C.getDriver().Diag(diag::err_drv_cuda_nvptx_host);
    return nullptr;
  }

  if (CompileHostOnly) {
    OffloadAction::HostDependence HDep(*HostAction, *HostTC,
                                       /*BoundArch=*/nullptr, Action::OFK_Cuda);
    return C.MakeAction<OffloadAction>(HDep);
  }

  // Collect all cuda_gpu_arch parameters, removing duplicates.
  SmallVector<CudaArch, 4> GpuArchList;
  llvm::SmallSet<CudaArch, 4> GpuArchs;
  for (Arg *A : Args) {
    if (!A->getOption().matches(options::OPT_cuda_gpu_arch_EQ))
      continue;
    A->claim();

    const auto &ArchStr = A->getValue();
    CudaArch Arch = StringToCudaArch(ArchStr);
    if (Arch == CudaArch::UNKNOWN)
      C.getDriver().Diag(clang::diag::err_drv_cuda_bad_gpu_arch) << ArchStr;
    else if (GpuArchs.insert(Arch).second)
      GpuArchList.push_back(Arch);
  }

  // Default to sm_20 which is the lowest common denominator for supported GPUs.
  // sm_20 code should work correctly, if suboptimally, on all newer GPUs.
  if (GpuArchList.empty())
    GpuArchList.push_back(CudaArch::SM_20);

  // Replicate inputs for each GPU architecture.
  Driver::InputList CudaDeviceInputs;
  for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I)
    CudaDeviceInputs.push_back(std::make_pair(types::TY_CUDA_DEVICE, InputArg));

  // Build actions for all device inputs.
  ActionList CudaDeviceActions;
  C.getDriver().BuildActions(C, Args, CudaDeviceInputs, CudaDeviceActions);
  assert(GpuArchList.size() == CudaDeviceActions.size() &&
         "Failed to create actions for all devices");

  // Check whether any of device actions stopped before they could generate PTX.
  bool PartialCompilation =
      llvm::any_of(CudaDeviceActions, [](const Action *a) {
        return a->getKind() != Action::AssembleJobClass;
      });

  const ToolChain *CudaTC = C.getSingleOffloadToolChain<Action::OFK_Cuda>();

  // Figure out what to do with device actions -- pass them as inputs to the
  // host action or run each of them independently.
  if (PartialCompilation || CompileDeviceOnly) {
    // In case of partial or device-only compilation results of device actions
    // are not consumed by the host action device actions have to be added to
    // top-level actions list with AtTopLevel=true and run independently.

    // -o is ambiguous if we have more than one top-level action.
    if (Args.hasArg(options::OPT_o) &&
        (!CompileDeviceOnly || GpuArchList.size() > 1)) {
      C.getDriver().Diag(
          clang::diag::err_drv_output_argument_with_multiple_files);
      return nullptr;
    }

    for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I) {
      OffloadAction::DeviceDependences DDep;
      DDep.add(*CudaDeviceActions[I], *CudaTC, CudaArchToString(GpuArchList[I]),
               Action::OFK_Cuda);
      Actions.push_back(
          C.MakeAction<OffloadAction>(DDep, CudaDeviceActions[I]->getType()));
    }
    // Kill host action in case of device-only compilation.
    if (CompileDeviceOnly)
      return nullptr;
    return HostAction;
  }

  // If we're not a partial or device-only compilation, we compile each arch to
  // ptx and assemble to cubin, then feed the cubin *and* the ptx into a device
  // "link" action, which uses fatbinary to combine these cubins into one
  // fatbin.  The fatbin is then an input to the host compilation.
  ActionList DeviceActions;
  for (unsigned I = 0, E = GpuArchList.size(); I != E; ++I) {
    Action* AssembleAction = CudaDeviceActions[I];
    assert(AssembleAction->getType() == types::TY_Object);
    assert(AssembleAction->getInputs().size() == 1);

    Action* BackendAction = AssembleAction->getInputs()[0];
    assert(BackendAction->getType() == types::TY_PP_Asm);

    for (auto &A : {AssembleAction, BackendAction}) {
      OffloadAction::DeviceDependences DDep;
      DDep.add(*A, *CudaTC, CudaArchToString(GpuArchList[I]), Action::OFK_Cuda);
      DeviceActions.push_back(C.MakeAction<OffloadAction>(DDep, A->getType()));
    }
  }
  auto FatbinAction =
      C.MakeAction<LinkJobAction>(DeviceActions, types::TY_CUDA_FATBIN);

  // Return a new host action that incorporates original host action and all
  // device actions.
  OffloadAction::HostDependence HDep(*HostAction, *HostTC,
                                     /*BoundArch=*/nullptr, Action::OFK_Cuda);
  OffloadAction::DeviceDependences DDep;
  DDep.add(*FatbinAction, *CudaTC, /*BoundArch=*/nullptr, Action::OFK_Cuda);
  return C.MakeAction<OffloadAction>(HDep, DDep);
}

void Driver::BuildActions(Compilation &C, DerivedArgList &Args,
                          const InputList &Inputs, ActionList &Actions) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation actions");

  if (!SuppressMissingInputWarning && Inputs.empty()) {
    Diag(clang::diag::err_drv_no_input_files);
    return;
  }

  Arg *FinalPhaseArg;
  phases::ID FinalPhase = getFinalPhase(Args, &FinalPhaseArg);

  if (FinalPhase == phases::Link && Args.hasArg(options::OPT_emit_llvm)) {
    Diag(clang::diag::err_drv_emit_llvm_link);
  }

  // Reject -Z* at the top level, these options should never have been exposed
  // by gcc.
  if (Arg *A = Args.getLastArg(options::OPT_Z_Joined))
    Diag(clang::diag::err_drv_use_of_Z_option) << A->getAsString(Args);

  // Diagnose misuse of /Fo.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_Fo)) {
    StringRef V = A->getValue();
    if (Inputs.size() > 1 && !V.empty() &&
        !llvm::sys::path::is_separator(V.back())) {
      // Check whether /Fo tries to name an output file for multiple inputs.
      Diag(clang::diag::err_drv_out_file_argument_with_multiple_sources)
          << A->getSpelling() << V;
      Args.eraseArg(options::OPT__SLASH_Fo);
    }
  }

  // Diagnose misuse of /Fa.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_Fa)) {
    StringRef V = A->getValue();
    if (Inputs.size() > 1 && !V.empty() &&
        !llvm::sys::path::is_separator(V.back())) {
      // Check whether /Fa tries to name an asm file for multiple inputs.
      Diag(clang::diag::err_drv_out_file_argument_with_multiple_sources)
          << A->getSpelling() << V;
      Args.eraseArg(options::OPT__SLASH_Fa);
    }
  }

  // Diagnose misuse of /o.
  if (Arg *A = Args.getLastArg(options::OPT__SLASH_o)) {
    if (A->getValue()[0] == '\0') {
      // It has to have a value.
      Diag(clang::diag::err_drv_missing_argument) << A->getSpelling() << 1;
      Args.eraseArg(options::OPT__SLASH_o);
    }
  }

  // Diagnose unsupported forms of /Yc /Yu. Ignore /Yc/Yu for now if:
  // * no filename after it
  // * both /Yc and /Yu passed but with different filenames
  // * corresponding file not also passed as /FI
  Arg *YcArg = Args.getLastArg(options::OPT__SLASH_Yc);
  Arg *YuArg = Args.getLastArg(options::OPT__SLASH_Yu);
  if (YcArg && YcArg->getValue()[0] == '\0') {
    Diag(clang::diag::warn_drv_ycyu_no_arg_clang_cl) << YcArg->getSpelling();
    Args.eraseArg(options::OPT__SLASH_Yc);
    YcArg = nullptr;
  }
  if (YuArg && YuArg->getValue()[0] == '\0') {
    Diag(clang::diag::warn_drv_ycyu_no_arg_clang_cl) << YuArg->getSpelling();
    Args.eraseArg(options::OPT__SLASH_Yu);
    YuArg = nullptr;
  }
  if (YcArg && YuArg && strcmp(YcArg->getValue(), YuArg->getValue()) != 0) {
    Diag(clang::diag::warn_drv_ycyu_different_arg_clang_cl);
    Args.eraseArg(options::OPT__SLASH_Yc);
    Args.eraseArg(options::OPT__SLASH_Yu);
    YcArg = YuArg = nullptr;
  }
  if (YcArg || YuArg) {
    StringRef Val = YcArg ? YcArg->getValue() : YuArg->getValue();
    bool FoundMatchingInclude = false;
    for (const Arg *Inc : Args.filtered(options::OPT_include)) {
      // FIXME: Do case-insensitive matching and consider / and \ as equal.
      if (Inc->getValue() == Val)
        FoundMatchingInclude = true;
    }
    if (!FoundMatchingInclude) {
      Diag(clang::diag::warn_drv_ycyu_no_fi_arg_clang_cl)
          << (YcArg ? YcArg : YuArg)->getSpelling();
      Args.eraseArg(options::OPT__SLASH_Yc);
      Args.eraseArg(options::OPT__SLASH_Yu);
      YcArg = YuArg = nullptr;
    }
  }
  if (YcArg && Inputs.size() > 1) {
    Diag(clang::diag::warn_drv_yc_multiple_inputs_clang_cl);
    Args.eraseArg(options::OPT__SLASH_Yc);
    YcArg = nullptr;
  }
  if (Args.hasArg(options::OPT__SLASH_Y_)) {
    // /Y- disables all pch handling.  Rather than check for it everywhere,
    // just remove clang-cl pch-related flags here.
    Args.eraseArg(options::OPT__SLASH_Fp);
    Args.eraseArg(options::OPT__SLASH_Yc);
    Args.eraseArg(options::OPT__SLASH_Yu);
    YcArg = YuArg = nullptr;
  }

  // Track the host offload kinds used on this compilation.
  unsigned CompilationActiveOffloadHostKinds = 0u;

  // Construct the actions to perform.
  ActionList LinkerInputs;

  llvm::SmallVector<phases::ID, phases::MaxNumberOfPhases> PL;
  for (auto &I : Inputs) {
    types::ID InputType = I.first;
    const Arg *InputArg = I.second;

    PL.clear();
    types::getCompilationPhases(InputType, PL);

    // If the first step comes after the final phase we are doing as part of
    // this compilation, warn the user about it.
    phases::ID InitialPhase = PL[0];
    if (InitialPhase > FinalPhase) {
      // Claim here to avoid the more general unused warning.
      InputArg->claim();

      // Suppress all unused style warnings with -Qunused-arguments
      if (Args.hasArg(options::OPT_Qunused_arguments))
        continue;

      // Special case when final phase determined by binary name, rather than
      // by a command-line argument with a corresponding Arg.
      if (CCCIsCPP())
        Diag(clang::diag::warn_drv_input_file_unused_by_cpp)
            << InputArg->getAsString(Args) << getPhaseName(InitialPhase);
      // Special case '-E' warning on a previously preprocessed file to make
      // more sense.
      else if (InitialPhase == phases::Compile &&
               FinalPhase == phases::Preprocess &&
               getPreprocessedType(InputType) == types::TY_INVALID)
        Diag(clang::diag::warn_drv_preprocessed_input_file_unused)
            << InputArg->getAsString(Args) << !!FinalPhaseArg
            << (FinalPhaseArg ? FinalPhaseArg->getOption().getName() : "");
      else
        Diag(clang::diag::warn_drv_input_file_unused)
            << InputArg->getAsString(Args) << getPhaseName(InitialPhase)
            << !!FinalPhaseArg
            << (FinalPhaseArg ? FinalPhaseArg->getOption().getName() : "");
      continue;
    }

    if (YcArg) {
      // Add a separate precompile phase for the compile phase.
      if (FinalPhase >= phases::Compile) {
        llvm::SmallVector<phases::ID, phases::MaxNumberOfPhases> PCHPL;
        types::getCompilationPhases(types::TY_CXXHeader, PCHPL);
        Arg *PchInputArg = MakeInputArg(Args, Opts, YcArg->getValue());

        // Build the pipeline for the pch file.
        Action *ClangClPch = C.MakeAction<InputAction>(*PchInputArg, InputType);
        for (phases::ID Phase : PCHPL)
          ClangClPch = ConstructPhaseAction(C, Args, Phase, ClangClPch);
        assert(ClangClPch);
        Actions.push_back(ClangClPch);
        // The driver currently exits after the first failed command.  This
        // relies on that behavior, to make sure if the pch generation fails,
        // the main compilation won't run.
      }
    }

    phases::ID CudaInjectionPhase =
        (phases::Compile < FinalPhase &&
         llvm::find(PL, phases::Compile) != PL.end())
            ? phases::Compile
            : FinalPhase;

    // Track the host offload kinds used on this input.
    unsigned InputActiveOffloadHostKinds = 0u;

    // Build the pipeline for this file.
    Action *Current = C.MakeAction<InputAction>(*InputArg, InputType);
    for (SmallVectorImpl<phases::ID>::iterator i = PL.begin(), e = PL.end();
         i != e; ++i) {
      phases::ID Phase = *i;

      // We are done if this step is past what the user requested.
      if (Phase > FinalPhase)
        break;

      // Queue linker inputs.
      if (Phase == phases::Link) {
        assert((i + 1) == e && "linking must be final compilation step.");
        LinkerInputs.push_back(Current);
        Current = nullptr;
        break;
      }

      // Some types skip the assembler phase (e.g., llvm-bc), but we can't
      // encode this in the steps because the intermediate type depends on
      // arguments. Just special case here.
      if (Phase == phases::Assemble && Current->getType() != types::TY_PP_Asm)
        continue;

      // Otherwise construct the appropriate action.
      Current = ConstructPhaseAction(C, Args, Phase, Current);

      if (InputType == types::TY_CUDA && Phase == CudaInjectionPhase) {
        Current = buildCudaActions(C, Args, InputArg, Current, Actions);
        if (!Current)
          break;

        // We produced a CUDA action for this input, so the host has to support
        // CUDA.
        InputActiveOffloadHostKinds |= Action::OFK_Cuda;
        CompilationActiveOffloadHostKinds |= Action::OFK_Cuda;
      }

      if (Current->getType() == types::TY_Nothing)
        break;
    }

    // If we ended with something, add to the output list. Also, propagate the
    // offload information to the top-level host action related with the current
    // input.
    if (Current) {
      if (InputActiveOffloadHostKinds)
        Current->propagateHostOffloadInfo(InputActiveOffloadHostKinds,
                                          /*BoundArch=*/nullptr);
      Actions.push_back(Current);
    }
  }

  // Add a link action if necessary and propagate the offload information for
  // the current compilation.
  if (!LinkerInputs.empty()) {
    Actions.push_back(
        C.MakeAction<LinkJobAction>(LinkerInputs, types::TY_Image));
    Actions.back()->propagateHostOffloadInfo(CompilationActiveOffloadHostKinds,
                                             /*BoundArch=*/nullptr);
  }

  // If we are linking, claim any options which are obviously only used for
  // compilation.
  if (FinalPhase == phases::Link && PL.size() == 1) {
    Args.ClaimAllArgs(options::OPT_CompileOnly_Group);
    Args.ClaimAllArgs(options::OPT_cl_compile_Group);
  }

  // Claim ignored clang-cl options.
  Args.ClaimAllArgs(options::OPT_cl_ignored_Group);

  // Claim --cuda-host-only and --cuda-compile-host-device, which may be passed
  // to non-CUDA compilations and should not trigger warnings there.
  Args.ClaimAllArgs(options::OPT_cuda_host_only);
  Args.ClaimAllArgs(options::OPT_cuda_compile_host_device);
}

Action *Driver::ConstructPhaseAction(Compilation &C, const ArgList &Args,
                                     phases::ID Phase, Action *Input) const {
  llvm::PrettyStackTraceString CrashInfo("Constructing phase actions");
  // Build the appropriate action.
  switch (Phase) {
  case phases::Link:
    llvm_unreachable("link action invalid here.");
  case phases::Preprocess: {
    types::ID OutputTy;
    // -{M, MM} alter the output type.
    if (Args.hasArg(options::OPT_M, options::OPT_MM)) {
      OutputTy = types::TY_Dependencies;
    } else {
      OutputTy = Input->getType();
      if (!Args.hasFlag(options::OPT_frewrite_includes,
                        options::OPT_fno_rewrite_includes, false) &&
          !CCGenDiagnostics)
        OutputTy = types::getPreprocessedType(OutputTy);
      assert(OutputTy != types::TY_INVALID &&
             "Cannot preprocess this input type!");
    }
    return C.MakeAction<PreprocessJobAction>(Input, OutputTy);
  }
  case phases::Precompile: {
    types::ID OutputTy = types::TY_PCH;
    if (Args.hasArg(options::OPT_fsyntax_only)) {
      // Syntax checks should not emit a PCH file
      OutputTy = types::TY_Nothing;
    }
    return C.MakeAction<PrecompileJobAction>(Input, OutputTy);
  }
  case phases::Compile: {
    if (Args.hasArg(options::OPT_fsyntax_only))
      return C.MakeAction<CompileJobAction>(Input, types::TY_Nothing);
    if (Args.hasArg(options::OPT_rewrite_objc))
      return C.MakeAction<CompileJobAction>(Input, types::TY_RewrittenObjC);
    if (Args.hasArg(options::OPT_rewrite_legacy_objc))
      return C.MakeAction<CompileJobAction>(Input,
                                            types::TY_RewrittenLegacyObjC);
    if (Args.hasArg(options::OPT__analyze, options::OPT__analyze_auto))
      return C.MakeAction<AnalyzeJobAction>(Input, types::TY_Plist);
    if (Args.hasArg(options::OPT__migrate))
      return C.MakeAction<MigrateJobAction>(Input, types::TY_Remap);
    if (Args.hasArg(options::OPT_emit_ast))
      return C.MakeAction<CompileJobAction>(Input, types::TY_AST);
    if (Args.hasArg(options::OPT_module_file_info))
      return C.MakeAction<CompileJobAction>(Input, types::TY_ModuleFile);
    if (Args.hasArg(options::OPT_verify_pch))
      return C.MakeAction<VerifyPCHJobAction>(Input, types::TY_Nothing);
    return C.MakeAction<CompileJobAction>(Input, types::TY_LLVM_BC);
  }
  case phases::Backend: {
    if (isUsingLTO()) {
      types::ID Output =
          Args.hasArg(options::OPT_S) ? types::TY_LTO_IR : types::TY_LTO_BC;
      return C.MakeAction<BackendJobAction>(Input, Output);
    }
    if (Args.hasArg(options::OPT_emit_llvm)) {
      types::ID Output =
          Args.hasArg(options::OPT_S) ? types::TY_LLVM_IR : types::TY_LLVM_BC;
      return C.MakeAction<BackendJobAction>(Input, Output);
    }
    return C.MakeAction<BackendJobAction>(Input, types::TY_PP_Asm);
  }
  case phases::Assemble:
    return C.MakeAction<AssembleJobAction>(std::move(Input), types::TY_Object);
  }

  llvm_unreachable("invalid phase in ConstructPhaseAction");
}

void Driver::BuildJobs(Compilation &C) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs");

  Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o);

  // It is an error to provide a -o option if we are making multiple output
  // files.
  if (FinalOutput) {
    unsigned NumOutputs = 0;
    for (const Action *A : C.getActions())
      if (A->getType() != types::TY_Nothing)
        ++NumOutputs;

    if (NumOutputs > 1) {
      Diag(clang::diag::err_drv_output_argument_with_multiple_files);
      FinalOutput = nullptr;
    }
  }

  // Collect the list of architectures.
  llvm::StringSet<> ArchNames;
  if (C.getDefaultToolChain().getTriple().isOSBinFormatMachO())
    for (const Arg *A : C.getArgs())
      if (A->getOption().matches(options::OPT_arch))
        ArchNames.insert(A->getValue());

  // Set of (Action, canonical ToolChain triple) pairs we've built jobs for.
  std::map<std::pair<const Action *, std::string>, InputInfo> CachedResults;
  for (Action *A : C.getActions()) {
    // If we are linking an image for multiple archs then the linker wants
    // -arch_multiple and -final_output <final image name>. Unfortunately, this
    // doesn't fit in cleanly because we have to pass this information down.
    //
    // FIXME: This is a hack; find a cleaner way to integrate this into the
    // process.
    const char *LinkingOutput = nullptr;
    if (isa<LipoJobAction>(A)) {
      if (FinalOutput)
        LinkingOutput = FinalOutput->getValue();
      else
        LinkingOutput = getDefaultImageName();
    }

    BuildJobsForAction(C, A, &C.getDefaultToolChain(),
                       /*BoundArch*/ nullptr,
                       /*AtTopLevel*/ true,
                       /*MultipleArchs*/ ArchNames.size() > 1,
                       /*LinkingOutput*/ LinkingOutput, CachedResults,
                       /*BuildForOffloadDevice*/ false);
  }

  // If the user passed -Qunused-arguments or there were errors, don't warn
  // about any unused arguments.
  if (Diags.hasErrorOccurred() ||
      C.getArgs().hasArg(options::OPT_Qunused_arguments))
    return;

  // Claim -### here.
  (void)C.getArgs().hasArg(options::OPT__HASH_HASH_HASH);

  // Claim --driver-mode, --rsp-quoting, it was handled earlier.
  (void)C.getArgs().hasArg(options::OPT_driver_mode);
  (void)C.getArgs().hasArg(options::OPT_rsp_quoting);

  for (Arg *A : C.getArgs()) {
    // FIXME: It would be nice to be able to send the argument to the
    // DiagnosticsEngine, so that extra values, position, and so on could be
    // printed.
    if (!A->isClaimed()) {
      if (A->getOption().hasFlag(options::NoArgumentUnused))
        continue;

      // Suppress the warning automatically if this is just a flag, and it is an
      // instance of an argument we already claimed.
      const Option &Opt = A->getOption();
      if (Opt.getKind() == Option::FlagClass) {
        bool DuplicateClaimed = false;

        for (const Arg *AA : C.getArgs().filtered(&Opt)) {
          if (AA->isClaimed()) {
            DuplicateClaimed = true;
            break;
          }
        }

        if (DuplicateClaimed)
          continue;
      }

      // In clang-cl, don't mention unknown arguments here since they have
      // already been warned about.
      if (!IsCLMode() || !A->getOption().matches(options::OPT_UNKNOWN))
        Diag(clang::diag::warn_drv_unused_argument)
            << A->getAsString(C.getArgs());
    }
  }
}
/// Collapse an offloading action looking for a job of the given type. The input
/// action is changed to the input of the collapsed sequence. If we effectively
/// had a collapse return the corresponding offloading action, otherwise return
/// null.
template <typename T>
static OffloadAction *collapseOffloadingAction(Action *&CurAction) {
  if (!CurAction)
    return nullptr;
  if (auto *OA = dyn_cast<OffloadAction>(CurAction)) {
    if (OA->hasHostDependence())
      if (auto *HDep = dyn_cast<T>(OA->getHostDependence())) {
        CurAction = HDep;
        return OA;
      }
    if (OA->hasSingleDeviceDependence())
      if (auto *DDep = dyn_cast<T>(OA->getSingleDeviceDependence())) {
        CurAction = DDep;
        return OA;
      }
  }
  return nullptr;
}
// Returns a Tool for a given JobAction.  In case the action and its
// predecessors can be combined, updates Inputs with the inputs of the
// first combined action. If one of the collapsed actions is a
// CudaHostAction, updates CollapsedCHA with the pointer to it so the
// caller can deal with extra handling such action requires.
static const Tool *selectToolForJob(Compilation &C, bool SaveTemps,
                                    bool EmbedBitcode, const ToolChain *TC,
                                    const JobAction *JA,
                                    const ActionList *&Inputs,
                                    ActionList &CollapsedOffloadAction) {
  const Tool *ToolForJob = nullptr;
  CollapsedOffloadAction.clear();

  // See if we should look for a compiler with an integrated assembler. We match
  // bottom up, so what we are actually looking for is an assembler job with a
  // compiler input.

  // Look through offload actions between assembler and backend actions.
  Action *BackendJA = (isa<AssembleJobAction>(JA) && Inputs->size() == 1)
                          ? *Inputs->begin()
                          : nullptr;
  auto *BackendOA = collapseOffloadingAction<BackendJobAction>(BackendJA);

  if (TC->useIntegratedAs() && !SaveTemps &&
      !C.getArgs().hasArg(options::OPT_via_file_asm) &&
      !C.getArgs().hasArg(options::OPT__SLASH_FA) &&
      !C.getArgs().hasArg(options::OPT__SLASH_Fa) && BackendJA &&
      isa<BackendJobAction>(BackendJA)) {
    // A BackendJob is always preceded by a CompileJob, and without -save-temps
    // or -fembed-bitcode, they will always get combined together, so instead of
    // checking the backend tool, check if the tool for the CompileJob has an
    // integrated assembler. For -fembed-bitcode, CompileJob is still used to
    // look up tools for BackendJob, but they need to match before we can split
    // them.

    // Look through offload actions between backend and compile actions.
    Action *CompileJA = *BackendJA->getInputs().begin();
    auto *CompileOA = collapseOffloadingAction<CompileJobAction>(CompileJA);

    assert(CompileJA && isa<CompileJobAction>(CompileJA) &&
           "Backend job is not preceeded by compile job.");
    const Tool *Compiler = TC->SelectTool(*cast<CompileJobAction>(CompileJA));
    if (!Compiler)
      return nullptr;
    // When using -fembed-bitcode, it is required to have the same tool (clang)
    // for both CompilerJA and BackendJA. Otherwise, combine two stages.
    if (EmbedBitcode) {
      JobAction *InputJA = cast<JobAction>(*Inputs->begin());
      const Tool *BackendTool = TC->SelectTool(*InputJA);
      if (BackendTool == Compiler)
        CompileJA = InputJA;
    }
    if (Compiler->hasIntegratedAssembler()) {
      Inputs = &CompileJA->getInputs();
      ToolForJob = Compiler;
      // Save the collapsed offload actions because they may still contain
      // device actions.
      if (CompileOA)
        CollapsedOffloadAction.push_back(CompileOA);
      if (BackendOA)
        CollapsedOffloadAction.push_back(BackendOA);
    }
  }

  // A backend job should always be combined with the preceding compile job
  // unless OPT_save_temps or OPT_fembed_bitcode is enabled and the compiler is
  // capable of emitting LLVM IR as an intermediate output.
  if (isa<BackendJobAction>(JA)) {
    // Check if the compiler supports emitting LLVM IR.
    assert(Inputs->size() == 1);

    // Look through offload actions between backend and compile actions.
    Action *CompileJA = *JA->getInputs().begin();
    auto *CompileOA = collapseOffloadingAction<CompileJobAction>(CompileJA);

    assert(CompileJA && isa<CompileJobAction>(CompileJA) &&
           "Backend job is not preceeded by compile job.");
    const Tool *Compiler = TC->SelectTool(*cast<CompileJobAction>(CompileJA));
    if (!Compiler)
      return nullptr;
    if (!Compiler->canEmitIR() ||
        (!SaveTemps && !EmbedBitcode)) {
      Inputs = &CompileJA->getInputs();
      ToolForJob = Compiler;

      if (CompileOA)
        CollapsedOffloadAction.push_back(CompileOA);
    }
  }

  // Otherwise use the tool for the current job.
  if (!ToolForJob)
    ToolForJob = TC->SelectTool(*JA);

  // See if we should use an integrated preprocessor. We do so when we have
  // exactly one input, since this is the only use case we care about
  // (irrelevant since we don't support combine yet).

  // Look through offload actions after preprocessing.
  Action *PreprocessJA = (Inputs->size() == 1) ? *Inputs->begin() : nullptr;
  auto *PreprocessOA =
      collapseOffloadingAction<PreprocessJobAction>(PreprocessJA);

  if (PreprocessJA && isa<PreprocessJobAction>(PreprocessJA) &&
      !C.getArgs().hasArg(options::OPT_no_integrated_cpp) &&
      !C.getArgs().hasArg(options::OPT_traditional_cpp) && !SaveTemps &&
      !C.getArgs().hasArg(options::OPT_rewrite_objc) &&
      ToolForJob->hasIntegratedCPP()) {
    Inputs = &PreprocessJA->getInputs();
    if (PreprocessOA)
      CollapsedOffloadAction.push_back(PreprocessOA);
  }

  return ToolForJob;
}

InputInfo Driver::BuildJobsForAction(
    Compilation &C, const Action *A, const ToolChain *TC, const char *BoundArch,
    bool AtTopLevel, bool MultipleArchs, const char *LinkingOutput,
    std::map<std::pair<const Action *, std::string>, InputInfo> &CachedResults,
    bool BuildForOffloadDevice) const {
  // The bound arch is not necessarily represented in the toolchain's triple --
  // for example, armv7 and armv7s both map to the same triple -- so we need
  // both in our map.
  std::string TriplePlusArch = TC->getTriple().normalize();
  if (BoundArch) {
    TriplePlusArch += "-";
    TriplePlusArch += BoundArch;
  }
  std::pair<const Action *, std::string> ActionTC = {A, TriplePlusArch};
  auto CachedResult = CachedResults.find(ActionTC);
  if (CachedResult != CachedResults.end()) {
    return CachedResult->second;
  }
  InputInfo Result = BuildJobsForActionNoCache(
      C, A, TC, BoundArch, AtTopLevel, MultipleArchs, LinkingOutput,
      CachedResults, BuildForOffloadDevice);
  CachedResults[ActionTC] = Result;
  return Result;
}

InputInfo Driver::BuildJobsForActionNoCache(
    Compilation &C, const Action *A, const ToolChain *TC, const char *BoundArch,
    bool AtTopLevel, bool MultipleArchs, const char *LinkingOutput,
    std::map<std::pair<const Action *, std::string>, InputInfo> &CachedResults,
    bool BuildForOffloadDevice) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs");

  InputInfoList OffloadDependencesInputInfo;
  if (const OffloadAction *OA = dyn_cast<OffloadAction>(A)) {
    // The offload action is expected to be used in four different situations.
    //
    // a) Set a toolchain/architecture/kind for a host action:
    //    Host Action 1 -> OffloadAction -> Host Action 2
    //
    // b) Set a toolchain/architecture/kind for a device action;
    //    Device Action 1 -> OffloadAction -> Device Action 2
    //
    // c) Specify a device dependences to a host action;
    //    Device Action 1  _
    //                      \
    //      Host Action 1  ---> OffloadAction -> Host Action 2
    //
    // d) Specify a host dependence to a device action.
    //      Host Action 1  _
    //                      \
    //    Device Action 1  ---> OffloadAction -> Device Action 2
    //
    // For a) and b), we just return the job generated for the dependence. For
    // c) and d) we override the current action with the host/device dependence
    // if the current toolchain is host/device and set the offload dependences
    // info with the jobs obtained from the device/host dependence(s).

    // If there is a single device option, just generate the job for it.
    if (OA->hasSingleDeviceDependence()) {
      InputInfo DevA;
      OA->doOnEachDeviceDependence([&](Action *DepA, const ToolChain *DepTC,
                                       const char *DepBoundArch) {
        DevA =
            BuildJobsForAction(C, DepA, DepTC, DepBoundArch, AtTopLevel,
                               /*MultipleArchs*/ !!DepBoundArch, LinkingOutput,
                               CachedResults, /*BuildForOffloadDevice=*/true);
      });
      return DevA;
    }

    // If 'Action 2' is host, we generate jobs for the device dependences and
    // override the current action with the host dependence. Otherwise, we
    // generate the host dependences and override the action with the device
    // dependence. The dependences can't therefore be a top-level action.
    OA->doOnEachDependence(
        /*IsHostDependence=*/BuildForOffloadDevice,
        [&](Action *DepA, const ToolChain *DepTC, const char *DepBoundArch) {
          OffloadDependencesInputInfo.push_back(BuildJobsForAction(
              C, DepA, DepTC, DepBoundArch, /*AtTopLevel=*/false,
              /*MultipleArchs*/ !!DepBoundArch, LinkingOutput, CachedResults,
              /*BuildForOffloadDevice=*/DepA->getOffloadingDeviceKind() !=
                  Action::OFK_None));
        });

    A = BuildForOffloadDevice
            ? OA->getSingleDeviceDependence(/*DoNotConsiderHostActions=*/true)
            : OA->getHostDependence();
  }

  if (const InputAction *IA = dyn_cast<InputAction>(A)) {
    // FIXME: It would be nice to not claim this here; maybe the old scheme of
    // just using Args was better?
    const Arg &Input = IA->getInputArg();
    Input.claim();
    if (Input.getOption().matches(options::OPT_INPUT)) {
      const char *Name = Input.getValue();
      return InputInfo(A, Name, /* BaseInput = */ Name);
    }
    return InputInfo(A, &Input, /* BaseInput = */ "");
  }

  if (const BindArchAction *BAA = dyn_cast<BindArchAction>(A)) {
    const ToolChain *TC;
    const char *ArchName = BAA->getArchName();

    if (ArchName)
      TC = &getToolChain(C.getArgs(),
                         computeTargetTriple(*this, DefaultTargetTriple,
                                             C.getArgs(), ArchName));
    else
      TC = &C.getDefaultToolChain();

    return BuildJobsForAction(C, *BAA->input_begin(), TC, ArchName, AtTopLevel,
                              MultipleArchs, LinkingOutput, CachedResults,
                              BuildForOffloadDevice);
  }


  const ActionList *Inputs = &A->getInputs();

  const JobAction *JA = cast<JobAction>(A);
  ActionList CollapsedOffloadActions;

  const Tool *T =
      selectToolForJob(C, isSaveTempsEnabled(), embedBitcodeEnabled(), TC, JA,
                       Inputs, CollapsedOffloadActions);
  if (!T)
    return InputInfo();

  // If we've collapsed action list that contained OffloadAction we
  // need to build jobs for host/device-side inputs it may have held.
  for (const auto *OA : CollapsedOffloadActions)
    cast<OffloadAction>(OA)->doOnEachDependence(
        /*IsHostDependence=*/BuildForOffloadDevice,
        [&](Action *DepA, const ToolChain *DepTC, const char *DepBoundArch) {
          OffloadDependencesInputInfo.push_back(BuildJobsForAction(
              C, DepA, DepTC, DepBoundArch, /* AtTopLevel */ false,
              /*MultipleArchs=*/!!DepBoundArch, LinkingOutput, CachedResults,
              /*BuildForOffloadDevice=*/DepA->getOffloadingDeviceKind() !=
                  Action::OFK_None));
        });

  // Only use pipes when there is exactly one input.
  InputInfoList InputInfos;
  for (const Action *Input : *Inputs) {
    // Treat dsymutil and verify sub-jobs as being at the top-level too, they
    // shouldn't get temporary output names.
    // FIXME: Clean this up.
    bool SubJobAtTopLevel =
        AtTopLevel && (isa<DsymutilJobAction>(A) || isa<VerifyJobAction>(A));
    InputInfos.push_back(BuildJobsForAction(
        C, Input, TC, BoundArch, SubJobAtTopLevel, MultipleArchs, LinkingOutput,
        CachedResults, BuildForOffloadDevice));
  }

  // Always use the first input as the base input.
  const char *BaseInput = InputInfos[0].getBaseInput();

  // ... except dsymutil actions, which use their actual input as the base
  // input.
  if (JA->getType() == types::TY_dSYM)
    BaseInput = InputInfos[0].getFilename();

  // Append outputs of offload device jobs to the input list
  if (!OffloadDependencesInputInfo.empty())
    InputInfos.append(OffloadDependencesInputInfo.begin(),
                      OffloadDependencesInputInfo.end());

  // Set the effective triple of the toolchain for the duration of this job.
  llvm::Triple EffectiveTriple;
  const ToolChain &ToolTC = T->getToolChain();
  const ArgList &Args = C.getArgsForToolChain(TC, BoundArch);
  if (InputInfos.size() != 1) {
    EffectiveTriple = llvm::Triple(ToolTC.ComputeEffectiveClangTriple(Args));
  } else {
    // Pass along the input type if it can be unambiguously determined.
    EffectiveTriple = llvm::Triple(
        ToolTC.ComputeEffectiveClangTriple(Args, InputInfos[0].getType()));
  }
  RegisterEffectiveTriple TripleRAII(ToolTC, EffectiveTriple);

  // Determine the place to write output to, if any.
  InputInfo Result;
  if (JA->getType() == types::TY_Nothing)
    Result = InputInfo(A, BaseInput);
  else
    Result = InputInfo(A, GetNamedOutputPath(C, *JA, BaseInput, BoundArch,
                                             AtTopLevel, MultipleArchs,
                                             TC->getTriple().normalize()),
                       BaseInput);

  if (CCCPrintBindings && !CCGenDiagnostics) {
    llvm::errs() << "# \"" << T->getToolChain().getTripleString() << '"'
                 << " - \"" << T->getName() << "\", inputs: [";
    for (unsigned i = 0, e = InputInfos.size(); i != e; ++i) {
      llvm::errs() << InputInfos[i].getAsString();
      if (i + 1 != e)
        llvm::errs() << ", ";
    }
    llvm::errs() << "], output: " << Result.getAsString() << "\n";
  } else {
    T->ConstructJob(C, *JA, Result, InputInfos,
                    C.getArgsForToolChain(TC, BoundArch), LinkingOutput);
  }
  return Result;
}

const char *Driver::getDefaultImageName() const {
  llvm::Triple Target(llvm::Triple::normalize(DefaultTargetTriple));
  return Target.isOSWindows() ? "a.exe" : "a.out";
}

/// \brief Create output filename based on ArgValue, which could either be a
/// full filename, filename without extension, or a directory. If ArgValue
/// does not provide a filename, then use BaseName, and use the extension
/// suitable for FileType.
static const char *MakeCLOutputFilename(const ArgList &Args, StringRef ArgValue,
                                        StringRef BaseName,
                                        types::ID FileType) {
  SmallString<128> Filename = ArgValue;

  if (ArgValue.empty()) {
    // If the argument is empty, output to BaseName in the current dir.
    Filename = BaseName;
  } else if (llvm::sys::path::is_separator(Filename.back())) {
    // If the argument is a directory, output to BaseName in that dir.
    llvm::sys::path::append(Filename, BaseName);
  }

  if (!llvm::sys::path::has_extension(ArgValue)) {
    // If the argument didn't provide an extension, then set it.
    const char *Extension = types::getTypeTempSuffix(FileType, true);

    if (FileType == types::TY_Image &&
        Args.hasArg(options::OPT__SLASH_LD, options::OPT__SLASH_LDd)) {
      // The output file is a dll.
      Extension = "dll";
    }

    llvm::sys::path::replace_extension(Filename, Extension);
  }

  return Args.MakeArgString(Filename.c_str());
}

const char *Driver::GetNamedOutputPath(Compilation &C, const JobAction &JA,
                                       const char *BaseInput,
                                       const char *BoundArch, bool AtTopLevel,
                                       bool MultipleArchs,
                                       StringRef NormalizedTriple) const {
  llvm::PrettyStackTraceString CrashInfo("Computing output path");
  // Output to a user requested destination?
  if (AtTopLevel && !isa<DsymutilJobAction>(JA) && !isa<VerifyJobAction>(JA)) {
    if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o))
      return C.addResultFile(FinalOutput->getValue(), &JA);
  }

  // For /P, preprocess to file named after BaseInput.
  if (C.getArgs().hasArg(options::OPT__SLASH_P)) {
    assert(AtTopLevel && isa<PreprocessJobAction>(JA));
    StringRef BaseName = llvm::sys::path::filename(BaseInput);
    StringRef NameArg;
    if (Arg *A = C.getArgs().getLastArg(options::OPT__SLASH_Fi))
      NameArg = A->getValue();
    return C.addResultFile(
        MakeCLOutputFilename(C.getArgs(), NameArg, BaseName, types::TY_PP_C),
        &JA);
  }

  // Default to writing to stdout?
  if (AtTopLevel && !CCGenDiagnostics &&
      (isa<PreprocessJobAction>(JA) || JA.getType() == types::TY_ModuleFile))
    return "-";

  // Is this the assembly listing for /FA?
  if (JA.getType() == types::TY_PP_Asm &&
      (C.getArgs().hasArg(options::OPT__SLASH_FA) ||
       C.getArgs().hasArg(options::OPT__SLASH_Fa))) {
    // Use /Fa and the input filename to determine the asm file name.
    StringRef BaseName = llvm::sys::path::filename(BaseInput);
    StringRef FaValue = C.getArgs().getLastArgValue(options::OPT__SLASH_Fa);
    return C.addResultFile(
        MakeCLOutputFilename(C.getArgs(), FaValue, BaseName, JA.getType()),
        &JA);
  }

  // Output to a temporary file?
  if ((!AtTopLevel && !isSaveTempsEnabled() &&
       !C.getArgs().hasArg(options::OPT__SLASH_Fo)) ||
      CCGenDiagnostics) {
    StringRef Name = llvm::sys::path::filename(BaseInput);
    std::pair<StringRef, StringRef> Split = Name.split('.');
    std::string TmpName = GetTemporaryPath(
        Split.first, types::getTypeTempSuffix(JA.getType(), IsCLMode()));
    return C.addTempFile(C.getArgs().MakeArgString(TmpName.c_str()));
  }

  SmallString<128> BasePath(BaseInput);
  StringRef BaseName;

  // Dsymutil actions should use the full path.
  if (isa<DsymutilJobAction>(JA) || isa<VerifyJobAction>(JA))
    BaseName = BasePath;
  else
    BaseName = llvm::sys::path::filename(BasePath);

  // Determine what the derived output name should be.
  const char *NamedOutput;

  if (JA.getType() == types::TY_Object &&
      C.getArgs().hasArg(options::OPT__SLASH_Fo, options::OPT__SLASH_o)) {
    // The /Fo or /o flag decides the object filename.
    StringRef Val =
        C.getArgs()
            .getLastArg(options::OPT__SLASH_Fo, options::OPT__SLASH_o)
            ->getValue();
    NamedOutput =
        MakeCLOutputFilename(C.getArgs(), Val, BaseName, types::TY_Object);
  } else if (JA.getType() == types::TY_Image &&
             C.getArgs().hasArg(options::OPT__SLASH_Fe,
                                options::OPT__SLASH_o)) {
    // The /Fe or /o flag names the linked file.
    StringRef Val =
        C.getArgs()
            .getLastArg(options::OPT__SLASH_Fe, options::OPT__SLASH_o)
            ->getValue();
    NamedOutput =
        MakeCLOutputFilename(C.getArgs(), Val, BaseName, types::TY_Image);
  } else if (JA.getType() == types::TY_Image) {
    if (IsCLMode()) {
      // clang-cl uses BaseName for the executable name.
      NamedOutput =
          MakeCLOutputFilename(C.getArgs(), "", BaseName, types::TY_Image);
    } else if (MultipleArchs && BoundArch) {
      SmallString<128> Output(getDefaultImageName());
      Output += JA.getOffloadingFileNamePrefix(NormalizedTriple);
      Output += "-";
      Output.append(BoundArch);
      NamedOutput = C.getArgs().MakeArgString(Output.c_str());
    } else {
      NamedOutput = getDefaultImageName();
    }
  } else if (JA.getType() == types::TY_PCH && IsCLMode()) {
    NamedOutput = C.getArgs().MakeArgString(GetClPchPath(C, BaseName).c_str());
  } else {
    const char *Suffix = types::getTypeTempSuffix(JA.getType(), IsCLMode());
    assert(Suffix && "All types used for output should have a suffix.");

    std::string::size_type End = std::string::npos;
    if (!types::appendSuffixForType(JA.getType()))
      End = BaseName.rfind('.');
    SmallString<128> Suffixed(BaseName.substr(0, End));
    Suffixed += JA.getOffloadingFileNamePrefix(NormalizedTriple);
    if (MultipleArchs && BoundArch) {
      Suffixed += "-";
      Suffixed.append(BoundArch);
    }
    // When using both -save-temps and -emit-llvm, use a ".tmp.bc" suffix for
    // the unoptimized bitcode so that it does not get overwritten by the ".bc"
    // optimized bitcode output.
    if (!AtTopLevel && C.getArgs().hasArg(options::OPT_emit_llvm) &&
        JA.getType() == types::TY_LLVM_BC)
      Suffixed += ".tmp";
    Suffixed += '.';
    Suffixed += Suffix;
    NamedOutput = C.getArgs().MakeArgString(Suffixed.c_str());
  }

  // Prepend object file path if -save-temps=obj
  if (!AtTopLevel && isSaveTempsObj() && C.getArgs().hasArg(options::OPT_o) &&
      JA.getType() != types::TY_PCH) {
    Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o);
    SmallString<128> TempPath(FinalOutput->getValue());
    llvm::sys::path::remove_filename(TempPath);
    StringRef OutputFileName = llvm::sys::path::filename(NamedOutput);
    llvm::sys::path::append(TempPath, OutputFileName);
    NamedOutput = C.getArgs().MakeArgString(TempPath.c_str());
  }

  // If we're saving temps and the temp file conflicts with the input file,
  // then avoid overwriting input file.
  if (!AtTopLevel && isSaveTempsEnabled() && NamedOutput == BaseName) {
    bool SameFile = false;
    SmallString<256> Result;
    llvm::sys::fs::current_path(Result);
    llvm::sys::path::append(Result, BaseName);
    llvm::sys::fs::equivalent(BaseInput, Result.c_str(), SameFile);
    // Must share the same path to conflict.
    if (SameFile) {
      StringRef Name = llvm::sys::path::filename(BaseInput);
      std::pair<StringRef, StringRef> Split = Name.split('.');
      std::string TmpName = GetTemporaryPath(
          Split.first, types::getTypeTempSuffix(JA.getType(), IsCLMode()));
      return C.addTempFile(C.getArgs().MakeArgString(TmpName.c_str()));
    }
  }

  // As an annoying special case, PCH generation doesn't strip the pathname.
  if (JA.getType() == types::TY_PCH && !IsCLMode()) {
    llvm::sys::path::remove_filename(BasePath);
    if (BasePath.empty())
      BasePath = NamedOutput;
    else
      llvm::sys::path::append(BasePath, NamedOutput);
    return C.addResultFile(C.getArgs().MakeArgString(BasePath.c_str()), &JA);
  } else {
    return C.addResultFile(NamedOutput, &JA);
  }
}

std::string Driver::GetFilePath(const char *Name, const ToolChain &TC) const {
  // Respect a limited subset of the '-Bprefix' functionality in GCC by
  // attempting to use this prefix when looking for file paths.
  for (const std::string &Dir : PrefixDirs) {
    if (Dir.empty())
      continue;
    SmallString<128> P(Dir[0] == '=' ? SysRoot + Dir.substr(1) : Dir);
    llvm::sys::path::append(P, Name);
    if (llvm::sys::fs::exists(Twine(P)))
      return P.str();
  }

  SmallString<128> P(ResourceDir);
  llvm::sys::path::append(P, Name);
  if (llvm::sys::fs::exists(Twine(P)))
    return P.str();

  for (const std::string &Dir : TC.getFilePaths()) {
    if (Dir.empty())
      continue;
    SmallString<128> P(Dir[0] == '=' ? SysRoot + Dir.substr(1) : Dir);
    llvm::sys::path::append(P, Name);
    if (llvm::sys::fs::exists(Twine(P)))
      return P.str();
  }

  return Name;
}

void Driver::generatePrefixedToolNames(
    const char *Tool, const ToolChain &TC,
    SmallVectorImpl<std::string> &Names) const {
  // FIXME: Needs a better variable than DefaultTargetTriple
  Names.emplace_back(DefaultTargetTriple + "-" + Tool);
  Names.emplace_back(Tool);

  // Allow the discovery of tools prefixed with LLVM's default target triple.
  std::string LLVMDefaultTargetTriple = llvm::sys::getDefaultTargetTriple();
  if (LLVMDefaultTargetTriple != DefaultTargetTriple)
    Names.emplace_back(LLVMDefaultTargetTriple + "-" + Tool);
}

static bool ScanDirForExecutable(SmallString<128> &Dir,
                                 ArrayRef<std::string> Names) {
  for (const auto &Name : Names) {
    llvm::sys::path::append(Dir, Name);
    if (llvm::sys::fs::can_execute(Twine(Dir)))
      return true;
    llvm::sys::path::remove_filename(Dir);
  }
  return false;
}

std::string Driver::GetProgramPath(const char *Name,
                                   const ToolChain &TC) const {
  SmallVector<std::string, 2> TargetSpecificExecutables;
  generatePrefixedToolNames(Name, TC, TargetSpecificExecutables);

  // Respect a limited subset of the '-Bprefix' functionality in GCC by
  // attempting to use this prefix when looking for program paths.
  for (const auto &PrefixDir : PrefixDirs) {
    if (llvm::sys::fs::is_directory(PrefixDir)) {
      SmallString<128> P(PrefixDir);
      if (ScanDirForExecutable(P, TargetSpecificExecutables))
        return P.str();
    } else {
      SmallString<128> P(PrefixDir + Name);
      if (llvm::sys::fs::can_execute(Twine(P)))
        return P.str();
    }
  }

  const ToolChain::path_list &List = TC.getProgramPaths();
  for (const auto &Path : List) {
    SmallString<128> P(Path);
    if (ScanDirForExecutable(P, TargetSpecificExecutables))
      return P.str();
  }

  // If all else failed, search the path.
  for (const auto &TargetSpecificExecutable : TargetSpecificExecutables)
    if (llvm::ErrorOr<std::string> P =
            llvm::sys::findProgramByName(TargetSpecificExecutable))
      return *P;

  return Name;
}

std::string Driver::GetTemporaryPath(StringRef Prefix,
                                     const char *Suffix) const {
  SmallString<128> Path;
  std::error_code EC = llvm::sys::fs::createTemporaryFile(Prefix, Suffix, Path);
  if (EC) {
    Diag(clang::diag::err_unable_to_make_temp) << EC.message();
    return "";
  }

  return Path.str();
}

std::string Driver::GetClPchPath(Compilation &C, StringRef BaseName) const {
  SmallString<128> Output;
  if (Arg *FpArg = C.getArgs().getLastArg(options::OPT__SLASH_Fp)) {
    // FIXME: If anybody needs it, implement this obscure rule:
    // "If you specify a directory without a file name, the default file name
    // is VCx0.pch., where x is the major version of Visual C++ in use."
    Output = FpArg->getValue();

    // "If you do not specify an extension as part of the path name, an
    // extension of .pch is assumed. "
    if (!llvm::sys::path::has_extension(Output))
      Output += ".pch";
  } else {
    Output = BaseName;
    llvm::sys::path::replace_extension(Output, ".pch");
  }
  return Output.str();
}

const ToolChain &Driver::getToolChain(const ArgList &Args,
                                      const llvm::Triple &Target) const {

  ToolChain *&TC = ToolChains[Target.str()];
  if (!TC) {
    switch (Target.getOS()) {
    case llvm::Triple::Haiku:
      TC = new toolchains::Haiku(*this, Target, Args);
      break;
    case llvm::Triple::CloudABI:
      TC = new toolchains::CloudABI(*this, Target, Args);
      break;
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
    case llvm::Triple::IOS:
    case llvm::Triple::TvOS:
    case llvm::Triple::WatchOS:
      TC = new toolchains::DarwinClang(*this, Target, Args);
      break;
    case llvm::Triple::DragonFly:
      TC = new toolchains::DragonFly(*this, Target, Args);
      break;
    case llvm::Triple::OpenBSD:
      TC = new toolchains::OpenBSD(*this, Target, Args);
      break;
    case llvm::Triple::Bitrig:
      TC = new toolchains::Bitrig(*this, Target, Args);
      break;
    case llvm::Triple::NetBSD:
      TC = new toolchains::NetBSD(*this, Target, Args);
      break;
    case llvm::Triple::FreeBSD:
      TC = new toolchains::FreeBSD(*this, Target, Args);
      break;
    case llvm::Triple::Minix:
      TC = new toolchains::Minix(*this, Target, Args);
      break;
    case llvm::Triple::Linux:
    case llvm::Triple::ELFIAMCU:
      if (Target.getArch() == llvm::Triple::hexagon)
        TC = new toolchains::HexagonToolChain(*this, Target, Args);
      else if ((Target.getVendor() == llvm::Triple::MipsTechnologies) &&
               !Target.hasEnvironment())
        TC = new toolchains::MipsLLVMToolChain(*this, Target, Args);
      else
        TC = new toolchains::Linux(*this, Target, Args);
      break;
    case llvm::Triple::NaCl:
      TC = new toolchains::NaClToolChain(*this, Target, Args);
      break;
    case llvm::Triple::Solaris:
      TC = new toolchains::Solaris(*this, Target, Args);
      break;
    case llvm::Triple::AMDHSA:
      TC = new toolchains::AMDGPUToolChain(*this, Target, Args);
      break;
    case llvm::Triple::Win32:
      switch (Target.getEnvironment()) {
      default:
        if (Target.isOSBinFormatELF())
          TC = new toolchains::Generic_ELF(*this, Target, Args);
        else if (Target.isOSBinFormatMachO())
          TC = new toolchains::MachO(*this, Target, Args);
        else
          TC = new toolchains::Generic_GCC(*this, Target, Args);
        break;
      case llvm::Triple::GNU:
        TC = new toolchains::MinGW(*this, Target, Args);
        break;
      case llvm::Triple::Itanium:
        TC = new toolchains::CrossWindowsToolChain(*this, Target, Args);
        break;
      case llvm::Triple::MSVC:
      case llvm::Triple::UnknownEnvironment:
        TC = new toolchains::MSVCToolChain(*this, Target, Args);
        break;
      }
      break;
    case llvm::Triple::CUDA:
      TC = new toolchains::CudaToolChain(*this, Target, Args);
      break;
    case llvm::Triple::PS4:
      TC = new toolchains::PS4CPU(*this, Target, Args);
      break;
    default:
      // Of these targets, Hexagon is the only one that might have
      // an OS of Linux, in which case it got handled above already.
      switch (Target.getArch()) {
      case llvm::Triple::tce:
        TC = new toolchains::TCEToolChain(*this, Target, Args);
        break;
      case llvm::Triple::hexagon:
        TC = new toolchains::HexagonToolChain(*this, Target, Args);
        break;
      case llvm::Triple::lanai:
        TC = new toolchains::LanaiToolChain(*this, Target, Args);
        break;
      case llvm::Triple::xcore:
        TC = new toolchains::XCoreToolChain(*this, Target, Args);
        break;
      case llvm::Triple::wasm32:
      case llvm::Triple::wasm64:
        TC = new toolchains::WebAssembly(*this, Target, Args);
        break;
      default:
        if (Target.getVendor() == llvm::Triple::Myriad)
          TC = new toolchains::MyriadToolChain(*this, Target, Args);
        else if (Target.isOSBinFormatELF())
          TC = new toolchains::Generic_ELF(*this, Target, Args);
        else if (Target.isOSBinFormatMachO())
          TC = new toolchains::MachO(*this, Target, Args);
        else
          TC = new toolchains::Generic_GCC(*this, Target, Args);
      }
    }
  }
  return *TC;
}

bool Driver::ShouldUseClangCompiler(const JobAction &JA) const {
  // Say "no" if there is not exactly one input of a type clang understands.
  if (JA.size() != 1 ||
      !types::isAcceptedByClang((*JA.input_begin())->getType()))
    return false;

  // And say "no" if this is not a kind of action clang understands.
  if (!isa<PreprocessJobAction>(JA) && !isa<PrecompileJobAction>(JA) &&
      !isa<CompileJobAction>(JA) && !isa<BackendJobAction>(JA))
    return false;

  return true;
}

/// GetReleaseVersion - Parse (([0-9]+)(.([0-9]+)(.([0-9]+)?))?)? and return the
/// grouped values as integers. Numbers which are not provided are set to 0.
///
/// \return True if the entire string was parsed (9.2), or all groups were
/// parsed (10.3.5extrastuff).
bool Driver::GetReleaseVersion(const char *Str, unsigned &Major,
                               unsigned &Minor, unsigned &Micro,
                               bool &HadExtra) {
  HadExtra = false;

  Major = Minor = Micro = 0;
  if (*Str == '\0')
    return false;

  char *End;
  Major = (unsigned)strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (*End != '.')
    return false;

  Str = End + 1;
  Minor = (unsigned)strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (*End != '.')
    return false;

  Str = End + 1;
  Micro = (unsigned)strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (Str == End)
    return false;
  HadExtra = true;
  return true;
}

/// Parse digits from a string \p Str and fulfill \p Digits with
/// the parsed numbers. This method assumes that the max number of
/// digits to look for is equal to Digits.size().
///
/// \return True if the entire string was parsed and there are
/// no extra characters remaining at the end.
bool Driver::GetReleaseVersion(const char *Str,
                               MutableArrayRef<unsigned> Digits) {
  if (*Str == '\0')
    return false;

  char *End;
  unsigned CurDigit = 0;
  while (CurDigit < Digits.size()) {
    unsigned Digit = (unsigned)strtol(Str, &End, 10);
    Digits[CurDigit] = Digit;
    if (*Str != '\0' && *End == '\0')
      return true;
    if (*End != '.' || Str == End)
      return false;
    Str = End + 1;
    CurDigit++;
  }

  // More digits than requested, bail out...
  return false;
}

std::pair<unsigned, unsigned> Driver::getIncludeExcludeOptionFlagMasks() const {
  unsigned IncludedFlagsBitmask = 0;
  unsigned ExcludedFlagsBitmask = options::NoDriverOption;

  if (Mode == CLMode) {
    // Include CL and Core options.
    IncludedFlagsBitmask |= options::CLOption;
    IncludedFlagsBitmask |= options::CoreOption;
  } else {
    ExcludedFlagsBitmask |= options::CLOption;
  }

  return std::make_pair(IncludedFlagsBitmask, ExcludedFlagsBitmask);
}

bool clang::driver::isOptimizationLevelFast(const ArgList &Args) {
  return Args.hasFlag(options::OPT_Ofast, options::OPT_O_Group, false);
}
