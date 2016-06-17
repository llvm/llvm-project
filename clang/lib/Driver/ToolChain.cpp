//===--- ToolChain.cpp - Collections of tools for one platform ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Tools.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Config/config.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang;
using namespace llvm;
using namespace llvm::opt;

static llvm::opt::Arg *GetRTTIArgument(const ArgList &Args) {
  return Args.getLastArg(options::OPT_mkernel, options::OPT_fapple_kext,
                         options::OPT_fno_rtti, options::OPT_frtti);
}

static ToolChain::RTTIMode CalculateRTTIMode(const ArgList &Args,
                                             const llvm::Triple &Triple,
                                             const Arg *CachedRTTIArg) {
  // Explicit rtti/no-rtti args
  if (CachedRTTIArg) {
    if (CachedRTTIArg->getOption().matches(options::OPT_frtti))
      return ToolChain::RM_EnabledExplicitly;
    else
      return ToolChain::RM_DisabledExplicitly;
  }

  // -frtti is default, except for the PS4 CPU.
  if (!Triple.isPS4CPU())
    return ToolChain::RM_EnabledImplicitly;

  // On the PS4, turning on c++ exceptions turns on rtti.
  // We're assuming that, if we see -fexceptions, rtti gets turned on.
  Arg *Exceptions = Args.getLastArgNoClaim(
      options::OPT_fcxx_exceptions, options::OPT_fno_cxx_exceptions,
      options::OPT_fexceptions, options::OPT_fno_exceptions);
  if (Exceptions &&
      (Exceptions->getOption().matches(options::OPT_fexceptions) ||
       Exceptions->getOption().matches(options::OPT_fcxx_exceptions)))
    return ToolChain::RM_EnabledImplicitly;

  return ToolChain::RM_DisabledImplicitly;
}

ToolChain::ToolChain(const Driver &D, const llvm::Triple &T,
                     const ArgList &Args)
    : D(D), Triple(T), Args(Args), CachedRTTIArg(GetRTTIArgument(Args)),
      CachedRTTIMode(CalculateRTTIMode(Args, Triple, CachedRTTIArg)) {
  if (Arg *A = Args.getLastArg(options::OPT_mthread_model))
    if (!isThreadModelSupported(A->getValue()))
      D.Diag(diag::err_drv_invalid_thread_model_for_target)
          << A->getValue() << A->getAsString(Args);
}

ToolChain::~ToolChain() {
}

vfs::FileSystem &ToolChain::getVFS() const { return getDriver().getVFS(); }

bool ToolChain::useIntegratedAs() const {
  return Args.hasFlag(options::OPT_fintegrated_as,
                      options::OPT_fno_integrated_as,
                      IsIntegratedAssemblerDefault());
}

const SanitizerArgs& ToolChain::getSanitizerArgs() const {
  if (!SanitizerArguments.get())
    SanitizerArguments.reset(new SanitizerArgs(*this, Args));
  return *SanitizerArguments.get();
}

namespace {
struct DriverSuffix {
  const char *Suffix;
  const char *ModeFlag;
};

const DriverSuffix *FindDriverSuffix(StringRef ProgName) {
  // A list of known driver suffixes. Suffixes are compared against the
  // program name in order. If there is a match, the frontend type is updated as
  // necessary by applying the ModeFlag.
  static const DriverSuffix DriverSuffixes[] = {
      {"clang", nullptr},
      {"clang++", "--driver-mode=g++"},
      {"clang-c++", "--driver-mode=g++"},
      {"clang-cc", nullptr},
      {"clang-cpp", "--driver-mode=cpp"},
      {"clang-g++", "--driver-mode=g++"},
      {"clang-gcc", nullptr},
      {"clang-cl", "--driver-mode=cl"},
      {"cc", nullptr},
      {"cpp", "--driver-mode=cpp"},
      {"cl", "--driver-mode=cl"},
      {"++", "--driver-mode=g++"},
  };

  for (size_t i = 0; i < llvm::array_lengthof(DriverSuffixes); ++i)
    if (ProgName.endswith(DriverSuffixes[i].Suffix))
      return &DriverSuffixes[i];
  return nullptr;
}

/// Normalize the program name from argv[0] by stripping the file extension if
/// present and lower-casing the string on Windows.
std::string normalizeProgramName(llvm::StringRef Argv0) {
  std::string ProgName = llvm::sys::path::stem(Argv0);
#ifdef LLVM_ON_WIN32
  // Transform to lowercase for case insensitive file systems.
  std::transform(ProgName.begin(), ProgName.end(), ProgName.begin(), ::tolower);
#endif
  return ProgName;
}

const DriverSuffix *parseDriverSuffix(StringRef ProgName) {
  // Try to infer frontend type and default target from the program name by
  // comparing it against DriverSuffixes in order.

  // If there is a match, the function tries to identify a target as prefix.
  // E.g. "x86_64-linux-clang" as interpreted as suffix "clang" with target
  // prefix "x86_64-linux". If such a target prefix is found, it may be
  // added via -target as implicit first argument.
  const DriverSuffix *DS = FindDriverSuffix(ProgName);

  if (!DS) {
    // Try again after stripping any trailing version number:
    // clang++3.5 -> clang++
    ProgName = ProgName.rtrim("0123456789.");
    DS = FindDriverSuffix(ProgName);
  }

  if (!DS) {
    // Try again after stripping trailing -component.
    // clang++-tot -> clang++
    ProgName = ProgName.slice(0, ProgName.rfind('-'));
    DS = FindDriverSuffix(ProgName);
  }
  return DS;
}
} // anonymous namespace

std::pair<std::string, std::string>
ToolChain::getTargetAndModeFromProgramName(StringRef PN) {
  std::string ProgName = normalizeProgramName(PN);
  const DriverSuffix *DS = parseDriverSuffix(ProgName);
  if (!DS)
    return std::make_pair("", "");
  std::string ModeFlag = DS->ModeFlag == nullptr ? "" : DS->ModeFlag;

  std::string::size_type LastComponent =
      ProgName.rfind('-', ProgName.size() - strlen(DS->Suffix));
  if (LastComponent == std::string::npos)
    return std::make_pair("", ModeFlag);

  // Infer target from the prefix.
  StringRef Prefix(ProgName);
  Prefix = Prefix.slice(0, LastComponent);
  std::string IgnoredError;
  std::string Target;
  if (llvm::TargetRegistry::lookupTarget(Prefix, IgnoredError)) {
    Target = Prefix;
  }
  return std::make_pair(Target, ModeFlag);
}

StringRef ToolChain::getDefaultUniversalArchName() const {
  // In universal driver terms, the arch name accepted by -arch isn't exactly
  // the same as the ones that appear in the triple. Roughly speaking, this is
  // an inverse of the darwin::getArchTypeForDarwinArchName() function, but the
  // only interesting special case is powerpc.
  switch (Triple.getArch()) {
  case llvm::Triple::ppc:
    return "ppc";
  case llvm::Triple::ppc64:
    return "ppc64";
  case llvm::Triple::ppc64le:
    return "ppc64le";
  default:
    return Triple.getArchName();
  }
}

bool ToolChain::IsUnwindTablesDefault() const {
  return false;
}

Tool *ToolChain::getClang() const {
  if (!Clang)
    Clang.reset(new tools::Clang(*this));
  return Clang.get();
}

Tool *ToolChain::buildAssembler() const {
  return new tools::ClangAs(*this);
}

Tool *ToolChain::buildLinker() const {
  llvm_unreachable("Linking is not supported by this toolchain");
}

Tool *ToolChain::getAssemble() const {
  if (!Assemble)
    Assemble.reset(buildAssembler());
  return Assemble.get();
}

Tool *ToolChain::getClangAs() const {
  if (!Assemble)
    Assemble.reset(new tools::ClangAs(*this));
  return Assemble.get();
}

Tool *ToolChain::getLink() const {
  if (!Link)
    Link.reset(buildLinker());
  return Link.get();
}

Tool *ToolChain::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::AssembleJobClass:
    return getAssemble();

  case Action::LinkJobClass:
    return getLink();

  case Action::InputClass:
  case Action::BindArchClass:
  case Action::CudaDeviceClass:
  case Action::CudaHostClass:
  case Action::LipoJobClass:
  case Action::DsymutilJobClass:
  case Action::VerifyDebugInfoJobClass:
    llvm_unreachable("Invalid tool kind.");

  case Action::CompileJobClass:
  case Action::PrecompileJobClass:
  case Action::PreprocessJobClass:
  case Action::AnalyzeJobClass:
  case Action::MigrateJobClass:
  case Action::VerifyPCHJobClass:
  case Action::BackendJobClass:
    return getClang();
  }

  llvm_unreachable("Invalid tool kind.");
}

static StringRef getArchNameForCompilerRTLib(const ToolChain &TC,
                                             const ArgList &Args) {
  const llvm::Triple &Triple = TC.getTriple();
  bool IsWindows = Triple.isOSWindows();

  if (Triple.isWindowsMSVCEnvironment() && TC.getArch() == llvm::Triple::x86)
    return "i386";

  if (TC.getArch() == llvm::Triple::arm || TC.getArch() == llvm::Triple::armeb)
    return (arm::getARMFloatABI(TC, Args) == arm::FloatABI::Hard && !IsWindows)
               ? "armhf"
               : "arm";

  return TC.getArchName();
}

std::string ToolChain::getCompilerRT(const ArgList &Args, StringRef Component,
                                     bool Shared) const {
  const llvm::Triple &TT = getTriple();
  const char *Env = TT.isAndroid() ? "-android" : "";
  bool IsITANMSVCWindows =
      TT.isWindowsMSVCEnvironment() || TT.isWindowsItaniumEnvironment();

  StringRef Arch = getArchNameForCompilerRTLib(*this, Args);
  const char *Prefix = IsITANMSVCWindows ? "" : "lib";
  const char *Suffix = Shared ? (Triple.isOSWindows() ? ".dll" : ".so")
                              : (IsITANMSVCWindows ? ".lib" : ".a");

  SmallString<128> Path(getDriver().ResourceDir);
  StringRef OSLibName = Triple.isOSFreeBSD() ? "freebsd" : getOS();
  llvm::sys::path::append(Path, "lib", OSLibName);
  llvm::sys::path::append(Path, Prefix + Twine("clang_rt.") + Component + "-" +
                                    Arch + Env + Suffix);
  return Path.str();
}

const char *ToolChain::getCompilerRTArgString(const llvm::opt::ArgList &Args,
                                              StringRef Component,
                                              bool Shared) const {
  return Args.MakeArgString(getCompilerRT(Args, Component, Shared));
}

bool ToolChain::needsProfileRT(const ArgList &Args) {
  if (Args.hasFlag(options::OPT_fprofile_arcs, options::OPT_fno_profile_arcs,
                   false) ||
      Args.hasArg(options::OPT_fprofile_generate) ||
      Args.hasArg(options::OPT_fprofile_generate_EQ) ||
      Args.hasArg(options::OPT_fprofile_instr_generate) ||
      Args.hasArg(options::OPT_fprofile_instr_generate_EQ) ||
      Args.hasArg(options::OPT_fcreate_profile) ||
      Args.hasArg(options::OPT_coverage))
    return true;

  return false;
}

Tool *ToolChain::SelectTool(const JobAction &JA) const {
  if (getDriver().ShouldUseClangCompiler(JA)) return getClang();
  Action::ActionClass AC = JA.getKind();
  if (AC == Action::AssembleJobClass && useIntegratedAs())
    return getClangAs();
  return getTool(AC);
}

std::string ToolChain::GetFilePath(const char *Name) const {
  return D.GetFilePath(Name, *this);
}

std::string ToolChain::GetProgramPath(const char *Name) const {
  return D.GetProgramPath(Name, *this);
}

std::string ToolChain::GetLinkerPath() const {
  if (Arg *A = Args.getLastArg(options::OPT_fuse_ld_EQ)) {
    StringRef UseLinker = A->getValue();

    if (llvm::sys::path::is_absolute(UseLinker)) {
      // If we're passed -fuse-ld= with what looks like an absolute path,
      // don't attempt to second-guess that.
      if (llvm::sys::fs::exists(UseLinker))
        return UseLinker;
    } else {
      // If we're passed -fuse-ld= with no argument, or with the argument ld,
      // then use whatever the default system linker is.
      if (UseLinker.empty() || UseLinker == "ld")
        return GetProgramPath("ld");

      llvm::SmallString<8> LinkerName("ld.");
      LinkerName.append(UseLinker);

      std::string LinkerPath(GetProgramPath(LinkerName.c_str()));
      if (llvm::sys::fs::exists(LinkerPath))
        return LinkerPath;
    }

    getDriver().Diag(diag::err_drv_invalid_linker_name) << A->getAsString(Args);
    return "";
  }

  return GetProgramPath(DefaultLinker);
}

types::ID ToolChain::LookupTypeForExtension(const char *Ext) const {
  return types::lookupTypeForExtension(Ext);
}

bool ToolChain::HasNativeLLVMSupport() const {
  return false;
}

bool ToolChain::isCrossCompiling() const {
  llvm::Triple HostTriple(LLVM_HOST_TRIPLE);
  switch (HostTriple.getArch()) {
  // The A32/T32/T16 instruction sets are not separate architectures in this
  // context.
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    return getArch() != llvm::Triple::arm && getArch() != llvm::Triple::thumb &&
           getArch() != llvm::Triple::armeb && getArch() != llvm::Triple::thumbeb;
  default:
    return HostTriple.getArch() != getArch();
  }
}

ObjCRuntime ToolChain::getDefaultObjCRuntime(bool isNonFragile) const {
  return ObjCRuntime(isNonFragile ? ObjCRuntime::GNUstep : ObjCRuntime::GCC,
                     VersionTuple());
}

bool ToolChain::isThreadModelSupported(const StringRef Model) const {
  if (Model == "single") {
    // FIXME: 'single' is only supported on ARM and WebAssembly so far.
    return Triple.getArch() == llvm::Triple::arm ||
           Triple.getArch() == llvm::Triple::armeb ||
           Triple.getArch() == llvm::Triple::thumb ||
           Triple.getArch() == llvm::Triple::thumbeb ||
           Triple.getArch() == llvm::Triple::wasm32 ||
           Triple.getArch() == llvm::Triple::wasm64;
  } else if (Model == "posix")
    return true;

  return false;
}

std::string ToolChain::ComputeLLVMTriple(const ArgList &Args,
                                         types::ID InputType) const {
  switch (getTriple().getArch()) {
  default:
    return getTripleString();

  case llvm::Triple::x86_64: {
    llvm::Triple Triple = getTriple();
    if (!Triple.isOSBinFormatMachO())
      return getTripleString();

    if (Arg *A = Args.getLastArg(options::OPT_march_EQ)) {
      // x86_64h goes in the triple. Other -march options just use the
      // vanilla triple we already have.
      StringRef MArch = A->getValue();
      if (MArch == "x86_64h")
        Triple.setArchName(MArch);
    }
    return Triple.getTriple();
  }
  case llvm::Triple::aarch64: {
    llvm::Triple Triple = getTriple();
    if (!Triple.isOSBinFormatMachO())
      return getTripleString();

    // FIXME: older versions of ld64 expect the "arm64" component in the actual
    // triple string and query it to determine whether an LTO file can be
    // handled. Remove this when we don't care any more.
    Triple.setArchName("arm64");
    return Triple.getTriple();
  }
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb: {
    // FIXME: Factor into subclasses.
    llvm::Triple Triple = getTriple();
    bool IsBigEndian = getTriple().getArch() == llvm::Triple::armeb ||
                       getTriple().getArch() == llvm::Triple::thumbeb;

    // Handle pseudo-target flags '-mlittle-endian'/'-EL' and
    // '-mbig-endian'/'-EB'.
    if (Arg *A = Args.getLastArg(options::OPT_mlittle_endian,
                                 options::OPT_mbig_endian)) {
      IsBigEndian = !A->getOption().matches(options::OPT_mlittle_endian);
    }

    // Thumb2 is the default for V7 on Darwin.
    //
    // FIXME: Thumb should just be another -target-feaure, not in the triple.
    StringRef MCPU, MArch;
    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      MCPU = A->getValue();
    if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
      MArch = A->getValue();
    std::string CPU =
        Triple.isOSBinFormatMachO()
            ? tools::arm::getARMCPUForMArch(MArch, Triple).str()
            : tools::arm::getARMTargetCPU(MCPU, MArch, Triple);
    StringRef Suffix =
      tools::arm::getLLVMArchSuffixForARM(CPU, MArch, Triple);
    bool IsMProfile = ARM::parseArchProfile(Suffix) == ARM::PK_M;
    bool ThumbDefault = IsMProfile || (ARM::parseArchVersion(Suffix) == 7 && 
                                       getTriple().isOSBinFormatMachO());
    // FIXME: this is invalid for WindowsCE
    if (getTriple().isOSWindows())
      ThumbDefault = true;
    std::string ArchName;
    if (IsBigEndian)
      ArchName = "armeb";
    else
      ArchName = "arm";

    // Assembly files should start in ARM mode, unless arch is M-profile.
    if ((InputType != types::TY_PP_Asm && Args.hasFlag(options::OPT_mthumb,
         options::OPT_mno_thumb, ThumbDefault)) || IsMProfile) {
      if (IsBigEndian)
        ArchName = "thumbeb";
      else
        ArchName = "thumb";
    }
    Triple.setArchName(ArchName + Suffix.str());

    return Triple.getTriple();
  }
  }
}

std::string ToolChain::ComputeEffectiveClangTriple(const ArgList &Args,
                                                   types::ID InputType) const {
  return ComputeLLVMTriple(Args, InputType);
}

void ToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                          ArgStringList &CC1Args) const {
  // Each toolchain should provide the appropriate include flags.
}

void ToolChain::addClangTargetOptions(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
}

void ToolChain::addClangWarningOptions(ArgStringList &CC1Args) const {}

void ToolChain::addProfileRTLibs(const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs) const {
  if (!needsProfileRT(Args)) return;

  CmdArgs.push_back(getCompilerRTArgString(Args, "profile"));
}

ToolChain::RuntimeLibType ToolChain::GetRuntimeLibType(
    const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_rtlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "compiler-rt")
      return ToolChain::RLT_CompilerRT;
    if (Value == "libgcc")
      return ToolChain::RLT_Libgcc;
    getDriver().Diag(diag::err_drv_invalid_rtlib_name)
      << A->getAsString(Args);
  }

  return GetDefaultRuntimeLibType();
}

static bool ParseCXXStdlibType(const StringRef& Name,
                               ToolChain::CXXStdlibType& Type) {
  if (Name == "libc++")
    Type = ToolChain::CST_Libcxx;
  else if (Name == "libstdc++")
    Type = ToolChain::CST_Libstdcxx;
  else
    return false;

  return true;
}

ToolChain::CXXStdlibType ToolChain::GetCXXStdlibType(const ArgList &Args) const{
  ToolChain::CXXStdlibType Type;
  bool HasValidType = false;
  bool ForcePlatformDefault = false;

  const Arg *A = Args.getLastArg(options::OPT_stdlib_EQ);
  if (A) {
    StringRef Value = A->getValue();
    HasValidType = ParseCXXStdlibType(Value, Type);

    // Only use in tests to override CLANG_DEFAULT_CXX_STDLIB!
    if (Value == "platform")
      ForcePlatformDefault = true;
    else if (!HasValidType)
      getDriver().Diag(diag::err_drv_invalid_stdlib_name)
        << A->getAsString(Args);
  }

  if (!HasValidType && (ForcePlatformDefault ||
      !ParseCXXStdlibType(CLANG_DEFAULT_CXX_STDLIB, Type)))
    Type = GetDefaultCXXStdlibType();

  return Type;
}

/// \brief Utility function to add a system include directory to CC1 arguments.
/*static*/ void ToolChain::addSystemInclude(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args,
                                            const Twine &Path) {
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(Path));
}

/// \brief Utility function to add a system include directory with extern "C"
/// semantics to CC1 arguments.
///
/// Note that this should be used rarely, and only for directories that
/// historically and for legacy reasons are treated as having implicit extern
/// "C" semantics. These semantics are *ignored* by and large today, but its
/// important to preserve the preprocessor changes resulting from the
/// classification.
/*static*/ void ToolChain::addExternCSystemInclude(const ArgList &DriverArgs,
                                                   ArgStringList &CC1Args,
                                                   const Twine &Path) {
  CC1Args.push_back("-internal-externc-isystem");
  CC1Args.push_back(DriverArgs.MakeArgString(Path));
}

void ToolChain::addExternCSystemIncludeIfExists(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args,
                                                const Twine &Path) {
  if (llvm::sys::fs::exists(Path))
    addExternCSystemInclude(DriverArgs, CC1Args, Path);
}

/// \brief Utility function to add a list of system include directories to CC1.
/*static*/ void ToolChain::addSystemIncludes(const ArgList &DriverArgs,
                                             ArgStringList &CC1Args,
                                             ArrayRef<StringRef> Paths) {
  for (StringRef Path : Paths) {
    CC1Args.push_back("-internal-isystem");
    CC1Args.push_back(DriverArgs.MakeArgString(Path));
  }
}

void ToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                             ArgStringList &CC1Args) const {
  // Header search paths should be handled by each of the subclasses.
  // Historically, they have not been, and instead have been handled inside of
  // the CC1-layer frontend. As the logic is hoisted out, this generic function
  // will slowly stop being called.
  //
  // While it is being called, replicate a bit of a hack to propagate the
  // '-stdlib=' flag down to CC1 so that it can in turn customize the C++
  // header search paths with it. Once all systems are overriding this
  // function, the CC1 flag and this line can be removed.
  DriverArgs.AddAllArgs(CC1Args, options::OPT_stdlib_EQ);
}

void ToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  CXXStdlibType Type = GetCXXStdlibType(Args);

  switch (Type) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    break;

  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    break;
  }
}

void ToolChain::AddFilePathLibArgs(const ArgList &Args,
                                   ArgStringList &CmdArgs) const {
  for (const auto &LibPath : getFilePaths())
    if(LibPath.length() > 0)
      CmdArgs.push_back(Args.MakeArgString(StringRef("-L") + LibPath));
}

void ToolChain::AddCCKextLibArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-lcc_kext");
}

bool ToolChain::AddFastMathRuntimeIfAvailable(const ArgList &Args,
                                              ArgStringList &CmdArgs) const {
  // Do not check for -fno-fast-math or -fno-unsafe-math when -Ofast passed
  // (to keep the linker options consistent with gcc and clang itself).
  if (!isOptimizationLevelFast(Args)) {
    // Check if -ffast-math or -funsafe-math.
    Arg *A =
        Args.getLastArg(options::OPT_ffast_math, options::OPT_fno_fast_math,
                        options::OPT_funsafe_math_optimizations,
                        options::OPT_fno_unsafe_math_optimizations);

    if (!A || A->getOption().getID() == options::OPT_fno_fast_math ||
        A->getOption().getID() == options::OPT_fno_unsafe_math_optimizations)
      return false;
  }
  // If crtfastmath.o exists add it to the arguments.
  std::string Path = GetFilePath("crtfastmath.o");
  if (Path == "crtfastmath.o") // Not found.
    return false;

  CmdArgs.push_back(Args.MakeArgString(Path));
  return true;
}

SanitizerMask ToolChain::getSupportedSanitizers() const {
  // Return sanitizers which don't require runtime support and are not
  // platform dependent.
  using namespace SanitizerKind;
  SanitizerMask Res = (Undefined & ~Vptr & ~Function) | (CFI & ~CFIICall) |
                      CFICastStrict | UnsignedIntegerOverflow | LocalBounds;
  if (getTriple().getArch() == llvm::Triple::x86 ||
      getTriple().getArch() == llvm::Triple::x86_64)
    Res |= CFIICall;
  return Res;
}

void ToolChain::AddCudaIncludeArgs(const ArgList &DriverArgs,
                                   ArgStringList &CC1Args) const {}

void ToolChain::AddIAMCUIncludeArgs(const ArgList &DriverArgs,
                                    ArgStringList &CC1Args) const {}
