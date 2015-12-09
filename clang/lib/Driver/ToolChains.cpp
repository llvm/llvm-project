//===--- ToolChains.cpp - ToolChain Implementations -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"
#include "clang/Basic/ObjCRuntime.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/VirtualFileSystem.h"
#include "clang/Config/config.h" // for GCC_INSTALL_PREFIX
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib> // ::getenv
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

MachO::MachO(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  // We expect 'as', 'ld', etc. to be adjacent to our install dir.
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);
}

/// Darwin - Darwin tool chain for i386 and x86_64.
Darwin::Darwin(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : MachO(D, Triple, Args), TargetInitialized(false) {}

types::ID MachO::LookupTypeForExtension(const char *Ext) const {
  types::ID Ty = types::lookupTypeForExtension(Ext);

  // Darwin always preprocesses assembly files (unless -x is used explicitly).
  if (Ty == types::TY_PP_Asm)
    return types::TY_Asm;

  return Ty;
}

bool MachO::HasNativeLLVMSupport() const { return true; }

/// Darwin provides an ARC runtime starting in MacOS X 10.7 and iOS 5.0.
ObjCRuntime Darwin::getDefaultObjCRuntime(bool isNonFragile) const {
  if (isTargetWatchOSBased())
    return ObjCRuntime(ObjCRuntime::WatchOS, TargetVersion);
  if (isTargetIOSBased())
    return ObjCRuntime(ObjCRuntime::iOS, TargetVersion);
  if (isNonFragile)
    return ObjCRuntime(ObjCRuntime::MacOSX, TargetVersion);
  return ObjCRuntime(ObjCRuntime::FragileMacOSX, TargetVersion);
}

/// Darwin provides a blocks runtime starting in MacOS X 10.6 and iOS 3.2.
bool Darwin::hasBlocksRuntime() const {
  if (isTargetWatchOSBased())
    return true;
  else if (isTargetIOSBased())
    return !isIPhoneOSVersionLT(3, 2);
  else {
    assert(isTargetMacOS() && "unexpected darwin target");
    return !isMacosxVersionLT(10, 6);
  }
}

// This is just a MachO name translation routine and there's no
// way to join this into ARMTargetParser without breaking all
// other assumptions. Maybe MachO should consider standardising
// their nomenclature.
static const char *ArmMachOArchName(StringRef Arch) {
  return llvm::StringSwitch<const char *>(Arch)
      .Case("armv6k", "armv6")
      .Case("armv6m", "armv6m")
      .Case("armv5tej", "armv5")
      .Case("xscale", "xscale")
      .Case("armv4t", "armv4t")
      .Case("armv7", "armv7")
      .Cases("armv7a", "armv7-a", "armv7")
      .Cases("armv7r", "armv7-r", "armv7")
      .Cases("armv7em", "armv7e-m", "armv7em")
      .Cases("armv7k", "armv7-k", "armv7k")
      .Cases("armv7m", "armv7-m", "armv7m")
      .Cases("armv7s", "armv7-s", "armv7s")
      .Default(nullptr);
}

static const char *ArmMachOArchNameCPU(StringRef CPU) {
  unsigned ArchKind = llvm::ARM::parseCPUArch(CPU);
  if (ArchKind == llvm::ARM::AK_INVALID)
    return nullptr;
  StringRef Arch = llvm::ARM::getArchName(ArchKind);

  // FIXME: Make sure this MachO triple mangling is really necessary.
  // ARMv5* normalises to ARMv5.
  if (Arch.startswith("armv5"))
    Arch = Arch.substr(0, 5);
  // ARMv6*, except ARMv6M, normalises to ARMv6.
  else if (Arch.startswith("armv6") && !Arch.endswith("6m"))
    Arch = Arch.substr(0, 5);
  // ARMv7A normalises to ARMv7.
  else if (Arch.endswith("v7a"))
    Arch = Arch.substr(0, 5);
  return Arch.data();
}

static bool isSoftFloatABI(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_msoft_float, options::OPT_mhard_float,
                           options::OPT_mfloat_abi_EQ);
  if (!A)
    return false;

  return A->getOption().matches(options::OPT_msoft_float) ||
         (A->getOption().matches(options::OPT_mfloat_abi_EQ) &&
          A->getValue() == StringRef("soft"));
}

StringRef MachO::getMachOArchName(const ArgList &Args) const {
  switch (getTriple().getArch()) {
  default:
    return getDefaultUniversalArchName();

  case llvm::Triple::aarch64:
    return "arm64";

  case llvm::Triple::thumb:
  case llvm::Triple::arm:
    if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
      if (const char *Arch = ArmMachOArchName(A->getValue()))
        return Arch;

    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      if (const char *Arch = ArmMachOArchNameCPU(A->getValue()))
        return Arch;

    return "arm";
  }
}

Darwin::~Darwin() {}

MachO::~MachO() {}

std::string MachO::ComputeEffectiveClangTriple(const ArgList &Args,
                                               types::ID InputType) const {
  llvm::Triple Triple(ComputeLLVMTriple(Args, InputType));

  return Triple.getTriple();
}

std::string Darwin::ComputeEffectiveClangTriple(const ArgList &Args,
                                                types::ID InputType) const {
  llvm::Triple Triple(ComputeLLVMTriple(Args, InputType));

  // If the target isn't initialized (e.g., an unknown Darwin platform, return
  // the default triple).
  if (!isTargetInitialized())
    return Triple.getTriple();

  SmallString<16> Str;
  if (isTargetWatchOSBased())
    Str += "watchos";
  else if (isTargetTvOSBased())
    Str += "tvos";
  else if (isTargetIOSBased())
    Str += "ios";
  else
    Str += "macosx";
  Str += getTargetVersion().getAsString();
  Triple.setOSName(Str);

  return Triple.getTriple();
}

void Generic_ELF::anchor() {}

Tool *MachO::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::LipoJobClass:
    if (!Lipo)
      Lipo.reset(new tools::darwin::Lipo(*this));
    return Lipo.get();
  case Action::DsymutilJobClass:
    if (!Dsymutil)
      Dsymutil.reset(new tools::darwin::Dsymutil(*this));
    return Dsymutil.get();
  case Action::VerifyDebugInfoJobClass:
    if (!VerifyDebug)
      VerifyDebug.reset(new tools::darwin::VerifyDebug(*this));
    return VerifyDebug.get();
  default:
    return ToolChain::getTool(AC);
  }
}

Tool *MachO::buildLinker() const { return new tools::darwin::Linker(*this); }

Tool *MachO::buildAssembler() const {
  return new tools::darwin::Assembler(*this);
}

DarwinClang::DarwinClang(const Driver &D, const llvm::Triple &Triple,
                         const ArgList &Args)
    : Darwin(D, Triple, Args) {}

void DarwinClang::addClangWarningOptions(ArgStringList &CC1Args) const {
  // For modern targets, promote certain warnings to errors.
  if (isTargetWatchOSBased() || getTriple().isArch64Bit()) {
    // Always enable -Wdeprecated-objc-isa-usage and promote it
    // to an error.
    CC1Args.push_back("-Wdeprecated-objc-isa-usage");
    CC1Args.push_back("-Werror=deprecated-objc-isa-usage");

    // For iOS and watchOS, also error about implicit function declarations,
    // as that can impact calling conventions.
    if (!isTargetMacOS())
      CC1Args.push_back("-Werror=implicit-function-declaration");
  }
}

/// \brief Determine whether Objective-C automated reference counting is
/// enabled.
static bool isObjCAutoRefCount(const ArgList &Args) {
  return Args.hasFlag(options::OPT_fobjc_arc, options::OPT_fno_objc_arc, false);
}

void DarwinClang::AddLinkARCArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  // Avoid linking compatibility stubs on i386 mac.
  if (isTargetMacOS() && getArch() == llvm::Triple::x86)
    return;

  ObjCRuntime runtime = getDefaultObjCRuntime(/*nonfragile*/ true);

  if ((runtime.hasNativeARC() || !isObjCAutoRefCount(Args)) &&
      runtime.hasSubscripting())
    return;

  CmdArgs.push_back("-force_load");
  SmallString<128> P(getDriver().ClangExecutable);
  llvm::sys::path::remove_filename(P); // 'clang'
  llvm::sys::path::remove_filename(P); // 'bin'
  llvm::sys::path::append(P, "lib", "arc", "libarclite_");
  // Mash in the platform.
  if (isTargetWatchOSSimulator())
    P += "watchsimulator";
  else if (isTargetWatchOS())
    P += "watchos";
  else if (isTargetTvOSSimulator())
    P += "appletvsimulator";
  else if (isTargetTvOS())
    P += "appletvos";
  else if (isTargetIOSSimulator())
    P += "iphonesimulator";
  else if (isTargetIPhoneOS())
    P += "iphoneos";
  else
    P += "macosx";
  P += ".a";

  CmdArgs.push_back(Args.MakeArgString(P));
}

void MachO::AddLinkRuntimeLib(const ArgList &Args, ArgStringList &CmdArgs,
                              StringRef DarwinLibName, bool AlwaysLink,
                              bool IsEmbedded, bool AddRPath) const {
  SmallString<128> Dir(getDriver().ResourceDir);
  llvm::sys::path::append(Dir, "lib", IsEmbedded ? "macho_embedded" : "darwin");

  SmallString<128> P(Dir);
  llvm::sys::path::append(P, DarwinLibName);

  // For now, allow missing resource libraries to support developers who may
  // not have compiler-rt checked out or integrated into their build (unless
  // we explicitly force linking with this library).
  if (AlwaysLink || getVFS().exists(P))
    CmdArgs.push_back(Args.MakeArgString(P));

  // Adding the rpaths might negatively interact when other rpaths are involved,
  // so we should make sure we add the rpaths last, after all user-specified
  // rpaths. This is currently true from this place, but we need to be
  // careful if this function is ever called before user's rpaths are emitted.
  if (AddRPath) {
    assert(DarwinLibName.endswith(".dylib") && "must be a dynamic library");

    // Add @executable_path to rpath to support having the dylib copied with
    // the executable.
    CmdArgs.push_back("-rpath");
    CmdArgs.push_back("@executable_path");

    // Add the path to the resource dir to rpath to support using the dylib
    // from the default location without copying.
    CmdArgs.push_back("-rpath");
    CmdArgs.push_back(Args.MakeArgString(Dir));
  }
}

void Darwin::addProfileRTLibs(const ArgList &Args,
                              ArgStringList &CmdArgs) const {
  if (!needsProfileRT(Args)) return;

  // TODO: Clean this up once autoconf is gone
  SmallString<128> P(getDriver().ResourceDir);
  llvm::sys::path::append(P, "lib", "darwin");
  const char *Library = "libclang_rt.profile_osx.a";

  // Select the appropriate runtime library for the target.
  if (isTargetWatchOS()) {
    Library = "libclang_rt.profile_watchos.a";
  } else if (isTargetWatchOSSimulator()) {
    llvm::sys::path::append(P, "libclang_rt.profile_watchossim.a");
    Library = getVFS().exists(P) ? "libclang_rt.profile_watchossim.a"
                                 : "libclang_rt.profile_watchos.a";
  } else if (isTargetTvOS()) {
    Library = "libclang_rt.profile_tvos.a";
  } else if (isTargetTvOSSimulator()) {
    llvm::sys::path::append(P, "libclang_rt.profile_tvossim.a");
    Library = getVFS().exists(P) ? "libclang_rt.profile_tvossim.a"
                                 : "libclang_rt.profile_tvos.a";
  } else if (isTargetIPhoneOS()) {
    Library = "libclang_rt.profile_ios.a";
  } else if (isTargetIOSSimulator()) {
    llvm::sys::path::append(P, "libclang_rt.profile_iossim.a");
    Library = getVFS().exists(P) ? "libclang_rt.profile_iossim.a"
                                 : "libclang_rt.profile_ios.a";
  } else {
    assert(isTargetMacOS() && "unexpected non MacOS platform");
  }
  AddLinkRuntimeLib(Args, CmdArgs, Library,
                    /*AlwaysLink*/ true);
  return;
}

void DarwinClang::AddLinkSanitizerLibArgs(const ArgList &Args,
                                          ArgStringList &CmdArgs,
                                          StringRef Sanitizer) const {
  if (!Args.hasArg(options::OPT_dynamiclib) &&
      !Args.hasArg(options::OPT_bundle)) {
    // Sanitizer runtime libraries requires C++.
    AddCXXStdlibLibArgs(Args, CmdArgs);
  }
  // ASan is not supported on watchOS.
  assert(isTargetMacOS() || isTargetIOSSimulator());
  StringRef OS = isTargetMacOS() ? "osx" : "iossim";
  AddLinkRuntimeLib(
      Args, CmdArgs,
      (Twine("libclang_rt.") + Sanitizer + "_" + OS + "_dynamic.dylib").str(),
      /*AlwaysLink*/ true, /*IsEmbedded*/ false,
      /*AddRPath*/ true);

  if (GetCXXStdlibType(Args) == ToolChain::CST_Libcxx) {
    // Add explicit dependcy on -lc++abi, as -lc++ doesn't re-export
    // all RTTI-related symbols that UBSan uses.
    CmdArgs.push_back("-lc++abi");
  }
}

void DarwinClang::AddLinkRuntimeLibArgs(const ArgList &Args,
                                        ArgStringList &CmdArgs) const {
  // Darwin only supports the compiler-rt based runtime libraries.
  switch (GetRuntimeLibType(Args)) {
  case ToolChain::RLT_CompilerRT:
    break;
  default:
    getDriver().Diag(diag::err_drv_unsupported_rtlib_for_platform)
        << Args.getLastArg(options::OPT_rtlib_EQ)->getValue() << "darwin";
    return;
  }

  // Darwin doesn't support real static executables, don't link any runtime
  // libraries with -static.
  if (Args.hasArg(options::OPT_static) ||
      Args.hasArg(options::OPT_fapple_kext) ||
      Args.hasArg(options::OPT_mkernel))
    return;

  // Reject -static-libgcc for now, we can deal with this when and if someone
  // cares. This is useful in situations where someone wants to statically link
  // something like libstdc++, and needs its runtime support routines.
  if (const Arg *A = Args.getLastArg(options::OPT_static_libgcc)) {
    getDriver().Diag(diag::err_drv_unsupported_opt) << A->getAsString(Args);
    return;
  }

  const SanitizerArgs &Sanitize = getSanitizerArgs();
  if (Sanitize.needsAsanRt())
    AddLinkSanitizerLibArgs(Args, CmdArgs, "asan");
  if (Sanitize.needsUbsanRt())
    AddLinkSanitizerLibArgs(Args, CmdArgs, "ubsan");
  if (Sanitize.needsTsanRt())
    AddLinkSanitizerLibArgs(Args, CmdArgs, "tsan");

  // Otherwise link libSystem, then the dynamic runtime library, and finally any
  // target specific static runtime library.
  CmdArgs.push_back("-lSystem");

  // Select the dynamic runtime library and the target specific static library.
  if (isTargetWatchOSBased()) {
    // We currently always need a static runtime library for watchOS.
    AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.watchos.a");
  } else if (isTargetTvOSBased()) {
    // We currently always need a static runtime library for tvOS.
    AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.tvos.a");
  } else if (isTargetIOSBased()) {
    // If we are compiling as iOS / simulator, don't attempt to link libgcc_s.1,
    // it never went into the SDK.
    // Linking against libgcc_s.1 isn't needed for iOS 5.0+
    if (isIPhoneOSVersionLT(5, 0) && !isTargetIOSSimulator() &&
        getTriple().getArch() != llvm::Triple::aarch64)
      CmdArgs.push_back("-lgcc_s.1");

    // We currently always need a static runtime library for iOS.
    AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.ios.a");
  } else {
    assert(isTargetMacOS() && "unexpected non MacOS platform");
    // The dynamic runtime library was merged with libSystem for 10.6 and
    // beyond; only 10.4 and 10.5 need an additional runtime library.
    if (isMacosxVersionLT(10, 5))
      CmdArgs.push_back("-lgcc_s.10.4");
    else if (isMacosxVersionLT(10, 6))
      CmdArgs.push_back("-lgcc_s.10.5");

    // For OS X, we thought we would only need a static runtime library when
    // targeting 10.4, to provide versions of the static functions which were
    // omitted from 10.4.dylib.
    //
    // Unfortunately, that turned out to not be true, because Darwin system
    // headers can still use eprintf on i386, and it is not exported from
    // libSystem. Therefore, we still must provide a runtime library just for
    // the tiny tiny handful of projects that *might* use that symbol.
    if (isMacosxVersionLT(10, 5)) {
      AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.10.4.a");
    } else {
      if (getTriple().getArch() == llvm::Triple::x86)
        AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.eprintf.a");
      AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.osx.a");
    }
  }
}

void Darwin::AddDeploymentTarget(DerivedArgList &Args) const {
  const OptTable &Opts = getDriver().getOpts();

  // Support allowing the SDKROOT environment variable used by xcrun and other
  // Xcode tools to define the default sysroot, by making it the default for
  // isysroot.
  if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
    // Warn if the path does not exist.
    if (!getVFS().exists(A->getValue()))
      getDriver().Diag(clang::diag::warn_missing_sysroot) << A->getValue();
  } else {
    if (char *env = ::getenv("SDKROOT")) {
      // We only use this value as the default if it is an absolute path,
      // exists, and it is not the root path.
      if (llvm::sys::path::is_absolute(env) && getVFS().exists(env) &&
          StringRef(env) != "/") {
        Args.append(Args.MakeSeparateArg(
            nullptr, Opts.getOption(options::OPT_isysroot), env));
      }
    }
  }

  Arg *OSXVersion = Args.getLastArg(options::OPT_mmacosx_version_min_EQ);
  Arg *iOSVersion = Args.getLastArg(options::OPT_miphoneos_version_min_EQ);
  Arg *TvOSVersion = Args.getLastArg(options::OPT_mtvos_version_min_EQ);
  Arg *WatchOSVersion = Args.getLastArg(options::OPT_mwatchos_version_min_EQ);

  if (OSXVersion && (iOSVersion || TvOSVersion || WatchOSVersion)) {
    getDriver().Diag(diag::err_drv_argument_not_allowed_with)
        << OSXVersion->getAsString(Args)
        << (iOSVersion ? iOSVersion :
            TvOSVersion ? TvOSVersion : WatchOSVersion)->getAsString(Args);
    iOSVersion = TvOSVersion = WatchOSVersion = nullptr;
  } else if (iOSVersion && (TvOSVersion || WatchOSVersion)) {
    getDriver().Diag(diag::err_drv_argument_not_allowed_with)
        << iOSVersion->getAsString(Args)
        << (TvOSVersion ? TvOSVersion : WatchOSVersion)->getAsString(Args);
    TvOSVersion = WatchOSVersion = nullptr;
  } else if (TvOSVersion && WatchOSVersion) {
     getDriver().Diag(diag::err_drv_argument_not_allowed_with)
        << TvOSVersion->getAsString(Args)
        << WatchOSVersion->getAsString(Args);
    WatchOSVersion = nullptr;
  } else if (!OSXVersion && !iOSVersion && !TvOSVersion && !WatchOSVersion) {
    // If no deployment target was specified on the command line, check for
    // environment defines.
    std::string OSXTarget;
    std::string iOSTarget;
    std::string TvOSTarget;
    std::string WatchOSTarget;

    if (char *env = ::getenv("MACOSX_DEPLOYMENT_TARGET"))
      OSXTarget = env;
    if (char *env = ::getenv("IPHONEOS_DEPLOYMENT_TARGET"))
      iOSTarget = env;
    if (char *env = ::getenv("TVOS_DEPLOYMENT_TARGET"))
      TvOSTarget = env;
    if (char *env = ::getenv("WATCHOS_DEPLOYMENT_TARGET"))
      WatchOSTarget = env;

    // If there is no command-line argument to specify the Target version and
    // no environment variable defined, see if we can set the default based
    // on -isysroot.
    if (OSXTarget.empty() && iOSTarget.empty() && WatchOSTarget.empty() &&
        Args.hasArg(options::OPT_isysroot)) {
      if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
        StringRef isysroot = A->getValue();
        // Assume SDK has path: SOME_PATH/SDKs/PlatformXX.YY.sdk
        size_t BeginSDK = isysroot.rfind("SDKs/");
        size_t EndSDK = isysroot.rfind(".sdk");
        if (BeginSDK != StringRef::npos && EndSDK != StringRef::npos) {
          StringRef SDK = isysroot.slice(BeginSDK + 5, EndSDK);
          // Slice the version number out.
          // Version number is between the first and the last number.
          size_t StartVer = SDK.find_first_of("0123456789");
          size_t EndVer = SDK.find_last_of("0123456789");
          if (StartVer != StringRef::npos && EndVer > StartVer) {
            StringRef Version = SDK.slice(StartVer, EndVer + 1);
            if (SDK.startswith("iPhoneOS") ||
                SDK.startswith("iPhoneSimulator"))
              iOSTarget = Version;
            else if (SDK.startswith("MacOSX"))
              OSXTarget = Version;
            else if (SDK.startswith("WatchOS") ||
                     SDK.startswith("WatchSimulator"))
              WatchOSTarget = Version;
            else if (SDK.startswith("AppleTVOS") ||
                     SDK.startswith("AppleTVSimulator"))
              TvOSTarget = Version;
          }
        }
      }
    }

    // If no OSX or iOS target has been specified, try to guess platform
    // from arch name and compute the version from the triple.
    if (OSXTarget.empty() && iOSTarget.empty() && TvOSTarget.empty() &&
        WatchOSTarget.empty()) {
      StringRef MachOArchName = getMachOArchName(Args);
      unsigned Major, Minor, Micro;
      if (MachOArchName == "armv7" || MachOArchName == "armv7s" ||
          MachOArchName == "arm64") {
        getTriple().getiOSVersion(Major, Minor, Micro);
        llvm::raw_string_ostream(iOSTarget) << Major << '.' << Minor << '.'
                                            << Micro;
      } else if (MachOArchName == "armv7k") {
        getTriple().getWatchOSVersion(Major, Minor, Micro);
        llvm::raw_string_ostream(WatchOSTarget) << Major << '.' << Minor << '.'
                                                << Micro;
      } else if (MachOArchName != "armv6m" && MachOArchName != "armv7m" &&
                 MachOArchName != "armv7em") {
        if (!getTriple().getMacOSXVersion(Major, Minor, Micro)) {
          getDriver().Diag(diag::err_drv_invalid_darwin_version)
              << getTriple().getOSName();
        }
        llvm::raw_string_ostream(OSXTarget) << Major << '.' << Minor << '.'
                                            << Micro;
      }
    }

    // Do not allow conflicts with the watchOS target.
    if (!WatchOSTarget.empty() && (!iOSTarget.empty() || !TvOSTarget.empty())) {
      getDriver().Diag(diag::err_drv_conflicting_deployment_targets)
        << "WATCHOS_DEPLOYMENT_TARGET"
        << (!iOSTarget.empty() ? "IPHONEOS_DEPLOYMENT_TARGET" :
            "TVOS_DEPLOYMENT_TARGET");
    }

    // Do not allow conflicts with the tvOS target.
    if (!TvOSTarget.empty() && !iOSTarget.empty()) {
      getDriver().Diag(diag::err_drv_conflicting_deployment_targets)
        << "TVOS_DEPLOYMENT_TARGET"
        << "IPHONEOS_DEPLOYMENT_TARGET";
    }

    // Allow conflicts among OSX and iOS for historical reasons, but choose the
    // default platform.
    if (!OSXTarget.empty() && (!iOSTarget.empty() ||
                               !WatchOSTarget.empty() ||
                               !TvOSTarget.empty())) {
      if (getTriple().getArch() == llvm::Triple::arm ||
          getTriple().getArch() == llvm::Triple::aarch64 ||
          getTriple().getArch() == llvm::Triple::thumb)
        OSXTarget = "";
      else
        iOSTarget = WatchOSTarget = TvOSTarget = "";
    }

    if (!OSXTarget.empty()) {
      const Option O = Opts.getOption(options::OPT_mmacosx_version_min_EQ);
      OSXVersion = Args.MakeJoinedArg(nullptr, O, OSXTarget);
      Args.append(OSXVersion);
    } else if (!iOSTarget.empty()) {
      const Option O = Opts.getOption(options::OPT_miphoneos_version_min_EQ);
      iOSVersion = Args.MakeJoinedArg(nullptr, O, iOSTarget);
      Args.append(iOSVersion);
    } else if (!TvOSTarget.empty()) {
      const Option O = Opts.getOption(options::OPT_mtvos_version_min_EQ);
      TvOSVersion = Args.MakeJoinedArg(nullptr, O, TvOSTarget);
      Args.append(TvOSVersion);
    } else if (!WatchOSTarget.empty()) {
      const Option O = Opts.getOption(options::OPT_mwatchos_version_min_EQ);
      WatchOSVersion = Args.MakeJoinedArg(nullptr, O, WatchOSTarget);
      Args.append(WatchOSVersion);
    }
  }

  DarwinPlatformKind Platform;
  if (OSXVersion)
    Platform = MacOS;
  else if (iOSVersion)
    Platform = IPhoneOS;
  else if (TvOSVersion)
    Platform = TvOS;
  else if (WatchOSVersion)
    Platform = WatchOS;
  else
    llvm_unreachable("Unable to infer Darwin variant");

  // Set the tool chain target information.
  unsigned Major, Minor, Micro;
  bool HadExtra;
  if (Platform == MacOS) {
    assert((!iOSVersion && !TvOSVersion && !WatchOSVersion) &&
           "Unknown target platform!");
    if (!Driver::GetReleaseVersion(OSXVersion->getValue(), Major, Minor, Micro,
                                   HadExtra) ||
        HadExtra || Major != 10 || Minor >= 100 || Micro >= 100)
      getDriver().Diag(diag::err_drv_invalid_version_number)
          << OSXVersion->getAsString(Args);
  } else if (Platform == IPhoneOS) {
    assert(iOSVersion && "Unknown target platform!");
    if (!Driver::GetReleaseVersion(iOSVersion->getValue(), Major, Minor, Micro,
                                   HadExtra) ||
        HadExtra || Major >= 10 || Minor >= 100 || Micro >= 100)
      getDriver().Diag(diag::err_drv_invalid_version_number)
          << iOSVersion->getAsString(Args);
  } else if (Platform == TvOS) {
    if (!Driver::GetReleaseVersion(TvOSVersion->getValue(), Major, Minor,
                                   Micro, HadExtra) || HadExtra ||
        Major >= 10 || Minor >= 100 || Micro >= 100)
      getDriver().Diag(diag::err_drv_invalid_version_number)
          << TvOSVersion->getAsString(Args);
  } else if (Platform == WatchOS) {
    if (!Driver::GetReleaseVersion(WatchOSVersion->getValue(), Major, Minor,
                                   Micro, HadExtra) || HadExtra ||
        Major >= 10 || Minor >= 100 || Micro >= 100)
      getDriver().Diag(diag::err_drv_invalid_version_number)
          << WatchOSVersion->getAsString(Args);
  } else
    llvm_unreachable("unknown kind of Darwin platform");

  // Recognize iOS targets with an x86 architecture as the iOS simulator.
  if (iOSVersion && (getTriple().getArch() == llvm::Triple::x86 ||
                     getTriple().getArch() == llvm::Triple::x86_64))
    Platform = IPhoneOSSimulator;
  if (TvOSVersion && (getTriple().getArch() == llvm::Triple::x86 ||
                      getTriple().getArch() == llvm::Triple::x86_64))
    Platform = TvOSSimulator;
  if (WatchOSVersion && (getTriple().getArch() == llvm::Triple::x86 ||
                         getTriple().getArch() == llvm::Triple::x86_64))
    Platform = WatchOSSimulator;

  setTarget(Platform, Major, Minor, Micro);
}

void DarwinClang::AddCXXStdlibLibArgs(const ArgList &Args,
                                      ArgStringList &CmdArgs) const {
  CXXStdlibType Type = GetCXXStdlibType(Args);

  switch (Type) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    break;

  case ToolChain::CST_Libstdcxx:
    // Unfortunately, -lstdc++ doesn't always exist in the standard search path;
    // it was previously found in the gcc lib dir. However, for all the Darwin
    // platforms we care about it was -lstdc++.6, so we search for that
    // explicitly if we can't see an obvious -lstdc++ candidate.

    // Check in the sysroot first.
    if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
      SmallString<128> P(A->getValue());
      llvm::sys::path::append(P, "usr", "lib", "libstdc++.dylib");

      if (!getVFS().exists(P)) {
        llvm::sys::path::remove_filename(P);
        llvm::sys::path::append(P, "libstdc++.6.dylib");
        if (getVFS().exists(P)) {
          CmdArgs.push_back(Args.MakeArgString(P));
          return;
        }
      }
    }

    // Otherwise, look in the root.
    // FIXME: This should be removed someday when we don't have to care about
    // 10.6 and earlier, where /usr/lib/libstdc++.dylib does not exist.
    if (!getVFS().exists("/usr/lib/libstdc++.dylib") &&
        getVFS().exists("/usr/lib/libstdc++.6.dylib")) {
      CmdArgs.push_back("/usr/lib/libstdc++.6.dylib");
      return;
    }

    // Otherwise, let the linker search.
    CmdArgs.push_back("-lstdc++");
    break;
  }
}

void DarwinClang::AddCCKextLibArgs(const ArgList &Args,
                                   ArgStringList &CmdArgs) const {

  // For Darwin platforms, use the compiler-rt-based support library
  // instead of the gcc-provided one (which is also incidentally
  // only present in the gcc lib dir, which makes it hard to find).

  SmallString<128> P(getDriver().ResourceDir);
  llvm::sys::path::append(P, "lib", "darwin");

  // Use the newer cc_kext for iOS ARM after 6.0.
  if (isTargetWatchOS()) {
    llvm::sys::path::append(P, "libclang_rt.cc_kext_watchos.a");
  } else if (isTargetTvOS()) {
    llvm::sys::path::append(P, "libclang_rt.cc_kext_tvos.a");
  } else if (isTargetIPhoneOS()) {
    llvm::sys::path::append(P, "libclang_rt.cc_kext_ios.a");
  } else {
    llvm::sys::path::append(P, "libclang_rt.cc_kext.a");
  }

  // For now, allow missing resource libraries to support developers who may
  // not have compiler-rt checked out or integrated into their build.
  if (getVFS().exists(P))
    CmdArgs.push_back(Args.MakeArgString(P));
}

DerivedArgList *MachO::TranslateArgs(const DerivedArgList &Args,
                                     const char *BoundArch) const {
  DerivedArgList *DAL = new DerivedArgList(Args.getBaseArgs());
  const OptTable &Opts = getDriver().getOpts();

  // FIXME: We really want to get out of the tool chain level argument
  // translation business, as it makes the driver functionality much
  // more opaque. For now, we follow gcc closely solely for the
  // purpose of easily achieving feature parity & testability. Once we
  // have something that works, we should reevaluate each translation
  // and try to push it down into tool specific logic.

  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT_Xarch__)) {
      // Skip this argument unless the architecture matches either the toolchain
      // triple arch, or the arch being bound.
      llvm::Triple::ArchType XarchArch =
          tools::darwin::getArchTypeForMachOArchName(A->getValue(0));
      if (!(XarchArch == getArch() ||
            (BoundArch &&
             XarchArch ==
                 tools::darwin::getArchTypeForMachOArchName(BoundArch))))
        continue;

      Arg *OriginalArg = A;
      unsigned Index = Args.getBaseArgs().MakeIndex(A->getValue(1));
      unsigned Prev = Index;
      std::unique_ptr<Arg> XarchArg(Opts.ParseOneArg(Args, Index));

      // If the argument parsing failed or more than one argument was
      // consumed, the -Xarch_ argument's parameter tried to consume
      // extra arguments. Emit an error and ignore.
      //
      // We also want to disallow any options which would alter the
      // driver behavior; that isn't going to work in our model. We
      // use isDriverOption() as an approximation, although things
      // like -O4 are going to slip through.
      if (!XarchArg || Index > Prev + 1) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_with_args)
            << A->getAsString(Args);
        continue;
      } else if (XarchArg->getOption().hasFlag(options::DriverOption)) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_isdriver)
            << A->getAsString(Args);
        continue;
      }

      XarchArg->setBaseArg(A);

      A = XarchArg.release();
      DAL->AddSynthesizedArg(A);

      // Linker input arguments require custom handling. The problem is that we
      // have already constructed the phase actions, so we can not treat them as
      // "input arguments".
      if (A->getOption().hasFlag(options::LinkerInput)) {
        // Convert the argument into individual Zlinker_input_args.
        for (const char *Value : A->getValues()) {
          DAL->AddSeparateArg(
              OriginalArg, Opts.getOption(options::OPT_Zlinker_input), Value);
        }
        continue;
      }
    }

    // Sob. These is strictly gcc compatible for the time being. Apple
    // gcc translates options twice, which means that self-expanding
    // options add duplicates.
    switch ((options::ID)A->getOption().getID()) {
    default:
      DAL->append(A);
      break;

    case options::OPT_mkernel:
    case options::OPT_fapple_kext:
      DAL->append(A);
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_static));
      break;

    case options::OPT_dependency_file:
      DAL->AddSeparateArg(A, Opts.getOption(options::OPT_MF), A->getValue());
      break;

    case options::OPT_gfull:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_g_Flag));
      DAL->AddFlagArg(
          A, Opts.getOption(options::OPT_fno_eliminate_unused_debug_symbols));
      break;

    case options::OPT_gused:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_g_Flag));
      DAL->AddFlagArg(
          A, Opts.getOption(options::OPT_feliminate_unused_debug_symbols));
      break;

    case options::OPT_shared:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_dynamiclib));
      break;

    case options::OPT_fconstant_cfstrings:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_mconstant_cfstrings));
      break;

    case options::OPT_fno_constant_cfstrings:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_mno_constant_cfstrings));
      break;

    case options::OPT_Wnonportable_cfstrings:
      DAL->AddFlagArg(A,
                      Opts.getOption(options::OPT_mwarn_nonportable_cfstrings));
      break;

    case options::OPT_Wno_nonportable_cfstrings:
      DAL->AddFlagArg(
          A, Opts.getOption(options::OPT_mno_warn_nonportable_cfstrings));
      break;

    case options::OPT_fpascal_strings:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_mpascal_strings));
      break;

    case options::OPT_fno_pascal_strings:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_mno_pascal_strings));
      break;
    }
  }

  if (getTriple().getArch() == llvm::Triple::x86 ||
      getTriple().getArch() == llvm::Triple::x86_64)
    if (!Args.hasArgNoClaim(options::OPT_mtune_EQ))
      DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_mtune_EQ),
                        "core2");

  // Add the arch options based on the particular spelling of -arch, to match
  // how the driver driver works.
  if (BoundArch) {
    StringRef Name = BoundArch;
    const Option MCpu = Opts.getOption(options::OPT_mcpu_EQ);
    const Option MArch = Opts.getOption(options::OPT_march_EQ);

    // This code must be kept in sync with LLVM's getArchTypeForDarwinArch,
    // which defines the list of which architectures we accept.
    if (Name == "ppc")
      ;
    else if (Name == "ppc601")
      DAL->AddJoinedArg(nullptr, MCpu, "601");
    else if (Name == "ppc603")
      DAL->AddJoinedArg(nullptr, MCpu, "603");
    else if (Name == "ppc604")
      DAL->AddJoinedArg(nullptr, MCpu, "604");
    else if (Name == "ppc604e")
      DAL->AddJoinedArg(nullptr, MCpu, "604e");
    else if (Name == "ppc750")
      DAL->AddJoinedArg(nullptr, MCpu, "750");
    else if (Name == "ppc7400")
      DAL->AddJoinedArg(nullptr, MCpu, "7400");
    else if (Name == "ppc7450")
      DAL->AddJoinedArg(nullptr, MCpu, "7450");
    else if (Name == "ppc970")
      DAL->AddJoinedArg(nullptr, MCpu, "970");

    else if (Name == "ppc64" || Name == "ppc64le")
      DAL->AddFlagArg(nullptr, Opts.getOption(options::OPT_m64));

    else if (Name == "i386")
      ;
    else if (Name == "i486")
      DAL->AddJoinedArg(nullptr, MArch, "i486");
    else if (Name == "i586")
      DAL->AddJoinedArg(nullptr, MArch, "i586");
    else if (Name == "i686")
      DAL->AddJoinedArg(nullptr, MArch, "i686");
    else if (Name == "pentium")
      DAL->AddJoinedArg(nullptr, MArch, "pentium");
    else if (Name == "pentium2")
      DAL->AddJoinedArg(nullptr, MArch, "pentium2");
    else if (Name == "pentpro")
      DAL->AddJoinedArg(nullptr, MArch, "pentiumpro");
    else if (Name == "pentIIm3")
      DAL->AddJoinedArg(nullptr, MArch, "pentium2");

    else if (Name == "x86_64")
      DAL->AddFlagArg(nullptr, Opts.getOption(options::OPT_m64));
    else if (Name == "x86_64h") {
      DAL->AddFlagArg(nullptr, Opts.getOption(options::OPT_m64));
      DAL->AddJoinedArg(nullptr, MArch, "x86_64h");
    }

    else if (Name == "arm")
      DAL->AddJoinedArg(nullptr, MArch, "armv4t");
    else if (Name == "armv4t")
      DAL->AddJoinedArg(nullptr, MArch, "armv4t");
    else if (Name == "armv5")
      DAL->AddJoinedArg(nullptr, MArch, "armv5tej");
    else if (Name == "xscale")
      DAL->AddJoinedArg(nullptr, MArch, "xscale");
    else if (Name == "armv6")
      DAL->AddJoinedArg(nullptr, MArch, "armv6k");
    else if (Name == "armv6m")
      DAL->AddJoinedArg(nullptr, MArch, "armv6m");
    else if (Name == "armv7")
      DAL->AddJoinedArg(nullptr, MArch, "armv7a");
    else if (Name == "armv7em")
      DAL->AddJoinedArg(nullptr, MArch, "armv7em");
    else if (Name == "armv7k")
      DAL->AddJoinedArg(nullptr, MArch, "armv7k");
    else if (Name == "armv7m")
      DAL->AddJoinedArg(nullptr, MArch, "armv7m");
    else if (Name == "armv7s")
      DAL->AddJoinedArg(nullptr, MArch, "armv7s");
  }

  return DAL;
}

void MachO::AddLinkRuntimeLibArgs(const ArgList &Args,
                                  ArgStringList &CmdArgs) const {
  // Embedded targets are simple at the moment, not supporting sanitizers and
  // with different libraries for each member of the product { static, PIC } x
  // { hard-float, soft-float }
  llvm::SmallString<32> CompilerRT = StringRef("libclang_rt.");
  CompilerRT +=
      (tools::arm::getARMFloatABI(*this, Args) == tools::arm::FloatABI::Hard)
          ? "hard"
          : "soft";
  CompilerRT += Args.hasArg(options::OPT_fPIC) ? "_pic.a" : "_static.a";

  AddLinkRuntimeLib(Args, CmdArgs, CompilerRT, false, true);
}

DerivedArgList *Darwin::TranslateArgs(const DerivedArgList &Args,
                                      const char *BoundArch) const {
  // First get the generic Apple args, before moving onto Darwin-specific ones.
  DerivedArgList *DAL = MachO::TranslateArgs(Args, BoundArch);
  const OptTable &Opts = getDriver().getOpts();

  // If no architecture is bound, none of the translations here are relevant.
  if (!BoundArch)
    return DAL;

  // Add an explicit version min argument for the deployment target. We do this
  // after argument translation because -Xarch_ arguments may add a version min
  // argument.
  AddDeploymentTarget(*DAL);

  // For iOS 6, undo the translation to add -static for -mkernel/-fapple-kext.
  // FIXME: It would be far better to avoid inserting those -static arguments,
  // but we can't check the deployment target in the translation code until
  // it is set here.
  if (isTargetWatchOSBased() ||
      (isTargetIOSBased() && !isIPhoneOSVersionLT(6, 0))) {
    for (ArgList::iterator it = DAL->begin(), ie = DAL->end(); it != ie; ) {
      Arg *A = *it;
      ++it;
      if (A->getOption().getID() != options::OPT_mkernel &&
          A->getOption().getID() != options::OPT_fapple_kext)
        continue;
      assert(it != ie && "unexpected argument translation");
      A = *it;
      assert(A->getOption().getID() == options::OPT_static &&
             "missing expected -static argument");
      it = DAL->getArgs().erase(it);
    }
  }

  // Default to use libc++ on OS X 10.9+ and iOS 7+.
  if (((isTargetMacOS() && !isMacosxVersionLT(10, 9)) ||
       (isTargetIOSBased() && !isIPhoneOSVersionLT(7, 0)) ||
       isTargetWatchOSBased()) &&
      !Args.getLastArg(options::OPT_stdlib_EQ))
    DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_stdlib_EQ),
                      "libc++");

  // Validate the C++ standard library choice.
  CXXStdlibType Type = GetCXXStdlibType(*DAL);
  if (Type == ToolChain::CST_Libcxx) {
    // Check whether the target provides libc++.
    StringRef where;

    // Complain about targeting iOS < 5.0 in any way.
    if (isTargetIOSBased() && isIPhoneOSVersionLT(5, 0))
      where = "iOS 5.0";

    if (where != StringRef()) {
      getDriver().Diag(clang::diag::err_drv_invalid_libcxx_deployment) << where;
    }
  }

  return DAL;
}

bool MachO::IsUnwindTablesDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool MachO::UseDwarfDebugFlags() const {
  if (const char *S = ::getenv("RC_DEBUG_OPTIONS"))
    return S[0] != '\0';
  return false;
}

bool Darwin::UseSjLjExceptions(const ArgList &Args) const {
  // Darwin uses SjLj exceptions on ARM.
  if (getTriple().getArch() != llvm::Triple::arm &&
      getTriple().getArch() != llvm::Triple::thumb)
    return false;

  // Only watchOS uses the new DWARF/Compact unwinding method.
  return !isTargetWatchOS();
}

bool MachO::isPICDefault() const { return true; }

bool MachO::isPIEDefault() const { return false; }

bool MachO::isPICDefaultForced() const {
  return (getArch() == llvm::Triple::x86_64 ||
          getArch() == llvm::Triple::aarch64);
}

bool MachO::SupportsProfiling() const {
  // Profiling instrumentation is only supported on x86.
  return getArch() == llvm::Triple::x86 || getArch() == llvm::Triple::x86_64;
}

void Darwin::addMinVersionArgs(const ArgList &Args,
                               ArgStringList &CmdArgs) const {
  VersionTuple TargetVersion = getTargetVersion();

  if (isTargetWatchOS())
    CmdArgs.push_back("-watchos_version_min");
  else if (isTargetWatchOSSimulator())
    CmdArgs.push_back("-watchos_simulator_version_min");
  else if (isTargetTvOS())
    CmdArgs.push_back("-tvos_version_min");
  else if (isTargetTvOSSimulator())
    CmdArgs.push_back("-tvos_simulator_version_min");
  else if (isTargetIOSSimulator())
    CmdArgs.push_back("-ios_simulator_version_min");
  else if (isTargetIOSBased())
    CmdArgs.push_back("-iphoneos_version_min");
  else {
    assert(isTargetMacOS() && "unexpected target");
    CmdArgs.push_back("-macosx_version_min");
  }

  CmdArgs.push_back(Args.MakeArgString(TargetVersion.getAsString()));
}

void Darwin::addStartObjectFileArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  // Derived from startfile spec.
  if (Args.hasArg(options::OPT_dynamiclib)) {
    // Derived from darwin_dylib1 spec.
    if (isTargetWatchOSBased()) {
      ; // watchOS does not need dylib1.o.
    } else if (isTargetIOSSimulator()) {
      ; // iOS simulator does not need dylib1.o.
    } else if (isTargetIPhoneOS()) {
      if (isIPhoneOSVersionLT(3, 1))
        CmdArgs.push_back("-ldylib1.o");
    } else {
      if (isMacosxVersionLT(10, 5))
        CmdArgs.push_back("-ldylib1.o");
      else if (isMacosxVersionLT(10, 6))
        CmdArgs.push_back("-ldylib1.10.5.o");
    }
  } else {
    if (Args.hasArg(options::OPT_bundle)) {
      if (!Args.hasArg(options::OPT_static)) {
        // Derived from darwin_bundle1 spec.
        if (isTargetWatchOSBased()) {
          ; // watchOS does not need bundle1.o.
        } else if (isTargetIOSSimulator()) {
          ; // iOS simulator does not need bundle1.o.
        } else if (isTargetIPhoneOS()) {
          if (isIPhoneOSVersionLT(3, 1))
            CmdArgs.push_back("-lbundle1.o");
        } else {
          if (isMacosxVersionLT(10, 6))
            CmdArgs.push_back("-lbundle1.o");
        }
      }
    } else {
      if (Args.hasArg(options::OPT_pg) && SupportsProfiling()) {
        if (Args.hasArg(options::OPT_static) ||
            Args.hasArg(options::OPT_object) ||
            Args.hasArg(options::OPT_preload)) {
          CmdArgs.push_back("-lgcrt0.o");
        } else {
          CmdArgs.push_back("-lgcrt1.o");

          // darwin_crt2 spec is empty.
        }
        // By default on OS X 10.8 and later, we don't link with a crt1.o
        // file and the linker knows to use _main as the entry point.  But,
        // when compiling with -pg, we need to link with the gcrt1.o file,
        // so pass the -no_new_main option to tell the linker to use the
        // "start" symbol as the entry point.
        if (isTargetMacOS() && !isMacosxVersionLT(10, 8))
          CmdArgs.push_back("-no_new_main");
      } else {
        if (Args.hasArg(options::OPT_static) ||
            Args.hasArg(options::OPT_object) ||
            Args.hasArg(options::OPT_preload)) {
          CmdArgs.push_back("-lcrt0.o");
        } else {
          // Derived from darwin_crt1 spec.
          if (isTargetWatchOSBased()) {
            ; // watchOS does not need crt1.o.
          } else if (isTargetIOSSimulator()) {
            ; // iOS simulator does not need crt1.o.
          } else if (isTargetIPhoneOS()) {
            if (getArch() == llvm::Triple::aarch64)
              ; // iOS does not need any crt1 files for arm64
            else if (isIPhoneOSVersionLT(3, 1))
              CmdArgs.push_back("-lcrt1.o");
            else if (isIPhoneOSVersionLT(6, 0))
              CmdArgs.push_back("-lcrt1.3.1.o");
          } else {
            if (isMacosxVersionLT(10, 5))
              CmdArgs.push_back("-lcrt1.o");
            else if (isMacosxVersionLT(10, 6))
              CmdArgs.push_back("-lcrt1.10.5.o");
            else if (isMacosxVersionLT(10, 8))
              CmdArgs.push_back("-lcrt1.10.6.o");

            // darwin_crt2 spec is empty.
          }
        }
      }
    }
  }

  if (!isTargetIPhoneOS() && Args.hasArg(options::OPT_shared_libgcc) &&
      !isTargetWatchOS() &&
      isMacosxVersionLT(10, 5)) {
    const char *Str = Args.MakeArgString(GetFilePath("crt3.o"));
    CmdArgs.push_back(Str);
  }
}

bool Darwin::SupportsObjCGC() const { return isTargetMacOS(); }

void Darwin::CheckObjCARC() const {
  if (isTargetIOSBased() || isTargetWatchOSBased() ||
      (isTargetMacOS() && !isMacosxVersionLT(10, 6)))
    return;
  getDriver().Diag(diag::err_arc_unsupported_on_toolchain);
}

SanitizerMask Darwin::getSupportedSanitizers() const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  if (isTargetMacOS() || isTargetIOSSimulator())
    Res |= SanitizerKind::Address;
  if (isTargetMacOS()) {
    if (!isMacosxVersionLT(10, 9))
      Res |= SanitizerKind::Vptr;
    Res |= SanitizerKind::SafeStack;
    Res |= SanitizerKind::Thread;
  }
  return Res;
}

/// Generic_GCC - A tool chain using the 'gcc' command to perform
/// all subcommands; this relies on gcc translating the majority of
/// command line options.

/// \brief Parse a GCCVersion object out of a string of text.
///
/// This is the primary means of forming GCCVersion objects.
/*static*/
Generic_GCC::GCCVersion Linux::GCCVersion::Parse(StringRef VersionText) {
  const GCCVersion BadVersion = {VersionText.str(), -1, -1, -1, "", "", ""};
  std::pair<StringRef, StringRef> First = VersionText.split('.');
  std::pair<StringRef, StringRef> Second = First.second.split('.');

  GCCVersion GoodVersion = {VersionText.str(), -1, -1, -1, "", "", ""};
  if (First.first.getAsInteger(10, GoodVersion.Major) || GoodVersion.Major < 0)
    return BadVersion;
  GoodVersion.MajorStr = First.first.str();
  if (Second.first.getAsInteger(10, GoodVersion.Minor) || GoodVersion.Minor < 0)
    return BadVersion;
  GoodVersion.MinorStr = Second.first.str();

  // First look for a number prefix and parse that if present. Otherwise just
  // stash the entire patch string in the suffix, and leave the number
  // unspecified. This covers versions strings such as:
  //   4.4
  //   4.4.0
  //   4.4.x
  //   4.4.2-rc4
  //   4.4.x-patched
  // And retains any patch number it finds.
  StringRef PatchText = GoodVersion.PatchSuffix = Second.second.str();
  if (!PatchText.empty()) {
    if (size_t EndNumber = PatchText.find_first_not_of("0123456789")) {
      // Try to parse the number and any suffix.
      if (PatchText.slice(0, EndNumber).getAsInteger(10, GoodVersion.Patch) ||
          GoodVersion.Patch < 0)
        return BadVersion;
      GoodVersion.PatchSuffix = PatchText.substr(EndNumber);
    }
  }

  return GoodVersion;
}

/// \brief Less-than for GCCVersion, implementing a Strict Weak Ordering.
bool Generic_GCC::GCCVersion::isOlderThan(int RHSMajor, int RHSMinor,
                                          int RHSPatch,
                                          StringRef RHSPatchSuffix) const {
  if (Major != RHSMajor)
    return Major < RHSMajor;
  if (Minor != RHSMinor)
    return Minor < RHSMinor;
  if (Patch != RHSPatch) {
    // Note that versions without a specified patch sort higher than those with
    // a patch.
    if (RHSPatch == -1)
      return true;
    if (Patch == -1)
      return false;

    // Otherwise just sort on the patch itself.
    return Patch < RHSPatch;
  }
  if (PatchSuffix != RHSPatchSuffix) {
    // Sort empty suffixes higher.
    if (RHSPatchSuffix.empty())
      return true;
    if (PatchSuffix.empty())
      return false;

    // Provide a lexicographic sort to make this a total ordering.
    return PatchSuffix < RHSPatchSuffix;
  }

  // The versions are equal.
  return false;
}

static llvm::StringRef getGCCToolchainDir(const ArgList &Args) {
  const Arg *A = Args.getLastArg(options::OPT_gcc_toolchain);
  if (A)
    return A->getValue();
  return GCC_INSTALL_PREFIX;
}

/// \brief Initialize a GCCInstallationDetector from the driver.
///
/// This performs all of the autodetection and sets up the various paths.
/// Once constructed, a GCCInstallationDetector is essentially immutable.
///
/// FIXME: We shouldn't need an explicit TargetTriple parameter here, and
/// should instead pull the target out of the driver. This is currently
/// necessary because the driver doesn't store the final version of the target
/// triple.
void Generic_GCC::GCCInstallationDetector::init(
    const llvm::Triple &TargetTriple, const ArgList &Args,
    ArrayRef<std::string> ExtraTripleAliases) {
  llvm::Triple BiarchVariantTriple = TargetTriple.isArch32Bit()
                                         ? TargetTriple.get64BitArchVariant()
                                         : TargetTriple.get32BitArchVariant();
  // The library directories which may contain GCC installations.
  SmallVector<StringRef, 4> CandidateLibDirs, CandidateBiarchLibDirs;
  // The compatible GCC triples for this particular architecture.
  SmallVector<StringRef, 16> CandidateTripleAliases;
  SmallVector<StringRef, 16> CandidateBiarchTripleAliases;
  CollectLibDirsAndTriples(TargetTriple, BiarchVariantTriple, CandidateLibDirs,
                           CandidateTripleAliases, CandidateBiarchLibDirs,
                           CandidateBiarchTripleAliases);

  // Compute the set of prefixes for our search.
  SmallVector<std::string, 8> Prefixes(D.PrefixDirs.begin(),
                                       D.PrefixDirs.end());

  StringRef GCCToolchainDir = getGCCToolchainDir(Args);
  if (GCCToolchainDir != "") {
    if (GCCToolchainDir.back() == '/')
      GCCToolchainDir = GCCToolchainDir.drop_back(); // remove the /

    Prefixes.push_back(GCCToolchainDir);
  } else {
    // If we have a SysRoot, try that first.
    if (!D.SysRoot.empty()) {
      Prefixes.push_back(D.SysRoot);
      Prefixes.push_back(D.SysRoot + "/usr");
    }

    // Then look for gcc installed alongside clang.
    Prefixes.push_back(D.InstalledDir + "/..");

    // And finally in /usr.
    if (D.SysRoot.empty())
      Prefixes.push_back("/usr");
  }

  // Loop over the various components which exist and select the best GCC
  // installation available. GCC installs are ranked by version number.
  Version = GCCVersion::Parse("0.0.0");
  for (const std::string &Prefix : Prefixes) {
    if (!D.getVFS().exists(Prefix))
      continue;
    for (StringRef Suffix : CandidateLibDirs) {
      const std::string LibDir = Prefix + Suffix.str();
      if (!D.getVFS().exists(LibDir))
        continue;
      for (StringRef Candidate : ExtraTripleAliases) // Try these first.
        ScanLibDirForGCCTriple(TargetTriple, Args, LibDir, Candidate);
      for (StringRef Candidate : CandidateTripleAliases)
        ScanLibDirForGCCTriple(TargetTriple, Args, LibDir, Candidate);
    }
    for (StringRef Suffix : CandidateBiarchLibDirs) {
      const std::string LibDir = Prefix + Suffix.str();
      if (!D.getVFS().exists(LibDir))
        continue;
      for (StringRef Candidate : CandidateBiarchTripleAliases)
        ScanLibDirForGCCTriple(TargetTriple, Args, LibDir, Candidate,
                               /*NeedsBiarchSuffix=*/ true);
    }
  }
}

void Generic_GCC::GCCInstallationDetector::print(raw_ostream &OS) const {
  for (const auto &InstallPath : CandidateGCCInstallPaths)
    OS << "Found candidate GCC installation: " << InstallPath << "\n";

  if (!GCCInstallPath.empty())
    OS << "Selected GCC installation: " << GCCInstallPath << "\n";

  for (const auto &Multilib : Multilibs)
    OS << "Candidate multilib: " << Multilib << "\n";

  if (Multilibs.size() != 0 || !SelectedMultilib.isDefault())
    OS << "Selected multilib: " << SelectedMultilib << "\n";
}

bool Generic_GCC::GCCInstallationDetector::getBiarchSibling(Multilib &M) const {
  if (BiarchSibling.hasValue()) {
    M = BiarchSibling.getValue();
    return true;
  }
  return false;
}

/*static*/ void Generic_GCC::GCCInstallationDetector::CollectLibDirsAndTriples(
    const llvm::Triple &TargetTriple, const llvm::Triple &BiarchTriple,
    SmallVectorImpl<StringRef> &LibDirs,
    SmallVectorImpl<StringRef> &TripleAliases,
    SmallVectorImpl<StringRef> &BiarchLibDirs,
    SmallVectorImpl<StringRef> &BiarchTripleAliases) {
  // Declare a bunch of static data sets that we'll select between below. These
  // are specifically designed to always refer to string literals to avoid any
  // lifetime or initialization issues.
  static const char *const AArch64LibDirs[] = {"/lib64", "/lib"};
  static const char *const AArch64Triples[] = {
      "aarch64-none-linux-gnu", "aarch64-linux-gnu", "aarch64-linux-android",
      "aarch64-redhat-linux"};
  static const char *const AArch64beLibDirs[] = {"/lib"};
  static const char *const AArch64beTriples[] = {"aarch64_be-none-linux-gnu",
                                                 "aarch64_be-linux-gnu"};

  static const char *const ARMLibDirs[] = {"/lib"};
  static const char *const ARMTriples[] = {"arm-linux-gnueabi",
                                           "arm-linux-androideabi"};
  static const char *const ARMHFTriples[] = {"arm-linux-gnueabihf",
                                             "armv7hl-redhat-linux-gnueabi"};
  static const char *const ARMebLibDirs[] = {"/lib"};
  static const char *const ARMebTriples[] = {"armeb-linux-gnueabi",
                                             "armeb-linux-androideabi"};
  static const char *const ARMebHFTriples[] = {
      "armeb-linux-gnueabihf", "armebv7hl-redhat-linux-gnueabi"};

  static const char *const X86_64LibDirs[] = {"/lib64", "/lib"};
  static const char *const X86_64Triples[] = {
      "x86_64-linux-gnu",       "x86_64-unknown-linux-gnu",
      "x86_64-pc-linux-gnu",    "x86_64-redhat-linux6E",
      "x86_64-redhat-linux",    "x86_64-suse-linux",
      "x86_64-manbo-linux-gnu", "x86_64-linux-gnu",
      "x86_64-slackware-linux", "x86_64-linux-android",
      "x86_64-unknown-linux"};
  static const char *const X32LibDirs[] = {"/libx32"};
  static const char *const X86LibDirs[] = {"/lib32", "/lib"};
  static const char *const X86Triples[] = {
      "i686-linux-gnu",       "i686-pc-linux-gnu",     "i486-linux-gnu",
      "i386-linux-gnu",       "i386-redhat-linux6E",   "i686-redhat-linux",
      "i586-redhat-linux",    "i386-redhat-linux",     "i586-suse-linux",
      "i486-slackware-linux", "i686-montavista-linux", "i686-linux-android",
      "i586-linux-gnu"};

  static const char *const MIPSLibDirs[] = {"/lib"};
  static const char *const MIPSTriples[] = {"mips-linux-gnu", "mips-mti-linux",
                                            "mips-mti-linux-gnu",
                                            "mips-img-linux-gnu"};
  static const char *const MIPSELLibDirs[] = {"/lib"};
  static const char *const MIPSELTriples[] = {
      "mipsel-linux-gnu", "mipsel-linux-android", "mips-img-linux-gnu"};

  static const char *const MIPS64LibDirs[] = {"/lib64", "/lib"};
  static const char *const MIPS64Triples[] = {
      "mips64-linux-gnu", "mips-mti-linux-gnu", "mips-img-linux-gnu",
      "mips64-linux-gnuabi64"};
  static const char *const MIPS64ELLibDirs[] = {"/lib64", "/lib"};
  static const char *const MIPS64ELTriples[] = {
      "mips64el-linux-gnu", "mips-mti-linux-gnu", "mips-img-linux-gnu",
      "mips64el-linux-android", "mips64el-linux-gnuabi64"};

  static const char *const PPCLibDirs[] = {"/lib32", "/lib"};
  static const char *const PPCTriples[] = {
      "powerpc-linux-gnu", "powerpc-unknown-linux-gnu", "powerpc-linux-gnuspe",
      "powerpc-suse-linux", "powerpc-montavista-linuxspe"};
  static const char *const PPC64LibDirs[] = {"/lib64", "/lib"};
  static const char *const PPC64Triples[] = {
      "powerpc64-linux-gnu", "powerpc64-unknown-linux-gnu",
      "powerpc64-suse-linux", "ppc64-redhat-linux"};
  static const char *const PPC64LELibDirs[] = {"/lib64", "/lib"};
  static const char *const PPC64LETriples[] = {
      "powerpc64le-linux-gnu", "powerpc64le-unknown-linux-gnu",
      "powerpc64le-suse-linux", "ppc64le-redhat-linux"};

  static const char *const SPARCv8LibDirs[] = {"/lib32", "/lib"};
  static const char *const SPARCv8Triples[] = {"sparc-linux-gnu",
                                               "sparcv8-linux-gnu"};
  static const char *const SPARCv9LibDirs[] = {"/lib64", "/lib"};
  static const char *const SPARCv9Triples[] = {"sparc64-linux-gnu",
                                               "sparcv9-linux-gnu"};

  static const char *const SystemZLibDirs[] = {"/lib64", "/lib"};
  static const char *const SystemZTriples[] = {
      "s390x-linux-gnu", "s390x-unknown-linux-gnu", "s390x-ibm-linux-gnu",
      "s390x-suse-linux", "s390x-redhat-linux"};

  // Solaris.
  static const char *const SolarisSPARCLibDirs[] = {"/gcc"};
  static const char *const SolarisSPARCTriples[] = {"sparc-sun-solaris2.11",
                                                    "i386-pc-solaris2.11"};

  using std::begin;
  using std::end;

  if (TargetTriple.getOS() == llvm::Triple::Solaris) {
    LibDirs.append(begin(SolarisSPARCLibDirs), end(SolarisSPARCLibDirs));
    TripleAliases.append(begin(SolarisSPARCTriples), end(SolarisSPARCTriples));
    return;
  }

  switch (TargetTriple.getArch()) {
  case llvm::Triple::aarch64:
    LibDirs.append(begin(AArch64LibDirs), end(AArch64LibDirs));
    TripleAliases.append(begin(AArch64Triples), end(AArch64Triples));
    BiarchLibDirs.append(begin(AArch64LibDirs), end(AArch64LibDirs));
    BiarchTripleAliases.append(begin(AArch64Triples), end(AArch64Triples));
    break;
  case llvm::Triple::aarch64_be:
    LibDirs.append(begin(AArch64beLibDirs), end(AArch64beLibDirs));
    TripleAliases.append(begin(AArch64beTriples), end(AArch64beTriples));
    BiarchLibDirs.append(begin(AArch64beLibDirs), end(AArch64beLibDirs));
    BiarchTripleAliases.append(begin(AArch64beTriples), end(AArch64beTriples));
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    LibDirs.append(begin(ARMLibDirs), end(ARMLibDirs));
    if (TargetTriple.getEnvironment() == llvm::Triple::GNUEABIHF) {
      TripleAliases.append(begin(ARMHFTriples), end(ARMHFTriples));
    } else {
      TripleAliases.append(begin(ARMTriples), end(ARMTriples));
    }
    break;
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb:
    LibDirs.append(begin(ARMebLibDirs), end(ARMebLibDirs));
    if (TargetTriple.getEnvironment() == llvm::Triple::GNUEABIHF) {
      TripleAliases.append(begin(ARMebHFTriples), end(ARMebHFTriples));
    } else {
      TripleAliases.append(begin(ARMebTriples), end(ARMebTriples));
    }
    break;
  case llvm::Triple::x86_64:
    LibDirs.append(begin(X86_64LibDirs), end(X86_64LibDirs));
    TripleAliases.append(begin(X86_64Triples), end(X86_64Triples));
    // x32 is always available when x86_64 is available, so adding it as
    // secondary arch with x86_64 triples
    if (TargetTriple.getEnvironment() == llvm::Triple::GNUX32) {
      BiarchLibDirs.append(begin(X32LibDirs), end(X32LibDirs));
      BiarchTripleAliases.append(begin(X86_64Triples), end(X86_64Triples));
    } else {
      BiarchLibDirs.append(begin(X86LibDirs), end(X86LibDirs));
      BiarchTripleAliases.append(begin(X86Triples), end(X86Triples));
    }
    break;
  case llvm::Triple::x86:
    LibDirs.append(begin(X86LibDirs), end(X86LibDirs));
    TripleAliases.append(begin(X86Triples), end(X86Triples));
    BiarchLibDirs.append(begin(X86_64LibDirs), end(X86_64LibDirs));
    BiarchTripleAliases.append(begin(X86_64Triples), end(X86_64Triples));
    break;
  case llvm::Triple::mips:
    LibDirs.append(begin(MIPSLibDirs), end(MIPSLibDirs));
    TripleAliases.append(begin(MIPSTriples), end(MIPSTriples));
    BiarchLibDirs.append(begin(MIPS64LibDirs), end(MIPS64LibDirs));
    BiarchTripleAliases.append(begin(MIPS64Triples), end(MIPS64Triples));
    break;
  case llvm::Triple::mipsel:
    LibDirs.append(begin(MIPSELLibDirs), end(MIPSELLibDirs));
    TripleAliases.append(begin(MIPSELTriples), end(MIPSELTriples));
    TripleAliases.append(begin(MIPSTriples), end(MIPSTriples));
    BiarchLibDirs.append(begin(MIPS64ELLibDirs), end(MIPS64ELLibDirs));
    BiarchTripleAliases.append(begin(MIPS64ELTriples), end(MIPS64ELTriples));
    break;
  case llvm::Triple::mips64:
    LibDirs.append(begin(MIPS64LibDirs), end(MIPS64LibDirs));
    TripleAliases.append(begin(MIPS64Triples), end(MIPS64Triples));
    BiarchLibDirs.append(begin(MIPSLibDirs), end(MIPSLibDirs));
    BiarchTripleAliases.append(begin(MIPSTriples), end(MIPSTriples));
    break;
  case llvm::Triple::mips64el:
    LibDirs.append(begin(MIPS64ELLibDirs), end(MIPS64ELLibDirs));
    TripleAliases.append(begin(MIPS64ELTriples), end(MIPS64ELTriples));
    BiarchLibDirs.append(begin(MIPSELLibDirs), end(MIPSELLibDirs));
    BiarchTripleAliases.append(begin(MIPSELTriples), end(MIPSELTriples));
    BiarchTripleAliases.append(begin(MIPSTriples), end(MIPSTriples));
    break;
  case llvm::Triple::ppc:
    LibDirs.append(begin(PPCLibDirs), end(PPCLibDirs));
    TripleAliases.append(begin(PPCTriples), end(PPCTriples));
    BiarchLibDirs.append(begin(PPC64LibDirs), end(PPC64LibDirs));
    BiarchTripleAliases.append(begin(PPC64Triples), end(PPC64Triples));
    break;
  case llvm::Triple::ppc64:
    LibDirs.append(begin(PPC64LibDirs), end(PPC64LibDirs));
    TripleAliases.append(begin(PPC64Triples), end(PPC64Triples));
    BiarchLibDirs.append(begin(PPCLibDirs), end(PPCLibDirs));
    BiarchTripleAliases.append(begin(PPCTriples), end(PPCTriples));
    break;
  case llvm::Triple::ppc64le:
    LibDirs.append(begin(PPC64LELibDirs), end(PPC64LELibDirs));
    TripleAliases.append(begin(PPC64LETriples), end(PPC64LETriples));
    break;
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
    LibDirs.append(begin(SPARCv8LibDirs), end(SPARCv8LibDirs));
    TripleAliases.append(begin(SPARCv8Triples), end(SPARCv8Triples));
    BiarchLibDirs.append(begin(SPARCv9LibDirs), end(SPARCv9LibDirs));
    BiarchTripleAliases.append(begin(SPARCv9Triples), end(SPARCv9Triples));
    break;
  case llvm::Triple::sparcv9:
    LibDirs.append(begin(SPARCv9LibDirs), end(SPARCv9LibDirs));
    TripleAliases.append(begin(SPARCv9Triples), end(SPARCv9Triples));
    BiarchLibDirs.append(begin(SPARCv8LibDirs), end(SPARCv8LibDirs));
    BiarchTripleAliases.append(begin(SPARCv8Triples), end(SPARCv8Triples));
    break;
  case llvm::Triple::systemz:
    LibDirs.append(begin(SystemZLibDirs), end(SystemZLibDirs));
    TripleAliases.append(begin(SystemZTriples), end(SystemZTriples));
    break;
  default:
    // By default, just rely on the standard lib directories and the original
    // triple.
    break;
  }

  // Always append the drivers target triple to the end, in case it doesn't
  // match any of our aliases.
  TripleAliases.push_back(TargetTriple.str());

  // Also include the multiarch variant if it's different.
  if (TargetTriple.str() != BiarchTriple.str())
    BiarchTripleAliases.push_back(BiarchTriple.str());
}

// \brief -- try common CUDA installation paths looking for files we need for
// CUDA compilation.

void Generic_GCC::CudaInstallationDetector::init(
    const llvm::Triple &TargetTriple, const llvm::opt::ArgList &Args) {
  SmallVector<std::string, 4> CudaPathCandidates;

  if (Args.hasArg(options::OPT_cuda_path_EQ))
    CudaPathCandidates.push_back(
        Args.getLastArgValue(options::OPT_cuda_path_EQ));
  else {
    CudaPathCandidates.push_back(D.SysRoot + "/usr/local/cuda");
    CudaPathCandidates.push_back(D.SysRoot + "/usr/local/cuda-7.5");
    CudaPathCandidates.push_back(D.SysRoot + "/usr/local/cuda-7.0");
  }

  for (const auto &CudaPath : CudaPathCandidates) {
    if (CudaPath.empty() || !D.getVFS().exists(CudaPath))
      continue;

    CudaInstallPath = CudaPath;
    CudaIncludePath = CudaInstallPath + "/include";
    CudaLibDevicePath = CudaInstallPath + "/nvvm/libdevice";
    CudaLibPath =
        CudaInstallPath + (TargetTriple.isArch64Bit() ? "/lib64" : "/lib");

    if (!(D.getVFS().exists(CudaIncludePath) &&
          D.getVFS().exists(CudaLibPath) &&
          D.getVFS().exists(CudaLibDevicePath)))
      continue;

    std::error_code EC;
    for (llvm::sys::fs::directory_iterator LI(CudaLibDevicePath, EC), LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef FilePath = LI->path();
      StringRef FileName = llvm::sys::path::filename(FilePath);
      // Process all bitcode filenames that look like libdevice.compute_XX.YY.bc
      const StringRef LibDeviceName = "libdevice.";
      if (!(FileName.startswith(LibDeviceName) && FileName.endswith(".bc")))
        continue;
      StringRef GpuArch = FileName.slice(
          LibDeviceName.size(), FileName.find('.', LibDeviceName.size()));
      CudaLibDeviceMap[GpuArch] = FilePath.str();
      // Insert map entries for specifc devices with this compute capability.
      if (GpuArch == "compute_20") {
        CudaLibDeviceMap["sm_20"] = FilePath;
        CudaLibDeviceMap["sm_21"] = FilePath;
      } else if (GpuArch == "compute_30") {
        CudaLibDeviceMap["sm_30"] = FilePath;
        CudaLibDeviceMap["sm_32"] = FilePath;
      } else if (GpuArch == "compute_35") {
        CudaLibDeviceMap["sm_35"] = FilePath;
        CudaLibDeviceMap["sm_37"] = FilePath;
      }
    }

    IsValid = true;
    break;
  }
}

void Generic_GCC::CudaInstallationDetector::print(raw_ostream &OS) const {
  if (isValid())
    OS << "Found CUDA installation: " << CudaInstallPath << "\n";
}

namespace {
// Filter to remove Multilibs that don't exist as a suffix to Path
class FilterNonExistent {
  StringRef Base;
  vfs::FileSystem &VFS;

public:
  FilterNonExistent(StringRef Base, vfs::FileSystem &VFS)
      : Base(Base), VFS(VFS) {}
  bool operator()(const Multilib &M) {
    return !VFS.exists(Base + M.gccSuffix() + "/crtbegin.o");
  }
};
} // end anonymous namespace

static void addMultilibFlag(bool Enabled, const char *const Flag,
                            std::vector<std::string> &Flags) {
  if (Enabled)
    Flags.push_back(std::string("+") + Flag);
  else
    Flags.push_back(std::string("-") + Flag);
}

static bool isMipsArch(llvm::Triple::ArchType Arch) {
  return Arch == llvm::Triple::mips || Arch == llvm::Triple::mipsel ||
         Arch == llvm::Triple::mips64 || Arch == llvm::Triple::mips64el;
}

static bool isMips32(llvm::Triple::ArchType Arch) {
  return Arch == llvm::Triple::mips || Arch == llvm::Triple::mipsel;
}

static bool isMips64(llvm::Triple::ArchType Arch) {
  return Arch == llvm::Triple::mips64 || Arch == llvm::Triple::mips64el;
}

static bool isMipsEL(llvm::Triple::ArchType Arch) {
  return Arch == llvm::Triple::mipsel || Arch == llvm::Triple::mips64el;
}

static bool isMips16(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_mips16, options::OPT_mno_mips16);
  return A && A->getOption().matches(options::OPT_mips16);
}

static bool isMicroMips(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_mmicromips, options::OPT_mno_micromips);
  return A && A->getOption().matches(options::OPT_mmicromips);
}

namespace {
struct DetectedMultilibs {
  /// The set of multilibs that the detected installation supports.
  MultilibSet Multilibs;

  /// The primary multilib appropriate for the given flags.
  Multilib SelectedMultilib;

  /// On Biarch systems, this corresponds to the default multilib when
  /// targeting the non-default multilib. Otherwise, it is empty.
  llvm::Optional<Multilib> BiarchSibling;
};
} // end anonymous namespace

static Multilib makeMultilib(StringRef commonSuffix) {
  return Multilib(commonSuffix, commonSuffix, commonSuffix);
}

static bool findMIPSMultilibs(const Driver &D, const llvm::Triple &TargetTriple,
                              StringRef Path, const ArgList &Args,
                              DetectedMultilibs &Result) {
  // Some MIPS toolchains put libraries and object files compiled
  // using different options in to the sub-directoris which names
  // reflects the flags used for compilation. For example sysroot
  // directory might looks like the following examples:
  //
  // /usr
  //   /lib      <= crt*.o files compiled with '-mips32'
  // /mips16
  //   /usr
  //     /lib    <= crt*.o files compiled with '-mips16'
  //   /el
  //     /usr
  //       /lib  <= crt*.o files compiled with '-mips16 -EL'
  //
  // or
  //
  // /usr
  //   /lib      <= crt*.o files compiled with '-mips32r2'
  // /mips16
  //   /usr
  //     /lib    <= crt*.o files compiled with '-mips32r2 -mips16'
  // /mips32
  //     /usr
  //       /lib  <= crt*.o files compiled with '-mips32'

  FilterNonExistent NonExistent(Path, D.getVFS());

  // Check for FSF toolchain multilibs
  MultilibSet FSFMipsMultilibs;
  {
    auto MArchMips32 = makeMultilib("/mips32")
                           .flag("+m32")
                           .flag("-m64")
                           .flag("-mmicromips")
                           .flag("+march=mips32");

    auto MArchMicroMips = makeMultilib("/micromips")
                              .flag("+m32")
                              .flag("-m64")
                              .flag("+mmicromips");

    auto MArchMips64r2 = makeMultilib("/mips64r2")
                             .flag("-m32")
                             .flag("+m64")
                             .flag("+march=mips64r2");

    auto MArchMips64 = makeMultilib("/mips64").flag("-m32").flag("+m64").flag(
        "-march=mips64r2");

    auto MArchDefault = makeMultilib("")
                            .flag("+m32")
                            .flag("-m64")
                            .flag("-mmicromips")
                            .flag("+march=mips32r2");

    auto Mips16 = makeMultilib("/mips16").flag("+mips16");

    auto UCLibc = makeMultilib("/uclibc").flag("+muclibc");

    auto MAbi64 =
        makeMultilib("/64").flag("+mabi=n64").flag("-mabi=n32").flag("-m32");

    auto BigEndian = makeMultilib("").flag("+EB").flag("-EL");

    auto LittleEndian = makeMultilib("/el").flag("+EL").flag("-EB");

    auto SoftFloat = makeMultilib("/sof").flag("+msoft-float");

    auto Nan2008 = makeMultilib("/nan2008").flag("+mnan=2008");

    FSFMipsMultilibs =
        MultilibSet()
            .Either(MArchMips32, MArchMicroMips, MArchMips64r2, MArchMips64,
                    MArchDefault)
            .Maybe(UCLibc)
            .Maybe(Mips16)
            .FilterOut("/mips64/mips16")
            .FilterOut("/mips64r2/mips16")
            .FilterOut("/micromips/mips16")
            .Maybe(MAbi64)
            .FilterOut("/micromips/64")
            .FilterOut("/mips32/64")
            .FilterOut("^/64")
            .FilterOut("/mips16/64")
            .Either(BigEndian, LittleEndian)
            .Maybe(SoftFloat)
            .Maybe(Nan2008)
            .FilterOut(".*sof/nan2008")
            .FilterOut(NonExistent)
            .setIncludeDirsCallback([](StringRef InstallDir,
                                       StringRef TripleStr, const Multilib &M) {
              std::vector<std::string> Dirs;
              Dirs.push_back((InstallDir + "/include").str());
              std::string SysRootInc =
                  InstallDir.str() + "/../../../../sysroot";
              if (StringRef(M.includeSuffix()).startswith("/uclibc"))
                Dirs.push_back(SysRootInc + "/uclibc/usr/include");
              else
                Dirs.push_back(SysRootInc + "/usr/include");
              return Dirs;
            });
  }

  // Check for Musl toolchain multilibs
  MultilibSet MuslMipsMultilibs;
  {
    auto MArchMipsR2 = makeMultilib("")
                           .osSuffix("/mips-r2-hard-musl")
                           .flag("+EB")
                           .flag("-EL")
                           .flag("+march=mips32r2");

    auto MArchMipselR2 = makeMultilib("/mipsel-r2-hard-musl")
                             .flag("-EB")
                             .flag("+EL")
                             .flag("+march=mips32r2");

    MuslMipsMultilibs = MultilibSet().Either(MArchMipsR2, MArchMipselR2);

    // Specify the callback that computes the include directories.
    MuslMipsMultilibs.setIncludeDirsCallback([](
        StringRef InstallDir, StringRef TripleStr, const Multilib &M) {
      std::vector<std::string> Dirs;
      Dirs.push_back(
          (InstallDir + "/../sysroot" + M.osSuffix() + "/usr/include").str());
      return Dirs;
    });
  }

  // Check for Code Sourcery toolchain multilibs
  MultilibSet CSMipsMultilibs;
  {
    auto MArchMips16 = makeMultilib("/mips16").flag("+m32").flag("+mips16");

    auto MArchMicroMips =
        makeMultilib("/micromips").flag("+m32").flag("+mmicromips");

    auto MArchDefault = makeMultilib("").flag("-mips16").flag("-mmicromips");

    auto UCLibc = makeMultilib("/uclibc").flag("+muclibc");

    auto SoftFloat = makeMultilib("/soft-float").flag("+msoft-float");

    auto Nan2008 = makeMultilib("/nan2008").flag("+mnan=2008");

    auto DefaultFloat =
        makeMultilib("").flag("-msoft-float").flag("-mnan=2008");

    auto BigEndian = makeMultilib("").flag("+EB").flag("-EL");

    auto LittleEndian = makeMultilib("/el").flag("+EL").flag("-EB");

    // Note that this one's osSuffix is ""
    auto MAbi64 = makeMultilib("")
                      .gccSuffix("/64")
                      .includeSuffix("/64")
                      .flag("+mabi=n64")
                      .flag("-mabi=n32")
                      .flag("-m32");

    CSMipsMultilibs =
        MultilibSet()
            .Either(MArchMips16, MArchMicroMips, MArchDefault)
            .Maybe(UCLibc)
            .Either(SoftFloat, Nan2008, DefaultFloat)
            .FilterOut("/micromips/nan2008")
            .FilterOut("/mips16/nan2008")
            .Either(BigEndian, LittleEndian)
            .Maybe(MAbi64)
            .FilterOut("/mips16.*/64")
            .FilterOut("/micromips.*/64")
            .FilterOut(NonExistent)
            .setIncludeDirsCallback([](StringRef InstallDir,
                                       StringRef TripleStr, const Multilib &M) {
              std::vector<std::string> Dirs;
              Dirs.push_back((InstallDir + "/include").str());
              std::string SysRootInc =
                  InstallDir.str() + "/../../../../" + TripleStr.str();
              if (StringRef(M.includeSuffix()).startswith("/uclibc"))
                Dirs.push_back(SysRootInc + "/libc/uclibc/usr/include");
              else
                Dirs.push_back(SysRootInc + "/libc/usr/include");
              return Dirs;
            });
  }

  MultilibSet AndroidMipsMultilibs =
      MultilibSet()
          .Maybe(Multilib("/mips-r2").flag("+march=mips32r2"))
          .Maybe(Multilib("/mips-r6").flag("+march=mips32r6"))
          .FilterOut(NonExistent);

  MultilibSet DebianMipsMultilibs;
  {
    Multilib MAbiN32 =
        Multilib().gccSuffix("/n32").includeSuffix("/n32").flag("+mabi=n32");

    Multilib M64 = Multilib()
                       .gccSuffix("/64")
                       .includeSuffix("/64")
                       .flag("+m64")
                       .flag("-m32")
                       .flag("-mabi=n32");

    Multilib M32 = Multilib().flag("-m64").flag("+m32").flag("-mabi=n32");

    DebianMipsMultilibs =
        MultilibSet().Either(M32, M64, MAbiN32).FilterOut(NonExistent);
  }

  MultilibSet ImgMultilibs;
  {
    auto Mips64r6 = makeMultilib("/mips64r6").flag("+m64").flag("-m32");

    auto LittleEndian = makeMultilib("/el").flag("+EL").flag("-EB");

    auto MAbi64 =
        makeMultilib("/64").flag("+mabi=n64").flag("-mabi=n32").flag("-m32");

    ImgMultilibs =
        MultilibSet()
            .Maybe(Mips64r6)
            .Maybe(MAbi64)
            .Maybe(LittleEndian)
            .FilterOut(NonExistent)
            .setIncludeDirsCallback([](StringRef InstallDir,
                                       StringRef TripleStr, const Multilib &M) {
              std::vector<std::string> Dirs;
              Dirs.push_back((InstallDir + "/include").str());
              Dirs.push_back(
                  (InstallDir + "/../../../../sysroot/usr/include").str());
              return Dirs;
            });
  }

  StringRef CPUName;
  StringRef ABIName;
  tools::mips::getMipsCPUAndABI(Args, TargetTriple, CPUName, ABIName);

  llvm::Triple::ArchType TargetArch = TargetTriple.getArch();

  Multilib::flags_list Flags;
  addMultilibFlag(isMips32(TargetArch), "m32", Flags);
  addMultilibFlag(isMips64(TargetArch), "m64", Flags);
  addMultilibFlag(isMips16(Args), "mips16", Flags);
  addMultilibFlag(CPUName == "mips32", "march=mips32", Flags);
  addMultilibFlag(CPUName == "mips32r2" || CPUName == "mips32r3" ||
                      CPUName == "mips32r5" || CPUName == "p5600",
                  "march=mips32r2", Flags);
  addMultilibFlag(CPUName == "mips32r6", "march=mips32r6", Flags);
  addMultilibFlag(CPUName == "mips64", "march=mips64", Flags);
  addMultilibFlag(CPUName == "mips64r2" || CPUName == "mips64r3" ||
                      CPUName == "mips64r5" || CPUName == "octeon",
                  "march=mips64r2", Flags);
  addMultilibFlag(isMicroMips(Args), "mmicromips", Flags);
  addMultilibFlag(tools::mips::isUCLibc(Args), "muclibc", Flags);
  addMultilibFlag(tools::mips::isNaN2008(Args, TargetTriple), "mnan=2008",
                  Flags);
  addMultilibFlag(ABIName == "n32", "mabi=n32", Flags);
  addMultilibFlag(ABIName == "n64", "mabi=n64", Flags);
  addMultilibFlag(isSoftFloatABI(Args), "msoft-float", Flags);
  addMultilibFlag(!isSoftFloatABI(Args), "mhard-float", Flags);
  addMultilibFlag(isMipsEL(TargetArch), "EL", Flags);
  addMultilibFlag(!isMipsEL(TargetArch), "EB", Flags);

  if (TargetTriple.isAndroid()) {
    // Select Android toolchain. It's the only choice in that case.
    if (AndroidMipsMultilibs.select(Flags, Result.SelectedMultilib)) {
      Result.Multilibs = AndroidMipsMultilibs;
      return true;
    }
    return false;
  }

  if (TargetTriple.getVendor() == llvm::Triple::MipsTechnologies &&
      TargetTriple.getOS() == llvm::Triple::Linux &&
      TargetTriple.getEnvironment() == llvm::Triple::UnknownEnvironment) {
    if (MuslMipsMultilibs.select(Flags, Result.SelectedMultilib)) {
      Result.Multilibs = MuslMipsMultilibs;
      return true;
    }
    return false;
  }

  if (TargetTriple.getVendor() == llvm::Triple::ImaginationTechnologies &&
      TargetTriple.getOS() == llvm::Triple::Linux &&
      TargetTriple.getEnvironment() == llvm::Triple::GNU) {
    // Select mips-img-linux-gnu toolchain.
    if (ImgMultilibs.select(Flags, Result.SelectedMultilib)) {
      Result.Multilibs = ImgMultilibs;
      return true;
    }
    return false;
  }

  // Sort candidates. Toolchain that best meets the directories goes first.
  // Then select the first toolchains matches command line flags.
  MultilibSet *candidates[] = {&DebianMipsMultilibs, &FSFMipsMultilibs,
                               &CSMipsMultilibs};
  std::sort(
      std::begin(candidates), std::end(candidates),
      [](MultilibSet *a, MultilibSet *b) { return a->size() > b->size(); });
  for (const auto &candidate : candidates) {
    if (candidate->select(Flags, Result.SelectedMultilib)) {
      if (candidate == &DebianMipsMultilibs)
        Result.BiarchSibling = Multilib();
      Result.Multilibs = *candidate;
      return true;
    }
  }

  {
    // Fallback to the regular toolchain-tree structure.
    Multilib Default;
    Result.Multilibs.push_back(Default);
    Result.Multilibs.FilterOut(NonExistent);

    if (Result.Multilibs.select(Flags, Result.SelectedMultilib)) {
      Result.BiarchSibling = Multilib();
      return true;
    }
  }

  return false;
}

static bool findBiarchMultilibs(const Driver &D,
                                const llvm::Triple &TargetTriple,
                                StringRef Path, const ArgList &Args,
                                bool NeedsBiarchSuffix,
                                DetectedMultilibs &Result) {
  // Some versions of SUSE and Fedora on ppc64 put 32-bit libs
  // in what would normally be GCCInstallPath and put the 64-bit
  // libs in a subdirectory named 64. The simple logic we follow is that
  // *if* there is a subdirectory of the right name with crtbegin.o in it,
  // we use that. If not, and if not a biarch triple alias, we look for
  // crtbegin.o without the subdirectory.

  Multilib Default;
  Multilib Alt64 = Multilib()
                       .gccSuffix("/64")
                       .includeSuffix("/64")
                       .flag("-m32")
                       .flag("+m64")
                       .flag("-mx32");
  Multilib Alt32 = Multilib()
                       .gccSuffix("/32")
                       .includeSuffix("/32")
                       .flag("+m32")
                       .flag("-m64")
                       .flag("-mx32");
  Multilib Altx32 = Multilib()
                        .gccSuffix("/x32")
                        .includeSuffix("/x32")
                        .flag("-m32")
                        .flag("-m64")
                        .flag("+mx32");

  FilterNonExistent NonExistent(Path, D.getVFS());

  // Determine default multilib from: 32, 64, x32
  // Also handle cases such as 64 on 32, 32 on 64, etc.
  enum { UNKNOWN, WANT32, WANT64, WANTX32 } Want = UNKNOWN;
  const bool IsX32 = TargetTriple.getEnvironment() == llvm::Triple::GNUX32;
  if (TargetTriple.isArch32Bit() && !NonExistent(Alt32))
    Want = WANT64;
  else if (TargetTriple.isArch64Bit() && IsX32 && !NonExistent(Altx32))
    Want = WANT64;
  else if (TargetTriple.isArch64Bit() && !IsX32 && !NonExistent(Alt64))
    Want = WANT32;
  else {
    if (TargetTriple.isArch32Bit())
      Want = NeedsBiarchSuffix ? WANT64 : WANT32;
    else if (IsX32)
      Want = NeedsBiarchSuffix ? WANT64 : WANTX32;
    else
      Want = NeedsBiarchSuffix ? WANT32 : WANT64;
  }

  if (Want == WANT32)
    Default.flag("+m32").flag("-m64").flag("-mx32");
  else if (Want == WANT64)
    Default.flag("-m32").flag("+m64").flag("-mx32");
  else if (Want == WANTX32)
    Default.flag("-m32").flag("-m64").flag("+mx32");
  else
    return false;

  Result.Multilibs.push_back(Default);
  Result.Multilibs.push_back(Alt64);
  Result.Multilibs.push_back(Alt32);
  Result.Multilibs.push_back(Altx32);

  Result.Multilibs.FilterOut(NonExistent);

  Multilib::flags_list Flags;
  addMultilibFlag(TargetTriple.isArch64Bit() && !IsX32, "m64", Flags);
  addMultilibFlag(TargetTriple.isArch32Bit(), "m32", Flags);
  addMultilibFlag(TargetTriple.isArch64Bit() && IsX32, "mx32", Flags);

  if (!Result.Multilibs.select(Flags, Result.SelectedMultilib))
    return false;

  if (Result.SelectedMultilib == Alt64 || Result.SelectedMultilib == Alt32 ||
      Result.SelectedMultilib == Altx32)
    Result.BiarchSibling = Default;

  return true;
}

void Generic_GCC::GCCInstallationDetector::scanLibDirForGCCTripleSolaris(
    const llvm::Triple &TargetArch, const llvm::opt::ArgList &Args,
    const std::string &LibDir, StringRef CandidateTriple,
    bool NeedsBiarchSuffix) {
  // Solaris is a special case. The GCC installation is under
  // /usr/gcc/<major>.<minor>/lib/gcc/<triple>/<major>.<minor>.<patch>/, so we
  // need to iterate twice.
  std::error_code EC;
  for (vfs::directory_iterator LI = D.getVFS().dir_begin(LibDir, EC), LE;
       !EC && LI != LE; LI = LI.increment(EC)) {
    StringRef VersionText = llvm::sys::path::filename(LI->getName());
    GCCVersion CandidateVersion = GCCVersion::Parse(VersionText);

    if (CandidateVersion.Major != -1) // Filter obviously bad entries.
      if (!CandidateGCCInstallPaths.insert(LI->getName()).second)
        continue; // Saw this path before; no need to look at it again.
    if (CandidateVersion.isOlderThan(4, 1, 1))
      continue;
    if (CandidateVersion <= Version)
      continue;

    GCCInstallPath =
        LibDir + "/" + VersionText.str() + "/lib/gcc/" + CandidateTriple.str();
    if (!D.getVFS().exists(GCCInstallPath))
      continue;

    // If we make it here there has to be at least one GCC version, let's just
    // use the latest one.
    std::error_code EEC;
    for (vfs::directory_iterator
             LLI = D.getVFS().dir_begin(GCCInstallPath, EEC),
             LLE;
         !EEC && LLI != LLE; LLI = LLI.increment(EEC)) {

      StringRef SubVersionText = llvm::sys::path::filename(LLI->getName());
      GCCVersion CandidateSubVersion = GCCVersion::Parse(SubVersionText);

      if (CandidateSubVersion > Version)
        Version = CandidateSubVersion;
    }

    GCCTriple.setTriple(CandidateTriple);

    GCCInstallPath += "/" + Version.Text;
    GCCParentLibPath = GCCInstallPath + "/../../../../";

    IsValid = true;
  }
}

void Generic_GCC::GCCInstallationDetector::ScanLibDirForGCCTriple(
    const llvm::Triple &TargetTriple, const ArgList &Args,
    const std::string &LibDir, StringRef CandidateTriple,
    bool NeedsBiarchSuffix) {
  llvm::Triple::ArchType TargetArch = TargetTriple.getArch();
  // There are various different suffixes involving the triple we
  // check for. We also record what is necessary to walk from each back
  // up to the lib directory. Specifically, the number of "up" steps
  // in the second half of each row is 1 + the number of path separators
  // in the first half.
  const std::string LibAndInstallSuffixes[][2] = {
      {"/gcc/" + CandidateTriple.str(), "/../../.."},

      // Debian puts cross-compilers in gcc-cross
      {"/gcc-cross/" + CandidateTriple.str(), "/../../.."},

      {"/" + CandidateTriple.str() + "/gcc/" + CandidateTriple.str(),
       "/../../../.."},

      // The Freescale PPC SDK has the gcc libraries in
      // <sysroot>/usr/lib/<triple>/x.y.z so have a look there as well.
      {"/" + CandidateTriple.str(), "/../.."},

      // Ubuntu has a strange mis-matched pair of triples that this happens to
      // match.
      // FIXME: It may be worthwhile to generalize this and look for a second
      // triple.
      {"/i386-linux-gnu/gcc/" + CandidateTriple.str(), "/../../../.."}};

  if (TargetTriple.getOS() == llvm::Triple::Solaris) {
    scanLibDirForGCCTripleSolaris(TargetTriple, Args, LibDir, CandidateTriple,
                                  NeedsBiarchSuffix);
    return;
  }

  // Only look at the final, weird Ubuntu suffix for i386-linux-gnu.
  const unsigned NumLibSuffixes = (llvm::array_lengthof(LibAndInstallSuffixes) -
                                   (TargetArch != llvm::Triple::x86));
  for (unsigned i = 0; i < NumLibSuffixes; ++i) {
    StringRef LibSuffix = LibAndInstallSuffixes[i][0];
    std::error_code EC;
    for (vfs::directory_iterator
             LI = D.getVFS().dir_begin(LibDir + LibSuffix, EC),
             LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef VersionText = llvm::sys::path::filename(LI->getName());
      GCCVersion CandidateVersion = GCCVersion::Parse(VersionText);
      if (CandidateVersion.Major != -1) // Filter obviously bad entries.
        if (!CandidateGCCInstallPaths.insert(LI->getName()).second)
          continue; // Saw this path before; no need to look at it again.
      if (CandidateVersion.isOlderThan(4, 1, 1))
        continue;
      if (CandidateVersion <= Version)
        continue;

      DetectedMultilibs Detected;

      // Debian mips multilibs behave more like the rest of the biarch ones,
      // so handle them there
      if (isMipsArch(TargetArch)) {
        if (!findMIPSMultilibs(D, TargetTriple, LI->getName(), Args, Detected))
          continue;
      } else if (!findBiarchMultilibs(D, TargetTriple, LI->getName(), Args,
                                      NeedsBiarchSuffix, Detected)) {
        continue;
      }

      Multilibs = Detected.Multilibs;
      SelectedMultilib = Detected.SelectedMultilib;
      BiarchSibling = Detected.BiarchSibling;
      Version = CandidateVersion;
      GCCTriple.setTriple(CandidateTriple);
      // FIXME: We hack together the directory name here instead of
      // using LI to ensure stable path separators across Windows and
      // Linux.
      GCCInstallPath =
          LibDir + LibAndInstallSuffixes[i][0] + "/" + VersionText.str();
      GCCParentLibPath = GCCInstallPath + LibAndInstallSuffixes[i][1];
      IsValid = true;
    }
  }
}

Generic_GCC::Generic_GCC(const Driver &D, const llvm::Triple &Triple,
                         const ArgList &Args)
    : ToolChain(D, Triple, Args), GCCInstallation(D), CudaInstallation(D) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);
}

Generic_GCC::~Generic_GCC() {}

Tool *Generic_GCC::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::PreprocessJobClass:
    if (!Preprocess)
      Preprocess.reset(new tools::gcc::Preprocessor(*this));
    return Preprocess.get();
  case Action::CompileJobClass:
    if (!Compile)
      Compile.reset(new tools::gcc::Compiler(*this));
    return Compile.get();
  default:
    return ToolChain::getTool(AC);
  }
}

Tool *Generic_GCC::buildAssembler() const {
  return new tools::gnutools::Assembler(*this);
}

Tool *Generic_GCC::buildLinker() const { return new tools::gcc::Linker(*this); }

void Generic_GCC::printVerboseInfo(raw_ostream &OS) const {
  // Print the information about how we detected the GCC installation.
  GCCInstallation.print(OS);
  CudaInstallation.print(OS);
}

bool Generic_GCC::IsUnwindTablesDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool Generic_GCC::isPICDefault() const {
  return getArch() == llvm::Triple::x86_64 && getTriple().isOSWindows();
}

bool Generic_GCC::isPIEDefault() const { return false; }

bool Generic_GCC::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64 && getTriple().isOSWindows();
}

bool Generic_GCC::IsIntegratedAssemblerDefault() const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::bpfel:
  case llvm::Triple::bpfeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::systemz:
    return true;
  default:
    return false;
  }
}

/// \brief Helper to add the variant paths of a libstdc++ installation.
bool Generic_GCC::addLibStdCXXIncludePaths(
    Twine Base, Twine Suffix, StringRef GCCTriple, StringRef GCCMultiarchTriple,
    StringRef TargetMultiarchTriple, Twine IncludeSuffix,
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (!getVFS().exists(Base + Suffix))
    return false;

  addSystemInclude(DriverArgs, CC1Args, Base + Suffix);

  // The vanilla GCC layout of libstdc++ headers uses a triple subdirectory. If
  // that path exists or we have neither a GCC nor target multiarch triple, use
  // this vanilla search path.
  if ((GCCMultiarchTriple.empty() && TargetMultiarchTriple.empty()) ||
      getVFS().exists(Base + Suffix + "/" + GCCTriple + IncludeSuffix)) {
    addSystemInclude(DriverArgs, CC1Args,
                     Base + Suffix + "/" + GCCTriple + IncludeSuffix);
  } else {
    // Otherwise try to use multiarch naming schemes which have normalized the
    // triples and put the triple before the suffix.
    //
    // GCC surprisingly uses *both* the GCC triple with a multilib suffix and
    // the target triple, so we support that here.
    addSystemInclude(DriverArgs, CC1Args,
                     Base + "/" + GCCMultiarchTriple + Suffix + IncludeSuffix);
    addSystemInclude(DriverArgs, CC1Args,
                     Base + "/" + TargetMultiarchTriple + Suffix);
  }

  addSystemInclude(DriverArgs, CC1Args, Base + Suffix + "/backward");
  return true;
}


void Generic_ELF::addClangTargetOptions(const ArgList &DriverArgs,
                                        ArgStringList &CC1Args) const {
  const Generic_GCC::GCCVersion &V = GCCInstallation.getVersion();
  bool UseInitArrayDefault =
      getTriple().getArch() == llvm::Triple::aarch64 ||
      getTriple().getArch() == llvm::Triple::aarch64_be ||
      (getTriple().getOS() == llvm::Triple::Linux &&
       (!V.isOlderThan(4, 7, 0) || getTriple().isAndroid())) ||
      getTriple().getOS() == llvm::Triple::NaCl ||
      (getTriple().getVendor() == llvm::Triple::MipsTechnologies &&
       !getTriple().hasEnvironment());

  if (DriverArgs.hasFlag(options::OPT_fuse_init_array,
                         options::OPT_fno_use_init_array, UseInitArrayDefault))
    CC1Args.push_back("-fuse-init-array");
}

/// Mips Toolchain
MipsLLVMToolChain::MipsLLVMToolChain(const Driver &D,
                                     const llvm::Triple &Triple,
                                     const ArgList &Args)
    : Linux(D, Triple, Args) {
  // Select the correct multilib according to the given arguments.
  DetectedMultilibs Result;
  findMIPSMultilibs(D, Triple, "", Args, Result);
  Multilibs = Result.Multilibs;
  SelectedMultilib = Result.SelectedMultilib;

  // Find out the library suffix based on the ABI.
  LibSuffix = tools::mips::getMipsABILibSuffix(Args, Triple);
  getFilePaths().clear();
  getFilePaths().push_back(computeSysRoot() + "/usr/lib" + LibSuffix);

  // Use LLD by default.
  DefaultLinker = "lld";
}

void MipsLLVMToolChain::AddClangSystemIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  const Driver &D = getDriver();

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  const auto &Callback = Multilibs.includeDirsCallback();
  if (Callback) {
    const auto IncludePaths =
        Callback(D.getInstalledDir(), getTripleString(), SelectedMultilib);
    for (const auto &Path : IncludePaths)
      addExternCSystemIncludeIfExists(DriverArgs, CC1Args, Path);
  }
}

Tool *MipsLLVMToolChain::buildLinker() const {
  return new tools::gnutools::Linker(*this);
}

std::string MipsLLVMToolChain::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot + SelectedMultilib.osSuffix();

  const std::string InstalledDir(getDriver().getInstalledDir());
  std::string SysRootPath =
      InstalledDir + "/../sysroot" + SelectedMultilib.osSuffix();
  if (llvm::sys::fs::exists(SysRootPath))
    return SysRootPath;

  return std::string();
}

ToolChain::CXXStdlibType
MipsLLVMToolChain::GetCXXStdlibType(const ArgList &Args) const {
  Arg *A = Args.getLastArg(options::OPT_stdlib_EQ);
  if (A) {
    StringRef Value = A->getValue();
    if (Value != "libc++")
      getDriver().Diag(diag::err_drv_invalid_stdlib_name)
          << A->getAsString(Args);
  }

  return ToolChain::CST_Libcxx;
}

void MipsLLVMToolChain::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  assert((GetCXXStdlibType(DriverArgs) == ToolChain::CST_Libcxx) &&
         "Only -lc++ (aka libcxx) is suported in this toolchain.");

  const auto &Callback = Multilibs.includeDirsCallback();
  if (Callback) {
    const auto IncludePaths = Callback(getDriver().getInstalledDir(),
                                       getTripleString(), SelectedMultilib);
    for (const auto &Path : IncludePaths) {
      if (llvm::sys::fs::exists(Path + "/c++/v1")) {
        addSystemInclude(DriverArgs, CC1Args, Path + "/c++/v1");
        break;
      }
    }
  }
}

void MipsLLVMToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                            ArgStringList &CmdArgs) const {
  assert((GetCXXStdlibType(Args) == ToolChain::CST_Libcxx) &&
         "Only -lc++ (aka libxx) is suported in this toolchain.");

  CmdArgs.push_back("-lc++");
  CmdArgs.push_back("-lc++abi");
  CmdArgs.push_back("-lunwind");
}

std::string MipsLLVMToolChain::getCompilerRT(const ArgList &Args,
                                             StringRef Component,
                                             bool Shared) const {
  SmallString<128> Path(getDriver().ResourceDir);
  llvm::sys::path::append(Path, SelectedMultilib.osSuffix(), "lib" + LibSuffix,
                          getOS());
  llvm::sys::path::append(Path, Twine("libclang_rt." + Component + "-" +
                                      "mips" + (Shared ? ".so" : ".a")));
  return Path.str();
}

/// Hexagon Toolchain

std::string HexagonToolChain::GetGnuDir(const std::string &InstalledDir,
                                        const ArgList &Args) const {
  // Locate the rest of the toolchain ...
  std::string GccToolchain = getGCCToolchainDir(Args);

  if (!GccToolchain.empty())
    return GccToolchain;

  std::string InstallRelDir = InstalledDir + "/../../gnu";
  if (getVFS().exists(InstallRelDir))
    return InstallRelDir;

  std::string PrefixRelDir = std::string(LLVM_PREFIX) + "/../gnu";
  if (getVFS().exists(PrefixRelDir))
    return PrefixRelDir;

  return InstallRelDir;
}

const char *HexagonToolChain::GetSmallDataThreshold(const ArgList &Args) {
  Arg *A;

  A = Args.getLastArg(options::OPT_G, options::OPT_G_EQ,
                      options::OPT_msmall_data_threshold_EQ);
  if (A)
    return A->getValue();

  A = Args.getLastArg(options::OPT_shared, options::OPT_fpic,
                      options::OPT_fPIC);
  if (A)
    return "0";

  return nullptr;
}

bool HexagonToolChain::UsesG0(const char *smallDataThreshold) {
  return smallDataThreshold && smallDataThreshold[0] == '0';
}

static void GetHexagonLibraryPaths(const HexagonToolChain &TC,
                                   const ArgList &Args, const std::string &Ver,
                                   const std::string &MarchString,
                                   const std::string &InstalledDir,
                                   ToolChain::path_list *LibPaths) {
  bool buildingLib = Args.hasArg(options::OPT_shared);

  //----------------------------------------------------------------------------
  // -L Args
  //----------------------------------------------------------------------------
  for (Arg *A : Args.filtered(options::OPT_L))
    for (const char *Value : A->getValues())
      LibPaths->push_back(Value);

  //----------------------------------------------------------------------------
  // Other standard paths
  //----------------------------------------------------------------------------
  const std::string MarchSuffix = "/" + MarchString;
  const std::string G0Suffix = "/G0";
  const std::string MarchG0Suffix = MarchSuffix + G0Suffix;
  const std::string RootDir = TC.GetGnuDir(InstalledDir, Args) + "/";

  // lib/gcc/hexagon/...
  std::string LibGCCHexagonDir = RootDir + "lib/gcc/hexagon/";
  if (buildingLib) {
    LibPaths->push_back(LibGCCHexagonDir + Ver + MarchG0Suffix);
    LibPaths->push_back(LibGCCHexagonDir + Ver + G0Suffix);
  }
  LibPaths->push_back(LibGCCHexagonDir + Ver + MarchSuffix);
  LibPaths->push_back(LibGCCHexagonDir + Ver);

  // lib/gcc/...
  LibPaths->push_back(RootDir + "lib/gcc");

  // hexagon/lib/...
  std::string HexagonLibDir = RootDir + "hexagon/lib";
  if (buildingLib) {
    LibPaths->push_back(HexagonLibDir + MarchG0Suffix);
    LibPaths->push_back(HexagonLibDir + G0Suffix);
  }
  LibPaths->push_back(HexagonLibDir + MarchSuffix);
  LibPaths->push_back(HexagonLibDir);
}

HexagonToolChain::HexagonToolChain(const Driver &D, const llvm::Triple &Triple,
                                   const ArgList &Args)
    : Linux(D, Triple, Args) {
  const std::string InstalledDir(getDriver().getInstalledDir());
  const std::string GnuDir = GetGnuDir(InstalledDir, Args);

  // Note: Generic_GCC::Generic_GCC adds InstalledDir and getDriver().Dir to
  // program paths
  const std::string BinDir(GnuDir + "/bin");
  if (D.getVFS().exists(BinDir))
    getProgramPaths().push_back(BinDir);

  // Determine version of GCC libraries and headers to use.
  const std::string HexagonDir(GnuDir + "/lib/gcc/hexagon");
  std::error_code ec;
  GCCVersion MaxVersion = GCCVersion::Parse("0.0.0");
  for (vfs::directory_iterator di = D.getVFS().dir_begin(HexagonDir, ec), de;
       !ec && di != de; di = di.increment(ec)) {
    GCCVersion cv = GCCVersion::Parse(llvm::sys::path::filename(di->getName()));
    if (MaxVersion < cv)
      MaxVersion = cv;
  }
  GCCLibAndIncVersion = MaxVersion;

  ToolChain::path_list *LibPaths = &getFilePaths();

  // Remove paths added by Linux toolchain. Currently Hexagon_TC really targets
  // 'elf' OS type, so the Linux paths are not appropriate. When we actually
  // support 'linux' we'll need to fix this up
  LibPaths->clear();

  GetHexagonLibraryPaths(*this, Args, GetGCCLibAndIncVersion(),
                         GetTargetCPU(Args), InstalledDir, LibPaths);
}

HexagonToolChain::~HexagonToolChain() {}

Tool *HexagonToolChain::buildAssembler() const {
  return new tools::hexagon::Assembler(*this);
}

Tool *HexagonToolChain::buildLinker() const {
  return new tools::hexagon::Linker(*this);
}

void HexagonToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                 ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  std::string Ver(GetGCCLibAndIncVersion());
  std::string GnuDir = GetGnuDir(D.InstalledDir, DriverArgs);
  std::string HexagonDir(GnuDir + "/lib/gcc/hexagon/" + Ver);
  addExternCSystemInclude(DriverArgs, CC1Args, HexagonDir + "/include");
  addExternCSystemInclude(DriverArgs, CC1Args, HexagonDir + "/include-fixed");
  addExternCSystemInclude(DriverArgs, CC1Args, GnuDir + "/hexagon/include");
}

void HexagonToolChain::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  const Driver &D = getDriver();
  std::string Ver(GetGCCLibAndIncVersion());
  SmallString<128> IncludeDir(GetGnuDir(D.InstalledDir, DriverArgs));

  llvm::sys::path::append(IncludeDir, "hexagon/include/c++/");
  llvm::sys::path::append(IncludeDir, Ver);
  addSystemInclude(DriverArgs, CC1Args, IncludeDir);
}

ToolChain::CXXStdlibType
HexagonToolChain::GetCXXStdlibType(const ArgList &Args) const {
  Arg *A = Args.getLastArg(options::OPT_stdlib_EQ);
  if (!A)
    return ToolChain::CST_Libstdcxx;

  StringRef Value = A->getValue();
  if (Value != "libstdc++") {
    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);
  }

  return ToolChain::CST_Libstdcxx;
}

static int getHexagonVersion(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_march_EQ, options::OPT_mcpu_EQ);
  // Select the default CPU (v4) if none was given.
  if (!A)
    return 4;

  // FIXME: produce errors if we cannot parse the version.
  StringRef WhichHexagon = A->getValue();
  if (WhichHexagon.startswith("hexagonv")) {
    int Val;
    if (!WhichHexagon.substr(sizeof("hexagonv") - 1).getAsInteger(10, Val))
      return Val;
  }
  if (WhichHexagon.startswith("v")) {
    int Val;
    if (!WhichHexagon.substr(1).getAsInteger(10, Val))
      return Val;
  }

  // FIXME: should probably be an error.
  return 4;
}

StringRef HexagonToolChain::GetTargetCPU(const ArgList &Args) {
  int V = getHexagonVersion(Args);
  // FIXME: We don't support versions < 4. We should error on them.
  switch (V) {
  default:
    llvm_unreachable("Unexpected version");
  case 5:
    return "v5";
  case 4:
    return "v4";
  case 3:
    return "v3";
  case 2:
    return "v2";
  case 1:
    return "v1";
  }
}
// End Hexagon

/// AMDGPU Toolchain
AMDGPUToolChain::AMDGPUToolChain(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args)
  : Generic_ELF(D, Triple, Args) { }

Tool *AMDGPUToolChain::buildLinker() const {
  return new tools::amdgpu::Linker(*this);
}
// End AMDGPU

/// NaCl Toolchain
NaClToolChain::NaClToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {

  // Remove paths added by Generic_GCC. NaCl Toolchain cannot use the
  // default paths, and must instead only use the paths provided
  // with this toolchain based on architecture.
  path_list &file_paths = getFilePaths();
  path_list &prog_paths = getProgramPaths();

  file_paths.clear();
  prog_paths.clear();

  // Path for library files (libc.a, ...)
  std::string FilePath(getDriver().Dir + "/../");

  // Path for tools (clang, ld, etc..)
  std::string ProgPath(getDriver().Dir + "/../");

  // Path for toolchain libraries (libgcc.a, ...)
  std::string ToolPath(getDriver().ResourceDir + "/lib/");

  switch (Triple.getArch()) {
  case llvm::Triple::x86:
    file_paths.push_back(FilePath + "x86_64-nacl/lib32");
    file_paths.push_back(FilePath + "i686-nacl/usr/lib");
    prog_paths.push_back(ProgPath + "x86_64-nacl/bin");
    file_paths.push_back(ToolPath + "i686-nacl");
    break;
  case llvm::Triple::x86_64:
    file_paths.push_back(FilePath + "x86_64-nacl/lib");
    file_paths.push_back(FilePath + "x86_64-nacl/usr/lib");
    prog_paths.push_back(ProgPath + "x86_64-nacl/bin");
    file_paths.push_back(ToolPath + "x86_64-nacl");
    break;
  case llvm::Triple::arm:
    file_paths.push_back(FilePath + "arm-nacl/lib");
    file_paths.push_back(FilePath + "arm-nacl/usr/lib");
    prog_paths.push_back(ProgPath + "arm-nacl/bin");
    file_paths.push_back(ToolPath + "arm-nacl");
    break;
  case llvm::Triple::mipsel:
    file_paths.push_back(FilePath + "mipsel-nacl/lib");
    file_paths.push_back(FilePath + "mipsel-nacl/usr/lib");
    prog_paths.push_back(ProgPath + "bin");
    file_paths.push_back(ToolPath + "mipsel-nacl");
    break;
  default:
    break;
  }

  NaClArmMacrosPath = GetFilePath("nacl-arm-macros.s");
}

void NaClToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  SmallString<128> P(D.Dir + "/../");
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
    // x86 is special because multilib style uses x86_64-nacl/include for libc
    // headers but the SDK wants i686-nacl/usr/include. The other architectures
    // have the same substring.
    llvm::sys::path::append(P, "i686-nacl/usr/include");
    addSystemInclude(DriverArgs, CC1Args, P.str());
    llvm::sys::path::remove_filename(P);
    llvm::sys::path::remove_filename(P);
    llvm::sys::path::remove_filename(P);
    llvm::sys::path::append(P, "x86_64-nacl/include");
    addSystemInclude(DriverArgs, CC1Args, P.str());
    return;
  case llvm::Triple::arm:
    llvm::sys::path::append(P, "arm-nacl/usr/include");
    break;
  case llvm::Triple::x86_64:
    llvm::sys::path::append(P, "x86_64-nacl/usr/include");
    break;
  case llvm::Triple::mipsel:
    llvm::sys::path::append(P, "mipsel-nacl/usr/include");
    break;
  default:
    return;
  }

  addSystemInclude(DriverArgs, CC1Args, P.str());
  llvm::sys::path::remove_filename(P);
  llvm::sys::path::remove_filename(P);
  llvm::sys::path::append(P, "include");
  addSystemInclude(DriverArgs, CC1Args, P.str());
}

void NaClToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                        ArgStringList &CmdArgs) const {
  // Check for -stdlib= flags. We only support libc++ but this consumes the arg
  // if the value is libc++, and emits an error for other values.
  GetCXXStdlibType(Args);
  CmdArgs.push_back("-lc++");
}

void NaClToolChain::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                                 ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  // Check for -stdlib= flags. We only support libc++ but this consumes the arg
  // if the value is libc++, and emits an error for other values.
  GetCXXStdlibType(DriverArgs);

  SmallString<128> P(D.Dir + "/../");
  switch (getTriple().getArch()) {
  case llvm::Triple::arm:
    llvm::sys::path::append(P, "arm-nacl/include/c++/v1");
    addSystemInclude(DriverArgs, CC1Args, P.str());
    break;
  case llvm::Triple::x86:
    llvm::sys::path::append(P, "x86_64-nacl/include/c++/v1");
    addSystemInclude(DriverArgs, CC1Args, P.str());
    break;
  case llvm::Triple::x86_64:
    llvm::sys::path::append(P, "x86_64-nacl/include/c++/v1");
    addSystemInclude(DriverArgs, CC1Args, P.str());
    break;
  case llvm::Triple::mipsel:
    llvm::sys::path::append(P, "mipsel-nacl/include/c++/v1");
    addSystemInclude(DriverArgs, CC1Args, P.str());
    break;
  default:
    break;
  }
}

ToolChain::CXXStdlibType
NaClToolChain::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;
    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);
  }

  return ToolChain::CST_Libcxx;
}

std::string
NaClToolChain::ComputeEffectiveClangTriple(const ArgList &Args,
                                           types::ID InputType) const {
  llvm::Triple TheTriple(ComputeLLVMTriple(Args, InputType));
  if (TheTriple.getArch() == llvm::Triple::arm &&
      TheTriple.getEnvironment() == llvm::Triple::UnknownEnvironment)
    TheTriple.setEnvironment(llvm::Triple::GNUEABIHF);
  return TheTriple.getTriple();
}

Tool *NaClToolChain::buildLinker() const {
  return new tools::nacltools::Linker(*this);
}

Tool *NaClToolChain::buildAssembler() const {
  if (getTriple().getArch() == llvm::Triple::arm)
    return new tools::nacltools::AssemblerARM(*this);
  return new tools::gnutools::Assembler(*this);
}
// End NaCl

/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
/// Currently does not support anything else but compilation.

TCEToolChain::TCEToolChain(const Driver &D, const llvm::Triple &Triple,
                           const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  // Path mangling to find libexec
  std::string Path(getDriver().Dir);

  Path += "/../libexec";
  getProgramPaths().push_back(Path);
}

TCEToolChain::~TCEToolChain() {}

bool TCEToolChain::IsMathErrnoDefault() const { return true; }

bool TCEToolChain::isPICDefault() const { return false; }

bool TCEToolChain::isPIEDefault() const { return false; }

bool TCEToolChain::isPICDefaultForced() const { return false; }

// CloudABI - CloudABI tool chain which can call ld(1) directly.

CloudABI::CloudABI(const Driver &D, const llvm::Triple &Triple,
                   const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  SmallString<128> P(getDriver().Dir);
  llvm::sys::path::append(P, "..", getTriple().str(), "lib");
  getFilePaths().push_back(P.str());
}

void CloudABI::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) &&
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  SmallString<128> P(getDriver().Dir);
  llvm::sys::path::append(P, "..", getTriple().str(), "include/c++/v1");
  addSystemInclude(DriverArgs, CC1Args, P.str());
}

void CloudABI::AddCXXStdlibLibArgs(const ArgList &Args,
                                   ArgStringList &CmdArgs) const {
  CmdArgs.push_back("-lc++");
  CmdArgs.push_back("-lc++abi");
  CmdArgs.push_back("-lunwind");
}

Tool *CloudABI::buildLinker() const {
  return new tools::cloudabi::Linker(*this);
}

/// OpenBSD - OpenBSD tool chain which can call as(1) and ld(1) directly.

OpenBSD::OpenBSD(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *OpenBSD::buildAssembler() const {
  return new tools::openbsd::Assembler(*this);
}

Tool *OpenBSD::buildLinker() const { return new tools::openbsd::Linker(*this); }

/// Bitrig - Bitrig tool chain which can call as(1) and ld(1) directly.

Bitrig::Bitrig(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *Bitrig::buildAssembler() const {
  return new tools::bitrig::Assembler(*this);
}

Tool *Bitrig::buildLinker() const { return new tools::bitrig::Linker(*this); }

ToolChain::CXXStdlibType Bitrig::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;

    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);
  }
  return ToolChain::CST_Libcxx;
}

void Bitrig::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                          ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx:
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/v1");
    break;
  case ToolChain::CST_Libstdcxx:
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/stdc++");
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/stdc++/backward");

    StringRef Triple = getTriple().str();
    if (Triple.startswith("amd64"))
      addSystemInclude(DriverArgs, CC1Args,
                       getDriver().SysRoot + "/usr/include/c++/stdc++/x86_64" +
                           Triple.substr(5));
    else
      addSystemInclude(DriverArgs, CC1Args, getDriver().SysRoot +
                                                "/usr/include/c++/stdc++/" +
                                                Triple);
    break;
  }
}

void Bitrig::AddCXXStdlibLibArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    CmdArgs.push_back("-lc++abi");
    CmdArgs.push_back("-lpthread");
    break;
  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    break;
  }
}

/// FreeBSD - FreeBSD tool chain which can call as(1) and ld(1) directly.

FreeBSD::FreeBSD(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {

  // When targeting 32-bit platforms, look for '/usr/lib32/crt1.o' and fall
  // back to '/usr/lib' if it doesn't exist.
  if ((Triple.getArch() == llvm::Triple::x86 ||
       Triple.getArch() == llvm::Triple::ppc) &&
      D.getVFS().exists(getDriver().SysRoot + "/usr/lib32/crt1.o"))
    getFilePaths().push_back(getDriver().SysRoot + "/usr/lib32");
  else
    getFilePaths().push_back(getDriver().SysRoot + "/usr/lib");
}

ToolChain::CXXStdlibType FreeBSD::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;

    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);
  }
  if (getTriple().getOSMajorVersion() >= 10)
    return ToolChain::CST_Libcxx;
  return ToolChain::CST_Libstdcxx;
}

void FreeBSD::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx:
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/v1");
    break;
  case ToolChain::CST_Libstdcxx:
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/4.2");
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/4.2/backward");
    break;
  }
}

Tool *FreeBSD::buildAssembler() const {
  return new tools::freebsd::Assembler(*this);
}

Tool *FreeBSD::buildLinker() const { return new tools::freebsd::Linker(*this); }

bool FreeBSD::UseSjLjExceptions(const ArgList &Args) const {
  // FreeBSD uses SjLj exceptions on ARM oabi.
  switch (getTriple().getEnvironment()) {
  case llvm::Triple::GNUEABIHF:
  case llvm::Triple::GNUEABI:
  case llvm::Triple::EABI:
    return false;

  default:
    return (getTriple().getArch() == llvm::Triple::arm ||
            getTriple().getArch() == llvm::Triple::thumb);
  }
}

bool FreeBSD::HasNativeLLVMSupport() const { return true; }

bool FreeBSD::isPIEDefault() const { return getSanitizerArgs().requiresPIE(); }

SanitizerMask FreeBSD::getSupportedSanitizers() const {
  const bool IsX86 = getTriple().getArch() == llvm::Triple::x86;
  const bool IsX86_64 = getTriple().getArch() == llvm::Triple::x86_64;
  const bool IsMIPS64 = getTriple().getArch() == llvm::Triple::mips64 ||
                        getTriple().getArch() == llvm::Triple::mips64el;
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::Vptr;
  if (IsX86_64 || IsMIPS64) {
    Res |= SanitizerKind::Leak;
    Res |= SanitizerKind::Thread;
  }
  if (IsX86 || IsX86_64) {
    Res |= SanitizerKind::SafeStack;
  }
  return Res;
}

/// NetBSD - NetBSD tool chain which can call as(1) and ld(1) directly.

NetBSD::NetBSD(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {

  if (getDriver().UseStdLib) {
    // When targeting a 32-bit platform, try the special directory used on
    // 64-bit hosts, and only fall back to the main library directory if that
    // doesn't work.
    // FIXME: It'd be nicer to test if this directory exists, but I'm not sure
    // what all logic is needed to emulate the '=' prefix here.
    switch (Triple.getArch()) {
    case llvm::Triple::x86:
      getFilePaths().push_back("=/usr/lib/i386");
      break;
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
      switch (Triple.getEnvironment()) {
      case llvm::Triple::EABI:
      case llvm::Triple::GNUEABI:
        getFilePaths().push_back("=/usr/lib/eabi");
        break;
      case llvm::Triple::EABIHF:
      case llvm::Triple::GNUEABIHF:
        getFilePaths().push_back("=/usr/lib/eabihf");
        break;
      default:
        getFilePaths().push_back("=/usr/lib/oabi");
        break;
      }
      break;
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
      if (tools::mips::hasMipsAbiArg(Args, "o32"))
        getFilePaths().push_back("=/usr/lib/o32");
      else if (tools::mips::hasMipsAbiArg(Args, "64"))
        getFilePaths().push_back("=/usr/lib/64");
      break;
    case llvm::Triple::ppc:
      getFilePaths().push_back("=/usr/lib/powerpc");
      break;
    case llvm::Triple::sparc:
      getFilePaths().push_back("=/usr/lib/sparc");
      break;
    default:
      break;
    }

    getFilePaths().push_back("=/usr/lib");
  }
}

Tool *NetBSD::buildAssembler() const {
  return new tools::netbsd::Assembler(*this);
}

Tool *NetBSD::buildLinker() const { return new tools::netbsd::Linker(*this); }

ToolChain::CXXStdlibType NetBSD::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;

    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);
  }

  unsigned Major, Minor, Micro;
  getTriple().getOSVersion(Major, Minor, Micro);
  if (Major >= 7 || (Major == 6 && Minor == 99 && Micro >= 49) || Major == 0) {
    switch (getArch()) {
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
      return ToolChain::CST_Libcxx;
    default:
      break;
    }
  }
  return ToolChain::CST_Libstdcxx;
}

void NetBSD::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                          ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  switch (GetCXXStdlibType(DriverArgs)) {
  case ToolChain::CST_Libcxx:
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/");
    break;
  case ToolChain::CST_Libstdcxx:
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/g++");
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/g++/backward");
    break;
  }
}

/// Minix - Minix tool chain which can call as(1) and ld(1) directly.

Minix::Minix(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *Minix::buildAssembler() const {
  return new tools::minix::Assembler(*this);
}

Tool *Minix::buildLinker() const { return new tools::minix::Linker(*this); }

static void addPathIfExists(const Driver &D, const Twine &Path,
                            ToolChain::path_list &Paths) {
  if (D.getVFS().exists(Path))
    Paths.push_back(Path.str());
}

/// Solaris - Solaris tool chain which can call as(1) and ld(1) directly.

Solaris::Solaris(const Driver &D, const llvm::Triple &Triple,
                 const ArgList &Args)
    : Generic_GCC(D, Triple, Args) {

  GCCInstallation.init(Triple, Args);

  path_list &Paths = getFilePaths();
  if (GCCInstallation.isValid())
    addPathIfExists(D, GCCInstallation.getInstallPath(), Paths);

  addPathIfExists(D, getDriver().getInstalledDir(), Paths);
  if (getDriver().getInstalledDir() != getDriver().Dir)
    addPathIfExists(D, getDriver().Dir, Paths);

  addPathIfExists(D, getDriver().SysRoot + getDriver().Dir + "/../lib", Paths);

  std::string LibPath = "/usr/lib/";
  switch (Triple.getArch()) {
  case llvm::Triple::x86:
  case llvm::Triple::sparc:
    break;
  case llvm::Triple::x86_64:
    LibPath += "amd64/";
    break;
  case llvm::Triple::sparcv9:
    LibPath += "sparcv9/";
    break;
  default:
    llvm_unreachable("Unsupported architecture");
  }

  addPathIfExists(D, getDriver().SysRoot + LibPath, Paths);
}

Tool *Solaris::buildAssembler() const {
  return new tools::solaris::Assembler(*this);
}

Tool *Solaris::buildLinker() const { return new tools::solaris::Linker(*this); }

void Solaris::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  // Include the support directory for things like xlocale and fudged system
  // headers.
  addSystemInclude(DriverArgs, CC1Args, "/usr/include/c++/v1/support/solaris");

  if (GCCInstallation.isValid()) {
    GCCVersion Version = GCCInstallation.getVersion();
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/gcc/" +
                     Version.MajorStr + "." +
                     Version.MinorStr +
                     "/include/c++/" + Version.Text);
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/gcc/" + Version.MajorStr +
                     "." + Version.MinorStr + "/include/c++/" +
                     Version.Text + "/" +
                     GCCInstallation.getTriple().str());
  }
}

/// Distribution (very bare-bones at the moment).

enum Distro {
  // NB: Releases of a particular Linux distro should be kept together
  // in this enum, because some tests are done by integer comparison against
  // the first and last known member in the family, e.g. IsRedHat().
  ArchLinux,
  DebianLenny,
  DebianSqueeze,
  DebianWheezy,
  DebianJessie,
  DebianStretch,
  Exherbo,
  RHEL4,
  RHEL5,
  RHEL6,
  RHEL7,
  Fedora,
  OpenSUSE,
  UbuntuHardy,
  UbuntuIntrepid,
  UbuntuJaunty,
  UbuntuKarmic,
  UbuntuLucid,
  UbuntuMaverick,
  UbuntuNatty,
  UbuntuOneiric,
  UbuntuPrecise,
  UbuntuQuantal,
  UbuntuRaring,
  UbuntuSaucy,
  UbuntuTrusty,
  UbuntuUtopic,
  UbuntuVivid,
  UbuntuWily,
  UbuntuXenial,
  UnknownDistro
};

static bool IsRedhat(enum Distro Distro) {
  return Distro == Fedora || (Distro >= RHEL4 && Distro <= RHEL7);
}

static bool IsOpenSUSE(enum Distro Distro) { return Distro == OpenSUSE; }

static bool IsDebian(enum Distro Distro) {
  return Distro >= DebianLenny && Distro <= DebianStretch;
}

static bool IsUbuntu(enum Distro Distro) {
  return Distro >= UbuntuHardy && Distro <= UbuntuXenial;
}

static Distro DetectDistro(const Driver &D, llvm::Triple::ArchType Arch) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> File =
      llvm::MemoryBuffer::getFile("/etc/lsb-release");
  if (File) {
    StringRef Data = File.get()->getBuffer();
    SmallVector<StringRef, 16> Lines;
    Data.split(Lines, "\n");
    Distro Version = UnknownDistro;
    for (StringRef Line : Lines)
      if (Version == UnknownDistro && Line.startswith("DISTRIB_CODENAME="))
        Version = llvm::StringSwitch<Distro>(Line.substr(17))
                      .Case("hardy", UbuntuHardy)
                      .Case("intrepid", UbuntuIntrepid)
                      .Case("jaunty", UbuntuJaunty)
                      .Case("karmic", UbuntuKarmic)
                      .Case("lucid", UbuntuLucid)
                      .Case("maverick", UbuntuMaverick)
                      .Case("natty", UbuntuNatty)
                      .Case("oneiric", UbuntuOneiric)
                      .Case("precise", UbuntuPrecise)
                      .Case("quantal", UbuntuQuantal)
                      .Case("raring", UbuntuRaring)
                      .Case("saucy", UbuntuSaucy)
                      .Case("trusty", UbuntuTrusty)
                      .Case("utopic", UbuntuUtopic)
                      .Case("vivid", UbuntuVivid)
                      .Case("wily", UbuntuWily)
                      .Case("xenial", UbuntuXenial)
                      .Default(UnknownDistro);
    return Version;
  }

  File = llvm::MemoryBuffer::getFile("/etc/redhat-release");
  if (File) {
    StringRef Data = File.get()->getBuffer();
    if (Data.startswith("Fedora release"))
      return Fedora;
    if (Data.startswith("Red Hat Enterprise Linux") ||
        Data.startswith("CentOS")) {
      if (Data.find("release 7") != StringRef::npos)
        return RHEL7;
      else if (Data.find("release 6") != StringRef::npos)
        return RHEL6;
      else if (Data.find("release 5") != StringRef::npos)
        return RHEL5;
      else if (Data.find("release 4") != StringRef::npos)
        return RHEL4;
    }
    return UnknownDistro;
  }

  File = llvm::MemoryBuffer::getFile("/etc/debian_version");
  if (File) {
    StringRef Data = File.get()->getBuffer();
    if (Data[0] == '5')
      return DebianLenny;
    else if (Data.startswith("squeeze/sid") || Data[0] == '6')
      return DebianSqueeze;
    else if (Data.startswith("wheezy/sid") || Data[0] == '7')
      return DebianWheezy;
    else if (Data.startswith("jessie/sid") || Data[0] == '8')
      return DebianJessie;
    else if (Data.startswith("stretch/sid") || Data[0] == '9')
      return DebianStretch;
    return UnknownDistro;
  }

  if (D.getVFS().exists("/etc/SuSE-release"))
    return OpenSUSE;

  if (D.getVFS().exists("/etc/exherbo-release"))
    return Exherbo;

  if (D.getVFS().exists("/etc/arch-release"))
    return ArchLinux;

  return UnknownDistro;
}

/// \brief Get our best guess at the multiarch triple for a target.
///
/// Debian-based systems are starting to use a multiarch setup where they use
/// a target-triple directory in the library and header search paths.
/// Unfortunately, this triple does not align with the vanilla target triple,
/// so we provide a rough mapping here.
static std::string getMultiarchTriple(const Driver &D,
                                      const llvm::Triple &TargetTriple,
                                      StringRef SysRoot) {
  llvm::Triple::EnvironmentType TargetEnvironment =
      TargetTriple.getEnvironment();

  // For most architectures, just use whatever we have rather than trying to be
  // clever.
  switch (TargetTriple.getArch()) {
  default:
    break;

  // We use the existence of '/lib/<triple>' as a directory to detect some
  // common linux triples that don't quite match the Clang triple for both
  // 32-bit and 64-bit targets. Multiarch fixes its install triples to these
  // regardless of what the actual target triple is.
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    if (TargetEnvironment == llvm::Triple::GNUEABIHF) {
      if (D.getVFS().exists(SysRoot + "/lib/arm-linux-gnueabihf"))
        return "arm-linux-gnueabihf";
    } else {
      if (D.getVFS().exists(SysRoot + "/lib/arm-linux-gnueabi"))
        return "arm-linux-gnueabi";
    }
    break;
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb:
    if (TargetEnvironment == llvm::Triple::GNUEABIHF) {
      if (D.getVFS().exists(SysRoot + "/lib/armeb-linux-gnueabihf"))
        return "armeb-linux-gnueabihf";
    } else {
      if (D.getVFS().exists(SysRoot + "/lib/armeb-linux-gnueabi"))
        return "armeb-linux-gnueabi";
    }
    break;
  case llvm::Triple::x86:
    if (D.getVFS().exists(SysRoot + "/lib/i386-linux-gnu"))
      return "i386-linux-gnu";
    break;
  case llvm::Triple::x86_64:
    // We don't want this for x32, otherwise it will match x86_64 libs
    if (TargetEnvironment != llvm::Triple::GNUX32 &&
        D.getVFS().exists(SysRoot + "/lib/x86_64-linux-gnu"))
      return "x86_64-linux-gnu";
    break;
  case llvm::Triple::aarch64:
    if (D.getVFS().exists(SysRoot + "/lib/aarch64-linux-gnu"))
      return "aarch64-linux-gnu";
    break;
  case llvm::Triple::aarch64_be:
    if (D.getVFS().exists(SysRoot + "/lib/aarch64_be-linux-gnu"))
      return "aarch64_be-linux-gnu";
    break;
  case llvm::Triple::mips:
    if (D.getVFS().exists(SysRoot + "/lib/mips-linux-gnu"))
      return "mips-linux-gnu";
    break;
  case llvm::Triple::mipsel:
    if (D.getVFS().exists(SysRoot + "/lib/mipsel-linux-gnu"))
      return "mipsel-linux-gnu";
    break;
  case llvm::Triple::mips64:
    if (D.getVFS().exists(SysRoot + "/lib/mips64-linux-gnu"))
      return "mips64-linux-gnu";
    if (D.getVFS().exists(SysRoot + "/lib/mips64-linux-gnuabi64"))
      return "mips64-linux-gnuabi64";
    break;
  case llvm::Triple::mips64el:
    if (D.getVFS().exists(SysRoot + "/lib/mips64el-linux-gnu"))
      return "mips64el-linux-gnu";
    if (D.getVFS().exists(SysRoot + "/lib/mips64el-linux-gnuabi64"))
      return "mips64el-linux-gnuabi64";
    break;
  case llvm::Triple::ppc:
    if (D.getVFS().exists(SysRoot + "/lib/powerpc-linux-gnuspe"))
      return "powerpc-linux-gnuspe";
    if (D.getVFS().exists(SysRoot + "/lib/powerpc-linux-gnu"))
      return "powerpc-linux-gnu";
    break;
  case llvm::Triple::ppc64:
    if (D.getVFS().exists(SysRoot + "/lib/powerpc64-linux-gnu"))
      return "powerpc64-linux-gnu";
    break;
  case llvm::Triple::ppc64le:
    if (D.getVFS().exists(SysRoot + "/lib/powerpc64le-linux-gnu"))
      return "powerpc64le-linux-gnu";
    break;
  case llvm::Triple::sparc:
    if (D.getVFS().exists(SysRoot + "/lib/sparc-linux-gnu"))
      return "sparc-linux-gnu";
    break;
  case llvm::Triple::sparcv9:
    if (D.getVFS().exists(SysRoot + "/lib/sparc64-linux-gnu"))
      return "sparc64-linux-gnu";
    break;
  case llvm::Triple::systemz:
    if (D.getVFS().exists(SysRoot + "/lib/s390x-linux-gnu"))
      return "s390x-linux-gnu";
    break;
  }
  return TargetTriple.str();
}

static StringRef getOSLibDir(const llvm::Triple &Triple, const ArgList &Args) {
  if (isMipsArch(Triple.getArch())) {
    // lib32 directory has a special meaning on MIPS targets.
    // It contains N32 ABI binaries. Use this folder if produce
    // code for N32 ABI only.
    if (tools::mips::hasMipsAbiArg(Args, "n32"))
      return "lib32";
    return Triple.isArch32Bit() ? "lib" : "lib64";
  }

  // It happens that only x86 and PPC use the 'lib32' variant of oslibdir, and
  // using that variant while targeting other architectures causes problems
  // because the libraries are laid out in shared system roots that can't cope
  // with a 'lib32' library search path being considered. So we only enable
  // them when we know we may need it.
  //
  // FIXME: This is a bit of a hack. We should really unify this code for
  // reasoning about oslibdir spellings with the lib dir spellings in the
  // GCCInstallationDetector, but that is a more significant refactoring.
  if (Triple.getArch() == llvm::Triple::x86 ||
      Triple.getArch() == llvm::Triple::ppc)
    return "lib32";

  if (Triple.getArch() == llvm::Triple::x86_64 &&
      Triple.getEnvironment() == llvm::Triple::GNUX32)
    return "libx32";

  return Triple.isArch32Bit() ? "lib" : "lib64";
}

Linux::Linux(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  CudaInstallation.init(Triple, Args);
  Multilibs = GCCInstallation.getMultilibs();
  llvm::Triple::ArchType Arch = Triple.getArch();
  std::string SysRoot = computeSysRoot();

  // Cross-compiling binutils and GCC installations (vanilla and openSUSE at
  // least) put various tools in a triple-prefixed directory off of the parent
  // of the GCC installation. We use the GCC triple here to ensure that we end
  // up with tools that support the same amount of cross compiling as the
  // detected GCC installation. For example, if we find a GCC installation
  // targeting x86_64, but it is a bi-arch GCC installation, it can also be
  // used to target i386.
  // FIXME: This seems unlikely to be Linux-specific.
  ToolChain::path_list &PPaths = getProgramPaths();
  PPaths.push_back(Twine(GCCInstallation.getParentLibPath() + "/../" +
                         GCCInstallation.getTriple().str() + "/bin")
                       .str());

  Distro Distro = DetectDistro(D, Arch);

  if (IsOpenSUSE(Distro) || IsUbuntu(Distro)) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("relro");
  }

  if (Arch == llvm::Triple::arm || Arch == llvm::Triple::thumb)
    ExtraOpts.push_back("-X");

  const bool IsAndroid = Triple.isAndroid();
  const bool IsMips = isMipsArch(Arch);

  if (IsMips && !SysRoot.empty())
    ExtraOpts.push_back("--sysroot=" + SysRoot);

  // Do not use 'gnu' hash style for Mips targets because .gnu.hash
  // and the MIPS ABI require .dynsym to be sorted in different ways.
  // .gnu.hash needs symbols to be grouped by hash code whereas the MIPS
  // ABI requires a mapping between the GOT and the symbol table.
  // Android loader does not support .gnu.hash.
  if (!IsMips && !IsAndroid) {
    if (IsRedhat(Distro) || IsOpenSUSE(Distro) ||
        (IsUbuntu(Distro) && Distro >= UbuntuMaverick))
      ExtraOpts.push_back("--hash-style=gnu");

    if (IsDebian(Distro) || IsOpenSUSE(Distro) || Distro == UbuntuLucid ||
        Distro == UbuntuJaunty || Distro == UbuntuKarmic)
      ExtraOpts.push_back("--hash-style=both");
  }

  if (IsRedhat(Distro))
    ExtraOpts.push_back("--no-add-needed");

  if ((IsDebian(Distro) && Distro >= DebianSqueeze) || IsOpenSUSE(Distro) ||
      (IsRedhat(Distro) && Distro != RHEL4 && Distro != RHEL5) ||
      (IsUbuntu(Distro) && Distro >= UbuntuKarmic))
    ExtraOpts.push_back("--build-id");

  if (IsOpenSUSE(Distro))
    ExtraOpts.push_back("--enable-new-dtags");

  // The selection of paths to try here is designed to match the patterns which
  // the GCC driver itself uses, as this is part of the GCC-compatible driver.
  // This was determined by running GCC in a fake filesystem, creating all
  // possible permutations of these directories, and seeing which ones it added
  // to the link paths.
  path_list &Paths = getFilePaths();

  const std::string OSLibDir = getOSLibDir(Triple, Args);
  const std::string MultiarchTriple = getMultiarchTriple(D, Triple, SysRoot);

  // Add the multilib suffixed paths where they are available.
  if (GCCInstallation.isValid()) {
    const llvm::Triple &GCCTriple = GCCInstallation.getTriple();
    const std::string &LibPath = GCCInstallation.getParentLibPath();
    const Multilib &Multilib = GCCInstallation.getMultilib();

    // Sourcery CodeBench MIPS toolchain holds some libraries under
    // a biarch-like suffix of the GCC installation.
    addPathIfExists(D, GCCInstallation.getInstallPath() + Multilib.gccSuffix(),
                    Paths);

    // GCC cross compiling toolchains will install target libraries which ship
    // as part of the toolchain under <prefix>/<triple>/<libdir> rather than as
    // any part of the GCC installation in
    // <prefix>/<libdir>/gcc/<triple>/<version>. This decision is somewhat
    // debatable, but is the reality today. We need to search this tree even
    // when we have a sysroot somewhere else. It is the responsibility of
    // whomever is doing the cross build targeting a sysroot using a GCC
    // installation that is *not* within the system root to ensure two things:
    //
    //  1) Any DSOs that are linked in from this tree or from the install path
    //     above must be present on the system root and found via an
    //     appropriate rpath.
    //  2) There must not be libraries installed into
    //     <prefix>/<triple>/<libdir> unless they should be preferred over
    //     those within the system root.
    //
    // Note that this matches the GCC behavior. See the below comment for where
    // Clang diverges from GCC's behavior.
    addPathIfExists(D, LibPath + "/../" + GCCTriple.str() + "/lib/../" +
                           OSLibDir + Multilib.osSuffix(),
                    Paths);

    // If the GCC installation we found is inside of the sysroot, we want to
    // prefer libraries installed in the parent prefix of the GCC installation.
    // It is important to *not* use these paths when the GCC installation is
    // outside of the system root as that can pick up unintended libraries.
    // This usually happens when there is an external cross compiler on the
    // host system, and a more minimal sysroot available that is the target of
    // the cross. Note that GCC does include some of these directories in some
    // configurations but this seems somewhere between questionable and simply
    // a bug.
    if (StringRef(LibPath).startswith(SysRoot)) {
      addPathIfExists(D, LibPath + "/" + MultiarchTriple, Paths);
      addPathIfExists(D, LibPath + "/../" + OSLibDir, Paths);
    }
  }

  // Similar to the logic for GCC above, if we currently running Clang inside
  // of the requested system root, add its parent library paths to
  // those searched.
  // FIXME: It's not clear whether we should use the driver's installed
  // directory ('Dir' below) or the ResourceDir.
  if (StringRef(D.Dir).startswith(SysRoot)) {
    addPathIfExists(D, D.Dir + "/../lib/" + MultiarchTriple, Paths);
    addPathIfExists(D, D.Dir + "/../" + OSLibDir, Paths);
  }

  addPathIfExists(D, SysRoot + "/lib/" + MultiarchTriple, Paths);
  addPathIfExists(D, SysRoot + "/lib/../" + OSLibDir, Paths);
  addPathIfExists(D, SysRoot + "/usr/lib/" + MultiarchTriple, Paths);
  addPathIfExists(D, SysRoot + "/usr/lib/../" + OSLibDir, Paths);

  // Try walking via the GCC triple path in case of biarch or multiarch GCC
  // installations with strange symlinks.
  if (GCCInstallation.isValid()) {
    addPathIfExists(D,
                    SysRoot + "/usr/lib/" + GCCInstallation.getTriple().str() +
                        "/../../" + OSLibDir,
                    Paths);

    // Add the 'other' biarch variant path
    Multilib BiarchSibling;
    if (GCCInstallation.getBiarchSibling(BiarchSibling)) {
      addPathIfExists(D, GCCInstallation.getInstallPath() +
                             BiarchSibling.gccSuffix(),
                      Paths);
    }

    // See comments above on the multilib variant for details of why this is
    // included even from outside the sysroot.
    const std::string &LibPath = GCCInstallation.getParentLibPath();
    const llvm::Triple &GCCTriple = GCCInstallation.getTriple();
    const Multilib &Multilib = GCCInstallation.getMultilib();
    addPathIfExists(D, LibPath + "/../" + GCCTriple.str() + "/lib" +
                           Multilib.osSuffix(),
                    Paths);

    // See comments above on the multilib variant for details of why this is
    // only included from within the sysroot.
    if (StringRef(LibPath).startswith(SysRoot))
      addPathIfExists(D, LibPath, Paths);
  }

  // Similar to the logic for GCC above, if we are currently running Clang
  // inside of the requested system root, add its parent library path to those
  // searched.
  // FIXME: It's not clear whether we should use the driver's installed
  // directory ('Dir' below) or the ResourceDir.
  if (StringRef(D.Dir).startswith(SysRoot))
    addPathIfExists(D, D.Dir + "/../lib", Paths);

  addPathIfExists(D, SysRoot + "/lib", Paths);
  addPathIfExists(D, SysRoot + "/usr/lib", Paths);
}

bool Linux::HasNativeLLVMSupport() const { return true; }

Tool *Linux::buildLinker() const { return new tools::gnutools::Linker(*this); }

Tool *Linux::buildAssembler() const {
  return new tools::gnutools::Assembler(*this);
}

std::string Linux::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot;

  if (!GCCInstallation.isValid() || !isMipsArch(getTriple().getArch()))
    return std::string();

  // Standalone MIPS toolchains use different names for sysroot folder
  // and put it into different places. Here we try to check some known
  // variants.

  const StringRef InstallDir = GCCInstallation.getInstallPath();
  const StringRef TripleStr = GCCInstallation.getTriple().str();
  const Multilib &Multilib = GCCInstallation.getMultilib();

  std::string Path =
      (InstallDir + "/../../../../" + TripleStr + "/libc" + Multilib.osSuffix())
          .str();

  if (getVFS().exists(Path))
    return Path;

  Path = (InstallDir + "/../../../../sysroot" + Multilib.osSuffix()).str();

  if (getVFS().exists(Path))
    return Path;

  return std::string();
}

void Linux::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  std::string SysRoot = computeSysRoot();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc))
    addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/local/include");

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> P(D.ResourceDir);
    llvm::sys::path::append(P, "include");
    addSystemInclude(DriverArgs, CC1Args, P);
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? StringRef(SysRoot) : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  // Lacking those, try to detect the correct set of system includes for the
  // target triple.

  // Add include directories specific to the selected multilib set and multilib.
  if (GCCInstallation.isValid()) {
    const auto &Callback = Multilibs.includeDirsCallback();
    if (Callback) {
      const auto IncludePaths = Callback(GCCInstallation.getInstallPath(),
                                         GCCInstallation.getTriple().str(),
                                         GCCInstallation.getMultilib());
      for (const auto &Path : IncludePaths)
        addExternCSystemIncludeIfExists(DriverArgs, CC1Args, Path);
    }
  }

  // Implement generic Debian multiarch support.
  const StringRef X86_64MultiarchIncludeDirs[] = {
      "/usr/include/x86_64-linux-gnu",

      // FIXME: These are older forms of multiarch. It's not clear that they're
      // in use in any released version of Debian, so we should consider
      // removing them.
      "/usr/include/i686-linux-gnu/64", "/usr/include/i486-linux-gnu/64"};
  const StringRef X86MultiarchIncludeDirs[] = {
      "/usr/include/i386-linux-gnu",

      // FIXME: These are older forms of multiarch. It's not clear that they're
      // in use in any released version of Debian, so we should consider
      // removing them.
      "/usr/include/x86_64-linux-gnu/32", "/usr/include/i686-linux-gnu",
      "/usr/include/i486-linux-gnu"};
  const StringRef AArch64MultiarchIncludeDirs[] = {
      "/usr/include/aarch64-linux-gnu"};
  const StringRef ARMMultiarchIncludeDirs[] = {
      "/usr/include/arm-linux-gnueabi"};
  const StringRef ARMHFMultiarchIncludeDirs[] = {
      "/usr/include/arm-linux-gnueabihf"};
  const StringRef MIPSMultiarchIncludeDirs[] = {"/usr/include/mips-linux-gnu"};
  const StringRef MIPSELMultiarchIncludeDirs[] = {
      "/usr/include/mipsel-linux-gnu"};
  const StringRef MIPS64MultiarchIncludeDirs[] = {
      "/usr/include/mips64-linux-gnu", "/usr/include/mips64-linux-gnuabi64"};
  const StringRef MIPS64ELMultiarchIncludeDirs[] = {
      "/usr/include/mips64el-linux-gnu",
      "/usr/include/mips64el-linux-gnuabi64"};
  const StringRef PPCMultiarchIncludeDirs[] = {
      "/usr/include/powerpc-linux-gnu"};
  const StringRef PPC64MultiarchIncludeDirs[] = {
      "/usr/include/powerpc64-linux-gnu"};
  const StringRef PPC64LEMultiarchIncludeDirs[] = {
      "/usr/include/powerpc64le-linux-gnu"};
  const StringRef SparcMultiarchIncludeDirs[] = {
      "/usr/include/sparc-linux-gnu"};
  const StringRef Sparc64MultiarchIncludeDirs[] = {
      "/usr/include/sparc64-linux-gnu"};
  const StringRef SYSTEMZMultiarchIncludeDirs[] = {
      "/usr/include/s390x-linux-gnu"};
  ArrayRef<StringRef> MultiarchIncludeDirs;
  switch (getTriple().getArch()) {
  case llvm::Triple::x86_64:
    MultiarchIncludeDirs = X86_64MultiarchIncludeDirs;
    break;
  case llvm::Triple::x86:
    MultiarchIncludeDirs = X86MultiarchIncludeDirs;
    break;
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
    MultiarchIncludeDirs = AArch64MultiarchIncludeDirs;
    break;
  case llvm::Triple::arm:
    if (getTriple().getEnvironment() == llvm::Triple::GNUEABIHF)
      MultiarchIncludeDirs = ARMHFMultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = ARMMultiarchIncludeDirs;
    break;
  case llvm::Triple::mips:
    MultiarchIncludeDirs = MIPSMultiarchIncludeDirs;
    break;
  case llvm::Triple::mipsel:
    MultiarchIncludeDirs = MIPSELMultiarchIncludeDirs;
    break;
  case llvm::Triple::mips64:
    MultiarchIncludeDirs = MIPS64MultiarchIncludeDirs;
    break;
  case llvm::Triple::mips64el:
    MultiarchIncludeDirs = MIPS64ELMultiarchIncludeDirs;
    break;
  case llvm::Triple::ppc:
    MultiarchIncludeDirs = PPCMultiarchIncludeDirs;
    break;
  case llvm::Triple::ppc64:
    MultiarchIncludeDirs = PPC64MultiarchIncludeDirs;
    break;
  case llvm::Triple::ppc64le:
    MultiarchIncludeDirs = PPC64LEMultiarchIncludeDirs;
    break;
  case llvm::Triple::sparc:
    MultiarchIncludeDirs = SparcMultiarchIncludeDirs;
    break;
  case llvm::Triple::sparcv9:
    MultiarchIncludeDirs = Sparc64MultiarchIncludeDirs;
    break;
  case llvm::Triple::systemz:
    MultiarchIncludeDirs = SYSTEMZMultiarchIncludeDirs;
    break;
  default:
    break;
  }
  for (StringRef Dir : MultiarchIncludeDirs) {
    if (D.getVFS().exists(SysRoot + Dir)) {
      addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + Dir);
      break;
    }
  }

  if (getTriple().getOS() == llvm::Triple::RTEMS)
    return;

  // Add an include of '/include' directly. This isn't provided by default by
  // system GCCs, but is often used with cross-compiling GCCs, and harmless to
  // add even when Clang is acting as-if it were a system compiler.
  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/include");

  addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/include");
}


static std::string DetectLibcxxIncludePath(StringRef base) {
  std::error_code EC;
  int MaxVersion = 0;
  std::string MaxVersionString = "";
  for (llvm::sys::fs::directory_iterator LI(base, EC), LE; !EC && LI != LE;
       LI = LI.increment(EC)) {
    StringRef VersionText = llvm::sys::path::filename(LI->path());
    int Version;
    if (VersionText[0] == 'v' &&
        !VersionText.slice(1, StringRef::npos).getAsInteger(10, Version)) {
      if (Version > MaxVersion) {
        MaxVersion = Version;
        MaxVersionString = VersionText;
      }
    }
  }
  return MaxVersion ? (base + "/" + MaxVersionString).str() : "";
}

void Linux::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  // Check if libc++ has been enabled and provide its include paths if so.
  if (GetCXXStdlibType(DriverArgs) == ToolChain::CST_Libcxx) {
    const std::string LibCXXIncludePathCandidates[] = {
        DetectLibcxxIncludePath(getDriver().Dir + "/../include/c++"),

        // We also check the system as for a long time this is the only place
        // Clang looked.
        // FIXME: We should really remove this. It doesn't make any sense.
        DetectLibcxxIncludePath(getDriver().SysRoot + "/usr/include/c++")};
    for (const auto &IncludePath : LibCXXIncludePathCandidates) {
      if (IncludePath.empty() || !getVFS().exists(IncludePath))
        continue;
      // Add the first candidate that exists.
      addSystemInclude(DriverArgs, CC1Args, IncludePath);
      break;
    }
    return;
  }

  // We need a detected GCC installation on Linux to provide libstdc++'s
  // headers. We handled the libc++ case above.
  if (!GCCInstallation.isValid())
    return;

  // By default, look for the C++ headers in an include directory adjacent to
  // the lib directory of the GCC installation. Note that this is expect to be
  // equivalent to '/usr/include/c++/X.Y' in almost all cases.
  StringRef LibDir = GCCInstallation.getParentLibPath();
  StringRef InstallDir = GCCInstallation.getInstallPath();
  StringRef TripleStr = GCCInstallation.getTriple().str();
  const Multilib &Multilib = GCCInstallation.getMultilib();
  const std::string GCCMultiarchTriple = getMultiarchTriple(
      getDriver(), GCCInstallation.getTriple(), getDriver().SysRoot);
  const std::string TargetMultiarchTriple =
      getMultiarchTriple(getDriver(), getTriple(), getDriver().SysRoot);
  const GCCVersion &Version = GCCInstallation.getVersion();

  // The primary search for libstdc++ supports multiarch variants.
  if (addLibStdCXXIncludePaths(LibDir.str() + "/../include",
                               "/c++/" + Version.Text, TripleStr,
                               GCCMultiarchTriple, TargetMultiarchTriple,
                               Multilib.includeSuffix(), DriverArgs, CC1Args))
    return;

  // Otherwise, fall back on a bunch of options which don't use multiarch
  // layouts for simplicity.
  const std::string LibStdCXXIncludePathCandidates[] = {
      // Gentoo is weird and places its headers inside the GCC install,
      // so if the first attempt to find the headers fails, try these patterns.
      InstallDir.str() + "/include/g++-v" + Version.MajorStr + "." +
          Version.MinorStr,
      InstallDir.str() + "/include/g++-v" + Version.MajorStr,
      // Android standalone toolchain has C++ headers in yet another place.
      LibDir.str() + "/../" + TripleStr.str() + "/include/c++/" + Version.Text,
      // Freescale SDK C++ headers are directly in <sysroot>/usr/include/c++,
      // without a subdirectory corresponding to the gcc version.
      LibDir.str() + "/../include/c++",
  };

  for (const auto &IncludePath : LibStdCXXIncludePathCandidates) {
    if (addLibStdCXXIncludePaths(IncludePath, /*Suffix*/ "", TripleStr,
                                 /*GCCMultiarchTriple*/ "",
                                 /*TargetMultiarchTriple*/ "",
                                 Multilib.includeSuffix(), DriverArgs, CC1Args))
      break;
  }
}

void Linux::AddCudaIncludeArgs(const ArgList &DriverArgs,
                               ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nocudainc))
    return;

  if (CudaInstallation.isValid()) {
    addSystemInclude(DriverArgs, CC1Args, CudaInstallation.getIncludePath());
    CC1Args.push_back("-include");
    CC1Args.push_back("cuda_runtime.h");
  }
}

bool Linux::isPIEDefault() const { return getSanitizerArgs().requiresPIE(); }

SanitizerMask Linux::getSupportedSanitizers() const {
  const bool IsX86 = getTriple().getArch() == llvm::Triple::x86;
  const bool IsX86_64 = getTriple().getArch() == llvm::Triple::x86_64;
  const bool IsMIPS64 = getTriple().getArch() == llvm::Triple::mips64 ||
                        getTriple().getArch() == llvm::Triple::mips64el;
  const bool IsPowerPC64 = getTriple().getArch() == llvm::Triple::ppc64 ||
                           getTriple().getArch() == llvm::Triple::ppc64le;
  const bool IsAArch64 = getTriple().getArch() == llvm::Triple::aarch64 ||
                         getTriple().getArch() == llvm::Triple::aarch64_be;
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::KernelAddress;
  Res |= SanitizerKind::Vptr;
  Res |= SanitizerKind::SafeStack;
  if (IsX86_64 || IsMIPS64 || IsAArch64)
    Res |= SanitizerKind::DataFlow;
  if (IsX86_64 || IsMIPS64 || IsAArch64)
    Res |= SanitizerKind::Leak;
  if (IsX86_64 || IsMIPS64 || IsAArch64 || IsPowerPC64)
    Res |= SanitizerKind::Thread;
  if (IsX86_64 || IsMIPS64 || IsPowerPC64 || IsAArch64)
    Res |= SanitizerKind::Memory;
  if (IsX86 || IsX86_64) {
    Res |= SanitizerKind::Function;
  }
  return Res;
}

void Linux::addProfileRTLibs(const llvm::opt::ArgList &Args,
                             llvm::opt::ArgStringList &CmdArgs) const {
  if (!needsProfileRT(Args)) return;

  // Add linker option -u__llvm_runtime_variable to cause runtime
  // initialization module to be linked in.
  if (!Args.hasArg(options::OPT_coverage))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("-u", llvm::getInstrProfRuntimeHookVarName())));
  ToolChain::addProfileRTLibs(Args, CmdArgs);
}

/// DragonFly - DragonFly tool chain which can call as(1) and ld(1) directly.

DragonFly::DragonFly(const Driver &D, const llvm::Triple &Triple,
                     const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {

  // Path mangling to find libexec
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
  if (D.getVFS().exists("/usr/lib/gcc47"))
    getFilePaths().push_back("/usr/lib/gcc47");
  else
    getFilePaths().push_back("/usr/lib/gcc44");
}

Tool *DragonFly::buildAssembler() const {
  return new tools::dragonfly::Assembler(*this);
}

Tool *DragonFly::buildLinker() const {
  return new tools::dragonfly::Linker(*this);
}

/// Stub for CUDA toolchain. At the moment we don't have assembler or
/// linker and need toolchain mainly to propagate device-side options
/// to CC1.

CudaToolChain::CudaToolChain(const Driver &D, const llvm::Triple &Triple,
                             const ArgList &Args)
    : Linux(D, Triple, Args) {}

void
CudaToolChain::addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const {
  Linux::addClangTargetOptions(DriverArgs, CC1Args);
  CC1Args.push_back("-fcuda-is-device");

  if (DriverArgs.hasArg(options::OPT_nocudalib))
    return;

  std::string LibDeviceFile = CudaInstallation.getLibDeviceFile(
      DriverArgs.getLastArgValue(options::OPT_march_EQ));
  if (!LibDeviceFile.empty()) {
    CC1Args.push_back("-mlink-cuda-bitcode");
    CC1Args.push_back(DriverArgs.MakeArgString(LibDeviceFile));

    // Libdevice in CUDA-7.0 requires PTX version that's more recent
    // than LLVM defaults to. Use PTX4.2 which is the PTX version that
    // came with CUDA-7.0.
    CC1Args.push_back("-target-feature");
    CC1Args.push_back("+ptx42");
  }
}

llvm::opt::DerivedArgList *
CudaToolChain::TranslateArgs(const llvm::opt::DerivedArgList &Args,
                             const char *BoundArch) const {
  DerivedArgList *DAL = new DerivedArgList(Args.getBaseArgs());
  const OptTable &Opts = getDriver().getOpts();

  for (Arg *A : Args) {
    if (A->getOption().matches(options::OPT_Xarch__)) {
      // Skip this argument unless the architecture matches BoundArch
      if (A->getValue(0) != StringRef(BoundArch))
        continue;

      unsigned Index = Args.getBaseArgs().MakeIndex(A->getValue(1));
      unsigned Prev = Index;
      std::unique_ptr<Arg> XarchArg(Opts.ParseOneArg(Args, Index));

      // If the argument parsing failed or more than one argument was
      // consumed, the -Xarch_ argument's parameter tried to consume
      // extra arguments. Emit an error and ignore.
      //
      // We also want to disallow any options which would alter the
      // driver behavior; that isn't going to work in our model. We
      // use isDriverOption() as an approximation, although things
      // like -O4 are going to slip through.
      if (!XarchArg || Index > Prev + 1) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_with_args)
            << A->getAsString(Args);
        continue;
      } else if (XarchArg->getOption().hasFlag(options::DriverOption)) {
        getDriver().Diag(diag::err_drv_invalid_Xarch_argument_isdriver)
            << A->getAsString(Args);
        continue;
      }
      XarchArg->setBaseArg(A);
      A = XarchArg.release();
      DAL->AddSynthesizedArg(A);
    }
    DAL->append(A);
  }

  DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ), BoundArch);
  return DAL;
}

/// XCore tool chain
XCoreToolChain::XCoreToolChain(const Driver &D, const llvm::Triple &Triple,
                               const ArgList &Args)
    : ToolChain(D, Triple, Args) {
  // ProgramPaths are found via 'PATH' environment variable.
}

Tool *XCoreToolChain::buildAssembler() const {
  return new tools::XCore::Assembler(*this);
}

Tool *XCoreToolChain::buildLinker() const {
  return new tools::XCore::Linker(*this);
}

bool XCoreToolChain::isPICDefault() const { return false; }

bool XCoreToolChain::isPIEDefault() const { return false; }

bool XCoreToolChain::isPICDefaultForced() const { return false; }

bool XCoreToolChain::SupportsProfiling() const { return false; }

bool XCoreToolChain::hasBlocksRuntime() const { return false; }

void XCoreToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                               ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;
  if (const char *cl_include_dir = getenv("XCC_C_INCLUDE_PATH")) {
    SmallVector<StringRef, 4> Dirs;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
    StringRef(cl_include_dir).split(Dirs, StringRef(EnvPathSeparatorStr));
    ArrayRef<StringRef> DirVec(Dirs);
    addSystemIncludes(DriverArgs, CC1Args, DirVec);
  }
}

void XCoreToolChain::addClangTargetOptions(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  CC1Args.push_back("-nostdsysteminc");
}

void XCoreToolChain::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;
  if (const char *cl_include_dir = getenv("XCC_CPLUS_INCLUDE_PATH")) {
    SmallVector<StringRef, 4> Dirs;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator, '\0'};
    StringRef(cl_include_dir).split(Dirs, StringRef(EnvPathSeparatorStr));
    ArrayRef<StringRef> DirVec(Dirs);
    addSystemIncludes(DriverArgs, CC1Args, DirVec);
  }
}

void XCoreToolChain::AddCXXStdlibLibArgs(const ArgList &Args,
                                         ArgStringList &CmdArgs) const {
  // We don't output any lib args. This is handled by xcc.
}

MyriadToolChain::MyriadToolChain(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args)
    : Generic_GCC(D, Triple, Args) {
  // If a target of 'sparc-myriad-elf' is specified to clang, it wants to use
  // 'sparc-myriad--elf' (note the unknown OS) as the canonical triple.
  // This won't work to find gcc. Instead we give the installation detector an
  // extra triple, which is preferable to further hacks of the logic that at
  // present is based solely on getArch(). In particular, it would be wrong to
  // choose the myriad installation when targeting a non-myriad sparc install.
  switch (Triple.getArch()) {
  default:
    D.Diag(diag::err_target_unsupported_arch) << Triple.getArchName()
                                              << "myriad";
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::shave:
    GCCInstallation.init(Triple, Args, {"sparc-myriad-elf"});
  }

  if (GCCInstallation.isValid()) {
    // The contents of LibDir are independent of the version of gcc.
    // This contains libc, libg (a superset of libc), libm, libstdc++, libssp.
    SmallString<128> LibDir(GCCInstallation.getParentLibPath());
    if (Triple.getArch() == llvm::Triple::sparcel)
      llvm::sys::path::append(LibDir, "../sparc-myriad-elf/lib/le");
    else
      llvm::sys::path::append(LibDir, "../sparc-myriad-elf/lib");
    addPathIfExists(D, LibDir, getFilePaths());

    // This directory contains crt{i,n,begin,end}.o as well as libgcc.
    // These files are tied to a particular version of gcc.
    SmallString<128> CompilerSupportDir(GCCInstallation.getInstallPath());
    // There are actually 4 choices: {le,be} x {fpu,nofpu}
    // but as this toolchain is for LEON sparc, it can assume FPU.
    if (Triple.getArch() == llvm::Triple::sparcel)
      llvm::sys::path::append(CompilerSupportDir, "le");
    addPathIfExists(D, CompilerSupportDir, getFilePaths());
  }
}

MyriadToolChain::~MyriadToolChain() {}

void MyriadToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args) const {
  if (!DriverArgs.hasArg(options::OPT_nostdinc))
    addSystemInclude(DriverArgs, CC1Args, getDriver().SysRoot + "/include");
}

void MyriadToolChain::AddClangCXXStdlibIncludeArgs(
    const ArgList &DriverArgs, ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  // Only libstdc++, for now.
  StringRef LibDir = GCCInstallation.getParentLibPath();
  const GCCVersion &Version = GCCInstallation.getVersion();
  StringRef TripleStr = GCCInstallation.getTriple().str();
  const Multilib &Multilib = GCCInstallation.getMultilib();

  addLibStdCXXIncludePaths(
      LibDir.str() + "/../" + TripleStr.str() + "/include/c++/" + Version.Text,
      "", TripleStr, "", "", Multilib.includeSuffix(), DriverArgs, CC1Args);
}

// MyriadToolChain handles several triples:
//  {shave,sparc{,el}}-myriad-{rtems,unknown}-elf
Tool *MyriadToolChain::SelectTool(const JobAction &JA) const {
  // The inherited method works fine if not targeting the SHAVE.
  if (!isShaveCompilation(getTriple()))
    return ToolChain::SelectTool(JA);
  switch (JA.getKind()) {
  case Action::PreprocessJobClass:
  case Action::CompileJobClass:
    if (!Compiler)
      Compiler.reset(new tools::SHAVE::Compiler(*this));
    return Compiler.get();
  case Action::AssembleJobClass:
    if (!Assembler)
      Assembler.reset(new tools::SHAVE::Assembler(*this));
    return Assembler.get();
  default:
    return ToolChain::getTool(JA.getKind());
  }
}

Tool *MyriadToolChain::buildLinker() const {
  return new tools::Myriad::Linker(*this);
}

bool WebAssembly::IsMathErrnoDefault() const { return false; }

bool WebAssembly::IsObjCNonFragileABIDefault() const { return true; }

bool WebAssembly::UseObjCMixedDispatch() const { return true; }

bool WebAssembly::isPICDefault() const { return false; }

bool WebAssembly::isPIEDefault() const { return false; }

bool WebAssembly::isPICDefaultForced() const { return false; }

bool WebAssembly::IsIntegratedAssemblerDefault() const { return true; }

// TODO: Support Objective C stuff.
bool WebAssembly::SupportsObjCGC() const { return false; }

bool WebAssembly::hasBlocksRuntime() const { return false; }

// TODO: Support profiling.
bool WebAssembly::SupportsProfiling() const { return false; }

void WebAssembly::addClangTargetOptions(const ArgList &DriverArgs,
                                        ArgStringList &CC1Args) const {
  if (DriverArgs.hasFlag(options::OPT_fuse_init_array,
                         options::OPT_fno_use_init_array, true))
    CC1Args.push_back("-fuse-init-array");
}

PS4CPU::PS4CPU(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  if (Args.hasArg(options::OPT_static))
    D.Diag(diag::err_drv_unsupported_opt_for_target) << "-static" << "PS4";

  // Determine where to find the PS4 libraries. We use SCE_PS4_SDK_DIR
  // if it exists; otherwise use the driver's installation path, which
  // should be <SDK_DIR>/host_tools/bin.

  SmallString<512> PS4SDKDir;
  if (const char *EnvValue = getenv("SCE_PS4_SDK_DIR")) {
    if (!llvm::sys::fs::exists(EnvValue))
      getDriver().Diag(clang::diag::warn_drv_ps4_sdk_dir) << EnvValue;
    PS4SDKDir = EnvValue;
  } else {
    PS4SDKDir = getDriver().Dir;
    llvm::sys::path::append(PS4SDKDir, "/../../");
  }

  // By default, the driver won't report a warning if it can't find
  // PS4's include or lib directories. This behavior could be changed if
  // -Weverything or -Winvalid-or-nonexistent-directory options are passed.
  // If -isysroot was passed, use that as the SDK base path.
  std::string PrefixDir;
  if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
    PrefixDir = A->getValue();
    if (!llvm::sys::fs::exists(PrefixDir))
      getDriver().Diag(clang::diag::warn_missing_sysroot) << PrefixDir;
  } else
    PrefixDir = PS4SDKDir.str();

  SmallString<512> PS4SDKIncludeDir(PrefixDir);
  llvm::sys::path::append(PS4SDKIncludeDir, "target/include");
  if (!Args.hasArg(options::OPT_nostdinc) &&
      !Args.hasArg(options::OPT_nostdlibinc) &&
      !Args.hasArg(options::OPT_isysroot) &&
      !Args.hasArg(options::OPT__sysroot_EQ) &&
      !llvm::sys::fs::exists(PS4SDKIncludeDir)) {
    getDriver().Diag(clang::diag::warn_drv_unable_to_find_directory_expected)
        << "PS4 system headers" << PS4SDKIncludeDir;
  }

  SmallString<512> PS4SDKLibDir(PS4SDKDir);
  llvm::sys::path::append(PS4SDKLibDir, "target/lib");
  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs) &&
      !Args.hasArg(options::OPT__sysroot_EQ) && !Args.hasArg(options::OPT_E) &&
      !Args.hasArg(options::OPT_c) && !Args.hasArg(options::OPT_S) &&
      !Args.hasArg(options::OPT_emit_ast) &&
      !llvm::sys::fs::exists(PS4SDKLibDir)) {
    getDriver().Diag(clang::diag::warn_drv_unable_to_find_directory_expected)
        << "PS4 system libraries" << PS4SDKLibDir;
    return;
  }
  getFilePaths().push_back(PS4SDKLibDir.str());
}

Tool *PS4CPU::buildAssembler() const {
  return new tools::PS4cpu::Assemble(*this);
}

Tool *PS4CPU::buildLinker() const { return new tools::PS4cpu::Link(*this); }

bool PS4CPU::isPICDefault() const { return true; }

bool PS4CPU::HasNativeLLVMSupport() const { return true; }

SanitizerMask PS4CPU::getSupportedSanitizers() const {
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::Vptr;
  return Res;
}
