//===--- ToolChains.cpp - ToolChain Implementations -----------------------===//
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/Program.h"

// FIXME: This needs to be listed last until we fix the broken include guards
// in these files and the LLVM config.h files.
#include "clang/Config/config.h" // for GCC_INSTALL_PREFIX

#include <cstdlib> // ::getenv

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

/// Darwin - Darwin tool chain for i386 and x86_64.

Darwin::Darwin(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : ToolChain(D, Triple, Args), TargetInitialized(false)
{
  // Compute the initial Darwin version from the triple
  unsigned Major, Minor, Micro;
  if (!Triple.getMacOSXVersion(Major, Minor, Micro))
    getDriver().Diag(diag::err_drv_invalid_darwin_version) <<
      Triple.getOSName();
  llvm::raw_string_ostream(MacosxVersionMin)
    << Major << '.' << Minor << '.' << Micro;

  // FIXME: DarwinVersion is only used to find GCC's libexec directory.
  // It should be removed when we stop supporting that.
  DarwinVersion[0] = Minor + 4;
  DarwinVersion[1] = Micro;
  DarwinVersion[2] = 0;

  // Compute the initial iOS version from the triple
  Triple.getiOSVersion(Major, Minor, Micro);
  llvm::raw_string_ostream(iOSVersionMin)
    << Major << '.' << Minor << '.' << Micro;
}

types::ID Darwin::LookupTypeForExtension(const char *Ext) const {
  types::ID Ty = types::lookupTypeForExtension(Ext);

  // Darwin always preprocesses assembly files (unless -x is used explicitly).
  if (Ty == types::TY_PP_Asm)
    return types::TY_Asm;

  return Ty;
}

bool Darwin::HasNativeLLVMSupport() const {
  return true;
}

/// Darwin provides an ARC runtime starting in MacOS X 10.7 and iOS 5.0.
ObjCRuntime Darwin::getDefaultObjCRuntime(bool isNonFragile) const {
  if (isTargetIPhoneOS())
    return ObjCRuntime(ObjCRuntime::iOS, TargetVersion);
  if (isNonFragile)
    return ObjCRuntime(ObjCRuntime::MacOSX, TargetVersion);
  return ObjCRuntime(ObjCRuntime::FragileMacOSX, TargetVersion);
}

/// Darwin provides a blocks runtime starting in MacOS X 10.6 and iOS 3.2.
bool Darwin::hasBlocksRuntime() const {
  if (isTargetIPhoneOS())
    return !isIPhoneOSVersionLT(3, 2);
  else
    return !isMacosxVersionLT(10, 6);
}

static const char *GetArmArchForMArch(StringRef Value) {
  return llvm::StringSwitch<const char*>(Value)
    .Case("armv6k", "armv6")
    .Case("armv6m", "armv6m")
    .Case("armv5tej", "armv5")
    .Case("xscale", "xscale")
    .Case("armv4t", "armv4t")
    .Case("armv7", "armv7")
    .Cases("armv7a", "armv7-a", "armv7")
    .Cases("armv7r", "armv7-r", "armv7")
    .Cases("armv7em", "armv7e-m", "armv7em")
    .Cases("armv7f", "armv7-f", "armv7f")
    .Cases("armv7k", "armv7-k", "armv7k")
    .Cases("armv7m", "armv7-m", "armv7m")
    .Cases("armv7s", "armv7-s", "armv7s")
    .Default(0);
}

static const char *GetArmArchForMCpu(StringRef Value) {
  return llvm::StringSwitch<const char *>(Value)
    .Cases("arm9e", "arm946e-s", "arm966e-s", "arm968e-s", "arm926ej-s","armv5")
    .Cases("arm10e", "arm10tdmi", "armv5")
    .Cases("arm1020t", "arm1020e", "arm1022e", "arm1026ej-s", "armv5")
    .Case("xscale", "xscale")
    .Cases("arm1136j-s", "arm1136jf-s", "arm1176jz-s", "arm1176jzf-s", "armv6")
    .Case("cortex-m0", "armv6m")
    .Cases("cortex-a5", "cortex-a7", "cortex-a8", "armv7")
    .Cases("cortex-a9", "cortex-a12", "cortex-a15", "armv7")
    .Cases("cortex-r4", "cortex-r5", "armv7r")
    .Case("cortex-a9-mp", "armv7f")
    .Case("cortex-m3", "armv7m")
    .Case("cortex-m4", "armv7em")
    .Case("swift", "armv7s")
    .Default(0);
}

StringRef Darwin::getDarwinArchName(const ArgList &Args) const {
  switch (getTriple().getArch()) {
  default:
    return getArchName();

  case llvm::Triple::thumb:
  case llvm::Triple::arm: {
    if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
      if (const char *Arch = GetArmArchForMArch(A->getValue()))
        return Arch;

    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      if (const char *Arch = GetArmArchForMCpu(A->getValue()))
        return Arch;

    return "arm";
  }
  }
}

Darwin::~Darwin() {
}

std::string Darwin::ComputeEffectiveClangTriple(const ArgList &Args,
                                                types::ID InputType) const {
  llvm::Triple Triple(ComputeLLVMTriple(Args, InputType));

  // If the target isn't initialized (e.g., an unknown Darwin platform, return
  // the default triple).
  if (!isTargetInitialized())
    return Triple.getTriple();

  if (Triple.getArchName() == "thumbv6m" ||
      Triple.getArchName() == "thumbv7m" ||
      Triple.getArchName() == "thumbv7em") {
    // OS is ios or macosx unless it's the v6m or v7m.
    Triple.setOS(llvm::Triple::Darwin);
    Triple.setEnvironment(llvm::Triple::EABI);
  } else {
    SmallString<16> Str;
    Str += isTargetIPhoneOS() ? "ios" : "macosx";
    Str += getTargetVersion().getAsString();
    Triple.setOSName(Str);
  }

  return Triple.getTriple();
}

void Generic_ELF::anchor() {}

Tool *Darwin::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::LipoJobClass:
    if (!Lipo)
      Lipo.reset(new tools::darwin::Lipo(*this));
    return Lipo.get();
  case Action::DsymutilJobClass:
    if (!Dsymutil)
      Dsymutil.reset(new tools::darwin::Dsymutil(*this));
    return Dsymutil.get();
  case Action::VerifyJobClass:
    if (!VerifyDebug)
      VerifyDebug.reset(new tools::darwin::VerifyDebug(*this));
    return VerifyDebug.get();
  default:
    return ToolChain::getTool(AC);
  }
}

Tool *Darwin::buildLinker() const {
  return new tools::darwin::Link(*this);
}

Tool *Darwin::buildAssembler() const {
  return new tools::darwin::Assemble(*this);
}

DarwinClang::DarwinClang(const Driver &D, const llvm::Triple& Triple,
                         const ArgList &Args)
  : Darwin(D, Triple, Args)
{
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  // We expect 'as', 'ld', etc. to be adjacent to our install dir.
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);
}

void DarwinClang::AddLinkARCArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {

  CmdArgs.push_back("-force_load");
  SmallString<128> P(getDriver().ClangExecutable);
  llvm::sys::path::remove_filename(P); // 'clang'
  llvm::sys::path::remove_filename(P); // 'bin'
  llvm::sys::path::append(P, "lib", "arc", "libarclite_");
  // Mash in the platform.
  if (isTargetIOSSimulator())
    P += "iphonesimulator";
  else if (isTargetIPhoneOS())
    P += "iphoneos";
  else
    P += "macosx";
  P += ".a";

  CmdArgs.push_back(Args.MakeArgString(P));
}

void DarwinClang::AddLinkRuntimeLib(const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    const char *DarwinStaticLib,
                                    bool AlwaysLink) const {
  SmallString<128> P(getDriver().ResourceDir);
  llvm::sys::path::append(P, "lib", "darwin", DarwinStaticLib);

  // For now, allow missing resource libraries to support developers who may
  // not have compiler-rt checked out or integrated into their build (unless
  // we explicitly force linking with this library).
  if (AlwaysLink || llvm::sys::fs::exists(P.str()))
    CmdArgs.push_back(Args.MakeArgString(P.str()));
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
    getDriver().Diag(diag::err_drv_unsupported_opt)
      << A->getAsString(Args);
    return;
  }

  // If we are building profile support, link that library in.
  if (Args.hasArg(options::OPT_fprofile_arcs) ||
      Args.hasArg(options::OPT_fprofile_generate) ||
      Args.hasArg(options::OPT_fcreate_profile) ||
      Args.hasArg(options::OPT_coverage)) {
    // Select the appropriate runtime library for the target.
    if (isTargetIPhoneOS()) {
      AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.profile_ios.a");
    } else {
      AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.profile_osx.a");
    }
  }

  const SanitizerArgs &Sanitize = getSanitizerArgs();

  // Add Ubsan runtime library, if required.
  if (Sanitize.needsUbsanRt()) {
    // FIXME: Move this check to SanitizerArgs::filterUnsupportedKinds.
    if (isTargetIPhoneOS()) {
      getDriver().Diag(diag::err_drv_clang_unsupported_per_platform)
        << "-fsanitize=undefined";
    } else {
      AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.ubsan_osx.a", true);

      // The Ubsan runtime library requires C++.
      AddCXXStdlibLibArgs(Args, CmdArgs);
    }
  }

  // Add ASAN runtime library, if required. Dynamic libraries and bundles
  // should not be linked with the runtime library.
  if (Sanitize.needsAsanRt()) {
    // FIXME: Move this check to SanitizerArgs::filterUnsupportedKinds.
    if (isTargetIPhoneOS() && !isTargetIOSSimulator()) {
      getDriver().Diag(diag::err_drv_clang_unsupported_per_platform)
        << "-fsanitize=address";
    } else {
      if (!Args.hasArg(options::OPT_dynamiclib) &&
          !Args.hasArg(options::OPT_bundle)) {
        // The ASAN runtime library requires C++.
        AddCXXStdlibLibArgs(Args, CmdArgs);
      }
      if (isTargetMacOS()) {
        AddLinkRuntimeLib(Args, CmdArgs,
                          "libclang_rt.asan_osx_dynamic.dylib",
                          true);
      } else {
        if (isTargetIOSSimulator()) {
          AddLinkRuntimeLib(Args, CmdArgs,
                            "libclang_rt.asan_iossim_dynamic.dylib",
                            true);
        }
      }
    }
  }

  // Otherwise link libSystem, then the dynamic runtime library, and finally any
  // target specific static runtime library.
  CmdArgs.push_back("-lSystem");

  // Select the dynamic runtime library and the target specific static library.
  if (isTargetIPhoneOS()) {
    // If we are compiling as iOS / simulator, don't attempt to link libgcc_s.1,
    // it never went into the SDK.
    // Linking against libgcc_s.1 isn't needed for iOS 5.0+
    if (isIPhoneOSVersionLT(5, 0) && !isTargetIOSSimulator())
      CmdArgs.push_back("-lgcc_s.1");

    // We currently always need a static runtime library for iOS.
    AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.ios.a");
  } else {
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
    if (!llvm::sys::fs::exists(A->getValue()))
      getDriver().Diag(clang::diag::warn_missing_sysroot) << A->getValue();
  } else {
    if (char *env = ::getenv("SDKROOT")) {
      // We only use this value as the default if it is an absolute path,
      // exists, and it is not the root path.
      if (llvm::sys::path::is_absolute(env) && llvm::sys::fs::exists(env) &&
          StringRef(env) != "/") {
        Args.append(Args.MakeSeparateArg(
                      0, Opts.getOption(options::OPT_isysroot), env));
      }
    }
  }

  Arg *OSXVersion = Args.getLastArg(options::OPT_mmacosx_version_min_EQ);
  Arg *iOSVersion = Args.getLastArg(options::OPT_miphoneos_version_min_EQ);
  Arg *iOSSimVersion = Args.getLastArg(
    options::OPT_mios_simulator_version_min_EQ);

  if (OSXVersion && (iOSVersion || iOSSimVersion)) {
    getDriver().Diag(diag::err_drv_argument_not_allowed_with)
          << OSXVersion->getAsString(Args)
          << (iOSVersion ? iOSVersion : iOSSimVersion)->getAsString(Args);
    iOSVersion = iOSSimVersion = 0;
  } else if (iOSVersion && iOSSimVersion) {
    getDriver().Diag(diag::err_drv_argument_not_allowed_with)
          << iOSVersion->getAsString(Args)
          << iOSSimVersion->getAsString(Args);
    iOSSimVersion = 0;
  } else if (!OSXVersion && !iOSVersion && !iOSSimVersion) {
    // If no deployment target was specified on the command line, check for
    // environment defines.
    StringRef OSXTarget;
    StringRef iOSTarget;
    StringRef iOSSimTarget;
    if (char *env = ::getenv("MACOSX_DEPLOYMENT_TARGET"))
      OSXTarget = env;
    if (char *env = ::getenv("IPHONEOS_DEPLOYMENT_TARGET"))
      iOSTarget = env;
    if (char *env = ::getenv("IOS_SIMULATOR_DEPLOYMENT_TARGET"))
      iOSSimTarget = env;

    // If no '-miphoneos-version-min' specified on the command line and
    // IPHONEOS_DEPLOYMENT_TARGET is not defined, see if we can set the default
    // based on -isysroot.
    if (iOSTarget.empty()) {
      if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
        StringRef first, second;
        StringRef isysroot = A->getValue();
        llvm::tie(first, second) = isysroot.split(StringRef("SDKs/iPhoneOS"));
        if (second != "")
          iOSTarget = second.substr(0,3);
      }
    }

    // If no OSX or iOS target has been specified and we're compiling for armv7,
    // go ahead as assume we're targeting iOS.
    if (OSXTarget.empty() && iOSTarget.empty() &&
        (getDarwinArchName(Args) == "armv7" ||
         getDarwinArchName(Args) == "armv7s"))
        iOSTarget = iOSVersionMin;

    // Handle conflicting deployment targets
    //
    // FIXME: Don't hardcode default here.

    // Do not allow conflicts with the iOS simulator target.
    if (!iOSSimTarget.empty() && (!OSXTarget.empty() || !iOSTarget.empty())) {
      getDriver().Diag(diag::err_drv_conflicting_deployment_targets)
        << "IOS_SIMULATOR_DEPLOYMENT_TARGET"
        << (!OSXTarget.empty() ? "MACOSX_DEPLOYMENT_TARGET" :
            "IPHONEOS_DEPLOYMENT_TARGET");
    }

    // Allow conflicts among OSX and iOS for historical reasons, but choose the
    // default platform.
    if (!OSXTarget.empty() && !iOSTarget.empty()) {
      if (getTriple().getArch() == llvm::Triple::arm ||
          getTriple().getArch() == llvm::Triple::thumb)
        OSXTarget = "";
      else
        iOSTarget = "";
    }

    if (!OSXTarget.empty()) {
      const Option O = Opts.getOption(options::OPT_mmacosx_version_min_EQ);
      OSXVersion = Args.MakeJoinedArg(0, O, OSXTarget);
      Args.append(OSXVersion);
    } else if (!iOSTarget.empty()) {
      const Option O = Opts.getOption(options::OPT_miphoneos_version_min_EQ);
      iOSVersion = Args.MakeJoinedArg(0, O, iOSTarget);
      Args.append(iOSVersion);
    } else if (!iOSSimTarget.empty()) {
      const Option O = Opts.getOption(
        options::OPT_mios_simulator_version_min_EQ);
      iOSSimVersion = Args.MakeJoinedArg(0, O, iOSSimTarget);
      Args.append(iOSSimVersion);
    } else {
      // Otherwise, assume we are targeting OS X.
      const Option O = Opts.getOption(options::OPT_mmacosx_version_min_EQ);
      OSXVersion = Args.MakeJoinedArg(0, O, MacosxVersionMin);
      Args.append(OSXVersion);
    }
  }

  // Reject invalid architecture combinations.
  if (iOSSimVersion && (getTriple().getArch() != llvm::Triple::x86 &&
                        getTriple().getArch() != llvm::Triple::x86_64)) {
    getDriver().Diag(diag::err_drv_invalid_arch_for_deployment_target)
      << getTriple().getArchName() << iOSSimVersion->getAsString(Args);
  }

  // Set the tool chain target information.
  unsigned Major, Minor, Micro;
  bool HadExtra;
  if (OSXVersion) {
    assert((!iOSVersion && !iOSSimVersion) && "Unknown target platform!");
    if (!Driver::GetReleaseVersion(OSXVersion->getValue(), Major, Minor,
                                   Micro, HadExtra) || HadExtra ||
        Major != 10 || Minor >= 100 || Micro >= 100)
      getDriver().Diag(diag::err_drv_invalid_version_number)
        << OSXVersion->getAsString(Args);
  } else {
    const Arg *Version = iOSVersion ? iOSVersion : iOSSimVersion;
    assert(Version && "Unknown target platform!");
    if (!Driver::GetReleaseVersion(Version->getValue(), Major, Minor,
                                   Micro, HadExtra) || HadExtra ||
        Major >= 10 || Minor >= 100 || Micro >= 100)
      getDriver().Diag(diag::err_drv_invalid_version_number)
        << Version->getAsString(Args);
  }

  bool IsIOSSim = bool(iOSSimVersion);

  // In GCC, the simulator historically was treated as being OS X in some
  // contexts, like determining the link logic, despite generally being called
  // with an iOS deployment target. For compatibility, we detect the
  // simulator as iOS + x86, and treat it differently in a few contexts.
  if (iOSVersion && (getTriple().getArch() == llvm::Triple::x86 ||
                     getTriple().getArch() == llvm::Triple::x86_64))
    IsIOSSim = true;

  setTarget(/*IsIPhoneOS=*/ !OSXVersion, Major, Minor, Micro, IsIOSSim);
}

void DarwinClang::AddCXXStdlibLibArgs(const ArgList &Args,
                                      ArgStringList &CmdArgs) const {
  CXXStdlibType Type = GetCXXStdlibType(Args);

  switch (Type) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    break;

  case ToolChain::CST_Libstdcxx: {
    // Unfortunately, -lstdc++ doesn't always exist in the standard search path;
    // it was previously found in the gcc lib dir. However, for all the Darwin
    // platforms we care about it was -lstdc++.6, so we search for that
    // explicitly if we can't see an obvious -lstdc++ candidate.

    // Check in the sysroot first.
    if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
      SmallString<128> P(A->getValue());
      llvm::sys::path::append(P, "usr", "lib", "libstdc++.dylib");

      if (!llvm::sys::fs::exists(P.str())) {
        llvm::sys::path::remove_filename(P);
        llvm::sys::path::append(P, "libstdc++.6.dylib");
        if (llvm::sys::fs::exists(P.str())) {
          CmdArgs.push_back(Args.MakeArgString(P.str()));
          return;
        }
      }
    }

    // Otherwise, look in the root.
    // FIXME: This should be removed someday when we don't have to care about
    // 10.6 and earlier, where /usr/lib/libstdc++.dylib does not exist.
    if (!llvm::sys::fs::exists("/usr/lib/libstdc++.dylib") &&
        llvm::sys::fs::exists("/usr/lib/libstdc++.6.dylib")) {
      CmdArgs.push_back("/usr/lib/libstdc++.6.dylib");
      return;
    }

    // Otherwise, let the linker search.
    CmdArgs.push_back("-lstdc++");
    break;
  }
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
  if (!isTargetIPhoneOS() || isTargetIOSSimulator() ||
      !isIPhoneOSVersionLT(6, 0)) {
    llvm::sys::path::append(P, "libclang_rt.cc_kext.a");
  } else {
    llvm::sys::path::append(P, "libclang_rt.cc_kext_ios5.a");
  }

  // For now, allow missing resource libraries to support developers who may
  // not have compiler-rt checked out or integrated into their build.
  if (llvm::sys::fs::exists(P.str()))
    CmdArgs.push_back(Args.MakeArgString(P.str()));
}

DerivedArgList *Darwin::TranslateArgs(const DerivedArgList &Args,
                                      const char *BoundArch) const {
  DerivedArgList *DAL = new DerivedArgList(Args.getBaseArgs());
  const OptTable &Opts = getDriver().getOpts();

  // FIXME: We really want to get out of the tool chain level argument
  // translation business, as it makes the driver functionality much
  // more opaque. For now, we follow gcc closely solely for the
  // purpose of easily achieving feature parity & testability. Once we
  // have something that works, we should reevaluate each translation
  // and try to push it down into tool specific logic.

  for (ArgList::const_iterator it = Args.begin(),
         ie = Args.end(); it != ie; ++it) {
    Arg *A = *it;

    if (A->getOption().matches(options::OPT_Xarch__)) {
      // Skip this argument unless the architecture matches either the toolchain
      // triple arch, or the arch being bound.
      llvm::Triple::ArchType XarchArch =
        tools::darwin::getArchTypeForDarwinArchName(A->getValue(0));
      if (!(XarchArch == getArch()  ||
            (BoundArch && XarchArch ==
             tools::darwin::getArchTypeForDarwinArchName(BoundArch))))
        continue;

      Arg *OriginalArg = A;
      unsigned Index = Args.getBaseArgs().MakeIndex(A->getValue(1));
      unsigned Prev = Index;
      Arg *XarchArg = Opts.ParseOneArg(Args, Index);

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
      A = XarchArg;

      DAL->AddSynthesizedArg(A);

      // Linker input arguments require custom handling. The problem is that we
      // have already constructed the phase actions, so we can not treat them as
      // "input arguments".
      if (A->getOption().hasFlag(options::LinkerInput)) {
        // Convert the argument into individual Zlinker_input_args.
        for (unsigned i = 0, e = A->getNumValues(); i != e; ++i) {
          DAL->AddSeparateArg(OriginalArg,
                              Opts.getOption(options::OPT_Zlinker_input),
                              A->getValue(i));

        }
        continue;
      }
    }

    // Sob. These is strictly gcc compatible for the time being. Apple
    // gcc translates options twice, which means that self-expanding
    // options add duplicates.
    switch ((options::ID) A->getOption().getID()) {
    default:
      DAL->append(A);
      break;

    case options::OPT_mkernel:
    case options::OPT_fapple_kext:
      DAL->append(A);
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_static));
      break;

    case options::OPT_dependency_file:
      DAL->AddSeparateArg(A, Opts.getOption(options::OPT_MF),
                          A->getValue());
      break;

    case options::OPT_gfull:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_g_Flag));
      DAL->AddFlagArg(A,
               Opts.getOption(options::OPT_fno_eliminate_unused_debug_symbols));
      break;

    case options::OPT_gused:
      DAL->AddFlagArg(A, Opts.getOption(options::OPT_g_Flag));
      DAL->AddFlagArg(A,
             Opts.getOption(options::OPT_feliminate_unused_debug_symbols));
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
      DAL->AddFlagArg(A,
                   Opts.getOption(options::OPT_mno_warn_nonportable_cfstrings));
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
      DAL->AddJoinedArg(0, Opts.getOption(options::OPT_mtune_EQ), "core2");

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
      DAL->AddJoinedArg(0, MCpu, "601");
    else if (Name == "ppc603")
      DAL->AddJoinedArg(0, MCpu, "603");
    else if (Name == "ppc604")
      DAL->AddJoinedArg(0, MCpu, "604");
    else if (Name == "ppc604e")
      DAL->AddJoinedArg(0, MCpu, "604e");
    else if (Name == "ppc750")
      DAL->AddJoinedArg(0, MCpu, "750");
    else if (Name == "ppc7400")
      DAL->AddJoinedArg(0, MCpu, "7400");
    else if (Name == "ppc7450")
      DAL->AddJoinedArg(0, MCpu, "7450");
    else if (Name == "ppc970")
      DAL->AddJoinedArg(0, MCpu, "970");

    else if (Name == "ppc64" || Name == "ppc64le")
      DAL->AddFlagArg(0, Opts.getOption(options::OPT_m64));

    else if (Name == "i386")
      ;
    else if (Name == "i486")
      DAL->AddJoinedArg(0, MArch, "i486");
    else if (Name == "i586")
      DAL->AddJoinedArg(0, MArch, "i586");
    else if (Name == "i686")
      DAL->AddJoinedArg(0, MArch, "i686");
    else if (Name == "pentium")
      DAL->AddJoinedArg(0, MArch, "pentium");
    else if (Name == "pentium2")
      DAL->AddJoinedArg(0, MArch, "pentium2");
    else if (Name == "pentpro")
      DAL->AddJoinedArg(0, MArch, "pentiumpro");
    else if (Name == "pentIIm3")
      DAL->AddJoinedArg(0, MArch, "pentium2");

    else if (Name == "x86_64")
      DAL->AddFlagArg(0, Opts.getOption(options::OPT_m64));
    else if (Name == "x86_64h") {
      DAL->AddFlagArg(0, Opts.getOption(options::OPT_m64));
      DAL->AddJoinedArg(0, MArch, "x86_64h");
    }

    else if (Name == "arm")
      DAL->AddJoinedArg(0, MArch, "armv4t");
    else if (Name == "armv4t")
      DAL->AddJoinedArg(0, MArch, "armv4t");
    else if (Name == "armv5")
      DAL->AddJoinedArg(0, MArch, "armv5tej");
    else if (Name == "xscale")
      DAL->AddJoinedArg(0, MArch, "xscale");
    else if (Name == "armv6")
      DAL->AddJoinedArg(0, MArch, "armv6k");
    else if (Name == "armv6m")
      DAL->AddJoinedArg(0, MArch, "armv6m");
    else if (Name == "armv7")
      DAL->AddJoinedArg(0, MArch, "armv7a");
    else if (Name == "armv7em")
      DAL->AddJoinedArg(0, MArch, "armv7em");
    else if (Name == "armv7f")
      DAL->AddJoinedArg(0, MArch, "armv7f");
    else if (Name == "armv7k")
      DAL->AddJoinedArg(0, MArch, "armv7k");
    else if (Name == "armv7m")
      DAL->AddJoinedArg(0, MArch, "armv7m");
    else if (Name == "armv7s")
      DAL->AddJoinedArg(0, MArch, "armv7s");

    else
      llvm_unreachable("invalid Darwin arch");
  }

  // Add an explicit version min argument for the deployment target. We do this
  // after argument translation because -Xarch_ arguments may add a version min
  // argument.
  if (BoundArch)
    AddDeploymentTarget(*DAL);

  // For iOS 6, undo the translation to add -static for -mkernel/-fapple-kext.
  // FIXME: It would be far better to avoid inserting those -static arguments,
  // but we can't check the deployment target in the translation code until
  // it is set here.
  if (isTargetIPhoneOS() && !isIPhoneOSVersionLT(6, 0)) {
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
       (isTargetIPhoneOS() && !isIPhoneOSVersionLT(7, 0))) &&
      !Args.getLastArg(options::OPT_stdlib_EQ))
    DAL->AddJoinedArg(0, Opts.getOption(options::OPT_stdlib_EQ), "libc++");

  // Validate the C++ standard library choice.
  CXXStdlibType Type = GetCXXStdlibType(*DAL);
  if (Type == ToolChain::CST_Libcxx) {
    // Check whether the target provides libc++.
    StringRef where;

    // Complain about targetting iOS < 5.0 in any way.
    if (isTargetIPhoneOS() && isIPhoneOSVersionLT(5, 0))
      where = "iOS 5.0";

    if (where != StringRef()) {
      getDriver().Diag(clang::diag::err_drv_invalid_libcxx_deployment)
        << where;
    }
  }

  return DAL;
}

bool Darwin::IsUnwindTablesDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool Darwin::UseDwarfDebugFlags() const {
  if (const char *S = ::getenv("RC_DEBUG_OPTIONS"))
    return S[0] != '\0';
  return false;
}

bool Darwin::UseSjLjExceptions() const {
  // Darwin uses SjLj exceptions on ARM.
  return (getTriple().getArch() == llvm::Triple::arm ||
          getTriple().getArch() == llvm::Triple::thumb);
}

bool Darwin::isPICDefault() const {
  return true;
}

bool Darwin::isPIEDefault() const {
  return false;
}

bool Darwin::isPICDefaultForced() const {
  return getArch() == llvm::Triple::x86_64;
}

bool Darwin::SupportsProfiling() const {
  // Profiling instrumentation is only supported on x86.
  return getArch() == llvm::Triple::x86 || getArch() == llvm::Triple::x86_64;
}

bool Darwin::SupportsObjCGC() const {
  // Garbage collection is supported everywhere except on iPhone OS.
  return !isTargetIPhoneOS();
}

void Darwin::CheckObjCARC() const {
  if (isTargetIPhoneOS() || !isMacosxVersionLT(10, 6))
    return;
  getDriver().Diag(diag::err_arc_unsupported_on_toolchain);
}

std::string
Darwin_Generic_GCC::ComputeEffectiveClangTriple(const ArgList &Args,
                                                types::ID InputType) const {
  return ComputeLLVMTriple(Args, InputType);
}

/// Generic_GCC - A tool chain using the 'gcc' command to perform
/// all subcommands; this relies on gcc translating the majority of
/// command line options.

/// \brief Parse a GCCVersion object out of a string of text.
///
/// This is the primary means of forming GCCVersion objects.
/*static*/
Generic_GCC::GCCVersion Linux::GCCVersion::Parse(StringRef VersionText) {
  const GCCVersion BadVersion = { VersionText.str(), -1, -1, -1, "", "", "" };
  std::pair<StringRef, StringRef> First = VersionText.split('.');
  std::pair<StringRef, StringRef> Second = First.second.split('.');

  GCCVersion GoodVersion = { VersionText.str(), -1, -1, -1, "", "", "" };
  if (First.first.getAsInteger(10, GoodVersion.Major) ||
      GoodVersion.Major < 0)
    return BadVersion;
  GoodVersion.MajorStr = First.first.str();
  if (Second.first.getAsInteger(10, GoodVersion.Minor) ||
      GoodVersion.Minor < 0)
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

static StringRef getGCCToolchainDir(const ArgList &Args) {
  const Arg *A = Args.getLastArg(options::OPT_gcc_toolchain);
  if (A)
    return A->getValue();
  return GCC_INSTALL_PREFIX;
}

/// \brief Construct a GCCInstallationDetector from the driver.
///
/// This performs all of the autodetection and sets up the various paths.
/// Once constructed, a GCCInstallationDetector is essentially immutable.
///
/// FIXME: We shouldn't need an explicit TargetTriple parameter here, and
/// should instead pull the target out of the driver. This is currently
/// necessary because the driver doesn't store the final version of the target
/// triple.
Generic_GCC::GCCInstallationDetector::GCCInstallationDetector(
    const Driver &D, const llvm::Triple &TargetTriple, const ArgList &Args)
    : IsValid(false), D(D) {
  llvm::Triple BiarchVariantTriple =
      TargetTriple.isArch32Bit() ? TargetTriple.get64BitArchVariant()
                                 : TargetTriple.get32BitArchVariant();
  llvm::Triple::ArchType TargetArch = TargetTriple.getArch();
  // The library directories which may contain GCC installations.
  SmallVector<StringRef, 4> CandidateLibDirs, CandidateBiarchLibDirs;
  // The compatible GCC triples for this particular architecture.
  SmallVector<StringRef, 10> CandidateTripleAliases;
  SmallVector<StringRef, 10> CandidateBiarchTripleAliases;
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
  for (unsigned i = 0, ie = Prefixes.size(); i < ie; ++i) {
    if (!llvm::sys::fs::exists(Prefixes[i]))
      continue;
    for (unsigned j = 0, je = CandidateLibDirs.size(); j < je; ++j) {
      const std::string LibDir = Prefixes[i] + CandidateLibDirs[j].str();
      if (!llvm::sys::fs::exists(LibDir))
        continue;
      for (unsigned k = 0, ke = CandidateTripleAliases.size(); k < ke; ++k)
        ScanLibDirForGCCTriple(TargetArch, Args, LibDir,
                               CandidateTripleAliases[k]);
    }
    for (unsigned j = 0, je = CandidateBiarchLibDirs.size(); j < je; ++j) {
      const std::string LibDir = Prefixes[i] + CandidateBiarchLibDirs[j].str();
      if (!llvm::sys::fs::exists(LibDir))
        continue;
      for (unsigned k = 0, ke = CandidateBiarchTripleAliases.size(); k < ke;
           ++k)
        ScanLibDirForGCCTriple(TargetArch, Args, LibDir,
                               CandidateBiarchTripleAliases[k],
                               /*NeedsBiarchSuffix=*/ true);
    }
  }
}

void Generic_GCC::GCCInstallationDetector::print(raw_ostream &OS) const {
  for (std::set<std::string>::const_iterator
           I = CandidateGCCInstallPaths.begin(),
           E = CandidateGCCInstallPaths.end();
       I != E; ++I)
    OS << "Found candidate GCC installation: " << *I << "\n";

  OS << "Selected GCC installation: " << GCCInstallPath << "\n";
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
  static const char *const AArch64LibDirs[] = { "/lib" };
  static const char *const AArch64Triples[] = { "aarch64-none-linux-gnu",
                                                "aarch64-linux-gnu" };

  static const char *const ARMLibDirs[] = { "/lib" };
  static const char *const ARMTriples[] = { "arm-linux-gnueabi",
                                            "arm-linux-androideabi" };
  static const char *const ARMHFTriples[] = { "arm-linux-gnueabihf",
                                              "armv7hl-redhat-linux-gnueabi" };

  static const char *const X86_64LibDirs[] = { "/lib64", "/lib" };
  static const char *const X86_64Triples[] = {
    "x86_64-linux-gnu", "x86_64-unknown-linux-gnu", "x86_64-pc-linux-gnu",
    "x86_64-redhat-linux6E", "x86_64-redhat-linux", "x86_64-suse-linux",
    "x86_64-manbo-linux-gnu", "x86_64-linux-gnu", "x86_64-slackware-linux"
  };
  static const char *const X86LibDirs[] = { "/lib32", "/lib" };
  static const char *const X86Triples[] = {
    "i686-linux-gnu", "i686-pc-linux-gnu", "i486-linux-gnu", "i386-linux-gnu",
    "i386-redhat-linux6E", "i686-redhat-linux", "i586-redhat-linux",
    "i386-redhat-linux", "i586-suse-linux", "i486-slackware-linux",
    "i686-montavista-linux"
  };

  static const char *const MIPSLibDirs[] = { "/lib" };
  static const char *const MIPSTriples[] = { "mips-linux-gnu",
                                             "mips-mti-linux-gnu" };
  static const char *const MIPSELLibDirs[] = { "/lib" };
  static const char *const MIPSELTriples[] = { "mipsel-linux-gnu",
                                               "mipsel-linux-android" };

  static const char *const MIPS64LibDirs[] = { "/lib64", "/lib" };
  static const char *const MIPS64Triples[] = { "mips64-linux-gnu",
                                               "mips-mti-linux-gnu" };
  static const char *const MIPS64ELLibDirs[] = { "/lib64", "/lib" };
  static const char *const MIPS64ELTriples[] = { "mips64el-linux-gnu",
                                                 "mips-mti-linux-gnu" };

  static const char *const PPCLibDirs[] = { "/lib32", "/lib" };
  static const char *const PPCTriples[] = {
    "powerpc-linux-gnu", "powerpc-unknown-linux-gnu", "powerpc-linux-gnuspe",
    "powerpc-suse-linux", "powerpc-montavista-linuxspe"
  };
  static const char *const PPC64LibDirs[] = { "/lib64", "/lib" };
  static const char *const PPC64Triples[] = { "powerpc64-linux-gnu",
                                              "powerpc64-unknown-linux-gnu",
                                              "powerpc64-suse-linux",
                                              "ppc64-redhat-linux" };
  static const char *const PPC64LELibDirs[] = { "/lib64", "/lib" };
  static const char *const PPC64LETriples[] = { "powerpc64le-linux-gnu",
                                                "powerpc64le-unknown-linux-gnu",
                                                "powerpc64le-suse-linux",
                                                "ppc64le-redhat-linux" };

  static const char *const SystemZLibDirs[] = { "/lib64", "/lib" };
  static const char *const SystemZTriples[] = {
    "s390x-linux-gnu", "s390x-unknown-linux-gnu", "s390x-ibm-linux-gnu",
    "s390x-suse-linux", "s390x-redhat-linux"
  };

  switch (TargetTriple.getArch()) {
  case llvm::Triple::aarch64:
    LibDirs.append(AArch64LibDirs,
                   AArch64LibDirs + llvm::array_lengthof(AArch64LibDirs));
    TripleAliases.append(AArch64Triples,
                         AArch64Triples + llvm::array_lengthof(AArch64Triples));
    BiarchLibDirs.append(AArch64LibDirs,
                         AArch64LibDirs + llvm::array_lengthof(AArch64LibDirs));
    BiarchTripleAliases.append(
        AArch64Triples, AArch64Triples + llvm::array_lengthof(AArch64Triples));
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    LibDirs.append(ARMLibDirs, ARMLibDirs + llvm::array_lengthof(ARMLibDirs));
    if (TargetTriple.getEnvironment() == llvm::Triple::GNUEABIHF) {
      TripleAliases.append(ARMHFTriples,
                           ARMHFTriples + llvm::array_lengthof(ARMHFTriples));
    } else {
      TripleAliases.append(ARMTriples,
                           ARMTriples + llvm::array_lengthof(ARMTriples));
    }
    break;
  case llvm::Triple::x86_64:
    LibDirs.append(X86_64LibDirs,
                   X86_64LibDirs + llvm::array_lengthof(X86_64LibDirs));
    TripleAliases.append(X86_64Triples,
                         X86_64Triples + llvm::array_lengthof(X86_64Triples));
    BiarchLibDirs.append(X86LibDirs,
                         X86LibDirs + llvm::array_lengthof(X86LibDirs));
    BiarchTripleAliases.append(X86Triples,
                               X86Triples + llvm::array_lengthof(X86Triples));
    break;
  case llvm::Triple::x86:
    LibDirs.append(X86LibDirs, X86LibDirs + llvm::array_lengthof(X86LibDirs));
    TripleAliases.append(X86Triples,
                         X86Triples + llvm::array_lengthof(X86Triples));
    BiarchLibDirs.append(X86_64LibDirs,
                         X86_64LibDirs + llvm::array_lengthof(X86_64LibDirs));
    BiarchTripleAliases.append(
        X86_64Triples, X86_64Triples + llvm::array_lengthof(X86_64Triples));
    break;
  case llvm::Triple::mips:
    LibDirs.append(MIPSLibDirs,
                   MIPSLibDirs + llvm::array_lengthof(MIPSLibDirs));
    TripleAliases.append(MIPSTriples,
                         MIPSTriples + llvm::array_lengthof(MIPSTriples));
    BiarchLibDirs.append(MIPS64LibDirs,
                         MIPS64LibDirs + llvm::array_lengthof(MIPS64LibDirs));
    BiarchTripleAliases.append(
        MIPS64Triples, MIPS64Triples + llvm::array_lengthof(MIPS64Triples));
    break;
  case llvm::Triple::mipsel:
    LibDirs.append(MIPSELLibDirs,
                   MIPSELLibDirs + llvm::array_lengthof(MIPSELLibDirs));
    TripleAliases.append(MIPSELTriples,
                         MIPSELTriples + llvm::array_lengthof(MIPSELTriples));
    TripleAliases.append(MIPSTriples,
                         MIPSTriples + llvm::array_lengthof(MIPSTriples));
    BiarchLibDirs.append(
        MIPS64ELLibDirs,
        MIPS64ELLibDirs + llvm::array_lengthof(MIPS64ELLibDirs));
    BiarchTripleAliases.append(
        MIPS64ELTriples,
        MIPS64ELTriples + llvm::array_lengthof(MIPS64ELTriples));
    break;
  case llvm::Triple::mips64:
    LibDirs.append(MIPS64LibDirs,
                   MIPS64LibDirs + llvm::array_lengthof(MIPS64LibDirs));
    TripleAliases.append(MIPS64Triples,
                         MIPS64Triples + llvm::array_lengthof(MIPS64Triples));
    BiarchLibDirs.append(MIPSLibDirs,
                         MIPSLibDirs + llvm::array_lengthof(MIPSLibDirs));
    BiarchTripleAliases.append(MIPSTriples,
                               MIPSTriples + llvm::array_lengthof(MIPSTriples));
    break;
  case llvm::Triple::mips64el:
    LibDirs.append(MIPS64ELLibDirs,
                   MIPS64ELLibDirs + llvm::array_lengthof(MIPS64ELLibDirs));
    TripleAliases.append(
        MIPS64ELTriples,
        MIPS64ELTriples + llvm::array_lengthof(MIPS64ELTriples));
    BiarchLibDirs.append(MIPSELLibDirs,
                         MIPSELLibDirs + llvm::array_lengthof(MIPSELLibDirs));
    BiarchTripleAliases.append(
        MIPSELTriples, MIPSELTriples + llvm::array_lengthof(MIPSELTriples));
    BiarchTripleAliases.append(
        MIPSTriples, MIPSTriples + llvm::array_lengthof(MIPSTriples));
    break;
  case llvm::Triple::ppc:
    LibDirs.append(PPCLibDirs, PPCLibDirs + llvm::array_lengthof(PPCLibDirs));
    TripleAliases.append(PPCTriples,
                         PPCTriples + llvm::array_lengthof(PPCTriples));
    BiarchLibDirs.append(PPC64LibDirs,
                         PPC64LibDirs + llvm::array_lengthof(PPC64LibDirs));
    BiarchTripleAliases.append(
        PPC64Triples, PPC64Triples + llvm::array_lengthof(PPC64Triples));
    break;
  case llvm::Triple::ppc64:
    LibDirs.append(PPC64LibDirs,
                   PPC64LibDirs + llvm::array_lengthof(PPC64LibDirs));
    TripleAliases.append(PPC64Triples,
                         PPC64Triples + llvm::array_lengthof(PPC64Triples));
    BiarchLibDirs.append(PPCLibDirs,
                         PPCLibDirs + llvm::array_lengthof(PPCLibDirs));
    BiarchTripleAliases.append(PPCTriples,
                               PPCTriples + llvm::array_lengthof(PPCTriples));
    break;
  case llvm::Triple::ppc64le:
    LibDirs.append(PPC64LELibDirs,
                   PPC64LELibDirs + llvm::array_lengthof(PPC64LELibDirs));
    TripleAliases.append(PPC64LETriples,
                         PPC64LETriples + llvm::array_lengthof(PPC64LETriples));
    break;
  case llvm::Triple::systemz:
    LibDirs.append(SystemZLibDirs,
                   SystemZLibDirs + llvm::array_lengthof(SystemZLibDirs));
    TripleAliases.append(SystemZTriples,
                         SystemZTriples + llvm::array_lengthof(SystemZTriples));
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

static bool isSoftFloatABI(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_msoft_float,
                           options::OPT_mhard_float,
                           options::OPT_mfloat_abi_EQ);
  if (!A) return false;

  return A->getOption().matches(options::OPT_msoft_float) ||
         (A->getOption().matches(options::OPT_mfloat_abi_EQ) &&
          A->getValue() == StringRef("soft"));
}

static bool isMipsArch(llvm::Triple::ArchType Arch) {
  return Arch == llvm::Triple::mips ||
         Arch == llvm::Triple::mipsel ||
         Arch == llvm::Triple::mips64 ||
         Arch == llvm::Triple::mips64el;
}

static bool isMips16(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_mips16,
                           options::OPT_mno_mips16);
  return A && A->getOption().matches(options::OPT_mips16);
}

static bool isMips32r2(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_march_EQ,
                           options::OPT_mcpu_EQ);

  return A && A->getValue() == StringRef("mips32r2");
}

static bool isMips64r2(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_march_EQ,
                           options::OPT_mcpu_EQ);

  return A && A->getValue() == StringRef("mips64r2");
}

static bool isMicroMips(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_mmicromips,
                           options::OPT_mno_micromips);
  return A && A->getOption().matches(options::OPT_mmicromips);
}

static bool isMipsFP64(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_mfp64, options::OPT_mfp32);
  return A && A->getOption().matches(options::OPT_mfp64);
}

static bool isMipsNan2008(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_mnan_EQ);
  return A && A->getValue() == StringRef("2008");
}

// FIXME: There is the same routine in the Tools.cpp.
static bool hasMipsN32ABIArg(const ArgList &Args) {
  Arg *A = Args.getLastArg(options::OPT_mabi_EQ);
  return A && (A->getValue() == StringRef("n32"));
}

static bool hasCrtBeginObj(Twine Path) {
  return llvm::sys::fs::exists(Path + "/crtbegin.o");
}

static bool findTargetBiarchSuffix(std::string &Suffix, StringRef Path,
                                   llvm::Triple::ArchType TargetArch,
                                   const ArgList &Args) {
  // FIXME: This routine was only intended to model bi-arch toolchains which
  // use -m32 and -m64 to swap between variants of a target. It shouldn't be
  // doing ABI-based builtin location for MIPS.
  if (hasMipsN32ABIArg(Args))
    Suffix = "/n32";
  else if (TargetArch == llvm::Triple::x86_64 ||
           TargetArch == llvm::Triple::ppc64 ||
           TargetArch == llvm::Triple::systemz ||
           TargetArch == llvm::Triple::mips64 ||
           TargetArch == llvm::Triple::mips64el)
    Suffix = "/64";
  else
    Suffix = "/32";

  return hasCrtBeginObj(Path + Suffix);
}

void Generic_GCC::GCCInstallationDetector::findMIPSABIDirSuffix(
    std::string &Suffix, llvm::Triple::ArchType TargetArch, StringRef Path,
    const llvm::opt::ArgList &Args) {
  if (!isMipsArch(TargetArch))
    return;

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
  //
  // Unfortunately different toolchains use different and partially
  // overlapped naming schemes. So we have to make a trick for detection
  // of using toolchain. We lookup a path which unique for each toolchains.

  bool IsMentorToolChain = hasCrtBeginObj(Path + "/mips16/soft-float");
  bool IsFSFToolChain = hasCrtBeginObj(Path + "/mips32/mips16/sof");

  if (IsMentorToolChain && IsFSFToolChain)
    D.Diag(diag::err_drv_unknown_toolchain);

  if (IsMentorToolChain) {
    if (isMips16(Args))
      Suffix += "/mips16";
    else if (isMicroMips(Args))
      Suffix += "/micromips";

    if (isSoftFloatABI(Args))
      Suffix += "/soft-float";

    if (TargetArch == llvm::Triple::mipsel ||
        TargetArch == llvm::Triple::mips64el)
      Suffix += "/el";
  } else if (IsFSFToolChain) {
    if (TargetArch == llvm::Triple::mips ||
        TargetArch == llvm::Triple::mipsel) {
      if (isMicroMips(Args))
        Suffix += "/micromips";
      else if (isMips32r2(Args))
        Suffix += "";
      else
        Suffix += "/mips32";

      if (isMips16(Args))
        Suffix += "/mips16";
    } else {
      if (isMips64r2(Args))
        Suffix += hasMipsN32ABIArg(Args) ? "/mips64r2" : "/mips64r2/64";
      else
        Suffix += hasMipsN32ABIArg(Args) ? "/mips64" : "/mips64/64";
    }

    if (TargetArch == llvm::Triple::mipsel ||
        TargetArch == llvm::Triple::mips64el)
      Suffix += "/el";

    if (isSoftFloatABI(Args))
      Suffix += "/sof";
    else {
      if (isMipsFP64(Args))
        Suffix += "/fp64";

      if (isMipsNan2008(Args))
        Suffix += "/nan2008";
    }
  }

  if (!hasCrtBeginObj(Path + Suffix))
    Suffix.clear();
}

void Generic_GCC::GCCInstallationDetector::ScanLibDirForGCCTriple(
    llvm::Triple::ArchType TargetArch, const ArgList &Args,
    const std::string &LibDir, StringRef CandidateTriple,
    bool NeedsBiarchSuffix) {
  // There are various different suffixes involving the triple we
  // check for. We also record what is necessary to walk from each back
  // up to the lib directory.
  const std::string LibSuffixes[] = {
    "/gcc/" + CandidateTriple.str(),
    // Debian puts cross-compilers in gcc-cross
    "/gcc-cross/" + CandidateTriple.str(),
    "/" + CandidateTriple.str() + "/gcc/" + CandidateTriple.str(),

    // The Freescale PPC SDK has the gcc libraries in
    // <sysroot>/usr/lib/<triple>/x.y.z so have a look there as well.
    "/" + CandidateTriple.str(),

    // Ubuntu has a strange mis-matched pair of triples that this happens to
    // match.
    // FIXME: It may be worthwhile to generalize this and look for a second
    // triple.
    "/i386-linux-gnu/gcc/" + CandidateTriple.str()
  };
  const std::string InstallSuffixes[] = {
    "/../../..",    // gcc/
    "/../../..",    // gcc-cross/
    "/../../../..", // <triple>/gcc/
    "/../..",       // <triple>/
    "/../../../.."  // i386-linux-gnu/gcc/<triple>/
  };
  // Only look at the final, weird Ubuntu suffix for i386-linux-gnu.
  const unsigned NumLibSuffixes =
      (llvm::array_lengthof(LibSuffixes) - (TargetArch != llvm::Triple::x86));
  for (unsigned i = 0; i < NumLibSuffixes; ++i) {
    StringRef LibSuffix = LibSuffixes[i];
    llvm::error_code EC;
    for (llvm::sys::fs::directory_iterator LI(LibDir + LibSuffix, EC), LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef VersionText = llvm::sys::path::filename(LI->path());
      GCCVersion CandidateVersion = GCCVersion::Parse(VersionText);
      if (CandidateVersion.Major != -1) // Filter obviously bad entries.
        if (!CandidateGCCInstallPaths.insert(LI->path()).second)
          continue; // Saw this path before; no need to look at it again.
      if (CandidateVersion.isOlderThan(4, 1, 1))
        continue;
      if (CandidateVersion <= Version)
        continue;

      std::string MIPSABIDirSuffix;
      findMIPSABIDirSuffix(MIPSABIDirSuffix, TargetArch, LI->path(), Args);

      // Some versions of SUSE and Fedora on ppc64 put 32-bit libs
      // in what would normally be GCCInstallPath and put the 64-bit
      // libs in a subdirectory named 64. The simple logic we follow is that
      // *if* there is a subdirectory of the right name with crtbegin.o in it,
      // we use that. If not, and if not a biarch triple alias, we look for
      // crtbegin.o without the subdirectory.

      std::string BiarchSuffix;
      if (findTargetBiarchSuffix(BiarchSuffix,
                                 LI->path() + MIPSABIDirSuffix,
                                 TargetArch, Args)) {
        GCCBiarchSuffix = BiarchSuffix;
      } else if (NeedsBiarchSuffix ||
                 !hasCrtBeginObj(LI->path() + MIPSABIDirSuffix)) {
        continue;
      } else {
        GCCBiarchSuffix.clear();
      }

      Version = CandidateVersion;
      GCCTriple.setTriple(CandidateTriple);
      // FIXME: We hack together the directory name here instead of
      // using LI to ensure stable path separators across Windows and
      // Linux.
      GCCInstallPath = LibDir + LibSuffixes[i] + "/" + VersionText.str();
      GCCParentLibPath = GCCInstallPath + InstallSuffixes[i];
      GCCMIPSABIDirSuffix = MIPSABIDirSuffix;
      IsValid = true;
    }
  }
}

Generic_GCC::Generic_GCC(const Driver &D, const llvm::Triple& Triple,
                         const ArgList &Args)
  : ToolChain(D, Triple, Args), GCCInstallation(getDriver(), Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);
}

Generic_GCC::~Generic_GCC() {
}

Tool *Generic_GCC::getTool(Action::ActionClass AC) const {
  switch (AC) {
  case Action::PreprocessJobClass:
    if (!Preprocess)
      Preprocess.reset(new tools::gcc::Preprocess(*this));
    return Preprocess.get();
  case Action::PrecompileJobClass:
    if (!Precompile)
      Precompile.reset(new tools::gcc::Precompile(*this));
    return Precompile.get();
  case Action::CompileJobClass:
    if (!Compile)
      Compile.reset(new tools::gcc::Compile(*this));
    return Compile.get();
  default:
    return ToolChain::getTool(AC);
  }
}

Tool *Generic_GCC::buildAssembler() const {
  return new tools::gcc::Assemble(*this);
}

Tool *Generic_GCC::buildLinker() const {
  return new tools::gcc::Link(*this);
}

void Generic_GCC::printVerboseInfo(raw_ostream &OS) const {
  // Print the information about how we detected the GCC installation.
  GCCInstallation.print(OS);
}

bool Generic_GCC::IsUnwindTablesDefault() const {
  return getArch() == llvm::Triple::x86_64;
}

bool Generic_GCC::isPICDefault() const {
  return false;
}

bool Generic_GCC::isPIEDefault() const {
  return false;
}

bool Generic_GCC::isPICDefaultForced() const {
  return false;
}

void Generic_GCC::addClangTargetOptions(const ArgList &DriverArgs,
                                        ArgStringList &CC1Args) const {
  const Generic_GCC::GCCVersion &V = GCCInstallation.getVersion();
  bool UseInitArrayDefault = 
      getTriple().getArch() == llvm::Triple::aarch64 ||
      (getTriple().getOS() == llvm::Triple::Linux && (
         !V.isOlderThan(4, 7, 0) ||
         getTriple().getEnvironment() == llvm::Triple::Android));

  if (DriverArgs.hasFlag(options::OPT_fuse_init_array,
                         options::OPT_fno_use_init_array,
                         UseInitArrayDefault))
    CC1Args.push_back("-fuse-init-array");
}

/// Hexagon Toolchain

std::string Hexagon_TC::GetGnuDir(const std::string &InstalledDir) {

  // Locate the rest of the toolchain ...
  if (strlen(GCC_INSTALL_PREFIX))
    return std::string(GCC_INSTALL_PREFIX);

  std::string InstallRelDir = InstalledDir + "/../../gnu";
  if (llvm::sys::fs::exists(InstallRelDir))
    return InstallRelDir;

  std::string PrefixRelDir = std::string(LLVM_PREFIX) + "/../gnu";
  if (llvm::sys::fs::exists(PrefixRelDir))
    return PrefixRelDir;

  return InstallRelDir;
}

static void GetHexagonLibraryPaths(
  const ArgList &Args,
  const std::string Ver,
  const std::string MarchString,
  const std::string &InstalledDir,
  ToolChain::path_list *LibPaths)
{
  bool buildingLib = Args.hasArg(options::OPT_shared);

  //----------------------------------------------------------------------------
  // -L Args
  //----------------------------------------------------------------------------
  for (arg_iterator
         it = Args.filtered_begin(options::OPT_L),
         ie = Args.filtered_end();
       it != ie;
       ++it) {
    for (unsigned i = 0, e = (*it)->getNumValues(); i != e; ++i)
      LibPaths->push_back((*it)->getValue(i));
  }

  //----------------------------------------------------------------------------
  // Other standard paths
  //----------------------------------------------------------------------------
  const std::string MarchSuffix = "/" + MarchString;
  const std::string G0Suffix = "/G0";
  const std::string MarchG0Suffix = MarchSuffix + G0Suffix;
  const std::string RootDir = Hexagon_TC::GetGnuDir(InstalledDir) + "/";

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

Hexagon_TC::Hexagon_TC(const Driver &D, const llvm::Triple &Triple,
                       const ArgList &Args)
  : Linux(D, Triple, Args) {
  const std::string InstalledDir(getDriver().getInstalledDir());
  const std::string GnuDir = Hexagon_TC::GetGnuDir(InstalledDir);

  // Note: Generic_GCC::Generic_GCC adds InstalledDir and getDriver().Dir to
  // program paths
  const std::string BinDir(GnuDir + "/bin");
  if (llvm::sys::fs::exists(BinDir))
    getProgramPaths().push_back(BinDir);

  // Determine version of GCC libraries and headers to use.
  const std::string HexagonDir(GnuDir + "/lib/gcc/hexagon");
  llvm::error_code ec;
  GCCVersion MaxVersion= GCCVersion::Parse("0.0.0");
  for (llvm::sys::fs::directory_iterator di(HexagonDir, ec), de;
       !ec && di != de; di = di.increment(ec)) {
    GCCVersion cv = GCCVersion::Parse(llvm::sys::path::filename(di->path()));
    if (MaxVersion < cv)
      MaxVersion = cv;
  }
  GCCLibAndIncVersion = MaxVersion;

  ToolChain::path_list *LibPaths= &getFilePaths();

  // Remove paths added by Linux toolchain. Currently Hexagon_TC really targets
  // 'elf' OS type, so the Linux paths are not appropriate. When we actually
  // support 'linux' we'll need to fix this up
  LibPaths->clear();

  GetHexagonLibraryPaths(
    Args,
    GetGCCLibAndIncVersion(),
    GetTargetCPU(Args),
    InstalledDir,
    LibPaths);
}

Hexagon_TC::~Hexagon_TC() {
}

Tool *Hexagon_TC::buildAssembler() const {
  return new tools::hexagon::Assemble(*this);
}

Tool *Hexagon_TC::buildLinker() const {
  return new tools::hexagon::Link(*this);
}

void Hexagon_TC::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                           ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  std::string Ver(GetGCCLibAndIncVersion());
  std::string GnuDir = Hexagon_TC::GetGnuDir(D.InstalledDir);
  std::string HexagonDir(GnuDir + "/lib/gcc/hexagon/" + Ver);
  addExternCSystemInclude(DriverArgs, CC1Args, HexagonDir + "/include");
  addExternCSystemInclude(DriverArgs, CC1Args, HexagonDir + "/include-fixed");
  addExternCSystemInclude(DriverArgs, CC1Args, GnuDir + "/hexagon/include");
}

void Hexagon_TC::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                              ArgStringList &CC1Args) const {

  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  const Driver &D = getDriver();
  std::string Ver(GetGCCLibAndIncVersion());
  SmallString<128> IncludeDir(Hexagon_TC::GetGnuDir(D.InstalledDir));

  llvm::sys::path::append(IncludeDir, "hexagon/include/c++/");
  llvm::sys::path::append(IncludeDir, Ver);
  addSystemInclude(DriverArgs, CC1Args, IncludeDir.str());
}

ToolChain::CXXStdlibType
Hexagon_TC::GetCXXStdlibType(const ArgList &Args) const {
  Arg *A = Args.getLastArg(options::OPT_stdlib_EQ);
  if (!A)
    return ToolChain::CST_Libstdcxx;

  StringRef Value = A->getValue();
  if (Value != "libstdc++") {
    getDriver().Diag(diag::err_drv_invalid_stdlib_name)
      << A->getAsString(Args);
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

StringRef Hexagon_TC::GetTargetCPU(const ArgList &Args)
{
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

/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
/// Currently does not support anything else but compilation.

TCEToolChain::TCEToolChain(const Driver &D, const llvm::Triple& Triple,
                           const ArgList &Args)
  : ToolChain(D, Triple, Args) {
  // Path mangling to find libexec
  std::string Path(getDriver().Dir);

  Path += "/../libexec";
  getProgramPaths().push_back(Path);
}

TCEToolChain::~TCEToolChain() {
}

bool TCEToolChain::IsMathErrnoDefault() const {
  return true;
}

bool TCEToolChain::isPICDefault() const {
  return false;
}

bool TCEToolChain::isPIEDefault() const {
  return false;
}

bool TCEToolChain::isPICDefaultForced() const {
  return false;
}

/// OpenBSD - OpenBSD tool chain which can call as(1) and ld(1) directly.

OpenBSD::OpenBSD(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *OpenBSD::buildAssembler() const {
  return new tools::openbsd::Assemble(*this);
}

Tool *OpenBSD::buildLinker() const {
  return new tools::openbsd::Link(*this);
}

/// Bitrig - Bitrig tool chain which can call as(1) and ld(1) directly.

Bitrig::Bitrig(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *Bitrig::buildAssembler() const {
  return new tools::bitrig::Assemble(*this);
}

Tool *Bitrig::buildLinker() const {
  return new tools::bitrig::Link(*this);
}

void Bitrig::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
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
                     getDriver().SysRoot + "/usr/include/c++/stdc++");
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/stdc++/backward");

    StringRef Triple = getTriple().str();
    if (Triple.startswith("amd64"))
      addSystemInclude(DriverArgs, CC1Args,
                       getDriver().SysRoot + "/usr/include/c++/stdc++/x86_64" +
                       Triple.substr(5));
    else
      addSystemInclude(DriverArgs, CC1Args,
                       getDriver().SysRoot + "/usr/include/c++/stdc++/" +
                       Triple);
    break;
  }
}

void Bitrig::AddCXXStdlibLibArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    CmdArgs.push_back("-lcxxrt");
    // Include supc++ to provide Unwind until provided by libcxx.
    CmdArgs.push_back("-lgcc");
    break;
  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    break;
  }
}

/// FreeBSD - FreeBSD tool chain which can call as(1) and ld(1) directly.

FreeBSD::FreeBSD(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {

  // When targeting 32-bit platforms, look for '/usr/lib32/crt1.o' and fall
  // back to '/usr/lib' if it doesn't exist.
  if ((Triple.getArch() == llvm::Triple::x86 ||
       Triple.getArch() == llvm::Triple::ppc) &&
      llvm::sys::fs::exists(getDriver().SysRoot + "/usr/lib32/crt1.o"))
    getFilePaths().push_back(getDriver().SysRoot + "/usr/lib32");
  else
    getFilePaths().push_back(getDriver().SysRoot + "/usr/lib");
}

ToolChain::CXXStdlibType
FreeBSD::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;

    getDriver().Diag(diag::err_drv_invalid_stdlib_name)
      << A->getAsString(Args);
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
  return new tools::freebsd::Assemble(*this);
}

Tool *FreeBSD::buildLinker() const {
  return new tools::freebsd::Link(*this);
}

bool FreeBSD::UseSjLjExceptions() const {
  // FreeBSD uses SjLj exceptions on ARM oabi.
  switch (getTriple().getEnvironment()) {
  case llvm::Triple::GNUEABI:
  case llvm::Triple::EABI:
    return false;

  default:
    return (getTriple().getArch() == llvm::Triple::arm ||
            getTriple().getArch() == llvm::Triple::thumb);
  }
}

/// NetBSD - NetBSD tool chain which can call as(1) and ld(1) directly.

NetBSD::NetBSD(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {

  if (getDriver().UseStdLib) {
    // When targeting a 32-bit platform, try the special directory used on
    // 64-bit hosts, and only fall back to the main library directory if that
    // doesn't work.
    // FIXME: It'd be nicer to test if this directory exists, but I'm not sure
    // what all logic is needed to emulate the '=' prefix here.
    if (Triple.getArch() == llvm::Triple::x86)
      getFilePaths().push_back("=/usr/lib/i386");

    getFilePaths().push_back("=/usr/lib");
  }
}

Tool *NetBSD::buildAssembler() const {
  return new tools::netbsd::Assemble(*this);
}

Tool *NetBSD::buildLinker() const {
  return new tools::netbsd::Link(*this);
}

ToolChain::CXXStdlibType
NetBSD::GetCXXStdlibType(const ArgList &Args) const {
  if (Arg *A = Args.getLastArg(options::OPT_stdlib_EQ)) {
    StringRef Value = A->getValue();
    if (Value == "libstdc++")
      return ToolChain::CST_Libstdcxx;
    if (Value == "libc++")
      return ToolChain::CST_Libcxx;

    getDriver().Diag(diag::err_drv_invalid_stdlib_name)
      << A->getAsString(Args);
  }

  unsigned Major, Minor, Micro;
  getTriple().getOSVersion(Major, Minor, Micro);
  if (Major >= 7 || (Major == 6 && Minor == 99 && Micro >= 23) || Major == 0) {
    if (getArch() == llvm::Triple::x86 || getArch() == llvm::Triple::x86_64)
      return ToolChain::CST_Libcxx;
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

Minix::Minix(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *Minix::buildAssembler() const {
  return new tools::minix::Assemble(*this);
}

Tool *Minix::buildLinker() const {
  return new tools::minix::Link(*this);
}

/// AuroraUX - AuroraUX tool chain which can call as(1) and ld(1) directly.

AuroraUX::AuroraUX(const Driver &D, const llvm::Triple& Triple,
                   const ArgList &Args)
  : Generic_GCC(D, Triple, Args) {

  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
  getFilePaths().push_back("/usr/sfw/lib");
  getFilePaths().push_back("/opt/gcc4/lib");
  getFilePaths().push_back("/opt/gcc4/lib/gcc/i386-pc-solaris2.11/4.2.4");

}

Tool *AuroraUX::buildAssembler() const {
  return new tools::auroraux::Assemble(*this);
}

Tool *AuroraUX::buildLinker() const {
  return new tools::auroraux::Link(*this);
}

/// Solaris - Solaris tool chain which can call as(1) and ld(1) directly.

Solaris::Solaris(const Driver &D, const llvm::Triple& Triple,
                 const ArgList &Args)
  : Generic_GCC(D, Triple, Args) {

  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool *Solaris::buildAssembler() const {
  return new tools::solaris::Assemble(*this);
}

Tool *Solaris::buildLinker() const {
  return new tools::solaris::Link(*this);
}

/// Distribution (very bare-bones at the moment).

enum Distro {
  ArchLinux,
  DebianLenny,
  DebianSqueeze,
  DebianWheezy,
  DebianJessie,
  Exherbo,
  RHEL4,
  RHEL5,
  RHEL6,
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
  UnknownDistro
};

static bool IsRedhat(enum Distro Distro) {
  return Distro == Fedora || (Distro >= RHEL4 && Distro <= RHEL6);
}

static bool IsOpenSUSE(enum Distro Distro) {
  return Distro == OpenSUSE;
}

static bool IsDebian(enum Distro Distro) {
  return Distro >= DebianLenny && Distro <= DebianJessie;
}

static bool IsUbuntu(enum Distro Distro) {
  return Distro >= UbuntuHardy && Distro <= UbuntuTrusty;
}

static Distro DetectDistro(llvm::Triple::ArchType Arch) {
  OwningPtr<llvm::MemoryBuffer> File;
  if (!llvm::MemoryBuffer::getFile("/etc/lsb-release", File)) {
    StringRef Data = File.get()->getBuffer();
    SmallVector<StringRef, 8> Lines;
    Data.split(Lines, "\n");
    Distro Version = UnknownDistro;
    for (unsigned i = 0, s = Lines.size(); i != s; ++i)
      if (Version == UnknownDistro && Lines[i].startswith("DISTRIB_CODENAME="))
        Version = llvm::StringSwitch<Distro>(Lines[i].substr(17))
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
          .Default(UnknownDistro);
    return Version;
  }

  if (!llvm::MemoryBuffer::getFile("/etc/redhat-release", File)) {
    StringRef Data = File.get()->getBuffer();
    if (Data.startswith("Fedora release"))
      return Fedora;
    else if (Data.startswith("Red Hat Enterprise Linux") &&
             Data.find("release 6") != StringRef::npos)
      return RHEL6;
    else if ((Data.startswith("Red Hat Enterprise Linux") ||
              Data.startswith("CentOS")) &&
             Data.find("release 5") != StringRef::npos)
      return RHEL5;
    else if ((Data.startswith("Red Hat Enterprise Linux") ||
              Data.startswith("CentOS")) &&
             Data.find("release 4") != StringRef::npos)
      return RHEL4;
    return UnknownDistro;
  }

  if (!llvm::MemoryBuffer::getFile("/etc/debian_version", File)) {
    StringRef Data = File.get()->getBuffer();
    if (Data[0] == '5')
      return DebianLenny;
    else if (Data.startswith("squeeze/sid") || Data[0] == '6')
      return DebianSqueeze;
    else if (Data.startswith("wheezy/sid")  || Data[0] == '7')
      return DebianWheezy;
    else if (Data.startswith("jessie/sid")  || Data[0] == '8')
      return DebianJessie;
    return UnknownDistro;
  }

  if (llvm::sys::fs::exists("/etc/SuSE-release"))
    return OpenSUSE;

  if (llvm::sys::fs::exists("/etc/exherbo-release"))
    return Exherbo;

  if (llvm::sys::fs::exists("/etc/arch-release"))
    return ArchLinux;

  return UnknownDistro;
}

/// \brief Get our best guess at the multiarch triple for a target.
///
/// Debian-based systems are starting to use a multiarch setup where they use
/// a target-triple directory in the library and header search paths.
/// Unfortunately, this triple does not align with the vanilla target triple,
/// so we provide a rough mapping here.
static std::string getMultiarchTriple(const llvm::Triple TargetTriple,
                                      StringRef SysRoot) {
  // For most architectures, just use whatever we have rather than trying to be
  // clever.
  switch (TargetTriple.getArch()) {
  default:
    return TargetTriple.str();

    // We use the existence of '/lib/<triple>' as a directory to detect some
    // common linux triples that don't quite match the Clang triple for both
    // 32-bit and 64-bit targets. Multiarch fixes its install triples to these
    // regardless of what the actual target triple is.
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    if (TargetTriple.getEnvironment() == llvm::Triple::GNUEABIHF) {
      if (llvm::sys::fs::exists(SysRoot + "/lib/arm-linux-gnueabihf"))
        return "arm-linux-gnueabihf";
    } else {
      if (llvm::sys::fs::exists(SysRoot + "/lib/arm-linux-gnueabi"))
        return "arm-linux-gnueabi";
    }
    return TargetTriple.str();
  case llvm::Triple::x86:
    if (llvm::sys::fs::exists(SysRoot + "/lib/i386-linux-gnu"))
      return "i386-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::x86_64:
    if (llvm::sys::fs::exists(SysRoot + "/lib/x86_64-linux-gnu"))
      return "x86_64-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::aarch64:
    if (llvm::sys::fs::exists(SysRoot + "/lib/aarch64-linux-gnu"))
      return "aarch64-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::mips:
    if (llvm::sys::fs::exists(SysRoot + "/lib/mips-linux-gnu"))
      return "mips-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::mipsel:
    if (llvm::sys::fs::exists(SysRoot + "/lib/mipsel-linux-gnu"))
      return "mipsel-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::ppc:
    if (llvm::sys::fs::exists(SysRoot + "/lib/powerpc-linux-gnuspe"))
      return "powerpc-linux-gnuspe";
    if (llvm::sys::fs::exists(SysRoot + "/lib/powerpc-linux-gnu"))
      return "powerpc-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::ppc64:
    if (llvm::sys::fs::exists(SysRoot + "/lib/powerpc64-linux-gnu"))
      return "powerpc64-linux-gnu";
  case llvm::Triple::ppc64le:
    if (llvm::sys::fs::exists(SysRoot + "/lib/powerpc64le-linux-gnu"))
      return "powerpc64le-linux-gnu";
    return TargetTriple.str();
  }
}

static void addPathIfExists(Twine Path, ToolChain::path_list &Paths) {
  if (llvm::sys::fs::exists(Path)) Paths.push_back(Path.str());
}

static StringRef getMultilibDir(const llvm::Triple &Triple,
                                const ArgList &Args) {
  if (isMipsArch(Triple.getArch())) {
    // lib32 directory has a special meaning on MIPS targets.
    // It contains N32 ABI binaries. Use this folder if produce
    // code for N32 ABI only.
    if (hasMipsN32ABIArg(Args))
      return "lib32";
    return Triple.isArch32Bit() ? "lib" : "lib64";
  }

  // It happens that only x86 and PPC use the 'lib32' variant of multilib, and
  // using that variant while targeting other architectures causes problems
  // because the libraries are laid out in shared system roots that can't cope
  // with a 'lib32' multilib search path being considered. So we only enable
  // them when we know we may need it.
  //
  // FIXME: This is a bit of a hack. We should really unify this code for
  // reasoning about multilib spellings with the lib dir spellings in the
  // GCCInstallationDetector, but that is a more significant refactoring.
  if (Triple.getArch() == llvm::Triple::x86 ||
      Triple.getArch() == llvm::Triple::ppc)
    return "lib32";

  return Triple.isArch32Bit() ? "lib" : "lib64";
}

Linux::Linux(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {
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
                         GCCInstallation.getTriple().str() + "/bin").str());

  Linker = GetProgramPath("ld");

  Distro Distro = DetectDistro(Arch);

  if (IsOpenSUSE(Distro) || IsUbuntu(Distro)) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("relro");
  }

  if (Arch == llvm::Triple::arm || Arch == llvm::Triple::thumb)
    ExtraOpts.push_back("-X");

  const bool IsAndroid = Triple.getEnvironment() == llvm::Triple::Android;
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

  if (Distro == DebianSqueeze || Distro == DebianWheezy ||
      Distro == DebianJessie || IsOpenSUSE(Distro) ||
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

  const std::string Multilib = getMultilibDir(Triple, Args);
  const std::string MultiarchTriple = getMultiarchTriple(Triple, SysRoot);

  // Add the multilib suffixed paths where they are available.
  if (GCCInstallation.isValid()) {
    const llvm::Triple &GCCTriple = GCCInstallation.getTriple();
    const std::string &LibPath = GCCInstallation.getParentLibPath();

    // Sourcery CodeBench MIPS toolchain holds some libraries under
    // a biarch-like suffix of the GCC installation.
    //
    // FIXME: It would be cleaner to model this as a variant of bi-arch. IE,
    // instead of a '64' biarch suffix it would be 'el' or something.
    if (IsAndroid && IsMips && isMips32r2(Args)) {
      assert(GCCInstallation.getBiarchSuffix().empty() &&
             "Unexpected bi-arch suffix");
      addPathIfExists(GCCInstallation.getInstallPath() + "/mips-r2", Paths);
    } else {
      addPathIfExists((GCCInstallation.getInstallPath() +
                       GCCInstallation.getMIPSABIDirSuffix() +
                       GCCInstallation.getBiarchSuffix()),
                      Paths);
    }

    // GCC cross compiling toolchains will install target libraries which ship
    // as part of the toolchain under <prefix>/<triple>/<libdir> rather than as
    // any part of the GCC installation in
    // <prefix>/<libdir>/gcc/<triple>/<version>. This decision is somewhat
    // debatable, but is the reality today. We need to search this tree even
    // when we have a sysroot somewhere else. It is the responsibility of
    // whomever is doing the cross build targetting a sysroot using a GCC
    // installation that is *not* within the system root to ensure two things:
    //
    //  1) Any DSOs that are linked in from this tree or from the install path
    //     above must be preasant on the system root and found via an
    //     appropriate rpath.
    //  2) There must not be libraries installed into
    //     <prefix>/<triple>/<libdir> unless they should be preferred over
    //     those within the system root.
    //
    // Note that this matches the GCC behavior. See the below comment for where
    // Clang diverges from GCC's behavior.
    addPathIfExists(LibPath + "/../" + GCCTriple.str() + "/lib/../" + Multilib +
                    GCCInstallation.getMIPSABIDirSuffix(),
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
      addPathIfExists(LibPath + "/" + MultiarchTriple, Paths);
      addPathIfExists(LibPath + "/../" + Multilib, Paths);
    }
  }
  addPathIfExists(SysRoot + "/lib/" + MultiarchTriple, Paths);
  addPathIfExists(SysRoot + "/lib/../" + Multilib, Paths);
  addPathIfExists(SysRoot + "/usr/lib/" + MultiarchTriple, Paths);
  addPathIfExists(SysRoot + "/usr/lib/../" + Multilib, Paths);

  // Try walking via the GCC triple path in case of biarch or multiarch GCC
  // installations with strange symlinks.
  if (GCCInstallation.isValid()) {
    addPathIfExists(SysRoot + "/usr/lib/" + GCCInstallation.getTriple().str() +
                    "/../../" + Multilib, Paths);

    // Add the non-multilib suffixed paths (if potentially different).
    const std::string &LibPath = GCCInstallation.getParentLibPath();
    const llvm::Triple &GCCTriple = GCCInstallation.getTriple();
    if (!GCCInstallation.getBiarchSuffix().empty())
      addPathIfExists(GCCInstallation.getInstallPath() +
                      GCCInstallation.getMIPSABIDirSuffix(), Paths);

    // See comments above on the multilib variant for details of why this is
    // included even from outside the sysroot.
    addPathIfExists(LibPath + "/../" + GCCTriple.str() +
                    "/lib" + GCCInstallation.getMIPSABIDirSuffix(), Paths);

    // See comments above on the multilib variant for details of why this is
    // only included from within the sysroot.
    if (StringRef(LibPath).startswith(SysRoot))
      addPathIfExists(LibPath, Paths);
  }
  addPathIfExists(SysRoot + "/lib", Paths);
  addPathIfExists(SysRoot + "/usr/lib", Paths);
}

bool FreeBSD::HasNativeLLVMSupport() const {
  return true;
}

bool Linux::HasNativeLLVMSupport() const {
  return true;
}

Tool *Linux::buildLinker() const {
  return new tools::gnutools::Link(*this);
}

Tool *Linux::buildAssembler() const {
  return new tools::gnutools::Assemble(*this);
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
  const StringRef MIPSABIDirSuffix = GCCInstallation.getMIPSABIDirSuffix();

  std::string Path = (InstallDir + "/../../../../" + TripleStr + "/libc" +
                      MIPSABIDirSuffix).str();

  if (llvm::sys::fs::exists(Path))
    return Path;

  Path = (InstallDir + "/../../../../sysroot" + MIPSABIDirSuffix).str();

  if (llvm::sys::fs::exists(Path))
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
    addSystemInclude(DriverArgs, CC1Args, P.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (SmallVectorImpl<StringRef>::iterator I = dirs.begin(), E = dirs.end();
         I != E; ++I) {
      StringRef Prefix = llvm::sys::path::is_absolute(*I) ? SysRoot : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + *I);
    }
    return;
  }

  // Lacking those, try to detect the correct set of system includes for the
  // target triple.

  // Sourcery CodeBench and modern FSF Mips toolchains put extern C
  // system includes under three additional directories.
  if (GCCInstallation.isValid() && isMipsArch(getTriple().getArch())) {
    addExternCSystemIncludeIfExists(
        DriverArgs, CC1Args, GCCInstallation.getInstallPath() + "/include");

    addExternCSystemIncludeIfExists(
        DriverArgs, CC1Args,
        GCCInstallation.getInstallPath() + "/../../../../" +
            GCCInstallation.getTriple().str() + "/libc/usr/include");

    addExternCSystemIncludeIfExists(
        DriverArgs, CC1Args,
        GCCInstallation.getInstallPath() + "/../../../../sysroot/usr/include");
  }

  // Implement generic Debian multiarch support.
  const StringRef X86_64MultiarchIncludeDirs[] = {
    "/usr/include/x86_64-linux-gnu",

    // FIXME: These are older forms of multiarch. It's not clear that they're
    // in use in any released version of Debian, so we should consider
    // removing them.
    "/usr/include/i686-linux-gnu/64", "/usr/include/i486-linux-gnu/64"
  };
  const StringRef X86MultiarchIncludeDirs[] = {
    "/usr/include/i386-linux-gnu",

    // FIXME: These are older forms of multiarch. It's not clear that they're
    // in use in any released version of Debian, so we should consider
    // removing them.
    "/usr/include/x86_64-linux-gnu/32", "/usr/include/i686-linux-gnu",
    "/usr/include/i486-linux-gnu"
  };
  const StringRef AArch64MultiarchIncludeDirs[] = {
    "/usr/include/aarch64-linux-gnu"
  };
  const StringRef ARMMultiarchIncludeDirs[] = {
    "/usr/include/arm-linux-gnueabi"
  };
  const StringRef ARMHFMultiarchIncludeDirs[] = {
    "/usr/include/arm-linux-gnueabihf"
  };
  const StringRef MIPSMultiarchIncludeDirs[] = {
    "/usr/include/mips-linux-gnu"
  };
  const StringRef MIPSELMultiarchIncludeDirs[] = {
    "/usr/include/mipsel-linux-gnu"
  };
  const StringRef PPCMultiarchIncludeDirs[] = {
    "/usr/include/powerpc-linux-gnu"
  };
  const StringRef PPC64MultiarchIncludeDirs[] = {
    "/usr/include/powerpc64-linux-gnu"
  };
  ArrayRef<StringRef> MultiarchIncludeDirs;
  if (getTriple().getArch() == llvm::Triple::x86_64) {
    MultiarchIncludeDirs = X86_64MultiarchIncludeDirs;
  } else if (getTriple().getArch() == llvm::Triple::x86) {
    MultiarchIncludeDirs = X86MultiarchIncludeDirs;
  } else if (getTriple().getArch() == llvm::Triple::aarch64) {
    MultiarchIncludeDirs = AArch64MultiarchIncludeDirs;
  } else if (getTriple().getArch() == llvm::Triple::arm) {
    if (getTriple().getEnvironment() == llvm::Triple::GNUEABIHF)
      MultiarchIncludeDirs = ARMHFMultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = ARMMultiarchIncludeDirs;
  } else if (getTriple().getArch() == llvm::Triple::mips) {
    MultiarchIncludeDirs = MIPSMultiarchIncludeDirs;
  } else if (getTriple().getArch() == llvm::Triple::mipsel) {
    MultiarchIncludeDirs = MIPSELMultiarchIncludeDirs;
  } else if (getTriple().getArch() == llvm::Triple::ppc) {
    MultiarchIncludeDirs = PPCMultiarchIncludeDirs;
  } else if (getTriple().getArch() == llvm::Triple::ppc64) {
    MultiarchIncludeDirs = PPC64MultiarchIncludeDirs;
  }
  for (ArrayRef<StringRef>::iterator I = MultiarchIncludeDirs.begin(),
                                     E = MultiarchIncludeDirs.end();
       I != E; ++I) {
    if (llvm::sys::fs::exists(SysRoot + *I)) {
      addExternCSystemInclude(DriverArgs, CC1Args, SysRoot + *I);
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

/// \brief Helper to add the three variant paths for a libstdc++ installation.
/*static*/ bool Linux::addLibStdCXXIncludePaths(Twine Base, Twine TargetArchDir,
                                                const ArgList &DriverArgs,
                                                ArgStringList &CC1Args) {
  if (!llvm::sys::fs::exists(Base))
    return false;
  addSystemInclude(DriverArgs, CC1Args, Base);
  addSystemInclude(DriverArgs, CC1Args, Base + "/" + TargetArchDir);
  addSystemInclude(DriverArgs, CC1Args, Base + "/backward");
  return true;
}

/// \brief Helper to add an extra variant path for an (Ubuntu) multilib
/// libstdc++ installation.
/*static*/ bool Linux::addLibStdCXXIncludePaths(Twine Base, Twine Suffix,
                                                Twine TargetArchDir,
                                                Twine BiarchSuffix,
                                                Twine MIPSABIDirSuffix,
                                                const ArgList &DriverArgs,
                                                ArgStringList &CC1Args) {
  if (!addLibStdCXXIncludePaths(Base + Suffix,
                                TargetArchDir + MIPSABIDirSuffix + BiarchSuffix,
                                DriverArgs, CC1Args))
    return false;

  addSystemInclude(DriverArgs, CC1Args, Base + "/" + TargetArchDir + Suffix
                   + MIPSABIDirSuffix + BiarchSuffix);
  return true;
}

void Linux::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  // Check if libc++ has been enabled and provide its include paths if so.
  if (GetCXXStdlibType(DriverArgs) == ToolChain::CST_Libcxx) {
    // libc++ is always installed at a fixed path on Linux currently.
    addSystemInclude(DriverArgs, CC1Args,
                     getDriver().SysRoot + "/usr/include/c++/v1");
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
  StringRef MIPSABIDirSuffix = GCCInstallation.getMIPSABIDirSuffix();
  StringRef BiarchSuffix = GCCInstallation.getBiarchSuffix();
  const GCCVersion &Version = GCCInstallation.getVersion();

  if (addLibStdCXXIncludePaths(LibDir.str() + "/../include",
                               "/c++/" + Version.Text, TripleStr, BiarchSuffix,
                               MIPSABIDirSuffix, DriverArgs, CC1Args))
    return;

  const std::string IncludePathCandidates[] = {
    // Gentoo is weird and places its headers inside the GCC install, so if the
    // first attempt to find the headers fails, try these patterns.
    InstallDir.str() + "/include/g++-v" + Version.MajorStr + "." +
        Version.MinorStr,
    InstallDir.str() + "/include/g++-v" + Version.MajorStr,
    // Android standalone toolchain has C++ headers in yet another place.
    LibDir.str() + "/../" + TripleStr.str() + "/include/c++/" + Version.Text,
    // Freescale SDK C++ headers are directly in <sysroot>/usr/include/c++,
    // without a subdirectory corresponding to the gcc version.
    LibDir.str() + "/../include/c++",
  };

  for (unsigned i = 0; i < llvm::array_lengthof(IncludePathCandidates); ++i) {
    if (addLibStdCXXIncludePaths(IncludePathCandidates[i],
                                 TripleStr + MIPSABIDirSuffix + BiarchSuffix,
                                 DriverArgs, CC1Args))
      break;
  }
}

bool Linux::isPIEDefault() const {
  return getSanitizerArgs().hasZeroBaseShadow();
}

/// DragonFly - DragonFly tool chain which can call as(1) and ld(1) directly.

DragonFly::DragonFly(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {

  // Path mangling to find libexec
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
  if (llvm::sys::fs::exists("/usr/lib/gcc47"))
    getFilePaths().push_back("/usr/lib/gcc47");
  else
    getFilePaths().push_back("/usr/lib/gcc44");
}

Tool *DragonFly::buildAssembler() const {
  return new tools::dragonfly::Assemble(*this);
}

Tool *DragonFly::buildLinker() const {
  return new tools::dragonfly::Link(*this);
}


/// XCore tool chain
XCore::XCore(const Driver &D, const llvm::Triple &Triple,
             const ArgList &Args) : ToolChain(D, Triple, Args) {
  // ProgramPaths are found via 'PATH' environment variable.
}

Tool *XCore::buildAssembler() const {
  return new tools::XCore::Assemble(*this);
}

Tool *XCore::buildLinker() const {
  return new tools::XCore::Link(*this);
}

bool XCore::isPICDefault() const {
  return false;
}

bool XCore::isPIEDefault() const {
  return false;
}

bool XCore::isPICDefaultForced() const {
  return false;
}

bool XCore::SupportsProfiling() const {
  return false;
}

bool XCore::hasBlocksRuntime() const {
  return false;
}


void XCore::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;
  if (const char *cl_include_dir = getenv("XCC_C_INCLUDE_PATH")) {
    SmallVector<StringRef, 4> Dirs;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator,'\0'};
    StringRef(cl_include_dir).split(Dirs, StringRef(EnvPathSeparatorStr));
    ArrayRef<StringRef> DirVec(Dirs);
    addSystemIncludes(DriverArgs, CC1Args, DirVec);
  }
}

void XCore::addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const {
  CC1Args.push_back("-nostdsysteminc");
}

void XCore::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;
  if (const char *cl_include_dir = getenv("XCC_CPLUS_INCLUDE_PATH")) {
    SmallVector<StringRef, 4> Dirs;
    const char EnvPathSeparatorStr[] = {llvm::sys::EnvPathSeparator,'\0'};
    StringRef(cl_include_dir).split(Dirs, StringRef(EnvPathSeparatorStr));
    ArrayRef<StringRef> DirVec(Dirs);
    addSystemIncludes(DriverArgs, CC1Args, DirVec);
  }
}

void XCore::AddCXXStdlibLibArgs(const ArgList &Args,
                                ArgStringList &CmdArgs) const {
  // We don't output any lib args. This is handled by xcc.
}
