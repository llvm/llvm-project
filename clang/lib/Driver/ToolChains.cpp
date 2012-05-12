//===--- ToolChains.cpp - ToolChain Implementations -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ToolChains.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/ObjCRuntime.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"
#include "clang/Basic/Version.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/system_error.h"

#include <cstdlib> // ::getenv

#include "clang/Config/config.h" // for GCC_INSTALL_PREFIX

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;

/// Darwin - Darwin tool chain for i386 and x86_64.

Darwin::Darwin(const Driver &D, const llvm::Triple& Triple)
  : ToolChain(D, Triple), TargetInitialized(false),
    ARCRuntimeForSimulator(ARCSimulator_None),
    LibCXXForSimulator(LibCXXSimulator_None)
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

bool Darwin::hasARCRuntime() const {
  // FIXME: Remove this once there is a proper way to detect an ARC runtime
  // for the simulator.
  switch (ARCRuntimeForSimulator) {
  case ARCSimulator_None:
    break;
  case ARCSimulator_HasARCRuntime:
    return true;
  case ARCSimulator_NoARCRuntime:
    return false;
  }

  if (isTargetIPhoneOS())
    return !isIPhoneOSVersionLT(5);
  else
    return !isMacosxVersionLT(10, 7);
}

bool Darwin::hasSubscriptingRuntime() const {
    return !isTargetIPhoneOS() && !isMacosxVersionLT(10, 8);
}

/// Darwin provides an ARC runtime starting in MacOS X 10.7 and iOS 5.0.
void Darwin::configureObjCRuntime(ObjCRuntime &runtime) const {
  if (runtime.getKind() != ObjCRuntime::NeXT)
    return ToolChain::configureObjCRuntime(runtime);

  runtime.HasARC = runtime.HasWeak = hasARCRuntime();
  runtime.HasSubscripting = hasSubscriptingRuntime();

  // So far, objc_terminate is only available in iOS 5.
  // FIXME: do the simulator logic properly.
  if (!ARCRuntimeForSimulator && isTargetIPhoneOS())
    runtime.HasTerminate = !isIPhoneOSVersionLT(5);
  else
    runtime.HasTerminate = false;
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
    .Case("armv5tej", "armv5")
    .Case("xscale", "xscale")
    .Case("armv4t", "armv4t")
    .Case("armv7", "armv7")
    .Cases("armv7a", "armv7-a", "armv7")
    .Cases("armv7r", "armv7-r", "armv7")
    .Cases("armv7m", "armv7-m", "armv7")
    .Default(0);
}

static const char *GetArmArchForMCpu(StringRef Value) {
  return llvm::StringSwitch<const char *>(Value)
    .Cases("arm9e", "arm946e-s", "arm966e-s", "arm968e-s", "arm926ej-s","armv5")
    .Cases("arm10e", "arm10tdmi", "armv5")
    .Cases("arm1020t", "arm1020e", "arm1022e", "arm1026ej-s", "armv5")
    .Case("xscale", "xscale")
    .Cases("arm1136j-s", "arm1136jf-s", "arm1176jz-s",
           "arm1176jzf-s", "cortex-m0", "armv6")
    .Cases("cortex-a8", "cortex-r4", "cortex-m3", "cortex-a9", "armv7")
    .Default(0);
}

StringRef Darwin::getDarwinArchName(const ArgList &Args) const {
  switch (getTriple().getArch()) {
  default:
    return getArchName();

  case llvm::Triple::thumb:
  case llvm::Triple::arm: {
    if (const Arg *A = Args.getLastArg(options::OPT_march_EQ))
      if (const char *Arch = GetArmArchForMArch(A->getValue(Args)))
        return Arch;

    if (const Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
      if (const char *Arch = GetArmArchForMCpu(A->getValue(Args)))
        return Arch;

    return "arm";
  }
  }
}

Darwin::~Darwin() {
  // Free tool implementations.
  for (llvm::DenseMap<unsigned, Tool*>::iterator
         it = Tools.begin(), ie = Tools.end(); it != ie; ++it)
    delete it->second;
}

std::string Darwin::ComputeEffectiveClangTriple(const ArgList &Args,
                                                types::ID InputType) const {
  llvm::Triple Triple(ComputeLLVMTriple(Args, InputType));

  // If the target isn't initialized (e.g., an unknown Darwin platform, return
  // the default triple).
  if (!isTargetInitialized())
    return Triple.getTriple();

  SmallString<16> Str;
  Str += isTargetIPhoneOS() ? "ios" : "macosx";
  Str += getTargetVersion().getAsString();
  Triple.setOSName(Str);

  return Triple.getTriple();
}

void Generic_ELF::anchor() {}

Tool &Darwin::SelectTool(const Compilation &C, const JobAction &JA,
                         const ActionList &Inputs) const {
  Action::ActionClass Key;

  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple())) {
    // Fallback to llvm-gcc for i386 kext compiles, we don't support that ABI.
    if (Inputs.size() == 1 &&
        types::isCXX(Inputs[0]->getType()) &&
        getTriple().isOSDarwin() &&
        getTriple().getArch() == llvm::Triple::x86 &&
        (C.getArgs().getLastArg(options::OPT_fapple_kext) ||
         C.getArgs().getLastArg(options::OPT_mkernel)))
      Key = JA.getKind();
    else
      Key = Action::AnalyzeJobClass;
  } else
    Key = JA.getKind();

  bool UseIntegratedAs = C.getArgs().hasFlag(options::OPT_integrated_as,
                                             options::OPT_no_integrated_as,
                                             IsIntegratedAssemblerDefault());

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::InputClass:
    case Action::BindArchClass:
      llvm_unreachable("Invalid tool kind.");
    case Action::PreprocessJobClass:
      T = new tools::darwin::Preprocess(*this); break;
    case Action::AnalyzeJobClass:
    case Action::MigrateJobClass:
      T = new tools::Clang(*this); break;
    case Action::PrecompileJobClass:
    case Action::CompileJobClass:
      T = new tools::darwin::Compile(*this); break;
    case Action::AssembleJobClass: {
      if (UseIntegratedAs)
        T = new tools::ClangAs(*this);
      else
        T = new tools::darwin::Assemble(*this);
      break;
    }
    case Action::LinkJobClass:
      T = new tools::darwin::Link(*this); break;
    case Action::LipoJobClass:
      T = new tools::darwin::Lipo(*this); break;
    case Action::DsymutilJobClass:
      T = new tools::darwin::Dsymutil(*this); break;
    case Action::VerifyJobClass:
      T = new tools::darwin::VerifyDebug(*this); break;
    }
  }

  return *T;
}


DarwinClang::DarwinClang(const Driver &D, const llvm::Triple& Triple)
  : Darwin(D, Triple)
{
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  // We expect 'as', 'ld', etc. to be adjacent to our install dir.
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);

  // For fallback, we need to know how to find the GCC cc1 executables, so we
  // also add the GCC libexec paths. This is legacy code that can be removed
  // once fallback is no longer useful.
  AddGCCLibexecPath(DarwinVersion[0]);
  AddGCCLibexecPath(DarwinVersion[0] - 2);
  AddGCCLibexecPath(DarwinVersion[0] - 1);
  AddGCCLibexecPath(DarwinVersion[0] + 1);
  AddGCCLibexecPath(DarwinVersion[0] + 2);
}

void DarwinClang::AddGCCLibexecPath(unsigned darwinVersion) {
  std::string ToolChainDir = "i686-apple-darwin";
  ToolChainDir += llvm::utostr(darwinVersion);
  ToolChainDir += "/4.2.1";

  std::string Path = getDriver().Dir;
  Path += "/../llvm-gcc-4.2/libexec/gcc/";
  Path += ToolChainDir;
  getProgramPaths().push_back(Path);

  Path = "/usr/llvm-gcc-4.2/libexec/gcc/";
  Path += ToolChainDir;
  getProgramPaths().push_back(Path);
}

void DarwinClang::AddLinkSearchPathArgs(const ArgList &Args,
                                       ArgStringList &CmdArgs) const {
  // The Clang toolchain uses explicit paths for internal libraries.

  // Unfortunately, we still might depend on a few of the libraries that are
  // only available in the gcc library directory (in particular
  // libstdc++.dylib). For now, hardcode the path to the known install location.
  // FIXME: This should get ripped out someday.  However, when building on
  // 10.6 (darwin10), we're still relying on this to find libstdc++.dylib.
  llvm::sys::Path P(getDriver().Dir);
  P.eraseComponent(); // .../usr/bin -> ../usr
  P.appendComponent("llvm-gcc-4.2");
  P.appendComponent("lib");
  P.appendComponent("gcc");
  switch (getTriple().getArch()) {
  default:
    llvm_unreachable("Invalid Darwin arch!");
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    P.appendComponent("i686-apple-darwin10");
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    P.appendComponent("arm-apple-darwin10");
    break;
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
    P.appendComponent("powerpc-apple-darwin10");
    break;
  }
  P.appendComponent("4.2.1");

  // Determine the arch specific GCC subdirectory.
  const char *ArchSpecificDir = 0;
  switch (getTriple().getArch()) {
  default:
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb: {
    std::string Triple = ComputeLLVMTriple(Args);
    StringRef TripleStr = Triple;
    if (TripleStr.startswith("armv5") || TripleStr.startswith("thumbv5"))
      ArchSpecificDir = "v5";
    else if (TripleStr.startswith("armv6") || TripleStr.startswith("thumbv6"))
      ArchSpecificDir = "v6";
    else if (TripleStr.startswith("armv7") || TripleStr.startswith("thumbv7"))
      ArchSpecificDir = "v7";
    break;
  }
  case llvm::Triple::ppc64:
    ArchSpecificDir = "ppc64";
    break;
  case llvm::Triple::x86_64:
    ArchSpecificDir = "x86_64";
    break;
  }

  if (ArchSpecificDir) {
    P.appendComponent(ArchSpecificDir);
    bool Exists;
    if (!llvm::sys::fs::exists(P.str(), Exists) && Exists)
      CmdArgs.push_back(Args.MakeArgString("-L" + P.str()));
    P.eraseComponent();
  }

  bool Exists;
  if (!llvm::sys::fs::exists(P.str(), Exists) && Exists)
    CmdArgs.push_back(Args.MakeArgString("-L" + P.str()));
}

void DarwinClang::AddLinkARCArgs(const ArgList &Args,
                                 ArgStringList &CmdArgs) const {

  CmdArgs.push_back("-force_load");
  llvm::sys::Path P(getDriver().ClangExecutable);
  P.eraseComponent(); // 'clang'
  P.eraseComponent(); // 'bin'
  P.appendComponent("lib");
  P.appendComponent("arc");
  P.appendComponent("libarclite_");
  std::string s = P.str();
  // Mash in the platform.
  if (isTargetIOSSimulator())
    s += "iphonesimulator";
  else if (isTargetIPhoneOS())
    s += "iphoneos";
  // FIXME: Remove this once we depend fully on -mios-simulator-version-min.
  else if (ARCRuntimeForSimulator != ARCSimulator_None)
    s += "iphonesimulator";
  else
    s += "macosx";
  s += ".a";

  CmdArgs.push_back(Args.MakeArgString(s));
}

void DarwinClang::AddLinkRuntimeLib(const ArgList &Args,
                                    ArgStringList &CmdArgs,
                                    const char *DarwinStaticLib) const {
  llvm::sys::Path P(getDriver().ResourceDir);
  P.appendComponent("lib");
  P.appendComponent("darwin");
  P.appendComponent(DarwinStaticLib);

  // For now, allow missing resource libraries to support developers who may
  // not have compiler-rt checked out or integrated into their build.
  bool Exists;
  if (!llvm::sys::fs::exists(P.str(), Exists) && Exists)
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
      << Args.getLastArg(options::OPT_rtlib_EQ)->getValue(Args) << "darwin";
    return;
  }

  // Darwin doesn't support real static executables, don't link any runtime
  // libraries with -static.
  if (Args.hasArg(options::OPT_static))
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

  // Add ASAN runtime library, if required. Dynamic libraries and bundles
  // should not be linked with the runtime library.
  if (Args.hasFlag(options::OPT_faddress_sanitizer,
                   options::OPT_fno_address_sanitizer, false)) {
    if (Args.hasArg(options::OPT_dynamiclib) ||
        Args.hasArg(options::OPT_bundle)) return;
    if (isTargetIPhoneOS()) {
      getDriver().Diag(diag::err_drv_clang_unsupported_per_platform)
        << "-faddress-sanitizer";
    } else {
      AddLinkRuntimeLib(Args, CmdArgs, "libclang_rt.asan_osx.a");

      // The ASAN runtime library requires C++ and CoreFoundation.
      AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-framework");
      CmdArgs.push_back("CoreFoundation");
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

static inline StringRef SimulatorVersionDefineName() {
  return "__IPHONE_OS_VERSION_MIN_REQUIRED";
}

/// \brief Parse the simulator version define:
/// __IPHONE_OS_VERSION_MIN_REQUIRED=([0-9])([0-9][0-9])([0-9][0-9])
// and return the grouped values as integers, e.g:
//   __IPHONE_OS_VERSION_MIN_REQUIRED=40201
// will return Major=4, Minor=2, Micro=1.
static bool GetVersionFromSimulatorDefine(StringRef define,
                                          unsigned &Major, unsigned &Minor,
                                          unsigned &Micro) {
  assert(define.startswith(SimulatorVersionDefineName()));
  StringRef name, version;
  llvm::tie(name, version) = define.split('=');
  if (version.empty())
    return false;
  std::string verstr = version.str();
  char *end;
  unsigned num = (unsigned) strtol(verstr.c_str(), &end, 10);
  if (*end != '\0')
    return false;
  Major = num / 10000;
  num = num % 10000;
  Minor = num / 100;
  Micro = num % 100;
  return true;
}

void Darwin::AddDeploymentTarget(DerivedArgList &Args) const {
  const OptTable &Opts = getDriver().getOpts();

  Arg *OSXVersion = Args.getLastArg(options::OPT_mmacosx_version_min_EQ);
  Arg *iOSVersion = Args.getLastArg(options::OPT_miphoneos_version_min_EQ);
  Arg *iOSSimVersion = Args.getLastArg(
    options::OPT_mios_simulator_version_min_EQ);

  // FIXME: HACK! When compiling for the simulator we don't get a
  // '-miphoneos-version-min' to help us know whether there is an ARC runtime
  // or not; try to parse a __IPHONE_OS_VERSION_MIN_REQUIRED
  // define passed in command-line.
  if (!iOSVersion && !iOSSimVersion) {
    for (arg_iterator it = Args.filtered_begin(options::OPT_D),
           ie = Args.filtered_end(); it != ie; ++it) {
      StringRef define = (*it)->getValue(Args);
      if (define.startswith(SimulatorVersionDefineName())) {
        unsigned Major = 0, Minor = 0, Micro = 0;
        if (GetVersionFromSimulatorDefine(define, Major, Minor, Micro) &&
            Major < 10 && Minor < 100 && Micro < 100) {
          ARCRuntimeForSimulator = Major < 5 ? ARCSimulator_NoARCRuntime
                                             : ARCSimulator_HasARCRuntime;
          LibCXXForSimulator = Major < 5 ? LibCXXSimulator_NotAvailable
                                         : LibCXXSimulator_Available;
        }
        break;
      }
    }
  }

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
        StringRef isysroot = A->getValue(Args);
        llvm::tie(first, second) = isysroot.split(StringRef("SDKs/iPhoneOS"));
        if (second != "")
          iOSTarget = second.substr(0,3);
      }
    }

    // If no OSX or iOS target has been specified and we're compiling for armv7,
    // go ahead as assume we're targeting iOS.
    if (OSXTarget.empty() && iOSTarget.empty())
      if (getDarwinArchName(Args) == "armv7")
        iOSTarget = "0.0";

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
      const Option *O = Opts.getOption(options::OPT_mmacosx_version_min_EQ);
      OSXVersion = Args.MakeJoinedArg(0, O, OSXTarget);
      Args.append(OSXVersion);
    } else if (!iOSTarget.empty()) {
      const Option *O = Opts.getOption(options::OPT_miphoneos_version_min_EQ);
      iOSVersion = Args.MakeJoinedArg(0, O, iOSTarget);
      Args.append(iOSVersion);
    } else if (!iOSSimTarget.empty()) {
      const Option *O = Opts.getOption(
        options::OPT_mios_simulator_version_min_EQ);
      iOSSimVersion = Args.MakeJoinedArg(0, O, iOSSimTarget);
      Args.append(iOSSimVersion);
    } else {
      // Otherwise, assume we are targeting OS X.
      const Option *O = Opts.getOption(options::OPT_mmacosx_version_min_EQ);
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
    if (!Driver::GetReleaseVersion(OSXVersion->getValue(Args), Major, Minor,
                                   Micro, HadExtra) || HadExtra ||
        Major != 10 || Minor >= 100 || Micro >= 100)
      getDriver().Diag(diag::err_drv_invalid_version_number)
        << OSXVersion->getAsString(Args);
  } else {
    const Arg *Version = iOSVersion ? iOSVersion : iOSSimVersion;
    assert(Version && "Unknown target platform!");
    if (!Driver::GetReleaseVersion(Version->getValue(Args), Major, Minor,
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
    bool Exists;
    if (const Arg *A = Args.getLastArg(options::OPT_isysroot)) {
      llvm::sys::Path P(A->getValue(Args));
      P.appendComponent("usr");
      P.appendComponent("lib");
      P.appendComponent("libstdc++.dylib");

      if (llvm::sys::fs::exists(P.str(), Exists) || !Exists) {
        P.eraseComponent();
        P.appendComponent("libstdc++.6.dylib");
        if (!llvm::sys::fs::exists(P.str(), Exists) && Exists) {
          CmdArgs.push_back(Args.MakeArgString(P.str()));
          return;
        }
      }
    }

    // Otherwise, look in the root.
    // FIXME: This should be removed someday when we don't have to care about
    // 10.6 and earlier, where /usr/lib/libstdc++.dylib does not exist.
    if ((llvm::sys::fs::exists("/usr/lib/libstdc++.dylib", Exists) || !Exists)&&
      (!llvm::sys::fs::exists("/usr/lib/libstdc++.6.dylib", Exists) && Exists)){
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

  llvm::sys::Path P(getDriver().ResourceDir);
  P.appendComponent("lib");
  P.appendComponent("darwin");
  P.appendComponent("libclang_rt.cc_kext.a");

  // For now, allow missing resource libraries to support developers who may
  // not have compiler-rt checked out or integrated into their build.
  bool Exists;
  if (!llvm::sys::fs::exists(P.str(), Exists) && Exists)
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
      //
      // FIXME: Canonicalize name.
      StringRef XarchArch = A->getValue(Args, 0);
      if (!(XarchArch == getArchName()  ||
            (BoundArch && XarchArch == BoundArch)))
        continue;

      Arg *OriginalArg = A;
      unsigned Index = Args.getBaseArgs().MakeIndex(A->getValue(Args, 1));
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
      } else if (XarchArg->getOption().isDriverOption()) {
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
      if (A->getOption().isLinkerInput()) {
        // Convert the argument into individual Zlinker_input_args.
        for (unsigned i = 0, e = A->getNumValues(); i != e; ++i) {
          DAL->AddSeparateArg(OriginalArg,
                              Opts.getOption(options::OPT_Zlinker_input),
                              A->getValue(Args, i));

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
                          A->getValue(Args));
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
    const Option *MCpu = Opts.getOption(options::OPT_mcpu_EQ);
    const Option *MArch = Opts.getOption(options::OPT_march_EQ);

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

    else if (Name == "ppc64")
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
    else if (Name == "armv7")
      DAL->AddJoinedArg(0, MArch, "armv7a");

    else
      llvm_unreachable("invalid Darwin arch");
  }

  // Add an explicit version min argument for the deployment target. We do this
  // after argument translation because -Xarch_ arguments may add a version min
  // argument.
  AddDeploymentTarget(*DAL);

  // Validate the C++ standard library choice.
  CXXStdlibType Type = GetCXXStdlibType(*DAL);
  if (Type == ToolChain::CST_Libcxx) {
    switch (LibCXXForSimulator) {
    case LibCXXSimulator_None:
      // Handle non-simulator cases.
      if (isTargetIPhoneOS()) {
        if (isIPhoneOSVersionLT(5, 0)) {
          getDriver().Diag(clang::diag::err_drv_invalid_libcxx_deployment)
            << "iOS 5.0";
        }
      }
      break;
    case LibCXXSimulator_NotAvailable:
      getDriver().Diag(clang::diag::err_drv_invalid_libcxx_deployment)
        << "iOS 5.0";
      break;
    case LibCXXSimulator_Available:
      break;
    }
  }

  return DAL;
}

bool Darwin::IsUnwindTablesDefault() const {
  // FIXME: Gross; we should probably have some separate target
  // definition, possibly even reusing the one in clang.
  return getArchName() == "x86_64";
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

const char *Darwin::GetDefaultRelocationModel() const {
  return "pic";
}

const char *Darwin::GetForcedPicModel() const {
  if (getArchName() == "x86_64")
    return "pic";
  return 0;
}

bool Darwin::SupportsProfiling() const {
  // Profiling instrumentation is only supported on x86.
  return getArchName() == "i386" || getArchName() == "x86_64";
}

bool Darwin::SupportsObjCGC() const {
  // Garbage collection is supported everywhere except on iPhone OS.
  return !isTargetIPhoneOS();
}

bool Darwin::SupportsObjCARC() const {
  return isTargetIPhoneOS() || !isMacosxVersionLT(10, 6);
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
  const GCCVersion BadVersion = { VersionText.str(), -1, -1, -1, "" };
  std::pair<StringRef, StringRef> First = VersionText.split('.');
  std::pair<StringRef, StringRef> Second = First.second.split('.');

  GCCVersion GoodVersion = { VersionText.str(), -1, -1, -1, "" };
  if (First.first.getAsInteger(10, GoodVersion.Major) ||
      GoodVersion.Major < 0)
    return BadVersion;
  if (Second.first.getAsInteger(10, GoodVersion.Minor) ||
      GoodVersion.Minor < 0)
    return BadVersion;

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
    if (unsigned EndNumber = PatchText.find_first_not_of("0123456789")) {
      // Try to parse the number and any suffix.
      if (PatchText.slice(0, EndNumber).getAsInteger(10, GoodVersion.Patch) ||
          GoodVersion.Patch < 0)
        return BadVersion;
      GoodVersion.PatchSuffix = PatchText.substr(EndNumber).str();
    }
  }

  return GoodVersion;
}

/// \brief Less-than for GCCVersion, implementing a Strict Weak Ordering.
bool Generic_GCC::GCCVersion::operator<(const GCCVersion &RHS) const {
  if (Major < RHS.Major) return true; if (Major > RHS.Major) return false;
  if (Minor < RHS.Minor) return true; if (Minor > RHS.Minor) return false;

  // Note that we rank versions with *no* patch specified is better than ones
  // hard-coding a patch version. Thus if the RHS has no patch, it always
  // wins, and the LHS only wins if it has no patch and the RHS does have
  // a patch.
  if (RHS.Patch == -1) return true;   if (Patch == -1) return false;
  if (Patch < RHS.Patch) return true; if (Patch > RHS.Patch) return false;
  if (PatchSuffix == RHS.PatchSuffix) return false;

  // Finally, between completely tied version numbers, the version with the
  // suffix loses as we prefer full releases.
  if (RHS.PatchSuffix.empty()) return true;
  return false;
}

static StringRef getGCCToolchainDir(const ArgList &Args) {
  const Arg *A = Args.getLastArg(options::OPT_gcc_toolchain);
  if (A)
    return A->getValue(Args);
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
    const Driver &D,
    const llvm::Triple &TargetTriple,
    const ArgList &Args)
    : IsValid(false) {
  llvm::Triple MultiarchTriple
    = TargetTriple.isArch32Bit() ? TargetTriple.get64BitArchVariant()
                                 : TargetTriple.get32BitArchVariant();
  llvm::Triple::ArchType TargetArch = TargetTriple.getArch();
  // The library directories which may contain GCC installations.
  SmallVector<StringRef, 4> CandidateLibDirs, CandidateMultiarchLibDirs;
  // The compatible GCC triples for this particular architecture.
  SmallVector<StringRef, 10> CandidateTripleAliases;
  SmallVector<StringRef, 10> CandidateMultiarchTripleAliases;
  CollectLibDirsAndTriples(TargetTriple, MultiarchTriple, CandidateLibDirs,
                           CandidateTripleAliases,
                           CandidateMultiarchLibDirs,
                           CandidateMultiarchTripleAliases);

  // Compute the set of prefixes for our search.
  SmallVector<std::string, 8> Prefixes(D.PrefixDirs.begin(),
                                       D.PrefixDirs.end());

  StringRef GCCToolchainDir = getGCCToolchainDir(Args);
  if (GCCToolchainDir != "") {
    if (GCCToolchainDir.back() == '/')
      GCCToolchainDir = GCCToolchainDir.drop_back(); // remove the /

    Prefixes.push_back(GCCToolchainDir);
  } else {
    Prefixes.push_back(D.SysRoot);
    Prefixes.push_back(D.SysRoot + "/usr");
    Prefixes.push_back(D.InstalledDir + "/..");
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
        ScanLibDirForGCCTriple(TargetArch, LibDir, CandidateTripleAliases[k]);
    }
    for (unsigned j = 0, je = CandidateMultiarchLibDirs.size(); j < je; ++j) {
      const std::string LibDir
        = Prefixes[i] + CandidateMultiarchLibDirs[j].str();
      if (!llvm::sys::fs::exists(LibDir))
        continue;
      for (unsigned k = 0, ke = CandidateMultiarchTripleAliases.size(); k < ke;
           ++k)
        ScanLibDirForGCCTriple(TargetArch, LibDir,
                               CandidateMultiarchTripleAliases[k],
                               /*NeedsMultiarchSuffix=*/true);
    }
  }
}

/*static*/ void Generic_GCC::GCCInstallationDetector::CollectLibDirsAndTriples(
    const llvm::Triple &TargetTriple,
    const llvm::Triple &MultiarchTriple,
    SmallVectorImpl<StringRef> &LibDirs,
    SmallVectorImpl<StringRef> &TripleAliases,
    SmallVectorImpl<StringRef> &MultiarchLibDirs,
    SmallVectorImpl<StringRef> &MultiarchTripleAliases) {
  // Declare a bunch of static data sets that we'll select between below. These
  // are specifically designed to always refer to string literals to avoid any
  // lifetime or initialization issues.
  static const char *const ARMLibDirs[] = { "/lib" };
  static const char *const ARMTriples[] = {
    "arm-linux-gnueabi",
    "arm-linux-androideabi"
  };

  static const char *const X86_64LibDirs[] = { "/lib64", "/lib" };
  static const char *const X86_64Triples[] = {
    "x86_64-linux-gnu",
    "x86_64-unknown-linux-gnu",
    "x86_64-pc-linux-gnu",
    "x86_64-redhat-linux6E",
    "x86_64-redhat-linux",
    "x86_64-suse-linux",
    "x86_64-manbo-linux-gnu",
    "x86_64-linux-gnu",
    "x86_64-slackware-linux"
  };
  static const char *const X86LibDirs[] = { "/lib32", "/lib" };
  static const char *const X86Triples[] = {
    "i686-linux-gnu",
    "i686-pc-linux-gnu",
    "i486-linux-gnu",
    "i386-linux-gnu",
    "i686-redhat-linux",
    "i586-redhat-linux",
    "i386-redhat-linux",
    "i586-suse-linux",
    "i486-slackware-linux"
  };

  static const char *const MIPSLibDirs[] = { "/lib" };
  static const char *const MIPSTriples[] = { "mips-linux-gnu" };
  static const char *const MIPSELLibDirs[] = { "/lib" };
  static const char *const MIPSELTriples[] = { "mipsel-linux-gnu" };

  static const char *const PPCLibDirs[] = { "/lib32", "/lib" };
  static const char *const PPCTriples[] = {
    "powerpc-linux-gnu",
    "powerpc-unknown-linux-gnu",
    "powerpc-suse-linux"
  };
  static const char *const PPC64LibDirs[] = { "/lib64", "/lib" };
  static const char *const PPC64Triples[] = {
    "powerpc64-linux-gnu",
    "powerpc64-unknown-linux-gnu",
    "powerpc64-suse-linux",
    "ppc64-redhat-linux"
  };

  switch (TargetTriple.getArch()) {
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    LibDirs.append(ARMLibDirs, ARMLibDirs + llvm::array_lengthof(ARMLibDirs));
    TripleAliases.append(
      ARMTriples, ARMTriples + llvm::array_lengthof(ARMTriples));
    break;
  case llvm::Triple::x86_64:
    LibDirs.append(
      X86_64LibDirs, X86_64LibDirs + llvm::array_lengthof(X86_64LibDirs));
    TripleAliases.append(
      X86_64Triples, X86_64Triples + llvm::array_lengthof(X86_64Triples));
    MultiarchLibDirs.append(
      X86LibDirs, X86LibDirs + llvm::array_lengthof(X86LibDirs));
    MultiarchTripleAliases.append(
      X86Triples, X86Triples + llvm::array_lengthof(X86Triples));
    break;
  case llvm::Triple::x86:
    LibDirs.append(X86LibDirs, X86LibDirs + llvm::array_lengthof(X86LibDirs));
    TripleAliases.append(
      X86Triples, X86Triples + llvm::array_lengthof(X86Triples));
    MultiarchLibDirs.append(
      X86_64LibDirs, X86_64LibDirs + llvm::array_lengthof(X86_64LibDirs));
    MultiarchTripleAliases.append(
      X86_64Triples, X86_64Triples + llvm::array_lengthof(X86_64Triples));
    break;
  case llvm::Triple::mips:
    LibDirs.append(
      MIPSLibDirs, MIPSLibDirs + llvm::array_lengthof(MIPSLibDirs));
    TripleAliases.append(
      MIPSTriples, MIPSTriples + llvm::array_lengthof(MIPSTriples));
    break;
  case llvm::Triple::mipsel:
    LibDirs.append(
      MIPSELLibDirs, MIPSELLibDirs + llvm::array_lengthof(MIPSELLibDirs));
    TripleAliases.append(
      MIPSELTriples, MIPSELTriples + llvm::array_lengthof(MIPSELTriples));
    break;
  case llvm::Triple::ppc:
    LibDirs.append(PPCLibDirs, PPCLibDirs + llvm::array_lengthof(PPCLibDirs));
    TripleAliases.append(
      PPCTriples, PPCTriples + llvm::array_lengthof(PPCTriples));
    MultiarchLibDirs.append(
      PPC64LibDirs, PPC64LibDirs + llvm::array_lengthof(PPC64LibDirs));
    MultiarchTripleAliases.append(
      PPC64Triples, PPC64Triples + llvm::array_lengthof(PPC64Triples));
    break;
  case llvm::Triple::ppc64:
    LibDirs.append(
      PPC64LibDirs, PPC64LibDirs + llvm::array_lengthof(PPC64LibDirs));
    TripleAliases.append(
      PPC64Triples, PPC64Triples + llvm::array_lengthof(PPC64Triples));
    MultiarchLibDirs.append(
      PPCLibDirs, PPCLibDirs + llvm::array_lengthof(PPCLibDirs));
    MultiarchTripleAliases.append(
      PPCTriples, PPCTriples + llvm::array_lengthof(PPCTriples));
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
  if (TargetTriple.str() != MultiarchTriple.str())
    MultiarchTripleAliases.push_back(MultiarchTriple.str());
}

void Generic_GCC::GCCInstallationDetector::ScanLibDirForGCCTriple(
    llvm::Triple::ArchType TargetArch, const std::string &LibDir,
    StringRef CandidateTriple, bool NeedsMultiarchSuffix) {
  // There are various different suffixes involving the triple we
  // check for. We also record what is necessary to walk from each back
  // up to the lib directory.
  const std::string LibSuffixes[] = {
    "/gcc/" + CandidateTriple.str(),
    "/" + CandidateTriple.str() + "/gcc/" + CandidateTriple.str(),

    // Ubuntu has a strange mis-matched pair of triples that this happens to
    // match.
    // FIXME: It may be worthwhile to generalize this and look for a second
    // triple.
    "/i386-linux-gnu/gcc/" + CandidateTriple.str()
  };
  const std::string InstallSuffixes[] = {
    "/../../..",
    "/../../../..",
    "/../../../.."
  };
  // Only look at the final, weird Ubuntu suffix for i386-linux-gnu.
  const unsigned NumLibSuffixes = (llvm::array_lengthof(LibSuffixes) -
                                   (TargetArch != llvm::Triple::x86));
  for (unsigned i = 0; i < NumLibSuffixes; ++i) {
    StringRef LibSuffix = LibSuffixes[i];
    llvm::error_code EC;
    for (llvm::sys::fs::directory_iterator LI(LibDir + LibSuffix, EC), LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef VersionText = llvm::sys::path::filename(LI->path());
      GCCVersion CandidateVersion = GCCVersion::Parse(VersionText);
      static const GCCVersion MinVersion = { "4.1.1", 4, 1, 1, "" };
      if (CandidateVersion < MinVersion)
        continue;
      if (CandidateVersion <= Version)
        continue;

      // Some versions of SUSE and Fedora on ppc64 put 32-bit libs
      // in what would normally be GCCInstallPath and put the 64-bit
      // libs in a subdirectory named 64. The simple logic we follow is that
      // *if* there is a subdirectory of the right name with crtbegin.o in it,
      // we use that. If not, and if not a multiarch triple, we look for
      // crtbegin.o without the subdirectory.
      StringRef MultiarchSuffix
        = (TargetArch == llvm::Triple::x86_64 ||
           TargetArch == llvm::Triple::ppc64) ? "/64" : "/32";
      if (llvm::sys::fs::exists(LI->path() + MultiarchSuffix + "/crtbegin.o")) {
        GCCMultiarchSuffix = MultiarchSuffix.str();
      } else {
        if (NeedsMultiarchSuffix ||
            !llvm::sys::fs::exists(LI->path() + "/crtbegin.o"))
          continue;
        GCCMultiarchSuffix.clear();
      }

      Version = CandidateVersion;
      GCCTriple.setTriple(CandidateTriple);
      // FIXME: We hack together the directory name here instead of
      // using LI to ensure stable path separators across Windows and
      // Linux.
      GCCInstallPath = LibDir + LibSuffixes[i] + "/" + VersionText.str();
      GCCParentLibPath = GCCInstallPath + InstallSuffixes[i];
      IsValid = true;
    }
  }
}

Generic_GCC::Generic_GCC(const Driver &D, const llvm::Triple& Triple,
                         const ArgList &Args)
  : ToolChain(D, Triple), GCCInstallation(getDriver(), Triple, Args) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir)
    getProgramPaths().push_back(getDriver().Dir);
}

Generic_GCC::~Generic_GCC() {
  // Free tool implementations.
  for (llvm::DenseMap<unsigned, Tool*>::iterator
         it = Tools.begin(), ie = Tools.end(); it != ie; ++it)
    delete it->second;
}

Tool &Generic_GCC::SelectTool(const Compilation &C,
                              const JobAction &JA,
                              const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::InputClass:
    case Action::BindArchClass:
      llvm_unreachable("Invalid tool kind.");
    case Action::PreprocessJobClass:
      T = new tools::gcc::Preprocess(*this); break;
    case Action::PrecompileJobClass:
      T = new tools::gcc::Precompile(*this); break;
    case Action::AnalyzeJobClass:
    case Action::MigrateJobClass:
      T = new tools::Clang(*this); break;
    case Action::CompileJobClass:
      T = new tools::gcc::Compile(*this); break;
    case Action::AssembleJobClass:
      T = new tools::gcc::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::gcc::Link(*this); break;

      // This is a bit ungeneric, but the only platform using a driver
      // driver is Darwin.
    case Action::LipoJobClass:
      T = new tools::darwin::Lipo(*this); break;
    case Action::DsymutilJobClass:
      T = new tools::darwin::Dsymutil(*this); break;
    case Action::VerifyJobClass:
      T = new tools::darwin::VerifyDebug(*this); break;
    }
  }

  return *T;
}

bool Generic_GCC::IsUnwindTablesDefault() const {
  // FIXME: Gross; we should probably have some separate target
  // definition, possibly even reusing the one in clang.
  return getArchName() == "x86_64";
}

const char *Generic_GCC::GetDefaultRelocationModel() const {
  return "static";
}

const char *Generic_GCC::GetForcedPicModel() const {
  return 0;
}
/// Hexagon Toolchain

Hexagon_TC::Hexagon_TC(const Driver &D, const llvm::Triple& Triple)
  : ToolChain(D, Triple) {
  getProgramPaths().push_back(getDriver().getInstalledDir());
  if (getDriver().getInstalledDir() != getDriver().Dir.c_str())
    getProgramPaths().push_back(getDriver().Dir);
}

Hexagon_TC::~Hexagon_TC() {
  // Free tool implementations.
  for (llvm::DenseMap<unsigned, Tool*>::iterator
         it = Tools.begin(), ie = Tools.end(); it != ie; ++it)
    delete it->second;
}

Tool &Hexagon_TC::SelectTool(const Compilation &C,
                             const JobAction &JA,
                             const ActionList &Inputs) const {
  Action::ActionClass Key;
  //   if (JA.getKind () == Action::CompileJobClass)
  //     Key = JA.getKind ();
  //     else

  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();
  //   if ((JA.getKind () == Action::CompileJobClass)
  //     && (JA.getType () != types::TY_LTO_BC)) {
  //     Key = JA.getKind ();
  //   }

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::InputClass:
    case Action::BindArchClass:
      assert(0 && "Invalid tool kind.");
    case Action::AnalyzeJobClass:
      T = new tools::Clang(*this); break;
    case Action::AssembleJobClass:
      T = new tools::hexagon::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::hexagon::Link(*this); break;
    default:
      assert(false && "Unsupported action for Hexagon target.");
    }
  }

  return *T;
}

bool Hexagon_TC::IsUnwindTablesDefault() const {
  // FIXME: Gross; we should probably have some separate target
  // definition, possibly even reusing the one in clang.
  return getArchName() == "x86_64";
}

const char *Hexagon_TC::GetDefaultRelocationModel() const {
  return "static";
}

const char *Hexagon_TC::GetForcedPicModel() const {
  return 0;
} // End Hexagon


/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
/// Currently does not support anything else but compilation.

TCEToolChain::TCEToolChain(const Driver &D, const llvm::Triple& Triple)
  : ToolChain(D, Triple) {
  // Path mangling to find libexec
  std::string Path(getDriver().Dir);

  Path += "/../libexec";
  getProgramPaths().push_back(Path);
}

TCEToolChain::~TCEToolChain() {
  for (llvm::DenseMap<unsigned, Tool*>::iterator
           it = Tools.begin(), ie = Tools.end(); it != ie; ++it)
      delete it->second;
}

bool TCEToolChain::IsMathErrnoDefault() const {
  return true;
}

bool TCEToolChain::IsUnwindTablesDefault() const {
  return false;
}

const char *TCEToolChain::GetDefaultRelocationModel() const {
  return "static";
}

const char *TCEToolChain::GetForcedPicModel() const {
  return 0;
}

Tool &TCEToolChain::SelectTool(const Compilation &C,
                            const JobAction &JA,
                               const ActionList &Inputs) const {
  Action::ActionClass Key;
  Key = Action::AnalyzeJobClass;

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::PreprocessJobClass:
      T = new tools::gcc::Preprocess(*this); break;
    case Action::AnalyzeJobClass:
      T = new tools::Clang(*this); break;
    default:
     llvm_unreachable("Unsupported action for TCE target.");
    }
  }
  return *T;
}

/// OpenBSD - OpenBSD tool chain which can call as(1) and ld(1) directly.

OpenBSD::OpenBSD(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool &OpenBSD::SelectTool(const Compilation &C, const JobAction &JA,
                          const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  bool UseIntegratedAs = C.getArgs().hasFlag(options::OPT_integrated_as,
                                             options::OPT_no_integrated_as,
                                             IsIntegratedAssemblerDefault());

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass: {
      if (UseIntegratedAs)
        T = new tools::ClangAs(*this);
      else
        T = new tools::openbsd::Assemble(*this);
      break;
    }
    case Action::LinkJobClass:
      T = new tools::openbsd::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
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

Tool &FreeBSD::SelectTool(const Compilation &C, const JobAction &JA,
                          const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  bool UseIntegratedAs = C.getArgs().hasFlag(options::OPT_integrated_as,
                                             options::OPT_no_integrated_as,
                                             IsIntegratedAssemblerDefault());

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      if (UseIntegratedAs)
        T = new tools::ClangAs(*this);
      else
        T = new tools::freebsd::Assemble(*this);
      break;
    case Action::LinkJobClass:
      T = new tools::freebsd::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
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

Tool &NetBSD::SelectTool(const Compilation &C, const JobAction &JA,
                         const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  bool UseIntegratedAs = C.getArgs().hasFlag(options::OPT_integrated_as,
                                             options::OPT_no_integrated_as,
                                             IsIntegratedAssemblerDefault());

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      if (UseIntegratedAs)
        T = new tools::ClangAs(*this);
      else
        T = new tools::netbsd::Assemble(*this);
      break;
    case Action::LinkJobClass:
      T = new tools::netbsd::Link(*this);
      break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
}

/// Minix - Minix tool chain which can call as(1) and ld(1) directly.

Minix::Minix(const Driver &D, const llvm::Triple& Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

Tool &Minix::SelectTool(const Compilation &C, const JobAction &JA,
                        const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      T = new tools::minix::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::minix::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
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

Tool &AuroraUX::SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      T = new tools::auroraux::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::auroraux::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
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

Tool &Solaris::SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      T = new tools::solaris::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::solaris::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
}

/// Linux toolchain (very bare-bones at the moment).

enum LinuxDistro {
  ArchLinux,
  DebianLenny,
  DebianSqueeze,
  DebianWheezy,
  Exherbo,
  RHEL4,
  RHEL5,
  RHEL6,
  Fedora13,
  Fedora14,
  Fedora15,
  Fedora16,
  FedoraRawhide,
  OpenSuse11_3,
  OpenSuse11_4,
  OpenSuse12_1,
  OpenSuse12_2,
  UbuntuHardy,
  UbuntuIntrepid,
  UbuntuJaunty,
  UbuntuKarmic,
  UbuntuLucid,
  UbuntuMaverick,
  UbuntuNatty,
  UbuntuOneiric,
  UbuntuPrecise,
  UnknownDistro
};

static bool IsRedhat(enum LinuxDistro Distro) {
  return (Distro >= Fedora13 && Distro <= FedoraRawhide) ||
         (Distro >= RHEL4    && Distro <= RHEL6);
}

static bool IsOpenSuse(enum LinuxDistro Distro) {
  return Distro >= OpenSuse11_3 && Distro <= OpenSuse12_2;
}

static bool IsDebian(enum LinuxDistro Distro) {
  return Distro >= DebianLenny && Distro <= DebianWheezy;
}

static bool IsUbuntu(enum LinuxDistro Distro) {
  return Distro >= UbuntuHardy && Distro <= UbuntuPrecise;
}

static LinuxDistro DetectLinuxDistro(llvm::Triple::ArchType Arch) {
  OwningPtr<llvm::MemoryBuffer> File;
  if (!llvm::MemoryBuffer::getFile("/etc/lsb-release", File)) {
    StringRef Data = File.get()->getBuffer();
    SmallVector<StringRef, 8> Lines;
    Data.split(Lines, "\n");
    LinuxDistro Version = UnknownDistro;
    for (unsigned i = 0, s = Lines.size(); i != s; ++i)
      if (Version == UnknownDistro && Lines[i].startswith("DISTRIB_CODENAME="))
        Version = llvm::StringSwitch<LinuxDistro>(Lines[i].substr(17))
          .Case("hardy", UbuntuHardy)
          .Case("intrepid", UbuntuIntrepid)
          .Case("jaunty", UbuntuJaunty)
          .Case("karmic", UbuntuKarmic)
          .Case("lucid", UbuntuLucid)
          .Case("maverick", UbuntuMaverick)
          .Case("natty", UbuntuNatty)
          .Case("oneiric", UbuntuOneiric)
          .Case("precise", UbuntuPrecise)
          .Default(UnknownDistro);
    return Version;
  }

  if (!llvm::MemoryBuffer::getFile("/etc/redhat-release", File)) {
    StringRef Data = File.get()->getBuffer();
    if (Data.startswith("Fedora release 16"))
      return Fedora16;
    else if (Data.startswith("Fedora release 15"))
      return Fedora15;
    else if (Data.startswith("Fedora release 14"))
      return Fedora14;
    else if (Data.startswith("Fedora release 13"))
      return Fedora13;
    else if (Data.startswith("Fedora release") &&
             Data.find("Rawhide") != StringRef::npos)
      return FedoraRawhide;
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
    return UnknownDistro;
  }

  if (!llvm::MemoryBuffer::getFile("/etc/SuSE-release", File))
    return llvm::StringSwitch<LinuxDistro>(File.get()->getBuffer())
      .StartsWith("openSUSE 11.3", OpenSuse11_3)
      .StartsWith("openSUSE 11.4", OpenSuse11_4)
      .StartsWith("openSUSE 12.1", OpenSuse12_1)
      .StartsWith("openSUSE 12.2", OpenSuse12_2)
      .Default(UnknownDistro);

  bool Exists;
  if (!llvm::sys::fs::exists("/etc/exherbo-release", Exists) && Exists)
    return Exherbo;

  if (!llvm::sys::fs::exists("/etc/arch-release", Exists) && Exists)
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
  case llvm::Triple::x86:
    if (llvm::sys::fs::exists(SysRoot + "/lib/i386-linux-gnu"))
      return "i386-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::x86_64:
    if (llvm::sys::fs::exists(SysRoot + "/lib/x86_64-linux-gnu"))
      return "x86_64-linux-gnu";
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
    if (llvm::sys::fs::exists(SysRoot + "/lib/powerpc-linux-gnu"))
      return "powerpc-linux-gnu";
    return TargetTriple.str();
  case llvm::Triple::ppc64:
    if (llvm::sys::fs::exists(SysRoot + "/lib/powerpc64-linux-gnu"))
      return "powerpc64-linux-gnu";
    return TargetTriple.str();
  }
}

static void addPathIfExists(Twine Path, ToolChain::path_list &Paths) {
  if (llvm::sys::fs::exists(Path)) Paths.push_back(Path.str());
}

Linux::Linux(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
  : Generic_ELF(D, Triple, Args) {
  llvm::Triple::ArchType Arch = Triple.getArch();
  const std::string &SysRoot = getDriver().SysRoot;

  // OpenSuse stores the linker with the compiler, add that to the search
  // path.
  ToolChain::path_list &PPaths = getProgramPaths();
  PPaths.push_back(Twine(GCCInstallation.getParentLibPath() + "/../" +
                         GCCInstallation.getTriple().str() + "/bin").str());

  Linker = GetProgramPath("ld");

  LinuxDistro Distro = DetectLinuxDistro(Arch);

  if (IsOpenSuse(Distro) || IsUbuntu(Distro)) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("relro");
  }

  if (Arch == llvm::Triple::arm || Arch == llvm::Triple::thumb)
    ExtraOpts.push_back("-X");

  const bool IsMips = Arch == llvm::Triple::mips ||
                      Arch == llvm::Triple::mipsel ||
                      Arch == llvm::Triple::mips64 ||
                      Arch == llvm::Triple::mips64el;

  const bool IsAndroid = Triple.getEnvironment() == llvm::Triple::ANDROIDEABI;

  // Do not use 'gnu' hash style for Mips targets because .gnu.hash
  // and the MIPS ABI require .dynsym to be sorted in different ways.
  // .gnu.hash needs symbols to be grouped by hash code whereas the MIPS
  // ABI requires a mapping between the GOT and the symbol table.
  // Android loader does not support .gnu.hash.
  if (!IsMips && !IsAndroid) {
    if (IsRedhat(Distro) || IsOpenSuse(Distro) ||
        (IsUbuntu(Distro) && Distro >= UbuntuMaverick))
      ExtraOpts.push_back("--hash-style=gnu");

    if (IsDebian(Distro) || IsOpenSuse(Distro) || Distro == UbuntuLucid ||
        Distro == UbuntuJaunty || Distro == UbuntuKarmic)
      ExtraOpts.push_back("--hash-style=both");
  }

  if (IsRedhat(Distro))
    ExtraOpts.push_back("--no-add-needed");

  if (Distro == DebianSqueeze || Distro == DebianWheezy ||
      IsOpenSuse(Distro) ||
      (IsRedhat(Distro) && Distro != RHEL4 && Distro != RHEL5) ||
      (IsUbuntu(Distro) && Distro >= UbuntuKarmic))
    ExtraOpts.push_back("--build-id");

  if (IsOpenSuse(Distro))
    ExtraOpts.push_back("--enable-new-dtags");

  // The selection of paths to try here is designed to match the patterns which
  // the GCC driver itself uses, as this is part of the GCC-compatible driver.
  // This was determined by running GCC in a fake filesystem, creating all
  // possible permutations of these directories, and seeing which ones it added
  // to the link paths.
  path_list &Paths = getFilePaths();

  const std::string Multilib = Triple.isArch32Bit() ? "lib32" : "lib64";
  const std::string MultiarchTriple = getMultiarchTriple(Triple, SysRoot);

  // Add the multilib suffixed paths where they are available.
  if (GCCInstallation.isValid()) {
    const llvm::Triple &GCCTriple = GCCInstallation.getTriple();
    const std::string &LibPath = GCCInstallation.getParentLibPath();
    addPathIfExists((GCCInstallation.getInstallPath() +
                     GCCInstallation.getMultiarchSuffix()),
                    Paths);

    // If the GCC installation we found is inside of the sysroot, we want to
    // prefer libraries installed in the parent prefix of the GCC installation.
    // It is important to *not* use these paths when the GCC installation is
    // outside of the system root as that can pick up unintended libraries.
    // This usually happens when there is an external cross compiler on the
    // host system, and a more minimal sysroot available that is the target of
    // the cross.
    if (StringRef(LibPath).startswith(SysRoot)) {
      addPathIfExists(LibPath + "/../" + GCCTriple.str() + "/lib/../" + Multilib,
                      Paths);
      addPathIfExists(LibPath + "/" + MultiarchTriple, Paths);
      addPathIfExists(LibPath + "/../" + Multilib, Paths);
    }
  }
  addPathIfExists(SysRoot + "/lib/" + MultiarchTriple, Paths);
  addPathIfExists(SysRoot + "/lib/../" + Multilib, Paths);
  addPathIfExists(SysRoot + "/usr/lib/" + MultiarchTriple, Paths);
  addPathIfExists(SysRoot + "/usr/lib/../" + Multilib, Paths);

  // Try walking via the GCC triple path in case of multiarch GCC
  // installations with strange symlinks.
  if (GCCInstallation.isValid())
    addPathIfExists(SysRoot + "/usr/lib/" + GCCInstallation.getTriple().str() +
                    "/../../" + Multilib, Paths);

  // Add the non-multilib suffixed paths (if potentially different).
  if (GCCInstallation.isValid()) {
    const std::string &LibPath = GCCInstallation.getParentLibPath();
    const llvm::Triple &GCCTriple = GCCInstallation.getTriple();
    if (!GCCInstallation.getMultiarchSuffix().empty())
      addPathIfExists(GCCInstallation.getInstallPath(), Paths);

    if (StringRef(LibPath).startswith(SysRoot)) {
      addPathIfExists(LibPath + "/../" + GCCTriple.str() + "/lib", Paths);
      addPathIfExists(LibPath, Paths);
    }
  }
  addPathIfExists(SysRoot + "/lib", Paths);
  addPathIfExists(SysRoot + "/usr/lib", Paths);
}

bool Linux::HasNativeLLVMSupport() const {
  return true;
}

Tool &Linux::SelectTool(const Compilation &C, const JobAction &JA,
                        const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  bool UseIntegratedAs = C.getArgs().hasFlag(options::OPT_integrated_as,
                                             options::OPT_no_integrated_as,
                                             IsIntegratedAssemblerDefault());

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      if (UseIntegratedAs)
        T = new tools::ClangAs(*this);
      else
        T = new tools::linuxtools::Assemble(*this);
      break;
    case Action::LinkJobClass:
      T = new tools::linuxtools::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
}

void Linux::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc))
    addSystemInclude(DriverArgs, CC1Args, D.SysRoot + "/usr/local/include");

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    llvm::sys::Path P(D.ResourceDir);
    P.appendComponent("include");
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
      StringRef Prefix = llvm::sys::path::is_absolute(*I) ? D.SysRoot : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + *I);
    }
    return;
  }

  // Lacking those, try to detect the correct set of system includes for the
  // target triple.

  // Implement generic Debian multiarch support.
  const StringRef X86_64MultiarchIncludeDirs[] = {
    "/usr/include/x86_64-linux-gnu",

    // FIXME: These are older forms of multiarch. It's not clear that they're
    // in use in any released version of Debian, so we should consider
    // removing them.
    "/usr/include/i686-linux-gnu/64",
    "/usr/include/i486-linux-gnu/64"
  };
  const StringRef X86MultiarchIncludeDirs[] = {
    "/usr/include/i386-linux-gnu",

    // FIXME: These are older forms of multiarch. It's not clear that they're
    // in use in any released version of Debian, so we should consider
    // removing them.
    "/usr/include/x86_64-linux-gnu/32",
    "/usr/include/i686-linux-gnu",
    "/usr/include/i486-linux-gnu"
  };
  const StringRef ARMMultiarchIncludeDirs[] = {
    "/usr/include/arm-linux-gnueabi"
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
  } else if (getTriple().getArch() == llvm::Triple::arm) {
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
    if (llvm::sys::fs::exists(D.SysRoot + *I)) {
      addExternCSystemInclude(DriverArgs, CC1Args, D.SysRoot + *I);
      break;
    }
  }

  if (getTriple().getOS() == llvm::Triple::RTEMS)
    return;

  // Add an include of '/include' directly. This isn't provided by default by
  // system GCCs, but is often used with cross-compiling GCCs, and harmless to
  // add even when Clang is acting as-if it were a system compiler.
  addExternCSystemInclude(DriverArgs, CC1Args, D.SysRoot + "/include");

  addExternCSystemInclude(DriverArgs, CC1Args, D.SysRoot + "/usr/include");
}

/// \brief Helper to add the thre variant paths for a libstdc++ installation.
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
  StringRef Version = GCCInstallation.getVersion();
  if (!addLibStdCXXIncludePaths(LibDir + "/../include/c++/" + Version,
                                (GCCInstallation.getTriple().str() +
                                 GCCInstallation.getMultiarchSuffix()),
                                DriverArgs, CC1Args)) {
    // Gentoo is weird and places its headers inside the GCC install, so if the
    // first attempt to find the headers fails, try this pattern.
    addLibStdCXXIncludePaths(InstallDir + "/include/g++-v4",
                             (GCCInstallation.getTriple().str() +
                              GCCInstallation.getMultiarchSuffix()),
                             DriverArgs, CC1Args);
  }
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
  getFilePaths().push_back("/usr/lib/gcc41");
}

Tool &DragonFly::SelectTool(const Compilation &C, const JobAction &JA,
                            const ActionList &Inputs) const {
  Action::ActionClass Key;
  if (getDriver().ShouldUseClangCompiler(C, JA, getTriple()))
    Key = Action::AnalyzeJobClass;
  else
    Key = JA.getKind();

  Tool *&T = Tools[Key];
  if (!T) {
    switch (Key) {
    case Action::AssembleJobClass:
      T = new tools::dragonfly::Assemble(*this); break;
    case Action::LinkJobClass:
      T = new tools::dragonfly::Link(*this); break;
    default:
      T = &Generic_GCC::SelectTool(C, JA, Inputs);
    }
  }

  return *T;
}
