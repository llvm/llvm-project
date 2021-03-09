//===--- Linux.h - Linux ToolChain Implementations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Linux.h"
#include "Arch/ARM.h"
#include "Arch/Mips.h"
#include "Arch/PPC.h"
#include "Arch/RISCV.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Distro.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

using tools::addPathIfExists;

/// Get our best guess at the multiarch triple for a target.
///
/// Debian-based systems are starting to use a multiarch setup where they use
/// a target-triple directory in the library and header search paths.
/// Unfortunately, this triple does not align with the vanilla target triple,
/// so we provide a rough mapping here.
std::string Linux::getMultiarchTriple(const Driver &D,
                                      const llvm::Triple &TargetTriple,
                                      StringRef SysRoot) const {
  llvm::Triple::EnvironmentType TargetEnvironment =
      TargetTriple.getEnvironment();
  bool IsAndroid = TargetTriple.isAndroid();
  bool IsMipsR6 = TargetTriple.getSubArch() == llvm::Triple::MipsSubArch_r6;
  bool IsMipsN32Abi = TargetTriple.getEnvironment() == llvm::Triple::GNUABIN32;

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
    if (IsAndroid) {
      return "arm-linux-androideabi";
    } else if (TargetEnvironment == llvm::Triple::GNUEABIHF) {
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
    if (IsAndroid)
      return "i686-linux-android";
    if (D.getVFS().exists(SysRoot + "/lib/i386-linux-gnu"))
      return "i386-linux-gnu";
    break;
  case llvm::Triple::x86_64:
    if (IsAndroid)
      return "x86_64-linux-android";
    // We don't want this for x32, otherwise it will match x86_64 libs
    if (TargetEnvironment != llvm::Triple::GNUX32 &&
        D.getVFS().exists(SysRoot + "/lib/x86_64-linux-gnu"))
      return "x86_64-linux-gnu";
    break;
  case llvm::Triple::aarch64:
    if (IsAndroid)
      return "aarch64-linux-android";
    if (D.getVFS().exists(SysRoot + "/lib/aarch64-linux-gnu"))
      return "aarch64-linux-gnu";
    break;
  case llvm::Triple::aarch64_be:
    if (D.getVFS().exists(SysRoot + "/lib/aarch64_be-linux-gnu"))
      return "aarch64_be-linux-gnu";
    break;

  case llvm::Triple::m68k:
    if (D.getVFS().exists(SysRoot + "/lib/m68k-linux-gnu"))
      return "m68k-linux-gnu";
    break;

  case llvm::Triple::mips: {
    std::string MT = IsMipsR6 ? "mipsisa32r6-linux-gnu" : "mips-linux-gnu";
    if (D.getVFS().exists(SysRoot + "/lib/" + MT))
      return MT;
    break;
  }
  case llvm::Triple::mipsel: {
    if (IsAndroid)
      return "mipsel-linux-android";
    std::string MT = IsMipsR6 ? "mipsisa32r6el-linux-gnu" : "mipsel-linux-gnu";
    if (D.getVFS().exists(SysRoot + "/lib/" + MT))
      return MT;
    break;
  }
  case llvm::Triple::mips64: {
    std::string MT = std::string(IsMipsR6 ? "mipsisa64r6" : "mips64") +
                     "-linux-" + (IsMipsN32Abi ? "gnuabin32" : "gnuabi64");
    if (D.getVFS().exists(SysRoot + "/lib/" + MT))
      return MT;
    if (D.getVFS().exists(SysRoot + "/lib/mips64-linux-gnu"))
      return "mips64-linux-gnu";
    break;
  }
  case llvm::Triple::mips64el: {
    if (IsAndroid)
      return "mips64el-linux-android";
    std::string MT = std::string(IsMipsR6 ? "mipsisa64r6el" : "mips64el") +
                     "-linux-" + (IsMipsN32Abi ? "gnuabin32" : "gnuabi64");
    if (D.getVFS().exists(SysRoot + "/lib/" + MT))
      return MT;
    if (D.getVFS().exists(SysRoot + "/lib/mips64el-linux-gnu"))
      return "mips64el-linux-gnu";
    break;
  }
  case llvm::Triple::ppc:
    if (D.getVFS().exists(SysRoot + "/lib/powerpc-linux-gnuspe"))
      return "powerpc-linux-gnuspe";
    if (D.getVFS().exists(SysRoot + "/lib/powerpc-linux-gnu"))
      return "powerpc-linux-gnu";
    break;
  case llvm::Triple::ppcle:
    if (D.getVFS().exists(SysRoot + "/lib/powerpcle-linux-gnu"))
      return "powerpcle-linux-gnu";
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
  if (Triple.isMIPS()) {
    if (Triple.isAndroid()) {
      StringRef CPUName;
      StringRef ABIName;
      tools::mips::getMipsCPUAndABI(Args, Triple, CPUName, ABIName);
      if (CPUName == "mips32r6")
        return "libr6";
      if (CPUName == "mips32r2")
        return "libr2";
    }
    // lib32 directory has a special meaning on MIPS targets.
    // It contains N32 ABI binaries. Use this folder if produce
    // code for N32 ABI only.
    if (tools::mips::hasMipsAbiArg(Args, "n32"))
      return "lib32";
    return Triple.isArch32Bit() ? "lib" : "lib64";
  }

  // It happens that only x86, PPC and SPARC use the 'lib32' variant of
  // oslibdir, and using that variant while targeting other architectures causes
  // problems because the libraries are laid out in shared system roots that
  // can't cope with a 'lib32' library search path being considered. So we only
  // enable them when we know we may need it.
  //
  // FIXME: This is a bit of a hack. We should really unify this code for
  // reasoning about oslibdir spellings with the lib dir spellings in the
  // GCCInstallationDetector, but that is a more significant refactoring.
  if (Triple.getArch() == llvm::Triple::x86 || Triple.isPPC32() ||
      Triple.getArch() == llvm::Triple::sparc)
    return "lib32";

  if (Triple.getArch() == llvm::Triple::x86_64 &&
      Triple.getEnvironment() == llvm::Triple::GNUX32)
    return "libx32";

  if (Triple.getArch() == llvm::Triple::riscv32)
    return "lib32";

  return Triple.isArch32Bit() ? "lib" : "lib64";
}

Linux::Linux(const Driver &D, const llvm::Triple &Triple, const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  GCCInstallation.init(Triple, Args);
  Multilibs = GCCInstallation.getMultilibs();
  SelectedMultilib = GCCInstallation.getMultilib();
  llvm::Triple::ArchType Arch = Triple.getArch();
  std::string SysRoot = computeSysRoot();
  ToolChain::path_list &PPaths = getProgramPaths();

  Generic_GCC::PushPPaths(PPaths);

  Distro Distro(D.getVFS(), Triple);

  if (Distro.IsAlpineLinux() || Triple.isAndroid()) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("now");
  }

  if (Distro.IsOpenSUSE() || Distro.IsUbuntu() || Distro.IsAlpineLinux() ||
      Triple.isAndroid()) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("relro");
  }

  // Android ARM/AArch64 use max-page-size=4096 to reduce VMA usage. Note, lld
  // from 11 onwards default max-page-size to 65536 for both ARM and AArch64.
  if ((Triple.isARM() || Triple.isAArch64()) && Triple.isAndroid()) {
    ExtraOpts.push_back("-z");
    ExtraOpts.push_back("max-page-size=4096");
  }

  if (GCCInstallation.getParentLibPath().find("opt/rh/devtoolset") !=
      StringRef::npos)
    // With devtoolset on RHEL, we want to add a bin directory that is relative
    // to the detected gcc install, because if we are using devtoolset gcc then
    // we want to use other tools from devtoolset (e.g. ld) instead of the
    // standard system tools.
    PPaths.push_back(Twine(GCCInstallation.getParentLibPath() +
                     "/../bin").str());

  if (Arch == llvm::Triple::arm || Arch == llvm::Triple::thumb)
    ExtraOpts.push_back("-X");

  const bool IsAndroid = Triple.isAndroid();
  const bool IsMips = Triple.isMIPS();
  const bool IsHexagon = Arch == llvm::Triple::hexagon;
  const bool IsRISCV = Triple.isRISCV();

  if (IsMips && !SysRoot.empty())
    ExtraOpts.push_back("--sysroot=" + SysRoot);

  // Do not use 'gnu' hash style for Mips targets because .gnu.hash
  // and the MIPS ABI require .dynsym to be sorted in different ways.
  // .gnu.hash needs symbols to be grouped by hash code whereas the MIPS
  // ABI requires a mapping between the GOT and the symbol table.
  // Android loader does not support .gnu.hash until API 23.
  // Hexagon linker/loader does not support .gnu.hash
  if (!IsMips && !IsHexagon) {
    if (Distro.IsRedhat() || Distro.IsOpenSUSE() || Distro.IsAlpineLinux() ||
        (Distro.IsUbuntu() && Distro >= Distro::UbuntuMaverick) ||
        (IsAndroid && !Triple.isAndroidVersionLT(23)))
      ExtraOpts.push_back("--hash-style=gnu");

    if (Distro.IsDebian() || Distro.IsOpenSUSE() ||
        Distro == Distro::UbuntuLucid || Distro == Distro::UbuntuJaunty ||
        Distro == Distro::UbuntuKarmic ||
        (IsAndroid && Triple.isAndroidVersionLT(23)))
      ExtraOpts.push_back("--hash-style=both");
  }

#ifdef ENABLE_LINKER_BUILD_ID
  ExtraOpts.push_back("--build-id");
#endif

  if (IsAndroid || Distro.IsOpenSUSE())
    ExtraOpts.push_back("--enable-new-dtags");

  // The selection of paths to try here is designed to match the patterns which
  // the GCC driver itself uses, as this is part of the GCC-compatible driver.
  // This was determined by running GCC in a fake filesystem, creating all
  // possible permutations of these directories, and seeing which ones it added
  // to the link paths.
  path_list &Paths = getFilePaths();

  const std::string OSLibDir = std::string(getOSLibDir(Triple, Args));
  const std::string MultiarchTriple = getMultiarchTriple(D, Triple, SysRoot);

  Generic_GCC::AddMultilibPaths(D, SysRoot, OSLibDir, MultiarchTriple, Paths);

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

  if (IsAndroid) {
    // Android sysroots contain a library directory for each supported OS
    // version as well as some unversioned libraries in the usual multiarch
    // directory.
    unsigned Major;
    unsigned Minor;
    unsigned Micro;
    Triple.getEnvironmentVersion(Major, Minor, Micro);
    addPathIfExists(D,
                    SysRoot + "/usr/lib/" + MultiarchTriple + "/" +
                        llvm::to_string(Major),
                    Paths);
  }

  addPathIfExists(D, SysRoot + "/usr/lib/" + MultiarchTriple, Paths);
  // 64-bit OpenEmbedded sysroots may not have a /usr/lib dir. So they cannot
  // find /usr/lib64 as it is referenced as /usr/lib/../lib64. So we handle
  // this here.
  if (Triple.getVendor() == llvm::Triple::OpenEmbedded &&
      Triple.isArch64Bit())
    addPathIfExists(D, SysRoot + "/usr/" + OSLibDir, Paths);
  else
    addPathIfExists(D, SysRoot + "/usr/lib/../" + OSLibDir, Paths);
  if (IsRISCV) {
    StringRef ABIName = tools::riscv::getRISCVABI(Args, Triple);
    addPathIfExists(D, SysRoot + "/" + OSLibDir + "/" + ABIName, Paths);
    addPathIfExists(D, SysRoot + "/usr/" + OSLibDir + "/" + ABIName, Paths);
  }

  Generic_GCC::AddMultiarchPaths(D, SysRoot, OSLibDir, Paths);

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

ToolChain::CXXStdlibType Linux::GetDefaultCXXStdlibType() const {
  if (getTriple().isAndroid())
    return ToolChain::CST_Libcxx;
  return ToolChain::CST_Libstdcxx;
}

bool Linux::HasNativeLLVMSupport() const { return true; }

Tool *Linux::buildLinker() const { return new tools::gnutools::Linker(*this); }

Tool *Linux::buildStaticLibTool() const {
  return new tools::gnutools::StaticLibTool(*this);
}

Tool *Linux::buildAssembler() const {
  return new tools::gnutools::Assembler(*this);
}

std::string Linux::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot;

  if (getTriple().isAndroid()) {
    // Android toolchains typically include a sysroot at ../sysroot relative to
    // the clang binary.
    const StringRef ClangDir = getDriver().getInstalledDir();
    std::string AndroidSysRootPath = (ClangDir + "/../sysroot").str();
    if (getVFS().exists(AndroidSysRootPath))
      return AndroidSysRootPath;
  }

  if (!GCCInstallation.isValid() || !getTriple().isMIPS())
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

std::string Linux::getDynamicLinker(const ArgList &Args) const {
  const llvm::Triple::ArchType Arch = getArch();
  const llvm::Triple &Triple = getTriple();

  const Distro Distro(getDriver().getVFS(), Triple);

  if (Triple.isAndroid())
    return Triple.isArch64Bit() ? "/system/bin/linker64" : "/system/bin/linker";

  if (Triple.isMusl()) {
    std::string ArchName;
    bool IsArm = false;

    switch (Arch) {
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      ArchName = "arm";
      IsArm = true;
      break;
    case llvm::Triple::armeb:
    case llvm::Triple::thumbeb:
      ArchName = "armeb";
      IsArm = true;
      break;
    default:
      ArchName = Triple.getArchName().str();
    }
    if (IsArm &&
        (Triple.getEnvironment() == llvm::Triple::MuslEABIHF ||
         tools::arm::getARMFloatABI(*this, Args) == tools::arm::FloatABI::Hard))
      ArchName += "hf";

    return "/lib/ld-musl-" + ArchName + ".so.1";
  }

  std::string LibDir;
  std::string Loader;

  switch (Arch) {
  default:
    llvm_unreachable("unsupported architecture");

  case llvm::Triple::aarch64:
    LibDir = "lib";
    Loader = "ld-linux-aarch64.so.1";
    break;
  case llvm::Triple::aarch64_be:
    LibDir = "lib";
    Loader = "ld-linux-aarch64_be.so.1";
    break;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb: {
    const bool HF =
        Triple.getEnvironment() == llvm::Triple::GNUEABIHF ||
        tools::arm::getARMFloatABI(*this, Args) == tools::arm::FloatABI::Hard;

    LibDir = "lib";
    Loader = HF ? "ld-linux-armhf.so.3" : "ld-linux.so.3";
    break;
  }
  case llvm::Triple::m68k:
    LibDir = "lib";
    Loader = "ld.so.1";
    break;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el: {
    bool IsNaN2008 = tools::mips::isNaN2008(Args, Triple);

    LibDir = "lib" + tools::mips::getMipsABILibSuffix(Args, Triple);

    if (tools::mips::isUCLibc(Args))
      Loader = IsNaN2008 ? "ld-uClibc-mipsn8.so.0" : "ld-uClibc.so.0";
    else if (!Triple.hasEnvironment() &&
             Triple.getVendor() == llvm::Triple::VendorType::MipsTechnologies)
      Loader =
          Triple.isLittleEndian() ? "ld-musl-mipsel.so.1" : "ld-musl-mips.so.1";
    else
      Loader = IsNaN2008 ? "ld-linux-mipsn8.so.1" : "ld.so.1";

    break;
  }
  case llvm::Triple::ppc:
    LibDir = "lib";
    Loader = "ld.so.1";
    break;
  case llvm::Triple::ppcle:
    LibDir = "lib";
    Loader = "ld.so.1";
    break;
  case llvm::Triple::ppc64:
    LibDir = "lib64";
    Loader =
        (tools::ppc::hasPPCAbiArg(Args, "elfv2")) ? "ld64.so.2" : "ld64.so.1";
    break;
  case llvm::Triple::ppc64le:
    LibDir = "lib64";
    Loader =
        (tools::ppc::hasPPCAbiArg(Args, "elfv1")) ? "ld64.so.1" : "ld64.so.2";
    break;
  case llvm::Triple::riscv32: {
    StringRef ABIName = tools::riscv::getRISCVABI(Args, Triple);
    LibDir = "lib";
    Loader = ("ld-linux-riscv32-" + ABIName + ".so.1").str();
    break;
  }
  case llvm::Triple::riscv64: {
    StringRef ABIName = tools::riscv::getRISCVABI(Args, Triple);
    LibDir = "lib";
    Loader = ("ld-linux-riscv64-" + ABIName + ".so.1").str();
    break;
  }
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
    LibDir = "lib";
    Loader = "ld-linux.so.2";
    break;
  case llvm::Triple::sparcv9:
    LibDir = "lib64";
    Loader = "ld-linux.so.2";
    break;
  case llvm::Triple::systemz:
    LibDir = "lib";
    Loader = "ld64.so.1";
    break;
  case llvm::Triple::x86:
    LibDir = "lib";
    Loader = "ld-linux.so.2";
    break;
  case llvm::Triple::x86_64: {
    bool X32 = Triple.getEnvironment() == llvm::Triple::GNUX32;

    LibDir = X32 ? "libx32" : "lib64";
    Loader = X32 ? "ld-linux-x32.so.2" : "ld-linux-x86-64.so.2";
    break;
  }
  case llvm::Triple::ve:
    return "/opt/nec/ve/lib/ld-linux-ve.so.1";
  }

  if (Distro == Distro::Exherbo &&
      (Triple.getVendor() == llvm::Triple::UnknownVendor ||
       Triple.getVendor() == llvm::Triple::PC))
    return "/usr/" + Triple.str() + "/lib/" + Loader;
  return "/" + LibDir + "/" + Loader;
}

void Linux::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args) const {
  const Driver &D = getDriver();
  std::string SysRoot = computeSysRoot();

  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nostdlibinc))
    addSystemInclude(DriverArgs, CC1Args, SysRoot + "/usr/local/include");

  SmallString<128> ResourceDirInclude(D.ResourceDir);
  llvm::sys::path::append(ResourceDirInclude, "include");
  if (!DriverArgs.hasArg(options::OPT_nobuiltininc) &&
      (!getTriple().isMusl() || DriverArgs.hasArg(options::OPT_nostdlibinc)))
    addSystemInclude(DriverArgs, CC1Args, ResourceDirInclude);

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? "" : StringRef(SysRoot);
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  // Lacking those, try to detect the correct set of system includes for the
  // target triple.

  AddMultilibIncludeArgs(DriverArgs, CC1Args);

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
  const StringRef ARMEBMultiarchIncludeDirs[] = {
      "/usr/include/armeb-linux-gnueabi"};
  const StringRef ARMEBHFMultiarchIncludeDirs[] = {
      "/usr/include/armeb-linux-gnueabihf"};
  const StringRef M68kMultiarchIncludeDirs[] = {"/usr/include/m68k-linux-gnu"};
  const StringRef MIPSMultiarchIncludeDirs[] = {"/usr/include/mips-linux-gnu"};
  const StringRef MIPSELMultiarchIncludeDirs[] = {
      "/usr/include/mipsel-linux-gnu"};
  const StringRef MIPS64MultiarchIncludeDirs[] = {
      "/usr/include/mips64-linux-gnuabi64"};
  const StringRef MIPS64ELMultiarchIncludeDirs[] = {
      "/usr/include/mips64el-linux-gnuabi64"};
  const StringRef MIPSN32MultiarchIncludeDirs[] = {
      "/usr/include/mips64-linux-gnuabin32"};
  const StringRef MIPSN32ELMultiarchIncludeDirs[] = {
      "/usr/include/mips64el-linux-gnuabin32"};
  const StringRef MIPSR6MultiarchIncludeDirs[] = {
      "/usr/include/mipsisa32-linux-gnu"};
  const StringRef MIPSR6ELMultiarchIncludeDirs[] = {
      "/usr/include/mipsisa32r6el-linux-gnu"};
  const StringRef MIPS64R6MultiarchIncludeDirs[] = {
      "/usr/include/mipsisa64r6-linux-gnuabi64"};
  const StringRef MIPS64R6ELMultiarchIncludeDirs[] = {
      "/usr/include/mipsisa64r6el-linux-gnuabi64"};
  const StringRef MIPSN32R6MultiarchIncludeDirs[] = {
      "/usr/include/mipsisa64r6-linux-gnuabin32"};
  const StringRef MIPSN32R6ELMultiarchIncludeDirs[] = {
      "/usr/include/mipsisa64r6el-linux-gnuabin32"};
  const StringRef PPCMultiarchIncludeDirs[] = {
      "/usr/include/powerpc-linux-gnu",
      "/usr/include/powerpc-linux-gnuspe"};
  const StringRef PPCLEMultiarchIncludeDirs[] = {
      "/usr/include/powerpcle-linux-gnu"};
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
  case llvm::Triple::thumb:
    if (getTriple().getEnvironment() == llvm::Triple::GNUEABIHF)
      MultiarchIncludeDirs = ARMHFMultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = ARMMultiarchIncludeDirs;
    break;
  case llvm::Triple::armeb:
  case llvm::Triple::thumbeb:
    if (getTriple().getEnvironment() == llvm::Triple::GNUEABIHF)
      MultiarchIncludeDirs = ARMEBHFMultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = ARMEBMultiarchIncludeDirs;
    break;
  case llvm::Triple::m68k:
    MultiarchIncludeDirs = M68kMultiarchIncludeDirs;
    break;
  case llvm::Triple::mips:
    if (getTriple().getSubArch() == llvm::Triple::MipsSubArch_r6)
      MultiarchIncludeDirs = MIPSR6MultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = MIPSMultiarchIncludeDirs;
    break;
  case llvm::Triple::mipsel:
    if (getTriple().getSubArch() == llvm::Triple::MipsSubArch_r6)
      MultiarchIncludeDirs = MIPSR6ELMultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = MIPSELMultiarchIncludeDirs;
    break;
  case llvm::Triple::mips64:
    if (getTriple().getSubArch() == llvm::Triple::MipsSubArch_r6)
      if (getTriple().getEnvironment() == llvm::Triple::GNUABIN32)
        MultiarchIncludeDirs = MIPSN32R6MultiarchIncludeDirs;
      else
        MultiarchIncludeDirs = MIPS64R6MultiarchIncludeDirs;
    else if (getTriple().getEnvironment() == llvm::Triple::GNUABIN32)
      MultiarchIncludeDirs = MIPSN32MultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = MIPS64MultiarchIncludeDirs;
    break;
  case llvm::Triple::mips64el:
    if (getTriple().getSubArch() == llvm::Triple::MipsSubArch_r6)
      if (getTriple().getEnvironment() == llvm::Triple::GNUABIN32)
        MultiarchIncludeDirs = MIPSN32R6ELMultiarchIncludeDirs;
      else
        MultiarchIncludeDirs = MIPS64R6ELMultiarchIncludeDirs;
    else if (getTriple().getEnvironment() == llvm::Triple::GNUABIN32)
      MultiarchIncludeDirs = MIPSN32ELMultiarchIncludeDirs;
    else
      MultiarchIncludeDirs = MIPS64ELMultiarchIncludeDirs;
    break;
  case llvm::Triple::ppc:
    MultiarchIncludeDirs = PPCMultiarchIncludeDirs;
    break;
  case llvm::Triple::ppcle:
    MultiarchIncludeDirs = PPCLEMultiarchIncludeDirs;
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

  const std::string AndroidMultiarchIncludeDir =
      std::string("/usr/include/") +
      getMultiarchTriple(D, getTriple(), SysRoot);
  const StringRef AndroidMultiarchIncludeDirs[] = {AndroidMultiarchIncludeDir};
  if (getTriple().isAndroid())
    MultiarchIncludeDirs = AndroidMultiarchIncludeDirs;

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

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc) && getTriple().isMusl())
    addSystemInclude(DriverArgs, CC1Args, ResourceDirInclude);
}

void Linux::addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const {
  // Try generic GCC detection first.
  if (Generic_GCC::addGCCLibStdCxxIncludePaths(DriverArgs, CC1Args))
    return;

  // We need a detected GCC installation on Linux to provide libstdc++'s
  // headers in odd Linuxish places.
  if (!GCCInstallation.isValid())
    return;

  StringRef LibDir = GCCInstallation.getParentLibPath();
  StringRef TripleStr = GCCInstallation.getTriple().str();
  const Multilib &Multilib = GCCInstallation.getMultilib();
  const GCCVersion &Version = GCCInstallation.getVersion();

  const std::string LibStdCXXIncludePathCandidates[] = {
      // Android standalone toolchain has C++ headers in yet another place.
      LibDir.str() + "/../" + TripleStr.str() + "/include/c++/" + Version.Text,
      // Freescale SDK C++ headers are directly in <sysroot>/usr/include/c++,
      // without a subdirectory corresponding to the gcc version.
      LibDir.str() + "/../include/c++",
      // Cray's gcc installation puts headers under "g++" without a
      // version suffix.
      LibDir.str() + "/../include/g++",
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
  CudaInstallation.AddCudaIncludeArgs(DriverArgs, CC1Args);
}

void Linux::AddHIPIncludeArgs(const ArgList &DriverArgs,
                              ArgStringList &CC1Args) const {
  RocmInstallation.AddHIPIncludeArgs(DriverArgs, CC1Args);
}

void Linux::AddIAMCUIncludeArgs(const ArgList &DriverArgs,
                                ArgStringList &CC1Args) const {
  if (GCCInstallation.isValid()) {
    CC1Args.push_back("-isystem");
    CC1Args.push_back(DriverArgs.MakeArgString(
        GCCInstallation.getParentLibPath() + "/../" +
        GCCInstallation.getTriple().str() + "/include"));
  }
}

bool Linux::isPIEDefault() const {
  return (getTriple().isAndroid() && !getTriple().isAndroidVersionLT(16)) ||
          getTriple().isMusl() || getSanitizerArgs().requiresPIE();
}

bool Linux::IsAArch64OutlineAtomicsDefault(const ArgList &Args) const {
  // Outline atomics for AArch64 are supported by compiler-rt
  // and libgcc since 9.3.1
  assert(getTriple().isAArch64() && "expected AArch64 target!");
  ToolChain::RuntimeLibType RtLib = GetRuntimeLibType(Args);
  if (RtLib == ToolChain::RLT_CompilerRT)
    return true;
  assert(RtLib == ToolChain::RLT_Libgcc && "unexpected runtime library type!");
  if (GCCInstallation.getVersion().isOlderThan(9, 3, 1))
    return false;
  return true;
}

bool Linux::isNoExecStackDefault() const {
    return getTriple().isAndroid();
}

bool Linux::IsMathErrnoDefault() const {
  if (getTriple().isAndroid())
    return false;
  return Generic_ELF::IsMathErrnoDefault();
}

SanitizerMask Linux::getSupportedSanitizers() const {
  const bool IsX86 = getTriple().getArch() == llvm::Triple::x86;
  const bool IsX86_64 = getTriple().getArch() == llvm::Triple::x86_64;
  const bool IsMIPS = getTriple().isMIPS32();
  const bool IsMIPS64 = getTriple().isMIPS64();
  const bool IsPowerPC64 = getTriple().getArch() == llvm::Triple::ppc64 ||
                           getTriple().getArch() == llvm::Triple::ppc64le;
  const bool IsAArch64 = getTriple().getArch() == llvm::Triple::aarch64 ||
                         getTriple().getArch() == llvm::Triple::aarch64_be;
  const bool IsArmArch = getTriple().getArch() == llvm::Triple::arm ||
                         getTriple().getArch() == llvm::Triple::thumb ||
                         getTriple().getArch() == llvm::Triple::armeb ||
                         getTriple().getArch() == llvm::Triple::thumbeb;
  const bool IsRISCV64 = getTriple().getArch() == llvm::Triple::riscv64;
  const bool IsSystemZ = getTriple().getArch() == llvm::Triple::systemz;
  SanitizerMask Res = ToolChain::getSupportedSanitizers();
  Res |= SanitizerKind::Address;
  Res |= SanitizerKind::PointerCompare;
  Res |= SanitizerKind::PointerSubtract;
  Res |= SanitizerKind::Fuzzer;
  Res |= SanitizerKind::FuzzerNoLink;
  Res |= SanitizerKind::KernelAddress;
  Res |= SanitizerKind::Memory;
  Res |= SanitizerKind::Vptr;
  Res |= SanitizerKind::SafeStack;
  if (IsX86_64 || IsMIPS64 || IsAArch64)
    Res |= SanitizerKind::DataFlow;
  if (IsX86_64 || IsMIPS64 || IsAArch64 || IsX86 || IsArmArch || IsPowerPC64 ||
      IsRISCV64 || IsSystemZ)
    Res |= SanitizerKind::Leak;
  if (IsX86_64 || IsMIPS64 || IsAArch64 || IsPowerPC64)
    Res |= SanitizerKind::Thread;
  if (IsX86_64)
    Res |= SanitizerKind::KernelMemory;
  if (IsX86 || IsX86_64)
    Res |= SanitizerKind::Function;
  if (IsX86_64 || IsMIPS64 || IsAArch64 || IsX86 || IsMIPS || IsArmArch ||
      IsPowerPC64)
    Res |= SanitizerKind::Scudo;
  if (IsX86_64 || IsAArch64) {
    Res |= SanitizerKind::HWAddress;
    Res |= SanitizerKind::KernelHWAddress;
  }
  return Res;
}

void Linux::addProfileRTLibs(const llvm::opt::ArgList &Args,
                             llvm::opt::ArgStringList &CmdArgs) const {
  // Add linker option -u__llvm_profile_runtime to cause runtime
  // initialization module to be linked in.
  if (needsProfileRT(Args))
    CmdArgs.push_back(Args.MakeArgString(
        Twine("-u", llvm::getInstrProfRuntimeHookVarName())));
  ToolChain::addProfileRTLibs(Args, CmdArgs);
}

llvm::DenormalMode
Linux::getDefaultDenormalModeForType(const llvm::opt::ArgList &DriverArgs,
                                     const JobAction &JA,
                                     const llvm::fltSemantics *FPType) const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
  case llvm::Triple::x86_64: {
    std::string Unused;
    // DAZ and FTZ are turned on in crtfastmath.o
    if (!DriverArgs.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles) &&
        isFastMathRuntimeAvailable(DriverArgs, Unused))
      return llvm::DenormalMode::getPreserveSign();
    return llvm::DenormalMode::getIEEE();
  }
  default:
    return llvm::DenormalMode::getIEEE();
  }
}

void Linux::addExtraOpts(llvm::opt::ArgStringList &CmdArgs) const {
  for (const auto &Opt : ExtraOpts)
    CmdArgs.push_back(Opt.c_str());
}
