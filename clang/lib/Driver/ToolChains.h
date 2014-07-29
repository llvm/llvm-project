//===--- ToolChains.h - ToolChain Implementations ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_DRIVER_TOOLCHAINS_H_
#define CLANG_LIB_DRIVER_TOOLCHAINS_H_

#include "Tools.h"
#include "clang/Basic/VersionTuple.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/Compiler.h"
#include <set>
#include <vector>

namespace clang {
namespace driver {
namespace toolchains {

/// Generic_GCC - A tool chain using the 'gcc' command to perform
/// all subcommands; this relies on gcc translating the majority of
/// command line options.
class LLVM_LIBRARY_VISIBILITY Generic_GCC : public ToolChain {
protected:
  /// \brief Struct to store and manipulate GCC versions.
  ///
  /// We rely on assumptions about the form and structure of GCC version
  /// numbers: they consist of at most three '.'-separated components, and each
  /// component is a non-negative integer except for the last component. For
  /// the last component we are very flexible in order to tolerate release
  /// candidates or 'x' wildcards.
  ///
  /// Note that the ordering established among GCCVersions is based on the
  /// preferred version string to use. For example we prefer versions without
  /// a hard-coded patch number to those with a hard coded patch number.
  ///
  /// Currently this doesn't provide any logic for textual suffixes to patches
  /// in the way that (for example) Debian's version format does. If that ever
  /// becomes necessary, it can be added.
  struct GCCVersion {
    /// \brief The unparsed text of the version.
    std::string Text;

    /// \brief The parsed major, minor, and patch numbers.
    int Major, Minor, Patch;

    /// \brief The text of the parsed major, and major+minor versions.
    std::string MajorStr, MinorStr;

    /// \brief Any textual suffix on the patch number.
    std::string PatchSuffix;

    static GCCVersion Parse(StringRef VersionText);
    bool isOlderThan(int RHSMajor, int RHSMinor, int RHSPatch,
                     StringRef RHSPatchSuffix = StringRef()) const;
    bool operator<(const GCCVersion &RHS) const {
      return isOlderThan(RHS.Major, RHS.Minor, RHS.Patch, RHS.PatchSuffix);
    }
    bool operator>(const GCCVersion &RHS) const { return RHS < *this; }
    bool operator<=(const GCCVersion &RHS) const { return !(*this > RHS); }
    bool operator>=(const GCCVersion &RHS) const { return !(*this < RHS); }
  };

  /// \brief This is a class to find a viable GCC installation for Clang to
  /// use.
  ///
  /// This class tries to find a GCC installation on the system, and report
  /// information about it. It starts from the host information provided to the
  /// Driver, and has logic for fuzzing that where appropriate.
  class GCCInstallationDetector {
    bool IsValid;
    llvm::Triple GCCTriple;

    // FIXME: These might be better as path objects.
    std::string GCCInstallPath;
    std::string GCCParentLibPath;

    /// The primary multilib appropriate for the given flags.
    Multilib SelectedMultilib;
    /// On Biarch systems, this corresponds to the default multilib when
    /// targeting the non-default multilib. Otherwise, it is empty.
    llvm::Optional<Multilib> BiarchSibling;

    GCCVersion Version;

    // We retain the list of install paths that were considered and rejected in
    // order to print out detailed information in verbose mode.
    std::set<std::string> CandidateGCCInstallPaths;

    /// The set of multilibs that the detected installation supports.
    MultilibSet Multilibs;

  public:
    GCCInstallationDetector() : IsValid(false) {}
    void init(const Driver &D, const llvm::Triple &TargetTriple,
                            const llvm::opt::ArgList &Args);

    /// \brief Check whether we detected a valid GCC install.
    bool isValid() const { return IsValid; }

    /// \brief Get the GCC triple for the detected install.
    const llvm::Triple &getTriple() const { return GCCTriple; }

    /// \brief Get the detected GCC installation path.
    StringRef getInstallPath() const { return GCCInstallPath; }

    /// \brief Get the detected GCC parent lib path.
    StringRef getParentLibPath() const { return GCCParentLibPath; }

    /// \brief Get the detected Multilib
    const Multilib &getMultilib() const { return SelectedMultilib; }

    /// \brief Get the whole MultilibSet
    const MultilibSet &getMultilibs() const { return Multilibs; }

    /// Get the biarch sibling multilib (if it exists).
    /// \return true iff such a sibling exists
    bool getBiarchSibling(Multilib &M) const;

    /// \brief Get the detected GCC version string.
    const GCCVersion &getVersion() const { return Version; }

    /// \brief Print information about the detected GCC installation.
    void print(raw_ostream &OS) const;

  private:
    static void
    CollectLibDirsAndTriples(const llvm::Triple &TargetTriple,
                             const llvm::Triple &BiarchTriple,
                             SmallVectorImpl<StringRef> &LibDirs,
                             SmallVectorImpl<StringRef> &TripleAliases,
                             SmallVectorImpl<StringRef> &BiarchLibDirs,
                             SmallVectorImpl<StringRef> &BiarchTripleAliases);

    void ScanLibDirForGCCTriple(const llvm::Triple &TargetArch,
                                const llvm::opt::ArgList &Args,
                                const std::string &LibDir,
                                StringRef CandidateTriple,
                                bool NeedsBiarchSuffix = false);
  };

  GCCInstallationDetector GCCInstallation;

public:
  Generic_GCC(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args);
  ~Generic_GCC();

  void printVerboseInfo(raw_ostream &OS) const override;

  bool IsUnwindTablesDefault() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;
  bool IsIntegratedAssemblerDefault() const override;

protected:
  Tool *getTool(Action::ActionClass AC) const override;
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;

  /// \name ToolChain Implementation Helper Functions
  /// @{

  /// \brief Check whether the target triple's architecture is 64-bits.
  bool isTarget64Bit() const { return getTriple().isArch64Bit(); }

  /// \brief Check whether the target triple's architecture is 32-bits.
  bool isTarget32Bit() const { return getTriple().isArch32Bit(); }

  /// @}

private:
  mutable std::unique_ptr<tools::gcc::Preprocess> Preprocess;
  mutable std::unique_ptr<tools::gcc::Compile> Compile;
};

class LLVM_LIBRARY_VISIBILITY MachO : public ToolChain {
protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
  Tool *getTool(Action::ActionClass AC) const override;
private:
  mutable std::unique_ptr<tools::darwin::Lipo> Lipo;
  mutable std::unique_ptr<tools::darwin::Dsymutil> Dsymutil;
  mutable std::unique_ptr<tools::darwin::VerifyDebug> VerifyDebug;

public:
  MachO(const Driver &D, const llvm::Triple &Triple,
             const llvm::opt::ArgList &Args);
  ~MachO();

  /// @name MachO specific toolchain API
  /// {

  /// Get the "MachO" arch name for a particular compiler invocation. For
  /// example, Apple treats different ARM variations as distinct architectures.
  StringRef getMachOArchName(const llvm::opt::ArgList &Args) const;


  /// Add the linker arguments to link the ARC runtime library.
  virtual void AddLinkARCArgs(const llvm::opt::ArgList &Args,
                              llvm::opt::ArgStringList &CmdArgs) const {}

  /// Add the linker arguments to link the compiler runtime library.
  virtual void AddLinkRuntimeLibArgs(const llvm::opt::ArgList &Args,
                                     llvm::opt::ArgStringList &CmdArgs) const;

  virtual void
  addStartObjectFileArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const {}

  virtual void addMinVersionArgs(const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs) const {}

  /// On some iOS platforms, kernel and kernel modules were built statically. Is
  /// this such a target?
  virtual bool isKernelStatic() const {
    return false;
  }

  /// Is the target either iOS or an iOS simulator?
  bool isTargetIOSBased() const {
    return false;
  }

  void AddLinkRuntimeLib(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs,
                         StringRef DarwinStaticLib,
                         bool AlwaysLink = false,
                         bool IsEmbedded = false) const;

  /// }
  /// @name ToolChain Implementation
  /// {

  std::string ComputeEffectiveClangTriple(const llvm::opt::ArgList &Args,
                                          types::ID InputType) const override;

  types::ID LookupTypeForExtension(const char *Ext) const override;

  bool HasNativeLLVMSupport() const override;

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const override;

  bool IsBlocksDefault() const override {
    // Always allow blocks on Apple; users interested in versioning are
    // expected to use /usr/include/Blocks.h.
    return true;
  }
  bool IsIntegratedAssemblerDefault() const override {
    // Default integrated assembler to on for Apple's MachO targets.
    return true;
  }

  bool IsMathErrnoDefault() const override {
    return false;
  }

  bool IsEncodeExtendedBlockSignatureDefault() const override {
    return true;
  }

  bool IsObjCNonFragileABIDefault() const override {
    // Non-fragile ABI is default for everything but i386.
    return getTriple().getArch() != llvm::Triple::x86;
  }

  bool UseObjCMixedDispatch() const override {
    return true;
  }

  bool IsUnwindTablesDefault() const override;

  RuntimeLibType GetDefaultRuntimeLibType() const override {
    return ToolChain::RLT_CompilerRT;
  }

  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;

  bool SupportsProfiling() const override;

  bool SupportsObjCGC() const override {
    return false;
  }

  bool UseDwarfDebugFlags() const override;

  bool UseSjLjExceptions() const override {
    return false;
  }

  /// }
};

  /// Darwin - The base Darwin tool chain.
class LLVM_LIBRARY_VISIBILITY Darwin : public MachO {
public:
  /// The host version.
  unsigned DarwinVersion[3];

  /// Whether the information on the target has been initialized.
  //
  // FIXME: This should be eliminated. What we want to do is make this part of
  // the "default target for arguments" selection process, once we get out of
  // the argument translation business.
  mutable bool TargetInitialized;

  enum DarwinPlatformKind {
    MacOS,
    IPhoneOS,
    IPhoneOSSimulator
  };

  mutable DarwinPlatformKind TargetPlatform;

  /// The OS version we are targeting.
  mutable VersionTuple TargetVersion;

private:
  /// The default macosx-version-min of this tool chain; empty until
  /// initialized.
  std::string MacosxVersionMin;

  /// The default ios-version-min of this tool chain; empty until
  /// initialized.
  std::string iOSVersionMin;

private:
  void AddDeploymentTarget(llvm::opt::DerivedArgList &Args) const;

public:
  Darwin(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);
  ~Darwin();

  std::string ComputeEffectiveClangTriple(const llvm::opt::ArgList &Args,
                                          types::ID InputType) const override;

  /// @name Apple Specific Toolchain Implementation
  /// {

  void
  addMinVersionArgs(const llvm::opt::ArgList &Args,
                    llvm::opt::ArgStringList &CmdArgs) const override;

  void
  addStartObjectFileArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const override;

  bool isKernelStatic() const override {
    return !isTargetIPhoneOS() || isIPhoneOSVersionLT(6, 0) ||
           getTriple().getArch() == llvm::Triple::arm64;
  }

protected:
  /// }
  /// @name Darwin specific Toolchain functions
  /// {

  // FIXME: Eliminate these ...Target functions and derive separate tool chains
  // for these targets and put version in constructor.
  void setTarget(DarwinPlatformKind Platform, unsigned Major, unsigned Minor,
                 unsigned Micro) const {
    // FIXME: For now, allow reinitialization as long as values don't
    // change. This will go away when we move away from argument translation.
    if (TargetInitialized && TargetPlatform == Platform &&
        TargetVersion == VersionTuple(Major, Minor, Micro))
      return;

    assert(!TargetInitialized && "Target already initialized!");
    TargetInitialized = true;
    TargetPlatform = Platform;
    TargetVersion = VersionTuple(Major, Minor, Micro);
  }

  bool isTargetIPhoneOS() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == IPhoneOS;
  }

  bool isTargetIOSSimulator() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == IPhoneOSSimulator;
  }

  bool isTargetIOSBased() const {
    assert(TargetInitialized && "Target not initialized!");
    return isTargetIPhoneOS() || isTargetIOSSimulator();
  }

  bool isTargetMacOS() const {
    return TargetPlatform == MacOS;
  }

  bool isTargetInitialized() const { return TargetInitialized; }

  VersionTuple getTargetVersion() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetVersion;
  }

  bool isIPhoneOSVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(isTargetIOSBased() && "Unexpected call for non iOS target!");
    return TargetVersion < VersionTuple(V0, V1, V2);
  }

  bool isMacosxVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(isTargetMacOS() && "Unexpected call for non OS X target!");
    return TargetVersion < VersionTuple(V0, V1, V2);
  }

public:
  /// }
  /// @name ToolChain Implementation
  /// {

  // Darwin tools support multiple architecture (e.g., i386 and x86_64) and
  // most development is done against SDKs, so compiling for a different
  // architecture should not get any special treatment.
  bool isCrossCompiling() const override { return false; }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const override;

  ObjCRuntime getDefaultObjCRuntime(bool isNonFragile) const override;
  bool hasBlocksRuntime() const override;

  bool UseObjCMixedDispatch() const override {
    // This is only used with the non-fragile ABI and non-legacy dispatch.

    // Mixed dispatch is used everywhere except OS X before 10.6.
    return !(isTargetMacOS() && isMacosxVersionLT(10, 6));
  }

  unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const override {
    // Stack protectors default to on for user code on 10.5,
    // and for everything in 10.6 and beyond
    if (isTargetIOSBased())
      return 1;
    else if (isTargetMacOS() && !isMacosxVersionLT(10, 6))
      return 1;
    else if (isTargetMacOS() && !isMacosxVersionLT(10, 5) && !KernelOrKext)
      return 1;

    return 0;
  }

  bool SupportsObjCGC() const override;

  void CheckObjCARC() const override;

  bool UseSjLjExceptions() const override;
};

/// DarwinClang - The Darwin toolchain used by Clang.
class LLVM_LIBRARY_VISIBILITY DarwinClang : public Darwin {
public:
  DarwinClang(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args);

  /// @name Apple ToolChain Implementation
  /// {

  void
  AddLinkRuntimeLibArgs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const override;

  void
  AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs) const override;

  void
  AddCCKextLibArgs(const llvm::opt::ArgList &Args,
                   llvm::opt::ArgStringList &CmdArgs) const override;

  virtual void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const;

  void
  AddLinkARCArgs(const llvm::opt::ArgList &Args,
                 llvm::opt::ArgStringList &CmdArgs) const override;
  /// }
};

class LLVM_LIBRARY_VISIBILITY Generic_ELF : public Generic_GCC {
  virtual void anchor();
public:
  Generic_ELF(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args)
      : Generic_GCC(D, Triple, Args) {}

  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const override;
};

class LLVM_LIBRARY_VISIBILITY AuroraUX : public Generic_GCC {
public:
  AuroraUX(const Driver &D, const llvm::Triple &Triple,
           const llvm::opt::ArgList &Args);

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY Solaris : public Generic_GCC {
public:
  Solaris(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  bool IsIntegratedAssemblerDefault() const override { return true; }
protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;

};


class LLVM_LIBRARY_VISIBILITY OpenBSD : public Generic_ELF {
public:
  OpenBSD(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  bool IsMathErrnoDefault() const override { return false; }
  bool IsObjCNonFragileABIDefault() const override { return true; }
  bool isPIEDefault() const override { return true; }

  unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const override {
    return 2;
  }

  virtual bool IsIntegratedAssemblerDefault() const {
    if (getTriple().getArch() == llvm::Triple::ppc ||
        getTriple().getArch() == llvm::Triple::sparc ||
        getTriple().getArch() == llvm::Triple::sparcv9)
      return true;
    return Generic_ELF::IsIntegratedAssemblerDefault();
  }

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY Bitrig : public Generic_ELF {
public:
  Bitrig(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);

  bool IsMathErrnoDefault() const override { return false; }
  bool IsObjCNonFragileABIDefault() const override { return true; }

  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                              llvm::opt::ArgStringList &CC1Args) const override;
  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;
  unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const override {
     return 1;
  }

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY FreeBSD : public Generic_ELF {
public:
  FreeBSD(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);
  bool HasNativeLLVMSupport() const override;

  bool IsMathErrnoDefault() const override { return false; }
  bool IsObjCNonFragileABIDefault() const override { return true; }

  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const override;
  bool IsIntegratedAssemblerDefault() const override {
    if (getTriple().getArch() == llvm::Triple::ppc ||
        getTriple().getArch() == llvm::Triple::ppc64)
      return true;
    return Generic_ELF::IsIntegratedAssemblerDefault();
  }

  bool UseSjLjExceptions() const override;
  bool isPIEDefault() const override;
protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY NetBSD : public Generic_ELF {
public:
  NetBSD(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);

  bool IsMathErrnoDefault() const override { return false; }
  bool IsObjCNonFragileABIDefault() const override { return true; }

  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;

  void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                              llvm::opt::ArgStringList &CC1Args) const override;
  bool IsUnwindTablesDefault() const override {
    return true;
  }
  bool IsIntegratedAssemblerDefault() const override {
    if (getTriple().getArch() == llvm::Triple::ppc)
      return true;
    return Generic_ELF::IsIntegratedAssemblerDefault();
  }

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY Minix : public Generic_ELF {
public:
  Minix(const Driver &D, const llvm::Triple &Triple,
        const llvm::opt::ArgList &Args);

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY DragonFly : public Generic_ELF {
public:
  DragonFly(const Driver &D, const llvm::Triple &Triple,
            const llvm::opt::ArgList &Args);

  bool IsMathErrnoDefault() const override { return false; }

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY Linux : public Generic_ELF {
public:
  Linux(const Driver &D, const llvm::Triple &Triple,
        const llvm::opt::ArgList &Args);

  bool HasNativeLLVMSupport() const override;

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const override;
  bool isPIEDefault() const override;

  std::string Linker;
  std::vector<std::string> ExtraOpts;

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;

private:
  static bool addLibStdCXXIncludePaths(Twine Base, Twine Suffix,
                                       Twine TargetArchDir, Twine IncludeSuffix,
                                       const llvm::opt::ArgList &DriverArgs,
                                       llvm::opt::ArgStringList &CC1Args);
  static bool addLibStdCXXIncludePaths(Twine Base, Twine TargetArchDir,
                                       const llvm::opt::ArgList &DriverArgs,
                                       llvm::opt::ArgStringList &CC1Args);

  std::string computeSysRoot() const;
};

class LLVM_LIBRARY_VISIBILITY Hexagon_TC : public Linux {
protected:
  GCCVersion GCCLibAndIncVersion;
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;

public:
  Hexagon_TC(const Driver &D, const llvm::Triple &Triple,
             const llvm::opt::ArgList &Args);
  ~Hexagon_TC();

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                              llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;

  StringRef GetGCCLibAndIncVersion() const { return GCCLibAndIncVersion.Text; }

  static std::string GetGnuDir(const std::string &InstalledDir);

  static StringRef GetTargetCPU(const llvm::opt::ArgList &Args);
};

/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
class LLVM_LIBRARY_VISIBILITY TCEToolChain : public ToolChain {
public:
  TCEToolChain(const Driver &D, const llvm::Triple &Triple,
               const llvm::opt::ArgList &Args);
  ~TCEToolChain();

  bool IsMathErrnoDefault() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;
};

class LLVM_LIBRARY_VISIBILITY Windows : public ToolChain {
public:
  Windows(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  bool IsIntegratedAssemblerDefault() const override;
  bool IsUnwindTablesDefault() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const override;

protected:
  Tool *buildLinker() const override;
  Tool *buildAssembler() const override;
};


class LLVM_LIBRARY_VISIBILITY XCore : public ToolChain {
public:
  XCore(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);
protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
public:
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;
  bool SupportsProfiling() const override;
  bool hasBlocksRuntime() const override;
  void AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                    llvm::opt::ArgStringList &CC1Args) const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                       llvm::opt::ArgStringList &CC1Args) const override;
  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
