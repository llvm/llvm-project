//===--- ToolChains.h - ToolChain Implementations ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_H

#include "Tools.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/VersionTuple.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
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
public:
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
    const Driver &D;

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
    explicit GCCInstallationDetector(const Driver &D) : IsValid(false), D(D) {}
    void init(const llvm::Triple &TargetTriple, const llvm::opt::ArgList &Args,
              ArrayRef<std::string> ExtraTripleAliases = None);

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

    void scanLibDirForGCCTripleSolaris(const llvm::Triple &TargetArch,
                                       const llvm::opt::ArgList &Args,
                                       const std::string &LibDir,
                                       StringRef CandidateTriple,
                                       bool NeedsBiarchSuffix = false);
  };

protected:
  GCCInstallationDetector GCCInstallation;

  // \brief A class to find a viable CUDA installation
  class CudaInstallationDetector {
  private:
    const Driver &D;
    bool IsValid = false;
    CudaVersion Version = CudaVersion::UNKNOWN;
    std::string InstallPath;
    std::string BinPath;
    std::string LibPath;
    std::string LibDevicePath;
    std::string IncludePath;
    llvm::StringMap<std::string> LibDeviceMap;

    // CUDA architectures for which we have raised an error in
    // CheckCudaVersionSupportsArch.
    mutable llvm::SmallSet<CudaArch, 4> ArchsWithVersionTooLowErrors;

  public:
    CudaInstallationDetector(const Driver &D) : D(D) {}
    void init(const llvm::Triple &TargetTriple, const llvm::opt::ArgList &Args);

    /// \brief Emit an error if Version does not support the given Arch.
    ///
    /// If either Version or Arch is unknown, does not emit an error.  Emits at
    /// most one error per Arch.
    void CheckCudaVersionSupportsArch(CudaArch Arch) const;

    /// \brief Check whether we detected a valid Cuda install.
    bool isValid() const { return IsValid; }
    /// \brief Print information about the detected CUDA installation.
    void print(raw_ostream &OS) const;

    /// \brief Get the detected Cuda install's version.
    CudaVersion version() const { return Version; }
    /// \brief Get the detected Cuda installation path.
    StringRef getInstallPath() const { return InstallPath; }
    /// \brief Get the detected path to Cuda's bin directory.
    StringRef getBinPath() const { return BinPath; }
    /// \brief Get the detected Cuda Include path.
    StringRef getIncludePath() const { return IncludePath; }
    /// \brief Get the detected Cuda library path.
    StringRef getLibPath() const { return LibPath; }
    /// \brief Get the detected Cuda device library path.
    StringRef getLibDevicePath() const { return LibDevicePath; }
    /// \brief Get libdevice file for given architecture
    std::string getLibDeviceFile(StringRef Gpu) const {
      return LibDeviceMap.lookup(Gpu);
    }
  };

  CudaInstallationDetector CudaInstallation;

public:
  Generic_GCC(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args);
  ~Generic_GCC() override;

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

  bool addLibStdCXXIncludePaths(Twine Base, Twine Suffix, StringRef GCCTriple,
                                StringRef GCCMultiarchTriple,
                                StringRef TargetMultiarchTriple,
                                Twine IncludeSuffix,
                                const llvm::opt::ArgList &DriverArgs,
                                llvm::opt::ArgStringList &CC1Args) const;

  /// @}

private:
  mutable std::unique_ptr<tools::gcc::Preprocessor> Preprocess;
  mutable std::unique_ptr<tools::gcc::Compiler> Compile;
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
  ~MachO() override;

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

  virtual void addStartObjectFileArgs(const llvm::opt::ArgList &Args,
                                      llvm::opt::ArgStringList &CmdArgs) const {
  }

  virtual void addMinVersionArgs(const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs) const {}

  /// On some iOS platforms, kernel and kernel modules were built statically. Is
  /// this such a target?
  virtual bool isKernelStatic() const { return false; }

  /// Is the target either iOS or an iOS simulator?
  bool isTargetIOSBased() const { return false; }

  void AddLinkRuntimeLib(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs,
                         StringRef DarwinLibName, bool AlwaysLink = false,
                         bool IsEmbedded = false, bool AddRPath = false) const;

  /// Add any profiling runtime libraries that are needed. This is essentially a
  /// MachO specific version of addProfileRT in Tools.cpp.
  void addProfileRTLibs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const override {
    // There aren't any profiling libs for embedded targets currently.
  }

  /// }
  /// @name ToolChain Implementation
  /// {

  types::ID LookupTypeForExtension(const char *Ext) const override;

  bool HasNativeLLVMSupport() const override;

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const override;

  bool IsBlocksDefault() const override {
    // Always allow blocks on Apple; users interested in versioning are
    // expected to use /usr/include/Block.h.
    return true;
  }
  bool IsIntegratedAssemblerDefault() const override {
    // Default integrated assembler to on for Apple's MachO targets.
    return true;
  }

  bool IsMathErrnoDefault() const override { return false; }

  bool IsEncodeExtendedBlockSignatureDefault() const override { return true; }

  bool IsObjCNonFragileABIDefault() const override {
    // Non-fragile ABI is default for everything but i386.
    return getTriple().getArch() != llvm::Triple::x86;
  }

  bool UseObjCMixedDispatch() const override { return true; }

  bool IsUnwindTablesDefault() const override;

  RuntimeLibType GetDefaultRuntimeLibType() const override {
    return ToolChain::RLT_CompilerRT;
  }

  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;

  bool SupportsProfiling() const override;

  bool SupportsObjCGC() const override { return false; }

  bool UseDwarfDebugFlags() const override;

  bool UseSjLjExceptions(const llvm::opt::ArgList &Args) const override {
    return false;
  }

  /// }
};

/// Darwin - The base Darwin tool chain.
class LLVM_LIBRARY_VISIBILITY Darwin : public MachO {
public:
  /// Whether the information on the target has been initialized.
  //
  // FIXME: This should be eliminated. What we want to do is make this part of
  // the "default target for arguments" selection process, once we get out of
  // the argument translation business.
  mutable bool TargetInitialized;

  enum DarwinPlatformKind {
    MacOS,
    IPhoneOS,
    IPhoneOSSimulator,
    TvOS,
    TvOSSimulator,
    WatchOS,
    WatchOSSimulator
  };

  mutable DarwinPlatformKind TargetPlatform;

  /// The OS version we are targeting.
  mutable VersionTuple TargetVersion;

private:
  void AddDeploymentTarget(llvm::opt::DerivedArgList &Args) const;

public:
  Darwin(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);
  ~Darwin() override;

  std::string ComputeEffectiveClangTriple(const llvm::opt::ArgList &Args,
                                          types::ID InputType) const override;

  /// @name Apple Specific Toolchain Implementation
  /// {

  void addMinVersionArgs(const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs) const override;

  void addStartObjectFileArgs(const llvm::opt::ArgList &Args,
                              llvm::opt::ArgStringList &CmdArgs) const override;

  bool isKernelStatic() const override {
    return (!(isTargetIPhoneOS() && !isIPhoneOSVersionLT(6, 0)) &&
            !isTargetWatchOS());
  }

  void addProfileRTLibs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const override;

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
    return TargetPlatform == IPhoneOS || TargetPlatform == TvOS;
  }

  bool isTargetIOSSimulator() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == IPhoneOSSimulator ||
           TargetPlatform == TvOSSimulator;
  }

  bool isTargetIOSBased() const {
    assert(TargetInitialized && "Target not initialized!");
    return isTargetIPhoneOS() || isTargetIOSSimulator();
  }

  bool isTargetTvOS() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == TvOS;
  }

  bool isTargetTvOSSimulator() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == TvOSSimulator;
  }

  bool isTargetTvOSBased() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == TvOS || TargetPlatform == TvOSSimulator;
  }

  bool isTargetWatchOS() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == WatchOS;
  }

  bool isTargetWatchOSSimulator() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == WatchOSSimulator;
  }

  bool isTargetWatchOSBased() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == WatchOS || TargetPlatform == WatchOSSimulator;
  }

  bool isTargetMacOS() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetPlatform == MacOS;
  }

  bool isTargetInitialized() const { return TargetInitialized; }

  VersionTuple getTargetVersion() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetVersion;
  }

  bool isIPhoneOSVersionLT(unsigned V0, unsigned V1 = 0,
                           unsigned V2 = 0) const {
    assert(isTargetIOSBased() && "Unexpected call for non iOS target!");
    return TargetVersion < VersionTuple(V0, V1, V2);
  }

  bool isMacosxVersionLT(unsigned V0, unsigned V1 = 0, unsigned V2 = 0) const {
    assert(isTargetMacOS() && "Unexpected call for non OS X target!");
    return TargetVersion < VersionTuple(V0, V1, V2);
  }

  StringRef getPlatformFamily() const;
  static StringRef getSDKName(StringRef isysroot);
  StringRef getOSLibraryNameSuffix() const;

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

  CXXStdlibType GetDefaultCXXStdlibType() const override;
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
    if (isTargetIOSBased() || isTargetWatchOSBased())
      return 1;
    else if (isTargetMacOS() && !isMacosxVersionLT(10, 6))
      return 1;
    else if (isTargetMacOS() && !isMacosxVersionLT(10, 5) && !KernelOrKext)
      return 1;

    return 0;
  }

  bool SupportsObjCGC() const override;

  void CheckObjCARC() const override;

  bool UseSjLjExceptions(const llvm::opt::ArgList &Args) const override;

  bool SupportsEmbeddedBitcode() const override;

  SanitizerMask getSupportedSanitizers() const override;
};

/// DarwinClang - The Darwin toolchain used by Clang.
class LLVM_LIBRARY_VISIBILITY DarwinClang : public Darwin {
public:
  DarwinClang(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args);

  /// @name Apple ToolChain Implementation
  /// {

  RuntimeLibType GetRuntimeLibType(const llvm::opt::ArgList &Args) const override;

  void AddLinkRuntimeLibArgs(const llvm::opt::ArgList &Args,
                             llvm::opt::ArgStringList &CmdArgs) const override;

  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  void AddCCKextLibArgs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const override;

  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;

  void AddLinkARCArgs(const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs) const override;

  unsigned GetDefaultDwarfVersion() const override { return 2; }
  // Until dtrace (via CTF) and LLDB can deal with distributed debug info,
  // Darwin defaults to standalone/full debug info.
  bool GetDefaultStandaloneDebug() const override { return true; }
  llvm::DebuggerKind getDefaultDebuggerTuning() const override {
    return llvm::DebuggerKind::LLDB;
  }

  /// }

private:
  void AddLinkSanitizerLibArgs(const llvm::opt::ArgList &Args,
                               llvm::opt::ArgStringList &CmdArgs,
                               StringRef Sanitizer) const;
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

class LLVM_LIBRARY_VISIBILITY CloudABI : public Generic_ELF {
public:
  CloudABI(const Driver &D, const llvm::Triple &Triple,
           const llvm::opt::ArgList &Args);
  bool HasNativeLLVMSupport() const override { return true; }

  bool IsMathErrnoDefault() const override { return false; }
  bool IsObjCNonFragileABIDefault() const override { return true; }

  CXXStdlibType
  GetCXXStdlibType(const llvm::opt::ArgList &Args) const override {
    return ToolChain::CST_Libcxx;
  }
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  bool isPIEDefault() const override;
  SanitizerMask getSupportedSanitizers() const override;
  SanitizerMask getDefaultSanitizers() const override;

protected:
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY Solaris : public Generic_GCC {
public:
  Solaris(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  bool IsIntegratedAssemblerDefault() const override { return true; }

  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

  unsigned GetDefaultDwarfVersion() const override { return 2; }

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY MinGW : public ToolChain {
public:
  MinGW(const Driver &D, const llvm::Triple &Triple,
        const llvm::opt::ArgList &Args);

  bool IsIntegratedAssemblerDefault() const override;
  bool IsUnwindTablesDefault() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;
  bool UseSEHExceptions() const;

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

protected:
  Tool *getTool(Action::ActionClass AC) const override;
  Tool *buildLinker() const override;
  Tool *buildAssembler() const override;

private:
  std::string Base;
  std::string GccLibDir;
  std::string Ver;
  std::string Arch;
  mutable std::unique_ptr<tools::gcc::Preprocessor> Preprocessor;
  mutable std::unique_ptr<tools::gcc::Compiler> Compiler;
  void findGccLibDir();
};

class LLVM_LIBRARY_VISIBILITY Haiku : public Generic_ELF {
public:
  Haiku(const Driver &D, const llvm::Triple &Triple,
          const llvm::opt::ArgList &Args);

  bool isPIEDefault() const override { return getTriple().getArch() == llvm::Triple::x86_64; }

  void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                              llvm::opt::ArgStringList &CC1Args) const override;
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
  unsigned GetDefaultDwarfVersion() const override { return 2; }

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

  CXXStdlibType GetDefaultCXXStdlibType() const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
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

  CXXStdlibType GetDefaultCXXStdlibType() const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  bool UseSjLjExceptions(const llvm::opt::ArgList &Args) const override;
  bool isPIEDefault() const override;
  SanitizerMask getSupportedSanitizers() const override;
  unsigned GetDefaultDwarfVersion() const override { return 2; }
  // Until dtrace (via CTF) and LLDB can deal with distributed debug info,
  // FreeBSD defaults to standalone/full debug info.
  bool GetDefaultStandaloneDebug() const override { return true; }

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

  CXXStdlibType GetDefaultCXXStdlibType() const override;

  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  bool IsUnwindTablesDefault() const override { return true; }

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
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  void AddCudaIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const override;
  void AddIAMCUIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                           llvm::opt::ArgStringList &CC1Args) const override;
  bool isPIEDefault() const override;
  SanitizerMask getSupportedSanitizers() const override;
  void addProfileRTLibs(const llvm::opt::ArgList &Args,
                        llvm::opt::ArgStringList &CmdArgs) const override;
  virtual std::string computeSysRoot() const;

  virtual std::string getDynamicLinker(const llvm::opt::ArgList &Args) const;

  std::vector<std::string> ExtraOpts;

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY CudaToolChain : public Linux {
public:
  CudaToolChain(const Driver &D, const llvm::Triple &Triple,
                const llvm::opt::ArgList &Args);

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const override;

  // Never try to use the integrated assembler with CUDA; always fork out to
  // ptxas.
  bool useIntegratedAs() const override { return false; }

  void AddCudaIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const override;

  const Generic_GCC::CudaInstallationDetector &cudaInstallation() const {
    return CudaInstallation;
  }
  Generic_GCC::CudaInstallationDetector &cudaInstallation() {
    return CudaInstallation;
  }

protected:
  Tool *buildAssembler() const override;  // ptxas
  Tool *buildLinker() const override;     // fatbinary (ok, not really a linker)
};

class LLVM_LIBRARY_VISIBILITY MipsLLVMToolChain : public Linux {
protected:
  Tool *buildLinker() const override;

public:
  MipsLLVMToolChain(const Driver &D, const llvm::Triple &Triple,
                    const llvm::opt::ArgList &Args);

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;

  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;

  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  std::string getCompilerRT(const llvm::opt::ArgList &Args, StringRef Component,
                            bool Shared = false) const override;

  std::string computeSysRoot() const override;

  RuntimeLibType GetDefaultRuntimeLibType() const override {
    return GCCInstallation.isValid() ? RuntimeLibType::RLT_Libgcc
                                     : RuntimeLibType::RLT_CompilerRT;
  }

private:
  Multilib SelectedMultilib;
  std::string LibSuffix;
};

class LLVM_LIBRARY_VISIBILITY LanaiToolChain : public Generic_ELF {
public:
  LanaiToolChain(const Driver &D, const llvm::Triple &Triple,
                 const llvm::opt::ArgList &Args)
      : Generic_ELF(D, Triple, Args) {}
  bool IsIntegratedAssemblerDefault() const override { return true; }
};

class LLVM_LIBRARY_VISIBILITY HexagonToolChain : public Linux {
protected:
  GCCVersion GCCLibAndIncVersion;
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;

public:
  HexagonToolChain(const Driver &D, const llvm::Triple &Triple,
                   const llvm::opt::ArgList &Args);
  ~HexagonToolChain() override;

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;

  StringRef GetGCCLibAndIncVersion() const { return GCCLibAndIncVersion.Text; }
  bool IsIntegratedAssemblerDefault() const override {
    return true;
  }

  std::string getHexagonTargetDir(
      const std::string &InstalledDir,
      const SmallVectorImpl<std::string> &PrefixDirs) const;
  void getHexagonLibraryPaths(const llvm::opt::ArgList &Args,
      ToolChain::path_list &LibPaths) const;

  static const StringRef GetDefaultCPU();
  static const StringRef GetTargetCPUVersion(const llvm::opt::ArgList &Args);

  static Optional<unsigned> getSmallDataThreshold(
      const llvm::opt::ArgList &Args);
};

class LLVM_LIBRARY_VISIBILITY AMDGPUToolChain : public Generic_ELF {
protected:
  Tool *buildLinker() const override;

public:
  AMDGPUToolChain(const Driver &D, const llvm::Triple &Triple,
            const llvm::opt::ArgList &Args);
  unsigned GetDefaultDwarfVersion() const override { return 2; }
  bool IsIntegratedAssemblerDefault() const override { return true; }
};

class LLVM_LIBRARY_VISIBILITY NaClToolChain : public Generic_ELF {
public:
  NaClToolChain(const Driver &D, const llvm::Triple &Triple,
                const llvm::opt::ArgList &Args);

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;

  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  bool IsIntegratedAssemblerDefault() const override {
    return getTriple().getArch() == llvm::Triple::mipsel;
  }

  // Get the path to the file containing NaCl's ARM macros.
  // It lives in NaClToolChain because the ARMAssembler tool needs a
  // const char * that it can pass around,
  const char *GetNaClArmMacrosPath() const { return NaClArmMacrosPath.c_str(); }

  std::string ComputeEffectiveClangTriple(const llvm::opt::ArgList &Args,
                                          types::ID InputType) const override;

protected:
  Tool *buildLinker() const override;
  Tool *buildAssembler() const override;

private:
  std::string NaClArmMacrosPath;
};

/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
class LLVM_LIBRARY_VISIBILITY TCEToolChain : public ToolChain {
public:
  TCEToolChain(const Driver &D, const llvm::Triple &Triple,
               const llvm::opt::ArgList &Args);
  ~TCEToolChain() override;

  bool IsMathErrnoDefault() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;
};

class LLVM_LIBRARY_VISIBILITY MSVCToolChain : public ToolChain {
public:
  MSVCToolChain(const Driver &D, const llvm::Triple &Triple,
                const llvm::opt::ArgList &Args);

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const override;

  bool IsIntegratedAssemblerDefault() const override;
  bool IsUnwindTablesDefault() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

  bool getWindowsSDKDir(std::string &path, int &major,
                        std::string &windowsSDKIncludeVersion,
                        std::string &windowsSDKLibVersion) const;
  bool getWindowsSDKLibraryPath(std::string &path) const;
  /// \brief Check if Universal CRT should be used if available
  bool useUniversalCRT(std::string &visualStudioDir) const;
  bool getUniversalCRTSdkDir(std::string &path, std::string &ucrtVersion) const;
  bool getUniversalCRTLibraryPath(std::string &path) const;
  bool getVisualStudioInstallDir(std::string &path) const;
  bool getVisualStudioBinariesFolder(const char *clangProgramPath,
                                     std::string &path) const;
  VersionTuple getMSVCVersionFromExe() const override;

  std::string ComputeEffectiveClangTriple(const llvm::opt::ArgList &Args,
                                          types::ID InputType) const override;
  SanitizerMask getSupportedSanitizers() const override;

protected:
  void AddSystemIncludeWithSubfolder(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args,
                                     const std::string &folder,
                                     const Twine &subfolder1,
                                     const Twine &subfolder2 = "",
                                     const Twine &subfolder3 = "") const;

  Tool *buildLinker() const override;
  Tool *buildAssembler() const override;
};

class LLVM_LIBRARY_VISIBILITY CrossWindowsToolChain : public Generic_GCC {
public:
  CrossWindowsToolChain(const Driver &D, const llvm::Triple &T,
                        const llvm::opt::ArgList &Args);

  bool IsIntegratedAssemblerDefault() const override { return true; }
  bool IsUnwindTablesDefault() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;

  unsigned int GetDefaultStackProtectorLevel(bool KernelOrKext) const override {
    return 0;
  }

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  SanitizerMask getSupportedSanitizers() const override;

protected:
  Tool *buildLinker() const override;
  Tool *buildAssembler() const override;
};

class LLVM_LIBRARY_VISIBILITY XCoreToolChain : public ToolChain {
public:
  XCoreToolChain(const Driver &D, const llvm::Triple &Triple,
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
  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;
};

/// MyriadToolChain - A tool chain using either clang or the external compiler
/// installed by the Movidius SDK to perform all subcommands.
class LLVM_LIBRARY_VISIBILITY MyriadToolChain : public Generic_ELF {
public:
  MyriadToolChain(const Driver &D, const llvm::Triple &Triple,
                  const llvm::opt::ArgList &Args);
  ~MyriadToolChain() override;

  void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  Tool *SelectTool(const JobAction &JA) const override;
  unsigned GetDefaultDwarfVersion() const override { return 2; }
  SanitizerMask getSupportedSanitizers() const override;

protected:
  Tool *buildLinker() const override;
  bool isShaveCompilation(const llvm::Triple &T) const {
    return T.getArch() == llvm::Triple::shave;
  }

private:
  mutable std::unique_ptr<Tool> Compiler;
  mutable std::unique_ptr<Tool> Assembler;
};

class LLVM_LIBRARY_VISIBILITY WebAssembly final : public ToolChain {
public:
  WebAssembly(const Driver &D, const llvm::Triple &Triple,
              const llvm::opt::ArgList &Args);

private:
  bool IsMathErrnoDefault() const override;
  bool IsObjCNonFragileABIDefault() const override;
  bool UseObjCMixedDispatch() const override;
  bool isPICDefault() const override;
  bool isPIEDefault() const override;
  bool isPICDefaultForced() const override;
  bool IsIntegratedAssemblerDefault() const override;
  bool hasBlocksRuntime() const override;
  bool SupportsObjCGC() const override;
  bool SupportsProfiling() const override;
  bool HasNativeLLVMSupport() const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const override;
  RuntimeLibType GetDefaultRuntimeLibType() const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  void AddClangSystemIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

  Tool *buildLinker() const override;
};

class LLVM_LIBRARY_VISIBILITY PS4CPU : public Generic_ELF {
public:
  PS4CPU(const Driver &D, const llvm::Triple &Triple,
         const llvm::opt::ArgList &Args);

  bool IsMathErrnoDefault() const override { return false; }
  bool IsObjCNonFragileABIDefault() const override { return true; }
  bool HasNativeLLVMSupport() const override;
  bool isPICDefault() const override;

  unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const override {
    return 2; // SSPStrong
  }

  llvm::DebuggerKind getDefaultDebuggerTuning() const override {
    return llvm::DebuggerKind::SCE;
  }

  SanitizerMask getSupportedSanitizers() const override;

protected:
  Tool *buildAssembler() const override;
  Tool *buildLinker() const override;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_H
