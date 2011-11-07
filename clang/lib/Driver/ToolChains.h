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

#include "clang/Driver/Action.h"
#include "clang/Driver/ToolChain.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Compiler.h"

#include "Tools.h"

namespace clang {
namespace driver {
namespace toolchains {

/// Generic_GCC - A tool chain using the 'gcc' command to perform
/// all subcommands; this relies on gcc translating the majority of
/// command line options.
class LLVM_LIBRARY_VISIBILITY Generic_GCC : public ToolChain {
protected:
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

public:
  Generic_GCC(const HostInfo &Host, const llvm::Triple& Triple);
  ~Generic_GCC();

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;

  virtual bool IsUnwindTablesDefault() const;
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;
};

/// Darwin - The base Darwin tool chain.
class LLVM_LIBRARY_VISIBILITY Darwin : public ToolChain {
public:
  /// The host version.
  unsigned DarwinVersion[3];

private:
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

  /// Whether the information on the target has been initialized.
  //
  // FIXME: This should be eliminated. What we want to do is make this part of
  // the "default target for arguments" selection process, once we get out of
  // the argument translation business.
  mutable bool TargetInitialized;

  // FIXME: Remove this once there is a proper way to detect an ARC runtime
  // for the simulator.
 public:
  mutable enum {
    ARCSimulator_None,
    ARCSimulator_HasARCRuntime,
    ARCSimulator_NoARCRuntime
  } ARCRuntimeForSimulator;

  mutable enum {
    LibCXXSimulator_None,
    LibCXXSimulator_NotAvailable,
    LibCXXSimulator_Available
  } LibCXXForSimulator;

private:
  /// Whether we are targeting iPhoneOS target.
  mutable bool TargetIsIPhoneOS;

  /// Whether we are targeting the iPhoneOS simulator target.
  mutable bool TargetIsIPhoneOSSimulator;

  /// The OS version we are targeting.
  mutable unsigned TargetVersion[3];

  /// The default macosx-version-min of this tool chain; empty until
  /// initialized.
  std::string MacosxVersionMin;

  bool hasARCRuntime() const;

private:
  void AddDeploymentTarget(DerivedArgList &Args) const;

public:
  Darwin(const HostInfo &Host, const llvm::Triple& Triple);
  ~Darwin();

  std::string ComputeEffectiveClangTriple(const ArgList &Args,
                                          types::ID InputType) const;

  /// @name Darwin Specific Toolchain API
  /// {

  // FIXME: Eliminate these ...Target functions and derive separate tool chains
  // for these targets and put version in constructor.
  void setTarget(bool IsIPhoneOS, unsigned Major, unsigned Minor,
                 unsigned Micro, bool IsIOSSim) const {
    assert((!IsIOSSim || IsIPhoneOS) && "Unexpected deployment target!");

    // FIXME: For now, allow reinitialization as long as values don't
    // change. This will go away when we move away from argument translation.
    if (TargetInitialized && TargetIsIPhoneOS == IsIPhoneOS &&
        TargetIsIPhoneOSSimulator == IsIOSSim &&
        TargetVersion[0] == Major && TargetVersion[1] == Minor &&
        TargetVersion[2] == Micro)
      return;

    assert(!TargetInitialized && "Target already initialized!");
    TargetInitialized = true;
    TargetIsIPhoneOS = IsIPhoneOS;
    TargetIsIPhoneOSSimulator = IsIOSSim;
    TargetVersion[0] = Major;
    TargetVersion[1] = Minor;
    TargetVersion[2] = Micro;
  }

  bool isTargetIPhoneOS() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetIsIPhoneOS;
  }

  bool isTargetIOSSimulator() const {
    assert(TargetInitialized && "Target not initialized!");
    return TargetIsIPhoneOSSimulator;
  }

  bool isTargetInitialized() const { return TargetInitialized; }

  void getTargetVersion(unsigned (&Res)[3]) const {
    assert(TargetInitialized && "Target not initialized!");
    Res[0] = TargetVersion[0];
    Res[1] = TargetVersion[1];
    Res[2] = TargetVersion[2];
  }

  /// getDarwinArchName - Get the "Darwin" arch name for a particular compiler
  /// invocation. For example, Darwin treats different ARM variations as
  /// distinct architectures.
  StringRef getDarwinArchName(const ArgList &Args) const;

  static bool isVersionLT(unsigned (&A)[3], unsigned (&B)[3]) {
    for (unsigned i=0; i < 3; ++i) {
      if (A[i] > B[i]) return false;
      if (A[i] < B[i]) return true;
    }
    return false;
  }

  bool isIPhoneOSVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(isTargetIPhoneOS() && "Unexpected call for OS X target!");
    unsigned B[3] = { V0, V1, V2 };
    return isVersionLT(TargetVersion, B);
  }

  bool isMacosxVersionLT(unsigned V0, unsigned V1=0, unsigned V2=0) const {
    assert(!isTargetIPhoneOS() && "Unexpected call for iPhoneOS target!");
    unsigned B[3] = { V0, V1, V2 };
    return isVersionLT(TargetVersion, B);
  }

  /// AddLinkSearchPathArgs - Add the linker search paths to \arg CmdArgs.
  ///
  /// \param Args - The input argument list.
  /// \param CmdArgs [out] - The command argument list to append the paths
  /// (prefixed by -L) to.
  virtual void AddLinkSearchPathArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const = 0;

  /// AddLinkARCArgs - Add the linker arguments to link the ARC runtime library.
  virtual void AddLinkARCArgs(const ArgList &Args,
                              ArgStringList &CmdArgs) const = 0;
  
  /// AddLinkRuntimeLibArgs - Add the linker arguments to link the compiler
  /// runtime library.
  virtual void AddLinkRuntimeLibArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const = 0;
  
  /// }
  /// @name ToolChain Implementation
  /// {

  virtual types::ID LookupTypeForExtension(const char *Ext) const;

  virtual bool HasNativeLLVMSupport() const;

  virtual void configureObjCRuntime(ObjCRuntime &runtime) const;
  virtual bool hasBlocksRuntime() const;

  virtual DerivedArgList *TranslateArgs(const DerivedArgList &Args,
                                        const char *BoundArch) const;

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;

  virtual bool IsBlocksDefault() const {
    // Always allow blocks on Darwin; users interested in versioning are
    // expected to use /usr/include/Blocks.h.
    return true;
  }
  virtual bool IsIntegratedAssemblerDefault() const {
#ifdef DISABLE_DEFAULT_INTEGRATED_ASSEMBLER
    return false;
#else
    // Default integrated assembler to on for x86.
    return (getTriple().getArch() == llvm::Triple::x86 ||
            getTriple().getArch() == llvm::Triple::x86_64);
#endif
  }
  virtual bool IsStrictAliasingDefault() const {
#ifdef DISABLE_DEFAULT_STRICT_ALIASING
    return false;
#else
    return ToolChain::IsStrictAliasingDefault();
#endif
  }
  
  virtual bool IsObjCDefaultSynthPropertiesDefault() const {
    return false;
  }

  virtual bool IsObjCNonFragileABIDefault() const {
    // Non-fragile ABI is default for everything but i386.
    return getTriple().getArch() != llvm::Triple::x86;
  }
  virtual bool IsObjCLegacyDispatchDefault() const {
    // This is only used with the non-fragile ABI.

    // Legacy dispatch is used everywhere except on x86_64.
    return getTriple().getArch() != llvm::Triple::x86_64;
  }
  virtual bool UseObjCMixedDispatch() const {
    // This is only used with the non-fragile ABI and non-legacy dispatch.

    // Mixed dispatch is used everywhere except OS X before 10.6.
    return !(!isTargetIPhoneOS() && isMacosxVersionLT(10, 6));
  }
  virtual bool IsUnwindTablesDefault() const;
  virtual unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const {
    // Stack protectors default to on for user code on 10.5,
    // and for everything in 10.6 and beyond
    return !isTargetIPhoneOS() &&
      (!isMacosxVersionLT(10, 6) ||
         (!isMacosxVersionLT(10, 5) && !KernelOrKext));
  }
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;

  virtual bool SupportsProfiling() const;

  virtual bool SupportsObjCGC() const;

  virtual bool UseDwarfDebugFlags() const;

  virtual bool UseSjLjExceptions() const;

  /// }
};

/// DarwinClang - The Darwin toolchain used by Clang.
class LLVM_LIBRARY_VISIBILITY DarwinClang : public Darwin {
private:
  void AddGCCLibexecPath(unsigned darwinVersion);

public:
  DarwinClang(const HostInfo &Host, const llvm::Triple& Triple);

  /// @name Darwin ToolChain Implementation
  /// {

  virtual void AddLinkSearchPathArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const;

  virtual void AddLinkRuntimeLibArgs(const ArgList &Args,
                                     ArgStringList &CmdArgs) const;
  void AddLinkRuntimeLib(const ArgList &Args, ArgStringList &CmdArgs, 
                         const char *DarwinStaticLib) const;
  
  virtual void AddCXXStdlibLibArgs(const ArgList &Args,
                                   ArgStringList &CmdArgs) const;

  virtual void AddCCKextLibArgs(const ArgList &Args,
                                ArgStringList &CmdArgs) const;

  virtual void AddLinkARCArgs(const ArgList &Args,
                              ArgStringList &CmdArgs) const;
  /// }
};

/// Darwin_Generic_GCC - Generic Darwin tool chain using gcc.
class LLVM_LIBRARY_VISIBILITY Darwin_Generic_GCC : public Generic_GCC {
public:
  Darwin_Generic_GCC(const HostInfo &Host, const llvm::Triple& Triple)
    : Generic_GCC(Host, Triple) {}

  std::string ComputeEffectiveClangTriple(const ArgList &Args,
                                          types::ID InputType) const;

  virtual const char *GetDefaultRelocationModel() const { return "pic"; }
};

class LLVM_LIBRARY_VISIBILITY Generic_ELF : public Generic_GCC {
 public:
  Generic_ELF(const HostInfo &Host, const llvm::Triple& Triple)
    : Generic_GCC(Host, Triple) {}

  virtual bool IsIntegratedAssemblerDefault() const {
    // Default integrated assembler to on for x86.
    return (getTriple().getArch() == llvm::Triple::x86 ||
            getTriple().getArch() == llvm::Triple::x86_64);
  }
};

class LLVM_LIBRARY_VISIBILITY AuroraUX : public Generic_GCC {
public:
  AuroraUX(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;
};

class LLVM_LIBRARY_VISIBILITY OpenBSD : public Generic_ELF {
public:
  OpenBSD(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;
};

class LLVM_LIBRARY_VISIBILITY FreeBSD : public Generic_ELF {
public:
  FreeBSD(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;
};

class LLVM_LIBRARY_VISIBILITY NetBSD : public Generic_ELF {
  const llvm::Triple ToolTriple;

public:
  NetBSD(const HostInfo &Host, const llvm::Triple& Triple,
         const llvm::Triple& ToolTriple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;
};

class LLVM_LIBRARY_VISIBILITY Minix : public Generic_GCC {
public:
  Minix(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;
};

class LLVM_LIBRARY_VISIBILITY DragonFly : public Generic_ELF {
public:
  DragonFly(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;
};

class LLVM_LIBRARY_VISIBILITY Linux : public Generic_ELF {
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

    /// \brief Any textual suffix on the patch number.
    std::string PatchSuffix;

    static GCCVersion Parse(StringRef VersionText);
    bool operator<(const GCCVersion &RHS) const;
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
    std::string GccTriple;

    // FIXME: These might be better as path objects.
    std::string GccInstallPath;
    std::string GccParentLibPath;

    GCCVersion Version;

  public:
    GCCInstallationDetector(const Driver &D);

    /// \brief Check whether we detected a valid GCC install.
    bool isValid() const { return IsValid; }

    /// \brief Get the GCC triple for the detected install.
    StringRef getTriple() const { return GccTriple; }

    /// \brief Get the detected GCC installation path.
    StringRef getInstallPath() const { return GccInstallPath; }

    /// \brief Get the detected GCC parent lib path.
    StringRef getParentLibPath() const { return GccParentLibPath; }

    /// \brief Get the detected GCC version string.
    StringRef getVersion() const { return Version.Text; }

  private:
    static void CollectLibDirsAndTriples(llvm::Triple::ArchType HostArch,
                                         SmallVectorImpl<StringRef> &LibDirs,
                                         SmallVectorImpl<StringRef> &Triples);

    void ScanLibDirForGCCTriple(const std::string &LibDir,
                                StringRef CandidateTriple);
  };

  GCCInstallationDetector GCCInstallation;

public:
  Linux(const HostInfo &Host, const llvm::Triple& Triple);

  virtual bool HasNativeLLVMSupport() const;

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;

  virtual void AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const;
  virtual void AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args) const;

  std::string Linker;
  std::vector<std::string> ExtraOpts;
};


/// TCEToolChain - A tool chain using the llvm bitcode tools to perform
/// all subcommands. See http://tce.cs.tut.fi for our peculiar target.
class LLVM_LIBRARY_VISIBILITY TCEToolChain : public ToolChain {
public:
  TCEToolChain(const HostInfo &Host, const llvm::Triple& Triple);
  ~TCEToolChain();

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;
  bool IsMathErrnoDefault() const;
  bool IsUnwindTablesDefault() const;
  const char* GetDefaultRelocationModel() const;
  const char* GetForcedPicModel() const;

private:
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

};

class LLVM_LIBRARY_VISIBILITY Windows : public ToolChain {
  mutable llvm::DenseMap<unsigned, Tool*> Tools;

public:
  Windows(const HostInfo &Host, const llvm::Triple& Triple);

  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const;

  virtual bool IsIntegratedAssemblerDefault() const;
  virtual bool IsUnwindTablesDefault() const;
  virtual const char *GetDefaultRelocationModel() const;
  virtual const char *GetForcedPicModel() const;

  virtual void AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const;
  virtual void AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args) const;

};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif
