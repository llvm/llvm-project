//===--- Darwin.h - Darwin ToolChain Implementations ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_DARWIN_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_DARWIN_H

#include "Cuda.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {

namespace toolchains {
class MachO;
} // end namespace toolchains

namespace tools {

namespace darwin {
llvm::Triple::ArchType getArchTypeForMachOArchName(StringRef Str);
void setTripleTypeForMachOArchName(llvm::Triple &T, StringRef Str);

class LLVM_LIBRARY_VISIBILITY MachOTool : public Tool {
  virtual void anchor();

protected:
  void AddMachOArch(const llvm::opt::ArgList &Args,
                    llvm::opt::ArgStringList &CmdArgs) const;

  const toolchains::MachO &getMachOToolChain() const {
    return reinterpret_cast<const toolchains::MachO &>(getToolChain());
  }

public:
  MachOTool(
      const char *Name, const char *ShortName, const ToolChain &TC,
      ResponseFileSupport ResponseSupport = RF_None,
      llvm::sys::WindowsEncodingMethod ResponseEncoding = llvm::sys::WEM_UTF8,
      const char *ResponseFlag = "@")
      : Tool(Name, ShortName, TC, ResponseSupport, ResponseEncoding,
             ResponseFlag) {}
};

class LLVM_LIBRARY_VISIBILITY Assembler : public MachOTool {
public:
  Assembler(const ToolChain &TC)
      : MachOTool("darwin::Assembler", "assembler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Linker : public MachOTool {
  bool NeedsTempPath(const InputInfoList &Inputs) const;
  void AddLinkArgs(Compilation &C, const llvm::opt::ArgList &Args,
                   llvm::opt::ArgStringList &CmdArgs,
                   const InputInfoList &Inputs) const;

public:
  Linker(const ToolChain &TC)
      : MachOTool("darwin::Linker", "linker", TC, RF_FileList,
                  llvm::sys::WEM_UTF8, "-filelist") {}

  bool hasIntegratedCPP() const override { return false; }
  bool isLinkJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Lipo : public MachOTool {
public:
  Lipo(const ToolChain &TC) : MachOTool("darwin::Lipo", "lipo", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Dsymutil : public MachOTool {
public:
  Dsymutil(const ToolChain &TC)
      : MachOTool("darwin::Dsymutil", "dsymutil", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  bool isDsymutilJob() const override { return true; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY VerifyDebug : public MachOTool {
public:
  VerifyDebug(const ToolChain &TC)
      : MachOTool("darwin::VerifyDebug", "dwarfdump", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};
} // end namespace darwin
} // end namespace tools

namespace toolchains {

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
  virtual void AddFuzzerLinkArgs(const llvm::opt::ArgList &Args,
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

  types::ID LookupTypeForExtension(StringRef Ext) const override;

  bool HasNativeLLVMSupport() const override;

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;

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

  CudaInstallationDetector CudaInstallation;

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

  /// Return true if c++17 aligned allocation/deallocation functions are not
  /// implemented in the c++ standard library of the deployment target we are
  /// targeting.
  bool isAlignedAllocationUnavailable() const;

  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                             llvm::opt::ArgStringList &CC1Args) const override;

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
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;

  CXXStdlibType GetDefaultCXXStdlibType() const override;
  ObjCRuntime getDefaultObjCRuntime(bool isNonFragile) const override;
  bool hasBlocksRuntime() const override;

  void AddCudaIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const override;

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

  void printVerboseInfo(raw_ostream &OS) const override;
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

  unsigned GetDefaultDwarfVersion() const override;
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

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_DARWIN_H
