//===--- ToolChain.h - Collections of tools for one platform ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_TOOLCHAIN_H
#define LLVM_CLANG_DRIVER_TOOLCHAIN_H

#include "clang/Basic/Sanitizers.h"
#include "clang/Driver/Action.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/Types.h"
#include "clang/Driver/Util.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetOptions.h"
#include <memory>
#include <string>

namespace llvm {
namespace opt {
  class ArgList;
  class DerivedArgList;
  class InputArgList;
}
}

namespace clang {
class ObjCRuntime;
namespace vfs {
class FileSystem;
}

namespace driver {
  class Compilation;
  class Driver;
  class JobAction;
  class SanitizerArgs;
  class Tool;

/// ToolChain - Access to tools for a single platform.
class ToolChain {
public:
  typedef SmallVector<std::string, 16> path_list;

  enum CXXStdlibType {
    CST_Libcxx,
    CST_Libstdcxx
  };

  enum RuntimeLibType {
    RLT_CompilerRT,
    RLT_Libgcc
  };

  enum RTTIMode {
    RM_EnabledExplicitly,
    RM_EnabledImplicitly,
    RM_DisabledExplicitly,
    RM_DisabledImplicitly
  };

private:
  const Driver &D;
  const llvm::Triple Triple;
  const llvm::opt::ArgList &Args;
  // We need to initialize CachedRTTIArg before CachedRTTIMode
  const llvm::opt::Arg *const CachedRTTIArg;
  const RTTIMode CachedRTTIMode;

  /// The list of toolchain specific path prefixes to search for
  /// files.
  path_list FilePaths;

  /// The list of toolchain specific path prefixes to search for
  /// programs.
  path_list ProgramPaths;

  mutable std::unique_ptr<Tool> Clang;
  mutable std::unique_ptr<Tool> Assemble;
  mutable std::unique_ptr<Tool> Link;
  Tool *getClang() const;
  Tool *getAssemble() const;
  Tool *getLink() const;
  Tool *getClangAs() const;

  mutable std::unique_ptr<SanitizerArgs> SanitizerArguments;

protected:
  MultilibSet Multilibs;
  const char *DefaultLinker = "ld";

  ToolChain(const Driver &D, const llvm::Triple &T,
            const llvm::opt::ArgList &Args);

  virtual Tool *buildAssembler() const;
  virtual Tool *buildLinker() const;
  virtual Tool *getTool(Action::ActionClass AC) const;

  /// \name Utilities for implementing subclasses.
  ///@{
  static void addSystemInclude(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args,
                               const Twine &Path);
  static void addExternCSystemInclude(const llvm::opt::ArgList &DriverArgs,
                                      llvm::opt::ArgStringList &CC1Args,
                                      const Twine &Path);
  static void
      addExternCSystemIncludeIfExists(const llvm::opt::ArgList &DriverArgs,
                                      llvm::opt::ArgStringList &CC1Args,
                                      const Twine &Path);
  static void addSystemIncludes(const llvm::opt::ArgList &DriverArgs,
                                llvm::opt::ArgStringList &CC1Args,
                                ArrayRef<StringRef> Paths);
  ///@}

public:
  virtual ~ToolChain();

  // Accessors

  const Driver &getDriver() const { return D; }
  vfs::FileSystem &getVFS() const;
  const llvm::Triple &getTriple() const { return Triple; }

  llvm::Triple::ArchType getArch() const { return Triple.getArch(); }
  StringRef getArchName() const { return Triple.getArchName(); }
  StringRef getPlatform() const { return Triple.getVendorName(); }
  StringRef getOS() const { return Triple.getOSName(); }

  /// \brief Provide the default architecture name (as expected by -arch) for
  /// this toolchain. Note t
  StringRef getDefaultUniversalArchName() const;

  std::string getTripleString() const {
    return Triple.getTriple();
  }

  path_list &getFilePaths() { return FilePaths; }
  const path_list &getFilePaths() const { return FilePaths; }

  path_list &getProgramPaths() { return ProgramPaths; }
  const path_list &getProgramPaths() const { return ProgramPaths; }

  const MultilibSet &getMultilibs() const { return Multilibs; }

  const SanitizerArgs& getSanitizerArgs() const;

  // Returns the Arg * that explicitly turned on/off rtti, or nullptr.
  const llvm::opt::Arg *getRTTIArg() const { return CachedRTTIArg; }

  // Returns the RTTIMode for the toolchain with the current arguments.
  RTTIMode getRTTIMode() const { return CachedRTTIMode; }

  /// \brief Return any implicit target and/or mode flag for an invocation of
  /// the compiler driver as `ProgName`.
  ///
  /// For example, when called with i686-linux-android-g++, the first element
  /// of the return value will be set to `"i686-linux-android"` and the second
  /// will be set to "--driver-mode=g++"`.
  ///
  /// \pre `llvm::InitializeAllTargets()` has been called.
  /// \param ProgName The name the Clang driver was invoked with (from,
  /// e.g., argv[0])
  /// \return A pair of (`target`, `mode-flag`), where one or both may be empty.
  static std::pair<std::string, std::string>
  getTargetAndModeFromProgramName(StringRef ProgName);

  // Tool access.

  /// TranslateArgs - Create a new derived argument list for any argument
  /// translations this ToolChain may wish to perform, or 0 if no tool chain
  /// specific translations are needed.
  ///
  /// \param BoundArch - The bound architecture name, or 0.
  virtual llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args,
                const char *BoundArch) const {
    return nullptr;
  }

  /// Choose a tool to use to handle the action \p JA.
  ///
  /// This can be overridden when a particular ToolChain needs to use
  /// a compiler other than Clang.
  virtual Tool *SelectTool(const JobAction &JA) const;

  // Helper methods

  std::string GetFilePath(const char *Name) const;
  std::string GetProgramPath(const char *Name) const;

  /// Returns the linker path, respecting the -fuse-ld= argument to determine
  /// the linker suffix or name.
  std::string GetLinkerPath() const;

  /// \brief Dispatch to the specific toolchain for verbose printing.
  ///
  /// This is used when handling the verbose option to print detailed,
  /// toolchain-specific information useful for understanding the behavior of
  /// the driver on a specific platform.
  virtual void printVerboseInfo(raw_ostream &OS) const {}

  // Platform defaults information

  /// \brief Returns true if the toolchain is targeting a non-native
  /// architecture.
  virtual bool isCrossCompiling() const;

  /// HasNativeLTOLinker - Check whether the linker and related tools have
  /// native LLVM support.
  virtual bool HasNativeLLVMSupport() const;

  /// LookupTypeForExtension - Return the default language type to use for the
  /// given extension.
  virtual types::ID LookupTypeForExtension(const char *Ext) const;

  /// IsBlocksDefault - Does this tool chain enable -fblocks by default.
  virtual bool IsBlocksDefault() const { return false; }

  /// IsIntegratedAssemblerDefault - Does this tool chain enable -integrated-as
  /// by default.
  virtual bool IsIntegratedAssemblerDefault() const { return false; }

  /// \brief Check if the toolchain should use the integrated assembler.
  bool useIntegratedAs() const;

  /// IsMathErrnoDefault - Does this tool chain use -fmath-errno by default.
  virtual bool IsMathErrnoDefault() const { return true; }

  /// IsEncodeExtendedBlockSignatureDefault - Does this tool chain enable
  /// -fencode-extended-block-signature by default.
  virtual bool IsEncodeExtendedBlockSignatureDefault() const { return false; }

  /// IsObjCNonFragileABIDefault - Does this tool chain set
  /// -fobjc-nonfragile-abi by default.
  virtual bool IsObjCNonFragileABIDefault() const { return false; }

  /// UseObjCMixedDispatchDefault - When using non-legacy dispatch, should the
  /// mixed dispatch method be used?
  virtual bool UseObjCMixedDispatch() const { return false; }

  /// GetDefaultStackProtectorLevel - Get the default stack protector level for
  /// this tool chain (0=off, 1=on, 2=strong, 3=all).
  virtual unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const {
    return 0;
  }

  /// GetDefaultRuntimeLibType - Get the default runtime library variant to use.
  virtual RuntimeLibType GetDefaultRuntimeLibType() const {
    return ToolChain::RLT_Libgcc;
  }

  virtual std::string getCompilerRT(const llvm::opt::ArgList &Args,
                                    StringRef Component,
                                    bool Shared = false) const;

  const char *getCompilerRTArgString(const llvm::opt::ArgList &Args,
                                     StringRef Component,
                                     bool Shared = false) const;
  /// needsProfileRT - returns true if instrumentation profile is on.
  static bool needsProfileRT(const llvm::opt::ArgList &Args);

  /// IsUnwindTablesDefault - Does this tool chain use -funwind-tables
  /// by default.
  virtual bool IsUnwindTablesDefault() const;

  /// \brief Test whether this toolchain defaults to PIC.
  virtual bool isPICDefault() const = 0;

  /// \brief Test whether this toolchain defaults to PIE.
  virtual bool isPIEDefault() const = 0;

  /// \brief Tests whether this toolchain forces its default for PIC, PIE or
  /// non-PIC.  If this returns true, any PIC related flags should be ignored
  /// and instead the results of \c isPICDefault() and \c isPIEDefault() are
  /// used exclusively.
  virtual bool isPICDefaultForced() const = 0;

  /// SupportsProfiling - Does this tool chain support -pg.
  virtual bool SupportsProfiling() const { return true; }

  /// Does this tool chain support Objective-C garbage collection.
  virtual bool SupportsObjCGC() const { return true; }

  /// Complain if this tool chain doesn't support Objective-C ARC.
  virtual void CheckObjCARC() const {}

  /// UseDwarfDebugFlags - Embed the compile options to clang into the Dwarf
  /// compile unit information.
  virtual bool UseDwarfDebugFlags() const { return false; }

  // Return the DWARF version to emit, in the absence of arguments
  // to the contrary.
  virtual unsigned GetDefaultDwarfVersion() const { return 4; }

  // True if the driver should assume "-fstandalone-debug"
  // in the absence of an option specifying otherwise,
  // provided that debugging was requested in the first place.
  // i.e. a value of 'true' does not imply that debugging is wanted.
  virtual bool GetDefaultStandaloneDebug() const { return false; }

  // Return the default debugger "tuning."
  virtual llvm::DebuggerKind getDefaultDebuggerTuning() const {
    return llvm::DebuggerKind::GDB;
  }

  /// UseSjLjExceptions - Does this tool chain use SjLj exceptions.
  virtual bool UseSjLjExceptions(const llvm::opt::ArgList &Args) const {
    return false;
  }

  /// getThreadModel() - Which thread model does this target use?
  virtual std::string getThreadModel() const { return "posix"; }

  /// isThreadModelSupported() - Does this target support a thread model?
  virtual bool isThreadModelSupported(const StringRef Model) const;

  /// ComputeLLVMTriple - Return the LLVM target triple to use, after taking
  /// command line arguments into account.
  virtual std::string
  ComputeLLVMTriple(const llvm::opt::ArgList &Args,
                    types::ID InputType = types::TY_INVALID) const;

  /// ComputeEffectiveClangTriple - Return the Clang triple to use for this
  /// target, which may take into account the command line arguments. For
  /// example, on Darwin the -mmacosx-version-min= command line argument (which
  /// sets the deployment target) determines the version in the triple passed to
  /// Clang.
  virtual std::string ComputeEffectiveClangTriple(
      const llvm::opt::ArgList &Args,
      types::ID InputType = types::TY_INVALID) const;

  /// getDefaultObjCRuntime - Return the default Objective-C runtime
  /// for this platform.
  ///
  /// FIXME: this really belongs on some sort of DeploymentTarget abstraction
  virtual ObjCRuntime getDefaultObjCRuntime(bool isNonFragile) const;

  /// hasBlocksRuntime - Given that the user is compiling with
  /// -fblocks, does this tool chain guarantee the existence of a
  /// blocks runtime?
  ///
  /// FIXME: this really belongs on some sort of DeploymentTarget abstraction
  virtual bool hasBlocksRuntime() const { return true; }

  /// \brief Add the clang cc1 arguments for system include paths.
  ///
  /// This routine is responsible for adding the necessary cc1 arguments to
  /// include headers from standard system header directories.
  virtual void
  AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const;

  /// \brief Add options that need to be passed to cc1 for this target.
  virtual void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                                     llvm::opt::ArgStringList &CC1Args) const;

  /// \brief Add warning options that need to be passed to cc1 for this target.
  virtual void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const;

  // GetRuntimeLibType - Determine the runtime library type to use with the
  // given compilation arguments.
  virtual RuntimeLibType
  GetRuntimeLibType(const llvm::opt::ArgList &Args) const;

  // GetCXXStdlibType - Determine the C++ standard library type to use with the
  // given compilation arguments.
  virtual CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const;

  /// AddClangCXXStdlibIncludeArgs - Add the clang -cc1 level arguments to set
  /// the include paths to use for the given C++ standard library type.
  virtual void
  AddClangCXXStdlibIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                               llvm::opt::ArgStringList &CC1Args) const;

  /// AddCXXStdlibLibArgs - Add the system specific linker arguments to use
  /// for the given C++ standard library type.
  virtual void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                                   llvm::opt::ArgStringList &CmdArgs) const;

  /// AddFilePathLibArgs - Add each thing in getFilePaths() as a "-L" option.
  void AddFilePathLibArgs(const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs) const;

  /// AddCCKextLibArgs - Add the system specific linker arguments to use
  /// for kernel extensions (Darwin-specific).
  virtual void AddCCKextLibArgs(const llvm::opt::ArgList &Args,
                                llvm::opt::ArgStringList &CmdArgs) const;

  /// AddFastMathRuntimeIfAvailable - If a runtime library exists that sets
  /// global flags for unsafe floating point math, add it and return true.
  ///
  /// This checks for presence of the -Ofast, -ffast-math or -funsafe-math flags.
  virtual bool AddFastMathRuntimeIfAvailable(
      const llvm::opt::ArgList &Args, llvm::opt::ArgStringList &CmdArgs) const;
  /// addProfileRTLibs - When -fprofile-instr-profile is specified, try to pass
  /// a suitable profile runtime library to the linker.
  virtual void addProfileRTLibs(const llvm::opt::ArgList &Args,
                                llvm::opt::ArgStringList &CmdArgs) const;

  /// \brief Add arguments to use system-specific CUDA includes.
  virtual void AddCudaIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                                  llvm::opt::ArgStringList &CC1Args) const;

  /// \brief Return sanitizers which are available in this toolchain.
  virtual SanitizerMask getSupportedSanitizers() const;
};

} // end namespace driver
} // end namespace clang

#endif
