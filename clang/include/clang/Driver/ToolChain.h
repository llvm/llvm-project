//===--- ToolChain.h - Collections of tools for one platform ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_TOOLCHAIN_H_
#define CLANG_DRIVER_TOOLCHAIN_H_

#include "clang/Driver/Util.h"
#include "clang/Driver/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Path.h"
#include <string>

namespace clang {
  class ObjCRuntime;

namespace driver {
  class ArgList;
  class Compilation;
  class DerivedArgList;
  class Driver;
  class InputArgList;
  class JobAction;
  class Tool;

/// ToolChain - Access to tools for a single platform.
class ToolChain {
public:
  typedef SmallVector<std::string, 4> path_list;

  enum CXXStdlibType {
    CST_Libcxx,
    CST_Libstdcxx
  };

  enum RuntimeLibType {
    RLT_CompilerRT,
    RLT_Libgcc
  };

private:
  const Driver &D;
  const llvm::Triple Triple;

  /// The list of toolchain specific path prefixes to search for
  /// files.
  path_list FilePaths;

  /// The list of toolchain specific path prefixes to search for
  /// programs.
  path_list ProgramPaths;

protected:
  ToolChain(const Driver &D, const llvm::Triple &T);

  /// \name Utilities for implementing subclasses.
  ///@{
  static void addSystemInclude(const ArgList &DriverArgs,
                               ArgStringList &CC1Args,
                               const Twine &Path);
  static void addExternCSystemInclude(const ArgList &DriverArgs,
                                      ArgStringList &CC1Args,
                                      const Twine &Path);
  static void addSystemIncludes(const ArgList &DriverArgs,
                                ArgStringList &CC1Args,
                                ArrayRef<StringRef> Paths);
  ///@}

public:
  virtual ~ToolChain();

  // Accessors

  const Driver &getDriver() const;
  const llvm::Triple &getTriple() const { return Triple; }

  llvm::Triple::ArchType getArch() const { return Triple.getArch(); }
  StringRef getArchName() const { return Triple.getArchName(); }
  StringRef getPlatform() const { return Triple.getVendorName(); }
  StringRef getOS() const { return Triple.getOSName(); }

  /// \brief Provide the default architecture name (as expected by -arch) for
  /// this toolchain. Note t
  std::string getDefaultUniversalArchName() const;

  std::string getTripleString() const {
    return Triple.getTriple();
  }

  path_list &getFilePaths() { return FilePaths; }
  const path_list &getFilePaths() const { return FilePaths; }

  path_list &getProgramPaths() { return ProgramPaths; }
  const path_list &getProgramPaths() const { return ProgramPaths; }

  // Tool access.

  /// TranslateArgs - Create a new derived argument list for any argument
  /// translations this ToolChain may wish to perform, or 0 if no tool chain
  /// specific translations are needed.
  ///
  /// \param BoundArch - The bound architecture name, or 0.
  virtual DerivedArgList *TranslateArgs(const DerivedArgList &Args,
                                        const char *BoundArch) const {
    return 0;
  }

  /// SelectTool - Choose a tool to use to handle the action \p JA with the
  /// given \p Inputs.
  virtual Tool &SelectTool(const Compilation &C, const JobAction &JA,
                           const ActionList &Inputs) const = 0;

  // Helper methods

  std::string GetFilePath(const char *Name) const;
  std::string GetProgramPath(const char *Name) const;

  // Platform defaults information

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

  /// IsStrictAliasingDefault - Does this tool chain use -fstrict-aliasing by
  /// default.
  virtual bool IsStrictAliasingDefault() const { return true; }

  /// IsMathErrnoDefault - Does this tool chain use -fmath-errno by default.
  virtual bool IsMathErrnoDefault() const { return true; }

  /// IsObjCDefaultSynthPropertiesDefault - Does this tool chain enable
  /// -fobjc-default-synthesize-properties by default.
  virtual bool IsObjCDefaultSynthPropertiesDefault() const { return false; }

  /// IsObjCNonFragileABIDefault - Does this tool chain set
  /// -fobjc-nonfragile-abi by default.
  virtual bool IsObjCNonFragileABIDefault() const { return false; }

  /// UseObjCMixedDispatchDefault - When using non-legacy dispatch, should the
  /// mixed dispatch method be used?
  virtual bool UseObjCMixedDispatch() const { return false; }

  /// GetDefaultStackProtectorLevel - Get the default stack protector level for
  /// this tool chain (0=off, 1=on, 2=all).
  virtual unsigned GetDefaultStackProtectorLevel(bool KernelOrKext) const {
    return 0;
  }

  /// GetDefaultRuntimeLibType - Get the default runtime library variant to use.
  virtual RuntimeLibType GetDefaultRuntimeLibType() const {
    return ToolChain::RLT_Libgcc;
  }

  /// IsUnwindTablesDefault - Does this tool chain use -funwind-tables
  /// by default.
  virtual bool IsUnwindTablesDefault() const;

  /// GetDefaultRelocationModel - Return the LLVM name of the default
  /// relocation model for this tool chain.
  virtual const char *GetDefaultRelocationModel() const = 0;

  /// GetForcedPicModel - Return the LLVM name of the forced PIC model
  /// for this tool chain, or 0 if this tool chain does not force a
  /// particular PIC mode.
  virtual const char *GetForcedPicModel() const = 0;

  /// SupportsProfiling - Does this tool chain support -pg.
  virtual bool SupportsProfiling() const { return true; }

  /// Does this tool chain support Objective-C garbage collection.
  virtual bool SupportsObjCGC() const { return true; }

  /// Complain if this tool chain doesn't support Objective-C ARC.
  virtual void CheckObjCARC() const {}

  /// UseDwarfDebugFlags - Embed the compile options to clang into the Dwarf
  /// compile unit information.
  virtual bool UseDwarfDebugFlags() const { return false; }

  /// UseSjLjExceptions - Does this tool chain use SjLj exceptions.
  virtual bool UseSjLjExceptions() const { return false; }

  /// ComputeLLVMTriple - Return the LLVM target triple to use, after taking
  /// command line arguments into account.
  virtual std::string ComputeLLVMTriple(const ArgList &Args,
                                 types::ID InputType = types::TY_INVALID) const;

  /// ComputeEffectiveClangTriple - Return the Clang triple to use for this
  /// target, which may take into account the command line arguments. For
  /// example, on Darwin the -mmacosx-version-min= command line argument (which
  /// sets the deployment target) determines the version in the triple passed to
  /// Clang.
  virtual std::string ComputeEffectiveClangTriple(const ArgList &Args,
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
  virtual void AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const;

  // addClangTargetOptions - Add options that need to be passed to cc1 for
  // this target.
  virtual void addClangTargetOptions(ArgStringList &CC1Args) const;

  // GetRuntimeLibType - Determine the runtime library type to use with the
  // given compilation arguments.
  virtual RuntimeLibType GetRuntimeLibType(const ArgList &Args) const;

  // GetCXXStdlibType - Determine the C++ standard library type to use with the
  // given compilation arguments.
  virtual CXXStdlibType GetCXXStdlibType(const ArgList &Args) const;

  /// AddClangCXXStdlibIncludeArgs - Add the clang -cc1 level arguments to set
  /// the include paths to use for the given C++ standard library type.
  virtual void AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                            ArgStringList &CC1Args) const;

  /// AddCXXStdlibLibArgs - Add the system specific linker arguments to use
  /// for the given C++ standard library type.
  virtual void AddCXXStdlibLibArgs(const ArgList &Args,
                                   ArgStringList &CmdArgs) const;

  /// AddCCKextLibArgs - Add the system specific linker arguments to use
  /// for kernel extensions (Darwin-specific).
  virtual void AddCCKextLibArgs(const ArgList &Args,
                                ArgStringList &CmdArgs) const;

  /// AddFastMathRuntimeIfAvailable - If a runtime library exists that sets
  /// global flags for unsafe floating point math, add it and return true.
  ///
  /// This checks for presence of the -ffast-math or -funsafe-math flags.
  virtual bool AddFastMathRuntimeIfAvailable(const ArgList &Args,
                                             ArgStringList &CmdArgs) const;
};

} // end namespace driver
} // end namespace clang

#endif
