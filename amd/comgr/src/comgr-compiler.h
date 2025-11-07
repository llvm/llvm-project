//===- comgr-compiler.h - Comgr compiler Action internals -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_COMPILER_H
#define COMGR_COMPILER_H

#include "comgr.h"
#include "clang/Driver/Driver.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace COMGR {

/// Manages executing Compiler-related actions.
///
/// @warning No more than one public method should be called on a constructed
/// object before it is destructed.
class AMDGPUCompiler {
  DataAction *ActionInfo;
  DataSet *InSet;
  amd_comgr_data_set_t OutSetT;
  /// Precompiled header file paths.
  llvm::SmallVector<llvm::SmallString<128>, 2> PrecompiledHeaders;
  /// Arguments common to all driver invocations in the current action.
  llvm::SmallVector<const char *, 128> Args;
  llvm::SmallString<128> TmpDir;
  llvm::SmallString<128> InputDir;
  llvm::SmallString<128> OutputDir;
  llvm::SmallString<128> IncludeDir;
  llvm::raw_ostream &LogS;
  /// Storage for other dynamic strings we need to include in Argv.
  llvm::BumpPtrAllocator Allocator;
  llvm::StringSaver Saver = Allocator;
  /// Whether we need to disable Clang's device-lib linking.
  bool NoGpuLib = true;
  bool UseVFS = false;

  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFS;

  amd_comgr_status_t createTmpDirs();
  amd_comgr_status_t removeTmpDirs();
  amd_comgr_status_t processFile(DataObject *Input, const char *InputFilePath,
                                 const char *OutputFilePath);
  /// Process each file in @c InSet individually, placing output in @c OutSet.
  amd_comgr_status_t processFiles(amd_comgr_data_kind_t OutputKind,
                                  const char *OutputSuffix);
  amd_comgr_status_t processFiles(amd_comgr_data_kind_t OutputKind,
                                  const char *OutputSuffix, DataSet *InSet);
  amd_comgr_status_t addIncludeFlags();
  amd_comgr_status_t addTargetIdentifierFlags(llvm::StringRef IdentStr,
                                              bool CompilingSrc);
  amd_comgr_status_t addCompilationFlags();
  amd_comgr_status_t addDeviceLibraries();
  amd_comgr_status_t extractSpirvFlags(DataSet *BcSet);

  amd_comgr_status_t executeInProcessDriver(llvm::ArrayRef<const char *> Args);

  amd_comgr_status_t translateSpirvToBitcodeImpl(DataSet *SpirvInSet,
                                                 DataSet *BcOutSet);

public:
  AMDGPUCompiler(DataAction *ActionInfo, DataSet *InSet, DataSet *OutSet,
                 llvm::raw_ostream &LogS);
  ~AMDGPUCompiler();

  amd_comgr_status_t preprocessToSource();
  amd_comgr_status_t compileToBitcode(bool WithDeviceLibs = false);
  amd_comgr_status_t compileToRelocatable();
  amd_comgr_status_t unbundle();
  amd_comgr_status_t linkBitcodeToBitcode();
  amd_comgr_status_t codeGenBitcodeToRelocatable();
  amd_comgr_status_t codeGenBitcodeToAssembly();
  amd_comgr_status_t assembleToRelocatable();
  amd_comgr_status_t linkToRelocatable();
  amd_comgr_status_t linkToExecutable();
  amd_comgr_status_t compileToExecutable();
  amd_comgr_status_t compileSpirvToRelocatable();
  amd_comgr_status_t translateSpirvToBitcode();
  amd_comgr_status_t compileSourceToSpirv();

  amd_comgr_language_t getLanguage() const { return ActionInfo->Language; }
};
} // namespace COMGR

#endif
