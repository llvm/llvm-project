//===----------- DeviceOffload.h - Device Offloading ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements classes required for offloading to CUDA devices.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H
#define LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H

#include "IncrementalParser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace clang {
struct PartialTranslationUnit;
class CompilerInstance;
class CodeGenOptions;
class TargetOptions;
class IncrementalAction;

class IncrementalCUDADeviceParser : public IncrementalParser {

public:
  IncrementalCUDADeviceParser(
      CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
      IncrementalAction *DeviceAct,
      llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS,
      llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs);

  // Generate PTX for the last PTU.
  llvm::Expected<llvm::StringRef> GeneratePTX();

  // Generate fatbinary contents in memory
  llvm::Error GenerateFatbinary();

  ~IncrementalCUDADeviceParser();

protected:
  int SMVersion;
  llvm::SmallString<1024> PTXCode;
  llvm::SmallVector<char, 1024> FatbinContent;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS;
  CodeGenOptions &CodeGenOpts; // Intentionally a reference.
  const TargetOptions &TargetOpts;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H
