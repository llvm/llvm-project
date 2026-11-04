//===----------- DeviceOffload.h - Device Offloading ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes required for offloading to devices.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H
#define LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H

#include "IncrementalParser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace clang {
struct PartialTranslationUnit;
class CompilerInstance;
class CodeGenOptions;
class TargetOptions;
class IncrementalAction;

class IncrementalDeviceParser : public IncrementalParser {

public:
  IncrementalDeviceParser(
      CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
      IncrementalAction *DeviceAct,
      llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS,
      llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs);

  virtual llvm::Error GenerateOffloadBinary() = 0;

  ~IncrementalDeviceParser() override;

protected:
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS;
  CodeGenOptions &CodeGenOpts;
  const TargetOptions &TargetOpts;
};

class IncrementalHIPDeviceParser : public IncrementalDeviceParser {
  CompilerInstance &DeviceCI;
  llvm::SmallVector<char, 0> ObjectContent;

  // Generate the AMDGPU object file for the last PTU.
  llvm::Expected<llvm::StringRef> GenerateDeviceObject();

public:
  IncrementalHIPDeviceParser(
      CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
      IncrementalAction *DeviceAct,
      llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS,
      llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs);

  llvm::Error GenerateOffloadBinary() override;

  ~IncrementalHIPDeviceParser() override;
};

class IncrementalCUDADeviceParser : public IncrementalDeviceParser {
  // Generate PTX for the last PTU.
  llvm::Expected<llvm::StringRef> GeneratePTX();

  // Generate fatbinary contents in memory
  llvm::Error GenerateFatbinary();

  int SMVersion;
  llvm::SmallString<1024> PTXCode;
  llvm::SmallVector<char, 1024> FatbinContent;

public:
  IncrementalCUDADeviceParser(
      CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
      IncrementalAction *DeviceAct,
      llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS,
      llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs);

  llvm::Error GenerateOffloadBinary() override;

  ~IncrementalCUDADeviceParser() override;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H
