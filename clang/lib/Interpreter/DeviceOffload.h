//===----------- DeviceOffload.h - Device Offloading ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements classes required for offloading to HIP and CUDA devices.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H
#define LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H

#include "IncrementalParser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <memory>

namespace llvm {
class TargetMachine;
} // namespace llvm

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
  CompilerInstance &DeviceCI;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS;
  CodeGenOptions &CodeGenOpts;
  const TargetOptions &TargetOpts;
};

class IncrementalHIPDeviceParser : public IncrementalDeviceParser {

public:
  IncrementalHIPDeviceParser(
      CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
      IncrementalAction *DeviceAct,
      llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS,
      llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs);

  llvm::Expected<TranslationUnitDecl *> Parse(llvm::StringRef Input) override;

  llvm::Error GenerateOffloadBinary() override;

  ~IncrementalHIPDeviceParser();

protected:
  // Generate the HSACO code object for the last PTU.
  llvm::Expected<llvm::StringRef> GenerateHSACO();

  // Bundle the HSACO into a HIP offload bundle in memory.
  llvm::Error GenerateOffloadBundle();

  llvm::SmallVector<char, 1024> HSACOContent;
};

class IncrementalCUDADeviceParser : public IncrementalDeviceParser {

public:
  IncrementalCUDADeviceParser(
      CompilerInstance &DeviceInstance, CompilerInstance &HostInstance,
      IncrementalAction *DeviceAct,
      llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> VFS,
      llvm::Error &Err, std::list<PartialTranslationUnit> &PTUs);

  llvm::Error GenerateOffloadBinary() override;

  ~IncrementalCUDADeviceParser();

protected:
  // Generate PTX for the last PTU.
  llvm::Expected<llvm::StringRef> GeneratePTX();

  // Generate fatbinary contents in memory
  llvm::Error GenerateFatbinary();

  int SMVersion;
  llvm::SmallString<1024> PTXCode;
  llvm::SmallVector<char, 1024> FatbinContent;
  std::unique_ptr<llvm::TargetMachine> TM;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_INTERPRETER_DEVICE_OFFLOAD_H
