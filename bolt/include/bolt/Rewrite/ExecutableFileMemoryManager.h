//===- bolt/Rewrite/ExecutableFileMemoryManager.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_EXECUTABLE_FILE_MEMORY_MANAGER_H
#define BOLT_REWRITE_EXECUTABLE_FILE_MEMORY_MANAGER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include <cstdint>
#include <string>

namespace llvm {

namespace bolt {
class BinaryContext;

/// Class responsible for allocating and managing code and data sections.
class ExecutableFileMemoryManager : public SectionMemoryManager {
private:
  uint8_t *allocateSection(intptr_t Size, unsigned Alignment,
                           unsigned SectionID, StringRef SectionName,
                           bool IsCode, bool IsReadOnly);
  BinaryContext &BC;
  bool AllowStubs;

public:
  // Our linker's main purpose is to handle a single object file, created
  // by RewriteInstance after reading the input binary and reordering it.
  // After objects finish loading, we increment this. Therefore, whenever
  // this is greater than zero, we are dealing with additional objects that
  // will not be managed by BinaryContext but only exist to support linking
  // user-supplied objects into the main input executable.
  uint32_t ObjectsLoaded{0};

  ExecutableFileMemoryManager(BinaryContext &BC, bool AllowStubs)
      : BC(BC), AllowStubs(AllowStubs) {}

  ~ExecutableFileMemoryManager();

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    return allocateSection(Size, Alignment, SectionID, SectionName,
                           /*IsCode=*/true, true);
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override {
    return allocateSection(Size, Alignment, SectionID, SectionName,
                           /*IsCode=*/false, IsReadOnly);
  }

  // Ignore TLS sections by treating them as a regular data section
  TLSSection allocateTLSSection(uintptr_t Size, unsigned Alignment,
                                unsigned SectionID,
                                StringRef SectionName) override {
    TLSSection Res;
    Res.Offset = 0;
    Res.InitializationImage = allocateDataSection(
        Size, Alignment, SectionID, SectionName, /*IsReadOnly=*/false);
    return Res;
  }

  bool allowStubAllocation() const override { return AllowStubs; }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override;
};

} // namespace bolt
} // namespace llvm

#endif
