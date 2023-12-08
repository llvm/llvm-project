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
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include <cstdint>
#include <string>

namespace llvm {

namespace bolt {
class BinaryContext;

/// Class responsible for allocating and managing code and data sections.
class ExecutableFileMemoryManager : public jitlink::JITLinkMemoryManager {
private:
  void updateSection(const jitlink::Section &Section, uint8_t *Contents,
                     size_t Size, size_t Alignment);

  BinaryContext &BC;

  // All new sections will be identified by the following prefix.
  std::string NewSecPrefix;

  // Name prefix used for sections from the input.
  std::string OrgSecPrefix;

public:
  // Our linker's main purpose is to handle a single object file, created
  // by RewriteInstance after reading the input binary and reordering it.
  // After objects finish loading, we increment this. Therefore, whenever
  // this is greater than zero, we are dealing with additional objects that
  // will not be managed by BinaryContext but only exist to support linking
  // user-supplied objects into the main input executable.
  uint32_t ObjectsLoaded{0};

  ExecutableFileMemoryManager(BinaryContext &BC) : BC(BC) {}

  void allocate(const jitlink::JITLinkDylib *JD, jitlink::LinkGraph &G,
                OnAllocatedFunction OnAllocated) override;

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override;
  using JITLinkMemoryManager::deallocate;

  /// Section name management.
  void setNewSecPrefix(StringRef Prefix) { NewSecPrefix = Prefix; }
  void setOrgSecPrefix(StringRef Prefix) { OrgSecPrefix = Prefix; }
};

} // namespace bolt
} // namespace llvm

#endif
