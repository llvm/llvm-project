//===- InMemoryModuleCache.cpp - Cache for loaded memory buffers ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/InMemoryModuleCache.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace clang;

llvm::MemoryBuffer &
InMemoryModuleCache::addBuffer(llvm::StringRef Filename,
                               std::unique_ptr<llvm::MemoryBuffer> Buffer) {
  auto Insertion = PCMs.insert({Filename, PCM{std::move(Buffer), NextIndex++}});
  assert(Insertion.second && "Already has a buffer");
  return *Insertion.first->second.Buffer;
}

llvm::MemoryBuffer *
InMemoryModuleCache::lookupBuffer(llvm::StringRef Filename) {
  auto I = PCMs.find(Filename);
  if (I == PCMs.end())
    return nullptr;
  return I->second.Buffer.get();
}

bool InMemoryModuleCache::isBufferFinal(llvm::StringRef Filename) {
  auto I = PCMs.find(Filename);
  if (I == PCMs.end())
    return false;
  return I->second.Index < FirstRemovableIndex;
}

bool InMemoryModuleCache::tryToRemoveBuffer(llvm::StringRef Filename) {
  auto I = PCMs.find(Filename);
  assert(I != PCMs.end() && "No buffer to remove...");
  if (I->second.Index < FirstRemovableIndex)
    return true;

  PCMs.erase(I);
  return false;
}

void InMemoryModuleCache::finalizeCurrentBuffers() {
  FirstRemovableIndex = NextIndex;
}
