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
InMemoryModuleCache::addPCM(llvm::StringRef Filename,
                            std::unique_ptr<llvm::MemoryBuffer> Buffer) {
  auto Insertion = PCMs.insert(std::make_pair(Filename, std::move(Buffer)));
  assert(Insertion.second && "Already has a PCM");
  return *Insertion.first->second.Buffer;
}

llvm::MemoryBuffer &
InMemoryModuleCache::addFinalPCM(llvm::StringRef Filename,
                                 std::unique_ptr<llvm::MemoryBuffer> Buffer) {
  auto &PCM = PCMs[Filename];
  assert(!PCM.IsFinal && "Trying to override finalized PCM?");
  assert(!PCM.Buffer && "Already has a non-final PCM");
  PCM.Buffer = std::move(Buffer);
  PCM.IsFinal = true;
  return *PCM.Buffer;
}

llvm::MemoryBuffer *
InMemoryModuleCache::lookupPCM(llvm::StringRef Filename) const {
  auto I = PCMs.find(Filename);
  if (I == PCMs.end())
    return nullptr;
  return I->second.Buffer.get();
}

bool InMemoryModuleCache::isPCMFinal(llvm::StringRef Filename) const {
  auto I = PCMs.find(Filename);
  if (I == PCMs.end())
    return false;
  return I->second.IsFinal;
}

bool InMemoryModuleCache::tryToRemovePCM(llvm::StringRef Filename) {
  auto I = PCMs.find(Filename);
  assert(I != PCMs.end() && "PCM to remove is unknown...");

  auto &PCM = I->second;
  if (PCM.IsFinal)
    return true;

  PCMs.erase(I);
  return false;
}

void InMemoryModuleCache::finalizePCM(llvm::StringRef Filename) {
  auto I = PCMs.find(Filename);
  assert(I != PCMs.end() && "PCM to finalize is unknown...");

  auto &PCM = I->second;
  assert(PCM.Buffer && "Trying to finalize a dropped PCM...");
  PCM.IsFinal = true;
}
