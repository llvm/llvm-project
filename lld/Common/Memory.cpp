//===- Memory.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Memory.h"

using namespace llvm;
using namespace lld;

BumpPtrAllocator lld::BAlloc;
StringSaver lld::Saver{BAlloc};
std::vector<SpecificAllocBase *> lld::SpecificAllocBase::Instances;

void lld::freeArena() {
  for (SpecificAllocBase *Alloc : SpecificAllocBase::Instances)
    Alloc->reset();
  BAlloc.Reset();
}
