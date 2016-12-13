//===- Memory.cpp -----------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Support/Memory.h"

using namespace llvm;

namespace lld {
BumpPtrAllocator BAlloc;
StringSaver Saver{BAlloc};

SpecificAllocBase::SpecificAllocBase() { Instances.push_back(this); }

std::vector<SpecificAllocBase *> SpecificAllocBase::Instances;

void freeArena() {
  for (SpecificAllocBase *Alloc : SpecificAllocBase::Instances)
    Alloc->reset();
  BAlloc.Reset();
}
}
