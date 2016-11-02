//===- Memory.cpp -----------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Memory.h"

using namespace llvm;
using namespace lld;
using namespace lld::elf;

namespace lld {
BumpPtrAllocator elf::BAlloc;
StringSaver elf::Saver{elf::BAlloc};

SpecificAllocBase::SpecificAllocBase() { Instances.push_back(this); }

std::vector<SpecificAllocBase *> SpecificAllocBase::Instances;

void elf::freeArena() {
  for (SpecificAllocBase *Alloc : SpecificAllocBase::Instances)
    Alloc->reset();
  BAlloc.Reset();
}
}
