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

namespace lld {
BumpPtrAllocator elf::BAlloc;
StringSaver elf::Saver{elf::BAlloc};

void elf::freeArena() { elf::BAlloc.Reset(); }
}
