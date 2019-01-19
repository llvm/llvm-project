//===- FuzzerShmemPosix.cpp - Posix shared memory ---------------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// SharedMemoryRegion.  For Fuchsia, this is just stubs as equivalence servers
// are not currently supported.
//===----------------------------------------------------------------------===//
#include "FuzzerDefs.h"

#if LIBFUZZER_FUCHSIA

#include "FuzzerShmem.h"

namespace fuzzer {

bool SharedMemoryRegion::Create(const char *Name) {
  return false;
}

bool SharedMemoryRegion::Open(const char *Name) {
  return false;
}

bool SharedMemoryRegion::Destroy(const char *Name) {
  return false;
}

void SharedMemoryRegion::Post(int Idx) {}

void SharedMemoryRegion::Wait(int Idx) {}

}  // namespace fuzzer

#endif  // LIBFUZZER_FUCHSIA
