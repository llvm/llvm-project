//===-- guarded_pool_allocator_posix.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gwp_asan/guarded_pool_allocator.h"
#include "gwp_asan/utilities.h"

#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef ANDROID
#include <sys/prctl.h>
#define PR_SET_VMA 0x53564d41
#define PR_SET_VMA_ANON_NAME 0
#endif // ANDROID

void MaybeSetMappingName(void *Mapping, size_t Size, const char *Name) {
#ifdef ANDROID
  prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, Mapping, Size, Name);
#endif // ANDROID
  // Anonymous mapping names are only supported on Android.
  return;
}

namespace gwp_asan {
void *GuardedPoolAllocator::mapMemory(size_t Size, const char *Name) const {
  void *Ptr =
      mmap(nullptr, Size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  Check(Ptr != MAP_FAILED, "Failed to map guarded pool allocator memory");
  MaybeSetMappingName(Ptr, Size, Name);
  return Ptr;
}

void GuardedPoolAllocator::unmapMemory(void *Ptr, size_t Size,
                                       const char *Name) const {
  Check(munmap(Ptr, Size) == 0,
        "Failed to unmap guarded pool allocator memory.");
  MaybeSetMappingName(Ptr, Size, Name);
}

void GuardedPoolAllocator::markReadWrite(void *Ptr, size_t Size,
                                         const char *Name) const {
  Check(mprotect(Ptr, Size, PROT_READ | PROT_WRITE) == 0,
        "Failed to set guarded pool allocator memory at as RW.");
  MaybeSetMappingName(Ptr, Size, Name);
}

void GuardedPoolAllocator::markInaccessible(void *Ptr, size_t Size,
                                            const char *Name) const {
  // mmap() a PROT_NONE page over the address to release it to the system, if
  // we used mprotect() here the system would count pages in the quarantine
  // against the RSS.
  Check(mmap(Ptr, Size, PROT_NONE, MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE, -1,
             0) != MAP_FAILED,
        "Failed to set guarded pool allocator memory as inaccessible.");
  MaybeSetMappingName(Ptr, Size, Name);
}

size_t GuardedPoolAllocator::getPlatformPageSize() {
  return sysconf(_SC_PAGESIZE);
}

void GuardedPoolAllocator::installAtFork() {
  auto Disable = []() {
    if (auto *S = getSingleton())
      S->disable();
  };
  auto Enable = []() {
    if (auto *S = getSingleton())
      S->enable();
  };
  pthread_atfork(Disable, Enable, Enable);
}

} // namespace gwp_asan
