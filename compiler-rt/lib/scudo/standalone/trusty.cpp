//===-- trusty.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

#if SCUDO_TRUSTY

#include "common.h"
#include "mutex.h"
#include "string_utils.h"
#include "trusty.h"

#include <errno.h>           // for errno
#include <stdio.h>           // for printf()
#include <stdlib.h>          // for getenv()
#include <sys/auxv.h>        // for getauxval()
#include <time.h>            // for clock_gettime()
#include <trusty_syscalls.h> // for _trusty_brk()

#define SBRK_ALIGN 32

namespace scudo {

uptr getPageSize() { return getauxval(AT_PAGESZ); }

void NORETURN die() { abort(); }

void *map(UNUSED void *Addr, uptr Size, UNUSED const char *Name, uptr Flags,
          UNUSED MapPlatformData *Data) {
  // Calling _trusty_brk(0) returns the current program break.
  uptr ProgramBreak = reinterpret_cast<uptr>(_trusty_brk(0));
  uptr Start;
  uptr End;

  Start = roundUp(ProgramBreak, SBRK_ALIGN);
  // Don't actually extend the heap if MAP_NOACCESS flag is set since this is
  // the case where Scudo tries to reserve a memory region without mapping
  // physical pages.
  if (Flags & MAP_NOACCESS)
    return reinterpret_cast<void *>(Start);

  // Attempt to extend the heap by Size bytes using _trusty_brk.
  End = roundUp(Start + Size, SBRK_ALIGN);
  ProgramBreak =
      reinterpret_cast<uptr>(_trusty_brk(reinterpret_cast<void *>(End)));
  if (ProgramBreak < End) {
    errno = ENOMEM;
    dieOnMapUnmapError(Size);
    return nullptr;
  }
  return reinterpret_cast<void *>(Start); // Base of new reserved region.
}

// Unmap is a no-op since Trusty uses sbrk instead of memory mapping.
void unmap(UNUSED void *Addr, UNUSED uptr Size, UNUSED uptr Flags,
           UNUSED MapPlatformData *Data) {}

void setMemoryPermission(UNUSED uptr Addr, UNUSED uptr Size, UNUSED uptr Flags,
                         UNUSED MapPlatformData *Data) {}

void releasePagesToOS(UNUSED uptr BaseAddress, UNUSED uptr Offset,
                      UNUSED uptr Size, UNUSED MapPlatformData *Data) {}

const char *getEnv(const char *Name) { return getenv(Name); }

// All mutex operations are a no-op since Trusty doesn't currently support
// threads.
bool HybridMutex::tryLock() { return true; }

void HybridMutex::lockSlow() {}

void HybridMutex::unlock() {}

void HybridMutex::assertHeldImpl() {}

u64 getMonotonicTime() {
  timespec TS;
  clock_gettime(CLOCK_MONOTONIC, &TS);
  return static_cast<u64>(TS.tv_sec) * (1000ULL * 1000 * 1000) +
         static_cast<u64>(TS.tv_nsec);
}

u32 getNumberOfCPUs() { return 0; }

u32 getThreadID() { return 0; }

bool getRandom(UNUSED void *Buffer, UNUSED uptr Length, UNUSED bool Blocking) {
  return false;
}

void outputRaw(const char *Buffer) { printf("%s", Buffer); }

void setAbortMessage(UNUSED const char *Message) {}

} // namespace scudo

#endif // SCUDO_TRUSTY
