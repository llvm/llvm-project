//===- bolt/runtime/hugify.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined (__x86_64__)
#if !defined(__APPLE__)

#include "common.h"
#include <sys/mman.h>

// Enables a very verbose logging to stderr useful when debugging
//#define ENABLE_DEBUG

// Function pointers to init routines in the binary, so we can resume
// regular execution of the function that we hooked.
extern void (*__bolt_hugify_init_ptr)();

// The __hot_start and __hot_end symbols set by Bolt. We use them to figure
// out the rage for marking huge pages.
extern uint64_t __hot_start;
extern uint64_t __hot_end;

#ifdef MADV_HUGEPAGE
/// Check whether the kernel supports THP via corresponding sysfs entry.
static bool has_pagecache_thp_support() {
  char buf[256] = {0};
  const char *madviseStr = "always [madvise] never";

  int fd = __open("/sys/kernel/mm/transparent_hugepage/enabled",
                  0 /* O_RDONLY */, 0);
  if (fd < 0)
    return false;

  size_t res = __read(fd, buf, 256);
  if (res < 0)
    return false;

  int cmp = strnCmp(buf, madviseStr, strLen(madviseStr));
  return cmp == 0;
}

static void hugify_for_old_kernel(uint8_t *from, uint8_t *to) {
  size_t size = to - from;

  uint8_t *mem = reinterpret_cast<uint8_t *>(
      __mmap(0, size, 0x3 /* PROT_READ | PROT_WRITE*/,
             0x22 /* MAP_PRIVATE | MAP_ANONYMOUS*/, -1, 0));

  if (mem == (void *)MAP_FAILED) {
    char msg[] = "Could not allocate memory for text move\n";
    reportError(msg, sizeof(msg));
  }
#ifdef ENABLE_DEBUG
  reportNumber("Allocated temporary space: ", (uint64_t)mem, 16);
#endif

  // Copy the hot code to a temproary location.
  memcpy(mem, from, size);

  // Maps out the existing hot code.
  if (__mmap(reinterpret_cast<uint64_t>(from), size,
             PROT_READ | PROT_WRITE | PROT_EXEC,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1,
             0) == (void *)MAP_FAILED) {
    char msg[] = "failed to mmap memory for large page move terminating\n";
    reportError(msg, sizeof(msg));
  }

  // Mark the hot code page to be huge page.
  if (__madvise(from, size, MADV_HUGEPAGE) == -1) {
    char msg[] = "failed to allocate large page\n";
    reportError(msg, sizeof(msg));
  }

  // Copy the hot code back.
  memcpy(from, mem, size);

  // Change permission back to read-only, ignore failure
  __mprotect(from, size, PROT_READ | PROT_EXEC);

  __munmap(mem, size);
}
#endif

extern "C" void __bolt_hugify_self_impl() {
#ifdef MADV_HUGEPAGE
  uint8_t *hotStart = (uint8_t *)&__hot_start;
  uint8_t *hotEnd = (uint8_t *)&__hot_end;
  // Make sure the start and end are aligned with huge page address
  const size_t hugePageBytes = 2L * 1024 * 1024;
  uint8_t *from = hotStart - ((intptr_t)hotStart & (hugePageBytes - 1));
  uint8_t *to = hotEnd + (hugePageBytes - 1);
  to -= (intptr_t)to & (hugePageBytes - 1);

#ifdef ENABLE_DEBUG
  reportNumber("[hugify] hot start: ", (uint64_t)hotStart, 16);
  reportNumber("[hugify] hot end: ", (uint64_t)hotEnd, 16);
  reportNumber("[hugify] aligned huge page from: ", (uint64_t)from, 16);
  reportNumber("[hugify] aligned huge page to: ", (uint64_t)to, 16);
#endif

  if (!has_pagecache_thp_support()) {
    hugify_for_old_kernel(from, to);
    return;
  }

  if (__madvise(from, (to - from), MADV_HUGEPAGE) == -1) {
    char msg[] = "failed to allocate large page\n";
    // TODO: allow user to control the failure behavior.
    reportError(msg, sizeof(msg));
  }
#endif
}

/// This is hooking ELF's entry, it needs to save all machine state.
extern "C" __attribute((naked)) void __bolt_hugify_self() {
  __asm__ __volatile__(SAVE_ALL
                       "call __bolt_hugify_self_impl\n"
                       RESTORE_ALL
                       "jmp *__bolt_hugify_init_ptr(%%rip)\n"
                       :::);
}

#endif
#endif
