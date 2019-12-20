// REQUIRES: native-run
// UNSUPPORTED: arm, aarch64
// RUN: %clang_builtins %s %librt -o %t && %run_nomprotect %t
// REQUIRES: librt_has_clear_cache
//===-- clear_cache_test.c - Test clear_cache -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include <stdio.h>
#include <string.h>
#include <stdint.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

extern void __clear_cache(void* start, void* end);

typedef int (*pfunc)(void);

// Make these static to avoid ILT jumps for incremental linking on Windows.
static int func1() { return 1; }
static int func2() { return 2; }

void *__attribute__((noinline))
memcpy_f(void *dst, const void *src, size_t n) {
// ARM and MIPS nartually align functions, but use the LSB for ISA selection
// (THUMB, MIPS16/uMIPS respectively).  Ensure that the ISA bit is ignored in
// the memcpy
#if defined(__arm__) || defined(__mips__)
  return (void *)((uintptr_t)memcpy(dst, (void *)((uintptr_t)src & ~1), n) |
                  ((uintptr_t)src & 1));
#else
  return memcpy(dst, (void *)((uintptr_t)src), n);
#endif
}

int main()
{
    const int kSize = 128;
#if !defined(_WIN32)
    uint8_t *execution_buffer = mmap(0, kSize,
                                     PROT_READ | PROT_WRITE | PROT_EXEC,
                                     MAP_ANON | MAP_PRIVATE, -1, 0);
    if (execution_buffer == MAP_FAILED)
      return 1;
#else
    HANDLE mapping = CreateFileMapping(INVALID_HANDLE_VALUE, NULL,
                                       PAGE_EXECUTE_READWRITE, 0, kSize, NULL);
    if (mapping == NULL)
        return 1;

    uint8_t* execution_buffer = MapViewOfFile(
        mapping, FILE_MAP_ALL_ACCESS | FILE_MAP_EXECUTE, 0, 0, 0);
    if (execution_buffer == NULL)
        return 1;
#endif

    // verify you can copy and execute a function
    pfunc f1 = (pfunc)memcpy_f(execution_buffer, func1, kSize);
    __clear_cache(execution_buffer, execution_buffer + kSize);
    if ((*f1)() != 1)
        return 1;

    // verify you can overwrite a function with another
    pfunc f2 = (pfunc)memcpy_f(execution_buffer, func2, kSize);
    __clear_cache(execution_buffer, execution_buffer + kSize);
    if ((*f2)() != 2)
        return 1;

    return 0;
}
