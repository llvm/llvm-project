// REQUIRES: native-run
// UNSUPPORTED: arm, aarch64
// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- clear_cache_test.c - Test clear_cache -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <stdio.h>
#include <string.h>
#include <stdint.h>
#if defined(_WIN32)
#include <windows.h>
static uintptr_t get_page_size() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
}
#else
#include <unistd.h>
#include <sys/mman.h>

static uintptr_t get_page_size() {
    return sysconf(_SC_PAGE_SIZE);
}
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

unsigned char execution_buffer[128];

int main()
{
    // make executable the page containing execution_buffer 
    uintptr_t page_size = get_page_size();
    char* start = (char*)((uintptr_t)execution_buffer & (-page_size));
    char* end = (char*)((uintptr_t)(&execution_buffer[128+page_size]) & (-page_size));
#if defined(_WIN32)
    DWORD dummy_oldProt;
    MEMORY_BASIC_INFORMATION b;
    if (!VirtualQuery(start, &b, sizeof(b)))
        return 1;
    if (!VirtualProtect(b.BaseAddress, b.RegionSize, PAGE_EXECUTE_READWRITE, &b.Protect))
#else
    if (mprotect(start, end-start, PROT_READ|PROT_WRITE|PROT_EXEC) != 0)
#endif
        return 1;

    // verify you can copy and execute a function
    pfunc f1 = (pfunc)memcpy_f(execution_buffer, func1, 128);
    __clear_cache(execution_buffer, &execution_buffer[128]);
    if ((*f1)() != 1)
        return 1;

    // verify you can overwrite a function with another
    pfunc f2 = (pfunc)memcpy_f(execution_buffer, func2, 128);
    __clear_cache(execution_buffer, &execution_buffer[128]);
    if ((*f2)() != 2)
        return 1;

    return 0;
}
