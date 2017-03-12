//===-- enable_execute_stack_test.c - Test __enable_execute_stack ----------===//
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
void __clear_cache(void* start, void* end)
{
    if (!FlushInstructionCache(GetCurrentProcess(), start, end-start))
        exit(1);
}
void __enable_execute_stack(void *addr)
{
    MEMORY_BASIC_INFORMATION b;

    if (!VirtualQuery(addr, &b, sizeof(b)))
        exit(1);
    if (!VirtualProtect(b.BaseAddress, b.RegionSize, PAGE_EXECUTE_READWRITE, &b.Protect))
        exit(1);
}
#else
#include <sys/mman.h>
extern void __clear_cache(void* start, void* end);
extern void __enable_execute_stack(void* addr);
#endif

typedef int (*pfunc)(void);

int func1() 
{
    return 1;
}

int func2() 
{
    return 2;
}

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
    unsigned char execution_buffer[128];
    // mark stack page containing execution_buffer to be executable
    __enable_execute_stack(execution_buffer);
	
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
