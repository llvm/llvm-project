/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

#define AL(P, O) __opencl_atomic_load(P, O, memory_scope_device)
#define ACE(P, E, V, O) __opencl_atomic_compare_exchange_strong(P, E, V, O, O, memory_scope_device)

#ifndef NULL
#define NULL 0
#endif

#define OFFSET 8

// Atomically reserves space to the printf data buffer and returns a pointer to it
__global char *
__printf_alloc(uint bytes)
{
    __global char *ptr;
    if (__oclc_ABI_version < 500) {
        ptr = (__global char *)((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[3];
    } else {
        ptr = (__global char *)((__constant size_t *)__builtin_amdgcn_implicitarg_ptr())[9];
    }

    uint size = ((__global uint *)ptr)[1];
    uint offset = AL((__global atomic_uint *)ptr, memory_order_relaxed);

    for (;;) {
        if (OFFSET + offset + bytes > size)
            return NULL;

        if (ACE((__global atomic_uint *)ptr, &offset, offset+bytes, memory_order_relaxed))
            break;
    }

    return ptr + OFFSET + offset;
}

// Return a controlDWord for nonhostcall printf scheme, format as follows
// Bit 0 (LSB) -> stream (1 if stderr, 0 if stdout)
// Bit 1 -> constant format string (1 if constant)
// Bit 2 - 31 -> size of printf data frame (controlDWord + format string/hash + args)

uint __printf_control_dword(uint len, bool is_const_fmt_str, bool is_stderr) {
  return (len << 2) | (is_const_fmt_str ? (uint)2 : 0) | (uint)is_stderr ;
}

// printf stub to resolve link time dependencies.
// Will be replaced by the compiler.
__attribute__((noinline))
__attribute__((optnone))
__attribute__((format(printf, 1, 2)))
int printf(__constant const char* st, ...) {
  __printf_alloc(0);
  return -1;
}
