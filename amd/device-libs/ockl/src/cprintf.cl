/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"

#define AL(P, O) __opencl_atomic_load(P, O, memory_scope_device)
#define ACE(P, E, V, O) __opencl_atomic_compare_exchange_strong(P, E, V, O, O, memory_scope_device)

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
