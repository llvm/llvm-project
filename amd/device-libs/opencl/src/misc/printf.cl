
#include "irif.h"

#ifndef NULL
#define NULL 0
#endif

#define OFFSET 8

// Atomically reserves space to the printf data buffer and returns a pointer to it
__global char *
__printf_alloc(uint bytes)
{
    __global char *ptr = (__global char *)(((__constant size_t *)__llvm_amdgcn_implicitarg_ptr())[3]);

    uint size = ((__global uint *)ptr)[1];
    uint offset = atomic_load_explicit((__global atomic_uint *)ptr, memory_order_relaxed, memory_scope_device);

    for (;;) {
        if (OFFSET + offset + bytes > size)
            return NULL;

        if (atomic_compare_exchange_strong_explicit((__global atomic_uint *)ptr, &offset, offset+bytes, memory_order_relaxed, memory_order_relaxed, memory_scope_device))
            break;
    }

    return ptr + OFFSET + offset;
}

