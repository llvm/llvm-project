
#include "irif.h"

#ifndef NULL
#define NULL 0
#endif

#define OFFSET 8

// Atomically reserves space to the printf data buffer and returns a pointer to it
__global char *
__printf_alloc(uint bytes)
{
    __global char *ptr = (__global char *)((__constant size_t *)__llvm_amdgcn_implicitarg_ptr())[3];

    uint size = ((__global uint *)ptr)[1];
    uint offset = __llvm_ld_atomic_a1_x_dev_i32((__global uint *)ptr);

    for (;;) {
        if (OFFSET + offset + bytes > size)
            return NULL;

        uint tmp = __llvm_cmpxchg_a1_x_x_dev_i32((__global uint *)ptr, offset, offset + bytes);
        if (tmp == offset)
            break;

        offset = tmp;
    }

    return ptr + OFFSET + offset;
}

