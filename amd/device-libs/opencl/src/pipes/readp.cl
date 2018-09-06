/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "pipes.h"

#define ATTR __attribute__((always_inline))

#define READ_PIPE_SIZE(SIZE, STYPE) \
ATTR int \
__read_pipe_2_##SIZE(__global struct pipeimp* p, STYPE* ptr) \
{ \
    size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device); \
    size_t ri = wave_reserve_1(&p->read_idx, wi); \
    if (ri == ~(size_t)0) \
        return -1; \
 \
    size_t pi = wrap(ri, p->end_idx); \
    *ptr = ((__global STYPE *)p->packets)[pi]; \
 \
    if (ri == wi-1) { \
        __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device); \
        __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device); \
    }\
\
    return 0; \
}

DO_PIPE_SIZE(READ_PIPE_SIZE)

ATTR int
__read_pipe_2(__global struct pipeimp* p, void* ptr, uint size, uint align)
{
    size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device);
    size_t ri = wave_reserve_1(&p->read_idx, wi);
    if (ri == ~(size_t)0)
        return -1;

    size_t pi = wrap(ri, p->end_idx);
    __memcpy_internal_aligned(ptr, p->packets + pi*size, size, align);

    if (ri == wi-1) {
        __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device);
        __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device);
    }

    return 0;
}

#define READ_PIPE_RESERVED_SIZE(SIZE, STYPE) \
ATTR int \
__read_pipe_4_##SIZE(__global struct pipeimp* p, reserve_id_t rid, uint i, STYPE* ptr)  \
{ \
    size_t rin = __builtin_astype(rid, size_t) + i; \
    size_t pi = wrap(rin, p->end_idx); \
    *ptr = ((__global STYPE *)p->packets)[pi]; \
 \
    return 0; \
}

DO_PIPE_SIZE(READ_PIPE_RESERVED_SIZE)

ATTR int
__read_pipe_4(__global struct pipeimp* p, reserve_id_t rid, uint i, void *ptr, uint size, uint align)
{
    size_t rin = __builtin_astype(rid, size_t) + i; \
    size_t pi = wrap(rin, p->end_idx);
    __memcpy_internal_aligned(ptr, p->packets + pi*size, size, align);

    return 0;
}

