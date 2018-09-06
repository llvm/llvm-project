/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "pipes.h"

#define ATTR __attribute__((always_inline))

#define WRITE_PIPE_SIZE(SIZE, STYPE) \
ATTR int \
__write_pipe_2_##SIZE(__global struct pipeimp* p, const STYPE* ptr) \
{ \
    size_t ri = atomic_load_explicit(&p->read_idx, memory_order_relaxed, memory_scope_device); \
    size_t ei = p->end_idx; \
    size_t wi = wave_reserve_1(&p->write_idx, ri+ei); \
    if (wi == ~(size_t)0) \
        return -1; \
 \
    size_t pi = wrap(wi, ei); \
    ((__global STYPE *)p->packets)[pi] = *ptr; \
    return 0; \
}

DO_PIPE_SIZE(WRITE_PIPE_SIZE)

ATTR int
__write_pipe_2(__global struct pipeimp* p, const void* ptr, uint size, uint align)
{
    size_t ri = atomic_load_explicit(&p->read_idx, memory_order_relaxed, memory_scope_device);
    size_t ei = p->end_idx;
    size_t wi = wave_reserve_1(&p->write_idx, ri+ei);
    if (wi == ~(size_t)0)
        return -1;

    size_t pi = wrap(wi, ei);
    __memcpy_internal_aligned(p->packets + pi*size, ptr, size, align);

    return 0;
}

#define WRITE_PIPE_RESERVED_SIZE(SIZE, STYPE) \
ATTR int \
__write_pipe_4_##SIZE(__global struct pipeimp* p, reserve_id_t rid, uint i, const STYPE* ptr)  \
{ \
    size_t rin = __builtin_astype(rid, size_t) + i; \
    size_t pi = wrap(rin, p->end_idx); \
    ((__global STYPE *)p->packets)[pi] = *ptr; \
    return 0; \
}

DO_PIPE_SIZE(WRITE_PIPE_RESERVED_SIZE)

ATTR int
__write_pipe_4(__global struct pipeimp* p, reserve_id_t rid, uint i, const void *ptr, uint size, uint align)
{
    size_t rin = __builtin_astype(rid, size_t) + i; \
    size_t pi = wrap(rin, p->end_idx);
    __memcpy_internal_aligned(p->packets + pi*size, ptr, size, align);

    return 0;
}

