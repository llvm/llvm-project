/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#include "pipes.h"
#include "wgscratch.h"

#define ATTR __attribute__((always_inline))

#define RESERVE_READ_PIPE_SIZE(SIZE, STYPE) \
ATTR reserve_id_t \
__reserve_read_pipe_##SIZE(__global struct pipeimp *p, uint num_packets) \
{ \
    size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device); \
    size_t rid = __amd_wresvn(&p->read_idx, wi, num_packets); \
 \
    if (rid + num_packets == wi) { \
        __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device); \
        __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device); \
    } \
 \
    return __builtin_astype(rid, reserve_id_t); \
}

// DO_PIPE_SIZE(RESERVE_READ_PIPE_SIZE)

ATTR reserve_id_t
__reserve_read_pipe(__global struct pipeimp *p, uint num_packets, uint size, uint align)
{
    size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device);
    size_t rid = __amd_wresvn(&p->read_idx, wi, num_packets);

    if (rid + num_packets == wi) {
        __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device);
        __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device);
    }

    return __builtin_astype(rid, reserve_id_t);
}

#define RESERVE_WRITE_PIPE_SIZE(SIZE, STYPE) \
ATTR reserve_id_t \
__reserve_write_pipe_##SIZE(__global struct pipeimp *p, uint num_packets) \
{ \
    size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device); \
    size_t ei = p->end_idx; \
    return __amd_wresvn(&p->write_idx, ri + ei, num_packets); \
}

// DO_PIPE_SIZE(RESERVE_WRITE_PIPE_SIZE)

ATTR reserve_id_t
__reserve_write_pipe(__global struct pipeimp *p, uint num_packets, uint size, uint align)
{
    size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device);
    size_t ei = p->end_idx;
    size_t rid = __amd_wresvn(&p->write_idx, ri + ei, num_packets);
    return __builtin_astype(rid, reserve_id_t);
}

// Work group functions

#define WORK_GROUP_RESERVE_READ_PIPE_SIZE(SIZE, STYPE) \
ATTR reserve_id_t \
__work_group_reserve_read_pipe_##SIZE(__global struct pipeimp *p, uint num_packets) \
{ \
    __local size_t *t = (__local size_t *)__get_scratch_lds(); \
 \
    if ((int)get_local_linear_id() == 0) { \
        size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device); \
        size_t rid = reserve(&p->read_idx, wi, num_packets); \
 \
        if (rid + num_packets == wi) { \
            __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device); \
            __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device); \
        } \
 \
        *t = rid; \
    } \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
 \
    return __builtin_astype(*t, reserve_id_t); \
}

// DO_PIPE_SIZE(WORK_GROUP_RESERVE_READ_PIPE_SIZE)

ATTR reserve_id_t
__work_group_reserve_read_pipe(__global struct pipeimp *p, uint num_packets, uint size, uint align)
{
    __local size_t *t = (__local size_t *)__get_scratch_lds();

    if ((int)get_local_linear_id() == 0) {
        size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device);
        size_t rid = reserve(&p->read_idx, wi, num_packets);

        if (rid + num_packets == wi) {
            __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device);
            __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device);
        }

        *t = rid;
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    return __builtin_astype(*t, reserve_id_t);
}

#define WORK_GROUP_RESERVE_WRITE_PIPE_SIZE(SIZE, STYPE) \
ATTR reserve_id_t \
__work_group_reserve_write_pipe_##SIZE(__global struct pipeimp *p, uint num_packets) \
{ \
    __local size_t *t = (__local size_t *)__get_scratch_lds(); \
 \
    if ((int)get_local_linear_id() == 0) { \
        size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device); \
        size_t ei = p->end_idx; \
        *t = reserve(&p->write_idx, ri + ei, num_packets); \
    } \
 \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
 \
    return __builtin_astype(*t, reserve_id_t); \
}

// DO_PIPE_SIZE(WORK_GROUP_RESERVE_WRITE_PIPE_SIZE)

ATTR reserve_id_t
__work_group_reserve_write_pipe(__global struct pipeimp *p, uint num_packets, uint size, uint align)
{
    __local size_t *t = (__local size_t *)__get_scratch_lds();

    if ((int)get_local_linear_id() == 0) {
        size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device);
        size_t ei = p->end_idx;
        *t = reserve(&p->write_idx, ri + ei, num_packets);
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    return __builtin_astype(*t, reserve_id_t);
}

// sub group functions

#define SUB_GROUP_RESERVE_READ_PIPE_SIZE(SIZE, STYPE) \
ATTR reserve_id_t \
__sub_group_reserve_read_pipe_##SIZE(__global struct pipeimp *p, uint num_packets) \
{ \
    size_t rid = ~(size_t)0; \
 \
    if (get_sub_group_local_id() == 0) { \
        size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device); \
        rid = reserve(&p->read_idx, wi, num_packets); \
 \
        if (rid + num_packets == wi) { \
            __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device); \
            __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device); \
        } \
    } \
 \
    return __builtin_astype(sub_group_broadcast(rid, 0), reserve_id_t); \
}

// DO_PIPE_SIZE(SUB_GROUP_RESERVE_READ_PIPE_SIZE)

ATTR reserve_id_t
__sub_group_reserve_read_pipe(__global struct pipeimp *p, uint num_packets, uint size, uint align)
{
    size_t rid = ~(size_t)0;

    if (get_sub_group_local_id() == 0) {
        size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device);
        rid = reserve(&p->read_idx, wi, num_packets);

        if (rid + num_packets == wi) {
            __opencl_atomic_store(&p->write_idx, 0, memory_order_relaxed, memory_scope_device);
            __opencl_atomic_store(&p->read_idx, 0, memory_order_relaxed, memory_scope_device);
        }
    }

    return __builtin_astype(sub_group_broadcast(rid, 0), reserve_id_t);
}

#define SUB_GROUP_RESERVE_WRITE_PIPE_SIZE(SIZE, STYPE) \
ATTR reserve_id_t \
__sub_group_reserve_write_pipe_##SIZE(__global struct pipeimp *p, uint num_packets) \
{ \
    size_t rid = ~(size_t)0; \
 \
    if (get_sub_group_local_id() == 0) { \
        size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device); \
        size_t ei = p->end_idx; \
        rid = reserve(&p->write_idx, ri + ei, num_packets); \
    } \
 \
    return __builtin_astype(sub_group_broadcast(rid, 0), reserve_id_t); \
}

// DO_PIPE_SIZE(SUB_GROUP_RESERVE_WRITE_PIPE_SIZE)

ATTR reserve_id_t
__sub_group_reserve_write_pipe(__global struct pipeimp *p, uint num_packets, uint size, uint align)
{
     size_t rid = ~(size_t)0;

    if (get_sub_group_local_id() == 0) {
        size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device);
        size_t ei = p->end_idx;
        rid = reserve(&p->write_idx, ri + ei, num_packets);
    }

    return __builtin_astype(sub_group_broadcast(rid, 0), reserve_id_t);
}

