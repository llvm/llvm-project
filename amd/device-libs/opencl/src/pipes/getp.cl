/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "pipes.h"

#define ATTR __attribute__((always_inline, pure))

#define GET_PIPE_NUM_PACKETS_SIZE(SIZE, STYPE) \
ATTR uint \
__get_pipe_num_packets_##SIZE(__global struct pipeimp* p) \
{ \
    size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device); \
    size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device); \
    return (uint)(wi - ri); \
}

// DO_PIPE_SIZE(GET_PIPE_NUM_PACKETS_SIZE)

ATTR uint
__get_pipe_num_packets(__global struct pipeimp* p, uint size, uint align)
{
    size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device);
    size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device);
    return (uint)(wi - ri);
}

#define GET_PIPE_MAX_PACKETS_SIZE(SIZE, STYPE) \
ATTR uint \
__get_pipe_max_packets_##SIZE(__global struct pipeimp* p) \
{ \
    return (uint)p->end_idx; \
}

// DO_PIPE_SIZE(GET_PIPE_MAX_PACKETS_SIZE)

ATTR uint
__get_pipe_max_packets(__global struct pipeimp* p, uint size, uint align)
{
    return (uint)p->end_idx;
}

