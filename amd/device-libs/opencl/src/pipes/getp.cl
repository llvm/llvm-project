/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "pipes.h"

#define ATTR __attribute__((always_inline, pure))

static ATTR uint
num_packets(__global struct pipeimp* p)
{
    size_t ri = __opencl_atomic_load(&p->read_idx, memory_order_relaxed, memory_scope_device);
    size_t wi = __opencl_atomic_load(&p->write_idx, memory_order_relaxed, memory_scope_device);
    return (uint)(wi - ri);
}

ATTR uint
__get_pipe_num_packets_ro(__global struct pipeimp* p, uint size, uint align)
{
    return num_packets(p);
}

ATTR uint
__get_pipe_num_packets_wo(__global struct pipeimp* p, uint size, uint align)
{
    return num_packets(p);
}

ATTR uint
__get_pipe_max_packets_ro(__global struct pipeimp* p, uint size, uint align)
{
    return (uint)p->end_idx;
}

ATTR uint
__get_pipe_max_packets_wo(__global struct pipeimp* p, uint size, uint align)
{
    return (uint)p->end_idx;
}

