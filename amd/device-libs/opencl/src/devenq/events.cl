
#include "devenq.h"

#define ATTR __attribute__((overloadable, always_inline))

ATTR void
retain_event(clk_event_t e)
{
    __global AmdEvent *ev = __builtin_astype(e, __global AmdEvent *);
    atomic_fetch_add_explicit((__global atomic_uint *)&ev->counter, (uint)1, memory_order_relaxed, memory_scope_device);
}

ATTR void
release_event(clk_event_t e)
{
    __global AmdEvent *ev = __builtin_astype(e, __global AmdEvent *);
    uint c = atomic_fetch_sub_explicit((__global atomic_uint *)&ev->counter, (uint)1, memory_order_relaxed, memory_scope_device);
    if (c == 1U) {
        __global AmdVQueueHeader *vq = get_vqueue();
        __global uint *emask = (__global uint *)vq->event_slot_mask;
        __global AmdEvent *eb = (__global AmdEvent *)vq->event_slots;
        uint i = ev - eb;
        release_slot(emask, i);
    }
}

ATTR clk_event_t
create_user_event(void)
{
    __global AmdVQueueHeader *vq = get_vqueue();
    __global uint *emask = (__global uint *)vq->event_slot_mask;
    int i = reserve_slot(emask, vq->event_slot_num, 1);

    if (i >= 0) {
        __global AmdEvent *ev = (__global AmdEvent *)vq->event_slots + i;
        ev->state = CL_SUBMITTED;
        ev->counter = 1;
        ev->capture_info = 0;
        return __builtin_astype(ev, clk_event_t);
    } else
        return __builtin_astype((ulong)0, clk_event_t);
}

ATTR bool
is_valid_event(clk_event_t e)
{
    return __builtin_astype(e, ulong) != (ulong)0;
}

ATTR void
set_user_event_status(clk_event_t e, int s)
{
    __global AmdEvent *ev = __builtin_astype(e, __global AmdEvent *);
    atomic_store_explicit((__global atomic_uint *)&ev->state, (uint)s, memory_order_release, memory_scope_device);
}

ATTR void
capture_event_profiling_info(clk_event_t e, clk_profiling_info n, __global void *p)
{
    // Currently the second argument must be CLK_PROFILING_COMMAND_EXEC_TIME
    __global AmdEvent *ev = __builtin_astype(e, __global AmdEvent *);

    // Set the pointer now in case we're racing with the scheduler
    atomic_store_explicit((__global atomic_ulong *)&ev->capture_info, (ulong)p, memory_order_relaxed, memory_scope_device);

    uint state = atomic_load_explicit((__global atomic_uint *)&ev->state, memory_order_acquire, memory_scope_device);
    if (state == CL_COMPLETE) {
        __global ulong *t = (__global ulong *)ev->timer;
        ((__global ulong *)p)[0] = t[PROFILING_COMMAND_END] - t[PROFILING_COMMAND_START];
        ((__global ulong *)p)[1] = t[PROFILING_COMMAND_COMPLETE] - t[PROFILING_COMMAND_START];
    }
}

