
#include "devenq.h"

#define LSIZE_LIMIT 65536U
#define LOCAL_ALIGN 16

struct rtinfo {
    __global char* kernel_object;
    uint private_segment_size;
    uint group_segment_size;
};

static inline void
copy_captured_context(__global void * restrict d, void * restrict s, uint size, uint align)
{
    if (align == 8) {
         __global ulong * restrict d8 = (__global ulong * restrict)d;
         ulong * restrict s8 = (ulong * restrict)s;
         uint n = size / align;
         uint r = size % align;
         for (uint i=0; i<n; ++i)
             d8[i] = s8[i];
         if (r != 0) {
             __global char * restrict dd = (__global char * restrict)(d8 + n);
             char * restrict ss = (char * restrict)(s8 + n);
             if (r > 3) {
                 *(__global uint * restrict)dd = *(uint * restrict)ss;
                 dd += 4;
                 ss += 4;
                 r -= 4;
             }
             if (r > 1) {
                 *(__global ushort * restrict)dd = *(ushort * restrict)ss;
                 dd += 2;
                 ss += 2;
                 r -= 2;
             }
             if (r > 0) {
                 *dd = *ss;
             }
        }
    } else if (align >= 16) {
        __global uint4 * restrict d16 = (__global uint4 * restrict)d;
        uint4 * restrict s16 = (uint4 * restrict)s;
        uint n = size / 16;
        uint r = size % 16;
        for (uint i=0; i<n; ++i)
            d16[i] = s16[i];
        if (r != 0) {
            __global char * restrict dd = (__global char * restrict)(d16 + n);
            char * restrict ss = (char * restrict)(s16 + n);
            if (r > 7) {
                *(__global ulong * restrict)dd = *(ulong * restrict)ss;
                dd += 8;
                ss += 8;
                r -= 8;
            }
            if (r > 3) {
                *(__global uint * restrict)dd = *(uint * restrict)ss;
                dd += 4;
                ss += 4;
                r -= 4;
            }
            if (r > 1) {
                *(__global ushort * restrict)dd = *(ushort * restrict)ss;
                dd += 2;
                ss += 2;
                r -= 2;
            }
            if (r > 0) {
                *dd = *ss;
            }
        }
    } else if (align == 4) {
        __global uint * restrict d4 = (__global uint * restrict)d;
        uint * restrict s4 = (uint * restrict)s;
        uint n = size / align;
        uint r = size % align;
        for (uint i=0; i<n; ++i)
            d4[i] = s4[i];
        if (r != 0) {
            __global char * restrict dd = (__global char * restrict)(d4 + n);
            char * restrict ss = (char * restrict)(s4 + n);
            if (r > 1) {
                *(__global ushort * restrict)dd = *(ushort * restrict)ss;
                dd += 2;
                ss += 2;
                r -= 2;
            }
            if (r > 0) {
                *dd = *ss;
            }
        }
    } else {
        __global char * restrict d1 = (__global char * restrict)d;
        char * restrict s1 = (char * restrict)s;
        for (uint i=0; i<size; ++i)
            d1[i] = s1[i];
    }
}

static inline void
copy_retain_waitlist(__global size_t *dst, const size_t *src, uint n)
{
    uint i;
    for (i=0; i<n; ++i) {
        __global AmdEvent *ev = (__global AmdEvent *)src[i];
        atomic_fetch_add_explicit((__global atomic_uint *)&ev->counter, (uint)1, memory_order_relaxed, memory_scope_device);
        dst[i] = src[i];
    }
}

__attribute__((overloadable, always_inline, const)) queue_t
get_default_queue(void)
{
    return __builtin_astype(get_vqueue(), queue_t);
}

__attribute__((overloadable)) int
enqueue_marker(queue_t q, uint nwl, const clk_event_t *wl, clk_event_t *ce)
{
    __global AmdVQueueHeader *vq = __builtin_astype(q, __global AmdVQueueHeader *);
    if (nwl > vq->wait_size)
        return CLK_ENQUEUE_FAILURE;

    // Get a wrap slot
    __global uint *amask = (__global uint *)vq->aql_slot_mask;
    int ai = reserve_slot(amask, vq->aql_slot_num, vq->mask_groups);
    if (ai < 0)
        return CLK_ENQUEUE_FAILURE;

    // Get a return event slot
    __global uint *emask = (__global uint *)vq->event_slot_mask;
    int ei = reserve_slot(emask, vq->event_slot_num, 1);
    if (ei < 0) {
        release_slot(amask, ai);
        return CLK_ENQUEUE_FAILURE;
    }

    // Initialize return event
    __global AmdEvent *ev = (__global AmdEvent *)vq->event_slots + ei;
    ev->state = CL_SUBMITTED;
    ev->counter = 2;
    ev->capture_info = 0;

    // Initialize wrap
    __global AmdAqlWrap *me = get_aql_wrap();
    __global AmdAqlWrap *aw = (__global AmdAqlWrap *)(vq + 1) + ai;

    aw->enqueue_flags = CLK_ENQUEUE_FLAGS_NO_WAIT;
    aw->command_id = atomic_fetch_add_explicit((__global atomic_uint *)&vq->command_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    aw->child_counter = 0;
    aw->completion = ev;
    aw->parent_wrap = me;

    if (nwl > 0)
        copy_retain_waitlist((__global size_t *)aw->wait_list, (const size_t *)wl, nwl);

    aw->wait_num = nwl;

    // A marker is never enqueued so ignore displatch packet

    // Tell the scheduler
    atomic_fetch_add_explicit((__global atomic_uint *)&me->child_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    atomic_store_explicit((__global atomic_uint *)&aw->state, AQL_WRAP_MARKER, memory_order_release, memory_scope_device);

    *ce = __builtin_astype(ev, clk_event_t);
    return 0;
}

int
__enqueue_kernel_basic(queue_t q, kernel_enqueue_flags_t f, const ndrange_t r, void *block, void *capture)
{
    uint csize = ((uint *)capture)[0];
    uint calign = ((uint *)capture)[1];
    __global AmdVQueueHeader *vq = __builtin_astype(q, __global AmdVQueueHeader *);

    if (align_up(csize, sizeof(size_t)) + NUM_IMPLICIT_ARGS*sizeof(size_t) > vq->arg_size ||
        mul24(mul24((uint)r.localWorkSize[0], (uint)r.localWorkSize[1]), (uint)r.localWorkSize[2]) > CL_DEVICE_MAX_WORK_GROUP_SIZE)
        return CLK_ENQUEUE_FAILURE;

    // Get a queue slot
    __global uint *amask = (__global uint *)vq->aql_slot_mask;
    int ai = reserve_slot(amask, vq->aql_slot_num, vq->mask_groups);
    if (ai < 0)
        return CLK_ENQUEUE_FAILURE;

    __global AmdAqlWrap *aw = (__global AmdAqlWrap *)(vq + 1) + ai;

    // Set up kernarg
    copy_captured_context(aw->aql.kernarg_address, capture, csize, calign);
    __global size_t *implicit = (__global size_t *)((__global char *)aw->aql.kernarg_address + align_up(csize, sizeof(size_t)));
    if (__oclc_ABI_version < 500) {
        implicit[0] = r.globalWorkOffset[0];
        implicit[1] = r.globalWorkOffset[1];
        implicit[2] = r.globalWorkOffset[2];
        implicit[3] = (size_t)get_printf_ptr();
        implicit[4] = (size_t)get_vqueue();
        implicit[5] = (size_t)aw;
    } else {
        implicit[0] = ((size_t)((uint)r.globalWorkSize[0] / (ushort)r.localWorkSize[0])) |
                      ((size_t)((uint)r.globalWorkSize[1] / (ushort)r.localWorkSize[1]) << 32);
        implicit[1] = ((size_t)((uint)r.globalWorkSize[2] / (ushort)r.localWorkSize[2])) |
                      ((size_t)(ushort)r.localWorkSize[0] << 32) |
                      ((size_t)(ushort)r.localWorkSize[1] << 48);
        implicit[2] = ((size_t)(ushort)r.localWorkSize[2]) |
                      ((size_t)((uint)r.globalWorkSize[0] % (ushort)r.localWorkSize[0]) << 16) |
                      ((size_t)((uint)r.globalWorkSize[1] % (ushort)r.localWorkSize[1]) << 32) |
                      ((size_t)((uint)r.globalWorkSize[2] % (ushort)r.localWorkSize[2]) << 48);
        implicit[5] = r.globalWorkOffset[0];
        implicit[6] = r.globalWorkOffset[1];
        implicit[7] = r.globalWorkOffset[2];
        implicit[8] = (size_t)(ushort)r.workDimension;
        implicit[9] = (size_t)get_printf_ptr();
        implicit[13] = (size_t)get_vqueue();
        implicit[14] = (size_t)aw;
        implicit[24] = get_bases();
        implicit[25] = get_hsa_queue();
    }

    const __global struct rtinfo *rti = (const __global struct rtinfo *)block;

    __global AmdAqlWrap *me = get_aql_wrap();

    aw->enqueue_flags = f;
    aw->command_id = atomic_fetch_add_explicit((__global atomic_uint *)&vq->command_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    aw->completion = 0UL;
    aw->parent_wrap = me;
    aw->wait_num = 0;
    aw->aql.header = (0x1 << 11) | (0x1 << 9) |(0x0 << 8) | (0x2 << 0);
    aw->aql.setup = r.workDimension;
    aw->aql.workgroup_size_x = (ushort)r.localWorkSize[0];
    aw->aql.workgroup_size_y = (ushort)r.localWorkSize[1];
    aw->aql.workgroup_size_z = (ushort)r.localWorkSize[2];
    aw->aql.grid_size_x = (uint)r.globalWorkSize[0];
    aw->aql.grid_size_y = (uint)r.globalWorkSize[1];
    aw->aql.grid_size_z = (uint)r.globalWorkSize[2];
    aw->aql.private_segment_size = rti->private_segment_size;
    aw->aql.group_segment_size = rti->group_segment_size;
    aw->aql.kernel_object = rti->kernel_object;
    aw->aql.completion_signal.handle = 0;

    atomic_fetch_add_explicit((__global atomic_uint *)&me->child_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    atomic_store_explicit((__global atomic_uint *)&aw->state, AQL_WRAP_READY, memory_order_release, memory_scope_device);
    return 0;
}

int
__enqueue_kernel_basic_events(queue_t q, kernel_enqueue_flags_t f, const ndrange_t r, uint nwl, const clk_event_t *wl, clk_event_t *ce, void *block, void *capture)
{
    uint csize = ((uint *)capture)[0];
    uint calign = ((uint *)capture)[1];
    __global AmdVQueueHeader *vq = __builtin_astype(q, __global AmdVQueueHeader *);

    if (align_up(csize, sizeof(size_t)) + NUM_IMPLICIT_ARGS*sizeof(size_t) > vq->arg_size ||
        nwl > vq->wait_size ||
        mul24(mul24((uint)r.localWorkSize[0], (uint)r.localWorkSize[1]), (uint)r.localWorkSize[2]) > CL_DEVICE_MAX_WORK_GROUP_SIZE)
        return CLK_ENQUEUE_FAILURE;

    __global uint *amask = (__global uint *)vq->aql_slot_mask;
    int ai = reserve_slot(amask, vq->aql_slot_num, vq->mask_groups);
    if (ai < 0)
        return CLK_ENQUEUE_FAILURE;

    __global AmdEvent *ev = (__global AmdEvent *)NULL;
    if (ce) {
        // Get a completion event slot
        __global uint *emask = (__global uint *)vq->event_slot_mask;
        int ei = reserve_slot(emask, vq->event_slot_num, 1);
        if (ei < 0) {
            release_slot(amask, ai);
            return CLK_ENQUEUE_FAILURE;
        }

        // Initialize completion event
        ev = (__global AmdEvent *)vq->event_slots + ei;
        ev->state = CL_SUBMITTED;
        ev->counter = 2;
        ev->capture_info = 0;
        *ce = __builtin_astype(ev, clk_event_t);
    }

    __global AmdAqlWrap *aw = (__global AmdAqlWrap *)(vq + 1) + ai;

    // Set up kernarg
    copy_captured_context(aw->aql.kernarg_address, capture, csize, calign);
    __global size_t *implicit = (__global size_t *)((__global char *)aw->aql.kernarg_address + align_up(csize, sizeof(size_t)));
    if (__oclc_ABI_version < 500) {
        implicit[0] = r.globalWorkOffset[0];
        implicit[1] = r.globalWorkOffset[1];
        implicit[2] = r.globalWorkOffset[2];
        implicit[3] = (size_t)get_printf_ptr();
        implicit[4] = (size_t)get_vqueue();
        implicit[5] = (size_t)aw;
    } else {
        implicit[0] = ((size_t)((uint)r.globalWorkSize[0] / (ushort)r.localWorkSize[0])) |
                      ((size_t)((uint)r.globalWorkSize[1] / (ushort)r.localWorkSize[1]) << 32);
        implicit[1] = ((size_t)((uint)r.globalWorkSize[2] / (ushort)r.localWorkSize[2])) |
                      ((size_t)(ushort)r.localWorkSize[0] << 32) |
                      ((size_t)(ushort)r.localWorkSize[1] << 48);
        implicit[2] = ((size_t)(ushort)r.localWorkSize[2]) |
                      ((size_t)((uint)r.globalWorkSize[0] % (ushort)r.localWorkSize[0]) << 16) |
                      ((size_t)((uint)r.globalWorkSize[1] % (ushort)r.localWorkSize[1]) << 32) |
                      ((size_t)((uint)r.globalWorkSize[2] % (ushort)r.localWorkSize[2]) << 48);
        implicit[5] = r.globalWorkOffset[0];
        implicit[6] = r.globalWorkOffset[1];
        implicit[7] = r.globalWorkOffset[2];
        implicit[8] = (size_t)(ushort)r.workDimension;
        implicit[9] = (size_t)get_printf_ptr();
        implicit[13] = (size_t)get_vqueue();
        implicit[14] = (size_t)aw;
        implicit[24] = get_bases();
        implicit[25] = get_hsa_queue();
    }

    const __global struct rtinfo *rti = (const __global struct rtinfo *)block;

    __global AmdAqlWrap *me = get_aql_wrap();

    aw->enqueue_flags = f;
    aw->command_id = atomic_fetch_add_explicit((__global atomic_uint *)&vq->command_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    aw->completion = ev;
    aw->parent_wrap = me;
    if (nwl > 0)
        copy_retain_waitlist(aw->wait_list, (const size_t *)wl, nwl);
    aw->wait_num = nwl;
    aw->aql.header = (ushort)((0x1 << 11) | (0x1 << 9) |(0x0 << 8) | (0x2 << 0));
    aw->aql.setup = (ushort)r.workDimension;
    aw->aql.workgroup_size_x = (ushort)r.localWorkSize[0];
    aw->aql.workgroup_size_y = (ushort)r.localWorkSize[1];
    aw->aql.workgroup_size_z = (ushort)r.localWorkSize[2];
    aw->aql.grid_size_x = (uint)r.globalWorkSize[0];
    aw->aql.grid_size_y = (uint)r.globalWorkSize[1];
    aw->aql.grid_size_z = (uint)r.globalWorkSize[2];
    aw->aql.private_segment_size = rti->private_segment_size;
    aw->aql.group_segment_size = rti->group_segment_size;
    aw->aql.kernel_object = rti->kernel_object;
    aw->aql.completion_signal.handle = 0;

    atomic_fetch_add_explicit((__global atomic_uint *)&me->child_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    atomic_store_explicit((__global atomic_uint *)&aw->state, AQL_WRAP_READY, memory_order_release, memory_scope_device);
    return 0;
}

int
__enqueue_kernel_varargs(queue_t q, kernel_enqueue_flags_t f, const ndrange_t r, void *block, void *capture, uint nl, __private size_t *ll)
{
    uint csize = ((uint *)capture)[0];
    uint calign = ((uint *)capture)[1];

    const __global struct rtinfo *rti = (const __global struct rtinfo *)block;
    uint lo = rti->group_segment_size;
    for (uint il=0; il<nl; ++il)
        lo = align_up(lo, LOCAL_ALIGN) + (uint)ll[il];

    __global AmdVQueueHeader *vq = __builtin_astype(q, __global AmdVQueueHeader *);

    if (lo > LSIZE_LIMIT ||
        align_up(align_up(csize, sizeof(uint)) + nl*sizeof(uint), sizeof(size_t)) + NUM_IMPLICIT_ARGS*sizeof(size_t) > vq->arg_size ||
        mul24(mul24((uint)r.localWorkSize[0], (uint)r.localWorkSize[1]), (uint)r.localWorkSize[2]) > CL_DEVICE_MAX_WORK_GROUP_SIZE)
        return CLK_ENQUEUE_FAILURE;

    // Get a queue slot
    __global uint *amask = (__global uint *)vq->aql_slot_mask;
    int ai = reserve_slot(amask, vq->aql_slot_num, vq->mask_groups);
    if (ai < 0)
        return CLK_ENQUEUE_FAILURE;

    __global AmdAqlWrap *aw = (__global AmdAqlWrap *)(vq + 1) + ai;

    // Set up kernarg
    copy_captured_context(aw->aql.kernarg_address, capture, csize, calign);

    __global uint *la = (__global uint *)((__global char *)aw->aql.kernarg_address + align_up(csize, sizeof(uint)));
    lo = rti->group_segment_size;
    for (uint il=0; il<nl; ++il)
        lo = (la[il] = align_up(lo, LOCAL_ALIGN)) + (uint)ll[il];

    __global size_t *implicit = (__global size_t *)((__global char *)aw->aql.kernarg_address +
            align_up(align_up(csize, sizeof(uint)) + nl*sizeof(uint), sizeof(size_t)));
    if (__oclc_ABI_version < 500) {
        implicit[0] = r.globalWorkOffset[0];
        implicit[1] = r.globalWorkOffset[1];
        implicit[2] = r.globalWorkOffset[2];
        implicit[3] = (size_t)get_printf_ptr();
        implicit[4] = (size_t)get_vqueue();
        implicit[5] = (size_t)aw;
    } else {
        implicit[0] = ((size_t)((uint)r.globalWorkSize[0] / (ushort)r.localWorkSize[0])) |
                      ((size_t)((uint)r.globalWorkSize[1] / (ushort)r.localWorkSize[1]) << 32);
        implicit[1] = ((size_t)((uint)r.globalWorkSize[2] / (ushort)r.localWorkSize[2])) |
                      ((size_t)(ushort)r.localWorkSize[0] << 32) |
                      ((size_t)(ushort)r.localWorkSize[1] << 48);
        implicit[2] = ((size_t)(ushort)r.localWorkSize[2]) |
                      ((size_t)((uint)r.globalWorkSize[0] % (ushort)r.localWorkSize[0]) << 16) |
                      ((size_t)((uint)r.globalWorkSize[1] % (ushort)r.localWorkSize[1]) << 32) |
                      ((size_t)((uint)r.globalWorkSize[2] % (ushort)r.localWorkSize[2]) << 48);
        implicit[5] = r.globalWorkOffset[0];
        implicit[6] = r.globalWorkOffset[1];
        implicit[7] = r.globalWorkOffset[2];
        implicit[8] = (size_t)(ushort)r.workDimension;
        implicit[9] = (size_t)get_printf_ptr();
        implicit[13] = (size_t)get_vqueue();
        implicit[14] = (size_t)aw;
        implicit[24] = get_bases();
        implicit[25] = get_hsa_queue();
    }

    __global AmdAqlWrap *me = get_aql_wrap();

    aw->enqueue_flags = f;
    aw->command_id = atomic_fetch_add_explicit((__global atomic_uint *)&vq->command_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    aw->completion = 0UL;
    aw->parent_wrap = me;
    aw->wait_num = 0;
    aw->aql.header = (0x1 << 11) | (0x1 << 9) |(0x0 << 8) | (0x2 << 0);
    aw->aql.setup = r.workDimension;
    aw->aql.workgroup_size_x = (ushort)r.localWorkSize[0];
    aw->aql.workgroup_size_y = (ushort)r.localWorkSize[1];
    aw->aql.workgroup_size_z = (ushort)r.localWorkSize[2];
    aw->aql.grid_size_x = (uint)r.globalWorkSize[0];
    aw->aql.grid_size_y = (uint)r.globalWorkSize[1];
    aw->aql.grid_size_z = (uint)r.globalWorkSize[2];
    aw->aql.private_segment_size = rti->private_segment_size;
    aw->aql.group_segment_size = lo;
    aw->aql.kernel_object = rti->kernel_object;
    aw->aql.completion_signal.handle = 0;

    atomic_fetch_add_explicit((__global atomic_uint *)&me->child_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    atomic_store_explicit((__global atomic_uint *)&aw->state, AQL_WRAP_READY, memory_order_release, memory_scope_device);
    return 0;
}


int
__enqueue_kernel_events_varargs(queue_t q, kernel_enqueue_flags_t f, const ndrange_t r, int nwl, const clk_event_t *wl, clk_event_t *ce, void *block, void *capture, uint nl, __private size_t *ll)
{
    uint csize = ((uint *)capture)[0];
    uint calign = ((uint *)capture)[1];

    const __global struct rtinfo *rti = (const __global struct rtinfo *)block;
    uint lo = rti->group_segment_size;
    for (uint il=0; il<nl; ++il)
        lo = align_up(lo, LOCAL_ALIGN) + (uint)ll[il];

    __global AmdVQueueHeader *vq = __builtin_astype(q, __global AmdVQueueHeader *);

    if (lo > LSIZE_LIMIT ||
        nwl > vq->wait_size ||
        align_up(align_up(csize, sizeof(uint)) + nl*sizeof(uint), sizeof(size_t)) + NUM_IMPLICIT_ARGS*sizeof(size_t) > vq->arg_size ||
        mul24(mul24((uint)r.localWorkSize[0], (uint)r.localWorkSize[1]), (uint)r.localWorkSize[2]) > CL_DEVICE_MAX_WORK_GROUP_SIZE)
        return CLK_ENQUEUE_FAILURE;

    // Get a queue slot
    __global uint *amask = (__global uint *)vq->aql_slot_mask;
    int ai = reserve_slot(amask, vq->aql_slot_num, vq->mask_groups);
    if (ai < 0)
        return CLK_ENQUEUE_FAILURE;

    __global AmdEvent *ev = (__global AmdEvent *)NULL;
    if (ce) {
        // Get a completion event slot
        __global uint *emask = (__global uint *)vq->event_slot_mask;
        int ei = reserve_slot(emask, vq->event_slot_num, 1);
        if (ei < 0) {
            release_slot(amask, ai);
            return CLK_ENQUEUE_FAILURE;
        }

        // Initialize completion event
        ev = (__global AmdEvent *)vq->event_slots + ei;
        ev->state = CL_SUBMITTED;
        ev->counter = 2;
        ev->capture_info = 0;
        *ce = __builtin_astype(ev, clk_event_t);
    }

    __global AmdAqlWrap *aw = (__global AmdAqlWrap *)(vq + 1) + ai;

    // Set up kernarg
    copy_captured_context(aw->aql.kernarg_address, capture, csize, calign);

    __global uint *la = (__global uint *)((__global char *)aw->aql.kernarg_address + align_up(csize, sizeof(uint)));
    lo = rti->group_segment_size;
    for (uint il=0; il<nl; ++il)
        lo = (la[il] = align_up(lo, LOCAL_ALIGN)) + (uint)ll[il];

    __global size_t *implicit = (__global size_t *)((__global char *)aw->aql.kernarg_address +
            align_up(align_up(csize, sizeof(uint)) + nl*sizeof(uint), sizeof(size_t)));
    if (__oclc_ABI_version < 500) {
        implicit[0] = r.globalWorkOffset[0];
        implicit[1] = r.globalWorkOffset[1];
        implicit[2] = r.globalWorkOffset[2];
        implicit[3] = (size_t)get_printf_ptr();
        implicit[4] = (size_t)get_vqueue();
        implicit[5] = (size_t)aw;
    } else {
        implicit[0] = ((size_t)((uint)r.globalWorkSize[0] / (ushort)r.localWorkSize[0])) |
                      ((size_t)((uint)r.globalWorkSize[1] / (ushort)r.localWorkSize[1]) << 32);
        implicit[1] = ((size_t)((uint)r.globalWorkSize[2] / (ushort)r.localWorkSize[2])) |
                      ((size_t)(ushort)r.localWorkSize[0] << 32) |
                      ((size_t)(ushort)r.localWorkSize[1] << 48);
        implicit[2] = ((size_t)(ushort)r.localWorkSize[2]) |
                      ((size_t)((uint)r.globalWorkSize[0] % (ushort)r.localWorkSize[0]) << 16) |
                      ((size_t)((uint)r.globalWorkSize[1] % (ushort)r.localWorkSize[1]) << 32) |
                      ((size_t)((uint)r.globalWorkSize[2] % (ushort)r.localWorkSize[2]) << 48);
        implicit[5] = r.globalWorkOffset[0];
        implicit[6] = r.globalWorkOffset[1];
        implicit[7] = r.globalWorkOffset[2];
        implicit[8] = (size_t)(ushort)r.workDimension;
        implicit[9] = (size_t)get_printf_ptr();
        implicit[13] = (size_t)get_vqueue();
        implicit[14] = (size_t)aw;
    }

    __global AmdAqlWrap *me = get_aql_wrap();

    aw->enqueue_flags = f;
    aw->command_id = atomic_fetch_add_explicit((__global atomic_uint *)&vq->command_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    aw->completion = ev;
    aw->parent_wrap = me;
    if (nwl > 0)
        copy_retain_waitlist((__global size_t *)aw->wait_list, (const size_t *)wl, nwl);
    aw->wait_num = nwl;
    aw->aql.header = (0x1 << 11) | (0x1 << 9) |(0x0 << 8) | (0x2 << 0);
    aw->aql.setup = r.workDimension;
    aw->aql.workgroup_size_x = (ushort)r.localWorkSize[0];
    aw->aql.workgroup_size_y = (ushort)r.localWorkSize[1];
    aw->aql.workgroup_size_z = (ushort)r.localWorkSize[2];
    aw->aql.grid_size_x = (uint)r.globalWorkSize[0];
    aw->aql.grid_size_y = (uint)r.globalWorkSize[1];
    aw->aql.grid_size_z = (uint)r.globalWorkSize[2];
    aw->aql.private_segment_size = rti->private_segment_size;
    aw->aql.group_segment_size = lo;
    aw->aql.kernel_object = rti->kernel_object;
    aw->aql.completion_signal.handle = 0;

    atomic_fetch_add_explicit((__global atomic_uint *)&me->child_counter, (uint)1, memory_order_relaxed, memory_scope_device);
    atomic_store_explicit((__global atomic_uint *)&aw->state, AQL_WRAP_READY, memory_order_release, memory_scope_device);
    return 0;
}

