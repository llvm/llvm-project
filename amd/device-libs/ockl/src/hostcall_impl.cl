/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl_hsa.h"

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define AC(P, E, V, O, R, S)                                                   \
    __opencl_atomic_compare_exchange_strong(P, E, V, O, R, S)
#define AL(P, O, S) __opencl_atomic_load(P, O, S)
#define AF(K, P, V, O, S) __opencl_atomic_fetch_##K(P, V, O, S)

typedef enum { STATUS_SUCCESS, STATUS_BUSY } status_t;

typedef enum {
    CONTROL_OFFSET_READY_FLAG = 0,
    CONTROL_OFFSET_RESERVED0 = 1,
} control_offset_t;

typedef enum {
    CONTROL_WIDTH_READY_FLAG = 1,
    CONTROL_WIDTH_RESERVED0 = 31,
} control_width_t;

typedef struct {
    ulong next;
    ulong activemask;
    uint service;
    uint control;
} header_t;

typedef struct {
    // 64 slots of 8 ulongs each
    ulong slots[64][8];
} payload_t;

// Note: Hostcall buffer struct defined here is not an exact
// match of runtime buffer layout but matches its prefix that
// this code tries to access.
typedef struct {
    __global header_t *headers;
    __global payload_t *payloads;
    hsa_signal_t doorbell;
    ulong free_stack;
    ulong ready_stack;
    ulong index_mask;
} buffer_t;

static void
send_signal(hsa_signal_t signal)
{
    __ockl_hsa_signal_add(signal, 1, __ockl_memory_order_release);
}

static __global header_t *
get_header(__global buffer_t *buffer, ulong ptr)
{
    return buffer->headers + (ptr & buffer->index_mask);
}

static __global payload_t *
get_payload(__global buffer_t *buffer, ulong ptr)
{
    return buffer->payloads + (ptr & buffer->index_mask);
}

static uint
get_control_field(uint control, uint offset, uint width)
{
    return (control >> offset) & ((1 << width) - 1);
}

static uint
get_ready_flag(uint control)
{
    return get_control_field(control, CONTROL_OFFSET_READY_FLAG,
                             CONTROL_WIDTH_READY_FLAG);
}

static uint
set_control_field(uint control, uint offset, uint width, uint value)
{
    uint mask = ~(((1 << width) - 1) << offset);
    return (control & mask) | (value << offset);
}

static uint
set_ready_flag(uint control)
{
    return set_control_field(control, CONTROL_OFFSET_READY_FLAG,
                             CONTROL_WIDTH_READY_FLAG, 1);
}

static uint
optimizationBarrierHack(uint in_val)
{
    uint out_val;
    __asm__ volatile("" : "=v"(out_val) : "0"(in_val));
    return out_val;
}

static ulong
pop(__global ulong *top, __global buffer_t *buffer)
{
    ulong F = AL((__global atomic_ulong *)top, memory_order_acquire,
                 memory_scope_all_svm_devices);
    // F is guaranteed to be non-zero, since there are at least as
    // many packets as there are waves, and each wave can hold at most
    // one packet.
    while (true) {
        __global header_t *P = get_header(buffer, F);
        ulong N = AL((__global atomic_ulong *)&P->next, memory_order_relaxed,
                     memory_scope_all_svm_devices);
        if (AC((__global atomic_ulong *)top, &F, N, memory_order_acquire,
               memory_order_relaxed, memory_scope_all_svm_devices)) {
            break;
        }
        __builtin_amdgcn_s_sleep(1);
    }

    return F;
}

/** \brief Use the first active lane to get a free packet and
 *         broadcast to the whole wave.
 */
static ulong
pop_free_stack(__global buffer_t *buffer, uint me, uint low)
{
    ulong packet_ptr = 0;
    if (me == low) {
        packet_ptr = pop(&buffer->free_stack, buffer);
    }

    uint ptr_lo = packet_ptr;
    uint ptr_hi = packet_ptr >> 32;
    ptr_lo = __builtin_amdgcn_readfirstlane(ptr_lo);
    ptr_hi = __builtin_amdgcn_readfirstlane(ptr_hi);

    return ((ulong)ptr_hi << 32) | ptr_lo;
}

static void
push(__global ulong *top, ulong ptr, __global buffer_t *buffer)
{
    ulong F = AL((__global const atomic_ulong *)top, memory_order_relaxed,
                 memory_scope_all_svm_devices);
    __global header_t *P = get_header(buffer, ptr);

    while (true) {
        P->next = F;
        if (AC((__global atomic_ulong *)top, &F, ptr, memory_order_release,
               memory_order_relaxed, memory_scope_all_svm_devices))
            break;
        __builtin_amdgcn_s_sleep(1);
    }
}

/** \brief Use the first active lane in a wave to submit a ready
 *         packet and signal the host.
 */
static void
push_ready_stack(__global buffer_t *buffer, ulong ptr, uint me, uint low)
{
    if (me == low) {
        push(&buffer->ready_stack, ptr, buffer);
        send_signal(buffer->doorbell);
    }
}

static ulong
inc_ptr_tag(ulong ptr, ulong index_mask)
{
    // Unit step for the tag.
    ulong inc = index_mask + 1;
    ptr += inc;
    // When the tag for index 0 wraps, increment the tag.
    return ptr == 0 ? inc : ptr;
}

/** \brief Return the packet after incrementing the ABA tag
 */
static void
return_free_packet(__global buffer_t *buffer, ulong ptr, uint me, uint low)
{
    if (me == low) {
        ptr = inc_ptr_tag(ptr, buffer->index_mask);
        push(&buffer->free_stack, ptr, buffer);
    }
}

static void
fill_packet(__global header_t *header, __global payload_t *payload,
            uint service_id, ulong arg0, ulong arg1, ulong arg2, ulong arg3,
            ulong arg4, ulong arg5, ulong arg6, ulong arg7, uint me, uint low)
{
    ulong active = __builtin_amdgcn_read_exec();
    if (me == low) {
        header->service = service_id;
        header->activemask = active;
        uint control = set_ready_flag(0);
        header->control = control;
    }

    __global ulong *ptr = payload->slots[me];
    ptr[0] = arg0;
    ptr[1] = arg1;
    ptr[2] = arg2;
    ptr[3] = arg3;
    ptr[4] = arg4;
    ptr[5] = arg5;
    ptr[6] = arg6;
    ptr[7] = arg7;
}

/** \brief Wait for the host response and return the first two ulong
 *         entries per workitem.
 *
 *  After the packet is submitted in READY state, the wave spins until
 *  the host changes the state to DONE. Each workitem reads the first
 *  two ulong elements in its slot and returns this.
 */
static long2
get_return_value(__global header_t *header, __global payload_t *payload,
                 uint me, uint low)
{
    // The while loop needs to be executed by all active
    // lanes. Otherwise, later reads from ptr are performed only by
    // the first thread, while other threads reuse a value cached from
    // previous operations. The use of readfirstlane in the while loop
    // prevents this reordering.
    //
    // In the absence of the readfirstlane, only one thread has a
    // sequenced-before relation from the atomic load on
    // header->control to the ordinary loads on ptr. As a result, the
    // compiler is free to reorder operations in such a way that the
    // ordinary loads are performed only by the first thread. The use
    // of readfirstlane provides a stronger code-motion barrier, and
    // it effectively "spreads out" the sequenced-before relation to
    // the ordinary stores in other threads too.
    while (true) {
        uint ready_flag = 1;
        if (me == low) {
            uint control =
                AL((__global const atomic_uint *)&header->control,
                   memory_order_acquire, memory_scope_all_svm_devices);
            ready_flag = get_ready_flag(control);
        }
        ready_flag = __builtin_amdgcn_readfirstlane(ready_flag);
        if (ready_flag == 0)
            break;
        __builtin_amdgcn_s_sleep(1);
    }

    __global ulong *ptr = (__global ulong *)(payload->slots + me);
    ulong value0 = *ptr++;
    ulong value1 = *ptr;

    long2 retval = {value0, value1};
    return retval;
}

/** \brief The implementation that should be hidden behind an ABI
 *
 *  The transaction is a wave-wide operation, where the service_id
 *  must be uniform, but the parameters are different for each
 *  workitem. Parameters from all active lanes are written into a
 *  hostcall packet. The hostcall blocks until the host processes the
 *  request, and returns the response it receiveds.
 *
 *  TODO: This function and everything above it should eventually move
 *  to a separate library that is loaded by the language runtime. The
 *  function itself will be exposed as an orindary function symbol to
 *  be linked into kernel objects that are loaded after this library.
 *
 *  *** INTERNAL USE ONLY ***
 *  Internal function, not safe for direct use in user
 *  code. Application kernels must only use __ockl_hostcall_preview()
 *  defined elsewhere.
 *
 */
long2
__ockl_hostcall_internal(void *_buffer, uint service_id, ulong arg0, ulong arg1,
                         ulong arg2, ulong arg3, ulong arg4, ulong arg5,
                         ulong arg6, ulong arg7)
{
    uint me = __ockl_lane_u32();
    me = optimizationBarrierHack(me);
    uint low = __builtin_amdgcn_readfirstlane(me);

    __global buffer_t *buffer = (__global buffer_t *)_buffer;
    ulong packet_ptr = pop_free_stack(buffer, me, low);
    __global header_t *header = get_header(buffer, packet_ptr);
    __global payload_t *payload = get_payload(buffer, packet_ptr);

    fill_packet(header, payload, service_id, arg0, arg1, arg2, arg3, arg4, arg5,
                arg6, arg7, me, low);
    push_ready_stack(buffer, packet_ptr, me, low);

    long2 retval = get_return_value(header, payload, me, low);
    return_free_packet(buffer, packet_ptr, me, low);
    return retval;
}
