
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "oclc.h"
#include "ockl_hsa.h"

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define ATTR

#define AL(T,P,O,S) __opencl_atomic_load(P,O,S)
#define AS(P,V,O,S) __opencl_atomic_store(P,V,O,S)
#define AF(T,K,P,V,O,S) __opencl_atomic_fetch_##K(P,V,O,S)
#define AX(T,P,V,O,S) __opencl_atomic_exchange(P,V,O,S)
#define AC(P,E,V,O,R,S) __opencl_atomic_compare_exchange_strong(P,E,V,O,R,S)

//
// HSA queue ops
//

ATTR ulong
OCKL_MANGLE_T(hsa_queue,load_read_index)(const __global hsa_queue_t *queue, __ockl_memory_order mem_order)
{
    const __global amd_queue_t *q = (const __global amd_queue_t *)queue;
    return AL(ulong, (__global atomic_ulong *)&q->read_dispatch_id, mem_order, memory_scope_all_svm_devices);
}

ATTR ulong
OCKL_MANGLE_T(hsa_queue,load_write_index)(const __global hsa_queue_t *queue, __ockl_memory_order mem_order)
{
    const __global amd_queue_t *q = (const __global amd_queue_t *)queue;
    return AL(ulong, (__global atomic_ulong *)&q->write_dispatch_id, mem_order, memory_scope_all_svm_devices);
}

ATTR ulong
OCKL_MANGLE_T(hsa_queue,add_write_index)(__global hsa_queue_t *queue, ulong value, __ockl_memory_order mem_order)
{
    __global amd_queue_t *q = (__global amd_queue_t *)queue;
    return AF(ulong, add, (__global atomic_ulong *)&q->write_dispatch_id, value, mem_order, memory_scope_all_svm_devices);
}

ATTR ulong
OCKL_MANGLE_T(hsa_queue,cas_write_index)(__global hsa_queue_t *queue, ulong expected, ulong value, __ockl_memory_order mem_order)
{
    __global amd_queue_t *q = (__global amd_queue_t *)queue;
    ulong e = expected;
    AC((__global atomic_ulong *)&q->write_dispatch_id, &e, value, mem_order, memory_order_relaxed, memory_scope_all_svm_devices);
    return e;
}

ATTR void
OCKL_MANGLE_T(hsa_queue,store_write_index)(__global hsa_queue_t *queue, ulong value, __ockl_memory_order mem_order)
{
    __global amd_queue_t *q = (__global amd_queue_t *)queue;
    AS((__global atomic_ulong *)&q->write_dispatch_id, value, mem_order, memory_scope_all_svm_devices);
}

//
// HSA signal ops
//

static ATTR void
update_mbox(const __global amd_signal_t *sig)
{
    __global atomic_ulong *mb = (__global atomic_ulong *)sig->event_mailbox_ptr;
    if (mb) {
        uint id = sig->event_id;
        AS(mb, id, memory_order_release, memory_scope_all_svm_devices);
        __builtin_amdgcn_s_sendmsg(1 | (0 << 4), __builtin_amdgcn_readfirstlane(id) & 0xff);
    }
}

ATTR long
OCKL_MANGLE_T(hsa_signal,load)(const hsa_signal_t sig, __ockl_memory_order mem_order)
{
    const __global amd_signal_t *s = (const __global amd_signal_t *)sig.handle;
    return AL(long, (__global atomic_long *)&s->value, mem_order, memory_scope_all_svm_devices);
}

ATTR void
OCKL_MANGLE_T(hsa_signal,add)(hsa_signal_t sig, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    AF(long, add, (__global atomic_long *)&s->value, value, mem_order, memory_scope_all_svm_devices);
    update_mbox(s);
}

ATTR void
OCKL_MANGLE_T(hsa_signal,and)(hsa_signal_t sig, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    AF(long, and, (__global atomic_long *)&s->value, value, mem_order, memory_scope_all_svm_devices);
    update_mbox(s);
}

ATTR void
OCKL_MANGLE_T(hsa_signal,or)(hsa_signal_t sig, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    AF(long, or, (__global atomic_long *)&s->value, value, mem_order, memory_scope_all_svm_devices);
    update_mbox(s);
}

ATTR void
OCKL_MANGLE_T(hsa_signal,xor)(hsa_signal_t sig, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    AF(long, xor, (__global atomic_long *)&s->value, value, mem_order, memory_scope_all_svm_devices);
    update_mbox(s);
}

ATTR long
OCKL_MANGLE_T(hsa_signal,exchange)(hsa_signal_t sig, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    long ret = AX(long, (__global atomic_long *)&s->value, value, mem_order, memory_scope_all_svm_devices);
    update_mbox(s);
    return ret;
}

ATTR void
OCKL_MANGLE_T(hsa_signal,subtract)(hsa_signal_t sig, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    AF(long, sub, (__global atomic_long *)&s->value, value, mem_order, memory_scope_all_svm_devices);
    update_mbox(s);
}

ATTR long
OCKL_MANGLE_T(hsa_signal,cas)(hsa_signal_t sig, long expected, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    long e = expected;
    if (AC((__global atomic_long *)&s->value, &e, value, mem_order, memory_order_relaxed, memory_scope_all_svm_devices))
        update_mbox(s);
    return e;
}

ATTR void
OCKL_MANGLE_T(hsa_signal,store)(hsa_signal_t sig, long value, __ockl_memory_order mem_order)
{
    __global amd_signal_t *s = (__global amd_signal_t *)sig.handle;
    if (s->kind == AMD_SIGNAL_KIND_USER) {
        AS((__global atomic_long *)&s->value, value, mem_order, memory_scope_all_svm_devices);
        update_mbox(s);
    } else if (__oclc_ISA_version >= 9000) {
        // Hardware doorbell supports AQL semantics.
        AS((__global atomic_ulong *)s->hardware_doorbell_ptr, (ulong)value, memory_order_release, memory_scope_all_svm_devices);
    } else {

        {
            __global amd_queue_t * q = s->queue_ptr;
            __global atomic_uint *lp = (__global atomic_uint *)&q->legacy_doorbell_lock;
            uint e = 0;
            while (!AC(lp, &e, (uint)1, memory_order_acquire, memory_order_relaxed, memory_scope_all_svm_devices)) {
                __builtin_amdgcn_s_sleep(1);
                e = 0;
            }

            ulong legacy_dispatch_id = value + 1;

            if (legacy_dispatch_id > q->max_legacy_doorbell_dispatch_id_plus_1) {
                AS((__global atomic_ulong *)&q->max_legacy_doorbell_dispatch_id_plus_1, legacy_dispatch_id, memory_order_relaxed, memory_scope_all_svm_devices);

                if (__oclc_ISA_version < 8000) {
                    legacy_dispatch_id = (ulong)(((uint)legacy_dispatch_id & ((q->hsa_queue.size << 1) - 1)) * 16);
                }

                *s->legacy_hardware_doorbell_ptr = (uint)legacy_dispatch_id;
            }

            AS(lp, 0, memory_order_release, memory_scope_all_svm_devices);
        }
    }
}

