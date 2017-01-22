
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "irif.h"
#include "oclc.h"
#include "ockl_hsa.h"

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define ATTR __attribute__((always_inline))

// TODO Remove this workaround when the compiler is ready

#define AL(T,P,O,S) ({ \
    T __l; \
    switch (O) { \
    case __ockl_memory_order_acquire: \
        __l = atomic_load_explicit(P, memory_order_acquire, S); \
        break; \
    case __ockl_memory_order_seq_cst: \
        __l = atomic_load_explicit(P, memory_order_seq_cst, S); \
        break; \
    default: \
        __l = atomic_load_explicit(P, memory_order_relaxed, S); \
        break; \
    } \
    __l; \
})

#define AS(P,V,O,S) ({ \
    switch (O) { \
    case __ockl_memory_order_release: \
        atomic_store_explicit(P, V, memory_order_release, S); \
        break; \
    case __ockl_memory_order_seq_cst: \
        atomic_store_explicit(P, V, memory_order_seq_cst, S); \
        break; \
    default: \
        atomic_store_explicit(P, V, memory_order_relaxed, S); \
        break; \
    } \
})

#define AF(T,K,P,V,O,S) ({ \
    T __f; \
    switch (O) { \
    case __ockl_memory_order_acquire: \
        __f = atomic_fetch_##K##_explicit(P, V, memory_order_acquire, S); \
        break; \
    case __ockl_memory_order_release: \
        __f = atomic_fetch_##K##_explicit(P, V, memory_order_release, S); \
        break; \
    case __ockl_memory_order_acq_rel: \
        __f = atomic_fetch_##K##_explicit(P, V, memory_order_acq_rel, S); \
        break; \
    case __ockl_memory_order_seq_cst: \
        __f = atomic_fetch_##K##_explicit(P, V, memory_order_seq_cst, S); \
        break; \
    default: \
        __f = atomic_fetch_##K##_explicit(P, V, memory_order_relaxed, S); \
        break; \
    } \
    __f; \
})

#define AX(T,P,V,O,S) ({ \
    T __e; \
    switch (O) { \
    case __ockl_memory_order_acquire: \
        __e = atomic_exchange_explicit(P, V, memory_order_acquire, S); \
        break; \
    case __ockl_memory_order_release: \
        __e = atomic_exchange_explicit(P, V, memory_order_release, S); \
        break; \
    case __ockl_memory_order_acq_rel: \
        __e = atomic_exchange_explicit(P, V, memory_order_acq_rel, S); \
        break; \
    case __ockl_memory_order_seq_cst: \
        __e = atomic_exchange_explicit(P, V, memory_order_seq_cst, S); \
        break; \
    default: \
        __e = atomic_exchange_explicit(P, V, memory_order_relaxed, S); \
        break; \
    } \
    __e; \
})

#define AC(P,E,V,O,R,S) ({ \
    bool __c; \
    switch (O) { \
    case __ockl_memory_order_acquire: \
        __c = atomic_compare_exchange_strong_explicit(P, E, V, memory_order_acquire, R, S); \
        break; \
    case __ockl_memory_order_release: \
        __c = atomic_compare_exchange_strong_explicit(P, E, V, memory_order_release, R, S); \
        break; \
    case __ockl_memory_order_acq_rel: \
        __c = atomic_compare_exchange_strong_explicit(P, E, V, memory_order_acq_rel, R, S); \
        break; \
    case __ockl_memory_order_seq_cst: \
        __c = atomic_compare_exchange_strong_explicit(P, E, V, memory_order_seq_cst, R, S); \
        break; \
    default: \
        __c = atomic_compare_exchange_strong_explicit(P, E, V, memory_order_relaxed, R, S); \
        break; \
    } \
    __c; \
})

//
// HSA queue ops
//

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
        atomic_store_explicit(mb, id, memory_order_release, memory_scope_all_svm_devices);
        __llvm_amdgcn_s_sendmsg(1 | (0 << 4), __llvm_amdgcn_readfirstlane(id) & 0xff);
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
    } else {

        {
            __global amd_queue_t * q = s->queue_ptr;
            __global atomic_uint *lp = (__global atomic_uint *)&q->legacy_doorbell_lock;
            uint e = 0;
            while (!atomic_compare_exchange_strong_explicit(lp, &e, (uint)1, memory_order_acquire, memory_order_relaxed, memory_scope_all_svm_devices)) {
                __llvm_amdgcn_s_sleep(1);
                e = 0;
            }

            ulong legacy_dispatch_id = value + 1;

            if (legacy_dispatch_id > q->max_legacy_doorbell_dispatch_id_plus_1) {
                atomic_store_explicit((__global atomic_ulong *)&q->max_legacy_doorbell_dispatch_id_plus_1, legacy_dispatch_id, memory_order_relaxed, memory_scope_all_svm_devices);

                if (__oclc_ISA_version() < 800) {
                    legacy_dispatch_id = (ulong)(((uint)legacy_dispatch_id & ((q->hsa_queue.size << 1) - 1)) * 16);
                }

                *s->legacy_hardware_doorbell_ptr = (uint)legacy_dispatch_id;
            }

            atomic_store_explicit(lp, 0, memory_order_release, memory_scope_all_svm_devices);
        }
    }
}

