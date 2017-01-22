/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef OCKL_HSA_H
#define OCKL_HSA_H

#include "ockl.h"
#include "device_amd_hsa.h"

typedef enum __ockl_memory_order_e {
  __ockl_memory_order_relaxed,
  __ockl_memory_order_acquire,
  __ockl_memory_order_release,
  __ockl_memory_order_acq_rel,
  __ockl_memory_order_seq_cst,
} __ockl_memory_order;

extern ulong OCKL_MANGLE_T(hsa_queue,load_write_index)(const __global hsa_queue_t *queue, __ockl_memory_order mem_order);
extern ulong OCKL_MANGLE_T(hsa_queue,add_write_index)(__global hsa_queue_t *queue, ulong value, __ockl_memory_order mem_order);
extern ulong OCKL_MANGLE_T(hsa_queue,cas_write_index)(__global hsa_queue_t *queue, ulong expected, ulong value, __ockl_memory_order mem_order);
extern void OCKL_MANGLE_T(hsa_queue,store_write_index)(__global hsa_queue_t *queue, ulong value, __ockl_memory_order mem_order);
 
extern long OCKL_MANGLE_T(hsa_signal,load)(const hsa_signal_t sig, __ockl_memory_order mem_order);
extern void OCKL_MANGLE_T(hsa_signal,add)(hsa_signal_t sig, long value, __ockl_memory_order mem_order);
extern void OCKL_MANGLE_T(hsa_signal,and)(hsa_signal_t sig, long value, __ockl_memory_order mem_order);
extern void OCKL_MANGLE_T(hsa_signal,or)(hsa_signal_t sig, long value, __ockl_memory_order mem_order);
extern void OCKL_MANGLE_T(hsa_signal,xor)(hsa_signal_t sig, long value, __ockl_memory_order mem_order);
extern long OCKL_MANGLE_T(hsa_signal,exchange)(hsa_signal_t sig, long value, __ockl_memory_order mem_order);
extern void OCKL_MANGLE_T(hsa_signal,subtract)(hsa_signal_t sig, long value, __ockl_memory_order mem_order);
extern long OCKL_MANGLE_T(hsa_signal,cas)(hsa_signal_t sig, long expected, long value, __ockl_memory_order mem_order);
extern void OCKL_MANGLE_T(hsa_signal,store)(hsa_signal_t sig, long value, __ockl_memory_order mem_order);

#endif // OCKL_HSA_H
