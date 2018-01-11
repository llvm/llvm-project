#include "csi.h"

WEAK void __csi_init() {}

__attribute__((always_inline))
WEAK void __csi_unit_init(const char * const file_name,
                          const instrumentation_counts_t counts) {}

__attribute__((always_inline))
WEAK void __csi_before_load(const csi_id_t load_id,
                            const void *addr,
                            const int32_t num_bytes,
                            const load_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_after_load(const csi_id_t load_id,
                           const void *addr,
                           const int32_t num_bytes,
                           const load_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_before_store(const csi_id_t store_id,
                             const void *addr,
                             const int32_t num_bytes,
                             const store_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_after_store(const csi_id_t store_id,
                            const void *addr,
                            const int32_t num_bytes,
                            const store_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_func_entry(const csi_id_t func_id, const func_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_func_exit(const csi_id_t func_exit_id,
                          const csi_id_t func_id, const func_exit_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_bb_entry(const csi_id_t bb_id, const bb_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_bb_exit(const csi_id_t bb_id, const bb_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_before_call(csi_id_t callsite_id, csi_id_t func_id,
                            const call_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_after_call(csi_id_t callsite_id, csi_id_t func_id,
                           const call_prop_t prop) {}

__attribute__((always_inline))
WEAK void __csi_detach(const csi_id_t detach_id) {}

__attribute__((always_inline))
WEAK void __csi_task(const csi_id_t task_id, const csi_id_t detach_id,
                     void *sp) {}

__attribute__((always_inline))
WEAK void __csi_task_exit(const csi_id_t task_exit_id,
                          const csi_id_t task_id,
                          const csi_id_t detach_id) {}

__attribute__((always_inline))
WEAK void __csi_detach_continue(const csi_id_t detach_continue_id,
                                const csi_id_t detach_id) {}

__attribute__((always_inline))
WEAK void __csi_sync(const csi_id_t sync_id) {}
