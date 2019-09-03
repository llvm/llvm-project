#pragma once 

#include <stdint.h>

#ifdef __cplusplus
#define EXTERN_C extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C
#define EXTERN_C_END
#include <stdbool.h> // for C99 bool type
#endif

#define WEAK __attribute__((weak))

// API function signatures
EXTERN_C

/**
 * Unless a type requires bitwise operations (e.g., property lists), we use
 * signed integers. We don't need the extra bit of data, and using unsigned
 * integers can lead to subtle bugs. See
 * http://www.soundsoftware.ac.uk/c-pitfall-unsigned
 */

typedef int64_t csi_id_t;

#define UNKNOWN_CSI_ID ((csi_id_t)-1)

typedef struct {
  csi_id_t num_func;
  csi_id_t num_func_exit;
  csi_id_t num_loop;
  csi_id_t num_loop_exit;
  csi_id_t num_bb;
  csi_id_t num_callsite;
  csi_id_t num_load;
  csi_id_t num_store;
  csi_id_t num_detach;
  csi_id_t num_task;
  csi_id_t num_task_exit;
  csi_id_t num_detach_continue;
  csi_id_t num_sync;
  csi_id_t num_alloca;
  csi_id_t num_allocfn;
  csi_id_t num_free;
} instrumentation_counts_t;

// Property bitfields.

typedef struct {
  // The function might spawn.
  unsigned may_spawn : 1;
  // Pad struct to 64 total bits.
  uint64_t _padding : 63;
} func_prop_t;

typedef struct {
  // The function might have spawned.
  unsigned may_spawn : 1;
  // This function exit returns an exception.
  unsigned eh_return : 1;
  // Pad struct to 64 total bits.
  uint64_t _padding : 62;
} func_exit_prop_t;

typedef struct {
  // The basic block is a landingpad.
  unsigned is_landingpad : 1;
  // The basic block is an exception-handling pad.
  unsigned is_ehpad : 1;
  // Pad struct to 64 total bits.
  uint64_t _padding : 62;
} bb_prop_t;

typedef struct {
  // The loop is a Tapir loop
  unsigned is_tapir_loop : 1;
  // The loop has one exiting block: the latch.
  unsigned has_one_exiting_block : 1;
  uint64_t _padding : 62;
} loop_prop_t;

typedef struct {
  // This exiting block of the loop is a latch.
  unsigned is_latch : 1;
  uint64_t _padding : 63;
} loop_exit_prop_t;

typedef struct {
  // The task is the body of a Tapir loop
  unsigned is_tapir_loop_body : 1;
  uint64_t _padding : 63;
} task_prop_t;

typedef struct {
  // The task is the body of a Tapir loop
  unsigned is_tapir_loop_body : 1;
  uint64_t _padding : 63;
} task_exit_prop_t;

typedef struct {
  // The call is indirect.
  unsigned is_indirect : 1;
  // Pad struct to 64 total bits.
  uint64_t _padding : 63;
} call_prop_t;

typedef struct {
  // The alignment of the load.
  unsigned alignment : 8;
  // The loaded address is in a vtable.
  unsigned is_vtable_access : 1;
  // The loaded address points to constant data.
  unsigned is_constant : 1;
  // The loaded address is on the stack.
  unsigned is_on_stack : 1;
  // The loaded address cannot be captured.
  unsigned may_be_captured : 1;
  // The loaded address is read before it is written in the same basic block.
  unsigned is_read_before_write_in_bb : 1;
  // Pad struct to 64 total bits.
  uint64_t _padding : 51;
} load_prop_t;

typedef struct {
  // The alignment of the store.
  unsigned alignment : 8;
  // The stored address is in a vtable.
  unsigned is_vtable_access : 1;
  // The stored address points to constant data.
  unsigned is_constant : 1;
  // The stored address is on the stack.
  unsigned is_on_stack : 1;
  // The stored address cannot be captured.
  unsigned may_be_captured : 1;
  // Pad struct to 64 total bits.
  uint64_t _padding : 52;
} store_prop_t;

typedef struct {
  // The alloca is static.
  unsigned is_static : 1;
  // Pad struct to 64 total bits.
  uint64_t _padding : 63;
} alloca_prop_t;

typedef struct {
  // Type of the allocation function (e.g., malloc, calloc, new).
  unsigned allocfn_ty : 8;
  // Pad struct to 64 total bits.
  uint64_t _padding : 56;
} allocfn_prop_t;

typedef struct {
  // Type of the free function (e.g., free, delete).
  unsigned free_ty : 8;
  // Pad struct to 64 total bits.
  uint64_t _padding : 56;
} free_prop_t;

WEAK void __csi_init();

WEAK void __csi_unit_init(const char *const file_name,
                          const instrumentation_counts_t counts);

///-----------------------------------------------------------------------------
/// Function entry/exit
WEAK void __csi_func_entry(const csi_id_t func_id, const func_prop_t prop);

WEAK void __csi_func_exit(const csi_id_t func_exit_id, const csi_id_t func_id,
                          const func_exit_prop_t prop);

///-----------------------------------------------------------------------------
/// Loop hooks.  The before_loop hook occurs just before the loop begins
/// execution, i.e., in a loop preheader block.  The loopbody_entry hook
/// executes at the beginning of each loop iteration.  The loopbody_exit hook
/// executes at the end of each iteration.  The after_loop hook occurs just
/// after the loop completes execution, i.e., in each dedicated exit from the
/// loop.  Loops are transformed into a canonical form before instrumentation is
/// inserted to ensure that before_loop and after_loop hooks are encountered as
/// a pair.
///
/// TODO: Pass loop IVs to the loopbody_entry hook?
WEAK void __csi_before_loop(const csi_id_t loop_id, const int64_t trip_count,
                            const loop_prop_t prop);
WEAK void __csi_after_loop(const csi_id_t loop_id, const loop_prop_t prop);
WEAK void __csi_loopbody_entry(const csi_id_t loop_id, const loop_prop_t prop);
WEAK void __csi_loopbody_exit(const csi_id_t loop_exit_id,
                              const csi_id_t loop_id,
                              const loop_exit_prop_t prop);

///-----------------------------------------------------------------------------
/// Basic block entry/exit.  The bb_entry hook comes after any PHI hooks in that
/// basic block.  The bb_exit hook comes before any hooks for terminators, e.g.,
/// for invoke instructions.
WEAK void __csi_bb_entry(const csi_id_t bb_id, const bb_prop_t prop);

WEAK void __csi_bb_exit(const csi_id_t bb_id, const bb_prop_t prop);

///-----------------------------------------------------------------------------
/// Callsite hooks
WEAK void __csi_before_call(const csi_id_t call_id, const csi_id_t func_id,
                            const call_prop_t prop);

WEAK void __csi_after_call(const csi_id_t call_id, const csi_id_t func_id,
                           const call_prop_t prop);

///-----------------------------------------------------------------------------
/// Hooks for loads and stores
WEAK void __csi_before_load(const csi_id_t load_id, const void *addr,
                            const int32_t num_bytes, const load_prop_t prop);

WEAK void __csi_after_load(const csi_id_t load_id, const void *addr,
                           const int32_t num_bytes, const load_prop_t prop);

WEAK void __csi_before_store(const csi_id_t store_id, const void *addr,
                             const int32_t num_bytes, const store_prop_t prop);

WEAK void __csi_after_store(const csi_id_t store_id, const void *addr,
                            const int32_t num_bytes, const store_prop_t prop);

///-----------------------------------------------------------------------------
/// Hooks for Tapir control flow.
WEAK void __csi_detach(const csi_id_t detach_id, const int32_t *has_spawned);

WEAK void __csi_task(const csi_id_t task_id, const csi_id_t detach_id,
                     const task_prop_t prop);

WEAK void __csi_task_exit(const csi_id_t task_exit_id, const csi_id_t task_id,
                          const csi_id_t detach_id,
                          const task_exit_prop_t prop);

WEAK void __csi_detach_continue(const csi_id_t detach_continue_id,
                                const csi_id_t detach_id);

WEAK void __csi_before_sync(const csi_id_t sync_id, const int32_t *has_spawned);
WEAK void __csi_after_sync(const csi_id_t sync_id, const int32_t *has_spawned);

///-----------------------------------------------------------------------------
/// Hooks for memory allocation
WEAK void __csi_before_alloca(const csi_id_t alloca_id, uint64_t num_bytes,
                              const alloca_prop_t prop);

WEAK void __csi_after_alloca(const csi_id_t alloca_id, const void *addr,
                             uint64_t num_bytes, const alloca_prop_t prop);

WEAK void __csi_before_allocfn(const csi_id_t allocfn_id, uint64_t size,
                               uint64_t num, uint64_t alignment,
                               const void *oldaddr, const allocfn_prop_t prop);

WEAK void __csi_after_allocfn(const csi_id_t alloca_id, const void *addr,
                              uint64_t size, uint64_t num, uint64_t alignment,
                              const void *oldaddr, const allocfn_prop_t prop);

WEAK void __csi_before_free(const csi_id_t free_id, const void *ptr,
                            const free_prop_t prop);

WEAK void __csi_after_free(const csi_id_t free_id, const void *ptr,
                           const free_prop_t prop);

// This struct is mirrored in ComprehensiveStaticInstrumentation.cpp,
// FrontEndDataTable::getSourceLocStructType.
typedef struct {
  char *name;
  // TODO(ddoucet): Why is this 32 bits?
  int32_t line_number;
  int32_t column_number;
  char *filename;
} source_loc_t;

typedef struct {
  int32_t full_ir_size;
  int32_t non_empty_size;
} sizeinfo_t;

// Front-end data (FED) table accessors.
// Front-end data (FED) table accessors.  All such accessors should look like
// accesses to constant data.
__attribute__((const))
const source_loc_t *__csi_get_func_source_loc(const csi_id_t func_id);
__attribute__((const))
const source_loc_t *__csi_get_func_exit_source_loc(const csi_id_t func_exit_id);
__attribute__((const))
const source_loc_t *__csi_get_loop_source_loc(const csi_id_t loop_id);
__attribute__((const))
const source_loc_t *__csi_get_loop_exit_source_loc(const csi_id_t loop_exit_id);
__attribute__((const))
const source_loc_t *__csi_get_bb_source_loc(const csi_id_t bb_id);
__attribute__((const))
const source_loc_t *__csi_get_callsite_source_loc(const csi_id_t call_id);
__attribute__((const))
const source_loc_t *__csi_get_load_source_loc(const csi_id_t load_id);
__attribute__((const))
const source_loc_t *__csi_get_store_source_loc(const csi_id_t store_id);
__attribute__((const))
const source_loc_t *__csi_get_detach_source_loc(const csi_id_t detach_id);
__attribute__((const))
const source_loc_t *__csi_get_task_source_loc(const csi_id_t task_id);
__attribute__((const))
const source_loc_t *__csi_get_task_exit_source_loc(const csi_id_t task_exit_id);
__attribute__((const))
const source_loc_t *
__csi_get_detach_continue_source_loc(const csi_id_t detach_continue_id);
__attribute__((const))
const source_loc_t *__csi_get_sync_source_loc(const csi_id_t sync_id);
__attribute__((const))
const source_loc_t *__csi_get_alloca_source_loc(const csi_id_t alloca_id);
__attribute__((const))
const source_loc_t *__csi_get_allocfn_source_loc(const csi_id_t allocfn_id);
__attribute__((const))
const source_loc_t *__csi_get_free_source_loc(const csi_id_t free_id);
__attribute__((const))
const sizeinfo_t *__csi_get_bb_sizeinfo(const csi_id_t bb_id);

__attribute__((const))
const char *__csan_get_allocfn_str(const allocfn_prop_t prop);
__attribute__((const))
const char *__csan_get_free_str(const free_prop_t prop);

EXTERN_C_END
