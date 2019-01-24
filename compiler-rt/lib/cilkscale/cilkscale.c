#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

/* #include <cilk/common.h> */
/* #include <internal/abi.h> */

#include <csi/csi.h>
#include "context_stack.h"

#define CILKTOOL_API __attribute__((visibility("default")))

#ifndef SERIAL_TOOL
#define SERIAL_TOOL 1
#endif

#ifndef TRACE_CALLS
#define TRACE_CALLS 0
#endif

#if !SERIAL_TOOL
#include <cilk/reducer.h>
#include "context_stack_reducer.h"
#endif

/*************************************************************************/
/**
 * Data structures for tracking work and span.
 */
#if SERIAL_TOOL
context_stack_t ctx_stack;
#else
CILK_C_DECLARE_REDUCER(context_stack_t) ctx_stack =
  CILK_C_INIT_REDUCER(context_stack_t,
		      reduce_context_stack,
		      identity_context_stack,
		      destroy_context_stack,
		      {NULL});
#endif

bool TOOL_INITIALIZED = false;

/*************************************************************************/

#if SERIAL_TOOL
// Ensure that this tool is run serially
static inline void ensure_serial_tool(void) {
  // assert(1 == __cilkrts_get_nworkers());
  fprintf(stderr, "Forcing CILK_NWORKERS=1.\n");
  char *e = getenv("CILK_NWORKERS");
  if (!e || 0!=strcmp(e, "1")) {
    // fprintf(err_io, "Setting CILK_NWORKERS to be 1\n");
    if( setenv("CILK_NWORKERS", "1", 1) ) {
      fprintf(stderr, "Error setting CILK_NWORKERS to be 1\n");
      exit(1);
    }
  }
}
#endif

void print_analysis(void) {
  assert(TOOL_INITIALIZED);
#if SERIAL_TOOL
  assert(NULL != ctx_stack.bot);

  uint64_t span = ctx_stack.bot->prefix_spn + ctx_stack.bot->contin_spn;
  uint64_t work = ctx_stack.running_wrk;
#else
  assert(MAIN == REDUCER_VIEW(ctx_stack).bot->func_type);
  assert(NULL != REDUCER_VIEW(ctx_stack).bot);

  uint64_t span = REDUCER_VIEW(ctx_stack).bot->prefix_spn + REDUCER_VIEW(ctx_stack).bot->contin_spn;
  uint64_t work = REDUCER_VIEW(ctx_stack).running_wrk;
#endif

  fprintf(stderr, "work ");  printtime(work);
  fprintf(stderr, ", span ");  printtime(span);
  fprintf(stderr, ", parallelism %f\n", work / (double)span);
}

void cilkscale_destroy(void) {
#if SERIAL_TOOL
  gettime(&(ctx_stack.stop));
#else
  gettime(&(REDUCER_VIEW(ctx_stack).stop));
#endif
#if TRACE_CALLS
  fprintf(stderr, "cilkscale_destroy()\n");
#endif

  print_analysis();

#if SERIAL_TOOL
#else
  CILK_C_UNREGISTER_REDUCER(ctx_stack);
#endif
  TOOL_INITIALIZED = false;
}

CILKTOOL_API void __csi_init() {
#if TRACE_CALLS
  fprintf(stderr, "__csi_init()\n");
#endif

  atexit(cilkscale_destroy);

  TOOL_INITIALIZED = true;

#if SERIAL_TOOL
  ensure_serial_tool();

  context_stack_init(&ctx_stack, MAIN);

  ctx_stack.in_user_code = true;

  gettime(&(ctx_stack.start));
#else
  context_stack_init(&(REDUCER_VIEW(ctx_stack)), MAIN);

  CILK_C_REGISTER_REDUCER(ctx_stack);

  REDUCER_VIEW(ctx_stack).in_user_code = true;

  gettime(&(REDUCER_VIEW(ctx_stack).start));
#endif
}

CILKTOOL_API void __csi_unit_init(const char *const file_name,
                                  const instrumentation_counts_t counts) {
  return;
}

/*************************************************************************/
/**
 * Hooks into runtime system.
 */

CILKTOOL_API
void __csi_func_entry(const csi_id_t func_id, const func_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_func_exit(const csi_id_t func_exit_id, const csi_id_t func_id,
                      const func_exit_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_bb_entry(const csi_id_t bb_id, const bb_prop_t prop) {
  context_stack_t *stack;

#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif  // SERIAL_TOOL

  get_bb_time(&(stack->running_wrk), &(stack->bot->contin_spn), bb_id);
  return;
}

CILKTOOL_API
void __csi_bb_exit(const csi_id_t bb_id, const bb_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_before_call(const csi_id_t call_id, const csi_id_t func_id,
                       const call_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_after_call(const csi_id_t call_id, const csi_id_t func_id,
                      const call_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_before_load(const csi_id_t load_id, const void *addr, int32_t size,
                       load_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_after_load(const csi_id_t load_id, const void *addr, int32_t size,
                       load_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_before_store(const csi_id_t store_id, const void *addr, int32_t size,
                        store_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_after_store(const csi_id_t store_id, const void *addr, int32_t size,
                       store_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_before_alloca(const csi_id_t alloca_id, uint64_t num_bytes,
                         const alloca_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_after_alloca(const csi_id_t alloca_id, const void *addr,
                        uint64_t num_bytes, const alloca_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_before_allocfn(const csi_id_t allocfn_id,
                          uint64_t size, uint64_t num, uint64_t alignment,
                          const void *oldaddr, const allocfn_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_after_allocfn(const csi_id_t allocfn_id, const void *addr,
                         uint64_t size, uint64_t num, uint64_t alignment,
                         const void *oldaddr, const allocfn_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_before_free(const csi_id_t free_id, const void *ptr,
                       const free_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_after_free(const csi_id_t free_id, const void *ptr,
                      const free_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_detach(const csi_id_t detach_id) {
  context_stack_t *stack;

#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif
  gettime(&(stack->stop));

#if TRACE_CALLS
  fprintf(stderr, "detach(%ld)\n", detach_id);
#endif

  uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start));
  stack->running_wrk += strand_time;
  stack->bot->contin_spn += strand_time;
}

CILKTOOL_API
void __csi_task(const csi_id_t task_id, const csi_id_t detach_id) {
  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif

  assert(NULL != stack->bot);

  /* Push new frame onto the stack */
  context_stack_push(stack, HELPER);
  gettime(&(stack->start));
}

CILKTOOL_API
void __csi_task_exit(const csi_id_t task_exit_id,
                     const csi_id_t task_id,
                     const csi_id_t detach_id) {
  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif
  gettime(&(stack->stop));

  uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start));
  stack->running_wrk += strand_time;
  stack->bot->contin_spn += strand_time;

  context_stack_frame_t *old_bottom;

#if TRACE_CALLS
  fprintf(stderr, "task_exit(%ld, %ld, %ld)\n",
          task_exit_id, task_id, detach_id);
#endif

  assert(0 == stack->bot->lchild_spn);
  stack->bot->prefix_spn += stack->bot->contin_spn;

  /* Pop the stack */
  old_bottom = context_stack_pop(stack);
  if (stack->bot->contin_spn + old_bottom->prefix_spn > stack->bot->lchild_spn) {
    stack->bot->prefix_spn += stack->bot->contin_spn;
    stack->bot->lchild_spn = old_bottom->prefix_spn;
    stack->bot->contin_spn = 0;
  }

  free(old_bottom);
}

CILKTOOL_API
void __csi_detach_continue(const csi_id_t detach_continue_id,
                           const csi_id_t detach_id) {
  // In the continuation
#if TRACE_CALLS
  fprintf(stderr, "detach_continue(%ld, %ld)\n",
          detach_continue_id, detach_id);
#endif

  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif

  gettime(&(stack->start));
}

CILKTOOL_API
void __csi_before_sync(const csi_id_t sync_id) {
  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif
  gettime(&(stack->stop));

  uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start));
  stack->running_wrk += strand_time;
  stack->bot->contin_spn += strand_time;

  if (stack->bot->lchild_spn > stack->bot->contin_spn) {
    stack->bot->prefix_spn += stack->bot->lchild_spn;
  } else {
    stack->bot->prefix_spn += stack->bot->contin_spn;
  }
  stack->bot->lchild_spn = 0;
  stack->bot->contin_spn = 0;
}

CILKTOOL_API
void __csi_after_sync(const csi_id_t sync_id) {
  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif

  gettime(&(stack->start));
}
