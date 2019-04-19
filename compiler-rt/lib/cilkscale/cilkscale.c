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
  context_stack_frame_t *bottom = context_stack_bot(&ctx_stack);
  assert(NULL != bottom);

  uint64_t span = bottom->prefix_spn + bottom->contin_spn;
  uint64_t work = ctx_stack.running_wrk;
#else
  context_stack_t *stack = &(REDUCER_VIEW(ctx_stack));
  context_stack_frame_t *bottom = context_stack_bot(stack);
  assert(NULL != bottom);

  uint64_t span = bottom->prefix_spn + bottom->contin_spn;
  uint64_t work = stack->running_wrk;
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
  free(ctx_stack.frames);
#else
  free(REDUCER_VIEW(ctx_stack).frames);
#endif

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

  gettime(&(ctx_stack.start));
#else
  context_stack_init(&(REDUCER_VIEW(ctx_stack)), MAIN);

  CILK_C_REGISTER_REDUCER(ctx_stack);

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
void __csi_bb_entry(const csi_id_t bb_id, const bb_prop_t prop) {
  if (!TOOL_INITIALIZED)
    return;

  context_stack_t *stack;

#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif  // SERIAL_TOOL

  get_bb_time(&(stack->running_wrk), &(context_stack_bot(stack)->contin_spn),
              bb_id);
  return;
}

CILKTOOL_API
void __csi_bb_exit(const csi_id_t bb_id, const bb_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_func_entry(const csi_id_t func_id, const func_prop_t prop) {
  if (!TOOL_INITIALIZED)
    return;

  context_stack_t *stack;

#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif
  gettime(&(stack->stop));

#if TRACE_CALLS
  fprintf(stderr, "func_entry(%ld)\n", func_id);
#endif

  uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start));
  stack->running_wrk += strand_time;
  context_stack_bot(stack)->contin_spn += strand_time;

  /* Push new frame onto the stack */
  context_stack_push(stack, SPAWN);
  // gettime(&(stack->start));
  // Because of the high overhead of calling gettime(), especially compared to
  // the running time of the operations in this hook, the work and span
  // measurements appear more stable if we simply use the recorded time as the
  // new start time.
  settime(&(stack->start), stack->stop);
}

CILKTOOL_API
void __csi_func_exit(const csi_id_t func_exit_id, const csi_id_t func_id,
                     const func_exit_prop_t prop) {
  if (!TOOL_INITIALIZED)
    return;

  context_stack_t *stack;

#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif
  gettime(&(stack->stop));

#if TRACE_CALLS
  fprintf(stderr, "func_exit(%ld)\n", func_id);
#endif

  uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start));
  stack->running_wrk += strand_time;

  assert(0 == context_stack_bot(stack)->lchild_spn);

  context_stack_frame_t *old_bottom, *new_bottom;
  // Pop the stack
  old_bottom = context_stack_pop(stack);
  new_bottom = context_stack_bot(stack);

  new_bottom->contin_spn +=
    old_bottom->prefix_spn + old_bottom->contin_spn + strand_time;
  /* gettime(&(stack->start)); */
  // Because of the high overhead of calling gettime(), especially compared to
  // the running time of the operations in this hook, the work and span
  // measurements appear more stable if we simply use the recorded time as the
  // new start time.
  settime(&(stack->start), stack->stop);
}

CILKTOOL_API
void __csi_detach(const csi_id_t detach_id, const int32_t *has_spawned) {
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
  context_stack_bot(stack)->contin_spn += strand_time;
}

CILKTOOL_API
void __csi_task(const csi_id_t task_id, const csi_id_t detach_id) {
  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif

#if TRACE_CALLS
  fprintf(stderr, "task(%ld, %ld)\n", task_id, detach_id);
#endif

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
  context_stack_bot(stack)->contin_spn += strand_time;

  context_stack_frame_t *old_bottom, *new_bottom;

#if TRACE_CALLS
  fprintf(stderr, "task_exit(%ld, %ld, %ld)\n",
          task_exit_id, task_id, detach_id);
#endif

  old_bottom = context_stack_bot(stack);
  assert(0 == old_bottom->lchild_spn);
  old_bottom->prefix_spn += old_bottom->contin_spn;

  /* Pop the stack */
  old_bottom = context_stack_pop(stack);
  new_bottom = context_stack_bot(stack);
  if (new_bottom->contin_spn + old_bottom->prefix_spn >
      new_bottom->lchild_spn) {
    new_bottom->prefix_spn += new_bottom->contin_spn;
    new_bottom->lchild_spn = old_bottom->prefix_spn;
    new_bottom->contin_spn = 0;
  }
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
void __csi_before_sync(const csi_id_t sync_id, const int32_t *has_spawned) {
  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif
  gettime(&(stack->stop));

#if TRACE_CALLS
  fprintf(stderr, "before_sync(%ld)\n", sync_id);
#endif

  uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start));
  stack->running_wrk += strand_time;
  context_stack_bot(stack)->contin_spn += strand_time;
}

CILKTOOL_API
void __csi_after_sync(const csi_id_t sync_id, const int32_t *has_spawned) {
  context_stack_t *stack;
#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif
#if TRACE_CALLS
  fprintf(stderr, "after_sync(%ld)\n", sync_id);
#endif

  context_stack_frame_t *new_bottom = context_stack_bot(stack);
  if (new_bottom->lchild_spn > new_bottom->contin_spn)
    new_bottom->prefix_spn += new_bottom->lchild_spn;
  else
    new_bottom->prefix_spn += new_bottom->contin_spn;
  new_bottom->lchild_spn = 0;
  new_bottom->contin_spn = 0;

  gettime(&(stack->start));
}
