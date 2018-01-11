#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/* #define _POSIX_C_SOURCE 200112L */
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

#if !SERIAL_TOOL
#include <cilk/reducer.h>
#include "context_stack_reducer.h"
#endif

#ifndef TRACE_CALLS
#define TRACE_CALLS 0
#endif

#ifndef INSTCOUNT
#define INSTCOUNT 1
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
/**
 * Data structures and helper methods for time of user strands.
 */

static inline uint64_t elapsed_nsec(const struct timespec *stop,
			     const struct timespec *start) {
  return (uint64_t)(stop->tv_sec - start->tv_sec) * 1000000000ll
    + (stop->tv_nsec - start->tv_nsec);
}

static inline void gettime(struct timespec *timer) {
#if INSTCOUNT
#else
  // TB 2014-08-01: This is the "clock_gettime" variant I could get
  // working with -std=c11.  I want to use TIME_MONOTONIC instead, but
  // it does not appear to be supported on my system.
  /* timespec_get(timer, TIME_UTC); */
  clock_gettime(CLOCK_MONOTONIC, timer);
#endif
}

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

#if INSTCOUNT
  fprintf(stderr, "work %f MInstructions, span %f MInstructions, parallelism %f\n",
	  work / (1000000.0),
	  span / (1000000.0),
	  work / (double)span);
#else
  fprintf(stderr, "work %fs, span %fs, parallelism %f\n",
	  work / (1000000000.0),
	  span / (1000000000.0),
	  work / (double)span);
#endif
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
#if INSTCOUNT
  context_stack_t *stack;

#if SERIAL_TOOL
  stack = &(ctx_stack);
#else
  stack = &(REDUCER_VIEW(ctx_stack));
#endif  // SERIAL_TOOL
  uint64_t inst_count = __csi_get_bb_sizeinfo(bb_id)->non_empty_size;
  stack->running_wrk += inst_count;
  stack->bot->contin_spn += inst_count;

#endif  // INSTCOUNT
  return;
}

CILKTOOL_API
void __csi_bb_exit(const csi_id_t bb_id, const bb_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_before_call(const csi_id_t call_id,
                       const csi_id_t func_id,
                       const call_prop_t prop) {
  return;
}

CILKTOOL_API
void __csi_after_call(const csi_id_t call_id,
                      const csi_id_t func_id,
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

/* void __csan_func_entry(const csi_id_t func_id, void *sp, */
/*                        const func_prop_t prop) */
/* { */
/*   context_stack_t *stack; */
/*   gettime(&(stack->stop)); */

/* #if TRACE_CALLS */
/*   fprintf(stderr, "cilk_enter_begin(%p, %p, %p)\n", sf, this_fn, rip); */
/* #endif */

/* #if SERIAL_TOOL */
/*   stack = &(ctx_stack); */
/* #else */
/*   stack = &(REDUCER_VIEW(ctx_stack)); */
/* #endif */

/*   assert(NULL != stack->bot); */

/*   if (stack->bot->func_type != HELPER) { */
/* #if TRACE_CALLS */
/*     if (MAIN == stack->bot->func_type) { */
/*       printf("parent is MAIN\n"); */
/*     } else { */
/*       printf("parent is SPAWN\n"); */
/*     } */
/* #endif */
/*     // TB 2014-12-18: This assert won't necessarily pass, if shrink-wrapping has */
/*     // taken place. */
/*     /\* assert(stack->in_user_code); *\/ */
      
/*     uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start)); */
/*     stack->running_wrk += strand_time; */
/*     stack->bot->contin_spn += strand_time; */
      
/*     stack->in_user_code = false; */
/*   } else { */
/*     assert(!(stack->in_user_code)); */
/*   } */

/*   /\* Push new frame onto the stack *\/ */
/*   context_stack_push(stack, SPAWN); */
/*   stack->in_user_code = true; */
/*   gettime(&(stack->start)); */
/* } */

/* void __csan_func_exit(const csi_id_t func_exit_id, */
/*                       const csi_id_t func_id, */
/*                       const func_exit_prop_t prop) { */
/*   context_stack_t *stack; */
/* #if SERIAL_TOOL */
/*   stack = &(ctx_stack); */
/* #else */
/*   stack = &(REDUCER_VIEW(ctx_stack)); */
/* #endif */

/*   context_stack_frame_t *old_bottom; */
  
/*   gettime(&(stack->stop)); */

/*   assert(stack->in_user_code); */
/*   stack->in_user_code = false; */

/*   if (SPAWN == stack->bot->func_type) { */
/* #if TRACE_CALLS */
/*     fprintf(stderr, "cilk_leave_begin(%p) from SPAWN\n", sf); */
/* #endif */
/*     uint64_t strand_time = elapsed_nsec(&(stack->stop), &(stack->start)); */
/*     stack->running_wrk += strand_time; */
/*     stack->bot->contin_spn += strand_time; */
/*     assert(NULL != stack->bot->parent); */

/*     assert(0 == stack->bot->lchild_spn); */
/*     stack->bot->prefix_spn += stack->bot->contin_spn; */

/*     /\* Pop the stack *\/ */
/*     old_bottom = context_stack_pop(stack); */
/*     stack->bot->contin_spn += old_bottom->prefix_spn; */
    
/*   } else { */
/* #if TRACE_CALLS */
/*     fprintf(stderr, "cilk_leave_begin(%p) from HELPER\n", sf); */
/* #endif */

/*     assert(HELPER != stack->bot->parent->func_type); */

/*     assert(0 == stack->bot->lchild_spn); */
/*     stack->bot->prefix_spn += stack->bot->contin_spn; */

/*     /\* Pop the stack *\/ */
/*     old_bottom = context_stack_pop(stack); */
/*     if (stack->bot->contin_spn + old_bottom->prefix_spn > stack->bot->lchild_spn) { */
/*       // fprintf(stderr, "updating longest child\n"); */
/*       stack->bot->prefix_spn += stack->bot->contin_spn; */
/*       stack->bot->lchild_spn = old_bottom->prefix_spn; */
/*       stack->bot->contin_spn = 0; */
/*     } */
    
/*   } */

/*   free(old_bottom); */
/*   gettime(&(stack->start)); */
/* } */

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
void __csi_task(const csi_id_t task_id, const csi_id_t detach_id, void *sp) {
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
    // fprintf(stderr, "updating longest child\n");
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
void __csi_sync(const csi_id_t sync_id) {
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

  gettime(&(stack->start));
}
