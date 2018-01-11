#ifndef INCLUDED_CONTEXT_STACK_H
#define INCLUDED_CONTEXT_STACK_H

#include <stdbool.h>
#include <time.h>
/* #define _POSIX_C_SOURCE 200112L */

/* Enum for types of functions */
typedef enum {
  MAIN,
  SPAWN,
  HELPER,
} cilk_function_type;

/* Type for a context stack frame */
typedef struct context_stack_frame_t {
  /* Function type */
  cilk_function_type func_type;

  /* Height of the function */
  int32_t height;

  /* Return address of this function */
  void* rip;

  /* Pointer to the frame's parent */
  struct context_stack_frame_t *parent;

  /* Span of the prefix of this function */
  uint64_t prefix_spn;
  /* Data associated with the function's prefix */
  void* prefix_data;

  /* Span of the longest spawned child of this function observed so
     far */
  uint64_t lchild_spn;
  /* Data associated with the function's longest child */
  void* lchild_data;

  /* Span of the continuation of the function since the spawn of its
     longest child */
  uint64_t contin_spn;
  /* Data associated with the function's continuation */
  void* contin_data;

} context_stack_frame_t;

/* Initializes the context stack frame *frame */
void context_stack_frame_init(context_stack_frame_t *frame, cilk_function_type func_type)
{
  frame->parent = NULL;
  frame->func_type = func_type;
  frame->rip = __builtin_extract_return_addr(__builtin_return_address(0));
  frame->height = 0;

  frame->prefix_spn = 0;
  frame->prefix_data = NULL;
  frame->lchild_spn = 0;
  frame->lchild_data = NULL;
  frame->contin_spn = 0;
  frame->contin_data = NULL;
}

/* Type for a context stack */
typedef struct {
  /* Flag to indicate whether user code is being executed.  This flag
     is mostly used for debugging. */
  bool in_user_code;

  /* Start and stop timers for measuring the execution time of a
     strand. */
  struct timespec start;
  struct timespec stop;

  /* Pointer to bottom of the stack, onto which frames are pushed. */
  context_stack_frame_t *bot;

  /* Running total of work. */
  uint64_t running_wrk;

  /* Data associated with the running work */
  void* running_wrk_data;

} context_stack_t;

/* Initializes the context stack */
void context_stack_init(context_stack_t *stack, cilk_function_type func_type)
{
  context_stack_frame_t *new_frame =
    (context_stack_frame_t *)malloc(sizeof(context_stack_frame_t));
  context_stack_frame_init(new_frame, func_type);
  stack->bot = new_frame;
  stack->running_wrk = 0;
  stack->running_wrk_data = NULL;
  stack->in_user_code = false;
}

/* Push new frame of function type func_type onto the stack *stack */
context_stack_frame_t* context_stack_push(context_stack_t *stack, cilk_function_type func_type)
{
  context_stack_frame_t *new_frame
    = (context_stack_frame_t *)malloc(sizeof(context_stack_frame_t));
  context_stack_frame_init(new_frame, func_type);
  new_frame->parent = stack->bot;
  stack->bot = new_frame;

  return new_frame;
}

/* Pops the bottommost frame off of the stack *stack, and returns a
   pointer to it. */
context_stack_frame_t* context_stack_pop(context_stack_t *stack)
{
  context_stack_frame_t *old_bottom = stack->bot;
  stack->bot = stack->bot->parent;
  if (stack->bot->height < old_bottom->height + 1) {
    stack->bot->height = old_bottom->height + 1;
  }

  return old_bottom;
}

#endif
