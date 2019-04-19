#ifndef INCLUDED_CONTEXT_STACK_H
#define INCLUDED_CONTEXT_STACK_H

#include <stdbool.h>
#include "cilkscale_timer.h"

#ifndef DEFAULT_STACK_SIZE
#define DEFAULT_STACK_SIZE 64
#endif

/* Enum for types of functions */
typedef enum {
  MAIN,
  SPAWN,
  HELPER,
} cilk_function_type;

/* Type for a context stack frame */
typedef struct context_stack_frame_t {
  /* Span of the prefix of this function */
  uint64_t prefix_spn;

  /* Span of the longest spawned child of this function observed so
     far */
  uint64_t lchild_spn;

  /* Span of the continuation of the function since the spawn of its
     longest child */
  uint64_t contin_spn;

  /* Function type */
  cilk_function_type func_type;

  /* /\* Height of the function *\/ */
  /* int32_t height; */
} context_stack_frame_t;

/* Initializes the context stack frame *frame */
void context_stack_frame_init(context_stack_frame_t *frame,
                              cilk_function_type func_type)
{
  frame->func_type = func_type;
  /* frame->height = 0; */

  frame->prefix_spn = 0;
  frame->lchild_spn = 0;
  frame->contin_spn = 0;
}

/* Type for a context stack */
typedef struct {
  /* Start and stop timers for measuring the execution time of a
     strand. */
  cilkscale_timer_t start;
  cilkscale_timer_t stop;

  /* Running total of work. */
  uint64_t running_wrk;

  /* Dynamic array of context-stack frames. */
  context_stack_frame_t *frames;
  int32_t capacity;

  /* Index of the context-stack frame for the function/task frame at the bottom
     of the stack. */
  int32_t bot;
} context_stack_t;

/* Returns a pointer to the bottommost frame of *stack. */
context_stack_frame_t *context_stack_bot(context_stack_t *stack)
{
  return &stack->frames[stack->bot];
}

/* Initializes the context stack */
void context_stack_init(context_stack_t *stack, cilk_function_type func_type)
{
  stack->frames =
    (context_stack_frame_t *)malloc(sizeof(context_stack_frame_t) *
                                    DEFAULT_STACK_SIZE);
  stack->capacity = DEFAULT_STACK_SIZE;
  stack->bot = 0;
  context_stack_frame_t *new_frame = context_stack_bot(stack);
  context_stack_frame_init(new_frame, func_type);
  stack->running_wrk = 0;
}

/* Push new frame of function type func_type onto *stack. */
context_stack_frame_t *context_stack_push(context_stack_t *stack,
                                          cilk_function_type func_type)
{
  stack->bot++;
  // Resize stack->frames if necessary
  if (stack->bot >= stack->capacity) {
    stack->capacity *= 2;
    stack->frames =
      (context_stack_frame_t *)realloc(
          stack->frames, sizeof(context_stack_frame_t) * stack->capacity);
  }
  context_stack_frame_t *new_frame = context_stack_bot(stack);
  context_stack_frame_init(new_frame, func_type);

  return new_frame;
}

/* Pops the bottommost frame off of *stack, and returns a pointer to it. */
context_stack_frame_t *context_stack_pop(context_stack_t *stack)
{
  assert(stack->bot > 0);
  context_stack_frame_t *old_bottom = context_stack_bot(stack);
  stack->bot--;
  /* if (stack->bot->height < old_bottom->height + 1) { */
  /*   stack->bot->height = old_bottom->height + 1; */
  /* } */

  return old_bottom;
}

#endif
