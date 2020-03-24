#ifndef INCLUDED_CONTEXT_STACK_REDUCER_H
#define INCLUDED_CONTEXT_STACK_REDUCER_H

#include <stdlib.h>
#include <assert.h>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer.h>

#include "context_stack.h"

/* Identity method for context stack reducer */
void identity_context_stack(void *reducer, void *view)
{
  context_stack_init((context_stack_t *)view, SPAWN);
#if TRACE_CALLS
  fprintf(stderr, "created new reducer view.\n");
#endif
  gettime(&((context_stack_t *)view)->start);
}

/* Reduce method for context stack reducer */
void reduce_context_stack(void *reducer, void *l, void *r)
{
  context_stack_t *left = (context_stack_t *)l;
  context_stack_t *right = (context_stack_t *)r;
  context_stack_frame_t *l_bot = context_stack_bot(left);
  context_stack_frame_t *r_bot = context_stack_bot(right);

  assert(SPAWN == r_bot->func_type);
  assert(r_bot->func_type == l_bot->func_type);

#if TRACE_CALLS
  fprintf(stderr, "left wrk = %ld\right work = %ld\n",
          left->running_wrk, right->running_wrk);
  fprintf(stderr, "left contin = %ld\nleft child = %ld\n"
          "right prefix = %ld\nright child = %ld\nright contin = %ld\n",
          l_bot->contin_spn, l_bot->lchild_spn, r_bot->prefix_spn,
          r_bot->lchild_spn, r_bot->contin_spn);
#endif

  /* /\* height is maintained as a max reducer *\/ */
  /* if (right->bot->height > left->bot->height) { */
  /*   left->bot->height = right->bot->height; */
  /* } */

  /* running_wrk is maintained as a sum reducer */
  left->running_wrk += right->running_wrk;

  if (l_bot->contin_spn + r_bot->prefix_spn + r_bot->lchild_spn
      > l_bot->lchild_spn) {
    l_bot->prefix_spn += l_bot->contin_spn + r_bot->prefix_spn;
    l_bot->lchild_spn = r_bot->lchild_spn;
    l_bot->contin_spn = r_bot->contin_spn;
  } else {
    l_bot->contin_spn += r_bot->prefix_spn + r_bot->contin_spn;
  }
}

/* Destructor for context stack reducer */
void destroy_context_stack(void *reducer, void *view)
{
  // free(((context_stack_t*)view)->bot);
  free(((context_stack_t *)view)->frames);
}

#endif
