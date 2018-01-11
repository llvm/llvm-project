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
  context_stack_init((context_stack_t*)view, SPAWN);
}

/* Reduce method for context stack reducer */
void reduce_context_stack(void *reducer, void *l, void *r)
{
  context_stack_t *left = (context_stack_t*)l;
  context_stack_t *right = (context_stack_t*)r;

  assert(NULL == right->bot->parent);
  assert(SPAWN == right->bot->func_type);
  assert(right->bot->func_type == left->bot->func_type);

  assert(!(left->in_user_code));
  assert(!(right->in_user_code));

  /* height is maintained as a max reducer */
  if (right->bot->height > left->bot->height) {
    left->bot->height = right->bot->height;
  }
  /* running_wrk is maintained as a sum reducer */
  left->running_wrk += right->running_wrk;

  /* assert(0 == left->bot->contin_spn); */

  if (left->bot->contin_spn + right->bot->prefix_spn + right->bot->lchild_spn
      > left->bot->lchild_spn) {
    left->bot->prefix_spn += left->bot->contin_spn + right->bot->prefix_spn;
    left->bot->lchild_spn = right->bot->lchild_spn;
    left->bot->contin_spn = right->bot->contin_spn;
  } else {
    left->bot->contin_spn += right->bot->prefix_spn + right->bot->contin_spn;
  }
}

/* Destructor for context stack reducer */
void destroy_context_stack(void *reducer, void *view)
{
  free(((context_stack_t*)view)->bot);
}

#endif
