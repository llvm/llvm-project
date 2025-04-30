/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* llmputil.c: OpenMP utility routines for our LLVM compilers */

#include "llmputil.h"
#include "error.h"
#include "symtab.h"
#include "dtypeutl.h"
#ifndef FE90
#include "iliutil.h"
#endif

/* Global container of uplevel pointers */
static struct {
  LLUplevel *base; /* Pointer to the allocated array of items */
  int size;        /* Total size including unused items */
  int avl;         /* Total items in use */
} llmp_all_uplevels;

/* Global container of task pointers */
static struct {
  LLTask *base; /* Pointer to the allocated array of items */
  int size;     /* Total size including unused items */
  int avl;      /* Total items in use */
} llmp_all_tasks;

static LLUplevel *
get_uplevel(int stblock_sptr)
{
  int key;
  LLUplevel *up;
  assert(STYPEG(stblock_sptr) == ST_BLOCK, "Uplevel key must be an ST_BLOCK",
         stblock_sptr, ERR_Fatal);

  /* Index */
  key = PARSYMSG(stblock_sptr);

  /* Locate uplevel pointer */
  up = NULL;
  if (key <= llmp_all_uplevels.avl)
    up = (LLUplevel *)(&llmp_all_uplevels.base[key]);

  assert(up && key, "Could not locate uplevel instance for stblock",
         stblock_sptr, ERR_Fatal);

  return up;
}

LLUplevel *
llmp_has_uplevel(int stblock_sptr)
{
  int key;
  LLUplevel *up;
  assert(STYPEG(stblock_sptr) == ST_BLOCK, "Uplevel key must be an ST_BLOCK",
         stblock_sptr, ERR_Fatal);

  /* Index */
  key = PARSYMSG(stblock_sptr);

  /* Locate uplevel pointer */
  up = NULL;
  if (key && key <= llmp_all_uplevels.avl)
    up = (LLUplevel *)(&llmp_all_uplevels.base[key]);

  return up;
}

LLUplevel *
llmp_create_uplevel(int stblock_sptr)
{
  int key;
  LLUplevel *up;

  assert(STYPEG(stblock_sptr) == ST_BLOCK, "Uplevel key must be an ST_BLOCK",
         stblock_sptr, ERR_Fatal);

  /* Avoid processing an already created uplevel */
  if (PARSYMSG(stblock_sptr))
    return get_uplevel(stblock_sptr);

  /* Make room if necessary */
  if (llmp_all_uplevels.avl == 0) {
    llmp_all_uplevels.avl = 2;
    key = 1;
  } else {
    key = llmp_all_uplevels.avl;
    ++llmp_all_uplevels.avl;
  }
  NEED(llmp_all_uplevels.avl, llmp_all_uplevels.base, LLUplevel,
       llmp_all_uplevels.size, llmp_all_uplevels.size + 8);

  up = (LLUplevel *)(&llmp_all_uplevels.base[key]);
  memset(up, 0, sizeof(LLUplevel));

  /* Add key and map it to stblock */
  PARSYMSP(stblock_sptr, key);

  return up;
}

LLUplevel *
llmp_get_uplevel(int stblock_sptr)
{
  return get_uplevel(stblock_sptr);
}

void
llmp_uplevel_set_dtype(LLUplevel *up, DTYPE dtype)
{
  up->dtype = dtype;
}

void
llmp_uplevel_set_parent(SPTR stblock_sptr, SPTR parent_sptr)
{
  LLUplevel *up = llmp_create_uplevel(stblock_sptr);
  up->parent = parent_sptr;
  if (llmp_has_uplevel(parent_sptr) == NULL)
    up = llmp_create_uplevel(parent_sptr);
}

/* Uniquely add shared_sptr to up */
int
llmp_add_shared_var(LLUplevel *up, int shared_sptr)
{
  int i;
  const int idx = up->vals_count;

  /* Unique add: I really wanted to make this a hashset... */
  for (i = 0; i < up->vals_count; ++i) {
    if (shared_sptr == 0)
      break;
    if (up->vals[i] == shared_sptr)
      return 0;
  }

  ++up->vals_count;
  NEED(up->vals_count, up->vals, int, up->vals_size, up->vals_size + 8);
  up->vals[idx] = shared_sptr;
  return 1;
}

/* add 0 as placeholder for character len sptr for shared_sptr */
void
llmp_add_shared_var_charlen(LLUplevel *up, int shared_sptr)
{
  int i;
  const int idx = up->vals_count;

  /* Unique add: I really wanted to make this a hashset... */
  for (i = 0; i < up->vals_count; ++i)
    if (up->vals[i] == shared_sptr) {
      ++up->vals_count;
      NEED(up->vals_count, up->vals, int, up->vals_size, up->vals_size + 8);
      up->vals[idx] = 0;
    }
}

/* Return a new key (index) into our table of all uplevels */
int
llmp_get_next_key(void)
{
  int key;
  if (llmp_all_uplevels.avl == 0) {
    llmp_all_uplevels.avl = 2;
    key = 1;
  } else {
    key = llmp_all_uplevels.avl;
    ++llmp_all_uplevels.avl;
  }
  NEED(llmp_all_uplevels.avl, llmp_all_uplevels.base, LLUplevel,
       llmp_all_uplevels.size, llmp_all_uplevels.size + 8);
  return key;
}

/* Return the uplevel for a specific key (index into our table of uplevels) */
LLUplevel *
llmp_create_uplevel_bykey(int key)
{
  LLUplevel *up;

  assert(key <= llmp_all_uplevels.avl, "Invalid uplevel key", key, ERR_Fatal);

  up = (LLUplevel *)(&llmp_all_uplevels.base[key]);
  memset(up, 0, sizeof(LLUplevel));

  return up;
}

void
llmp_reset_uplevel(void)
{
  if (llmp_all_uplevels.avl) {
    for (int i = 1; i < llmp_all_uplevels.avl; ++i) {
      LLUplevel *up = (LLUplevel *)(&llmp_all_uplevels.base[i]);
      if (up->vals_count)
        FREE(up->vals);
    }
    FREE(llmp_all_uplevels.base);
    memset(&llmp_all_uplevels, 0, sizeof(llmp_all_uplevels));
  }
  if (llmp_all_tasks.avl) {
    for (int i = 0; llmp_all_tasks.avl; ++i) {
      LLTask *task = (LLTask *)(&llmp_all_tasks.base[i]);
      if (task->privs_count) {
        FREE(task->privs);
      }
      FREE(llmp_all_tasks.base);
      memset(&llmp_all_tasks, 0, sizeof(llmp_all_tasks));
    }
  }
  llmp_all_uplevels.avl = 0;
  llmp_all_tasks.avl = 0;
}

LLUplevel *
llmp_outermost_uplevel(SPTR child)
{
  LLUplevel *up = get_uplevel(child);
  while (up->parent) {
    up = llmp_get_uplevel(up->parent);
  }
  return up;
}

LLUplevel *
llmp_parent_uplevel(SPTR child)
{
  LLUplevel *up = get_uplevel(child);
  if (up->parent) {
    up = llmp_get_uplevel(up->parent);
  } else {
    up = NULL;
  }
  return up;
}

SPTR
llmp_get_parent_sptr(SPTR child)
{
  LLUplevel *up = get_uplevel(child);
  return up->parent;
}

LLTask *
llmp_get_task(int scope_sptr)
{
  for (int i = 0; i < llmp_all_tasks.avl; ++i) {
    LLTask *task = (LLTask *)&llmp_all_tasks.base[i];
    if (task->scope_sptr == scope_sptr)
      return task;
  }
  return NULL;
}

LLTask *
llmp_create_task(int scope_sptr)
{
  NEED(llmp_all_tasks.avl + 1, llmp_all_tasks.base, LLTask, llmp_all_tasks.size,
       llmp_all_tasks.size + 4);

  LLTask *task = (LLTask *)(&llmp_all_tasks.base[llmp_all_tasks.avl]);
  ++llmp_all_tasks.avl;
  memset(task, 0, sizeof(LLTask));
  task->actual_size = llmp_task_get_base_task_size();
  task->scope_sptr = scope_sptr;
  return task;
}

/* Return the size of an empty KMPC task (no shared variables):
 * Pointer + Pointer + int32(+pad) +
 * kmp_cmplrdata_t(data1) + kmp_cmplrdata_t(data2)
 * see kmp.h
 */
int
llmp_task_get_base_task_size(void)
{
  int pad = sizeof(void *) - sizeof(int);
#ifdef TARGET_WIN
  return sizeof(void *) + sizeof(void *) + sizeof(int) + pad +
         sizeof(void *) * 2;
#else
  return sizeof(void *) + sizeof(void *) + sizeof(int32_t) + pad +
         sizeof(void *) * 2;
#endif
}

/* Return the size of a KMPC equivalent task (base + size of privates) */
int
llmp_task_get_size(LLTask *task)
{
  return task->actual_size;
}

/* Set the fnsptr that belongs to the outlined task */
void
llmp_task_set_fnsptr(LLTask *task, int task_sptr)
{
  task->task_sptr = task_sptr;
}

/* Return the task object associated with 'task_sptr' */
LLTask *
llmp_task_get_by_fnsptr(int task_sptr)
{
  for (int i = 0; i < llmp_all_tasks.avl; ++i) {
    LLTask *task = (LLTask *)&llmp_all_tasks.base[i];
    if (task->task_sptr == task_sptr) {
      return task;
    }
  }
  return NULL;
}

int
llmp_task_add_private(LLTask *task, int shared_sptr, SPTR private_sptr)
{
#ifndef FE90
  int pad = 0;
  int size;
  int align;
  DTYPE dtype;
#endif
  int offset = 0;
  LLFirstPrivate *fp;
  int idx = task->privs_count;

  NEED(++task->privs_count, task->privs, LLFirstPrivate, task->privs_size,
       task->privs_size + 4);

  /* Create the private object */
  fp = (LLFirstPrivate *)&(task->privs[idx]);
  fp->private_sptr = private_sptr;
  fp->shared_sptr = shared_sptr;

/* Bump up the size of the task to contain private_sptr */
#ifdef FE90
  task->actual_size += size_of_var(private_sptr);
#else
  dtype = DTYPEG(private_sptr);
  if (dtype) {
    size = zsize_of(dtype);
    align = alignment(dtype);
    pad = ALIGN(task->actual_size, align) - task->actual_size;
    task->actual_size += pad;
  }
  offset = task->actual_size;
  task->actual_size += size_of_sym(private_sptr);
#endif
  return offset;
}

int
llmp_task_add_loopvar(LLTask *task, int num, DTYPE dtype)
/* put loop variables on task_alloc array after private vars */
{
#ifndef FE90
  int pad = 0;
  int size;
  int align;
#endif
  int offset = 0;
#ifdef FE90
  /* we add it to backend only */
#else
  /* Bump up the size of the task to contain loop var and make sure
   * it is integer*64 aligned.
   */
  size = zsize_of(dtype) * num;
  align = alignment(dtype);
  pad = ALIGN(task->actual_size, align) - task->actual_size;
  task->actual_size += pad;
  offset = task->actual_size;
  task->actual_size += size;
#endif
  return offset;
}

void
llmp_task_add(int scope_sptr, int shared_sptr, SPTR private_sptr)
{
  LLTask *task;
  assert(scope_sptr && STYPEG(scope_sptr) == ST_BLOCK,
         "Task key must be a scope sptr (ST_BLOCK)", scope_sptr, ERR_Fatal);

  task = llmp_get_task(scope_sptr);
  if (!task)
    task = llmp_create_task(scope_sptr);
  llmp_task_add_private(task, shared_sptr, private_sptr);
}

int
llmp_task_get_private(const LLTask *task, int sptr, int encl)
{
  for (int i = 0; i < task->privs_count; ++i) {
    const int pr = task->privs[i].private_sptr;
    if (sptr == pr && TASKG(sptr)
#ifndef FE90
        && is_llvm_local_private(sptr)
#endif
    )
      return pr;
  }

  return 0;
}

/* should call in taskdup only */
INT
llmp_task_get_privoff(int sptr, const LLTask *task)
{
  for (int i = 0; i < task->privs_count; ++i) {
    const int pr = task->privs[i].shared_sptr;
    if (sptr == pr)
      return ADDRESSG(task->privs[i].private_sptr);
  }
  return 0;
}

void
llmp_concur_add_shared_var(int stblock_sptr, int shared_sptr)
{
  LLUplevel *up = llmp_create_uplevel(stblock_sptr);
  (void)llmp_add_shared_var(up, shared_sptr);
}

bool is_omp_mode_target(OMP_TARGET_MODE mode) {
  switch (mode)
  {
    case mode_target:
    case mode_target_teams:
    case mode_target_teams_distribute:
    case mode_target_teams_distribute_simd:
    case mode_target_teams_distribute_parallel_for:
    case mode_target_teams_distribute_parallel_for_simd:
    case mode_target_parallel:
    case mode_target_parallel_for:
    case mode_target_parallel_for_simd:
    case mode_target_simd:
      return true;
    default:
      return false;
  }
}
