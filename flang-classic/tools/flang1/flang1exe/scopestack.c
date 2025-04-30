/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief Manage the scope stack.
*/

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "ccffinfo.h"
#include "semant.h"

static SCOPESTACK *push_scope(void);
static void pop_scope(void);
static const char *kind_to_string(SCOPEKIND kind);

/** \brief Initialize the scope stack.

    It starts with one frame representing the outer scope,
    with no associated symbol.
 */
void
scopestack_init()
{
  SCOPESTACK *scope;
  if (sem.scope_stack == NULL) {
    sem.scope_size = 10;
    NEW(sem.scope_stack, SCOPESTACK, sem.scope_size);
  }
  sem.scope_level = 0;
  sem.next_unnamed_scope = 6; // probably some arbitrary small value > 1 or 2
  scope = curr_scope();
  BZERO(scope, SCOPESTACK, 1);
  scope->kind = SCOPE_OUTER;
  scope->closed = true;
  scope->symavl = stb.stg_avail;
}

/** \brief Return the scope at the top of the scope stack. */
SCOPESTACK *
curr_scope()
{
  return get_scope(0);
}

/** \brief Get an entry in the scope stack.
    \param level Level of the entry to return: 0 means top and negative is
           relative to the top.
 */
SCOPESTACK *
get_scope(int level)
{
  if (level <= 0) {
    level += sem.scope_level;
  }
  if (level < 0 || level >= sem.scope_size) {
#if DEBUG
    dumpscope(gbl.dbgfil);
#endif
    interr("bad scope stack level", level, ERR_Fatal);
  }
  return &sem.scope_stack[level];
}

/** \brief Return the level of this entry in the scope stack
           or -1 if \a scope is null.
 */
int
get_scope_level(SCOPESTACK *scope)
{
  if (scope == 0) {
    return -1;
  } else {
    int level = scope - sem.scope_stack;
    assert(level >= 0 && level <= sem.scope_level, "bad scope stack level",
           level, ERR_Fatal);
    return level;
  }
}

/** \brief Return the next entry below this one in the scope stack; 0 if none.
    If scope is 0, return the top of the stack.
 */
SCOPESTACK *
next_scope(SCOPESTACK *scope)
{
  int sl = get_scope_level(scope);
  if (sl < 0) {
    return curr_scope();
  } else if (sl == 0) {
    return 0;
  } else {
    return scope - 1;
  }
}

/** \brief Return the next entry below scope that has this sptr assocated with
   it.
           If scope is 0, search from the top of the stack.
 */
SCOPESTACK *
next_scope_sptr(SCOPESTACK *scope, int sptr)
{
  while ((scope = next_scope(scope)) != 0) {
    if (scope->sptr == sptr) {
      return scope;
    }
  }
  return 0;
}

/** \brief Return the next entry below scope that has this kind.
           If scope is 0, search from the top of the stack.
 */
SCOPESTACK *
next_scope_kind(SCOPESTACK *scope, SCOPEKIND kind)
{
  while ((scope = next_scope(scope)) != 0) {
    if (scope->kind == kind) {
      return scope;
    }
  }
  return 0;
}

/** \brief Return the next entry below scope that has this kind and sptr.
           If scope is 0, search from the top of the stack.
 */
SCOPESTACK *
next_scope_kind_sptr(SCOPESTACK *scope, SCOPEKIND kind, int sptr)
{
  while ((scope = next_scope_kind(scope, kind)) != 0) {
    if (scope->sptr == sptr) {
      return scope;
    }
  }
  return 0;
}

/** \brief Return the next entry below scope that has this kind and symbol name.
           If scope is 0, search from the top of the stack.
 */
SCOPESTACK *
next_scope_kind_symname(SCOPESTACK *scope, SCOPEKIND kind, const char *symname)
{
  while ((scope = next_scope_kind(scope, kind)) != 0) {
    if (strcmp(symname, SYMNAME(scope->sptr)) == 0) {
      return scope;
    }
  }
  return 0;
}

/** \brief Return the USE module scope for the module associated with this
           symbol, or -1 if none.
 */
int
have_use_scope(int sptr)
{
  SCOPESTACK *scope = 0;
  if (sem.scope_stack == NULL)
    return -1;
  while ((scope = next_scope(scope)) != 0) {
    if (scope->kind == SCOPE_USE && scope->sptr == sptr)
      return get_scope_level(scope);
    if (scope->closed)
      return -1;
  }
  return -1;
}

/** \brief Return TRUE if sptr is in the exception list for this scope
    at this level.
 */
LOGICAL
is_except_in_scope(SCOPESTACK *scope, int sptr)
{
  return sym_in_sym_list(sptr, scope->except);
}

/** \brief Return TRUE if scope is private and has sptr in its 'only' list.
 */
LOGICAL
is_private_in_scope(SCOPESTACK *scope, int sptr)
{
  return scope->Private && !sym_in_sym_list(sptr, scope->only);
}

/** \brief Push a slot on the scope stack.
 *  \param sptr For most scopes, scope symbol.  For interface and parallel
                scopes, a small (unnamed) numeric index.
 *  \param kind Scope kind.
 */
void
push_scope_level(int sptr, SCOPEKIND kind)
{
  SCOPESTACK *scope;

  if (sem.scope_stack == NULL)
    return;

  scope = push_scope();
  scope->kind = kind;
  scope->sptr = sptr;
  scope->symavl = stb.stg_avail;

  switch (kind) {
  case SCOPE_NORMAL:
  case SCOPE_MODULE:
  case SCOPE_BLOCK:
  case SCOPE_USE:
  case SCOPE_PAR:
    break;
  case SCOPE_SUBPROGRAM:
    if (sem.which_pass == 1)
      setfile(1, SYMNAME(sptr), 0);
    break;
  case SCOPE_INTERFACE:
    scope->closed = TRUE;
    break;
  default:
    interr("push_scope_level: invalid scope kind", kind, ERR_Warning);
  }

  /* When entering a parallel scope, the current scope is left as the
   * enclosing scope of the parallel region. */
  if (kind != SCOPE_PAR)
    stb.curr_scope = sptr;

#if DEBUG
  if (DBGBIT(5, 0x200)) {
    fprintf(gbl.dbgfil, "\n++++++++  push_%ccope_level(%s)  pass=%d  line=%d\n",
            sem.which_pass ? 'S' : 's', kind_to_string(kind), sem.which_pass,
            gbl.lineno);
    dumpscope(gbl.dbgfil);
  }
#endif
}

/* \brief Pop scope stack up to and including a slot of the given kind. */
void
pop_scope_level(SCOPEKIND kind)
{
  SCOPEKIND curr_kind;
  SPTR sptr;

  if (sem.scope_stack == NULL)
    return;

  if (sem.scope_level == 0) {
    interr("pop_scope_level: stack underflow", sem.scope_level, ERR_Severe);
    return;
  }

  do {
    SCOPESTACK *scope = curr_scope();
    int top = scope->symavl;
    int scope_id = scope->sptr;
    curr_kind = scope->kind;
    switch (curr_kind) {
    case SCOPE_BLOCK:
      for (sptr = top; sptr < stb.stg_avail; ++sptr) {
        if (!CONSTRUCTSYMG(sptr) || HIDDENG(sptr))
          continue;
        if (ST_ISVAR(STYPEG(sptr))) {
          // Some secondary syms are marked as construct syms when created.
          // Assuming that isn't guaranteed for all cases, re/do that here.
          SPTR parent = ENCLFUNCG(sptr), sptr2;
          if ((sptr2 = MIDNUMG(sptr)) != 0) {
            CONSTRUCTSYMP(sptr2, true);
            ENCLFUNCP(sptr2, parent);
          }
          if ((sptr2 = SDSCG(sptr)) != 0) {
            CONSTRUCTSYMP(sptr2, true);
            ENCLFUNCP(sptr2, parent);
          }
          if ((sptr2 = PTROFFG(sptr)) != 0) {
            CONSTRUCTSYMP(sptr2, true);
            ENCLFUNCP(sptr2, parent);
          }
        }
        // Avoid repeat processing in an ancestor block.
        HIDDENP(sptr, true);
        // Secondary syms must remain on hash chains to avoid conflicts with
        // same-named syms in other blocks.  Remove primary syms.
        if (!HCCSYMG(sptr))
          pop_sym(sptr);
      }
      break;
    case SCOPE_INTERFACE: {
      int parent_scope_id = get_scope(-1)->sptr;
      if (parent_scope_id != scope_id)
        for (sptr = stb.stg_avail - 1; sptr >= top; --sptr)
          if (SCOPEG(sptr) == scope_id)
            SCOPEP(sptr, parent_scope_id); // rehost to enclosing scope
      break;
    }
    default:
      if (sem.interface && STYPEG(scope_id) != ST_MODULE)
        for (sptr = stb.stg_avail - 1; sptr >= top; --sptr)
          if (SCOPEG(sptr) == scope_id)
            IGNOREP(sptr, 1); // remove interface scope symbol
      break;
    }
    pop_scope();
  } while (curr_kind != kind);

  /*
   * When leaving a parallel scope, the current scope doesn't need to be
   * reset since it should always be the scope of the nonparallel region
   * containing the parallel region.
   */
  if (kind != SCOPE_PAR && sem.scope_stack[sem.scope_level].kind != SCOPE_PAR)
    stb.curr_scope = sem.scope_level > 0 ?
      curr_scope()->sptr : sem.scope_stack[1].sptr;

#if DEBUG
  if (DBGBIT(5, 0x200)) {
    fprintf(gbl.dbgfil, "\n--------  pop_%ccope_level(%s)  pass=%d  line=%d\n",
            sem.which_pass ? 'S' : 's', kind_to_string(kind), sem.which_pass,
            gbl.lineno);
    dumpscope(gbl.dbgfil);
  }
#endif
}

static SCOPESTACK saved_scope_stack[1];
static int count_scope_saved = 0;

/** \brief Pop the current scope into a save area; restore with
 * restore_scope_level() */
void
save_scope_level(void)
{
  if (count_scope_saved >= 1) {
    interr("trying to save too many scope levels", count_scope_saved, 3);
    return;
  }
  saved_scope_stack[count_scope_saved++] = *curr_scope();
  pop_scope();
  stb.curr_scope = curr_scope()->sptr;
#if DEBUG
  if (DBGBIT(5, 0x200)) {
    fprintf(gbl.dbgfil, "\n--------  save_%ccope_level  pass=%d  line=%d\n",
            sem.which_pass ? 'S' : 's', sem.which_pass, gbl.lineno);
    dumpscope(gbl.dbgfil);
  }
#endif
}

/** \brief Restore the scope that was saved by save_scope_level() */
void
restore_scope_level(void)
{
  if (count_scope_saved <= 0) {
    interr("trying to restore too many scope levels", count_scope_saved, 3);
    return;
  }
  *push_scope() = saved_scope_stack[--count_scope_saved];
  stb.curr_scope = curr_scope()->sptr;
#if DEBUG
  if (DBGBIT(5, 0x200)) {
    fprintf(gbl.dbgfil, "\n++++++++  restore_%ccope_level  pass=%d  line=%d\n",
            sem.which_pass ? 'S' : 's', sem.which_pass, gbl.lineno);
    dumpscope(gbl.dbgfil);
  }
#endif
}

void
par_push_scope(LOGICAL bind_to_outer)
{
  SCOPESTACK *scope, *next_scope;
  SC_KIND prev_sc = sem.sc;
  if (curr_scope()->kind != SCOPE_PAR && sem.parallel >= 1) {
    sem.sc = SC_PRIVATE;
  } else if (sem.task) {
    sem.sc = SC_PRIVATE;
  }
  else if (sem.teams >= 1) {
    sem.sc = SC_PRIVATE;
  } else if (sem.target && sem.parallel >= 1) {
    sem.sc = SC_PRIVATE;
  }
  push_scope_level(sem.next_unnamed_scope++, SCOPE_PAR);
  scope = curr_scope();
  next_scope = get_scope(-1); /* next to top of stack */
  if (!bind_to_outer || next_scope->kind != SCOPE_PAR) {
    scope->rgn_scope = sem.scope_level;
    scope->par_scope = PAR_SCOPE_SHARED;
  } else {
    scope->rgn_scope = next_scope->rgn_scope;
    scope->par_scope = next_scope->par_scope;
    scope->end_prologue = next_scope->end_prologue;
  }
  scope->di_par = sem.doif_depth;
  scope->shared_list = NULL;
  scope->prev_sc = prev_sc;
  enter_lexical_block(flg.debug && !XBIT(123, 0x400));
}

void
par_pop_scope(void)
{
  SCOPE_SYM *symp;
  /*
   * Restore the scope of any symbols that appeared in a SHARED clause.
   * This is only needed if the DEFAULT scope is 'PRIVATE' or 'NONE".
   */
  for (symp = curr_scope()->shared_list; symp != NULL; symp = symp->next) {
    SCOPEP(symp->sptr, symp->scope);
  }
  if (BLK_SYM(sem.scope_level)) {
    exit_lexical_block(flg.debug && !XBIT(123, 0x400));
  }

  sem.sc = curr_scope()->prev_sc;
  pop_scope_level(SCOPE_PAR);
  if (curr_scope()->kind != SCOPE_PAR) {
    sem.sc = SC_LOCAL;
  }
}

static SCOPESTACK *
push_scope(void)
{
  ++sem.scope_level;
  NEED(sem.scope_level + 1, sem.scope_stack, SCOPESTACK, sem.scope_size,
       sem.scope_size + 10);
  BZERO(sem.scope_stack + sem.scope_level, SCOPESTACK, 1);
  return curr_scope();
}

static void
pop_scope(void)
{
  --sem.scope_level;
  assert(sem.scope_level >= 0, "scope stack underflow", sem.scope_level,
         ERR_Fatal);
}

#if DEBUG
void
dumpscope(FILE *f)
{
  int sl;
  if (f == NULL) {
    f = stderr;
  }
  if (sem.scope_stack == NULL) {
    fprintf(f, "no scope stack\n");
    return;
  }
  for (sl = 0; sl <= sem.scope_level; ++sl) {
    dump_one_scope(sl, f);
  }
}

void
dump_one_scope(int sl, FILE *f)
{
  SCOPESTACK *scope;
  SPTR sptr;
  if (f == NULL) {
    f = stderr;
  }
  if (sl < 0 || sl >= sem.scope_size) {
    interr("dump_one_scope: bad scope stack level", sl, ERR_Warning);
    return;
  }
  scope = sem.scope_stack + sl;
  sptr = scope->sptr;
  fprintf(f, "%ccope %2d. %-11s %-7s %-8s symavl=%3d  %d=%s\n",
          sem.which_pass ? 'S' : 's', sl, kind_to_string(scope->kind),
          scope->closed ? "closed" : "open",
          scope->Private ? "private" : "public",
          scope->symavl, sptr,
          sptr >= stb.firstosym ? SYMNAME(sptr) : "");
  if (scope->except) {
    int ex;
    fprintf(f, "+ except");
    for (ex = scope->except; ex; ex = SYMI_NEXT(ex)) {
      fprintf(f, " %d(%s)", SYMI_SPTR(ex), SYMNAME(SYMI_SPTR(ex)));
    }
    fprintf(f, "\n");
  }
  if (scope->import) {
    int im;
    fprintf(f, "+ import");
    for (im = scope->import; im; im = SYMI_NEXT(im)) {
      fprintf(f, " %d(%s)", SYMI_SPTR(im), SYMNAME(SYMI_SPTR(im)));
    }
    fprintf(f, "\n");
  }
}

static const char *
kind_to_string(SCOPEKIND kind)
{
  switch (kind) {
  case SCOPE_OUTER:      return "Outer";
  case SCOPE_NORMAL:     return "Normal";
  case SCOPE_MODULE:     return "Module";
  case SCOPE_SUBPROGRAM: return "Subprogram";
  case SCOPE_BLOCK:      return "Block";
  case SCOPE_INTERFACE:  return "Interface";
  case SCOPE_USE:        return "Use";
  case SCOPE_PAR:        return "Par";
  }
}

#endif
