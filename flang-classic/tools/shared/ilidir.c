/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief ILI directive module
 *
 * Utility routines are valid after the semantic analyzer and after basic
 * blocks have been created for a function.
 */

#include "ilidir.h"
#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "fdirect.h"

#if DEBUG
#define TR0(s)         \
  if (DBGBIT(1, 2048)) \
    fprintf(gbl.dbgfil, s);
#define TR1(s, a)      \
  if (DBGBIT(1, 2048)) \
    fprintf(gbl.dbgfil, s, a);
#define TR2(s, a, b)   \
  if (DBGBIT(1, 2048)) \
    fprintf(gbl.dbgfil, s, a, b);
#define TR3(s, a, b, c) \
  if (DBGBIT(1, 2048))  \
    fprintf(gbl.dbgfil, s, a, b, c);

#else
#define TR0(s)
#define TR1(s, a)
#define TR2(s, a, b)
#define TR3(s, a, b, c)

#endif

static int first;      /* beginning loopset index for function; 0 => doesn't
                        * exist */
static bool opened; /* TRUE if open_pragma found a set which applies to a
                        * loop */

static int find_lpprg(int);

/** \brief All blocks have been created for a function (called at end of
 * expand).
 */
void
ili_lpprg_init(void)
{
  int i;
  LPPRG *lpprg;

  if (direct.lpg.avail <= 1) {
    first = 0;
    return;
  }
  first = 1;
  TR1("ili_lpprg_init: %s\n", SYMNAME(GBL_CURRFUNC));
  for (i = first; i < direct.lpg.avail; i++) {
    lpprg = direct.lpg.stgb + i;
    if (lpprg->beg_line < 0)
      break;
    TR3("indx %d, begline %d, endline %d\n", i, lpprg->beg_line,
        lpprg->end_line);
  }

}

/** \brief Find and open the set of loop pragmas associated with a line number.
 *
 * SHOULD have a matching close_pragma().
 */
void
open_pragma(int line)
{
  LPPRG *lpprg;
  int match;

  opened = false;
  match = find_lpprg(line);
  if (match) {
    lpprg = direct.lpg.stgb + match;
    TR2("    begline %d, endline %d\n", lpprg->beg_line, lpprg->end_line);
    load_dirset(&lpprg->dirset);
    opened = true;
#ifdef FE90
    direct.indep = lpprg->indep;
    direct.index_reuse_list = lpprg->index_reuse_list;
#endif
  }

}

/** \brief Close the current pragmas => load the pragmas which have routine
 * scope.
 */
void
close_pragma(void)
{
  if (opened) {
    load_dirset(&direct.rou_begin);
    opened = false;
#ifdef FE90
    direct.indep = NULL;
    direct.index_reuse_list = NULL;
#endif
  }

}

#ifdef FE90
void
open_dynpragma(int std, int lineno)
{
  int i;
  LPPRG *lpprg;

  for (i = 1; i < direct.dynlpg.avail; i++) {
    lpprg = direct.dynlpg.stgb + i;
    if (lpprg->beg_line != std)
      continue;
    load_dirset(&lpprg->dirset);
    direct.indep = lpprg->indep;
    direct.index_reuse_list = lpprg->index_reuse_list;
    opened = TRUE;
    return;
  }
  open_pragma(lineno);
}

/** \brief Save the current pragma for std. */
void
save_dynpragma(int std)
{
  int i;
  LPPRG *lpprg;

  for (i = 1; i < direct.dynlpg.avail; i++) {
    lpprg = direct.dynlpg.stgb + i;
    if (lpprg->beg_line != std)
      continue;
    store_dirset(&lpprg->dirset);
    lpprg->indep = direct.indep;
    lpprg->index_reuse_list = direct.index_reuse_list;
    opened = TRUE; /* so close_pragma will work */
    close_pragma();
    return;
  }
  i = direct.dynlpg.avail++;
  NEED(direct.dynlpg.avail, direct.dynlpg.stgb, LPPRG, direct.dynlpg.size,
       direct.dynlpg.size + 8);
  lpprg = direct.dynlpg.stgb + i;
  lpprg->beg_line = std;
  store_dirset(&lpprg->dirset);
  lpprg->indep = direct.indep;
  lpprg->index_reuse_list = direct.index_reuse_list;
  opened = TRUE; /* so close_pragma will work */
  close_pragma();
}
#endif

/** \brief Find the set of loop pragmas associated with a line number (the
 * set's beginning and ending line numbers enclose the requested line number).
 *
 * Since the loop sets occur in lexical order, the last loop set found is the
 * set with beginning line number is the floor of the requested line number.
 */
static int
find_lpprg(int line)
{
  int i;
  LPPRG *lpprg;
  int match;

  if (first == 0 || line == 0)
    return 0;

  match = 0;
  for (i = first; i < direct.lpg.avail; i++) {
    lpprg = direct.lpg.stgb + i;
    if (lpprg->beg_line < 0 || lpprg->beg_line > line)
      break;
    if (lpprg->beg_line <= line && line <= lpprg->end_line)
      match = i;
  }
  TR2("find_lpprg:for line %d, match = %d\n", line, match);

  return match;
}

#define STK_SZ 128

static int stk[STK_SZ]; /* should be dynamic ? */
static int top = 0;     /* first entry in stack is reserved -- the
                         * value is zero => use routine's pragmas */

/** \brief Find and open and stack the set of loop pragmas associated with a
 * line number.
 *
 * MUST have a matching pop_pragma().  If a match isn't found, push the
 * routine's set.
 */
void
push_pragma(int line)
{
  LPPRG *lpprg;
  int match;

  top++;
  if (top >= STK_SZ) {
#if DEBUG
    interr("push_pragma:stkovflw", line, ERR_Severe);
#endif
    top = STK_SZ - 1;
  }
  match = find_lpprg(line);
  stk[top] = match;
  if (match) {
    lpprg = direct.lpg.stgb + match;
    TR2("    push: begline %d, endline %d\n", lpprg->beg_line, lpprg->end_line);
    load_dirset(&lpprg->dirset);
  } else {
    TR0("    push: using routine\n");
    load_dirset(&direct.rou_begin);
  }

}

/** \brief Pop the set from the stack and open/restore the set which is on the
 * top of stack.
 */
void
pop_pragma(void)
{
  LPPRG *lpprg;
  int match;

#if DEBUG
  if (top <= 0) {
    interr("pop_pragma:stkuflw", gbl.lineno, ERR_Severe);
    load_dirset(&direct.rou_begin);
    return;
  }
#endif
  top--;
  match = stk[top];
  if (match) {
    lpprg = direct.lpg.stgb + match;
    TR1("    pop: using %d\n", match);
    load_dirset(&lpprg->dirset);
  } else {
    TR0("    pop: using routine\n");
    load_dirset(&direct.rou_begin);
  }

}
