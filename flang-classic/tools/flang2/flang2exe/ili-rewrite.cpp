/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "gbldefs.h"
#include "error.h"
#include "fastset.h"
#include "global.h"
#include "go.h"
#include "symtab.h"
#include "ili.h"
#include "ili-rewrite.h"

#if DEBUG + 0
#define DEBUGGING (DBGBIT(64, 8) != 0)
#else
#define DEBUGGING 0
#endif

/* Determines whether an ILI that has been passed to addili() was stored
 * without change.  If this be not the case, addili() can be assumed to
 * have rewritten the expression, possibly to improve it by having
 * recognized and replaced an expression (but possibly also just to
 * canonicalize it).
 */
static bool
addili_changed_ili(int ili, const ILI *saved)
{
  int j, opc = ILI_OPC(ili), opnds = IL_OPRS(opc);
  if (opc != saved->opc)
    return true;
  for (j = 1; j <= opnds; ++j)
    if (ILI_OPND(ili, j) != saved->opnd[j - 1])
      return true;
  return ILI_ALT(ili) != saved->alt;
}

/* JSRs have special semantics: their ILIs must be treated as
 * common subexpressions and evaluated at most once in any
 * non-extended basic block.
 */
static bool
is_jsr(int opc)
{
  switch (opc) {
  case IL_JSR:
  case IL_JSRA:
/* case IL_QJSR: no, QJSR is discretionary */
  case IL_GJSR:
  case IL_GJSRA:
#ifdef LONG_DOUBLE_FLOAT128
  case IL_FLOAT128RESULT:
#endif
    return true;
  }
  return false;
}

struct hidden_state {
  int *ili_map;     /* maps at->original_ili to new ili */
  int *ili_bih;     /* most recent BIH containing ili */
  int *ili_backmap; /* maps new ili to at->original_ili */
  int limit;        /* extent of these arrays */
  bool is_context_insensitive;
};

static int visit_ili_operands(bool *result_cse, const ILI_coordinates *at,
                              struct hidden_state *state);

/* Visits ILI that are referenced as subscripts in name table (NME)
 * operands of load/store ILIs.
 */
static int
visit_nme_subscript(int nme, bool *operand_cse, ILI_coordinates *at,
                    struct hidden_state *state)
{
  if (nme > NME_VOL /* && state->is_context_insensitive */) {
    if (NME_TYPE(nme) == NT_IND || NME_TYPE(nme) == NT_MEM) {
      int base = NME_NM(nme);
      int new_base = visit_nme_subscript(base, operand_cse, at, state);
      if (new_base != base)
        nme = addnme(NME_TYPE(nme), NME_SYM(nme), new_base, NME_CNST(nme));
    } else if (NME_TYPE(nme) == NT_ARR) {
      int base = NME_NM(nme);
      int sub = NME_SUB(nme);
      int new_base = visit_nme_subscript(base, operand_cse, at, state);
      int new_sub;
      at->ili = sub;
      at->original_ili = sub;
      new_sub = visit_ili_operands(operand_cse, at, state);
      if (new_base != base || new_sub != sub)
        nme = add_arrnme(NT_ARR, NME_SYM(nme), new_base, NME_CNST(nme), new_sub,
                         NME_INLARR(nme));
    }
  }
  return nme;
}

/* Recursive traversal of ILI expression trees with a visitation callback.
 * Returns the ILI index of the replacement operand.
 */
static int
visit_ili_operands(bool *result_cse, const ILI_coordinates *at,
                   struct hidden_state *state)
{
  ILI_OP opc = ILI_OPC(at->ili);
  int opnds = IL_OPRS(opc);
  ILI new_ili;
  bool any_change = false;
  ILI_coordinates here = *at;
  int orig_ili = at->original_ili;
  int result = state->ili_map[orig_ili];
  bool operand_cse = false;
  int j;

  if (DEBUGGING)
    dbgprintf("visit_ili_operands: begin ili %d original %d, ilt %d, "
              "bih %d\n",
              at->ili, at->original_ili, at->ilt, at->bih);

  CHECK(orig_ili >= 0 && orig_ili < state->limit);

  /* Respect JSR and CSE block semantics if this expression has already
   * been visited in this block.
   * 1) Each distinct JSR ILI number produces at most one call per block.
   *    JSRs are automatically common subexpressions.  Distinct JSR ILIs
   *	  are distinct calls.
   * 2) CSE ILIs re-use the value produced by the most recent evaluation
   *    of their operand ILI in the same block.  If the operand ILI appears
   *	  outside a CSE ILI, it is re-evaluated.  Expressions that are used
   *	  as CSEs often appear in FREE ILIs but are not required to do so.
   *    The expander and optimizer should take care to avoid potential
   *	  ambiguity, such as having the same ILI appear within and without
   *	  CSE ILIs in the same statement.
   */
  if (result >= 0) {
    if (state->is_context_insensitive) {
      if (DEBUGGING)
        dbgprintf("visit_ili_operands: end ili %d (orig %d), "
                  "cached result %d\n",
                  at->ili, orig_ili, result);
      return result;
    } else if (state->ili_bih[orig_ili] != at->bih) {
      result = -1; /* cached result no longer valid */
    } else if (is_jsr(opc)) {
      if (DEBUGGING)
        dbgprintf("visit_ili_operands: end JSR ili %d (orig %d), "
                  "forced block CSE %d\n",
                  at->ili, orig_ili, result);
      CHECK(result > 0 && result < state->limit);
      CHECK(state->ili_backmap[result] == orig_ili);
      *result_cse = true;
      return result;
    } else if (at->parent && at->parent_opnd == 1 &&
               is_cseili_opcode(ILI_OPC(at->parent->ili))) {
      /* This ILI is the operand of a CSE ILI. */
      if (DEBUGGING)
        dbgprintf("visit_ili_operands: end ili %d (orig %d), "
                  "CSE result %d\n",
                  at->ili, orig_ili, result);
      *result_cse = true;
      return result;
    }
  }

  /* Check for circular dependence */
  CHECK(state->ili_map[orig_ili] >= -1);
  state->ili_map[orig_ili] = -2;

  /* Recursively visit operands */
  here.parent = at;
  for (j = 1; j <= opnds; ++j) {
    int opd = ILI_OPND(at->ili, j), original_opd = opd;
    if (IL_ISLINK(opc, j)) {
      here.ili = opd;
      here.original_ili = opd;
      here.parent_opnd = j;
      opd = visit_ili_operands(&operand_cse, &here, state);
      any_change |= opd != original_opd;
    } else if (IL_OPRFLAG(opc, j) == ILIO_NME) {
      here.parent_opnd = j;
      opd = visit_nme_subscript(opd, &operand_cse, &here, state);
      any_change |= opd != original_opd;
    }
    new_ili.opnd[j - 1] = opd;
  }

  /* Visit alt code, too */
  here.ili = ILI_ALT(at->ili);
  if (here.ili > 0) {
    here.original_ili = here.ili;
    here.parent_opnd = 0;
    new_ili.alt = visit_ili_operands(&operand_cse, &here, state);
    any_change |= new_ili.alt != here.ili;
  } else {
    new_ili.alt = 0;
  }

  if (!any_change) {
    result = at->visitor(at);
    if (DEBUGGING)
      dbgprintf("visit_ili_operands: ili %d, no operand changes, "
                "visitor returned result %d\n",
                at->ili, result);
  } else {
    /* At least one of the operands to this ILI has changed.
     * Construct a new ILI and see whether it enjoys any improvement
     * in addili().
     */
    new_ili.opc = opc;
    here = *at;
    here.original_ili = at->ili;
    if (is_jsr(opc)) {
      /* Observe special semantics for JSRs, which are *always* CSEs
       * in the blocks that contain them.  Create a new distinct JSR
       * so that we don't accidentally alias another JSR in the same
       * block.
       */
      here.ili = get_ili_ns(&new_ili);
      CHECK(here.ili > 0);
      if (new_ili.alt != 0) {
        if (ILI_ALT(here.ili) == 0)
          ILI_ALT(here.ili) = new_ili.alt;
        else
          CHECK(ILI_ALT(here.ili) == new_ili.alt);
      }
    } else {
      here.ili = addili(&new_ili);
    }
    if (here.ili == 0) {
      /* conditional jump was nullified */
      if (DEBUGGING)
        dbgprintf("visit_ili_operands: ili %d, result is 0 after "
                  "addili() with changed operands\n",
                  at->ili);
      result = 0;
    } else {
      /* Improvement test:
       * Traversal has visited the operands of this ILI instance, at least
       * one of which has changed, and the ILI has been reconstituted.
       * It will be characterized as having been "improved" if the
       * expression finally stored by addili() differs from what was
       * passed into it, and so it may be assumed that a local pattern
       * recognition and replacement transformation has affected the
       * expression.  If an operand changed was a forced move due to
       * JSR or CSE semantics, that is also conveyed.
       */
      if (new_ili.alt > 0 && new_ili.alt != ILI_ALT(at->ili)) {
        here.ili = get_ili_ns(&new_ili);
        if (ILI_ALT(here.ili) == 0)
          ILI_ALT(here.ili) = new_ili.alt;
        else
          CHECK(ILI_ALT(here.ili) == new_ili.alt);
      }
      here.this_ili_improved = addili_changed_ili(here.ili, &new_ili);
      here.has_cse = operand_cse;
      result = at->visitor(&here);
      if (DEBUGGING)
        dbgprintf("visit_ili_operands: ili %d, operands changed; "
                  "addili() returned %d (improved? %d forced? %d), "
                  "visitor result %d\n",
                  at->ili, here.ili, here.this_ili_improved, operand_cse,
                  result);
    }
  }

  /* Bookkeeping of results */
  if (result >= state->limit) {
    int new_limit = 2 * result;
    int dummy = state->limit; /* so NEED can overwrite it */
    NEED(new_limit, state->ili_map, int, dummy, new_limit);
    dummy = state->limit;
    NEED(new_limit, state->ili_bih, int, dummy, new_limit);
    dummy = state->limit;
    NEED(new_limit, state->ili_backmap, int, dummy, new_limit);
    for (j = state->limit; j < new_limit; ++j) {
      state->ili_map[j] = -1;
      state->ili_bih[j] = -1;
      state->ili_backmap[j] = -1;
    }
    state->limit = new_limit;
  }

  if (DEBUGGING)
    dbgprintf("visit_ili_operands: end ili %d original %d, result %d\n",
              at->ili, orig_ili, result);
  state->ili_map[orig_ili] = result;
  state->ili_bih[orig_ili] = at->bih;
  if (result >= 0)
    state->ili_backmap[result] = orig_ili;
  *result_cse |= operand_cse;
  return result;
}

/* Driver for context-sensitive ILI traversal.
 * Returns true if any ILI changed.
 */
bool
visit_ilis(ILI_visitor visitor, void *visitor_context,
           bool is_context_insensitive)
{
  int j;
  bool any = false;
  ILI_coordinates at;
  struct hidden_state state;

  memset(&state, 0, sizeof state);
  state.limit = 2 * ilib.stg_avail;
  NEW(state.ili_map, int, state.limit);
  NEW(state.ili_bih, int, state.limit);
  NEW(state.ili_backmap, int, state.limit);
  state.is_context_insensitive = is_context_insensitive;
  for (j = 0; j < state.limit; ++j) {
    state.ili_map[j] = -1;
    state.ili_bih[j] = -1;
    state.ili_backmap[j] = -1;
  }

  at.visitor = visitor;
  at.context = visitor_context;
  at.parent = NULL;
  at.parent_opnd = 0;
  at.this_ili_improved = false;
  for (at.bih = gbl.entbih; at.bih > 0; at.bih = BIH_NEXT(at.bih)) {
    int next_ilt;
    rdilts(at.bih);
    for (at.ilt = BIH_ILTFIRST(at.bih); at.ilt > 0; at.ilt = next_ilt) {
      int new_ili;
      bool has_cse = false;
      next_ilt = ILT_NEXT(at.ilt);
      at.ili = ILT_ILIP(at.ilt);
      at.original_ili = at.ili;
      new_ili = visit_ili_operands(&has_cse, &at, &state);
      if (new_ili != at.original_ili) {
        /* ILI has changed for this statement */
        any = true;
        if (new_ili == 0) {
          /* conditional branch was nullified */
          new_ili = ad1ili(IL_NULL, 0);
          ILT_BR(at.ilt) = 0;
          if (DEBUGGING)
            dbgprintf("visit_ilis: ilt %d nullified\n", at.ilt);
        } else {
          if (ILI_OPC(new_ili) == IL_JMP)
            BIH_FT(at.bih) = 0;
          if (DEBUGGING)
            dbgprintf("visit_ilis: ilt %d: ili %d -> %d "
                      "(cses? %d)\n",
                      at.ilt, at.ili, new_ili, has_cse);
        }
        ILT_ILIP(at.ilt) = new_ili;
      }
    }
    wrilts(at.bih);
    if (BIH_LAST(at.bih))
      break;
  }

  FREE(state.ili_backmap);
  FREE(state.ili_bih);
  FREE(state.ili_map);
  return any;
}

/* Create a new ILI that is a copy of another with a potential
 * change at one operand index.  This function is suitable for
 * use by a visitation callback.  "opnd_index" is one-based,
 * and -1 refers to alt code.
 */
int
update_ili_operand(int ili, int opnd_index, int new_opnd)
{
  if (ILI_OPND(ili, opnd_index) != new_opnd) {
    int j;
    ILI_OP opc = ILI_OPC(ili);
    int opnds = IL_OPRS(opc);
    ILI newili;
    newili.opc = opc;
    if (opnd_index == -1)
      newili.alt = new_opnd;
    else
      newili.alt = ILI_ALT(ili);
    for (j = 1; j <= opnds; ++j) {
      if (j == opnd_index)
        newili.opnd[j - 1] = new_opnd;
      else
        newili.opnd[j - 1] = ILI_OPND(ili, j);
    }
    ili = addili(&newili);
    if (newili.alt != 0 && newili.alt != ILI_ALT(ili)) {
      ili = get_ili_ns(&newili);
      if (ILI_ALT(ili) == 0)
        ILI_ALT(ili) = newili.alt;
    }
  }
  return ili;
}

/* Collect the set of ILIs that are ILT expression roots. */
void
collect_root_ilis(fastset *root_ilis)
{
  int bih, ilt;

  fastset_vacate(root_ilis);
  for (bih = gbl.entbih; bih > 0; bih = BIH_NEXT(bih)) {
    rdilts(bih);
    for (ilt = BIH_ILTFIRST(bih); ilt > 0; ilt = ILT_NEXT(ilt)) {
      int ili = ILT_ILIP(ilt);
      fastset_add(root_ilis, ili);
    }
    wrilts(bih);
    if (BIH_LAST(bih))
      break;
  }
}

/* Find all live ILIs as a no-op context-insensitive traversal. */
static int
noop_visitor(const ILI_coordinates *at)
{
  fastset *live_ilis = GetLiveILIs(at);
  fastset_add(live_ilis, at->ili);
  return at->ili;
}

void
collect_live_ilis(fastset *live)
{
  bool any;
  fastset_vacate(live);
  any = visit_ilis(noop_visitor, live, true /* context-insensitive */);
  CHECK(!any);
}

bool
scan_ili_tree(ILI_tree_scan_visitor visitor, void *visitor_context, int ili)
{
  int j;
  ILI_OP opc;
  int opnds;

  if (visitor(visitor_context, ili))
    return true;
  if (is_cseili_opcode(ILI_OPC(ili)))
    return false;
  opc = ILI_OPC(ili);
  opnds = IL_OPRS(opc);
  for (j = 1; j <= opnds; ++j)
    if (IL_ISLINK(opc, j))
      if (scan_ili_tree(visitor, visitor_context, ILI_OPND(ili, j)))
        return true;
  if (ILI_ALT(ili))
    return scan_ili_tree(visitor, visitor_context, ILI_ALT(ili));
  return false;
}

void
collect_tree_ilis(fastset *tree_ilis, int ili, bool scan_cses)
{
  if (!fastset_contains(tree_ilis, ili)) {
    ILI_OP opc = ILI_OPC(ili);
    fastset_add(tree_ilis, ili);
    if (scan_cses || !is_cseili_opcode(opc)) {
      int j;
      int opnds = IL_OPRS(opc);
      for (j = 1; j <= opnds; ++j)
        if (IL_ISLINK(opc, j))
          collect_tree_ilis(tree_ilis, ILI_OPND(ili, j), scan_cses);
      if (ILI_ALT(ili))
        collect_tree_ilis(tree_ilis, ILI_ALT(ili), scan_cses);
    }
  }
}
