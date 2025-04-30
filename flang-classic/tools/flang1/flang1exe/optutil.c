/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  optutil.c - miscellaneous optimizer utility routines

    LOGICAL is_optsym(int)
        determine if nme represents an optimizable sym

    LOGICAL is_sym_optsafe(int)
        determine if there are side-effect or alias conflicts for nme

    LOGICAL is_call_safe(int)
        determine if nme can be affected by a call

    LOGICAL is_based_safe(int, int, int)
        determine if based symbol is safe to optimize

    int     pred_of_loop(int)
        find a flowgraph node which is a predecessor of a loop

    int     find_rdef(int, int, LOGICAL)
        find a single def of a nme which reaches a flowgraph node (either the
        beginning or end of the node)

    LOGICAL is_sym_entry_live(int)
        determine if a symbol is live upon entry to a function/subprogram

    LOGICAL is_sym_exit_live(int)
        determine if a symbol is live upon exit from a function/subprogram

    LOGICAL is_sym_imp_live(int)
        determine if a symbol is live because of implicit uses

    LOGICAL can_copy_def(int, int, LOGICAL)
        determine if a def's value can be copied to a flowgraph node (either
        the beginning or end of the node).

    LOIGCAL def_ok(int, int, int)
    LOGICAL is_avail_expr(int, int, int, int, LOGICAL)  (static)
    LOGICAL avail_expr(int)  (static)

    LOGICAL single_ud(int)
        determine if a single def reaches a use.
    LOGICAL only_one_ud(int)
        similar to single_ud except left-hand side must be available rather
        than the right-hand side.

    LOGICAL is_def_imp_live(int)
        determine if a definition is live because of implicit uses

    void    rm_def_rloop(int, int)
        remove a def, where its uses in a loop have been deleted, if it has
        no other uses.
*/
#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "semant.h"
#include "nme.h"
#include "ast.h"
#include "optimize.h"

int basesym_of(int);

/* forward declarations */

/*

   The bit vector representation for a set of size N is a sequence of
   bits ordered right to left.  For i, 1<=i<=N,
      w = (i-1) / #bits in a BV unit
      r = (i-1) % #bits in a BV unit
   then the rth bit in the wth BV unit represents the set membership
   for i.

   Bit vector support routines:
      bv_zero   -  a = 0
      bv_copy   -  a = b
      bv_union  -  a = a U b
      bv_sub    -  a = a - b
      bv_set    -  add element to a
      bv_off    -  remove element from a
      bv_notequal  -  a != b
      bv_mem	-  is element a member of a
*/

void
bv_zero(BV *a, int len)
{
  while (len--)
    *a++ = 0;
}

void
bv_copy(BV *a, BV *b, int len)
{
  while (len--)
    *a++ = *b++;
}

void
bv_union(BV *a, BV *b, int len)
{
  while (len--)
    *a++ |= *b++;
}

void
bv_sub(BV *a, BV *b, int len)
{
  while (len--)
    *a++ &= ~(*b++);
}

void
bv_intersect(BV *a, BV *b, UINT len)
{
  while (len--)
    *a++ &= *b++;
}

void
bv_intersect3(BV *a, BV *b, BV *c, UINT len)
{
  while (len--)
    *a++ = (*b++) & (*c++);
}

void
bv_set(BV *a, int elem)
{
  int w, r;

  w = (elem - 1) / BV_BITS;
  r = (elem - 1) - w * BV_BITS;
  *(a + w) |= 1 << r;
}

void
bv_off(BV *a, int elem)
{
  int w, r;

  w = (elem - 1) / BV_BITS;
  r = (elem - 1) - w * BV_BITS;
  *(a + w) &= (BV) ~(1 << r);
}

LOGICAL
bv_notequal(BV *a, BV *b, int len)
{
  while (len--)
    if (*a++ != *b++)
      return (TRUE);
  return (FALSE);
}

LOGICAL
bv_mem(BV *a, int elem)
{
  int w, r;

#if DEBUG
  assert(a != NULL, "bv_mem: bv null", 0, 3);
#endif
  w = (elem - 1) / BV_BITS;
  r = (elem - 1) - w * BV_BITS;
  if (*(a + w) & (BV)(1 << r))
    return (TRUE);
  return (FALSE);
}

void
bv_print(BV *bv, int maxlen)
{
  int i, j, w;

  j = 0;
  w = *bv++;
  for (i = 1; i <= maxlen; i++) {
    if (w & 1) {
      if (j == 11) {
        fprintf(gbl.dbgfil, "\n           ");
        j = 0;
      }
      j++;
      fprintf(gbl.dbgfil, " %5d", i);
    }
    w = w >> 1;
    if (i % BV_BITS == 0)
      w = *bv++;
  }
  fprintf(gbl.dbgfil, "\n");

}

/*  this routine checks the names entry to determine if it is allowed
 *  to enter in the global flow analysis
 */
LOGICAL
is_optsym(int nme)
{
  if (NME_TYPE(nme) != NT_VAR) /* var names only	 */
    return (FALSE);
  return (TRUE);
}

LOGICAL
is_sym_optsafe(int nme, int lpx)
{
  int sym;
  /*
   * a symbol is not safe (potential "side-effect" or alias conflicts
   * exist) if:
   * 1.  for C and Fortran, the symbol is volatile
   *
   * 2.  for Fortran, the symbol is equivalenced
   *
   * 3.  loop contains a call AND
   *     a.  sym's address has been taken, or
   *     b.  sym's storage class is not auto (it is static or extern), or
   *     c.  for c++, sym is used in two or more nested scope levels
   *
   * 4.  loop contains a store or load via a pointer AND the sym could
   *     conflict with a pointer (see optutil.c:is_sym_ptrsafe()).
   *
   * 6.  the symbol is private and the loop is not a parallel loop.
   *
   * 7.  the symbol is not a private variable and the loop contains
   *     a parallel section.
   *
   * 8.  the symbol is threadprivate and the current region is 0; in
   *     region 0, its address can't computed before the call to mp_cdecl().
   *
   * WARNING - Any new constraints may need to be added to
   *     invar.c:is_sym_invariant_safe() as well.
   *
   */
  sym = NME_SYM(nme);

  if (VOLG(sym))
    return (FALSE);

  if (!XBIT(19, 0x1) && SOCPTRG(sym)) /* noeqvchk => XBIT(19,0x1) set */
    return (FALSE);

  if (LP_CALLFG(lpx) && (ADDRTKNG(sym)
                         || !IS_LCL_OR_DUM(sym))) {
    return (FALSE);
  }

  if ((LP_PTR_STORE(lpx) || LP_PTR_LOAD(lpx)) && !is_sym_ptrsafe(sym))
    return (FALSE);

  if (IS_PRIVATE(sym) && !LP_PARLOOP(lpx))
    return (FALSE);

  if ((LP_CSECT(lpx) || LP_PARREGN(lpx)) && !IS_PRIVATE(sym)) {
    Q_ITEM *q;
    for (q = LP_STL_PAR(lpx); q != NULL; q = q->next)
      if (q->info == nme)
        return (FALSE);
  }

  if (LP_PARSECT(lpx) && !IS_PRIVATE(sym)) {
    return (FALSE);
  }

  if (lpx == 0 && THREADG(sym)) /* not going to qualify any further */
    return FALSE;

  return (TRUE);
}

LOGICAL
is_sym_live_safe(int nme, int lpx)
{
  int sym;
/*
 * Less conservative form of is_sym_optsafe() for flow.c:is_live_out()/
 * is_live_in().  The is_live routines used to call is_sym_optsafe()
 * with a loop value of 0 to catch cases such as passing the address
 * of a variable to a subroutine (i.e., its ADDRTKN flag is set and
 * LP_CALLFG(0) is set).  Using 0 had the effect of returning FALSE
 * for any variable appearing in a critical section; however, it's
 * sufficient to use the actual loop index for which the is_live
 * inquiry.  In this function, explicitly use a loop value of 0 (LPZ)
 * where it's necesary to be convservative; flow.c will call this
 * function with the actual loop index.
 */
#define LPZ 0
  /*
   * a symbol is not safe (potential "side-effect" or alias conflicts
   * exist) if:
   * 1.  for C and Fortran, the symbol is volatile
   *
   * 2.  for Fortran, the symbol is equivalenced
   *
   * 3.  loop contains a call AND
   *     a.  sym's address has been taken, or
   *     b.  sym's storage class is not auto (it is static or extern), or
   *     c.  for c++, sym is used in two or more nested scope levels
   *
   * 4.  loop contains a store or load via a pointer AND the sym could
   *     conflict with a pointer (see optutil.c:is_sym_ptrsafe()).
   *
   * 6.  the symbol is private and the loop is not a parallel loop.
   *
   * 7.  the symbol is not a private variable and the loop contains
   *     a parallel section.
   *
   * 8.  the symbol is threadprivate and the current region is 0; in
   *     region 0, its address can't computed before the call to mp_cdecl().
   *
   * WARNING - Any new constraints may need to be added to
   *     invar.c:is_sym_invariant_safe() as well.
   *
   */
  sym = NME_SYM(nme);

  if (VOLG(sym))
    return (FALSE);

  if (!XBIT(19, 0x1) && SOCPTRG(sym)) /* noeqvchk => XBIT(19,0x1) set */
    return (FALSE);

  if (LP_CALLFG(LPZ) && (ADDRTKNG(sym)
                         || !IS_LCL_OR_DUM(sym))) {
    return (FALSE);
  }

  if ((LP_PTR_STORE(LPZ) || LP_PTR_LOAD(LPZ)) && !is_sym_ptrsafe(sym))
    return (FALSE);

  if (IS_PRIVATE(sym) && !LP_PARLOOP(lpx))
    return (FALSE);

  if ((LP_CSECT(lpx) || LP_PARREGN(lpx)) && !IS_PRIVATE(sym)) {
    Q_ITEM *q;
    for (q = LP_STL_PAR(lpx); q != NULL; q = q->next)
      if (q->info == nme)
        return (FALSE);
  }

  if (LP_PARSECT(lpx) && !IS_PRIVATE(sym)) {
    return (FALSE);
  }

  return (TRUE);
}

LOGICAL
is_call_safe(int nme)
{
  int sym;
  /*
   * a symbol can be modified by a call if:
   * 1.  for C, the symbol is extern or file static
   *
   * 2.  for Fortran, the symbol is extern (common block)
   *
   * 3.  its address is taken
   *
   * 4.  for C++, sym is accessed from separate scoping units
   *
   * 5.  for Fortran, the call is to an contained subprogram
   *     Unfortunately, we don't distinguish between internal and
   *     external calls as yet, so if there are contained subprograms,
   *     outer-block symbols are marked as not call-safe.
   */
  sym = NME_SYM(nme);

  if (IS_CMNBLK(sym))
    return FALSE;
  if ((POINTERG(sym) || ALLOCATTRG(sym)) && MIDNUMG(sym) &&
      IS_CMNBLK(MIDNUMG(sym))) {
    return FALSE;
  }
  if (gbl.internal && !INTERNALG(sym))
    return FALSE;
  if (ADDRTKNG(sym))
    return FALSE;

  return TRUE;
}

/*
 * determine, given an pointer nme (type NT_IND), if it's safe to ignore
 * any pointer conflicts.  This is based on a combination of the storage
 * class of the pointer variable involved in the reference and the value
 * of the -x 2 flag.
 */
LOGICAL
is_ptr_safe(int nme)
{
  int sym;
  int sc;

#if DEBUG
  assert(NME_TYPE(nme) == NT_IND || NME_TYPE(nme) == NT_VAR ||
             NME_TYPE(nme) == NT_MEM,
         "is_ptr_safe: nme not ind", nme, 3);
#endif
  if (0 != (sym = basesym_of(nme))) {
    sc = SCG(sym);
    if ((XBIT(2, 0x1) && sc == SC_DUMMY) || (XBIT(2, 0x2) && sc == SC_LOCAL) ||
        (XBIT(2, 0x4) && sc == SC_STATIC) ||
        (XBIT(2, 0x8) && (sc == SC_CMBLK || sc == SC_EXTERN)))
      return TRUE;
  }

  return FALSE;
}

/*
 * Determine if a symbol does not conflict with a pointer reference.
 * A pointer can conflict with a symbol if:
 * 1.  the symbol's addrtkn flag is set,
 *
 * 2.  for C, the symbol is extern or file static,
 *
 * 3.  for fortran, the symbol is in common or is an external,
 *
 * 4.  for C++, sym is accessed from separate scoping units
 *
 * The storage class tests may be inhibited by the -x 2 flag or the
 * presence of -Mnodepchk.
 */
LOGICAL
is_sym_ptrsafe(int sym)
{
  int sc;

  if (ADDRTKNG(sym))
    return FALSE;
  sc = SCG(sym);
  if (flg.depchk && (sc == SC_CMBLK || sc == SC_EXTERN) && !XBIT(2, 0x8))
    return FALSE;

  return TRUE;
}

/*
 * find the flowgraph node which is the predecessor of the indicated loop.
 * If there is more than 1, 0 is returned.
 */
int
pred_of_loop(int lpx)
{
  PSI_P pred;
  int fgx;

  fgx = 0;
  pred = FG_PRED(LP_HEAD(lpx));
  for (; pred != PSI_P_NULL; pred = PSI_NEXT(pred)) {
    if (FG_LOOP(PSI_NODE(pred)) == lpx)
      continue;
    if (fgx) {
      fgx = 0;
      break;
    }
    fgx = PSI_NODE(pred);
  }

  return fgx;
}

/*
 * find a single definition for the symbol indicated by its names entry
 * which reaches the beginning or the end of the flowgraph node, fgx.
 * If a definition isn't found or if more than one def reaches the node,
 * 0 is returned.
 */
/* TRUE if def is to reach the beginning */
int
find_rdef(int nme, int fgx, LOGICAL begin)
{
  int def;
  int rdef;
  BV *inout;

  /* ensure that the variable is an "optimizable" symbol */

  if (!is_optsym(nme))
    return 0;

  inout = begin ? FG_IN(fgx) : FG_OUT(fgx);
  /*
   * if the node doesn't have a IN/OUT set (node added after flow analysis),
   * return "no def".
   */
  if (inout == NULL)
    return 0;
  /*
   * scan the defs for the induction variable for a single reaching
   * definition.
   */
  rdef = 0;
  for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
    if (bv_mem(inout, def)) {
      if (rdef) {
        rdef = 0;
        break;
      }
      rdef = def;
    }
  }

  return (rdef);
}

/*
 * determine if a symbol is live upon exit from a function/
 * subprogram.
 */
LOGICAL
is_sym_exit_live(int nme)
{
  int sym;

#if DEBUG
  assert(nme > 0 && nme <= nmeb.stg_avail, "is_sym_exit_live, bad nme", nme, 3);
#endif
  sym = NME_SYM(nme);
#if DEBUG
  assert(sym <= stb.stg_avail, "is_sym_exit_live, bad sym", nme, 3);
#endif
  if (IS_STATIC(sym) || IS_EXTERN(sym))
    return TRUE;
  if (SAVEG(sym))
    return TRUE;
  if (IS_DUM(sym) && INTENTG(sym) != INTENT_IN)
    return TRUE;
  /* if this is the result variable */
  if (RESULTG(sym))
    return TRUE;
  /* in a program that contains others, or in a contained program,
   * outer-block variables are live */
  if (gbl.internal && !INTERNALG(sym))
    return TRUE;
  if (SCG(sym) == SC_LOCAL && DINITG(sym) && (ADDRTKNG(sym) || ASSNG(sym)) &&
      gbl.rutype != RU_PROG)
    /*
     * if a local variable is data initialized, disallow it if it
     * has been stored; the thinking is that it falls into the
     * same category of a saved variable -- someday, may want
     * to override this if XBIT(124,0x80) is set (also expreg.c)
     */
    return TRUE;
  if (SCG(sym) == SC_BASED) {
    int s;
    for (s = MIDNUMG(sym); s; s = MIDNUMG(s))
      if (IS_DUM(s) || IS_STATIC(s) || IS_EXTERN(s) || DINITG(s) || SAVEG(s))
        return TRUE;
  }
  if (THREADG(sym)) /* not going to qualify any further */
    return TRUE;

  return FALSE;
}

/*
 * determine if a symbol is live because of implicit uses -- live upon
 * exit from a function, has its address taken, is volatile, etc.
 */
LOGICAL
is_sym_imp_live(int nme)
{
  int sym;

#if DEBUG
  assert(nme > 0 && nme <= nmeb.stg_avail, "is_sym_imp_live, bad nme", nme, 3);
#endif
  if (is_sym_exit_live(nme))
    return TRUE;
  sym = NME_SYM(nme);
  if (ADDRTKNG(sym) || VOLG(sym))
    return TRUE;
  if (ARGG(sym))
    return TRUE;
#ifdef PTRSTOREP
  if (PTRSTOREG(sym))
    return TRUE;
#endif
  if (!XBIT(19, 0x1) && SOCPTRG(sym)) /* noeqvchk => XBIT(19,0x1) set */
    return TRUE;
  if (SCG(sym) == SC_BASED && MIDNUMG(sym))
    return TRUE;
  if (PTRVG(sym))
    return TRUE;
  if (STYPEG(sym) == ST_VAR && HCCSYMG(sym) && !VCSYMG(sym))
    /* ...implicit uses of compiler-created bounds variables. */
    return TRUE;
  return FALSE;
}

/*
 * determine if a symbol is live upon entry to a function/
 * subprogram.
 */
LOGICAL
is_sym_entry_live(int nme)
{
  int sym;

#if DEBUG
  assert(nme > 0 && nme <= nmeb.stg_avail, "is_sym_entry_live, bad nme", nme,
         3);
#endif
  sym = NME_SYM(nme);
#if DEBUG
  assert(sym <= stb.stg_avail, "is_sym_entry_live, bad sym", nme, 3);
#endif
  if (IS_STATIC(sym) || IS_EXTERN(sym))
    return TRUE;
  if (IS_DUM(sym))
    return (INTENTG(sym) != INTENT_OUT);
  if (DINITG(sym) || SAVEG(sym))
    return TRUE;
  if (SCG(sym) == SC_BASED) {
    int s;
    for (s = MIDNUMG(sym); s; s = MIDNUMG(s))
      if (IS_DUM(s) || IS_STATIC(s) || IS_EXTERN(s) || DINITG(s) || SAVEG(s))
        return TRUE;
  }
  if (THREADG(sym)) /* not going to qualify any further */
    return TRUE;

  return FALSE;
}

LOGICAL
is_store_via_ptr(int astx)
{
  int nme;

  if (A_TYPEG(astx) != A_ASN)
    return FALSE;
  astx = A_DESTG(astx);
  if ((nme = A_NMEG(astx)) == 0)
    return FALSE;

  for (nme = A_NMEG(astx); TRUE; nme = NME_NM(nme)) {
    switch (NME_TYPE(nme)) {
    case NT_ARR:
    case NT_MEM:
      continue;
    case NT_VAR:
      if (POINTERG(NME_SYM(nme)))
        return (!is_ptr_safe(nme));
      return FALSE;
    case NT_IND:
      return (!is_ptr_safe(nme));
    default:
      break;
    }
    break;
  }

  return TRUE;
}

static struct {/* global info needed when checking expressions */
  struct {     /* defines start of path */
    int stmt;  /* ilt */
    int fg;    /* flowgraph node containing stmt */
  } start;
  struct {/* defines end of path */
    int stmt;
    int fg;
  } end;
  int eob; /* checking to end of block */
} srch_ae;

static int visit_list; /* list of nodes visited in in_path(),
                        * call_in_path() */

extern LOGICAL def_ok(int def, int fgx, int stmt);
extern LOGICAL is_avail_expr(int expr, int start_ilt, int start_fg, int end_ilt,
                             int end_fg);
static LOGICAL avail_expr(int expr);
static LOGICAL is_in_path(int fg);
static LOGICAL in_path(int cur, int fg);
static LOGICAL iscall_in_path(void);
static LOGICAL isptr_in_path(void);
static LOGICAL call_in_path(int cur);

/*
 * determine if it's safe to copy the value of the def to either the
 * beginning of the flowgraph node or to the end of the flowgraph
 * node. We've already determined that the def reaches the flowgraph node.
 */
LOGICAL
can_copy_def(int def, int fgx, LOGICAL begin)
{
  int end_ilt;

#if DEBUG
  assert(def, "can_copy_def: def is 0", 0, 3);
#endif
  if (OPTDBG(9, 16384))
    fprintf(gbl.dbgfil, "can_copy_def trace for %s, def %d, expr %d, to fg %d",
            getprint(basesym_of(DEF_NM(def))), def, DEF_RHS(def), fgx);
  if (begin)
    end_ilt = BIH_ILTFIRST(FG_TO_BIH(fgx));
  else
    end_ilt = BIH_ILTLAST(FG_TO_BIH(fgx));
  if (OPTDBG(9, 16384))
    fprintf(gbl.dbgfil, ", ilt %d\n", end_ilt);

  if (!def_ok(def, fgx, end_ilt))
    return FALSE;

  if (DEF_CONST(def)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "can copy const def %d\n", def);
    return TRUE;
  }

  if (XBIT(7, 1)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "can't copy def %d, inhibited\n", def);
    return FALSE;
  }

  if (DEF_DOINIT(def))
    srch_ae.start.stmt = FG_STDLAST(DEF_FG(def));
  else
    srch_ae.start.stmt = DEF_STD(def);
  srch_ae.start.fg = DEF_FG(def);
  srch_ae.end.stmt = end_ilt;
  srch_ae.end.fg = fgx;
  srch_ae.eob = TRUE;
  return (avail_expr((int)DEF_RHS(def)));
}

/*
 * determine if it's safe to copy the value of the def to a statement
 * by checking attributes about the def.
 * We've already determined that the def reaches the flowgraph node.
 */
LOGICAL
def_ok(int def, int fgx, int stmt)
{
  int nme;
  int sym;
  int def_fg, def_std;
  int iltx;

#if DEBUG
  assert(def, "def_ok: def is 0", 0, 3);
#endif
  if (OPTDBG(9, 16384))
    fprintf(gbl.dbgfil, "def_ok trace for def %d of %s\n", def,
            getprint(basesym_of(DEF_NM(def))));

  /* does the storing expr contain a use of the symbol being defined */

  if (DEF_SELF(def)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not ok, self.\n", def);
    return FALSE;
  }

  def_fg = DEF_FG(def);

  if (def_fg != fgx && BIH_CS(FG_TO_BIH(fgx))) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not ok, use in critical sec\n", def);
    return FALSE;
  }

  nme = DEF_NM(def);
  sym = NME_SYM(nme);
  if (is_sym_entry_live(nme) && !is_dominator(def_fg, fgx)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not ok, not dom.\n", def);
    return FALSE;
  }
  def_std = DEF_STD(def);
  if (STD_EX(def_std)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not, ilt_ex\n", def);
    return FALSE;
  }

  if (VOLG(sym)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not ok, sym %d VOL\n", def, sym);
    return FALSE;
  }
  if (!XBIT(19, 0x1) && SOCPTRG(sym)) { /* noeqvchk => XBIT(19,0x1) set */
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not ok, sym %d EQUIV\n", def, sym);
    return FALSE;
  }

  /* If sym is a common block variable (and potentially aliased), do not propagate
   * its definition, like how variables in an equivalent statement are handled.
   */
  if (SCG(sym) == SC_CMBLK && MODCMNG(CMBLKG(sym)) == 0) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not ok, sym %d is common block "
              "variable\n", def, sym);
    return FALSE;
  }

  if (FG_IN(fgx) == NULL) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "can't copy def %d, no in of fg %d\n", def, fgx);
    return FALSE;
  }

  /* does stmt precede the def?
   *
   * if def and stmt are in the same block, it's necessary to
   * scan backwards from the def.
   *
   * if they aren't in the same block, the dominator test is
   * sufficient.
   *
   */
  if (def_fg == fgx) {
    if (DEF_DOINIT(def))
      /* defs which initialize do variables never precedes a stmt */
      ;
    else
      for (iltx = (def_std); iltx; iltx = STD_PREV(iltx)) {
        if (iltx == stmt) {
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil, "def %d not ok, ilt %d precedes def\n", def,
                    stmt);
          return FALSE;
        }
      }
  } else if (!is_dominator(def_fg, fgx)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "def %d not ok, def does not dominator ilt %d\n", def,
              stmt);
    return FALSE;
  }

  /*
   * If variable is "call" unsafe, then:
   * 1. a call cannot exist between the def and the point.
   * 2. If the variable has its address taken, then a store via a ptr
   *    cannot exist between the def and the point.
   */
  if (!is_call_safe(nme)) {
    if (is_call_in_path(def_std, def_fg, stmt, fgx)) {
      if (OPTDBG(9, 16384))
        fprintf(gbl.dbgfil, "def %d not ok, call in path (%d, %d), (%d, %d)\n",
                def, def_fg, def_std, fgx, stmt);
      return FALSE;
    }
    if (!is_sym_ptrsafe(sym)) {
      if (is_ptr_in_path(def_std, def_fg, stmt, fgx)) {
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "def %d not ok, ptr in path (%d, %d), (%d, %d)\n",
                  def, def_fg, def_std, fgx, stmt);
        return FALSE;
      }
    }
  }

  return TRUE;
}

/*
 * determine if an expression is available along a path which is defined as
 * two points in a flowgraph node, (start_ilt, start_fg) and (end_ilt, end_ilt)
 * NOTES:
 * 1.  starting point is always the <fg, ilt> of a def
 * 2.  the end of the path "follows" the def (i.e., def_ok has been called
 *     to ensure that the end of the path does not precede the def).
 */
LOGICAL
is_avail_expr(int expr, int start_ilt, int start_fg, int end_ilt, int end_fg)
{

  if (OPTDBG(9, 16384))
    fprintf(gbl.dbgfil, "is_avail_expr(expr:%d, s:%d, sfg:%d, e:%d, efg:%d)\n",
            expr, start_ilt, start_fg, end_ilt, end_fg);

  if (A_TYPEG(expr) == A_CNST) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "const expr is avail.\n");
    return TRUE;
  }
  srch_ae.start.stmt = start_ilt;
  srch_ae.start.fg = start_fg;
  srch_ae.end.stmt = end_ilt;
  srch_ae.end.fg = end_fg;
  srch_ae.eob = FALSE;

  return (avail_expr(expr));
}

static LOGICAL _avail(int expr, LOGICAL *av_p);

/*
 * determine if an expression reaches the beginning or end of a block
 * starting at the end of a block containing a given point.
 * Note that this recurses thru the original expression passed to
 * is_avail_expr.

 * For each "variable" found in the expression, it's determined if its
 * value is available at the end of the path.
 */
static LOGICAL
avail_expr(int expr)
{
  LOGICAL av;
  ast_visit(1, 1);
  av = TRUE;
  ast_traverse(expr, _avail, NULL, &av);
  ast_unvisit();
  if (OPTDBG(9, 16384) && av)
    fprintf(gbl.dbgfil, "expr avail.\n");
  return av;
}

static LOGICAL
_avail(int expr, LOGICAL *av_p)
{
  int opc;
  int nme, sym;
  int i;
  int def;
  int iltx;

  if (!*av_p)
    return TRUE; /* don't continue if already not available */
  if (OPTDBG(9, 16384))
    fprintf(gbl.dbgfil, "_avail(%d)\n", expr);
  opc = A_TYPEG(expr);
  switch (opc) {
  case A_CNST:
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "const expr is avail.\n");
    return TRUE; /* stop */
  case A_ID:
    sym = A_SPTRG(expr);
    if (!ST_ISVAR(STYPEG(sym)))
      return TRUE; /* ignore ST_PROC, ST_INTRIN, etc. */
    nme = A_NMEG(expr);
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "avail_expr. nme %d, load %d\n", nme, expr);
    if (!is_optsym(nme)) {
      switch (NME_TYPE(nme)) {
      case NT_VAR:
        if ((SCG(NME_SYM(nme)) == SC_BASED || POINTERG(NME_SYM(nme))) &&
            isptr_in_path()) {
          *av_p = FALSE;
          return TRUE;
        }
        FLANG_FALLTHROUGH;
      case NT_ARR:
      case NT_MEM:
      case NT_UNK:
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "expr not avail %d, nme.\n", expr);
        *av_p = FALSE;
        return TRUE;
      case NT_IND:
        if (isptr_in_path()) {
          *av_p = FALSE;
          return TRUE;
        }
        break;
      default:
        interr("avail_expr: unk nme", expr, 3);
        break;
      }
      break; /* recurse */
    }
    /*
     * A load of an optimizable symbol has been found.
     * Step 1.  If variable is "call" unsafe, then:
     * 1. a call cannot be in the path defined from start to end
     *    (precise analysis).
     * 2. If the variable has its address taken, then a store via a ptr
     *    cannot be in the path defined from start to end
     */
    sym = NME_SYM(nme);
#ifdef RESHAPEDG
    if (SCG(sym) == SC_BASED && RESHAPEDG(sym)) {
      *av_p = FALSE;
      return TRUE;
    }
#endif
    if (!is_call_safe(nme)) {
      if (iscall_in_path()) {
        *av_p = FALSE;
        return TRUE;
      }
      if (ADDRTKNG(sym)) {
        if (isptr_in_path()) {
          *av_p = FALSE;
          return TRUE;
        }
      }
    }
    /*
     * Step2. determine if there are any defs of the variable after the
     * point in the flowgraph node to the end of the same node.
     * Note that if the path is in one block, we'll terminate the
     * search at the end of the path; otherwise, we check until the
     * end of the block.
     */
    if (srch_ae.eob)
      iltx = 0; /* ==> end of block */
    else if (srch_ae.start.fg == srch_ae.end.fg)
      iltx = srch_ae.end.stmt;
    else
      iltx = 0; /* ==> end of block */
                /*
                 * scan the defs in the node to find the stmt containing the expression.
                 * NOTE: works only because we've defined the point in the node to
                 * be a def of an optimizable symbol.
                 */
    def = FG_FDEF(srch_ae.start.fg);
    for (; def; def = DEF_LNEXT(def))
      if (DEF_STD(def) == srch_ae.start.stmt) {
        if (DEF_NM(def) == nme && DEF_SELF(def)) {
          /* At the starting point, there is a def which defines
           * the variable and includes itself; normally, this is
           * caught by def_ok(), but is_avail_expr() may be called
           * without first calling def_ok().
           */
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil, "expr not avail. DEF_SELF(%d) for nme %d\n",
                    def, nme);
          *av_p = FALSE;
          return TRUE;
        }
        break;
      }

    /* scan all the defs after the point */
    rdilts(FG_TO_BIH(srch_ae.start.fg));
    for (def = DEF_LNEXT(def); def; def = DEF_LNEXT(def))
      if (DEF_NM(def) == nme) {
        for (i = STD_NEXT(srch_ae.start.stmt);; i = STD_NEXT(i)) {
          if (i == iltx)
            break;
          if (DEF_STD(def) == i) {
            if (OPTDBG(9, 16384))
              fprintf(gbl.dbgfil, "expr not avail. def %d after\n", def);
            wrilts(FG_TO_BIH(srch_ae.start.fg));
            *av_p = FALSE;
            return TRUE;
          }
        }
      }
    wrilts(FG_TO_BIH(srch_ae.start.fg));

    /*
     * Step 3.  scan all defs of the variable. If one reaches the end
     * of the path, then the expression is unavailble.
     */
    for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
      if (DEF_FG(def) == srch_ae.start.fg)
        continue; /* def is in same block -- it must precede */
                  /*
                   * if def is in the flow graph node containing end point, we need
                   * to ensure that the def does not precede end.stmt.  WARNING:
                   * steps should have already been taken to ensure that start
                   * dominates end.
                   */
      if (DEF_FG(def) == srch_ae.end.fg) {
        if (DEF_DOINIT(def) && srch_ae.eob) {
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil, "expr not avail. dodef %d bef\n", def);
          *av_p = FALSE;
          return TRUE;
        }
        if (srch_ae.eob)
          iltx = 0; /* ==> end of block */
        else
          iltx = srch_ae.end.stmt;
        rdilts(FG_TO_BIH(srch_ae.end.fg));
        for (iltx = STD_PREV(iltx); iltx; iltx = STD_PREV(iltx))
          if (DEF_STD(def) == iltx) {
            if (OPTDBG(9, 16384))
              fprintf(gbl.dbgfil, "expr not avail. def %d bef\n", def);
            wrilts(FG_TO_BIH(srch_ae.end.fg));
            *av_p = FALSE;
            return TRUE;
          }
        wrilts(FG_TO_BIH(srch_ae.end.fg));
        continue;
      }
      if (is_dominator((int)DEF_FG(def), srch_ae.start.fg))
        continue; /* if def dominates point, it's ok */
      if (bv_mem(FG_IN(srch_ae.end.fg), def)) {
        if (srch_ae.start.fg == srch_ae.end.fg &&
            !bv_mem(FG_OUT(srch_ae.end.fg), def)) {
          /*
           * start and end are in the same block.  A def is in its
           * IN set but is killed.  check where the def is killed.
           * if it's at the start or before, then all is ok.
           */
          int d;
          d = FG_FDEF(srch_ae.end.fg);
          while (d) {
            if (DEF_NM(d) == nme) {
              for (iltx = srch_ae.start.stmt; iltx; iltx = STD_PREV(iltx)) {
                if (DEF_STD(d) == iltx)
                  goto next_def;
              }
            }
            d = DEF_LNEXT(d);
          }
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil, "expr not avail. def %d IN.1\n", def);
          *av_p = FALSE;
          return TRUE;
        }
        /*
         * although def is in IN(end), is it in the path (start, end).
         */
        if (is_dominator(srch_ae.start.fg, srch_ae.end.fg) &&
            !is_in_path((int)DEF_FG(def)))
          ;
        else {
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil, "expr not avail. def %d IN.2\n", def);
          *av_p = FALSE;
          return TRUE;
        }
      }
    next_def:;
    }

    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "expr avail.\n");
    return (TRUE);

  case A_CALL:
  case A_FUNC:
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "expr %d not avail. contains call\n", expr);
    *av_p = FALSE;
    return TRUE;

  default:
    break;
  }

  return FALSE; /* recurse on expr's operands */
}

/*
 * is node fg in the path (srch_ae.start.fg, srch_ae.end.fg) where
 * start dominates end.
 */
static LOGICAL
is_in_path(int fg)
{
  LOGICAL ret;

  if (srch_ae.start.fg == fg)
    return TRUE;

  /* use the "natnxt" field to link together nodes visited during in_path */
  visit_list = srch_ae.start.fg;
  FG_NATNXT(visit_list) = 0;
  FG_VISITED(srch_ae.start.fg) = 1;

  ret = in_path(srch_ae.end.fg, fg);

  /*  unvisit the nodes visited during in_path */
  while (visit_list) {
    FG_VISITED(visit_list) = 0;
    visit_list = FG_NATNXT(visit_list);
  }

  return ret;
}

static LOGICAL
in_path(int cur, int fg)
{
  PSI_P pred;
  int pred_fg;

  if (FG_VISITED(cur))
    return FALSE;
  if (cur == fg)
    return TRUE;

  FG_VISITED(cur) = 1;
  FG_NATNXT(cur) = visit_list;
  visit_list = cur;

  for (pred = FG_PRED(cur); pred != PSI_P_NULL; pred = PSI_NEXT(pred)) {
    pred_fg = PSI_NODE(pred);
    if (in_path(pred_fg, fg))
      return TRUE;
  }
  return FALSE;
}

/*
 * is a call in the path <srch_ae.start.fg, srch_ae.start.stmt>,
 * <srch_ae.end.fg, srch_ae.end.stmt>.
 */
static LOGICAL
iscall_in_path(void)
{
  int iltx;
  int save_ex;
  LOGICAL ret;
  int term_std;
  /*
   * examine the ilts from the start stmt to the end of the start node.
   */
  if (srch_ae.start.fg == srch_ae.end.fg) {
    /* nodes are the same; check ilts after the start and before the
     * end stmt.
     */
    if (srch_ae.start.stmt == srch_ae.end.stmt)
      return FALSE;
    iltx = STD_NEXT(srch_ae.start.stmt);
    while (iltx) { /* WARNING: end.stmt could have been deleted */
      if (iltx == srch_ae.end.stmt)
        break;
      if (STD_EX(iltx)) {
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "call is in path, after def in fg %d, ilt %d\n",
                  srch_ae.start.fg, iltx);
        return TRUE;
      }
      iltx = STD_NEXT(iltx);
    }
    return FALSE;
  }
  /*
   * nodes are not the same; check ilts after the start and thru the
   * end of the block.
   */
  iltx = srch_ae.start.stmt;
  term_std = FG_STDLAST(srch_ae.start.fg);
  while (term_std != iltx) {
    iltx = STD_NEXT(iltx);
    if (iltx == 0)
      break;
    if (STD_EX(iltx)) {
      if (OPTDBG(9, 16384))
        fprintf(gbl.dbgfil, "call is in path, after def in fg %d, ilt %d\n",
                srch_ae.start.fg, iltx);
      return TRUE;
    }
  }

  /* nodes are different, examine ilts between start of the end node and
   * the end statement.
   */
  term_std = FG_STDFIRST(srch_ae.end.fg);
  for (iltx = (srch_ae.end.stmt); iltx; iltx = STD_PREV(iltx)) {
    if (STD_EX(iltx)) {
      if (OPTDBG(9, 16384))
        fprintf(gbl.dbgfil, "call is in path, before end of fg %d, ilt %d\n",
                srch_ae.start.fg, iltx);
      return TRUE;
    }
    if (iltx == term_std)
      break;
  }

  /* if start does not dominate end, check if a call (the call "def")
   * reaches then end node.
   */
  if (!is_dominator(srch_ae.start.fg, srch_ae.end.fg)) {
    if (bv_mem(FG_IN(srch_ae.end.fg), CALL_DEF)) {
      if (OPTDBG(9, 16384))
        fprintf(gbl.dbgfil, "call is in path, calldef IN fg %d\n",
                srch_ae.end.fg);
      return TRUE;
    }
    return FALSE;
  }

  /*
   * nodes are not the same & start dominates the end.
   * if the end node is in a loop and start is NOT in the same loop
   * then it's necessary to check the block's * external flag.
   * Even if the call exists after the end stmt, it can
   * still reach the statement.
   */
  if (FG_LOOP(srch_ae.end.fg) &&
      FG_LOOP(srch_ae.end.fg) != FG_LOOP(srch_ae.start.fg) &&
      BIH_EX(FG_TO_BIH(srch_ae.end.fg))) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "call is in path, fg %d in loop %d contains call\n",
              srch_ae.end.fg, FG_LOOP(srch_ae.end.fg));
    return TRUE;
  }

  /* traverse the path start to end bottom-up, checking if a call exists
   * in each node.  To setup, start's visit flag is set (=> if a call exists
   * at the beginning of the block, it will be ignored), and the "EX" flag
   * of the end's bih must be cleared then restored.
   */
  /* use the "natnxt" field to link together nodes visited during in_path */
  visit_list = srch_ae.start.fg;
  FG_NATNXT(visit_list) = 0;
  FG_VISITED(srch_ae.start.fg) = 1;
  save_ex = BIH_EX(FG_TO_BIH(srch_ae.end.fg));
  BIH_EX(FG_TO_BIH(srch_ae.end.fg)) = 0;

  ret = call_in_path(srch_ae.end.fg);

  /*  unvisit the nodes visited during in_path */
  while (visit_list) {
    FG_VISITED(visit_list) = 0;
    visit_list = FG_NATNXT(visit_list);
  }

  BIH_EX(FG_TO_BIH(srch_ae.end.fg)) = save_ex;

  return ret;
}

static LOGICAL
call_in_path(int cur)
{
  PSI_P pred;
  int pred_fg;

  if (FG_VISITED(cur))
    return FALSE;
  if (BIH_EX(FG_TO_BIH(cur)))
    return TRUE;

  FG_VISITED(cur) = 1;
  FG_NATNXT(cur) = visit_list;
  visit_list = cur;

  for (pred = FG_PRED(cur); pred != PSI_P_NULL; pred = PSI_NEXT(pred)) {
    pred_fg = PSI_NODE(pred);
    if (call_in_path(pred_fg)) {
      if (OPTDBG(9, 16384))
        fprintf(gbl.dbgfil, "call is in path, call in fg %d\n", pred_fg);
      return TRUE;
    }
  }
  return FALSE;
}

/*
 * external version is iscall_in_path(). sets up search structure (srch_ae).
 * for iscall_in_path().
 */
LOGICAL
is_call_in_path(int start_ilt, int start_fg, int end_ilt, int end_fg)
{

  srch_ae.start.stmt = start_ilt;
  srch_ae.start.fg = start_fg;
  srch_ae.end.stmt = end_ilt;
  srch_ae.end.fg = end_fg;

  return (iscall_in_path());
}

/*
 * is a ptr store in the path <srch_ae.start.fg, srch_ae.start.stmt>,
 * <srch_ae.end.fg, srch_ae.end.stmt>.
 */
static LOGICAL
isptr_in_path(void)
{
  int iltx;
  int term_std;
  /*
   * examine the ilts from the start stmt to the end of the start node.
   * only search if there exists a store via a pointer in the node.
   */
  if (srch_ae.start.fg == srch_ae.end.fg) {
    if (srch_ae.start.stmt == srch_ae.end.stmt)
      return FALSE;
    if (FG_PTR_STORE(srch_ae.start.fg)) {
      /* nodes are the same; check ilts after the start and before the
       * end stmt.
       */
      iltx = STD_NEXT(srch_ae.start.stmt);
      while (iltx) { /* WARNING: end.stmt could have been deleted */
        if (iltx == srch_ae.end.stmt)
          break;
        if (is_store_via_ptr((int)STD_AST(iltx))) {
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil, "ptr in path, ptr def in start fg %d\n",
                    srch_ae.end.fg);
          return TRUE;
        }
        iltx = STD_NEXT(iltx);
      }
    }
    return FALSE;
  }

  if (FG_PTR_STORE(srch_ae.start.fg)) {
    /* nodes are not the same; check ilts after the start and thru the
     * end of the block.
     */
    iltx = srch_ae.start.stmt;
    term_std = FG_STDLAST(srch_ae.start.fg);
    while (term_std != iltx) {
      iltx = STD_NEXT(iltx);
      if (iltx == 0)
        break;
      if (is_store_via_ptr((int)STD_AST(iltx))) {
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "ptr in path, ptr def in start fg %d\n",
                  srch_ae.end.fg);
        return TRUE;
      }
    }
  }
  /*
   * nodes are different. the "ptr store" def cannot reach the
   * beginning of the end node (i.e., it's not a member of the node's
   * IN set).
   */
  if (bv_mem(FG_IN(srch_ae.end.fg), PTR_STORE_DEF)) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "ptr in path, ptr def IN\n");
    return TRUE;
  }
  /*
   * nodes are different. examine ilts between the start of the end
   * node and the end statement, inclusive.
   * only search if there exists a store via a pointer in the node.
   */
  if (FG_PTR_STORE(srch_ae.end.fg)) {
    term_std = FG_STDFIRST(srch_ae.end.fg);
    for (iltx = (srch_ae.end.stmt); iltx; iltx = STD_PREV(iltx)) {
      if (is_store_via_ptr((int)STD_AST(iltx))) {
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "ptr in path, ptr def in end fg %d\n",
                  srch_ae.end.fg);
        return TRUE;
      }
      if (iltx == term_std)
        break;
    }
  }

  return FALSE;
}

/*
 * external version is isptr_in_path(). sets up search structure (srch_ae).
 * for isptr_in_path().
 */
LOGICAL
is_ptr_in_path(int start_ilt, int start_fg, int end_ilt, int end_fg)
{

  srch_ae.start.stmt = start_ilt;
  srch_ae.start.fg = start_fg;
  srch_ae.end.stmt = end_ilt;
  srch_ae.end.fg = end_fg;

  return (isptr_in_path());
}

/*
 * determine if a use has a single reaching def which can be copied
 */
LOGICAL
single_ud(int use)
{
  UD *ud;
  int def;
  int nme;
  int use_fg;
  int use_std;

#if DEBUG
  assert(use > 0 && use <= opt.useb.stg_avail, "single_ud: bad use", use, 3);
#endif
  nme = USE_NM(use);
  use_fg = USE_FG(use);
  use_std = USE_STD(use);
  if (OPTDBG(9, 16384))
    fprintf(gbl.dbgfil, "single_ud trace for %s, use %d, ilt %d, to fg %d\n",
            getprint(basesym_of(nme)), use, use_std, use_fg);

  if ((ud = USE_UD(use)) == NULL || ud->next != NULL) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "single_ud, mult/0 reaching defs\n");
    return FALSE;
  }

  def = ud->def;
  if (!def_ok(def, use_fg, use_std))
    return FALSE;

  return (is_avail_expr((int)DEF_RHS(def), (int)DEF_STD(def), (int)DEF_FG(def),
                        use_std, use_fg));
}

/*
 * determine if a use has a single reaching def.
 * like single_ud except only lhs must be avail, not rhs
 * (since we're not copying it).
 */
LOGICAL
only_one_ud(int use)
{
  UD *ud;
  int def;
  int nme;
  int use_fg;
  int use_std;
  int t;

#if DEBUG
  assert(use > 0 && use <= opt.useb.stg_avail, "only_one_ud: bad use", use, 3);
#endif
  nme = USE_NM(use);
  use_fg = USE_FG(use);
  use_std = USE_STD(use);
  if ((ud = USE_UD(use)) == NULL || ud->next != NULL) {
    if (OPTDBG(9, 16384))
      fprintf(gbl.dbgfil, "only_one_ud, mult/0 reaching defs\n");
    return FALSE;
  }

  def = ud->def;
  if (OPTDBG(9, 16384))
    fprintf(gbl.dbgfil, "only_one_ud trace for %s, use %d, ilt %d, to fg %d\n",
            getprint(basesym_of(nme)), use, use_std, use_fg);
  /* HACK HACK! */
  if (0 != (t = DEF_SELF(def))) {
    DU *du;
    for (du = DEF_DU(def); du != NULL; du = du->next)
      if (USE_STD(du->use) == DEF_STD(def))
        return FALSE;
  }
  DEF_SELF(def) = 0;
  if (!def_ok(def, use_fg, use_std)) {
    DEF_SELF(def) = t;
    return FALSE;
  }
  DEF_SELF(def) = t;
  return (is_avail_expr((int)USE_AST(use), (int)DEF_STD(def), (int)DEF_FG(def),
                        use_std, use_fg));
}

LOGICAL
is_def_imp_live(int def)
{
  int nme, sym;

#if DEBUG
  assert(def && def < opt.defb.stg_avail, "is_def_imp_live: bad def", def, 3);
#endif
  /*
   * first, if the def is of a symbol which is live upon exit (i.e., due to
   * storage class), it's live at exit if it's a member of the exit's
   * IN set.
   */
  nme = DEF_NM(def);
  if (is_sym_exit_live(nme)) {
    BV *in_exit;

    in_exit = FG_IN(opt.exitfg);
    if (in_exit == NULL) {
      /* situation arises if the function can't exit, i.e., an
       * infinite loop which dominates the exit
       */
      if (OPTDBG(9, 8192))
        fprintf(gbl.dbgfil,
                "is_def_imp_live - can't exit, def %d is dead at exit\n", def);
      return FALSE;
    }
    if (bv_mem(in_exit, def)) {
      if (OPTDBG(9, 8192))
        fprintf(gbl.dbgfil, "is_def_imp_live - def %d is live at exit\n", def);
      return TRUE;
    }
  }
  if (is_sym_exit_live(nme) && bv_mem(FG_IN(opt.exitfg), def)) {
    if (OPTDBG(9, 8192))
      fprintf(gbl.dbgfil, "is_def_imp_live - def %d is live at exit\n", def);
    return TRUE;
  }
  /*
   * second, check a few attributes about the symbol (see the second
   * part of is_sym_imp_live).
   */
  sym = NME_SYM(nme);
  if (ADDRTKNG(sym) || VOLG(sym))
    return TRUE;
#ifdef PTRSTOREP
  if (PTRSTOREG(sym))
    return TRUE;
#endif
  if (!XBIT(19, 0x1) && SOCPTRG(sym))
    return TRUE;
  return FALSE;
}

/*
 * a def's uses have been removed from a loop. check if there are any
 * other uses of the def; if none and it meets other criteria, delete
 * the definition.
 */
void
rm_def_rloop(int def, int lpx)
{
  int def_fg, def_std;
  int use;
  int count; /* # of uses of def in same block */
  int i;
  DU *du;

#if DEBUG
  assert(def && def < opt.defb.stg_avail, "rm_def_rloop: bad def", def, 3);
  assert(lpx, "rm_def_rloop: lpx", 0, 3);
#endif
  /*
   * first, if the def is of a symbol which is live upon exit (i.e., due to
   * storage class), it can't be deleted if it reaches the exit from the
   * function.
   */
  if (is_def_imp_live(def)) {
    if (OPTDBG(9, 8192))
      fprintf(gbl.dbgfil, "rm_def_rloop - def %d is live at exit\n", def);
    return;
  }

  def_fg = DEF_FG(def);
  def_std = DEF_STD(def);
  count = 0;
  for (du = DEF_DU(def); du != NULL; du = du->next) {
    int use_fg;
    use_fg = USE_FG(use = du->use);
    if (FG_LOOP(use_fg) == lpx)
      continue;
    if (use_fg != def_fg) {
      if (OPTDBG(9, 8192))
        fprintf(gbl.dbgfil, "rm_def_rloop - def %d has other uses\n", def);
      return;
    }
    /* use in same block as def: scan ilts before use, searching for def */
    for (i = (USE_STD(use)); i; i = STD_PREV(i)) {
      if (i == def_std) {
        count++;
        goto next_use;
      }
    }
    if (OPTDBG(9, 8192))
      fprintf(gbl.dbgfil, "rm_def_rloop - def %d has uses bef\n", def);
    return;
  next_use:;
  }

  if (count) {
    STD_DELETE(def_std) = 1;
    DEF_DELETE(def) = 1;
    if (OPTDBG(9, 8192))
      fprintf(gbl.dbgfil, "rm_def_rloop - def %d, ilt %d marked deleted\n", def,
              def_std);
    return;
  }

  DEF_DELETE(def) = 1;
  unlnkilt(def_std, (int)FG_TO_BIH(def_fg), FALSE);
  if (OPTDBG(9, 8192))
    fprintf(gbl.dbgfil, "rm_def_rloop - def %d, ilt %d deleted\n", def,
            def_std);
}

/* structure to aid copy propagation */

static struct {
  int lp;
  int new;
  int *stg_base; /* table of candidate loads to replace */
  int stg_size;
  int stg_avail;
} copyb;

static LOGICAL collect_loads(int, int *);
static LOGICAL cp_loop(int);

/*
 * propagate the definitions of any variables in 'expr' which is
 * considered to be at the point immediately preceding the loop.
 */
int
copy_to_loop(int tree, int lp)
{
  int i;
  LOGICAL changes;

  copyb.lp = lp;
  copyb.new = tree;
  if (OPTDBG(9, 65536)) {
    fprintf(gbl.dbgfil, "copy_to_loop %d, ast %d:\n", lp, tree);
    dbg_print_ast(tree, gbl.dbgfil);
  }
  OPT_ALLOC(copyb, int, 32);
  while (TRUE) {
    /*
     * Recursively search 'tree' and collect all loads which are
     * candidates for copy propagation.
     */
    copyb.stg_avail = 0;
    ast_visit(1, 1);
    ast_traverse(copyb.new, collect_loads, NULL, NULL);
    ast_unvisit();
    changes = FALSE;
    /*
     * For all candidate loads, attempt to copy their values.  cp_loop()
     * updates copyb.new if replacement occurs.
     */
    for (i = 0; i < copyb.stg_avail; i++) {
      if (cp_loop(copyb.stg_base[i])) {
        if (OPTDBG(9, 65536)) {
          fprintf(gbl.dbgfil, "recur copy_to_loop %d, ast: %d\n", lp,
                  copyb.new);
          dbg_print_ast(copyb.new, gbl.dbgfil);
        }
        changes = TRUE;
      }
    }
    if (!changes)
      break;
  }

  OPT_FREE(copyb);
  return copyb.new;
}

static LOGICAL
collect_loads(int expr, int *dummy)
{
  int opc;
  int nme;
  int i;

  opc = A_TYPEG(expr);
  switch (opc) {
  case A_ID:
  case A_MEM:
  case A_SUBSCR:
    nme = A_NMEG(expr);
    if (OPTDBG(9, 65536))
      fprintf(gbl.dbgfil, "    collect_loads. nme %d, load %d\n", nme, expr);
    if (is_optsym(nme)) {
      i = copyb.stg_avail++;
      OPT_NEED(copyb, int, 32);
      copyb.stg_base[i] = expr;
    }
    return TRUE; /* stop traversal */

  default:
    break;
  }

  return FALSE; /* continue to traverse */
}

static LOGICAL
cp_loop(int expr)
{
  int nme;
  int i;
  int rdef;
  PSI_P pred;

  nme = A_NMEG(expr);
  if (OPTDBG(9, 65536))
    fprintf(gbl.dbgfil, "    cp_loop. nme %d, load %d\n", nme, expr);
  /*
   * A load of an optimizable symbol has been found.
   * for the predecessors of the loop, determine if there is one
   * and only one def of the variable which reaches the end of
   * the predecessors.  This def must be live for all paths to the
   * predecessors.
   */
  rdef = 0;
  pred = FG_PRED(LP_HEAD(copyb.lp));
  for (; pred != PSI_P_NULL; pred = PSI_NEXT(pred)) {
    if (FG_LOOP(PSI_NODE(pred)) == copyb.lp)
      continue;
    i = find_rdef(nme, PSI_NODE(pred), FALSE /* end of block */);
    if (i == 0) {
      rdef = 0;
      if (OPTDBG(9, 65536))
        fprintf(gbl.dbgfil, "    cp_loop: rdef nfnd to end of fg %d\n",
                PSI_NODE(pred));
      break;
    }
    if (rdef) {
      if (i != rdef) {
        if (OPTDBG(9, 65536))
          fprintf(gbl.dbgfil, "    cp_loop: mult rdefs fnd, %d & %d\n", rdef,
                  i);
        rdef = 0;
        break;
      }
    } else
      rdef = i;

    if (!can_copy_def(rdef, PSI_NODE(pred), FALSE /* end of block */)) {
      if (OPTDBG(9, 65536))
        fprintf(gbl.dbgfil,
                "    cp_loop: def %d cannot be copied to end of %d\n", rdef,
                PSI_NODE(pred));
      rdef = 0;
      break;
    }
  }
  if (rdef) {
    if (OPTDBG(9, 65536))
      fprintf(gbl.dbgfil, "    cp_loop: copy def %d\n", rdef);
    i = DEF_RHS(rdef);
    if (A_TYPEG(i) == A_CONV) {
      switch (DTY(A_DTYPEG(i))) {
      case TY_INT:
      case TY_SINT:
      case TY_BINT:
        if (OPTDBG(9, 65536))
          fprintf(gbl.dbgfil, "    cp_loop: def %d - I_INT(rhs %d)\n", rdef, i);
        i = ast_intr(I_INT, DT_INT, 1, A_LOPG(i));
        break;
      case TY_REAL:
        if (OPTDBG(9, 65536))
          fprintf(gbl.dbgfil, "    cp_loop: def %d - I_REAL(rhs %d)\n", rdef,
                  i);
        i = ast_intr(I_REAL, DT_REAL4, 1, A_LOPG(i));
        break;
      case TY_DBLE:
        if (OPTDBG(9, 65536))
          fprintf(gbl.dbgfil, "    cp_loop: def %d - I_DBLE(rhs %d)\n", rdef,
                  i);
        i = ast_intr(I_DBLE, DT_DBLE, 1, A_LOPG(i));
        break;
      default:
        if (OPTDBG(9, 65536))
          fprintf(gbl.dbgfil,
                  "    cp_loop: def %d cannot be copied - ugly A_CONV %d\n",
                  rdef, i);
        return FALSE;
      }
    }
    if (OPTDBG(9, 65536))
      fprintf(gbl.dbgfil, "    cp_loop: replace ast %d with %d\n", expr, i);
    ast_visit(1, 1);
    ast_replace(expr, i);
    ast_rewrite(copyb.new);
    copyb.new = A_REPLG(copyb.new);
    ast_unvisit();
    return TRUE; /* replacement occurred */
  }
  return FALSE; /* replacement did not occur */
}

typedef struct ptrnmestr {
  int nme;
  int std;     /* unused */
  int isdummy; /* is dummy or global */
} ptrdefstr;

#define MAX_DEF 10
static ptrdefstr lhsdefs[MAX_DEF];
static int lhscount = 0;

/* if def == 0, then start at NME_DEF
 * if def == x, then start at DEF_NEXT(def)
 */
int
find_next_reaching_def(int nme, int fgx, int def)
{
  int nextdef;
  BV *inout;

  if (def) {
    nextdef = DEF_NEXT(def);
  } else {
    nextdef = NME_DEF(nme);
  }

  /* find def reaching this fg */
  inout = FG_IN(fgx);

  for (; nextdef; nextdef = DEF_NEXT(nextdef)) {
    if (bv_mem(inout, nextdef)) {
      return nextdef;
    }
  }
  return 0;
}

static int
ast_grandparent(int astptr)
{
  while (1) {
    switch (A_TYPEG(astptr)) {
    case A_SUBSCR:
      astptr = A_LOPG(astptr);
      break;
    case A_ID:
      return astptr;
    case A_MEM:
      astptr = A_PARENTG(astptr);
      break;
    default:
      return astptr;
    }
  }
  return astptr;
}

static int
get_next_parent(int astptr, int myparent)
{
  int currast;
  while (1) {
    switch (A_TYPEG(astptr)) {
    case A_SUBSCR:
      astptr = A_LOPG(astptr);
      break;
    case A_ID:
      return astptr;
      break;
    case A_MEM:
      currast = A_PARENTG(astptr);
      if (A_TYPEG(currast) == A_SUBSCR) {
        currast = A_LOPG(currast);
      }
      if (currast == myparent)
        return astptr;
      else
        astptr = A_PARENTG(astptr);
      break;
    default:
      return astptr;
      break;
    }
  }
  return astptr;
}

#ifdef FLANG_OPTUTIL_UNUSED
/*  All of following should consider true
 *  astptr is                          p%p1%p2%p3
 *  astx may be one of the following:  p, p%p1, p%p1, p%p1%p2%p3
 *  Currently ignore subscript.
 */
static LOGICAL
is_this_astptr(int astptr, int astx, int std)
{
  int i = 0;
  int astptr_p, astx_p;

  /* strip off subscript so that we can use it to break a while loop */
  if (A_TYPEG(astptr) == A_SUBSCR)
    astptr = A_LOPG(astptr);
  if (A_TYPEG(astx) == A_SUBSCR)
    astx = A_LOPG(astx);

  /* top level parent */
  astptr_p = ast_grandparent(astptr);
  astx_p = ast_grandparent(astx);

  if (astptr_p != astx_p)
    return FALSE;

  while (1) {
    ++i;
    if (i > 100) {
#if DEBUG
      interr("is_this_astptr: infinite loop ", astptr, 3);
#endif
      return FALSE;
    }
    astptr_p = get_next_parent(astptr, astptr_p);
    astx_p = get_next_parent(astx, astx_p);

    /* member must be the same */
    if (A_TYPEG(astptr_p) == A_MEM && A_TYPEG(astx_p) == A_MEM) {
      if (A_MEMG(astptr_p) != A_MEMG(astx_p))
        return FALSE;
    }

    /* reach the leaf ast */
    if (astptr_p == astptr || astx_p == astx)
      return TRUE;

    if (astptr_p != astx_p)
      return FALSE;
  }

  return FALSE;
}
#endif

/*
 *  it check if astc is a child of astp, ignoring subscript
 *  the expression of astp must be equeal or shorter than astc
 *  For example:
 *  astp:    a%b%c%d, a%b, a%b%c  true for following ast expr
 *  astc:    a%b%c%d, a%b%c%d%p
 */
static LOGICAL
is_parentof_ast(int astp, int astc)
{
  int i = 0;
  int astp_p, astc_p;

  /* strip off subscript so that we can use it to break a while loop */
  if (A_TYPEG(astp) == A_SUBSCR)
    astp = A_LOPG(astp);
  if (A_TYPEG(astc) == A_SUBSCR)
    astc = A_LOPG(astc);

  /* top level parent */
  astp_p = ast_grandparent(astp);
  astc_p = ast_grandparent(astc);

  if (astp_p != astc_p)
    return FALSE;

  if (A_TYPEG(astp) == A_ID) /* astp is top most parent */
    return TRUE;
  else if (A_TYPEG(astc) == A_ID) /* astc is top most parent */
    return FALSE;

  while (1) {
    ++i;
    if (i > 100) {
#if DEBUG
      interr("is_parentof_ast: infinite loop ", astp, 3);
#endif
      return FALSE;
    }

    astp_p = get_next_parent(astp, astp_p);
    astc_p = get_next_parent(astc, astc_p);

    /* member must be the same */
    if (A_TYPEG(astp_p) == A_MEM && A_TYPEG(astc_p) == A_MEM) {
      if (A_MEMG(astp_p) != A_MEMG(astc_p))
        return FALSE;
      else if (astp_p != astp && astc_p != astc)
        continue;
    }

    /* reach the leaf ast, astp covers ast */
    if (astp_p == astp) {
      int p_mem, c_mem;
#if DEBUG
      /* expect a mem ast */
      if (A_TYPEG(astp_p) != A_MEM || A_TYPEG(astc_p) != A_MEM) {
        interr("is_parentof_ast: expect member ast ", astp_p, 3);
      }
#endif
      c_mem = A_MEMG(astp_p);
      p_mem = A_MEMG(astc_p);
      if (c_mem != p_mem)
        return FALSE;

      return TRUE;
    }

    /* astp expression is longer than ast expression */
    if (astc_p == astc)
      return FALSE;

    if (astp_p != astc_p)
      return FALSE;
  }

  return FALSE;
}

/* Return the base nme of ast
 * Note: this routine can be expanded to handle other type of ast
 */
int
nme_of_ast(int ast)
{
  while (1) {
    switch (A_TYPEG(ast)) {
    case A_ID:
      return basenme_of(A_NMEG(ast));
    case A_SUBSCR:
    case A_SUBSTR:
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      ast = A_PARENTG(ast);
      break;
    default:
      return 0;
    }
  }
  return 0;
}

/* ast is a func call, ptr is an ast of lhs pointer
 * up to this point, descriptor has been added but the arguments has
 * not been rearranged, it is in the foram or
 * sub(ptr1, ptr2, ,ptr3, ptr1$sd, ptr2$sd, ptr3$sd) */
static LOGICAL
is_ptrast_arg(int ptrast, int ast)
{
  int i, nargs, argt, ele = 0, sptr, astx;
  int iface, entry, dscptr, fval, paramcnt;
  int inface_arg = 0;

  switch (A_TYPEG(ast)) {
  case A_CALL:
  case A_FUNC:
    nargs = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    break;
  default:
#if DEBUG
    interr("is_astptr_arg: expect call  ", ptrast, 3);
#endif
    return TRUE;
  }

  entry = procsym_of_ast(A_LOPG(ast));
  proc_arginfo(entry, NULL, &dscptr, &iface);
  if (iface && PUREG(iface))
    return TRUE;
  if (A_TYPEG(ast) == A_INTR && INKINDG(entry == IK_ELEMENTAL))
    return FALSE;

  /*
      if (!is_procedure_ptr(entry)) {
          is_ent = 1;
      } else {
          is_ent = 0;
      }
      if (is_ent && NODESCG(entry))
          return FALSE;
  */

  /* don't handle type bound procedure for now */
  if (STYPEG(entry) == ST_MEMBER && CLASSG(entry) && CCSYMG(entry) &&
      VTABLEG(entry) && NOPASSG(entry))
    return TRUE;

  /* get number of parameter */
  fval = A_SPTRG(A_LOPG(ast));
  paramcnt = PARAMCTG(fval);

  if (!dscptr) {
    for (i = 0; i < nargs && i < paramcnt; ++i) {
      if (DTY(DDTG(A_DTYPEG(ele))) == TY_DERIVED)
        if (is_parentof_ast(ele, ptrast))
          return TRUE;
    }
    return FALSE;
  }

  for (i = 0; i < nargs && i < paramcnt; ++i) {
    inface_arg = aux.dpdsc_base[dscptr + i];
    if (inface_arg) {
      ele = ARGT_ARG(argt, i);
      if (ele == 0)
        continue;
      if (POINTERG(inface_arg)) {
        switch (A_TYPEG(ele)) {
        case A_ID:
          sptr = memsym_of_ast(ele);
          if (CLASSG(sptr) && VTABLEG(sptr) && BINDG(sptr))
            sptr = pass_sym_of_ast(ele);
          if (STYPEG(sptr) == ST_PROC)
            break;
          astx = ptrast;
          if (A_TYPEG(ptrast) == A_SUBSCR)
            astx = A_LOPG(ptrast);
          if (memsym_of_ast(astx) == sptr)
            return FALSE; /* not safe */
          FLANG_FALLTHROUGH;
        case A_MEM:
          if (is_parentof_ast(ele, ptrast))
            return FALSE;
          FLANG_FALLTHROUGH;
        case A_SUBSCR:
        default:
          break;
        }
      } else {
        if (DTY(DDTG(A_DTYPEG(ele))) == TY_DERIVED)
          if (is_parentof_ast(ele, ptrast))
            return TRUE;
      }
    }
  }
  return FALSE;
}

#ifdef FLANG_OPTUTIL_UNUSED
/* 1) a=>b return 1
 * 2) call(a) return 2
 * 3) all else return 0
 */
static int
isstd_ptrdef(int std)
{
  int astx = STD_AST(std);
  if (A_TYPEG(astx) == A_ICALL) {
    if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
      return 1;
    }
  } else if (A_TYPEG(astx) == A_CALL || A_TYPEG(astx) == A_FUNC) {
    return 2; /* This is def std for pointer */
  }
  return 0;
}
#endif

/*
 * is a ptr def in the path <srch_ae.start.fg, srch_ae.start.stmt>,
 * <srch_ae.end.fg, srch_ae.end.stmt>.
 * If nme is 0, return if there is any ptr def in the path
 * to do: member_ast, only for member - check if for this particular member
 *        assume ast to be in the form of a%b%x...  (can be subscript)
 *        the nme must be the parent nme of member_ast.
 */
static LOGICAL
isptrdef_in_path(int nme, int ptrast)
{
  int iltx;
  int term_std;
  int astx;
  /*
   * examine the ilts from the start stmt to the end of the start node.
   * only search if there exists a store via a pointer in the node.
   */
  if (srch_ae.start.fg == srch_ae.end.fg) {
    if (srch_ae.start.stmt == srch_ae.end.stmt)
      return FALSE;
    iltx = STD_NEXT(srch_ae.start.stmt);
    while (iltx) { /* WARNING: end.stmt could have been deleted */
      if (iltx == srch_ae.end.stmt)
        break;
      astx = STD_AST(iltx);
      switch (A_TYPEG(astx)) {
      case A_ICALL:
        if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
          int args, lastx, rastx;
          args = A_ARGSG(astx);
          lastx = ARGT_ARG(args, 0);
          rastx = ARGT_ARG(args, 2);
          if (nme) {
            /* being conservative, for derived type-consider the nme
             * of parent instead of looking at the particular member
             * It is on to-do-list to check on this particular member.
             */
            if (nme_of_ast(astx) != nme)
              break;
          }
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil, "ptrdef in path, ptr def in start fg %d\n",
                    srch_ae.end.fg);
          return TRUE;
        }
        break;
      case A_CALL:
      case A_FUNC:
        if (is_ptrast_arg(ptrast, astx)) {
          if (OPTDBG(9, 16384))
            fprintf(gbl.dbgfil,
                    "ptrdef in path -- arg, ptr def in start fg %d\n",
                    srch_ae.end.fg);
          return TRUE;
        }
        break;
      }
      iltx = STD_NEXT(iltx);
    }
    return FALSE;
  }

  iltx = srch_ae.start.stmt;
  term_std = FG_STDLAST(srch_ae.start.fg);
  while (term_std != iltx) {
    iltx = STD_NEXT(iltx);
    if (iltx == 0)
      break;
    astx = STD_AST(iltx);
    switch (A_TYPEG(astx)) {
    case A_ICALL:
      if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
        int args, lastx, rastx;
        args = A_ARGSG(astx);
        lastx = ARGT_ARG(args, 0);
        rastx = ARGT_ARG(args, 2);
        if (nme) {
          if (nme_of_ast(astx) != nme) {
            break;
          }
        }
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "ptrdef in path, ptr def in start fg %d\n",
                  srch_ae.end.fg);
        return TRUE;
      }
      break;
    case A_CALL:
    case A_FUNC:
      if (is_ptrast_arg(ptrast, astx)) {
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "ptrdef in path -- arg, ptr def in start fg %d\n",
                  srch_ae.end.fg);
        return TRUE;
      }
      break;
    }
  }
  /*
   * nodes are different. examine ilts between the start of the end
   * node and the end statement, inclusive.
   * only search if there exists a store via a pointer in the node.
   */
  term_std = FG_STDFIRST(srch_ae.end.fg);
  for (iltx = (srch_ae.end.stmt); iltx; iltx = STD_PREV(iltx)) {
    astx = STD_AST(iltx);
    switch (A_TYPEG(astx)) {
    case A_ICALL:
      if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
        int args, lastx, rastx;
        args = A_ARGSG(astx);
        lastx = ARGT_ARG(args, 0);
        rastx = ARGT_ARG(args, 2);
        if (nme) {
          if (nme_of_ast(astx) != nme) {
            break;
          }
        }
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "ptrdef in path, ptr def in start fg %d\n",
                  srch_ae.end.fg);
        return TRUE;
      }
      break;
    case A_CALL:
    case A_FUNC:
      if (is_ptrast_arg(ptrast, astx)) {
        if (OPTDBG(9, 16384))
          fprintf(gbl.dbgfil, "ptrdef in path -- arg, ptr def in start fg %d\n",
                  srch_ae.end.fg);
        return TRUE;
      }
      break;
    }

    if (iltx == term_std)
      break;
  }

  return FALSE;
}

/*
 * If there is a pointer def(a=>b) or sub(b)
 * If there is the lhs is a member, the nme must be the nme of parent
 */
LOGICAL
is_ptrdef_in_path(int start_ilt, int start_fg, int end_ilt, int end_fg, int nme,
                  int ast)
{

  srch_ae.start.stmt = start_ilt;
  srch_ae.start.fg = start_fg;
  srch_ae.end.stmt = end_ilt;
  srch_ae.end.fg = end_fg;

  return (isptrdef_in_path(basenme_of(nme), ast));
}

/* check if all def of this ast is in allocate statement and it is safe */
LOGICAL
alldefs_allocsafe(int ast, int stmt)
{
  int def = 0;
  int fgx = STD_FG(stmt);
  int std = 0;
  int hasalloc = 0;
  BV *bv = NULL;
  LOGICAL is_inited = FALSE;

  int nme = nme_of_ast(ast);

  if (!fgx)
    return FALSE;
  if (nme) {
    bv = FG_UNINITED(STD_FG(stmt));
    is_inited = is_initialized(bv, nme);
    if (!is_inited) /* must be defined in this routine */
      return FALSE;
    while ((def = find_next_reaching_def(nme, fgx, def))) {
      std = DEF_STD(def);
      if (std == stmt) {
        continue;
      }
      if (is_alloc_std(std)) {
        int allocast = STD_AST(std);

        if (!is_parentof_ast(A_SRCG(allocast), ast))
          continue;

        hasalloc = 1;
        /* check if there is a pointer def or a call with this pointer */
        if (is_ptrdef_in_path(std, STD_FG(std), stmt, STD_FG(stmt), nme, ast)) {
          return FALSE;
        }
      } else {
        /* an assignment of a parent will cause an unsafe. */
        if (A_ASN == A_TYPEG(STD_AST(std))) {
          int destast = STD_AST(std);
          if (is_parentof_ast(A_DESTG(destast), ast)) {
            if (A_TYPEG(destast) == A_SUBSCR) {
              destast = A_LOPG(destast);
              if (ast == destast) /* normal assignment */
                continue;
            }
            return FALSE;
          }
          continue;
        } else if (A_TYPEG(STD_AST(std)) == A_ICALL &&
                   A_OPTYPEG(STD_AST(std)) == I_PTR2_ASSIGN) {
          /* pointer associate constructs are ok; we already processed them,
             e.g., in ptrdefs_has_lhsconflict */
          hasalloc = 1;
          continue;
        } else
          return FALSE;
      }
    }
    /* all pointer defs are in allocate stmt and it is safe */
    if (hasalloc)
      return TRUE;
  }
  return FALSE;
}

/* find all reaching def of this nme and check if it has conflict(nme appears in
 * lhsdefs
 * struct
 */
static LOGICAL
ptrdefs_has_lhsconflict(int nme, int std, int def)
{
  int i, astx, rhsnme, rastx, args;

  def = find_next_reaching_def(nme, STD_FG(std), def);
  if (!def) return FALSE;

  /* if it is not initialized, then need temp */
  if (!is_initialized(FG_UNINITED(STD_FG(std)), nme))
    return TRUE;

  while (def) {
    int def_std = DEF_STD(def);
    if (def_std == std) {
      def = find_next_reaching_def(nme, STD_FG(std), def);
      continue;
    }
    astx = STD_AST(def_std);
    switch (A_TYPEG(astx)) {
    case A_ICALL:
      if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
        for (i = 0; i < lhscount; ++i) {
          if (nme == lhsdefs[i].nme) {
            return TRUE;
          }
        }
        /* recursive find its def's def, i.e., a=>b, b=>c, current=>d ,
         * may be a also point by lhs.
         */
        args = A_ARGSG(astx);
        rastx = ARGT_ARG(args, 2);
        rhsnme = nme_of_ast(rastx);

        if (!rhsnme || ptrdefs_has_lhsconflict(rhsnme, def_std, def))
          return TRUE;
      }
      break;
    case A_CALL:
    case A_FUNC:
      return TRUE;
      break;
    case A_ALLOC:
      break;
    case A_ASN:
      /* if it is an assignment of its parent, then it is not safe */
      /* check if it is the same type */
      break;
    default:
      return TRUE;
      break;
    }
    def = find_next_reaching_def(nme, STD_FG(std), def);
  }
  return FALSE;
}

#ifdef FLANG_OPTUTIL_UNUSED
static LOGICAL
is_member_ast(int ast)
{
  while (ast) {
    switch (A_TYPEG(ast)) {
    case A_MEM:
      return TRUE;
    case A_ID:
      return FALSE;
    case A_SUBSCR:
      ast = A_LOPG(ast);
      break;
    default:
      return FALSE;
    }
  }
  return FALSE;
}
#endif

static void
_find_rhs_def_conflict(int ast, int *args)
{
  int i, sptr, ast_opnd, lhs_opnd;
  int std = args[0];
  int lhs = args[2];
  int allochk = args[3];
  int nme = nme_of_ast(ast);
  int lhs_sptr = basesym_of(nme_of_ast(lhs));
  if (lhscount >= MAX_DEF) {
    args[1] = 1;
    return;
  }

  if (nme && args[1] == 0) { /* args[1] == 0 -- no conflict found so far */
    switch (A_TYPEG(ast)) {
    case A_ID:
      if (lhs == ast) {
        return; /* impossible to get here, lhs is supposed to be subscript */
      }
      sptr = A_SPTRG(ast);
      if (sptr == lhs_sptr) /* will check at A_SUBSCR */
        return;
      if (TARGETG(sptr)) {
        args[1] = 1;
        return;
      } else if (POINTERG(sptr)) {
        if (allochk) {
          if (!alldefs_allocsafe(ast, std)) {
            args[1] = 1;
            return;
          }
        }
        /* check current nme against all the nme's on the lhs */
        for (i = 0; i < lhscount; ++i) {
          if (nme == lhsdefs[i].nme) {
            args[1] = 1;
            return;
          }
        }

        /* check all of the defs of the current nme */
        if (ptrdefs_has_lhsconflict(nme, std, 0)) {
          args[1] = 1;
          return;
        }
      }
      break;
    case A_SUBSCR:
      ast_opnd = A_LOPG(ast);
      lhs_opnd = A_LOPG(lhs);
      if (A_TYPEG(lhs_opnd) == A_MEM) {
        if (A_TYPEG(ast_opnd) == A_MEM) {
          if (A_MEMG(ast_opnd) == A_MEMG(lhs_opnd))
            if (A_ASDG(lhs) != A_ASDG(ast))
              args[1] = 1;
        }
      } else if (A_TYPEG(lhs_opnd) != A_MEM) {
        if (lhs_opnd == ast_opnd)
          if (A_ASDG(lhs) != A_ASDG(ast))
            args[1] = 1;
      }

      break;
    default:
      break;
    }
  }
}

static void
_find_lhs_on_rhs_conflict(int ast, int *args)
{
  int sptr, ast_opnd, lhs_opnd;
  int lhs = args[2];
  int nme = nme_of_ast(ast);
  int lhs_sptr = basesym_of(nme_of_ast(lhs));
  if (lhscount >= MAX_DEF) {
    args[1] = 1;
    return;
  }

  /*
   * Possible conflict:
   *    abc = sub(abc)
   *    abc%mem = sub(abc)
   *    abc%mem = sub(abc%mem)
   *    abc(1:3) = abc(2:4) op ..
   *
   */

  if (nme && args[1] == 0) { /* args[1] == 0 -- no conflict found so far */
    switch (A_TYPEG(ast)) {
    case A_ID:
      if (lhs == ast) {
        return; /* impossible to get here, lhs is supposed to be subscript */
      }
      sptr = A_SPTRG(ast);
      if (sptr == lhs_sptr) /* will check at A_SUBSCR */
        return;
      if (TARGETG(sptr)) {
        args[1] = 1;
        return;
      }
      /* if it is a parent, then it is not safe */
      if (DTY(DDTG(DTYPEG(sptr))) == TY_DERIVED && is_parentof_ast(ast, lhs)) {
        args[1] = 1;
        return;
      }
      break;
    case A_SUBSCR:
      /* next 2 lines checking may not be sufficient -- need to find example for
       * it
       * if (ast == lhs)
       *    break;
       */
      ast_opnd = A_LOPG(ast);
      lhs_opnd = A_LOPG(lhs);
      if (A_TYPEG(lhs_opnd) == A_MEM) {
        if (A_TYPEG(ast_opnd) == A_MEM) {
          if (A_MEMG(ast_opnd) == A_MEMG(lhs_opnd))
            if (A_ASDG(lhs) != A_ASDG(ast))
              args[1] = 1;
        }
      } else if (A_TYPEG(lhs_opnd) != A_MEM) {
        if (lhs_opnd == ast_opnd)
          if (A_ASDG(lhs) != A_ASDG(ast))
            args[1] = 1;
      }

      break;
    case A_MEM:
      /* if it is a parent of lhs, then it is not safe */
      if (is_parentof_ast(ast, lhs)) {
        args[1] = 1;
        return;
      }
      break;
    default:
      break;
    }
  }
}

static void
find_rhs_conflict(int lhs, int rhs, int stmt, int allochk, int *result)
{

  int args[4];
  args[0] = stmt, args[1] = 0;
  args[2] = lhs;
  args[3] = allochk;
  if (lhscount >= MAX_DEF) {
    *result = 1;
    return;
  }
  ast_visit(1, 1);
  ast_traverse(rhs, NULL, _find_rhs_def_conflict, args);
  *result = args[1];
  ast_unvisit();
}

static void
find_lhs_on_rhs_conflict(int lhs, int rhs, int stmt, int allochk, int *result)
{

  int args[4];
  args[0] = stmt, args[1] = 0;
  args[2] = lhs;
  args[3] = allochk;
  if (lhscount >= MAX_DEF) {
    *result = 1;
    return;
  }
  ast_visit(1, 1);
  ast_traverse(rhs, NULL, _find_lhs_on_rhs_conflict, args);
  *result = args[1];
  ast_unvisit();
}

int
add_lhs_nme(int nme, int std, int isdummy)
{
  if (lhscount >= MAX_DEF)
    return MAX_DEF + 1;

  lhsdefs[lhscount].nme = nme;
  lhsdefs[lhscount].std = std;
  lhsdefs[lhscount].isdummy = isdummy;
  lhscount++;

  return lhscount;
}

#if DEBUG
#ifdef FLANG_OPTUTIL_UNUSED
static void
dump_lhs_nme(int nme, int std, int isdummy)
{
  int i;
  for (i = 0; i < lhscount; ++i) {
    fprintf(gbl.dbgfil, "std: %d\n", std);
    print_nme(nme);
  }
}
#endif
#endif

/* find all origin of lhs defs */
static void
get_lhs_first_defs(int stmt, int lhs)
{
  int a, nme;
  int forall_lhs;
  int astx = STD_AST(stmt);
  if (lhscount >= MAX_DEF)
    return;
  switch (A_TYPEG(astx)) {
  case A_ICALL:
    /* intrinsic call, see if it is ptr assignment */
    if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
      /* pointer assignment */
      int args, lhsastx, rhsastx;
      args = A_ARGSG(astx);
      lhsastx = ARGT_ARG(args, 0);
      rhsastx = ARGT_ARG(args, 2);
      nme = 0;
      switch (A_TYPEG(rhsastx)) {
      case A_ID:
        nme = nme_of_ast(rhsastx);
        break;
      case A_MEM:
        nme = nme_of_ast(rhsastx);
        a = A_PARENTG(rhsastx);
        break;
      case A_SUBSCR:
        nme = nme_of_ast(rhsastx);
        a = A_LOPG(rhsastx);
        break;
      default:
        lhscount = MAX_DEF + 1; /* don't handle other kinds if any */
        return;
      }
      if (nme) {
        int def, isdummy;
        int fgx = STD_FG(stmt);
        int sym = basesym_of(nme);
        isdummy = FALSE;
        def = 0;
        if (sym) {
          isdummy = (SCG(sym) == SC_DUMMY) ? TRUE : FALSE;
          if (!isdummy)
            isdummy = !(is_sym_ptrsafe(sym));
        }
        def = find_next_reaching_def(nme, fgx, def);
        if (def) {
          while ((def = find_next_reaching_def(nme, fgx, def))) {
            int std = DEF_STD(def);
            if (std == stmt)
              continue;
            if (lhs != lhsastx)
              if (!is_parentof_ast(lhsastx, lhs))
                continue;
            lhscount++; /* not safe , too many defs */
            get_lhs_first_defs(std, rhsastx);
            if (lhscount >= MAX_DEF)
              return;
          }
        } else {
          /* very first defs in this function */
          add_lhs_nme(nme, stmt, isdummy);
          return;
        }
      }
    }
    break;
  case A_CALL:
  case A_FUNC:
    lhscount = MAX_DEF + 1; /* not safe */
    break;
  case A_ALLOC:
    nme = nme_of_ast(lhs);
    add_lhs_nme(nme, stmt, 0);
    break;
  case A_ASN:
    break;
  case A_FORALL:
    forall_lhs = A_IFSTMTG(astx); /* look for lhs of the forall assignment */
    if (A_TYPEG(forall_lhs) == A_ASN) {
      forall_lhs = A_DESTG(forall_lhs);
      if (A_TYPEG(forall_lhs) == A_SUBSCR) {
        nme = nme_of_ast(forall_lhs);
        add_lhs_nme(nme, stmt, 0);
      }
      else
        lhscount = MAX_DEF + 1; /* error */
    }
    else
      lhscount = MAX_DEF + 1; /* error */
    break;
  default:
    lhscount = MAX_DEF + 1; /* error */
    break;
  }
} /* _find_pointer_defs */

/* determine if the lhs need temp array */
LOGICAL
lhs_needtmp(int lhs, int rhs, int stmt)
{
  int def = 0;
  int fgx = STD_FG(stmt);
  int result;

  if (!fgx)
    return FALSE;

  /* init lhscount for each stmt*/
  lhscount = 0;

  /* if all pointer defs are allocated and along the path from all allocate
   * stmt(s) to
   * current statement there is no pointer def where lhs is on the rhs and lhs
   * is not pointer argument to a call which has interface, then it it safe
   */

  if (alldefs_allocsafe(lhs, stmt)) {
    /* if all def of lhs is allocated and it is available from the allocated
     * stmt to forall stmt
     * Then check if lhs also presents in rhs in this forall and have different
     * index(overlapped).
     */
    find_lhs_on_rhs_conflict(lhs, rhs, stmt, 0, &result);
    if (result == 0)
      return FALSE;
  } else {
    /* find and store the nme's of all lhs defs, to be checking with rhs later
     */
    int nme = nme_of_ast(lhs);
    if (nme) {
      while ((def = find_next_reaching_def(nme, fgx, def))) {
        int std = DEF_STD(def);
        if (std == stmt)
          continue;
        get_lhs_first_defs(std, lhs);
      }
    } else {
      return TRUE;
    }
  }

  /* do check on the rhs */
  find_rhs_conflict(lhs, rhs, stmt, 1, &result);
  if (result != 1)
    return FALSE;
  /* yes need temp */
  return TRUE;
}
