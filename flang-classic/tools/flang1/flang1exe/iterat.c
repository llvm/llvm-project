/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* comm.c - PGHPF communications module */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "gramtk.h"
#include "symutl.h"
#include "extern.h"
#include "comm.h"
#include "ccffinfo.h"
#include "hlvect.h"

/* This routine  is to check array subs whether it is array element.
 *  That can be confirmed if forall index does not appear any subscripts
 *  of array subs.
 */
int
is_array_element_in_forall(int subs, int std)
{
  int ast;
  int list;
  int ndim;
  int i;
  int j;
  int asd;
  int sub_expr;

  ast = STD_AST(std);
  list = A_LISTG(ast);
  asd = A_ASDG(subs);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++)
    for (j = list; j != 0; j = ASTLI_NEXT(j)) {
      sub_expr = ASD_SUBS(asd, i);
      if (is_name_in_expr(sub_expr, ASTLI_SPTR(j)))
        return 0;
    }
  return 1;
}

/* This routine is to search isptr in expr. If it finds,
*  return 1, else return 0.
*/

LOGICAL
is_name_in_expr(int expr, int isptr)
{
  int i, n, argt, asd, sptr;

  if (expr == 0)
    return FALSE;
  switch (A_TYPEG(expr)) {
  case A_ID:
    sptr = A_SPTRG(expr);
    if (isptr == sptr)
      return TRUE;
    if (is_equivalence(isptr, sptr))
      return TRUE;
    break;
  case A_BINOP:
    if (is_name_in_expr(A_LOPG(expr), isptr))
      return TRUE;
    if (is_name_in_expr(A_ROPG(expr), isptr))
      return TRUE;
    break;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    if (is_name_in_expr(A_LOPG(expr), isptr))
      return TRUE;
    break;
  case A_LABEL:
  case A_CMPLXC:
  case A_CNST:
    break;
  case A_MEM:
    if (is_name_in_expr(A_PARENTG(expr), isptr))
      return TRUE;
    break;
  case A_SUBSTR:
    if (is_name_in_expr(A_LOPG(expr), isptr))
      return TRUE;
    if (is_name_in_expr(A_LEFTG(expr), isptr))
      return TRUE;
    if (is_name_in_expr(A_RIGHTG(expr), isptr))
      return TRUE;
    break;
  case A_ICALL:
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(expr);
    n = A_ARGCNTG(expr);
    for (i = 0; i < n; ++i) {
      if (is_name_in_expr(ARGT_ARG(argt, i), isptr))
        return TRUE;
    }
    break;

  case A_TRIPLE:
    if (is_name_in_expr(A_LBDG(expr), isptr))
      return TRUE;
    if (is_name_in_expr(A_UPBDG(expr), isptr))
      return TRUE;
    if (is_name_in_expr(A_STRIDEG(expr), isptr))
      return TRUE;
    break;

  case A_SUBSCR:
    asd = A_ASDG(expr);
    n = ASD_NDIM(asd);
    for (i = 0; i < n; ++i) {
      if (is_name_in_expr(ASD_SUBS(asd, i), isptr))
        return TRUE;
    }
    if (is_name_in_expr(A_LOPG(expr), isptr))
      return TRUE;
    break;
  default:
    interr("is_name_in_expr: unknown ast type", expr, 3);
    break;
  }
  return FALSE;
} /* is_name_in_expr */

/* This function return TRUE if it find isptr is in expression
 * However, It does not count isptr if it appears as subscript.
 * For example,  expr=i.eq.a(i), isptr=i return TRUE;
 *               expr=0.eq.a(i), isptr=i return FALSE;
 *               expr=j.eq.i, isptr=i    return TRUE;
 */
LOGICAL
is_lonely_idx(int expr, int isptr)
{
  int i;
  int n;
  int find1;
  int find2;
  int argt;

  if (expr == 0)
    return FALSE;
  switch (A_TYPEG(expr)) {
  case A_ID:
    if (isptr == A_SPTRG(expr))
      return TRUE;
    else
      return FALSE;
  case A_BINOP:
    find1 = is_lonely_idx(A_LOPG(expr), isptr);
    find2 = is_lonely_idx(A_ROPG(expr), isptr);
    if (find1 || find2)
      return TRUE;
    else
      return FALSE;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    find1 = is_lonely_idx(A_LOPG(expr), isptr);
    return find1;
  case A_LABEL:
  case A_CMPLXC:
  case A_CNST:
    return FALSE;
  case A_MEM:
    find1 = is_lonely_idx((int)A_PARENTG(expr), isptr);
    return find1;
  case A_SUBSTR:
    if (is_lonely_idx((int)A_LOPG(expr), isptr))
      return TRUE;
    if (is_lonely_idx((int)A_LEFTG(expr), isptr))
      return TRUE;
    if (is_lonely_idx((int)A_RIGHTG(expr), isptr))
      return TRUE;
    return FALSE;
  case A_ICALL:
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(expr);
    n = A_ARGCNTG(expr);
    for (i = 0; i < n; ++i) {
      if (is_lonely_idx(ARGT_ARG(argt, i), isptr))
        return TRUE;
    }
    return FALSE;

  case A_TRIPLE:
    if (is_lonely_idx(A_LBDG(expr), isptr))
      return TRUE;
    if (is_lonely_idx(A_UPBDG(expr), isptr))
      return TRUE;
    if (is_lonely_idx(A_STRIDEG(expr), isptr))
      return TRUE;
    return FALSE;

  case A_SUBSCR:
    return FALSE;
  default:
    interr("is_lonely_idx: unknown type", expr, 3);
    return FALSE;
  }
}

static struct {
  int *elist;
  int *dlist;
  int esize, eavl;
  int dsize, davl;
  int estdx, dstdx;
  int dexpr, forall_list;
} dep = {NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/*
 * Given an AST tree (such as FORALL LHS), save the ID and member names on the
 * LHS
 * these will be used to determine if any RHS or subscript expression
 * might be changed by this forall assignment
 *  forall(i=1:10) A(i)%MEM(10) = B(i)
 * this will save A%MEM, so we can check whether B conflicts with A%MEM
 */
static void
build_dlist(int depends, int forall_list, int lstdx, int rstdx)
{
  int d;
  dep.forall_list = forall_list;
  dep.dstdx = lstdx;
  dep.estdx = rstdx;
  dep.dexpr = depends;
  /* build list of depends names */
  if (dep.dsize == 0) {
    dep.dsize = 20;
    NEW(dep.dlist, int, dep.dsize);
  }
  dep.davl = 0;
  for (d = depends; d;) {
    switch (A_TYPEG(d)) {
    case A_ID:
      NEED(dep.davl + 1, dep.dlist, int, dep.dsize, dep.dsize * 2);
      dep.dlist[dep.davl] = A_SPTRG(d);
      ++dep.davl;
      d = 0;
      break;
    case A_MEM:
      NEED(dep.davl + 1, dep.dlist, int, dep.dsize, dep.dsize * 2);
      dep.dlist[dep.davl] = A_SPTRG(A_MEMG(d));
      ++dep.davl;
      d = A_PARENTG(d);
      break;
    case A_SUBSTR:
    case A_SUBSCR:
      d = A_LOPG(d);
      break;
    default:
      interr("build_dlist: unexpected ast", d, 3);
      d = 0;
    }
  }
} /* build_dlist */

static LOGICAL subscr_dependent_check(int expr);

/*
 * allocate and fill a name buffer
 */
static char *
build_name_buffer(int *list, int avl)
{
  int i, len;
  char *buffer;
  len = 0;
  for (i = 1; i <= avl; ++i) {
    len += strlen(SYMNAME(list[avl - i])) + 1;
  }
  len += 1;
  buffer = (char *)malloc(len);
  len = 0;
  for (i = 1; i <= avl; ++i) {
    strcpy(buffer + len, SYMNAME(list[avl - i]));
    len += strlen(SYMNAME(list[avl - i]));
    if (i < avl) {
      buffer[len++] = '%';
    }
    buffer[len] = '\0';
  }
  return buffer;
} /* build_name_buffer */

/*
 * see whether the AST reference at expr conflicts with the previously
 * saved names from build_dlist
 */
static LOGICAL
name_dependent_check(int expr)
{
  int e, d;
  int esptr, dsptr, edtype, ddtype;
  LOGICAL etarget, dtarget;
  /* expr must be A_MEM or A_ID */
  if (expr == 0)
    return FALSE;
  /* build list of expr names */
  if (dep.esize == 0) {
    dep.esize = 20;
    NEW(dep.elist, int, dep.esize);
  }
  dep.eavl = 0;
  for (e = expr; e;) {
    switch (A_TYPEG(e)) {
    case A_ID:
      NEED(dep.eavl + 1, dep.elist, int, dep.esize, dep.esize * 2);
      dep.elist[dep.eavl] = A_SPTRG(e);
      ++dep.eavl;
      e = 0;
      break;
    case A_MEM:
      NEED(dep.eavl + 1, dep.elist, int, dep.esize, dep.esize * 2);
      dep.elist[dep.eavl] = A_SPTRG(A_MEMG(e));
      ++dep.eavl;
      e = A_PARENTG(e);
      break;
    case A_SUBSTR:
    case A_SUBSCR:
      e = A_LOPG(e);
      break;
    case A_FUNC:
      e = 0;
      break;
    default:
      interr("name_dependent: unexpected ast", e, 3);
      e = 0;
    }
  }

  if (dep.eavl == 0 || dep.davl == 0) {
    interr("name_dependent: no names", 0, 3);
    return FALSE;
  }
  /* compare elist to the previously built dlist */
  etarget = FALSE;
  for (e = dep.eavl; e > 0; --e) {
    esptr = dep.elist[e - 1];
    edtype = DDTG(DTYPEG(esptr));
    if (TARGETG(esptr))
      etarget = TRUE;
    dtarget = FALSE;
    for (d = dep.davl; d > 0; --d) {
      dsptr = dep.dlist[d - 1];
      ddtype = DDTG(DTYPEG(dsptr));
      if (TARGETG(dsptr))
        dtarget = TRUE;
      /* can 'esptr' overlap with 'dsptr'? */
      /* yes if they are the same variable,
       * esptr is a pointer and dsptr is a pointer or target of same type,
       * esptr is a target and dsptr is a pointer of same type,
       * both variables and equivalenced (handled later)
       */
      if (edtype == ddtype) {
        /* ### for F2003, allow extended datatypes as well */
        int overlap;
        overlap = 0;
        if (esptr == dsptr && e == dep.eavl && d == dep.davl) {
          overlap = 1;
        } else if (POINTERG(esptr) && (POINTERG(dsptr) || dtarget)) {
          overlap = 2;
          if (flg.opt >= 2 && XBIT(53, 2)) {
            /* see if we have any PTA information about
             * the RHS in elist with respect to the LHS in dlist */
            if (!pta_conflict(dep.estdx, dep.elist[dep.eavl - 1], dep.dstdx,
                              dep.dlist[dep.davl - 1], POINTERG(dsptr),
                              dtarget)) {
              overlap = 0;
            }
          }
          if (!flg.depchk)
            overlap = 0;
        } else if (etarget && POINTERG(dsptr)) {
          overlap = 2;
          if (flg.opt >= 2 && XBIT(53, 2)) {
            if (!pta_conflict(dep.dstdx, dep.dlist[dep.davl - 1], dep.estdx,
                              dep.elist[dep.eavl - 1], POINTERG(esptr),
                              etarget)) {
              overlap = 0;
            }
          }
          if (!flg.depchk)
            overlap = 0;
        }
        if (DTY(ddtype) == TY_UNION) {
          /* assume overlap */
        } else {
          int ee, dd;
          /* compare the rest of the list */
          for (dd = d - 1, ee = e - 1; overlap > 0 && dd > 0 && ee > 0;
               --dd, --ee) {
            /* AAA%b%c%d%rest
             * BBB%b%c%d%more
             * compare 'rest' and 'more' */
            int ddsptr, eesptr;
            ddsptr = dep.dlist[dd - 1];
            eesptr = dep.elist[ee - 1];
            if (ddsptr != eesptr) {
              overlap = 0;
            } else {
              int dddtype;
              /* same member; if this is a union, stop here */
              dddtype = DDTG(DTYPEG(ddsptr));
              if (DTY(dddtype) == TY_UNION)
                break;
            }
          }
          if (overlap == 1 && dep.forall_list) {
            /* allow (=,=,=) dependence */
            if (dep.dexpr == expr) {
              overlap = 0;
              /* check subscripts, if any */
            } else if (!dd_array_conflict(A_LISTG(dep.forall_list), dep.dexpr,
                                          expr, -1)) {
              overlap = 0;
            } else {
              /* flg.depchk MORE WORK HERE */
            }
          }
          if (overlap > 0) {
            int i, needmsg;
            if (dep.eavl != dep.davl) {
              needmsg = 1;
            } else {
              needmsg = 0;
              for (i = 0; i < dep.davl; ++i) {
                if (dep.dlist[i] != dep.elist[i]) {
                  needmsg = 1;
                  break;
                }
              }
            }
            if (needmsg) {
              char *dname, *ename;
              dname = build_name_buffer(dep.dlist, dep.davl);
              ename = build_name_buffer(dep.elist, dep.eavl);
              ccff_info(MSGFTN, "FTN019", 1, STD_LINENO(dep.dstdx),
                        "Conflict or overlap between %var1 and %var2",
                        "var1=%s", dname, "var2=%s", ename, NULL);
              free(ename);
              free(dname);
            }
            return TRUE;
          }
        }
      }
    }
  }
  esptr = dep.elist[dep.eavl - 1];
  dsptr = dep.dlist[dep.davl - 1];
  if (is_equivalence(dsptr, esptr))
    return TRUE;

  /* now check any subscripts */
  if (subscr_dependent_check(expr))
    return TRUE;
  return FALSE;
  /* do we need a special case for ST_ARRDSC? */
} /* name_dependent_check */

/*
 * recursively visit the tree expr to see whether any name in the tree
 * conflicts with the previously saved names from build_dlist
 */
static LOGICAL
expr_dependent_check(int expr)
{
  int i, n, argt;

  if (expr == 0)
    return FALSE;
  switch (A_TYPEG(expr)) {
  case A_ID:
  case A_MEM:
  case A_SUBSCR:
    if (name_dependent_check(expr))
      return TRUE;
    break;
  case A_BINOP:
    if (expr_dependent_check(A_LOPG(expr)))
      return TRUE;
    if (expr_dependent_check(A_ROPG(expr)))
      return TRUE;
    break;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    if (expr_dependent_check(A_LOPG(expr)))
      return TRUE;
    break;
  case A_LABEL:
  case A_CMPLXC:
  case A_CNST:
    break;
  case A_SUBSTR:
    if (expr_dependent_check(A_LOPG(expr)))
      return TRUE;
    if (expr_dependent_check(A_LEFTG(expr)))
      return TRUE;
    if (expr_dependent_check(A_RIGHTG(expr)))
      return TRUE;
    break;
  case A_ICALL:
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(expr);
    n = A_ARGCNTG(expr);
    for (i = 0; i < n; ++i) {
      if (expr_dependent_check(ARGT_ARG(argt, i)))
        return TRUE;
    }
    break;

  case A_TRIPLE:
    if (expr_dependent_check(A_LBDG(expr)))
      return TRUE;
    if (expr_dependent_check(A_UPBDG(expr)))
      return TRUE;
    if (expr_dependent_check(A_STRIDEG(expr)))
      return TRUE;
    break;
  case A_MP_ATOMICREAD:
    if (expr_dependent_check(A_SRCG(expr)))
      return TRUE;
    break;

  default:
    interr("expr_dependent_check: unknown type", expr, 3);
    break;
  }
  return FALSE;
} /* expr_dependent_check */

/*
 * recursively visit the tree expr to see whether any name in the tree
 * conflicts with the previously saved names from build_dlist
 * Like expr_dependent_check, but only really checks expressions in
 * subscript or substring selectors.
 */
static LOGICAL
subscr_dependent_check(int expr)
{
  int i, n, argt, asd;

  if (expr == 0)
    return FALSE;
  switch (A_TYPEG(expr)) {
  case A_ID:
    break;
  case A_MEM:
    if (subscr_dependent_check(A_PARENTG(expr)))
      return TRUE;
    break;
  case A_BINOP:
    if (subscr_dependent_check(A_LOPG(expr)))
      return TRUE;
    if (subscr_dependent_check(A_ROPG(expr)))
      return TRUE;
    break;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    if (subscr_dependent_check(A_LOPG(expr)))
      return TRUE;
    break;
  case A_LABEL:
  case A_CMPLXC:
  case A_CNST:
    break;
  case A_SUBSTR:
    if (subscr_dependent_check(A_LOPG(expr)))
      return TRUE;
    if (expr_dependent_check(A_LEFTG(expr)))
      return TRUE;
    if (expr_dependent_check(A_RIGHTG(expr)))
      return TRUE;
    break;
  case A_ICALL:
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(expr);
    n = A_ARGCNTG(expr);
    for (i = 0; i < n; ++i) {
      if (subscr_dependent_check(ARGT_ARG(argt, i)))
        return TRUE;
    }
    break;

  case A_TRIPLE:
    if (subscr_dependent_check(A_LBDG(expr)))
      return TRUE;
    if (subscr_dependent_check(A_UPBDG(expr)))
      return TRUE;
    if (subscr_dependent_check(A_STRIDEG(expr)))
      return TRUE;
    break;

  case A_SUBSCR:
    asd = A_ASDG(expr);
    n = ASD_NDIM(asd);
    for (i = 0; i < n; ++i) {
      if (expr_dependent_check(ASD_SUBS(asd, i)))
        return TRUE;
    }
    break;
  case A_MP_ATOMICREAD:
    if (subscr_dependent_check(A_SRCG(expr)))
      return TRUE;
    break;
  default:
    interr("subscr_dependent_check: unknown type", expr, 3);
    break;
  }
  return FALSE;
} /* subscr_dependent_check */

LOGICAL
expr_dependent(int expr, int depends, int lstdx, int rstdx)
{
  build_dlist(depends, 0, lstdx, rstdx);
  return expr_dependent_check(expr);
} /* expr_dependent */

LOGICAL
is_dependent(int lhs, int rhs, int forall, int lstdx, int rstdx)
{
  if (rhs == 0)
    return FALSE;
  build_dlist(lhs, forall, lstdx, rstdx);
  return expr_dependent_check(rhs);
} /* is_dependent */

LOGICAL
subscr_dependent(int expr, int depends, int lstdx, int rstdx)
{
  build_dlist(depends, 0, lstdx, rstdx);
  return subscr_dependent_check(expr);
} /* subscr_dependent */

/* find isptr symbol in sub_expr and replace with expr
   if indirection=0, it does not change  array subscripts
*/

int
replace_expr(int sub_expr, int isptr, int expr, int indirection)
{
  int i;
  int asd;
  int ndim;
  int expr1;
  int expr2;
  int sptr;
  int subs[7];
  int nargs, argt;

  if (sub_expr == 0)
    return sub_expr;
  switch (A_TYPEG(sub_expr)) {
  case A_ID:
    if (isptr == A_SPTRG(sub_expr))
      return expr;
    else
      return sub_expr;
  case A_BINOP:
    expr1 = replace_expr(A_LOPG(sub_expr), isptr, expr, indirection);
    expr2 = replace_expr(A_ROPG(sub_expr), isptr, expr, indirection);
    return mk_binop(A_OPTYPEG(sub_expr), expr1, expr2, A_DTYPEG(sub_expr));
  case A_UNOP:
    expr1 = replace_expr(A_LOPG(sub_expr), isptr, expr, indirection);
    return mk_unop(A_OPTYPEG(sub_expr), expr1, A_DTYPEG(sub_expr));
  case A_CONV:
    expr1 = replace_expr(A_LOPG(sub_expr), isptr, expr, indirection);
    return mk_convert(expr1, A_DTYPEG(sub_expr));
  case A_CMPLXC:
  case A_CNST:
    return sub_expr;
  case A_PAREN:
    expr1 = replace_expr(A_LOPG(sub_expr), isptr, expr, indirection);
    return mk_paren(expr1, A_DTYPEG(sub_expr));
  case A_SUBSCR:
    if (indirection) {
      sptr = sptr_of_subscript(sub_expr);
      /*
      assert(!ALIGNG(sptr),
             "replace_expr: indirection should not distri", sptr, 999);
       */
      asd = A_ASDG(sub_expr);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; i++) {
        expr1 = replace_expr(ASD_SUBS(asd, i), isptr, expr, indirection);
        subs[i] = expr1;
      }
      sub_expr = mk_subscr(A_LOPG(sub_expr), subs, ndim, A_DTYPEG(sub_expr));
    }
    return sub_expr;
  case A_FUNC:
  case A_INTR:
    argt = A_ARGSG(sub_expr);
    nargs = A_ARGCNTG(sub_expr);
    for (i = 0; i < nargs; ++i)
      if (ARGT_ARG(argt, i))
        ARGT_ARG(argt, i) =
            replace_expr(ARGT_ARG(argt, i), isptr, expr, indirection);
    A_ARGSP(sub_expr, argt);
    return sub_expr;
  default:
    return sub_expr;
  }
}

LOGICAL
is_equivalence(int sptr1, int sptr2)
{
  int stype1;
  int stype2;
  int p;

  stype1 = STYPEG(sptr1);
  stype2 = STYPEG(sptr2);
  if (stype1 != ST_ARRAY && stype1 != ST_VAR)
    return FALSE;
  if (stype2 != ST_ARRAY && stype2 != ST_VAR)
    return FALSE;

  for (p = SOCPTRG(sptr1); p; p = SOC_NEXT(p)) {
    if (SOC_SPTR(p) == sptr2)
      return TRUE;
  }

  for (p = SOCPTRG(sptr2); p; p = SOC_NEXT(p)) {
    if (SOC_SPTR(p) == sptr1)
      return TRUE;
  }
  return FALSE;
}
