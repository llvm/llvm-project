/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Rewrite subscript vectors for lhs and rhs, etc.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "gramtk.h"
#include "comm.h"
#include "extern.h"
#include "hpfutl.h"
#include "commopt.h"
#include "rte.h"

static int reference_for_temp_lhs_indirection(int, int, int);
#ifdef FLANG_VSUB_UNUSED
static int newforall_list(int arr, int forall);
#endif
static int forall_semantic(int std);
static void forall_with_mask(int std);
static void forall_loop_interchange(int std);
static void forall_with_shape(int std);
static void forall_list_call(int std);
static void forall_bound_dependence(int std);
static void forall_bound_dependence_fix(int prevstd, int nextstd);
static LOGICAL is_mask_for_rhs(int std, int ast);
static LOGICAL is_legal_lhs_for_mask(int, int);
#ifdef FLANG_VSUB_UNUSED
static int make_dos(int std);
static void make_enddos(int n, int std);
#endif
static void scalar_lhs_dependency(int std);
static void scatter_dependency(int std);
static void scatter_dependency_assumsz(int std);
static int take_out_assumsz_array(int expr, int std, int sptr);
static LOGICAL is_one_idx_for_dim(int, int);
static LOGICAL is_sequentialize_pure(int std);
static LOGICAL is_ugly_pure(int ast);
static LOGICAL find_scatter_rhs(int expr, int forall, int *rhs);
static LOGICAL is_all_idx_in_subscript(int list, int a);

static LOGICAL ptr_subs_olap(int, int);
static LOGICAL can_ptr_olap(int, int);

/** \brief This routine rewrites foralls

    1. forall with shape sec as A(i,:)
    2. forall with dependency,
    3. forall with distributed indirection array at rhs.
 */
void
rewrite_forall(void)
{
  int std, stdnext;
  int ast;
  int parallel_depth, task_depth;

  parallel_depth = 0;
  task_depth = 0;
  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    gbl.lineno = STD_LINENO(std);
    arg_gbl.std = std;
    arg_gbl.lhs = 0;
    arg_gbl.used = FALSE;
    arg_gbl.inforall = FALSE;
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_MP_PARALLEL:
      ++parallel_depth;
      /*symutl.sc = SC_PRIVATE;*/
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_ENDPARALLEL:
      --parallel_depth;
      if (parallel_depth == 0 && task_depth == 0) {
        /*symutl.sc = SC_LOCAL;*/
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    case A_MP_TASKLOOPREG:
    case A_MP_ETASKLOOPREG:
      break;
    case A_MP_TASK:
    case A_MP_TASKLOOP:
      ++task_depth;
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_ENDTASK:
    case A_MP_ETASKLOOP:
      --task_depth;
      if (parallel_depth == 0 && task_depth == 0) {
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    case A_FORALL:
      process_forall(std);
      break;
    }
  }
}

static int process_forall_recursion = 0;

int
process_forall(int std)
{
  int forall, asn, lhs, rhs, save_process_forall_recursion;
  int prevstd, nextstd;

  forall = STD_AST(std);
  assert(A_TYPEG(forall) == A_FORALL, "process_forall: not a FORALL", forall,
         3);
  asn = A_IFSTMTG(forall);
  if (A_TYPEG(asn) != A_ASN) {
    sequentialize(std, STD_AST(std), FALSE);
    return 0;
  }
  lhs = A_DESTG(asn);
  rhs = A_SRCG(asn);
  if (A_TYPEG(lhs) == A_MEM && !A_SHAPEG(lhs) && A_TYPEG(rhs) == A_ID &&
      HCCSYMG(A_SPTRG(rhs)) && !A_SRCG(forall)) {
    sequentialize(std, STD_AST(std), FALSE);
    return 0;
  }
  /* sequentialize string forall */
  if (A_TYPEG(lhs) == A_SUBSTR) {
    /*scalarize(std,STD_AST(std),FALSE);*/
    sequentialize(std, STD_AST(std), FALSE);
    return 0;
  }
  rhs = A_SRCG(asn);
  if (A_TYPEG(rhs) == A_FUNC && SEQUENTG(A_SPTRG(A_LOPG(rhs)))) {
    sequentialize(std, STD_AST(std), FALSE);
    return 0;
  }
  save_process_forall_recursion = process_forall_recursion;
  process_forall_recursion = 1;
  (void)forall_semantic(std);
  if (!save_process_forall_recursion) {
    forall_bound_dependence(std);
    prevstd = STD_PREV(std);
    nextstd = STD_NEXT(std);
  }
  forall_loop_interchange(std);
  forall_with_shape(std);
  /*    forall_list_normalize(std); */
  forall_with_mask(std);
  forall_lhs_indirection(std);
  /*    forall_rhs_indirection(std);   */
  if (!save_process_forall_recursion) {
    forall_bound_dependence_fix(prevstd, nextstd);
  }
  process_forall_recursion = save_process_forall_recursion;
  return 0;
}

static int
forall_semantic(int std)
{
  int forall;
  int asn;
  int list;
  int first_lhs;
  int j;

  forall = STD_AST(std);
  assert(A_TYPEG(forall) == A_FORALL, "forall_semantic: not a FORALL", forall,
         3);
  list = A_LISTG(forall);
  asn = A_IFSTMTG(forall);
  if (A_TYPEG(asn) != A_ASN)
    return 0;

  first_lhs = A_DESTG(asn);
  for (j = list; j != 0; j = ASTLI_NEXT(j)) {
    LOGICAL found;
    int isptr, lhs;
    isptr = ASTLI_SPTR(j);
    lhs = first_lhs;
    found = FALSE;
    while (!found && A_TYPEG(lhs) != A_ID) {
      if (A_TYPEG(lhs) == A_MEM) {
        lhs = A_PARENTG(lhs);
      } else if (A_TYPEG(lhs) == A_SUBSCR) {
        int asd;
        int i, ndim;
        asd = A_ASDG(lhs);
        ndim = ASD_NDIM(asd);
        for (i = 0; i < ndim; i++)
          if (is_name_in_expr(ASD_SUBS(asd, i), isptr))
            found = TRUE;
        /* see if there's a subscripted parent */
        lhs = A_LOPG(lhs);

      } else if (A_TYPEG(lhs) == A_SUBSTR) {
        if (is_name_in_expr(A_RIGHTG(lhs), isptr) ||
            is_name_in_expr(A_LEFTG(lhs), isptr)) {
          scalarize(std, STD_AST(std), FALSE);
          if (A_TYPEG(STD_AST(std)) == A_COMMENT)
            return 1;
          found = TRUE;
        }
        lhs = A_LOPG(lhs);
      } else {
        interr("forall_semantic: LHS not subscr or member", lhs, 3);
        return 0;
      }
    }
    if (!found && (A_TYPEG(lhs) != A_ID || !HCCSYMG(A_SPTRG(lhs)))) {
      error(487, 4, STD_LINENO(std), SYMNAME(isptr), CNULL);
      /* NOTREACHED */
      return 0;
    }
  }

  return 0;
}

int
assign_scalar(int std, int ast)
{
  int sptr;
  int asn, dest;

  sptr = sym_get_scalar("ii", "s", A_DTYPEG(ast));
  asn = mk_stmt(A_ASN, 0);
  dest = mk_id(sptr);
  A_DESTP(asn, dest);
  A_SRCP(asn, ast);
  add_stmt_before(asn, std);
  return mk_id(sptr);
}

static void
forall_list_call(int std)
{
  int forall;
  int list;
  int j;
  int triple;
  int l, u, s;

  forall = STD_AST(std);
  list = A_LISTG(forall);
  for (j = list; j != 0; j = ASTLI_NEXT(j)) {
    triple = ASTLI_TRIPLE(j);
    l = A_LBDG(triple);
    u = A_UPBDG(triple);
    s = A_STRIDEG(triple);
    if (l && contains_call(l))
      l = assign_scalar(std, l);
    if (u && contains_call(u))
      u = assign_scalar(std, u);
    if (s && contains_call(s))
      u = assign_scalar(std, s);
    triple = mk_triple(l, u, s);
    ASTLI_TRIPLE(j) = triple;
  }
}

static void
forall_with_mask(int std)
{

  int forall;
  int asn;
  int lhs;
  int src;
  int temp_ast, sptr;
  int newforall;
  int newasn;
  int mask;
  int stdf;
  int align;
  int list;

  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);
  src = A_SRCG(asn);
  lhs = A_DESTG(asn);
  mask = A_IFEXPRG(forall);
  if (!mask)
    return;
  if (A_TYPEG(mask) == A_SUBSCR)
    return;
  if (!is_legal_lhs_for_mask(lhs, forall))
    return;
  if (!is_indirection_in_it(lhs) && !is_mask_for_rhs(std, src))
    return;

  list = A_LISTG(forall);
  if (is_multiple_idx_in_list(list))
    return;
  if (!is_one_idx_for_dim(lhs, list))
    return;

  align = ALIGNG(left_array_symbol(lhs));
  if (!align)
    return;
  /* split forall */
  sptr = get_temp_forall(forall, lhs, std, std, DT_LOG, 0);
  temp_ast = reference_for_temp_lhs_indirection(sptr, lhs, forall);
  newforall = mk_stmt(A_FORALL, 0);
  A_LISTP(newforall, A_LISTG(forall));
  A_SRCP(newforall, A_SRCG(forall));
  newasn = mk_stmt(A_ASN, 0);
  A_DESTP(newasn, temp_ast);
  A_SRCP(newasn, mask);
  A_IFSTMTP(newforall, newasn);
  A_IFEXPRP(newforall, 0);
  stdf = add_stmt_before(newforall, std);
  process_forall(stdf);

  A_IFEXPRP(forall, temp_ast);
  STD_AST(std) = forall;
}

static LOGICAL
is_mask_for_rhs(int std, int ast)
{
  int shape;
  int l, r;
  int dtype;
  int i;
  int asn;
  int forall;
  int lhs;

  if (ast == 0)
    return 0;
  shape = A_SHAPEG(ast);
  dtype = A_DTYPEG(ast);
  switch (A_TYPEG(ast)) {
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_SUBSTR:
  case A_MEM:
    return FALSE;
  case A_BINOP:
    l = is_mask_for_rhs(std, A_LOPG(ast));
    r = is_mask_for_rhs(std, A_ROPG(ast));
    return (l || r);
  case A_UNOP:
    l = is_mask_for_rhs(std, A_LOPG(ast));
    return l;
  case A_PAREN:
  case A_CONV:
    l = is_mask_for_rhs(std, A_LOPG(ast));
    return l;
  case A_SUBSCR:
    forall = STD_AST(std);
    asn = A_IFSTMTG(forall);
    lhs = A_DESTG(asn);
    if (is_indirection_in_it(ast) && is_legal_rhs(lhs, ast, forall))
      return TRUE;
    return FALSE;
  case A_TRIPLE:
    l = is_mask_for_rhs(std, A_LBDG(ast));
    r = is_mask_for_rhs(std, A_UPBDG(ast));
    i = is_mask_for_rhs(std, A_STRIDEG(ast));
    return (l || r || i);
  case A_INTR:
  case A_FUNC:
  case A_LABEL:
  default:
    return FALSE;
  }
}

/* This is routine does some transformations if lhs array has an indirection
 * subscript. There are two transformations.
 *  1-) Bring indirection array section into form which will be acceptable
 *      by pghpf_scatter such as A(V(V(i))) is not acceptable.
 *       - no indirection of indirection
 *       - it has to be one dimension vector
 *  2-) assign rhs of original assignment into TMP such that
 *      TMP has the same shape as lhs and the same distribution as lhs.
 *      optz.: if rhs has one array and rhs does not have indirection
 *             don't create TMP for rhs.
 *  For example:
 *       forall(i=,j=)  A(V(i),j) = rhs + ..
 *   will be
 *       forall(i=,j=)  TMP(i,j) = rhs + ...
 *       forall(i=,j=)  A(V(i),j) = TMP(i,j)
 */

void
forall_lhs_indirection(int std)
{
  int forall;
  int asn;
  int lhs;
  int src;
  int temp_ast, sptr;
  int newforall;
  int newasn;
  int optype;
  int align;
  int stdf;
  int list;
  int home;

  scalar_lhs_dependency(std);
  scatter_dependency(std);
  forall = STD_AST(std);
  list = A_LISTG(forall);
  asn = A_IFSTMTG(forall);
  src = A_SRCG(asn);
  lhs = A_DESTG(asn);
  align = ALIGNG(left_array_symbol(lhs));
  if (!align)
    return;
  /*    if(!is_indirection_in_it(lhs)) return; */
  if (!is_vector_indirection_in_it(lhs, list))
    return;
  if (!is_legal_lhs(lhs, forall))
    return;
  if (is_duplicate(lhs, list))
    return;
  if (!is_one_idx_for_dim(lhs, list))
    return;
  if (is_multiple_idx_in_list(list))
    return;
  /* if there is mask find a home array from rhs */
  home = 0;
  if (A_IFEXPRG(forall)) {
    if (!find_scatter_rhs(src, forall, &home))
      return;
  } else
    home = lhs;

  optype = -1;
  if (!scatter_class(std)) {
    /* split forall */
    sptr = get_temp_forall(forall, home, std, std, 0, left_subscript_ast(home));
    temp_ast = reference_for_temp_lhs_indirection(sptr, home, forall);
    newforall = mk_stmt(A_FORALL, 0);
    A_LISTP(newforall, A_LISTG(forall));
    A_IFEXPRP(newforall, A_IFEXPRG(forall));
    A_SRCP(newforall, A_SRCG(forall));

    newasn = mk_stmt(A_ASN, 0);
    A_DESTP(newasn, temp_ast);
    A_SRCP(newasn, src);
    A_IFSTMTP(newforall, newasn);
    stdf = add_stmt_before(newforall, std);
    process_forall(stdf);
    A_SRCP(asn, temp_ast);
  }

  A_DESTP(asn, lhs);
  A_IFSTMTP(forall, asn);
  STD_AST(std) = forall;
}

/* This routine checks is whether lhs is in parallizibale form:
 * We can distribute iteration only lhs subscript are:
 *    - forall index,
 *    - scalar,
 *    - vector subscript,
 *    - no indirection of indirection.
 *    - can be legal array section.
 */

LOGICAL
is_legal_lhs(int a, int forall)
{
  int list;
  int i;
  int ndim;
  int asd;
  ADSC *ad;
  int lb;
  int sptr;

  list = A_LISTG(forall);
  do {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      sptr = sptr_of_subscript(a);
      assert(is_array_type(sptr), "is_legal_lhs: must be array", sptr, 4);
      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; i++) {
        if (!is_scalar(ASD_SUBS(asd, i), list) &&
            !is_idx(ASD_SUBS(asd, i), list) &&
            !is_vector_subscript(ASD_SUBS(asd, i), list))
          return FALSE;
        /* don't let LBOUND(A, i) != 1, if there is indirection
         * This will be optimized later, */
        if (is_vector_subscript(ASD_SUBS(asd, i), list)) {
          ad = AD_DPTR(DTYPEG(sptr));
          lb = AD_LWBD(ad, i);
          if (lb != 0 && lb != astb.i1)
            return FALSE;
        }
      }
      a = A_LOPG(a);
    } else {
      interr("is_legal_lhs: must be array or member", a, 4);
    }
  } while (A_TYPEG(a) != A_ID);
  return TRUE;
}

static LOGICAL
is_legal_lhs_for_mask(int a, int forall)
{
  int ast, list;

  list = A_LISTG(forall);
  ast = a;
  do {
    if (A_TYPEG(ast) == A_MEM) {
      ast = A_PARENTG(ast);
    } else if (A_TYPEG(ast) == A_SUBSCR) {
      int i;
      int ndim;
      int asd;
      asd = A_ASDG(ast);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; ++i) {
        if (!is_scalar(ASD_SUBS(asd, i), list) &&
            !is_idx(ASD_SUBS(asd, i), list) &&
            !is_vector_subscript(ASD_SUBS(asd, i), list))
          return FALSE;
      }
      ast = A_LOPG(ast);
    } else {
      interr("is_legal_lhs_for_mask: not subscr or member", A_TYPEG(ast), 3);
    }
  } while (A_TYPEG(ast) != A_ID);
  if (is_duplicate(a, list))
    return FALSE;
  return TRUE;
}

/* don't allow forall(i=1:n,j=istart(i):istop(i) */
LOGICAL
is_multiple_idx_in_list(int list)
{
  int triplet, triplet1;
  int list0, list1;
  int isptr;

  list0 = list;
  for (; list; list = ASTLI_NEXT(list)) {
    triplet = ASTLI_TRIPLE(list);
    isptr = ASTLI_SPTR(list);
    list1 = list0;
    for (; list1; list1 = ASTLI_NEXT(list1)) {
      triplet1 = ASTLI_TRIPLE(list1);
      if (is_name_in_expr(triplet1, isptr))
        return TRUE;
    }
  }
  return FALSE;
}

/* This will return FALSE cases like u(nodes(i,j))
 * Each dimension should have less than equal 1 idx
 * Othervise return false.
 */
static LOGICAL
is_one_idx_for_dim(int a, int list)
{
  while (A_TYPEG(a) != A_ID) {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      int i, ndim, asd;
      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; ++i) {
        int astli, nidx;
        astli = 0;
        nidx = 0;
        search_forall_idx(ASD_SUBS(asd, i), list, &astli, &nidx);
        if (astli == 0)
          continue;
        if (nidx > 1)
          return FALSE;
      }
      a = A_LOPG(a);
    } else {
      interr("is_one_idx_for_dim: not subscript or member", A_TYPEG(a), 3);
    }
  }
  return TRUE;
}

LOGICAL
is_duplicate(int a, int list)
{
  for (; list > 0; list = ASTLI_NEXT(list)) {
    int sptr, found, ast;
    sptr = ASTLI_SPTR(list);
    found = 0;
    ast = a;
    while (A_TYPEG(ast) != A_ID) {
      if (A_TYPEG(ast) == A_MEM) {
        ast = A_PARENTG(ast);
      } else if (A_TYPEG(ast) == A_SUBSCR) {
        int i;
        int ndim;
        int asd;

        asd = A_ASDG(ast);
        ndim = ASD_NDIM(asd);
        for (i = 0; i < ndim; ++i) {
          if (is_name_in_expr(ASD_SUBS(asd, i), sptr))
            ++found;
        }
        ast = A_LOPG(ast);
      } else {
        interr("is_duplicate: not member or subscript", A_TYPEG(ast), 3);
        return FALSE;
      }
    }
    if (found > 1)
      return TRUE;
  }
  return FALSE;
}

LOGICAL
is_scalar(int a, int list)
{
  int astli;
  int nidx;

  astli = 0;
  nidx = 0;
  search_forall_idx(a, list, &astli, &nidx);
  if (nidx == 0 && astli == 0)
    return TRUE;
  return FALSE;
}

LOGICAL
is_idx(int a, int list)
{
  int astli;
  int nidx;

  astli = 0;
  nidx = 0;
  search_forall_idx(a, list, &astli, &nidx);
  if (nidx == 1 && astli) {
    if (mk_id(ASTLI_SPTR(astli)) == a)
      return TRUE;
  }
  return FALSE;
}

static LOGICAL
is_triplet(int a, int list)
{
  int astli;
  int nidx;
  int base, stride;

  astli = 0;
  nidx = 0;
  search_idx(a, list, &astli, &base, &stride);
  if (base && stride && astli)
    return TRUE;
  return FALSE;
}
LOGICAL
is_vector_subscript(int a, int list)
{
  int count;
  int i;
  int asd;
  int ndim;

  if (A_TYPEG(a) != A_SUBSCR)
    return FALSE;
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  count = 0;
  for (i = 0; i < ndim; i++) {
    if (!is_scalar(ASD_SUBS(asd, i), list) && !(is_idx(ASD_SUBS(asd, i), list)))
      return FALSE;
  }

  if (is_scalar(a, list))
    return FALSE;
  return TRUE;
}

/* order2: used for pghpf_permute_section */
/* no: number of elements returned in order2 */
LOGICAL
is_ordered(int lhs, int rhs, int list, int order2[MAXDIMS], int *no)
{
  int asd, ndim;
  int i, j, r, l;
  int count, count1;
  int order[MAXDIMS], order1[MAXDIMS];
  LOGICAL found;
  int astli, nidx;

  /* rhs */
  count = 0;
  for (r = rhs; A_TYPEG(r) != A_ID;) {
    switch (A_TYPEG(r)) {
    case A_MEM:
      r = A_PARENTG(r);
      break;
    case A_SUBSCR:
      asd = A_ASDG(r);
      ndim = ASD_NDIM(asd);
      for (j = 0; j < ndim; ++j) {
        astli = 0;
        nidx = 0;
        search_forall_idx(ASD_SUBS(asd, j), list, &astli, &nidx);
        if (nidx == 1 && astli) {
          assert(count < MAXDIMS, "is_ordered: dimensions > MAXDIMS", count, 4);
          order[count] = ASTLI_SPTR(astli);
          ++count;
        }
      }
      r = A_LOPG(r);
      break;
    default:
      interr("LHS is not subscript, id, or member", r, 4);
    }
  }

  /* lhs */
  count1 = 0;
  for (l = lhs; A_TYPEG(l) != A_ID;) {
    switch (A_TYPEG(l)) {
    case A_MEM:
      l = A_PARENTG(l);
      break;
    case A_SUBSCR:
      asd = A_ASDG(l);
      ndim = ASD_NDIM(asd);
      for (j = 0; j < ndim; ++j) {
        astli = 0;
        nidx = 0;
        search_forall_idx(ASD_SUBS(asd, j), list, &astli, &nidx);
        if (nidx == 1 && astli) {
          assert(count1 < MAXDIMS, "is_ordered: dimensions > MAXDIMS", count1,
                 4);
          order1[count1] = ASTLI_SPTR(astli);
          count1++;
        }
      }
      l = A_LOPG(l);
    }
  }

  for (j = 0; j < count1; ++j)
    for (i = 0; i < count; i++)
      if (order1[j] == order[i])
        order2[j] = i;
  *no = count1;

  /* no transpose accesses between lhs and rhs */
  /* Algorithm:
   * lhs(i,j,k) = rhs(k,i),
   * start with rhs indices,
   * kill lhs indices upto rhs indices you are looking for.
   * if you can not find rhs, this means you are ready to kill it
   * that means it appears before previous rhs index.
   * that is a transpose access.
   */

  for (i = 0; i < count; i++) {
    found = FALSE;
    for (j = 0; j < count1; j++) {
      if (order[i] != order1[j]) {
        order1[j] = 0;
      } else {
        order1[j] = 0;
        found = TRUE;
        break;
      }
    }
    if (!found)
      return FALSE;
  }
  *no = 0;
  return TRUE;
}

/* This routine finds out the dimension of sptr.
 * It takes subscript a(f(i),5,f(j)). It eliminates scalar dimension.
 * It makes an ast for reference sptr.
 *  a(f(i),5,f(j)) --> sptr(f(i),f(j))
 */

static int
reference_for_temp_lhs_indirection(int sptr, int a, int forall)
{
  int subs[MAXDIMS];
  int list;
  int i, j;
  int asd;
  int ndim;
  int astnew;
  int astli;
  int nidx;
  int index_var;
  int triple;
  ADSC *ad;
  int l, u, s;
  int lb, t;

  list = A_LISTG(forall);
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  j = 0;
  /* array will be referenced after communication as follows  */
  for (i = 0; i < ndim; i++) {
    astli = 0;
    nidx = 0;
    search_forall_idx(ASD_SUBS(asd, i), list, &astli, &nidx);
    if (nidx == 1 && astli) {
      index_var = ASTLI_SPTR(astli);
      subs[j] = mk_id(index_var);
      /* normalize astli according to new tmp*/
      /* integer ind(6); integer A(3,6); tmp for A(ind(3:6),3) */
      if (is_vector_subscript(ASD_SUBS(asd, i), list)) {
        triple = ASTLI_TRIPLE(astli);
        l = A_LBDG(triple);
        u = A_UPBDG(triple);
        s = A_STRIDEG(triple);
        ad = AD_DPTR(DTYPEG(sptr));
        lb = AD_LWBD(ad, j);
        if (!lb)
          lb = astb.i1;
        if (!s)
          s = astb.i1;
        t = opt_binop(OP_SUB, subs[j], l, DT_INT);
        t = opt_binop(OP_DIV, t, s, DT_INT);
        t = opt_binop(OP_ADD, t, lb, DT_INT);
        subs[j] = t;
      }
      j++;
    }
  }
  assert(j == rank_of_sym(sptr), "reference_for_temp: rank mismatched", sptr,
         4);
  astnew = mk_subscr(mk_id(sptr), subs, j, DDTG(DTYPEG(sptr)));
  return astnew;
}

/* ast to search */
/* list = pointer of forall indices */
void
search_forall_idx(int ast, int list, int *astli, int *nidx)
{
  int argt, n, i;
  int asd;

  if (!ast)
    return;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    search_forall_idx(A_LOPG(ast), list, astli, nidx);
    search_forall_idx(A_ROPG(ast), list, astli, nidx);
    break;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    search_forall_idx(A_LOPG(ast), list, astli, nidx);
    break;
  case A_CMPLXC:
  case A_CNST:
    break;

  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i)
      search_forall_idx(ARGT_ARG(argt, i), list, astli, nidx);
    break;
  case A_TRIPLE:
    search_forall_idx(A_LBDG(ast), list, astli, nidx);
    search_forall_idx(A_UPBDG(ast), list, astli, nidx);
    if (A_STRIDEG(ast))
      search_forall_idx(A_STRIDEG(ast), list, astli, nidx);
    break;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    n = ASD_NDIM(asd);
    for (i = 0; i < n; ++i)
      search_forall_idx(ASD_SUBS(asd, i), list, astli, nidx);
    search_forall_idx(A_LOPG(ast), list, astli, nidx);
    break;
  case A_SUBSTR:
    search_forall_idx(A_LEFTG(ast), list, astli, nidx);
    search_forall_idx(A_RIGHTG(ast), list, astli, nidx);
    search_forall_idx(A_LOPG(ast), list, astli, nidx);
    break;
  case A_MEM:
    search_forall_idx(A_PARENTG(ast), list, astli, nidx);
    break;
  case A_ID:
    for (i = list; i != 0; i = ASTLI_NEXT(i)) {
      if (A_SPTRG(ast) == ASTLI_SPTR(i)) {
        if (*astli != i) {
          *astli = i;
          (*nidx)++;
        }
      }
    }
    break;
  default:
    interr("search_forall_idx: bad ast type", A_TYPEG(ast), 3);
    break;
  }
}

LOGICAL
is_legal_rhs(int lhs, int rhs, int forall)
{
  int list;
  int i;
  int ndim;
  int asd;

  list = A_LISTG(forall);
  asd = A_ASDG(rhs);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++) {
    if (!is_scalar(ASD_SUBS(asd, i), list) &&
        !is_triplet(ASD_SUBS(asd, i), list) &&
        !is_vector_subscript(ASD_SUBS(asd, i), list))
      return FALSE;
  }
  /*
      if (is_duplicate(rhs, list)) return FALSE;
      if (!is_ordered(lhs, rhs, list, order2, &no)) return FALSE;
  */
  return TRUE;
}

#ifdef FLANG_VSUB_UNUSED
/* This routine takes an array and forall,
 * It returns a list which only has forall index appears
 * in the array subscripts. A(i), forall(i=,j=), return i=..
 */
static int
newforall_list(int arr, int forall)
{
  int astli, base, stride;
  int list;
  int numdim;
  int asd;
  int i;
  int newlist;

  list = A_LISTG(forall);
  asd = A_ASDG(arr);
  numdim = ASD_NDIM(asd);
  start_astli();
  for (i = 0; i < numdim; ++i) {
    astli = 0;
    search_idx(ASD_SUBS(asd, i), list, &astli, &base, &stride);
    if (astli) {
      newlist = add_astli();
      ASTLI_SPTR(newlist) = ASTLI_SPTR(astli);
      ASTLI_TRIPLE(newlist) = ASTLI_TRIPLE(astli);
    }
  }
  return ASTLI_HEAD;
}
#endif

static void
forall_loop_interchange(int std)
{
  int forall, list;
  int asn, lhs;

  forall = STD_AST(std);
  list = A_LISTG(forall);
  if (is_multiple_idx_in_list(list))
    return;

  asn = A_IFSTMTG(forall);
  lhs = A_DESTG(asn);
  if (A_SHAPEG(lhs))
    return;
  start_astli();
  do {
    if (A_TYPEG(lhs) == A_MEM) {
      lhs = A_PARENTG(lhs);
    } else if (A_TYPEG(lhs) == A_SUBSCR) {
      int asd, ndim, i;
      asd = A_ASDG(lhs);
      ndim = ASD_NDIM(asd);
      for (i = ndim - 1; i >= 0; --i) {
        int astli, base, stride;
        /* must look like: c2 +/- c1 * i where i is an index. */
        /* search for an index & do the recursion */
        astli = 0;
        search_idx(ASD_SUBS(asd, i), list, &astli, &base, &stride);
        if (base == 0) {
          /* hopeless */
          return;
        }
        if (astli) {
          int newlist;
          list = delete_astli(list, astli); /* a(i,i) */
          newlist = add_astli();
          ASTLI_SPTR(newlist) = ASTLI_SPTR(astli);
          ASTLI_TRIPLE(newlist) = ASTLI_TRIPLE(astli);
        }
      }
      lhs = A_LOPG(lhs);
    } else if (A_TYPEG(lhs) == A_SUBSTR) {
      return;
    } else {
      interr("forall_loop_interchange: not member/subscript", lhs, 3);
    }
  } while (A_TYPEG(lhs) != A_ID);

  A_LISTP(forall, ASTLI_HEAD);
  A_STDP(forall, std);
  STD_AST(std) = forall;
}

/* this will delete astli from list */
int
delete_astli(int list, int astli)
{
  int newlist;
  int listp;

  start_astli();
  for (listp = list; listp != 0; listp = ASTLI_NEXT(listp))
    if (listp != astli) {
      newlist = add_astli();
      ASTLI_SPTR(newlist) = ASTLI_SPTR(listp);
      ASTLI_TRIPLE(newlist) = ASTLI_TRIPLE(listp);
    }
  return ASTLI_HEAD;
}

/* This routine changes forall whose has a sahpe.
 * For example, forall (j=0:my, k=0:mz) dXc(1,:,j,k) = dXc(1,:,0,0)
 * It uses the same routine with array assignment conversion into forall.
 * That is, fist change A_ASN of forall into forall
 * and then first add original forall indices
 * and then the second forall indices. This makes,
 * forall (j=0:my, k=0:mz, i_1=0:mz) dXc(1,i_1,j,k) = dXc(1,i_1,0,0)
 * OPTIMIZATION:
 * The above algorithm may not access the array with column major order.
 * The order of indices does not  effect the semantic of forall but
 * may effect the performance in some systems.
 */

static void
forall_with_shape(int std)
{
  int shape;
  int asn;
  int src, dest;
  int ast1, ast2;
  int mask;
  int ast;
  int lc;
  int list;

  ast = STD_AST(std);
  asn = A_IFSTMTG(ast);
  src = A_SRCG(asn);
  dest = A_DESTG(asn);
  shape = A_SHAPEG(dest);
  mask = A_IFEXPRG(ast);
  list = A_LISTG(ast);
  lc = 0;
  for (; list; list = ASTLI_NEXT(list))
    lc++;

  if (shape) {
    /* this is an array assignment */
    /* need to create a forall */
    int list;
    ast1 = make_forall(shape, dest, 0, lc);
    ast2 = normalize_forall(ast1, asn, 0);
    A_IFSTMTP(ast1, ast2);
    if (mask)
      mask = normalize_forall(ast1, mask, 0);
    A_IFEXPRP(ast1, mask);
    /* add original forall indices */
    list = concatenate_list(A_LISTG(ast), A_LISTG(ast1));
    A_LISTP(ast1, list);
    A_STDP(ast1, std);
    STD_AST(std) = ast1;
    A_SRCP(ast1, A_SRCG(ast));
    A_OPT1P(ast1, A_OPT1G(ast));
    A_ARRASNP(ast1, A_ARRASNG(ast));
    A_STARTP(ast1, A_STARTG(ast));
    A_NCOUNTP(ast1, A_NCOUNTG(ast));
  }
}

/* this routine take two lists and concatenates them and make a new list */
int
concatenate_list(int list1, int list2)
{
  int listp, newlist;

  start_astli();
  for (listp = list1; listp != 0; listp = ASTLI_NEXT(listp)) {
    newlist = add_astli();
    ASTLI_SPTR(newlist) = ASTLI_SPTR(listp);
    ASTLI_TRIPLE(newlist) = ASTLI_TRIPLE(listp);
  }

  /* add new forall indices */
  for (listp = list2; listp != 0; listp = ASTLI_NEXT(listp)) {
    newlist = add_astli();
    ASTLI_SPTR(newlist) = ASTLI_SPTR(listp);
    ASTLI_TRIPLE(newlist) = ASTLI_TRIPLE(listp);
  }
  return ASTLI_HEAD;
}

/* This routine rewrites those foralls with transformational intrinsics,
 * It takes intrinsic outside of forall and but inside do loop which
 * is constructed from forall statement.
 */

static struct {
  int first;
  int lhs;
  int n;
  int std;
  int pre_std;
} intr_info;

void
rewrite_forall_pure(void)
{
  int std, ast, asn;
  int stdnext;
  int expr;

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    gbl.lineno = STD_LINENO(std);
    ast = STD_AST(std);
    if (A_TYPEG(ast) == A_ASN)
      scatter_dependency_assumsz(std);
    if (A_TYPEG(ast) == A_FORALL) {
      int sclrzd;
      forall_list_call(std);
      sclrzd = forall_semantic(std);
      if (sclrzd) {
        continue;
      }
      if (A_TYPEG(A_IFSTMTG(STD_AST(std))) != A_ASN) {
        scalarize(std, STD_AST(std), FALSE);
        continue;
      }

      init_ftb();
      forall_opt1(ast);
      put_forall_pcalls(std);

      asn = A_IFSTMTG(ast);
      expr = A_IFEXPRG(ast);
      intr_info.first = 1;
      intr_info.lhs = A_DESTG(asn);
      intr_info.std = std;
      intr_info.pre_std = STD_PREV(std);
      if (is_sequentialize_pure(std)) {
        report_comm(std, UGLYPURE_CAUSE);
        scalarize(std, STD_AST(std), FALSE);
      }
      A_OPT1P(ast, 0);
      FREE(ftb.base);
    }
  }
}

static LOGICAL
is_sequentialize_pure(int std)
{
  int forall;
  int asn;
  int dest, src;
  int expr;
  int pstd, past;
  int nd;
  int i;

  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);
  dest = A_DESTG(asn);
  src = A_SRCG(asn);
  expr = A_IFEXPRG(forall);

  if (is_ugly_pure(src) || is_ugly_pure(dest) || is_ugly_pure(expr))
    return TRUE;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NPCALL(nd); i++) {
    pstd = glist(FT_PCALL(nd), i);
    STD_PURE(pstd) = FALSE;
    past = STD_AST(pstd);
    if (is_ugly_pure(past))
      return TRUE;
  }
  return FALSE;
}

/*
 * This routine takes transformational intrinsics out of forall stmt. and
 * puts into do loops. func will returns a  A_ASN which has transformation
 * intrinsic.
 */

static LOGICAL
is_ugly_pure(int ast)
{
  int lhs;
  int std;
  int shape;
  LOGICAL l, r;
  int dtype;
  int asd;
  int numdim;
  int i;
  int sptr;
  int iface;
  int argt, nargs;
  int arg;

  if (ast == 0)
    return FALSE;
  lhs = intr_info.lhs;
  std = intr_info.std;
  shape = A_SHAPEG(ast);
  dtype = A_DTYPEG(ast);
  switch (A_TYPEG(ast)) {
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_SUBSTR:
  case A_MEM:
    return FALSE;
  case A_BINOP:
    l = is_ugly_pure(A_LOPG(ast));
    if (l)
      return TRUE;
    r = is_ugly_pure(A_ROPG(ast));
    if (r)
      return TRUE;
    return FALSE;
  case A_UNOP:
  case A_PAREN:
  case A_CONV:
    l = is_ugly_pure(A_LOPG(ast));
    if (l)
      return TRUE;
    return FALSE;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd);
    assert(numdim > 0 && numdim <= MAXDIMS, "is_ugly_pure: bad numdim", ast, 4);
    for (i = 0; i < numdim; ++i) {
      l = is_ugly_pure(ASD_SUBS(asd, i));
      if (l)
        return TRUE;
    }
    return FALSE;
  case A_TRIPLE:
    l = is_ugly_pure(A_LBDG(ast));
    if (l)
      return TRUE;
    r = is_ugly_pure(A_UPBDG(ast));
    if (r)
      return TRUE;
    l = is_ugly_pure(A_STRIDEG(ast));
    if (l)
      return TRUE;
    return FALSE;
  case A_CALL:
  case A_INTR:
  case A_FUNC:
    sptr = procsym_of_ast(A_LOPG(ast));
    if (A_TYPEG(ast) == A_INTR && INKINDG(sptr) == IK_ELEMENTAL) {
      argt = A_ARGSG(ast);
      nargs = A_ARGCNTG(ast);
      for (i = 0; i < nargs; ++i) {
        l = is_ugly_pure(ARGT_ARG(argt, i));
        if (l)
          return TRUE;
      }
      return FALSE;
    }
    proc_arginfo(sptr, NULL, NULL, &iface);
    if (A_TYPEG(ast) == A_FUNC && iface && is_impure(iface))
      error(488, ERR_Severe, STD_LINENO(std), "subprogram call in FORALL",
            SYMNAME(sptr));

    argt = A_ARGSG(ast);
    nargs = A_ARGCNTG(ast);
    for (i = 0; i < nargs; ++i) {
      arg = ARGT_ARG(argt, i);
      l = is_ugly_pure(arg);
      if (l)
        return TRUE;

      shape = A_SHAPEG(arg);
      /* does not like pure(A(1:n) + b(1:n)) */
      if (shape) {
        if (A_TYPEG(arg) != A_ID && A_TYPEG(arg) != A_SUBSCR &&
            A_TYPEG(arg) != A_MEM && A_TYPEG(arg) != A_INTR &&
            A_TYPEG(arg) != A_FUNC)
          return TRUE;
        /* don't like elemental arg with shape, pure(abs(a(:,i))) */
        if (A_TYPEG(arg) == A_INTR &&
            INKINDG(A_SPTRG(A_LOPG(arg))) == IK_ELEMENTAL)
          return TRUE;
      }
    }
    return FALSE;
  default:
    interr("is_ugly_pure: unexpected ast", ast, 2);
    return TRUE;
  }
}

#ifdef FLANG_VSUB_UNUSED
/* This is to calculate how many DO statements have to be made
   from forall statement and add those before std              */

static int
make_dos(int std)
{
  int forall;
  int newast;
  int stdnext;
  int triplet_list;
  int triplet;
  int index_var;
  int n;

  forall = STD_AST(std);
  stdnext = STD_NEXT(std);

  n = 0;
  triplet_list = A_LISTG(forall);
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    int dovar;
    n++;
    index_var = ASTLI_SPTR(triplet_list);
    triplet = ASTLI_TRIPLE(triplet_list);
    newast = mk_stmt(A_DO, 0);
    dovar = mk_id(index_var);
    A_DOVARP(newast, dovar);
    A_M1P(newast, A_LBDG(triplet));
    A_M2P(newast, A_UPBDG(triplet));
    A_M3P(newast, A_STRIDEG(triplet));
    A_M4P(newast, 0);
    add_stmt_before(newast, std);
  }
  return n;
}
#endif

#ifdef FLANG_VSUB_UNUSED
/* this is to add n enddo statements before std */

static void
make_enddos(int n, int std)
{
  int newast;
  int i;

  for (i = 0; i < n; i++) {
    newast = mk_stmt(A_ENDDO, 0);
    add_stmt_before(newast, std);
  }
}
#endif

static LOGICAL
_contains_call(int astx, LOGICAL *pflag)
{
  if (A_TYPEG(astx) == A_INTR &&
      INKINDG(A_SPTRG(A_LOPG(astx))) != IK_ELEMENTAL) {
    *pflag = TRUE;
    return TRUE;
  }
  return FALSE;
}

/* Return TRUE if AST astx contains an intrinsic or external call. */
LOGICAL
contains_call(int astx)
{
  LOGICAL flag = FALSE;

  if (A_CALLFGG(astx))
    return TRUE;

  ast_visit(1, 1);
  ast_traverse(astx, _contains_call, NULL, &flag);
  ast_unvisit();
  return flag;
}

static LOGICAL
appears_in_expr(int sptr, int expr)
{
  int asd;
  int numdim, i;
  int nargs, argt;

  switch (A_TYPEG(expr)) {
  case A_CMPLXC:
  case A_CNST:
    return FALSE;
  case A_ID:
    if (A_SPTRG(expr) == sptr)
      return TRUE;
    if (is_pointer_dependent(sptr, A_SPTRG(expr)))
      return TRUE;
    if (is_equivalence(sptr, A_SPTRG(expr)))
      return TRUE;
    return FALSE;
  case A_MEM:
    return appears_in_expr(sptr, A_PARENTG(expr));
  case A_BINOP:
    if (appears_in_expr(sptr, A_LOPG(expr)))
      return TRUE;
    if (appears_in_expr(sptr, A_ROPG(expr)))
      return TRUE;
    return FALSE;
  case A_SUBSTR:
  case A_UNOP:
  case A_PAREN:
  case A_CONV:
    return appears_in_expr(sptr, A_LOPG(expr));
  case A_SUBSCR:
    if (appears_in_expr(sptr, A_LOPG(expr)))
      return TRUE;
    asd = A_ASDG(expr);
    numdim = ASD_NDIM(asd);
    assert(numdim > 0 && numdim <= MAXDIMS, "is_dependent: bad numdim", expr,
           4);
    for (i = 0; i < numdim; ++i) {
      if (appears_in_expr(sptr, ASD_SUBS(asd, i)))
        return TRUE;
    }
    return FALSE;
  case A_TRIPLE:
    if (appears_in_expr(sptr, A_LBDG(expr)))
      return TRUE;
    if (appears_in_expr(sptr, A_UPBDG(expr)))
      return TRUE;
    if (A_STRIDEG(expr))
      return appears_in_expr(sptr, A_STRIDEG(expr));
    return FALSE;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      if (appears_in_expr(sptr, ARGT_ARG(argt, i)))
        return TRUE;
    }
    return FALSE;
  case A_LABEL:
  default:
    interr("appears_in_expr: unexpected ast", expr, 2);
    return FALSE;
  }
} /* appears_in_expr */

/* recursive traversal; removes scalar subscripts (assigns to temp)
 * that contain no forall indices and do contain reference to 'sptr' */
static int
remove_scalar_lhs_dependency(int ast, int list, int sptr, int std)
{
  int asd, ndim, i, lop, nlop, nast, changes, subscr[MAXDIMS];
  switch (A_TYPEG(ast)) {
  default:
    return ast;
  case A_SUBSTR:
    lop = A_LOPG(ast);
    nlop = remove_scalar_lhs_dependency(lop, list, sptr, std);
    if (nlop == lop)
      return ast;
    nast = mk_substr(nlop, A_LEFTG(ast), A_RIGHTG(ast), A_DTYPEG(ast));
    return nast;
  case A_MEM:
    lop = A_PARENTG(ast);
    nlop = remove_scalar_lhs_dependency(lop, list, sptr, std);
    if (nlop == lop)
      return ast;
    nast = mk_member(nlop, A_MEMG(ast), A_DTYPEG(ast));
    return nast;
  case A_SUBSCR:
    lop = A_LOPG(ast);
    nlop = remove_scalar_lhs_dependency(lop, list, sptr, std);
    changes = 0;
    if (nlop != lop)
      ++changes;
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      int ss;
      ss = ASD_SUBS(asd, i);
      subscr[i] = ss;
      /* is this a 'scalar' subscript? */
      if (A_SHAPEG(ss) == 0) {
        int astli, nidx;
        astli = nidx = 0;
        search_forall_idx(ss, list, &astli, &nidx);
        if (nidx == 0) {
          /* truly a scalar subscript, no FORALL indices either */
          if (appears_in_expr(sptr, ss)) {
            int temp, tempast, asn;
            temp = sym_get_scalar(SYMNAME(sptr), "ss", DT_INT);
            asn = mk_stmt(A_ASN, 0);
            tempast = mk_id(temp);
            A_DESTP(asn, tempast);
            A_SRCP(asn, ss);
            add_stmt_before(asn, std);
            subscr[i] = tempast;
            ++changes;
          }
        }
      }
    }
    if (changes == 0)
      return ast;
    nast = mk_subscr(nlop, subscr, ndim, A_DTYPEG(ast));
    return nast;
  }
} /* remove_scalar_lhs_dependency */

/* This routine removes any scalar subscripts that might
 * depend on the LHS variable
 * For example,
 *              forall(j=1:N) i(i(1,2),j) = 0
 *  or
 *              forall(j=1:N) i(i(1)%m(1))%m(j) = 0
 * will be rewritten
 *               temp = i(1,2)
 *               forall(j=1:N) i(temp,j) = 0
 *  or
 *		temp = i(1)%m(1)
 *              forall(j=1:N) i(temp)%m(j) = 0
 */

static void
scalar_lhs_dependency(int std)
{
  int forall, list, asn, lhs, sptrlhs, newlhs;
  forall = STD_AST(std);
  list = A_LISTG(forall);
  asn = A_IFSTMTG(forall);
  lhs = A_DESTG(asn);
  sptrlhs = sym_of_ast(lhs);
  newlhs = remove_scalar_lhs_dependency(lhs, list, sptrlhs, std);
  A_DESTP(asn, newlhs);
} /* scalar_lhs_dependency */


/* This routine  is to check whether forall has scatter dependency.
 * Scatter dependency means that same lhs array used as subscript of lhs
 * If it has, it creates temp which is shape array with lhs.
 * For example,
 *              forall(j=1:N) i(i(j)) = 0
 *  or
 *              forall(j=1:N) i(i(j)%m)%m = 0
 * will be rewritten
 *               temp(:) = i(:)
 *               forall(j=1:N) temp(i(j)) = 0
 *               i(:) = temp(:)
 *  or
 *		temp(:) = i(:)%m
 *              forall(j=1:N) temp(i(j)%m) = 0
 *              i(:)%m = temp(:)
 *  or (for SMP)
 *		forall(j=1:N) temp(i(j)%m) = i(i(j)%m)
 *              forall(j=1:N) temp(i(j)%m) = 0
 *		forall(j=1:N) i(i(j)%m) = temp(i(j)%m)
 *              * where j is Openmp do loop index, we cannot 
 *                copy the whole array temp back to array i
 *                because it may overwrite other thread
 *                work-sharing 
 */

static int scatter_dependency_recursion = 0;

static void
scatter_dependency(int std)
{
  int lhs, leftlhs, newleftlhs, l;
  int ast, ast2;
  int asn;
  int asd;
  int subs[MAXDIMS];
  int i;
  int ndim;
  int sptr;
  int temp_ast;
  int src, dest;
  int destsptr;
  int eledtype;
  int forall;
  int subscr[MAXDIMS];
  int shape;
  int std1, forall1, forall2, orig_lhs;
  LOGICAL pointer_dependent;

  if (scatter_dependency_recursion)
    return;

  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);
  lhs = A_DESTG(asn);
  l = lhs;
  leftlhs = 0;
  do {
    switch (A_TYPEG(l)) {
    case A_ID:
      l = 0;
      break;
    case A_MEM:
      l = A_PARENTG(l);
      break;
    case A_SUBSTR:
      l = A_LOPG(l);
      break;
    case A_SUBSCR:
      leftlhs = l;
      l = A_LOPG(l);
      break;
    default:
      interr("scatter_dependency: unexpected ast", l, 4);
      l = 0;
      break;
    }
  } while (l > 0);
  if (leftlhs == 0)
    return;

  sptr = sptr_of_subscript(leftlhs);
  pointer_dependent = FALSE;
  /* this can be improved such that
     only POINTER indirection in LHS */
  if (POINTERG(sptr) && ptr_subs_olap(sptr, lhs))
    pointer_dependent = TRUE;

  if (pointer_dependent || subscr_dependent(lhs, lhs, std, std)) {
    src = A_LOPG(leftlhs);
    eledtype = DDTG(DTYPEG(sptr));
    dest = 0;
    scatter_dependency_recursion = 1;
    /* assume size array must be handled earlier
     */
    if (ASUMSZG(sptr))
      return;
    destsptr = mk_assign_sptr(src, "sc", subscr, eledtype, &dest);
    mk_mem_allocate(mk_id(destsptr), subscr, std, src);

    temp_ast = 0;
    if (STD_PAR(std)) {
      int asn1;

      /* We must keep triplet the same as the index might be omp loop index.
       * The transformation is similar to non-SMP but we must keep the
       * loop indexes the same as original.
       */
      asd = A_ASDG(leftlhs);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; i++) {
        subs[i] = ASD_SUBS(asd, i);
      }
      temp_ast = mk_subscr(mk_id(destsptr), subs, ndim, 
                           DDTG(DTYPEG(destsptr)));
      temp_ast = replace_ast_subtree(lhs, leftlhs, temp_ast);
      forall1 = mk_stmt(A_FORALL, 0);
      A_LISTP(forall1, A_LISTG(forall));
      asn1 = mk_stmt(A_ASN,0);
      A_DESTP(asn1, temp_ast);
      A_SRCP(asn1, lhs);
      A_IFSTMTP(forall1, asn1);
      add_stmt_before(forall1, std);
      orig_lhs = lhs;
    } else {
      /* tmp = leftlhs */
      ast = mk_assn_stmt(dest, src, eledtype);

      /* need to create a forall */
      shape = A_SHAPEG(dest);
      forall1 = make_forall(shape, dest, 0, 0);
      ast2 = normalize_forall(forall1, ast, 0);
      A_IFSTMTP(forall1, ast2);
      A_IFEXPRP(forall1, 0);
      std1 = add_stmt_before(forall1, std);
      process_forall(std1);

    }

    /* change original forall */
    asd = A_ASDG(leftlhs);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++)
      subs[i] = ASD_SUBS(asd, i);
    newleftlhs = mk_subscr(mk_id(destsptr), subs, ndim, DDTG(DTYPEG(destsptr)));
    lhs = replace_ast_subtree(lhs, leftlhs, newleftlhs);
    A_DESTP(asn, lhs);

    if (temp_ast) {
      int asn2;
      forall2 = mk_stmt(A_FORALL, 0);
      A_LISTP(forall2, A_LISTG(forall1));
      asn2 = mk_stmt(A_ASN,0);
      A_DESTP(asn2, orig_lhs);
      A_SRCP(asn2, temp_ast);
      A_IFSTMTP(forall2, asn2);
      std1 = add_stmt_after(forall2, std);
    } else {
      /* leftlhs = TMP */
      ast = mk_assn_stmt(src, dest, eledtype);
      /* need to create a forall */
      shape = A_SHAPEG(src);
      forall2 = make_forall(shape, src, 0, 0);
      ast2 = normalize_forall(forall2, ast, 0);
      A_IFSTMTP(forall2, ast2);
      A_IFEXPRP(forall2, 0);
      std1 = add_stmt_after(forall2, std);
      process_forall(std1);
    }
    mk_mem_deallocate(mk_id(destsptr), std1);
    scatter_dependency_recursion = 0;
  }
}

/* this function is to take scatter_dependency for only assumed size array
 * The other arrays are handle at scatter_dependency()
 * because it is impossible to find upper bound of assumed size array
 * For example,
 * IVEC1(IVEC1(1:5)) = 0 will be
 * allocate(tmp(1:5)
 * tmp = ivec1(1:5)
 * ivec1(tmp) = 0
 */
static void
scatter_dependency_assumsz(int std)
{
  int asn;
  int sptr;
  int shape;
  int lhs, l, leftlhs, newleftlhs;
  int asd;
  int ndim;
  int subs[MAXDIMS];
  int i;

  asn = STD_AST(std);
  lhs = A_DESTG(asn);
  l = lhs;
  leftlhs = 0;
  do {
    switch (A_TYPEG(l)) {
    case A_ID:
      l = 0;
      break;
    case A_MEM:
      l = A_PARENTG(l);
      break;
    case A_SUBSTR:
      l = A_LOPG(l);
      break;
    case A_SUBSCR:
      leftlhs = l;
      l = A_LOPG(l);
      break;
    default:
      interr("scatter_dependency_assumsz: unexpected ast", l, 4);
      l = 0;
      break;
    }
  } while (l > 0);
  if (leftlhs == 0)
    return;
  shape = A_SHAPEG(leftlhs);
  if (shape == 0)
    return;
  sptr = sptr_of_subscript(leftlhs);
  if (!ASUMSZG(sptr))
    return;
  asd = A_ASDG(leftlhs);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++) {
    subs[i] = ASD_SUBS(asd, i);
    subs[i] = take_out_assumsz_array(subs[i], std, sptr);
  }
  newleftlhs = mk_subscr(A_LOPG(leftlhs), subs, ndim, A_DTYPEG(leftlhs));
  lhs = replace_ast_subtree(lhs, leftlhs, newleftlhs);
  A_DESTP(asn, lhs);
}

static int
take_out_assumsz_array(int expr, int std, int sptr)
{
  int l, r, d, o;
  int i, nargs, argt;
  int sptr1;
  int eledtype;
  int dest, destsptr;
  int subscr[MAXDIMS];
  int shape;
  int ast;

  if (expr == 0)
    return expr;
  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = take_out_assumsz_array(A_LOPG(expr), std, sptr);
    r = take_out_assumsz_array(A_ROPG(expr), std, sptr);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = take_out_assumsz_array(A_LOPG(expr), std, sptr);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(expr);
    l = take_out_assumsz_array(A_LOPG(expr), std, sptr);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(expr);
    l = take_out_assumsz_array(A_LOPG(expr), std, sptr);
    return mk_paren(l, d);
  case A_SUBSTR:
    return expr;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = take_out_assumsz_array(ARGT_ARG(argt, i), std, sptr);
    }
    return expr;
  case A_CNST:
  case A_CMPLXC:
  case A_ID:
    return expr;
  case A_MEM:
  case A_SUBSCR:
    shape = A_SHAPEG(expr);
    if (!shape)
      return expr;
    if (sptr != sym_of_ast(expr)) {
      int e;
      /* check any subscripts */
      for (e = expr; e;) {
        int asd, ndim, i, ch;
        switch (A_TYPEG(e)) {
        case A_MEM:
          e = A_PARENTG(e);
          break;
        case A_SUBSCR:
          asd = A_ASDG(e);
          ndim = ASD_NDIM(asd);
          ch = 0;
          for (i = 0; i < ndim; ++i) {
            int ss = ASD_SUBS(asd, i);
            subscr[i] = take_out_assumsz_array(ss, std, sptr);
            if (subscr[i] != ss)
              ch = 1;
          }
          if (ch) {
            int ne;
            ne = mk_subscr(A_LOPG(e), subscr, ndim, A_DTYPEG(e));
            expr = replace_ast_subtree(expr, e, ne);
          }
          e = A_LOPG(e);
          break;
        case A_ID:
          e = 0;
          break;
        default:
          interr("take_out_assumsz_array: unexpected ast", e, 3);
          e = 0;
          break;
        }
      }
      return expr;
    }
    sptr1 = memsym_of_ast(expr);

    eledtype = DDTG(A_DTYPEG(expr));
    destsptr = mk_assign_sptr(expr, "sc", subscr, eledtype, &dest);
    mk_mem_allocate(mk_id(destsptr), subscr, std, expr);
    /* tmp = lhs */
    ast = mk_assn_stmt(dest, expr, eledtype);
    add_stmt_before(ast, std);
    mk_mem_deallocate(mk_id(destsptr), std);
    return dest;

  default:
    return expr;
  }
}

/* This routine is to find an array from expr
 * such that it is going to be used as a rhs for scatter communication.
 * all forall index must appear on rhs arra
 */

static LOGICAL
find_scatter_rhs(int expr, int forall, int *rhs)
{
  int i, nargs, argt;
  int asd;
  int ndim;
  int list;
  LOGICAL find1;

  if (expr == 0)
    return FALSE;

  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    find1 = find_scatter_rhs(A_LOPG(expr), forall, rhs);
    if (find1)
      return TRUE;
    return find_scatter_rhs(A_ROPG(expr), forall, rhs);
  case A_UNOP:
    return find_scatter_rhs(A_LOPG(expr), forall, rhs);
  case A_CONV:
    return find_scatter_rhs(A_LOPG(expr), forall, rhs);
  case A_PAREN:
    return find_scatter_rhs(A_LOPG(expr), forall, rhs);
  case A_MEM:
    return FALSE;
  case A_SUBSTR:
    return FALSE;
  case A_INTR:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      find1 = find_scatter_rhs(ARGT_ARG(argt, i), forall, rhs);
      if (find1)
        return TRUE;
    }
    return FALSE;
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      find1 = find_scatter_rhs(ARGT_ARG(argt, i), forall, rhs);
      if (find1)
        return TRUE;
    }
    return TRUE;
  case A_CNST:
  case A_CMPLXC:
  case A_ID:
    return FALSE;
  case A_SUBSCR:
    list = A_LISTG(forall);
    if (is_one_idx_for_dim(expr, list) && is_all_idx_in_subscript(list, expr)) {
      *rhs = expr;
      return TRUE;
    }

    asd = A_ASDG(expr);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++) {
      find1 = find_scatter_rhs(ASD_SUBS(asd, i), forall, rhs);
      if (find1)
        return TRUE;
    }
    return FALSE;
  case A_TRIPLE:
    return FALSE;
  default:
    interr("find_scatter_rhs: unknown expression", expr, 2);
    return FALSE;
  }
}

static LOGICAL
is_all_idx_in_subscript(int list, int a)
{
  int ndim;
  int asd;
  int i, j;
  int isptr;
  LOGICAL found;

  assert(A_TYPEG(a) == A_SUBSCR, "is_all_idx_in_subscript:must be subscript", a,
         3);
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  for (j = list; j != 0; j = ASTLI_NEXT(j)) {
    isptr = ASTLI_SPTR(j);
    found = FALSE;
    for (i = 0; i < ndim; i++)
      if (is_name_in_expr(ASD_SUBS(asd, i), isptr))
        found = TRUE;
    if (!found)
      return FALSE;
  }
  return TRUE;
}

static int
copy_to_scalar(int ast, int std, int sym)
{
  int nsym, nsymast, asn;
  if (ast == 0)
    return 0;
  nsym = sym_get_scalar(SYMNAME(sym), "ss", DT_INT);
  nsymast = mk_id(nsym);
  asn = mk_stmt(A_ASN, DT_INT);
  A_DESTP(asn, nsymast);
  A_SRCP(asn, ast);
  add_stmt_before(asn, std);
  return nsymast;
} /* copy_to_scalar */

/* check whether the forall bounds might be changed by the forall LHS.
 * if so, copy them to TEMPs */
static int save_list = 0;
static void
forall_bound_dependence(int std)
{
  int forall, list, asn, lhs, sptrlhs, astli, li;
  forall = STD_AST(std);
  list = A_LISTG(forall);
  asn = A_IFSTMTG(forall);
  lhs = A_DESTG(asn);
  sptrlhs = sym_of_ast(lhs);
  li = 0;
  for (astli = list; astli != 0; astli = ASTLI_NEXT(astli)) {
    int triple, lw, up, st, ntriple, nlw, nup, nst;
    triple = ASTLI_TRIPLE(astli);
    nlw = lw = A_LBDG(triple);
    start_astli();
    if (lw != 0 && appears_in_expr(sptrlhs, lw)) {
      /* assign lw to temp */
      nlw = copy_to_scalar(lw, std, ASTLI_SPTR(astli));
      li = add_astli();
      ASTLI_AST(li) = nlw;
      ASTLI_PT(li) = lw;
    }
    nup = up = A_UPBDG(triple);
    if (up != 0 && appears_in_expr(sptrlhs, up)) {
      /* assign up to temp */
      nup = copy_to_scalar(up, std, ASTLI_SPTR(astli));
      li = add_astli();
      ASTLI_AST(li) = nup;
      ASTLI_PT(li) = up;
    }
    nst = st = A_STRIDEG(triple);
    if (st != 0 && appears_in_expr(sptrlhs, st)) {
      /* assign st to temp */
      nst = copy_to_scalar(st, std, ASTLI_SPTR(astli));
      li = add_astli();
      ASTLI_AST(li) = nst;
      ASTLI_PT(li) = st;
    }
    if (nlw != lw || nup != up || nst != st) {
      ntriple = mk_triple(nlw, nup, nst);
      ASTLI_TRIPLE(astli) = ntriple;
    }
  }
  if (li == 0) {
    save_list = 0;
  } else {
    save_list = ASTLI_HEAD;
  }
} /* forall_bound_dependence */

extern int rewrite_opfields;
static void
forall_bound_dependence_fix(int prevstd, int nextstd)
{
  /* visit statements between prevstd and nextstd.
   * replace any appearances of the forall limits by the temps created */
  int std, li;
  ast_visit(1, 1);
  rewrite_opfields = 0x3; /* copy opt1 and opt2 fields */
  for (li = save_list; li; li = ASTLI_NEXT(li)) {
    ast_replace(ASTLI_PT(li), ASTLI_AST(li));
  }
  for (std = STD_NEXT(prevstd); std != nextstd; std = STD_NEXT(std)) {
    int ast;
    ast = STD_AST(std);
    ast = ast_rewrite(ast);
    A_STDP(ast, std);
    STD_AST(std) = ast;
  }
  ast_unvisit();
} /* forall_bound_dependence_fix */

/* inquire whether a pointer array has subscripts which may overlap */

static LOGICAL
ptr_subs_olap(int parr, int a)
{
  do {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      int asd;
      int ndim, i;
      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; ++i)
        if (can_ptr_olap(parr, ASD_SUBS(asd, i)))
          return TRUE;
      a = A_LOPG(a);
    } else if (A_TYPEG(a) == A_ID) {
      return FALSE;
    } else {
      interr("ptr_subs_olap: LHS not subscript or member", a, 4);
    }
  } while (1);
}

/* inquire whether expression has array */
static LOGICAL
can_ptr_olap(int parr, int ast)
{

  int argt, n, i;
  int sptr, lop;
  int rank;
  int dtype;

  if (ast == 0)
    return FALSE;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    if (can_ptr_olap(parr, A_LOPG(ast)))
      return TRUE;
    return can_ptr_olap(parr, A_ROPG(ast));
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return can_ptr_olap(parr, A_LOPG(ast));
  case A_CMPLXC:
  case A_CNST:
    break;
  case A_MEM:
    if (can_ptr_olap(parr, A_MEMG(ast)))
      return TRUE;
    return can_ptr_olap(parr, A_PARENTG(ast));
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      if (can_ptr_olap(parr, ARGT_ARG(argt, i)))
        return TRUE;
    }
    break;

  case A_TRIPLE:
    if (can_ptr_olap(parr, A_LBDG(ast)))
      return TRUE;
    if (can_ptr_olap(parr, A_UPBDG(ast)))
      return TRUE;
    if (can_ptr_olap(parr, A_STRIDEG(ast)))
      return TRUE;
    break;
  case A_SUBSCR:
    lop = A_LOPG(ast);
    switch (A_TYPEG(lop)) {
    case A_ID:
      sptr = A_SPTRG(lop);
      break;
    case A_MEM:
      sptr = A_SPTRG(A_MEMG(lop));
      break;
    default:
      return FALSE;
    }
    if (STYPEG(sptr) == ST_DESCRIPTOR || DESCARRAYG(sptr))
      /* set in rte.c */
      return FALSE;
    if (sptr == parr)
      return TRUE;
    if (XBIT(58, 0x80000000))
      return TRUE;
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY) {
      rank = ADD_NUMDIM(DTYPEG(parr));
      if (POINTERG(sptr)) {
        if (rank == ADD_NUMDIM(dtype))
          return TRUE;
      }
    }
    break;
  case A_ID:
    sptr = A_SPTRG(ast);
    if (sptr == parr)
      return TRUE;
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY) {
      if (XBIT(58, 0x80000000))
        return TRUE;
      rank = ADD_NUMDIM(DTYPEG(parr));
      if (POINTERG(sptr)) {
        if (rank == ADD_NUMDIM(dtype))
          return TRUE;
      }
    }
    break;
  default:
    interr("can_ptr_olap: bad opc", ast, 3);
    return TRUE;
  }
  return FALSE;
}
