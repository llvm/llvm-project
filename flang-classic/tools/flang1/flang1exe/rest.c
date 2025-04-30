/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file rest.c
    \brief various ast transformations
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
#include "ccffinfo.h"
#include "pragma.h"
#include "dpm_out.h"
#include "optimize.h"
#define RTE_C
#include "rte.h"
#undef RTE_C
#include "rtlRtns.h"

static int _transform_func(int, int);
static LOGICAL stride_1_dummy(int, int, int);
static LOGICAL stride_1_section(int, int, int, int);
#ifdef FLANG_REST_UNUSED
static LOGICAL is_expr_has_function(int);
static void transform_extrinsic(int, int);
#endif
void remove_alias(int, int);
static int first_element_from_section(int);
static void copy_arg_to_seq_tmp(int, int, int, int, int, int, int *, int *,
                                LOGICAL, LOGICAL, LOGICAL);
static int temp_type_descriptor(int ast, int std);
static LOGICAL is_seq_dummy(int, int, int);
static LOGICAL needs_type_in_descr(SPTR, int);
static LOGICAL is_optional_char_dummy(int, int, int);
#ifdef FLANG_REST_UNUSED
static void check_nonseq_element(int, int, int, int);
#endif
static void check_pure_interface(int, int, int);
static void handle_seq_section(int, int, int, int, int *, int *, LOGICAL, int);
static int mk_descr_from_section(int, DTYPE, int);
static int need_copyout(int entry, int loc);
static LOGICAL is_desc_needed(int, int, int);
static LOGICAL continuous_section(int, int, int, int);
static int transform_all_call(int std, int ast);
static int remove_subscript_expressions(int ast, int std, int sym);
static void set_descr_tag(int descr, int tag, int std);
static int get_descr_arg(int ele, SPTR sptr, int std);
static int get_descr_or_placeholder_arg(SPTR inface_arg, int ele, int std);

/* rhs_is_dist argument seems to be useless at this point */
int
insert_comm_before(int std, int ast, LOGICAL *rhs_is_dist, LOGICAL is_subscript)
{
  int l, r, d, o;
  int l1, l2, l3, l4;
  int a;
  int i, nargs, argt, ag;
  int asd;
  int ndim;
  int subs[MAXDIMS];
  int dest, nextd, newd;

  if (!ast)
    return ast;
  if (ast < astb.firstuast)
    return ast;

  a = ast;
  switch (A_TYPEG(ast)) {
  /* statements */
  case A_ASN:
    dest = A_DESTG(a);
    for (d = dest; d; d = nextd) {
      nextd = 0;
      switch (A_TYPEG(d)) {
      case A_SUBSTR:
        nextd = A_LOPG(d);
        break;
      case A_MEM:
        nextd = A_PARENTG(d);
        break;
      case A_SUBSCR:
        asd = A_ASDG(d);
        ndim = ASD_NDIM(asd);
        for (i = 0; i < ndim; ++i) {
          subs[i] =
              insert_comm_before(std, ASD_SUBS(asd, i), rhs_is_dist, TRUE);
        }
        newd = mk_subscr(A_LOPG(d), subs, ndim, A_DTYPEG(d));
        dest = replace_ast_subtree(dest, d, newd);
        nextd = A_LOPG(d);
        break;
      }
    }
    A_DESTP(a, dest);
    l = insert_comm_before(std, A_SRCG(a), rhs_is_dist, is_subscript);
    A_SRCP(a, l);
    return a;
  case A_IF:
  case A_IFTHEN:
    l = insert_comm_before(std, A_IFEXPRG(a), rhs_is_dist, is_subscript);
    A_IFEXPRP(a, l);
    return a;

  case A_ELSE:
  case A_ELSEIF:
  case A_ENDIF:
    return a;
  case A_AIF:
    l = insert_comm_before(std, A_IFEXPRG(a), rhs_is_dist, is_subscript);
    A_IFEXPRP(a, l);
    return a;
  case A_GOTO:
    return a;
  case A_CGOTO:
    l = insert_comm_before(std, A_LOPG(a), rhs_is_dist, is_subscript);
    A_LOPP(a, l);
    return a;
  case A_AGOTO:
  case A_ASNGOTO:
    return a;
  case A_DO:
    l1 = insert_comm_before(std, A_M1G(a), rhs_is_dist, is_subscript);
    l2 = insert_comm_before(std, A_M2G(a), rhs_is_dist, is_subscript);
    if (A_M3G(a))
      l3 = insert_comm_before(std, A_M3G(a), rhs_is_dist, is_subscript);
    else
      l3 = 0;
    if (A_M4G(a))
      l4 = insert_comm_before(std, A_M4G(a), rhs_is_dist, is_subscript);
    else
      l4 = 0;
    A_M1P(a, l1);
    A_M2P(a, l2);
    A_M3P(a, l3);
    A_M4P(a, l4);
    return a;
  case A_DOWHILE:
    l = insert_comm_before(std, A_IFEXPRG(a), rhs_is_dist, is_subscript);
    A_IFEXPRP(a, l);
    return a;
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
  case A_ENTRY:
    return a;
  case A_ICALL:
  case A_CALL:
    return a;
  case A_REDISTRIBUTE:
  case A_REALIGN:
    return a;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    return a;
  case A_ALLOC:
    /*	interr("insert_comm_before: ALLOC not handled", std, 2); */
    return a;
  case A_WHERE:
  case A_ELSEWHERE:
  case A_ENDWHERE:
    interr("insert_comm_before: WHERE stmt found", std, 3);
    return a;
  case A_FORALL:
  case A_ENDFORALL:
    interr("insert_comm_before: FORALL stmt found", std, 3);
    return a;
  case A_COMMENT:
  case A_COMSTR:
    return a;
  case A_LABEL:
    return a;
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = insert_comm_before(std, A_LOPG(a), rhs_is_dist, is_subscript);
    r = insert_comm_before(std, A_ROPG(a), rhs_is_dist, is_subscript);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = insert_comm_before(std, A_LOPG(a), rhs_is_dist, is_subscript);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(a);
    l = insert_comm_before(std, A_LOPG(a), rhs_is_dist, is_subscript);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(a);
    l = insert_comm_before(std, A_LOPG(a), rhs_is_dist, is_subscript);
    return mk_paren(l, d);
  case A_MEM:
    d = A_DTYPEG(a);
    l = insert_comm_before(std, A_PARENTG(a), rhs_is_dist, is_subscript);
    return mk_member(l, A_MEMG(a), A_DTYPEG(A_MEMG(a)));
  case A_SUBSTR:
    d = A_DTYPEG(a);
    l1 = insert_comm_before(std, A_LOPG(a), rhs_is_dist, is_subscript);
    l2 = l3 = 0;
    if (A_LEFTG(a))
      l2 = insert_comm_before(std, A_LEFTG(a), rhs_is_dist, is_subscript);
    if (A_RIGHTG(a))
      l3 = insert_comm_before(std, A_RIGHTG(a), rhs_is_dist, is_subscript);
    return mk_substr(l1, l2, l3, d);
  case A_INTR:
    if (INKINDG(A_SPTRG(A_LOPG(a))) == IK_INQUIRY)
      return a;
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ag =
          insert_comm_before(std, ARGT_ARG(argt, i), rhs_is_dist, is_subscript);
      ARGT_ARG(argt, i) = ag;
    }
    A_ARGSP(a, argt);
    return a;
  case A_FUNC:
    return a;
  case A_CNST:
  case A_CMPLXC:
    return a;
  case A_ID:
    return a;
  case A_SUBSCR:
    asd = A_ASDG(a);
    ndim = ASD_NDIM(asd);
    assert(ndim <= MAXDIMS, "insert_comm_before: ndim too big", std, 4);
    for (i = 0; i < ndim; ++i) {
      subs[i] = insert_comm_before(std, ASD_SUBS(asd, i), rhs_is_dist, TRUE);
    }
    l = A_LOPG(a);
    a = mk_subscr(l, subs, ndim, A_DTYPEG(a));
    return a;

  case A_TRIPLE:
    l1 = insert_comm_before(std, A_LBDG(a), rhs_is_dist, is_subscript);
    l2 = insert_comm_before(std, A_UPBDG(a), rhs_is_dist, is_subscript);
    l3 = insert_comm_before(std, A_STRIDEG(a), rhs_is_dist, is_subscript);
    return mk_triple(l1, l2, l3);
  case A_HOFFSET:
    return a;
  case A_HOWNERPROC:
  case A_HLOCALOFFSET:
  case A_HLOCALIZEBNDS:
  case A_HCYCLICLP:
  case A_HOVLPSHIFT:
  case A_HCSTART:
  case A_HCFINISH:
  case A_HGETSCLR:
    return a;
  case A_BARRIER:
    return a;
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
  case A_ENDATOMIC:
    return a;
  case A_NOBARRIER:
    return a;
  case A_MP_PARALLEL:
  case A_MP_ENDPARALLEL:
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_BARRIER:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_PDO:
  case A_MP_ENDPDO:
  case A_MP_SECTIONS:
  case A_MP_ENDSECTIONS:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_PRE_TLS_COPY:
  case A_MP_BCOPYIN:
  case A_MP_COPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_COPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_TASK:
  case A_MP_TASKLOOP:
  case A_MP_TASKFIRSTPRIV:
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_ETASKDUP:
  case A_MP_TASKLOOPREG:
  case A_MP_ETASKLOOPREG:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
  case A_MP_BMPSCOPE:
  case A_MP_EMPSCOPE:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_FLUSH:
  case A_PREFETCH:
  case A_PRAGMA:
  case A_MP_TASKGROUP:
  case A_MP_ETASKGROUP:
  case A_MP_TARGET:
  case A_MP_ENDTARGET:
  case A_MP_TEAMS:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_TARGETUPDATE:
  case A_MP_TARGETDATA:
  case A_MP_ENDTARGETDATA:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_CANCEL:
  case A_MP_CANCELLATIONPOINT:
  case A_MP_ATOMICREAD:
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
  case A_MP_MAP:
  case A_MP_TARGETLOOPTRIPCOUNT:
  case A_MP_EMAP:
  case A_MP_EREDUCTION:
  case A_MP_BREDUCTION:
  case A_MP_REDUCTIONITEM:
    return a;
  default:
    interr("insert_comm_before: unknown expression", std, 2);
    return a;
  }
}

static int
_transform_func(int std, int ast)
{
  int l, r, d, o;
  int l1, l2, l3;
  int a;
  int i, nargs, argt, ag;
  int asd;
  int ndim;
  int subs[MAXDIMS];

  a = ast;
  if (!a)
    return a;
  switch (A_TYPEG(ast)) {
  case A_LABEL:
    return a;
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = _transform_func(std, A_LOPG(a));
    r = _transform_func(std, A_ROPG(a));
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = _transform_func(std, A_LOPG(a));
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(a);
    l = _transform_func(std, A_LOPG(a));
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(a);
    l = _transform_func(std, A_LOPG(a));
    return mk_paren(l, d);
  case A_MEM:
    d = A_DTYPEG(a);
    l = _transform_func(std, A_PARENTG(a));
    return mk_member(l, A_MEMG(a), A_DTYPEG(A_MEMG(a)));
  case A_SUBSTR:
    d = A_DTYPEG(a);
    l1 = _transform_func(std, A_LOPG(a));
    l2 = l3 = 0;
    if (A_LEFTG(a))
      l2 = _transform_func(std, A_LEFTG(a));
    if (A_RIGHTG(a))
      l3 = _transform_func(std, A_RIGHTG(a));
    return mk_substr(l1, l2, l3, d);
  case A_INTR:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ag = _transform_func(std, ARGT_ARG(argt, i));
      ARGT_ARG(argt, i) = ag;
    }
    return a;
  case A_FUNC:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ag = _transform_func(std, ARGT_ARG(argt, i));
      ARGT_ARG(argt, i) = ag;
    }
    transform_call(std, a);
    return a;
  case A_CNST:
  case A_CMPLXC:
    return a;
  case A_ID:
    return a;
  case A_SUBSCR:
    asd = A_ASDG(a);
    ndim = ASD_NDIM(asd);
    assert(ndim <= MAXDIMS, "_transform_func: ndim too big", std, 4);
    for (i = 0; i < ndim; ++i)
      subs[i] = _transform_func(std, ASD_SUBS(asd, i));
    l = A_LOPG(a);
    a = mk_subscr(l, subs, ndim, A_DTYPEG(a));
    return a;

  case A_TRIPLE:
    l1 = _transform_func(std, A_LBDG(a));
    l2 = _transform_func(std, A_UPBDG(a));
    l3 = _transform_func(std, A_STRIDEG(a));
    return mk_triple(l1, l2, l3);

  default:
    interr("_transform_func: unknown expression", std, 2);
    return a;
  }
}

static void
insert_comm(int std, int ast, LOGICAL lhs_is_dist)
{
  LOGICAL rhs_is_dist;

  /* This is for a scalar statement, so we just need to retrieve the
   * values into temp scalars.
   */
  rhs_is_dist = FALSE;
  ast = insert_comm_before(std, ast, &rhs_is_dist, FALSE);
  STD_AST(std) = ast;
  A_STDP(ast, std);
}

void
transform_ast(int std, int ast)
{
  LOGICAL lhs_is_dist = FALSE;

  /* transform a single ast; involves inserting guards,
     communication, etc. */
  /* This ast isn't in a forall */
  /* Now go through & determine what communication is needed for this
   * statement.
   */
  /* guard the AST if the LHS isn't local */
  switch (A_TYPEG(ast)) {
  case A_ASN:
    break;
  default:
    insert_comm(std, ast, lhs_is_dist);
    break;
  }
}

#ifdef FLANG_REST_UNUSED
/* This routine is to search function from expression. */
static LOGICAL
is_expr_has_function(int expr)
{
  int i, nargs, argt;
  int asd;
  int ndim;
  LOGICAL find1;

  if (expr == 0)
    return FALSE;

  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    find1 = is_expr_has_function(A_LOPG(expr));
    if (find1)
      return TRUE;
    return is_expr_has_function(A_ROPG(expr));
  case A_UNOP:
    return is_expr_has_function(A_LOPG(expr));
  case A_CONV:
    return is_expr_has_function(A_LOPG(expr));
  case A_PAREN:
    return is_expr_has_function(A_LOPG(expr));
  case A_MEM:
    return is_expr_has_function(A_PARENTG(expr));
  case A_SUBSTR:
    find1 = is_expr_has_function(A_LOPG(expr));
    if (find1)
      return TRUE;
    if (A_LEFTG(expr)) {
      find1 = is_expr_has_function(A_LOPG(expr));
      if (find1)
        return TRUE;
    }
    if (A_RIGHTG(expr)) {
      find1 = is_expr_has_function(A_LOPG(expr));
      if (find1)
        return TRUE;
    }
    return FALSE;
  case A_INTR:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      find1 = is_expr_has_function(ARGT_ARG(argt, i));
      if (find1)
        return TRUE;
    }
    return FALSE;
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      find1 = is_expr_has_function(ARGT_ARG(argt, i));
      if (find1)
        return TRUE;
    }
    return TRUE;
  case A_CNST:
  case A_CMPLXC:
    return FALSE;
  case A_ID:
    return FALSE;
  case A_SUBSCR:
    asd = A_ASDG(expr);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++) {
      find1 = is_expr_has_function(ASD_SUBS(asd, i));
      if (find1)
        return TRUE;
    }
    return FALSE;
  case A_TRIPLE:
    find1 = is_expr_has_function(A_LBDG(expr));
    if (find1)
      return TRUE;
    find1 = is_expr_has_function(A_UPBDG(expr));
    if (find1)
      return TRUE;
    find1 = is_expr_has_function(A_STRIDEG(expr));
    if (find1)
      return TRUE;
    return FALSE;
  default:
    interr("is_expr_has_function: unknown expression", expr, 2);
    return FALSE;
  }
}
#endif

int pghpf_type_sptr = 0;

static void
declare_type(void)
{
  int sptr;
  int commonsptr;
  int dtype;

  commonsptr = getsymbol("pghpf_type");
  if (gbl.internal > 1 && INTERNALG(commonsptr)) {
    commonsptr = insert_sym(commonsptr);
  }
  if (STYPEG(commonsptr) == ST_CMBLK) {
    pghpf_type_sptr = CMEMFG(commonsptr);
    return;
  }
  if (STYPEG(commonsptr) == ST_VAR && SCG(commonsptr) == SC_CMBLK) {
    pghpf_type_sptr = commonsptr;
    return;
  }
  if (XBIT(70, 0x80000000) && STYPEG(commonsptr) == ST_VAR &&
      SCG(commonsptr) == SC_BASED) {
    pghpf_type_sptr = commonsptr;
    return;
  }
  STYPEP(commonsptr, ST_CMBLK);
  SCOPEP(commonsptr, stb.curr_scope);
  if (gbl.internal > 1) {
    INTERNALP(commonsptr, 1);
  }
  DTYPEP(commonsptr, DT_INT);
  DCLDP(commonsptr, 1);
  HCCSYMP(commonsptr, 1);
  SCP(commonsptr, SC_CMBLK);
  SYMLKP(commonsptr, gbl.cmblks); /* link into list of common blocks */
  gbl.cmblks = commonsptr;

  dtype = get_array_dtype(1, DT_INT);
  ADD_ZBASE(dtype) = ADD_LWBD(dtype, 0) = ADD_LWAST(dtype, 0) =
      mk_isz_cval(-43, astb.bnd.dtype);
  ADD_UPBD(dtype, 0) = ADD_UPAST(dtype, 0) = mk_isz_cval(43, astb.bnd.dtype);
  ADD_MLPYR(dtype, 0) = astb.bnd.one;
  ADD_NUMELM(dtype) = mk_isz_cval(67, astb.bnd.dtype);

  sptr = insert_sym(commonsptr);
  STYPEP(sptr, ST_VAR);
  SCOPEP(sptr, stb.curr_scope);
  if (gbl.internal > 1) {
    INTERNALP(sptr, 1);
  }
  DTYPEP(sptr, dtype);
  DCLDP(sptr, 1);
  HCCSYMP(sptr, 1);
  SCP(sptr, SC_CMBLK);
  pghpf_type_sptr = sptr;
#if defined(TARGET_WIN)
  if (!XBIT(70, 0x80000000)) {
    DLLP(commonsptr, DLL_IMPORT);
  }
#endif
  if (XBIT(70, 0x80000000)) {
    int sptr1;
    sptr1 = getsymbol("pghpf_typep");
    if (gbl.internal > 1) {
      INTERNALP(sptr1, 1);
    }
    SCP(sptr, SC_BASED);
    MIDNUMP(sptr, sptr1);
    STYPEP(sptr1, ST_VAR);
    DTYPEP(sptr1, DT_PTR);
    DCLDP(sptr1, 1);
    HCCSYMP(sptr1, 1);
    SCP(sptr1, SC_CMBLK);
    sptr = sptr1;
  }
  CMEMFP(commonsptr, sptr);
  CMEMLP(commonsptr, sptr);
  CMBLKP(sptr, commonsptr);
  SYMLKP(sptr, NOSYM);
} /* declare_type */

static short __pghpf_type[] = {
    /*
     *  These values should be the same as what's in
     *      rte/hpf/src/const.c
     *  Also, these values must correspond to the values of the
     *  enum, _pghpf_type, in dtypeutl.c in the increasing order
     *  from -<max enum value> to <max enum value>, inclusive.
     */
    -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29,
    -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14,
    -13, -12, -11, -10, -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,   1,
    2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
    17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43};

static int
pghpf_type(int ss)
{
  int ast, subs[MAXDIMS];

  if (!XBIT(57, 0x8000) && ss >= -43 && ss <= 43) {
    ast = mk_isz_cval(__pghpf_type[ss + 43], astb.bnd.dtype);
    return ast;
  }
  ast = mk_isz_cval(ss, astb.bnd.dtype);

  if (ss >= -43 && ss <= 43) {
    subs[0] = ast;
    if (pghpf_type_sptr == 0)
      declare_type();
    ast = mk_id(pghpf_type_sptr);
    ast = mk_subscr(ast, subs, 1, astb.bnd.dtype);
  }
  return ast;
} /* pghpf_type */

static int
copy_to_scalar(int ast, int std, int sym)
{
  int nsym, nsymast, asn;
  int sptr;

  if (ast == 0)
    return 0;
  switch (A_TYPEG(ast)) {
  case A_ID:   /* leave alone */
  case A_CNST: /* leave alone */
    break;
  case A_SUBSCR: /* leave alone if this is a section descriptor */
    if (A_TYPEG(A_LOPG(ast)) == A_ID) {
      int sptr;
      sptr = A_SPTRG(A_LOPG(ast));
      if (DESCARRAYG(sptr))
        break;
    } else if (A_TYPEG(A_LOPG(ast)) == A_MEM) {
      sptr = A_SPTRG(A_MEMG(A_LOPG(ast)));
      if (DESCARRAYG(sptr))
        break;
    }
    FLANG_FALLTHROUGH;
  default:
    if (!pure_gbl.local_mode) {
      LOGICAL rhs_is_dist = FALSE;
      ast = insert_comm_before(std, ast, &rhs_is_dist, FALSE);
    }
    nsym = sym_get_scalar(SYMNAME(sym), "ss", astb.bnd.dtype);
    nsymast = mk_id(nsym);
    asn = mk_stmt(A_ASN, astb.bnd.dtype);
    A_DESTP(asn, nsymast);
    A_SRCP(asn, ast);
    add_stmt_before(asn, std);
    ast = nsymast;
    break;
  }
  return ast;
} /* copy_to_scalar */

/*
 * given an AST pointer to a variable reference, go through any * subscripts:
 * 1. scalar subscript nontrivial expression: copy expression to a temp,
 *    replace by the temp.
 * 2. triplet: go through the triplet elements, look for nontrivial expressions
 *    as above.
 * 3. otherwise, leave alone.
 */
static int
remove_subscript_expressions(int ast, int std, int sym)
{
  int parent, nparent;
  int asd, ndim, i, ss, nss, changes;
  int subs[MAXDIMS];
  switch (A_TYPEG(ast)) {
  case A_MEM:
    parent = A_PARENTG(ast);
    nparent = remove_subscript_expressions(parent, std, sym);
    if (nparent != parent) {
      ast = mk_member(nparent, A_MEMG(ast), A_DTYPEG(ast));
    }
    break;
  case A_SUBSCR:
    changes = 0;
    parent = A_LOPG(ast);
    nparent = remove_subscript_expressions(parent, std, sym);
    if (parent != nparent)
      ++changes;
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      ss = ASD_SUBS(asd, i);
      nss = ss;
      if (A_TYPEG(ss) == A_TRIPLE) {
        int lb, ub, st, nlb, nub, nst;
        lb = A_LBDG(ss);
        ub = A_UPBDG(ss);
        st = A_STRIDEG(ss);
        nlb = copy_to_scalar(lb, std, sym);
        if (A_TYPEG(ub) == A_BINOP && A_LOPG(ub) == nlb) {
          nub = ub;
        } else {
          nub = copy_to_scalar(ub, std, sym);
        }
        nst = copy_to_scalar(st, std, sym);
        if (nlb == lb && nub == ub && nst == st) {
          nss = ss;
        } else {
          nss = mk_triple(nlb, nub, nst);
        }
      } else if (A_SHAPEG(ss) == 0) {
        nss = copy_to_scalar(ss, std, sym);
      }
      subs[i] = nss;
      if (ss != nss)
        ++changes;
    }
    if (changes) {
      ast = mk_subscr(nparent, subs, ndim, A_DTYPEG(ast));
    }
    break;
  }
  return ast;
} /* remove_subscript_expressions */

/* given ast, find the A_SUBSCR that has a section subscript */
static int
find_section_subscript(int ast)
{
  int a;
  for (a = ast; a != 0;) {
    int lop;
    switch (A_TYPEG(a)) {
    case A_SUBSCR:
      /* it must have a shape; its parent
       * must be an A_ID, or an A_MEM whose parent has no shape */
      if (!A_SHAPEG(a)) {
        return a; /* probably an error */
      }
      lop = A_LOPG(a);
      if (A_TYPEG(lop) == A_ID) {
        return a;
      } else if (A_TYPEG(lop) == A_MEM) {
        int parent;
        parent = A_PARENTG(lop);
        if (!A_SHAPEG(parent)) {
          /* this is the A_SUBSCR we're looking for */
          return a;
        }
        a = parent;
      } else {
        return ast; /* error */
      }
      break;
    case A_MEM:
      a = A_PARENTG(a);
      break;
    case A_ID:
    default:
      return ast; /* shouldn't get here */
    }
  }
  return ast;
} /* find_section_subscript */

static int
transform_section_arg(int ele, int std, int callast, int entry, int *descr,
                      int argnbr)
{
  int sptr;
  int dummy_sptr;
  int sec;
  int retval;
  int secss, secsptr;

  dummy_sptr = 0;
  retval = 0;
  *descr = 0;
  if (A_TYPEG(ele) != A_SUBSCR || !A_SHAPEG(ele)) {
    return ele;
  }

  sptr = sptr_of_subscript(ele);
  ele = remove_subscript_expressions(ele, std, sym_of_ast(ele));
  if (A_TYPEG(callast) != A_ICALL && is_seq_dummy(entry, ele, argnbr)) {
    handle_seq_section(entry, ele, argnbr, std, &retval, descr, 0, 0);
  } else if (XBIT(57, 0x10000) && A_TYPEG(callast) != A_ICALL &&
             stride_1_dummy(entry, ele, argnbr)) {
    if (!stride_1_section(entry, ele, argnbr, std)) {
      handle_seq_section(entry, ele, argnbr, std, &retval, descr, 1, 0);
    } else if (XBIT(57, 0x100000) &&
               continuous_section(entry, ele, argnbr, 1)) {
      /* pghpf_template(newdescriptor,1,extent,1,extent...)
       * pghpf_instance(newdescriptor,appropriate)
       * pass a(1,1,1)+newdescriptor */
      ele = convert_subscript(ele);
      secss = find_section_subscript(ele);
      sec = make_simple_template_from_ast(secss, std,
                                          needs_type_in_descr(entry, argnbr));
      retval = first_element_from_section(ele);
      if (SCG(sptr) == SC_DUMMY && CLASSG(sptr)) {
        retval = gen_poly_element_arg(retval, sptr, std);
      }
      if (POINTERG(sptr) && is_whole_array(ele)) {
        retval = A_LOPG(retval);
      }
      *descr = check_member(A_LOPG(secss), mk_id(sec));
    } else {
      /* pghpf_sect(section,olddescriptor,bounds...)
       *  pass array+section  */
      ele = convert_subscript(ele);
      secss = find_section_subscript(ele);
      if (XBIT(57, 0x200000)) {
#define SECTZBASE 0x00400000
        sec = make_sec_from_ast(secss, std, std, 0, SECTZBASE);
        retval = first_element_from_section(ele);
        if (POINTERG(sptr) && is_whole_array(ele)) {
          retval = A_LOPG(retval);
        }
      } else {
        sec = make_sec_from_ast(secss, std, std, 0, 0);
        secsptr = sptr_of_subscript(secss);
        retval = check_member(A_LOPG(secss), mk_id(secsptr));
        retval = replace_ast_subtree(ele, secss, retval);
      }
      *descr = check_member(A_LOPG(secss), mk_id(sec));
    }
  }
#ifdef RESHAPEDG
  else if (SCG(sptr) == SC_BASED && RESHAPEDG(sptr) &&
           continuous_section(entry, ele, argnbr, 1)) {
    ele = convert_subscript(ele);
    secss = find_section_subscript(ele);
    sec = make_simple_template_from_ast(secss, std,
                                        needs_type_in_descr(entry, argnbr));
    retval = first_element_from_section(ele);
    *descr = check_member(A_LOPG(secss), mk_id(sec));
  }
#endif
  else {
    secss = find_section_subscript(ele);
    sec = make_sec_from_ast(secss, std, std, 0, 0);
    secsptr = sptr_of_subscript(secss);
    retval = check_member(A_LOPG(secss), mk_id(secsptr));
    retval = replace_ast_subtree(ele, secss, retval);
    *descr = check_member(A_LOPG(secss), mk_id(sec));
  }
  return retval;
}

void
copy_surrogate_to_bnds_vars(DTYPE dt_dest, int parent_dest, DTYPE dt_src,
                            int parent_src, int std)
{
  ADSC *addest;
  ADSC *adsrc;
  int ndim;
  int i;
  int mult = astb.bnd.one;

  if (DTY(dt_dest) != TY_ARRAY)
    return;
  addest = AD_DPTR(dt_dest);
  adsrc = AD_DPTR(dt_src);
  ndim = AD_NUMDIM(addest);
  for (i = 0; i < ndim; i++) {
    int lwast_dest = add_parent_to_bounds(parent_dest, AD_LWAST(addest, i));
    int lwast_src = add_parent_to_bounds(parent_src, AD_LWAST(adsrc, i));
    int upast_dest = add_parent_to_bounds(parent_dest, AD_UPAST(addest, i));
    int upast_src = add_parent_to_bounds(parent_src, AD_UPAST(adsrc, i));
    int extntast_dest =
        add_parent_to_bounds(parent_dest, AD_EXTNTAST(addest, i));
    int astasgn = mk_assn_stmt(lwast_dest, lwast_src, astb.bnd.dtype);
    add_stmt_before(astasgn, std);
    astasgn = mk_assn_stmt(upast_dest, upast_src, astb.bnd.dtype);
    add_stmt_before(astasgn, std);
    astasgn = mk_assn_stmt(
        extntast_dest, mk_extent_expr(lwast_dest, upast_dest), astb.bnd.dtype);
    add_stmt_before(astasgn, std);
    if (i) {
      int lwast_dest =
          add_parent_to_bounds(parent_dest, AD_LWAST(addest, i - 1));
      int upast_dest =
          add_parent_to_bounds(parent_dest, AD_UPAST(addest, i - 1));
      int extent = mk_binop(OP_SUB, upast_dest, lwast_dest, astb.bnd.dtype);
      extent = mk_binop(OP_ADD, extent, astb.bnd.one, astb.bnd.dtype);
      mult = mk_binop(OP_MUL, extent, mult, astb.bnd.dtype);
      astasgn = mk_assn_stmt(AD_MLPYR(addest, i), mult, astb.bnd.dtype);
      add_stmt_before(astasgn, std);
      mult = AD_MLPYR(addest, i);
    }
  }
  {
    int zbase_src = add_parent_to_bounds(parent_src, AD_ZBASE(adsrc));
    int zbase_dest = add_parent_to_bounds(parent_dest, AD_ZBASE(addest));
    int zbase = mk_binop(OP_SUB, astb.bnd.one, zbase_src, astb.bnd.dtype);
    int astasgn = mk_assn_stmt(zbase_dest, zbase, astb.bnd.dtype);
    add_stmt_before(astasgn, std);
  }
}

void
copy_desc_to_bnds_vars(int sptrdest, int desc, int memdesc, int std)
{
  int dt;
  int astasgn;
  ADSC *addest;
  int ndim;
  int i;
  int mult;
  int dest_sdsc;

  dt = DTYPEG(sptrdest);
  if (DTY(dt) != TY_ARRAY)
    return;
  addest = AD_DPTR(dt);
  ndim = AD_NUMDIM(addest);
  mult = astb.bnd.one;
  dest_sdsc = SDSCG(sptrdest);
  for (i = 0; i < ndim; i++) {
    int a;
    a = AD_LWAST(addest, i);
    if (A_TYPEG(a) == A_ID) {
      astasgn = mk_assn_stmt(AD_LWAST(addest, i),
                             check_member(memdesc, get_global_lower(desc, i)),
                             astb.bnd.dtype);
      add_stmt_before(astasgn, std);
    }

    a = AD_UPAST(addest, i);
    if (A_TYPEG(a) == A_ID) {
      a = check_member(memdesc, get_extent(desc, i));
      a = mk_binop(OP_SUB, a, mk_isz_cval(1, astb.bnd.dtype), astb.bnd.dtype);
      a = mk_binop(OP_ADD, AD_LWAST(addest, i), a, astb.bnd.dtype);

      astasgn = mk_assn_stmt(AD_UPAST(addest, i), a, astb.bnd.dtype);
      add_stmt_before(astasgn, std);
    }

    if (ALLOCATTRG(sptrdest)) {
      a = AD_EXTNTAST(addest, i);
      if (A_TYPEG(a) == A_ID) {
        astasgn = mk_assn_stmt(
            AD_EXTNTAST(addest, i),
            mk_extent_expr(AD_LWAST(addest, i), AD_UPAST(addest, i)),
            astb.bnd.dtype);
        add_stmt_before(astasgn, std);
      }
    }
  }
}

static void
get_invoking_proc_desc(int sptr, int ast, int std)
{
  int tmp, dtype, sdsc, newargt;
  int func, astnew;

  if (!is_procedure_ptr(sptr))
    return;
  sdsc = SDSCG(sptr);
  if (STYPEG(sptr) == ST_MEMBER) {
    sdsc  = get_member_descriptor(sptr);
    if (sdsc <= NOSYM) {
      sdsc = SDSCG(sptr);
    }
  } else {
    sdsc = SDSCG(sptr);
  }

  if (sdsc <= NOSYM) {
    get_static_descriptor(sptr);
    sdsc = SDSCG(sptr);
  }

  if (STYPEG(sdsc) == ST_MEMBER) {
    dtype = DTYPEG(sptr);
    tmp  = getcctmp_sc('d', sem.dtemps++, ST_VAR, dtype, sem.sc);
    POINTERP(tmp, 1);
    DTYPEP(tmp, dtype);
    get_static_descriptor(tmp); 
    A_INVOKING_DESCP(ast, mk_id(SDSCG(tmp)));
    newargt = mk_argt(2);
    ARGT_ARG(newargt, 0) = A_INVOKING_DESCG(ast);
    ARGT_ARG(newargt, 1) = check_member(A_LOPG(ast), mk_id(sdsc));
    func = mk_id(sym_mkfunc_nodesc(mkRteRtnNm(RTE_copy_proc_desc), DT_NONE));
    astnew = mk_func_node(A_CALL, func, 2, newargt);
    add_stmt_before(astnew, std);
  }
}

/** \brief The following code takes a function or subroutine call and adds
   section
    descriptors for the array arguments at the end of the argument list,
    if the callee has an explicit interface.

    For example, call `foo(a1,a2,a3)` will be transformed
    call `foo(a1,a2,a3, sec_a1, sec_a2, sec_a3)`.

    It does not accept array expressions as argument nor array funcs.
    For accelerator routines, it also adds arguments to match the
    device-reflected copies

    Rule:
    <pre>
    Argument Type           Address         Descriptor
    ___________________     ___________     ___________________
    scalar                  &scalar         --
    array elem              &array_elem     --
    array expression        &temp           section descriptor (if
   assumed-shape)
    array expression        &temp           -- (else)
    array section           &array          section descriptor (if assumed-shape
                                            and stride-1)
    array section           &temp           section descriptor (if assumed-shape
                                            and not-stride-1)
    array section           &array          -- (else if contiguous)
    array section           &temp           -- (else)
    ptr array section       &ptr            static descriptor (if pointer dummy)
    ptr array section       &temp           section descriptor (if
   assumed-shape,
                                            call added)
    ptr array section       &temp           -- (else, call added)
    func/subroutine dummy   &function       --
    </pre>

    If there is an explicit interface, and the dummy argument has the REFLECTED
    attribute, then the actual argument must have a device resident copy,
    either from a region, data region, implicit data region, or must be itself
    reflected.  In that case, the address of the reflected copy is passed just
    after the dummy, and if there is a section descriptor, the address of the
    section descriptor for the reflected copy is passed just after the section
    descriptor.
 */
void
transform_call(int std, int ast)
{
  int ele;
  int argt, nargs;
  int i, newj, newi;
  int sptr;
  int newargt, newnargs;
  int ty;
  int entry;
  int dscptr;
  int iface; /* sptr of explicit interface; could be zero */
  int inface_arg;
  int retval, descr;
  int is_hcc;
  int is_ent;
  int istart;
  int f_descr;
  int tbp_inv; /* passed object argument number. 0 if N/A */

  entry = procsym_of_ast(A_LOPG(ast));
  if (STYPEG(entry) == ST_MEMBER && CLASSG(entry) && CCSYMG(entry) &&
      VTABLEG(entry)) {
    fix_class_args(VTABLEG(entry));
    if (!NOPASSG(entry)) {
      tbp_inv = find_dummy_position(VTABLEG(entry), PASSG(entry));
      if (tbp_inv == 0)
        tbp_inv = max_binding_invobj(VTABLEG(entry), INVOBJG(BINDG(entry)));
    } else {
      tbp_inv = 0;
    }
  }
  else {
    tbp_inv = 0;
  }
  if (STYPEG(entry) == ST_MEMBER && CLASSG(entry) && CCSYMG(entry) &&
      VTABLEG(entry) && NOPASSG(entry)) {
    int sptr3;
    entry = VTABLEG(entry);

    sptr3 = pass_sym_of_ast(A_LOPG(ast));
    if (sptr3 != sym_of_ast(A_LOPG(ast)) && CLASSG(sptr3) &&
        STYPEG(sptr3) == ST_MEMBER) {
      /* The type bound procedure is invoked by a member of a
       * derived type. Since it's declared CLASS, it must have a
       * type descriptor in the derived type. Need to assign the
       * type from the embedded type descriptor to the section
       * descriptor.
       */
      int tmpv = get_tmp_descr(DTYPEG(sptr3));
      int sptrsdsc = SDSCG(tmpv);
      int  dest_ast = mk_id(sptrsdsc);

      gen_set_type(dest_ast, ast, std, TRUE, FALSE);
      A_INVOKING_DESCP(ast, dest_ast);

    }
  } else if (STYPEG(entry) == ST_MEMBER && CLASSG(entry) && CCSYMG(entry) &&
             VTABLEG(entry)) {
    int sptr3;
    sptr3 = pass_sym_of_ast(A_LOPG(ast));
    if (!CLASSG(sptr3) && STYPEG(sptr3) == ST_ARRAY && !SDSCG(sptr3) &&
        DESCRG(sptr3)) {
      /* Hack to resolve dynamic call when there is no descriptor to
       * hold the type. Use direct call in this case.
       */
      int entry_ast = mk_id((entry = VTABLEG(entry)));
      A_LOPP(ast, entry_ast);
    }
  }
  /* pure routine can only call other pure routines */
  check_pure_interface(entry, std, ast);

  if (!is_procedure_ptr(entry)) {
    is_hcc = HCCSYMG(entry);
    is_ent = 1;
  } else {
    is_hcc = 0;
    is_ent = 0;
    get_invoking_proc_desc(entry,ast,std);
  }
  if (is_ent && NODESCG(entry)) {
    if (STYPEG(entry) != ST_INTRIN && !EXPSTG(entry))
      return;
    argt = A_ARGSG(ast);
    nargs = A_ARGCNTG(ast);
    newargt = mk_argt(nargs);
    for (i = 0; i < nargs; ++i) {
      ele = ARGT_ARG(argt, i);
      ARGT_ARG(newargt, i) = ele;
      if (ele == 0)
        continue;
      switch (A_TYPEG(ele)) {
      case A_SUBSCR:
        /*
        sptr = sptr_of_subscript(ele);
        */
        if (A_SHAPEG(ele)) { /* Array Section  */
          ARGT_ARG(newargt, i) = A_LOPG(ele);
        }
        break;
      default:
        break;
      }
    }
    A_ARGSP(ast, newargt);
    A_ARGCNTP(ast, nargs);
    return;
  }
  if (A_TYPEG(ast) == A_INTR && INKINDG(entry) == IK_ELEMENTAL)
    return;

  proc_arginfo(entry, NULL, &dscptr, &iface);
  nargs = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);
  newnargs = nargs;
  istart = 0;
  f_descr = 0;

  if (dscptr && MVDESCG(entry)) {
    f_descr = aux.dpdsc_base[dscptr + 0];
    if (f_descr && needs_descriptor(f_descr)) {
      istart = 1;
    }
  }
  /* how many section descriptors, reflected copies do we need? */
  for (i = 0; i < nargs; ++i) {
    int needdescr = 0;
    inface_arg = 0;
    if (dscptr)
      inface_arg = aux.dpdsc_base[dscptr + i];
    needdescr = needs_descriptor(inface_arg);
    if (inface_arg && CLASSG(inface_arg)) {
      ++newnargs;
    } else if (needdescr || is_hcc) {
      ++newnargs;
    }
#if DEBUG
    else if (is_ent && strcmp(SYMNAME(entry), mkRteRtnNm(RTE_show)) == 0) {
      ++newnargs;
    }
#endif
    else if (XBIT(57, 0x4000000) && i == 0 && is_ent &&
             (strcmp(SYMNAME(entry), "pgi_get_descriptor") == 0 ||
              strcmp(SYMNAME(entry), "pgi_put_descriptor") == 0))
      ++newnargs;
  }

  if (!CFUNCG(entry) && !dscptr) {
    /* check to see if we need implicit procedure descriptors */
    for (i = 0; i < nargs; ++i) {
      ele = ARGT_ARG(argt, i);
      if (A_TYPEG(ele) == A_ID && IS_PROC(STYPEG(sym_of_ast(ele))) ) {
        ++newnargs;
      }
    }
  } else if (sem.which_pass > 0 && dscptr) {
    /* make sure interfaces of procedure arguments match interfaces of
     * dummy arguments.
     */
    for (i = 0; i < nargs; ++i) {
      ele = ARGT_ARG(argt, i);
      if (A_TYPEG(ele) == A_ID && IS_PROC(STYPEG(sym_of_ast(ele))) ) {
        int proc = sym_of_ast(ele);
        int dpdsc = 0, dpdsc2 = 0;
        inface_arg = aux.dpdsc_base[dscptr + i];
        proc_arginfo(proc, NULL, &dpdsc, NULL);
        proc_arginfo(inface_arg, NULL, &dpdsc2, NULL);
        if (IS_PROC(STYPEG(inface_arg)) && (!TYPDG(proc) || dpdsc != 0) &&
            (!TYPDG(inface_arg) || dpdsc2 != 0) &&
            !cmp_interfaces_strict(proc, inface_arg, 
                                  (IGNORE_ARG_NAMES | RELAX_STYPE_CHK |
                                   RELAX_INTENT_CHK | RELAX_PURE_CHK_1))) {
          error(1009,ERR_Severe,gbl.lineno,SYMNAME(proc),SYMNAME(inface_arg));
        }
      }
    }
  }
        
  newargt = mk_argt(newnargs);
  newi = 0;
  /* put original arguments, and the in-line reflected copies */
  for (i = 0; i < nargs; ++i) {
    ele = ARGT_ARG(argt, i);
    if (!dscptr) {
      inface_arg = 0;
    } else {
      inface_arg = aux.dpdsc_base[dscptr + i];
    }
    /* initialize */
    if (A_TYPEG(ele) == A_FUNC && CFUNCG(entry)) {
      SPTR arg = memsym_of_ast(ele);
      if (CFUNCG(arg)) {
        /* When a BIND(C) function result is used as an argument
         * to another BIND(C) function, assign the result to a temp
         * to ensure that the result is passed by value, not by reference.
         */
        int tmp_ast, assn_ast;
        SPTR tmp = getccsym_sc('d', sem.dtemps++, DTY(DTYPEG(arg)) == TY_ARRAY 
                               ? ST_ARRAY : ST_VAR, SC_LOCAL);
        DTYPEP(tmp, DTYPEG(arg));
        tmp_ast = mk_id(tmp);
        assn_ast = mk_assn_stmt(tmp_ast, ele, DTYPEG(arg));
        add_stmt_before(assn_ast, std);
        ele = tmp_ast;
        ARGT_ARG(argt, i) = ele;
      }
    }
    ARGT_ARG(newargt, newi) = ele;
    ++newi;

  }
  newj = newi;
  newi = istart;
  for (i = istart; i < nargs; ++i) {
    int needdescr = 0;
    ele = ARGT_ARG(argt, i);
    if (!dscptr) {
      inface_arg = 0;
      needdescr = 0;
      if (is_hcc)
        needdescr = 1;

      if (!CFUNCG(entry) && !dscptr && A_TYPEG(ele) == A_ID && 
          IS_PROC(STYPEG(sym_of_ast(ele)))) {
        /* add implicit procedure descriptor argument */
        int tmp = get_proc_ptr(sym_of_ast(ele));
        if (INTERNALG(sym_of_ast(ele))) {
          add_ptr_assign(mk_id(tmp), ele, std);
        }
        ARGT_ARG(newargt, newj) = mk_id(SDSCG(tmp));
        ++newj;
      }

#if DEBUG
      if (is_ent && strcmp(SYMNAME(entry), mkRteRtnNm(RTE_show)) == 0) {
        needdescr = 1;
      }
#endif
      if (XBIT(57, 0x4000000) && i == 0 && is_ent &&
          (strcmp(SYMNAME(entry), "pgi_get_descriptor") == 0 ||
           strcmp(SYMNAME(entry), "pgi_put_descriptor") == 0))
        needdescr = 1;
    } else {
      inface_arg = aux.dpdsc_base[dscptr + i];
      needdescr = needs_descriptor(inface_arg);
      /* actually, only for pointer or assumed-shape arguments dummy */
      if (XBIT(54, 0x80) && inface_arg > NOSYM && ast_is_sym(ele) &&
          needs_descriptor(memsym_of_ast(ele))) {
        /* Generate contiguity check at call-site */
        gen_contig_check(mk_id(inface_arg), ele, 0, gbl.lineno, true, std); 
      }
    }
    if (ele == 0) {
      ARGT_ARG(newargt, newi) = ele;
      ++newi;
      continue;
    }
    switch (A_TYPEG(ele)) {
    case A_LABEL: /* Alternate return call sub(*10) */
      ARGT_ARG(newargt, newi) = ele;
      ++newi;
      if (needdescr) {
        ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg, ele,
                                                               std);
        ++newj;
      }
      break;

    /* expressions */
    case A_UNOP:
      if (A_OPTYPEG(ele) == OP_REF) {
        A_LOPP(ele,
               transform_section_arg(A_LOPG(ele), std, ast, entry, &descr, i));
      }
      if (A_OPTYPEG(ele) == OP_VAL || A_OPTYPEG(ele) == OP_LOC ||
          A_OPTYPEG(ele) == OP_REF) {
        /* optional arguments come here as well */
        /* if char, ptr0 becomes ptr0c */
        if (is_optional_char_dummy(entry, ele, i))
          ele = astb.ptr0c;
        ARGT_ARG(newargt, newi) = ele;
        ++newi;
        if (needdescr || (inface_arg && CLASSG(inface_arg))) {
          if (ele == astb.ptr0 || ele == astb.ptr0c)
            ty = 0;
          else
            ty = dtype_to_arg(A_DTYPEG(ele));
          ARGT_ARG(newargt, newj) = pghpf_type(ty);
          ++newj;
        }
        break;
      }
      FLANG_FALLTHROUGH;
    case A_CNST:
      if (DTY(A_DTYPEG(ele)) == TY_HOLL) {
        /* treat as sequence-associated */
        ARGT_ARG(newargt, newi) = ele;
        ++newi;
        if (needdescr) {
          ty = -1 * dtype_to_arg(A_DTYPEG(ele));
          ARGT_ARG(newargt, newj) = pghpf_type(ty);
          ++newj;
        }
        break;
      }
      FLANG_FALLTHROUGH;
    case A_BINOP:
    case A_PAREN:
    case A_CMPLXC:
    case A_INTR:
    case A_CONV:
    case A_SUBSTR:
      /* should have been assigned to a temp already */
      assert(!A_SHAPEG(ele), "transform_call:Array Expression can't be here",
             ele, 3);
      if (A_TYPEG(ele) == A_INTR && A_OPTYPEG(ele) == I_NULL) {
        /* NULL() call */
        ARGT_ARG(newargt, newi) = mk_cnst(stb.k0);
        ++newi;
        if (needdescr) {
          ARGT_ARG(newargt, newj) = mk_cnst(stb.k0);
          newj++;
        }
        break;
      }
      ARGT_ARG(newargt, newi) = ele;
      ++newi;
      if (needdescr) {
        ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg, ele,
                                                               std);
        ++newj;
      }
      if (is_unl_poly(inface_arg)) {
        /* this happens with expr passed in */
        int descr = temp_type_descriptor(ele, std);
        ARGT_ARG(newargt, newj) = descr;
        ++newj;
      }
      break;

    case A_FUNC:
      /* should have been assigned to a temp already */
      ARGT_ARG(newargt, newi) = ele;
      ++newi;
      assert(!A_SHAPEG(ele), "transform_call:Array Expression can't be here",
             ele, 3);
      if (needdescr) {
        ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg, ele,
                                                               std);
        ++newj;
      } else if (CLASSG(inface_arg)) {
        int descr;
        DTYPE dty = A_DTYPEG(ele);
        if (DTY(dty) != TY_DERIVED) {
          descr = temp_type_descriptor(ele, std);
        } else {
          /* Use function dtype result, else use dtype of function */
          SPTR st_type;
          SPTR fval = FVALG(sym_of_ast(A_LOPG(ele)));
          if (fval) {
            sptr = fval;
          } else {
            DTYPE dty = A_DTYPEG(ele);
            sptr = DTY(dty + 3);
          }
          st_type = get_static_type_descriptor(sptr);
          assert(st_type != 0, "failed to get type descriptor", 0, ERR_Fatal);
          descr = mk_id(st_type);
        }
        ARGT_ARG(newargt, newj) = descr;
        ++newj;
      }
      break;

    case A_MEM:
    case A_ID:
      /* might be a whole array */
      sptr = memsym_of_ast(ele);
      if (CLASSG(sptr) && VTABLEG(sptr) && BINDG(sptr)) {
        /* member is a type bound procedure, so get the pass arg */
        sptr = pass_sym_of_ast(ele);
      }
      /* function or subroutine argument */
      if (STYPEG(sptr) == ST_PROC) {
        ARGT_ARG(newargt, newi) = ele;
        ++newi;
        needdescr = needs_descriptor(inface_arg);
        if (needdescr) {
          if (STYPEG(sptr) == ST_PROC && (SCG(sptr) != SC_DUMMY || 
              SDSCG(sptr))) {
            if (SCG(sptr) != SC_DUMMY) {
              int tmp = get_proc_ptr(sptr);
              if (INTERNALG(sptr)) {
                add_ptr_assign(mk_id(tmp), ele, std);
              }
              ARGT_ARG(newargt, newj) = mk_id(SDSCG(tmp));
            } else {
              ARGT_ARG(newargt, newj) = mk_id(SDSCG(sptr));
            }
          } else {
            ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg,
                                                                   ele, std);
          }
          if (INTERNALG(entry)) {
            int tmp = get_proc_ptr(entry);
            A_INVOKING_DESCP(ast, mk_id(SDSCG(tmp)));
          }
          ++newj;
        }
        break;
      }
      /* was there an interface */
      if (inface_arg) {
        int sptrsdsc;
        int need_surr;
        int scope;
        int dty;
        int unl_poly;
      class_arg:
        unl_poly = is_unl_poly(inface_arg);
        if (!ALLOCDESCG(sptr) &&
            (unl_poly ||
             ((ALLOCATTRG(sptr) || POINTERG(sptr)) && !ALLOCDESCG(sptr) &&
              (needdescr || ALLOCATTRG(sptr)) &&
              (CLASSG(inface_arg) || ALLOCDESCG(inface_arg)) &&
              (DTY(DTYPEG(inface_arg)) != TY_ARRAY)))) {
          /* Set type to descriptor if we're dealing
           * with an allocatable/pointer descriptor or if
           * we have an unlimited polymorphic dummy arg. In
           * the latter case, we call check_alloc_ptr_type()
           * with flag 2 which forces a descriptor on the
           * actual argument with type information. This descriptor
           * is then passed to the unlimited polymorphic's dummy
           * descriptor below.
           */
          /* Also execute this code if the interface arg
           * does not need a descriptor, but the actual arg is
           * allocatable. That way, the type gets added to the actual's
           * descriptor.
           */
          if (!needdescr && !CLASSG(sptr) && DTY(DTYPEG(sptr)) == TY_DERIVED) {
            /* Actual argument is not polymorphic and does not require a
             * section descriptor. So, we will just give it a static
             * type descriptor if it does not already have one. Also note
             * that this only applies to derived type scalars. If sptr's
             * dtype is not derived type, then we may need to execute one of
             * the else branches which handles arrays, polymorphic ojects, and
             * simple scalars like reals and integers (which can occur
             * when the dummy argument is class(*)).
             */
            if (!SDSCG(sptr)) {
              get_static_type_descriptor(sptr);
            }
          } else { 
            /* make sure we assign the type of the actual argument to a
             * descriptor argument. This descriptor argument may be a 
             * new descriptor if the actual argument does not normally take
             * a descriptor or the argument's (previously assigned) descriptor
             * if the argument requires a descriptor.
             */
            check_alloc_ptr_type(sptr, std, 0, unl_poly ? 2 : 1, 0, 0,
                                 STYPEG(sptr) == ST_MEMBER ? ele : 0);
            if (!needdescr && unl_poly) { 
              /* initialize the descriptor only if it's a new descriptor
               * (i.e., the actual argument normally does not take a 
               *  descriptor).
               */
              int descr_length_ast =
                    symbol_descriptor_length_ast(sptr, 0 /*no AST*/);
              if (descr_length_ast > 0) {
                int length_ast = get_value_length_ast(DT_NONE, 0, sptr,
                                                      DTYPEG(sptr), 0);
                if (length_ast > 0)
                  add_stmt_before(mk_assn_stmt(descr_length_ast, length_ast,
                                               astb.bnd.dtype), std);
              }
            }
          }
        }
        dty = DTYPEG(sptr);
        if (DTY(dty) == TY_ARRAY)
          dty = DTY(dty + 1);
        if ((!unl_poly || !DESCRG(sptr) || CLASSG(sptr) || !needdescr ||
             SDSCG(sptr) || DTY(dty) == TY_DERIVED) &&
            /*(CLASSG(inface_arg) && !needdescr) ||
                    (ALLOCDESCG(inface_arg) && needdescr)*/ CLASSG(inface_arg)) {
          int tmp = 0;
          if (i == (tbp_inv-1) && CLASSG(sptr) && !MONOMORPHICG(sptr) &&
              A_TYPEG(ele) == A_SUBSCR && !ELEMENTALG(VTABLEG(entry))) {
            /* We have a polymorphic subscripted pass argument. Need to
             * compute its address based on the polymorphic size of the
             * object.
             */
            /* Use temporary variable instead of assign directly.
             * Compiling with gcc may cause the direct assignment to fail,
             * but clang not. because the function gen_poly_element_arg()
             * will change astb.argt.stg_base in mk_argt() when there are not
             * enough room for "argt".
             */
            int element_addr = gen_poly_element_arg(ele, sptr, std);
            ARGT_ARG(newargt, newi) = element_addr;
          } else if (A_TYPEG(ele) == A_SUBSCR) {
            /* This case occurs when we branch from
             * the A_SUBSCR case below to the class_arg label above.
             */
            ARGT_ARG(newargt, newi) = ele;
          } else {
            ARGT_ARG(newargt, newi) = check_member(ele, mk_id(sptr));
          }
          ++newi;
          scope = SCOPEG(sptr); /* get scope of dummy arg */
          while (STYPEG(scope) == ST_ALIAS)
            scope = SYMLKG(scope);
          if (A_TYPEG(ele) == A_SUBSCR && CLASSG(inface_arg) &&
              STYPEG(sptr) != ST_MEMBER && unl_poly && !SDSCG(sptr)) {
            /* Handle unlimited polymorphic array element */
            int tmpv =
                getcctmp_sc('d', sem.dtemps++, ST_VAR, A_DTYPEG(ele), sem.sc);
            check_alloc_ptr_type(tmpv, std, 0, 2, 0, 0, 0);
            sptrsdsc = SDSCG(tmpv);
          } else if (DESCRG(sptr) && !SDSCG(sptr)) {
            /* Use Array Descriptor since section descriptor
             * has not yet been set.
             */
            if (A_TYPEG(ele) == A_SUBSCR && CLASSG(inface_arg) &&
                !CLASSG(sptr) && DTY(DTYPEG(inface_arg)) != TY_ARRAY) {
              sptrsdsc = get_static_type_descriptor(sptr);
            } else {
              sptrsdsc = DESCRG(sptr);
              DESCUSEDP(sptr, 1);
            }
          } else if (CLASSG(sptr) && DTY(DTYPEG(sptr)) == TY_ARRAY &&
                     /*STYPEG(sptr) == ST_MEMBER &&*/ SDSCG(sptr)) {
            /* Need to get descriptor member */
            if (STYPEG(sptr) == ST_MEMBER) {
              if (STYPEG(SDSCG(sptr)) == ST_MEMBER) {
                sptrsdsc = SDSCG(sptr);
              } else {
                sptrsdsc = get_member_descriptor(sptr);
              }
            } else {
              sptrsdsc = SDSCG(sptr);
            }
            /* Create temporary descriptor if the argument is subscripted or
             * if the passed object argument (denoted with tbp_inv) is a
             * derived type component and the declared type is abstract.
             */
            if (A_TYPEG(ele) == A_SUBSCR ||
                (i == (tbp_inv-1) && (STYPEG(sptrsdsc) == ST_MEMBER ||
                                      ABSTRACTG(VTABLEG(entry))))) {
              /* Create temporary descriptor argument for the
               * the element.
               */

              int dest_ast;
              int dtype = A_DTYPEG(ele);
              int tmpv = get_tmp_descr(dtype);
              int src_ast = check_member(ele, mk_id(sptrsdsc));

              sptrsdsc = SDSCG(tmpv);
              dest_ast = mk_id(sptrsdsc);
              if (i == (tbp_inv-1)) {
                A_INVOKING_DESCP(ast, dest_ast);
              }
              gen_set_type(dest_ast, src_ast, std, TRUE, FALSE);
            }
          } else {
            sptrsdsc = get_type_descr_arg2(scope, sptr);
          }

          if (!sptrsdsc) {
            sptrsdsc = SDSCG(sptr);
          }
          if (CLASSG(sptr) && STYPEG(sptr) == ST_MEMBER &&
              STYPEG(sptrsdsc) != ST_MEMBER) {
            int sdsc_mem = get_member_descriptor(sptr);
            tmp = check_member(ele, mk_id(sdsc_mem));
          } else {
            if (!sptrsdsc) {
              /* Occurs when the dummy argument expects a descriptor but the
               * actual argument does not normally hold a descriptor. This
               * typically occurs with a class(*) dummy argument and a scalar
               * actual argument. Call check_alloc_ptr_type() to generate
               * a descriptor argument for the actual argument.
               */
              check_alloc_ptr_type(sptr, std, 0, unl_poly ? 2 : 1, 0, 0,
                                   STYPEG(sptr) == ST_MEMBER ? ele : 0);
              sptrsdsc = SDSCG(sptr);
            }
            if (sptrsdsc)
              tmp = mk_id(sptrsdsc);
          }
          if( STYPEG(sptrsdsc) != ST_MEMBER &&
              DTY(DTYPEG(sptr)) != TY_ARRAY && CLASSG(sptr) &&
              STYPEG(sptr) == ST_MEMBER && FVALG(entry) != inface_arg) {
            int newargt2, astnew, func;
            int sdsc_mem = get_member_descriptor(sptr);

            newargt2 = mk_argt(2);
            ARGT_ARG(newargt2, 0) = mk_id(sptrsdsc);

            ARGT_ARG(newargt2, 1) = check_member(ele, mk_id(sdsc_mem));

            func = mk_id(
                sym_mkfunc_nodesc(mkRteRtnNm(RTE_test_and_set_type), DT_NONE));
            astnew = mk_func_node(A_CALL, func, 2, newargt2);
            add_stmt_before(astnew, std);

          } else if (ALLOCDESCG(sptr) && needdescr && !CLASSG(inface_arg) &&
                     FVALG(entry) == inface_arg) {
            /* Need to assign type of function result to the
             * argument after the function call.
             * This occurs when we're returning a pointer
             * to a derived type (or returning an allocatable
             * derived type). This is a special case when the
             * return argument is not declared class and the
             * type is statically known at compile-time.
             */
            DTYPE dtype = DTYPEG(inface_arg);
            if (DTY(dtype) == TY_DERIVED) {
              LOGICAL is_inline;
              int tag = DTY(dtype + 3);
              int st_type = get_static_type_descriptor(tag);
              is_inline =
                  inline_RTE_set_type(sptrsdsc, st_type, std, 1, dtype, 0);
              if (!is_inline) {
                gen_set_type(mk_id(sptrsdsc), mk_id(st_type), std, TRUE, FALSE);
              }
            }
          }
          if (sptrsdsc != 0 && !CLASSG(sptrsdsc) && !CLASSG(sptr) &&
              CLASSG(inface_arg)) {
            /* non-polymorphic object with a regular descriptor (not a
             * type descriptor which would be the case if sptrsdsc's CLASS
             * field were TRUE), so make sure we set its type field.
             */
            int dtype;

            dtype = DTYPEG(sptr);
            if (DTY(dtype) == TY_DERIVED) {
              LOGICAL is_inline;
              int tag = DTY(dtype + 3);
              int st_type = get_static_type_descriptor(tag);
              is_inline = (STYPEG(sptrsdsc) == ST_MEMBER)
                              ? 0
                              : inline_RTE_set_type(sptrsdsc, st_type, std, 0,
                                                      dtype, 0);
              if (!is_inline) {
                gen_set_type(mk_id(sptrsdsc), mk_id(st_type), std, TRUE, FALSE);
              }
            }
          }
          if (tmp != 0) {
            ARGT_ARG(newargt, newj) = check_member(ele, tmp);
            ++newj;
          }
          break;
        }
        if ((POINTERG(sptr) && POINTERG(inface_arg)) ||
            (ALLOCATTRG(sptr) && ALLOCATTRG(inface_arg))) {
          /* pointer dummy and actual, need a descriptor */
            ARGT_ARG(newargt, newi) = ele;
          ++newi;
          if (SDSCG(sptr) > NOSYM && DESCRG(sptr) <= NOSYM &&
              ALLOCDESCG(sptr) && !CLASSG(inface_arg) && needdescr &&
              POINTERG(sptr) && FVALG(entry) == inface_arg) {
            /* Make sure we pass a full descriptor for
             * this argument. That way there's space for a type
             * in the resulting descriptor.
             */
            DTYPE dtype = DTYPEG(sptr);
            if (is_array_dtype(dtype))
              dtype = array_element_dtype(dtype);
            ARGT_ARG(newargt, newj) = check_member(ele, mk_id(SDSCG(sptr)));
            if (DTY(dtype) == TY_DERIVED && INTENTG(inface_arg) != INTENT_OUT) {
              int tag = DTY(dtype + 3);
              int st_type = get_static_type_descriptor(tag);
              gen_set_type(ARGT_ARG(newargt, newj), mk_id(st_type), std, TRUE,
                           FALSE);
            }
            ++newj;
            break;
          }
          sptrsdsc = SDSCG(sptr);
          need_surr = 0;
          if (ALLOCATTRG(inface_arg)) {
            assert(needdescr, "allocatable dummy argument but !needdescr", 0, 3);
            need_surr = 1;
            if (i == 0 && sptrsdsc && iface && FVALG(iface) &&
                ALLOCATTRG(FVALG(iface))) {
              /* If the allocatable represents the result of
               * a function call, then reference the real
               * descriptor.
               */
              need_surr = 0;
            } else if (A_TYPEG(ele) == A_MEM || THREADG(sptr))
              need_surr = 0;
            else if (MIDNUMG(sptr) && SCG(MIDNUMG(sptr)) == SC_PRIVATE)
              need_surr = 0;
            else if (MDALLOCG(sptr) || SCG(sptr) == SC_DUMMY ||
                     (MIDNUMG(sptr) && SCG(MIDNUMG(sptr)) == SC_CMBLK)) {
              /* reference the real descriptor if
               * + the allocatable is a module variable,
               * + the allocatable is a dummy argument,
               * + the allocatable is in a common block
               *   (this one is actually illegal, but may be
               *    allowed as an extension? or the MDALLOC
               *    flag may not always be set?).
               */
              need_surr = 0;
            } else if (needdescr && DTY(DTYPEG(inface_arg)) != TY_ARRAY) {
              /* allocatable scalar */
              if (!sptrsdsc) {
                get_static_descriptor(sptr);
                get_all_descriptors(sptr);
                sptrsdsc = SDSCG(sptr);
              }
              need_surr = 0;
            } else if (gbl.internal == 1 ||
                       (gbl.internal > 1 && !INTERNALG(sptr))) {
              need_surr = 2;
              if (DESCUSEDG(sptr) && ALLOCDESCG(sptr)) {
                need_surr = 0;
              }
            } else if (DESCUSEDG(sptr) && sptrsdsc) {
              /*
               * this check MUST BE DONE after checking if
               * need_surr must be set to 2
               */
              need_surr = 0;
            } else if (ALLOCDESCG(sptr) || flg.debug || TARGETG(sptr) ||
                       unl_poly) {
              need_surr = 2;
            } else if (!XBIT(47, 0x80000000)) {
              /* Default usage surrogates disabled, there's bugs.
               * Some situations where a real descriptor must be
               * passed are not caught by the farrago of special
               * case tests above.
               */
              need_surr = 2;
            }
          }
          if (need_surr == 1 /*&& sptrsdsc == 0*/) {
            char nm[50];
            static int nmctr = 0;
            DTYPE dtype = DTYPEG(sptr);
            int sptrtmp;
            sprintf(nm, "surrogate%d_%d", A_SPTRG(ele), nmctr++);
            sptrtmp = sym_get_array(nm, "_", DDTG(dtype),
                                    SHD_NDIM(A_SHAPEG(ele)));
            get_static_descriptor(sptrtmp);
            get_all_descriptors(sptrtmp);
            init_sdsc_from_dtype(sptrtmp, dtype, std);
            copy_surrogate_to_bnds_vars(dtype, 0, DTYPEG(sptrtmp), 0, STD_NEXT(std));
            sptrsdsc = SDSCG(sptrtmp);
            ARGT_ARG(newargt, newj) = check_member(ele, mk_id(sptrsdsc));
            ++newj;
          } else if (need_surr == 2) {
            /* DESCR field is 0 if sptr is a scalar */
            if (DESCRG(sptr) > NOSYM) {
              sptrsdsc = DESCRG(sptr);
              DESCUSEDP(sptr, TRUE);
              NODESCP(sptr, FALSE);
            } else if (sptrsdsc <= NOSYM) {
              get_static_descriptor(sptr);
              sptrsdsc = SDSCG(sptr);
            }
            copy_desc_to_bnds_vars(sptr, sptrsdsc, 0, STD_NEXT(std));
            ARGT_ARG(newargt, newj) = check_member(ele, mk_id(sptrsdsc));
            ++newj;
          } else if (STYPEG(sptr) != ST_MEMBER ||
                     STYPEG(SDSCG(sptr)) == ST_MEMBER) {
            /* don't pass SDSC for pointer derived type member
             * if the SDSC is not itself a member */
            descr = check_member(ele, mk_id(sptrsdsc));
              ARGT_ARG(newargt, newj) = descr;
            DESCUSEDP(sptr, 1);
            NODESCP(sptr, 0);
            ++newj;
          } else if (needdescr && !DT_ISBASIC(A_DTYPEG(ele))) {
            int sptrsdsc;
            sptr = memsym_of_ast(ele);
            if (!SDSCG(sptr))
              get_static_descriptor(sptr);
            sptrsdsc = get_member_descriptor(sptr);
            if (sptrsdsc <= NOSYM) {
              sptrsdsc = SDSCG(sptr);
            }
            ARGT_ARG(newargt, newj) = check_member(ele, mk_id(sptrsdsc));
            ++newj;
          } else {
            ty = dtype_to_arg(A_DTYPEG(ele));
            ARGT_ARG(newargt, newj) = pghpf_type(ty);
            ++newj;
          }
          break;
        }
      }

      if (is_ent &&
          (
#if DEBUG
              strcmp(SYMNAME(entry), mkRteRtnNm(RTE_show)) == 0 ||
#endif
              (XBIT(57, 0x4000000) && i == 0 &&
               (strcmp(SYMNAME(entry), "pgi_get_descriptor") == 0 ||
                strcmp(SYMNAME(entry), "pgi_put_descriptor") == 0)))) {
        if (strcmp(SYMNAME(entry), "pgi_put_descriptor") != 0) {
          ARGT_ARG(newargt, newi) = ele;
          ++newi;
        } else {
          /* here, we want to pass the address of the pointer,
           * not the value of the pointer */
          int sptr;
          if (A_TYPEG(ele) == A_ID)
            sptr = A_SPTRG(ele);
          else if (A_TYPEG(ele) == A_MEM)
            sptr = A_SPTRG(A_MEMG(ele));
          else
            sptr = 0;
          if (sptr && MIDNUMG(sptr)) {
            ARGT_ARG(newargt, newi) = check_member(ele, mk_id(MIDNUMG(sptr)));
            ++newi;
          } else {
            interr("unexpected leading argument to pgi_put_descriptor()", 0, 4);
          }
        }
        if (CLASSG(sptr)) {
          int sptrsdsc;
          if (SDSCG(sptr) && STYPEG(sptr) == ST_MEMBER) {
            sptrsdsc = get_member_descriptor(sptr);
          } else {
            sptrsdsc = get_type_descr_arg(gbl.currsub, sptr);
          }
          ARGT_ARG(newargt, newj) = check_member(ele, mk_id(sptrsdsc));
          ++newj;
        } else if (!A_SHAPEG(ele) && 
                   DTY(A_DTYPEG(ele)) != TY_PTR) { /* scalar */
          ty = dtype_to_arg(A_DTYPEG(ele));
          ARGT_ARG(newargt, newj) = pghpf_type(ty);
          ++newj;
        } else if (SDSCG(sptr)) {
         /*
          * If the array descriptor comes from the parent subprogram (Fortran
          * term, host subprogram), the INTERNREF flag of array descriptor must
          * be set. The missing case here is when the array is a static member
          * of derived type var and the derived type var is defined in parent
          * routine, the array descriptor is not set.  The following code is to
          * detect such case.
          */

          SPTR sdsdsptr = SDSCG(sptr); 
          
         /* Condition gbl.internal > 1 is to make sure that the current
          * function is a contained subprogram. 
          *   gbl.internal = 0, there is no contained subprogram. 
          *   gbl.internal = 1, the current routine at least has a contain 
          *   statement.
          *   gbl.internal > 1, the current routine is contained subprogram. 
          * A_TYPEG(ele) == A_MEM && needdescr && gbl.internal > 1
          * is to make sure that the descriptor is for an array
          * member in a derived-type variable if it is not an array variable
          * member in the derived-type, the INTERNREF flag of its descriptor
          * should be set in routine set_internref_flag in semsym.c.
          * 
          * SCOPEG(sdsdsptr) && STYPEG(SCOPEG(sdsdsptr)) != ST_MODULE &&
          * SCOPEG(sdsdsptr) == SCOPEG(gbl.currsub) is to make sure that the
          * sdsc is a reference from the parent routine.
          *
          * Note that the fortran can only contain one-level depth of
          * subprogram. The contained subprogram cannot contain any
          * subprograms.
          */       
          if (gbl.internal > 1 && A_TYPEG(ele) == A_MEM && needdescr &&
             SCOPEG(sdsdsptr) && STYPEG(SCOPEG(sdsdsptr)) != ST_MODULE && 
             SCOPEG(sdsdsptr) == SCOPEG(gbl.currsub)) {
             /* Pointer to the section descriptor created by the front-end 
              * (or any phase before transform()). If the field is non-zero, 
              * the transformer uses the descriptor located by this field; 
              * the actual symbol located by this field is a based/allocatable
              * array. 
              */
             if (SECDSCG(sdsdsptr))
               sdsdsptr = SECDSCG(sdsdsptr);
             INTERNREFP(sdsdsptr, TRUE);
          }
          ARGT_ARG(newargt, newj) = check_member(ele, mk_id(SDSCG(sptr)));
          DESCUSEDP(sptr, 1);
          NODESCP(sptr, 0);
          ++newj;
        } else {
          ARGT_ARG(newargt, newj) = check_member(ele, mk_id(DESCRG(sptr)));
          DESCUSEDP(sptr, 1);
          NODESCP(sptr, 0);
          ++newj;
        }

        break;
      }
      /* used for minloc/maxloc */
      if (NODESCG(sptr)) {
        ARGT_ARG(newargt, newi) = ele;
        ++newi;
        if (needdescr) {
          ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg,
                                                                 ele, std);
          ++newj;
        }
        break;
      }
      if (!A_SHAPEG(ele)) { /* scalar */
          ARGT_ARG(newargt, newi) = ele;
        ++newi;
        if (needdescr) {
          ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg,
                                                                 ele, std);
          ++newj;
        }
      } else { /* whole array */
        retval = descr = 0;
        ele = remove_subscript_expressions(ele, std, sym_of_ast(ele));
        if (is_seq_dummy(entry, ele, i) && A_TYPEG(ast) != A_ICALL) {
          handle_seq_section(entry, ele, i, std, &retval, &descr, 0,
                             inface_arg);
        } else if (XBIT(57, 0x10000) && A_TYPEG(ast) != A_ICALL &&
                   stride_1_dummy(entry, ele, i) &&
                   !stride_1_section(entry, ele, i, std)) {
          handle_seq_section(entry, ele, i, std, &retval, &descr, 1,
                             inface_arg);
        } else {
          if (!DESCRG(sptr)) {
            get_static_descriptor(sptr);
            get_all_descriptors(sptr);
          }
          SPTR descr_sptr = DESCRG(sptr);
          /* Set the INTERNREF flag of array descriptor to make sure host
             subroutines' array descriptor is accessible for contained
             subroutines. 
           */
          if (gbl.internal > 1 && A_TYPEG(ele) == A_MEM && needdescr && 
              descr_sptr > NOSYM && SCOPEG(descr_sptr) && 
              STYPEG(SCOPEG(descr_sptr)) != ST_MODULE &&
              SCOPEG(descr_sptr) == SCOPEG(gbl.currsub)) {
             if (SECDSCG(descr_sptr))
               descr_sptr = SECDSCG(descr_sptr);

             INTERNREFP(descr_sptr, TRUE);
          }
          retval = ele;
          descr = check_member(retval, mk_id(DESCRG(sptr)));
        }
          ARGT_ARG(newargt, newi) = retval;
        ++newi;
        if (needdescr) {
          int s;

          DESCUSEDP(sptr, 1);
          NODESCP(sptr, 0);

          /* Fix for assumed-shape arrays arguments where callee has the
           * argument marked as target (originally discovered at customer). 
           * In this case the whole array is sent, but the callee code 
           * still needs to use the address calculation method such that 
           * the lower bound is folded into the lbase field of the descriptor 
           * to make zero-based  offsets work (so-called; in practice 
           * usually 1-based with Fortran).
           * [see exp_ftn.c: compute_sdsc_subscr() & add_ptr_subscript()]
           *
           * Since a new section descriptor has not been generated in this
           * case where we send the whole array as an argument we need to
           * create a new, temporary, argument array descriptor which
           * translates bounds to zero-based (per the Fortran standard with
           * assumed-shape arguments) and adjusts the lbase field.
           *
           * NB: The new runtime routine, pgf90_tmp_desc(), used to create the
           * argument desriptor is similar to a pointer assignment (which
           * makes the ptr_assn() call), and which follows the same rules
           * of zero-based array bounds and lbase calculation. Also, this type
           * of lbase fixup can also be found in runtime routines like
           * ptr_fix_assumeshp(), where a whole array, instead of a section,
           * is identified. 
           */
          if(XBIT(58,0x400000) && ASSUMSHPG(inface_arg) && TARGETG(inface_arg))
          {
            char nd[50];    /* new, substitute, descriptor for this arg */
            static int ndctr = 0;
            DTYPE dtype = DTYPEG(sptr);
            SPTR sptrtmp, sptrdesc;
            sprintf(nd, "ndesc%d_%d", A_SPTRG(ele), ndctr++);
            sptrtmp = sym_get_array(nd, "", DDTG(dtype),
                                    SHD_NDIM(A_SHAPEG(ele)));
            get_static_descriptor(sptrtmp); /* add sdsc to sptrtmp; necessary */
            get_all_descriptors(sptrtmp); /* add desc & bounds in dtype */
            /* generate the runtime call pgf90_tmp_desc) */
            make_temp_descriptor(ele, sptr, sptrtmp, std);
            sptrdesc = DESCRG(sptrtmp);
            ARGT_ARG(newargt, newj) = check_member(ele, mk_id(sptrdesc));
          }
          else
            ARGT_ARG(newargt, newj) = descr;
          ++newj;
          s = memsym_of_ast(descr);
          if (s)
            ARGP(s, 1);
          if (s && STYPEG(s) == ST_ARRDSC && ARRAYG(s)) {
            DESCUSEDP(ARRAYG(s), 1);
            NODESCP(ARRAYG(s), 0);
            if (flg.smp) {
              set_parref_flag2(s, 0, std);
            }
          }
        }
      }
      break;

    case A_SUBSCR:
      sptr = sptr_of_subscript(ele);
      /* run-time can not handle the following array type */
      if (is_bad_dtype(DTYPEG(sptr))) {
        if (A_SHAPEG(ele)) {
          interr("transform_call: character/struct/union array section is not "
                 "allowed",
                 std, 4);
        }
          ARGT_ARG(newargt, newi) = ele;
        ++newi;
        if (needdescr) {
          ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg,
                                                                 ele, std);
          ++newj;
        }
        break;
      }

      if (!A_SHAPEG(ele)) { /* Array Element  */
        if (inface_arg && CLASSG(inface_arg))
          goto class_arg;
          ARGT_ARG(newargt, newi) = ele;
        ++newi;
        if (needdescr) {
          ARGT_ARG(newargt, newj) = get_descr_or_placeholder_arg(inface_arg, 
                                                                 ele, std);
          ++newj;
        }
      } else { /* Array Section  */
        int aa;
        aa = transform_section_arg(ele, std, ast, entry, &descr, i);
          ARGT_ARG(newargt, newi) = aa;
        ++newi;
        if (needdescr) {
          int s;
          /* situation where we are sending the whole array using
           * subscript notation, e.g. x(:,:), but semantically 
           * equivalent to the above A_ID case of 'x'. We know that 
           * the whole array is being passed when the descriptor 
           * generated from the call to transform_section_arg() 
           * above (descr) is unchanged from the sptr descriptor.
           */
          if(XBIT(58,0x400000) && ASSUMSHPG(inface_arg) &&
             TARGETG(inface_arg) && A_SPTRG(descr) &&
             (DESCRG(sptr) == A_SPTRG(descr)) )
          {
            char nsd[50];    /* new, substitute, descriptor for this arg */
            static int nsdctr = 0;
            DTYPE dtype = DTYPEG(sptr);
            SPTR sptrtmp, sptrdesc;
            /* In this case ele is an A_SUBSCR, so need to use its A_LOP */
            int lop_ele = A_LOPG(ele);

            sprintf(nsd, "n2desc%d_%d", A_SPTRG(lop_ele), nsdctr++);
            sptrtmp = sym_get_array(nsd, "", DDTG(dtype),
                                    SHD_NDIM(A_SHAPEG(lop_ele)));
            get_static_descriptor(sptrtmp); /* add sdsc to sptrtmp; necessary */
            get_all_descriptors(sptrtmp); /* add desc & bounds in dtype */
            /* generate the runtime call pgf90_tmp_desc) */
            make_temp_descriptor(ele, sptr, sptrtmp, std);
            sptrdesc = DESCRG(sptrtmp);
            ARGT_ARG(newargt, newj) = check_member(ele, mk_id(sptrdesc));
          }
          else
            ARGT_ARG(newargt, newj) = descr;
          ++newj;
          s = memsym_of_ast(descr);
          if (s)
            ARGP(s, 1);
          if (s && STYPEG(s) == ST_ARRDSC && ARRAYG(s)) {
            DESCUSEDP(ARRAYG(s), 1);
            NODESCP(ARRAYG(s), 0);
            if (flg.smp) {
              set_parref_flag2(s, 0, std);
            }
          }
          if (inface_arg && CLASSG(inface_arg))
            set_type_in_descriptor(descr, sptr, DT_NONE, ele, std);
        }
      }
      break;
    default:
      interr("transform_call: unknown expression", std, 2);
    }
  }
  if (istart) {
    retval = ARGT_ARG(argt, 0);
    sptr = memsym_of_ast(retval);
    descr = check_member(retval, mk_id(DESCRG(sptr)));
    ARGT_ARG(newargt, newj) = descr;
  }
  A_ARGSP(ast, newargt);
  A_ARGCNTP(ast, newnargs);
} /* transform_call Fortran */

/**
 * \brief Set the tag field in a descriptor expression.
 *
 * \param descr is an ast representing the descriptor expression.
 *
 * \param tag is the tag field value (typically __TAGDESC or __TAGPOLY).
 *
 * \param std is where we want to insert the assignment.
 */
static void
set_descr_tag(int descr, int tag, int std)
{
  int val,dast, assn;
  SPTR sdsc;

  if (!ast_is_sym(descr))
    return;

  sdsc = memsym_of_ast(descr);

  val = mk_cval1(tag, DT_INT);
  dast = check_member(descr, get_desc_tag(sdsc));
  assn = mk_assn_stmt(dast, val, DT_INT);
  add_stmt_before(assn, std);

}

/**
 * \brief Generate an ast that represents a descriptor actual argument for a
 * polymorphic or monomorphic actual argument.
 *
 * \param ele is the ast of the symbol expression
 *
 * \param sptr is the symbol table pointer of the symbol (could differ from
 *        ele).
 *
 * \param std is where to insert the assignment statements.
 *
 * \return an ast representing the descriptor actual argument
 */
static int
get_descr_arg(int ele, SPTR sptr, int std)
{
  
  SPTR sptrsdsc;
  int arg_ast;

  if (SDSCG(sptr) && STYPEG(sptr) == ST_MEMBER) {
    sptrsdsc = get_member_descriptor(sptr);
  } else {
    sptrsdsc = get_type_descr_arg(gbl.currsub, sptr);
  }

  arg_ast = check_member(ele, mk_id(sptrsdsc));

  if (STYPEG(sptr) != ST_ARRAY) {
    set_descr_tag(arg_ast, CLASSG(sptr) ? __TAGPOLY : __TAGDESC, std); 
  }

  return arg_ast;
}

/**
 * \brief Called by transform_call() to get either the descriptor argument
 *        or a placeholder descriptor argument for a procedure call.
 *
 * \param inface_arg is the symbol table pointer of the interface's argument.
 * 
 * \param ele is an ast representing the actual argument.
 * 
 * \param std is where to insert the assignment statements.
 *
 * \return an ast represting the descriptor/placeholder argument.
 */
static int
get_descr_or_placeholder_arg(SPTR inface_arg, int ele, int std)
{
  int ast;
  DTYPE ty;
  SPTR actual = (ast_is_sym(ele)) ? memsym_of_ast(ele) : 0;

  if (CLASSG(actual) || (inface_arg > NOSYM && needs_descriptor(actual) &&
      needs_descriptor(inface_arg))) {
    ast = get_descr_arg(ele, actual, std);
  } else {
    ty = dtype_to_arg(A_DTYPEG(ele));
    ast = pghpf_type(ty);
  }

  return ast;
}

/** 
 * \brief Create a temporary type descriptor for a procedure argument. 
 *
 * \param ast is the ast that represents a procedure argument.
 *
 * \param std is where to insert the assignment statements.
 *
 * \return an ast representing the temporary type descriptor.
 */
static int
temp_type_descriptor(int ast, int std)
{
  SPTR sdsc;
  int descr;
  DTYPE dtype = A_DTYPEG(ast);
  SPTR tmp = getcctmp_sc('d', sem.dtemps++, ST_VAR, dtype, sem.sc);
  check_alloc_ptr_type(tmp, std, 0, 2, 0, 0, 0);
  sdsc = SDSCG(tmp);
  assert(sdsc > NOSYM, "expect SDSC on generated temp", tmp, ERR_Fatal);
  descr = check_member(ast, mk_id(sdsc));
  if (DTY(dtype) == TY_CHAR) {
    int val;
    int dast, assn;
    /* copy string length into unlimited polymorphic descriptor argument */
    if (string_length(dtype)) {
      val = mk_cval1(string_length(dtype), DT_INT);
    } else {
      val = DTY(dtype + 1);
    }
    if (val) {
      dast = check_member(descr, get_byte_len(sdsc));
      assn = mk_assn_stmt(dast, val, DT_INT);
      add_stmt_before(assn, std);
    }

    set_descr_tag(descr, __TAGDESC, std);

    val = mk_cval1(0, DT_INT);
    dast = check_member(descr, get_desc_rank(sdsc));
    assn = mk_assn_stmt(dast, val, DT_INT);
    add_stmt_before(assn, std);

    dast = check_member(descr, get_desc_lsize(sdsc));
    assn = mk_assn_stmt(dast, val, DT_INT);
    add_stmt_before(assn, std);

    dast = check_member(descr, get_desc_gsize(sdsc));
    assn = mk_assn_stmt(dast, val, DT_INT);
    add_stmt_before(assn, std);

    val = mk_cval1(__TAGPOLY, DT_INT);
    dast = check_member(descr, get_kind(sdsc));
    assn = mk_assn_stmt(dast, val, DT_INT);
    add_stmt_before(assn, std);
  }
  return descr;
}

static LOGICAL
is_seq_dummy(int entry, int arr, int loc)
{
  int dscptr, iface;
  int dummy_sptr;

  proc_arginfo(entry, NULL, &dscptr, &iface);
  if (iface && HCCSYMG(iface)) {
    return FALSE;
  }
  if (!dscptr) { /* implicit interface, F77, all sequential */
    return TRUE;
  }
  dummy_sptr = aux.dpdsc_base[dscptr + loc];
  if (SEQG(dummy_sptr))
    return TRUE;
  if (CONTIGATTRG(dummy_sptr))
    return TRUE;
  if (ASSUMSHPG(dummy_sptr))
    return FALSE;
  return TRUE;
}

static LOGICAL
needs_type_in_descr(SPTR entry, int loc)
{
  int dscptr;
  SPTR iface;

  proc_arginfo(entry, NULL, &dscptr, &iface);
  if (iface > NOSYM && HCCSYMG(iface)) {
    return FALSE;
  }
  if (dscptr > 0) {
    SPTR dummy_sptr = aux.dpdsc_base[dscptr + loc];
    if (dummy_sptr > NOSYM) {
      DTYPE dtype = DTYPEG(dummy_sptr);
      if (is_array_dtype(dtype))
        dtype = array_element_dtype(dtype);
      return CLASSG(dummy_sptr) || DTY(dtype) == TY_DERIVED;
    }
  }
  return FALSE;
}

/*
 * return TRUE if dummy argument must be stride-1
 * in the leftmost dimension, but need not be contiguous
 * i.e., return TRUE for assumed-shape dummies
 */
static LOGICAL
stride_1_dummy(int entry, int arr, int pos)
{
  int sptr, dummy_sptr;

  dummy_sptr = find_dummy(entry, pos);
  if (!dummy_sptr || SEQG(dummy_sptr))
    return TRUE;
  sptr = memsym_of_ast(arr);
  if (POINTERG(sptr)) {
    /* if the actual argument is a pointer, it may be noncontiguous.
     * if the dummy is a pointer, we're ok.  Otherwise, sequentialize */
    if (POINTERG(dummy_sptr))
      return FALSE;
    return TRUE;
  }
  if (XBIT(57, 0x10000) && ASSUMSHPG(dummy_sptr) && SDSCS1G(dummy_sptr) &&
      !XBIT(54, 2) &&
      !(XBIT(58, 0x400000) && TARGETG(dummy_sptr))) {
    /* assumed-shape dummies usually must be stride-1 */
    return TRUE;
  }
  return FALSE;
} /* stride_1_dummy */

static LOGICAL
is_optional_char_dummy(int entry, int arr, int pos)
{
  int dummy_sptr;

  if (arr != astb.ptr0)
    return FALSE;
  dummy_sptr = find_dummy(entry, pos);
  if (dummy_sptr && DTYG(DTYPEG(dummy_sptr)) == TY_CHAR)
    return TRUE;
  return FALSE;
}

#ifdef FLANG_REST_UNUSED
/*
 * A scalar element of a non-sequence array can be passed as an actual
 * argument if and only if the dummy is scalar.  pghpf should enforce
 * this rule but does not.
 */
static void
check_nonseq_element(int std, int entry, int arr, int pos)
{
  int sptr, dummy_sptr;

  if (A_TYPEG(arr) != A_SUBSCR)
    return;
  if (A_SHAPEG(arr))
    return;
  sptr = sym_of_ast(arr);
  if (SEQG(sptr))
    return;
  dummy_sptr = find_dummy(entry, pos);
  if (dummy_sptr && is_array_type(dummy_sptr))
    error(472, 3, STD_LINENO(std), SYMNAME(sptr), CNULL);
}
#endif

static int
pure_procedure(int ast)
{
  if (pure_func_call(ast) || elemental_func_call(ast)) {
    return TRUE;
  }
  if (A_OPTYPEG(ast) == I_MOVE_ALLOC || A_OPTYPEG(ast) == I_MVBITS ||
      (A_OPTYPEG(ast) == I_PTR2_ASSIGN &&
       A_OPTYPEG(ARGT_ARG(A_ARGSG(ast), 2)) == I_NULL)) {
    return TRUE;
  }

  return FALSE;
}
/* pure can not have distributed dummy
 * pure dummy can only align with another pure dummy
 */
static void
check_pure_interface(int entry, int std, int ast)
{
  if (!is_impure(gbl.currsub) && !HCCSYMG(entry) && !pure_procedure(ast)) {
    switch (STYPEG(entry)) {
    case ST_INTRIN:
    case ST_GENERIC:
    case ST_PD:
      break;
    default:
      error(473, 2, gbl.lineno, SYMNAME(entry), CNULL);
    }
  }
}

/* This routine is to handle sequential array section.
 * It will try to find correct retval and descr
 * If it is continuous section
 *    retval = first_array_element,
 *    descr = new descriptor based on shape
 * else it will copy to tmp before and after
 *    retval = first_element_of_(tmp)
 *    descr = tmp_descriptor
 */
static void
handle_seq_section(int entry, int arr, int loc, int std, int *retval,
                   int *descr, LOGICAL stride1, int dummysptr)
{
  int sec;
  int arraysptr;
  int arrayalign;
  int arrayast;

  int topsptr;
  int topdtype;
  LOGICAL simplewholearray;
  LOGICAL is_seq_pointer, is_pointer;
  LOGICAL continuous, desc_needed;
  int is_hcc;
  int iface; /* sptr of explicit interface; could be zero */

  /* find a distributed array, if any, and its alignment */
  arrayast = arr;
  arraysptr = 0;
  topsptr = 0;
  arrayalign = 0;
  is_seq_pointer = FALSE;
  is_pointer = FALSE;
  simplewholearray = TRUE;
  do {
    switch (A_TYPEG(arrayast)) {
    case A_ID:
      arraysptr = A_SPTRG(arrayast);
      if (topsptr == 0)
        topsptr = arraysptr;
      arrayalign = ALIGNG(arraysptr);
      if (POINTERG(arraysptr)) {
        is_pointer = TRUE;
        if (!arrayalign)
          is_seq_pointer = TRUE;
      }
      if (TARGETG(arraysptr) && XBIT(58,0x400000))
          is_seq_pointer = TRUE;
      /* for F90, an assumed-shape dummy array looks like
       * a sequential pointer, if copy-ins are removed */
      if (XBIT(57, 0x10000) && ASSUMSHPG(arraysptr) && SDSCS1G(arraysptr) &&
          !XBIT(54, 2))
        is_seq_pointer = TRUE;
      break;
    case A_SUBSCR:
      arrayast = A_LOPG(arrayast);
      simplewholearray = FALSE;
      break;
    case A_MEM:
      arraysptr = A_SPTRG(A_MEMG(arrayast));
      if (topsptr == 0)
        topsptr = arraysptr;
      /* is the array shape from the member or from the parent */
      if (A_SHAPEG(A_PARENTG(arrayast))) {
        /* parent is array shaped, member should not be */
        arraysptr = 0;
        arrayast = A_PARENTG(arrayast);
        simplewholearray = FALSE;
      } else {
        /* right now, no members can be distributed anyway */
        arrayalign = ALIGNG(arraysptr);
       if (POINTERG(arraysptr)) {
         is_pointer = TRUE;
         if (!arrayalign)
           is_seq_pointer = TRUE;
       }
      }
      break;
    default:
      interr("handle_seq_section: unknown ast", arr, 4);
      break;
    }
  } while (arraysptr == 0);

  topdtype = DTYPEG(topsptr);
  if (DTY(topdtype) == TY_ARRAY)
    topdtype = DTY(topdtype + 1);

  if (simplewholearray && !is_pointer && CONTIGATTRG(arraysptr)) {
    /* Note: The call to first_element() uses the descriptor of the declared
     * dtype of arr which is fine for simple regular arrays. But it does not 
     * work for pointers since a pointer's descriptor can change depending on 
     * the pointer target. Typically this is accomplished by creating a section
     * descriptor first (i.e., call mk_descr_from_section()) which takes into 
     * account the runtime shape of the pointer target. We handle this and
     * other pointer cases below. 
     */ 
    *retval = first_element(arr); 
    *descr = DESCRG(arraysptr) > NOSYM ?
      check_member(arrayast, mk_id(DESCRG(arraysptr))) : 0;
    return;
  }

  /* whole array with no distribution */
  if (!is_seq_pointer
      /* for F90, pointers may not be contiguous */
      && !(is_pointer && XBIT(58, 0x10000)) && simplewholearray) {
    *retval = arr;
    *descr = DESCRG(arraysptr) > NOSYM ? check_member(arrayast, mk_id(DESCRG(arraysptr))) : 0;
    return;
  }

  proc_arginfo(entry, NULL, NULL, &iface);
  if (!is_procedure_ptr(entry)) {
    is_hcc = HCCSYMG(entry);
  } else {
    is_hcc = 0;
  }

  /* undistributed pointer, no alignment, whole array, F90 callee */
  if (is_seq_pointer && !XBIT(58, 0x10000) && simplewholearray) {
    if (is_hcc || (XBIT(28, 0x10))) {
      *retval = arr;
      *descr = DESCRG(arraysptr) > NOSYM ? check_member(arrayast, mk_id(DESCRG(arraysptr))) : 0;
      return;
    }
  }
  assert(is_array_type(arraysptr), "handle_seq_section: symbol not array",
         arraysptr, 3);
  arr = convert_subscript(arr);

  desc_needed = is_desc_needed(entry, arr, loc);
  continuous = continuous_section(entry, arr, loc, 0);

  if (is_seq_pointer) {
    if (XBIT(58, 0x10000)) {
      if (continuous) {
        if (CONTIGATTRG(arraysptr)) {
          if (!desc_needed) {
            *descr = pghpf_type(0);
          }
          else {
            *descr = mk_descr_from_section(arr, topdtype, std);
          }
          *retval = first_element_from_section(arr);
          return;
        }
        if (!desc_needed)
          copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                              descr, TRUE, stride1, simplewholearray);
        else
          copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                              descr, FALSE, stride1, simplewholearray);
      } else {
        copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                            descr, FALSE, stride1, simplewholearray);
      }
    } else {
      /* If the formal parameter does not need (allow) a runtime descriptor,
       * call copy_arg_to_seq_tmp to determine whether the actual parameter
       * should be copied or not.  Also, adjust `retval' to refer to the
       * first entry if the actual is not the whole array.
       *
       * Otherwise, insure that the runtime descriptor exists and set `retval'
       * to first element of the array (section).
       */
      if (simplewholearray) {
        if (desc_needed) {
          /* undistributed whole pointer array */
          *retval = arrayast;
          *descr = check_member(arrayast, mk_id(DESCRG(arraysptr)));
        } else {
          copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                              descr, TRUE, stride1, simplewholearray);
        }
      } else if (!continuous) {
        copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                            descr, FALSE, stride1, simplewholearray);
      } else if (!desc_needed) {
        copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                            descr, TRUE, stride1, simplewholearray);
        *retval = first_element_from_section(arr);
      } else {
        if (SDSCG(arraysptr)) {
          int secss, secsptr;
          secss = find_section_subscript(arr);
          sec = make_sec_from_ast(secss, std, std, 0, 0);
          secsptr = sptr_of_subscript(secss);
          *retval = check_member(A_LOPG(secss), mk_id(secsptr));
          *retval = replace_ast_subtree(arr, secss, *retval);
          *descr = check_member(A_LOPG(secss), mk_id(sec));
        } else {
          *descr = mk_descr_from_section(arr, topdtype, std);
          *retval = first_element_from_section(arr);
        }
      }
    }
    return;
  }

  /* f90 language */
  if (!desc_needed && (!continuous || (is_pointer && XBIT(58, 0x10000)))) {
    copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                        descr, FALSE, stride1, simplewholearray);
    *descr = pghpf_type(0);
    if (!DESCUSEDG(A_SPTRG(*retval)))
      NODESCP(A_SPTRG(*retval), 1);
    return;
  }

  if (!desc_needed) {
    *descr = pghpf_type(0);
    *retval = first_element_from_section(arr);
  } else if (continuous && !arrayalign) {
    /* unused variable, it just made for new descriptor */
    *descr = mk_descr_from_section(arr, topdtype, std);
    *retval = first_element_from_section(arr);
  } else {
    int dd;
    dd = dummysptr;
    if (!dd) {
      /*
       * Would have liked to have set dummysptr, but its purpose for
       * error checking and when/where is determined by the context
       * of calling handle_seq_section() ... in my situation, I'm
       * trying to capture an array section where we only know if
       * it's contiguous at run-time - the REFLECTED check below is
       * invalid when the actual argument is an array section ....
       */
      dd = find_dummy(entry, loc);
    }
    if ( dd && ASSUMSHPG(dd) && 
        (ignore_tkr(dd, IGNORE_C) || 
        (XBIT(58, 0x400000) && TARGETG(dd)))) {
      int secss, secsptr;
      secss = find_section_subscript(arr);
      if (XBIT(57, 0x200000)) {
        sec = make_sec_from_ast_chk(secss, std, std, 0, SECTZBASE, 1);
        *retval = first_element_from_section(arr);
      } else {
        /*
         * WE want the beginning of the array rather than the first
         * element:
         */
        sec = make_sec_from_ast_chk(secss, std, std, 0, 0, 1);
        secsptr = sptr_of_subscript(secss);
        *retval = check_member(A_LOPG(secss), mk_id(secsptr));
        *retval = replace_ast_subtree(arr, secss, *retval);
      }
      *descr = check_member(A_LOPG(secss), mk_id(sec));
      return;
    }
    copy_arg_to_seq_tmp(entry, loc, topdtype, arraysptr, arr, std, retval,
                        descr, FALSE, stride1, simplewholearray);
  }
}

static int
mk_descr_from_section(int section, DTYPE topdtype, int std)
{
  int tmp;
  int subscr[MAXDIMS];
  tmp = mk_shape_sptr(A_SHAPEG(section), subscr, topdtype);
  DESCUSEDP(tmp, 1);
  NODESCP(tmp, 0);
  emit_alnd_secd(tmp, 0, TRUE, std, 0);
  return mk_id(INS_DESCR(SECDG(DESCRG(tmp))));
}

/*
 * is this a small-constant shape?
 */
static int
small_shape(int shd)
{
  int ndim, i, astlw, astup, aststride;
  ISZ_T size, lw, up, stride;
  ndim = SHD_NDIM(shd);
  size = 1;
  for (i = 0; i < ndim; ++i) {
    if (!A_ALIASG(SHD_LWB(shd, i)))
      return 0;
    if (!A_ALIASG(SHD_UPB(shd, i)))
      return 0;
    if (SHD_STRIDE(shd, i) && !A_ALIASG(SHD_STRIDE(shd, i)))
      return 0;
    astlw = A_ALIASG(SHD_LWB(shd, i));
    lw = ad_val_of(A_SPTRG(astlw));
    astup = A_ALIASG(SHD_UPB(shd, i));
    up = ad_val_of(A_SPTRG(astup));
    stride = 1;
    if (SHD_STRIDE(shd, i)) {
      aststride = A_ALIASG(SHD_STRIDE(shd, i));
      stride = ad_val_of(A_SPTRG(aststride));
    }
    size *= (up - lw + stride) / stride;
    if (size > 20) {
      return 0;
    }
  }
  return 1;
} /* small_shape */

/*
 * allocate a temporary array and its pointer
 */
static int
new_seq_temp_array(int shd, int small, int dtype, int cvlen, int oldsptr,
                   int mustallocate)
{
  int ndim, i;
  int sptr, sptrdtype, ptr, astlw, astup, aststride;

  /* determine how many dimensions are needed, and which ones they are */
  ndim = SHD_NDIM(shd);
  assert(ndim <= MAXDIMS, "new_seq_temp_array: too many dimensions", ndim, 4);
  /* get the temporary */
  sptr = sym_get_array("tmp", "r", dtype, ndim);
  /* make the descriptors for the temporary */
  trans_mkdescr(sptr);
  /* mark as compiler created */
  HCCSYMP(sptr, 1);
  if (cvlen) {
    CVLENP(sptr, cvlen);
    ADJLENP(sptr, 1);
  }
  if (!ADJLENG(sptr) && !mustallocate && small) {
    SCP(sptr, symutl.sc);
    /* fill in the bounds */
    sptrdtype = DTYPEG(sptr);
    ADD_MLPYR(sptrdtype, 0) = astb.bnd.one;
    for (i = 0; i < ndim; ++i) {
      astlw = A_ALIASG(SHD_LWB(shd, i));
      astup = A_ALIASG(SHD_UPB(shd, i));
      if (SHD_STRIDE(shd, i)) {
        aststride = A_ALIASG(SHD_STRIDE(shd, i));
      }
      if (aststride == astb.i1 || aststride == astb.bnd.one)
        aststride = 0;
      ADD_LWBD(sptrdtype, i) = ADD_LWAST(sptrdtype, i) = astb.bnd.one;
      astup = mk_binop(OP_SUB, astup, astlw, astb.bnd.dtype);
      if (!aststride) {
        astup = mk_binop(OP_ADD, astup, astb.bnd.one, astb.bnd.dtype);
      } else {
        astup = mk_binop(OP_ADD, astup, aststride, astb.bnd.dtype);
        astup = mk_binop(OP_DIV, astup, aststride, astb.bnd.dtype);
      }
      ADD_UPBD(sptrdtype, i) = ADD_UPAST(sptrdtype, i) = astup;
      ADD_EXTNTAST(sptrdtype, i) = mk_extent(astb.bnd.one, astup, i);
      ADD_MLPYR(sptrdtype, i + 1) =
          mk_binop(OP_MUL, ADD_MLPYR(sptrdtype, i), astup, astb.bnd.dtype);
      ALLOCP(sptr, 0);
    }
  } else {
    /* make it allocatable */
    ptr = get_next_sym(SYMNAME(sptr), "p");
    STYPEP(ptr, ST_VAR);
    SCP(ptr, symutl.sc);
    DTYPEP(ptr, DT_PTR);
    DCLDP(ptr, 0);
    SCP(sptr, SC_BASED);
    MIDNUMP(sptr, ptr);
    ALLOCP(sptr, 1);
  }
  return sptr;
} /* new_seq_temp_array */

#define TEMP_AREA 6

typedef struct T_LIST {
  struct T_LIST *next;
  int temp, shd, dtype, cvlen, std, sc;
} T_LIST;

#define GET_T_LIST(q) q = (T_LIST *)getitem(TEMP_AREA, sizeof(T_LIST))
static T_LIST *templist;

/*
 * try to reuse an old temp array
 * the goal here is to create fewer temp arrays and fewer descriptors,
 * so the frame size doesn't get too big
 */
static int
make_seq_temp_array(int shd, int dtype, int oldsptr, int mustallocate, int std)
{
  int sptr, small, cvlen, ndim, i;
  T_LIST *q;
  small = small_shape(shd);
  cvlen = 0;
  if (DTY(dtype) == TY_CHAR && ADJLENG(oldsptr) && !F90POINTERG(oldsptr)) {
    cvlen = CVLENG(oldsptr);
    small = 0;
  }
  ndim = SHD_NDIM(shd);
  for (q = templist; q; q = q->next) {
    if (q->std == std || q->dtype != dtype || q->cvlen != cvlen ||
        q->sc != symutl.sc)
      continue;
    if (ndim != SHD_NDIM(q->shd))
      continue;
    /* see if this is allocatable */
    if (mustallocate || !small) {
      if (!ALLOCG(q->temp))
        continue;
    } else {
      /* must have the right fixed shape */
      if (ALLOCG(q->temp))
        continue;
      for (i = 0; i < ndim; ++i) {
        if (SHD_LWB(shd, i) != SHD_LWB(q->shd, i) ||
            SHD_UPB(shd, i) != SHD_UPB(q->shd, i) ||
            SHD_STRIDE(shd, i) != SHD_STRIDE(q->shd, i))
          break;
      }
      if (i < ndim)
        continue;
    }
    q->std = std;
    return q->temp;
  }
  /* see if we have a temp array that will fill the bill */
  sptr = new_seq_temp_array(shd, small, dtype, cvlen, oldsptr, mustallocate);
  GET_T_LIST(q);
  q->next = templist;
  templist = q;
  q->temp = sptr;
  q->shd = shd;
  q->dtype = dtype;
  q->cvlen = cvlen;
  q->std = std;
  q->sc = symutl.sc;
  return sptr;
} /* make_seq_temp_array */

/*
 * if the section dimension is only the leading dimension, return TRUE
 *  else, FALSE
 */
static LOGICAL
leading_section(int ast)
{
  if (A_TYPEG(ast) == A_SUBSCR) {
    int asd, numdim, i, ss;
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd);
    for (i = 0; i < numdim; ++i) {
      ss = ASD_SUBS(asd, i);
      if (A_TYPEG(ss) == A_TRIPLE) {
        /* triple in any but first dimension is bad */
        if (i != 0)
          return FALSE;
      } else {
        /* not having a triple in the first dimension is bad */
        if (i == 0)
          return FALSE;
        /* any dimension having a vector subscript is bad */
        if (A_SHAPEG(ss) != 0)
          return FALSE;
      }
    }
    /* here, must have a triple in 1st dimension and no others */
    return TRUE;
  }
  return FALSE;
} /* leading_section */

static void
copy_arg_to_seq_tmp(int entry, int loc, int eledtype, int arraysptr,
                    int arr_ast, int std, int *retval, int *descr,
                    LOGICAL actual_is_contiguous, LOGICAL stride1,
                    LOGICAL wholearray)
{
  int tmp, tmpptr = 0, tmp_id, tmp_ast, array_ptr_ast = 0, iftest = 0, ifast;
  int copy_args = 0;
  int asn, ast, forall, std1;
  int shape, ndim, i;
  int align;
  int subscr[MAXDIMS];

  /* save alignment so that tmp will not aligned */
  align = ALIGNG(arraysptr);

#if DEBUG
  if (DBGBIT(51, 1)) {
    fprintf(gbl.dbgfil, "copy_arg_to_seq_tmp, call to %s, argument %d, array "
                        "%s\n    contiguous=%d, stride1=%d, wholearray=%d\n",
            SYMNAME(entry), loc, SYMNAME(arraysptr), actual_is_contiguous,
            stride1, wholearray);
    fprintf(gbl.dbgfil, " copy_f77_argl test is POINTER(%s)=%d && "
                        "MIDNUM(%s)=%d && contiguous=%d\n",
            SYMNAME(arraysptr), (int)POINTERG(arraysptr),
            SYMNAME(arraysptr), MIDNUMG(arraysptr), actual_is_contiguous);
    fprintf(gbl.dbgfil, " copy_f90_arg  test is POINTER(%s)=%d && "
                        "MIDNUM(%s)=%d && stride1=%d && !HCCSYMG(%s)=%d\n",
            SYMNAME(arraysptr), (int)POINTERG(arraysptr),
            SYMNAME(arraysptr), MIDNUMG(arraysptr), stride1, SYMNAME(entry),
            (int)HCCSYMG(entry));
  }
#endif

  if (POINTERG(arraysptr) && MIDNUMG(arraysptr) &&
      actual_is_contiguous) {
    int sec, secss, copyfunc, size, ddd;
    tmp = make_seq_temp_array(A_SHAPEG(arr_ast), eledtype, arraysptr, 1, std);
    tmp_id = mk_id(tmp);
    tmpptr = MIDNUMG(tmp);

    array_ptr_ast = check_member(arr_ast, mk_id(MIDNUMG(arraysptr)));

    if (OPTARGG(arraysptr)) {
      int present = ast_intr(I_PRESENT, stb.user.dt_log, 1, array_ptr_ast);
      ast = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(ast, present);
      add_stmt_before(ast, std);
    }
    /*
     * call runtime to see if copy is necessary
     */
    copyfunc = mk_id(sym_mkfunc(mkRteRtnNm(RTE_copy_f77_argl), DT_ADDR));
    copy_args = mk_argt(6);
    ARGT_ARG(copy_args, 0) = array_ptr_ast;
    if (!wholearray) {
      secss = find_section_subscript(arr_ast);
      sec = make_sec_from_ast(secss, std, std, 0, 0);
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(sec));
    } else {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(SDSCG(arraysptr)));
    }
    ddd = first_element_from_section(arr_ast);
    ARGT_ARG(copy_args, 2) = ddd;
    ARGT_ARG(copy_args, 3) = mk_id(tmpptr);
    ARGT_ARG(copy_args, 4) = astb.i1;
    size = size_ast(arraysptr, DDTG(DTYPEG(arraysptr)));
    ARGT_ARG(copy_args, 5) = size;

    ast = mk_func_node(A_CALL, copyfunc, 6, copy_args);
    add_stmt_before(ast, std);

    if (OPTARGG(arraysptr)) {
      ast = mk_stmt(A_ELSE, 0);
      add_stmt_before(ast, std);

      ast = mk_assn_stmt(mk_id(tmpptr), mk_unop(OP_LOC, astb.ptr0, DT_PTR),
                         DT_PTR);
      add_stmt_before(ast, std);

      ast = mk_stmt(A_ENDIF, 0);
      add_stmt_before(ast, std);
    }
    copy_args = mk_argt(6);
    ARGT_ARG(copy_args, 0) = array_ptr_ast;
    if (!wholearray) {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(sec));
    } else {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(SDSCG(arraysptr)));
    }
    ARGT_ARG(copy_args, 2) = astb.i0;
    ARGT_ARG(copy_args, 3) = mk_id(tmpptr);
    if (need_copyout(entry, loc)) {
      ARGT_ARG(copy_args, 4) = astb.i0;
    } else {
      ARGT_ARG(copy_args, 4) = mk_cval(2, DT_INT);
    }
    ARGT_ARG(copy_args, 5) = size;

    ast = mk_func_node(A_CALL, copyfunc, 6, copy_args);
    add_stmt_after(ast, std);
#if DEBUG
    if (DBGBIT(51, 1)) {
      fprintf(gbl.dbgfil, "...add call to RTE_copy_f77_argl\n\n");
    }
#endif
    if (!HCCSYMG(entry) && !CCSYMG(entry)) {
      ccff_info(MSGFTN, "FTN020", 1, gbl.lineno,
                "Possible copy in and copy out of %sptr in call to %entry",
                "sptr=%s", SYMNAME(arraysptr), "entry=%s", SYMNAME(entry),
                NULL);
    }
  } else if (ASSUMSHPG(arraysptr) && actual_is_contiguous) {
    int dtype;
    int sec, secss, copyfunc, size, ddd;

    dtype = DTYPEG(arraysptr);
    if (DTY(dtype) == TY_ARRAY &&
        (ADD_NUMDIM(dtype) == 1 || leading_section(arr_ast))) {
      *retval = first_element_from_section(arr_ast);
      *descr = mk_id(DESCRG(arraysptr));
      if (OPTARGG(arraysptr) && eledtype != DT_ASSCHAR) {
        tmp =
            make_seq_temp_array(A_SHAPEG(arr_ast), eledtype, arraysptr, 1, std);
        tmp_id = mk_id(tmp);
        tmpptr = MIDNUMG(tmp);

        copyfunc =
            mk_id(sym_mkfunc(mkRteRtnNm(RTE_addr_1_dim_1st_elem), DT_ADDR));
        copy_args = mk_argt(3);
        ARGT_ARG(copy_args, 0) = A_LOPG(arr_ast);
        ARGT_ARG(copy_args, 1) = *retval;
        ARGT_ARG(copy_args, 2) = mk_id(tmpptr);
        *retval = tmp_id;
        ast = mk_func_node(A_CALL, copyfunc, 3, copy_args);
        add_stmt_before(ast, std);
      }
      return;
    }
    tmp = make_seq_temp_array(A_SHAPEG(arr_ast), eledtype, arraysptr, 1, std);
    tmp_id = mk_id(tmp);
    tmpptr = MIDNUMG(tmp);

    if (OPTARGG(arraysptr)) {
      int present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(arraysptr));
      ast = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(ast, present);
      add_stmt_before(ast, std);
    }

    /*
     * call runtime to see if copy is necessary
     */
    copyfunc = mk_id(sym_mkfunc(mkRteRtnNm(RTE_copy_f77_argsl), DT_ADDR));
    copy_args = mk_argt(6);
    ARGT_ARG(copy_args, 0) = mk_id(arraysptr);
    if (!wholearray) {
      secss = find_section_subscript(arr_ast);
      sec = make_sec_from_ast(secss, std, std, 0, 0);
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(sec));
    } else {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(DESCRG(arraysptr)));
    }
    ddd = first_element_from_section(arr_ast);
    ARGT_ARG(copy_args, 2) = ddd;
    ARGT_ARG(copy_args, 3) = mk_id(tmpptr);
    ARGT_ARG(copy_args, 4) = astb.i1;
    size = size_ast(arraysptr, DDTG(DTYPEG(arraysptr)));
    ARGT_ARG(copy_args, 5) = size;

    ast = mk_func_node(A_CALL, copyfunc, 6, copy_args);
    add_stmt_before(ast, std);

    if (OPTARGG(arraysptr)) {
      ast = mk_stmt(A_ELSE, 0);
      add_stmt_before(ast, std);

      ast = mk_assn_stmt(mk_id(tmpptr), mk_unop(OP_LOC, astb.ptr0, DT_PTR),
                         DT_PTR);
      add_stmt_before(ast, std);

      ast = mk_stmt(A_ENDIF, 0);
      add_stmt_before(ast, std);
    }

    copy_args = mk_argt(6);
    ARGT_ARG(copy_args, 0) = mk_id(arraysptr);
    if (!wholearray) {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(sec));
    } else {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(DESCRG(arraysptr)));
    }
    ARGT_ARG(copy_args, 2) = astb.i0;
    ARGT_ARG(copy_args, 3) = mk_id(tmpptr);
    if (need_copyout(entry, loc)) {
      ARGT_ARG(copy_args, 4) = astb.i0;
    } else {
      ARGT_ARG(copy_args, 4) = mk_cval(2, DT_INT);
    }
    ARGT_ARG(copy_args, 5) = size;

    ast = mk_func_node(A_CALL, copyfunc, 6, copy_args);
    add_stmt_after(ast, std);
#if DEBUG
    if (DBGBIT(51, 1)) {
      fprintf(gbl.dbgfil, "...add call to RTE_copy_f77_argsl\n\n");
    }
#endif
    if (!HCCSYMG(entry) && !CCSYMG(entry)) {
      ccff_info(MSGFTN, "FTN020", 1, gbl.lineno,
                "Possible copy in and copy out of %sptr in call to %entry",
                "sptr=%s", SYMNAME(arraysptr), "entry=%s", SYMNAME(entry),
                NULL);
    }
  } else if (POINTERG(arraysptr) && MIDNUMG(arraysptr) &&
             stride1 && !HCCSYMG(entry)) {
    int sec, secss, secd, copyfunc, size;
    int dd;

    tmp = make_seq_temp_array(A_SHAPEG(arr_ast), eledtype, arraysptr, 1, std);
    tmp_id = mk_id(tmp);
    tmpptr = MIDNUMG(tmp);

    array_ptr_ast = check_member(arr_ast, mk_id(MIDNUMG(arraysptr)));

    if (!wholearray) {
      secss = find_section_subscript(arr_ast);
      sec = make_sec_from_ast(secss, std, std, 0, 0);
      secd = check_member(arr_ast, mk_id(sec));
    } else {
      secd = check_member(arr_ast, mk_id(SDSCG(arraysptr)));
    }
    dd = find_dummy(entry, loc);
    if (dd && ASSUMSHPG(dd) && ignore_tkr(dd, IGNORE_C)) {
      *retval = A_LOPG(arr_ast);
      *descr = secd;
      return;
    }
    /*
     * call runtime to see if copy is necessary
     */
    copyfunc = mk_id(sym_mkfunc(mkRteRtnNm(RTE_copy_f90_argl), DT_ADDR));
    copy_args = mk_argt(6);
    ARGT_ARG(copy_args, 0) = array_ptr_ast;
    ARGT_ARG(copy_args, 1) = secd;
    ARGT_ARG(copy_args, 2) = mk_id(tmpptr);
    ARGT_ARG(copy_args, 3) = mk_id(DESCRG(tmp));
    ARGT_ARG(copy_args, 4) = astb.i1;
    size = size_ast(arraysptr, DDTG(DTYPEG(arraysptr)));
    ARGT_ARG(copy_args, 5) = size;

    ast = mk_func_node(A_CALL, copyfunc, 6, copy_args);
    add_stmt_before(ast, std);

    copy_args = mk_argt(6);
    ARGT_ARG(copy_args, 0) = array_ptr_ast;
    if (!wholearray) {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(sec));
    } else {
      ARGT_ARG(copy_args, 1) = check_member(arr_ast, mk_id(SDSCG(arraysptr)));
    }
    ARGT_ARG(copy_args, 2) = mk_id(tmpptr);
    ARGT_ARG(copy_args, 3) = mk_id(DESCRG(tmp));
    if (need_copyout(entry, loc)) {
      ARGT_ARG(copy_args, 4) = astb.i0;
    } else {
      ARGT_ARG(copy_args, 4) = mk_cval(2, DT_INT);
    }
    ARGT_ARG(copy_args, 5) = size;

    ast = mk_func_node(A_CALL, copyfunc, 6, copy_args);
    add_stmt_after(ast, std);
    DESCUSEDP(tmp, 1);
    NODESCP(tmp, 0);
#if DEBUG
    if (DBGBIT(51, 1)) {
      fprintf(gbl.dbgfil, "...add call to RTE_copy_f90_arg\n\n");
    }
#endif
    if (!HCCSYMG(entry) && !CCSYMG(entry)) {
      ccff_info(MSGFTN, "FTN020", 1, gbl.lineno,
                "Possible copy in and copy out of %sptr in call to %entry",
                "sptr=%s", SYMNAME(arraysptr), "entry=%s", SYMNAME(entry),
                NULL);
    }
  } else {
    int cico;

    cico = 0;
    /*
     * for now, there's always a copy-in -- someday, can check for
     * intent(out)
     */
    cico |= 0x1;
    tmp = make_seq_temp_array(A_SHAPEG(arr_ast), eledtype, arraysptr, 0, std);
    tmp_id = mk_id(tmp);
    /*
     * generate unconditional inline copy
     */
    if (XBIT(28, 0x20) && POINTERG(arraysptr) && MIDNUMG(arraysptr)) {
      /* make the tmp be based, conditionally allocate it */
      array_ptr_ast = check_member(arr_ast, mk_id(MIDNUMG(arraysptr)));

      /* IF( array$p .eq. 0 )THEN */
      iftest = mk_binop(OP_NE, array_ptr_ast, astb.i0, DT_LOG);
      ifast = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(ifast, iftest);
      add_stmt_before(ifast, std);
    }
    shape = A_SHAPEG(arr_ast);
    ndim = SHD_NDIM(shape);
    for (i = 0; i < ndim; ++i) {
      int astlw, astup, aststride;
      astlw = SHD_LWB(shape, i);
      if (A_ALIASG(astlw))
        astlw = A_ALIASG(astlw);
      astup = SHD_UPB(shape, i);
      if (A_ALIASG(astup))
        astup = A_ALIASG(astup);
      aststride = SHD_STRIDE(shape, i);
      if (aststride && A_ALIASG(aststride))
        aststride = A_ALIASG(aststride);
      if (aststride == astb.bnd.one || aststride == astb.i1)
        aststride = 0;
      astup = mk_binop(OP_SUB, astup, astlw, astb.bnd.dtype);
      if (!aststride) {
        astup = mk_binop(OP_ADD, astup, astb.bnd.one, astb.bnd.dtype);
      } else {
        astup = mk_binop(OP_ADD, astup, aststride, astb.bnd.dtype);
        astup = mk_binop(OP_DIV, astup, aststride, astb.bnd.dtype);
      }
      subscr[i] = mk_triple(astb.bnd.one, astup, 0);
    }
    tmp_ast = mk_subscr(tmp_id, subscr, ndim, eledtype);
    if (ALLOCG(tmp)) {
      /* ALLOCATE(tmp(...)) */
      mk_mem_allocate(tmp_id, subscr, std, arr_ast);
    }
    /* tmp = array */
    asn = mk_assn_stmt(tmp_ast, arr_ast, eledtype);
    shape = A_SHAPEG(tmp_ast);
    forall = make_forall(shape, tmp_ast, 0, 0);
    forall = rename_forall_list(forall);
    ast = normalize_forall(forall, asn, 0);
    A_IFSTMTP(forall, ast);
    A_IFEXPRP(forall, 0);
    std1 = add_stmt_before(forall, std);
    if (XBIT(28, 0x20) && POINTERG(arraysptr) && MIDNUMG(arraysptr)) {
      /* ENDIF */
      ifast = mk_stmt(A_ENDIF, 0);
      add_stmt_before(ifast, std);
    }
    forall_opt1(forall);
    process_forall(std1);
    if (pure_gbl.local_mode) {
      /* no communication */
      scalarize(std1, forall, TRUE);
    } else {
      transform_forall(std1, forall);
    }

    if (XBIT(28, 0x20) && POINTERG(arraysptr) && MIDNUMG(arraysptr)) {
      /* ENDIF */
      ifast = mk_stmt(A_ENDIF, 0);
      add_stmt_after(ifast, std);
    }
    if (ALLOCG(tmp)) {
      mk_mem_deallocate(tmp_id, std);
    }

    if (need_copyout(entry, loc)) {
      asn = mk_assn_stmt(arr_ast, tmp_ast, eledtype);

      shape = A_SHAPEG(arr_ast);
      forall = make_forall(shape, arr_ast, 0, 0);
      forall = rename_forall_list(forall);
      ast = normalize_forall(forall, asn, 0);
      A_IFSTMTP(forall, ast);
      A_IFEXPRP(forall, 0);
      /*forall = rename_forall_list(forall);*/
      std1 = add_stmt_after(forall, std);
      cico |= 0x2;
      forall_opt1(forall);
      process_forall(std1);
      if (pure_gbl.local_mode) {
        /* no communication */
        scalarize(std1, forall, TRUE);
      } else {
        transform_forall(std1, forall);
      }
    }

    if (XBIT(28, 0x20) && POINTERG(arraysptr) && MIDNUMG(arraysptr)) {
      /* IF( array$p .ne. 0 )THEN */
      iftest = mk_binop(OP_NE, array_ptr_ast, astb.i0, DT_LOG);
      ifast = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(ifast, iftest);
      add_stmt_after(ifast, std);
    }

#if DEBUG
    if (DBGBIT(51, 1)) {
      fprintf(gbl.dbgfil, "...add allocate/free\n\n");
    }
#endif
    if (!HCCSYMG(entry) && !CCSYMG(entry)) {
      if (cico == 1)
        ccff_info(MSGFTN, "FTN021", 1, gbl.lineno,
                  "Copy in of %sptr in call to %entry", "sptr=%s",
                  SYMNAME(arraysptr), "entry=%s", SYMNAME(entry), NULL);
      else if (cico == 2)
        ccff_info(MSGFTN, "FTN022", 1, gbl.lineno,
                  "Copy out of %sptr in call to %entry", "sptr=%s",
                  SYMNAME(arraysptr), "entry=%s", SYMNAME(entry), NULL);
      else
        ccff_info(MSGFTN, "FTN023", 1, gbl.lineno,
                  "Copy in and copy out of %sptr in call to %entry", "sptr=%s",
                  SYMNAME(arraysptr), "entry=%s", SYMNAME(entry), NULL);
    }
  }

  *retval = tmp_id;
  *descr = mk_id(DESCRG(tmp));
} /* copy_arg_to_seq_tmp */

int
rename_forall_list(int forall)
{
  int isptr, new_isptr;
  int oldast, newast;
  int newforall;
  int list, listp;

  list = A_LISTG(forall);
  ast_visit(1, 1);
  for (listp = list; listp != 0; listp = ASTLI_NEXT(listp)) {
    isptr = ASTLI_SPTR(listp);
    new_isptr = sym_get_scalar(SYMNAME(isptr), "i", astb.bnd.dtype);
    newast = mk_id(new_isptr);
    oldast = mk_id(isptr);
    ast_replace(oldast, newast);
  }

  newforall = ast_rewrite(forall);
  ast_unvisit();
  A_ARRASNP(newforall, A_ARRASNG(forall));
  A_STARTP(newforall, A_STARTG(forall));
  A_NCOUNTP(newforall, A_NCOUNTG(forall));
  A_CONSTBNDP(newforall, A_CONSTBNDG(forall));

  return newforall;
}

/* This routine is to make array element from array section.
 * For example, A(3:20,7:30) will be A(3,7). It only uses lower bounds.
 * This routine is used to implement sequence association.
 */

static int
first_element_from_section(int arr)
{
  ADSC *ad;
  int i;
  int glb;
  int asd;
  int triple, lop, parent;
  int sptr;
  int ndim, numdim;
  int newarr;
  int subs[MAXDIMS];

  switch (A_TYPEG(arr)) {
  case A_ID:
    return arr;
  case A_MEM:
    /* do the parent */
    parent = first_element_from_section(A_PARENTG(arr));
    return mk_member(parent, A_MEMG(arr), A_DTYPEG(A_MEMG(arr)));
  case A_SUBSCR:
    lop = A_LOPG(arr);
    if (A_TYPEG(lop) == A_MEM) {
      parent = first_element_from_section(A_PARENTG(lop));
      lop = mk_member(parent, A_MEMG(lop), A_DTYPEG(A_MEMG(lop)));
      A_LOPP(arr, lop);
    }
    sptr = sptr_of_subscript(arr);
    assert(is_array_type(sptr), "first_element_from_section: arg not array", 0,
           4);
    ndim = rank_of(DTYPEG(sptr));
    ad = AD_DPTR(DTYPEG(sptr));
    asd = A_ASDG(arr);
    numdim = ASD_NDIM(asd);
    assert(numdim == ndim, "first_element_from_sectio: numdim from ST", sptr,
           3);
    for (i = 0; i < ndim; ++i) {
      if (A_TYPEG(triple = ASD_SUBS(asd, i)) == A_TRIPLE) {
        glb = A_LBDG(triple);
        if (glb == 0)
          glb = AD_LWAST(ad, i);
        if (glb == 0)
          glb = mk_isz_cval(1, astb.bnd.dtype);
      } else {
        glb = ASD_SUBS(asd, i);
      }
      subs[i] = glb;
    }
    newarr = mk_subscr(A_LOPG(arr), subs, ndim, DTY(A_DTYPEG(A_LOPG(arr)) + 1));
    return newarr;
  default:
    interr("first_element_from_section: unknown ast", arr, 4);
    return 0;
  }
}

/* assign 'ast' to a temp integer before 'std' */
static int
tmp_assign(int ast, int std)
{
  int tmp, tmpast, asn, newstd;
  LOGICAL rhs_is_dist = FALSE;
  tmp = sym_get_scalar("sub", "a", DT_INT);
  /* make assignment to temp_sclr */
  asn = mk_stmt(A_ASN, DT_INT);
  tmpast = mk_id(tmp);
  A_DESTP(asn, tmpast);
  A_SRCP(asn, ast);

  newstd = add_stmt_before(asn, std);
  STD_AST(newstd) = asn;
  A_STDP(asn, newstd);
  ast = insert_comm_before(newstd, ast, &rhs_is_dist, FALSE);
  A_SRCP(asn, ast);
  return tmp;
} /* tmp_assign */

/* where 'ast' is the expression we are looking for,
 * look at all arguments; if the argument is an array or member,
 * look at all scalar subscripts.  We don't need to worry about
 * vector subscripts, they are handled by creating a section.
 * If ast appears in any of the scalar subscripts, remove that
 * subscript to a temp */
static void
remove_from_scalar_subscript(int ast, int argt, int nargs, int std)
{
  int i, ele, asd, ndim, d, firsttmp;
  for (i = 0; i < nargs; ++i) {
    ele = ARGT_ARG(argt, i);
    /* look at all scalar subscripts */
    firsttmp = 1;
    while (ele) {
      switch (A_TYPEG(ele)) {
      default: /* operators, simple ID, leave alone */
        ele = 0;
        break;
      case A_MEM: /* look for other subscripts */
        ele = A_PARENTG(ele);
        break;
      case A_SUBSCR: /* look for 'ast' in any scalar subscripts */
        asd = A_ASDG(ele);
        ndim = ASD_NDIM(asd);
        for (d = 0; d < ndim; ++d) {
          int sub;
          sub = ASD_SUBS(asd, d);
          /* ignore vector subscripts here */
          if (A_SHAPEG(sub))
            continue;
          if (A_TYPEG(sub) == A_TRIPLE)
            continue;
          /* if already replaced don't need to do it again */
          if (A_REPLG(sub))
            continue;
          if (expr_dependent(sub, ast, std, std)) {
            int tmp;
            tmp = tmp_assign(sub, std);
            SYMLKP(tmp, firsttmp);
            firsttmp = tmp;
            ADDRESSP(tmp, sub); /* stuff it here */
          }
        }
        /* go to subscript parent */
        ele = A_LOPG(ele);
        break;
      }
    }
    if (firsttmp > 1) {
      int t, nt, sub;
      ast_visit(1, 1);
      for (t = firsttmp; t > 1; t = nt) {
        nt = SYMLKG(t);
        SYMLKP(t, 1);
        sub = ADDRESSG(t);
        ADDRESSP(t, 0);
        ast_replace(sub, mk_id(t));
      }
      ele = ARGT_ARG(argt, i);
      ele = ast_rewrite(ele);
      ARGT_ARG(argt, i) = ele;
      ast_unvisit();
    }
  }
} /* remove_from_scalar_subscript */

static void
is_function_common(int ast, LOGICAL *any)
{
  int sptr;
  switch (A_TYPEG(ast)) {
  case A_FUNC:
    *any = TRUE;
    break;
  case A_ID:
    sptr = A_SPTRG(ast);
    switch (STYPEG(sptr)) {
    case ST_VAR:
    case ST_ARRAY:
      if (SCG(sptr) == SC_CMBLK) {
        *any = TRUE;
      }
      break;
    default:;
    }
    break;
  }
} /* is_function_common */

static LOGICAL
any_functions(int ast)
{
  LOGICAL any = FALSE;
  ast_visit(1, 1);
  ast_traverse(ast, NULL, is_function_common, &any);
  ast_unvisit();
  return any;
} /* any_functions */

/* look at all arguments; if the argument is an array or member,
 * look at all scalar subscripts.  We don't need to worry about
 * vector subscripts, they are handled by creating a section.
 * If a function call or common variable
 * appears in any of the scalar subscripts, remove that
 * subscript to a temp */
static void
remove_function_common_from_scalar_subscript(int argt, int nargs, int std)
{
  int i, ele, asd, ndim, d, firsttmp;
  for (i = 0; i < nargs; ++i) {
    ele = ARGT_ARG(argt, i);
    /* look at all scalar subscripts */
    firsttmp = 1;
    while (ele) {
      switch (A_TYPEG(ele)) {
      default: /* operators, simple ID, leave alone */
        ele = 0;
        break;
      case A_MEM: /* look for other subscripts */
        ele = A_PARENTG(ele);
        break;
      case A_SUBSCR: /* look for 'ast' in any scalar subscripts */
        asd = A_ASDG(ele);
        ndim = ASD_NDIM(asd);
        for (d = 0; d < ndim; ++d) {
          int sub;
          sub = ASD_SUBS(asd, d);
          /* ignore vector subscripts here */
          if (A_SHAPEG(sub))
            continue;
          if (A_TYPEG(sub) == A_TRIPLE)
            continue;
          /* if already replaced don't need to do it again */
          if (A_REPLG(sub))
            continue;
          if (any_functions(sub)) {
            int tmp;
            tmp = tmp_assign(sub, std);
            SYMLKP(tmp, firsttmp);
            firsttmp = tmp;
            ADDRESSP(tmp, sub); /* stuff it here */
          }
        }
        /* go to subscript parent */
        ele = A_LOPG(ele);
        break;
      }
    }
    if (firsttmp > 1) {
      int t, nt, sub;
      ast_visit(1, 1);
      for (t = firsttmp; t > 1; t = nt) {
        nt = SYMLKG(t);
        SYMLKP(t, 1);
        sub = ADDRESSG(t);
        ADDRESSP(t, 0);
        ast_replace(sub, mk_id(t));
      }
      ele = ARGT_ARG(argt, i);
      ele = ast_rewrite(ele);
      ARGT_ARG(argt, i) = ele;
      ast_unvisit();
    }
  }
} /* remove_function_common_from_scalar_subscript */

/* take out scalar arguments if they are used as part of subscript arguments
 * For example here: call sub(i,j, a(i,j))
 * subroutine may change the value of i and j
 */
void
remove_alias(int std, int ast)
{
  int argt;
  int nargs;
  int i;
  int ele;

  if (pure_gbl.local_mode)
    return;
  argt = A_ARGSG(ast);
  nargs = A_ARGCNTG(ast);
  for (i = 0; i < nargs; ++i) {
    ele = ARGT_ARG(argt, i);
    switch (A_TYPEG(ele)) {
    case A_ID:
    case A_SUBSCR:
    case A_MEM:
      break;
    default:
      continue;
    }

    /* we have a scalar variable, array element, or member
     * see if it is used in the scalar subscript of another argument. */
    remove_from_scalar_subscript(ele, argt, nargs, std);
  }
  /* look for function calls in scalar subscripts, remove them, too. */
  remove_function_common_from_scalar_subscript(argt, nargs, std);
}

static LOGICAL
is_desc_needed(int entry, int arr_ast, int loc)
{
  int iface; /* sptr of explicit interface; could be zero */
  int dscptr;
  int sptr1;

  proc_arginfo(entry, NULL, &dscptr, &iface);

  /* only user procedure may not need descr */
  if (iface && HCCSYMG(iface))
    return TRUE;
  if (iface && STYPEG(entry) != ST_PROC && STYPEG(entry) != ST_ENTRY
  && (STYPEG(entry) != ST_MEMBER || !VTABLEG(entry)) && !is_procedure_ptr(entry))
    return TRUE;
  /* for F90, F77, C, need descriptor if copy-in is needed */
  if (!dscptr)
    return FALSE;
  /* sptr = sptr_of_subscript(arr_ast);*/
  sptr1 = aux.dpdsc_base[dscptr + loc];
  if (sptr1 && is_kopy_in_needed(sptr1))
    return TRUE;
  /* for F90, need descriptor for assumed-shape arrays */
  if (DTY(DTYPEG(sptr1)) == TY_ARRAY && ASSUMSHPG(sptr1))
    return TRUE;
  return FALSE;
}

/*
 * Need a copy-out?
 *  if we know the argument is intent(in), no;
 *  otherwise, default is yes
 */
static int
need_copyout(int entry, int pos)
{
  int sptr;
  if (HCCSYMG(entry) && INTENTG(entry) == INTENT_IN) {
    return 0;
  }
  sptr = find_dummy(entry, pos);
  if (sptr && INTENTG(sptr) == INTENT_IN)
    return 0;
  return 1;
} /* need_copyout */

/*  Continuous memory rule:
 *   Trailing dimensions are scalars, preceded by at most one non-full
 *   dimension (whose stride is 1 and lower bound equal to the dimension's
 *   lower bound), preceded by full dimensions (a full dimension is a triple
 *   spanning the dimension's full extent).
 *   When calling F77_LOCAL, all but the last non-scalar dimension must be
 *   undistributed.
 */
static LOGICAL
continuous_section(int entry, int arr_ast, int loc, int onlyfirst)
{
  int asd;
  int ndims, dim;
  int astsub;
  int sptr;

  if (!A_SHAPEG(arr_ast))
    return TRUE;
  asd = A_ASDG(arr_ast);
  ndims = ASD_NDIM(asd);

  /* Find the 1st non-scalar dimension. */
  for (dim = ndims - 1; dim >= 0; dim--)
    if (A_TYPEG(ASD_SUBS(asd, dim)) == A_TRIPLE) {
      break;
    }
  if (dim < 0)
    return TRUE;
  sptr = sym_of_ast(arr_ast);
  if (onlyfirst && SCG(sptr) == SC_DUMMY && ASSUMSHPG(sptr) && dim > 0)
    return FALSE;
  astsub = ASD_SUBS(asd, dim);
  if (A_STRIDEG(astsub) && A_STRIDEG(astsub) != astb.i1 &&
      A_STRIDEG(astsub) != astb.bnd.one)
    return FALSE;

  /* Leading dimensions must be full. */
  for (--dim; dim >= 0; dim--) {
    if (!is_whole_dim(arr_ast, dim))
      return FALSE;
  }
  return TRUE;
}

/*  stride-1 memory rule:
 *   Leftmost dimension has no stride.
 */
static LOGICAL
stride_1_section(int entry, int arr_ast, int pos, int std)
{
  int asd;
  int ndims, dim;
  int astsub, sptr;

  if (!A_SHAPEG(arr_ast))
    return TRUE;
  if (A_TYPEG(arr_ast) == A_SUBSCR) {
    asd = A_ASDG(arr_ast);
    ndims = ASD_NDIM(asd);

    /* Find the 1st non-scalar dimension. */
    for (dim = ndims - 1; dim >= 0; --dim) {
      if (A_TYPEG(ASD_SUBS(asd, dim)) == A_TRIPLE)
        break;
    }

    if (dim < 0)
      return TRUE; /* no triplets */
    dim = 0;
    astsub = ASD_SUBS(asd, dim);
    if (A_TYPEG(astsub) != A_TRIPLE)
      return FALSE; /* some triplets, but not in leftmost dimension */
    if (A_STRIDEG(astsub) && A_STRIDEG(astsub) != astb.i1 &&
        A_STRIDEG(astsub) != astb.k1)
      return FALSE; /* leftmost triplet is not stride 1 */
  }
  sptr = memsym_of_ast(arr_ast);
  if (POINTERG(sptr) || (TARGETG(sptr) && XBIT(58,0x400000))) {
      /*
       * Is this a stride-1 pointer array section?  If the corresponding
       * dummy is assumed-shape, we cannot omit the copy arg calls.  The
       * pointer will locate beginning address of the target and the lbase
       * of the descriptor can be non-zero; lbase, if non-zero, is the
       * distance from the beginning of the target to start of the section.
       * Eventually, we'll pass the pointer & descriptor 'as-is'; however,
       * we expected the assumed-shape dummmy to correspond to the first
       * element of the section.  Even if we passed the address of the
       * first element, the descriptor's lbase could still be non-zero.
       */
    int dummy_sptr;
    dummy_sptr = find_dummy(entry, pos);
    if (dummy_sptr == 0 || !ASSUMSHPG(dummy_sptr)) {
      if (pta_stride1(std, sptr)) {
        return TRUE;
      }
    }
    return FALSE;
  }
  return TRUE;
} /* stride_1_section */

void
call_analyze(void)
{
  int std, stdnext;
  int ast;
  int parallel_depth;
  int task_depth;
  int target_depth;

  init_region();
  parallel_depth = 0;
  task_depth = 0;
  target_depth = 0;
  templist = NULL;
  if (flg.opt >= 2 && XBIT(53, 2)) {
    points_to();
  }
  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    gbl.lineno = STD_LINENO(std);
    if (STD_PURE(std))
      continue;
    if (STD_LOCAL(std))
      pure_gbl.local_mode = 1; /* don't process for DO-INDEPENDENT */
    else
      pure_gbl.local_mode = 0;
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_MP_PARALLEL:
      ++parallel_depth;
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_TARGET:
      ++target_depth;
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_ENDPARALLEL:
      --parallel_depth;
      if (parallel_depth == 0 && task_depth == 0 && target_depth == 0) {
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    case A_MP_ENDTARGET:
      --target_depth;
      if (parallel_depth == 0 && task_depth == 0 && target_depth == 0) {
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    case A_MP_TASK:
    case A_MP_TASKLOOP:
      ++task_depth;
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_ENDTASK:
    case A_MP_ETASKLOOP:
      --task_depth;
      if (parallel_depth == 0 && task_depth == 0 && target_depth == 0) {
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    }
    transform_all_call(std, ast);
    check_region(std);
  }
  if (flg.opt >= 2 && XBIT(53, 2)) {
    f90_fini_pointsto();
  }

  freearea(TEMP_AREA);
  templist = NULL;
}

static int
transform_all_call(int std, int ast)
{
  int l, r, d, o;
  int l1, l2, l3, l4;
  int a;
  int astnew, stdnew;
  int i, nargs, argt, ag;
  int asd;
  int ndim;
  int subs[MAXDIMS];
  int dest;

  if (!ast)
    return ast;

  a = ast;
  switch (A_TYPEG(a)) {
  /* statements */
  case A_ASN:
    dest = A_DESTG(a);
    dest = transform_all_call(std, A_DESTG(a));
    A_DESTP(a, dest);
    l = transform_all_call(std, A_SRCG(a));
    A_SRCP(a, l);
    return a;
  case A_IF:
  case A_IFTHEN:
    /* if there is any copy-in/copy-out code generated,
     * it must come BEFORE this statement;
     * insert a dummy statement before this one */
    astnew = mk_stmt(A_ASN, DT_LOG);
    stdnew = add_stmt_before(astnew, std);
    l = transform_all_call(stdnew, A_IFEXPRG(a));
    if (STD_NEXT(stdnew) == std) {
      /* didn't need the new statement, delete it */
      STD_PREV(std) = STD_PREV(stdnew);
      STD_NEXT(STD_PREV(stdnew)) = std;
      STD_PREV(stdnew) = STD_NEXT(stdnew) = 0;
      if (astb.std.stg_avail == stdnew + 1) {
        /* no statements added, make this one available again */
        STG_RETURN(astb.std);
      }
      A_IFEXPRP(a, l);
    } else {
      /* have to finish the assignment */
      int sptr, asptr;
      sptr = sym_get_scalar("if", "tmp", DT_LOG);
      asptr = mk_id(sptr);
      A_DESTP(astnew, asptr);
      A_SRCP(astnew, l);
      A_IFEXPRP(a, asptr);
    }
    return a;

  case A_ELSE:
  case A_ELSEIF:
  case A_ENDIF:
    return a;
  case A_AIF:
    l = transform_all_call(std, A_IFEXPRG(a));
    A_IFEXPRP(a, l);
    return a;
  case A_GOTO:
    return a;
  case A_CGOTO:
    l = transform_all_call(std, A_LOPG(a));
    A_LOPP(a, l);
    return a;
  case A_AGOTO:
  case A_ASNGOTO:
    return a;
  case A_DO:
    l1 = transform_all_call(std, A_M1G(a));
    l2 = transform_all_call(std, A_M2G(a));
    if (A_M3G(a))
      l3 = transform_all_call(std, A_M3G(a));
    else
      l3 = 0;
    if (A_M4G(a))
      l4 = transform_all_call(std, A_M4G(a));
    else
      l4 = 0;
    A_M1P(a, l1);
    A_M2P(a, l2);
    A_M3P(a, l3);
    A_M4P(a, l4);
    return a;
  case A_DOWHILE:
    l = transform_all_call(std, A_IFEXPRG(a));
    A_IFEXPRP(a, l);
    return a;
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
  case A_ENTRY:
    return a;
  case A_ICALL:
  case A_CALL:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      if (!ARGT_ARG(argt, i))
        continue;
      ag = _transform_func(std, ARGT_ARG(argt, i));
      ARGT_ARG(argt, i) = ag;
    }
    transform_call(std, a);
    return a;
  case A_REDISTRIBUTE:
  case A_REALIGN:
    return a;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    return a;
  case A_ALLOC:
    /*	interr("transform_all_call: ALLOC not handled", std, 2); */
    return a;
  case A_WHERE:
  case A_ELSEWHERE:
  case A_ENDWHERE:
    interr("transform_all_call: WHERE stmt found", std, 3);
    return a;
  case A_FORALL:
  case A_ENDFORALL:
    return a;
  case A_COMMENT:
  case A_COMSTR:
    return a;
  case A_LABEL:
    return a;
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = transform_all_call(std, A_LOPG(a));
    r = transform_all_call(std, A_ROPG(a));
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = transform_all_call(std, A_LOPG(a));
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(a);
    l = transform_all_call(std, A_LOPG(a));
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(a);
    l = transform_all_call(std, A_LOPG(a));
    return mk_paren(l, d);
  case A_MEM:
    d = A_DTYPEG(a);
    l = transform_all_call(std, A_PARENTG(a));
    return mk_member(l, A_MEMG(a), A_DTYPEG(A_MEMG(a)));
  case A_SUBSTR:
    d = A_DTYPEG(a);
    l1 = transform_all_call(std, A_LOPG(a));
    l2 = l3 = 0;
    if (A_LEFTG(a))
      l2 = transform_all_call(std, A_LEFTG(a));
    if (A_RIGHTG(a))
      l3 = transform_all_call(std, A_RIGHTG(a));
    return mk_substr(l1, l2, l3, d);
  case A_INTR:
    if (INKINDG(A_SPTRG(A_LOPG(a))) == IK_INQUIRY)
      return a;
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ag = transform_all_call(std, ARGT_ARG(argt, i));
      ARGT_ARG(argt, i) = ag;
    }
    A_ARGSP(a, argt);
    return a;
  case A_FUNC:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ag = _transform_func(std, ARGT_ARG(argt, i));
      ARGT_ARG(argt, i) = ag;
    }
    transform_call(std, a);
    return a;
  case A_CNST:
  case A_CMPLXC:
    return a;
  case A_ID:
    return a;
  case A_SUBSCR:
    asd = A_ASDG(a);
    ndim = ASD_NDIM(asd);
    assert(ndim <= MAXDIMS, "transform_all_call: ndim too big", std, 4);
    for (i = 0; i < ndim; ++i) {
      subs[i] = transform_all_call(std, ASD_SUBS(asd, i));
    }
    l = A_LOPG(a);
    a = mk_subscr(l, subs, ndim, A_DTYPEG(a));
    return a;

  case A_TRIPLE:
    l1 = transform_all_call(std, A_LBDG(a));
    l2 = transform_all_call(std, A_UPBDG(a));
    l3 = transform_all_call(std, A_STRIDEG(a));
    return mk_triple(l1, l2, l3);
  case A_HOFFSET:
  case A_HOWNERPROC:
  case A_HLOCALOFFSET:
  case A_HLOCALIZEBNDS:
  case A_HCYCLICLP:
  case A_HALLOBNDS:
  case A_HSECT:
  case A_HCOPYSECT:
  case A_HGATHER:
  case A_HOVLPSHIFT:
  case A_HCSTART:
  case A_HCFINISH:
  case A_HGETSCLR:
  case A_BARRIER:
  case A_MASTER:
  case A_ENDMASTER:
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
  case A_ENDATOMIC:
  case A_CRITICAL:
  case A_ENDCRITICAL:
  case A_NOBARRIER:
  case A_MP_PARALLEL:
  case A_MP_ENDPARALLEL:
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_BARRIER:
  case A_MP_TASKWAIT:
  case A_MP_TASKGROUP:
  case A_MP_ETASKGROUP:
  case A_MP_TASKYIELD:
  case A_MP_PDO:
  case A_MP_ENDPDO:
  case A_MP_SECTIONS:
  case A_MP_ENDSECTIONS:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_PRE_TLS_COPY:
  case A_MP_BCOPYIN:
  case A_MP_COPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_COPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_TASK:
  case A_MP_TASKLOOP:
  case A_MP_TASKFIRSTPRIV:
  case A_MP_TASKDUP:
  case A_MP_ETASKDUP:
  case A_MP_TASKLOOPREG:
  case A_MP_ETASKLOOPREG:
  case A_MP_TASKREG:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
  case A_MP_BMPSCOPE:
  case A_MP_EMPSCOPE:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_FLUSH:
  case A_PREFETCH:
  case A_MP_TARGET:
  case A_MP_ENDTARGET:
  case A_MP_TEAMS:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_ENDTARGETDATA:
  case A_MP_TARGETDATA:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_TARGETUPDATE:
  case A_MP_CANCEL:
  case A_MP_CANCELLATIONPOINT:
  case A_MP_ATOMICREAD:
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
  case A_MP_MAP:
  case A_MP_TARGETLOOPTRIPCOUNT:
  case A_MP_EMAP:
  case A_MP_EREDUCTION:
  case A_MP_BREDUCTION:
  case A_MP_REDUCTIONITEM:
    return a;
  case A_PRAGMA:
    return a;
  default:
    interr("transform_all_call: unknown expression", std, 2);
    return a;
  }
}
