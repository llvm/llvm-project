/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief rewrite function args, etc
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "comm.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "transfrm.h"
#include "gramtk.h"
#include "extern.h"
#include "hpfutl.h"
#include "ccffinfo.h"
#include "dinit.h"
#include "rte.h"
#include "fdirect.h"
#include "mach.h"
#include "rtlRtns.h"
#include "ilidir.h" /* for open_pragma, close_pragma */

static LOGICAL matmul_use_lhs(int, int, int);
static int triplet_extent(int);
static int misalignment(int, int, int);

static LOGICAL is_another_shift(int, int, int, int);
static LOGICAL _is_another_shift(int, LOGICAL *);
static int transform_associated(int, int);
static void transform_mvbits(int, int);
static void transform_merge(int, int);
static void transform_elemental(int, int);
static void transform_c_f_pointer(int, int);
static void transform_c_f_procpointer(int, int);
static void transform_move_alloc(int, int);

static void check_arg_isalloc(int);
static int rewrite_func_ast(int, int, int);
static int rewrite_intr_allocatable(int, int, int);
static LOGICAL ast_has_allocatable_member(int);
static int rewrite_sub_ast(int, int);
static int mk_result_sptr(int, int, int *, int, int, int *);
static LOGICAL take_out_user_def_func(int);
static int matmul(int, int, int);
static int mmul(int, int, int); /* fast matmul */
static int reshape(int, int, int);
static int _reshape(int, DTYPE, int);

static int inline_reduction_f90(int ast, int dest, int lc, LOGICAL *doremove);

static void nop_dealloc(int, int);
static void handle_shift(int s);

/*------ Argument & Expression Rewriting ----------*/
int
gen_islocal_index(int ast, int sptr, int dim, int subAst)
{
  int nargs, argt;
  int newast;
  int align;
  int descr;
  int olb, oub;
  int tmp1, tmp2;

  align = ALIGNG(sptr);
  descr = DESCRG(sptr);
  DESCUSEDP(sptr, TRUE);
  if (!XBIT(47, 0x80) && align) {
    /* inline it; if (idx.ge.sd$desc(olb).and.idx.le.sd$descr(oub)) then */
    olb = check_member(ast, get_owner_lower(descr, dim));
    oub = check_member(ast, get_owner_upper(descr, dim));
    if (normalize_bounds(sptr)) {
      olb = add_lbnd(DTYPEG(sptr), dim, olb, ast);
      oub = add_lbnd(DTYPEG(sptr), dim, oub, ast);
    }
    tmp1 = mk_binop(OP_GE, subAst, olb, DT_LOG);
    tmp2 = mk_binop(OP_LE, subAst, oub, DT_LOG);
    newast = mk_binop(OP_LAND, tmp1, tmp2, DT_LOG);
    return newast;
  }

  nargs = 3;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = check_member(ast, mk_id(descr));
  ARGT_ARG(argt, 1) = mk_cval(dim + 1, astb.bnd.dtype);
  newast = mk_default_int(subAst);
  if (normalize_bounds(sptr))
    newast = sub_lbnd(DTYPEG(sptr), dim, newast, ast);
  ARGT_ARG(argt, 2) = newast;
  newast = mk_func_node(A_FUNC,
                        mk_id(sym_mkfunc(mkRteRtnNm(RTE_islocal_idx), DT_LOG)),
                        nargs, argt);
  NODESCP(A_SPTRG(A_LOPG(newast)), 1);
  A_DTYPEP(newast, DT_LOG);
  return newast;
} /* gen_islocal_index */

#ifdef FLANG_FUNC_UNUSED
static int
gen_scalar_mask(int ast, int list)
{
  return 0;
} /* gen_scalar_mask */
#endif

#ifdef FLANG_FUNC_UNUSED
/*
 * SUM and PRODUCT reductions use a longer datatype for
 * the reduction temporary; for instance, they use
 * REAL*8 for a REAL*4 SUM call
 */
static int
reduction_type(DTYPE dtype)
{
  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    return DT_INT;
  case TY_INT8:
    return dtype;

  case TY_REAL:
    return DT_REAL8;
  case TY_DBLE:
    if (XBIT(57, 0x14) || XBIT(51, 0x80)) {
      /* no real*16, or map real*16 to real*8,
       * or don't use quad precision accumulators */
      return dtype;
    } else {
      return DT_QUAD;
    }
  case TY_QUAD:
    return dtype;

  case TY_CMPLX:
    return DT_CMPLX16;
  case TY_DCMPLX:
    if (XBIT(57, 0x18) || XBIT(51, 0x80)) {
      /* no complex*32, or map complex*32 to complex*16,
       * or don't use quad precision accumulators */
      return dtype;
    } else {
      return DT_QCMPLX;
    }
  case TY_QCMPLX:
    return dtype;
  default:
    return dtype;
  }
} /* reduction_type */
#endif

#ifdef FLANG_FUNC_UNUSED
static int
assign_result(int sptr, int ast, DTYPE dtype, DTYPE dtyperes, int stdnext,
              int lineno)
{
  int tsclr, tsclrAst, asn, std;
  if (dtyperes == dtype)
    return ast;
  /* we had a SUM or PRODUCT where we used a REAL*8 temp for a REAL*4
   * reduction, for instance.  Now, coerce back to REAL*4 */
  tsclr = sym_get_scalar(SYMNAME(sptr), "rr", dtyperes);
  tsclrAst = mk_id(tsclr);
  asn = mk_assn_stmt(tsclrAst, ast, dtyperes);
  std = add_stmt_before(asn, stdnext);
  STD_LINENO(std) = lineno;
  STD_LOCAL(std) = 1;
  STD_PAR(std) = STD_PAR(stdnext);
  STD_TASK(std) = STD_TASK(stdnext);
  STD_ACCEL(std) = STD_ACCEL(stdnext);
  STD_KERNEL(std) = STD_KERNEL(stdnext);
  return tsclrAst;
} /* assign_result */
#endif

/* this will check whether cshift or eoshift needs any communication. */
static LOGICAL
is_no_comm_shift(int func_ast, int func_args)
{
  return TRUE;
}

/*
 * generate inline loops for CSHIFT and EOSHIFT
 */
#define SHIFTMAX 7
/* shift structure */
static struct {
  int shift;                    /* shift distance */
  int dim, cdim;                /* which dimension being shifted */
  int boundary;                 /* for EOSHIFT, boundary value */
  int shifttype;                /* CSHIFT or EOSHIFT */
  int dim_dest, dim_src;        /* which dimensions get shifted */
  int n, m, k;                  /* extent, positive shift amount */
  int nc, mc, kc;               /* constant value of above */
  LOGICAL lt;                   /* less than */
  LOGICAL then_part, else_part; /* nonzero shift, zero shift */
} ss[SHIFTMAX];                 /* shift data */

static struct {
  int shiftcount; /* how many nested shifts */
  int subssrc[MAXSUBS], subsdest[MAXSUBS];
  int src, dest;
  int ndimsrc, ndimdest;
} sg; /* shift global data */

static void
recurse_shift(int s)
{
  if (s < sg.shiftcount) {
    handle_shift(s);
  } else {
    int ast_lhs, ast_rhs, ast;
    ast_lhs =
        mk_subscr(A_LOPG(sg.dest), sg.subsdest, sg.ndimdest, A_DTYPEG(sg.dest));
    ast_rhs =
        mk_subscr(A_LOPG(sg.src), sg.subssrc, sg.ndimsrc, A_DTYPEG(sg.src));
    ast = mk_assn_stmt(ast_lhs, ast_rhs, DTY(A_DTYPEG(A_LOPG(sg.dest)) + 1));
    add_stmt_before(ast, arg_gbl.std);
  }
} /* recurse_shift */

static void
recurse_eoshift(int s)
{
  if (s < sg.shiftcount) {
    handle_shift(s);
  } else {
    int ast_lhs, ast_rhs, ast;
    ast_lhs =
        mk_subscr(A_LOPG(sg.dest), sg.subsdest, sg.ndimdest, A_DTYPEG(sg.dest));
    ast_rhs = ss[s - 1].boundary;
    ast = mk_assn_stmt(ast_lhs, ast_rhs, DTY(A_DTYPEG(A_LOPG(sg.dest)) + 1));
    add_stmt_before(ast, arg_gbl.std);
  }
} /* recurse_eoshift */

static void
handle_shift(int s)
{
  if (A_TYPEG(ss[s].m) != A_CNST) {
    int ast, expr;
    ast = mk_stmt(A_IFTHEN, 0);
    expr = mk_binop(OP_NE, ss[s].m, astb.bnd.zero, DT_LOG);
    A_IFEXPRP(ast, expr);
    add_stmt_before(ast, arg_gbl.std);
  }
  if (ss[s].then_part) {
    int ta, la, ua, xa, lla, uua, sa;
    int tb, lb, ub, xb, llb, uub, sb;
    int tmp1, tmp2;
    ta = sg.subsdest[ss[s].dim_dest];
    la = A_LBDG(ta);
    ua = A_UPBDG(ta);
    sa = A_STRIDEG(ta);
    xa = triplet_extent(ta);
    tb = sg.subssrc[ss[s].dim_src];
    lb = A_LBDG(tb);
    ub = A_UPBDG(tb);
    sb = A_STRIDEG(tb);
    xb = triplet_extent(tb);
    if (ss[s].shifttype == I_CSHIFT) {
      /*  a(la : ua - m*sa : sa) = b(lb + m*sb : ub : sb)  */
      tmp1 = opt_binop(OP_MUL, ss[s].m, sa, astb.bnd.dtype);
      uua = opt_binop(OP_SUB, ua, tmp1, astb.bnd.dtype);
      sg.subsdest[ss[s].dim_dest] = mk_triple(la, uua, sa);

      tmp1 = opt_binop(OP_MUL, ss[s].m, sb, astb.bnd.dtype);
      llb = opt_binop(OP_ADD, lb, tmp1, astb.bnd.dtype);
      sg.subssrc[ss[s].dim_src] = mk_triple(llb, ub, sb);
      recurse_shift(s + 1);

      /* a(la + (n - m)*sa : ua : sa) = b(lb : ub - (n - m)*sb : sb) */
      tmp1 = opt_binop(OP_SUB, xa, ss[s].m, astb.bnd.dtype);
      tmp2 = opt_binop(OP_MUL, tmp1, sa, astb.bnd.dtype);
      lla = opt_binop(OP_ADD, la, tmp2, astb.bnd.dtype);
      sg.subsdest[ss[s].dim_dest] = mk_triple(lla, ua, sa);

      tmp1 = opt_binop(OP_SUB, xb, ss[s].m, astb.bnd.dtype);
      tmp2 = opt_binop(OP_MUL, tmp1, sb, astb.bnd.dtype);
      uub = opt_binop(OP_SUB, ub, tmp2, astb.bnd.dtype);
      sg.subssrc[ss[s].dim_src] = mk_triple(lb, uub, sb);
      recurse_shift(s + 1);
    } else if (ss[s].shifttype == I_EOSHIFT) {
      int ast_lhs, ast_rhs, ast, x;
      /* handle case with m > 0 */
      x = 0;
      if (A_TYPEG(ss[s].m) == A_CNST) {
        if (ss[s].mc > 0) {
          x = 1;
        }
      } else {
        int ast, expr;
        x = 1;
        /* test whether the shift distance is < 0 or > 0 */
        ast = mk_stmt(A_IFTHEN, 0);
        expr = mk_binop(OP_GT, ss[s].m, astb.bnd.zero, DT_LOG);
        A_IFEXPRP(ast, expr);
        add_stmt_before(ast, arg_gbl.std);
      }
      if (x) {
        /*  a(la : ua - m*sa : sa) = b(lb + m*sb : ub : sb)  */
        tmp1 = opt_binop(OP_MUL, ss[s].m, sa, astb.bnd.dtype);
        uua = opt_binop(OP_SUB, ua, tmp1, astb.bnd.dtype);
        sg.subsdest[ss[s].dim_dest] = mk_triple(la, uua, sa);

        tmp1 = opt_binop(OP_MUL, ss[s].m, sb, astb.bnd.dtype);
        llb = opt_binop(OP_ADD, lb, tmp1, astb.bnd.dtype);
        sg.subssrc[ss[s].dim_src] = mk_triple(llb, ub, sb);
        recurse_shift(s + 1);

        /* a(la + (n - m)*sa : ua : sa) = boundary */
        tmp1 = opt_binop(OP_SUB, xa, ss[s].m, astb.bnd.dtype);
        tmp2 = opt_binop(OP_MUL, tmp1, sa, astb.bnd.dtype);
        lla = opt_binop(OP_ADD, la, tmp2, astb.bnd.dtype);
        sg.subsdest[ss[s].dim_dest] = mk_triple(lla, ua, sa);

        ast_lhs = mk_subscr(A_LOPG(sg.dest), sg.subsdest, sg.ndimdest,
                            A_DTYPEG(sg.dest));
        ast_rhs = ss[s].boundary; /* boundary have to be spread if array */
        if (A_SHAPEG(ast_rhs)) {
          /* add spread call */
          int newargt, spread;
          newargt = mk_argt(3);
          ARGT_ARG(newargt, 0) = ast_rhs;
          ARGT_ARG(newargt, 1) = mk_cval(ss[s].dim_dest + 1, astb.bnd.dtype);
          tmp2 = opt_binop(OP_SUB, ua, lla, astb.bnd.dtype);
          if (sa != astb.i1 && sa != astb.bnd.one) {
            tmp2 = opt_binop(OP_DIV, tmp2, sa, astb.bnd.dtype);
          }
          ARGT_ARG(newargt, 2) = mk_cval(tmp2, astb.bnd.dtype);
          spread = mk_id(intast_sym[I_SPREAD]);
          ast_rhs = mk_func_node(A_INTR, spread, 3, newargt);
          A_OPTYPEP(ast_rhs, I_SPREAD);
        }
        ast =
            mk_assn_stmt(ast_lhs, ast_rhs, DTY(A_DTYPEG(A_LOPG(sg.dest)) + 1));
        add_stmt_before(ast, arg_gbl.std);
      }
      /* handle case with m < 0 */
      x = 0;
      if (A_TYPEG(ss[s].m) == A_CNST) {
        if (ss[s].mc < 0) {
          x = 1;
        }
      } else {
        int ast;
        x = 1;
        ast = mk_stmt(A_ELSE, 0);
        add_stmt_before(ast, arg_gbl.std);
      }
      if (x) {
        /* a(la - m*sa : ua : sa) = b(lb : ub - m*sb : sb) */
        tmp1 = opt_binop(OP_MUL, ss[s].m, sa, astb.bnd.dtype);
        lla = opt_binop(OP_SUB, la, tmp1, astb.bnd.dtype);
        sg.subsdest[ss[s].dim_dest] = mk_triple(lla, ua, sa);

        tmp1 = opt_binop(OP_MUL, ss[s].m, sb, astb.bnd.dtype);
        uub = opt_binop(OP_SUB, ub, tmp1, astb.bnd.dtype);
        sg.subssrc[ss[s].dim_src] = mk_triple(lb, uub, sb);
        recurse_shift(s + 1);

        /* a(la : (la-m*sa)-1 : sa) = boundary */
        lla = opt_binop(OP_SUB, lla, astb.bnd.one, astb.bnd.dtype);
        sg.subsdest[ss[s].dim_dest] = mk_triple(la, lla, sa);

        ast_lhs = mk_subscr(A_LOPG(sg.dest), sg.subsdest, sg.ndimdest,
                            A_DTYPEG(sg.dest));
        ast_rhs = ss[s].boundary; /* boundary have to be spread if array */
        if (A_SHAPEG(ast_rhs)) {
          /* add spread call */
          int newargt, spread;
          newargt = mk_argt(3);
          ARGT_ARG(newargt, 0) = ast_rhs;
          ARGT_ARG(newargt, 1) = mk_cval(ss[s].dim_dest + 1, astb.bnd.dtype);
          tmp2 = opt_binop(OP_SUB, ua, lla, astb.bnd.dtype);
          if (sa != astb.i1 && sa != astb.bnd.one) {
            tmp2 = opt_binop(OP_DIV, tmp2, sa, astb.bnd.dtype);
          }
          ARGT_ARG(newargt, 2) = mk_cval(tmp2, astb.bnd.dtype);
          spread = mk_id(intast_sym[I_SPREAD]);
          ast_rhs = mk_func_node(A_INTR, spread, 3, newargt);
          A_OPTYPEP(ast_rhs, I_SPREAD);
        }
        ast =
            mk_assn_stmt(ast_lhs, ast_rhs, DTY(A_DTYPEG(A_LOPG(sg.dest)) + 1));
        add_stmt_before(ast, arg_gbl.std);
      }
      if (A_TYPEG(ss[s].m) != A_CNST) {
        int ast;
        ast = mk_stmt(A_ENDIF, 0);
        add_stmt_before(ast, arg_gbl.std);
      }
    }

    sg.subsdest[ss[s].dim_dest] = ta;
    sg.subssrc[ss[s].dim_src] = tb;
  }

  if (A_TYPEG(ss[s].m) != A_CNST) {
    int ast;
    ast = mk_stmt(A_ELSE, 0);
    add_stmt_before(ast, arg_gbl.std);
  }

  if (ss[s].else_part) {
    /* a(la:ua:sa) = b(lb:ub:sb) */
    if (ss[s].shifttype == I_EOSHIFT)
      recurse_eoshift(s + 1);
    else
      recurse_shift(s + 1);
  }

  if (A_TYPEG(ss[s].m) != A_CNST) {
    int ast;
    ast = mk_stmt(A_ENDIF, 0);
    add_stmt_before(ast, arg_gbl.std);
  }
} /* handle_shift */

/*
 * for an EOSHIFT call with an omitted boundary value,
 * use zero.  This functions returns an AST referencing
 * an appropriate 'zero' value for the given array datatype.
 */
static int
_makezero(DTYPE dtype)
{
  int v[4], sptr;
  INT V;
  int sub, ndims, i;
  int firstast, lastast, ast, member;
  char *str;
  int l, len;
  switch (DTY(dtype)) {
  case TY_HOLL:
  case TY_WORD:
  case TY_INT:
  case TY_LOG:
  case TY_REAL:
  case TY_SINT:
  case TY_BINT:
  case TY_SLOG:
  case TY_BLOG:
    V = 0;
    return mk_cval1(V, dtype);

  case TY_DBLE:
  case TY_QUAD:
  case TY_DWORD:
  case TY_LOG8:
  case TY_INT8:
    v[0] = v[1] = v[2] = v[3] = 0;
    sptr = getcon(v, dtype);
    return mk_cval1((INT)sptr, dtype);

  case TY_CMPLX:
    v[0] = stb.flt0;
    v[1] = stb.flt0;
    sptr = getcon(v, dtype);
    return mk_cval(sptr, dtype);
  case TY_DCMPLX:
    v[0] = stb.dbl0;
    v[1] = stb.dbl0;
    sptr = getcon(v, dtype);
    return mk_cval1(sptr, dtype);
  case TY_QCMPLX:
    v[0] = v[1] = v[2] = v[3] = 0;
    v[0] = getcon(v, DT_QUAD);
    v[1] = v[0];
    sptr = getcon(v, dtype);
    return mk_cval1(sptr, dtype);

  case TY_CHAR:
  case TY_NCHAR:
    /* make blank */
    len = DTY(dtype + 1);
    if (!A_ALIASG(len)) {
      len = 1;
    } else {
      len = A_ALIASG(len);
      len = A_SPTRG(len);
      len = CONVAL2G(len);
    }
    str = (char *)malloc(len + 1);
    for (l = 0; l < len; ++l)
      str[l] = ' ';
    str[len] = '\0';
    sptr = getstring(str, len);
    free(str);
    return mk_id(sptr);
    break;

  case TY_ARRAY:
    /* make an array of zeros */
    sub = _makezero(DTY(dtype + 1));
    ndims = ADD_NUMDIM(dtype);
    for (i = 0; i < ndims; ++i) {
      int j, extent, prevast, ast;
      extent = ADD_EXTNTAST(dtype, i);
      if (!A_ALIASG(extent)) {
        extent = 1;
      } else {
        extent = A_ALIASG(extent);
        extent = A_SPTRG(extent);
        extent = CONVAL2G(extent);
      }
      prevast = 0;
      for (j = 0; j < extent; ++j) {
        ast = mk_init(sub, DTY(dtype + 1));
        A_RIGHTP(ast, prevast);
        prevast = ast;
      }
      sub = ast;
    }
    return sub;

  case TY_STRUCT:
  case TY_DERIVED:
    /* make a structure of zeros */
    firstast = 0;
    lastast = 0;
    for (member = DTY(dtype + 1); member > NOSYM; member = SYMLKG(member)) {
      sub = _makezero(DTYPEG(member));
      ast = mk_init(sub, DTYPEG(member));
      if (firstast == 0) {
        firstast = ast;
      } else {
        A_RIGHTP(lastast, ast);
      }
      lastast = ast;
      A_SPTRP(ast, member);
    }
    return firstast;

  case TY_UNION:
  case TY_PTR:
  case TY_NONE:
  default:
    interr("makezero: unknown datatype", DTY(dtype), 4);
    break;
  }
  return 0;
} /* _makezero */

/*
 * write data-initialization to dinit file for array/structure
 */
static void
putzero(int ast)
{
  /* derived type? */
  for (; ast; ast = A_RIGHTG(ast)) {
    int a, dtype, sptr;
    a = A_LEFTG(ast);
    switch (A_TYPEG(a)) {
    case A_INIT:
      dtype = A_DTYPEG(a);
      if (DTY(dtype) == TY_DERIVED || DTY(dtype) == TY_STRUCT) {
        if (DTY(dtype + 3)) {
          dinit_put(DINIT_TYPEDEF, DTY(dtype + 3));
        }
      }
      putzero(a);
      if (DTY(dtype) == TY_DERIVED || DTY(dtype) == TY_STRUCT) {
        if (DTY(dtype + 3)) {
          dinit_put(DINIT_ENDTYPE, DTY(dtype + 3));
        }
      }
      break;
    case A_ID:
    case A_CNST:
      sptr = A_SPTRG(a);
      dtype = DTYPEG(sptr);
      switch (DTY(dtype)) {
      case TY_BINT:
      case TY_SINT:
      case TY_INT:
      case TY_BLOG:
      case TY_SLOG:
      case TY_LOG:
      case TY_FLOAT:
        dinit_put(dtype, CONVAL2G(sptr));
        break;
      case TY_DBLE:
      case TY_CMPLX:
      case TY_DCMPLX:
      case TY_QUAD:
      case TY_QCMPLX:
      case TY_INT8:
      case TY_LOG8:
      case TY_CHAR:
        dinit_put(dtype, sptr);
        break;
      }
      break;
    }
  }
} /* putzero */

/*
 * for an EOSHIFT call with an omitted boundary value,
 * use zero.  This functions returns an AST referencing
 * an appropriate 'zero' value for the given array datatype.
 */
static int
makezero(DTYPE dtype)
{
  int sub, sptr;
  switch (DTY(dtype)) {
  default:
    return _makezero(dtype);

  case TY_ARRAY:
    /* make an array of zeros */
    sub = _makezero(dtype);
    sptr = get_next_sym("init", "r");
    STYPEP(sptr, ST_ARRAY);
    DTYPEP(sptr, dtype);
    SCP(sptr, SC_STATIC);
    DINITP(sptr, 1);
    SEQP(sptr, 1);
    PARAMP(sptr, 1);
    PARAMVALP(sptr, sub);
    dinit_put(DINIT_LOC, sptr);
    putzero(sub);
    dinit_put(DINIT_END, 0);
    return mk_id(sptr);

  case TY_STRUCT:
  case TY_UNION:
  case TY_DERIVED:
    /* make an array of zeros */
    sub = _makezero(dtype);
    sptr = get_next_sym("init", "r");
    STYPEP(sptr, ST_VAR);
    DTYPEP(sptr, dtype);
    SCP(sptr, SC_STATIC);
    DINITP(sptr, 1);
    SEQP(sptr, 1);
    PARAMP(sptr, 1);
    PARAMVALP(sptr, sub);
    /* dump out the values to the data initialization file */
    dinit_put(DINIT_LOC, sptr);
    if (DTY(dtype + 3)) {
      dinit_put(DINIT_TYPEDEF, DTY(dtype + 3));
    }
    putzero(sub);
    if (DTY(dtype + 3)) {
      dinit_put(DINIT_ENDTYPE, DTY(dtype + 3));
    }
    dinit_put(DINIT_END, 0);
    return mk_id(sptr);
  }
} /* makezero */

static void
inline_shifts(int func_ast, int func_args, int lhs)
{
  int srcarray;
  int s;

  int sptrsrc, sptrdest;
  int asdsrc, asddest;
  int count;
  int i;
  int args;

  sg.shiftcount = 0;
  srcarray = func_ast;
  args = func_args;
  /* find all nested cshift/eoshift calls */
  while (A_TYPEG(srcarray) == A_INTR) {
    if (A_OPTYPEG(srcarray) == I_CSHIFT) {
      /* cshift(array, shift, [dim]) */
      assert(sg.shiftcount < SHIFTMAX, "inline_shifts: too many nested shifts",
             func_ast, 3);
      srcarray = ARGT_ARG(args, 0);
      s = sg.shiftcount;
      ss[s].shift = ARGT_ARG(args, 1);
      ss[s].dim = ARGT_ARG(args, 2);
      ss[s].shifttype = I_CSHIFT;
    } else if (A_OPTYPEG(srcarray) == I_EOSHIFT) {
      /* eoshift(array, shift, [boundary, dim]); */
      assert(sg.shiftcount < SHIFTMAX, "inline_shifts: too many nested shifts",
             func_ast, 3);
      srcarray = ARGT_ARG(args, 0);
      s = sg.shiftcount;
      ss[s].shift = ARGT_ARG(args, 1);
      ss[s].boundary = ARGT_ARG(args, 2);
      if (ss[s].boundary == 0) {
        /* must create a 'zero' */
        if (DTY(A_DTYPEG(srcarray)) == TY_ARRAY) {
          ss[s].boundary = makezero(DTY(A_DTYPEG(srcarray) + 1));
        } else {
          ss[s].boundary = makezero(A_DTYPEG(srcarray));
        }
      }
      ss[s].dim = ARGT_ARG(args, 3);
      ss[s].shifttype = I_EOSHIFT;
    } else {
      interr("inline_shifts: must be CSHIFT or EOSHIFT", srcarray, 3);
    }
    if (ss[s].dim == 0)
      ss[s].dim = mk_cval(1, astb.bnd.dtype);
    assert(A_TYPEG(ss[s].dim) == A_CNST,
           "inline_shifts: variable dimension not implemented", srcarray, 3);
    ss[s].cdim = get_int_cval(A_SPTRG(A_ALIASG(ss[s].dim)));
    ++sg.shiftcount;
    args = A_ARGSG(srcarray);
  }
  assert(lhs, "inline_shifts: no lhs", func_ast, 3);
  assert(A_TYPEG(lhs) == A_ID || A_TYPEG(lhs) == A_SUBSCR ||
             A_TYPEG(lhs) == A_MEM,
         "inline_shifts: bad lhs type", func_ast, 3);
  assert(A_TYPEG(srcarray) == A_ID || A_TYPEG(srcarray) == A_SUBSCR ||
             A_TYPEG(srcarray) == A_MEM,
         "inline_shifts: bad source type", func_ast, 3);

  sg.src = convert_subscript(srcarray);
  sg.dest = convert_subscript(lhs);
  sptrsrc = memsym_of_ast(sg.src);
  sptrdest = memsym_of_ast(sg.dest);

  asdsrc = A_ASDG(sg.src);
  sg.ndimsrc = ASD_NDIM(asdsrc);
  for (s = 0; s < sg.shiftcount; ++s) {
    if (ss[s].cdim > sg.ndimsrc || (ss[s].cdim < 1 || ss[s].cdim > 7)) {
      error(504, 3, gbl.lineno, SYMNAME(sptrsrc), CNULL);
      ss[s].cdim = 1;
    }
  }
  count = 0;
  for (i = 0; i < sg.ndimsrc; ++i) {
    if (A_TYPEG(ASD_SUBS(asdsrc, i)) == A_TRIPLE ||
        A_SHAPEG(ASD_SUBS(asdsrc, i))) {
      ++count;
      for (s = 0; s < sg.shiftcount; ++s) {
        if (count == ss[s].cdim) {
          ss[s].dim_src = i;
          break;
        }
      }
    }
  }

  asddest = A_ASDG(sg.dest);
  sg.ndimdest = ASD_NDIM(asddest);
  count = 0;
  for (i = 0; i < sg.ndimdest; ++i) {
    if (A_TYPEG(ASD_SUBS(asddest, i)) == A_TRIPLE ||
        A_SHAPEG(ASD_SUBS(asddest, i))) {
      ++count;
      for (s = 0; s < sg.shiftcount; ++s) {
        if (count == ss[s].cdim) {
          ss[s].dim_dest = i;
          break;
        }
      }
    }
  }

  /* Determine the section extent */
  for (s = 0; s < sg.shiftcount; ++s) {
    ss[s].n = triplet_extent(ASD_SUBS(asdsrc, ss[s].dim_src));
    if (A_TYPEG(ss[s].n) != A_CNST) {
      int tmp, ast;
      tmp = sym_get_scalar("n", "s", astb.bnd.dtype);
      ast = mk_assn_stmt(mk_id(tmp), ss[s].n, astb.bnd.dtype);
      add_stmt_before(ast, arg_gbl.std);
      ss[s].n = mk_id(tmp);
    } else {
      ss[s].nc = get_int_cval(A_SPTRG(A_ALIASG(ss[s].n)));
    }

    /*    Determine the net positive shift amount for CSHIFT
     *    m = MOD(k, n)
     *    if (m .lt. 0) then
     *         m = n + m
     *    endif
     */

    ss[s].k = ss[s].shift;
    if (A_TYPEG(ss[s].k) == A_CNST && A_TYPEG(ss[s].n) == A_CNST) {
      int result;
      ss[s].kc = get_int_cval(A_SPTRG(A_ALIASG(ss[s].k)));
      result = ss[s].kc % ss[s].nc;
      ss[s].m = mk_cval(result, astb.bnd.dtype);
    } else {
      int mod, tmp, ast;
      mod = ast_intr(I_MOD, DT_INT, 2, ss[s].k, ss[s].n);
      tmp = sym_get_scalar("m", "s", astb.bnd.dtype);
      ss[s].m = mk_id(tmp);
      ast = mk_assn_stmt(ss[s].m, mod, astb.bnd.dtype);
      add_stmt_before(ast, arg_gbl.std);
    }
    ss[s].lt = TRUE;
    if (A_TYPEG(ss[s].m) == A_CNST) {
      ss[s].mc = get_int_cval(A_SPTRG(A_ALIASG(ss[s].m)));
      if (ss[s].mc >= 0) {
        ss[s].lt = FALSE;
      } else if (ss[s].shifttype == I_CSHIFT) {
        if (A_TYPEG(ss[s].n) == A_CNST) {
          ss[s].mc = ss[s].mc + ss[s].nc;
          ss[s].m = mk_cval(ss[s].mc, astb.bnd.dtype);
          ss[s].lt = FALSE;
        } else {
          int ast, tmp;
          ast = opt_binop(OP_ADD, ss[s].m, ss[s].n, astb.bnd.dtype);
          tmp = sym_get_scalar("m", "s", astb.bnd.dtype);
          ss[s].m = mk_id(tmp);
          ast = mk_assn_stmt(ss[s].m, ast, astb.bnd.dtype);
          add_stmt_before(ast, arg_gbl.std);
        }
      }
    }

    if (ss[s].lt && ss[s].shifttype == I_CSHIFT) {
      int ast, expr;
      ast = mk_stmt(A_IFTHEN, 0);
      expr = mk_binop(OP_LT, ss[s].m, astb.bnd.zero, DT_LOG);
      A_IFEXPRP(ast, expr);
      add_stmt_before(ast, arg_gbl.std);
      ast = mk_assn_stmt(ss[s].m,
                         opt_binop(OP_ADD, ss[s].n, ss[s].m, astb.bnd.dtype),
                         astb.bnd.dtype);
      add_stmt_before(ast, arg_gbl.std);
      ast = mk_stmt(A_ENDIF, 0);
      add_stmt_before(ast, arg_gbl.std);
    }

    ss[s].then_part = FALSE;
    ss[s].else_part = FALSE;
    if (A_TYPEG(ss[s].m) != A_CNST) {
      ss[s].then_part = TRUE;
      ss[s].else_part = TRUE;
    } else if (ss[s].mc != 0) {
      ss[s].then_part = TRUE;
    } else {
      ss[s].else_part = TRUE;
    }
  }
  for (i = 0; i < sg.ndimdest; ++i) {
    sg.subsdest[i] = ASD_SUBS(asddest, i);
  }
  for (i = 0; i < sg.ndimsrc; ++i) {
    sg.subssrc[i] = ASD_SUBS(asdsrc, i);
  }

  handle_shift(0);

} /* inline_shifts */

/*   Determine the section extent
 *   n = (ub - lb + sb) / sb
 */
static int
triplet_extent(int t)
{
  int lb, ub, sb;
  int tmp1, tmp2, tmp3;

  assert(A_TYPEG(t) == A_TRIPLE, "triplet_extent: should be triplet", t, 3);
  lb = A_LBDG(t);
  ub = A_UPBDG(t);
  sb = A_STRIDEG(t);
  tmp1 = opt_binop(OP_SUB, ub, lb, astb.bnd.dtype);
  tmp2 = opt_binop(OP_ADD, tmp1, sb, astb.bnd.dtype);
  tmp3 = opt_binop(OP_DIV, tmp2, sb, astb.bnd.dtype);
  return tmp3;
}

static LOGICAL
is_inline_overlap_shifts(int func_ast, int func_args, int lhs)
{
  return FALSE;
}

LOGICAL
is_shift_conflict(int func_ast, int func_args, int expr)
{
  int srcarray;
  int boundary;
  int sptr;

  srcarray = ARGT_ARG(func_args, 0);
  sptr = memsym_of_ast(srcarray);
  boundary = -1;
  if (A_OPTYPEG(func_ast) == I_EOSHIFT)
    boundary = ARGT_ARG(func_args, 2);
  if (A_OPTYPEG(func_ast) == I_CSHIFT)
    if (expr && is_another_shift(expr, sptr, I_EOSHIFT, boundary))
      return TRUE;
  if (A_OPTYPEG(func_ast) == I_EOSHIFT) {
    if (expr && is_another_shift(expr, sptr, I_CSHIFT, boundary))
      return TRUE;
    if (expr && is_another_shift(expr, sptr, I_EOSHIFT, boundary))
      return TRUE;
  }
  return FALSE;
}

static struct {
  int sptr;
  int type;
  int boundary;
} expp;

static LOGICAL
is_another_shift(int expr, int sptr, int type, int boundary)
{
  LOGICAL result = FALSE;

  expp.sptr = sptr;
  expp.type = type;
  expp.boundary = boundary;
  ast_visit(1, 1);
  ast_traverse(expr, _is_another_shift, NULL, &result);
  ast_unvisit();
  return result;
}

static LOGICAL
_is_another_shift(int targast, LOGICAL *pflag)
{
  int boundary;
  int sptr;
  int type;
  int srcarray;
  int args;
  int check_boundary;

  if (A_TYPEG(targast) == A_INTR) {
    if (A_OPTYPEG(targast) == I_CSHIFT || A_OPTYPEG(targast) == I_EOSHIFT) {
      type = A_OPTYPEG(targast);
      args = A_ARGSG(targast);
      srcarray = ARGT_ARG(args, 0);
      boundary = 0;
      if (type == I_EOSHIFT)
        boundary = ARGT_ARG(args, 2);
      sptr = 0;
      switch (A_TYPEG(srcarray)) {
      case A_ID:
      case A_SUBSCR:
        sptr = memsym_of_ast(srcarray);
        break;
      }
      check_boundary = 1;
      if (expp.boundary != -1)
        if (expp.boundary == boundary)
          check_boundary = 0;
      if (expp.sptr == sptr && expp.type == type && check_boundary) {
        *pflag = TRUE;
        return TRUE;
      }
    }
  }
  return FALSE;
}

static int
stride_one(int lw, int up)
{
  if (A_TYPEG(lw) == A_CNST && A_TYPEG(up) == A_CNST &&
      ad_val_of(A_SPTRG(lw)) > ad_val_of(A_SPTRG(up)))
    return mk_isz_cval((ISZ_T)-1, astb.bnd.dtype);
  return astb.bnd.one;
}

int
convert_subscript(int a)
{
  ADSC *ad;
  int sptr, parent, member;
  int ndim;
  int lb, ub, st;
  int i;
  int subs[MAXSUBS];
  int asd;

  if (A_TYPEG(a) == A_ID) {
    sptr = A_SPTRG(a);
    if (!is_array_type(sptr))
      return a;
    ad = AD_DPTR(DTYPEG(sptr));
    ndim = AD_NUMDIM(ad);
    for (i = 0; i < ndim; i++) {
      subs[i] = mk_triple(AD_LWAST(ad, i), AD_UPAST(ad, i),
                          stride_one(AD_LWAST(ad, i), AD_UPAST(ad, i)));
    }
    return mk_subscr(mk_id(sptr), subs, ndim, A_DTYPEG(a));
  }

  if (A_TYPEG(a) == A_MEM) {
    /* do the parent first */
    parent = convert_subscript(A_PARENTG(a));
    member = A_MEMG(a);
    a = mk_member(parent, member, A_DTYPEG(member));
    sptr = A_SPTRG(member);
    if (!is_array_type(sptr))
      return a;
    ad = AD_DPTR(DTYPEG(sptr));
    ndim = AD_NUMDIM(ad);
    for (i = 0; i < ndim; i++) {
      subs[i] = mk_triple(check_member(a, AD_LWAST(ad, i)),
                          check_member(a, AD_UPAST(ad, i)), astb.bnd.one);
    }
    return mk_subscr(a, subs, ndim, A_DTYPEG(a));
  }

  if (A_TYPEG(a) == A_SUBSCR) {
    int lop, anytriple;
    sptr = sptr_of_subscript(a);
    assert(is_array_type(sptr), "convert_subscript: must be array", 2, a);
    lop = A_LOPG(a);
    ad = AD_DPTR(DTYPEG(sptr));
    asd = A_ASDG(a);
    ndim = ASD_NDIM(asd);
    anytriple = 0;
    for (i = 0; i < ndim; i++) {
      subs[i] = ASD_SUBS(asd, i);
      if (A_TYPEG(subs[i]) == A_TRIPLE) {
        anytriple = 1;
        lb = A_LBDG(subs[i]);
        if (!lb)
          lb = AD_LWAST(ad, i);
        ub = A_UPBDG(subs[i]);
        if (!ub)
          ub = AD_UPAST(ad, i);
        st = A_STRIDEG(subs[i]);
        if (!st)
          st = astb.bnd.one;
        subs[i] = mk_triple(lb, ub, st);
      }
    }
    /* was the triplet at this level? */
    if (anytriple)
      return mk_subscr(lop, subs, ndim, A_DTYPEG(a));

    if (A_TYPEG(lop) == A_MEM) {
      parent = convert_subscript(A_PARENTG(lop));
      member = A_MEMG(lop);
      lop = mk_member(parent, member, A_DTYPEG(member));
    }
    return mk_subscr(lop, subs, ndim, A_DTYPEG(a));
  }
  assert(0, "convert_subscript: it must be array", 0, a);
  return a;
}

static int
convert_subscript_in_expr(int expr)
{
  int l, r, d, o;
  int i, nargs, argt;
  int newargt;

  if (expr == 0)
    return expr;
  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = convert_subscript_in_expr(A_LOPG(expr));
    r = convert_subscript_in_expr(A_ROPG(expr));
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = convert_subscript_in_expr(A_LOPG(expr));
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(expr);
    l = convert_subscript_in_expr(A_LOPG(expr));
    if (DT_ISSCALAR(A_DTYPEG(l)) && DTY(d) == TY_ARRAY) {
      return mk_promote_scalar(l, d, A_SHAPEG(expr));
    } else {
      return mk_convert(l, d);
    }
  case A_PAREN:
    d = A_DTYPEG(expr);
    l = convert_subscript_in_expr(A_LOPG(expr));
    return mk_paren(l, d);
  case A_SUBSTR:
    d = A_DTYPEG(expr);
    o = convert_subscript_in_expr(A_LOPG(expr));
    l = convert_subscript_in_expr(A_LEFTG(expr));
    r = convert_subscript_in_expr(A_RIGHTG(expr));
    return mk_substr(o, l, r, d);
  case A_INTR:
    /* some intrinsic calls get shared trees, so make new tree */
    /* leave present alone */
    if (A_OPTYPEG(expr) == I_PRESENT)
      return expr;
    nargs = A_ARGCNTG(expr);
    newargt = mk_argt(nargs);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(newargt, i) = convert_subscript_in_expr(ARGT_ARG(argt, i));
    }
    l = mk_func_node(A_INTR, A_LOPG(expr), nargs, newargt);
    A_DTYPEP(l, A_DTYPEG(expr));
    A_OPTYPEP(l, A_OPTYPEG(expr));
    A_SHAPEP(l, A_SHAPEG(expr));
    return l;
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = convert_subscript_in_expr(ARGT_ARG(argt, i));
    }
    return expr;
  case A_CNST:
  case A_CMPLXC:
    return expr;
  case A_MEM:
  case A_ID:
  case A_SUBSCR:
    if (!A_SHAPEG(expr))
      return expr;
    expr = convert_subscript(expr);
    return expr;
  default:
    interr("convert_subscript_in_expr: unknown expression", expr, 2);
    return expr;
  }
}

static LOGICAL
stride1_triple(int triple)
{
#if DEBUG
  assert(A_TYPEG(triple) == A_TRIPLE, "stride1_triple: not A_TRIPLE", triple,
         4);
#endif
  if (A_STRIDEG(triple) && A_STRIDEG(triple) != astb.i1 &&
      A_STRIDEG(triple) != astb.bnd.one)
    return FALSE;
  return TRUE;
}

LOGICAL
contiguous_section(int arr_ast)
{
  int asd;
  int ndims, dim;
  int astsub;
  int sptr;
  int ast1;
  LOGICAL nonfull = FALSE;

  /* only for data references */
  if (A_TYPEG(arr_ast) != A_ID && A_TYPEG(arr_ast) != A_SUBSCR &&
      A_TYPEG(arr_ast) != A_MEM)
    return FALSE;

  for (ast1 = arr_ast; A_TYPEG(ast1) == A_MEM || A_TYPEG(ast1) == A_SUBSCR;
       ast1 = A_PARENTG(ast1)) {
    if (!A_SHAPEG(ast1))
      return TRUE; /* everything is contiguous so far and no more subscripting
                    */
    if (A_TYPEG(ast1) == A_MEM) {
      /* must be the first and only member */
      sptr = A_SPTRG(A_MEMG(ast1));
      if (ADDRESSG(sptr) != 0 || SYMLKG(sptr) != NOSYM)
        return FALSE;
    } else if (A_TYPEG(ast1) == A_SUBSCR) {
      /* must be contiguous subscripting */
      asd = A_ASDG(ast1);
      ndims = ASD_NDIM(asd);
      /* Find the 1st non-scalar dimension. */
      for (dim = ndims - 1; dim >= 0; --dim) {
        int ss = ASD_SUBS(asd, dim);
        if (A_TYPEG(ss) == A_TRIPLE)
          break;
        if (A_SHAPEG(ss))
          return FALSE; /* non-triplet shaped subscript */
      }
      if (dim < 0) {
        nonfull = TRUE; /* all parent subscripts must be scalar as well */
      } else if (nonfull) {
        return FALSE; /* already had a deeper non-full dimension */
      } else {
        astsub = ASD_SUBS(asd, dim);
        sptr = memsym_of_ast(ast1);
        if (!stride1_triple(astsub))
          return FALSE; /* not-stride-1 */
        if (!is_whole_dim(ast1, dim))
          nonfull = TRUE;
        /* Leading dimensions must be full. */
        for (--dim; dim >= 0; --dim) {
          if (!is_whole_dim(ast1, dim))
            return FALSE;
        }
      }
    }
  }
  if (A_TYPEG(ast1) != A_ID)
    return FALSE;
  return TRUE;
}

/* Check if array section is contiguous, does not have to be whole array */
static LOGICAL
contiguous_section_array(int arr_ast)
{
  int asd, ss;
  int ndims, dim;
  int ast1 = A_TYPEG(arr_ast) == A_MEM ? A_MEMG(arr_ast) : arr_ast;

  if (!ast1)
    return FALSE;

  if (!A_SHAPEG(ast1) || A_TYPEG(ast1) == A_ID)
    return TRUE;
  asd = A_ASDG(ast1);
  ndims = ASD_NDIM(asd);
  for (dim = ndims - 1; dim >= 0; dim--) {
    ss = ASD_SUBS(asd, dim);
    if (A_TYPEG(ss) == A_TRIPLE) {
      continue;
    }
    if (A_TYPEG(ss) == A_SUBSCR) {
      if (!is_whole_dim(arr_ast, dim))
        return FALSE;
    }
    if (A_TYPEG(ss) == A_ID && (DTY(A_DTYPEG(ss))) == TY_ARRAY) {
      if (!is_whole_dim(arr_ast, dim))
        return FALSE;
    }
  }
  return TRUE;
}

static int
extract_shape_from_args(int func_ast)
{
  int funcsptr, iface;
  int dscptr;
  int dummy_sptr;
  int shape = A_SHAPEG(func_ast);
  int arg_shape;
  int argt;
  int nargs;
  int i;

  funcsptr = procsym_of_ast(A_LOPG(func_ast));
  proc_arginfo(funcsptr, NULL, &dscptr, &iface);
  nargs = A_ARGCNTG(func_ast);
  argt = A_ARGSG(func_ast);
  for (i = 0; i < nargs; ++i) {
    if (dscptr) {
      dummy_sptr = aux.dpdsc_base[dscptr + i];
      if (ARGT_ARG(argt, i) == astb.ptr0 && OPTARGG(dummy_sptr)) {
        continue;
      }
    }
    arg_shape = A_SHAPEG(ARGT_ARG(argt, i));
    /* scalars are always conformable */
    if (arg_shape) {
      if (shape) {
        if (!conform_shape(arg_shape, shape) &&
            ((iface && FVALG(iface)) || !dummy_sptr ||
             INTENTG(dummy_sptr) != INTENT_IN)) {
          error(508, 3, gbl.lineno, SYMNAME(funcsptr), CNULL);
          break;
        }
      } else {
        shape = arg_shape;
      }
    }
  }
  return shape;
}

static int alloc_char_temp(int, const char *, int, int, int);
static int get_charintrin_temp(int, const char *);

static struct {
  int continue_std, func_std;
} difficult = {0, 0};

void
check_pointer_type(int past, int tast, int stmt, LOGICAL is_sourced_allocation)
{
  /* For type pointers, we want to set the type field of its
   * descriptor to whatever type we're assigning it to. Used for
   * polymorphic entities. The flag argument is set when we call this
   * function due to a sourced allocation.
   */

  int psptr, tsptr, dt1, dt2, desc1, type2;
  int astnew, is_inline, intrin_type;
  static int tmp = 0;
  int nullptr;
  bool isNullAssn = false;

  if (DT_PTR == DT_INT8)
    nullptr = astb.k0;
  else
    nullptr = astb.i0;
  if (A_TYPEG(tast) == A_SUBSCR)
    tast = A_LOPG(tast);

  dt1 = A_DTYPEG(past);
  if (DTY(dt1) == TY_ARRAY) {
    dt1 = DTY(dt1 + 1);
  }
  dt2 = A_DTYPEG(tast);
  if (DTY(dt2) == TY_ARRAY) {
    dt2 = DTY(dt2 + 1);
  }

  if (DTY(dt1) != TY_DERIVED) {
    return;
  }

  if (DTY(dt2) != TY_DERIVED) {
    if (!UNLPOLYG(DTY(dt1 + 3))) {
      return;
    }
    intrin_type = 1;
  } else {
    intrin_type = 0;
  }

  psptr = memsym_of_ast(past);

  if (!CLASSG(psptr)) {
    return;
  }

  switch (A_TYPEG(tast)) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    tsptr = memsym_of_ast(tast);
    break;
  case A_INTR:
    if (A_OPTYPEG(tast) == I_NULL) {
      tsptr = psptr;
      isNullAssn = true;
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    return;
  }

  if (ALLOCDESCG(psptr)) {
    desc1 = DESCRG(psptr);
    DESCUSEDP(psptr, TRUE);
    if (!desc1 || SDSCG(psptr)) {
      desc1 = SDSCG(psptr);
    }
    if (!intrin_type) {
      if (CLASSG(tsptr) || (is_sourced_allocation && ALLOCATTRG(tsptr))) {
        type2 = get_type_descr_arg(gbl.currsub, tsptr);
      } else {
        type2 = getccsym('P', tmp++, ST_VAR);
        DTYPEP(type2, dt2);
        type2 = get_static_type_descriptor(type2);
      }
    } else {
      type2 = dtype_to_arg(dt2);
      type2 = mk_cval1(type2, DT_INT);
      type2 = mk_unop(OP_VAL, type2, DT_INT);
    }

    /*
     *  Beware!  If intrin_type is TRUE, 'type2' is the index of an AST (that
     *  corresponds to the code number of the intrinsic type).  But if it's
     *  false, 'type2' is a symbol table pointer (to a descriptor).
     */
    if (desc1 && type2 && !XBIT(68, 0x4)) {

      if (isNullAssn) {
        int src_ast, astnew;
        if (intrin_type) {
          src_ast = type2;
        } else {
          type2 = getccsym('P', tmp++, ST_VAR);
          DTYPEP(type2, dt2);
          type2 = get_static_type_descriptor(type2);
          src_ast = mk_id(type2);
        }
        if (STYPEG(psptr) != ST_MEMBER) {
          astnew = mk_set_type_call(mk_id(desc1), src_ast, intrin_type);
        } else {
          int sdsc_mem = get_member_descriptor(psptr);
          int dest_ast = check_member(past, mk_id(sdsc_mem));
          astnew = mk_set_type_call(dest_ast, src_ast, intrin_type);
        }
        add_stmt_after(astnew, stmt);
        return;
      }

      if (STYPEG(psptr) != ST_MEMBER &&
          (STYPEG(tsptr) != ST_MEMBER || !CLASSG(tsptr))) {
        is_inline = (!intrin_type)
                        ? inline_RTE_set_type(desc1, type2, stmt, 1, dt2, 0)
                        : 0;
        if (!is_inline) {
          int dest_ast = mk_id(desc1);
          int src_ast =
              intrin_type ? type2 : check_member(dest_ast, mk_id(type2));

          gen_set_type(dest_ast, src_ast, stmt, FALSE, intrin_type);
        }
      } else if ((STYPEG(psptr) == ST_MEMBER && (STYPEG(tsptr) != ST_MEMBER)) ||
                 !CLASSG(tsptr)) {
        int sdsc_mem = get_member_descriptor(psptr);
        assert(sdsc_mem > NOSYM, "no descriptor for member", psptr, 3);
        is_inline = 0; /* TBD: inline_RTE_set_type( ) */
        if (!is_inline) {
          int nz_ast, if_ast, ptr_ast;
          int dest_ast = check_member(past, mk_id(sdsc_mem));
          int src_ast =
              intrin_type ? type2 : check_member(dest_ast, mk_id(type2));
          astnew = mk_set_type_call(dest_ast, src_ast, intrin_type);
          ptr_ast = mk_unop(OP_LOC, A_PARENTG(past), DT_PTR);
          nz_ast = mk_binop(OP_NE, ptr_ast, nullptr, DT_LOG);
          if_ast = mk_stmt(A_IF, 0);
          A_IFEXPRP(if_ast, nz_ast);
          A_IFSTMTP(if_ast, astnew);
          /* Use add_stmt_after() instead of add_stmt_before() below.
           * This appears to be the right thing to do in the event that you
           * have something like recordPtr%next => recordPtr2.
           * We want to access next's descriptor (embedded in recordPtr),
           * but we have to do it before we assign/change recordPtr%next
           * address.
           */
          add_stmt_before(if_ast, stmt);
        }
      } else if (STYPEG(psptr) != ST_MEMBER && STYPEG(tsptr) == ST_MEMBER) {
        int sdsc_mem = get_member_descriptor(tsptr);
        assert(sdsc_mem > NOSYM, "no descriptor for member", tsptr, 3);
        is_inline = 0; /* TBD: inline_RTE_set_type( ) */
        if (!is_inline) {
          int nz_ast, if_ast, ptr_ast;
          int dest_ast = mk_id(desc1);
          int src_ast =
              intrin_type
                  ? type2
                  : mk_member(A_PARENTG(tast), mk_id(sdsc_mem), A_DTYPEG(tast));
          astnew = mk_set_type_call(dest_ast, src_ast, intrin_type);

          /* if (tast .ne. 0) */

          ptr_ast = mk_unop(OP_LOC, A_PARENTG(tast), DT_PTR);
          nz_ast = mk_binop(OP_NE, ptr_ast, nullptr, DT_LOG);
          if_ast = mk_stmt(A_IF, 0);
          A_IFEXPRP(if_ast, nz_ast);
          A_IFSTMTP(if_ast, astnew);
          /* Use add_stmt_after() instead of add_stmt_before() below.
           * This appears to be the right thing to do in the event that you
           * have something like recordPtr => recordPtr%next. We want to
           * access next's descriptor (embedded in recordPtr), but we have to
           * do it before we assign/change recordPtr's address.
           */
          add_stmt_before(if_ast, stmt);
        }
      } else {
        int sdsc_mem = get_member_descriptor(tsptr);
        int sdsc_mem2 = get_member_descriptor(psptr);
        assert(sdsc_mem > NOSYM, "no descriptor for member", tsptr, 3);
        assert(sdsc_mem2 > NOSYM, "no descriptor for member", psptr, 3);
        is_inline = 0; /* TBD: inline_RTE_set_type( ) */
        if (!is_inline) {
          int nz_ast, if_ast, ptr_ast;
          int dest_ast =
              mk_member(A_PARENTG(past), mk_id(sdsc_mem2), A_DTYPEG(past));
          int src_ast =
              intrin_type
                  ? type2
                  : mk_member(A_PARENTG(tast), mk_id(sdsc_mem), A_DTYPEG(tast));
          astnew = mk_set_type_call(dest_ast, src_ast, intrin_type);

          /* if (tast .ne. 0) */
          ptr_ast = mk_unop(OP_LOC, A_PARENTG(tast), DT_PTR);
          nz_ast = mk_binop(OP_NE, ptr_ast, nullptr, DT_LOG);
          if_ast = mk_stmt(A_IF, 0);
          A_IFEXPRP(if_ast, nz_ast);
          A_IFSTMTP(if_ast, astnew);
          /* Use add_stmt_after() instead of add_stmt_before() below.
           * This appears to be the right thing to do in the event that you
           * have something like recordPtr%next => recordPtr%next%next.
           * We want to access next's descriptor (embedded in recordPtr),
           * but we have to do it before we assign/change recordPtr%next
           * address.
           */
          add_stmt_before(if_ast, stmt);
        }
      }
    }
  }

  if (!is_sourced_allocation && POINTERG(psptr) && UNLPOLYG(DTY(dt1 + 3)) &&
      UNLPOLYG(DTY(dt2 + 3)) && SDSCG(psptr) && SDSCG(tsptr)) {
    /* init unlimited polymorphic descriptor for pointer.
     * We do not have to do this for the sourced allocation case since
     * the sourced allocation case is handled in semant3.c with the
     * ALLOCATE productions.
     */
    int psdsc, tsdsc, dest_sdsc_ast, src_sdsc_ast;
    int fsptr, argt, val, ast;
    if (STYPEG(psptr) == ST_MEMBER) {
      psdsc = get_member_descriptor(psptr);
    } else {
      psdsc = SDSCG(psptr);
    }
    assert(psdsc > NOSYM, "no descriptor for psptr", psptr, 3);
    if (STYPEG(tsptr) == ST_MEMBER) {
      tsdsc = get_member_descriptor(tsptr);
    } else if (SCG(tsptr) == SC_DUMMY) {
      tsdsc = get_type_descr_arg(gbl.currsub, tsptr);
    } else {
      tsdsc = SDSCG(tsptr);
    }
    assert(tsdsc > NOSYM, "no descriptor for tsptr", tsptr, 3);
    fsptr = sym_mkfunc_nodesc(mkRteRtnNm(RTE_init_unl_poly_desc), DT_NONE);
    dest_sdsc_ast = check_member(past, mk_id(psdsc));
    src_sdsc_ast = check_member(tast, mk_id(tsdsc));

    argt = mk_argt(3);
    ARGT_ARG(argt, 0) = dest_sdsc_ast;
    ARGT_ARG(argt, 1) = src_sdsc_ast;
    val = mk_cval1(43, DT_INT);
    val = mk_unop(OP_VAL, val, DT_INT);
    ARGT_ARG(argt, 2) = val;
    ast = mk_id(fsptr);
    ast = mk_func_node(A_CALL, ast, 3, argt);
    add_stmt_after(ast, stmt);
  }
}

/* Given one of the arguments to move_alloc (either from or to), return the
 * corresponding symbol and pointer to the arg. */
static void
move_alloc_arg(int arg, SPTR *sptr, int *pvar)
{
  if (A_TYPEG(arg) == A_ID)
    *sptr = A_SPTRG(arg);
  else if (A_TYPEG(arg) == A_MEM)
    *sptr = A_SPTRG(A_MEMG(arg));
  else
    *sptr = 0;

  if (MIDNUMG(*sptr)) {
    *pvar = check_member(arg, mk_id(MIDNUMG(*sptr)));
  } else if (!ALLOCATTRG(*sptr)) {
    error(507, ERR_Fatal, gbl.lineno, SYMNAME(*sptr), 0);
  } else {
    *pvar = mk_unop(OP_LOC, mk_id(*sptr), DT_PTR);
  }
}

void
check_alloc_ptr_type(int psptr, int stmt, DTYPE dt1, int flag, LOGICAL after,
                     int past, int astmem)
{
  /* For allocatable/pointer objects, we assign a type to its dynamic type.
   * The psptr is the sptr of the allocatable/pointer object.
   * The stmt arg is the current statement to insert the type assign.
   * The typespec is the dynamic type. If it's 0, we assign the object's
   * declared type to its dynamic type.
   * The flag arg is set when we want to assign type to psptr's descriptor. It's
   * also set to 2 when psptr is used as an actual arg passed to a unlimited
   * polymorphic argument.
   * If flag is not set, then we just want to reserve space for type in
   * psptr's descriptor.
   * The after flag is set when we want to insert the type assignment after
   * the current statement. If it's 0, then we insert it before current stmt.
   */

  LOGICAL intrin_type;
  LOGICAL no_alloc_ptr = FALSE;

  if (dt1 <= DT_NONE)
    dt1 = DTYPEG(psptr);
  if (is_array_dtype(dt1))
    dt1 = array_element_dtype(dt1);
  intrin_type = DTY(dt1) != TY_DERIVED;

  if (!ALLOCDESCG(psptr) && !is_array_dtype(DTYPEG(psptr))) {
    if (!SDSCG(psptr) || DTY(DTYPEG(psptr)) == TY_DERIVED) {
      set_descriptor_rank(TRUE);
      get_static_descriptor(psptr);
      set_descriptor_rank(FALSE);
      ALLOCDESCP(psptr, TRUE);
      no_alloc_ptr = TRUE;
    } else if (flag == 2 && (ALLOCATTRG(psptr) || POINTERG(psptr))) {
      /* allocatable or pointer actual and unlimited polymorphic dummy */
      set_descriptor_rank(TRUE);
      get_static_descriptor(psptr);
      set_descriptor_rank(FALSE);
      if (ALLOCATTRG(psptr))
        ALLOCDESCP(psptr, TRUE);
    }
  }

  if (intrin_type) {
    DTYPE dt2 = DTYPEG(psptr);
    if (is_array_dtype(dt2))
      dt2 = array_element_dtype(dt2);
    if (flag != 2 && (DTY(dt2) != TY_DERIVED || !UNLPOLYG(DTY(dt2 + 3)))) {
      /* ignore non-derived type and unlimited polymorphic objects
       * unless flag is set to 2.
       */
      flag = 0;
    }
    /* otherwise we are allocating an intrinsic type to an unlimited polymorphic
     * object */
  }

  if (flag != 0 && (ALLOCDESCG(psptr) || intrin_type)) {
    int desc1_sptr = 0;
    LOGICAL is_member = past && STYPEG(psptr) == ST_MEMBER &&
                        (CLASSG(psptr) || FINALIZEDG(psptr));
    if (is_member) {
      /* copy type into member type descriptor.*/
      desc1_sptr = get_member_descriptor(psptr);
    } else {
      desc1_sptr = SDSCG(psptr);
      if (!desc1_sptr)
        desc1_sptr = DESCRG(psptr);
      if (desc1_sptr)
        DESCUSEDP(psptr, TRUE);
    }
    if (desc1_sptr) {
      int type2_sptr = 0, type2_ast = 0;
      if (intrin_type) {
        type2_ast = mk_cval1(dtype_to_arg(dt1), DT_INT);
        type2_ast = mk_unop(OP_VAL, type2_ast, DT_INT);
      } else {
        static int tmp = 0;
        type2_sptr = getccsym('A', tmp++, ST_VAR);
        DTYPEP(type2_sptr, dt1);
        type2_sptr = get_static_type_descriptor(type2_sptr);
        if (type2_sptr > NOSYM)
          type2_ast = mk_id(type2_sptr);
      }
      if (is_member ||
          (type2_ast && !XBIT(68, 0x4) &&
           (intrin_type || !inline_RTE_set_type(desc1_sptr, type2_sptr, stmt,
                                                after, dt1, astmem)))) {
        int desc1_ast = get_desc_tag(desc1_sptr);
        int tagdesc = get_desc_tag(desc1_sptr);
        if (is_member) {
          desc1_ast = check_member(past, desc1_ast);
          tagdesc = check_member(past, tagdesc);

        } else if (astmem) {
          desc1_ast = check_member(astmem, desc1_ast);
          tagdesc = check_member(astmem, tagdesc);
        }
        stmt = gen_set_type(desc1_ast, type2_ast, stmt, !after, intrin_type);
        if (no_alloc_ptr) {
          int tag = mk_isz_cval(intrin_type ? __TAGPOLY : __TAGDESC, DT_INT);
          int astnew = mk_assn_stmt(tagdesc, tag, 0);
          stmt = add_stmt_before(astnew, stmt);
        }
      }
    }
  }
}

/* if argument(s) is non-member allocatable, ALLOCDESC must be
 * set because RTE_sect2 can be called, then full descriptor must
 * be passed.  They can be arguments to other routine before matmul
 * and can be allocated in the subroutine.
 */
static void
check_arg_isalloc(int arg)
{
  int lop;
  int sptr = 0;
  if (A_TYPEG(arg) == A_SUBSCR) {
    lop = A_LOPG(arg);
    if (A_TYPEG(lop) == A_ID)
      sptr = A_SPTRG(lop);
  } else if (A_TYPEG(arg) == A_ID) {
    sptr = A_SPTRG(arg);
  }
  if (sptr && ALLOCATTRG(sptr)) {
    ALLOCDESCP(sptr, 1);
  }
}

static int forall_indx[MAXSUBS];

static LOGICAL
id_dep_in_forall_idxlist(int ast)
{
  int i;

  for (i = 0; forall_indx[i] && i < MAXSUBS; i++) {
    if (A_SPTRG(ast) == forall_indx[i]) {
      return TRUE;
    }
  }
  return 0;
}

static LOGICAL
_arg_forall_depnd(int ast, int *is_dep)
{
  if (A_TYPEG(ast) == A_ID) {
    *is_dep = id_dep_in_forall_idxlist(ast);
    return TRUE;
  }

  return FALSE;
}

static void
init_idx_list(int forall)
{
  int triplet_list;
  int i;

  for (i = 0; i < MAXSUBS; i++)
    forall_indx[i] = 0;

  triplet_list = A_LISTG(forall);
  for (i = 0; i < MAXSUBS && triplet_list;
       i++, triplet_list = ASTLI_NEXT(triplet_list)) {
    forall_indx[i] = ASTLI_SPTR(triplet_list);
  }
}

static LOGICAL
charintr_arg_forall_depnd(int ast_arg)
{
  LOGICAL is_dep = FALSE;
  int asd;
  int ndims;
  int i;

  if (A_TYPEG(ast_arg) != A_SUBSCR) {
    return FALSE;
  }

  init_idx_list(STD_AST(arg_gbl.std));

  asd = A_ASDG(ast_arg);
  ndims = ASD_NDIM(asd);
  for (i = 0; i < ndims && !is_dep; i++) {
    ast_visit(1, 1);
    ast_traverse(ASD_SUBS(asd, i), _arg_forall_depnd, NULL, &is_dep);
    ast_unvisit();
  }
  return is_dep;
}

/** \brief func_ast is a function or intrinsic call.  If it is a
    transformational intrinsic, create an appropriate temp, rewrite, and return
    a load of that temp.
    For now, don't do anything with user-defined functions.
    \param func_ast  A_INTR, A_FUNC, or A_ICALL
    \param func_args rewritten args for the function
    \param lhs ast for lhs (temp) if non-zero

    If lhs is non-zero, check lhs to see if it is OK for the intended
    use; if so, return 0.
 */
static int
rewrite_func_ast(int func_ast, int func_args, int lhs)
{
  int shape = A_SHAPEG(func_ast);
  DTYPE dtype = A_DTYPEG(func_ast);
  int dim, ndims, cdim;
  int shift;
  int newsym;
  int temp_arr = 0;
  int newargt;
  int srcarray;
  int rank;
  int retval = 0;
  int ast;
  int nargs;
  int mask;
  int value;
  LOGICAL back;
  int is_back_true;
  int vector;
  FtnRtlEnum rtlRtn;
  const char *root;
  int i;
  int subscr[MAXSUBS];
  int sptr;
  int astnew;
  int temp_sptr;
  LOGICAL is_icall; /* iff its first arg is changable */
  int ast_from_len = 0;
  int arg1;
  int dtnew;
  LOGICAL forall_depnd_intrin;
  const int type = A_TYPEG(func_ast);
  const int optype = A_OPTYPEG(func_ast);

  /* it only handles calls */
  if (type != A_INTR && type != A_FUNC && type != A_ICALL) {
    return func_ast;
  }
  if (type == A_FUNC) {
    if (elemental_func_call(func_ast)) {
      shape = extract_shape_from_args(func_ast);
    }
    goto ret_norm;
  }
  if (type == A_ICALL) {
    switch (optype) {
    case I_MOVE_ALLOC:
      transform_move_alloc(func_ast, func_args);
      return -1;
    case I_MVBITS:
      transform_mvbits(func_ast, func_args);
      return -1;
    case I_MERGE:
      transform_merge(func_ast, func_args);
      return -1;
    case I_NULLIFY:
      return -1;
#ifdef I_C_F_POINTER
    case I_C_F_POINTER:
      transform_c_f_pointer(func_ast, func_args);
      return -1;
#endif
#ifdef I_C_F_POINTER
    case I_C_F_PROCPOINTER:
      transform_c_f_procpointer(func_ast, func_args);
      return -1;
#endif
    }
  }
  if (type == A_INTR && optype == I_ASSOCIATED) {
    return transform_associated(arg_gbl.std, func_ast);
  }

  if (type == A_INTR) {
    switch (optype) {
    case I_ADJUSTL: /* adjustl(string) */
    case I_ADJUSTR: /* adjustr(string) */
      if (STYPEG(A_SPTRG(A_LOPG(func_ast))) == ST_PD)
        /* it's an IK_ELEMENTAL, but needs special processing */
        break;
      /*
       * ADJUSTL/ADJUSTR has been replaced, so this A_INTR
       * is just a function call
       */
      goto ret_norm;
    default:
      if (INKINDG(A_SPTRG(A_LOPG(func_ast))) == IK_ELEMENTAL)
        goto ret_norm;
    }
  }
  is_icall = TRUE;
  switch (optype) {
  case I_NUMBER_OF_PROCESSORS:
    retval = mk_id(sym_mknproc());
    A_DTYPEP(retval, DT_INT);
    A_SHAPEP(retval, 0);
    return retval;
  case I_ALL:   /* all(mask, [dim]) */
  case I_ANY:   /* any(mask, [dim]) */
  case I_COUNT: /* count(mask, [dim]) */
    srcarray = ARGT_ARG(func_args, 0);
    dim = ARGT_ARG(func_args, 1);

    /* check dim range if constant */
    cdim = -1;
    if (dim != 0 && A_TYPEG(dim) == A_CNST) {
      cdim = get_int_cval(A_SPTRG(A_ALIASG(dim)));
      if (A_SHAPEG(srcarray) &&
          ((int)SHD_NDIM(A_SHAPEG(srcarray)) < cdim || 1 > cdim))
        error(505, 3, gbl.lineno, SYMNAME(A_SPTRG(A_LOPG(func_ast))), CNULL);
    }

    if (shape == 0 && (dim == 0 || cdim != -1)) {
      /*E.g.,  pghpf_anys(result, mask) */
      rtlRtn =
          optype == I_ALL ? RTE_alls : optype == I_ANY ? RTE_anys : RTE_counts;
      nargs = 2;
    } else {
      /* E.g., pghpf_any(result, mask, dim) */
      rtlRtn =
          optype == I_ALL ? RTE_all : optype == I_ANY ? RTE_any : RTE_count;
      nargs = 3;
    }
    newargt = mk_argt(nargs);
    if (dim == 0) {
      dim = mk_cval(0, DT_INT);
    }
    ARGT_ARG(newargt, 1) = srcarray;
    if (nargs == 3) {
      ARGT_ARG(newargt, 2) = dim;
    }
    goto ret_new;
  case I_PRODUCT: /* product(array, [dim, mask]) */
  case I_SUM:     /* sum(array, [dim, mask]) */
    mask = ARGT_ARG(func_args, 2);

    srcarray = ARGT_ARG(func_args, 0);
    dim = ARGT_ARG(func_args, 1);

    /* check dim range if constant */
    cdim = -1;
    if (dim != 0 && A_TYPEG(dim) == A_CNST) {
      cdim = get_int_cval(A_SPTRG(A_ALIASG(dim)));
      if (A_SHAPEG(srcarray) &&
          ((int)SHD_NDIM(A_SHAPEG(srcarray)) < cdim || 1 > cdim))
        error(505, 3, gbl.lineno, SYMNAME(A_SPTRG(A_LOPG(func_ast))), CNULL);
      if (!XBIT(47, 0x80) && !XBIT(70, 0x1000000) && cdim == 1 && mask == 0) {
        /* Other than meeting the usual requirements, continue with
         * transforming the call if we inhibit inlining reductions
         * controlled by XBIT(47,0x80); otherwise, an ICE,
         * "rewrite_func_ast: bad dim for sum/prod" will occur
         * in an ensuing call
         */
        return func_ast;
      }
    }
    if (mask == 0) {
      mask = mk_cval(1, DT_LOG);
    }

    if (shape == 0 && (dim == 0 || cdim != -1)) {
      /* E.g,. pghpf_sums(result, array, mask) */
      rtlRtn = optype == I_PRODUCT ? RTE_products : RTE_sums;
      nargs = 3;
    } else {
      /* E.g., pghpf_sum(result, array, mask, dim) */
      rtlRtn = optype == I_PRODUCT ? RTE_product : RTE_sum;
      nargs = 4;
    }

    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    mask = misalignment(srcarray, mask, arg_gbl.std);
    ARGT_ARG(newargt, 2) = mask;
    if (nargs == 4) {
      assert(dim != 0, "rewrite_func_ast: bad dim for sum/prod", func_ast, 4);
      ARGT_ARG(newargt, 3) = dim;
    }
    goto ret_new;
  case I_EXECUTE_COMMAND_LINE:
    nargs = 7;
    rtlRtn = RTE_execcmdline;
    newsym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_INT);
    newargt = mk_argt(nargs);
    for (i = 0; i < nargs - 1; i++) {
      int arg = ARGT_ARG(func_args, i);
      ARGT_ARG(newargt, i) = arg != 0 ? arg : i == 0 ? astb.ptr0c : astb.ptr0;
    }
    /* Add two extra arguments at the end of the execute_command_line argument
       list. Those two integer kind for the exitstat and cmdstat argument
       respectively.
     */
    ARGT_ARG(newargt, nargs - 2) = mk_cval(size_of(stb.user.dt_int), DT_INT4);    
    ARGT_ARG(newargt, nargs - 1) = mk_cval(size_of(stb.user.dt_int), DT_INT4);    
    is_icall = FALSE;
    goto ret_call;
  case I_NORM2:     /* norm2(array, [dim]) */
    srcarray = ARGT_ARG(func_args, 0);
    dim = ARGT_ARG(func_args, 1);
    rank = get_ast_rank(srcarray);
    shape = dim ? A_SHAPEG(srcarray) : 0;

    // If dim is supplied for a one dimensional array, result is still a scalar.
    shape  = (shape && (rank == 1)) ? 0 : shape;

    if (dim == 0) {
      rtlRtn = RTE_norm2_nodim;
      nargs = 3;
    } else {
      rtlRtn = RTE_norm2;
      nargs = 4;
    }
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;

    if (!flg.ieee) { // fast. Currently also mapped to relaxed
      ARGT_ARG(newargt, 2) = mk_cval(1, DT_INT4);
    } else  { // Precise
      ARGT_ARG(newargt, 2) = mk_cval(2, DT_INT4);
    }

    if (nargs == 4) {
      ARGT_ARG(newargt, 3) = dim;      
    }
    goto ret_new;
  case I_MAXVAL: /* maxval(array, [dim, mask]) */
  case I_MINVAL: /* minval(array, [dim, mask]) */
    mask = ARGT_ARG(func_args, 2);
    srcarray = ARGT_ARG(func_args, 0);
    dim = ARGT_ARG(func_args, 1);

    if (mask == 0) {
      mask = mk_cval(1, DT_LOG);
    }
    mask = misalignment(srcarray, mask, arg_gbl.std);

    if (dim == 0) {
      rtlRtn = optype == I_MAXVAL ? RTE_maxvals : RTE_minvals;
      nargs = 3;
    } else {
      rtlRtn = optype == I_MAXVAL ? RTE_maxval : RTE_minval;
      nargs = 4;
    }
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = mask;
    if (nargs == 4) {
      ARGT_ARG(newargt, 3) = dim;
    }
    goto ret_new;
  case I_CSHIFT: /* cshift(array, shift, [dim]) */
    if (A_SHAPEG(ARGT_ARG(func_args, 1)))
      goto unch;
    dim = ARGT_ARG(func_args, 2);
    if (dim == 0)
      dim = mk_cval(1, DT_INT);
    if (A_TYPEG(dim) != A_CNST)
      goto unch;
    /* don't inline forall(i=1:n) a(i,:) = cshift(b(i,:)) */

    if (!arg_gbl.inforall &&
        is_inline_overlap_shifts(func_ast, func_args, lhs))
      goto ret_norm;
    if (!is_no_comm_shift(func_ast, func_args))
      goto unch;
    if (arg_gbl.inforall)
      goto unch;
    /* the following can inline cshift and eoshift
     * (without no_comm or comm restriction )
     * but it is restricted no_comm shift for performance reason only
     */

    assert(shape != 0, "expected non-zero shape", 0, ERR_Fatal);
    /* need to put this into a temp */
    temp_arr = mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1), lhs,
                              &retval);
    if (temp_arr != 0) {
      mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, 0);
      mk_mem_deallocate(mk_id(temp_arr), arg_gbl.std);
    }
    inline_shifts(func_ast, func_args, retval);
    return temp_arr == 0 && lhs != 0 ? 0 : retval;

  unch:
    srcarray = ARGT_ARG(func_args, 0);
    dim = ARGT_ARG(func_args, 2);
    if (dim == 0)
      dim = mk_cval(1, DT_INT);
    shift = ARGT_ARG(func_args, 1);
    nargs = 4;
    if (A_SHAPEG(shift) == 0) {
      shift = convert_int(shift, astb.bnd.dtype);
      rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_cshiftsca : RTE_cshifts;
    } else {
      rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_cshiftca : RTE_cshift;
    }
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = shift;
    ARGT_ARG(newargt, 3) = convert_int(dim, astb.bnd.dtype);
    goto ret_new;

  case I_DOT_PRODUCT: /* dot_product(vector_a, vector_b) */
    nargs = 3;
    rtlRtn = RTE_dotpr;
    newargt = mk_argt(nargs);
    srcarray = ARGT_ARG(func_args, 0);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = ARGT_ARG(func_args, 1);
    goto ret_new;
  case I_EOSHIFT: /* eoshift(array, shift, [boundary, dim]); */
    if (A_SHAPEG(ARGT_ARG(func_args, 1)))
      goto eoshiftcall; /* shift not a scalar */

    if (!arg_gbl.inforall &&
        is_inline_overlap_shifts(func_ast, func_args, lhs))
      goto ret_norm;

    if (!is_no_comm_shift(func_ast, func_args))
      goto eoshiftcall;
    if (A_TYPEG(ARGT_ARG(func_args, 3)) != A_CNST)
      goto eoshiftcall;
    if (arg_gbl.inforall)
      goto eoshiftcall;
    /* the following can inline cshift and eoshift
     * (without no_comm or comm restriction )
     * but it is restricted no_comm shift for performance reason only
     */

    if (shape) {
      /* need to put this into a temp */
      temp_arr = mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1),
                                lhs, &retval);
      if (temp_arr != 0) {
        mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, 0);
        mk_mem_deallocate(mk_id(temp_arr), arg_gbl.std);
      }
    }
    inline_shifts(func_ast, func_args, retval);
    return temp_arr == 0 && lhs != 0 ? 0 : retval;

  eoshiftcall:
    srcarray = ARGT_ARG(func_args, 0);
    dim = ARGT_ARG(func_args, 3);
    if (dim == 0)
      dim = mk_cval(1, DT_INT);
    nargs = 5;
    shift = ARGT_ARG(func_args, 1);
    if (A_SHAPEG(shift) == 0) {
      /* shift is scalar */
      shift = convert_int(shift, astb.bnd.dtype);
      /* boundary is... */
      if (ARGT_ARG(func_args, 2) == 0) { /* absent */
        rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_eoshiftszca : RTE_eoshiftsz;
        --nargs;
      } else if (A_SHAPEG(ARGT_ARG(func_args, 2)) == 0) /* scalar */
        rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_eoshiftssca : RTE_eoshiftss;
      else /* array */
        rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_eoshiftsaca : RTE_eoshiftsa;
    } else {
      /* shift is array */
      /* boundary is... */
      if (ARGT_ARG(func_args, 2) == 0) { /* absent */
        rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_eoshiftzca : RTE_eoshiftz;
        --nargs;
      } else if (A_SHAPEG(ARGT_ARG(func_args, 2)) == 0) /* scalar */
        rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_eoshiftsca : RTE_eoshifts;
      else /* array */
        rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_eoshiftca : RTE_eoshift;
    }
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = shift;
    ARGT_ARG(newargt, 3) = convert_int(dim, astb.bnd.dtype);
    if (nargs == 5)
      ARGT_ARG(newargt, 4) = ARGT_ARG(func_args, 2);
    goto ret_new;
  case I_MATMUL:           /* matmul(matrix_a, matrix_b) */
  case I_MATMUL_TRANSPOSE: /* matmul((transpose(matrix_a), matrix_b) */
    return matmul(func_ast, func_args, lhs);
  case I_FINDLOC: /* minloc(array, [dim, mask]) */
    srcarray = ARGT_ARG(func_args, 0);
    value = ARGT_ARG(func_args, 1);
    back = ARGT_ARG(func_args, 4);
    mask = ARGT_ARG(func_args, 3);
    mask = misalignment(srcarray, mask, arg_gbl.std);
    if (mask == 0)
      mask = mk_cval(1, DT_LOG);
    dim = ARGT_ARG(func_args, 2);

    if (DTY(A_DTYPEG(value)) == TY_CHAR || DTY(A_DTYPEG(value)) == TY_NCHAR) {
      temp_sptr = memsym_of_ast(value);
      /* e.g., pghpf_any(result, mask, dim) */
      if (dim == 0) {
        newsym = sym_mkfunc(mkRteRtnNm(RTE_findlocstrs), DT_NONE);
        nargs = 6;
        /* scalar findloc, result must be replicated */
        /* get the temp and add the necessary statements */
        temp_arr = mk_maxloc_sptr(
            shape, DDTG(dtype) == DT_INT8 ? DT_INT8 : astb.bnd.dtype);
        retval = mk_id(temp_arr);
        /* add args */
        newargt = mk_argt(nargs);
        ARGT_ARG(newargt, 0) = retval;
        ARGT_ARG(newargt, 1) = srcarray;
        ARGT_ARG(newargt, 2) = value;
        ARGT_ARG(newargt, 3) = size_ast(temp_sptr, DTYPEG(temp_sptr));
        ARGT_ARG(newargt, 4) = mask;
        ARGT_ARG(newargt, 5) = back;
        goto ret_call;
      } else {
        /* pghpf_findloc(result, array, mask, dim) */
        rtlRtn = RTE_findlocstr;
        nargs = 7;
        newargt = mk_argt(nargs);
        ARGT_ARG(newargt, 1) = srcarray;
        ARGT_ARG(newargt, 2) = value;
        ARGT_ARG(newargt, 3) = size_ast(temp_sptr, DTYPEG(temp_sptr));
        ARGT_ARG(newargt, 4) = mask;
        ARGT_ARG(newargt, 5) = dim;
        ARGT_ARG(newargt, 6) = back;
        goto ret_new;
      }
    } else {
      if (dim == 0) {
        nargs = 5;
        newsym = sym_mkfunc(mkRteRtnNm(RTE_findlocs), DT_NONE);
        /* scalar findloc, result must be replicated */
        /* get the temp and add the necessary statements */
        temp_arr = mk_maxloc_sptr(
            shape, DDTG(dtype) == DT_INT8 ? DT_INT8 : astb.bnd.dtype);
        retval = mk_id(temp_arr);
        /* add args */
        newargt = mk_argt(nargs);
        ARGT_ARG(newargt, 0) = retval;
        ARGT_ARG(newargt, 1) = srcarray;
        ARGT_ARG(newargt, 2) = value;
        ARGT_ARG(newargt, 3) = mask;
        ARGT_ARG(newargt, 4) = back;
        goto ret_call;
      } else {
        /* pghpf_findloc(result, array, mask, dim) */
        rtlRtn = RTE_findloc;
        nargs = 6;
        newargt = mk_argt(nargs);
        ARGT_ARG(newargt, 1) = srcarray;
        ARGT_ARG(newargt, 2) = value;
        ARGT_ARG(newargt, 3) = mask;
        ARGT_ARG(newargt, 4) = dim;
        ARGT_ARG(newargt, 5) = back;
        goto ret_new;
      }
    }

  case I_MAXLOC: /* maxloc(array, [dim, mask]) */
  case I_MINLOC: /* minloc(array, [dim, mask]) */
    srcarray = ARGT_ARG(func_args, 0);
    back = ARGT_ARG(func_args, 3);
    is_back_true = get_int_cval(sym_of_ast(back));
    mask = ARGT_ARG(func_args, 2);
    mask = misalignment(srcarray, mask, arg_gbl.std);
    if (mask == 0)
      mask = mk_cval(1, DT_LOG);
    dim = ARGT_ARG(func_args, 1);
    if (dim == 0) {
      if (is_back_true) {
        rtlRtn = optype == I_MAXLOC ? RTE_maxlocs_b : RTE_minlocs_b;
      } else {
        rtlRtn = optype == I_MAXLOC ? RTE_maxlocs : RTE_minlocs;
      }
      newsym = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
      nargs = is_back_true ? 4 : 3;
      /* get the temp and add the necessary statements */
      temp_arr = mk_maxloc_sptr(shape, DDTG(dtype) == DT_INT8 ? DT_INT8
                                                              : astb.bnd.dtype);
      retval = mk_id(temp_arr);
      /* add args */
      newargt = mk_argt(nargs);
      ARGT_ARG(newargt, 0) = retval;
      ARGT_ARG(newargt, 1) = srcarray;
      ARGT_ARG(newargt, 2) = mask;
      if (is_back_true)
        ARGT_ARG(newargt, 3) = back;
      goto ret_call;
    } else {
      /* pghpf_minloc(result, array, mask, dim) */
      if (is_back_true) {
        rtlRtn = optype == I_MAXLOC ? RTE_maxloc_b : RTE_minloc_b;
      } else {
        rtlRtn = optype == I_MAXLOC ? RTE_maxloc : RTE_minloc;
      }
      nargs = is_back_true ? 5 : 4;
      newargt = mk_argt(nargs);
      ARGT_ARG(newargt, 1) = srcarray;
      ARGT_ARG(newargt, 2) = mask;
      ARGT_ARG(newargt, 3) = dim;
      if (is_back_true)
        ARGT_ARG(newargt, 4) = back;
      goto ret_new;
    }
  case I_PACK: /* pack(array, mask, [vector]) */
    srcarray = ARGT_ARG(func_args, 0);
    mask = ARGT_ARG(func_args, 1);
    vector = ARGT_ARG(func_args, 2);

    if (vector == 0) {
      rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_packzca : RTE_packz;
    } else {
      rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_packca : RTE_pack;
    }

    if (mask == 0)
      mask = mk_cval(1, DT_LOG);
    if (DTYG(dtype) == TY_CHAR) {
      ast_from_len = srcarray;
    }
    if (vector == 0) {
      nargs = 3;
      /* pghpf_packz(result, array, mask) */
    } else {
      nargs = 4;
      /* pghpf_pack(result, array, mask, vector) */
    }
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = mask;
    if (nargs == 4) {
      ARGT_ARG(newargt, 3) = vector;
    }
    goto ret_new;
  case I_RESHAPE: /* reshape(source, shape, [pad, order]) */
    return reshape(func_ast, func_args, lhs);
  case I_SPREAD: /* spread(source, dim, ncopies) */
    dim = ARGT_ARG(func_args, 1);
    srcarray = ARGT_ARG(func_args, 0);
    if (!A_SHAPEG(srcarray))
      dim = astb.i1;
    if (A_ALIASG(dim) != 0) {
      int temp_arr = rewrite_intr_allocatable(func_ast, func_args, lhs);
      if (temp_arr != 0) {
        return temp_arr;
      }
      goto ret_norm;
    }
    if (DTYG(dtype) == TY_CHAR) {
      rtlRtn = A_SHAPEG(srcarray) == 0 ? RTE_spreadcs : RTE_spreadca;
      ast_from_len = srcarray;
    } else {
      rtlRtn = A_SHAPEG(srcarray) == 0 ? RTE_spreadsa : RTE_spread;
    }
    nargs = 4;
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = ARGT_ARG(func_args, 1);
    ARGT_ARG(newargt, 3) = ARGT_ARG(func_args, 2);
    goto ret_new;
  case I_TRANSPOSE: /* transpose(matrix) */
    temp_arr = rewrite_intr_allocatable(func_ast, func_args, lhs);
    if (temp_arr != 0) {
      return temp_arr;
    }
    goto ret_norm;
  case I_UNPACK: /* unpack(vector, mask, field) */
    rtlRtn = DTYG(dtype) == TY_CHAR ? RTE_unpackca : RTE_unpack;
    nargs = 4;
    srcarray = ARGT_ARG(func_args, 0);

    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = ARGT_ARG(func_args, 1);
    ARGT_ARG(newargt, 3) = ARGT_ARG(func_args, 2);
    goto ret_new;
  case I_TRANSFER: /* transfer(source, mold [, size]) */
                   /* If the result is an array, then the size is either taken
                    * from the size argument, or is based on the size of the source
                    * and the mold.
                    */
    srcarray = ARGT_ARG(func_args, 0);
    mask = ARGT_ARG(func_args, 1);   /* mold */
    vector = ARGT_ARG(func_args, 2); /* size */
    /* pghpf_transfer(result, src, sizeof(src), sizeof(mold)) */
    nargs = 4;
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = size_ast(sym_of_ast(mask), DDTG(A_DTYPEG(mask)));
    ARGT_ARG(newargt, 3) = size_ast_of(srcarray, DDTG(A_DTYPEG(srcarray)));
    /* get the name of the library routine */
    newsym = sym_mkfunc(mkRteRtnNm(RTE_transfer), DT_NONE);
    /* get the temp and add the necessary statements */
    if (shape) {
      /* need to put this into a temp */
      temp_arr = mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1), 0,
                                &retval);
      /* add temp_arr as argument */
      ARGT_ARG(newargt, 0) = retval;
      if (ALLOCG(temp_arr)) {
        int ddtg = DDTG(A_DTYPEG(mask));
        if (ddtg == DT_ASSCHAR || ddtg == DT_ASSNCHAR || ddtg == DT_DEFERCHAR ||
            ddtg == DT_DEFERNCHAR)
          mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, mask);
        else
          mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, 0);
        mk_mem_deallocate(mk_id(temp_arr), arg_gbl.std);
      }
    } else if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR
               || dtype == DT_ASSNCHAR || dtype == DT_DEFERNCHAR
    ) {
      retval = alloc_char_temp(dtype, "transfer", ARGT_ARG(newargt, 2),
                               arg_gbl.std, 0);
      ARGT_ARG(newargt, 0) = retval;
    } else if ((DTY(dtype) == TY_CHAR
                || DTY(dtype) == TY_NCHAR
                ) &&
               A_ALIASG(DTY(dtype + 1)) == 0) {
      /* the result has adjustable length */
      retval = alloc_char_temp(dtype, "transfer", ARGT_ARG(newargt, 2),
                               arg_gbl.std, 0);
      ARGT_ARG(newargt, 0) = retval;
    } else {
      /* need to put this into a scalar temp */
      int temp_sclr = sym_get_scalar("transfer", "r", dtype);
      /* add temp_sclr as argument */
      retval = mk_id(temp_sclr);
      ARGT_ARG(newargt, 0) = retval;
    }
    goto ret_call;

  case I_ADJUSTL: /* adjustl(string) */
  case I_ADJUSTR: /* adjustr(string) */
    if (optype == I_ADJUSTL) {
      rtlRtn = DTY(DDTG(dtype)) == TY_CHAR ? RTE_adjustla : RTE_nadjustl;
      root = "adjl";
    } else {
      rtlRtn = DTY(DDTG(dtype)) == TY_CHAR ? RTE_adjustra : RTE_nadjustr;
      root = "adjr";
    }
    newsym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_INT);
    arg1 = ARGT_ARG(func_args, 0);
    /* len = RTE_[n]adjust[lr](string) */
    nargs = 2;
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = arg1;

    /* the result has adjustable length */
    forall_depnd_intrin = arg_gbl.inforall && charintr_arg_forall_depnd(arg1);
    if (forall_depnd_intrin) {
      /* ADJUST[rl] in a FORALL, need an array temp subscr'd using
       * the subscripts on the original assign LHS */
      ast = A_LOPG(arg1);
      shape = A_SHAPEG(ast);
      retval = get_charintrin_temp(ast, root);
      retval = mk_subscr_copy(retval, A_ASDG(arg1), A_DTYPEG(ast));
    } else {
      ast = arg1;
      retval = get_charintrin_temp(ast, root);
    }

    if (A_TYPEG(ast) == A_SUBSTR) {
      /* We need to preserve the substring argument unless the
       * string that we're taking the substring of is adjustable.
       */
      switch (A_DTYPEG(A_LOPG(ast))) {
      case DT_ASSCHAR:
      case DT_ASSNCHAR:
      case DT_DEFERCHAR:
      case DT_DEFERNCHAR:
        break;
      default:
        /*
         * First, create a temporary and then propagate the substring
         * expression normalized to 1 to the temporary.  Normalization
         * is required since for adjustr(aaa(ii:jj)), the temp space
         * requirement will be computed as (jj - ii + 1) and the result
         * will be expressed as tmp(ii:jj), thus exceeded the space
         * allocated.  Need to express the result as tmp(1:sz), where
         * sz is 'jj - ii + 1'.
         */
        if (A_LEFTG(ast) && A_LEFTG(ast) != astb.i1) {
          int r = A_RIGHTG(ast);
          int temp_ast;
          if (!r) {
            r = string_expr_length(A_LOPG(ast));
          }
          temp_ast = mk_binop(OP_SUB, r, A_LEFTG(ast), DT_INT);
          temp_ast = mk_binop(OP_ADD, temp_ast, astb.i1, DT_INT);
          retval = mk_substr(retval, 0, temp_ast, A_DTYPEG(retval));
        } else
          retval = mk_substr(retval, 0, A_RIGHTG(ast), A_DTYPEG(retval));
      }
    }

    ARGT_ARG(newargt, 0) = retval;
    if (shape) {
      ADSC *ad;
      dtnew = get_array_dtype(SHD_NDIM(shape), DT_INT);
      ad = AD_DPTR(dtnew);
      for (i = 0; i < (int)SHD_NDIM(shape); i++) {
        AD_LWBD(ad, i) = AD_LWAST(ad, i) = SHD_LWB(shape, i);
        AD_UPBD(ad, i) = AD_UPAST(ad, i) = SHD_UPB(shape, i);
        AD_EXTNTAST(ad, i) = mk_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
      }
      temp_sptr = get_adjlr_arr_temp(dtnew);
      astnew = mk_id(temp_sptr);
      ast = mk_func_node(A_INTR, mk_id(newsym), nargs, newargt);
      A_OPTYPEP(ast, optype);
    } else {
      dtnew = DT_INT;
      astnew = mk_id(get_temp(DT_INT));
      ast = mk_func_node(A_FUNC, mk_id(newsym), nargs, newargt);
    }

    A_DTYPEP(ast, dtnew);
    A_SHAPEP(ast, shape);

    if (forall_depnd_intrin) {
      /* ADJUST[rl] in a FORALL, generate the a FORALL that assigns
       * the ADJUST[rl] to the subscr'd temp */
      int newforall;
      int forall = STD_AST(arg_gbl.std);
      astnew = mk_subscr_copy(astnew, A_ASDG(arg1), A_DTYPEG(ast));

      ast = mk_assn_stmt(astnew, ast, dtnew);
      newforall = mk_stmt(A_FORALL, 0);
      A_LISTP(newforall, A_LISTG(forall));
      A_IFEXPRP(newforall, 0);
      A_IFSTMTP(newforall, ast);
      add_stmt_before(newforall, arg_gbl.std);
    } else {
      ast = mk_assn_stmt(astnew, ast, dtnew);
      add_stmt_before(ast, arg_gbl.std);
    }
    return retval;

  case I_TRIM: /* trim(string) */
    arg1 = ARGT_ARG(func_args, 0);
    /* len = RTE_[n]trim(string) */
    nargs = 2;
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 1) = arg1;
    rtlRtn = DTY(dtype) == TY_CHAR ? RTE_trima : RTE_ntrim;
    newsym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_INT);
    /* the result has adjustable length */
    if (arg_gbl.inforall && charintr_arg_forall_depnd(arg1)) {
      /* The  call to RTE_trim must be in
       * a FORALL and the result(s) must be arrays */
      int forall = STD_AST(arg_gbl.std);
      int newforall;
      ADSC *ad;

      ast = A_LOPG(arg1);
      retval = get_charintrin_temp(ast, "trim");
      retval = mk_subscr_copy(retval, A_ASDG(arg1), A_DTYPEG(ast));
      ARGT_ARG(newargt, 0) = retval;

      shape = A_SHAPEG(ast);
      dtnew = get_array_dtype(SHD_NDIM(shape), DT_INT);
      ad = AD_DPTR(dtnew);
      for (i = 0; i < (int)SHD_NDIM(shape); i++) {
        AD_LWBD(ad, i) = AD_LWAST(ad, i) = SHD_LWB(shape, i);
        AD_UPBD(ad, i) = AD_UPAST(ad, i) = SHD_UPB(shape, i);
        AD_EXTNTAST(ad, i) = mk_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
      }
      temp_sptr = get_adjlr_arr_temp(dtnew);
      astnew = mk_id(temp_sptr);

      mk_mem_allocate(astnew, 0, arg_gbl.std, 0);
      mk_mem_deallocate(astnew, arg_gbl.std);
      astnew = mk_subscr_copy(astnew, A_ASDG(arg1), DT_INT);

      ast = mk_func_node(A_INTR, mk_id(newsym), nargs, newargt);
      A_DTYPEP(ast, DT_INT);
      A_SHAPEP(ast, 0);
      A_OPTYPEP(ast, I_TRIM);
      ast = mk_assn_stmt(astnew, ast, DT_INT);

      retval = mk_substr(retval, 0, astnew, A_DTYPEG(retval));

      newforall = mk_stmt(A_FORALL, 0);
      A_LISTP(newforall, A_LISTG(forall));
      A_IFEXPRP(newforall, 0);
      A_IFSTMTP(newforall, ast);
      add_stmt_before(newforall, arg_gbl.std);
    } else {
      int len_ast;
      retval = get_charintrin_temp(arg1, "trim");
      ARGT_ARG(newargt, 0) = retval;
      temp_sptr = A_SPTRG(retval);
      if (DTY(DTYPEG(temp_sptr)) == DT_DEFERCHAR ||
          DTY(DTYPEG(temp_sptr)) == DT_DEFERNCHAR) {
        len_ast = get_len_of_deferchar_ast(retval);
      } else if (SCG(temp_sptr) == SC_BASED) {
        len_ast = mk_id(CVLENG(temp_sptr));
      } else {
        int len_sptr = get_next_sym(SYMNAME(temp_sptr), "cl");
        STYPEP(len_sptr, ST_VAR);
        DTYPEP(len_sptr, DT_INT);
        SCP(len_sptr, symutl.sc);
        len_ast = mk_id(len_sptr);
      }
      /* add call to function; function returns the len */
      ast = mk_func_node(A_FUNC, mk_id(newsym), nargs, newargt);
      A_DTYPEP(ast, DT_INT);
      A_SHAPEP(ast, 0);
      ast = mk_assn_stmt(len_ast, ast, DT_INT);
      add_stmt_before(ast, arg_gbl.std);
      retval = mk_substr(retval, 0, len_ast, dtype);
    }
    return retval;

  case I_DATE_AND_TIME:
    rtlRtn = RTE_dandta;
    is_icall = FALSE;
    nargs = 4;
    goto opt_common;
  case I_SYSTEM_CLOCK:
    rtlRtn = RTE_sysclk;
    is_icall = FALSE;
    nargs = 3;
    goto opt_common;
  case I_CPU_TIME:
    is_icall = FALSE;
    arg1 = ARGT_ARG(func_args, 0);
    if (DTYG(A_DTYPEG(arg1)) == TY_DBLE)
      rtlRtn = RTE_cpu_timed;
#ifdef TARGET_SUPPORTS_QUADFP
    else if (DTYG(A_DTYPEG(arg1)) == TY_QUAD)
      rtlRtn = RTE_cpu_timeq;
#endif
    else
      rtlRtn = RTE_cpu_time;
    nargs = 1;
    goto opt_common;
  case I_RANDOM_NUMBER:
    is_icall = FALSE;
    arg1 = ARGT_ARG(func_args, 0);
    if (DTYG(A_DTYPEG(arg1)) == TY_DBLE)
      rtlRtn = RTE_rnumd;
#ifdef TARGET_SUPPORTS_QUADFP
    else if (DTYG(A_DTYPEG(arg1)) == TY_QUAD)
      rtlRtn = RTE_rnumq;
#endif
    else
      rtlRtn = RTE_rnum;
    nargs = 1;
    goto opt_common;
  case I_RANDOM_SEED:
    rtlRtn = RTE_rseed;
    is_icall = FALSE;
    nargs = 3;
  opt_common:
    newargt = mk_argt(nargs);
    for (i = 0; i < nargs; ++i) {
      if (ARGT_ARG(func_args, i) == 0)
        ARGT_ARG(newargt, i) = astb.ptr0;
      else
        ARGT_ARG(newargt, i) = ARGT_ARG(func_args, i);
    }
    newsym = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
    retval = 0;
    goto ret_call;
  case I_PRESENT:
    /* present(a) will be present(a$b) a$b base of dummy */
    srcarray = ARGT_ARG(func_args, 0);
    if (A_TYPEG(srcarray) == A_ID && (sptr = A_SPTRG(srcarray)) &&
        SCG(sptr) == SC_DUMMY &&
        !HCCSYMG(sptr) && /* compiler's PRESENT is correct */
        STYPEG(sptr) == ST_ARRAY) {
      if (!normalize_bounds(sptr) || needs_redim(sptr)) {
        sptr = NEWARGG(sptr);
      }
      assert(sptr, "rewrite_func_ast: no formal symbol", func_ast, 3);
      ARGT_ARG(func_args, 0) = mk_id(sptr);
    }
    goto ret_norm;
  case I_SECNDS:
    nargs = 1;
    is_icall = FALSE;
    arg1 = ARGT_ARG(func_args, 0);
    rtlRtn = DTY(A_DTYPEG(arg1)) == TY_DBLE ? RTE_secndsd : RTE_secnds;
    newsym = sym_mkfunc(mkRteRtnNm(rtlRtn), dtype);
    retval = mk_func_node(A_FUNC, mk_id(newsym), nargs, func_args);
    A_DTYPEP(retval, dtype);
    A_SHAPEP(retval, 0);
    return retval;
  case I_TIME:
    is_icall = FALSE;
    arg1 = ARGT_ARG(func_args, 0);
    rtlRtn = DTY(A_DTYPEG(arg1)) == TY_CHAR ? RTE_ftimea : RTE_ftimew;
    goto sub_common;
  case I_DATE:
    is_icall = FALSE;
    arg1 = ARGT_ARG(func_args, 0);
    rtlRtn = DTY(A_DTYPEG(arg1)) == TY_CHAR ? RTE_datea : RTE_datew;
    goto sub_common;
  case I_IDATE:
    is_icall = FALSE;
    arg1 = ARGT_ARG(func_args, 0);
    rtlRtn = DTY(A_DTYPEG(arg1)) == TY_SINT ? RTE_idate : RTE_jdate;
    goto sub_common;
  case I_LASTVAL:
    rtlRtn = RTE_lastval;
    is_icall = FALSE;
    goto sub_common;
  case I_REDUCE_SUM:
    rtlRtn = RTE_global_sum;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_PRODUCT:
    rtlRtn = RTE_global_product;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_ANY:
    rtlRtn = RTE_global_any;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_ALL:
    rtlRtn = RTE_global_all;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_PARITY:
    rtlRtn = RTE_global_parity;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_IANY:
    rtlRtn = RTE_global_iany;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_IALL:
    rtlRtn = RTE_global_iall;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_IPARITY:
    rtlRtn = RTE_global_iparity;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_MINVAL:
    rtlRtn = RTE_global_minval;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_MAXVAL:
    rtlRtn = RTE_global_maxval;
    is_icall = TRUE;
    goto sub_common;
  case I_REDUCE_FIRSTMAX:
    rtlRtn = RTE_global_firstmax;
    is_icall = FALSE;
    /*********************************************
    ====================================
    POSSIBLY NEED THIS SINCE is_icall = FALSE...
    THIS IS OFTEN IN OTHER SUCH CASES.  IN THIS CASE, NEED TO OVER-RIDE WHAT'S
    DONE IN sub_common).
    *BUT*, NOT DONE FOR _SECNDS, I_TIME, I_IDATE OR I_LASTVAL (THE LAST OF
    WHICH LOOKS JUST LIKE REDUCE_MAXVAL.)
    HENCE, TRY WITHOUT THE FOLLOWING TO START WITH!
    ====================================
            newargt = mk_argt(nargs);
            for (i = 0; i < nargs; ++i) {
                ARGT_ARG(newargt, i) = ARGT_ARG(func_args, i);
            }
    *********************************************/
    goto sub_common;
  case I_REDUCE_FIRSTMIN:
    rtlRtn = RTE_global_firstmin;
    is_icall = FALSE;
    goto sub_common;
  case I_REDUCE_LASTMAX:
    rtlRtn = RTE_global_lastmax;
    is_icall = FALSE;
    goto sub_common;
  case I_REDUCE_LASTMIN:
    rtlRtn = RTE_global_lastmin;
    is_icall = FALSE;
    goto sub_common;
  sub_common:
    nargs = ARGT_CNT(func_args);
    newargt = func_args;
    newsym = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
    retval = 0;
    goto ret_call;
  case I_PTR2_ASSIGN:
    check_pointer_type(ARGT_ARG(func_args, 0), ARGT_ARG(func_args, 1),
                       arg_gbl.std, 0);
    if (!XBIT(58, 0x22)) {
      /* ...no changes unless caller remapping. */
      return -1;
    }
    ast = ARGT_ARG(func_args, 1);
    if (A_TYPEG(ast) != A_ID || STYPEG(A_SPTRG(ast)) != ST_ARRAY ||
        POINTERG(A_SPTRG(ast))) {
      /* ...no changes unless pointer assigned to whole array. */
      return -1;
    }
    /* Create call:
     * pghpf_ptr_asgn[_char](ptr_base, ptr_desc, arr_base, arr_desc, vlb),
     * where vlb is a vector of lower bounds of arr_base. */
    sptr = A_SPTRG(ARGT_ARG(func_args, 0));
    nargs = 5;
    if (XBIT(70, 0x20)) {
      if (MIDNUMG(sptr))
        ++nargs;
      if (PTROFFG(sptr))
        ++nargs;
    }
    rtlRtn = DTYG(A_DTYPEG(ast)) == TY_CHAR ? RTE_ptr_asgn_chara : RTE_ptr_asgn;
    newsym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtype);
    newargt = mk_argt(nargs);
    ARGT_ARG(newargt, 0) = ARGT_ARG(func_args, 0);
    ARGT_ARG(newargt, 1) = mk_id(DESCRG(sptr));
    DESCUSEDP(sptr, TRUE);
    ARGT_ARG(newargt, 2) = ARGT_ARG(func_args, 1);
    temp_sptr = A_SPTRG(ARGT_ARG(func_args, 1));
    ARGT_ARG(newargt, 3) = mk_id(DESCRG(temp_sptr));
    DESCUSEDP(temp_sptr, TRUE);
    temp_arr = sym_get_array(SYMNAME(temp_sptr), "v", DT_INT, 1);
    NODESCP(temp_arr, TRUE);
    ALLOCP(temp_arr, FALSE);
    dtype = DTYPEG(temp_arr);
    ADD_NOBOUNDS(dtype) = 0;
    ADD_MLPYR(dtype, 0) = astb.i1;
    ADD_LWAST(dtype, 0) = ADD_LWBD(dtype, 0) = astb.i1;
    ndims = rank_of_sym(temp_sptr);
    ADD_UPAST(dtype, 0) = ADD_UPBD(dtype, 0) = mk_cval(ndims, DT_INT);
    ARGT_ARG(newargt, 4) = mk_id(temp_arr);
    nargs = 5;
    if (XBIT(70, 0x20)) {
      /* add pointer, offset to argument list */
      if (MIDNUMG(sptr)) {
        ARGT_ARG(newargt, nargs) = mk_id(MIDNUMG(sptr));
        ++nargs;
      }
      if (PTROFFG(sptr)) {
        ARGT_ARG(newargt, nargs) = mk_id(PTROFFG(sptr));
        ++nargs;
      }
    }
    dtype = DTYPEG(temp_sptr);
    for (dim = 0; dim < ndims; dim++) {
      subscr[0] = mk_cval(dim + 1, DT_INT);
      ast = mk_subscr(mk_id(temp_arr), subscr, 1, DT_INT);
      ast = mk_assn_stmt(ast, ADD_LWAST(dtype, dim), DT_INT);
      add_stmt_before(ast, arg_gbl.std);
    }
    if (XBIT(49, 0x8000)) {
      /* ...no Cray pointers. */
      /* Set the offset to 1 because every destination pointer P will
       * be transformed later to P(offset). */
      temp_sptr = A_SPTRG(ARGT_ARG(func_args, 0));
      temp_sptr = PTROFFG(temp_sptr);
      assert(temp_sptr, "rewrite_func_ast: no pointer offset", func_ast, 3);
      ast = mk_assn_stmt(mk_id(temp_sptr), astb.i1, DT_INT);
      add_stmt_before(ast, arg_gbl.std);
    }
    is_icall = FALSE;
    goto ret_call;
  case I_GETARG:
  case I_GET_COMMAND:
  case I_GET_COMMAND_ARGUMENT:
    if (optype == I_GETARG) {
      rtlRtn = RTE_getarga;
      nargs = 3;
    } else if (optype == I_GET_COMMAND) {
      rtlRtn = RTE_get_cmda;
      nargs = 4;
    } else {
      rtlRtn = RTE_get_cmd_arga;
      nargs = 5;
    }
    newsym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_INT);
    newargt = mk_argt(nargs);
    for (i = 0; i < nargs - 1; i++) {
      int arg = ARGT_ARG(func_args, i);
      ARGT_ARG(newargt, i) = arg != 0 ? arg : i == 0 ? astb.ptr0c : astb.ptr0;
    }
    ARGT_ARG(newargt, nargs - 1) =
        mk_cval(size_of(stb.user.dt_int), astb.bnd.dtype);
    is_icall = FALSE;
    goto ret_call;
  case I_GET_ENVIRONMENT_VARIABLE:
    newsym = sym_mkfunc_nodesc(mkRteRtnNm(RTE_get_env_vara), DT_INT);
    nargs = 6;
    newargt = mk_argt(nargs);
    for (i = 0; i < nargs - 1; i++) {
      int arg = ARGT_ARG(func_args, i);
      ARGT_ARG(newargt, i) = arg != 0 ? arg : i == 1 ? astb.ptr0c : astb.ptr0;
    }
    ARGT_ARG(newargt, 5) = mk_cval(size_of(stb.user.dt_int), DT_INT4);
    is_icall = FALSE;
    goto ret_call;
  case I_UBOUND: /* ubound(array[, dim, kind]) */
  case I_LBOUND: /* lbound(array[, dim, kind]) */
    return rewrite_lbound_ubound(func_ast, 0, arg_gbl.std);
  default:
    goto ret_norm;
  }

ret_new:
  /* get the name of the library routine */
  newsym = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
  /* get the temp and add the necessary statements */
  if (shape != 0) {
    /* need to put this into a temp */
    temp_arr = mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1), lhs,
                              &retval);
    if (temp_arr != 0) {
      /* add temp_arr as argument */
      ARGT_ARG(newargt, 0) = retval;
      if (ALLOCG(temp_arr)) {
        mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, ast_from_len);
        mk_mem_deallocate(mk_id(temp_arr), arg_gbl.std);
      }
    } else {
      /* lhs was distributed properly for this intr */
      ARGT_ARG(newargt, 0) = lhs;
      retval = 0;
    }
  } else {
    /* need to put this into a scalar temp */
    int temp_sclr = sym_get_scalar("tmp", "r", dtype);
    /* add temp_sclr as argument */
    retval = mk_id(temp_sclr);
    ARGT_ARG(newargt, 0) = retval;
  }

ret_call:
  /* add call to function */
  /* make every call ICALL iff call changes the first argument and
     no side effect, this will help optimizer */
  ast =
      mk_func_node(is_icall ? A_ICALL : A_CALL, mk_id(newsym), nargs, newargt);
  A_OPTYPEP(ast, optype);
  add_stmt_before(ast, arg_gbl.std);
  return retval;

ret_norm:
  retval = mk_func_node(type, A_LOPG(func_ast), A_ARGCNTG(func_ast), func_args);
  if (A_SRCG(func_ast)) { /* type bound procedure pass_arg%member part */
    A_SRCP(retval, A_SRCG(func_ast));
  }
  A_DTYPEP(retval, dtype);
  A_SHAPEP(retval, shape);
  A_OPTYPEP(retval, optype);

  if (shape == 0 && take_out_user_def_func(func_ast)) {
    int temp_ast, temp_sptr;
    if (arg_gbl.lhs == 0) {
      int func = procsym_of_ast(A_LOPG(func_ast));
      if (STYPEG(func) == ST_MEMBER && CLASSG(func) && CCSYMG(func) &&
          VTABLEG(func)) {
        func = VTABLEG(func);
      }
      sptr = func;
    } else if (A_TYPEG(arg_gbl.lhs) == A_SUBSCR) {
      sptr = sptr_of_subscript(arg_gbl.lhs);
    } else {
      sptr = sym_of_ast(arg_gbl.lhs);
    }
    temp_sptr = sym_get_scalar(SYMNAME(sptr), "scl", A_DTYPEG(retval));
    temp_ast = mk_id(temp_sptr);
    astnew = mk_assn_stmt(temp_ast, retval, 0);
    add_stmt_before(astnew, arg_gbl.std);
    retval = temp_ast;
  }

  return retval;
}

/* func_ast is an intrinsic that might be computed directly into its LHS
 * (e.g. TRANPOSE, SPREAD, UNPACK).
 * If lhs has an allocatable member, compute into a temp and return it.
 * Otherwise return 0.
 * This allows allocatable assignments to be handled correctly.
 */
static int
rewrite_intr_allocatable(int func_ast, int func_args, int lhs)
{
  if (!ast_has_allocatable_member(lhs)) {
    return 0;
  } else {
    /* compute into a temp and copy that to lhs to handle allocatables */
    int new_rhs, assn_ast;
    int subscr[MAXSUBS];
    int tmp_ast = 0;
    DTYPE dtype = A_DTYPEG(func_ast);
    int tmp_sptr = mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1),
                                  lhs, &tmp_ast);
    assert(tmp_sptr != 0, "sptr=0 from mk_result_sptr", 0, ERR_Fatal);
    assert(tmp_ast != 0, "tmp_ast=0 from mk_result_sptr", 0, ERR_Fatal);
    mk_mem_allocate(mk_id(tmp_sptr), subscr, arg_gbl.std, 0);
    mk_mem_deallocate(mk_id(tmp_sptr), arg_gbl.std);
    new_rhs = rewrite_func_ast(func_ast, func_args, tmp_ast);
    if (new_rhs != 0) {
      assn_ast = mk_assn_stmt(tmp_ast, new_rhs, dtype);
      add_stmt_before(assn_ast, arg_gbl.std);
    }
    return tmp_ast;
  }
}

static LOGICAL
ast_has_allocatable_member(int ast)
{
  if (ast == 0) {
    return FALSE;
  } else {
    int sptr = memsym_of_ast(ast);
    return !HCCSYMG(sptr) && allocatable_member(sptr);
  }
}

/* take out user-defined function to eliminate multiple invocation of function
 */
static LOGICAL
take_out_user_def_func(int func_ast)
{
  if (A_TYPEG(func_ast) == A_FUNC && arg_gbl.lhs != 0 &&
      A_SHAPEG(arg_gbl.lhs) != 0 && !arg_gbl.inforall) {
    return TRUE;
  }

  /* if the function call is in a difficult statement, like an IF or
   * DO or computed GOTO, difficult.continue_std holds the temporary
   * CONTINUE statement inserted around which temp statements were
   * inserted, and difficult.func_std holds the original statement.
   * If any statements were inserted between the CONTINUE and the original
   * statement, these statements should follow the function call,
   * so we must move the function call, store the result, and then
   * use the result in the IF, DO, etc. */
  if (difficult.continue_std != 0 && difficult.func_std != 0 &&
      STD_NEXT(difficult.continue_std) != difficult.func_std) {
    return TRUE;
  }
  return FALSE;
}

/*
 * Create an alloctable char temp of length 'len' within the context of
 * of a statement. The temp's len assignment and allocate statements are
 * added before 'std'; the temp's deallocate statement is added after 'std'.
 */
static int
alloc_char_temp(int basetype, const char *basename, int len, int std,
                int use_basetype)
{
  int dtype;
  int tempsptr;
  int tempast;
  int tempbase, templen, alloc, lenasn;

  if (!use_basetype)
    dtype = get_type(2, DTY(basetype), len);
  else
    dtype = basetype;
  tempsptr = get_next_sym(basename, "c");
  DTYPEP(tempsptr, dtype);
  STYPEP(tempsptr, ST_VAR);
  DCLDP(tempsptr, 1);
  SCP(tempsptr, SC_BASED);
  tempast = mk_id(tempsptr);

  /* create a pointer variable */
  tempbase = get_next_sym(SYMNAME(tempsptr), "cp");
  templen = get_next_sym(SYMNAME(tempsptr), "cl");

  /* make the pointer point to sptr */
  STYPEP(tempbase, ST_VAR);
  DTYPEP(tempbase, DT_PTR);
  SCP(tempbase, symutl.sc);

  /* set length variable */
  STYPEP(templen, ST_VAR);
  DTYPEP(templen, DT_INT);
  SCP(templen, symutl.sc);

  MIDNUMP(tempsptr, tempbase);
  CVLENP(tempsptr, templen);
  ADJLENP(tempsptr, 1);

  /* add char length variable assignment */
  lenasn = mk_assn_stmt(mk_id(templen), len, 0);
  add_stmt_before(lenasn, std);

  /* add an allocate statement */
  alloc = mk_stmt(A_ALLOC, 0);
  A_TKNP(alloc, TK_ALLOCATE);
  A_LOPP(alloc, 0);
  A_SRCP(alloc, tempast);
  add_stmt_before(alloc, std);

  alloc = mk_stmt(A_ALLOC, 0);
  A_TKNP(alloc, TK_DEALLOCATE);
  A_LOPP(alloc, 0);
  A_SRCP(alloc, tempast);
  add_stmt_after(alloc, std);

  return tempast;
}

static int
get_charintrin_temp(int arg, const char *nm)
{
  int adt;
  int dtype;
  int shape;
  int temp;
  int ast;
  int len;

  adt = A_DTYPEG(arg);
  dtype = adjust_ch_length(adt, arg);
  shape = A_SHAPEG(arg);

  /* get the temp and add the necessary statements */
  if (shape) {
    int subscr[MAXSUBS];
    /* need to put this into a temp */

    temp = mk_shape_sptr(shape, subscr, dtype);
    ast = mk_id(temp);
    if (ALLOCG(temp)) {
      mk_mem_allocate(ast, subscr, arg_gbl.std, 0);
      mk_mem_deallocate(ast, arg_gbl.std);
    }
  } else if (A_ALIASG(DTY(dtype + 1))) {
    temp = get_next_sym(nm, "c");
    DTYPEP(temp, dtype);
    STYPEP(temp, ST_VAR);
    DCLDP(temp, 1);
    SCP(temp, symutl.sc);
    ast = mk_id(temp);
  } else {
    if (A_TYPEG(arg) == A_ID) {
      /* check if arg has early spec */
      int sptr = A_SPTRG(arg);
      if (sptr && (EARLYSPECG(sptr) ||
                   (HCCSYMG(sptr) && ADJLENG(sptr) && CVLENG(sptr)))) {
        int clen = CVLENG(sptr);
        ast = alloc_char_temp(dtype, "trim", mk_id(clen), arg_gbl.std, 1);
        return ast;
      }
    }
    len = rewrite_sub_ast(DTY(dtype + 1), 0);
    ast = alloc_char_temp(dtype, nm, len, arg_gbl.std, 1);
  }

  return ast;
}

/* This routine takes two array section, dest and src.
 * if there is communication from src to destination.
 * it creates a new temporary which is same shape and subscript
 * and alignment and assign src to that temp and return the temp.
 */

static int
misalignment(int dest, int src, int std)
{
  return src;
}

/* arr:	array ast */
/* arg_ast: call ast */
/* argn: argument number */
static void
check_assumed_size(int arr, int arg_ast, int argn)
{
  /* In the presence of an interface, need to check if the formal
   * argument is assumed-size, and mark the array sequential. */
}

static int rewrite_sub_args(int arg_ast, int lc);

/* keep track of which dimensions have been as dim= for CSHIFT/EOSHIFT calls */
static int inshift[8] = {0, 0, 0, 0, 0, 0, 0, 0};

/*
 * return '1' for a simple reference (scalar, member, array element)
 * return '1' for unary or binary op of simple reference operands
 * return '0' otherwise
 */
static int
simple_reference(int ast)
{
  switch (A_TYPEG(ast)) {
  case A_MEM:
  case A_ID:
  case A_SUBSCR:
  case A_CNST:
    return 1;
  case A_BINOP:
    if (!simple_reference(A_LOPG(ast)))
      return 0;
    if (!simple_reference(A_ROPG(ast)))
      return 0;
    return 1;
  case A_UNOP:
  case A_PAREN:
    if (!simple_reference(A_LOPG(ast)))
      return 0;
    return 1;
  default:
    return 0;
  }
} /* simple_reference */

/*
 * return '1' if the argument should not be rewritten;
 * This occurs for nested CSHIFT or EOSHIFT calls.
 * in that case, call rewrite_sub_args for the nested call.
 */
static int
leave_arg(int ast, int i, int *parg, int lc)
{
  int arg;
  arg = *parg;
  /* 'ast', the calling ast, must be EOSHIFT or CSHIFT
   * if the first argument is also EOSHIFT or CSHIFT, return 1 */
  if (ast && (A_TYPEG(ast) == A_INTR) &&
      (A_OPTYPEG(ast) == I_EOSHIFT || A_OPTYPEG(ast) == I_CSHIFT) && (i == 0) &&
      (arg) && (A_TYPEG(arg) == A_INTR) &&
      (A_OPTYPEG(arg) == I_EOSHIFT || A_OPTYPEG(arg) == I_CSHIFT)) {
    int astarglist, argarglist, astdim = 0, argdim = 0, save;
    astarglist = A_ARGSG(ast);
    argarglist = A_ARGSG(arg);

    if (A_OPTYPEG(ast) == I_CSHIFT) {
      astdim = ARGT_ARG(astarglist, 2);
    } else if (A_OPTYPEG(ast) == I_EOSHIFT) {
      astdim = ARGT_ARG(astarglist, 3);
    }
    if (astdim == 0) {
      astdim = 1;
    } else {
      assert(A_TYPEG(astdim) == A_CNST,
             "inline_shifts: variable dim not implemented", ast, 3);
      astdim = get_int_cval(A_SPTRG(A_ALIASG(astdim)));
    }
    if (A_OPTYPEG(arg) == I_CSHIFT) {
      argdim = ARGT_ARG(argarglist, 2);
    } else if (A_OPTYPEG(arg) == I_EOSHIFT) {
      argdim = ARGT_ARG(argarglist, 3);
    }
    if (argdim == 0) {
      argdim = 1;
    } else {
      assert(A_TYPEG(argdim) == A_CNST,
             "inline_shifts: variable dim not implemented", ast, 3);
      argdim = get_int_cval(A_SPTRG(A_ALIASG(argdim)));
    }
    save = inshift[astdim];
    inshift[astdim] = 1;
    if (inshift[argdim]) {
      /* there may be further nested shifts as well */
      arg = rewrite_sub_ast(arg, lc);
      *parg = arg;
    } else {
      int args;
      args = rewrite_sub_args(arg, lc);
      A_ARGSP(arg, args);
    }
    inshift[astdim] = save;
    return 1;
  }
  if (!XBIT(70, 0x200000) && ast && (A_TYPEG(ast) == A_INTR)) {
    int astdim, dim, args, dtype, mask;
    mask = 0;
    switch (A_OPTYPEG(ast)) {
    case I_SUM:
    case I_PRODUCT:
    case I_MAXVAL:
    case I_MINVAL:
    case I_ALL:
    case I_ANY:
    case I_COUNT:
      if (i != 0)
        return 0;
      args = A_ARGSG(ast);
      astdim = ARGT_ARG(args, 1);
      mask = ARGT_ARG(args, 2);
      break;
    case I_NORM2:
      if (i != 0)
        return 0;
      // Argument with expression has to be rewritten
      switch(A_TYPEG(arg)) {
        default:
          break;
        case A_BINOP:
        case A_UNOP:
        case A_PAREN:
          return 0;
      }

      args = A_ARGSG(ast);
      astdim = ARGT_ARG(args, 1);
      break;
    case I_DOT_PRODUCT:
      if (i > 1)
        return 0;
      dtype = A_DTYPEG(ast);
      if (DT_ISCMPLX(DDTG(dtype)) && (XBIT(70, 0x4000000)
                                      || DDTG(dtype) == DT_QCMPLX
                                      ))
        return 0;
      astdim = 0;
      break;
    default:
      return 0;
    }
    if (mask)
      return 0;
    /* for a reduction function, 1st argument, leave it alone
     * if the 'dim' argument (if any) is '1' */
    if (astdim != 0) {
      if (A_TYPEG(astdim) != A_CNST)
        return 0;
      if (!XBIT(70, 0x400000)) {
        dim = get_int_cval(A_SPTRG(astdim));
        if (dim != 1)
          return 0;
      }
    }
    /* make sure the argument is an array, or expression of array
     * (no function calls) */
    if (!simple_reference(arg)) {
      return 0;
    }
    return 1;
  }
  return 0;
} /* leave_arg */

/*
 * return TRUE for TRANSPOSE, and for 1st argument of SPREAD
 * these arguments can be left as expressions
 */
static LOGICAL
leave_elemental_argument(int func_ast, int argnum)
{
  if (A_TYPEG(func_ast) == A_INTR) {
    if (A_OPTYPEG(func_ast) == I_TRANSPOSE ||
        (A_OPTYPEG(func_ast) == I_SPREAD && argnum == 0)) {
      return TRUE;
    }
  }
  return FALSE;
} /* leave_elemental_argument */

/*
 * if the actual argument is a scalar of intrinsic type
 * and the dummy argument is a pass-by-reference intent(in) argument,
 * then copy the scalar to a temp
 */
static int
copy_scalar_intent_in(int arg, int dummy_sptr, int std)
{
  int dtype, sptr, newsptr, destast, asnast;
  if (!dummy_sptr)
    return arg;
  if (INTENTG(dummy_sptr) != INTENT_IN)
    return arg;
  if (PASSBYVALG(dummy_sptr))
    return arg;
  if (ALLOCATTRG(dummy_sptr))
    return arg;
  if (POINTERG(dummy_sptr))
    return arg;
  if (OPTARGG(dummy_sptr))
    return arg;
  if (ALLOCG(dummy_sptr))
    return arg;
  dtype = A_DTYPEG(arg);
  if (!DT_ISBASIC(dtype))
    return arg;
  if (DTY(dtype) == TY_CHAR)
    return arg;
  if (A_SHAPEG(arg))
    return arg;
  if (A_TYPEG(arg) != A_ID)
    return arg;
  sptr = A_SPTRG(arg);
  if (OPTARGG(sptr))
    return arg; /* may be a missing argument */
  newsptr = sym_get_scalar(SYMNAME(sptr), "a", dtype);
  destast = mk_id(newsptr);
  asnast = mk_assn_stmt(destast, arg, dtype);
  add_stmt_before(asnast, std);
  return mk_id(newsptr);
} /* copy_scalar_intent_in */

/*
 * rewrite arguments of a function or subroutine call
 */
static int
rewrite_sub_args(int arg_ast, int lc)
{
  int argt;
  int newargt = 0;
  int arg, subarg;
  int shape;
  int nargs;
  int i, j, n;
  int asd;
  int temp_arr;
  int dtype, eldtype;
  int asn_ast;
  int ast;
  int std;
  int arr;
  int subscr[MAXSUBS];
  int func_args;
  int retval;
  int dscptr;
  int dummy_sptr;
  int func_sptr;
  int iface;
  LOGICAL caller_copies;
  int cloc_ast;

  std = arg_gbl.std;
  argt = A_ARGSG(arg_ast);
  nargs = A_ARGCNTG(arg_ast);
  func_sptr = procsym_of_ast(A_LOPG(arg_ast));
  if (STYPEG(func_sptr) == ST_MEMBER && CLASSG(func_sptr) &&
      CCSYMG(func_sptr) && VTABLEG(func_sptr)) {
    func_sptr = VTABLEG(func_sptr);
  }
  proc_arginfo(func_sptr, NULL, &dscptr, &iface);
  newargt = mk_argt(nargs);
  for (i = 0; i < nargs; ++i) {
    if (ARGT_ARG(argt, i) == 0) {
      ARGT_ARG(newargt, i) = 0;
      continue;
    }
    caller_copies = FALSE;
    arg = ARGT_ARG(argt, i);
    dummy_sptr = 0;
    if (dscptr && i < PARAMCTG(func_sptr))
      dummy_sptr = aux.dpdsc_base[dscptr + i];
    if (leave_arg(arg_ast, i, &arg, lc)) {
      ARGT_ARG(newargt, i) = arg;
      continue;
    }
    /* iso_c  c_loc , c_funloc are noops as function arguments:
       pass their arg up to this func as an arg
     */
    if (is_iso_cloc(arg)) {
      cloc_ast = ARGT_ARG(A_ARGSG(arg), 0);
      /* take out CLOC for both byval and byref arguments */
      if ((dummy_sptr == 0) || (func_sptr == 0)) {

        ARGT_ARG(newargt, i) = cloc_ast;
        continue;
      }
    }

    if (A_TYPEG(arg_ast) == A_INTR && A_OPTYPEG(arg_ast) == I_DOT_PRODUCT &&
        i == 2 && arg == ARGT_ARG(argt, 0)) {
      /* optimize the case of DOTPRODUCT(a(:)%mem,a(:)%mem) */
      ARGT_ARG(newargt, i) = ARGT_ARG(newargt, 0);
      continue;
    }
    arg = rewrite_sub_ast(arg, lc);
    /*	arg = rewrite_interface_args(arg_ast, arg, i);*/
    /* leave elementals alone */
    if (A_TYPEG(arg_ast) == A_INTR && INKINDG(func_sptr) == IK_ELEMENTAL) {
      ARGT_ARG(newargt, i) = arg;
      continue;
    }
    /* leave pointer assign alone */
    if (A_TYPEG(arg_ast) == A_ICALL && A_OPTYPEG(arg_ast) == I_PTR2_ASSIGN) {
      ARGT_ARG(newargt, i) = arg;
      continue;
    }
    if (A_TYPEG(arg_ast) == A_INTR) {
      /* leave elementals alone, leave pointer assign alone */
      if (INKINDG(func_sptr) == IK_ELEMENTAL ||
          A_OPTYPEG(arg_ast) == I_PTR2_ASSIGN) {
        ARGT_ARG(newargt, i) = arg;
        continue;
      }
    }
    if (iface && ELEMENTALG(iface)) {
      /* leave alone if arg is not an elemental function,
       * else process function below
       */
      if (A_TYPEG(arg) == A_FUNC) {
        int sym;
        switch (A_TYPEG(A_LOPG(arg))) {
        case A_ID:
        case A_LABEL:
        case A_ENTRY:
        case A_SUBSCR:
        case A_SUBSTR:
        case A_MEM:
          sym = memsym_of_ast(A_LOPG(arg));
          if (CLASSG(sym) && VTABLEG(sym) && BINDG(sym)) {
            sym = VTABLEG(sym);
            break;
          }
          FLANG_FALLTHROUGH;
        default:
          sym = A_SPTRG(A_LOPG(arg));
        }
        if (ELEMENTALG(sym)) {
          ARGT_ARG(newargt, i) = arg;
          continue;
        }
      } else if (A_TYPEG(arg) != A_FUNC || !ELEMENTALG(A_SPTRG(A_LOPG(arg)))) {
        ARGT_ARG(newargt, i) = arg;
        continue;
      }
    }
    /* don't touch %val, %loc, and %ref operators even their shape is
     * not NULL
     */
    if (A_TYPEG(arg) == A_UNOP) {
      if (A_OPTYPEG(arg) == OP_VAL || A_OPTYPEG(arg) == OP_BYVAL ||
          A_OPTYPEG(arg) == OP_LOC || A_OPTYPEG(arg) == OP_REF) {
        ARGT_ARG(newargt, i) = arg;
        continue;
      }
    }
    /* if this is a scalar expression variable passed to
     * a non-value intent(in) argument, copy to a temp
     * so we don't have to mark the variable as ADDRTKN */
    if (dummy_sptr && XBIT(68, 8))
      arg = copy_scalar_intent_in(arg, dummy_sptr, std);
    shape = A_SHAPEG(arg);
    dtype = A_DTYPEG(arg);
    subarg = arg;
    if (A_TYPEG(subarg) == A_SUBSTR)
      subarg = A_LOPG(subarg);
    if (A_TYPEG(subarg) == A_ID) {
      ARGT_ARG(newargt, i) = arg;
      continue;
    }
    if (A_TYPEG(subarg) == A_MEM) {
      /* if this is an array of derived types, then it needs
       * to be rewritten */
      if (A_SHAPEG(A_PARENTG(subarg))) {
        caller_copies = TRUE;
        goto rewrite_this;
      }
      if (A_TYPEG(A_MEMG(subarg)) == A_ID) {
        ARGT_ARG(newargt, i) = arg;
        continue;
      }
    }
    if (shape) {
      /* for  transpose(elementalexpression) or
       *      spread(elementalexpression,dim,size),
       * leave the elemental expressions in place, don't assign
       * to a temp.  They will be expanded when the transpose or spread
       * are inlined */
      if (leave_elemental_argument(arg_ast, i)) {
        ARGT_ARG(newargt, i) = arg;
        continue;
      }
      /* argument may be an array, but not a whole array */
      /* check for a(:)%b(9) */
      if (A_TYPEG(subarg) == A_SUBSCR) {
        int lop = A_LOPG(subarg);
        if (A_TYPEG(lop) == A_MEM && A_SHAPEG(A_PARENTG(lop))) {
          /* shape comes from parent of A_MEM; copy */
          caller_copies = TRUE;
          goto rewrite_this;
        }
      }

      /* need to check for vector subscripts here */
      if (subarg == arg && A_TYPEG(subarg) == A_SUBSCR) {
        asd = A_ASDG(subarg);
        n = ASD_NDIM(asd);
        for (j = 0; j < n; ++j)
          if (A_TYPEG(ASD_SUBS(asd, j)) != A_TRIPLE &&
              A_SHAPEG(ASD_SUBS(asd, j)) != 0)
            goto rewrite_this;
        ARGT_ARG(newargt, i) = arg;
        continue;
      }
    rewrite_this:
      assert(!arg_gbl.inforall, "rewrite_sub_args: can not handle PURE arg",
             arg, 2);
      if (arg_gbl.inforall) {
        ARGT_ARG(newargt, i) = arg;
        continue;
      }

      /* either vector subscript, or array expression */
      /* need to put this into a temp */
      ast = search_conform_array(subarg, FALSE);
      if (ast == 0)
        ast = search_conform_array(subarg, TRUE);
      assert(ast != 0, "rewrite_sub_args: can't find array", arg, 4);
      eldtype = DDTG(dtype);
      if (eldtype == DT_ASSCHAR || eldtype == DT_ASSNCHAR ||
          eldtype == DT_DEFERCHAR || eldtype == DT_DEFERNCHAR) {
        /* make up fake datatype with actual length */
        if (A_TYPEG(ast) == A_INTR) {
          eldtype =
              fix_dtype(memsym_of_ast(ARGT_ARG(A_ARGSG(ast), 0)), eldtype);
        } else {
          eldtype = get_type(2, DTY(eldtype), string_expr_length(arg));
        }
      }

      if (A_TYPEG(ast) == A_INTR) {
        func_args = A_ARGSG(ast);
        temp_arr = mk_result_sptr(ast, func_args, subscr, eldtype, 0, &retval);
        ast = retval;
      } else {
        temp_arr = mk_assign_sptr(ast, "a", subscr, eldtype, &ast);
      }
      /* make assignment to temp_arr */
      asn_ast = mk_assn_stmt(ast, arg, dtype);
      ARGT_ARG(newargt, i) = ast;
      if (ALLOCG(temp_arr)) {
        mk_mem_allocate(mk_id(temp_arr), subscr, std, 0);
      }
      add_stmt_before(asn_ast, std);
      if (ALLOCG(temp_arr))
        mk_mem_deallocate(mk_id(temp_arr), std);
      if (caller_copies && (!dummy_sptr || INTENTG(dummy_sptr) != INTENT_IN)) {
        /* make assignment from temp_arr */

        asn_ast = mk_assn_stmt(arg, ast, dtype);
        add_stmt_after(asn_ast, std);
      }
    } else if (A_TYPEG(subarg) == A_SUBSCR) {
      /*
       * argument is a subscripted reference. If the array is
       * distributed, then this needs to be put into a scalar temp
       * before the call and copied back to the array element after
       * the call. Note, this should probably be done in a later
       * phase
       */

      arr = A_LOPG(subarg);
      check_assumed_size(arr, arg_ast, i);
      if (A_TYPEG(arr) != A_ID || !ALIGNG(A_SPTRG(arr)))
        goto lval;
      ARGT_ARG(newargt, i) = subarg;
    } else if (A_ISLVAL(A_TYPEG(subarg))) {
    lval:
      /* This reference is an lvalue. We want to leave it alone.
       * However, it may be necessary to pull out subcomponents
       * of it. Example: substr(idx(1):idx(2)) where idx is distributed.
       */
      ARGT_ARG(newargt, i) = arg;
    } else
      ARGT_ARG(newargt, i) = arg;
  }
  return newargt;
}

/*
 * rewrite subprogram call
 */
static int
rewrite_sub_ast(int ast, int lc)
{
  int shape;
  int l, r, lop;
  int dtype;
  int args;
  int asd;
  int numdim;
  int i;
  int subs[MAXSUBS];

  if (ast == 0)
    return 0;
  shape = A_SHAPEG(ast);
  switch (A_TYPEG(ast)) {
  case A_NULL:
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_LABEL:
    return ast;
  case A_MP_ATOMICREAD:
    dtype = A_DTYPEG(ast);
    r = rewrite_sub_ast(A_SRCG(ast), lc);
    r = mk_atomic(A_MP_ATOMICREAD, 0, r, dtype);
    A_MEM_ORDERP(r, A_MEM_ORDERG(ast));
    return r;
  case A_MEM:
    dtype = A_DTYPEG(ast);
    r = rewrite_sub_ast((int)A_MEMG(ast), lc);
    l = rewrite_sub_ast(A_PARENTG(ast), lc);
    return mk_member(l, r, dtype);
  case A_BINOP:
    dtype = A_DTYPEG(ast);
    l = rewrite_sub_ast(A_LOPG(ast), lc);
    r = rewrite_sub_ast(A_ROPG(ast), lc);
    return mk_binop(A_OPTYPEG(ast), l, r, dtype);
  case A_UNOP:
    dtype = A_DTYPEG(ast);
    l = rewrite_sub_ast(A_LOPG(ast), lc);
    return mk_unop(A_OPTYPEG(ast), l, dtype);
  case A_PAREN:
    dtype = A_DTYPEG(ast);
    l = rewrite_sub_ast(A_LOPG(ast), lc);
    return mk_paren(l, dtype);
  case A_CONV:
    dtype = A_DTYPEG(ast);
    l = rewrite_sub_ast(A_LOPG(ast), lc);
    /* If the operand is a scalar and the result has a shape, we
     * can't use mk_convert */
    if (!A_SHAPEG(l) && A_SHAPEG(ast)) {
      r = mk_promote_scalar(l, dtype, A_SHAPEG(ast));
      A_DTYPEP(r, dtype);
    } else
      r = mk_convert(l, dtype);
    return r;
  case A_SUBSTR:
    lop = rewrite_sub_ast(A_LOPG(ast), lc);
    l = rewrite_sub_ast(A_LEFTG(ast), lc);
    r = rewrite_sub_ast(A_RIGHTG(ast), lc);
    return mk_substr(lop, l, r, A_DTYPEG(ast));
  case A_SUBSCR:
    dtype = A_DTYPEG(ast);
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd);
    assert(numdim > 0 && numdim <= 7, "rewrite_sub_ast: bad numdim", ast, 4);
    lop = rewrite_sub_ast(A_LOPG(ast), lc);
    for (i = 0; i < numdim; ++i) {
      l = rewrite_sub_ast(ASD_SUBS(asd, i), lc);
      subs[i] = l;
    }
    /*	return mk_subscr(A_LOPG(ast), subs, numdim, DTY(dtype+1)); */
    return mk_subscr(lop, subs, numdim, dtype);
  case A_TRIPLE:
    l = rewrite_sub_ast(A_LBDG(ast), lc);
    r = rewrite_sub_ast(A_UPBDG(ast), lc);
    i = rewrite_sub_ast(A_STRIDEG(ast), lc);
    return mk_triple(l, r, i);
  case A_INTR:
  case A_FUNC:
    ast = inline_reduction_f90(ast, 0, lc, NULL);
    if (A_TYPEG(ast) != A_INTR && A_TYPEG(ast) != A_FUNC)
      return ast;
    args = rewrite_sub_args(ast, lc);

    /* try again to inline it */
    ast = inline_reduction_f90(ast, 0, lc, NULL);
    l = rewrite_func_ast(ast, args, 0);
    return l;
  case A_ICALL:
    ast = inline_reduction_f90(ast, 0, lc, NULL);
    if (A_TYPEG(ast) != A_ICALL)
      return ast;
    args = rewrite_sub_args(ast, lc);
    A_ARGSP(ast, args);
    /* try again to inline it */
    ast = inline_reduction_f90(ast, 0, lc, NULL);
    l = rewrite_func_ast(ast, args, 0);
    return l;
  case A_CALL:
    assert(elemental_func_call(ast),
           "rewrite_sub_ast: attempt to rewrite call to non elemental subr",
           ast, 3);
    args = rewrite_sub_args(ast, lc);
    A_ARGSP(ast, args);
    transform_elemental(ast, args);
    return -1;
  default:
    interr("rewrite_sub_ast: unexpected ast", ast, 2);
    return ast;
  }
}

/* We are using the lhs for the result of an inline intrinsic.
 * Allocate it if necessary. */
static void
allocate_lhs_if_needed(int lhs, int rhs, int std)
{
  int astif, new_lhs;
  if (!XBIT(54, 1))
    return;
  if (A_TYPEG(lhs) == A_SUBSCR)
    return;
  if (!ast_is_sym(lhs) || !ALLOCATTRG(sym_of_ast(lhs)))
    return;
  astif = mk_conformable_test(lhs, rhs, OP_LE);
  add_stmt_before(astif, std);
  gen_dealloc_if_allocated(lhs, std);
  new_lhs = add_shapely_subscripts(lhs, rhs, A_DTYPEG(rhs),
                                   array_element_dtype(A_DTYPEG(lhs)));
  add_stmt_before(mk_allocate(new_lhs), std);
  add_stmt_before(mk_stmt(A_ENDIF, 0), std);
}

void
rewrite_asn(int ast, int std, bool flag, int lc)
{
  int rhs, lhs;
  int args;
  int l;
  int asd, j, n;
  int new_rhs;
  LOGICAL doremove;

  rhs = A_SRCG(ast);
  lhs = A_DESTG(ast);
  arg_gbl.lhs = lhs;

  lhs = rewrite_sub_ast(A_DESTG(ast), lc);
  A_DESTP(ast, lhs);
  arg_gbl.lhs = lhs;

  if (A_TYPEG(rhs) == A_MP_ATOMICREAD)
    return;

  /* If this is an assignment of an intrinsic directly into
   * the LHS, avoid the temp */
  if (flag && A_SHAPEG(lhs) &&
      (A_TYPEG(rhs) == A_FUNC || A_TYPEG(rhs) == A_INTR)) {
    int std_prev = STD_PREV(std); /* for allocate_lhs_if_needed case */
    if (A_TYPEG(lhs) == A_SUBSCR) {
      asd = A_ASDG(lhs);
      n = ASD_NDIM(asd);
      for (j = 0; j < n; ++j)
        if (A_TYPEG(ASD_SUBS(asd, j)) != A_TRIPLE &&
            A_SHAPEG(ASD_SUBS(asd, j)) != 0)
          goto rewrite_this; /* vector subscript */
    }
    /* Otherwise, we can use lhs directly */
    args = rewrite_sub_args(rhs, lc);
    A_ARGSP(rhs, args);
    new_rhs = inline_reduction_f90(rhs, lhs, lc, &doremove);
    if (new_rhs == rhs) {
      new_rhs = rewrite_func_ast(rhs, args, lhs);
      doremove = new_rhs == 0;
    }
    if (doremove) {
      allocate_lhs_if_needed(lhs, rhs, STD_NEXT(std_prev));
      if (std)
        delete_stmt(std);
    } else {
      A_SRCP(ast, new_rhs);
    }
    return;
  }

rewrite_this:
  l = rewrite_sub_ast(rhs, lc);
  A_SRCP(ast, l);
}

void
rewrite_calls(void)
{
  int std, stdnext, stdnew;
  int ast, rhs, lhs, astnew;
  int args, a;
  int type;
  int sptr_lhs;
  int prevstd, src;
  int parallel_depth;
  int task_depth;
  /*
   * Transform subroutine/function call arguments.
   * 1. If they contain array expressions, a temp must be allocated and
   *    the expression must be copied into the temp.
   * 2. If they contain references to array elements, then the elements must
   *    be copied into a scalar, the scalar passed, and the elements
   *    copied back.  Some of this can be avoided if the INTENT is known.
   * 3. Scalars aren't interfered with, except scalar arguments to
   *    intent(in) dummy arguments are copied to a temp, and the temp
   *    is passed instead.
   */

  parallel_depth = 0;
  task_depth = 0;
  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    arg_gbl.std = std;
    arg_gbl.lhs = 0;
    arg_gbl.used = FALSE;
    arg_gbl.inforall = FALSE;
    gbl.lineno = STD_LINENO(std);
    ast = STD_AST(std);
    switch (type = A_TYPEG(ast)) {
    case A_ASN:
      rhs = A_SRCG(ast);
      if (A_TYPEG(rhs) == A_HOVLPSHIFT || A_TYPEG(rhs) == A_HCSTART)
        break;
      lhs = A_DESTG(ast);
      sptr_lhs = sym_of_ast(lhs);
      open_pragma(STD_LINENO(std));
      if (expr_dependent(A_SRCG(ast), lhs, std, std))
        arg_gbl.used = TRUE;
      close_pragma();
      rewrite_asn(ast, std, TRUE, 0);
      break;
    case A_WHERE:
      lhs = A_DESTG(A_IFSTMTG(ast));
      sptr_lhs = sym_of_ast(lhs);
      if (expr_dependent(A_IFEXPRG(ast), lhs, std, std))
        arg_gbl.used = TRUE;
      assert(A_IFSTMTG(ast), "rewrite_calls: block where", 0, 4);
      rewrite_asn(A_IFSTMTG(ast), std, FALSE, 0);
      a = rewrite_sub_ast(A_IFEXPRG(ast), 0);
      A_IFEXPRP(ast, a);
      break;
    case A_IFTHEN:
    case A_IF:
    case A_AIF:
    case A_ELSEIF:
    case A_DOWHILE:
    case A_CGOTO:
    case A_DO:
    case A_MP_PDO:
      /* If the expression requires a temporary as part of its
       * evaluation, must make sure that the temp is freed before
       * the IF statement.  Insert a dummy statement, then delete it.
       */
      astnew = mk_stmt(A_CONTINUE, 0);
      stdnew = add_stmt_before(astnew, std);
      arg_gbl.std = stdnew;

      difficult.continue_std = stdnew;
      difficult.func_std = std;
      switch (type) {
        extern int assign_scalar(int, int); /* vsub.c */
      case A_IF:
      case A_ELSEIF:
      case A_AIF:
      case A_DOWHILE:
      case A_IFTHEN:
        a = rewrite_sub_ast(A_IFEXPRG(ast), 0);
        A_IFEXPRP(ast, a);
        break;
      case A_CGOTO:
        a = rewrite_sub_ast(A_LOPG(ast), 0);
        A_LOPP(ast, a);
        break;
      case A_DO:
      case A_MP_PDO:
        a = rewrite_sub_ast(A_M1G(ast), 0);
        if (a && contains_call(a)) {
          a = assign_scalar(std, a);
        }
        A_M1P(ast, a);
        a = rewrite_sub_ast(A_M2G(ast), 0);
        if (a && contains_call(a)) {
          a = assign_scalar(std, a);
        }
        A_M2P(ast, a);
        a = rewrite_sub_ast(A_M3G(ast), 0);
        if (a && contains_call(a)) {
          a = assign_scalar(std, a);
        }
        A_M3P(ast, a);
        a = rewrite_sub_ast(A_M4G(ast), 0);
        if (a && contains_call(a)) {
          a = assign_scalar(std, a);
        }
        A_M4P(ast, a);
        if (type == A_MP_PDO) {
          a = rewrite_sub_ast(A_LASTVALG(ast), 0);
          if (a && contains_call(a)) {
            a = assign_scalar(std, a);
          }
          A_LASTVALP(ast, a);
        }
        break;
      default:
        interr("rewrite_calls: unknown control ", ast, 4);
        break;
      }
      difficult.continue_std = difficult.func_std = 0;
      /* unlink the dummy statement */
      STD_NEXT(STD_PREV(stdnew)) = STD_NEXT(stdnew);
      STD_PREV(STD_NEXT(stdnew)) = STD_PREV(stdnew);
      arg_gbl.std = std;
      break;
    case A_ICALL:
      if (rewrite_sub_ast(ast, 0) != -1)
        ast_to_comment(ast);
      break;
    case A_CALL:
      if (elemental_func_call(ast)) {
        if (rewrite_sub_ast(ast, 0) != -1)
          ast_to_comment(ast);
      } else {
        args = rewrite_sub_args(ast, 0);
        A_ARGSP(ast, args);
      }
      break;
    case A_ALLOC:
      if (A_TKNG(ast) == TK_DEALLOCATE && !A_DALLOCMEMG(ast)) {
        if (A_TYPEG(A_SRCG(ast)) == A_SUBSCR) {
          A_SRCP(ast, A_LOPG(A_SRCG(ast)));
        }
        sptr_lhs = memsym_of_ast(A_SRCG(ast));
        if (allocatable_member(sptr_lhs)) {
          /* Has allocatable members but item itself is not
           * allocatable nor pointer
           */
          bool lhs_not_allocatable =
            !ALLOCG(sptr_lhs) && !ALLOCATTRG(sptr_lhs) && !POINTERG(sptr_lhs);
          /* HCCSYMG() checks if sptr_lhs is a compiler-generated temporary
           * variable or not. It is used inside rewrite_allocatable_assignment()
           * to guard handle_allocatable_members() which performs rewriting.
           * As a result, compiler-generated temporary allocatable variables
           * will only undergo shallow copying. HCCSYMG() is used here to
           * prevent deep deallocation for temporary allocatable variables.
           */
          if (!HCCSYMG(sptr_lhs) || lhs_not_allocatable)
            rewrite_deallocate(A_SRCG(ast), false, std);
          if (lhs_not_allocatable)
            nop_dealloc(sptr_lhs, ast);
        }
      } else if (A_TKNG(ast) == TK_ALLOCATE) {
        int a, sptr2, astmem;
        sptr_lhs = memsym_of_ast(A_SRCG(ast));
        if (STYPEG(sptr_lhs) == ST_MEMBER) {
          astmem = A_SRCG(ast);
        } else {
          astmem = 0;
        }
        switch (A_TYPEG(A_STARTG(ast))) {
        case A_ID:
        case A_LABEL:
        case A_ENTRY:
        case A_SUBSCR:
        case A_SUBSTR:
        case A_MEM:
          sptr2 = (A_STARTG(ast)) ? memsym_of_ast(A_STARTG(ast)) : 0;
          break;
        default:
          sptr2 = 0;
        }
        if (sptr2 > NOSYM &&
            (CLASSG(sptr2) || (CLASSG(sptr_lhs) && ALLOCATTRG(sptr2)))) {
          check_pointer_type(A_SRCG(ast), A_STARTG(ast), std, 1);
        } else {
          a = A_DTYPEG(ast);
          if (DTY(a) == TY_ARRAY)
            a = DTY(a + 1);

          if (CLASSG(sptr_lhs) || ALLOCDESCG(sptr_lhs) ||
              has_tbp_or_final(DTYPEG(sptr_lhs)) || has_tbp_or_final(a) ||
              is_or_has_poly(sptr_lhs) ||
              has_length_type_parameter_use(DTYPEG(sptr_lhs)) ||
              (sptr2 && !ALLOCATTRG(sptr_lhs) && has_poly_mbr(sptr_lhs, 1))) {
            int alloc_source;
            DTYPE source_dtype;
            check_alloc_ptr_type(sptr_lhs, std, a, 1, 1, A_SRCG(ast), astmem);
            alloc_source = A_STARTG(ast);
            source_dtype = DTYPEG(sptr2);
            if (alloc_source > 0 && (DTY(DDTG(source_dtype)) == TY_CHAR ||
                DTY(DDTG(source_dtype)) == TY_NCHAR)) {
              /* This is a sourced allocation with a character source argument. 
               * Need to make sure we assign the character object's length to
               * the receiver's descriptor.
               */
              int len = ast_intr(I_LEN, astb.bnd.dtype, 1, 
                                 A_TYPEG(alloc_source) == A_SUBSCR ?
                                 A_LOPG(alloc_source) : alloc_source);
              len = gen_set_len_ast(A_SRCG(ast), SDSCG(sptr_lhs), len);
              add_stmt_after(len, std);
            }
          }
        }
      }
      break;
    case A_ELSEWHERE:
    case A_ENDWHERE:
    case A_END:
    case A_STOP:
    case A_RETURN:
    case A_ELSE:
    case A_ENDIF:
    case A_ENDDO:
    case A_CONTINUE:
    case A_GOTO:
    case A_ASNGOTO:
    case A_AGOTO:
    case A_ENTRY:
    case A_PAUSE:
    case A_COMMENT:
    case A_COMSTR:
    case A_REDISTRIBUTE:
    case A_REALIGN:
    case A_HCFINISH:
    case A_MASTER:
    case A_ENDMASTER:
    case A_CRITICAL:
    case A_ENDCRITICAL:
    case A_ATOMIC:
    case A_ATOMICCAPTURE:
    case A_ATOMICREAD:
    case A_ATOMICWRITE:
    case A_ENDATOMIC:
    case A_BARRIER:
    case A_NOBARRIER:
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
    case A_MP_ENDPDO:
    case A_MP_ENDSECTIONS:
    case A_MP_WORKSHARE:
    case A_MP_ENDWORKSHARE:
    case A_MP_BPDO:
    case A_MP_EPDO:
    case A_MP_SECTION:
    case A_MP_LSECTION:
    case A_MP_PRE_TLS_COPY:
    case A_MP_BCOPYIN:
    case A_MP_COPYIN:
    case A_MP_ECOPYIN:
    case A_MP_BCOPYPRIVATE:
    case A_MP_COPYPRIVATE:
    case A_MP_ECOPYPRIVATE:
    case A_MP_EMPSCOPE:
    case A_MP_BORDERED:
    case A_MP_EORDERED:
    case A_MP_FLUSH:
    case A_MP_TASKGROUP:
    case A_MP_ETASKGROUP:
    case A_MP_DISTRIBUTE:
    case A_MP_ENDDISTRIBUTE:
    case A_MP_ENDTARGETDATA:
    case A_MP_TASKREG:
    case A_MP_TASKDUP:
    case A_MP_ETASKDUP:
      break;
    case A_MP_TASKLOOPREG:
    case A_MP_ETASKLOOPREG:
      break;
    case A_MP_TASK:
    case A_MP_TASKLOOP:
      a = rewrite_sub_ast(A_IFPARG(ast), 0);
      A_IFPARP(ast, a);
      a = rewrite_sub_ast(A_FINALPARG(ast), 0);
      A_FINALPARP(ast, a);
      a = rewrite_sub_ast(A_PRIORITYG(ast), 0);
      A_PRIORITYP(ast, a);
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
    case A_MP_BMPSCOPE:
      a = rewrite_sub_ast(A_STBLKG(ast), 0);
      A_STBLKP(ast, a);
      break;
    case A_MP_TASKFIRSTPRIV:
      a = rewrite_sub_ast(A_LOPG(ast), 0);
      A_LOPP(ast, a);
      a = rewrite_sub_ast(A_ROPG(ast), 0);
      A_ROPP(ast, a);
      break;
    case A_MP_PARALLEL:
      a = rewrite_sub_ast(A_IFPARG(ast), 0);
      A_IFPARP(ast, a);
      a = rewrite_sub_ast(A_NPARG(ast), 0);
      A_NPARP(ast, a);
      /* proc_bind is constant
      a = rewrite_sub_ast(A_PROCBINDG(ast), 0);
      A_PROCBINDP(ast, a);
      */
      ++parallel_depth;
      /*symutl.sc = SC_PRIVATE;*/
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_TEAMS:
      a = rewrite_sub_ast(A_NTEAMSG(ast), 0);
      A_NTEAMSP(ast, a);
      a = rewrite_sub_ast(A_THRLIMITG(ast), 0);
      A_THRLIMITP(ast, a);
      break;
    case A_MP_ENDPARALLEL:
      --parallel_depth;
      if (parallel_depth == 0 && task_depth == 0) {
        /*symutl.sc = SC_LOCAL;*/
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    case A_MP_ATOMICREAD:
      a = rewrite_sub_ast(A_SRCG(ast), 0);
      A_SRCP(ast, a);
      break;
    case A_MP_ATOMICWRITE:
    case A_MP_ATOMICUPDATE:
    case A_MP_ATOMICCAPTURE:
      a = rewrite_sub_ast(A_LOPG(ast), 0);
      A_LOPP(ast, a);
      a = rewrite_sub_ast(A_ROPG(ast), 0);
      A_ROPP(ast, a);
      break;
    case A_MP_ENDTEAMS:
    case A_MP_ENDTARGET:
    case A_MP_TARGET:
      break;
    case A_MP_CANCEL:
      a = rewrite_sub_ast(A_IFPARG(ast), 0);
      A_IFPARP(ast, a);
      FLANG_FALLTHROUGH;
    case A_MP_SECTIONS:
    case A_MP_CANCELLATIONPOINT:
      a = rewrite_sub_ast(A_ENDLABG(ast), 0);
      A_ENDLABP(ast, a);
      break;
    case A_MP_TARGETDATA:
    case A_MP_TARGETENTERDATA:
    case A_MP_TARGETEXITDATA:
    case A_MP_TARGETUPDATE:
      a = rewrite_sub_ast(A_IFPARG(ast), 0);
      A_IFPARP(ast, a);
      break;
    case A_FORALL:
      arg_gbl.used = TRUE; /* don't use lhs for intrinsics */
      arg_gbl.inforall = TRUE;
      src = A_SRCG(ast);
      prevstd = STD_PREV(std);
      a = rewrite_sub_ast(A_IFEXPRG(ast), 0);
      A_IFEXPRP(ast, a);
      rewrite_asn(A_IFSTMTG(ast), std, TRUE, 0);
      arg_gbl.inforall = FALSE;

      /* there is no std created  from forall before, if it is
       * created now, show the first one */
      if (src == 0 && STD_PREV(std) != prevstd) {
        A_SRCP(ast, STD_NEXT(prevstd));
        assert(STD_NEXT(prevstd) != std, "rewrite_calls: something is wrong",
               std, 3);
      }
      break;
    case A_HLOCALIZEBNDS:
    case A_HCYCLICLP:
      lhs = A_LOPG(ast);
      assert(A_TYPEG(lhs) == A_ID, "rewrite_calls: id not found", ast, 3);
      sptr_lhs = A_SPTRG(lhs);
      assert(STYPEG(sptr_lhs) == ST_ARRDSC || STYPEG(sptr_lhs) == ST_ARRAY,
             "rewrite_calls: array not found", ast, 3);
      break;
    case A_HGETSCLR:
    case A_HOWNERPROC:
      break;
    case A_PREFETCH:
      break;
    case A_PRAGMA:
      a = rewrite_sub_ast(A_LOPG(ast), 0);
      A_LOPP(ast, a);
      a = rewrite_sub_ast(A_ROPG(ast), 0);
      A_ROPP(ast, a);
      break;
    case A_MP_EMAP:
    case A_MP_MAP:
    case A_MP_TARGETLOOPTRIPCOUNT:
    case A_MP_EREDUCTION:
    case A_MP_BREDUCTION:
    case A_MP_REDUCTIONITEM:
      break;
    default:
      interr("rewrite_subroutine: unknown stmt found", ast, 4);
      break;
    }
  }
}

static void
nop_dealloc(int sptr, int ast)
{
  if (SCG(sptr) == SC_LOCAL && AUTOBJG(sptr) && has_allocattr(sptr))
    return;
  ast_to_comment(ast);
}

/*
 *  call pghpf_reduce_descriptor(result$sd, kind, len, array$sd, dim)
 *
 *  set up result descriptor for reduction intrinsic -- used when the
 *  dim arg is variable.  result dimensions are aligned with the
 *  corresponding source dimensions and the result array becomes
 * replicated over the reduction dimension.
 */

static void
add_reduce_descriptor(int temp_sptr, int arr_sptr, int arr_ast, int dim)
{
  DTYPE dtype = DTYPEG(temp_sptr);
  int kind = mk_cval(dtype_to_arg(DTY(dtype + 1)), astb.bnd.dtype);
  int len = size_ast(temp_sptr, DDTG(dtype));
  int sptrFunc = sym_mkfunc_nodesc(mkRteRtnNm(RTE_reduce_descriptor), 0);
  int astStmt = begin_call(A_CALL, sptrFunc, 5);
  add_arg(mk_id(DESCRG(temp_sptr)));
  add_arg(kind);
  add_arg(len);
  add_arg(check_member(arr_ast, mk_id(DESCRG(arr_sptr))));
  add_arg(convert_int(dim, astb.bnd.dtype));
  add_stmt_before(astStmt, arg_gbl.std);
}

/* call pghpf_spread_descriptor(result$sd, source$sd, dim, ncopies)
 *
 * set up result descriptor for spread intrinsic -- used when the dim
 * arg is variable.  the added spread dimension is given a collapsed
 * distribution and the remaining dimensions are aligned with the
 * corresponding source dimensions.  overlap allowances are set to
 * zero.
 */

static void
add_spread_descriptor(int temp_sptr, int arr_sptr, int arr_ast, int dim,
                      int ncopies)
{
  int sptrFunc;
  int astStmt;

  dim = convert_int(dim, astb.bnd.dtype);
  ncopies = convert_int(ncopies, astb.bnd.dtype);
  sptrFunc = sym_mkfunc_nodesc(mkRteRtnNm(RTE_spread_descriptor), 0);
  astStmt = begin_call(A_CALL, sptrFunc, 4);
  add_arg(mk_id(DESCRG(temp_sptr)));
  add_arg(check_member(arr_ast, mk_id(DESCRG(arr_sptr))));
  add_arg(dim);
  add_arg(ncopies);
  add_stmt_before(astStmt, arg_gbl.std);
}

/** \brief Make a temporary to be used as the argument to an intrinsic that
    returns an array.
    \param func_ast  ast for the intrinsic call
    \param func_args rewritten args for the function
    \param subscr    returned subscripts
    \param elem_dty  data type of elements
    \param lhs       passed lhs or zero
    \param retval    returned ast for lhs

    The actual size of this temp is derived from the
    arguments to the intrinsic.  The subscripts of the temp may not
    be the entire temp; this is derived from the arguments as well.

    If lhs is non-zero, check lhs to see if it is OK for the intended
    use; if so, return 0.
 */
static int
mk_result_sptr(int func_ast, int func_args, int *subscr, int elem_dty, int lhs,
               int *retval)
{
  int temp_sptr = 0;
  int dim;
  int shape;
  int shape1;
  int rank, rank1;
  int arg;
  int ncopies;

  shape = A_SHAPEG(func_ast);
  switch (A_OPTYPEG(func_ast)) {
  case I_MINLOC:
  case I_MAXLOC:
  case I_FINDLOC:
  case I_ALL:
  case I_ANY:
  case I_COUNT:
  case I_MAXVAL:
  case I_MINVAL:
  case I_PRODUCT:
  case I_SUM:
  case I_NORM2:
    arg = ARGT_ARG(func_args, 0);
    /* first arg with dimension removed */
    dim = A_OPTYPEG(func_ast) == I_FINDLOC ? ARGT_ARG(func_args, 2)
                                           : ARGT_ARG(func_args, 1);
    assert(dim != 0, "mk_result_sptr: dim must be constant", 0, 4);
    /* We know that the first argument is an array section or whole
     * array, so we can squeeze out the dimension & just use the
     * existing subscripts.
     */
    temp_sptr = chk_reduc_sptr(arg, "r", subscr, elem_dty, dim, lhs, retval);
    /* non-constant DIM */
    if (!A_ALIASG(dim) && temp_sptr && A_SHAPEG(arg)) {
      int array, arrayast;
      array = find_array(arg, &arrayast);
      add_reduce_descriptor(temp_sptr, array, arrayast, dim);
    }

    /* make the subscripts for the result */
    break;
  case I_UNPACK:
    /* mask (second arg) */
    arg = ARGT_ARG(func_args, 1);
    goto easy;
  case I_CSHIFT:
  case I_EOSHIFT:
    arg = ARGT_ARG(func_args, 0);
    while (A_TYPEG(arg) == A_INTR &&
           (A_OPTYPEG(arg) == I_CSHIFT || A_OPTYPEG(arg) == I_EOSHIFT)) {
      int fargs = A_ARGSG(arg);
      arg = ARGT_ARG(fargs, 0);
    }
    if (lhs == 0)
      goto easy;
    rank = SHD_NDIM(shape);
    if (arg_gbl.lhs) {
      shape1 = A_SHAPEG(arg_gbl.lhs);
      rank1 = SHD_NDIM(shape1);
      if (rank == rank1 && !arg_gbl.used &&
          DTY(A_DTYPEG(arg_gbl.lhs) + 1) == elem_dty) {
        *retval = arg_gbl.lhs;
        temp_sptr = 0;
        arg_gbl.used = TRUE;
        break;
      }
      if (rank == rank1) {
        temp_sptr =
            chk_assign_sptr(arg_gbl.lhs, "r", subscr, elem_dty, lhs, retval);
        break;
      }
    }
    goto easy;

  easy:
    if (ast_has_allocatable_member(lhs)) {
      goto temp_from_shape;
    }
    temp_sptr = chk_assign_sptr(arg, "r", subscr, elem_dty, lhs, retval);
    break;
  case I_SPREAD:
    /* first arg with dimension added */
    arg = ARGT_ARG(func_args, 0);
    dim = ARGT_ARG(func_args, 1);
    ncopies = ARGT_ARG(func_args, 2);
    assert(dim != 0, "mk_result_sptr: dim must be constant", 0, 4);

    temp_sptr =
        mk_spread_sptr(arg, "r", subscr, elem_dty, dim, ncopies, lhs, retval);
    /* non-constant DIM */
    if (!A_ALIASG(dim) && temp_sptr && A_SHAPEG(arg)) {
      int array, arrayast;
      array = find_array(arg, &arrayast);
      add_spread_descriptor(temp_sptr, array, arrayast, dim, ncopies);
    }

    break;
  case I_MATMUL:
  case I_MATMUL_TRANSPOSE:
    rank = SHD_NDIM(shape);
    if (matmul_use_lhs(lhs, rank, elem_dty)) {
      *retval = arg_gbl.lhs;
      temp_sptr = 0;
      arg_gbl.used = TRUE;
      break;
    }
    if (A_OPTYPEG(func_ast) == I_MATMUL_TRANSPOSE) {
      /* NOTE: this assumes that I_MATMUL_TRANSPOSE is
       * generated for the transpose of the first arg only
       */
      int tmp_shape = A_SHAPEG(ARGT_ARG(func_args, 0));
      arg = mk_id(mk_shape_sptr(tmp_shape, subscr, elem_dty));
      arg = mk_id(mk_transpose_sptr(arg, "r", subscr, elem_dty, retval));
    } else {
      arg = ARGT_ARG(func_args, 0);
    }

    /* first and second args */
    temp_sptr = mk_matmul_sptr(arg, ARGT_ARG(func_args, 1), "r", subscr,
                               elem_dty, retval);
    break;
  case I_TRANSPOSE:
    /* first arg */
    goto temp_from_shape;
  case I_PACK:
    /* problem */
    /* just make a 1-d temp with the appropriate size and no dist */
    temp_sptr = mk_pack_sptr(shape, elem_dty);
    subscr[0] = mk_triple(SHD_LWB(shape, 0), SHD_UPB(shape, 0), 0);
    *retval = mk_id(temp_sptr);
    break;
  case I_RESHAPE:
  case I_TRANSFER:
  temp_from_shape:
    /* make a temp out of the shape, no distribution */
    temp_sptr = mk_shape_sptr(shape, subscr, elem_dty);
    *retval = mk_id(temp_sptr);
    break;
  default:
    interr("mk_result_sptr: can't handle intrinsic", func_ast, 4);
    break;
  }
  return temp_sptr;
}

static LOGICAL
matmul_use_lhs(int lhs, int rank, int elem_dty)
{
  if (lhs && arg_gbl.lhs) {
    /*
     * the LHS cannot be a member whose shape comes froms a parent
     */
    int array;
    if (A_TYPEG(arg_gbl.lhs) == A_MEM && A_SHAPEG(A_PARENTG(arg_gbl.lhs)) != 0)
      return FALSE;
    /*
     * the LHS cannot be an allocatable if -Mallocatable=03 is enabled
     */
    array = find_array(arg_gbl.lhs, NULL);
    if (XBIT(54, 0x1) && ALLOCATTRG(array))
      return FALSE;
    if (rank == SHD_NDIM(A_SHAPEG(arg_gbl.lhs)) && arg_gbl.used == 0 &&
        DTY(A_DTYPEG(arg_gbl.lhs) + 1) == elem_dty) {
      return TRUE;
    }
  }
  return FALSE;
}

int
search_conform_array(int ast, int flag)
{
  int i;
  int argt;
  int nargs;
  int j;

  switch (A_TYPEG(ast)) {
  case A_SUBSCR:
    if (A_SHAPEG(ast) != 0 && flag &&
        (A_TYPEG(A_LOPG(ast)) == A_ID || A_TYPEG(A_LOPG(ast)) == A_MEM))
      return ast;
    return 0;
  case A_SUBSTR:
    return search_conform_array(A_LOPG(ast), flag);
  case A_ID:
    if (A_SHAPEG(ast))
      return ast;
    return 0;
  case A_BINOP:
    i = search_conform_array(A_LOPG(ast), flag);
    if (i != 0)
      return i;
    return search_conform_array(A_ROPG(ast), flag);
  case A_UNOP:
  case A_CONV:
    return search_conform_array(A_LOPG(ast), flag);
  case A_MEM:
    if (A_SHAPEG(A_MEMG(ast)))
      return ast;
    return search_conform_array(A_PARENTG(ast), flag);
  case A_INTR:
    argt = A_ARGSG(ast);
    nargs = A_ARGCNTG(ast);
    if (INKINDG(A_SPTRG(A_LOPG(ast))) != IK_ELEMENTAL) {
      switch (A_OPTYPEG(ast)) {
      case I_CSHIFT:
      case I_EOSHIFT:
        return search_conform_array(ARGT_ARG(argt, 0), flag);
      case I_SPREAD:
      case I_SUM:
      case I_PRODUCT:
      case I_MAXVAL:
      case I_MINVAL:
      case I_DOT_PRODUCT:
      case I_ALL:
      case I_ANY:
      case I_COUNT:
        return ast;
      case I_TRANSPOSE:
        return ast;
      default:
        return 0;
      }
    }
    for (i = 0; i < nargs; ++i)
      if (A_SHAPEG(ARGT_ARG(argt, i)))
        if ((j = search_conform_array(ARGT_ARG(argt, i), flag)) != 0)
          return j;
    FLANG_FALLTHROUGH;
  case A_FUNC:
    if (elemental_func_call(ast)) {
      /* search up to all arguments of elemental function for
       * a conformable array -- not just the first argument.
       */
      argt = A_ARGSG(ast);
      nargs = A_ARGCNTG(ast);
      for (i = 0; i < nargs; ++i) {
        if ((j = search_conform_array(ARGT_ARG(argt, i), flag)))
          return j;
      }
    }
    return 0;
  default:
    return 0;
  }
}

/* Pointer association status (logical function):
 * associated(pv [, target] )
 * external pghpf_associated
 * logical  pghpf_associated
 * ( pghpf_associated(pv, pv$sdsc, target, target$d) )
 */
static int
transform_associated(int std, int ast)
{
  int ast1;
  int argt, nargs;
  int pv, arr;
  int pv_sptr = 0, arr_sptr;
  int arr_desc, static_desc;
  int dtype;
  int func;
  int ty;
  int with_target;

  assert(A_TYPEG(ast) == A_INTR && A_OPTYPEG(ast) == I_ASSOCIATED,
         "transform_associated: not ASSOCIATED call", 2, ast);

  with_target = 0;
  argt = A_ARGSG(ast);
  nargs = A_ARGCNTG(ast);
  assert(nargs == 2,
         "transform_associated: ASSOCIATED with wrong number arguments", 2,
         ast);
  pv = ARGT_ARG(argt, 0);
  arr = ARGT_ARG(argt, 1);
  arr_desc = 0;
  assert(A_TYPEG(pv) == A_ID || A_TYPEG(pv) == A_MEM,
         "transform_associated: ASSOCIATED(V) where V is not an ID", 2, ast);
  if (A_TYPEG(pv) == A_ID) {
    pv_sptr = A_SPTRG(pv);
  } else if (A_TYPEG(pv) == A_MEM) {
    pv_sptr = A_SPTRG(A_MEMG(pv));
  }
  dtype = DTYPEG(pv_sptr);
  DESCUSEDP(pv_sptr, 1);

  arr_sptr = 0;
  if (arr) {
    switch (A_TYPEG(arr)) {
    case A_SUBSCR:
    case A_MEM:
    case A_ID:
      arr_sptr = memsym_of_ast(arr);
      break;
    default:
      assert(0, "transform_associated: ASSOCIATED(V,P) where P is not an ID", 2,
             ast);
    }
  }

  if (!arr)
    return ast;
  /* if this is an undistributed scalar pointer,
   * and there is no array 2nd argument, leave this as it is */
  if (DTY(dtype) != TY_ARRAY) {
    /* 2nd argument must also be scalar */
    switch (A_TYPEG(arr)) {
    case A_ID:
    case A_MEM:
      /* must not be an array */
      if (DTY(DTYPEG(arr_sptr)) != TY_ARRAY)
        return ast;
      break;
    }
  }

  if (arr) {
    with_target = 1;
    DESCUSEDP(arr_sptr, 1);
    if (A_SHAPEG(arr) && A_TYPEG(arr) == A_SUBSCR) {
      arr_desc = mk_id(make_sec_from_ast(arr, std, std, 0, 0));
      arr = A_LOPG(arr);
    } else if (A_SHAPEG(arr) && (A_TYPEG(arr) == A_ID || A_TYPEG(arr) == A_MEM))
      arr_desc = mk_id(DESCRG(arr_sptr));
    else {
      ty = dtype_to_arg(A_DTYPEG(arr));
      arr_desc = mk_isz_cval(ty, astb.bnd.dtype);
    }
  } else {
    if (DTYG(dtype) == TY_CHAR)
      arr = astb.ptr0c;
    else
      arr = astb.ptr0;
    arr_desc = astb.bnd.one;
  }
  assert(arr_desc, "transform_associated: need descriptor", 2, arr);

  if (!POINTERG(pv_sptr))
    error(506, 3, STD_LINENO(std), SYMNAME(pv_sptr), CNULL);
  static_desc = mk_id(SDSCG(pv_sptr));

  nargs = 4;
  if (XBIT(70, 0x20)) {
    if (MIDNUMG(pv_sptr))
      ++nargs;
    if (PTROFFG(pv_sptr))
      ++nargs;
  }
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = pv;
  ARGT_ARG(argt, 1) = check_member(pv, static_desc);
  ARGT_ARG(argt, 2) = arr;
  ARGT_ARG(argt, 3) = check_member(arr, arr_desc);
  nargs = 4;
  if (XBIT(70, 0x20)) {
    if (MIDNUMG(pv_sptr)) {
      ARGT_ARG(argt, nargs) = check_member(pv, mk_id(MIDNUMG(pv_sptr)));
      ++nargs;
    }
    if (PTROFFG(pv_sptr)) {
      ARGT_ARG(argt, nargs) = check_member(pv, mk_id(PTROFFG(pv_sptr)));
      ++nargs;
    }
  }

  if (with_target) {
    if (DTYG(dtype) == TY_CHAR)
      func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_associated_tchara), DT_LOG));
    else
      func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_associated_t), DT_LOG));
  } else {
    if (DTYG(dtype) == TY_CHAR)
      func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_associated_chara), DT_LOG));
    else
      func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_associated), DT_LOG));
  }

  ast1 = mk_func_node(A_FUNC, func, nargs, argt);

  NODESCP(A_SPTRG(A_LOPG(ast1)), 1);
  A_DTYPEP(ast1, DT_LOG);
  return ast1;
}

/* func_ast: A_FUNC or A_INTR */
/* func_args: rewritten args */
static void
transform_mvbits(int func_ast, int func_args)
{
  int lb, ub, st;
  int forall, dovar;
  int ast;
  int lineno;
  int stdnext, std;
  int newast;
  int to;
  int shape;
  int i, n;
  int triplet_list, index_var;
  int triplet;
  int newargt;
  int nargs;

  assert(A_TYPEG(func_ast) == A_ICALL && A_OPTYPEG(func_ast) == I_MVBITS,
         "transform_mvbits: something is wrong", 2, func_ast);

  stdnext = arg_gbl.std;
  lineno = STD_LINENO(stdnext);

  to = ARGT_ARG(func_args, 3);
  shape = A_SHAPEG(to);
  if (!shape) {
    return;
  }

  forall = make_forall(shape, to, 0, 0);

  n = 0;
  triplet_list = A_LISTG(forall);
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    n++;
    newast = mk_stmt(A_DO, 0);
    index_var = ASTLI_SPTR(triplet_list);
    triplet = ASTLI_TRIPLE(triplet_list);
    dovar = mk_id(index_var);
    A_DOVARP(newast, dovar);
    lb = A_LBDG(triplet);
    ub = A_UPBDG(triplet);
    st = A_STRIDEG(triplet);

    A_M1P(newast, lb);
    A_M2P(newast, ub);
    A_M3P(newast, st);
    A_M4P(newast, 0);

    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }

  nargs = 5;
  newargt = mk_argt(nargs);

  for (i = 0; i < 5; i++) {
    ast = ARGT_ARG(func_args, i);
    ast = normalize_forall(forall, ast, 0);
    ARGT_ARG(newargt, i) = ast;
  }

  newast = mk_func_node(A_ICALL, A_LOPG(func_ast), nargs, newargt);
  A_OPTYPEP(newast, A_OPTYPEG(func_ast));
  std = add_stmt_before(newast, stdnext);
  STD_LINENO(std) = lineno;
  STD_PAR(std) = STD_PAR(stdnext);
  STD_TASK(std) = STD_TASK(stdnext);
  STD_ACCEL(std) = STD_ACCEL(stdnext);
  STD_KERNEL(std) = STD_KERNEL(stdnext);

  for (i = 0; i < n; i++) {
    newast = mk_stmt(A_ENDDO, 0);
    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }
  delete_stmt(arg_gbl.std);
}

/* func_ast: A_FUNC or A_INTR */
/* func_args: rewritten args */
static void
transform_merge(int func_ast, int func_args)
{
  int lb, ub, st;
  int forall, dovar;
  int ast;
  int lineno;
  int stdnext, std;
  int newast;
  int temp;
  int shape;
  int i, n;
  int triplet_list, index_var;
  int triplet;
  int newargt;
  int nargs;

  assert(A_TYPEG(func_ast) == A_ICALL && A_OPTYPEG(func_ast) == I_MERGE,
         "transform_merge: something is wrong", 2, func_ast);

  stdnext = arg_gbl.std;
  lineno = STD_LINENO(stdnext);

  temp = ARGT_ARG(func_args, 0);
  shape = A_SHAPEG(temp);
  if (!shape) {
    A_TYPEP(func_ast, A_CALL);
    return;
  }

  forall = make_forall(shape, temp, 0, 0);

  n = 0;
  triplet_list = A_LISTG(forall);
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    n++;
    newast = mk_stmt(A_DO, 0);
    index_var = ASTLI_SPTR(triplet_list);
    triplet = ASTLI_TRIPLE(triplet_list);
    dovar = mk_id(index_var);
    A_DOVARP(newast, dovar);
    lb = A_LBDG(triplet);
    ub = A_UPBDG(triplet);
    st = A_STRIDEG(triplet);

    A_M1P(newast, lb);
    A_M2P(newast, ub);
    A_M3P(newast, st);
    A_M4P(newast, 0);

    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }

  nargs = ARGT_CNT(func_args);
  newargt = mk_argt(nargs);

  for (i = 0; i < nargs; i++) {
    ast = ARGT_ARG(func_args, i);
    ast = normalize_forall(forall, ast, 0);
    ARGT_ARG(newargt, i) = ast;
  }

  newast = mk_func_node(A_CALL, A_LOPG(func_ast), nargs, newargt);
  A_OPTYPEP(newast, A_OPTYPEG(func_ast));
  std = add_stmt_before(newast, stdnext);
  STD_LINENO(std) = lineno;
  STD_PAR(std) = STD_PAR(stdnext);
  STD_TASK(std) = STD_TASK(stdnext);
  STD_ACCEL(std) = STD_ACCEL(stdnext);
  STD_KERNEL(std) = STD_KERNEL(stdnext);

  for (i = 0; i < n; i++) {
    newast = mk_stmt(A_ENDDO, 0);
    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }
  delete_stmt(arg_gbl.std);
}

static void
transform_elemental(int func_ast, int func_args)
{
  int lb, ub, st;
  int forall, dovar;
  int ast;
  int lineno;
  int stdnext, std;
  int newast;
  int temp;
  int shape;
  int i, n;
  int triplet_list, index_var;
  int triplet;
  int newargt;
  int nargs;

  assert(A_TYPEG(func_ast) == A_CALL && elemental_func_call(func_ast),
         "transform_merge: something is wrong", func_ast, 3);

  stdnext = arg_gbl.std;
  lineno = STD_LINENO(stdnext);

  temp = ARGT_ARG(func_args, 0);
  shape = extract_shape_from_args(func_ast);
  if (!shape) {
    A_TYPEP(func_ast, A_CALL);
    return;
  }

  forall = make_forall(shape, temp, 0, 0);

  n = 0;
  triplet_list = A_LISTG(forall);
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    n++;
    newast = mk_stmt(A_DO, 0);
    index_var = ASTLI_SPTR(triplet_list);
    triplet = ASTLI_TRIPLE(triplet_list);
    dovar = mk_id(index_var);
    A_DOVARP(newast, dovar);
    lb = A_LBDG(triplet);
    ub = A_UPBDG(triplet);
    st = A_STRIDEG(triplet);

    A_M1P(newast, lb);
    A_M2P(newast, ub);
    A_M3P(newast, st);
    A_M4P(newast, 0);

    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }

  nargs = ARGT_CNT(func_args);
  newargt = mk_argt(nargs);

  for (i = 0; i < nargs; i++) {
    ast = ARGT_ARG(func_args, i);
    ast = normalize_forall(forall, ast, 0);
    ARGT_ARG(newargt, i) = ast;
  }

  newast = mk_func_node(A_CALL, A_LOPG(func_ast), nargs, newargt);
  A_OPTYPEP(newast, A_OPTYPEG(func_ast));
  A_INVOKING_DESCP(newast, A_INVOKING_DESCG(func_ast));
  std = add_stmt_before(newast, stdnext);
  STD_LINENO(std) = lineno;
  STD_PAR(std) = STD_PAR(stdnext);
  STD_TASK(std) = STD_TASK(stdnext);
  STD_ACCEL(std) = STD_ACCEL(stdnext);
  STD_KERNEL(std) = STD_KERNEL(stdnext);

  for (i = 0; i < n; i++) {
    newast = mk_stmt(A_ENDDO, 0);
    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }
  delete_stmt(arg_gbl.std);
}

/* move_alloc(from, to) */
static void
transform_move_alloc(int func_ast, int func_args)
{
  int std;
  int pvar, pvar2;
  int desc, desc2;
  SPTR sptr, sptr2;
  int func, nargs, newast, newargt;
  int stdnext = arg_gbl.std;
  int lineno = STD_LINENO(stdnext);
  int fptr = ARGT_ARG(func_args, 0);
  int fptr2 = ARGT_ARG(func_args, 1);

  move_alloc_arg(fptr, &sptr, &pvar);
  move_alloc_arg(fptr2, &sptr2, &pvar2);

  desc = find_descriptor_ast(sptr, fptr);
  assert(desc, "transform_move_alloc: invalid 'from' descriptor", sptr,
         ERR_Fatal);
  desc2 = find_descriptor_ast(sptr2, fptr2);
  assert(desc2, "transform_move_alloc: invalid 'to' descriptor", sptr2,
         ERR_Fatal);

  func = mk_id(sym_mkfunc_nodesc_expst(mkRteRtnNm(RTE_move_alloc), DT_INT));
  nargs = 4;
  newargt = mk_argt(nargs);
  ARGT_ARG(newargt, 0) = pvar;  /* from ptr */
  ARGT_ARG(newargt, 1) = desc;  /* from descriptor */
  ARGT_ARG(newargt, 2) = pvar2; /* to ptr */
  ARGT_ARG(newargt, 3) = desc2; /* to descriptor */
  newast = mk_func_node(A_CALL, func, nargs, newargt);
  std = add_stmt_before(newast, stdnext);

  STD_LINENO(std) = lineno;
  STD_PAR(std) = STD_PAR(stdnext);
  STD_TASK(std) = STD_TASK(stdnext);
  STD_ACCEL(std) = STD_ACCEL(stdnext);
  STD_KERNEL(std) = STD_KERNEL(stdnext);
  if (A_SHAPEG(fptr2) && sptr != sptr2 && !SDSCG(sptr2)) {
    int parent = STYPEG(sptr) == ST_MEMBER ? A_PARENTG(fptr) : 0;
    int parent2 = STYPEG(sptr2) == ST_MEMBER ? A_PARENTG(fptr2) : 0;
    copy_surrogate_to_bnds_vars(DTYPEG(sptr2), parent2, DTYPEG(sptr), parent,
                                STD_NEXT(std));
  }

  delete_stmt(arg_gbl.std);
}

static void
transform_c_f_pointer(int func_ast, int func_args)
{
  /*
   * c_f_pointer(cptr, fptr)        -- fptr is scalar
   * c_f_pointer(cptr, fptr, shape) -- fptr is array
   */
  int lineno;
  int stdnext, std;
  int newast;
  int rank;
  int fptr;
  int cptr, newcptrarg;
  int pvar = 0;
  int shape;
  int desc;
  int fty;
  int dtype;
  int func;
  int nargs;
  int newargt;
  int sptr;
  int shpty;
  int sz;

  stdnext = arg_gbl.std;
  lineno = STD_LINENO(stdnext);
  fptr = ARGT_ARG(func_args, 1);
  cptr = ARGT_ARG(func_args, 0);
  /*
   * pass the address of fptr$p instead of just referencing fptr.
   */
  dtype = A_DTYPEG(fptr);
  if (A_TYPEG(fptr) == A_ID)
    sptr = A_SPTRG(fptr);
  else if (A_TYPEG(fptr) == A_MEM)
    sptr = A_SPTRG(A_MEMG(fptr));
  else
    sptr = 0;
  if (sptr && MIDNUMG(sptr)) {
    pvar = check_member(fptr, mk_id(MIDNUMG(sptr)));
  } else {
    interr("FPTR error in c_f_pointer()", fptr, 4);
  }

  /* if argument:cptr does not have type(c_ptr), create a temporary
   * and assign its location to that temp.  Pass that temp to
   * c_f_pointer.
   */
  if (!is_iso_c_ptr(A_DTYPEG(cptr)) && !is_cuf_c_devptr(A_DTYPEG(cptr))) {
    DTYPE dt = get_iso_c_ptr();
    if (dt <= DT_NONE)
      interr("Error in c_f_pointer() - unable to find c_ptr type", fptr, 4);
    newcptrarg = mk_id(get_temp(dt));
    cptr = mk_unop(OP_LOC, cptr, DT_PTR);
    cptr = mk_assn_stmt(newcptrarg, cptr, dt);
    add_stmt_before(cptr, arg_gbl.std);
    cptr = newcptrarg;
  }

  shape = A_SHAPEG(fptr);
  if (!shape) { /* scalar */
    rank = 0;
    desc = astb.i0;
    shape = astb.i0;
    shpty = astb.i0;
  } else {
    /*
     * pass the address of fptr$sd
     */
    rank = SHD_NDIM(shape);
    if (SDSCG(sptr)) {
      desc = check_member(fptr, mk_id(SDSCG(sptr)));
      DESCUSEDP(sptr, 1);
      NODESCP(sptr, 0);
    } else {
      desc = check_member(fptr, mk_id(DESCRG(sptr)));
      DESCUSEDP(sptr, 1);
      NODESCP(sptr, 0);
    }
    shape = ARGT_ARG(func_args, 2);
    shpty = dtype_to_arg(DTY(A_DTYPEG(shape) + 1));
    shpty = mk_cval(shpty, astb.bnd.dtype);
  }

  dtype = DDTG(dtype);
  fty = dtype_to_arg(dtype);
  fty = mk_cval(fty, astb.bnd.dtype);
  switch (DTY(dtype)) {
  case TY_CHAR:
  case TY_NCHAR:
    sz = ast_intr(I_LEN, astb.bnd.dtype, 1, fptr);
    break;
  default:
    sz = mk_cval(size_of(dtype), astb.bnd.dtype);
    break;
  }
  func = mk_id(sym_mkfunc_nodesc_expst(mkRteRtnNm(RTE_c_f_ptr), DT_INT));

  nargs = 8;
  newargt = mk_argt(nargs);
  ARGT_ARG(newargt, 0) = cptr;                          /* cptr    */
  ARGT_ARG(newargt, 1) = mk_cval(rank, astb.bnd.dtype); /* rank    */
  ARGT_ARG(newargt, 2) = sz;                            /* len/size of fptr */
  ARGT_ARG(newargt, 3) = pvar;                          /* fptr$p  */
  ARGT_ARG(newargt, 4) = desc;                          /* fptr$sd */
  ARGT_ARG(newargt, 5) = fty;                           /* eltype of fptr */
  ARGT_ARG(newargt, 6) = shape;                         /* shape   */
  ARGT_ARG(newargt, 7) = shpty;                         /* eltype of shape */
  newast = mk_func_node(A_CALL, func, nargs, newargt);
  std = add_stmt_before(newast, stdnext);
  STD_LINENO(std) = lineno;
  STD_PAR(std) = STD_PAR(stdnext);
  STD_TASK(std) = STD_TASK(stdnext);
  STD_ACCEL(std) = STD_ACCEL(stdnext);
  STD_KERNEL(std) = STD_KERNEL(stdnext);
  delete_stmt(arg_gbl.std);
}

static void
transform_c_f_procpointer(int func_ast, int func_args)
{
  /*
   * c_f_procpointer(cptr, fptr)
   * call RTE_c_f_procptr, passing the address of cptr and fptr$p.
   * lower() could turn this into an assignment of the form:
   *     fptr$p = cptr%val
   * But today, I do not want to deal with assigning an integer (cptr%val)
   * to a pointer variable.
   */
  int lineno;
  int stdnext, std;
  int newast;
  int fptr;
  int pvar = 0;
  int dtype;
  int func;
  int nargs;
  int newargt;
  int sptr;

  stdnext = arg_gbl.std;
  lineno = STD_LINENO(stdnext);
  fptr = ARGT_ARG(func_args, 1);
  /*
   * pass the address of fptr$p instead of just referencing fptr.
   */
  dtype = A_DTYPEG(fptr);
  if (A_TYPEG(fptr) == A_ID)
    sptr = A_SPTRG(fptr);
  else if (A_TYPEG(fptr) == A_MEM)
    sptr = A_SPTRG(A_MEMG(fptr));
  else
    sptr = 0;
  if (sptr && MIDNUMG(sptr)) {
    pvar = check_member(fptr, mk_id(MIDNUMG(sptr)));
  } else {
    interr("FPTR error in c_f_procpointer()", fptr, 4);
  }

  func = mk_id(sym_mkfunc_nodesc_expst(mkRteRtnNm(RTE_c_f_procptr), DT_INT));
  nargs = 2;
  newargt = mk_argt(nargs);
  ARGT_ARG(newargt, 0) = ARGT_ARG(func_args, 0); /* cptr    */
  ARGT_ARG(newargt, 1) = pvar;                   /* fptr$p  */
  newast = mk_func_node(A_CALL, func, nargs, newargt);
  std = add_stmt_before(newast, stdnext);
  STD_LINENO(std) = lineno;
  STD_PAR(std) = STD_PAR(stdnext);
  STD_TASK(std) = STD_TASK(stdnext);
  STD_ACCEL(std) = STD_ACCEL(stdnext);
  STD_KERNEL(std) = STD_KERNEL(stdnext);
  delete_stmt(arg_gbl.std);
}

static void
_rewrite_scalar_fuctions(int astx, int *std)
{
  int sptrretval;
  int sptrtmp;
  int funcsptr;
  int iface;
  int ast;
  int asttmp;
  int args;

  if (A_TYPEG(astx) == A_FUNC && DT_ISSCALAR(A_DTYPEG(astx))) {
    funcsptr = procsym_of_ast(A_LOPG(astx));
    proc_arginfo(funcsptr, NULL, NULL, &iface);
    if (iface && FVALG(iface)) {
      args = rewrite_sub_args(astx, 0);
      A_ARGSP(astx, args);
      sptrretval = FVALG(iface);
      sptrtmp = sym_get_scalar(SYMNAME(sptrretval), "r", A_DTYPEG(astx));
      asttmp = mk_id(sptrtmp);
      ast = mk_assn_stmt(asttmp, astx, A_DTYPEG(astx));
      add_stmt_before(ast, *std);
      ast_replace(astx, asttmp);
    }
  }
}

static int
rewrite_scalar_functions(int astx, int std)
{
  int ast;

  ast_visit(1, 1);
  ast_traverse(astx, NULL, _rewrite_scalar_fuctions, &std);
  ast = ast_rewrite(astx);
  ast_unvisit();
  return ast;
}

/*
 * Return TRUE if AST astx contains an intrinsic or external call.
 * allow calls to user or intrinsic elementals
 */
static LOGICAL
_contains_any_call(int astx, LOGICAL *pflag)
{
  if (A_TYPEG(astx) == A_INTR) {
    /* allow elemental intrinsic call s*/
    if (INKINDG(procsym_of_ast(A_LOPG(astx))) == IK_ELEMENTAL) {
      return FALSE;
    }
    *pflag = TRUE;
    return TRUE;
  } else if (A_TYPEG(astx) == A_CALL || A_TYPEG(astx) == A_FUNC) {
    if (elemental_func_call(astx)) {
      return FALSE;
    }
    *pflag = TRUE;
    return TRUE;

  } else if (A_TYPEG(astx) == A_ICALL) {
    *pflag = TRUE;
    return TRUE;
  }
  return FALSE;
}

/*
 * Return TRUE if AST astx contains an intrinsic or external call.
 * allow calls to user or intrinsic elementals
 */
static LOGICAL
contains_any_call(int astx)
{
  LOGICAL flag = FALSE;
  ast_visit(1, 1);
  ast_traverse(astx, _contains_any_call, NULL, &flag);
  ast_unvisit();
  return flag;
}

static int subscript_lhs(int, int *, int, DTYPE, int, int);

static LOGICAL
ast_cval(int ast, ISZ_T *value)
{
  if (ast && A_ALIASG(ast))
    ast = A_ALIASG(ast);
  if (ast && A_TYPEG(ast) == A_CNST) {
    int sptr = A_SPTRG(ast);
    if (sptr && STYPEG(sptr) == ST_CONST) {
      *value = get_isz_cval(sptr);
      return TRUE;
    }
  }
  return FALSE;
} /* ast_cval */

/*
 * from a(1:3:1,2:4:2) given offsets 'i' and 'j' for subscripts 'si' and 'sj',
 * build the reference a(1+i, 2+j*2) and return that
 * This routine does the array reference rewrite
 */
static int
build_array_reference(int ast, int si, int vi, int sj, int vj)
{
  int asd, numdim, k, ss, iss;
  int subs[MAXSUBS];
  asd = A_ASDG(ast);
  numdim = ASD_NDIM(asd);
  iss = 0;
  for (k = 0; k < numdim; ++k) {
    ss = ASD_SUBS(asd, k);
    if (A_TYPEG(ss) == A_TRIPLE) {
      int v, a;
      if (iss == si) {
        v = vi;
      } else if (iss == sj) {
        v = vj;
      } else {
        return 0;
      }
      /* return A_LBDG(ss) + A_STRIDEG(ss) * v */
      a = mk_cval(v, A_DTYPEG(A_STRIDEG(ss)));
      a = mk_binop(OP_MUL, a, A_STRIDEG(ss), A_DTYPEG(A_STRIDEG(ss)));
      a = mk_binop(OP_ADD, a, A_LBDG(ss), A_DTYPEG(A_LBDG(ss)));
      subs[k] = a;
      ++iss;
    } else if (A_SHAPEG(ss)) {
      return 0;
    } else {
      subs[k] = ss;
    }
  }
  ast = mk_subscr(A_LOPG(ast), subs, numdim, DDTG(A_DTYPEG(ast)));
  return ast;
} /* build_array_reference */

/*
 * from a(1:3:1,2:4:2) given offsets 'i' and 'j' for subscripts 'si' and 'sj',
 * build the reference a(1+i, 2+j*2) and return that
 * This routine walks the expression tree to find the array reference(s)
 */
static int
build_array_ref(int inast, int si, int vi, int sj, int vj)
{
  int ast1, ast2, dtype, args, arg1;
  int shape, argt, nargs, k;
  switch (A_TYPEG(inast)) {
  case A_BINOP:
    ast1 = build_array_ref(A_LOPG(inast), si, vi, sj, vj);
    if (ast1 == 0)
      return 0;
    ast2 = build_array_ref(A_ROPG(inast), si, vi, sj, vj);
    if (ast2 == 0)
      return 0;
    dtype = A_DTYPEG(inast);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    return mk_binop(A_OPTYPEG(inast), ast1, ast2, dtype);
  case A_UNOP:
    ast1 = build_array_ref(A_LOPG(inast), si, vi, sj, vj);
    if (ast1 == 0)
      return 0;
    dtype = A_DTYPEG(inast);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    return mk_unop(A_OPTYPEG(inast), ast1, dtype);
  case A_CONV:
    ast1 = build_array_ref(A_LOPG(inast), si, vi, sj, vj);
    if (ast1 == 0)
      return 0;
    dtype = A_DTYPEG(inast);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    return mk_convert(ast1, dtype);
  case A_CMPLXC:
  case A_CNST:
    return inast;
  case A_SUBSTR:
    ast1 = build_array_ref(A_LOPG(inast), si, vi, sj, vj);
    if (ast1 == 0)
      return 0;
    return mk_substr(ast1, A_LEFTG(inast), A_RIGHTG(inast), A_DTYPEG(inast));
  case A_PAREN:
    ast1 = build_array_ref(A_LOPG(inast), si, vi, sj, vj);
    if (ast1 == 0)
      return 0;
    return mk_paren(ast1, A_DTYPEG(ast1));

  case A_FUNC:
    shape = A_SHAPEG(inast);
    if (shape) {
      argt = A_ARGSG(inast);
      nargs = A_ARGCNTG(inast);
      for (k = 0; k < nargs; ++k) {
        ast1 = build_array_ref(ARGT_ARG(argt, k), si, vi, sj, vj);
        if (ast1 == 0)
          return 0;
      }
      /* now for real */
      for (k = 0; k < nargs; ++k) {
        ARGT_ARG(argt, k) = build_array_ref(ARGT_ARG(argt, k), si, vi, sj, vj);
      }
      dtype = A_DTYPEG(inast);
      if (DTY(dtype) == TY_ARRAY && elemental_func_call(inast)) {
        A_DTYPEP(inast, DTY(dtype + 1));
        A_SHAPEP(inast, 0);
      }
    }
    return inast;
  case A_SUBSCR:
    /* does this subscript have any triplet entries */
    if (vector_member(inast)) {
      inast = build_array_reference(inast, si, vi, sj, vj);
    }
    if (A_TYPEG(A_LOPG(inast)) == A_MEM) {
      /* the parent might have an array index */
      int asd = A_ASDG(inast);
      ast1 = build_array_ref(A_PARENTG(A_LOPG(inast)), si, vi, sj, vj);
      if (ast1 == 0)
        return 0;
      if (ast1 != A_PARENTG(A_LOPG(inast))) {
        DTYPE dtype = A_DTYPEG(A_MEMG(A_LOPG(inast)));
        ast1 = mk_member(ast1, A_MEMG(A_LOPG(inast)), dtype);
        if (is_array_dtype(dtype))
          dtype = array_element_dtype(dtype);
        /* add the member subscripts */
        inast = mk_subscr_copy(ast1, asd, dtype);
      }
    }
    return inast;
  case A_MEM:
    /* the parent might have an array index */
    ast1 = build_array_ref(A_PARENTG(inast), si, vi, sj, vj);
    if (ast1 == 0)
      return 0;
    /* member should be a scalar here */
    return mk_member(ast1, A_MEMG(inast), A_DTYPEG(A_MEMG(inast)));
  case A_ID:
    return inast;
  case A_INTR:
    /* allow transpose() call */
    if (A_OPTYPEG(inast) != I_TRANSPOSE) {
      return 0;
    }
    args = A_ARGSG(inast);
    arg1 = ARGT_ARG(args, 0);
    ast1 = build_array_ref(arg1, sj, vi, si, vj);
    return ast1;
  default:
    return 0;
  }

} /* build_array_ref */

/*
 *  a = matmul( b, c )
 *  where the extent of a, b, c is less than 3 in each dimension
 *  inline to
 *   a(i,j) = sum(b(i,k) * c(k,j))
 *  where we expand i, j, k at compile time from 1 to the extent.
 *  for I_MATMUL_TRANSPOSE, we transpose the first argument:
 *   a(i,j) = sum(b(k,i) * c(k,j))
 *  if dest is zero, we have to create a temp array of the appropriate size
 *  and return a reference to that array.
 */

static int
inline_small_matmul(int ast, int dest)
{
  ISZ_T ilow, ihigh, istride, iextent;
  ISZ_T jlow, jhigh, jstride, jextent;
  ISZ_T klow, khigh, kstride, kextent;
  ISZ_T klowx, khighx, kstridex, kextentx;
  int ii, kk;
  int args, arg1, arg2, array1, array2, arraydest;
  int shape1, shape2;
  int stdnext, lineno;
  int i, j, k;
  int mulop, addop;
  int stdprev;
  if (XBIT(47, 0x200))
    return ast;
  args = A_ARGSG(ast);
  arg1 = ARGT_ARG(args, 0);
  arg2 = ARGT_ARG(args, 1);
  if (!arg1 || !arg2)
    return ast;

  stdprev = STD_PREV(arg_gbl.std);
  arg1 = rewrite_scalar_functions(arg1, arg_gbl.std);
  if (contains_any_call(arg1)) {
    arg1 = rewrite_sub_ast(arg1, 0);
    if (arg1 == -1)
      return ast;
  }
  arg2 = rewrite_scalar_functions(arg2, arg_gbl.std);
  if (contains_any_call(arg2)) {
    arg2 = rewrite_sub_ast(arg2, 0);
    if (arg2 == -1)
      return ast;
  }
  if (stdprev != STD_PREV(arg_gbl.std)) {
    /*
     * Allocatable temps could have been created while processing
     * the arguments and would degrade performance if we don't cleanup.
     * So, if any statements were created for the * arguments, just
     * make a new matmul ast
     */
    int argtnew, astnew;
    argtnew = mk_argt(2);
    ARGT_ARG(argtnew, 0) = arg1;
    ARGT_ARG(argtnew, 1) = arg2;
    astnew = mk_func_node(A_TYPEG(ast), A_LOPG(ast), 2, argtnew);
    A_OPTYPEP(astnew, A_OPTYPEG(ast));
    A_SHAPEP(astnew, A_SHAPEG(ast));
    A_DTYPEP(astnew, A_DTYPEG(ast));
    ast = astnew;
  }
  shape1 = A_SHAPEG(arg1);
  shape2 = A_SHAPEG(arg2);
  /* must be (n,k)x(k,m), or (k)x(k,m) or (n,k)x(k) */
  if (SHD_NDIM(shape1) != 2 && SHD_NDIM(shape1) != 1)
    return ast;
  if (SHD_NDIM(shape2) != 2 && SHD_NDIM(shape2) != 1)
    return ast;
  if (SHD_NDIM(shape1) == 1 && SHD_NDIM(shape2) == 1)
    return ast;
  /* check for transposed 1st argument */
  ii = 0;
  kk = 1;
  if (A_OPTYPEG(ast) == I_MATMUL_TRANSPOSE) {
    ii = 1;
    kk = 0;
  }
  /* the shapes must be constant sizes */
  if (SHD_NDIM(shape1) == 1) {
    ilow = 0;
    ihigh = 0;
    istride = 1;
    ii = 1;
    kk = 0;
    if (!ast_cval(SHD_LWB(shape1, kk), &klow))
      return ast;
    if (!ast_cval(SHD_UPB(shape1, kk), &khigh))
      return ast;
    if (!ast_cval(SHD_STRIDE(shape1, kk), &kstride))
      return ast;
  } else {
    if (!ast_cval(SHD_LWB(shape1, ii), &ilow))
      return ast;
    if (!ast_cval(SHD_UPB(shape1, ii), &ihigh))
      return ast;
    if (!ast_cval(SHD_STRIDE(shape1, ii), &istride))
      return ast;
    if (!ast_cval(SHD_LWB(shape1, kk), &klow))
      return ast;
    if (!ast_cval(SHD_UPB(shape1, kk), &khigh))
      return ast;
    if (!ast_cval(SHD_STRIDE(shape1, kk), &kstride))
      return ast;
  }
  if (SHD_NDIM(shape2) == 1) {
    jlow = 0;
    jhigh = 0;
    jstride = 1;
    if (!ast_cval(SHD_LWB(shape2, 0), &klowx))
      return ast;
    if (!ast_cval(SHD_UPB(shape2, 0), &khighx))
      return ast;
    if (!ast_cval(SHD_STRIDE(shape2, 0), &kstridex))
      return ast;
  } else {
    if (!ast_cval(SHD_LWB(shape2, 0), &klowx))
      return ast;
    if (!ast_cval(SHD_UPB(shape2, 0), &khighx))
      return ast;
    if (!ast_cval(SHD_STRIDE(shape2, 0), &kstridex))
      return ast;
    if (!ast_cval(SHD_LWB(shape2, 1), &jlow))
      return ast;
    if (!ast_cval(SHD_UPB(shape2, 1), &jhigh))
      return ast;
    if (!ast_cval(SHD_STRIDE(shape2, 1), &jstride))
      return ast;
  }
  if (istride == 0 || kstride == 0 || kstridex == 0 || jstride == 0)
    return ast;
  iextent = (ihigh - ilow + istride) / istride;
  jextent = (jhigh - jlow + jstride) / jstride;
  kextent = (khigh - klow + kstride) / kstride;
  kextentx = (khighx - klowx + kstridex) / kstridex;
  if (kextent != kextentx)
    return ast;

  /* See if it's small enough */
  if (iextent <= 0 || iextent > 4)
    return ast;
  if (jextent <= 0 || jextent > 4)
    return ast;
  if (kextent <= 0 || kextent > 4)
    return ast;
  if (iextent * jextent * kextent > 32)
    return ast;

  array1 = convert_subscript_in_expr(arg1);
  array2 = convert_subscript_in_expr(arg2);
  stdnext = arg_gbl.std;
  lineno = STD_LINENO(stdnext);
  if (1 || !dest) {
    int sptr, dtnew, eldtype;
    ADSC *ad;
    eldtype = DDTG(A_DTYPEG(ast));
    if (SHD_NDIM(shape1) == 1) {
      dtnew = get_array_dtype(1, eldtype);
      ad = AD_DPTR(dtnew);
      AD_LWBD(ad, 0) = AD_LWAST(ad, 0) = mk_cval(1, DT_INT);
      AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = mk_cval(jextent, DT_INT);
      AD_EXTNTAST(ad, 0) = AD_UPBD(ad, 0);
    } else if (SHD_NDIM(shape2) == 1) {
      dtnew = get_array_dtype(1, eldtype);
      ad = AD_DPTR(dtnew);
      AD_LWBD(ad, 0) = AD_LWAST(ad, 0) = mk_cval(1, DT_INT);
      AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = mk_cval(iextent, DT_INT);
      AD_EXTNTAST(ad, 0) = AD_UPBD(ad, 0);
    } else {
      dtnew = get_array_dtype(2, eldtype);
      ad = AD_DPTR(dtnew);
      AD_LWBD(ad, 0) = AD_LWAST(ad, 0) = mk_cval(1, DT_INT);
      AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = mk_cval(iextent, DT_INT);
      AD_EXTNTAST(ad, 0) = AD_UPBD(ad, 0);
      AD_LWBD(ad, 1) = AD_LWAST(ad, 1) = mk_cval(1, DT_INT);
      AD_UPBD(ad, 1) = AD_UPAST(ad, 1) = mk_cval(jextent, DT_INT);
      AD_EXTNTAST(ad, 1) = AD_UPBD(ad, 1);
    }
    sptr = get_arr_temp(dtnew, TRUE, FALSE, FALSE);
    trans_mkdescr(sptr);
    dest = mk_id(sptr);
  }
  arraydest = convert_subscript_in_expr(dest);
  mulop = OP_MUL;
  addop = OP_ADD;
  if (TY_ISLOG(DTYG(A_DTYPEG(ast)))) {
    mulop = OP_LAND;
    addop = OP_LOR;
  } else if (!TY_ISNUMERIC(DTYG(A_DTYPEG(ast)))) {
    return ast;
  }
  /* build assignment statements */
  for (j = 0; j < jextent; ++j) {
    for (i = 0; i < iextent; ++i) {
      int lhs, rhs, std;
      if (SHD_NDIM(shape1) == 1) {
        lhs = build_array_ref(arraydest, 0, j, 1, i);
      } else {
        lhs = build_array_ref(arraydest, 0, i, 1, j);
      }
      if (lhs == 0)
        return ast;
      rhs = 0;
      for (k = 0; k < kextent; ++k) {
        int opnd1, opnd2;
        opnd1 = build_array_ref(array1, ii, i, kk, k);
        if (opnd1 == 0)
          return ast;
        opnd2 = build_array_ref(array2, 0, k, 1, j);
        if (opnd2 == 0)
          return ast;
        opnd1 = mk_binop(mulop, opnd1, opnd2, A_DTYPEG(opnd1));
        if (!rhs) {
          rhs = opnd1;
        } else {
          rhs = mk_binop(addop, rhs, opnd1, A_DTYPEG(opnd1));
        }
      }
      lhs = mk_assn_stmt(lhs, rhs, A_DTYPEG(rhs));
      std = add_stmt_before(lhs, stdnext);
      STD_LINENO(std) = lineno;
      STD_PAR(std) = STD_PAR(stdnext);
      STD_TASK(std) = STD_TASK(stdnext);
      STD_ACCEL(std) = STD_ACCEL(stdnext);
      STD_KERNEL(std) = STD_KERNEL(stdnext);
    }
  }
  /* return the destination array */
  return arraydest;
} /* inline_small_matmul */

static int
inline_reduction_f90(int ast, int dest, int lc, LOGICAL *doremove)
{
  int astdim, dim, mask, astmask;
  int args;
  int src1, src2, std;
  int dtype, dtypetmp, dtyperes, dtsclr, eldtype;
  int dtypetmpval, sptrtmpval, asttmpval, dtypeval, astsubscrtmpval;
  int dealloc_tmpval = FALSE;
  int srcarray;
  int home, homeforall;
  int lb, ub, st;
  int forall;
  int asn;
  int lineno;
  int stdnext;
  int newast;
  int ast2;
  int allocobj;
  int sptrtmp, asttmp, astsubscrtmp;
  int tmpndim;
  int i, j, n;
  int triplet_list, index_var;
  int triplet;
  int shape;
  int dest_shape;
  int sptr;
  int ndim, asd;
  int list;
  int endif_ast, ifastnew;
  char sReduc[128];
  int ReducType;
  int astInit;
  int operator, operand;
  int ifast, endif;
  int dovar;
  int subs[MAXSUBS];
  int loopidx[MAXSUBS];
  int DOs[MAXSUBS];
  int curloop;
  int tmpidx[MAXSUBS];
  int nbrloops;
  int destndim;
  int destsub;
  int destsptr;
  int destref;
  ADSC *ad;
  int dealloc_dest = FALSE;

  if (XBIT(47, 0x80))
    return ast;
  if (A_TYPEG(ast) != A_INTR)
    return ast;

  /* if not reduction, return */
  switch (A_OPTYPEG(ast)) {
  case I_ALL:
  case I_ANY:
  case I_COUNT:
  case I_DOT_PRODUCT:
  case I_MAXVAL:
  case I_MINVAL:
  case I_PRODUCT:
  case I_SUM:
    if (doremove)
      *doremove = TRUE;
    break;
  case I_MAXLOC:
  case I_MINLOC:
      return ast;
    /* simple cases only */
    if (dest) {
      if (A_TYPEG(dest) == A_SUBSCR) {
        shape = A_SHAPEG(dest);
        if (SHD_NDIM(shape) != 1 || SHD_LWB(shape, 0) != SHD_UPB(shape, 0))
          return ast;
      } else if (A_TYPEG(dest) != A_ID)
        return ast;
    }
    if (doremove)
      *doremove = TRUE;
    break;
  case I_MATMUL:
  case I_MATMUL_TRANSPOSE:
    if (doremove)
      *doremove = FALSE;
    return inline_small_matmul(ast, dest);
  default:
    return ast;
  }

  /* collect args */
  mask = 0;
  strcpy(sReduc, SYMNAME(A_SPTRG(A_LOPG(ast))));
  dtype = A_DTYPEG(ast);
  dtyperes = DDTG(dtype);
  args = A_ARGSG(ast);
  switch (A_OPTYPEG(ast)) {
  case I_SUM:
  case I_PRODUCT:
    astdim = ARGT_ARG(args, 1);
    mask = ARGT_ARG(args, 2);
    srcarray = ARGT_ARG(args, 0);
    if (arg_gbl.inforall)
      if (contiguous_section_array(srcarray))
        return ast;
    break;
  case I_MAXLOC:
  case I_MINLOC:
    dtypeval = DDTG(A_DTYPEG(ARGT_ARG(args, 0)));
    FLANG_FALLTHROUGH;
  case I_MAXVAL:
  case I_MINVAL:
    astdim = ARGT_ARG(args, 1);
    mask = ARGT_ARG(args, 2);
    srcarray = ARGT_ARG(args, 0);
    if (DTYG(dtype) == TY_CHAR || DTYG(dtype) == TY_NCHAR)
      return ast;
    if (arg_gbl.inforall)
      if (contiguous_section_array(srcarray))
        return ast;
    break;
  case I_DOT_PRODUCT:
    astdim = 0;
    src1 = ARGT_ARG(args, 0);
    src2 = ARGT_ARG(args, 1);
    if (DT_ISCMPLX(DDTG(dtype)) && (XBIT(70, 0x4000000)
                                    || dtyperes == DT_QCMPLX
                                    ))
      return ast;
    if (arg_gbl.inforall) {
      if (contiguous_section_array(src1) && contiguous_section_array(src2))
        return ast;
    }
    if (DT_ISLOG(DDTG(dtype)))
      operator= OP_LAND;
    else
      operator= OP_MUL;
    if (DT_ISCMPLX(DDTG(dtype))) {
      int newargt, conjg, nast;
      if (dtyperes == DT_CMPLX) {
        conjg = I_CONJG;
      } else if (dtyperes == DT_CMPLX16) {
        conjg = I_DCONJG;
      } else {
        return ast;
      }
      newargt = mk_argt(1);
      ARGT_ARG(newargt, 0) = src1;
      nast = mk_func_node(A_INTR, mk_id(intast_sym[conjg]), 1, newargt);
      A_OPTYPEP(nast, conjg);
      A_DTYPEP(nast, A_DTYPEG(src1));
      src1 = nast;
    }
    srcarray = mk_binop(operator, src1, src2, dtype);
    break;
  case I_ALL:
  case I_ANY:
  case I_COUNT:
    astdim = ARGT_ARG(args, 1);
    srcarray = ARGT_ARG(args, 0);
    if (arg_gbl.inforall)
      if (contiguous_section_array(srcarray))
        return ast;
    break;
  }

  if (astdim) {
    if (A_TYPEG(astdim) != A_CNST) {
      return ast;
    }
    dim = get_int_cval(A_SPTRG(astdim));
  } else {
    dim = 0;
  }

  if ((A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC) && dim > 1)
    return ast;

  if (!XBIT(70, 0x1000000) && dim == 1 && arg_gbl.inforall) {
    return ast;
  }

  srcarray = rewrite_scalar_functions(srcarray, arg_gbl.std);
  if (contains_any_call(srcarray)) { /* return ast; */
    srcarray = rewrite_sub_ast(srcarray, 0);
    if (srcarray == -1)
      /* source is not something convert_subscript can handle and
       * computing it into an allocated temp is probably too
       * expensive.  Don't inline it; call the subroutine.
       */
      return ast;
    home = search_conform_array(srcarray, TRUE);
    if (!home)
      /* source is not something convert_subscript can handle and
       * computing it into an allocated temp is probably too
       * expensive.  Don't inline it; call the subroutine.
       */
      return ast;
    if (A_TYPEG(home) != A_ID && A_TYPEG(home) != A_MEM &&
        A_TYPEG(home) != A_TRIPLE && A_TYPEG(home) != A_SUBSCR)
      /* source is not something convert_subscript can handle and
       * computing it into an allocated temp is probably too
       * expensive.  Don't inline it; call the subroutine.
       */
      return ast;
    /*
    fprintf(STDERR,
        "%s:%s:%d - inline_reduction_f90 change in behavior\n",
        gbl.src_file,
        SYMNAME(gbl.currsub), gbl.lineno);
    dbg_print_ast(srcarray, 0);
    dump_one_ast(srcarray);
    */
  }
  home = search_conform_array(srcarray, TRUE);
  if (!home)
    return ast;
  if (mask) {
    mask = rewrite_scalar_functions(mask, arg_gbl.std);
    if (contains_any_call(mask)) { /* return ast; */
      mask = rewrite_sub_ast(mask, 0);
      if (mask == -1) {
        /* source is not something convert_subscript can handle and
         * computing it into an allocated temp is probably too
         * expensive.  Don't inline it; call the subroutine.
         */
        return ast;
      }
    }
  }
  ast2 = convert_subscript_in_expr(srcarray);
  home = convert_subscript(home);
  if (mask) {
    astmask = convert_subscript_in_expr(mask);
  } else {
    astmask = 0;
  }

  sptr = sptr_of_subscript(home);

  shape = A_SHAPEG(home);
  forall = make_forall(shape, home, astmask,
                       lc + SHD_NDIM(shape)); /*TODO: need correct triple */
  homeforall = normalize_forall(forall, home, 0);
  ast2 = normalize_forall(forall, ast2, 0);
  if (mask) {
    astmask = normalize_forall(forall, astmask, 0);
  }
  list = A_LISTG(forall);
  asd = A_ASDG(homeforall);
  ndim = ASD_NDIM(asd); /* MORE ndim and nbrloops are NOT the same!!! */
  nbrloops = SHD_NDIM(shape);

  stdnext = arg_gbl.std;
  lineno = STD_LINENO(stdnext);

  if (A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC) {
    /* build temp */
    sptrtmp = sym_get_scalar(SYMNAME(sptr), "r", dtyperes);
    dtypetmp = DTYPEG(sptrtmp);
    asttmp = mk_id(sptrtmp);
    dtypetmp = DTYPEG(sptrtmp);
    asttmp = mk_id(sptrtmp);

    /* build temp to hold values for I_MAXLOC, I_MINLOC */
    if (dim <= 1 || nbrloops == 1) {
      sptrtmpval = sym_get_scalar(SYMNAME(sptr), "vr", dtypeval);
      dtypetmpval = DTYPEG(sptrtmpval);
      asttmpval = mk_id(sptrtmpval);
    } else {
      reset_init_idx();
      dest_shape = A_SHAPEG(ast);
      sptrtmpval = sym_get_array(SYMNAME(sptr), "vr", dtypeval, dim - 1);
      dtypetmpval = DTYPEG(sptrtmpval);
      for (i = 0; i < dim - 1; ++i) {
        ADD_LWBD(dtypetmpval, i) = ADD_LWAST(dtypetmpval, i) =
            SHD_LWB(dest_shape, i);
        ADD_UPBD(dtypetmpval, i) = ADD_UPAST(dtypetmpval, i) =
            SHD_UPB(dest_shape, i);
        ADD_EXTNTAST(dtypetmpval, i) =
            mk_extent(ADD_LWAST(dtypetmpval, i), ADD_UPAST(dtypetmpval, i), i);
        subs[i] = mk_triple(SHD_LWB(dest_shape, i), SHD_UPB(dest_shape, i),
                            astb.bnd.one);
      }
      dtypetmpval = DTYPEG(sptrtmpval);
      NODESCP(sptrtmpval, 1);
      check_small_allocatable(sptrtmpval);
      asttmpval = mk_id(sptrtmpval);

      if (ALLOCG(sptrtmpval)) {
        allocobj = mk_subscr(asttmpval, subs, dim - 1, DDTG(dtypetmpval));
        newast = mk_stmt(A_ALLOC, 0);
        A_TKNP(newast, TK_ALLOCATE);
        A_LOPP(newast, 0);
        A_SRCP(newast, allocobj);
        std = add_stmt_before(newast, stdnext);
        STD_LINENO(std) = lineno;
        STD_LOCAL(std) = 1;
        STD_PAR(std) = STD_PAR(stdnext);
        STD_TASK(std) = STD_TASK(stdnext);
        STD_ACCEL(std) = STD_ACCEL(stdnext);
        STD_KERNEL(std) = STD_KERNEL(stdnext);
        if (STD_ACCEL(std))
          STD_RESCOPE(std) = 1;
        dealloc_tmpval = TRUE;
      }
    }
  } else {
    /* build temp */
    if (dim <= 1 || nbrloops == 1) {
      sptrtmp = sym_get_scalar(SYMNAME(sptr), "r", dtyperes);
      dtypetmp = DTYPEG(sptrtmp);
      asttmp = mk_id(sptrtmp);
    } else {
      reset_init_idx();
      dest_shape = A_SHAPEG(ast);
      sptrtmp = sym_get_array(SYMNAME(sptr), "r", dtyperes, dim - 1);
      dtypetmp = DTYPEG(sptrtmp);
      ad = AD_DPTR(dtype);
      for (i = 0; i < dim - 1; ++i) {
        if (SHD_STRIDE(dest_shape, i) == astb.i1 ||
            SHD_STRIDE(dest_shape, i) == astb.bnd.one) {
          ADD_LWBD(dtypetmp, i) = ADD_LWAST(dtypetmp, i) =
              SHD_LWB(dest_shape, i);
          ADD_UPBD(dtypetmp, i) = ADD_UPAST(dtypetmp, i) =
              SHD_UPB(dest_shape, i);
          ADD_EXTNTAST(dtypetmp, i) =
              mk_extent(ADD_LWAST(dtypetmp, i), ADD_UPAST(dtypetmp, i), i);
          subs[i] = mk_triple(SHD_LWB(dest_shape, i), SHD_UPB(dest_shape, i),
                              astb.bnd.one);
        } else {
          ADD_LWBD(dtypetmp, i) = ADD_LWAST(dtypetmp, i) =
              SHD_LWB(dest_shape, i);
          ADD_UPBD(dtypetmp, i) = ADD_UPAST(dtypetmp, i) = mk_binop(
              OP_DIV,
              mk_binop(OP_ADD,
                       mk_binop(OP_SUB, SHD_UPB(dest_shape, i),
                                SHD_LWB(dest_shape, i), astb.bnd.dtype),
                       SHD_STRIDE(dest_shape, i), astb.bnd.dtype),
              SHD_STRIDE(dest_shape, i), astb.bnd.dtype);

          ADD_EXTNTAST(dtypetmp, i) =
              mk_extent(ADD_LWAST(dtypetmp, i), ADD_UPAST(dtypetmp, i), i);
          subs[i] = mk_triple(ADD_LWAST(dtypetmp, i), ADD_UPAST(dtypetmp, i),
                              astb.bnd.one);
        }
      }
      dtypetmp = DTYPEG(sptrtmp);
      NODESCP(sptrtmp, 1);
      check_small_allocatable(sptrtmp);
      asttmp = mk_id(sptrtmp);

      if (ALLOCG(sptrtmp)) {
        allocobj = mk_subscr(asttmp, subs, dim - 1, DDTG(dtypetmp));
        newast = mk_stmt(A_ALLOC, 0);
        A_TKNP(newast, TK_ALLOCATE);
        A_LOPP(newast, 0);
        A_SRCP(newast, allocobj);
        std = add_stmt_before(newast, stdnext);
        STD_LINENO(std) = lineno;
        STD_LOCAL(std) = 1;
        STD_PAR(std) = STD_PAR(stdnext);
        STD_TASK(std) = STD_TASK(stdnext);
        STD_ACCEL(std) = STD_ACCEL(stdnext);
        STD_KERNEL(std) = STD_KERNEL(stdnext);
        if (STD_ACCEL(std))
          STD_RESCOPE(std) = 1;
      }
    }
  }

  /* if necessary, build destination */
  if (!dest) {
    if (DTY(dtype) == TY_ARRAY) {
      if (DTY(dtypetmp) == TY_ARRAY && ADD_NUMDIM(dtypetmp) == ndim - 1) {
        /* use temp from above as dest */
        destsptr = sptrtmp;
        dest = asttmp;
        NODESCP(sptrtmp, 0);
        trans_mkdescr(destsptr); /* MORE is this needed??? */
      } else {
        ADSC *addest;
        reset_init_idx();
        destsptr = sym_get_array(SYMNAME(sptr), "tr", dtyperes, nbrloops - 1);
        addest = AD_DPTR(DTYPEG(destsptr));
        AD_NUMDIM(addest) = nbrloops - 1;
        j = 0;
        shape = A_SHAPEG(home);
        for (i = 0; i < nbrloops; ++i) {
          if (i != dim - 1) {
            AD_LWAST(addest, j) = AD_LWBD(addest, j) = SHD_LWB(shape, i);
            AD_UPAST(addest, j) = AD_UPBD(addest, j) = SHD_UPB(shape, i);
            AD_EXTNTAST(addest, j) =
                mk_extent(AD_LWAST(addest, j), AD_UPAST(addest, j), j);
            subs[j] = mk_triple(AD_LWBD(addest, j), AD_UPBD(addest, j),
                                SHD_STRIDE(shape, i));
            j++;
          }
        }
        dest = mk_id(destsptr);
        A_SHAPEP(dest, reduc_shape(shape, astdim, STD_PREV(stdnext)));

        trans_mkdescr(destsptr); /* MORE is this needed??? */
        check_small_allocatable(destsptr);

        if (ALLOCG(destsptr)) {
          allocobj = mk_subscr(dest, subs, nbrloops - 1, dtyperes);
          newast = mk_stmt(A_ALLOC, 0);
          A_TKNP(newast, TK_ALLOCATE);
          A_LOPP(newast, 0);
          A_SRCP(newast, allocobj);
          std = add_stmt_before(newast, stdnext);
          STD_LINENO(std) = lineno;
          STD_LOCAL(std) = 1;
          STD_PAR(std) = STD_PAR(stdnext);
          STD_TASK(std) = STD_TASK(stdnext);
          STD_ACCEL(std) = STD_ACCEL(stdnext);
          STD_KERNEL(std) = STD_KERNEL(stdnext);
          if (STD_ACCEL(std))
            STD_RESCOPE(std) = 1;
          dealloc_dest = TRUE;
        }
      }
    } else {
      dest = asttmp;
    }
  }

  /* select reduction type */

  switch (A_OPTYPEG(ast)) {
  case I_SUM:
  case I_COUNT:
    ReducType = I_REDUCE_SUM;
    astInit = mk_convert(astb.i0, DDTG(dtypetmp));
    break;
  case I_DOT_PRODUCT:
    ReducType = I_REDUCE_SUM;
    if (DT_ISLOG(DDTG(dtypetmp)))
      astInit = mk_cval(SCFTN_FALSE, DT_LOG);
    else
      astInit = mk_convert(astb.i0, DDTG(dtypetmp));
    break;
  case I_PRODUCT:
    ReducType = I_REDUCE_PRODUCT;
    astInit = mk_convert(astb.i1, DDTG(dtypetmp));
    break;
  case I_MAXVAL:
    ReducType = I_REDUCE_MAXVAL;
    astInit = mk_smallest_val(DDTG(dtypetmp));
    break;
  case I_MAXLOC:
    ReducType = I_REDUCE_MAXVAL;
    astInit = mk_smallest_val(DDTG(dtypetmpval));
    break;
  case I_MINVAL:
    ReducType = I_REDUCE_MINVAL;
    astInit = mk_largest_val(DDTG(dtypetmp));
    break;
  case I_MINLOC:
    ReducType = I_REDUCE_MINVAL;
    astInit = mk_largest_val(DDTG(dtypetmpval));
    break;
  case I_ALL:
    ReducType = I_REDUCE_ALL;
    astInit = mk_cval(SCFTN_TRUE, DDTG(dtypetmp));
    break;
  case I_ANY:
    ReducType = I_REDUCE_ANY;
    astInit = mk_cval(SCFTN_FALSE, DDTG(dtypetmp));
    break;
  default:
    assert(0, "inline_reduction_f90: unknown type", ast, 4);
  }

  if (dim == 0) {
    /* initialize temp */
    if (A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC)
      asn = mk_assn_stmt(asttmpval, astInit, dtypetmpval);
    else
      asn = mk_assn_stmt(asttmp, astInit, dtypetmp);
    std = add_stmt_before(asn, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }

  n = nbrloops;
  j = nbrloops - 1;
  triplet_list = A_LISTG(forall);
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    index_var = ASTLI_SPTR(triplet_list);
    /* find the matching home dimension */
    for (i = 0; i < ndim; i++)
      if (is_name_in_expr(ASD_SUBS(asd, i), index_var))
        break;
    triplet = ASTLI_TRIPLE(triplet_list);
    st = A_STRIDEG(triplet);
    if (!st)
      st = astb.i1;

    newast = mk_stmt(A_DO, 0);
    lb = A_LBDG(triplet);
    ub = A_UPBDG(triplet);

    dovar = mk_id(index_var);
    loopidx[j] = dovar;
    A_DOVARP(newast, dovar);
    A_M1P(newast, lb);
    A_M2P(newast, ub);
    A_M3P(newast, st);
    A_M4P(newast, 0);
    DOs[j] = newast;

    if (n-- == dim) {
      /* initialize temp */
      if (A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC)
        asn = mk_assn_stmt(asttmpval, astInit, dtypetmpval);
      else
        asn = mk_assn_stmt(asttmp, astInit, dtypetmp);
      std = add_stmt_before(asn, stdnext);
      STD_LINENO(std) = lineno;
      STD_LOCAL(std) = 1;
      STD_PAR(std) = STD_PAR(stdnext);
      STD_TASK(std) = STD_TASK(stdnext);
      STD_ACCEL(std) = STD_ACCEL(stdnext);
      STD_KERNEL(std) = STD_KERNEL(stdnext);
    } else {
      tmpidx[j] = dovar;
    }

    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    i++;
    j--;
  }

  if (mask) {
    ifastnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ifastnew, astmask);
    std = add_stmt_before(ifastnew, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }

  /* select reduction stmt */
  if (dim > 1 && nbrloops != 1) {
    ad = AD_DPTR(DTYPEG(sptrtmp));
    tmpndim = AD_NUMDIM(ad);
    for (j = 0; j < tmpndim; i++, j++) {
      if (SHD_STRIDE(dest_shape, j) == astb.i1 ||
          SHD_STRIDE(dest_shape, j) == astb.bnd.one) {
        subs[j] = loopidx[j];
      } else
        subs[j] = mk_binop(OP_ADD,
                           mk_binop(OP_DIV, loopidx[j],
                                    SHD_STRIDE(dest_shape, j), astb.bnd.dtype),
                           SHD_LWB(dest_shape, j), astb.bnd.dtype);
    }
    astsubscrtmp = mk_subscr(asttmp, subs, tmpndim, DDTG(dtypetmp));
    A_SHAPEP(astsubscrtmp, 0);
    if (A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC) {
      astsubscrtmpval = mk_subscr(asttmpval, subs, tmpndim, DDTG(dtypetmpval));
      A_SHAPEP(astsubscrtmpval, 0);
    }
  } else {
    if (A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC) {
      astsubscrtmpval = asttmpval;
      astsubscrtmp = dest;
    } else
      astsubscrtmp = asttmp;
    if (A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC ||
        A_OPTYPEG(ast) == I_MAXVAL || A_OPTYPEG(ast) == I_MINVAL) {
      /* if the expression being reduced is nontrivial, assign to a temp */
      if (A_TYPEG(ast2) == A_SUBSCR || A_TYPEG(ast2) == A_ID) {
      } else {
        /* create a temporary scalar */
        int temprhs = sym_get_scalar(SYMNAME(sptr), "l", dtyperes);
        /* assign the RHS to temprhs */
        int std = mk_assn_stmt(mk_id(temprhs), ast2, dtyperes);
        add_stmt_before(std, stdnext);
        ast2 = mk_id(temprhs);
      }
    }
  }
  dtsclr = DDTG(dtypetmp);
  switch (A_OPTYPEG(ast)) {
  case I_SUM:
  case I_DOT_PRODUCT:
    if (DT_ISLOG(dtsclr))
      operator= OP_LOR;
    else
      operator= OP_ADD;
    newast = mk_binop(operator, astsubscrtmp, ast2, dtsclr);
    asn = mk_assn_stmt(astsubscrtmp, newast, dtsclr);

    std = add_stmt_before(asn, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    break;
  case I_COUNT:
    newast = mk_binop(OP_ADD, astsubscrtmp, astb.i1, dtsclr);
    asn = mk_assn_stmt(astsubscrtmp, newast, dtsclr);

    ifast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ifast, ast2);
    std = add_stmt_before(ifast, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);

    std = add_stmt_before(asn, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);

    endif = mk_stmt(A_ENDIF, 0);
    std = add_stmt_before(endif, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    break;
  case I_PRODUCT:
    newast = mk_binop(OP_MUL, astsubscrtmp, ast2, dtsclr);
    asn = mk_assn_stmt(astsubscrtmp, newast, dtsclr);
    std = add_stmt_before(asn, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    break;
  case I_MAXVAL:
    newast = mk_binop(OP_GT, ast2, astsubscrtmp, DT_LOG);
    asn = mk_assn_stmt(astsubscrtmp, ast2, dtsclr);
    goto max_min_common;
  case I_MINVAL:
    newast = mk_binop(OP_LT, ast2, astsubscrtmp, DT_LOG);
    asn = mk_assn_stmt(astsubscrtmp, ast2, dtsclr);
    goto max_min_common;
  case I_MAXLOC:
    newast = mk_binop(OP_GT, ast2, astsubscrtmpval, DT_LOG);
    asn = mk_assn_stmt(astsubscrtmpval, ast2, DDTG(dtypetmpval));
    goto max_min_common;
  case I_MINLOC:
    newast = mk_binop(OP_LT, ast2, astsubscrtmpval, DT_LOG);
    asn = mk_assn_stmt(astsubscrtmpval, ast2, DDTG(dtypetmpval));

  max_min_common:
    ifast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ifast, newast);
    std = add_stmt_before(ifast, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);

    std = add_stmt_before(asn, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);

    if (A_OPTYPEG(ast) == I_MAXLOC || A_OPTYPEG(ast) == I_MINLOC) {
      if (nbrloops > 1) {
        for (j = 0; j < nbrloops; j++) {
          int subscr;

          subscr = mk_cval(j + 1, astb.bnd.dtype);
          ast2 = mk_subscr(astsubscrtmp, &subscr, 1, dtyperes);
          asn = mk_assn_stmt(ast2, A_DOVARG(DOs[j]), dtyperes);
          std = add_stmt_before(asn, stdnext);
          STD_LINENO(std) = lineno;
          STD_LOCAL(std) = 1;
          STD_PAR(std) = STD_PAR(stdnext);
          STD_TASK(std) = STD_TASK(stdnext);
          STD_ACCEL(std) = STD_ACCEL(stdnext);
          STD_KERNEL(std) = STD_KERNEL(stdnext);
        }
      } else {
        asn = mk_assn_stmt(astsubscrtmp, A_DOVARG(DOs[0]), dtyperes);
        std = add_stmt_before(asn, stdnext);
        STD_LINENO(std) = lineno;
        STD_LOCAL(std) = 1;
        STD_PAR(std) = STD_PAR(stdnext);
        STD_TASK(std) = STD_TASK(stdnext);
        STD_ACCEL(std) = STD_ACCEL(stdnext);
        STD_KERNEL(std) = STD_KERNEL(stdnext);
      }
    }

    endif = mk_stmt(A_ENDIF, 0);
    std = add_stmt_before(endif, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    break;
  case I_ALL:
  case I_ANY:
    if (A_OPTYPEG(ast) == I_ALL) {
      newast = mk_unop(OP_LNOT, ast2, DT_LOG);
      operand = mk_cval(SCFTN_FALSE, DT_LOG);
    } else {
      newast = ast2;
      operand = mk_cval(SCFTN_TRUE, DT_LOG);
    }
    asn = mk_assn_stmt(astsubscrtmp, operand, dtsclr);

    ifast = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(ifast, newast);
    std = add_stmt_before(ifast, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);

    std = add_stmt_before(asn, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);

    endif = mk_stmt(A_ENDIF, 0);
    std = add_stmt_before(endif, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    break;
  default:
    assert(0, "inline_reduction_f90: unknown type", ast, 4);
  }

  if (mask) {
    endif_ast = mk_stmt(A_ENDIF, 0);
    std = add_stmt_before(endif_ast, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
  }

  destref = dest;
  eldtype = dtypetmp; /* assume subscripted object is the immediate lhs */
  destsptr = memsym_of_ast(dest);
  ast2 = search_conform_array(dest, TRUE);
  if (ast2) {
    /* array-valued result.  The result could be something like
     *   dt(:)%mem, du%amem(:), arr(:)
     * Need to locate the array in the lhs which needs to be subscripted.
     */
    int ss;
    if (A_TYPEG(ast2) == A_SUBSCR)
      ss = sptr_of_subscript(ast2);
    else
      ss = memsym_of_ast(ast2);
    if (ss != destsptr) {
      /* subscripted object is some aggregate */
      destsptr = ss;
      eldtype = DTY(DTYPEG(destsptr) + 1);
      destref = ast2;
    }
  }
  ast2 = mk_id(destsptr);
  ast2 = check_member(ast_is_sym(dest) &&
                              (sym_of_ast(dest) != pass_sym_of_ast(dest))
                          ? A_PARENTG(dest)
                          : dest,
                      ast2);
  ad = AD_DPTR(DTYPEG(destsptr));
  destndim = AD_NUMDIM(ad);
  for (i = 1; i <= nbrloops; i++) {
    newast = mk_stmt(A_ENDDO, 0);
    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    if (i == dim && destref != asttmp) {
      if (nbrloops > 1) {
        if (A_TYPEG(destref) == A_SUBSCR) {
          asd = A_ASDG(destref);
          curloop = 0;
          for (j = 0; j < destndim; j++) {
            destsub = ASD_SUBS(asd, j);
            if (A_TYPEG(destsub) != A_TRIPLE) {
              subs[j] = destsub;
            } else if (curloop < dim - 1) {
              subs[j] = destsub;
              curloop++;
            } else {
              /*
               *  for DO i$a = m1, m2, m3
               *  the subscripting of
               *     dest(lb:ub:st)
               *
               *  ( (i$a - m1)/m3 ) * st + lb
               *
               */
              int o;
              int mdo;
              subs[j] = loopidx[++curloop];
              mdo = DOs[curloop];
              o = mk_binop(OP_SUB, subs[j], A_M1G(mdo), astb.bnd.dtype);
              if ((A_M3G(mdo) != astb.i1) && (A_M3G(mdo) != astb.k1))
                o = mk_binop(OP_DIV, o, A_M3G(mdo), astb.bnd.dtype);
              if (A_STRIDEG(destsub))
                o = mk_binop(OP_MUL, o, A_STRIDEG(destsub), astb.bnd.dtype);
              o = mk_binop(OP_ADD, o, A_LBDG(destsub), astb.bnd.dtype);
              subs[j] = o;
            }
          }
        } else {
          for (j = 0; j < destndim; j++) {
            if (j < dim - 1) {
              int lb, ub;
              lb = check_member(destref, AD_LWBD(ad, j));
              ub = check_member(destref, AD_UPBD(ad, j));
              subs[j] = mk_triple(lb, ub, astb.bnd.one);
            } else {
              subs[j] = loopidx[j + 1];
            }
          }
        }
        ast2 = subscript_lhs(ast2, subs, destndim, eldtype, dest, destref);
        ast2 = convert_subscript_in_expr(ast2);
        ast2 = mk_assn_stmt(ast2, asttmp, dtypetmp);
        std = add_stmt_before(ast2, stdnext);
        STD_LINENO(std) = lineno;
        STD_LOCAL(std) = 1;
        STD_PAR(std) = STD_PAR(stdnext);
        STD_TASK(std) = STD_TASK(stdnext);
        STD_ACCEL(std) = STD_ACCEL(stdnext);
        STD_KERNEL(std) = STD_KERNEL(stdnext);
      }
    }
  }

  if (ALLOCG(sptrtmp)) {
    newast = mk_stmt(A_ALLOC, 0);
    A_TKNP(newast, TK_DEALLOCATE);
    A_LOPP(newast, 0);
    A_SRCP(newast, asttmp);
    if (dest != asttmp)
      std = add_stmt_before(newast, stdnext);
    else
      std = add_stmt_before(newast, STD_NEXT(stdnext));
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    if (STD_ACCEL(std))
      STD_RESCOPE(std) = 1;
  }

  if (dealloc_tmpval) {
    newast = mk_stmt(A_ALLOC, 0);
    A_TKNP(newast, TK_DEALLOCATE);
    A_LOPP(newast, 0);
    A_SRCP(newast, asttmpval);
    std = add_stmt_before(newast, stdnext);
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    if (STD_ACCEL(std))
      STD_RESCOPE(std) = 1;
  }

  if (dealloc_dest) {
    newast = mk_stmt(A_ALLOC, 0);
    A_TKNP(newast, TK_DEALLOCATE);
    A_LOPP(newast, 0);
    A_SRCP(newast, dest);
    std = add_stmt_before(newast, STD_NEXT(stdnext));
    STD_LINENO(std) = lineno;
    STD_LOCAL(std) = 1;
    STD_PAR(std) = STD_PAR(stdnext);
    STD_TASK(std) = STD_TASK(stdnext);
    STD_ACCEL(std) = STD_ACCEL(stdnext);
    STD_KERNEL(std) = STD_KERNEL(stdnext);
    if (STD_ACCEL(std))
      STD_RESCOPE(std) = 1;
  }

  ccff_info(MSGOPT, "OPT022", 1, STD_LINENO(arg_gbl.std),
            "%reduction reduction inlined", "reduction=%s", sReduc, NULL);

  return dest;
}

static int
subscript_lhs(int arr, int *subs, int dim, DTYPE dtype, int origlhs,
              int destref)
{
  /*
   * need to subscript an array in the lhs.  The origlhs could be something
   * like dt(:)%mem, du%amem(:), arr(:).
   * If the array is an aggregate, then need to just replace the array
   * in the origlhs with the subscripted form of the array and then apply
   * the remaining portion of the lhs; e.g.,
   *    arr%m1%m2%...mem becomes arr(i$a)%m1%m2%...mem
   */
  int ast = mk_subscr(arr, subs, dim, dtype);
  if (origlhs == destref)
    return ast;
  ast = replace_ast_subtree(origlhs, destref, ast);
  return ast;
}

/*
 * func_ast: A_FUNC or A_INTR
 * func_args: rewritten args
 * lhs: ast for lhs (temp) if non-zero
 */
static int
matmul(int func_ast, int func_args, int lhs)
{
  /* func_ast is a function or intrinsic call.  If it is a transformational
   * intrinsic, create an appropriate temp, rewrite, and return a load
   * of that temp.
   * For now, don't do anything with user-defined functions.
   */
  int shape;
  DTYPE dtype;
  int newsym;
  int temp_arr;
  int newargt;
  int srcarray;
  int retval;
  int ast;
  int nargs;
  const char *name;
  FtnRtlEnum rtlRtn = RTE_no_rtn;
  int subscr[MAXSUBS];
  int arg1, arg2;
  LOGICAL tmp_lhs_array;
  LOGICAL matmul_transpose;

  retval = mmul(func_ast, func_args, lhs);
  if (retval >= 0)
    return retval;

  tmp_lhs_array = FALSE;
  /* it only handles calls */
  shape = A_SHAPEG(func_ast);
  dtype = A_DTYPEG(func_ast);

  matmul_transpose = A_OPTYPEG(func_ast) == I_MATMUL_TRANSPOSE ? TRUE : FALSE;

  /*
   * A_OPTYPEG(func_ast):
   * case I_MATMUL:	         matmul(matrix_a, matrix_b)
   * case I_MATMUL_TRANSPOSE:	 matmul(transpose(matrix_a), matrix_b)
   */
  switch (DTYG(A_DTYPEG(func_ast))) {
  case TY_BINT:
    rtlRtn = RTE_matmul_int1;
    break;
  case TY_SINT:
    rtlRtn = RTE_matmul_int2;
    break;
  case TY_INT:
    rtlRtn = RTE_matmul_int4;
    break;
  case TY_INT8:
    rtlRtn = RTE_matmul_int8;
    break;
  case TY_REAL:
    if (matmul_transpose) {
      rtlRtn = RTE_matmul_real4mxv_t;
    } else {
      rtlRtn = RTE_matmul_real4;
    }
    break;
  case TY_DBLE:
    if (matmul_transpose) {
      rtlRtn = RTE_matmul_real8mxv_t;
    } else {
      rtlRtn = RTE_matmul_real8;
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    if (matmul_transpose) {
      rtlRtn = RTE_matmul_real16mxv_t;
    } else {
      rtlRtn = RTE_matmul_real16;
    }
    break;
#endif
  case TY_CMPLX:
    if (matmul_transpose) {
      rtlRtn = RTE_matmul_cplx8mxv_t;
    } else {
      rtlRtn = RTE_matmul_cplx8;
    }
    break;
  case TY_DCMPLX:
    if (matmul_transpose) {
      rtlRtn = RTE_matmul_cplx16mxv_t;
    } else {
      rtlRtn = RTE_matmul_cplx16;
    }
    break;
  case TY_BLOG:
    rtlRtn = RTE_matmul_log1;
    break;
  case TY_SLOG:
    rtlRtn = RTE_matmul_log2;
    break;
  case TY_LOG:
    rtlRtn = RTE_matmul_log4;
    break;
  case TY_LOG8:
    rtlRtn = RTE_matmul_log8;
    break;
  default:
    error(456, 3, gbl.lineno, CNULL, CNULL);
  }

  /* MORE if shape is set appropriately, the requirement that lhs is
   *      contiguous can be dropped
   */
  arg1 = ARGT_ARG(func_args, 0);
  arg2 = ARGT_ARG(func_args, 1);
  check_arg_isalloc(arg1);
  check_arg_isalloc(arg2);

  if (matmul_transpose) {
    nargs = 4;
    newargt = mk_argt(nargs);
    srcarray = ARGT_ARG(func_args, 0);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = ARGT_ARG(func_args, 1);
    ARGT_ARG(newargt, 3) = astb.i1; /* place holder in case we recognize
                                     * more than this one case
                                     */
  } else {
    /* use general purpose F90 matmul */
    nargs = 3;
    newargt = mk_argt(nargs);
    srcarray = ARGT_ARG(func_args, 0);
    ARGT_ARG(newargt, 1) = srcarray;
    ARGT_ARG(newargt, 2) = ARGT_ARG(func_args, 1);
  }

  name = mkRteRtnNm(rtlRtn);

  newsym = sym_mkfunc(name, DT_NONE);
  /* get the temp and add the necessary statements */
  temp_arr =
      mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1), lhs, &retval);
  if (temp_arr != 0) {
    /* add temp_arr as argument */
    ARGT_ARG(newargt, 0) = retval;
    if (ALLOCG(temp_arr)) {
      mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, 0);
      mk_mem_deallocate(mk_id(temp_arr), arg_gbl.std);
    }
    tmp_lhs_array = TRUE;
  } else {
    /* lhs was distributed properly for this intr */
    ARGT_ARG(newargt, 0) = lhs;
    retval = 0;
  }
  /* add call to function */
  /* make every call ICALL iff call changes the first argument and
     no side effect, this will help optimizer
     */
  ast = mk_func_node(A_ICALL, mk_id(newsym), nargs, newargt);
  A_OPTYPEP(ast, A_OPTYPEG(func_ast));
  add_stmt_before(ast, arg_gbl.std);
  return retval;
}

typedef struct { /* info for each fast matmul array/vector argument */
  int rank;      /* at most 2 */
  int ldim;      /* "leading dimension" */
  int extent[2]; /* number of elements for each dimension */
  int addr;      /* beginning address of the argument */
} MMUL;
static LOGICAL mmul_arg(int, int, MMUL *);
static LOGICAL mmul_array(int);
static int add_byval(int);

/*
 * func_ast: A_FUNC or A_INTR
 * func_args: rewritten args
 * lhs: ast for lhs (temp) if non-zero
 */
static int
mmul(int func_ast, int func_args, int lhs)
{
  /* func_ast is a function or intrinsic call.  If it is a transformational
   * intrinsic, create an appropriate temp, rewrite, and return a load
   * of that temp.
   * For now, don't do anything with user-defined functions.
   *
   * RTE_mmul_real4(ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
   * performs
   *
   * C = alpha*MATMUL(op(A), op(B)) + beta*C
   * where
   *     op(X) = X
   *     op(X) = TRANSPOSE(X)
   *     op(X) = CONJG(X)
   *
   * V   ta   : Integer(32 bits)
   *            0: no TRANSPOSE nor CONJG
   *            1: TRANSPOSE(A)
   *            2: CONJG(A)
   * V   tb   : Integer(32 bits)
   *            0: no TRANSPOSE nor CONJG
   *            1: TRANSPOSE(B)
   *            2: CONJG(B)
   * V   m    : Integer
   *            The number of rows of (transposed) A and C
   * V   n    : Integer
   *            The number of columns of B and C
   * V   k    : Integer
   *            The number of columns of (transposed) A and the number of
   *            rows of B
   * R   alpha: <matrix element type>
   *            The scalar alpha.
   * R   a    : <matrix element type>
   *            Matrix A.
   * V   lda  : Integer
   *            Leading dimension of (pre-transposed) A
   * R   b    : <matrix element type>
   *            Matrix B.
   * V   ldb  : Integer
   *            Leading dimension of B
   * R   beta : <matrix element type>
   *            The scalar beta.
   * R   c    : <matrix element type>
   *            Output Matrix C.
   * V   ldc  : Integer
   *            Leading dimension of C
   *
   * V - pass by value; unless specified, value is a 64-bit integer
   *     for a 64-bit target and 32-bit, otherwise,
   * R - pass by reference
   *
   * Our interface allows for
   * VxM - matmul(vectorA, matrixB) -> vectorC
   * MxV - matmul(matrixA, vectorB) -> vectorC
   *
   * For VxM:
   *   m   = 1
   *   k   = length of A & number of rows of B
   *   n   = number of columns of B and the length of C
   *   lda = 1
   *   ldb = as before
   *   ldc = 1
   *
   * For MxV:
   *   m   = number of rows of A and the length of C
   *   k   = number of columns of A and the length of B
   *   n   = 1
   *   lda = as before
   *   ldb = k
   *   ldc = m
   */
  int shape, rank;
  int dtype, elem_dty;
  int newsym;
  int temp_arr;
  int newargt;
  int arrA, arrB;
  INT ta, tb; /* transpose flags, actual values */
  MMUL mmA, mmB, mmC;
  int alpha, beta; /* ST_CONST symtab entries */
  INT num[2];
  int retval;
  int ast;
  int nargs;
  int subscr[MAXSUBS];
  FtnRtlEnum rtlRtn;

  retval = -1;
  if (XBIT(47, 0x10000000))
    return -1;
  /*
   * A_OPTYPEG(func_ast):
   * case I_MATMUL:	         matmul(matrix_a, matrix_b)
   * case I_MATMUL_TRANSPOSE:	 matmul(transpose(matrix_a), matrix_b)
   */
  dtype = A_DTYPEG(func_ast);
  elem_dty = DTY(dtype + 1);
  switch (elem_dty) {
  case DT_REAL4:
    alpha = stb.flt1;
    beta = stb.flt0;
    rtlRtn = RTE_mmul_real4;
    break;
  case DT_REAL8:
    alpha = stb.dbl1;
    beta = stb.dbl0;
    rtlRtn = RTE_mmul_real8;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case DT_QUAD:
    alpha = stb.quad1;
    beta = stb.quad0;
    rtlRtn = RTE_mmul_real16;
    break;
#endif
  case DT_CMPLX8:
    num[0] = CONVAL2G(stb.flt1);
    num[1] = CONVAL2G(stb.flt0);
    alpha = getcon(num, DT_CMPLX8);
    num[0] = CONVAL2G(stb.flt0);
    num[1] = CONVAL2G(stb.flt0);
    beta = getcon(num, DT_CMPLX8);
    rtlRtn = RTE_mmul_cmplx8;
    break;
  case DT_CMPLX16:
    num[0] = stb.dbl1;
    num[1] = stb.dbl0;
    alpha = getcon(num, DT_CMPLX16);
    num[0] = stb.dbl0;
    num[1] = stb.dbl0;
    beta = getcon(num, DT_CMPLX16);
    rtlRtn = RTE_mmul_cmplx16;
    break;
  default:
    return -1;
  }
  ta = tb = 0;
  if (A_OPTYPEG(func_ast) == I_MATMUL_TRANSPOSE) {
    /*
     * First  argument is a transpose of a 2D matrix.
     * Second argument is a vector.
     */
    ta = 1;
  }
  /* it only handles calls */
  shape = A_SHAPEG(func_ast);
  rank = SHD_NDIM(shape);

  /* MORE if shape is set appropriately, the requirement that lhs is
   *      contiguous can be dropped
   */
  arrA = ARGT_ARG(func_args, 0);
  arrB = ARGT_ARG(func_args, 1);
  if (!mmul_arg(arrA, ta, &mmA))
    return -1;
  if (!mmul_arg(arrB, 0, &mmB))
    return -1;
  if (matmul_use_lhs(lhs, rank, elem_dty)) {
    if (!mmul_arg(lhs, 0, &mmC))
      return -1;
    /*
     * A question here is if the lhs is not suitable as C, should
     * we go ahead and create a temp and call the fast matmul at
     * expense of 2 sets of copying memory, i.e.,
     *  tmp = matmu(A, B);
     *  C = tmp;
     * If YES, need to restructure when/how we perform
     *  temp_arr = mk_result_sptr(func_ast, ... ;
     * which is currently done below ...
     */
  }
  if (mmA.rank == 1) {
    /*  VxM  */
    mmA.extent[0] = mmA.extent[1]; /* m is 1 */
    mmA.extent[1] = mmB.extent[0]; /* k from B */
    mmA.ldim = mmA.extent[0];      /* 1 */
  } else if (mmB.rank == 1) {
    /*  MxV  */
    /* n is 1 */
    mmB.extent[0] = mmA.extent[1]; /* k */
  }
  nargs = 13;
  newargt = mk_argt(nargs);
  newsym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_NONE);
  ARGT_ARG(newargt, 0) = add_byval(mk_cval1(ta, DT_INT4));
  ARGT_ARG(newargt, 1) = add_byval(mk_cval1(tb, DT_INT4));
  ARGT_ARG(newargt, 2) = add_byval(mmA.extent[0]); /* m */
  ARGT_ARG(newargt, 3) = add_byval(mmB.extent[1]); /* n */
  ARGT_ARG(newargt, 4) = add_byval(mmA.extent[1]); /* k */
  ARGT_ARG(newargt, 5) = mk_cnst(alpha);
  ARGT_ARG(newargt, 6) = mmA.addr;
  ARGT_ARG(newargt, 7) = add_byval(mmA.ldim);
  ARGT_ARG(newargt, 8) = mmB.addr;
  ARGT_ARG(newargt, 9) = add_byval(mmB.ldim);
  ARGT_ARG(newargt, 10) = mk_cnst(beta);

  /* get the temp and add the necessary statements */
  temp_arr =
      mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1), lhs, &retval);
  if (temp_arr != 0) {
    /* add temp_arr as argument */
    (void)mmul_arg(retval, 0, &mmC);
    if (ALLOCG(temp_arr)) {
      mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, 0);
      mk_mem_deallocate(mk_id(temp_arr), arg_gbl.std);
    }
  } else {
    /* lhs was distributed properly for this intr */
    ARGT_ARG(newargt, 11) = lhs;
    retval = 0;
  }
  if (mmA.rank == 1) {
    mmC.ldim = mmA.extent[0]; /* 1 */
  }
  ARGT_ARG(newargt, 11) = mmC.addr;
  ARGT_ARG(newargt, 12) = add_byval(mmC.ldim);

  /* add call to function */
  /* make every call ICALL iff call changes the first argument and
     no side effect, this will help optimizer
     */
  ast = mk_func_node(A_ICALL, mk_id(newsym), nargs, newargt);
  A_OPTYPEP(ast, A_OPTYPEG(func_ast));
  add_stmt_before(ast, arg_gbl.std);
#if DEBUG
  ccff_info(MSGOPT, "OPT049", 1, STD_LINENO(arg_gbl.std),
            "MATMUL replaced by call to %mmul", "mmul=%s", mkRteRtnNm(rtlRtn),
            NULL);
#endif

  return retval;
}

static LOGICAL
mmul_arg(int arr, int transpose, MMUL *mm)
{
  int sptr;
  int shape;
  int ldim;
  int rank, i;
  int lb, ub, stride;
  int m;

  sptr = find_array(arr, NULL);
  if (POINTERG(sptr)
#ifdef CONTIGATTRG
      && !CONTIGATTRG(sptr)
#endif
  )
    return FALSE;
  shape = A_SHAPEG(arr);
  if (!shape)
    return FALSE;
  mm->rank = SHD_NDIM(shape);
  if (ASSUMSHPG(sptr) && mm->rank != 1
#ifdef CONTIGATTRG
      && !CONTIGATTRG(sptr)
#endif
  ) {
    /*
     * assumed-shaped arrays are guaranteed to be stride 1 in
     * just the first dimension.
     */
    return FALSE;
  }
  if (A_TYPEG(arr) == A_ID) {
    /*  whole */
    mm->addr = arr;
  } else if (A_TYPEG(arr) == A_MEM) {
    /*  whole -- allowing unsubscripted members is new as of 5/25/2012;
     *  so to back out, just add 'return FALSE;' here.
     */
    mm->addr = arr;
  } else if (mmul_array(arr)) {
    int asd;
    int subscr[MAXSUBS];
    asd = A_ASDG(arr);
    rank = ASD_NDIM(asd);
    for (i = 0; i < rank; ++i) {
      int ss;
      ss = ASD_SUBS(asd, i);
      if (A_TYPEG(ss) == A_TRIPLE) {
        subscr[i] = A_LBDG(ss);
      } else {
        subscr[i] = ss;
      }
    }
    mm->addr = mk_subscr(A_LOPG(arr), subscr, rank, DDTG(A_DTYPEG(arr)));
  } else
    return FALSE;

  for (i = 0; i < mm->rank; i++) {
    lb = SHD_LWB(shape, i);
    ub = SHD_UPB(shape, i);
    stride = SHD_STRIDE(shape, i);
    m = mk_binop(OP_SUB, ub, lb, astb.bnd.dtype);
    m = mk_binop(OP_ADD, m, stride, astb.bnd.dtype);
    mm->extent[i] = m;
  }
  /* ldim must be before any tranpose */
  if (STYPEG(sptr) == ST_MEMBER) {
    return FALSE;
  }
#ifdef NOEXTENTG
  else if (HCCSYMG(sptr) && SCG(sptr) == SC_LOCAL && ALLOCG(sptr) &&
           (NOEXTENTG(sptr) || simply_contiguous(arr))) {
    /*
     * the EXTNTAST temp may not be defined for compiler-created
     * allocatable temps assigned the value of the argument.
     */
    ADSC *tad;
    tad = AD_DPTR(DTYPEG(sptr));
    ldim = mk_extent_expr(AD_LWBD(tad, 0), AD_UPBD(tad, 0));
  }
#endif
#ifdef CONTIGATTRG
  else if (CONTIGATTRG(sptr)) {
    ADSC *tad;
    tad = AD_DPTR(DTYPEG(sptr));
    ldim = mk_extent_expr(AD_LWBD(tad, 0), AD_UPBD(tad, 0));
  }
#endif
  else
    return FALSE;
  if (transpose) {
    /*  extents are post-tranposed */
    m = mm->extent[0];
    mm->extent[0] = mm->extent[1];
    mm->extent[1] = m;
  }
  if (astb.bnd.dtype != DT_INT8) {
    ldim = mk_convert(ldim, DT_INT8);
    for (i = 0; i < mm->rank; i++) {
      mm->extent[i] = mk_convert(mm->extent[i], DT_INT8);
    }
  }
  if (mm->rank == 1)
    mm->extent[1] = astb.k1;
  mm->ldim = ldim;
  return TRUE;
}

/* Check if each section is contiguous or whole */
static LOGICAL
mmul_array(int arr_ast)
{
  int asd, ss;
  int ndims, dim;
  int ast1;
  LOGICAL any;

  ast1 = A_TYPEG(arr_ast) == A_MEM ? A_MEMG(arr_ast) : arr_ast;
  if (!ast1)
    return FALSE;

  if (!A_SHAPEG(ast1) || A_TYPEG(ast1) == A_ID)
    return TRUE;
  asd = A_ASDG(ast1);
  ndims = ASD_NDIM(asd);
  any = FALSE;
  for (dim = ndims - 1; dim >= 0; dim--) {
    ss = ASD_SUBS(asd, dim);
    if (A_TYPEG(ss) == A_TRIPLE) {
      if (!stride1_triple(ss)) {
        return FALSE;
      }
      any = TRUE;
      continue;
    }
    if (DTY(A_DTYPEG(ss)) == TY_ARRAY) {
      /*
       * No vector indexing ...
       */
      return FALSE;
    }
    if (any) {
      /*
       * The sections must be in consecutive leading dimensions
       */
      return FALSE;
    }
  }
  return TRUE;
}

static int
add_byval(int arg)
{
  int ast;
  ast = mk_unop(OP_VAL, arg, A_DTYPEG(arg));
  return ast;
}

/* reshape(source, shape, [pad, order]) */
static int
reshape(int func_ast, int func_args, int lhs)
{
  int dtype;
  int newsym;
  int temp_arr;
  int newargt;
  int srcarray;
  int retval;
  int ast;
  int nargs;
  FtnRtlEnum rtlRtn;
  int subscr[MAXSUBS];
  int ast_from_len;
  LOGICAL tmp_lhs_array;

  dtype = A_DTYPEG(func_ast);
  retval = _reshape(func_args, dtype, lhs);
  if (retval > 0) {
    return retval;
  }
  ast_from_len = 0;
  tmp_lhs_array = FALSE;
  if (DTYG(dtype) == TY_CHAR) {
    rtlRtn = RTE_reshapeca;
    if (DDTG(dtype) == DT_ASSCHAR || DDTG(dtype) == DT_ASSNCHAR ||
        DDTG(dtype) == DT_DEFERCHAR || DDTG(dtype) == DT_DEFERNCHAR) {
      ast_from_len = ARGT_ARG(func_args, 0);
    }
  } else
    rtlRtn = RTE_reshape;
  nargs = 5;
  srcarray = ARGT_ARG(func_args, 0);
  newargt = mk_argt(nargs);
  ARGT_ARG(newargt, 1) = srcarray;
  ARGT_ARG(newargt, 2) = ARGT_ARG(func_args, 1);
  if (ARGT_ARG(func_args, 2) == 0)
    if (DTYG(dtype) == TY_CHAR)
      ARGT_ARG(newargt, 3) = astb.ptr0c;
    else
      ARGT_ARG(newargt, 3) = astb.ptr0;
  else
    ARGT_ARG(newargt, 3) = ARGT_ARG(func_args, 2);
  if (ARGT_ARG(func_args, 3) == 0)
    ARGT_ARG(newargt, 4) = astb.ptr0;
  else
    ARGT_ARG(newargt, 4) = ARGT_ARG(func_args, 3);
  /* get the name of the library routine */
  newsym = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
  /* get the temp and add the necessary statements */
  /* need to put this into a temp */
  temp_arr =
      mk_result_sptr(func_ast, func_args, subscr, DTY(dtype + 1), lhs, &retval);
  if (temp_arr != 0) {
    /* add temp_arr as argument */
    ARGT_ARG(newargt, 0) = retval;
    if (ALLOCG(temp_arr)) {
      mk_mem_allocate(mk_id(temp_arr), subscr, arg_gbl.std, ast_from_len);
      mk_mem_deallocate(mk_id(temp_arr), arg_gbl.std);
    }
    tmp_lhs_array = TRUE;
  } else {
    /* lhs was distributed properly for this intr */
    ARGT_ARG(newargt, 0) = lhs;
    retval = 0;
  }
  /* add call to function */
  /* make every call ICALL iff call changes the first argument and
   * no side effect, this will help optimizer
   */
  ast = mk_func_node(A_ICALL, mk_id(newsym), nargs, newargt);
  A_OPTYPEP(ast, A_OPTYPEG(func_ast));
  add_stmt_before(ast, arg_gbl.std);
  return retval;
}

/* reshape(source, shape, [pad, order])
 *
 * Attempt to optimize reshape by representing the result of the reshape
 * as a (Cray) pointer of the source argument.  The requirements for this
 * optimization are:
 * o  pad & order are not present
 * o  the source:
 * o  +  is not pointer
 * o  +  is not assumed-shape array with rank > 1 unless the shape is in the
 *       first dimension
 * o  +  is contiguous
 * o  +  if character, has constant length
 * o  +  if member, shape is not in the parent
 * o  the extent of the shape array is constant
 */
static int
_reshape(int func_args, DTYPE dtype, int lhs)
{
  int retval;
  int srcarr, shparr; /* source & shape arguments, resp. */
  int sptr;
  int i, extnt;
  int shpdt, edt;
  int arrelem;
  int subs, subs_dt, stride;
  int ast, ast2, asn;
  int subscr[MAXSUBS];
  int temp;
  int temp_p;
  ADSC *ad;
  int mult;
  int zbase;

  retval = 0;
  if (XBIT(47, 0x20000000))
    return 0;
  if (ARGT_ARG(func_args, 2) || ARGT_ARG(func_args, 3))
    /* pad and order must not be present */
    return 0;
  if (DTYG(dtype) == TY_CHAR) {
    if (DDTG(dtype) == DT_ASSCHAR || DDTG(dtype) == DT_ASSNCHAR ||
        DDTG(dtype) == DT_DEFERCHAR || DDTG(dtype) == DT_DEFERNCHAR) {
      return 0;
    }
  }
  srcarr = ARGT_ARG(func_args, 0);
  sptr = find_array(srcarr, NULL);
  if (POINTERG(sptr))
    return 0;
  if (STYPEG(sptr) != ST_MEMBER && SCG(sptr) == SC_DUMMY && ASSUMSHPG(sptr) &&
      rank_of_sym(sptr) > 1) {
    int shd;
    shd = A_SHAPEG(srcarr);
    if (SHD_NDIM(shd) > 1)
      return 0;
    /*
     * is the shape in the first dimension and contiguous?
     * will be decided a few lines below by the call to
     * contiguous_section()
     */
  }
  /*
   * Ignore member reference whose shape is in the parent.
   */
  if (A_TYPEG(srcarr) == A_MEM && !A_SHAPEG(A_MEMG(srcarr)))
    return 0;
  /*
   * if subscripted, make sure the source is contiguous.
   */
  if (A_TYPEG(srcarr) == A_SUBSCR && !contiguous_section(srcarr))
    return 0;
  shparr = ARGT_ARG(func_args, 1);
  if (A_TYPEG(shparr) == A_MEM && !A_SHAPEG(A_MEMG(shparr)))
    /*
     * At this time, ignore if the parent has 'shape'; generating the
     * subscripted refs  of the shape array is currently relatively simple.
     */
    return 0;

  shpdt = A_DTYPEG(shparr);
  extnt = extent_of_shape(A_SHAPEG(shparr), 0);
  if (!extnt || !A_ALIASG(extnt))
    return 0;
  extnt = get_int_cval(A_SPTRG(A_ALIASG(extnt)));
  edt = DTY(shpdt + 1);
  /*
   * Someday, it sure would be nice if we could detect that the shape
   * array represents an array constructor of 'contant' values.
   * But for now, just make the 'shape' adjustable.
   *
   * Create a adjustable array (Cray) pointer temp.  It will by
   * marked 'RESHAPED' indicating that it will be representing a
   * section of memory that has been reshape and that the address
   * will be stored in its 'hidden' the pointer variable.
   */
  temp = sym_get_array("reshap", "r", DTY(A_DTYPEG(srcarr) + 1), extnt);
  SCP(temp, SC_BASED);
  RESHAPEDP(temp, 1);
  /*
   * Create the 'hidden' pointer that will locate the beginning of the
   * memory.
   */
  temp_p = sym_get_ptr(temp);
  MIDNUMP(temp, temp_p);
  ADJARRP(temp, 1);
  SEQP(temp, 1);
  /*
   * Generate the subscripted references of the shape argument to
   * represent the upper bounds of each dimension of the result.
   * The bounds will be:
   *  ( 1:SHAPE(1), 1:SHAPE(2), ... )
   * Also, create the bounds temps for the upper bound(s), multiplier(s),
   * and 'zbase'
   */
  /*fprintf(STDERR, "RESHAPE SHP ");dbg_print_ast(shparr,0);*/
  arrelem = first_element(shparr);
  /*fprintf(STDERR, "RESHAPE SHP1");dbg_print_ast(arrelem,0);*/
  subs = ASD_SUBS(A_ASDG(arrelem), 0); /*  the first subscript value */
  subs_dt = A_DTYPEG(subs);
  stride = SHD_STRIDE(A_SHAPEG(shparr), 0);
  if (!stride || stride == astb.bnd.one)
    stride = mk_cval(1, subs_dt);
  else if (A_DTYPEG(stride) != subs_dt) {
    stride = mk_convert(stride, subs_dt);
  }
  ad = AD_DPTR(DTYPEG(temp));
  AD_ADJARR(ad) = 1;
  i = 0;
  while (1) {
    AD_LWBD(ad, i) = 0;
    AD_LWAST(ad, i) = astb.bnd.one;
    if (A_DTYPEG(arrelem) == astb.bnd.dtype)
      AD_UPBD(ad, i) = arrelem;
    else
      AD_UPBD(ad, i) = mk_convert(arrelem, astb.bnd.dtype);
    AD_UPAST(ad, i) = mk_bnd_ast();
    AD_EXTNTAST(ad, i) = AD_UPAST(ad, i);
    if (i == 0) {
      AD_MLPYR(ad, i) = astb.bnd.one;
    } else {
      AD_MLPYR(ad, i) = mk_bnd_ast();
    }
    i++;
    if (i >= extnt)
      break;
    subs = mk_binop(OP_ADD, subs, stride, subs_dt);
    subscr[0] = subs;
    arrelem = mk_subscr(A_LOPG(arrelem), subscr, 1, edt);
  }
  /*
   * Generate
   *   'hidden pointer' = loc(source)
   */
  ast = ast_intr(I_LOC, DT_PTR, 1, first_element(srcarr));
  ast2 = mk_id(temp_p);
  asn = mk_assn_stmt(ast2, ast, DT_PTR);
  add_stmt_before(asn, arg_gbl.std);
  /*fprintf(STDERR, "RESHAPE LOC");dbg_print_ast(asn,0);*/
  /*
   * Generate
   *   the assignments to the upper bound and zbase temps
   */
  mult = astb.bnd.one;
  AD_MLPYR(ad, 0) = mult;
  for (i = 0; i < extnt; i++) {
    asn = mk_assn_stmt(AD_UPAST(ad, i), AD_UPBD(ad, i), astb.bnd.dtype);
    add_stmt_before(asn, arg_gbl.std);
    if (i) {
      mult = mk_mlpyr_expr(astb.bnd.one, AD_UPAST(ad, i - 1), mult);
      asn = mk_assn_stmt(AD_MLPYR(ad, i), mult, astb.bnd.dtype);
      add_stmt_before(asn, arg_gbl.std);
    }
  }
  zbase = mk_zbase_expr(ad);
  if (A_ALIASG(zbase)) {
    AD_ZBASE(ad) = zbase;
  } else {
    AD_ZBASE(ad) = mk_bnd_ast();
    asn = mk_assn_stmt(AD_ZBASE(ad), zbase, astb.bnd.dtype);
    add_stmt_before(asn, arg_gbl.std);
  }
  /*
   * Return the temp, expressed as a whole section in each dimension,
   * Simply returning 'temp' is not sufficient if we need to build a
   * descriptor, such as in
   *    print *, reshape(yy,[3,4])  !!! need descriptor for reshape
   */
  retval = mk_id(temp);
  retval = convert_subscript_in_expr(retval);
  /*fprintf(STDERR, "RESHAPE"); dbg_print_ast(retval,0);*/

  return retval;
}

/** \brief Rewrite intrinsic LBOUND/UBOUND to runtime call.
 *
 *  \param func_ast ast for the intrinsic call
 *  \param actual   corresponding actual for a assumed-shape formal that is
 *                  used as the array parameter in the intrinsic call
 *  \param nextstd  insert the generated stmts before this stmt
 */
int
rewrite_lbound_ubound(int func_ast, int actual, int nextstd)
{
  DTYPE dtype, arrdtype;
  int func_args, optype, array, dim, nargs, newargt, subscr[MAXDIMS],
      result, ast;
  SPTR sptr, actual_sptr, hpf_sym, temp_arr;
  FtnRtlEnum rtlRtn;

  func_args = A_ARGSG(func_ast);
  optype = A_OPTYPEG(func_ast);
  dtype = A_DTYPEG(func_ast);
  array = ARGT_ARG(func_args, 0);
  arrdtype = A_DTYPEG(array);
  /* The KIND parameter has been eliminated and is represented in dtype. */
  if (ARGT_CNT(func_args) == 2)
    dim = ARGT_ARG(func_args, 1);
  else
    dim = 0;
  sptr = get_whole_array_sym(array);
  result = 0;
  if (sptr && SDSCG(sptr) &&
      (POINTERG(sptr) || ALLOCG(sptr) || ASSUMRANKG(sptr))) {
    /* Get bound info from section descriptor. */
    if (dim) {
      if (optype == I_LBOUND) {
        switch (dtype) {
        case DT_BINT:
          rtlRtn = RTE_lbound1Dsc;
          break;
        case DT_SINT:
          rtlRtn = RTE_lbound2Dsc;
          break;
        case DT_INT4:
          rtlRtn = RTE_lbound4Dsc;
          break;
        case DT_INT8:
          rtlRtn = RTE_lbound8Dsc;
          break;
        default:
          error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                "invalid kind argument for lbound");
          rtlRtn = RTE_lboundDsc;
          break;
        }
      } else {
        switch (dtype) {
        case DT_BINT:
          rtlRtn = RTE_ubound1Dsc;
          break;
        case DT_SINT:
          rtlRtn = RTE_ubound2Dsc;
          break;
        case DT_INT4:
          rtlRtn = RTE_ubound4Dsc;
          break;
        case DT_INT8:
          rtlRtn = RTE_ubound8Dsc;
          break;
        default:
          error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                "invalid kind argument for ubound");
          rtlRtn = RTE_uboundDsc;
          break;
        }
      }
      /* pghpf...bound(dim, static_desciptor) */
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), dtype);
      nargs = 2;
      newargt = mk_argt(nargs);
      ARGT_ARG(newargt, 0) = dim;
      ARGT_ARG(newargt, 1) = check_member(array, mk_id(SDSCG(sptr)));
      DESCUSEDP(sptr, 1);
      goto ret_func;
    } else {
      if (!XBIT(68, 0x1) || XBIT(68, 0x2)) {
        if (optype == I_LBOUND) {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_lbounda1Dsc;
            break;
          case DT_SINT:
            rtlRtn = RTE_lbounda2Dsc;
            break;
          case DT_INT4:
            rtlRtn = RTE_lbounda4Dsc;
            break;
          case DT_INT8:
            rtlRtn = RTE_lbounda8Dsc;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for lbound");
            rtlRtn = RTE_lboundaDsc;
            break;
          }
        } else {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_ubounda1Dsc;
            break;
          case DT_SINT:
            rtlRtn = RTE_ubounda2Dsc;
            break;
          case DT_INT4:
            rtlRtn = RTE_ubounda4Dsc;
            break;
          case DT_INT8:
            rtlRtn = RTE_ubounda8Dsc;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for ubound");
            rtlRtn = RTE_uboundaDsc;
            break;
          }
        }
      } else {
        if (optype == I_LBOUND) {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_lboundaz1Dsc;
            break;
          case DT_SINT:
            rtlRtn = RTE_lboundaz2Dsc;
            break;
          case DT_INT4:
            rtlRtn = RTE_lboundaz4Dsc;
            break;
          case DT_INT8:
            rtlRtn = RTE_lboundaz8Dsc;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for lbound");
            rtlRtn = RTE_lboundazDsc;
            break;
          }
        } else {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_uboundaz1Dsc;
            break;
          case DT_SINT:
            rtlRtn = RTE_uboundaz2Dsc;
            break;
          case DT_INT4:
            rtlRtn = RTE_uboundaz4Dsc;
            break;
          case DT_INT8:
            rtlRtn = RTE_uboundaz8Dsc;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for ubound");
            rtlRtn = RTE_uboundazDsc;
            break;
          }
        }
      }
      /* pghpf...bounda(temp, sd) or
       * pghpf...boundaz(temp, sd) for -Mlarge_arrays
       */
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_NONE);
      nargs = 2;
      newargt = mk_argt(nargs);
      ARGT_ARG(newargt, 1) = check_member(array, mk_id(SDSCG(sptr)));
      DESCUSEDP(sptr, 1);
      goto ret_call;
    }
  } else {
    /* Get bound info from dtype or shape. */
    int rank = rank_of_ast(array);
    if (actual)
      actual_sptr = get_whole_array_sym(actual);
    else
      actual_sptr = SPTR_NULL;
    if (dim) {
      if (A_ALIASG(dim)) {
        int i = get_int_cval(A_SPTRG(A_ALIASG(dim)));
        if (actual) {
          int lb, extent, mask;
          if (actual_sptr && SDSCG(actual_sptr) &&
              (POINTERG(actual_sptr) || ALLOCG(actual_sptr))) {
            /* The whole array actual_sptr corresponding to an assumed-shape
             * formal cannot be assumed-rank. */
            extent = get_extent(SDSCG(actual_sptr), i - 1);
          } else {
            extent = extent_of_shape(A_SHAPEG(actual), i - 1);
          }
          lb = ADD_LWBD(arrdtype, i - 1);
          lb = ast_rewrite(lb); /* Replace formal in boundary */
          mask = mk_binop(OP_GT, extent, astb.bnd.zero, DT_LOG);
          lb = mk_merge(lb, astb.bnd.one, mask, astb.bnd.dtype);
          if (optype == I_LBOUND) {
            result = lb;
          } else {
            /* The extent of formal parameter is equal to the extent of actual
             * parameter. */
            result = mk_binop(OP_ADD, lb, extent, astb.bnd.dtype);
            result = mk_binop(OP_SUB, result, astb.bnd.one, astb.bnd.dtype);
          }
        } else if (sptr) {
          int lb, ub;
          lb = ADD_LWAST(arrdtype, i - 1);
          if (!lb)
            lb = astb.bnd.one;
          ub = ADD_UPAST(arrdtype, i - 1);
          if (optype == I_LBOUND) {
            if (ADD_ASSUMSZ(arrdtype) && i == rank) {
              result = lb;
            } else {
              int mask = mk_binop(OP_LE, lb, ub, DT_LOG);
              result = mk_merge(lb, astb.bnd.one, mask, astb.bnd.dtype);
            }
          } else {
            int mask = mk_binop(OP_LE, lb, ub, DT_LOG);
            result = mk_merge(ub, astb.bnd.zero, mask, astb.bnd.dtype);
          }
        } else {
          if (optype == I_LBOUND)
            result = astb.bnd.one;
          else
            result = extent_of_shape(A_SHAPEG(array), i - 1);
        }
        goto ret_val;
      } else {
        if (optype == I_LBOUND) {
          switch (dtype) {
          case DT_BINT:
            rtlRtn = RTE_lb1;
            break;
          case DT_SINT:
            rtlRtn = RTE_lb2;
            break;
          case DT_INT4:
            rtlRtn = RTE_lb4;
            break;
          case DT_INT8:
            rtlRtn = RTE_lb8;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for lbound");
            rtlRtn = RTE_lb;
            break;
          }
        } else {
          switch (dtype) {
          case DT_BINT:
            rtlRtn = RTE_ub1;
            break;
          case DT_SINT:
            rtlRtn = RTE_ub2;
            break;
          case DT_INT4:
            rtlRtn = RTE_ub4;
            break;
          case DT_INT8:
            rtlRtn = RTE_ub8;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for ubound");
            rtlRtn = RTE_ub;
            break;
          }
        }
        /* f90...bound(rank, dim, l1, u1, l2, u2, ..., l<rank>, u<rank>) */
        hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), astb.bnd.dtype);
        nargs = 2 + 2 * rank;
        newargt = mk_argt(nargs);
        ARGT_ARG(newargt, 0) = mk_isz_cval(rank, astb.bnd.dtype);
        if (actual)
          dim = ast_rewrite(dim); /* Replace formal in DIM */
        ARGT_ARG(newargt, 1) = dim;
      }
    } else {
      if (!XBIT(68, 0x1) || XBIT(68, 0x2)) {
        if (optype == I_LBOUND) {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_lba1;
            break;
          case DT_SINT:
            rtlRtn = RTE_lba2;
            break;
          case DT_INT4:
            rtlRtn = RTE_lba4;
            break;
          case DT_INT8:
            rtlRtn = RTE_lba8;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for lbound");
            rtlRtn = RTE_lba;
            break;
          }
        } else {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_uba1;
            break;
          case DT_SINT:
            rtlRtn = RTE_uba2;
            break;
          case DT_INT4:
            rtlRtn = RTE_uba4;
            break;
          case DT_INT8:
            rtlRtn = RTE_uba8;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for ubound");
            rtlRtn = RTE_uba;
            break;
          }
        }
      } else {
        /* -Mlarge_arrays, but the result is default integer */
        if (optype == I_LBOUND) {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_lbaz1;
            break;
          case DT_SINT:
            rtlRtn = RTE_lbaz2;
            break;
          case DT_INT4:
            rtlRtn = RTE_lbaz4;
            break;
          case DT_INT8:
            rtlRtn = RTE_lbaz8;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for lbound");
            rtlRtn = RTE_lbaz;
            break;
          }
        } else {
          switch (DDTG(dtype)) {
          case DT_BINT:
            rtlRtn = RTE_ubaz1;
            break;
          case DT_SINT:
            rtlRtn = RTE_ubaz2;
            break;
          case DT_INT4:
            rtlRtn = RTE_ubaz4;
            break;
          case DT_INT8:
            rtlRtn = RTE_ubaz8;
            break;
          default:
            error(155, 3, gbl.lineno, SYMNAME(gbl.currsub),
                  "invalid kind argument for ubound");
            rtlRtn = RTE_ubaz;
            break;
          }
        }
      }
      /* f90...bounda(temp, rank, l1, u1, l2, u2, ..., l<rank>, u<rank>) or
       * f90...boundaz(temp, rank, l1, u1, l2, u2, ..., l<rank>, u<rank>) for
       * -Mlarge_arrays
       */
      hpf_sym = sym_mkfunc_nodesc(mkRteRtnNm(rtlRtn), DT_NONE);
      nargs = 2 + 2 * rank;
      newargt = mk_argt(nargs);
      ARGT_ARG(newargt, 1) = mk_isz_cval(rank, astb.bnd.dtype);
    }
    /* l1, u1, l2, u2, ..., l<rank>, u<rank> */
    for (int i = 0; i < rank; i++) {
      int lb, ub;
      if (actual) {
        int extent, mask;
        if (actual_sptr && SDSCG(actual_sptr) &&
            (POINTERG(actual_sptr) || ALLOCG(actual_sptr))) {
          /* The whole array actual_sptr corresponding to an assumed-shape
           * formal cannot be assumed-rank. */
          extent = get_extent(SDSCG(actual_sptr), i);
        } else {
          extent = extent_of_shape(A_SHAPEG(actual), i);
        }
        lb = ADD_LWBD(arrdtype, i);
        lb = ast_rewrite(lb); /* Replace formal in boundary */
        mask = mk_binop(OP_GT, extent, astb.bnd.zero, DT_LOG);
        lb = mk_merge(lb, astb.bnd.one, mask, astb.bnd.dtype);
        /* The extent of formal parameter is equal to the extent of actual
         * parameter. */
        ub = mk_binop(OP_ADD, lb, extent, astb.bnd.dtype);
        ub = mk_binop(OP_SUB, ub, astb.bnd.one, astb.bnd.dtype);
      } else if (sptr) {
        lb = ADD_LWAST(arrdtype, i);
        if (!lb)
          lb = astb.bnd.one;
        if (ADD_ASSUMSZ(arrdtype) && i == rank - 1)
          ub = astb.ptr0;
        else
          ub = ADD_UPAST(arrdtype, i);
      } else {
        lb = astb.bnd.one;
        ub = extent_of_shape(A_SHAPEG(array), i);
      }
      ARGT_ARG(newargt, 2 + i * 2) = lb;
      ARGT_ARG(newargt, 3 + i * 2) = ub;
    }
    if (dim)
      goto ret_func;
    else
      goto ret_call;
  }
ret_func:
  ast = mk_func_node(A_FUNC, mk_id(hpf_sym), nargs, newargt);
  A_DTYPEP(ast, A_DTYPEG(func_ast));
  A_SHAPEP(ast, A_SHAPEG(func_ast));
  A_OPTYPEP(ast, optype);
  return ast;
ret_call:
  if (ADD_ASSUMRANK(arrdtype)) {
    temp_arr = mk_shape_sptr(A_SHAPEG(func_ast), subscr, DDTG(dtype));
    if (ALLOCG(temp_arr)) {
      mk_mem_allocate(mk_id(temp_arr), subscr, nextstd, 0);
      mk_mem_deallocate(mk_id(temp_arr), nextstd);
    }
  } else {
    temp_arr = get_arr_temp(dtype, TRUE, FALSE, FALSE);
    trans_mkdescr(temp_arr);
  }
  ARGT_ARG(newargt, 0) = mk_id(temp_arr);
  ast = mk_func_node(A_CALL, mk_id(hpf_sym), nargs, newargt);
  A_OPTYPEP(ast, optype);
  add_stmt_before(ast, nextstd);
  return mk_id(temp_arr);
ret_val:
  return result;
}
