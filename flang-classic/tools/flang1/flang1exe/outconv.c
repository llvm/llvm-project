/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
*   \brief Routines for descriptor optimizatons and forall transformations
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
#include "pragma.h"
#include "gramtk.h"
#include "extern.h"
#include "commopt.h"
#include "dpm_out.h"
#include "nme.h"
#include "optimize.h"
#include "pd.h"
#include "ccffinfo.h"
#define RTE_C
#include "rte.h"
#undef RTE_C
#include "comm.h"
#include "fdirect.h"
#include "rtlRtns.h"
#include "ilidir.h" /* for open_pragma, close_pragma */

static void convert_statements(void);
static void convert_simple(void);
static int conv_allocate(int std);
static int conv_deallocate(int std);
static LOGICAL is_same_mask(int expr, int expr1);
static LOGICAL no_effect_forall(int std);
static void init_collapse(void);
static void collapse_arrays(void);
static void end_collapse(void);
static void find_collapse_allocs(void);
static void find_collapse_defs(void);
static void delete_collapse(int ci);
static void find_collapse_uses(void);
static LOGICAL is_parent_loop(int lpParent, int lp);
static void collapse_loops(void);
static void find_descrs(void);
static void collapse_allocates(LOGICAL bDescr);
static void report_collapse(int lp);
#if DEBUG
#ifdef FLANG_OUTCONV_UNUSED
static void dump_collapse(void);
#endif
#endif
static int position_finder(int forall, int ast);
static void find_calls_pos(int std, int forall, int must_pos);
static void find_mask_calls_pos(int forall);
static void find_stmt_calls_pos(int forall, int mask_pos);
static int find_max_of_mask_calls_pos(int forall);
static void add_mask_calls(int pos, int forall, int stdnext);
static void add_stmt_calls(int pos, int forall, int stdnext);
static void forall_dependency(int std);
static void put_calls(int pos, int std, int stdnext);
static void search_pure_function(int stdfirst, int stdlast);
static int transform_pure_function(int expr, int std);
static void eliminate_barrier(void);
static void remove_mask_calls(int forall);
static void remove_stmt_calls(int forall);
static void move_mask_calls(int forall);
static LOGICAL is_stmt_call_dependent(int forall, int lhs);
static LOGICAL is_mask_call_dependent(int forall, int lhs);
static LOGICAL is_call_dependent(int std, int forall, int lhs);
static void convert_omp_workshare(void);
static void insert_assign(int lhs, int rhs, int beforestd);

static void convert_template_instance(void);
#define NO_PTR XBIT(49, 0x8000)
#define NO_CHARPTR XBIT(58, 0x1)
#define NO_DERIVEDPTR XBIT(58, 0x40000)

#undef MKASSN
#define MKASSN(d, s) mk_assn_stmt(d, s, 0)

void
convert_output(void)
{
  if (XBIT(49, 1))
    return;

  if (flg.opt >= 2 && !XBIT(47, 0x10)) {
    init_collapse();
    collapse_arrays();
  }
  convert_statements();
  FREE(ftb.base);
  freearea(FORALL_AREA);
  if (flg.opt >= 2 && !XBIT(47, 0x10)) {
    collapse_allocates(TRUE);
    end_collapse();
  }
  eliminate_barrier();
  free_brtbl();
  transform_wrapup();
  comm_fini();
  convert_simple();
  if (XBIT(58, 0x10000000))
    convert_template_instance();
}

/*
 *  keep track of forall temp arrays
 *
 */
#define TEMP_AREA 6

typedef struct T_LIST {
  struct T_LIST *next;
  int temp, asd, dtype, cvlen, sc, std, astd, dstd;
} T_LIST;

#define GET_T_LIST(q) q = (T_LIST *)getitem(TEMP_AREA, sizeof(T_LIST))
static T_LIST *templist;
static int beforestd;
static int newsymnum;

static void
early_flow_init(void)
{
  optshrd_init();
  flowgraph();
  findloop(0);
  flow();
}

static void
early_flow_fini(void)
{
  optshrd_fend();
  optshrd_end();
}

void
forall_dependency_analyze(void)
{
  int std;
  int ast;
  int parallel_depth;
  int task_depth;

  templist = NULL;
  parallel_depth = 0;
  task_depth = 0;
  for (std = STD_NEXT(0); std;) {
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_MP_PARALLEL:
      ++parallel_depth;
      set_descriptor_sc(SC_PRIVATE);
      break;
    case A_MP_ENDPARALLEL:
      --parallel_depth;
      if (parallel_depth == 0 && task_depth == 0) {
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
      if (parallel_depth == 0 && task_depth == 0) {
        set_descriptor_sc(SC_LOCAL);
      }
      break;
    case A_FORALL:
      if (STD_DELETE(std)) {
        ast_to_comment(ast);
        std = STD_NEXT(std);
        continue;
      }
      forall_dependency(std);
      break;
    }
    std = STD_NEXT(std);
  }
  freearea(TEMP_AREA);
  templist = NULL;
}

void
convert_forall(void)
{
  int std;
  int ast;

  if (XBIT(49, 2))
    return;

  if (flg.opt >= 2 && XBIT(53, 2)) {
    points_to();
  }
  /*
   * need to do early flow analysis to determine if lhs really need temp.
   * NOTE: -Hx,4,0x200000 is not useful at all; eventually a crash could
   * occur in nmeutil because a NME table is not created (nmeb.stg_base is
   * null).
   */
  if (flg.opt >= 2 && !XBIT(4, 0x200000)) {
    early_flow_init();
  }

  if (!XBIT(4, 0x100000))
    forall_dependency_analyze();
  /* we need to redo the flow graph forall_dependency_analyze can add more node
   * into the flow */
  init_region();
  for (std = STD_NEXT(0); std;) {
    arg_gbl.inforall = FALSE;
    ast = STD_AST(std);
    check_region(std);
    switch (A_TYPEG(ast)) {
    case A_FORALL:
      arg_gbl.inforall = TRUE;
      std = conv_forall(std);
      break;
    default:
      std = STD_NEXT(std);
      break;
    }
  }
  if (flg.smp) {
    convert_omp_workshare();
  }
  if (flg.opt >= 2 && !XBIT(4, 0x200000)) {
    early_flow_fini();
  }
  if (flg.opt >= 2 && XBIT(53, 2)) {
    f90_fini_pointsto();
  }
}

#define NO_WRKSHR 0
#define IN_WRKSHR 1
#define IN_PDO 2
#define IN_SINGLE 3
#define IN_PARALLEL 4
#define IN_CRITICAL 5

static int
gen_pdo(int do_ast)
{
  int ast;

  ast = mk_stmt(A_MP_PDO, 0);
  A_DOVARP(ast, A_DOVARG(do_ast));
  A_LASTVALP(ast, A_LASTVALG(do_ast));
  A_M1P(ast, A_M1G(do_ast));
  A_M2P(ast, A_M2G(do_ast));
  A_M3P(ast, A_M3G(do_ast));
  A_CHUNKP(ast, 0);
  A_SCHED_TYPEP(ast, 0); /* STATIC */
  A_ORDEREDP(ast, 0);
  A_LASTVALP(ast, 0);
  A_DISTRIBUTEP(ast, 0);
  A_DISTPARDOP(ast, 0);
  A_ENDLABP(ast, 0);
  A_DISTCHUNKP(ast, 0);
  A_TASKLOOPP(ast, 0);

  return ast;
}

static void
gen_endsingle(int std, int single, int presinglebarrier)
{
  int ompast;
  int ompstd;
  int singlestd = A_STDG(single);

  if (presinglebarrier &&
      A_TYPEG(STD_AST(STD_PREV(singlestd))) != A_MP_BARRIER) {
    add_stmt_before(mk_stmt(A_MP_BARRIER, 0), singlestd);
  }

  ompast = mk_stmt(A_MP_ENDSINGLE, 0);
  A_LOPP(single, ompast);
  A_LOPP(ompast, single);
  ompstd = add_stmt_before(ompast, std);
  add_stmt_after(mk_stmt(A_MP_BARRIER, 0), ompstd);
}

static void
convert_omp_workshare(void)
{
  int std;
  int newstd = 0;
  int ast;
  int lsptr;
  int prevast;
  int state = NO_WRKSHR;
  int dolevel = 0;
  int parallellevel = 0;
  int wherelevel = 0;
  int ompast;
  int ompstd;
  int single;
  int presinglebarrier = 0;
  int parallel_depth = 0;

  for (std = STD_NEXT(0); std; std = STD_NEXT(std)) {
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_MP_PARALLEL:
      ++parallel_depth;
      break;
    case A_MP_ENDPARALLEL:
      --parallel_depth;
      break;
    case A_MP_WORKSHARE:
    case A_MP_ENDWORKSHARE:
      if (parallel_depth > 1) {
        ast_to_comment(ast);
        ast = STD_AST(std);
      }
      break;
    }

    if (state != NO_WRKSHR && A_TYPEG(ast) == A_ALLOC &&
        A_TKNG(ast) == TK_DEALLOCATE) {
      int sptr = sym_of_ast(A_SRCG(ast));
      if (CCSYMG(sptr) || HCCSYMG(sptr)) {
        /* dealloc of a compiler generated temp, make sure
         * any OMP SINGLEs are preceded by a barrier */
        presinglebarrier++;
      }
    }

    switch (state) {
    case NO_WRKSHR:
      if (A_TYPEG(ast) == A_MP_WORKSHARE) {
        state = IN_WRKSHR;
      }
      break;
    case IN_WRKSHR:
      switch (A_TYPEG(ast)) {
      case A_MP_ENDWORKSHARE:
        state = NO_WRKSHR;
        break;
      case A_DO:
        prevast = STD_AST(STD_PREV(std));
        if (A_TYPEG(prevast) == A_COMMENT &&
            A_TYPEG(A_LOPG(prevast)) == A_FORALL) {
          ompast = gen_pdo(ast);
          newstd = add_stmt_before(ompast, std);
          if (parallel_depth > 1)
            STD_PAR(newstd) = 1;
          dolevel++;
          state = IN_PDO;
          ast_to_comment(ast);
        } else {
          /* probably an elemental intrinsic */
          single = mk_stmt(A_MP_SINGLE, 0);
          add_stmt_before(single, std);
          dolevel++;
          state = IN_SINGLE;
        }
        break;
      case A_MP_PARALLEL:
        single = mk_stmt(A_MP_SINGLE, 0);
        add_stmt_before(single, std);
        parallellevel++;
        state = IN_PARALLEL;
        break;
      case A_MP_CRITICAL:
        single = mk_stmt(A_MP_SINGLE, 0);
        add_stmt_before(single, std);
        state = IN_CRITICAL;
        break;
      case A_COMMENT:
        switch (A_TYPEG(A_LOPG(ast))) {
        case A_WHERE:
          wherelevel++;
          break;
        case A_ENDWHERE:
          wherelevel--;
          break;
        }
        break;
      case A_ALLOC:
        break;
      case A_ASN:
        lsptr = sym_of_ast(A_DESTG(ast));
        if (wherelevel) {
          if (HCCSYMG(lsptr)) {
            THREADP(lsptr, 1);
            break;
          }
        } else if (HCCSYMG(lsptr) && SCG(lsptr) == SC_PRIVATE) {
          break;
        }
        FLANG_FALLTHROUGH;
      default:
        single = mk_stmt(A_MP_SINGLE, 0);
        add_stmt_before(single, std);
        state = IN_SINGLE;
        break;
      }
      break;
    case IN_PDO:
      switch (A_TYPEG(ast)) {
      case A_DO:
        dolevel++;
        break;
      case A_ENDDO:
        if (--dolevel == 0) {
          ompast = mk_stmt(A_MP_ENDPDO, 0);
          ompstd = add_stmt_after(ompast, std);
          add_stmt_after(mk_stmt(A_MP_BARRIER, 0), ompstd);
          std = STD_NEXT(ompstd);
          state = IN_WRKSHR;
          ast_to_comment(ast);
        }
        break;
      case A_COMMENT:
        /* This case (WHERE or ENDWHERE in a DO) may never happen,
         * but the comment STDs can sometimes get shuffled and may
         * be out of order.  Just to be safe */
        switch (A_TYPEG(A_LOPG(ast))) {
        case A_WHERE:
          wherelevel++;
          break;
        case A_ENDWHERE:
          wherelevel--;
          break;
        }
        break;
      }
      break;
    case IN_SINGLE:
      switch (A_TYPEG(ast)) {
      case A_MP_ENDWORKSHARE:
        gen_endsingle(std, single, presinglebarrier);
        presinglebarrier = 0;
        state = NO_WRKSHR;
        break;
      case A_DO:
        prevast = STD_AST(STD_PREV(std));
        if (A_TYPEG(prevast) == A_COMMENT &&
            A_TYPEG(A_LOPG(prevast)) == A_FORALL) {
          gen_endsingle(STD_PREV(std), single, presinglebarrier);
          presinglebarrier = 0;
          ompast = gen_pdo(ast);
          newstd = add_stmt_before(ompast, std);
          if (parallel_depth > 1)
            STD_PAR(newstd) = 1;
          dolevel++;
          state = IN_PDO;
          ast_to_comment(ast);
        } else {
          dolevel++;
        }
        break;
      case A_ENDDO:
          dolevel--;
        break;
      case A_COMMENT:
        switch (A_TYPEG(A_LOPG(ast))) {
        case A_FORALL:
          gen_endsingle(std, single, presinglebarrier);
          presinglebarrier = 0;
          state = IN_WRKSHR;
          break;
        }
        break;
      case A_MP_PARALLEL:
        state = IN_PARALLEL;
        parallellevel++;
        break;
      case A_MP_CRITICAL:
        state = IN_CRITICAL;
        break;
      }
      break;
    case IN_PARALLEL:
      switch (A_TYPEG(ast)) {
      case A_MP_PARALLEL:
        parallellevel++;
        break;
      case A_MP_ENDPARALLEL:
        if (--parallellevel == 0) {
          state = IN_SINGLE;
        }
        break;
      }
      if (newstd)
        STD_PAR(newstd) = 1;

      break;
    case IN_CRITICAL:
      if (A_TYPEG(ast) == A_MP_ENDCRITICAL) {
        state = IN_SINGLE;
      }
      break;
    }
  }
}

static LOGICAL
no_effect_forall(int std)
{
  int forall;
  int asn;
  int count;
  int fusedstd;
  int nd;
  int i;

  count = 0;
  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);
  if (A_SRCG(asn) == A_DESTG(asn))
    count++;
  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NFUSE(nd, 0); i++) {
    fusedstd = FT_FUSEDSTD(nd, 0, i);
    forall = STD_AST(fusedstd);
    asn = A_IFSTMTG(forall);
    if (A_SRCG(asn) == A_DESTG(asn))
      count++;
  }

  if (count == FT_NFUSE(nd, 0) + 1) {
    delete_stmt(std);
    return TRUE;
  }
  return FALSE;
}

/*
 * replace pghpf_lbound/pghpf_ubound(dim,descriptor)
 */
static int
_pghpf_bound(int lbound, int ast)
{
  int argt, arg0, arg1, dim, ss[1], dtype, newast, offset;
  newast = ast;
  argt = A_ARGSG(ast);
  arg0 = ARGT_ARG(argt, 0);
  arg1 = ARGT_ARG(argt, 1);
  if ((A_TYPEG(arg1) == A_ID && DESCARRAYG(A_SPTRG(arg1))) ||
      (A_TYPEG(arg1) == A_MEM && DESCARRAYG(A_SPTRG(A_MEMG(arg1))))) {
    /* arg1 is a section descriptor */
    dtype = A_DTYPEG(arg1);
    if (A_ALIASG(arg0)) {
      arg0 = A_ALIASG(arg0);
      /* get constant value */
      dim = get_int_cval(A_SPTRG(arg0));
      offset = get_global_lower_index(dim - 1);
      ss[0] = mk_cval((INT)offset, DT_INT);
      newast = mk_subscr(arg1, ss, 1, DDTG(dtype));
      if (!lbound) {
        int a, b;
        offset = get_global_extent_index(dim - 1);
        ss[0] = mk_cval((INT)offset, DT_INT);
        b = mk_subscr(arg1, ss, 1, DDTG(dtype));
        a = mk_cval(1, astb.bnd.dtype);
        b = mk_binop(OP_SUB, b, a, astb.bnd.dtype);
        newast = mk_binop(OP_ADD, b, newast, astb.bnd.dtype);
      }
    } else {
      /* dimension is not constant, compute offset */
      int base;
      int arg0decr = mk_binop(OP_SUB, arg0, astb.i1, DT_INT);
      base = get_global_lower_index(0);
      offset = get_global_lower_index(1);
      offset = offset - base;
      ss[0] = mk_cval((INT)(offset), DT_INT);
      ss[0] = mk_binop(OP_MUL, arg0decr, ss[0], DT_INT);
      ss[0] = mk_binop(OP_ADD, mk_cval((INT)base, DT_INT), ss[0], DT_INT);
      newast = mk_subscr(arg1, ss, 1, DDTG(dtype));
      if (!lbound) {
        int a, b;
        base = get_global_extent_index(0);
        ss[0] = mk_cval((INT)(offset), DT_INT);
        ss[0] = mk_binop(OP_MUL, arg0decr, ss[0], DT_INT);
        ss[0] = mk_binop(OP_ADD, mk_cval((INT)base, DT_INT), ss[0], DT_INT);
        b = mk_subscr(arg1, ss, 1, DDTG(dtype));
        a = mk_cval(1, astb.bnd.dtype);
        b = mk_binop(OP_SUB, b, a, astb.bnd.dtype);
        newast = mk_binop(OP_ADD, b, newast, astb.bnd.dtype);
      }
    }
  }
  return newast;
} /* _pghpf_bound */

/*
 * replace pghpf_size(dim,descriptor)/pghpf_extent(descriptor,dim)
 */
static int
_pghpf_size(int size, int ast)
{
  int argt, arg0, arg1, dim, ss[1], dtype, newast, offset;
  newast = ast;
  argt = A_ARGSG(ast);
  if (size) {
    arg0 = ARGT_ARG(argt, 0); /* dim */
    arg1 = ARGT_ARG(argt, 1); /* section descriptor */
  } else {
    arg0 = ARGT_ARG(argt, 1); /* dim */
    arg1 = ARGT_ARG(argt, 0); /* section descriptor */
  }
  if ((A_TYPEG(arg1) == A_ID && DESCARRAYG(A_SPTRG(arg1))) ||
      (A_TYPEG(arg1) == A_MEM && DESCARRAYG(A_SPTRG(A_MEMG(arg1))))) {
    /* arg1 is a section descriptor */
    dtype = A_DTYPEG(arg1);
    if (arg0 == astb.ptr0) {
      /* global size */
      ss[0] = mk_cval(get_desc_gsize_index(), DT_INT);
      newast = mk_subscr(arg1, ss, 1, DDTG(dtype));
    } else if (A_ALIASG(arg0)) {
      arg0 = A_ALIASG(arg0);
      /* get constant value */
      dim = get_int_cval(A_SPTRG(arg0));
      offset = get_global_extent_index(dim - 1);
      ss[0] = mk_cval((INT)offset, DT_INT);
      newast = mk_subscr(arg1, ss, 1, DDTG(dtype));
    } else {
      /* dimension is not constant, compute offset */
      int base;
      int arg0decr = mk_binop(OP_SUB, arg0, astb.i1, DT_INT);
      base = get_global_extent_index(0);
      offset = get_global_extent_index(1);
      ss[0] = mk_cval((INT)(offset - base), DT_INT);
      ss[0] = mk_binop(OP_MUL, arg0decr, ss[0], DT_INT);
      ss[0] = mk_binop(OP_ADD, mk_cval((INT)base, DT_INT), ss[0], DT_INT);
      newast = mk_subscr(arg1, ss, 1, DDTG(dtype));
    }
  }
  return newast;
} /* _pghpf_size */

/*
 * replace RTE_size(rank,dim,l1,u1,s1,l2,u2,s2,...)
 */
static int
_RTE_size(int ast)
{
  int argt, arg0, arg1, argl, argu, args, rank, dim, newast, i;
  newast = ast;
  argt = A_ARGSG(ast);
  arg0 = ARGT_ARG(argt, 0); /* rank */
  arg1 = ARGT_ARG(argt, 1); /* dim */
  if (A_ALIASG(arg0)) {
    arg0 = A_ALIASG(arg0);
    rank = get_int_cval(A_SPTRG(arg0));
    if (A_ARGCNTG(ast) == rank * 3 + 2) {
      if (arg1 == astb.ptr0) {
        newast = 0;
        for (i = 0; i < rank; ++i) {
          int a;
          argl = ARGT_ARG(argt, i * 3 + 2);
          argu = ARGT_ARG(argt, i * 3 + 3);
          args = ARGT_ARG(argt, i * 3 + 4);
          a = mk_binop(OP_SUB, argu, argl, A_DTYPEG(argl));
          a = mk_binop(OP_ADD, a, args, A_DTYPEG(argl));
          if (args != astb.i1 && args != astb.bnd.one) {
            a = mk_binop(OP_DIV, a, args, A_DTYPEG(argl));
          }
          if (!newast) {
            newast = a;
          } else {
            newast = mk_binop(OP_MUL, newast, a, A_DTYPEG(a));
          }
        }
      } else if (A_ALIASG(arg1)) {
        arg1 = A_ALIASG(arg1);
        dim = get_int_cval(A_SPTRG(arg1));
        if (dim >= 1 && dim <= rank) {
          int a;
          argl = ARGT_ARG(argt, (dim - 1) * 3 + 2);
          argu = ARGT_ARG(argt, (dim - 1) * 3 + 3);
          args = ARGT_ARG(argt, (dim - 1) * 3 + 4);
          a = mk_binop(OP_SUB, argu, argl, A_DTYPEG(argl));
          a = mk_binop(OP_ADD, a, args, A_DTYPEG(argl));
          if (args != astb.i1 && args != astb.bnd.one) {
            a = mk_binop(OP_DIV, a, args, A_DTYPEG(argl));
          }
          newast = a;
        }
      }
    }
  }
  return newast;
} /* _RTE_size */

/*
 * replace pgi_element_size( array )
 */
static int
_pgi_element_size(int ast)
{
  int argt, arg0, sptr, dtype, ret;
  argt = A_ARGSG(ast);
  arg0 = ARGT_ARG(argt, 0); /* variable or array */
  sptr = memsym_of_ast(arg0);
  if (sptr <= NOSYM) {
    return astb.i0;
  }
  dtype = DDTG(DTYPEG(sptr));
  ret = mk_cval(size_of(dtype), DT_INT);
  return ret;
} /* _pgi_element_size */

/*
 * replace pgi_kind( array )
 */
static int
_pgi_kind(int ast)
{
  int argt, arg0, sptr, dtype, ret;
  argt = A_ARGSG(ast);
  arg0 = ARGT_ARG(argt, 0); /* variable or array */
  sptr = memsym_of_ast(arg0);
  if (sptr <= NOSYM) {
    return astb.i0;
  }
  dtype = DDTG(DTYPEG(sptr));
  ret = mk_cval(dtype_to_arg(dtype), DT_INT);
  return ret;
} /* _pgi_kind */

/*
 * return an expression that gives the size of dimension i of a shape
 * descriptor
 */
static int
size_shape(int shape, int i)
{
  int a, mask;
  int args = SHD_STRIDE(shape, i);
  int argl = SHD_LWB(shape, i);
  int argu = SHD_UPB(shape, i);
  a = mk_binop(OP_SUB, argu, argl, astb.bnd.dtype);
  a = mk_binop(OP_ADD, a, args, astb.bnd.dtype);
  a = mk_binop(OP_DIV, a, args, astb.bnd.dtype);
  /* 'a' is calculated as ((ub - lb + s) / s)
   * which works for negative strides as well.
   * Negative results are converted to zero.
   */
  mask = mk_binop(OP_GE, a, astb.bnd.zero, DT_LOG);
  a = mk_merge(a, astb.bnd.zero, mask, astb.bnd.dtype);
  if (astb.bnd.dtype != stb.user.dt_int) {
    /* -i8: type of size is integer*8 so convert result */
    a = mk_convert(a, stb.user.dt_int);
  }
  return a;
} /* size_shape */

/*
 * replace size(array,dim) (from shape descriptor)
 */
static int
_PDsize(int ast)
{
  int argt, argdim, arg, dim, ss[1], dtype, newast, offset, argsym, argsdsc;
  int rank;
  newast = ast;
  argt = A_ARGSG(ast);
  arg = ARGT_ARG(argt, 0);    /* section descriptor */
  argdim = ARGT_ARG(argt, 1); /* dim */
  argsym = 0;
  argsdsc = 0;
  if (A_TYPEG(arg) == A_ID) {
    argsym = A_SPTRG(arg);
  } else if (A_TYPEG(arg) == A_MEM) {
    argsym = A_SPTRG(A_MEMG(arg));
  }
  if (argsym) {
    argsdsc = SDSCG(argsym);
    if (!argsdsc || !DESCUSEDG(argsdsc) || !DESCARRAYG(argsdsc) ||
        DTY(DTYPEG(argsym)) != TY_ARRAY) {
      argsdsc = 0;
    }
  }
  dtype = A_DTYPEG(arg);
  if (argsdsc) {
    /* arg is an array and has a section descriptor */
    if (argdim == astb.ptr0) {
      /* global size */
      ss[0] = mk_cval(get_desc_gsize_index(), DT_INT);
    } else if (A_ALIASG(argdim)) {
      argdim = A_ALIASG(argdim);
      /* get constant value */
      dim = get_int_cval(A_SPTRG(argdim));
      offset = get_global_extent_index(dim - 1);
      ss[0] = mk_cval((INT)offset, DT_INT);
    } else {
      /* dimension is not constant, compute offset */
      int base;
      base = get_global_extent_index(0);
      offset = get_global_extent_index(1);
      ss[0] = mk_cval((INT)(offset - base), DT_INT);
      ss[0] = mk_binop(OP_MUL, argdim, ss[0], DT_INT);
      ss[0] = mk_binop(OP_ADD, mk_cval(base - (offset - base), DT_INT), ss[0],
                       DT_INT);
    }
    newast = mk_subscr(mk_id(argsdsc), ss, 1, DTYPEG(argsdsc));
    newast = check_member(arg, newast);
  } else {
    /* compute size from the shape descriptor */
    int shape, i;

    shape = A_SHAPEG(arg); /* this shape is always stride one */
    rank = SHD_NDIM(shape);
    if (argdim == astb.ptr0) {
      /* global size */
      newast = 0;
      for (i = 0; i < rank; ++i) {
        int a, args;
        args = SHD_STRIDE(shape, i);
        if (args != astb.i1 && args != astb.bnd.one) {
          return ast;
        }
        a = size_shape(shape, i);
        if (!newast) {
          newast = a;
        } else {
          newast = mk_binop(OP_MUL, newast, a, A_DTYPEG(a));
        }
      }
    } else if (A_ALIASG(argdim)) {
      argdim = A_ALIASG(argdim);
      /* get constant value */
      dim = get_int_cval(A_SPTRG(argdim));
      newast = size_shape(shape, dim - 1);
    } else {
      /* dimension is not constant, give up */
      newast = ast;
    }
  }
  return newast;
} /* _PDsize */

/**
 * \brief Used to simplify PD_lbound or PD_ubound call ast nodes to the value
 *        from shape descriptor of adjustable array.
 * \param lbound Flag which represents whether this is a call to lbound routine.
 *               When set to zero, it means call is to ubound.
 * \param ast    The AST node representing the call to lbound/ubound.
 * \return       AST node representing value extracted from shape.
 */
static int
_PDbound(int lbound, int ast)
{
  int argt, argdim, arg, dim;
  int rank, shape, bound;
  argt = A_ARGSG(ast);
  arg = ARGT_ARG(argt, 0);
  argdim = ARGT_ARG(argt, 1);
  /* The implementation requires that argument is an array and that dimension argument
     is a constant. */
  if (A_TYPEG(arg) == A_ID &&
      DTY(A_DTYPEG(arg)) == TY_ARRAY &&
      A_ALIASG(argdim)) {
    shape = A_SHAPEG(arg);
    /* Replacement of bound call can only happen if shape is known */
    if (shape) {
      rank = SHD_NDIM(shape);
      argdim = A_ALIASG(argdim);
      dim = get_int_cval(A_SPTRG(argdim));
      if (lbound) {
        bound = SHD_LWB(shape, dim - 1);
      } else {
        bound = SHD_UPB(shape, dim - 1);
      }
      return bound;
    }
  }

  return ast;
} /* _PDbound */

/*
 * replace RTE_lbound/RTE_ubound(rank,dim,b1,b2,b3,...)
 */
static int
_RTE_bound(int lbound, int ast)
{
  int argt, arg0, arg1, rank, dim, newast;
  newast = ast;
  argt = A_ARGSG(ast);
  arg0 = ARGT_ARG(argt, 0); /* rank */
  arg1 = ARGT_ARG(argt, 1); /* dim */
  if (A_ALIASG(arg0)) {
    arg0 = A_ALIASG(arg0);
    rank = get_int_cval(A_SPTRG(arg0));
    if (A_ARGCNTG(ast) == rank + 2) {
      if (A_ALIASG(arg1)) {
        arg1 = A_ALIASG(arg1);
        dim = get_int_cval(A_SPTRG(arg1));
        if (dim >= 1 && dim <= rank) {
          newast = ARGT_ARG(argt, (dim - 1) + 2 + (lbound ? 0 : 1));
        }
      }
    }
  }
  return newast;
} /* _RTE_bound */

/*
 * replace RTE_lb/RTE_ub(rank,dim,l1,u1,l1,u2,...)
 */
static int
_RTE_xb(int lbound, int ast, int rdt, int dolong)
{
  int argt, arg0, arg1, rank, dim, newast;
  newast = ast;
  argt = A_ARGSG(ast);
  arg0 = ARGT_ARG(argt, 0); /* rank */
  arg1 = ARGT_ARG(argt, 1); /* dim */
  if (A_ALIASG(arg0)) {
    arg0 = A_ALIASG(arg0);
    rank = get_int_cval(A_SPTRG(arg0));
    if (A_ARGCNTG(ast) == 2 * rank + 2) {
      if (A_ALIASG(arg1)) {
        arg1 = A_ALIASG(arg1);
        dim = get_int_cval(A_SPTRG(arg1));
        if (dim >= 1 && dim <= rank) {
          int tsource, fsource, mask; /* merge arguemnts */
          int ub, lb;
          lb = ARGT_ARG(argt, 2 * (dim - 1) + 2);
          ub = ARGT_ARG(argt, 2 * (dim - 1) + 2 + 1);
          if (lbound) {
            tsource = lb;
            fsource = astb.bnd.one;
          } else {
            tsource = ub;
            fsource = astb.bnd.zero;
          }
          mask = mk_binop(OP_LE, lb, ub, DT_LOG);
          newast = mk_merge(tsource, fsource, mask, dolong ? DT_INT8 : DT_INT);
          if (rdt) {
            newast = mk_convert(newast, rdt);
          }
        }
      }
    }
  }
  return newast;
} /* _RTE_xb */

/*
 * replace RTE_uba and RTE_lba
 */
static int
_RTE_ba(int lbound, int ast)
{
  int argt, arg0, arg1, rank, dim, lhs, rhs, ss[1];
  int ub, lb, newif, cmp, newstd;

  argt = A_ARGSG(ast);
  arg0 = ARGT_ARG(argt, 0); /* result array */
  arg1 = ARGT_ARG(argt, 1); /* rank */
  if (A_ALIASG(arg1)) {
    arg1 = A_ALIASG(arg1);
    rank = get_int_cval(A_SPTRG(arg1));
    if (A_ARGCNTG(ast) == rank * 2 + 2) {
      for (dim = 1; dim <= rank; dim++) {
        lb = ARGT_ARG(argt, (dim - 1) * 2 + 2);
        ub = ARGT_ARG(argt, (dim - 1) * 2 + 3);
        ss[0] = mk_cval((INT)dim, DT_INT);
        lhs = mk_subscr(arg0, ss, 1, DT_INT);

        if (dim == rank && ub == astb.ptr0) {
          /*
           * Special case for F77 assumed size arrays which
           * have no upper bound in the last dimension.
           */
          rhs = (lbound ? lb : astb.bnd.zero);
          insert_assign(lhs, rhs, beforestd);
        } else if (lbound && (lb == astb.i1 || lb == astb.bnd.one)) {
          /*
           * Special case for constant one lower bound.
           * No need for the if-then-else.
           */
          insert_assign(lhs, lb, beforestd);
        } else {
          /* if (lb <= ub) ... */
          newif = mk_stmt(A_IFTHEN, 0);
          cmp = mk_binop(OP_LE, lb, ub, DT_LOG);
          A_IFEXPRP(newif, cmp);
          newstd = add_stmt_before(newif, beforestd);
          STD_PAR(newstd) = STD_PAR(beforestd);
          STD_TASK(newstd) = STD_TASK(beforestd);
          STD_ACCEL(newstd) = STD_ACCEL(beforestd);
          STD_KERNEL(newstd) = STD_KERNEL(beforestd);

          /* lhs = (lbound ? lb : ub) */
          rhs = (lbound ? lb : ub);
          insert_assign(lhs, rhs, beforestd);

          /* else */
          newif = mk_stmt(A_ELSE, 0);
          newstd = add_stmt_before(newif, beforestd);
          STD_PAR(newstd) = STD_PAR(beforestd);
          STD_TASK(newstd) = STD_TASK(beforestd);
          STD_ACCEL(newstd) = STD_ACCEL(beforestd);
          STD_KERNEL(newstd) = STD_KERNEL(beforestd);

          /* lhs = (lbound ? 1 : 0) */
          rhs = (lbound ? astb.bnd.one : astb.bnd.zero);
          insert_assign(lhs, rhs, beforestd);

          /* end if */
          newif = mk_stmt(A_ENDIF, 0);
          newstd = add_stmt_before(newif, beforestd);
          STD_PAR(newstd) = STD_PAR(beforestd);
          STD_TASK(newstd) = STD_TASK(beforestd);
          STD_ACCEL(newstd) = STD_ACCEL(beforestd);
          STD_KERNEL(newstd) = STD_KERNEL(beforestd);
        }
      }
      ast_to_comment(ast);
    }
  }
  return ast;
}

/*
 * return operand of %val(), else just return the ast
 */
static int
value(int ast)
{
  if (ast > 0 && A_TYPEG(ast) == A_UNOP &&
      (A_OPTYPEG(ast) == OP_VAL || A_OPTYPEG(ast) == OP_BYVAL))
    ast = A_LOPG(ast);
  if (ast > 0 && A_ALIASG(ast))
    ast = A_ALIASG(ast);
  return ast;
} /* value */

/*
 * put value in a symbol if it's an expression
 */
static int
symvalue(int ast, char c, int num, int *ptemp, int var, int sdsc)
{
  int temp, newasn, newstd, a;
  ast = value(ast);
  if (!var && (A_TYPEG(ast) == A_ID || A_TYPEG(ast) == A_CNST))
    return ast;
  if (A_TYPEG(ast) == A_SUBSCR && A_TYPEG(A_LOPG(ast)) == A_ID &&
      (sdsc > 0 ? A_SPTRG(A_LOPG(ast)) == sdsc
                : DESCARRAYG(A_SPTRG(A_LOPG(ast)))))
    return ast;
  if (*ptemp == 0) {
    *ptemp = temp = getnewccsymf(ST_VAR, ".c%d_%d", num, newsymnum++);
    SCP(temp, SC_LOCAL);
    DTYPEP(temp, astb.bnd.dtype);
    if (STD_PAR(beforestd) || STD_TASK(beforestd))
      SCP(temp, SC_PRIVATE);
  }
  a = mk_id(*ptemp);
  if (ast == a)
    return ast;
  newasn = mk_stmt(A_ASN, 0);
  A_DESTP(newasn, a);
  A_SRCP(newasn, ast);
  newstd = add_stmt_before(newasn, beforestd);
  STD_PAR(newstd) = STD_PAR(beforestd);
  STD_TASK(newstd) = STD_TASK(beforestd);
  STD_ACCEL(newstd) = STD_ACCEL(beforestd);
  STD_KERNEL(newstd) = STD_KERNEL(beforestd);
  return a;
} /* symvalue */

/*
 * see above
 */
static void
_simple_replacements(int ast, int *pany)
{
  if (A_TYPEG(ast) == A_FUNC || A_TYPEG(ast) == A_CALL) {
    int lop;
    lop = A_LOPG(ast);
    if (lop && A_TYPEG(lop) == A_ID) {
      int fsptr;
      fsptr = A_SPTRG(lop);
      if (HCCSYMG(fsptr) && STYPEG(fsptr) == ST_PROC) {
        /* compiler created function */
        int newast;
        char *fname;
        int in_device_code;
        fname = SYMNAME(fsptr);
        newast = ast;
        in_device_code = 0;
        if (strcmp(fname, mkRteRtnNm(RTE_lboundDsc)) == 0 ||
            strcmp(fname, mkRteRtnNm(RTE_lbound1Dsc)) == 0 ||
            strcmp(fname, mkRteRtnNm(RTE_lbound2Dsc)) == 0 ||
            strcmp(fname, mkRteRtnNm(RTE_lbound4Dsc)) == 0 ||
            strcmp(fname, mkRteRtnNm(RTE_lbound8Dsc)) == 0) {
          newast = _pghpf_bound(1, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_uboundDsc)) == 0 ||
                   strcmp(fname, mkRteRtnNm(RTE_ubound1Dsc)) == 0 ||
                   strcmp(fname, mkRteRtnNm(RTE_ubound2Dsc)) == 0 ||
                   strcmp(fname, mkRteRtnNm(RTE_ubound4Dsc)) == 0 ||
                   strcmp(fname, mkRteRtnNm(RTE_ubound8Dsc)) == 0) {
          newast = _pghpf_bound(0, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_extent)) == 0) {
          newast = _pghpf_size(0, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_sizeDsc)) == 0) {
          newast = ARGT_ARG(A_ARGSG(ast), 1);
          if (A_TYPEG(newast) == A_ID && ASSUMRANKG(A_SPTRG(newast))) {
            return;
          }
          newast = _pghpf_size(1, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_size)) == 0) {
          newast = _RTE_size(ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_lbound)) == 0) {
          newast = _RTE_bound(1, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_ubound)) == 0) {
          newast = _RTE_bound(0, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_lba)) == 0) {
          if (in_device_code || XBIT(137, 0x20))
            newast = _RTE_ba(1, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_uba)) == 0) {
          if (in_device_code || XBIT(137, 0x20))
            newast = _RTE_ba(0, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_extent)) == 0) {
          newast = _pghpf_size(0, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_sizeDsc)) == 0) {
          newast = _pghpf_size(1, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_size)) == 0) {
          newast = _RTE_size(ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_lbound)) == 0) {
          newast = _RTE_bound(1, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_ubound)) == 0) {
          newast = _RTE_bound(0, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_lba)) == 0) {
          if (in_device_code || XBIT(137, 0x20))
            newast = _RTE_ba(1, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_uba)) == 0) {
          if (in_device_code || XBIT(137, 0x20))
            newast = _RTE_ba(0, ast);
        } else if (strcmp(fname, mkRteRtnNm(RTE_lb)) == 0) {
          /* Last arg:
           *  large arrays || ub/lb retval is 8 byte int || int is 8 byte
           */
          newast = _RTE_xb(1, ast, 0,
                             XBIT(68, 0x1) || XBIT(86, 0x2) || XBIT(128, 0x10));
        } else if (strcmp(fname, mkRteRtnNm(RTE_ub)) == 0) {
          /* Last arg:
           *  large arrays || ub/lb retval is 8 byte int || int is 8 byte
           */
          newast = _RTE_xb(0, ast, 0,
                             XBIT(68, 0x1) || XBIT(86, 0x2) || XBIT(128, 0x10));
        }
        if (newast != ast) {
          if (A_DTYPEG(newast) != A_DTYPEG(ast))
            newast = mk_convert(newast, A_DTYPEG(ast));
          ast_replace(ast, newast);
          *pany = *pany + 1;
        }
      } else if (XBIT(57, 0x4000000)) {
        int newast;
        char *fname;
        fname = SYMNAME(fsptr);
        newast = ast;
        if (strcmp(fname, "pgi_element_size") == 0) {
          newast = _pgi_element_size(ast);
        } else if (strcmp(fname, "pgi_kind") == 0) {
          newast = _pgi_kind(ast);
        }
        if (newast != ast) {
          if (A_DTYPEG(newast) != A_DTYPEG(ast))
            newast = mk_convert(newast, A_DTYPEG(ast));
          ast_replace(ast, newast);
          *pany = *pany + 1;
        }
      }
    }
  } else if (A_TYPEG(ast) == A_INTR) {
    int lop;
    lop = A_LOPG(ast);
    if (lop && A_TYPEG(lop) == A_ID) {
      int fsptr;
      fsptr = A_SPTRG(lop);
      if (STYPEG(fsptr) == ST_PD) {
        /* predeclared procedure */
        int newast;
        newast = ast;
        if (PDNUMG(fsptr) == PD_size) {
          /*  size(array,dim) ==> array$sd( extent(dim) ) if there is a $sd
           *		   ==> ubound(array,dim)-lbound(array,dim)+1 else
           *  size(array,<0>) ==> array$sd( gsize ) if there is a $sd
           *		   ==> product(ubound(array,dim)-lbound(array,dim)+1)
           *else
           *  size(expr,dim) ==> ubound(shape,dim)-lbound(shape,dim)+1
           *  size(expr,dim) ==> product(ubound(shape,dim)-lbound(shape,dim)+1)
           */
          newast = _PDsize(ast);
        } else if (PDNUMG(fsptr) == PD_lbound) {
          newast = _PDbound(1, ast);
        } else if (PDNUMG(fsptr) == PD_ubound) {
          newast = _PDbound(0, ast);
        }
        if (newast != ast) {
          if (A_DTYPEG(newast) != A_DTYPEG(ast))
            newast = mk_convert(newast, A_DTYPEG(ast));
          ast_replace(ast, newast);
          *pany = *pany + 1;
        }
      }
    }
  }
} /* _simple_replacements */

static void
convert_simple(void)
{
  int std, stdnext;
  int ast, any;

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    ast = STD_AST(std);
    ast_visit(1, 1);
    any = 0; /* any replacements found? */
    beforestd = std;
    ast_traverse(ast, NULL, _simple_replacements, &any);
    if (any) {
      while (any > 0) {
        ast = ast_rewrite(ast);
        any--;
      }
      STD_AST(std) = ast;
      A_STDP(ast, std);
    }
    ast_unvisit();
  }
} /* convert_simple */

/*
 * check that this is a single subscript with constant value as given
 */
static int
check_subscript(int ast, int value)
{
  int asd, ss, val;
  asd = A_ASDG(ast);
  if (ASD_NDIM(asd) != 1)
    return 0;
  ss = ASD_SUBS(asd, 0);
  if (A_TYPEG(ss) != A_CNST)
    return 0;
  val = get_int_cval(A_SPTRG(ss));
  if (value != val)
    return 0;
  return 1;
} /* check_subscript */

/*
 * check that the constant value matches what we expect
 */
static int
check_value(int ast, int value)
{
  int val;
  if (A_TYPEG(ast) != A_CNST)
    return 0;
  val = get_int_cval(A_SPTRG(ast));
  if (value != val)
    return 0;
  return 1;
} /* check_value */

/*
 * for RTE_sect calls, see if the lower bound / upper bound / stride
 * arguments for this dimension are the corresponding full dimension.
 *  lower bound = section descriptor(lbound)
 *  upper bound = section descriptor(ubound) OR
 *  upper bound = section descriptor(lbound) + (section descriptor(extent)-1)
 *  stride = 1
 */
static int
full_dimension(int astlower, int astupper, int aststride, int dim)
{
  int sdsc = 0;
  if (!check_value(aststride, 1))
    return 0;
  if (A_TYPEG(astlower) == A_SUBSCR) {
    if (A_TYPEG(A_LOPG(astlower)) != A_ID)
      return 0;
    sdsc = A_SPTRG(A_LOPG(astlower));
    if (!DESCARRAYG(sdsc))
      return 0;
    if (!check_subscript(astlower, get_global_lower_index(dim)))
      return 0;
  } else {
    return 0;
  }
  if (A_TYPEG(astupper) == A_SUBSCR) {
    if (A_TYPEG(A_LOPG(astupper)) != A_ID)
      return 0;
    if (A_SPTRG(A_LOPG(astupper)) != sdsc)
      return 0;
    if (!check_subscript(astupper, get_global_upper_index(dim)))
      return 0;
  } else if (A_TYPEG(astupper) == A_BINOP && A_OPTYPEG(astupper) == OP_ADD) {
    int astleft, astright;
    astleft = A_LOPG(astupper);
    astright = A_ROPG(astupper);
    if (A_TYPEG(astleft) == A_SUBSCR) {
      if (A_TYPEG(A_LOPG(astleft)) != A_ID)
        return 0;
      if (A_SPTRG(A_LOPG(astleft)) != sdsc)
        return 0;
      if (!check_subscript(astleft, get_global_lower_index(dim)))
        return 0;
    } else {
      return 0;
    }
    if (A_TYPEG(astright) == A_BINOP && A_OPTYPEG(astright) == OP_SUB) {
      astleft = A_LOPG(astright);
      astright = A_ROPG(astright);
      if (A_TYPEG(astleft) == A_SUBSCR) {
        if (A_TYPEG(A_LOPG(astleft)) != A_ID)
          return 0;
        if (A_SPTRG(A_LOPG(astleft)) != sdsc)
          return 0;
        if (!check_subscript(astleft, get_global_extent_index(dim)))
          return 0;
      } else {
        return 0;
      }
      if (!check_value(astright, 1))
        return 0;
    } else {
      return 0;
    }
  } else {
    return 0;
  }
  return sdsc;
} /* full_dimension */

/*
 * insert an assignment statement
 */
static void
insert_assign(int lhs, int rhs, int beforestd)
{
  int newasn, newstd;
  if (lhs == rhs)
    return;
  newasn = MKASSN(lhs, rhs);
  newstd = add_stmt_before(newasn, beforestd);
  STD_PAR(newstd) = STD_PAR(beforestd);
  STD_TASK(newstd) = STD_TASK(beforestd);
  STD_ACCEL(newstd) = STD_ACCEL(beforestd);
  STD_KERNEL(newstd) = STD_KERNEL(beforestd);
} /* insert_assign */

/*
 * replace RTE_sect calls
 * RTE_sect( newsd, oldsd, dims, [lower, upper, stride,]... flags )
 *
 * newsd.rank = rank	-- must be constant
 * newsd.kind = oldsd.kind
 * newsd.bytelen = oldsd.bytelen
 * flagstemp = oldsd.flags	-- handle constant case here
 * newsd.lsize = oldsd.lsize
 * newsd.gbase = oldsd.gbase
 * d=0
 * if flagstemp & SECTZBASE
 *  lbasetemp = 1
 * else
 *  lbasetemp = oldsd.lbase
 * endif
 * gsizetemp = 1
 * for r = 0 to rank-1 do
 *  if flags & (1<<r) then  -- section dimension
 *   upper = oldsd.upper[r]
 *   lower = oldsd.lower[r]
 *   stride = oldsd.stride[r]
 *   set extent=upper-lower+stride
 *   if stride == -1 then extent = -extent
 *   elseif stride != 1 then extent /= stride; endif
 *   if flags & SECTZBASE then
 *    if extent < 0 then extent = 0 endif
 *    newsd[d].lbound = 1
 *    newsd[d].ubound = extent
 *    newsd[d].lstride = stride * oldsd[r].lstride
 *    lbasetemp -= newsd[d].lstride
 *   else
 *    if extent < 0 then extent = 0; upper = lower-1; stride=1; endif
 *    newsd[d].extent = extent
 *    if flags & NOREINDEX and stride == 1 then
 *     newsd[d].lbound = lower
 *     newsd[d].ubound = upper
 *     set myoffset=0
 *    else
 *     newsd[d].lbound = 1
 *     newsd[d].ubound = extent
 *     set myoffset = lower-stride
 *    endif
 *    newsd[d].lstride = stride * oldsd[r].lstride
 *    lbasetemp += myoffset * oldsd[r].lstride
 *   endif
 *   newsd[d].sstride = 1
 *   newsd[d].soffset = 0
 *   if newsd[d].lstride != gsizetemp then reset flagstemp -= SEQUENTIAL_SECTION
 *endif
 *   set gsizetemp *= extent
 *   ++d
 *  else
 *   set lidx = oldsd[r].sstride * oldsd[r].lbound + oldsd[r].soffset =
 *oldsd[r].lbound
 *   set k = oldsd[r].lstride * ( lidx - oldsd[r].lbound )
 *         = oldsd[r].lstride * ( lower - oldsd[r].lbound )
 *         = oldsd[r].lstride * lower - oldsd[r].lstride * oldsd[r].lbound
 *   lbasetemp += k + (oldsd[r].lstride * oldsd[r].lbound)
 *             += oldsd[r].lstride * lower - oldsd[r].lstride * oldsd[r].lbound
 *			+ (oldsd[r].lstride * oldsd[r].lbound)
 *             += oldsd[r].lstride * lower
 *  endif
 * endfor
 * newsd.flags = flagstemp
 * newsd.lbase = lbasetemp
 * newsd.tag = DESCRIPTOR
 */

#define VALUE_ARGT_ARG(a, b) value(ARGT_ARG(a, b))

static int
_sect(int ast, int i8)
{
#define TAGDESC 35
#define SECTZBASE 0x00400000
#define SEQSECTION 0x20000000
#define TEMPLATE 0x00010000
  int argt, newargt, f, funcast;
  int astoldsd, astnewsd, astrank, astflags;
  int sptroldsd, sptrnewsd;
  int rank, flags, dims;
  int newstd, gsizeast, astgsize, lbaseast, astlbase;
  int flagstemp = 0, flagsast = 0, flagsseq = 1, gsizetemp = 0, lbasetemp = 0;
  int needgsize;
  int lowertemp = 0, uppertemp = 0, stridetemp = 0, extenttemp = 0;
  int myoffset = 0, astoffset = 0;
  int newif, cmp, mightbesequential = 1, leading, leadingfull,
                  computesequential;
  int r, d;
  int dtype = DT_INT;
  if (i8)
    dtype = DT_INT8;
  argt = A_ARGSG(ast);
  astnewsd = ARGT_ARG(argt, 0);
  astoldsd = ARGT_ARG(argt, 1);
  if (A_TYPEG(astnewsd) != A_ID || A_TYPEG(astoldsd) != A_ID)
    return 0;
  sptrnewsd = A_SPTRG(astnewsd);
  sptroldsd = A_SPTRG(astoldsd);
  if (CLASSG(sptrnewsd) || CLASSG(sptroldsd))
    return 0;
  astrank = VALUE_ARGT_ARG(argt, 2);
  if (astrank <= 0)
    return 0;
  if (A_TYPEG(astrank) != A_CNST)
    return 0;
  rank = CONVAL2G(A_SPTRG(astrank));
  if (A_ARGCNTG(ast) != 3 * rank + 4)
    return 0;
  astflags = VALUE_ARGT_ARG(argt, 3 * rank + 3);
  if (astflags <= 0)
    return 0;
  if (A_TYPEG(astflags) != A_CNST)
    return 0;
  flags = CONVAL2G(A_SPTRG(astflags));
  if (flags & 0x100) /* BOGUSFLAG */
    return 0;
  /* output dimensions is the pop count of flags */
  dims = (flags & 0x55) + ((flags >> 1) & 0x15);
  dims = (dims & 0x33) + ((dims >> 2) & 0x13);
  dims += (dims >> 4);
  dims = dims & 0xf;
  if (dims > rank || dims <= 0)
    return 0;
  needgsize = 0;
  if (XBIT(47, 0x1000000) || SCG(sptroldsd) == SC_CMBLK || gbl.internal == 1 ||
      (gbl.internal > 1 && INTERNALG(sptroldsd)) || ARGG(sptroldsd))
    needgsize = 1;

  /* set newsd.rank = rank */
  insert_assign(get_desc_rank(sptrnewsd), mk_isz_cval(dims, astb.bnd.dtype),
                beforestd);
  /* copy newsd.kind = oldsd.kind */
  insert_assign(get_kind(sptrnewsd), get_kind(sptroldsd), beforestd);
/* copy newsd.len = oldsd.len */
#ifdef SDSCCONTIGG
  if (SDSCCONTIGG(sptroldsd)) {
    insert_assign(get_byte_len(sptrnewsd),
                  mk_isz_cval(BYTELENG(sptroldsd), astb.bnd.dtype), beforestd);
  } else
#endif
  {
    insert_assign(get_byte_len(sptrnewsd), get_byte_len(sptroldsd), beforestd);
  }
  /* copy flags_temp = oldsd.flags */
  flagsast = get_desc_flags(sptroldsd);
  flagsseq = 1;
  /* copy newsd.gbase = oldsd.gbase */
  insert_assign(get_gbase(sptrnewsd), get_gbase(sptroldsd), beforestd);
  if (XBIT(49, 0x100) && !XBIT(49, 0x80000000) && !XBIT(68, 0x1)) {
    /* pointers are two ints long */
    insert_assign(get_gbase2(sptrnewsd), get_gbase2(sptroldsd), beforestd);
  }
  /* r runs through old rank; d runs through new dims */
  d = 0;
  if (flags & SECTZBASE) {
    /* set lbasetemp = 1 */
    lbaseast = astb.bnd.one;
  } else {
    /* copy lbasetemp = oldsd.lbase */
    lbaseast = get_xbase(sptroldsd);
  }

  /* might this be a sequential section?
   * only if all leading dimensions are sections with stride == 1
   */
  leading = 1;
  leadingfull = 1;
  computesequential = 1;
  for (r = 0; r < rank; ++r) {
    if (!(flags & (1 << r))) {
      /* nonvector dimension */
      leading = 0;
      needgsize = 1;
    } else {
      int aststride, astlower, astupper;
      if (!leading) {
        /* vector dimension after nonvector dimension
         * like a(:,2,:) can't be sequential */
        mightbesequential = 0;
        computesequential = 0;
        needgsize = 1;
        break;
      }
      aststride = VALUE_ARGT_ARG(argt, 5 + 3 * r);
      if (!check_value(aststride, 1)) {
        /* a(1:n:2) can't be sequential */
        mightbesequential = 0;
        computesequential = 0;
        needgsize = 1;
        break;
      }
      if (!leadingfull) {
        /* a(:,1:n,:) might be sequential */
        computesequential = 1;
        needgsize = 1;
      }
      astlower = VALUE_ARGT_ARG(argt, 3 + 3 * r);
      astupper = VALUE_ARGT_ARG(argt, 4 + 3 * r);
      if (!full_dimension(astlower, astupper, aststride, r)) {
        leadingfull = 0;
        needgsize = 1;
      }
    }
  }
  if (computesequential)
    needgsize = 1;
  if (needgsize) {
    /* create temp to hold global size */
    gsizetemp = getnewccsymf(ST_VAR, ".g%d_%d", ast, newsymnum++);
    SCP(gsizetemp, SC_LOCAL);
    DTYPEP(gsizetemp, astb.bnd.dtype);
    if (STD_PAR(beforestd) || STD_TASK(beforestd)) {
      SCP(gsizetemp, SC_PRIVATE);
    }
    gsizeast = astb.bnd.one;
  }
  if (!mightbesequential && flagsseq) {
    f = SEQSECTION;
    f = ~f;
    newargt = mk_argt(2);
    ARGT_ARG(newargt, 0) = flagsast;
    ARGT_ARG(newargt, 1) = mk_isz_cval(f, dtype);
    flagsast = mk_func_node(A_INTR, mk_id(intast_sym[I_AND]), 2, newargt);
    A_OPTYPEP(flagsast, I_AND);
    A_DTYPEP(flagsast, dtype);
    flagsseq = 0;
  }
  for (r = 0; r < rank; ++r) {
    int astlower = 0, astupper = 0, aststride = 0, astextent = 0, sdsc;
    ISZ_T extent, stride;
    astlower = VALUE_ARGT_ARG(argt, 3 + 3 * r);
    astupper = VALUE_ARGT_ARG(argt, 4 + 3 * r);
    aststride = VALUE_ARGT_ARG(argt, 5 + 3 * r);
    if (flags & (1 << r)) {
      if ((sdsc = full_dimension(astlower, astupper, aststride, r))) {
        astlower = symvalue(astlower, 'l', sptrnewsd, &lowertemp, 0, 0);
        if ((flags & NOREINDEX) && XBIT(70, 0x800000)) {
          /* going to need the upper bound */
          astupper =
              symvalue(astupper, 'u', sptrnewsd, &uppertemp, 0, sptrnewsd);
        }
        astextent = get_extent(sdsc, r);
      } else {
        astlower = symvalue(astlower, 'l', sptrnewsd, &lowertemp, 0, 0);
        if (XBIT(70, 0x800000)) {
          astupper =
              symvalue(astupper, 'u', sptrnewsd, &uppertemp, 0, sptrnewsd);
        }
        aststride = symvalue(aststride, 's', sptrnewsd, &stridetemp, 0, 0);
        /* section dimension */
        if (astlower == aststride) {
          astextent = astupper;
        } else {
          /* this is carefully orchestrated.
           * if the RTE_sect call was to create a section of another
           * descriptor, for instance when we pass a section of an
           * array to a subprogram, the call looks like:
           *  call RTE_sect(..,a$sd(lower),extent+(a$sd(lower)-a$sd(stride))..
           * where the upper bound of the section is lower+extent-stride.
           * here, we want to organize the expression to cancel out the
           * (lower-stride) if we can. */
          astextent =
              mk_binop(OP_SUB, astlower, aststride, A_DTYPEG(aststride));
          astextent =
              mk_binop(OP_SUB, astupper, astextent, A_DTYPEG(astextent));
        }
        astextent =
            symvalue(astextent, 'x', sptrnewsd, &extenttemp, 0, sptrnewsd);
        if (A_TYPEG(astextent) == A_CNST && A_TYPEG(aststride) == A_CNST) {
          extent = CONVAL2G(A_SPTRG(astextent));
          stride = CONVAL2G(A_SPTRG(aststride));
          if (stride == -1) {
            extent = -extent;
          } else {
            extent = extent / stride;
          }
          if (extent <= 0) {
            stride = 1;
            aststride = astb.bnd.one;
            if (XBIT(70, 0x800000)) {
              astupper =
                  mk_binop(OP_SUB, astlower, astb.bnd.one, A_DTYPEG(astlower));
            }
            extent = 0;
            astextent = astb.bnd.zero;
          } else {
            astextent = mk_isz_cval(extent, A_DTYPEG(astextent));
          }
        } else {
          if (A_TYPEG(aststride) == A_CNST) {
            stride = CONVAL2G(A_SPTRG(aststride));
            if (stride == -1) {
              astextent = mk_unop(OP_NEG, astextent, A_DTYPEG(astextent));
            } else if (stride != 1) {
              astextent =
                  mk_binop(OP_DIV, astextent, aststride, A_DTYPEG(astextent));
            }
            astextent =
                symvalue(astextent, 'x', sptrnewsd, &extenttemp, 1, sptrnewsd);
          } else {
            /* generate code to do the divide */
            /* if( stride .eq. -1 ) then */

            if (A_TYPEG(astextent) == A_CNST) {
              astextent = symvalue(astextent, 'x', sptrnewsd, &extenttemp, 1,
                                   sptrnewsd);
            }
            newif = mk_stmt(A_IFTHEN, 0);
            cmp = mk_binop(OP_EQ, aststride,
                           mk_isz_cval(-1, A_DTYPEG(aststride)), DT_LOG);
            A_IFEXPRP(newif, cmp);
            newstd = add_stmt_before(newif, beforestd);
            STD_PAR(newstd) = STD_PAR(beforestd);
            STD_TASK(newstd) = STD_TASK(beforestd);
            STD_ACCEL(newstd) = STD_ACCEL(beforestd);
            STD_KERNEL(newstd) = STD_KERNEL(beforestd);
            /* extent = -extent */
            insert_assign(astextent,
                          mk_unop(OP_NEG, astextent, A_DTYPEG(astextent)),
                          beforestd);
            /* else if( stride .ne. 1 )then */
            newif = mk_stmt(A_ELSEIF, 0);
            cmp = mk_binop(OP_NE, aststride, astb.bnd.one, DT_LOG);
            A_IFEXPRP(newif, cmp);
            newstd = add_stmt_before(newif, beforestd);
            STD_PAR(newstd) = STD_PAR(beforestd);
            STD_TASK(newstd) = STD_TASK(beforestd);
            STD_ACCEL(newstd) = STD_ACCEL(beforestd);
            STD_KERNEL(newstd) = STD_KERNEL(beforestd);
            /* extent = extent / stride */
            insert_assign(astextent, mk_binop(OP_DIV, astextent, aststride,
                                              A_DTYPEG(astextent)),
                          beforestd);
            /* endif */
            newif = mk_stmt(A_ENDIF, 0);
            newstd = add_stmt_before(newif, beforestd);
            STD_PAR(newstd) = STD_PAR(beforestd);
            STD_TASK(newstd) = STD_TASK(beforestd);
            STD_ACCEL(newstd) = STD_ACCEL(beforestd);
            STD_KERNEL(newstd) = STD_KERNEL(beforestd);
          }
          /* make sure upper bound is in a variable */
          if (XBIT(70, 0x800000)) {
            astupper =
                symvalue(astupper, 'u', sptrnewsd, &uppertemp, 1, sptrnewsd);
          }
          /* if( extent < 0 )then */
          newif = mk_stmt(A_IFTHEN, 0);
          cmp = mk_binop(OP_LE, astextent, astb.bnd.zero, DT_LOG);
          A_IFEXPRP(newif, cmp);
          newstd = add_stmt_before(newif, beforestd);
          STD_PAR(newstd) = STD_PAR(beforestd);
          STD_TASK(newstd) = STD_TASK(beforestd);
          STD_ACCEL(newstd) = STD_ACCEL(beforestd);
          STD_KERNEL(newstd) = STD_KERNEL(beforestd);
          /* extent = 0 */
          insert_assign(astextent, astb.bnd.zero, beforestd);
          if (XBIT(70, 0x800000)) {
            /* upper = lower-1 */
            insert_assign(astupper, mk_binop(OP_SUB, astlower, astb.bnd.one,
                                             A_DTYPEG(astlower)),
                          beforestd);
          }
          /* endif */
          newif = mk_stmt(A_ENDIF, 0);
          newstd = add_stmt_before(newif, beforestd);
          STD_PAR(newstd) = STD_PAR(beforestd);
          STD_TASK(newstd) = STD_TASK(beforestd);
          STD_ACCEL(newstd) = STD_ACCEL(beforestd);
          STD_KERNEL(newstd) = STD_KERNEL(beforestd);
        }
      }
      /* newsd[d].extent = extent */
      insert_assign(get_extent(sptrnewsd, d), astextent, beforestd);

      if (flags & SECTZBASE) {
        /* newsd[d].lbound = 1 */
        insert_assign(get_global_lower(sptrnewsd, d), astb.bnd.one, beforestd);
        if (XBIT(70, 0x800000)) {
          /* newsd[d].ubound = extent */
          insert_assign(get_global_upper(sptrnewsd, d), astextent, beforestd);
        }
        /* newsd[d].lstride = stride * oldsd[r].lstride */
        insert_assign(get_local_multiplier(sptrnewsd, d),
                      mk_binop(OP_MUL, aststride,
                               get_local_multiplier(sptroldsd, r),
                               A_DTYPEG(aststride)),
                      beforestd);
        /* lbasetemp -= newsd[d].lstride */
        astlbase =
            mk_binop(OP_SUB, lbaseast, get_local_multiplier(sptrnewsd, d),
                     A_DTYPEG(aststride));
        lbaseast = symvalue(astlbase, 'b', sptroldsd, &lbasetemp, 1, 0);
      } else if ((flags & NOREINDEX) && A_TYPEG(aststride) == A_CNST &&
                 CONVAL2G(A_SPTRG(aststride)) == 1) {
        /* newsd[d].lbound = lower */
        insert_assign(get_global_lower(sptrnewsd, d), astlower, beforestd);
        if (XBIT(70, 0x800000)) {
          /* newsd[d].ubound = upper */
          insert_assign(get_global_upper(sptrnewsd, d), astupper, beforestd);
        }
        /* newsd[d].lstride = stride * oldsd[r].lstride */
        insert_assign(get_local_multiplier(sptrnewsd, d),
                      mk_binop(OP_MUL, aststride,
                               get_local_multiplier(sptroldsd, r),
                               A_DTYPEG(aststride)),
                      beforestd);
      } else if ((flags & NOREINDEX)) {
        /* if stride == 1 then */
        newif = mk_stmt(A_IFTHEN, 0);
        cmp = mk_binop(OP_EQ, aststride, astb.bnd.one, DT_LOG);
        A_IFEXPRP(newif, cmp);
        newstd = add_stmt_before(newif, beforestd);
        STD_PAR(newstd) = STD_PAR(beforestd);
        STD_TASK(newstd) = STD_TASK(beforestd);
        STD_ACCEL(newstd) = STD_ACCEL(beforestd);
        STD_KERNEL(newstd) = STD_KERNEL(beforestd);
        /* newsd[d].lbound = lower */
        insert_assign(get_global_lower(sptrnewsd, d), astlower, beforestd);
        if (XBIT(70, 0x800000)) {
          /* newsd[d].ubound = upper */
          insert_assign(get_global_upper(sptrnewsd, d), astupper, beforestd);
        }
        /* set myoffset=0 */
        if (myoffset == 0) {
          myoffset = getnewccsymf(ST_VAR, ".o%d_%d", ast, newsymnum++);
          astlower = symvalue(astlower, 'l', sptrnewsd, &lowertemp, 0, 0);
          SCP(myoffset, SC_LOCAL);
          DTYPEP(myoffset, astb.bnd.dtype);
          if (STD_PAR(beforestd) || STD_TASK(beforestd)) {
            SCP(myoffset, SC_PRIVATE);
          }
          astoffset = mk_id(myoffset);
        }
        insert_assign(astoffset, astb.bnd.zero, beforestd);
        /* else */
        newif = mk_stmt(A_ELSE, 0);
        newstd = add_stmt_before(newif, beforestd);
        STD_PAR(newstd) = STD_PAR(beforestd);
        STD_TASK(newstd) = STD_TASK(beforestd);
        STD_ACCEL(newstd) = STD_ACCEL(beforestd);
        STD_KERNEL(newstd) = STD_KERNEL(beforestd);
        /* newsd[d].lbound = 1 */
        insert_assign(get_global_lower(sptrnewsd, d), astb.bnd.one, beforestd);
        if (XBIT(70, 0x800000)) {
          /* newsd[d].ubound = extent */
          insert_assign(get_global_upper(sptrnewsd, d), astextent, beforestd);
        }
        /* set myoffset = lower-stride */
        if (astlower == aststride) {
          insert_assign(astoffset, astb.bnd.zero, beforestd);
        } else {
          insert_assign(astoffset,
                        mk_binop(OP_SUB, astlower, aststride, astb.bnd.dtype),
                        beforestd);
        }
        /* endif */
        newif = mk_stmt(A_ENDIF, 0);
        newstd = add_stmt_before(newif, beforestd);
        STD_PAR(newstd) = STD_PAR(beforestd);
        STD_TASK(newstd) = STD_TASK(beforestd);
        STD_ACCEL(newstd) = STD_ACCEL(beforestd);
        STD_KERNEL(newstd) = STD_KERNEL(beforestd);
        /* newsd[d].lstride = stride * oldsd[r].lstride */
        insert_assign(get_local_multiplier(sptrnewsd, d),
                      mk_binop(OP_MUL, aststride,
                               get_local_multiplier(sptroldsd, r),
                               A_DTYPEG(aststride)),
                      beforestd);
        /* lbasetemp += myoffset * oldsd[r].lstride */
        astlbase = mk_binop(OP_ADD, lbaseast,
                            mk_binop(OP_MUL, astoffset,
                                     get_local_multiplier(sptroldsd, r),
                                     A_DTYPEG(aststride)),
                            A_DTYPEG(aststride));
        lbaseast = symvalue(astlbase, 'b', sptroldsd, &lbasetemp, 1, 0);
      } else {
        int newstride;
        /* newsd[d].lbound = 1 */
        insert_assign(get_global_lower(sptrnewsd, d), astb.bnd.one, beforestd);
        if (XBIT(70, 0x800000)) {
          /* newsd[d].ubound = extent */
          insert_assign(get_global_upper(sptrnewsd, d), astextent, beforestd);
        }
        /* newsd[d].lstride = stride * oldsd[r].lstride */
        if (r == 0 && SDSCS1G(sptroldsd)) {
          /* linear stride of 1st dimension here is always 1 */
          newstride = aststride;
#ifdef SDSCCONTIGG
        } else if (r == 0 && SDSCCONTIGG(sptroldsd)) {
          /* linear stride of 1st dimension here is always 1 */
          newstride = aststride;
#endif
        } else {
          newstride =
              mk_binop(OP_MUL, aststride, get_local_multiplier(sptroldsd, r),
                       A_DTYPEG(aststride));
        }
        insert_assign(get_local_multiplier(sptrnewsd, d), newstride, beforestd);
        if (astlower != aststride) {
          /* lbasetemp += (lower-stride) * oldsd[r].lstride */
          astlbase = mk_binop(
              OP_ADD, lbaseast,
              mk_binop(OP_MUL, get_local_multiplier(sptroldsd, r),
                       mk_binop(OP_SUB, astlower, aststride, astb.bnd.dtype),
                       astb.bnd.dtype),
              astb.bnd.dtype);
          lbaseast = symvalue(astlbase, 'b', sptroldsd, &lbasetemp, 1, 0);
        }
      }
      if (XBIT(70, 0x800000)) {
        /* newsd[d].sstride = 1 */
        insert_assign(get_section_stride(sptrnewsd, d), astb.bnd.one,
                      beforestd);
        /* newsd[d].soffset = 0 */
        insert_assign(get_section_offset(sptrnewsd, d), astb.bnd.zero,
                      beforestd);
      }
      if (computesequential && flagsseq) {
        if (flagstemp == 0) {
          int sptrfunc;
          newsymnum++;
          flagstemp = getnewccsym('f', newsymnum, ST_VAR);
          SCP(flagstemp, SC_LOCAL);
          DTYPEP(flagstemp, astb.bnd.dtype);
          if (STD_PAR(beforestd) || STD_TASK(beforestd)) {
            SCP(flagstemp, SC_PRIVATE);
          }
          /* flags = oldflags */
          insert_assign(mk_id(flagstemp), flagsast, beforestd);
          flagsast = mk_id(flagstemp);

          /* if( descriptor_length == datatype_length ) then
           * flags = flags | SEQUENTIAL
           * endif */
          newif = mk_stmt(A_IFTHEN, 0);
          newargt = mk_argt(1);
          ARGT_ARG(newargt, 0) = get_kind(sptrnewsd);
          sptrfunc = sym_mkfunc("__get_size_of", DT_INT);
          funcast = mk_func_node(A_FUNC, mk_id(sptrfunc), 1, newargt);
          cmp = mk_binop(OP_EQ, get_byte_len(sptrnewsd), funcast, DT_LOG);
          A_IFEXPRP(newif, cmp);
          newstd = add_stmt_before(newif, beforestd);
          STD_PAR(newstd) = STD_PAR(beforestd);
          STD_TASK(newstd) = STD_TASK(beforestd);
          STD_ACCEL(newstd) = STD_ACCEL(beforestd);
          STD_KERNEL(newstd) = STD_KERNEL(beforestd);

          newargt = mk_argt(2);
          ARGT_ARG(newargt, 0) = flagsast;
          f = SEQSECTION;
          ARGT_ARG(newargt, 1) = mk_isz_cval(f, dtype);
          funcast = mk_func_node(A_INTR, mk_id(intast_sym[I_OR]), 2, newargt);
          A_OPTYPEP(funcast, I_OR);
          A_DTYPEP(funcast, dtype);
          insert_assign(flagsast, funcast, beforestd);

          newif = mk_stmt(A_ENDIF, 0);
          newstd = add_stmt_before(newif, beforestd);
          STD_PAR(newstd) = STD_PAR(beforestd);
          STD_TASK(newstd) = STD_TASK(beforestd);
          STD_ACCEL(newstd) = STD_ACCEL(beforestd);
          STD_KERNEL(newstd) = STD_KERNEL(beforestd);
        }
        /* if newsd[d].lstride != gsizetemp then  */
        newif = mk_stmt(A_IFTHEN, 0);
        cmp = mk_binop(OP_NE, get_local_multiplier(sptrnewsd, d), gsizeast,
                       DT_LOG);
        A_IFEXPRP(newif, cmp);
        newstd = add_stmt_before(newif, beforestd);
        STD_PAR(newstd) = STD_PAR(beforestd);
        STD_TASK(newstd) = STD_TASK(beforestd);
        STD_ACCEL(newstd) = STD_ACCEL(beforestd);
        STD_KERNEL(newstd) = STD_KERNEL(beforestd);
        /* flags &= ~SEQUENTIAL_SECTION */
        newargt = mk_argt(2);
        ARGT_ARG(newargt, 0) = flagsast;
        f = SEQSECTION;
        f = ~f;
        ARGT_ARG(newargt, 1) = mk_isz_cval(f, dtype);
        funcast = mk_func_node(A_INTR, mk_id(intast_sym[I_AND]), 2, newargt);
        A_OPTYPEP(funcast, I_AND);
        A_DTYPEP(funcast, dtype);
        insert_assign(flagsast, funcast, beforestd);
        /* endif */
        newif = mk_stmt(A_ENDIF, 0);
        newstd = add_stmt_before(newif, beforestd);
        STD_PAR(newstd) = STD_PAR(beforestd);
        STD_TASK(newstd) = STD_TASK(beforestd);
        STD_ACCEL(newstd) = STD_ACCEL(beforestd);
        STD_KERNEL(newstd) = STD_KERNEL(beforestd);
      }
      if (needgsize) {
        /* gsizetemp *= extent */
        astgsize = mk_binop(OP_MUL, gsizeast, astextent, astb.bnd.dtype);
        gsizeast = symvalue(astgsize, 'g', sptroldsd, &gsizetemp, 1, 0);
      }
      ++d;
    } else if (!(flags & SECTZBASE)) {
      /* single dimension */
      /* lbasetemp += oldsd[r].lstride * lower */
      astlbase = mk_binop(OP_ADD, lbaseast,
                          mk_binop(OP_MUL, get_local_multiplier(sptroldsd, r),
                                   astlower, astb.bnd.dtype),
                          astb.bnd.dtype);
      lbaseast = symvalue(astlbase, 'b', sptroldsd, &lbasetemp, 1, 0);
    }
  }
  /* newsd.flags = flags */
  insert_assign(get_desc_flags(sptrnewsd), flagsast, beforestd);
  /* newsd.lbase = lbasetemp */
  insert_assign(get_xbase(sptrnewsd), lbaseast, beforestd);
  if (needgsize) {
    /* newsd.gsize = gsizetemp */
    insert_assign(get_desc_gsize(sptrnewsd), gsizeast, beforestd);
    /* newsd.lsize = gsizetemp */
    insert_assign(get_desc_lsize(sptrnewsd), gsizeast, beforestd);
  } else {
    /* copy newsd.gsize = oldsd.gsize */
    insert_assign(get_desc_gsize(sptrnewsd), get_desc_gsize(sptroldsd),
                  beforestd);
    /* copy newsd.lsize = oldsd.lsize */
    insert_assign(get_desc_lsize(sptrnewsd), get_desc_lsize(sptroldsd),
                  beforestd);
  }
  /* newsd.tag = DESCRIPTOR */
  insert_assign(get_desc_tag(sptrnewsd), mk_isz_cval(TAGDESC, dtype),
                beforestd);
  return 1;
} /* _sect */

/*
 * replace RTE_template[123] calls
 * RTE_template[123]( newsd, flags, kind, bytelen [,lower, upper] )
 *
 * newsd.rank = rank	-- must be constant
 * newsd.kind = kind
 * newsd.bytelen = bytelen
 * newsd.gbase = 0
 * d=0
 * lbasetemp = 1
 * gsizetemp = 1
 * for r = 0 to rank-1 do
 *   upper = upper[r]
 *   lower = lower[r]
 *   set extent=upper-lower+1
 *   if upper < lower then extent = 0; upper = lower-1; endif
 *   newsd[d].extent = extent
 *   newsd[d].lbound = lower
 *   newsd[d].ubound = upper
 *   newsd[d].lstride = gsizetemp
 *   lbasetemp -= lower * gsizetemp
 *   newsd[d].sstride = 1
 *   newsd[d].soffset = 0
 *   set gsizetemp *= extent
 * endfor
 * newsd.flags = flags
 * newsd.lbase = lbasetemp
 * newsd.lsize = gsizetemp
 * newsd.gsize = gsizetemp
 * newsd.tag = DESCRIPTOR
 */
static int
_template(int ast, int rank, LOGICAL usevalue, int i8)
{
  int argt;
  int astnewsd, astflags, argbase;
  int sptrnewsd;
  int flags;
  int newstd, astgsize, gsizeast, lbaseast, astlbase;
  int gsizetemp = 0, lbasetemp = 0;
  int lowertemp = 0, uppertemp = 0, extenttemp = 0;
  int newif, cmp;
  int r;
  int dtype = DT_INT;
  if (i8)
    dtype = DT_INT8;
  argt = A_ARGSG(ast);
  astnewsd = ARGT_ARG(argt, 0);
  if (A_TYPEG(astnewsd) != A_ID)
    return 0;
  sptrnewsd = A_SPTRG(astnewsd);
  if (rank > 0) {
    /* known number of dimensions */
    argbase = 0;
    if (A_ARGCNTG(ast) != 2 * rank + 4)
      return 0;
  } else {
    int astrank;
    argbase = 1;
    astrank = VALUE_ARGT_ARG(argt, argbase);
    if (astrank <= 0)
      return 0;
    if (A_TYPEG(astrank) != A_CNST)
      return 0;
    rank = CONVAL2G(A_SPTRG(astrank));
    if (A_ARGCNTG(ast) != 2 * rank + 5)
      return 0;
  }
  astflags = VALUE_ARGT_ARG(argt, argbase + 1);
  if (astflags <= 0)
    return 0;
  if (A_TYPEG(astflags) != A_CNST)
    return 0;
  flags = CONVAL2G(A_SPTRG(astflags));
  if (flags & 0x100) /* BOGUSFLAG */
    return 0;
  flags |= TEMPLATE | SEQSECTION;

  /* set newsd.rank = rank */
  insert_assign(get_desc_rank(sptrnewsd), mk_isz_cval(rank, astb.bnd.dtype),
                beforestd);
  /* copy newsd.kind = kind */
  insert_assign(get_kind(sptrnewsd), VALUE_ARGT_ARG(argt, argbase + 2),
                beforestd);
  /* copy newsd.len = len */
  insert_assign(get_byte_len(sptrnewsd), VALUE_ARGT_ARG(argt, argbase + 3),
                beforestd);
  /* initialize lbasetemp */
  lbaseast = astb.bnd.one;

  gsizeast = astb.bnd.one;
  for (r = 0; r < rank; ++r) {
    int astextent;
    int astlower = VALUE_ARGT_ARG(argt, argbase + 4 + 2 * r);
    int astupper = VALUE_ARGT_ARG(argt, argbase + 5 + 2 * r);
    astlower = symvalue(astlower, 'l', sptrnewsd, &lowertemp, 0, 0);
    if (XBIT(70, 0x800000)) {
      astupper = symvalue(astupper, 'u', sptrnewsd, &uppertemp, 0, sptrnewsd);
    }
    /* section dimension */
    if (astlower == astb.bnd.one) {
      astextent = astupper;
    } else {
      astextent = mk_binop(OP_SUB, astupper, astlower, A_DTYPEG(astupper));
      astextent =
          mk_binop(OP_ADD, astextent, astb.bnd.one, A_DTYPEG(astextent));
    }
    if (A_TYPEG(astextent) == A_CNST) {
      ISZ_T extent = CONVAL2G(A_SPTRG(astextent));
      if (extent <= 0) {
        if (XBIT(70, 0x800000)) {
          astupper =
              mk_binop(OP_SUB, astlower, astb.bnd.one, A_DTYPEG(astlower));
        }
        extent = 0;
        astextent = astb.bnd.zero;
      } else {
        astextent = mk_isz_cval(extent, A_DTYPEG(astextent));
      }
    } else {
      astextent =
          symvalue(astextent, 'x', sptrnewsd, &extenttemp, 1, sptrnewsd);
      /* make sure upper bound is in a variable */
      if (XBIT(70, 0x800000)) {
        astupper = symvalue(astupper, 'u', sptrnewsd, &uppertemp, 1, sptrnewsd);
      }
      /* if(ub < lb) */
      newif = mk_stmt(A_IFTHEN, 0);
      cmp = mk_binop(OP_LT, astupper, astlower, DT_LOG);
      A_IFEXPRP(newif, cmp);
      newstd = add_stmt_before(newif, beforestd);
      STD_PAR(newstd) = STD_PAR(beforestd);
      STD_TASK(newstd) = STD_TASK(beforestd);
      STD_ACCEL(newstd) = STD_ACCEL(beforestd);
      STD_KERNEL(newstd) = STD_KERNEL(beforestd);
      /* extent = 0 */
      insert_assign(astextent, astb.bnd.zero, beforestd);
      if (XBIT(70, 0x800000)) {
        /* upper = lower-1 */
        insert_assign(astupper, mk_binop(OP_SUB, astlower, astb.bnd.one,
                                         A_DTYPEG(astlower)),
                      beforestd);
      }
      /* endif */
      newif = mk_stmt(A_ENDIF, 0);
      newstd = add_stmt_before(newif, beforestd);
      STD_PAR(newstd) = STD_PAR(beforestd);
      STD_TASK(newstd) = STD_TASK(beforestd);
      STD_ACCEL(newstd) = STD_ACCEL(beforestd);
      STD_KERNEL(newstd) = STD_KERNEL(beforestd);
    }
    /* newsd[r].extent = extent */
    insert_assign(get_extent(sptrnewsd, r), astextent, beforestd);

    /* newsd[r].lbound = lower */
    insert_assign(get_global_lower(sptrnewsd, r), astlower, beforestd);
    if (XBIT(70, 0x800000)) {
      /* newsd[r].ubound = upper */
      insert_assign(get_global_upper(sptrnewsd, r), astupper, beforestd);
    }
    /* newsd[r].lstride = stride * oldsd[r].lstride */
    insert_assign(get_local_multiplier(sptrnewsd, r), gsizeast, beforestd);
    if (astlower != astb.bnd.zero) {
      astlbase = mk_binop(OP_SUB, lbaseast,
                          mk_binop(OP_MUL, gsizeast, astlower, astb.bnd.dtype),
                          astb.bnd.dtype);
      lbaseast = symvalue(astlbase, 'b', sptrnewsd, &lbasetemp, 0, 0);
    }
    if (XBIT(70, 0x800000)) {
      /* newsd[r].sstride = 1 */
      insert_assign(get_section_stride(sptrnewsd, r), astb.bnd.one, beforestd);
      /* newsd[r].soffset = 0 */
      insert_assign(get_section_offset(sptrnewsd, r), astb.bnd.zero, beforestd);
    }
    /* gsizetemp *= extent */
    astgsize = mk_binop(OP_MUL, gsizeast, astextent, astb.bnd.dtype);
    gsizeast = symvalue(astgsize, 'g', sptrnewsd, &gsizetemp, 1, 0);
  }
  /* newsd.flags = flags */
  insert_assign(get_desc_flags(sptrnewsd), mk_isz_cval(flags, astb.bnd.dtype),
                beforestd);
  /* newsd.lbase = lbasetemp */
  insert_assign(get_xbase(sptrnewsd), lbaseast, beforestd);
  /* newsd.gbase = 0 */
  insert_assign(get_gbase(sptrnewsd), astb.bnd.zero, beforestd);
  if (XBIT(49, 0x100) && !XBIT(49, 0x80000000) && !XBIT(68, 0x1)) {
    /* pointers are two ints long */
    insert_assign(get_gbase2(sptrnewsd), astb.bnd.zero, beforestd);
  }
  /* newsd.gsize = gsizetemp */
  insert_assign(get_desc_gsize(sptrnewsd), gsizeast, beforestd);
  /* newsd.lsize = gsizetemp */
  insert_assign(get_desc_lsize(sptrnewsd), gsizeast, beforestd);
  /* newsd.tag = DESCRIPTOR */
  insert_assign(get_desc_tag(sptrnewsd), mk_isz_cval(TAGDESC, dtype),
                beforestd);
  return 1;
} /* _template */

/*
 * section descriptor member
 */
static int
_sd_member(int subscript, int sdx, int sdtype)
{
  int subscr[2];
  subscr[0] = mk_isz_cval(subscript, sdtype);
  return mk_subscr(sdx, subscr, 1, sdtype);
} /* _sd_member */

/** \brief This routine assigns a size to a descriptor's length field.
 *
 * \param ast is the expression that has the descriptor 
 *        (e.g., an A_ID, A_MEM, etc.).
 * \param ddesc is the symbol table pointer of the descriptor.
 * \param sz is the AST representing the size.
 *
 * \return the resulting assignment AST.
 */
int
gen_set_len_ast(int ast, SPTR ddesc, int sz)
{

  DTYPE dtype;
  int ast2;

  dtype = astb.bnd.dtype;

  ast2 = mk_id(ddesc);
  ast2 = _sd_member(DESC_HDR_BYTE_LEN, ast2, dtype);
  A_DTYPEP(ast2, dtype);

  ast2 = check_member(ast, ast2);

  return mk_assn_stmt(ast2, sz, dtype);

}



LOGICAL
inline_RTE_set_type(int ddesc, int sdesc, int stmt, int after,
                      DTYPE src_dtype, int astmem)
{
  /* This function inlines RTE_set_type calls. Returns TRUE if successful,
   * else FALSE. The src_dtype is the declared type of the source object.
   */

  int stdx, asn;
  int subscript;
  int ast1, ast2;
  DTYPE sdtype, dtype;

  if (is_array_dtype(src_dtype))
    src_dtype = array_element_dtype(src_dtype);

  if (SCG(sdesc) == SC_DUMMY || SCG(ddesc) == SC_DUMMY) {
    /* TBD */
    return FALSE;
  }

  sdtype = astb.bnd.dtype;

  if (XBIT(49, 0x100) && !XBIT(49, 0x80000000) && !XBIT(68, 0x1)) {
    subscript = DESC_HDR_GBASE + 2;
    dtype = DT_INT8;
  } else {
    subscript = DESC_HDR_GBASE + 1;
    dtype = astb.bnd.dtype;
  }

  ast1 = mk_id(ddesc);
  ast1 = _sd_member(subscript, ast1, sdtype);
  A_DTYPEP(ast1, dtype);

  if (CLASSG(sdesc)) {
    ast2 = mk_id(sdesc);
    ast2 = mk_unop(OP_LOC, ast2, dtype);
  } else {
    ast2 = mk_id(sdesc);
    ast2 = _sd_member(subscript, ast2, sdtype);
    A_DTYPEP(ast2, dtype);
  }
  if (ast1 && astmem && STYPEG(ddesc) == ST_MEMBER) {
    ast1 = check_member(astmem, ast1);
  }

  if (ast1 && ast2) {
    asn = mk_assn_stmt(ast1, ast2, dtype);
    if (SCG(ddesc) != SC_EXTERN)
      ADDRTKNP(ddesc, 1);
    if (SCG(sdesc) != SC_EXTERN)
      ADDRTKNP(sdesc, 1);
  } else {
    return FALSE;
  }
  if (after)
    stdx = add_stmt_after(asn, stmt);
  else
    stdx = add_stmt_before(asn, stmt);

  return TRUE;
}

/*
 * copy one element from target section descriptor to pointer descriptor
 */
static void
_ptrassign_copy(int subscript, int ptrsdx, int tgtsdx, int sdtype)
{
  int stdx, asn;
  asn = MKASSN(_sd_member(subscript, ptrsdx, sdtype),
               _sd_member(subscript, tgtsdx, sdtype));
  stdx = add_stmt_before(asn, beforestd);
} /* _ptrassign_copy */

/*
 * set one element in pointer section descriptor
 */
static void
_ptrassign_set(int subscript, int ptrsdx, int value, int sdtype)
{
  int stdx, asn;
  asn =
      MKASSN(_sd_member(subscript, ptrsdx, sdtype), mk_isz_cval(value, sdtype));
  stdx = add_stmt_before(asn, beforestd);
} /* _ptrassign_set */

/*
 * set one element in pointer section descriptor
 */
static void
_ptrassign_set_ast(int subscript, int ptrsdx, int valastx, int sdtype)
{
  int stdx, asn;
  asn = MKASSN(_sd_member(subscript, ptrsdx, sdtype), valastx);
  stdx = add_stmt_before(asn, beforestd);
} /* _ptrassign_set_ast */

/*
 * if this is a ptr_assign call that is for the whole array (sectflag == 0)
 * then replace by inline code.
 *  if the pointer target is itself a pointer or allocatable,
 *   copy the pointer value
 *  else
 *   replace by %loc(pointer target)
 *  generate a loop to copy the descriptor
 */
static int
_ptrassign(int astx)
{
  int argt, sectflagx;
  int ptrx, ptrsdx, ptrsptr = 0, ptrsdsptr, ptrsdtype, tgtx, tgtsdx, tgtsptr;
  int asn, stdx, i, rank;
  argt = A_ARGSG(astx);
  sectflagx = ARGT_ARG(argt, 4);
  ptrx = ARGT_ARG(argt, 0);
  ptrsdx = ARGT_ARG(argt, 1);
  tgtx = ARGT_ARG(argt, 2);
  tgtsdx = ARGT_ARG(argt, 3);
  /* if the target is not an ID or MEMBER, give up */
  if (A_TYPEG(tgtx) != A_ID && A_TYPEG(tgtx) != A_MEM)
    return 0;
  /* if the target section descriptor is not an ID or MEMBER or CONST, give up
   */
  if (A_TYPEG(tgtsdx) != A_ID && A_TYPEG(tgtsdx) != A_MEM &&
      A_TYPEG(tgtsdx) != A_CNST)
    return 0;
  /* if the destination pointer is not an ID or MEMBER, give up */
  if (A_TYPEG(ptrx) != A_ID && A_TYPEG(ptrx) != A_MEM)
    return 0;
  /* if the destination section descriptor is not an ID or MEMBER, give up */
  if (A_TYPEG(ptrsdx) != A_ID && A_TYPEG(ptrsdx) != A_MEM)
    return 0;
  if (sectflagx != astb.i0 && sectflagx != astb.k0 &&
      (tgtsdx != ptrsdx /*|| XBIT(1,0x800)*/))
    /* leave the call in place */
    return 0;
  /* if the target is itself a pointer, we can simply copy the pointer value */
  if (A_TYPEG(tgtx) == A_ID) {
    tgtsptr = A_SPTRG(tgtx);
  } else if (A_TYPEG(tgtx) == A_MEM) {
    tgtsptr = A_SPTRG(A_MEMG(tgtx));
  }
  if (A_TYPEG(ptrx) == A_ID) {
    ptrsptr = A_SPTRG(ptrx);
  } else if (A_TYPEG(ptrx) == A_MEM) {
    ptrsptr = A_SPTRG(A_MEMG(ptrx));
  }
#ifdef TEXTUREG
  if (ptrsptr && TEXTUREG(ptrsptr))
    return 0;
#endif
#ifdef DEVICEG
  if (ptrsptr && DEVICEG(ptrsptr))
    return 0;
#endif
  if (A_TYPEG(ptrsdx) == A_ID) {
    ptrsdsptr = A_SPTRG(ptrsdx);
    if (STYPEG(ptrsptr) == ST_MEMBER && STYPEG(ptrsdsptr) != ST_MEMBER) {
      ptrsdsptr = 0;
    }
  } else if (A_TYPEG(ptrsdx) == A_MEM) {
    ptrsdsptr = A_SPTRG(A_MEMG(ptrsdx));
  }
  if (MIDNUMG(ptrsptr) == 0)
    return 0;
  if (POINTERG(tgtsptr)) {
    if (MIDNUMG(tgtsptr) == 0)
      return 0;
    /* copy the pointer */
    asn = MKASSN(check_member(ptrx, mk_id(MIDNUMG(ptrsptr))),
                 check_member(tgtx, mk_id(MIDNUMG(tgtsptr))));
  } else {
    /* must take %loc(arg) */
    asn = MKASSN(check_member(ptrx, mk_id(MIDNUMG(ptrsptr))),
                 mk_unop(OP_LOC, tgtx, DT_PTR));
  }
  stdx = add_stmt_before(asn, beforestd);
  if (ptrsdsptr) {
    ptrsdtype = DDTG(DTYPEG(ptrsdsptr));
    if (A_TYPEG(tgtsdx) == A_ID) {
      tgtsdx = mk_id(A_SPTRG(tgtsdx));
    } else if (A_TYPEG(tgtsdx) == A_MEM) {
      tgtsdx = mk_member(A_PARENTG(tgtsdx), mk_id(A_SPTRG(A_MEMG(tgtsdx))),
                         A_DTYPEG(tgtsdx));
    }
    if (A_TYPEG(tgtsdx) == A_CNST) {
      /* PTRSD(tag) = value */
      _ptrassign_set_ast(DESC_HDR_TAG, ptrsdx, tgtsdx, ptrsdtype);
    } else if (ptrsdx != tgtsdx) {
      /* PTRSD(tag) = Descriptor */
      /* PTRSD(rank) = TGTSD(rank) */
      /* PTRSD(kind) = TGTSD(kind) */
      /* PTRSD(len) = TGTSD(len) */
      /* PTRSD(flags) = TGTSD(flags) */
      /* PTRSD(lsize) = TGTSD(lsize) */
      /* PTRSD(gsize) = TGTSD(gsize) */
      /* PTRSD(lbase) = TGTSD(lbase) */
      /* PTRSD(gbase) = TGTSD(gbase) */
      /* for i = 0; i < rank ++i */
      /*  PTRSD(lower(i)) = 1 */
      /*  PTRSD(extent(i)) = TGTSD(extent(i)) */
      /*  PTRSD(upper(i)) = TGTSD(extent(i)) */
      /*  PTRSD(lstride(i)) = TGTSD(lstride(i)) */
      /*  PTRSD(soffset(i)) = 0 */
      /*  PTRSD(sstride(i)) = 0 */
      _ptrassign_set(DESC_HDR_TAG, ptrsdx, TAGDESC, ptrsdtype);
      _ptrassign_copy(DESC_HDR_RANK, ptrsdx, tgtsdx, ptrsdtype);
      _ptrassign_copy(DESC_HDR_KIND, ptrsdx, tgtsdx, ptrsdtype);
      _ptrassign_copy(DESC_HDR_BYTE_LEN, ptrsdx, tgtsdx, ptrsdtype);
      _ptrassign_copy(DESC_HDR_FLAGS, ptrsdx, tgtsdx, ptrsdtype);
      _ptrassign_copy(DESC_HDR_LSIZE, ptrsdx, tgtsdx, ptrsdtype);
      _ptrassign_copy(DESC_HDR_GSIZE, ptrsdx, tgtsdx, ptrsdtype);
      if (ASSUMSHPG(tgtsptr) && !XBIT(58, 0x400000)) {
        _ptrassign_set(DESC_HDR_LBASE, ptrsdx, 1, ptrsdtype);
      } else {
        _ptrassign_copy(DESC_HDR_LBASE, ptrsdx, tgtsdx, ptrsdtype);
      }
      _ptrassign_copy(DESC_HDR_GBASE, ptrsdx, tgtsdx, ptrsdtype);
      if (XBIT(49, 0x100) && !XBIT(49, 0x80000000)
          && !XBIT(68, 0x1)
              ) {
        /* pointers are two ints long */
        _ptrassign_copy(DESC_HDR_GBASE + 1, ptrsdx, tgtsdx, ptrsdtype);
      }
      rank = ADD_NUMDIM(DTYPEG(ptrsptr));
      for (i = 0; i < rank; ++i) {
        int lb;
        if (!ASSUMSHPG(tgtsptr) || XBIT(58, 0x400000)) {
          _ptrassign_copy(get_global_lower_index(i), ptrsdx, tgtsdx, ptrsdtype);
        } else {
          /* for assumed-shape arguments, use the declared bounds */
          lb = ADD_LWAST(DTYPEG(tgtsptr), i);
          _ptrassign_set_ast(get_global_lower_index(i), ptrsdx, lb, ptrsdtype);
        }
        _ptrassign_copy(get_global_extent_index(i), ptrsdx, tgtsdx, ptrsdtype);
        _ptrassign_set(get_section_stride_index(i), ptrsdx, 0, ptrsdtype);
        _ptrassign_set(get_section_offset_index(i), ptrsdx, 0, ptrsdtype);
        _ptrassign_copy(get_multiplier_index(i), ptrsdx, tgtsdx, ptrsdtype);
        if (ASSUMSHPG(tgtsptr) && !XBIT(58, 0x400000)) {
          /* adjust the LBASE */
          int a;
          a = mk_binop(OP_MUL,
                       _sd_member(get_multiplier_index(i), ptrsdx, ptrsdtype),
                       lb, ptrsdtype);
          a = mk_binop(OP_SUB, _sd_member(DESC_HDR_LBASE, ptrsdx, ptrsdtype), a,
                       ptrsdtype);
          _ptrassign_set_ast(DESC_HDR_LBASE, ptrsdx, a, ptrsdtype);
        }
        /* we could copy the upper bound, but it's never used by the runtime
         * anyway */
        /* _ptrassign_copy( get_global_upper_index(i), ptrsdx, tgtsdx, ptrsdtype
         * );*/
      }
    }
  }
  return 1; /* ### */
} /* _ptrassign */

/*
 * inline RTE_sect calls, where possible
 * also inline simple ptr2_assign calls, where the pointee is the whole array
 *  do this after sectfloat
 */
void
sectinline(void)
{
  int std, stdnext;
  int ast;

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    ast = STD_AST(std);
    beforestd = std;
    if (A_TYPEG(ast) == A_CALL) {
      int lop;
      lop = A_LOPG(ast);
      if (lop && A_TYPEG(lop) == A_ID) {
        int fsptr;
        fsptr = A_SPTRG(lop);
        if (HCCSYMG(fsptr) && STYPEG(fsptr) == ST_PROC) {
          int i;
          i = getF90TmplSectRtn(SYMNAME(fsptr));
          switch (i & FTYPE_MASK) {
          case FTYPE_SECT:
            /* found one of the names */
            if (_sect(ast, i & FTYPE_I8)) {
              ast_to_comment(ast);
            }
            break;
          case FTYPE_TEMPLATE:
            if (_template(ast, -1, FALSE, i & FTYPE_I8))
              ast_to_comment(ast);
            break;
          case FTYPE_TEMPLATE1:
            if (_template(ast, 1, FALSE, i & FTYPE_I8))
              ast_to_comment(ast);
            break;
          case FTYPE_TEMPLATE1V:
            if (_template(ast, 1, TRUE, i & FTYPE_I8))
              ast_to_comment(ast);
            break;
          case FTYPE_TEMPLATE2:
            if (_template(ast, 2, FALSE, i & FTYPE_I8))
              ast_to_comment(ast);
            break;
          case FTYPE_TEMPLATE2V:
            if (_template(ast, 2, TRUE, i & FTYPE_I8))
              ast_to_comment(ast);
            break;
          case FTYPE_TEMPLATE3:
            if (_template(ast, 3, FALSE, i & FTYPE_I8))
              ast_to_comment(ast);
            break;
          case FTYPE_TEMPLATE3V:
            if (_template(ast, 3, TRUE, i & FTYPE_I8))
              ast_to_comment(ast);
            break;
          }
        }
      }
    } else if (A_TYPEG(ast) == A_ICALL) {
      switch (A_OPTYPEG(ast)) {
      case I_PTR2_ASSIGN:
        /* see if this can be inlined */
        if (_ptrassign(ast)) {
          ast_to_comment(ast);
        }
        break;
      }
    }
  }
} /* sectinline */

static void
convert_statements(void)
{
  int std, stdnext;
  int ast;
  int parallel_depth;
  int task_depth;

  init_tbl();
  unvisit_every_sptr();

  parallel_depth = 0;
  task_depth = 0;
  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_ALLOC:
      if (A_TKNG(ast) == TK_ALLOCATE) {
        stdnext = conv_allocate(std);
      } else {
        assert(A_TKNG(ast) == TK_DEALLOCATE, "conv_statements: bad dealloc",
               std, 4);
        stdnext = conv_deallocate(std);
      }
      break;
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
    default:
      break;
    }
  }
  free_tbl();
}

static void
_mark_descr(int ast, int *dummy)
{
  if (A_TYPEG(ast) == A_MEM)
    ast = A_MEMG(ast);
  if (A_TYPEG(ast) == A_ID) {
    int sptr, stype;
    sptr = A_SPTRG(ast);
    stype = STYPEG(sptr);
    if ((stype == ST_ARRAY || stype == ST_MEMBER) && DESCARRAYG(sptr)) {
      VISITP(sptr, 1);
    }
  }
} /* _mark_descr */

static void
convert_template_instance(void)
{
  int sptr, std, stdnext;
  /* we are looking for cases where we have a RTE_template call
   * followed by a pghpf_instance call, and the pghpf_instance call
   * is the ONLY use of the RTE_template output template.
   * for instance
   *  call RTE_template(aa$sd,1,2,0,0,1,20)
   *  call pghpf_instance(aa$sd1,aa$sd,27,4,0)
   * replace aa$sd by aa$sd1 here
   */

  /* reset VISIT flags */
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    VISITP(sptr, 0);
  }

  /* Look for all uses of section descriptors anywhere
   * outside of calls to RTE_template and pghpf_instance */
  for (std = STD_NEXT(0); std; std = stdnext) {
    int ast, sptr, argcnt, dummy;
    stdnext = STD_NEXT(std);
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_CALL:
      sptr = memsym_of_ast(A_LOPG(ast));
      if (STYPEG(sptr) != ST_PROC)
        break;
      argcnt = A_ARGCNTG(ast);
      if (STYPEG(sptr) == ST_PROC) {
        /* don't look at RTE_template calls */
        if (strcmp(SYMNAME(sptr), mkRteRtnNm(RTE_template)) == 0)
          break;
        /* don't look at pghpf_instance calls
         * if the previous statement is a RTE_template call */
        if (strcmp(SYMNAME(sptr), mkRteRtnNm(RTE_instance)) == 0) {
          int stdprev, astprev;
          stdprev = STD_PREV(std);
          astprev = STD_AST(stdprev);
          if (A_TYPEG(astprev) == A_CALL) {
            int sptrprev;
            sptrprev = A_SPTRG(A_LOPG(astprev));
            if (STYPEG(sptrprev) == ST_PROC &&
                strcmp(SYMNAME(sptrprev), mkRteRtnNm(RTE_template)) == 0)
              break;
          }
        }
      }
      FLANG_FALLTHROUGH;
    default:
      ast_visit(1, 1);
      ast_traverse(ast, NULL, _mark_descr, &dummy);
      ast_unvisit();
      break;
    }
  }
  /* Look for pghpf_instance calls where the previous statement
   * is a RTE_template call, or the previous two statements are
   * a RTE_set_intrin_type call and a RTE_template call, and where
   * the input descriptor to the instance is the output descriptor of the
   * template call, and the descriptor has no other uses */
  for (std = STD_NEXT(0); std; std = stdnext) {
    int ast, sptr, argcnt;
    stdnext = STD_NEXT(std);
    ast = STD_AST(std);
    if (A_TYPEG(ast) == A_CALL) {
      sptr = memsym_of_ast(A_LOPG(ast));
      argcnt = A_ARGCNTG(ast);
      if (STYPEG(sptr) == ST_PROC && argcnt == 5 &&
          strcmp(SYMNAME(sptr), mkRteRtnNm(RTE_instance)) == 0) {
        int stdprev, astprev;
        stdprev = STD_PREV(std);
        astprev = STD_AST(stdprev);
        if (A_TYPEG(astprev) == A_CALL) {
          int sptrprev, set_intrin_type_std = 0;
          sptrprev = A_SPTRG(A_LOPG(astprev));
          if (STYPEG(sptrprev) == ST_PROC &&
              strcmp(SYMNAME(sptrprev), mkRteRtnNm(RTE_set_intrin_type)) == 0) {
            int stdtmp, asttmp;
            stdtmp = STD_PREV(stdprev);
            asttmp = STD_AST(stdtmp);
            if (A_TYPEG(asttmp) == A_CALL) {
              int sptrtmp = A_SPTRG(A_LOPG(asttmp));
              if (STYPEG(sptrtmp) == ST_PROC &&
                  strcmp(SYMNAME(sptrtmp), mkRteRtnNm(RTE_template)) == 0) {
                set_intrin_type_std = stdprev;
                stdprev = stdtmp;
                astprev = asttmp;
                sptrprev = sptrtmp;
              }
            }
          }
          if (STYPEG(sptrprev) == ST_PROC &&
              strcmp(SYMNAME(sptrprev), mkRteRtnNm(RTE_template)) == 0) {
            /* get argument lists */
            int argsi, insd, outsd, argst, tempsd, sptrsd, collapse;
            argsi = A_ARGSG(ast);
            outsd = ARGT_ARG(argsi, 0);
            insd = ARGT_ARG(argsi, 1);
            argst = A_ARGSG(astprev);
            tempsd = ARGT_ARG(argst, 0);
            sptrsd = sym_of_ast(tempsd);
            collapse = ARGT_ARG(argsi, 4);
            if (sptrsd && !VISITG(sptrsd) && tempsd == insd &&
                tempsd != outsd) {
              if (collapse == astb.i0 || collapse == astb.k0) {
                /* replace
                 *  call RTE_template(aa$sd,1,2,0,0,1,20)
                 *  call pghpf_instance(aa$sd1,aa$sd,27,4,0)
                 * by
                 *  call RTE_template(aa$sd1,1,2,27,4,1,20)
                 */
                ARGT_ARG(argst, 0) = outsd;
                ARGT_ARG(argst, 3) = ARGT_ARG(argsi, 2);
                ARGT_ARG(argst, 4) = ARGT_ARG(argsi, 3);
                delete_stmt(std);
                if (set_intrin_type_std != 0)
                  delete_stmt(set_intrin_type_std);
              } else {
                /* replace
                 *  call RTE_template(aa$sd,1,2,0,0,1,20)
                 *  call pghpf_instance(aa$sd1,aa$sd,27,4,0)
                 * by
                 *  call RTE_template(aa$sd1,1,2,0,0,1,20)
                 *  call pghpf_instance(aa$sd1,aa$sd1,27,4,0)
                 */
                ARGT_ARG(argsi, 1) = outsd;
                ARGT_ARG(argst, 0) = outsd;
              }
              STYPEP(sptrsd, ST_UNKNOWN);
            }
            if (sptrsd && tempsd == insd && tempsd == outsd &&
                (collapse == astb.i0 || collapse == astb.k0)) {
              /* replace
               *  call RTE_template(aa$sd,1,2,0,0,1,20)
               *  call pghpf_instance(aa$sd,aa$sd,27,4,0)
               * by
               *  call RTE_template(aa$sd,1,2,27,4,1,20)
               */
              ARGT_ARG(argst, 3) = ARGT_ARG(argsi, 2);
              ARGT_ARG(argst, 4) = ARGT_ARG(argsi, 3);
              delete_stmt(std);
              if (set_intrin_type_std != 0)
                delete_stmt(set_intrin_type_std);
            }
          }
        }
      }
    }
  }

  /* go back and reset VISIT flags again */
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
    VISITP(sptr, 0);
  }

  /* look for pghpf_instance calls where the input/output descriptors
   * are identical; replace by assignments
   * look for RTE_template calls;
   * if the rank is constant, replace by
   * RTE_template1/RTE_template2/RTE_template3 or
   * RTE_template1v/RTE_template2v/RTE_template3v calls, as appropriate */
  for (std = STD_NEXT(0); std; std = stdnext) {
    int ast, sptr, argcnt;
    stdnext = STD_NEXT(std);
    ast = STD_AST(std);
    if (A_TYPEG(ast) == A_CALL) {
      sptr = memsym_of_ast(A_LOPG(ast));
      argcnt = A_ARGCNTG(ast);
      if (STYPEG(sptr) == ST_PROC && argcnt == 5 &&
          strcmp(SYMNAME(sptr), mkRteRtnNm(RTE_instance)) == 0) {
        /* replace call pghpf_instance(a$sd,a$sd,kind,len,0)
         * by direct assignment
         *  a$sd(kindoffset) = kind
         *  a$sd(lenoffset) = len
         */
        int argsi, outsd, insd, collapse;
        argsi = A_ARGSG(ast);
        outsd = ARGT_ARG(argsi, 0);
        insd = ARGT_ARG(argsi, 1);
        collapse = ARGT_ARG(argsi, 4);
        if (outsd == insd && A_TYPEG(insd) == A_ID &&
            (collapse == astb.i0 || collapse == astb.k0)) {
          int kind, len, lhs, newasn, newstd;
          insd = A_SPTRG(insd);
          kind = ARGT_ARG(argsi, 2);
          len = ARGT_ARG(argsi, 3);
          lhs = get_kind(insd);
          newasn = mk_stmt(A_ASN, 0);
          A_DESTP(newasn, lhs);
          A_SRCP(newasn, kind);
          newstd = add_stmt_before(newasn, std);
          lhs = get_byte_len(insd);
          newasn = mk_stmt(A_ASN, 0);
          A_DESTP(newasn, lhs);
          A_SRCP(newasn, len);
          STD_AST(std) = newasn;
        }
      } else if (STYPEG(sptr) == ST_PROC &&
                 strcmp(SYMNAME(sptr), mkRteRtnNm(RTE_template)) == 0) {
        /*  call RTE_template(aa$sd,rank,flags,kind,len,lb1,lb2)
         *   turn into
         *  call RTE_template1(aa$sd,flags,kind,len,lb1,lb2) */
        int args, rank, ii;
        args = A_ARGSG(ast);
        rank = ARGT_ARG(args, 1);
        if (A_ALIASG(rank)) {
          rank = A_ALIASG(rank);
          rank = get_int_cval(A_SPTRG(rank));
          if (rank >= 1 && rank <= 3) {
            int fsptr, a;
            FtnRtlEnum rtlRtn;
            /* one fewer argument */
            --argcnt;
            for (a = 1; a < argcnt; ++a) {
              ARGT_ARG(args, a) = ARGT_ARG(args, a + 1);
            }
            ARGT_CNT(args) = argcnt;
            A_ARGCNTP(ast, argcnt);
            if (size_of(DT_PTR) != size_of(DT_INT)) {
              /* on hammer, seems faster to pass by ref */
              switch (rank) {
              case 1:
                rtlRtn = RTE_template1;
                break;
              case 2:
                rtlRtn = RTE_template2;
                break;
              case 3:
                rtlRtn = RTE_template3;
                break;
              }
            } else {
              switch (rank) {
              case 1:
                rtlRtn = RTE_template1v;
                break;
              case 2:
                rtlRtn = RTE_template2v;
                break;
              case 3:
                rtlRtn = RTE_template3v;
                break;
              }
              for (a = 1; a < argcnt; ++a) {
                ARGT_ARG(args, a) = mk_unop(OP_VAL, ARGT_ARG(args, a), DT_INT);
              }
            }
            fsptr = sym_mkfunc(mkRteRtnNm(rtlRtn), DT_NONE);
            NODESCP(fsptr, 1);
            ii = mk_id(fsptr);
            A_LOPP(ast, ii);
            /*
             * tpr 3569:  a call to RTE_template() without the
             * upperbound of the last dimension is generated
             * for an assumed-size array.  But, the rank-specific
             * template functions accesses the upper bound
             * which could cause a segfault.   Just add the
             * the dimension's lowerbound as the upper bound.
             */
            if (argcnt < rank * 2 + 4) {
              ARGT_ARG(args, argcnt) = ARGT_ARG(args, argcnt - 1);
              argcnt++;
              ARGT_CNT(args) = argcnt;
              A_ARGCNTP(ast, argcnt);
            }
          }
        }
      }
    }
  }
} /* convert_template_instance */

static int
conv_deallocate(int std)
{
  int dealloc_ast, idast;
  int ast;
  int sptr = 0;
  int argt;
  int secd;
  int arrdsc;
  LITEMF *list;
  int i;
  int nargs;

  dealloc_ast = A_SRCG(STD_AST(std));
again:
  switch (A_TYPEG(dealloc_ast)) {
  case A_ID:
    sptr = A_SPTRG(dealloc_ast);
    idast = dealloc_ast;
    break;
  case A_MEM:
    sptr = A_SPTRG(A_MEMG(dealloc_ast));
    idast = dealloc_ast;
    break;
  case A_SUBSCR:
    dealloc_ast = A_LOPG(dealloc_ast);
    goto again;
  default:
    interr("conv_deallocate: unexpected ast", dealloc_ast, 4);
  }

  /* free the section and the align */
  arrdsc = DESCRG(sptr);
  if (arrdsc == 0)
    goto exit_;
  secd = SECDG(arrdsc);
  if (secd == 0)
    goto exit_;

  list = 0;
  for (i = 0; i < tbl.avl; i++) {
    if (tbl.base[i].f1 == sptr)
      list = tbl.base[i].f3;
    else if (STYPEG(sptr) == ST_MEMBER && STYPEG(tbl.base[i].f1) == ST_MEMBER &&
             ENCLDTYPEG(sptr) == ENCLDTYPEG(tbl.base[i].f1) &&
             strcmp(SYMNAME(sptr), SYMNAME(tbl.base[i].f1)) == 0) {
      /* This occurs with parameterized derived types */
      list = tbl.base[i].f3;
    }
  }

  /*
   * f22379: there are cases where 'all' descriptors are created where there
   * may not be a matching allocate, such as a a pointer member of a
   * polymorphic typ> For now, I'm just removing the assert for now -- in the
   * future, we may want to qualify the assert.
  assert(list, "conv_deallocate: did not find corresponding allocate", sptr, 3);
   */
  if (!list || list->nitem == 0)
    goto exit_;
  nargs = list->nitem + 1;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = mk_cval(list->nitem, DT_INT);
  for (i = 0; i < list->nitem; i++) {
    int mast;
    mast = check_member(idast, mk_id(glist(list, i)));
    ARGT_ARG(argt, list->nitem - i) = mast;
  }
  ast = mk_func_node(A_CALL, mk_id(sym_mkfunc(mkRteRtnNm(RTE_freen), DT_NONE)),
                     nargs, argt);
  add_stmt_before(ast, std);
exit_:
  std = STD_NEXT(std);
  if (STD_IGNORE(STD_PREV(std)))
    delete_stmt(STD_PREV(std));
  return std;
}

/* Algorithm:
 * This routine converts allocatable arrays.
 * allocate(a(a$sd(33):a$sd(34)))
 * This allocate stmt is user defined statement
 * not compiler define allocates.
 * It calls emit_alnd_secd to set align and section descriptor
 * for allocatable arrays.
 * emit_alnd_secd has to generate algn and sec just before allocate stmt
 * unlike non-array.
 */
extern LOGICAL want_descriptor_anyway(int sptr);

static int
conv_allocate(int std)
{
  int alloc_ast, idast;
  int sptr = 0;
  int subsc, memast;
  int i;
  int asd;
  int dtype;
  int align;
  LITEMF *list;
  int nd;
  int a_dtype;

  alloc_ast = STD_AST(std);
  memast = subsc = A_SRCG(alloc_ast);
  /* only set a_dtype for typed allocation, not sourced allocation, etc. */
  a_dtype = (!A_STARTG(alloc_ast)) ? A_DTYPEG(alloc_ast) : 0;
  switch (A_TYPEG(subsc)) {
  case A_SUBSCR:
    sptr = sptr_of_subscript(subsc);
    memast = idast = A_LOPG(subsc);
    asd = A_ASDG(subsc);
    if (!HCCSYMG(sptr)) {
      /* if the sptr is not a compiler generated temp, skip the "UGLY HACK"
       * This became necessary with the addition of ALLOCATE SOURCE/MOLD.
       * TODO: Think this needs to be revisited.
       */
      break;
    }
    /* UGLY HACK:
     * if this is a temporary that was created on behalf of
     * a derived type member, use the member as the 'idast' */
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY) {
      int lower = ADD_LWBD(dtype, 0);
      if (lower && A_TYPEG(lower) == A_SUBSCR)
        lower = A_LOPG(lower);
      if (lower && A_TYPEG(lower) == A_MEM) {
        idast = lower;
      } else if (lower && A_TYPEG(lower) == A_ID &&
                 STYPEG(A_SPTRG(lower)) == ST_MEMBER) {
        /* candidate case */
        int subs;
        subs = ASD_SUBS(asd, 0);
        if (subs && A_TYPEG(subs) == A_TRIPLE)
          subs = A_LBDG(subs);
        if (subs && A_TYPEG(subs) == A_SUBSCR)
          subs = A_LOPG(subs);
        if (subs && A_TYPEG(subs) == A_MEM)
          idast = subs;
      }
    }
    break;
  case A_ID:
    sptr = A_SPTRG(subsc);
    idast = subsc;
    subsc = 0;
    break;
  case A_MEM:
    sptr = A_SPTRG(A_MEMG(subsc));
    idast = subsc;
    subsc = 0;
    break;
  default:
    interr("conv_allocate: unexpected ast", alloc_ast, 4);
  }

  if (DTY(DTYPEG(sptr)) != TY_ARRAY)
    goto exit_;
  /* pointer based but not allocatable variables */
  if (SCG(sptr) == SC_BASED && !ALLOCG(sptr))
    goto exit_;
  if (NODESCG(sptr))
    goto exit_;

  dtype = DTYPEG(sptr);
  /* put out the array bounds assignments */

  align = ALIGNG(sptr);
  if (want_descriptor_anyway(sptr))
    DESCUSEDP(sptr, 1);

  init_fl();

  /* if this is a host subprogram, it
   * may be passed as argument in a contained subprogram,
   * but we don't know here */
  if (gbl.internal == 1 || STYPEG(sptr) == ST_MEMBER)
    DESCUSEDP(sptr, 1);
  if (DESCUSEDG(sptr) && !TPALLOCG(sptr)) {
    set_typed_alloc(a_dtype);
    emit_alnd_secd(sptr, idast, TRUE, std, subsc);
    set_typed_alloc(DT_NONE);
  }

  /* allocating an array pointer, need to plug the runtime desc gbase field */
  if (DTY(DTYPEG(sptr)) == TY_ARRAY && POINTERG(sptr) &&
      (flg.debug || XBIT(70, 0x2000000))) {
    int src;
    int dest;
    int stmt;
    if (STYPEG(sptr) == ST_MEMBER) {
      src = mk_member(A_PARENTG(idast), mk_id(MIDNUMG(sptr)),
                      DTYPEG(MIDNUMG(sptr)));
      dest = check_member(idast, get_gbase(SDSCG(sptr)));
    } else {
      src = mk_id(MIDNUMG(sptr));
      dest = get_gbase(SDSCG(sptr));
    }
    /*
     * For the time being, the pointer is copied to the gbase field
     * by the runtime routine, RTE_ptrcp().  This has always been the
     * behavior for 64-bit; however, for 32-bit, we were generating
     * assignments.  Unfortuately, there is an ili mismatch (the
     * source is 'AR' and the store expects 'IR') caught by dump_ili().
     */
    stmt = begin_call(A_CALL, sym_mkfunc_nodesc(mkRteRtnNm(RTE_ptrcp), DT_NONE),
                      2);
    add_arg(dest);
    add_arg(src);

    add_stmt_after(stmt, std);
  }

  /* may have to reset 'visit' flag */

  nd = get_tbl();
  list = clist();
  for (i = 0; i < fl.avl; i++) {
    plist(list, fl.base[i]);
  }
  tbl.base[nd].f1 = sptr;
  tbl.base[nd].f3 = list;
  FREE(fl.base);

exit_:
  std = STD_NEXT(std);
  if (STD_IGNORE(STD_PREV(std)))
    delete_stmt(STD_PREV(std));
  return std;
}

static int
lhs_dim(int forall, int astli)
{
  int lhs, lhsd;
  int ndim, asd;
  int nd;
  CTYPE *ct;
  int i;

  nd = A_OPT1G(forall);
  ct = FT_CYCLIC(nd);
  lhs = ct->lhs;
  lhsd = left_subscript_ast(lhs);
  asd = A_ASDG(lhsd);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++) {
    if (ct->idx[i])
      if (ASTLI_SPTR(ct->idx[i]) == ASTLI_SPTR(astli))
        return i;
  }
  return -1;
}

static void
conv_fused_forall(int std, int ast, int *stdnextp)
{
  int i;
  int forall;
  int nd;
  int stmt;
  int expr;
  int fusedstd;
  int exprp, exprn;
  int forallp, foralln;
  int stdnext = *stdnextp;

  nd = A_OPT1G(ast);
  if (FT_NFUSE(nd, 0) == 0)
    return;

  for (i = 0; i < FT_NFUSE(nd, 0); i++) {
    fusedstd = FT_FUSEDSTD(nd, 0, i);
    forall = STD_AST(fusedstd);
    if (i == 0)
      exprp = 0;
    else {
      forallp = STD_AST(FT_FUSEDSTD(nd, 0, i - 1));
      exprp = A_IFEXPRG(forallp);
    }

    if (i == FT_NFUSE(nd, 0) - 1)
      exprn = 0;
    else {
      foralln = STD_AST(FT_FUSEDSTD(nd, 0, i + 1));
      exprn = A_IFEXPRG(foralln);
    }

    if (A_TYPEG(forall) != A_FORALL)
      continue;
    expr = A_IFEXPRG(forall);
    if (expr && !is_same_mask(expr, exprp)) {
      insert_mask(expr, STD_PREV(stdnext));
    }

    stmt = A_IFSTMTG(forall);
    if (stmt)
      rewrite_asn(stmt, 0, FALSE, MAXSUBS);
    if (stmt) {
      if (A_SRCG(stmt) != A_DESTG(stmt)) {
        add_stmt_before(stmt, stdnext);
      }
      if (expr && !is_same_mask(expr, exprn))
        insert_endmask(expr, STD_PREV(stdnext));
    }

    if (fusedstd == stdnext) {
      *stdnextp = STD_NEXT(stdnext);
    }
    if (STD_LINENO(std) && STD_LINENO(fusedstd))
      ccff_info(MSGFUSE, "FUS030", gbl.findex, STD_LINENO(std),
                "Array assignment / Forall at line %linelist fused",
                "linelist=%d", STD_LINENO(fusedstd), NULL);
    delete_stmt(fusedstd);
  }
}

static LOGICAL
is_same_mask(int expr, int expr1)
{

  LOGICAL l, r;
  int argt, argt1;
  int sptr, sptr1;
  int dim, dim1;
  int ast, ast1;
  int ndim, ndim1;
  int asd, asd1;

  if (expr == 0 || expr1 == 0)
    return FALSE;
  if (A_TYPEG(expr) != A_TYPEG(expr1))
    return FALSE;
  switch (A_TYPEG(expr)) {
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_SUBSTR:
  case A_MEM:
  case A_TRIPLE:
  case A_LABEL:
    if (expr == expr1)
      return TRUE;
    else
      return FALSE;
  case A_SUBSCR:
    if (expr == expr1)
      return TRUE;

    sptr = sym_of_ast(expr);
    sptr1 = sym_of_ast(expr1);
    /* compare a$arrdsc(41) with b$arrdsc(41)
     * if a and b distributed the same way this should be equal */
    if (STYPEG(sptr) == ST_ARRDSC && STYPEG(sptr1) == ST_ARRDSC) {
      asd = A_ASDG(expr);
      ndim = ASD_NDIM(asd);
      asd1 = A_ASDG(expr1);
      ndim1 = ASD_NDIM(asd1);
      assert(ndim == 1 && ndim == ndim1, "is_same_mask: unmatched ndim", expr,
             3);
      if (ndim != ndim1)
        return FALSE;
      if (ASD_SUBS(asd, 0) != ASD_SUBS(asd1, 0))
        return FALSE;
      sptr = ARRAYG(sptr);
      sptr1 = ARRAYG(sptr1);
      assert(sptr && sptr1, "is_same_mask: can not find original array", sptr,
             3);
      if (is_same_array_alignment(sptr, sptr1))
        return TRUE;
    }
    return FALSE;

  case A_BINOP:
    if (A_DTYPEG(expr) != A_DTYPEG(expr1))
      return FALSE;
    if (A_OPTYPEG(expr) != A_OPTYPEG(expr1))
      return FALSE;
    l = is_same_mask(A_LOPG(expr), A_LOPG(expr1));
    r = is_same_mask(A_ROPG(expr), A_ROPG(expr1));
    return l && r;
  case A_UNOP:
    if (A_DTYPEG(expr) != A_DTYPEG(expr1))
      return FALSE;
    if (A_OPTYPEG(expr) != A_OPTYPEG(expr1))
      return FALSE;
    return is_same_mask(A_LOPG(expr), A_LOPG(expr1));
  case A_PAREN:
    return is_same_mask(A_LOPG(expr), A_LOPG(expr1));
  case A_CONV:
    if (A_DTYPEG(expr) != A_DTYPEG(expr1))
      return FALSE;
    return is_same_mask(A_LOPG(expr), A_LOPG(expr1));
  case A_INTR:
  case A_FUNC:
    if (expr == expr1)
      return TRUE;
    sptr = A_SPTRG(A_LOPG(expr));
    sptr1 = A_SPTRG(A_LOPG(expr1));
    if (sptr != sptr1)
      return FALSE;
    if (strcmp(SYMNAME(sptr), mkRteRtnNm(RTE_islocal_idx)) != 0)
      return FALSE;
    argt = A_ARGSG(expr);
    argt1 = A_ARGSG(expr1);

    sptr = ARRAYG(memsym_of_ast(ARGT_ARG(argt, 0)));
    dim = get_int_cval(A_SPTRG(ARGT_ARG(argt, 1)));
    ast = ARGT_ARG(argt, 2);

    sptr1 = ARRAYG(memsym_of_ast(ARGT_ARG(argt1, 0)));
    dim1 = get_int_cval(A_SPTRG(ARGT_ARG(argt1, 1)));
    ast1 = ARGT_ARG(argt1, 2);

    if (ast != ast1)
      return FALSE;

    return TRUE;
  default:
    interr("is_same_mask: unexpected ast", expr, 2);
    return FALSE;
  }
}

static LOGICAL
is_same_mask_in_fused(int std, int *pos)
{
  int forall, forall1;
  int fusedstd;
  int nd;
  int expr, expr1;
  int i;
  CTYPE *ct;
  int max;
  int ast, src;

  /* put all the mask first */
  forall = STD_AST(std);
  nd = A_OPT1G(forall);
  ct = FT_CYCLIC(nd);

  expr = A_IFEXPRG(forall);
  for (i = 0; i < FT_NFUSE(nd, 0); i++) {
    fusedstd = FT_FUSEDSTD(nd, 0, i);
    forall1 = STD_AST(fusedstd);
    expr1 = A_IFEXPRG(forall1);
    if (!is_same_mask(expr, expr1))
      return FALSE;
  }

  /* don't let cyclic and block-cyclic to be mask fused */
  /*    for (i=0;i<7;i++)
      if (ct->cb_block[i] || ct->c_init[i]) return FALSE;
      */

  *pos = position_finder(forall, expr);

  /* Find the position of GETs calls at forall */
  for (i = 0; i < FT_NMGET(nd); i++) {
    ast = glist(FT_MGET(nd), i);
    assert(A_TYPEG(ast) == A_HGETSCLR, "find_mask_calls_pos: wrong ast type",
           ast, 3);
    src = A_SRCG(ast);
    max = position_finder(forall, src);
    if (max > *pos)
      *pos = max;
  }

  max = find_max_of_mask_calls_pos(forall);
  if (max > *pos)
    *pos = max;

  for (i = 0; i < FT_NFUSE(nd, 0); i++) {
    fusedstd = FT_FUSEDSTD(nd, 0, i);
    forall1 = STD_AST(fusedstd);
    A_IFEXPRP(forall1, 0);
  }

  return TRUE;
}

#ifdef FLANG_OUTCONV_UNUSED
/* Register the barrier at stdBar for all FORALL statements fused with
 * astForall. If bBefore = TRUE, the barrier occurs before the loop. */
static void
record_fused_barriers(LOGICAL bBefore, int astForall, int stdBar)
{
  int ift;
  int nFused, iFused;
  int stdFused;
  int astFused;

  ift = A_OPT1G(astForall);
  if (!ift)
    return;
  nFused = FT_NFUSE(ift, 0);

  for (iFused = 0; iFused < nFused; iFused++) {
    stdFused = FT_FUSEDSTD(ift, 0, iFused);
    astFused = STD_AST(stdFused);
    if (!astFused)
      continue;
    record_barrier(bBefore, astFused, stdBar);
  }
}
#endif

int
conv_forall(int std)
{
  int forall;
  int stmt;
  int newast;
  int stdnext;
  int triplet_list;
  int triplet;
  int index_var;
  int n;
  int expr;
  int std1;
  int ldim;
  int nd;
  CTYPE *ct;
  int i;
  int revers[7];
  int pos, cnt;
  LOGICAL samemask;
  int lhs_sptr, lhs_ast;
  int doifstmt, ifexpr, zero;
  int stride, tmp_ifexpr;

  stdnext = STD_NEXT(std);
  if (no_effect_forall(std))
    return stdnext;

  forall = STD_AST(std);
  n = 0;
  triplet_list = A_LISTG(forall);
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list))
    n++;
  find_mask_calls_pos(forall);
  pos = n;
  samemask = is_same_mask_in_fused(std, &pos);
  find_stmt_calls_pos(forall, pos);

  n = 0;
  triplet_list = A_LISTG(forall);
  nd = A_OPT1G(forall);
  lhs_ast = left_subscript_ast(A_DESTG(A_IFSTMTG(forall)));
  lhs_sptr = memsym_of_ast(lhs_ast);

  ct = FT_CYCLIC(nd);
  if (ct->ifast)
    insert_mask(A_IFEXPRG(ct->ifast), STD_PREV(stdnext));

  doifstmt = 1; /* only place if stmt if stride is 1 for now */
  ifexpr = 0;
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    revers[n] = triplet_list;
    n++;
    ldim = 0;
    triplet = ASTLI_TRIPLE(triplet_list);

    if (!XBIT(34, 0x8000000)) {
      if (DTY(DT_INT) != TY_INT8 && !XBIT(68, 0x1)) {
        zero = astb.i0;
      } else {
        zero = astb.bnd.zero;
      }
      tmp_ifexpr = mk_binop(OP_SUB, A_UPBDG(triplet), A_LBDG(triplet),
                            A_DTYPEG(A_LBDG(triplet)));
      if (A_STRIDEG(triplet) != astb.i1) {
        stride = A_STRIDEG(triplet);
        if (stride == 0)
          stride = astb.i1;
      } else {
        stride = astb.i1;
      }
      tmp_ifexpr =
          mk_binop(OP_ADD, tmp_ifexpr, stride, A_DTYPEG(A_LBDG(triplet)));
      tmp_ifexpr =
          mk_binop(OP_DIV, tmp_ifexpr, stride, A_DTYPEG(A_LBDG(triplet)));
      tmp_ifexpr = mk_binop(OP_LE, tmp_ifexpr, zero, DT_LOG);
      if (ifexpr) {
        ifexpr = mk_binop(OP_LOR, tmp_ifexpr, ifexpr, DT_LOG);
      } else {
        ifexpr = tmp_ifexpr;
      }
    }

    if (ct->lhs)
      ldim = lhs_dim(forall, triplet_list);
    if (ldim >= 0) {
      if (ct->cb_init[ldim])
        add_stmt_before(ct->cb_init[ldim], stdnext);
      if (ct->cb_do[ldim])
        add_stmt_before(ct->cb_do[ldim], stdnext);
      if (ct->cb_block[ldim]) {
        int astBlock = ct->cb_block[ldim];
        int astCall, ast1;
        int argt;
        int dim;

        if (normalize_bounds(lhs_sptr)) {
          assert(A_TYPEG(astBlock) == A_CALL && A_ARGCNTG(astBlock) == 8,
                 "conv_forall: missing block_loop", std, 4);
          argt = A_ARGSG(astBlock);
          dim = get_int_cval(A_SPTRG(ARGT_ARG(argt, 1))) - 1;
          assert(ldim == dim, "conv_forall: missing dim in block_loop", std, 4);

          astCall = begin_call(A_CALL, sym_of_ast(A_LOPG(astBlock)), 8);
          add_arg(ARGT_ARG(argt, 0)); /* descriptor */
          add_arg(ARGT_ARG(argt, 1)); /* dimension */

          /* Normalize the lower bound. */
          ast1 = ARGT_ARG(argt, 2);
          ast1 = sub_lbnd(DTYPEG(lhs_sptr), dim, ast1, lhs_ast);
          add_arg(ast1); /* lower bound */

          /* Normalize the upper bound. */
          ast1 = ARGT_ARG(argt, 3);
          ast1 = sub_lbnd(DTYPEG(lhs_sptr), dim, ast1, lhs_ast);
          add_arg(ast1); /* upper bound */

          add_arg(ARGT_ARG(argt, 4)); /* stride */

          add_arg(ARGT_ARG(argt, 5)); /* cycle # */

          add_arg(ARGT_ARG(argt, 6)); /* output lower bound */
          add_arg(ARGT_ARG(argt, 7)); /* output upper bound */
          add_stmt_before(astCall, stdnext);

          ast1 = add_lbnd(DTYPEG(lhs_sptr), dim, ARGT_ARG(argt, 6), lhs_ast);
          ast1 = mk_assn_stmt(ARGT_ARG(argt, 6), ast1, DT_INT);
          add_stmt_before(ast1, stdnext);

          ast1 = add_lbnd(DTYPEG(lhs_sptr), dim, ARGT_ARG(argt, 7), lhs_ast);
          ast1 = mk_assn_stmt(ARGT_ARG(argt, 7), ast1, DT_INT);
          add_stmt_before(ast1, stdnext);
        } else
          add_stmt_before(astBlock, stdnext);
      }
    }
  }
  /* don't do one dimension */
  if (n <= 1 || STD_ZTRIP(std) != 1)
    doifstmt = 0;

  triplet_list = A_LISTG(forall);

  cnt = 0;
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    int dovar, tstd;
    ldim = 0;
    if (ct->lhs)
      ldim = lhs_dim(forall, triplet_list);
    if (ldim >= 0 && ct->c_init[ldim])
      add_stmt_before(ct->c_init[ldim], stdnext);

    add_mask_calls(cnt, forall, stdnext);

    if (samemask && cnt == pos) {
      expr = A_IFEXPRG(forall);
      if (expr)
        insert_mask(expr, STD_PREV(stdnext));
    }

    add_stmt_calls(cnt, forall, stdnext);

    index_var = ASTLI_SPTR(triplet_list);
    triplet = ASTLI_TRIPLE(triplet_list);

    newast = mk_stmt(A_DO, 0);
    dovar = mk_id(index_var);
    A_DOVARP(newast, dovar);
    A_M1P(newast, A_LBDG(triplet));
    A_M2P(newast, A_UPBDG(triplet));
    if (A_STRIDEG(triplet) != astb.i1) {
      A_M3P(newast, A_STRIDEG(triplet));
    } else {
      A_M3P(newast, astb.i1);
    }
    A_M4P(newast, ifexpr);

    tstd = add_stmt_before(newast, stdnext);

    STD_ZTRIP(tstd) = 0;
    if (doifstmt && !XBIT(34, 0x8000000)) {
      STD_ZTRIP(tstd) = 1;
    }

    cnt++;
  }

  add_mask_calls(cnt, forall, stdnext);

  if (cnt == pos) {
    expr = A_IFEXPRG(forall);
    if (expr)
      insert_mask(expr, STD_PREV(stdnext));
  }

  add_stmt_calls(cnt, forall, stdnext);

  if (ct->inner_cyclic)
    for (i = 0; i < ct->inner_cyclic->nitem; i++)
      add_stmt_before(glist(ct->inner_cyclic, i), stdnext);

  stmt = A_IFSTMTG(forall);

  /*
  plist = FT_PCALL(nd);
  for(ip = 0; ip< FT_NPCALL(nd); ip++) {
    pstd = plist->item;
    plist = plist->next;
    past = STD_AST(pstd);
    delete_stmt(pstd);
    pstd=add_stmt_before(past, stdnext);
    pghpf_local_mode = 1;
    transform_ast(pstd, past);
    pghpf_local_mode = 0;
  }
  */
  arg_gbl.std = stdnext;
  rewrite_asn(stmt, 0, FALSE, MAXSUBS);
  if (stmt) {
    /* perhaps should move this part related to elemental function
     * to another function.
     * At this point, a function is already converted to a subroutine call.
     * It was done in semfunc.c in func_call().
     */
    int ast;
    int rhs = A_SRCG(stmt);
    int lhs = A_DESTG(stmt);
    int func_ast = 0;
    int func_sptr = 0;
    int dt = 0;
    int afunc = 0;

    if ((afunc = (A_TYPEG(rhs) == A_FUNC))) {
      func_ast = A_LOPG(rhs);
      func_sptr = A_SPTRG(func_ast);
      dt = DTYPEG(func_sptr);
    }
    if (afunc && func_sptr && ELEMENTALG(func_sptr) && ADJLENG(func_sptr)) {
      int argcnt, argt;
      int result_sptr = A_SPTRG(ARGT_ARG(A_ARGSG(rhs), 0));
      int result_ast = mk_id(result_sptr);

      /* make A_CALL instead of A_FUNC */
      argcnt = A_ARGCNTG(rhs);
      argt = mk_argt(argcnt);
      ast = mk_func_node(A_CALL, mk_id(func_sptr), argcnt, A_ARGSG(rhs));
      std = add_stmt_before(ast, stdnext);

      /* b(i) = scalar_temp */
      ast = mk_assn_stmt(lhs, result_ast, dt);
      std = add_stmt_after(ast, std);
      rewrite_asn(ast, 0, FALSE, MAXSUBS);
    } else if (A_TYPEG(rhs) == A_INTR &&
               (A_OPTYPEG(rhs) == I_ADJUSTL || A_OPTYPEG(rhs) == I_ADJUSTR)) {
      /* make a scalar temp instead of an array to avoid allocating memory. In
         the case of adjust(l/r) the size of result string is same as incoming
         string. So, storing the return value can be optimized out. Hence, the
         use of a scalar temp.
      */
      lhs = mk_id(get_temp(DT_INT));
      ast = mk_assn_stmt(lhs, rhs, dt);
      add_stmt_before(ast, stdnext);
    } else if (A_TYPEG(rhs) == A_INTR && A_OPTYPEG(rhs) == I_TRIM) {
      /* In case of trim, the return value needs to be retained as the size
         of the returning string may change, hence the incoming lhs with an
         array of temps need to be retained.
      */
      ast = mk_assn_stmt(lhs, rhs, dt);
      add_stmt_before(ast, stdnext);
    } else if (A_SRCG(stmt) != A_DESTG(stmt)) {
      add_stmt_before(stmt, stdnext);
    }
    if (!samemask && expr)
      insert_endmask(expr, STD_PREV(stdnext));

    conv_fused_forall(std, forall, &stdnext);

    for (i = n - 1; i >= 0; i--) {
      int tstd;
      triplet_list = revers[i];
      if (samemask && i + 1 == pos && expr)
        insert_endmask(expr, STD_PREV(stdnext));

      ldim = 0;
      if (ct->lhs)
        ldim = lhs_dim(forall, triplet_list);
      if (ldim >= 0 && ct->c_inc[ldim])
        add_stmt_before(ct->c_inc[ldim], stdnext);

      newast = mk_stmt(A_ENDDO, 0);
      tstd = add_stmt_before(newast, stdnext);
      if (doifstmt)
        STD_ZTRIP(tstd) = 1;
    }

    if (samemask && i + 1 == pos && expr)
      insert_endmask(expr, STD_PREV(stdnext));
    for (i = n - 1; i >= 0; i--) {
      triplet_list = revers[i];
      ldim = 0;
      if (ct->lhs)
        ldim = lhs_dim(forall, triplet_list);
      if (ldim >= 0) {
        if (ct->cb_inc[ldim])
          add_stmt_before(ct->cb_inc[ldim], stdnext);
        if (ct->cb_enddo[ldim])
          add_stmt_before(ct->cb_enddo[ldim], stdnext);
      }
    }
    if (ct->endifast)
      insert_endmask(A_IFEXPRG(ct->ifast), STD_PREV(stdnext));
  } else {
    int tstd;
    while (TRUE) {
      std1 = stdnext;
      stmt = STD_AST(stdnext);
      stdnext = STD_NEXT(stdnext);
      if (A_TYPEG(stmt) == A_ENDFORALL) {
        if (expr)
          insert_endmask(expr, STD_PREV(stdnext));
        newast = mk_stmt(A_ENDDO, 0);
        while (n--) {
          tstd = add_stmt_before(newast, stdnext);
          if (doifstmt)
            STD_ZTRIP(tstd) = 1;
        }
        delete_stmt(std);  /* delete forall */
        delete_stmt(std1); /* delede endforall */
        break;
      } else if (A_TYPEG(stmt) == A_FORALL)
        stdnext = conv_forall(std);
      assert(stdnext, "conv_forall:unmatched forall", std, 4);
    }
  }

  /* fix up line numbers and propagate par flag */
  for (i = std; i != stdnext; i = STD_NEXT(i)) {
    STD_LINENO(i) = STD_LINENO(std);
    STD_PAR(i) = STD_PAR(std);
    STD_TASK(i) = STD_TASK(std);
    STD_ACCEL(i) = STD_ACCEL(std);
    STD_KERNEL(i) = STD_KERNEL(std);
  }

  /* for parallel PURE calls */
  pure_gbl.local_mode = 1;
  search_pure_function(std, stdnext);
  pure_gbl.local_mode = 0;
  ast_to_comment(forall);
  return stdnext;
}

static void
replace_loop_on_fuse_list(int oldloop, int maskloop)
{
  int nd = A_OPT1G(STD_AST(oldloop));
  int head = FT_HEADER(nd);
  int nfused;
  int i;
  nd = A_OPT1G(STD_AST(head));
  nfused = FT_NFUSE(nd, 0);
  for (i = 0; i < nfused; i++) {
    if (FT_FUSEDSTD(nd, 0, i) == oldloop) {
      FT_FUSEDSTD(nd, 0, i) = maskloop;
      break;
    }
  }
}

/* ast for forall */
/* ast for subscript expression */
/* statement before which to allocate temp */
/* statement after which to deallocate temp */
/* datatype, or zero */
/* ast with data type of element required */
static int
get_temp_forall2(int forall_ast, int subscr_ast, int alloc_stmt,
                 int dealloc_stmt, int dty, int ast_dty)
{
  int sptr = 0, astd, dstd, asd = 0;
  int subscr[MAXSUBS];
  int par, ndim, lp, std, fg, fg2, lp2;
  int save_sc;
  int dtype = dty ? dty : (DDTG(A_DTYPEG(ast_dty)));
  int cvlen = 0;
  T_LIST *q;
  lp = 0;
  cvlen = 0;
  std = alloc_stmt;

  fg = STD_FG(std);
  if (A_TYPEG(subscr_ast) == A_MEM) {
    goto new_sptr;
    /* subscr_ast = A_PARENTG(subscr_ast); */
  }
  asd = A_ASDG(subscr_ast);
  ndim = ASD_NDIM(asd);

  if (fg)
    lp = FG_LOOP(fg);
  else
    goto new_sptr;

  if (!lp)
    goto new_sptr;

  if (LP_MEXITS(lp))
    goto new_sptr;

  /* don't do char for now */
  if (DTY(dtype) == TY_CHAR)
    goto new_sptr;

  add_loop_hd(lp);

  /* notes that loop may change when we re-init */
  for (q = templist; q; q = q->next) {
    fg2 = STD_FG(q->std);
    if (fg2)
      lp2 = FG_LOOP(fg2);
    else
      continue;
    if (!lp2)
      continue;

    if (q->std == std || q->dtype != dtype || q->cvlen != cvlen ||
        q->sc != symutl.sc || LP_PARENT(lp2) != LP_PARENT(lp))
      continue;

    if (ndim != ASD_NDIM(q->asd))
      continue;
    if (same_forall_size(lp2, lp, 0)) {
#if DEBUG
      if (DBGBIT(43, 0x800)) {
        fprintf(gbl.dbgfil, "Reuse tmp array ostd:%d new:%d sptr:%d\n", q->std,
                std, sptr);
      }
#endif

      /* add and remove stmts to flowgraph */
      rdilts(fg);
      dstd = mk_mem_deallocate(mk_id(q->temp), dealloc_stmt);
      FG_STDLAST(fg) = dstd;
      wrilts(fg);

      rdilts(fg2);
      FG_STDLAST(fg2) = STD_PREV(FG_STDLAST(fg2));
      wrilts(fg2);

      ast_to_comment(STD_AST(q->dstd));
      q->dstd = dstd;
      q->std = std;
      STD_HSTBLE(q->astd) = dstd;
      STD_HSTBLE(q->dstd) = q->astd;
      par = STD_PAR(alloc_stmt) || STD_TASK(alloc_stmt);
      if (par) {
        save_sc = symutl.sc;
        set_descriptor_sc(SC_PRIVATE);
      }
      if (dty) {
        sptr = get_forall_subscr(forall_ast, subscr_ast, subscr, dty);
      } else {
        sptr = get_forall_subscr(forall_ast, subscr_ast, subscr,
                                 DDTG(A_DTYPEG(ast_dty)));
      }
      if (par) {
        set_descriptor_sc(save_sc);
      }
      return q->temp;
    }
  }
new_sptr:
  par = STD_PAR(alloc_stmt) || STD_TASK(alloc_stmt);
  if (par) {
    save_sc = symutl.sc;
    set_descriptor_sc(SC_PRIVATE);
  }

  if (dty) {
    sptr = mk_forall_sptr(forall_ast, subscr_ast, subscr, dty);
  } else {
    sptr =
        mk_forall_sptr(forall_ast, subscr_ast, subscr, DDTG(A_DTYPEG(ast_dty)));
  }
  if (par) {
    set_descriptor_sc(save_sc);
  }

  if (fg) {
    rdilts(fg);
  }
  astd = mk_mem_allocate(mk_id(sptr), subscr, alloc_stmt, ast_dty);
  dstd = mk_mem_deallocate(mk_id(sptr), dealloc_stmt);
  if (fg)
    wrilts(fg);

  if (!par) {
    STD_HSTBLE(astd) = dstd;
    STD_HSTBLE(dstd) = astd;
    if (STD_ACCEL(alloc_stmt))
      STD_RESCOPE(astd) = 1;
    if (STD_ACCEL(dealloc_stmt))
      STD_RESCOPE(dstd) = 1;
  }

  GET_T_LIST(q);
  q->next = templist;
  templist = q;
  q->temp = sptr;
  q->asd = asd;
  q->dtype = dtype;
  q->cvlen = cvlen;
  q->std = std;
  q->sc = symutl.sc;
  q->astd = astd;
  q->dstd = dstd;

  return sptr;
}

static LOGICAL
is_pointer(int ast)
{
  if (A_TYPEG(ast) == A_SUBSCR)
    ast = A_LOPG(ast);
  if (A_TYPEG(ast) == A_MEM)
    ast = A_MEMG(ast);
  if (A_TYPEG(ast) != A_ID)
    return FALSE;
  if (POINTERG(A_SPTRG(ast)))
    return TRUE;
  return FALSE;
}

/* This routine  is to check whether forall has dependency.
 * If it has, it creates temp which is shape array with lhs.
 * For example,
 *              forall(i=1:N) a(i) = a(i-1)+.....
 * will be rewritten
 *              forall(i=1:N) temp(i) = a(i-1)+.....
 *              forall(i=1:N) a(i) = temp(i)
 */

/*
 * This routine assumes that input is block forall with an assignment
 * statement in it.
 */
static void
forall_dependency(int std)
{
  int lhs, rhs;
  int asn;
  int sptr;
  int temp_ast;
  int newasn;
  int forall;
  int newforall;
  int newstd;
  int nd;
  int header;
  int lineno;
  LOGICAL bIndep, isdepend;
  int sptr_lhs;
  CTYPE *ct;
  int lhso;
  int par;
  int task;
  int expr;

  forall = STD_AST(std);
  par = STD_PAR(std);
  task = STD_TASK(std);
  asn = A_IFSTMTG(forall);
  lhs = A_DESTG(asn);
  sptr_lhs = sym_of_ast(lhs);
  rhs = A_SRCG(asn);
  expr = A_IFEXPRG(forall);

  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  /* find pointer original lhs */
  if (POINTERG(sptr_lhs)) {
    ct = FT_CYCLIC(nd);
    if (ct && ct->lhs)
      lhso = ct->lhs;
    else
      lhso = lhs;
  } else
    lhso = lhs;

  /* forall-independent */
  lineno = STD_LINENO(std);
  open_pragma(lineno);
  bIndep = XBIT(19, 0x100) != 0;
  if (bIndep) {
    close_pragma();
    return;
  }

  /* take conditional expr, if there is dependency */
  if (expr)
    if (is_dependent(lhs, expr, forall, std, std) ||
        is_mask_call_dependent(forall, lhs)) {
      if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
        if (is_pointer(lhs) && !lhs_needtmp(lhs, rhs, std))
          return;
        /* get_temp_forall2() is defined in this file */
        sptr = get_temp_forall2(forall, lhs, header, std, DT_LOG, 0);
      } else {
        /* symutl.c:get_temp_forall() */
        sptr = get_temp_forall(forall, lhs, header, std, DT_LOG, 0);
      }
      if (flg.opt >= 2 && !XBIT(2, 0x400000))
        temp_ast = reference_for_temp(sptr, lhs, forall);
      else
        temp_ast = reference_for_temp(sptr, lhso, forall);
      A_IFEXPRP(forall, temp_ast);
      newforall = mk_stmt(A_FORALL, 0);
      A_LISTP(newforall, A_LISTG(forall));
      A_OPT1P(newforall, A_OPT1G(forall));
      A_IFEXPRP(newforall, 0);
      newasn = mk_stmt(A_ASN, 0);
      A_DESTP(newasn, temp_ast);
      A_SRCP(newasn, expr);
      A_IFSTMTP(newforall, newasn);
      move_mask_calls(newforall);
      remove_mask_calls(newforall);
      remove_mask_calls(forall);
      newstd = add_stmt_before(newforall, std);
      STD_PAR(newstd) = par;
      STD_TASK(newstd) = task;

      /* add the newstd to the std fuse list */
      replace_loop_on_fuse_list(std, newstd);

      report_comm(std, DEPENDENCY_CAUSE);
      un_fuse(forall);
      un_fuse(newforall);

      /* need to add this to flow graph otherwise add_loop_hd will drop it */
      if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
        int fg = STD_FG(std);
        int newfg = add_fg(FG_LPREV(fg));
        FG_STDLAST(newfg) = newstd;
        FG_STDFIRST(newfg) = newstd;
      }
    }

  isdepend = is_dependent(lhs, rhs, forall, std, std);
  if (isdepend || is_stmt_call_dependent(forall, lhs)) {
    if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
      if (is_pointer(lhs) && !lhs_needtmp(lhs, rhs, std)) {
        return;
      }
      /* get_temp_forall2() is defined in this file */
      sptr = get_temp_forall2(forall, lhs, header, std, 0, lhs);
    } else {
      /* symutl.c:get_temp_forall() */
      sptr = get_temp_forall(forall, lhs, header, std, 0, lhs);
    }
    if (flg.opt >= 2 && !XBIT(2, 0x400000))
      temp_ast = reference_for_temp(sptr, lhs, forall);
    else
      temp_ast = reference_for_temp(sptr, lhso, forall);
    A_DESTP(asn, temp_ast);
    A_IFSTMTP(forall, asn);
    newforall = mk_stmt(A_FORALL, 0);
    A_LISTP(newforall, A_LISTG(forall));
    A_OPT1P(newforall, A_OPT1G(forall));
    A_IFEXPRP(newforall, A_IFEXPRG(forall));
    newasn = mk_stmt(A_ASN, 0);
    A_DESTP(newasn, lhs);
    A_SRCP(newasn, temp_ast);
    A_IFSTMTP(newforall, newasn);
    remove_mask_calls(newforall);
    remove_stmt_calls(newforall);
    newstd = add_stmt_after(newforall, std);
    STD_PAR(newstd) = par;
    STD_TASK(newstd) = task;
    report_comm(std, DEPENDENCY_CAUSE);
    un_fuse(forall);
    un_fuse(newforall);
  }
  close_pragma();
}

static LOGICAL
is_stmt_call_dependent(int forall, int lhs)
{
  int nd;
  int cstd;
  int i;
  LOGICAL l;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NSCALL(nd); i++) {
    cstd = glist(FT_SCALL(nd), i);
    l = is_call_dependent(cstd, forall, lhs);
    if (l)
      return TRUE;
  }
  return FALSE;
}

static LOGICAL
is_mask_call_dependent(int forall, int lhs)
{
  int nd;
  int cstd;
  int i;
  LOGICAL l;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    l = is_call_dependent(cstd, forall, lhs);
    if (l)
      return TRUE;
  }
  return FALSE;
}

static LOGICAL
is_call_dependent(int std, int forall, int lhs)
{
  int ast, ast1;
  int std1;
  int nd, nd1;
  int i;
  int argt;
  int nargs;
  LOGICAL l;

  ast = STD_AST(std);
  nd = A_OPT1G(ast);
  assert(nd, "is_call_dependent: uninitialized pure call", ast, 3);
  nargs = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);
  for (i = 0; i < nargs; ++i) {
    l = is_dependent(lhs, ARGT_ARG(argt, i), forall, std, std);
    if (l)
      return TRUE;
  }

  for (i = 0; i < FT_CALL_NCALL(nd); i++) {
    std1 = glist(FT_CALL_CALL(nd), i);
    ast1 = STD_AST(std1);
    nd1 = A_OPT1G(ast1);
    assert(nd1, "is_call_dependent: uninitialized pure call", ast1, 3);
    l = is_call_dependent(std1, forall, lhs);
    if (l)
      return TRUE;
  }
  return FALSE;
}

static void
move_mask_calls(int forall)
{
  int nd;
  int nd1;

  nd = A_OPT1G(forall);
  nd1 = mk_ftb();
  BCOPY(ftb.base + nd1, ftb.base + nd, FT, 1);
  FT_NSCALL(nd1) = FT_NMCALL(nd);
  FT_SCALL(nd1) = FT_MCALL(nd);
  FT_NSGET(nd1) = FT_NMGET(nd);
  FT_SGET(nd1) = FT_MGET(nd);
  A_OPT1P(forall, nd1);
}
static void
remove_mask_calls(int forall)
{
  int nd;
  int nd1;

  nd = A_OPT1G(forall);
  nd1 = mk_ftb();
  BCOPY(ftb.base + nd1, ftb.base + nd, FT, 1);
  FT_NMCALL(nd1) = 0;
  FT_MCALL(nd1) = clist();
  FT_NMGET(nd1) = 0;
  FT_MGET(nd1) = clist();
  A_OPT1P(forall, nd1);
}

static void
remove_stmt_calls(int forall)
{
  int nd;
  int nd1;

  nd = A_OPT1G(forall);
  nd1 = mk_ftb();
  BCOPY(ftb.base + nd1, ftb.base + nd, FT, 1);
  FT_NSCALL(nd1) = 0;
  FT_SCALL(nd1) = clist();
  FT_NSGET(nd1) = 0;
  FT_SGET(nd1) = clist();
  A_OPT1P(forall, nd1);
}

/* This routine return TRUE if there is a possiblity that
 * sptr is pointer and points sptr1 or
 * sptr1 is pointer and points sptr
 * otherwise return FALSE;
 * ### add pointer target information here
 */
LOGICAL
is_pointer_dependent(int sptr, int sptr1)
{
  if (DTY(DTYPEG(sptr)) != DTY(DTYPEG(sptr1)))
    return FALSE;
  if (POINTERG(sptr))
    if (POINTERG(sptr1) || TARGETG(sptr1))
      return TRUE;

  if (POINTERG(sptr1))
    if (POINTERG(sptr) || TARGETG(sptr))
      return TRUE;
  return FALSE;
}

/* ARRAY COLLAPSING */

/* typedefs for array collapsing */
typedef struct {
  int astArr;     /* SUBSCR AST of compiler-created array */
  int stdAlloc;   /* STD of allocate statement for astArr */
  int stdDealloc; /* STD of deallocate statement for astArr */
  int lp;         /* loop # defs of astArr */
  int astSclr;    /* AST of new scalar */
  union {
    INT16 all;
    struct {
      unsigned descr : 1;  /* found use of the array's descriptor */
      unsigned delete : 1; /* entry has been deleted */
    } bits;
  } flags;
} COLLAPSE;

/* macros for array collapsing */
#define COLLAPSE_ASTARR(i) collapse.base[i].astArr
#define COLLAPSE_STDALLOC(i) collapse.base[i].stdAlloc
#define COLLAPSE_STDDEALLOC(i) collapse.base[i].stdDealloc
#define COLLAPSE_LP(i) collapse.base[i].lp
#define COLLAPSE_ASTSCLR(i) collapse.base[i].astSclr
#define COLLAPSE_DESCR(i) collapse.base[i].flags.bits.descr
#define COLLAPSE_DELETE(i) collapse.base[i].flags.bits.delete

/* local storage for array collapsing */
static struct {
  COLLAPSE *base; /* the COLLAPSE table */
  int size;       /* size of the COLLAPSE table */
  int avail;      /* next available struct in the COLLAPSE table */
  int lp;         /* current loop */
  int std;        /* current STD */
} collapse;

static void
init_collapse(void)
{
  /* Initialize local storage. */
  collapse.size = 100;
  NEW(collapse.base, COLLAPSE, collapse.size);
  collapse.avail = 1;
}

/* Replace all compiler-created temp arrays that are used only within
 * one loop with scalars. */
static void
collapse_arrays(void)
{
  int ast;
  int ci;
  int sptrArr, sptrSclr;
  int nscalars;

  /* Scan STDs looking for ALLOCATE/DEALLOCATE statements. */
  find_collapse_allocs();

  /* Build the loop table. */
  hlopt_init(0);
#if DEBUG
  if (DBGBIT(43, 1))
    dump_flowgraph();
#endif
#if DEBUG
  if (DBGBIT(43, 4))
    dump_loops();
#endif

  /* Determine if all defs of each array are within a single loop. */
  find_collapse_defs();

  /* Determine if all uses of each array are within their defining loops. */
  find_collapse_uses();

  /* Create new scalars */
  nscalars = 0;
  for (ci = 1; ci < collapse.avail; ci++) {
    if (COLLAPSE_DELETE(ci))
      continue;
    if (!COLLAPSE_ASTARR(ci) || A_TYPEG(COLLAPSE_ASTARR(ci)) != A_SUBSCR) {
      delete_collapse(ci);
      continue;
    }
    sptrArr = memsym_of_ast(COLLAPSE_ASTARR(ci));
    sptrSclr = sym_get_scalar(SYMNAME(sptrArr), "s", DDTG(DTYPEG(sptrArr)));
    COLLAPSE_ASTSCLR(ci) = mk_id(sptrSclr);
    nscalars++;
  }

  if (nscalars)
    /* Collapse all arrays within the current program unit. */
    collapse_loops();

/* List loops containing collapsed arrays. */
#if DEBUG
  if (DBGBIT(43, 256))
    report_collapse(0);
#endif

  /* Mark arrays with uses of array descriptors. */
  find_descrs();

  /* Delete ALLOCATE/DEALLOCATE statements for arrays without uses of
   * array descriptors. */
  collapse_allocates(FALSE);

  /* Reclaim storage. */
  for (ast = 1; ast < astb.stg_avail; ast++)
    A_OPT2P(ast, 0);
  hlopt_end(0, 0);

#if DEBUG
  if (DBGBIT(43, 128)) {
    fprintf(gbl.dbgfil, "----- Statements after array collapsing -----\n");
    dump_std();
  }
#endif
}

/* Frees memory used to collapse arrays. */
static void
end_collapse(void)
{
  FREE(collapse.base);
}

/* For each ALLOCATE of a compiler-created array, initialize an entry
 * within the COLLAPSE table. Set the OPT2 field of the
 * array's AST to the index of its COLLAPSE table entry. */
static void
find_collapse_allocs(void)
{
  int std;
  int ast, astSrc, astArr;
  int ci;

  for (std = STD_NEXT(0); std; std = STD_NEXT(std)) {
    ast = STD_AST(std);
    if (A_TYPEG(ast) != A_ALLOC)
      continue;
    astSrc = A_SRCG(ast);
    if (A_TKNG(ast) == TK_ALLOCATE) {
      if (A_TYPEG(astSrc) != A_SUBSCR)
        continue; /* ...must be pointer ALLOCATE. */
      astArr = A_LOPG(astSrc);
      if (A_TYPEG(astArr) != A_ID)
        continue;
      if (!HCCSYMG(A_SPTRG(astArr)) || !VCSYMG(A_SPTRG(astArr)))
        continue; /* array not compiler created */
      ci = A_OPT2G(astArr);
      if (ci) {
        delete_collapse(ci); /* multiple ALLOCATEs found */
        continue;
      }

      /* Create a new COLLAPSE structure. */
      ci = collapse.avail++;
      NEED(collapse.avail, collapse.base, COLLAPSE, collapse.size,
           collapse.size + 100);
      BZERO(&collapse.base[ci], COLLAPSE, 1);
      COLLAPSE_ASTARR(ci) = astArr;
      COLLAPSE_STDALLOC(ci) = std;

      /* Set the OPT2 field in the ID AST to point to the COLLAPSE
       * structure. */
      A_OPT2P(astArr, ci);
    } else /* A_TKNG(ast) == TK_DEALLOCATE */ {
      astArr = astSrc;
      ci = A_OPT2G(astArr);
      if (!ci || COLLAPSE_DELETE(ci))
        continue; /* array doesn't qualify */
      if (COLLAPSE_STDDEALLOC(ci)) {
        delete_collapse(ci); /* multiple DEALLOCATEs found */
        continue;
      }
      COLLAPSE_STDDEALLOC(ci) = std;
    }
  }
}

/* Delete COLLAPSE table entry #ci. */
static void
delete_collapse(int ci)
{
  COLLAPSE_DELETE(ci) = TRUE;
}

/* Find the loops containing definitions of arrays within the COLLAPSE
 * table. */
static void
find_collapse_defs(void)
{
  int def;
  int nme;
  int ci;
  int lpDef;

  for (ci = 1; ci < collapse.avail; ci++) {
    if (COLLAPSE_DELETE(ci))
      continue;
    if (!COLLAPSE_STDALLOC(ci) || !COLLAPSE_STDDEALLOC(ci)) {
      delete_collapse(ci);
      continue;
    }
    nme = A_NMEG(COLLAPSE_ASTARR(ci));
    for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
      if (DEF_STD(def) == COLLAPSE_STDALLOC(ci) ||
          DEF_STD(def) == COLLAPSE_STDDEALLOC(ci))
        continue;
      lpDef = FG_LOOP(DEF_FG(def));
      if (LP_CALLFG(lpDef)) {
        delete_collapse(ci);
        break;
      }
      if (COLLAPSE_LP(ci)) {
        if (lpDef != COLLAPSE_LP(ci) || DEF_LHS(def) != COLLAPSE_ASTARR(ci)) {
          /* array assigned in multiple loops or
           * different assignments in the same loop */
          delete_collapse(ci);
          break;
        }
      } else {
        COLLAPSE_LP(ci) = lpDef;
        COLLAPSE_ASTARR(ci) = DEF_ADDR(def);
      }
    }
  }
}

/* Determine if uses of arrays in the COLLAPSE table are within the same
 * loops in which they are defined. */
static void
find_collapse_uses(void)
{
  int ci;
  int astArr;
  int nme;
  int def;
  DU *du;
  int use;

  for (ci = 1; ci < collapse.avail; ci++) {
    if (COLLAPSE_DELETE(ci))
      continue;
    astArr = COLLAPSE_ASTARR(ci);
    if (A_TYPEG(astArr) == A_SUBSCR)
      astArr = A_LOPG(astArr);
    assert(A_TYPEG(astArr) == A_ID, "find_collapse_uses: unknown array type",
           ci, 4);
    nme = A_NMEG(astArr);
    for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
      if (DEF_STD(def) == COLLAPSE_STDALLOC(ci) ||
          DEF_STD(def) == COLLAPSE_STDDEALLOC(ci))
        continue;
      for (du = DEF_DU(def); du; du = du->next) {
        use = du->use;
        if (is_parent_loop(COLLAPSE_LP(ci), FG_LOOP(USE_FG(use))) &&
            COLLAPSE_ASTARR(ci) == USE_ADDR(use))
          continue;
        delete_collapse(ci);
        goto next_ci;
      }
    }
  next_ci:;
  }
}

/* Return TRUE if lpParent is a parent loop of loop lp. */
static LOGICAL
is_parent_loop(int lpParent, int lp)
{
  if (lpParent == 0)
    return TRUE; /* all loops are descendents of loop #0 */
  for (; lp; lp = LP_PARENT(lp))
    if (lp == lpParent)
      return TRUE;
  return FALSE;
}

/* Replace collapsible arrays with scalars in all loops within loop lp. */
static void
collapse_loops(void)
{
  int std;
  int ast, astArr;
  int ci;
  int nme;
  int def;
  DU *du;
  int use;

  for (ci = 1; ci < collapse.avail; ci++) {
    if (COLLAPSE_DELETE(ci))
      continue;
    astArr = COLLAPSE_ASTARR(ci);
    if (A_TYPEG(astArr) == A_SUBSCR)
      astArr = A_LOPG(astArr);
    assert(A_TYPEG(astArr) == A_ID, "collapse_loops: unknown array type", ci,
           4);
    nme = A_NMEG(astArr);
    for (def = NME_DEF(nme); def; def = DEF_NEXT(def)) {
      if (DEF_STD(def) == COLLAPSE_STDALLOC(ci) ||
          DEF_STD(def) == COLLAPSE_STDDEALLOC(ci))
        continue;
      ast_visit(1, 1);
      ast_replace(COLLAPSE_ASTARR(ci), COLLAPSE_ASTSCLR(ci));
      std = DEF_STD(def);
      ast = ast_rewrite(STD_AST(std));
      STD_AST(std) = ast;
      A_STDP(ast, std);
      ast_unvisit();
      for (du = DEF_DU(def); du; du = du->next) {
        ast_visit(1, 1);
        use = du->use;
        ast_replace(COLLAPSE_ASTARR(ci), COLLAPSE_ASTSCLR(ci));
        std = USE_STD(use);
        ast = ast_rewrite(STD_AST(std));
        STD_AST(std) = ast;
        A_STDP(ast, std);
        ast_unvisit();
      }
    }
  }
}

static int global_astArrdsc, global_flag;

static void
look_for_descriptor(int ast, int *unused)
{
  if (ast == global_astArrdsc)
    global_flag = 1;
} /* look_for_descriptor */

/* Set the COLLAPSE_DESCR flag to TRUE for all arrays for which a descriptor
 * appears within the program. */
static void
find_descrs(void)
{
  int ci;
  int astArr, astArrdsc, ast;
  int sptrArr, sptrArrdsc;
  int std, stdend;
  int nargs, arg;
  int args;
  int src;

  for (ci = 1; ci < collapse.avail; ci++) {
    if (COLLAPSE_DELETE(ci))
      continue;
    astArr = A_LOPG(COLLAPSE_ASTARR(ci));
    sptrArr = A_SPTRG(astArr);
    if (NODESCG(sptrArr))
      continue;
    sptrArrdsc = DESCRG(sptrArr);
    astArrdsc = mk_id(sptrArrdsc);
    global_astArrdsc = astArrdsc;
    global_flag = 0;

    /* Search through STDs for an occurrence of astArrdsc in a CALL. */
    stdend = STD_NEXT(COLLAPSE_STDDEALLOC(ci));
    ast_visit(1, 1);
    for (std = COLLAPSE_STDALLOC(ci); global_flag == 0 && std != stdend;
         std = STD_NEXT(std)) {
      ast = STD_AST(std);
      if (A_TYPEG(ast) == A_CALL) {
        nargs = A_ARGCNTG(ast);
        args = A_ARGSG(ast);
        for (arg = 0; arg < nargs; arg++) {
          if (ARGT_ARG(args, arg) == astArrdsc) {
            global_flag = 1;
            break;
          }
        }
      } else if (A_TYPEG(ast) == A_ASN) {
        src = A_SRCG(ast);
        if (A_TYPEG(src) == A_SUBSCR) {
          if (A_LOPG(src) == astArrdsc) {
            global_flag = 1;
          }
        }
      } else if (A_TYPEG(ast) == A_IFTHEN) {
        /* descriptor might be used by 'gen_single' */
        ast_traverse(ast, NULL, look_for_descriptor, NULL);
      }
    }
    if (global_flag)
      COLLAPSE_DESCR(ci) = TRUE;
    ast_unvisit();
  }
}

/* If bDescr is FALSE, remove ALLOCATE/DEALLOCATE statements of
 * collapsed arrays without array descriptors. If bDescr is TRUE
 * delete ALLOCATE/DEALLOCATE statements of collapsed arrays with
 * array descriptors. */
static void
collapse_allocates(LOGICAL bDescr)
{
  int ci;

  for (ci = 1; ci < collapse.avail; ci++) {
    if (COLLAPSE_DELETE(ci) || bDescr != COLLAPSE_DESCR(ci))
      continue;
    delete_stmt(COLLAPSE_STDALLOC(ci));
    delete_stmt(COLLAPSE_STDDEALLOC(ci));
  }
}

static void
report_collapse(int lp)
{
  int ci;
  int lpi;
  int std;
  int lineno;

  for (ci = 1; ci < collapse.avail; ci++)
    if (!COLLAPSE_DELETE(ci) && COLLAPSE_LP(ci) == lp)
      break;
  if (ci < collapse.avail) {
    for (std = FG_STDFIRST(LP_HEAD(lp)); std; std = STD_PREV(std))
      if (STD_LINENO(std))
        break;
    lineno = (std ? STD_LINENO(std) : 1);
    ccff_info(MSGOPT, "OPT044", gbl.findex, lineno,
              "Temp arrays collapsed to scalars", NULL);
  }

  for (lpi = LP_CHILD(lp); lpi; lpi = LP_SIBLING(lpi))
    report_collapse(lpi);
}

#if DEBUG
#ifdef FLANG_OUTCONV_UNUSED
/* Dump the COLLAPSE table. */
static void
dump_collapse(void)
{
  int ci;

  for (ci = 1; ci < collapse.avail; ci++) {
    fprintf(gbl.dbgfil, "Entry %d:\n", ci);
    if (COLLAPSE_DELETE(ci)) {
      fprintf(gbl.dbgfil, "  DELETED\n");
      continue;
    }
    fprintf(gbl.dbgfil, "  Temp array: ");
    dbg_print_ast(COLLAPSE_ASTARR(ci), gbl.dbgfil);
    fprintf(gbl.dbgfil, "  Allocate STD %d, Deallocate STD %d, Defining loop "
                        "%d, Descriptor %d:1\n",
            COLLAPSE_STDALLOC(ci), COLLAPSE_STDDEALLOC(ci), COLLAPSE_LP(ci),
            COLLAPSE_DESCR(ci));
    if (!COLLAPSE_ASTSCLR(ci))
      continue;
    fprintf(gbl.dbgfil, "  New scalar: ");
    dbg_print_ast(COLLAPSE_ASTSCLR(ci), gbl.dbgfil);
  }
}
#endif
#endif

/* END OF ARRAY COLLAPSING */

static int
position_finder(int forall, int ast)
{
  int list1, listp;
  int isptr;
  int i;
  int reverse[7];
  int n;
  int pos;

  n = 0;
  list1 = A_LISTG(forall);
  for (listp = list1; listp != 0; listp = ASTLI_NEXT(listp)) {
    reverse[n] = ASTLI_SPTR(listp);
    n++;
  }

  pos = n;
  for (i = n - 1; i >= 0; i--) {
    isptr = reverse[i];
    if (!contains_ast(ast, mk_id(isptr)))
      pos = pos - 1;
    else
      break;
  }

  return pos;
}

static void
find_calls_pos(int std, int forall, int must_pos)
{
  int ast, ast1;
  int std1;
  int pos, pos1;
  int nd, nd1;
  int i;

  ast = STD_AST(std);
  nd = A_OPT1G(ast);
  assert(nd, "find_calls_pos: something is wrong", ast, 3);
  pos = position_finder(forall, ast);
  if (must_pos > pos)
    pos = must_pos;
  for (i = 0; i < FT_CALL_NCALL(nd); i++) {
    std1 = glist(FT_CALL_CALL(nd), i);
    ast1 = STD_AST(std1);
    nd1 = A_OPT1G(ast1);
    assert(nd1, "find_calls_pos: something is wrong", ast1, 3);
    find_calls_pos(std1, forall, must_pos);
    pos1 = FT_CALL_POS(nd1);
    if (pos1 > pos)
      pos = pos1;
  }
  FT_CALL_POS(nd) = pos;
}

static void
find_mask_calls_pos(int forall)
{
  int nd;
  int i;
  int cstd;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    find_calls_pos(cstd, forall, 0);
  }
}

static void
find_stmt_calls_pos(int forall, int mask_pos)
{
  int nd;
  int cstd;
  int i;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NSCALL(nd); i++) {
    cstd = glist(FT_SCALL(nd), i);
    find_calls_pos(cstd, forall, mask_pos);
  }
}

static int
find_max_of_mask_calls_pos(int forall)
{

  int nd, nd1;
  int i;
  int cstd;
  int ast;
  int max;
  int pos;

  max = 0;
  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    ast = STD_AST(cstd);
    nd1 = A_OPT1G(ast);
    assert(nd1, "find_calls_pos: something is wrong", ast, 3);
    pos = FT_CALL_POS(nd1);
    if (pos > max)
      max = pos;
  }
  return max;
}

static void
put_calls(int pos, int std, int stdnext)
{
  int ast, ast1;
  int std1;
  int pos1;
  int nd, nd1;
  int i;

  ast = STD_AST(std);
  nd = A_OPT1G(ast);
  assert(nd, "put_calls: something is wrong", ast, 3);
  for (i = 0; i < FT_CALL_NCALL(nd); i++) {
    std1 = glist(FT_CALL_CALL(nd), i);
    ast1 = STD_AST(std1);
    nd1 = A_OPT1G(ast1);
    assert(nd1, "put_calls: something is wrong", ast1, 3);
    put_calls(pos, std1, stdnext);
  }
  pos1 = FT_CALL_POS(nd);
  if (pos == pos1) {
    delete_stmt(std);
    std = add_stmt_before(ast, stdnext);
    pure_gbl.local_mode = 1;
    transform_call(std, ast);
    pure_gbl.local_mode = 0;
  }
}

static void
add_mask_calls(int pos, int forall, int stdnext)
{
  int nd;
  int cstd;
  int i;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    put_calls(pos, cstd, stdnext);
  }
}

static void
add_stmt_calls(int pos, int forall, int stdnext)
{
  int nd;
  int cstd;
  int i;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NSCALL(nd); i++) {
    cstd = glist(FT_SCALL(nd), i);
    put_calls(pos, cstd, stdnext);
  }
}

/* To enter local mode:
 *      pghpf_saved_local_mode = pghpf_local_mode
 *      pghpf_local_mode = 1
 */
void
enter_local_mode(int std)
{
  int ast, dest, src;
  int sptr = getsymbol("pghpf_local_mode");
  int sptr1 = getsymbol("pghpf_saved_local_mode");

  STYPEP(sptr1, ST_VAR);
  DTYPEP(sptr1, DT_INT);
  DCLDP(sptr1, 1);
  SCP(sptr1, SC_LOCAL);

  ast = mk_stmt(A_ASN, DT_INT);
  dest = mk_id(sptr1);
  A_DESTP(ast, dest);
  src = mk_id(sptr);
  A_SRCP(ast, src);
  add_stmt_before(ast, std);

  ast = mk_stmt(A_ASN, DT_INT);
  A_DESTP(ast, src);
  A_SRCP(ast, astb.i1);
  add_stmt_before(ast, std);
}

/* To exit local mode:
 *     pghpf_local_mode = pghpf_saved_local_mode
 */
void
exit_local_mode(int std)
{
  int ast, dest, src;
  int sptr = getsymbol("pghpf_local_mode");
  int sptr1 = getsymbol("pghpf_saved_local_mode");

  STYPEP(sptr1, ST_VAR);
  DTYPEP(sptr1, DT_INT);
  DCLDP(sptr1, 1);
  SCP(sptr1, SC_LOCAL);

  ast = mk_stmt(A_ASN, DT_INT);
  dest = mk_id(sptr);
  A_DESTP(ast, dest);
  src = mk_id(sptr1);
  A_SRCP(ast, src);
  add_stmt_before(ast, std);
}

static void
search_pure_function(int stdfirst, int stdlast)
{
  int std;
  int expr, newexpr;
  int ast;
  int lhs;
  int std1, ast1;
  int cnt;

  for (std = stdfirst; std != stdlast; std = STD_NEXT(std)) {
    ast = STD_AST(std);
    /* must be forall mask */
    if (A_TYPEG(ast) == A_IFTHEN) {
      /* find endif */
      cnt = 0;
      for (std1 = STD_NEXT(std); std1 != stdlast; std1 = STD_NEXT(std1)) {
        ast1 = STD_AST(std1);
        if (A_TYPEG(ast1) == A_IFTHEN)
          cnt++;
        if (A_TYPEG(ast1) == A_ENDIF) {
          if (cnt == 0)
            break;
          else
            cnt--;
        }
      }
      expr = A_IFEXPRG(ast);
      newexpr = transform_pure_function(expr, std);
      A_IFEXPRP(ast, newexpr);
    }
    /* must be forall asn */
    else if (A_TYPEG(ast) == A_ASN) {
      lhs = A_DESTG(ast);
      if (A_TYPEG(lhs) == A_SUBSCR) {
        expr = A_SRCG(ast);
        newexpr = transform_pure_function(expr, std);
        A_SRCP(ast, newexpr);
      }
    }
  }
}

static int
transform_pure_function(int expr, int std)
{

  int l, r, d, o;
  int i, nargs, argt;
  int newexpr;

  if (expr == 0)
    return expr;
  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = transform_pure_function(A_LOPG(expr), std);
    r = transform_pure_function(A_ROPG(expr), std);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = transform_pure_function(A_LOPG(expr), std);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(expr);
    l = transform_pure_function(A_LOPG(expr), std);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(expr);
    l = transform_pure_function(A_LOPG(expr), std);
    return mk_paren(l, d);
  case A_MEM:
    l = transform_pure_function(A_PARENTG(expr), std);
    r = A_MEMG(expr);
    d = A_DTYPEG(r);
    return mk_member(l, r, d);
  case A_SUBSTR:
    return expr;
  case A_INTR:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = transform_pure_function(ARGT_ARG(argt, i), std);
    }
    newexpr = mk_func_node((int)A_TYPEG(expr), A_LOPG(expr), nargs, argt);
    A_OPTYPEP(newexpr, A_OPTYPEG(expr));
    A_SHAPEP(newexpr, A_SHAPEG(expr));
    A_DTYPEP(newexpr, A_DTYPEG(expr));
    return newexpr;
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = transform_pure_function(ARGT_ARG(argt, i), std);
    }
    newexpr = mk_func_node((int)A_TYPEG(expr), A_LOPG(expr), nargs, argt);
    A_SHAPEP(newexpr, A_SHAPEG(expr));
    A_DTYPEP(newexpr, A_DTYPEG(expr));
    transform_call(std, newexpr);
    return newexpr;
  case A_CNST:
  case A_CMPLXC:
  case A_ID:
  case A_SUBSCR:
    return expr;
  default:
    interr("transform_pure_function: unknown expression", expr, 2);
    return expr;
  }
}

/*
 * return +1 at local mode exit, -1 at local mode entry
 * local mode exit is 'pghpf_local_mode = saved_pghpf_local_mode'
 * local mode entry is 'pghpf_local_mode = 1'
 */
static LOGICAL
at_local_mode(int ast)
{
  int sptr = getsymbol("pghpf_local_mode");
  int mkid = mk_id(sptr);
  if (A_DESTG(ast) == mkid) {
    int src = A_SRCG(ast);
    if (src == astb.i1) {
      return -1;
    } else {
      return +1;
    }
  }
  return 0;
} /* at_local_mode */

/*
 * eliminate barrier statements that are followed immediately by another
 * barrier statement.
 * also, eliminate barrier statements inside a 'private' mode loop.
 */
static void
eliminate_barrier(void)
{
  int std, stdPrev;
  int ast, bLocal, at;
  LOGICAL bFound;

  bFound = FALSE;
  bLocal = 0;
  for (std = STD_LAST; std; std = stdPrev) {
    stdPrev = STD_PREV(std);
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_BARRIER:
      if (bLocal) {
        /* eliminate all barrier statements in local region, */
        delete_stmt(std);
      } else if (!bFound) {
        bFound = TRUE;
      } else if (!STD_LABEL(std)) {
        delete_stmt(std);
      }
      break;
    case A_CONTINUE:
      /* eliminate useless CONTINUE statements */
      if (!STD_LABEL(std)) {
        delete_stmt(std);
      }
      break;
    case A_ASN:
      /* see if we are at the bottom or
       * top of a pghpf_local_mode region */
      at = at_local_mode(ast);
      bLocal += at;
      FLANG_FALLTHROUGH;
    default:
      bFound = FALSE;
      break;
    }
  }
}

static LOGICAL
use_offset(int sptr)
{
  LOGICAL retval;
  int dtype;
  retval = FALSE;
  if (SCG(sptr) == SC_BASED || ALLOCG(sptr) || LNRZDG(sptr)) {
    int dty;
    dtype = DTYPEG(sptr);
    dty = DTYG(dtype);
    if (NO_PTR || (NO_CHARPTR && dty == TY_CHAR) ||
        (NO_DERIVEDPTR && dty == TY_DERIVED)) {
      retval = TRUE;
    }
  }
  return retval;
} /* use_offset */

static LOGICAL
needs_linearization(int sptr)
{
  LOGICAL retval, alloc;
  int dtype;
  retval = FALSE;
  alloc = FALSE;
  if (F90POINTERG(sptr))
    return FALSE;
  if (ALLOCG(sptr)) {
    alloc = TRUE;
  } else if (F77OUTPUT) {
    dtype = DTYPEG(sptr);
    if ((DTY(dtype) == TY_ARRAY && (ADD_DEFER(dtype) || ADD_NOBOUNDS(dtype))) ||
        ALIGNG(sptr)) {
      alloc = TRUE;
    }
  }
  if (LNRZDG(sptr)) {
    retval = TRUE;
  } else if (F77OUTPUT) {
    if (alloc || use_offset(sptr)) {
      retval = TRUE;
    }
  } else if (alloc && (SCG(sptr) == SC_BASED || STYPEG(sptr) == ST_MEMBER) &&
             (MDALLOCG(sptr) || PTROFFG(sptr))) {
    retval = TRUE;
  }
  return retval;
} /* needs_linearization */

static LOGICAL linearize_any;

static void
_linearize(int ast, int *dummy)
{
  /* At an A_SUBSCR?  Should it be linearized? */
  if (A_TYPEG(ast) == A_SUBSCR && A_SHAPEG(ast) == 0) {
    int lop, sptr;

    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_ID) {
      sptr = A_SPTRG(lop);
    } else if (A_TYPEG(lop) == A_MEM) {
      sptr = A_SPTRG(A_MEMG(lop));
    } else {
      return;
    }

    if (needs_linearization(sptr)) {
      /* replace the subscript by the linearized subscripts */
      int asd, ndim, sdsc, ss, subscr[1], dtype, eldtype, newast;
      lop = ast_rewrite(lop);
      linearize_any = TRUE;
      asd = A_ASDG(ast);
      ndim = ASD_NDIM(asd);
      sdsc = SDSCG(sptr);
      dtype = DTYPEG(sptr);
      eldtype = DDTG(dtype);
      if (sdsc && !NODESCG(sptr)) {
        int i, simple;
        if (!POINTERG(sptr) && SCG(sptr) != SC_DUMMY) {
          simple = 1;
        } else {
          simple = 0;
        }
        ss = check_member(lop, get_xbase(sdsc));
        for (i = 0; i < ndim; ++i) {
          int s, stride;
          s = ASD_SUBS(asd, i);
          s = ast_rewrite(s);
          if (XBIT(58, 0x22) && !POINTERG(sptr)) {
            int lw;
            lw = ADD_LWAST(dtype, i);
            if (lw) {
              lw = ast_rewrite(lw);
              lw = mk_binop(OP_SUB, lw, astb.i1, DT_INT);
              s = mk_binop(OP_SUB, s, lw, DT_INT);
            }
          }
          if (i > 0 || !simple) {
            stride = check_member(lop, get_local_multiplier(sdsc, i));
            s = mk_binop(OP_MUL, s, stride, DT_INT);
          }

          if (ss == 0) {
            ss = s;
          } else {
            ss = mk_binop(OP_ADD, ss, s, DT_INT);
          }
        }
      } else {
        int dsym, ddtype, i;
        dsym = DESCRG(sptr);
        if (dsym) {
          ddtype = DTYPEG(dsym);
          if (DTY(ddtype) == TY_ARRAY) {
            dtype = ddtype;
          }
        }
        ss = 0;
        for (i = ndim; i > 0; --i) {
          int s, lw;
          lw = ADD_LWAST(dtype, i - 1);
          lw = ast_rewrite(lw);
          if (lw == 0) {
            lw = astb.i1;
          }
          if (i < ndim && ss != 0) {
            int up, stride;
            up = ADD_UPAST(dtype, i - 1);
            if (up == 0) {
              up = astb.i1;
            } else {
              up = ast_rewrite(up);
            }
            if (up == lw) {
              stride = astb.i1;
            } else if (lw == astb.i1) {
              stride = up;
            } else {
              stride = mk_binop(OP_SUB, up, lw, DT_INT);
              stride = mk_binop(OP_ADD, stride, astb.i1, DT_INT);
            }
            if (stride != astb.i1) {
              ss = mk_binop(OP_MUL, ss, stride, DT_INT);
            }
          }
          s = ASD_SUBS(asd, i - 1);
          s = ast_rewrite(s);
          if (ss == 0) {
            ss = s;
          } else {
            ss = mk_binop(OP_ADD, ss, s, DT_INT);
          }
          if (lw != astb.i0) {
            ss = mk_binop(OP_SUB, ss, lw, DT_INT);
          }
        }
        ss = mk_binop(OP_ADD, ss, astb.i1, DT_INT);
      }
      if (use_offset(sptr)) {
        /* add in the offset variable */
        int off;
        if ((STYPEG(sptr) != ST_MEMBER || POINTERG(sptr)) && PTROFFG(sptr)) {
          off = check_member(lop, mk_id(PTROFFG(sptr)));
        } else if (MIDNUMG(sptr)) {
          off = check_member(lop, mk_id(MIDNUMG(sptr)));
        } else {
          off = astb.i1;
        }
        ss = mk_binop(OP_ADD, ss, off, DT_INT);
        ss = mk_binop(OP_SUB, ss, astb.i1, DT_INT);
      }
      subscr[0] = ss;
      newast = mk_subscr(lop, subscr, 1, eldtype);
      ast_replace(ast, newast);
    }
  } else if (A_TYPEG(ast) == A_INTR) {
    int arg0, argcnt, argt, argtnew, i, diff, parent;
    switch (A_OPTYPEG(ast)) {
    case I_LBOUND:
    case I_UBOUND:
    case I_SIZE:
    case I_ALLOCATED:
      /* leave first argument as is, take the second argument */
      argt = A_ARGSG(ast);
      arg0 = ARGT_ARG(argt, 0);
      if (A_TYPEG(arg0) == A_MEM) {
        parent = ast_rewrite(A_PARENTG(arg0));
        diff = 0;
        if (parent != A_PARENTG(arg0)) {
          arg0 = mk_member(parent, A_MEMG(arg0), A_DTYPEG(arg0));
          ++diff;
        }
      }
      argcnt = A_ARGCNTG(ast);
      argtnew = mk_argt(argcnt);
      ARGT_ARG(argtnew, 0) = arg0;
      for (i = 1; i < argcnt; ++i) {
        ARGT_ARG(argtnew, i) = ast_rewrite(ARGT_ARG(argt, i));
        if (ARGT_ARG(argtnew, i) != ARGT_ARG(argt, i))
          ++diff;
      }
      if (!diff) {
        unmk_argt(argcnt);
        ast_replace(ast, ast);
      } else {
        int newast;
        newast = mk_func_node(A_TYPEG(ast), A_LOPG(ast), argcnt, argtnew);
        A_OPTYPEP(newast, A_OPTYPEG(ast));
        A_SHAPEP(newast, A_SHAPEG(ast));
        A_DTYPEP(newast, A_DTYPEG(ast));
        ast_replace(ast, newast);
      }
      break;
    }
  }
} /* _linearize */

static void
_linearize_all(int ast)
{
  int dummy = 0;
  ast_traverse(ast, NULL, _linearize, &dummy);
} /* _linearize_all */

static void
_linearize_sub(int ast)
{
  int lop, asd, i;
  switch (A_TYPEG(ast)) {
  case A_ID:
    break;
  case A_SUBSCR:
    /* look at subscripts, look at parent */
    asd = A_ASDG(ast);
    for (i = 0; i < ASD_NDIM(asd); ++i) {
      _linearize_all(ASD_SUBS(asd, i));
    }
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_MEM) {
      _linearize_all(A_PARENTG(lop));
    }
    break;
  case A_MEM:
    _linearize_all(A_PARENTG(ast));
    break;
  default:
    _linearize_all(ast);
    break;
  }
} /* _linearize_sub */

static void
_linearize_func(int ast, int *dummy)
{
  int argcnt, args, i, dont;
  int paramct, dpdsc, sptr, param;
  dont = -1;
  args = A_ARGSG(ast);
  switch (A_TYPEG(ast)) {
  case A_CALL:
  case A_FUNC:
  case A_ICALL:
    switch (A_OPTYPEG(ast)) {
    case I_NULLIFY:
      return;
    case I_COPYIN:
      if (XBIT(57, 0x80)) {
        int arg2, arg4;
        arg2 = ARGT_ARG(args, 2);
        arg4 = ARGT_ARG(args, 4);
        if (arg2 == arg4) {
          dont = 4;
        } else if (A_TYPEG(arg2) == A_SUBSCR && A_LOPG(arg2) == arg4) {
          dont = 4;
        }
      }
      break;
    case I_COPYOUT:
      if (XBIT(57, 0x80)) {
        int arg0, arg1;
        arg0 = ARGT_ARG(args, 0);
        arg1 = ARGT_ARG(args, 1);
        if (arg0 == arg1) {
          dont = 0;
        } else if (A_TYPEG(arg1) == A_SUBSCR && A_LOPG(arg1) == arg0) {
          dont = 0;
        }
      }
      break;
    case I_PTR2_ASSIGN:
      dont = 0;
      break;
    case I_PTR_COPYIN:
      dont = 3;
      break;
    case I_PTR_COPYOUT:
      dont = 0;
      break;
    }
    break;
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_SIZE:
    case I_LBOUND:
    case I_UBOUND:
    case I_PRESENT:
      return;
    }
    break;
  default:
    return;
  }
  argcnt = A_ARGCNTG(ast);
  sptr = A_SPTRG(A_LOPG(ast));
  if (STYPEG(sptr) == ST_PROC) {
    dpdsc = DPDSCG(sptr);
    paramct = PARAMCTG(sptr);
  } else {
    dpdsc = 0;
    paramct = 0;
  }
  for (i = 0; i < argcnt; ++i) {
    int arg, sptr;
    if (i == dont)
      continue;
    arg = ARGT_ARG(args, i);
    if (arg != 0) {
      param = 0;
      if (i < paramct && dpdsc) {
        param = aux.dpdsc_base[dpdsc + i];
      }
      switch (A_TYPEG(arg)) {
      case A_ID:
        sptr = A_SPTRG(arg);
        break;
      case A_MEM:
        sptr = A_SPTRG(A_MEMG(arg));
        /* see if remove_distributed_member will fix this */
        if (DTY(DTYPEG(sptr)) != TY_ARRAY && /* scalar */
            XBIT(70, 0x08) && /*remove_distributed_member is called*/
            ((POINTERG(sptr) && !F90POINTERG(sptr)) || ALIGNG(sptr)))
          /* and will replace this with a temp */
          continue;
        break;
      default:
        continue;
      }
      if (needs_linearization(sptr) && use_offset(sptr)) {
        int subscr[7];
        if (param && POINTERG(param)) {
          subscr[0] = astb.i1;
        } else if ((STYPEG(sptr) != ST_MEMBER || POINTERG(sptr)) &&
                   PTROFFG(sptr)) {
          subscr[0] = check_member(arg, mk_id(PTROFFG(sptr)));
        } else if (MIDNUMG(sptr)) {
          subscr[0] = check_member(arg, mk_id(MIDNUMG(sptr)));
        } else {
          subscr[0] = astb.i1;
        }
        ARGT_ARG(args, i) = mk_subscr(arg, subscr, 1, DDTG(DTYPEG(sptr)));
      }
    }
  }
} /* _linearize_func */

void
linearize_arrays(void)
{
  int std;
  int dummy = 0;
  deferred_to_pointer();
  /* linearize all subscripts */
  for (std = STD_NEXT(0); std; std = STD_NEXT(std)) {
    int ast;
    linearize_any = FALSE;
    ast = STD_AST(std);
    ast_visit(1, 1);
    switch (A_TYPEG(ast)) {
    case A_ALLOC:
      /* for ALLOCATEs, don't modify the allocate target directly */
      if (A_LOPG(ast) != 0) {
        _linearize_all(A_LOPG(ast));
      }
      if (A_DESTG(ast) != 0) {
        _linearize_all(A_DESTG(ast));
      }
      if (A_M3G(ast) != 0) {
        _linearize_all(A_M3G(ast));
      }
      if (A_STARTG(ast) != 0) {
        _linearize_all(A_STARTG(ast));
      }
      _linearize_sub(A_SRCG(ast));
      break;
    case A_REDIM: /* skip REDIM statements */
      break;
    default:
      _linearize_all(ast);
      break;
    }
    if (linearize_any) {
      ast = ast_rewrite(ast);
      STD_AST(std) = ast;
    }
    ast_unvisit();
    ast_visit(1, 1);
    ast_traverse(ast, NULL, _linearize_func, &dummy);
    ast_unvisit();
  }
} /* linearize_arrays */

/*
 * head of linked list of DEFs in each STD
 */
static int *stddeflist;
/*
 * head of linked list of DEFs in each LOOP
 */
static int *loopdeflist;

typedef struct syminfostruct {
  int loop, defs;
} syminfostruct;

static syminfostruct *syminfo;

static int clean;
static int always_executed;
static int chk_assign;
static int chk_subscr;

/*
 * set clean=0 and return immediately (with TRUE value) if
 *  we find a symbol which was modified in this loop
 *  we find an operation that is not clean:
 *   user function call
 *   divide
 *   non-integer multiply
 *
 */
static LOGICAL
_check_clean(int ast, int *pl)
{
  int l, o, sptr;
  int asd, i;

  l = *pl;
  switch (A_TYPEG(ast)) {
  case A_ID:
    sptr = A_SPTRG(ast);
    if (syminfo[sptr].loop == l) {
      /* must have been a def in this loop */
      clean = 0;
    } else if (SCG(sptr) == SC_BASED && MIDNUMG(sptr) &&
               syminfo[MIDNUMG(sptr)].loop == l) {
      /* must have been a def in this loop */
      clean = 0;
    } else if (SCG(sptr) == SC_BASED && !always_executed) {
      /* pointer may be null */
      clean = 0;
    } else if (POINTERG(sptr) && !always_executed) {
      /* pointer may be null */
      clean = 0;
    } else if (chk_assign && chk_subscr && !always_executed) {
      clean = 0;
    } else if (LP_CALLFG(l)) {
      /*
       * The LP_CALLFG check must be last; need to check 'sptr' as above.
       * if there is a call in the loop, and this is a COMMON symbol, unclean
       */
      if (SCG(sptr) == SC_CMBLK || (SCG(sptr) == SC_BASED && MIDNUMG(sptr) &&
                                    SCG(MIDNUMG(sptr)) == SC_CMBLK)) {
        clean = 0;
      }

      if (ALLOCDESCG(sptr)) {
        clean = 0;
      }
    }
    break;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    chk_subscr = 1;
    for (i = 0; i < (int)ASD_NDIM(asd); i++) {
      ast_traverse((int)ASD_SUBS(asd, i), _check_clean, NULL, pl);
      if (clean == 0)
        break;
    }
    chk_subscr = 0;
    break;
  case A_BINOP:
    o = A_OPTYPEG(ast);
    if (o == OP_DIV) {
      clean = 0;
    } else if (o == OP_MUL) {
      int d;
      d = A_DTYPEG(ast);
      if (!DT_ISINT(d)) {
        clean = 0;
      }
    }
    break;
  case A_FUNC:
  case A_CALL:
    clean = 0;
    break;
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_RAN:
    case I_RANDOM_NUMBER:
    case I_RANDOM_SEED:
      clean = 0;
      break;
    }
    break;
  }
  if (clean == 0)
    return TRUE;
  return FALSE;
} /* _check_clean */

/*
 * float a statement out of a loop
 */
static void
sfloat_stmt(int std, int fg, int l)
{
  int next, prev, head, prehead;
#if DEBUG
  if (DBGBIT(43, 0x800)) {
    fprintf(gbl.dbgfil, "FLOAT std:%d out of fnode:%d in loop:%d\n", std, fg,
            l);
  }
#endif

  /* remove stmt from std list */
  next = STD_NEXT(std);
  prev = STD_PREV(std);

  STD_PREV(next) = prev;
  STD_NEXT(prev) = next;

  /* remove stmt from fg's statement list */
  if (std == FG_STDFIRST(fg)) {
    if (std != FG_STDLAST(fg)) {
      FG_STDFIRST(fg) = next;
    } else {
      /* we've moved the only statement out */
      FG_STDFIRST(fg) = 0;
      FG_STDLAST(fg) = 0;
    }
  } else if (std == FG_STDLAST(fg)) {
    FG_STDLAST(fg) = prev;
  }

  /* find FG node into which to insert the statement */
  head = LP_HEAD(l);
  prehead = FG_LPREV(head);
  STD_FG(std) = prehead;

  if (FG_STDFIRST(prehead) == 0) {
    FG_STDFIRST(prehead) = std;
  }
  FG_STDLAST(prehead) = std;

  do {
    /* should iterate only once, DO is the top of the loop */
    next = FG_STDFIRST(head);
    head = FG_LNEXT(head);
  } while (next == 0);

  prev = STD_PREV(next);
  STD_NEXT(prev) = std;
  STD_PREV(next) = std;
  STD_NEXT(std) = next;
  STD_PREV(std) = prev;
} /* sfloat_stmt */

/*
 * move a statement out of a loop downward
 *
 */
static void
sdrop_stmt(int std, int fg, int l)
{
  int next, prev;
#if DEBUG
  if (DBGBIT(43, 0x800)) {
    fprintf(gbl.dbgfil, "DROP2 std:%d out of fnode:%d in loop:%d\n", std, fg,
            l);
  }
#endif

  /* remove stmt from std list */
  next = STD_NEXT(std);
  prev = STD_PREV(std);

  STD_PREV(next) = prev;
  STD_NEXT(prev) = next;

  /* remove stmt from fg's statement list */
  if (std == FG_STDFIRST(fg)) {
    if (std != FG_STDLAST(fg)) {
      FG_STDFIRST(fg) = next;
    } else {
      /* we've moved the only statement out */
      FG_STDFIRST(fg) = 0;
      FG_STDLAST(fg) = 0;
    }
  }
  if (std == FG_STDLAST(fg)) {
    FG_STDLAST(fg) = prev;
  }

  STD_NEXT(std) = 0;
  STD_PREV(std) = 0;
  /* put new std at the end of the list */
  if (LP_DSTDF(l)) {
    int tstd = LP_DSTDF(l);
    while (STD_NEXT(tstd)) {
      tstd = STD_NEXT(tstd);
    }
    STD_NEXT(tstd) = std;
    STD_PREV(std) = tstd;

  } else {
    LP_DSTDF(l) = std;
  }
}

static void
sfloat_stmt2(int std, int fg, int l)
{
  int next, prev;
#if DEBUG
  if (DBGBIT(43, 0x800)) {
    fprintf(gbl.dbgfil, "FLOAT2 std:%d out of fnode:%d in loop:%d\n", std, fg,
            l);
  }
#endif

  /* remove stmt from std list */
  next = STD_NEXT(std);
  prev = STD_PREV(std);

  STD_PREV(next) = prev;
  STD_NEXT(prev) = next;

  /* remove stmt from fg's statement list */
  if (std == FG_STDFIRST(fg)) {
    if (std != FG_STDLAST(fg)) {
      FG_STDFIRST(fg) = next;
    } else {
      /* we've moved the only statement out */
      FG_STDFIRST(fg) = 0;
      FG_STDLAST(fg) = 0;
    }
  } else if (std == FG_STDLAST(fg)) {
    FG_STDLAST(fg) = prev;
  }
  STD_NEXT(std) = 0;
  STD_PREV(std) = 0;
  /* append new std at the end of the list */
  if (LP_HSTDF(l)) {
    int tstd = LP_HSTDF(l);
    while (STD_NEXT(tstd)) {
      tstd = STD_NEXT(tstd);
    }
    STD_NEXT(tstd) = std;
    STD_PREV(std) = tstd;
  } else {
    LP_HSTDF(l) = std;
  }
}

void
hoist_stmt(int std, int fg, int l)
{
  if (STD_VISIT(std))
    return;
  /* don't do multiple exits  not yet */
  if (LP_MEXITS(l))
    return;

  STD_VISIT(std) = 1;

  if (is_dealloc_std(std))
    sdrop_stmt(std, STD_FG(std), l);
  else
    sfloat_stmt2(std, STD_FG(std), l);
}

void
restore_hoist_stmt(int lp)
{
  int next, prev, tail, head, laststd, posttail, prehead, tstd;

  int std = LP_HSTDF(lp);
  if (std) {
    laststd = std;
    STD_VISIT(std) = 0;
    while (STD_NEXT(laststd)) {
      laststd = STD_NEXT(laststd);
      STD_VISIT(laststd) = 0;
    }
    /* find FG node into which to insert the statement */
    head = LP_HEAD(lp);
    prehead = FG_LPREV(head);
    STD_FG(std) = prehead;

    if (FG_STDFIRST(prehead) == 0) {
      FG_STDFIRST(prehead) = std;
    }
    FG_STDLAST(prehead) = laststd;

    do {
      /* should iterate only once, DO is the top of the loop */
      next = FG_STDFIRST(head);
      head = FG_LNEXT(head);
    } while (next == 0);

    prev = STD_PREV(next);
    STD_NEXT(prev) = std;
    STD_PREV(next) = laststd;
    STD_NEXT(laststd) = next;
    STD_PREV(std) = prev;
  }

  std = LP_DSTDF(lp);
  if (std) {
    STD_VISIT(std) = 0;
    laststd = std;
    while (STD_NEXT(laststd)) {
      laststd = STD_NEXT(laststd);
    }

    /* find FG node into which to insert the statement */
    tail = LP_TAIL(lp);
    posttail = FG_LNEXT(tail);
    next = FG_STDFIRST(posttail);
    while (next == 0) {
      posttail = FG_LNEXT(posttail);
      next = FG_STDFIRST(posttail);
    }

    for (tstd = std; tstd; tstd = STD_NEXT(tstd)) {
      STD_FG(tstd) = posttail;
      STD_VISIT(tstd) = 0;
    }

    prev = STD_PREV(next);
    STD_PREV(std) = prev;
    STD_NEXT(prev) = std;
    STD_PREV(next) = laststd;
    STD_NEXT(laststd) = next;
  }
}

/*
 * record the def of a symbol; also, record any equivalenced defs.
 */
static void
add_def_syminfo(int sptr, int l)
{
  int socptr;
  int ss;

  if (syminfo[sptr].loop != l) {
    syminfo[sptr].loop = l;
    syminfo[sptr].defs = 0;
  }
  ++syminfo[sptr].defs;

  for (socptr = SOCPTRG(sptr); socptr; socptr = SOC_NEXT(socptr)) {
    ss = SOC_SPTR(socptr);
    if (syminfo[ss].loop != l) {
      syminfo[ss].loop = l;
      syminfo[ss].defs = 0;
    }
    ++syminfo[ss].defs;
  }
}

/*
 * given a loop, look at the loop header node.
 * it should have only one non-loop predecessor, which should have only
 * one successor, the loop header.  That node is then a valid preheader.
 */
static LOGICAL
have_preheader(int l)
{
  int h, n, ph, v;
  PSI_P pred;
  PSI_P succ;
  h = LP_HEAD(l);
  n = 0;
  ph = 0;
  for (pred = FG_PRED(h); pred; pred = PSI_NEXT(pred)) {
    v = PSI_NODE(pred);
    if (FG_LOOP(v) != l) {
      ++n;
      if (n > 1)
        return FALSE;
      ph = v;
    }
  }
  if (n != 1)
    return FALSE;
  succ = FG_SUCC(ph);
  if (succ == PSI_P_NULL || PSI_NEXT(succ))
    return FALSE;
  if (PSI_NODE(succ) != h)
    return FALSE;
  /* only one predecessor outside the loop, it has only one successor */
  return TRUE;
} /* have_preheader */

/*
 * if 'l' is an inner loop, look at the nodes in the loop
 * look at assignments and section descriptor function calls in those nodes
 * if this is the only assignment to the LHS and the RHS is loop invariant,
 * (for section descriptor functions, LHS is 1st argument, RHS is other args)
 * then float the statement out of the loop.
 * if the node is not control-equivalent to the loop entry, then require
 * the LHS to be a compiler temp, and the RHS to be 'safe'
 *  safe means no faults (no divides unless denominator is constant)
 */
static void
sfloat(int l)
{
  int lc, fg, std, ast, firstd, lastd;
  /* inner loops first */
  for (lc = LP_CHILD(l); lc; lc = LP_SIBLING(lc)) {
    sfloat(lc);
  }

  /* count how many defs of each variable in the loop */
  /* look at flow graph nodes in this loop */
  for (fg = LP_FG(l); fg; fg = FG_NEXT(fg)) {
    /* look at statements in this flow graph node */
    int std, stdlast;
    stdlast = FG_STDLAST(fg);
    for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
      int d;
      for (d = stddeflist[std]; d; d = DEF_NEXT(d)) {
        int nm;
        for (nm = DEF_NM(d); nm; nm = NME_NM(nm)) {
          int sptr;
          sptr = NME_SYM(nm);
          if (sptr > NOSYM) {
            add_def_syminfo(sptr, l);
          }
        }
      }
      if (std == stdlast)
        break;
    }
  }
  for (lc = LP_CHILD(l); lc; lc = LP_SIBLING(lc)) {
    int d;
    for (d = loopdeflist[lc]; d; d = DEF_NEXT(d)) {
      int nm;
      for (nm = DEF_NM(d); nm; nm = NME_NM(nm)) {
        int sptr;
        sptr = NME_SYM(nm);
        if (sptr > NOSYM) {
          add_def_syminfo(sptr, l);
        }
      }
    }
  }

  /* focus on DO loops */
  fg = LP_HEAD(l);
  std = FG_STDFIRST(fg);
  ast = STD_AST(std);
  if (A_TYPEG(ast) == A_DO || (XBIT(70, 0x800) && have_preheader(l))) {
    /* look at flow graph nodes in this loop */
    for (fg = LP_FG(l); fg; fg = FG_NEXT(fg)) {
      /* look at statements in this flow graph node */
      int std, nextstd, stdlast;
      stdlast = FG_STDLAST(fg);
      for (std = FG_STDFIRST(fg); std; std = nextstd) {
        /* is this an assignment that can be floated out,
         * or is this a template call that can be floated out */
        int ast, lhs, rhs, sptr, funcast, ll, nme;
        nextstd = STD_NEXT(std);
        ast = STD_AST(std);
        switch (A_TYPEG(ast)) {
        case A_ASN:
          lhs = A_DESTG(ast);
          rhs = A_SRCG(ast);
          if (A_TYPEG(lhs) != A_ID)
            break;
          sptr = A_SPTRG(lhs);
          if (SCG(sptr) != SC_LOCAL || (gbl.internal == 1 && LP_CALLFG(l)))
            break;
          if (gbl.internal > 1 && !INTERNALG(sptr) && LP_CALLFG(l))
            break;
          /*
           *  must be unconditional or dead after the loop.
           *  the only definition of this symbol in the loop,
           *  free of side effects or faults,
           *  loop-invariant RHS
           */
          nme = add_arrnme(NT_VAR, sptr, 0, (INT)0, 0, FALSE);
          if ((!FG_CTLEQUIV(fg) && is_live_out(nme, l)) || is_live_in(nme, l))
            break;
          if (syminfo[sptr].loop != l || syminfo[sptr].defs != 1)
            break;
          /* loop invariant, side-effect-free RHS */
          clean = 1;
          always_executed = FG_CTLEQUIV(fg);
          ll = l;
          chk_assign = 1;
          ast_visit(1, 1);
          ast_traverse(rhs, _check_clean, NULL, &ll);
          ast_unvisit();
          chk_assign = 0;
          if (clean) {
            /* move this statement to the loop preheader. */
            sfloat_stmt(std, fg, l);
          }
          break;
        case A_CALL:
          funcast = A_LOPG(ast);
          if (A_TYPEG(funcast) == A_ID &&
              getF90TmplSectRtn(SYMNAME(A_SPTRG(funcast)))) {
            int argcnt, args, i;
            argcnt = A_ARGCNTG(ast);
            args = A_ARGSG(ast);
            lhs = ARGT_ARG(args, 0);
            if (A_TYPEG(lhs) != A_ID)
              break;
            sptr = A_SPTRG(lhs);
            if (SCG(sptr) != SC_LOCAL || (gbl.internal == 1 && LP_CALLFG(l)))
              break;
            if (gbl.internal > 1 && !INTERNALG(sptr) && LP_CALLFG(l))
              break;
            /* must be unconditional or a compiler temp array for which this
             *  is a section descriptor,
             *  the only definition of this descriptor in the loop,
             *  free of side effects or faults,
             *  loop-invariant arguments
             */
            nme = add_arrnme(NT_VAR, sptr, 0, (INT)0, 0, FALSE);
            if (!FG_CTLEQUIV(fg) && (is_live_out(nme, l) || is_live_in(nme, l)))
              /*if( PUREG(sptr) && !FG_CTLEQUIV(fg) )*/
              break;
            if (syminfo[sptr].loop != l || syminfo[sptr].defs != 1)
              break;
            clean = 1;
            ll = l;
            /*always_executed = FG_CTLEQUIV(fg);*/
            /* for now, just assume calls are always executed;
             * other checks for side-effects are sufficient ...?
             */
            always_executed = 1;
            ast_visit(1, 1);
            for (i = 1; clean && i < argcnt; ++i) {
              rhs = ARGT_ARG(args, i);
              ast_traverse(rhs, _check_clean, NULL, &ll);
            }
            ast_unvisit();
            if (clean) {
              /* move this statement to the loop preheader. */
              sfloat_stmt(std, fg, l);
            }
          }
          break;
        }
        if (std == stdlast)
          break;
      }
    }
  }

  /* put the DEFs from the statements in the loop
   * onto the list of DEFs for this loop */
  /* look at flow graph nodes in this loop */
  firstd = 0;
  lastd = 0;
  for (fg = LP_FG(l); fg; fg = FG_NEXT(fg)) {
    /* look at statements in this flow graph node */
    int std;
    for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
      int d;
      d = stddeflist[std];
      if (d) {
        if (firstd == 0) {
          firstd = d;
        } else {
          DEF_NEXT(lastd) = d;
        }
        for (; d; d = DEF_NEXT(d)) {
          lastd = d;
        }
        /* here, lastd points to the end of the list */
        stddeflist[std] = 0; /* no vestigial pointers */
      }
      if (std == FG_STDLAST(fg))
        break;
    }
  }
  for (lc = LP_CHILD(l); lc; lc = LP_SIBLING(lc)) {
    int d;
    d = loopdeflist[lc];
    if (d) {
      if (firstd == 0) {
        firstd = d;
      } else {
        DEF_NEXT(lastd) = d;
      }
      for (; d; d = DEF_NEXT(d)) {
        lastd = d;
      }
      /* here, lastd points to the end of the list */
      loopdeflist[lc] = 0; /* no vestigial pointers */
    }
  }
  loopdeflist[l] = firstd;
} /* sfloat */

/*
 * look for section descriptor manipulations,
 *  such as RTE_template, pghpf_sect calls,
 * float these out of loops if possible
 */
void
sectfloat(void)
{
  int savex, l, fg, nm, s;
  optshrd_init();
  induction_init();
  optshrd_finit();
  savex = flg.x[6]; /* disable flow graph changes here */
  flg.x[6] |= 0x80000000;
  /* build the flowgraph for the function */
  flowgraph();
  postdominators();
  /* build the loop data structure */
  findlooptopsort();
  reorderloops();
  /* do flow analysis on the loops */
  flow();

  /* find control-equivalent nodes in loops */
  for (fg = 1; fg < opt.num_nodes; ++fg) {
    l = FG_LOOP(fg);
    if (l) {
      int head;
      head = LP_HEAD(l);
      if (fg == head) {
        /* this IS the loop head */
        FG_CTLEQUIV(fg) = 1;
      } else {
        int dom;
        dom = FG_DOM(fg);
        if (dom && FG_LOOP(dom) == l && FG_CTLEQUIV(dom) &&
            FG_PDOM(dom) == fg) {
          /* simple case, control equivalent to a control equivalent node */
          FG_CTLEQUIV(fg) = 1;
        } else if (is_dominator(head, fg) && is_post_dominator(fg, head)) {
          /* harder case; see if LP_HEAD dominates this node and this
           * node post-dominates LP_HEAD */
          FG_CTLEQUIV(fg) = 1;
        }
      }
    }
  }

#if DEBUG
  if (DBGBIT(56, 2)) {
    dumpfgraph();
    dumploops();
    dumpnmes();
    dumpdefs();
    dumpuses();
  }
#endif
  /* unlink DEF_NEXT list from NME, link into a list based on STD */
  NEW(stddeflist, int, astb.std.stg_size);
  BZERO(stddeflist, int, astb.std.stg_size);
  NEW(loopdeflist, int, opt.nloops + 1);
  BZERO(loopdeflist, int, opt.nloops + 1);
  NEW(syminfo, syminfostruct, stb.stg_avail);
  BZERO(syminfo, syminfostruct, stb.stg_avail);
  for (nm = 1; nm < nmeb.stg_avail; ++nm) {
    int d, nextd;
    for (d = NME_DEF(nm); d; d = nextd) {
      int std;
      nextd = DEF_NEXT(d);
      std = DEF_STD(d);
      DEF_NEXT(d) = stddeflist[std];
      stddeflist[std] = d;
    }
  }
  /* mark those section descriptor arrays that are
   * section descriptors for user symbols */
  for (s = stb.firstusym; s < stb.stg_avail; ++s) {
    switch (STYPEG(s)) {
    case ST_ARRAY:
    case ST_DESCRIPTOR:
    case ST_STRUCT:
    case ST_MEMBER:
      if (!CCSYMG(s) && !HCCSYMG(s)) {
        int sdsc;
        sdsc = SDSCG(s);
        if (sdsc) {
          PUREP(sdsc, 1);
        }
      }
      break;
    default:;
    }
  }
  for (l = LP_CHILD(0); l; l = LP_SIBLING(l)) {
    sfloat(l);
  }

  FREE(syminfo);
  FREE(loopdeflist);
  FREE(stddeflist);
  optshrd_fend();
  induction_end();
  optshrd_end();
  flg.x[6] = savex; /* disable flow graph changes here */
} /* sectfloat */
