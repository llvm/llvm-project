/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Fortran communications module
 */

#include "comm.h"
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
#include "extern.h"
#include "hpfutl.h"
#include "commopt.h"
#include "ccffinfo.h"
#include "dinit.h"
#include "fdirect.h"
#include "rte.h"
#include "rtlRtns.h"
#include "ilidir.h" /* for open_pragma, close_pragma */

struct cs_table {
  LOGICAL is_used_lhs;
};

static struct cs_table cs_table;

static void comm_init(void);
static void transform_ptr(int std, int ast);
static int normalize_forall_triplet(int std, int forall);
static void emit_overlap(int a);
static int emit_permute_section(int a, int std);
static int eliminate_extra_idx(int lhs, int a, int forall);
static int emit_copy_section(int a, int std);
static int canonical_conversion(int ast);
static void forall_dependency_scalarize(int std, int *std1, int *std2);
static LOGICAL is_use_lhs(int a, LOGICAL, LOGICAL, int);
static int emit_gatherx(int a, int std, LOGICAL opt);
static void fix_guard_forall(int std);
static void emit_sum_scatterx(int);
static void emit_scatterx(int);
static void emit_scatterx_gatherx(int std, int result, int array, int mask,
                                  int allocstd, int tempast0, int lhssec,
                                  int comm_type);
static void compute_permute(int lhs, int rhs, int list, int order[7]);
static int put_data(int permute[7], int no);
static LOGICAL is_permuted(int array, int per[7], int per1[7], int *nper1);
static int scalar_communication(int ast, int std);
static int tag_call_comm(int std, int forall);
static void call_comm(int cstd, int fstd, int forall);
static void insert_call_comm(int std, int forall);
static void put_call_comm(int cstd, int fstd, int forall);
static void shape_communication(int std, int forall);
static void shape_comm(int cstd, int fstd, int forall);
static int sequentialize_mask_call(int forall, int stdnext);
static int sequentialize_stmt_call(int forall, int stdnext);
static int sequentialize_call(int cstd, int stdnext, int forall);
#ifdef FLANG_COMM_UNUSED
static int gen_shape_comm(int arg, int forall, int std, int nomask);
static int reference_for_pure_temp(int sptr, int lhs, int arg, int forall);
#endif
static void init_pertbl(void);
static void free_pertbl(void);
static int get_pertbl(void);
static int copy_section_temp_before(int sptr, int rhs, int forall);
static CTYPE *getcyclic(void);
static void init_opt_tables(void);
static LOGICAL is_scatter(int std);
static void opt_overlap(void);
static int insert_forall_comm(int ast);
#ifdef FLANG_COMM_UNUSED
static int construct_list_for_pure(int arg, int mask, int list);
static LOGICAL is_pure_temp_too_large(int list, int arg);
static int handle_pure_temp_too_large(int expr, int std);
#endif
static int forall_2_sec(int a, int forall);
static int make_sec_ast(int arr, int std, int allocstd, int sectflag);
static int temp_copy_section(int std, int forall, int lhs, int rhs, int dty,
                             int *allocast);
static int temp_gatherx(int std, int forall, int lhs, int rhs, int dty,
                        int *allocast);
static int gatherx_temp_before(int sptr, int rhs, int forall);
static int simple_reference_for_temp(int sptr, int a, int forall);

/**
   \brief Finalize the phase and free allocated memory.
 */
void
comm_fini(void)
{
  TRANS_FREE(trans.subb);
  trans.subb.stg_base = NULL;
  TRANS_FREE(trans.arrb);
  trans.subb.stg_base = NULL;
  TRANS_FREE(trans.tdescb);
  trans.tdescb.stg_base = NULL;
  FREE(finfot.base);
  finfot.base = NULL;
  free_pertbl();
}

/**
   \brief Communication analyzer entry point.
 */
void
comm_analyze(void)
{
  int std, stdnext;
  int ast;
  int parallel_depth;
  int task_depth;
  int type;

  comm_init();
  init_region();
  parallel_depth = 0;
  task_depth = 0;
  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    gbl.lineno = STD_LINENO(std);
    if (STD_PURE(std))
      continue;
    if (STD_LOCAL(std) || pure_gbl.end_master_region != 0)
      pure_gbl.local_mode = 1; /* don't process for DO-INDEPENDENT */
    else
      pure_gbl.local_mode = 0;
    ast = STD_AST(std);
    switch (type = A_TYPEG(ast)) {
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
    default:
      break;
    }
    if (type == A_FORALL) {
      if (STD_LOCAL(std))
        continue; /* don't process for DO-INDEPENDENT */
      transform_forall(std, ast);
    } else if (type == A_ICALL && A_OPTYPEG(ast) == I_PTR2_ASSIGN)
      transform_ptr(std, ast);
    else
      transform_ast(std, ast);
    check_region(std);
  }
}

/**
   \brief Keep track of STD of endcritical or endmaster statement
 */
void
init_region(void)
{
  pure_gbl.end_master_region = 0;
  pure_gbl.end_critical_region = 0;
} /* init_region */

/**
   \brief Check a region is valid.
 */
void
check_region(int std)
{
  int ast = STD_AST(std);
  if (A_TYPEG(ast) == A_MASTER && pure_gbl.end_master_region == 0) {
    /* get endmaster ast */
    int endmasterast = A_LOPG(ast);
    pure_gbl.end_master_region = A_STDG(endmasterast);
    if (pure_gbl.end_critical_region == 0) {
      pure_gbl.end_critical_region = pure_gbl.end_master_region;
    }
  } else if (A_TYPEG(ast) == A_CRITICAL && pure_gbl.end_critical_region == 0) {
    /* get endcritical ast */
    int endcriticalast = A_LOPG(ast);
    pure_gbl.end_critical_region = A_STDG(endcriticalast);
  }
  if (pure_gbl.end_critical_region == std) {
    pure_gbl.end_critical_region = 0;
  }
  if (pure_gbl.end_master_region == std) {
    pure_gbl.end_master_region = 0;
  }
} /* check_region */

/**
   \brief Create mask statements for conditional expression ast and insert them
   after stdstart. Return the STD of the last statement added.
 */
int
insert_mask(int ast, int stdstart)
{
  int std;
  int aststmt;

  if (A_TYPEG(ast) == A_BINOP && A_OPTYPEG(ast) == OP_SCAND) {
    std = insert_mask(A_LOPG(ast), stdstart);
    std = insert_mask(A_ROPG(ast), std);
    return std;
  }
  aststmt = mk_stmt(A_IFTHEN, 0);
  A_IFEXPRP(aststmt, ast);
  std = add_stmt_after(aststmt, stdstart);
  return std;
}

/**
   \brief Create ENDIF statements corresponding to conditional statements
   emitted
   for mask expression ast. Insert the ENDIFs after stdstart.
   Return the STD of the last statement added.
 */
int
insert_endmask(int ast, int stdstart)
{
  int std;
  int aststmt;

  if (A_TYPEG(ast) == A_BINOP && A_OPTYPEG(ast) == OP_SCAND) {
    std = insert_endmask(A_LOPG(ast), stdstart);
    std = insert_endmask(A_ROPG(ast), std);
    return std;
  }
  aststmt = mk_stmt(A_ENDIF, 0);
  std = add_stmt_after(aststmt, stdstart);
  return std;
}

/**
   \brief Dump compiler internal information for the communication analyzer.
 */
void
report_comm(int std, int cause)
{
  int ln;

  if (!XBIT(0, 2))
    return;

  if (STD_MINFO(std))
    return;

  STD_MINFO(std) = 1;

  ln = STD_LINENO(std);
  switch (cause) {
  case CANONICAL_CAUSE:
    ccff_info(MSGFTN, "FTN001", 1, ln, "Forall scalarized", NULL);
    break;
  case INTRINSIC_CAUSE:
    ccff_info(MSGFTN, "FTN002", 1, ln,
              "Forall scalarized: transformational intrinsic call", NULL);
    break;
  case UGLYCOMM_CAUSE:
    ccff_info(MSGFTN, "FTN003", 1, ln,
              "Forall scalarized: complex communication", NULL);
    break;
  case DEPENDENCY_CAUSE:
    ccff_info(MSGFTN, "FTN004", 1, ln, "Forall split in two: data dependence",
              NULL);
    break;
  case GETSCALAR_CAUSE:
    ccff_info(MSGFTN, "FTN005", 1, ln, "Expensive scalar communication", NULL);
    break;
  case COPYSCALAR_CAUSE:
    ccff_info(MSGFTN, "FTN006", 1, ln, "Expensive scalar copy communication",
              NULL);
    break;
  case COPYSECTION_CAUSE:
    ccff_info(MSGFTN, "FTN007", 1, ln,
              "Expensive all-to-all section copy communication", NULL);
    break;
  case PURECOMM_CAUSE:
    ccff_info(MSGFTN, "FTN008", 1, ln,
              "Communication generated: Forall pure arguments", NULL);
    break;
  case UGLYPURE_CAUSE:
    ccff_info(MSGFTN, "FTN009", 1, ln,
              "Forall scalarized: complex pure argument", NULL);
    break;
  case UGLYMASK_CAUSE:
    ccff_info(MSGFTN, "FTN010", 1, ln,
              "Forall scalarized: complex mask expression", NULL);
    break;
  case MANYRUNTIME_CAUSE:
    assert(A_TYPEG(STD_AST(std)) == A_FORALL, "report_comm: forall is expected",
           std, 2);
    ccff_info(MSGFTN, "FTN011", 1, ln, "Too many runtime calls", NULL);
    break;
  }
}

/**
   \brief Construct an AST to add the lower bound of dimension dim
   for array datatype dtyp to ast, and return the new AST.
 */
int
add_lbnd(int dtyp, int dim, int ast, int astmember)
{
  int astBnd = ADD_LWAST(dtyp, dim);
  int ast1;

  if (!astBnd || astBnd == astb.bnd.one)
    return ast;

  ast1 = mk_binop(OP_ADD, ast, check_member(astmember, astBnd), astb.bnd.dtype);
  ast1 = mk_binop(OP_SUB, ast1, astb.bnd.one, astb.bnd.dtype);
  return ast1;
}

/**
   \brief Construct an AST to subtract the lower bound of dimension dim
   for array datatype dtyp to ast, and return the new AST.
 */
int
sub_lbnd(int dtyp, int dim, int ast, int astmember)
{
  int astBnd = ADD_LWAST(dtyp, dim);
  int ast1;

  if (!astBnd || astBnd == astb.bnd.one)
    return ast;

  ast1 = mk_binop(OP_SUB, ast, check_member(astmember, astBnd), astb.bnd.dtype);
  ast1 = mk_binop(OP_ADD, ast1, astb.bnd.one, astb.bnd.dtype);
  return ast1;
}

/**
   \brief Return TRUE if the bounds of array sptr should be 1-based with
   respect to the runtime.
 */
LOGICAL
normalize_bounds(int sptr)
{
  if (STYPEG(sptr) != ST_ARRAY)
    return FALSE;
  return (XBIT(58, 0x22) && !POINTERG(sptr));
}

LOGICAL
is_same_number_of_idx(int dest, int src, int list)
{
  int count, count1;
  int asd;
  int j, ndim;

  count = 0;
  count1 = 0;

  /* dest */
  while (dest) {
    switch (A_TYPEG(dest)) {
    case A_ID:
      dest = 0;
      break;
    case A_SUBSTR:
      dest = A_LOPG(dest);
      break;
    case A_MEM:
      dest = A_PARENTG(dest);
      break;
    case A_SUBSCR:
      asd = A_ASDG(dest);
      ndim = ASD_NDIM(asd);

      for (j = 0; j < ndim; ++j) {
        if (search_forall_var(ASD_SUBS(asd, j), list))
          count++;
      }
      dest = A_LOPG(dest);
      break;
    default:
      dest = 0;
      break;
    }
  }
  while (src) {
    switch (A_TYPEG(src)) {
    case A_ID:
      src = 0;
      break;
    case A_SUBSTR:
      src = A_LOPG(src);
      break;
    case A_MEM:
      src = A_PARENTG(src);
      break;
    case A_SUBSCR:
      /* src */
      asd = A_ASDG(src);
      ndim = ASD_NDIM(asd);
      for (j = 0; j < ndim; ++j) {
        if (search_forall_var(ASD_SUBS(asd, j), list))
          count1++;
      }
      src = A_LOPG(src);
      break;
    default:
      src = 0;
      break;
    }
  }

  if (count1 == count)
    return TRUE;
  else
    return FALSE;
}

/**
   \brief This routine finds the dimension of sptr.

   It takes subscript `a(f(i),5,f(j))`. It eliminates scalar dimension.
   It makes an ast for reference sptr: `a(f(i),5,f(j)) --> sptr(f(i),f(j))`

   NOTE: This is always called after get_temp_forall(), which calls
   mk_forall_sptr().  The subscripts are not always as simple
   as `sptr(f(i),f(j))`, especially if the stride is not known.
   if the stride is not +1 or -1, the subscript will be normalized.
 */
int
reference_for_temp(int sptr, int a, int forall)
{
  int subs[7];
  int list;
  int i, ndim, k;
  int astnew, vector;

  list = A_LISTG(forall);
  ndim = 0;
  vector = 0;
  do {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      int asd, adim;
      asd = A_ASDG(a);
      adim = ASD_NDIM(asd);
      /* array will be referenced after communication as follows  */
      for (i = 0; i < adim; i++) {
        int ast;
        ast = ASD_SUBS(asd, i);
        if (XBIT(58, 0x20000)) {
          extern int constant_stride(int a, int *value);
          int c, stride, lw, up;
          if (A_TYPEG(ast) == A_TRIPLE) {
            lw = check_member(a, A_LBDG(ast));
            up = check_member(a, A_UPBDG(ast));
            c = constant_stride(A_STRIDEG(ast), &stride);
            if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
              stride = A_STRIDEG(ast);
              if (stride == 0)
                stride = astb.i1;
              up = mk_binop(OP_DIV, mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw,
                                                              stb.user.dt_int),
                                             stride, stb.user.dt_int),
                            stride, stb.user.dt_int);
              lw = astb.i1;
              subs[ndim] = mk_triple(lw, up, 0);
            } else if (c && stride == 1) {
              subs[ndim] = ast;
            } else if (c && stride == -1) {
              subs[ndim] = ast;
            } else {
              stride = A_STRIDEG(ast);
              if (stride == 0)
                stride = astb.i1;
              up = mk_binop(OP_DIV, mk_binop(OP_ADD, mk_binop(OP_SUB, up, lw,
                                                              stb.user.dt_int),
                                             stride, stb.user.dt_int),
                            stride, stb.user.dt_int);
              lw = astb.i1;
              subs[ndim] = mk_triple(lw, up, 0);
            }
            ++ndim;
            vector = 1;
          } else if (A_SHAPEG(ast)) {
            subs[ndim] = ast;
            ++ndim;
            vector = 1;
          } else if ((k = search_forall_var(ast, list))) {
            if (other_forall_var(ast, list, k))
              /*f2731*/
              subs[ndim] = ast;
            else {
              lw = A_LBDG(ASTLI_TRIPLE(k));
              up = A_UPBDG(ASTLI_TRIPLE(k));
              c = constant_stride(A_STRIDEG(ASTLI_TRIPLE(k)), &stride);
              if (flg.opt >= 2 && !XBIT(2, 0x400000)) {
                stride = A_STRIDEG(ASTLI_TRIPLE(k));
                if (stride == 0)
                  stride = astb.i1;
                subs[ndim] = mk_binop(
                    OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, mk_id(ASTLI_SPTR(k)), lw,
                                              stb.user.dt_int),
                             stride, stb.user.dt_int),
                    stride, stb.user.dt_int);
              } else if (c && stride == 1) {
                subs[ndim] = mk_id(ASTLI_SPTR(k));
              } else if (c && stride == -1) {
                subs[ndim] = mk_id(ASTLI_SPTR(k));
              } else {
                stride = A_STRIDEG(ASTLI_TRIPLE(k));
                if (stride == 0)
                  stride = astb.i1;
                subs[ndim] = mk_binop(
                    OP_DIV,
                    mk_binop(OP_ADD, mk_binop(OP_SUB, mk_id(ASTLI_SPTR(k)), lw,
                                              stb.user.dt_int),
                             stride, stb.user.dt_int),
                    stride, stb.user.dt_int);
              }
            }
            ++ndim;
          }
        } else if (A_TYPEG(ast) == A_TRIPLE || A_SHAPEG(ast)) {
          /* include this dimension */
          subs[ndim] = ast;
          ++ndim;
          vector = 1;
        } else if (search_forall_var(ASD_SUBS(asd, i), list)) {
          /* include this dimension */
          subs[ndim] = ast;
          ++ndim;
        }
      }
      a = A_LOPG(a);
    } else {
      interr("reference_for_temp: not subscr or member", a, 3);
    }
  } while (A_TYPEG(a) != A_ID);
  assert(ndim == rank_of_sym(sptr), "reference_for_temp: rank mismatched", sptr,
         4);
  if (vector) {
    astnew = mk_subscr(mk_id(sptr), subs, ndim, DTYPEG(sptr));
  } else {
    astnew = mk_subscr(mk_id(sptr), subs, ndim, DTY(DTYPEG(sptr) + 1));
  }
  return astnew;
}

/**
   \brief This routine a barrier statement in the barrier table.
 */
int
record_barrier(LOGICAL bBefore, int astStmt, int std)
{
  int i;
  int sptr;
  LITEMF *pl;

  switch (A_TYPEG(astStmt)) {
  case A_ASN:
    sptr = sym_of_ast(A_DESTG(astStmt));
    pl = clist();
    pl->item = sptr;
    break;
  case A_FORALL:
    sptr = sym_of_ast(A_DESTG(A_IFSTMTG(astStmt)));
    pl = clist();
    pl->item = sptr;
    break;
  default:
    return 0;
  }
  i = get_brtbl();
  brtbl.base[i].f1 = bBefore;
  brtbl.base[i].f2 = std;
  brtbl.base[i].f3 = pl;
  return i;
}

/**
   \brief This routine is to read distributed array element at forall by
   using get scalar primitive.
 */
int
emit_get_scalar(int a, int std)
{
  int lsptr, ld;
  int asd;
  int ndim;
  int temp, tempast;
  int ast;
  int commstd;

  if (STD_LOCAL(std))
    return a; /* don't process for DO-INDEPENDENT */
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  ld = dist_ast(a);
  if (ld == 0)
    return a;
  lsptr = memsym_of_ast(ld);
  if (!DISTG(lsptr) && !ALIGNG(lsptr))
    return a;

  /* It is distributed.  Create a temp to hold the value */
  temp = sym_get_scalar(SYMNAME(lsptr), "s", DTY(DTYPEG(lsptr) + 1));
  tempast = mk_id(temp);
  ast = new_node(A_HGETSCLR);
  A_SRCP(ast, a);
  A_DESTP(ast, tempast);
  if (DESCRG(lsptr)) {
    int lop;
    lop = check_member(a, mk_id(DESCRG(lsptr)));
    A_LOPP(ast, lop);
  }
  commstd = add_stmt_before(ast, std);
  A_STDP(ast, commstd);
  return replace_ast_subtree(a, ld, tempast);
}

/**
   <pre>
   Algorithm:
   * gather information about lhs array.
   * tag communications for rhs array.
   * optimize overlap_shift if there is same array shift.
   * optimize copy_section
   * convert to forall into block forall since owner computes rule distribution
     for cyclic require complicated statement insertion.
   * forall_gbl.s0....
   * forall_gbl.s1 forall(i=.
   * forall_gbl.s2   A(i)=
   * forall_gbl.s3 endforall
   * forall_gbl.s4, forall_gbl.s5 ...

   These variables are globals.
   * forall_gbl.s1 moves up.
   * forall_gbl.s4 moves down.
   </pre>
 */
void
forall_opt1(int ast)
{
  int std;
  int i, j;
  int nd;

  std = A_STDG(ast);
  if (A_OPT1G(ast))
    return;
  nd = mk_ftb();
  FT_NRT(nd) = 0;
  FT_RTL(nd) = clist();
  FT_NMCALL(nd) = 0;
  FT_MCALL(nd) = clist();
  FT_NSCALL(nd) = 0;
  FT_SCALL(nd) = clist();
  FT_NMGET(nd) = 0;
  FT_MGET(nd) = clist();
  FT_NSGET(nd) = 0;
  FT_SGET(nd) = clist();
  FT_NPCALL(nd) = 0;
  FT_PCALL(nd) = clist();
  FT_IGNORE(nd) = 0;
  FT_SECTL(nd) = 0;
  FT_CYCLIC(nd) = getcyclic();
  for (i = 0; i < 7; i++) {
    FT_NFUSE(nd, i) = 0;
    for (j = 0; j < MAXFUSE; j++)
      FT_FUSELP(nd, i, j) = 0;
  }
  FT_FUSED(nd) = 0;
  FT_HEADER(nd) = std;
  FT_BARR1(nd) = 0;
  FT_BARR2(nd) = 0;
  FT_FG(nd) = 0;
  A_OPT1P(ast, nd);
}

void
transform_forall(int std, int ast)
{
  int asn;
  int src, dest;
  int astnew;
  int test1, test2;

  comminfo.std = std;
  comminfo.usedstd = std;
  comminfo.forall = ast;
  trans.rhsbase = 0;

  init_opt_tables();
  forall_opt1(ast);

  if (pure_gbl.end_critical_region != 0) {
    scalarize(std, ast, TRUE);
    return;
  }

  shape_communication(std, ast);

  comminfo.std = std;
  comminfo.usedstd = std;
  comminfo.forall = ast;

  asn = A_IFSTMTG(ast);
  dest = scalar_communication(A_DESTG(asn), std);
  src = scalar_communication(A_SRCG(asn), std);
  A_DESTP(asn, dest);
  A_SRCP(asn, src);

  /* if the lhs is distributed, adjust the forall bounds; insert the
   * communication for the forall statement; adjust the rhs bounds
   */
  comminfo.mask_phase = 0;
  if (normalize_forall_triplet(std, ast) == 0) {
    report_comm(std, CANONICAL_CAUSE);
    scalarize(std, ast, TRUE);
    return;
  }

  if (is_scatter(std))
    return;

  if (canonical_conversion(ast) == 0) {
    report_comm(std, CANONICAL_CAUSE);
    scalarize(std, ast, TRUE);
    return;
  }

  asn = A_IFSTMTG(ast);
  if (process_lhs_sub(std, ast) == 0) {
    scalarize(std, ast, TRUE);
    return;
  }
  test1 = tag_forall_comm(A_SRCG(A_IFSTMTG(ast)));
  comminfo.mask_phase = 1;
  if (!comminfo.unstruct && A_IFEXPRG(ast))
    test2 = tag_forall_comm(A_IFEXPRG(ast));
  if (!comminfo.unstruct)
    test1 = tag_call_comm(std, ast);

  if (comminfo.unstruct) {
    report_comm(std, UGLYCOMM_CAUSE);
    scalarize(std, ast, TRUE);
    return;
  }
  if (comminfo.ugly_mask) {
    report_comm(std, UGLYMASK_CAUSE);
    scalarize(std, ast, TRUE);
    return;
  }
  comminfo.mask_phase = 0;
  opt_overlap();
  astnew = insert_forall_comm(A_SRCG(asn));
  A_SRCP(asn, astnew);
  comminfo.mask_phase = 1;
  if (A_IFEXPRG(ast)) {
    astnew = insert_forall_comm(A_IFEXPRG(ast));
    A_IFEXPRP(ast, astnew);
  }
  insert_call_comm(std, ast);

  /* guard_forall(std); */
  fix_guard_forall(std);

  /* give information if more than 40 run-time calls generated
   * for this forall
   */
  if (FT_NRT(A_OPT1G(STD_AST(std))) > 40)
    report_comm(std, MANYRUNTIME_CAUSE);
}

/**
   \brief The forall should be treated like a serial statement.

   Turn it into a block-forall so the IF stuff works OK.
 */
void
scalarize(int std, int forall, LOGICAL after_transformer)
{
  int std1;
  int std2;

  std1 = 0;
  std2 = 0;
  forall_dependency_scalarize(std, &std1, &std2);
  forall = STD_AST(std);
  sequentialize(std, forall, after_transformer);
  if (std1) {
    forall = STD_AST(std1);
    if (after_transformer)
      transform_forall(std1, forall);
  }

  if (std2) {
    forall = STD_AST(std2);
    if (after_transformer)
      transform_forall(std2, forall);
  }
}

/**
   \brief This is necessary, if forall sequentialized.
 */
void
un_fuse(int forall)
{
  int nd, nd1;
  int forall1;
  int fusedstd;
  int i;
  int forallstd;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NFUSE(nd, 0); i++) {
    fusedstd = FT_FUSEDSTD(nd, 0, i);
    forall1 = STD_AST(fusedstd);
    nd1 = A_OPT1G(forall1);
    FT_HEADER(nd1) = fusedstd;
  }
  FT_NFUSE(nd, 0) = 0;
  forallstd = A_STDG(forall);
  assert(forallstd, "un_fuse: it must be forall", forall, 3);
  assert(STD_AST(forallstd) == forall, "un_fuse: it must be forall", forall, 3);
  FT_HEADER(nd) = forallstd;
}

void
sequentialize(int std, int forall, LOGICAL after_transformer)
{
  int asn;
  int newast;
  int stdnext, stdnext1;
  int n, i;
  int triplet_list, index_var;
  int triplet;
  int expr;
  int lineno;

  if (after_transformer)
    un_fuse(forall);

  ast_to_comment(forall);
  asn = A_IFSTMTG(forall);
  if (!asn) {
    asn = mk_stmt(A_CONTINUE, 0);
  }
  lineno = STD_LINENO(std);
  stdnext = STD_NEXT(std);
  delete_stmt(A_STDG(forall));

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
    stdnext = add_stmt_before(newast, stdnext);
    STD_LINENO(stdnext) = lineno;
    if (after_transformer)
      transform_ast(stdnext, newast);
    stdnext = STD_NEXT(stdnext);
  }

  if (after_transformer)
    stdnext = sequentialize_mask_call(forall, stdnext);

  expr = A_IFEXPRG(forall);
  if (expr) {
    stdnext = STD_PREV(stdnext);
    stdnext1 = insert_mask(expr, stdnext);
    stdnext1 = STD_NEXT(stdnext1);
    if (after_transformer) {
      int nextnext;
      stdnext = STD_NEXT(stdnext);
      for (; stdnext != stdnext1; stdnext = nextnext) {
        nextnext = STD_NEXT(stdnext);
        transform_ast(stdnext, STD_AST(stdnext));
      }
    }
    stdnext = stdnext1;
  }

  if (after_transformer)
    stdnext = sequentialize_stmt_call(forall, stdnext);

  stdnext = add_stmt_before(asn, stdnext);
  stdnext1 = STD_NEXT(stdnext);
  STD_LINENO(stdnext) = lineno;
  if (after_transformer)
    transform_ast(stdnext, asn);
  stdnext = stdnext1;

  if (expr) {
    stdnext = insert_endmask(expr, STD_PREV(stdnext));
    stdnext = STD_NEXT(stdnext);
  }

  for (i = 0; i < n; i++) {
    newast = mk_stmt(A_ENDDO, 0);
    stdnext = add_stmt_before(newast, stdnext);
    STD_LINENO(stdnext) = lineno;
    stdnext = STD_NEXT(stdnext);
  }
}

/**
   \brief Initialize the communication analyzer phase.
 */
static void
comm_init(void)
{
  TRANS_ALLOC(trans.subb, SUBINFO, 1000);
  TRANS_ALLOC(trans.arrb, ARREF, 100);
  TRANS_ALLOC(trans.tdescb, TDESC, 50);
  init_pertbl();
  init_brtbl();
}

static LOGICAL
is_scatter(int std)
{
  if (!scatter_class(std))
    return FALSE;
  if (!comminfo.scat.base && !comminfo.scat.array_simple)
    return FALSE;
  emit_sum_scatterx(std);
  emit_scatterx(std);
  return TRUE;
}

/**
   \brief Like reference_for_temp(), this routine finds the dimension of sptr.

   It takes subscript `a(f(i),5,f(j))`. It eliminates scalar dimensions.
   It makes an ast to reference sptr: `a(f(i),5,f(j)) --> sptr(i,j)`
 */
static int
simple_reference_for_temp(int sptr, int a, int forall)
{
  int subs[7];
  int list;
  int i, ndim, k;
  int astnew;

  list = A_LISTG(forall);
  ndim = 0;
  do {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      int asd, adim;
      asd = A_ASDG(a);
      adim = ASD_NDIM(asd);
      /* array will be referenced after communication as follows  */
      for (i = 0; i < adim; i++) {
        int ast;
        ast = ASD_SUBS(asd, i);
        if (XBIT(58, 0x20000)) {
          if (A_TYPEG(ast) == A_TRIPLE) {
            subs[ndim] = ast;
            ++ndim;
          } else if ((k = search_forall_var(ast, list))) {
            subs[ndim] = mk_id(ASTLI_SPTR(k));
            ++ndim;
          } else if (A_SHAPEG(ast)) {
            subs[ndim] = ast;
            ++ndim;
          }
        } else if ((k = search_forall_var(ast, list))) {
          subs[ndim] = mk_id(ASTLI_SPTR(k));
          ++ndim;
        } else if (A_TYPEG(ast) == A_TRIPLE || A_SHAPEG(ast)) {
          /* include this dimension */
          subs[ndim] = ast;
          ++ndim;
        }
      }
      a = A_LOPG(a);
    } else {
      interr("simple_reference_for_temp: not subscr or member", a, 3);
    }
  } while (A_TYPEG(a) != A_ID);
  assert(ndim == rank_of_sym(sptr),
         "simple_reference_for_temp: rank mismatched", sptr, 4);
  astnew = mk_subscr(mk_id(sptr), subs, ndim, DTY(DTYPEG(sptr) + 1));
  return astnew;
}

static int
temp_gatherx(int std, int forall, int lhs, int rhs, int dty, int *allocast)
{
  int sptr;
  int subscr[7];
  int ast;
  int nd;
  int astnew;
  int header;

  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  sptr = mk_forall_sptr_gatherx(forall, lhs, rhs, subscr, dty);

  astnew =
      mk_subscr(mk_id(sptr), subscr, rank_of_sym(sptr), DTY(DTYPEG(sptr) + 1));
  ast = new_node(A_HALLOBNDS);
  A_LOPP(ast, astnew);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_ALLOC_SPTR(nd) = sptr;
  FT_ALLOC_FREE(nd) = header;
  FT_ALLOC_SAME(nd) = 0;
  FT_ALLOC_REUSE(nd) = 0;
  FT_ALLOC_USED(nd) = 0;
  FT_ALLOC_OUT(nd) = sptr;
  A_OPT1P(ast, nd);
  *allocast = ast;
  return sptr;
}

static int
temp_copy_section(int std, int forall, int lhs, int rhs, int dty, int *allocast)
{
  int sptr;
  int subscr[7];
  int ast;
  int nd;
  int astnew;
  int header;

  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  sptr = mk_forall_sptr_copy_section(forall, lhs, rhs, subscr, dty);

  astnew =
      mk_subscr(mk_id(sptr), subscr, rank_of_sym(sptr), DTY(DTYPEG(sptr) + 1));
  ast = new_node(A_HALLOBNDS);
  A_LOPP(ast, astnew);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_ALLOC_SPTR(nd) = sptr;
  FT_ALLOC_FREE(nd) = header;
  FT_ALLOC_SAME(nd) = 0;
  FT_ALLOC_REUSE(nd) = 0;
  FT_ALLOC_USED(nd) = 0;
  FT_ALLOC_OUT(nd) = sptr;
  A_OPT1P(ast, nd);
  *allocast = ast;
  return sptr;
}

/**
   \brief Just like copy_section_temp_before() except it does not eliminate
   scalar dimension.

   This means that it makes a new array with sptr by using subscript of rhs.
 */
static int
gatherx_temp_before(int sptr, int rhs, int forall)
{
  int subs[7];
  int k, j;
  int asd;
  int ndim;
  int astnew;
  int astli;
  int nidx;
  int list;

  asd = A_ASDG(rhs);
  ndim = ASD_NDIM(asd);
  list = A_LISTG(forall);

  j = 0;
  /* array will be referenced after communication as follows  */
  for (k = 0; k < ndim; ++k) {
    astli = 0;
    nidx = 0;
    search_forall_idx(ASD_SUBS(asd, k), list, &astli, &nidx);
    if (nidx == 1 && astli) {
      /* include this dimension */
      subs[j] = mk_id(ASTLI_SPTR(astli));
      j++;
    } else if (nidx == 0 && astli == 0) {
      /* include scalar dimension too */
      subs[j] = ASD_SUBS(asd, k);
      j++;
    }
  }
  assert(j == rank_of_sym(sptr), "gatherx_temp_before: rank mismatched", sptr,
         4);
  astnew = mk_subscr(mk_id(sptr), subs, j, DTY(DTYPEG(sptr) + 1));
  return astnew;
}

static int
make_sec_ast(int arr, int std, int allocstd, int sectflag)
{
  int asn;
  int ast;
  int nd;
  int sec, secast;
  int sectstd;
  int forall;
  int sptr;
  int header;
  int bogus;
  int shape;
  int rank;

  forall = STD_AST(std);
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);

  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  ast = new_node(A_HSECT);
  sptr = sptr_of_subscript(arr);
  A_LOPP(ast, arr);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_SECT_ARR(nd) = arr;
  FT_SECT_SPTR(nd) = sptr;
  FT_SECT_ALLOC(nd) = allocstd;
  FT_SECT_FREE(nd) = header;
  FT_SECT_FLAG(nd) = sectflag;
  bogus = getbit(sectflag, 8);
  shape = A_SHAPEG(arr);
  assert(shape, "make_sec_ast: ast has no shape", arr, 4);
  rank = SHD_NDIM(shape);
  if (is_whole_array(arr) && !bogus) {
    DESCUSEDP(sptr, 1);
    sec = DESCRG(sptr);
    secast = check_member(arr, mk_id(sec));
  } else {
    sec = sym_get_sdescr(sptr, rank); /* ZB */
    secast = mk_id(sec);
  }
  FT_SECT_SAME(nd) = 0;
  FT_SECT_REUSE(nd) = 0;
  FT_SECT_OUT(nd) = sec;
  A_OPT1P(ast, nd);

  A_DESTP(asn, secast);
  A_SRCP(asn, ast);

  sectstd = add_stmt_before(asn, header);
  A_STDP(asn, sectstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), sectstd);
  FT_NRT(nd)++;

  return sectstd;
}

/**
   \brief This routine takes an array in a forall statement with its subinfo
   and replaces all forall indexes.

   E.g., `forall(i=1:10:2) a(i+1)` will become `a(2:11:2)`.

   Note that this assumes each forall index appears in array subscripts.
   If not, something is wrong in the communication detection algorithm.
 */
static int
forall_2_sec(int a, int forall)
{
  int list;
  int ndim;
  int i;
  int j;
  int asd;
  int sub_expr;
  int triple;
  int l, u, s;
  int t1, t2, t3;
  int subs[7];
  int sptr;
  int astli;
  int base;
  int stride;
  int shape;
  int nd;
  int nidx;
  int changed;

  assert(A_TYPEG(a) == A_SUBSCR, "forall_2_sec: not SUBSCR", a, 4);
  list = A_LISTG(forall);
  asd = A_ASDG(a);
  sptr = sptr_of_subscript(a);
  ndim = ASD_NDIM(asd);
  shape = 0;
  if (A_ARRASNG(forall)) {
    nd = get_finfo(forall, a);
    if (nd)
      shape = FINFO_SHAPE(nd);
  }

  /* If it was an array assignment, use the original section info */
  if (A_ARRASNG(forall) && shape) {
    j = 0;
    for (i = 0; i < ndim; i++) {
      sub_expr = ASD_SUBS(asd, i);
      astli = 0;
      nidx = 0;
      search_forall_idx(sub_expr, list, &astli, &nidx);
      if (nidx == 1) {
        t1 = check_member(a, SHD_LWB(shape, j));
        t2 = check_member(a, SHD_UPB(shape, j));
        t3 = check_member(a, SHD_STRIDE(shape, j));
        j++;
        subs[i] = mk_triple(t1, t2, t3);
      } else
        subs[i] = ASD_SUBS(asd, i);
    }
    assert(j == SHD_NDIM(shape), "forall_2_sec: something is wrong", a, 4);
    return mk_subscr(A_LOPG(a), subs, ndim, DTYPEG(sptr));
  }
  /* If it was a forall, calculate the section info */
  changed = 0;
  for (i = 0; i < ndim; i++) {
    sub_expr = ASD_SUBS(asd, i);
    astli = 0;
    search_idx(sub_expr, list, &astli, &base, &stride);
    assert(base, "forall_2_sec: something is wrong", a, 4);
    if (astli) {
      triple = ASTLI_TRIPLE(astli);
      l = A_LBDG(triple);
      u = A_UPBDG(triple);
      s = A_STRIDEG(triple);
      t1 = replace_expr(sub_expr, ASTLI_SPTR(astli), l, 1);
      t2 = replace_expr(sub_expr, ASTLI_SPTR(astli), u, 1);
      if (s == 0)
        s = astb.bnd.one;
      t3 = opt_binop(OP_MUL, s, stride, astb.bnd.dtype);
      subs[i] = mk_triple(t1, t2, t3);
      changed = 1;
    } else
      subs[i] = ASD_SUBS(asd, i);
  }
  if (changed)
    return mk_subscr(A_LOPG(a), subs, ndim, DTYPEG(sptr));
  else
    return a;
}

/* give a%b(1:n)%c, return pointer to a%b%c in 'pnewast',
 * pointer to a%b(1:n) in 'psectast', pointer to b in 'psptr'. */
static void
remove_section(int ast, int *pnewast, int *psectast, int *psptr, int *panydist,
               int *pnontrivial)
{
  int lop, sptr = 0;
  switch (A_TYPEG(ast)) {
  case A_SUBSTR:
    remove_section(A_LOPG(ast), pnewast, psectast, psptr, panydist,
                   pnontrivial);
    *pnewast = mk_substr(*pnewast, A_LEFTG(ast), A_RIGHTG(ast), A_DTYPEG(ast));
    break;
  case A_INTR:
    *pnewast = ast;
    *psectast = 0;
    *psptr = 0;
    break;
  case A_ID:
    sptr = A_SPTRG(ast);
    *psptr = sptr;
    *psectast = ast;
    *pnewast = ast;
    break;
  case A_MEM:
    lop = A_PARENTG(ast);
    remove_section(lop, pnewast, psectast, psptr, panydist, pnontrivial);
    *pnewast = mk_member(*pnewast, A_MEMG(ast), A_DTYPEG(ast));
    sptr = A_SPTRG(A_MEMG(ast));
    if (A_SHAPEG(lop) != 0) {
      /* psectast, psptr already set by parent */
      *pnontrivial = 1;
    } else {
      *psectast = ast;
      *psptr = sptr;
    }
    break;
  case A_SUBSCR:
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_ID) {
      sptr = A_SPTRG(lop);
    } else if (A_TYPEG(lop) == A_MEM) {
      sptr = A_SPTRG(A_MEMG(lop));
    }
    remove_section(lop, pnewast, psectast, psptr, panydist, pnontrivial);
    if (A_SHAPEG(ast) == 0) {
      *pnewast = mk_subscr_copy(*pnewast, A_ASDG(ast), A_DTYPEG(ast));
      *psectast = ast;
      *psptr = sptr;
    } else if (A_TYPEG(lop) == A_ID ||
               (A_TYPEG(lop) == A_MEM && A_SHAPEG(A_PARENTG(lop)) == 0)) {
      /* if the 'lop' is an ID, or
       * if the 'lop' is an member whose parent has no shape,
       * shape comes from this subscript */
      *psectast = ast;
      *psptr = sptr;
    } else {
      /* section comes from A_MEM parent; psectast, psptr already set */
      *pnewast = mk_subscr_copy(*pnewast, A_ASDG(ast), A_DTYPEG(ast));
      *pnontrivial = 1;
    }
    break;
  default:
    *pnewast = 0;
    *psectast = 0;
    *psptr = 0;
    break;
  }
  if (sptr && ALIGNG(sptr))
    *panydist = 1;
} /* remove_section */

/*     pv => ar
 *     pv => ar(lower:upper:stride,...)
 *     call pghpf_ptr_assign(pv, pv$sdsc, ar, ar$d, sectflag)
 *  pv: base.
 *  pv$sdsc:            pv's (new) static descriptor
 *  ar:                 ar's base address (ar or ar(ar$o))
 *  ar$d:               ar's (old) descriptor
 *  sectflag:           integer, 0 if whole array, 1 if section
 */
static void
transform_ptr(int std, int ast)
{
  int argt, nargs;
  int newargt;
  int src, dest, newsrc, sectast, src_sptr, anydist;
  int dest_sptr = 0, nontrivial;
  int array_desc;
  int section;
  int ptr_reshape_dest = 0;
  int dtype;

  assert(A_TYPEG(ast) == A_ICALL && A_OPTYPEG(ast) == I_PTR2_ASSIGN,
         "transform_ptr: something is wrong", 2, ast);
  NODESCP(find_pointer_variable(A_LOPG(ast)), 1);
  argt = A_ARGSG(ast);
  nargs = A_ARGCNTG(ast);
  assert(nargs == 2, "transform_ptr: something is wrong", 2, ast);
  src = ARGT_ARG(argt, 1);
  dest = ARGT_ARG(argt, 0);

  anydist = 0;
  nontrivial = 0;
  remove_section(src, &newsrc, &sectast, &src_sptr, &anydist, &nontrivial);

/* sectast points to subtree with A_SHAPE() != 0.
 * src_sptr is the section sptr */
again:
  if (A_TYPEG(dest) == A_ID) {
    dest_sptr = A_SPTRG(dest);
  } else if (A_TYPEG(dest) == A_MEM) {
    dest_sptr = A_SPTRG(A_MEMG(dest));
  } else if (A_TYPEG(dest) == A_SUBSCR) { /* ptr reshape */
    ptr_reshape_dest = dest;
    dest = A_LOPG(dest);
    goto again;
  } else
    assert(0, "transform_ptr: bad pointer assignment target", ast, 3);

  /* don't let scalar pointer point to distributed array */
  if (DTY(DTYPEG(dest_sptr)) != TY_ARRAY && DTY(DTYPEG(src_sptr)) == TY_ARRAY &&
      anydist)
    error(155, 4, STD_LINENO(std), SYMNAME(dest_sptr),
          "- scalar POINTER associated with distributed object is unsupported");

  DESCUSEDP(src_sptr, 1);
  DESCUSEDP(dest_sptr, 1);
  if (!POINTERG(dest_sptr))
    error(155, 3, STD_LINENO(std), "must be POINTER", SYMNAME(dest_sptr));

  array_desc = 0;
  section = 0;
  dtype = DDTG(DTYPEG(dest_sptr));
  if (DTY(dtype) == TY_PTR && DTY(DTY(dtype + 1)) == TY_PROC &&
      STYPEG(src_sptr) == ST_PROC) {
    /* No array descriptor for procedure name target in a
     * procedure pointer assignment.
     */
  } else if (ptr_reshape_dest && bnds_remap_list(ptr_reshape_dest) &&
             simply_contiguous(src)) {
    emit_alnd_secd(dest_sptr, dest, TRUE, std, ptr_reshape_dest);
  } else if (A_TYPEG(sectast) == A_SUBSCR && A_SHAPEG(sectast) != 0) {
    int d;
    array_desc = check_member(dest, mk_id(SDSCG(dest_sptr)));
    d = make_sec_from_ast(sectast, std, std, array_desc, 0);
    /* if this was the whole array, we use the descriptor
     * of the source, not target */
    if (d == DESCRG(src_sptr)) {
      array_desc = check_member(sectast, mk_id(d));
    }
    section = 1;
  } else if (A_TYPEG(src) == A_MEM && A_SHAPEG(A_PARENTG(src))) {
    section = 1;
    array_desc = DESCRG(src_sptr);
    array_desc = check_member(sectast, mk_id(array_desc));
  } else {
    if (POINTERG(src_sptr) && A_SHAPEG(sectast)) {
      array_desc = SDSCG(src_sptr); /* section descriptor */
      array_desc = check_member(sectast, mk_id(array_desc));
    } else if (DTY(DTYPEG(src_sptr)) == TY_ARRAY && A_SHAPEG(sectast)) {
      array_desc = DESCRG(src_sptr);
      array_desc = check_member(sectast, mk_id(array_desc));
    } else {
      array_desc = 0;
    }
  }

  nargs = nontrivial ? 7 : 5;
  if (A_TYPEG(ptr_reshape_dest) == A_SUBSCR) {
    /* ptr reshape
     * compute number of additional args
     */
    int shd, nd, asd, i, sub;

    if (ptr_reshape_dest && bnds_remap_list(ptr_reshape_dest) &&
        simply_contiguous(src)) {
      newsrc = first_element(src);
    }
    shd = A_SHAPEG(ptr_reshape_dest);
    nd = SHD_NDIM(shd);
    nargs = 8; /* num dimensions */
    asd = A_ASDG(ptr_reshape_dest);
    for (i = 0; i < nd; ++i) {
      sub = ASD_SUBS(asd, i);
      if (A_LBDG(sub))
        ++nargs; /* lowerbound */
      if (A_UPBDG(sub))
        ++nargs; /* upperbound */
    }
  }
  newargt = mk_argt(nargs);
  ARGT_ARG(newargt, 0) = ARGT_ARG(argt, 0);
  /* this will need some changes when dest_sptr is a derived type member */
  if ((STYPEG(dest_sptr) == ST_VAR || STYPEG(dest_sptr) == ST_ARRAY) &&
      DSCASTG(dest_sptr)) {
    ARGT_ARG(newargt, 1) = DSCASTG(dest_sptr);
  } else {
    SPTR sdsc = SDSCG(dest_sptr);
    if (sdsc) {
      ARGT_ARG(newargt, 1) = check_member(dest, mk_id(sdsc));
    } else {
      ARGT_ARG(newargt, 1) = astb.bnd.zero;
    }
  }
  ARGT_ARG(newargt, 2) = newsrc;
  if (array_desc)
    ARGT_ARG(newargt, 3) = array_desc;
  else
    ARGT_ARG(newargt, 3) =
        mk_isz_cval(dtype_to_arg(DTYPEG(dest_sptr)), astb.bnd.dtype);

  /* section flag argument */
  if (!section)
    ARGT_ARG(newargt, 4) = astb.bnd.zero;
  else
    ARGT_ARG(newargt, 4) = astb.bnd.one;

  if (nontrivial) {
    /* add datatype argument */
    ARGT_ARG(newargt, 5) =
        mk_isz_cval(size_of(DDTG(DTYPEG(dest_sptr))), astb.bnd.dtype);
    ARGT_ARG(newargt, 6) =
        mk_isz_cval(ty_to_lib[DTYG(DTYPEG(dest_sptr))], astb.bnd.dtype);
  }

  if (A_TYPEG(ptr_reshape_dest) == A_SUBSCR) {
    /* ptr reshape
     * generate additional args
     */
    int shd, nd, asd, i, sub, val[4] = {0, 0, 0, 0}, tmp;
    int lbast, ubast, argcnt = 7;

    if (!nontrivial) {
      ARGT_ARG(newargt, 5) = astb.bnd.zero;
      ARGT_ARG(newargt, 6) = astb.bnd.zero;
    }
    shd = A_SHAPEG(ptr_reshape_dest);
    nd = SHD_NDIM(shd);
    val[1] = nd;
    tmp = getcon(val, DT_INT4);
    ARGT_ARG(newargt, argcnt++) = mk_cnst(tmp); /* num dimensions */
    asd = A_ASDG(ptr_reshape_dest);
    for (i = 0; i < nd; ++i) {
      sub = ASD_SUBS(asd, i);
      lbast = A_LBDG(sub);
      ubast = A_UPBDG(sub);
      if (lbast) {
        ARGT_ARG(newargt, argcnt++) = lbast; /* lowerbound */
      }
      if (ubast) {
        ARGT_ARG(newargt, argcnt++) = ubast; /* upperbound */
      }
    }
  }
  A_ARGCNTP(ast, nargs);
  A_ARGSP(ast, newargt);
}

static int
insert_forall_comm(int ast)
{
  /* go through and add the communication & rewrite the AST */
  int std;
  int l, r, d, o;
  int a;
  int i, nargs, argt;
  int arref;
  int header;
  int forall;
  int rhs_is_dist;
  int sptr;
  int asd, ndim;
  int subs[7];
  int nd, nd2;
  int src;
  int cnt;
  int commstd, commasn, comm;
  int lhs;

  a = ast;
  if (!a)
    return a;
  std = comminfo.std;
  forall = STD_AST(std);
  switch (A_TYPEG(ast)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = insert_forall_comm(A_LOPG(a));
    r = insert_forall_comm(A_ROPG(a));
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = insert_forall_comm(A_LOPG(a));
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(a);
    l = insert_forall_comm(A_LOPG(a));
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(a);
    l = insert_forall_comm(A_LOPG(a));
    return mk_paren(l, d);
  case A_MEM:
    r = A_MEMG(a);
    d = A_DTYPEG(r);
    l = insert_forall_comm(A_PARENTG(a) /*, forall, std*/);
    return mk_member(l, r, d);
  case A_SUBSTR:
    return a;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = insert_forall_comm(ARGT_ARG(argt, i));
    }
    /* remove cshift and eoshift, since they become overlap comm */
    if (A_OPTYPEG(a) == I_CSHIFT || A_OPTYPEG(a) == I_EOSHIFT) {
      src = ARGT_ARG(argt, 0);
      nd = A_OPT1G(comminfo.forall);
      cnt = FT_NRT(nd) - 2;
      commstd = glist(FT_RTL(nd), cnt);
      commasn = STD_AST(commstd);
      comm = A_SRCG(commasn);
      assert(A_TYPEG(comm) == A_HOVLPSHIFT,
             "insert_forall_comm: CSHIFT/EOSHIFT must be overlap", a, 2);
      nd2 = A_OPT1G(comm);
      FT_SHIFT_TYPE(nd2) = A_OPTYPEG(a);
      if (A_OPTYPEG(a) == I_EOSHIFT)
        FT_SHIFT_BOUNDARY(nd2) = ARGT_ARG(argt, 2);
      return src;
    }
    return a;
  case A_CNST:
  case A_CMPLXC:
    return a;
  case A_ID:
    return a;
  case A_SUBSCR:
    if (A_SHAPEG(a))
      return a;
    sptr = sptr_of_subscript(a);
    if (!ALIGNG(sptr)) {
      int parent;
      parent = A_LOPG(a);
      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; i++) {
        subs[i] = insert_forall_comm(ASD_SUBS(asd, i));
      }
      parent = insert_forall_comm(parent);
      return mk_subscr(parent, subs, ndim, A_DTYPEG(a));
    }

    if (!A_SHAPEG(a) && is_array_element_in_forall(a, std)) {
      nd = A_OPT1G(forall);
      header = FT_HEADER(nd);
      /*             a = emit_get_scalar(a, header); */
      rhs_is_dist = FALSE;
      a = insert_comm_before(header, a, &rhs_is_dist, FALSE);
      return a;
    }
    /* don't generate communication iff lhs == rhs */
    lhs = A_DESTG(A_IFSTMTG(forall));
    if (lhs == a)
      return a;

    arref = A_RFPTRG(a);

    switch (ARREF_CLASS(arref)) {
    case NO_COMM:
      break;
    case OVERLAP:
      emit_overlap(a);
      break;
    case COPY_SECTION:
      a = emit_copy_section(a, std);
      break;
    case GATHER:
      a = emit_gatherx(a, std, FALSE);
      break;
    case IRREGULAR:
      /*		a = emit_irregular(a, std);*/
      break;
    default:
      interr("insert_forall_comm: unknown comm tag", std, 2);
      return 0;
    }

    return a;

  default:
    interr("insert_forall_comm: unknown expression", std, 2);
    return 0;
  }
}

static void
init_opt_tables(void)
{
  cs_table.is_used_lhs = FALSE;
}

/* return TRUE if the LHS variable can be used for this RHS communication
 * target */
static LOGICAL
is_use_lhs(int a, LOGICAL sameidx, LOGICAL independent, int std)
{
  int lhs;
  int list;
  int src;

  if (cs_table.is_used_lhs)
    return FALSE;
  if (A_IFEXPRG(comminfo.forall))
    return FALSE;
  lhs = comminfo.sub;
  list = A_LISTG(comminfo.forall);
  src = A_SRCG(A_IFSTMTG(comminfo.forall));
  if (DTY(A_DTYPEG(a)) != DTY(A_DTYPEG(lhs)))
    return FALSE;
  if (sameidx && !is_same_number_of_idx(lhs, a, list))
    return FALSE;
  if (!independent && expr_dependent(a, lhs, std, std))
    return FALSE;

  cs_table.is_used_lhs = TRUE;
  return TRUE;
} /* is_use_lhs */

/* this is used to decide if section created for forall
 * to check whether index is out of bounds .
 * This does not occur iff:
 *     1-) forall from array-assignment or where statement
 *     2-) forall without mask
 */
static LOGICAL
is_bogus_forall(int forall)
{
  int mask;

  if (A_ARRASNG(forall))
    return FALSE;
  mask = A_IFEXPRG(forall);
  if (!mask)
    return FALSE;
  return TRUE;
}

static int
emit_copy_section(int a, int std)
{
  int ast;
  int astnew;
  int asn;
  int tempast;
  int tempast0;
  int dest, lop;
  int forall;
  int list;
  int lhs;
  int allocstd;
  int startstd;
  int commstd;
  int sectlstd;
  int sectrstd;
  int cp, xfer;
  int nd;
  int sptr;
  int allocast;
  int order2[MAXDIMS];
  int no;
  int header;
  int lhssec;
  int sectflag;
  LOGICAL independent;

  forall = STD_AST(std);
  lhs = comminfo.sub;
  list = A_LISTG(forall);
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);

  sectflag = 0;
  if (is_bogus_forall(forall))
    sectflag |= BOGUSFLAG;

  if (!is_ordered(lhs, a, list, order2, &no)) {
    tempast = emit_permute_section(a, std);
    return tempast;
  }

  open_dynpragma(std, STD_LINENO(std));
  independent = (flg.x[19] & 0x100) != 0;
  close_pragma();
  sectlstd = 0;
  lhssec = 0;
  if (is_use_lhs(a, TRUE, independent, std)) {
    sptr = sptr_of_subscript(comminfo.sub);
    tempast = lhs;
    lhssec = tempast = forall_2_sec(tempast, forall);
    sectlstd = make_sec_ast(tempast, std, 0, sectflag);
    nd = A_OPT1G(forall);
    FT_SECTL(nd) = sectlstd;
  }

  sptr = temp_copy_section(std, forall, lhs, a,
                           DTY(DTYPEG(sptr_of_subscript(a)) + 1), &allocast);
  tempast0 = tempast = copy_section_temp_before(sptr, a, forall);

  allocstd = add_stmt_before(allocast, header);
  A_STDP(allocast, allocstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), allocstd);
  FT_NRT(nd)++;

  tempast = forall_2_sec(tempast, forall);
  sectlstd = make_sec_ast(tempast, std, allocstd, sectflag);

  astnew = forall_2_sec(a, forall);
  sectrstd = make_sec_ast(astnew, std, 0, sectflag);

  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  ast = new_node(A_HCOPYSECT);
  A_SRCP(ast, astnew);
  A_SDESCP(ast, 0);
  A_DESTP(ast, tempast);
  A_DDESCP(ast, 0);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_CCOPY_LHS(nd) = lhs;
  FT_CCOPY_RHS(nd) = a;
  FT_CCOPY_TSPTR(nd) = sptr;
  FT_CCOPY_SECTR(nd) = sectrstd;
  FT_CCOPY_SECTL(nd) = sectlstd;
  FT_CCOPY_ALLOC(nd) = allocstd;
  FT_CCOPY_FREE(nd) = header;
  FT_CCOPY_REUSE(nd) = 0;
  FT_CCOPY_USELHS(nd) = 0;
  FT_CCOPY_SAME(nd) = 0;
  FT_CCOPY_LHSSEC(nd) = lhssec;
  FT_CCOPY_NOTLHS(nd) = (lhssec) ? 0 : 1;
  A_OPT1P(ast, nd);
  cp = sym_get_cp();
  FT_CCOPY_OUT(nd) = cp;
  dest = mk_id(cp);
  A_DESTP(asn, dest);
  A_SRCP(asn, ast);

  commstd = add_stmt_before(asn, header);
  A_STDP(asn, commstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), commstd);
  FT_NRT(nd)++;

  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  ast = new_node(A_HCSTART);
  lop = mk_id(cp);
  A_LOPP(ast, lop);
  A_SRCP(ast, astnew);
  A_DESTP(ast, tempast);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_CSTART_COMM(nd) = commstd;
  FT_CSTART_RHS(nd) = a;
  FT_CSTART_USEDSTD(nd) = comminfo.usedstd;
  xfer = sym_get_xfer();
  FT_CSTART_OUT(nd) = xfer;
  FT_CSTART_SECTR(nd) = sectrstd;
  FT_CSTART_SECTL(nd) = sectlstd;
  FT_CSTART_ALLOC(nd) = allocstd;
  FT_CSTART_FREE(nd) = header;
  FT_CSTART_REF(nd) = tempast0;
  FT_CSTART_TYPE(nd) = A_HCOPYSECT;
  FT_CSTART_REUSE(nd) = 0;
  FT_CSTART_INVMVD(nd) = 0;
  FT_CSTART_USELHS(nd) = 0;
  FT_CSTART_SAME(nd) = 0;
  A_OPT1P(ast, nd);
  dest = mk_id(xfer);
  A_DESTP(asn, dest);
  A_SRCP(asn, ast);

  startstd = add_stmt_before(asn, header);
  A_STDP(asn, startstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), startstd);
  FT_NRT(nd)++;

  return a;
}

/*
 * pghpf_permute_section(void *rb, void *sb, section *rs, section *ss, ...)
 *
 *chdr *
 * pghpf_comm_permute(void *rb, void *sb, section *rs, section *ss, ...)
 * ... = int x1, .., int xN,  where N = section rank
 * The axis arguments (x1, .., xN) is a permutation of the integers 1..N.
 * The permutation applies to the dimensions on the right hand side (like
 * a gather operation).
 * For example:
 *	forall (i=1:2, j=1:4, k=1:5) a(i,3,j,k) = b(k,i,j)
 *	pghpf_permute_section(a, b, a$s, b$s, 2, 3, 1)
 */
static int
emit_permute_section(int a, int std)
{
  int sptr, sptrast;
  int asd;
  int ndim;
  int ast1;
  int astnew;
  int tempast, tempast0;
  int argt, nargs;
  int i;
  int src, dest;
  int forall;
  int list;
  int lhs;
  LOGICAL use_lhs;
  int order2[MAXDIMS];
  int no;
  int func;
  int new_a;
  int nd, header;
  int sectflag;

  forall = STD_AST(std);
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  lhs = comminfo.sub;
  list = A_LISTG(forall);
  asd = A_ASDG(comminfo.sub);
  ndim = ASD_NDIM(asd);

  sectflag = 0;
  if (is_bogus_forall(forall))
    sectflag |= BOGUSFLAG;

  if (cs_table.is_used_lhs) {
    use_lhs = FALSE;
  } else {
    use_lhs = is_use_lhs_final(a, forall, TRUE, FALSE, std);
  }
  if (use_lhs) {
    sptr = sptr_of_subscript(comminfo.sub);
    sptrast = A_LOPG(comminfo.sub);
    tempast = lhs;
    cs_table.is_used_lhs = TRUE;
  } else {
    new_a = eliminate_extra_idx(lhs, a, forall);
    sptr = get_temp_copy_section(forall, lhs, new_a, header, header, a);
    sptrast = mk_id(sptr);
    tempast0 = tempast = copy_section_temp_before(sptr, new_a, forall);
  }

  if (is_ordered(tempast, a, list, order2, &no)) {
    assert(0, "emit_permute_section: something is wrong", 3, a);
  }

  tempast = forall_2_sec(tempast, forall);
  dest = make_sec_from_ast(tempast, header, header, 0, sectflag);

  astnew = forall_2_sec(a, forall);
  src = make_sec_from_ast(astnew, header, header, 0, sectflag);

  nargs = 4 + no;
  func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_permute_section), DT_NONE));
  NODESCP(A_SPTRG(func), 1);

  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = sptrast;
  ARGT_ARG(argt, 1) = A_LOPG(a);

  ARGT_ARG(argt, 2) = check_member(sptrast, mk_id(dest));
  ARGT_ARG(argt, 3) = check_member(A_LOPG(a), mk_id(src));

  for (i = 0; i < no; i++)
    ARGT_ARG(argt, 4 + i) = mk_isz_cval(order2[i] + 1, astb.bnd.dtype);

  ast1 = mk_stmt(A_CALL, 0);
  A_LOPP(ast1, func);
  A_ARGCNTP(ast1, nargs);
  A_ARGSP(ast1, argt);
  add_stmt_before(ast1, header);

  /* temp will be referenced after communication as follows  */
  if (use_lhs)
    return lhs; /* forall is totally removed no need to access */
  else {
    process_rhs_sub(tempast0);
    return tempast0;
  }
}

/* This routine finds out the dimension of sptr.
 * It takes subscript a(f(i),5,f(j)). It eliminates scalar dimension.
 * It makes an ast for reference sptr.
 *  a(f(i),5,f(j)) --> sptr(i,j)
 */
static int
copy_section_temp_before(int sptr, int rhs, int forall)
{
  int subs[7];
  int k, j;
  int asd;
  int ndim;
  int astnew;
  int astli;
  int nidx;
  int list;

  asd = A_ASDG(rhs);
  ndim = ASD_NDIM(asd);
  list = A_LISTG(forall);

  j = 0;
  /* array will be referenced after communication as follows  */
  for (k = 0; k < ndim; ++k) {
    astli = 0;
    nidx = 0;
    search_forall_idx(ASD_SUBS(asd, k), list, &astli, &nidx);
    if (nidx == 1 && astli) {
      /* include this dimension */
      subs[j] = mk_id(ASTLI_SPTR(astli));
      j++;
    }
  }
  assert(j == rank_of_sym(sptr), "copy_section_temp_before: rank mismatched",
         sptr, 4);
  astnew = mk_subscr(mk_id(sptr), subs, j, DTY(DTYPEG(sptr) + 1));
  return astnew;
}

/* It takes  forall(i=,j=,k=) a(i,j,k) =  b(j,i) , return a(i,j,1) */
static int
eliminate_extra_idx(int lhs, int a, int forall)
{
  int subs[7];
  int k, i;
  int asd;
  int ndim;
  int asd1;
  int ndim1;
  int astnew;
  int astli;
  int nidx;
  int list;
  LOGICAL found;
  int sptr;

  sptr = sptr_of_subscript(lhs);
  asd = A_ASDG(lhs);
  ndim = ASD_NDIM(asd);
  list = A_LISTG(forall);

  asd1 = A_ASDG(a);
  ndim1 = ASD_NDIM(asd1);

  for (k = 0; k < ndim; ++k) {
    subs[k] = ASD_SUBS(asd, k);
    astli = 0;
    nidx = 0;
    search_forall_idx(ASD_SUBS(asd, k), list, &astli, &nidx);
    if (nidx == 1 && astli) {
      found = FALSE;
      for (i = 0; i < ndim1; ++i)
        if (is_name_in_expr(ASD_SUBS(asd1, i), ASTLI_SPTR(astli)))
          found = TRUE;
      if (!found)
        subs[k] = astb.i1;
    }
  }
  astnew = mk_subscr(mk_id(sptr), subs, ndim, DTY(DTYPEG(sptr) + 1));
  return astnew;
}

/* This  routine is to find out how index is permuted at result
 * based on array. used by scatterx/gatherx to perform axis ordering.
 * It creates axis array for indirection subscripts.
 *
 * For an indirectly indexed dimension, the axis vector indicates which
 * combination of the index variables is used to subscript the index
 * vector.  The size of the axis vector is equal to the rank of the index
 * vector.  If the order of the index variables is not permuted, i.e. the
 * axis vector is (/1, 2, 3, .. N/), then the corresponding permuted bit
 * can be zeroed and the axis argument omitted.

 * For a directly indexed dimension, the axis argument indicates which
 * index variable is used to subscript that dimension.  If the axis
 * number matches the dimension number, then the corresponding permuted
 * bit can be zeroed and the axis argument omitted.
 */
static void
permute_axis(int result, int array, int list, int permute[7])
{
  int subs[7];
  int newresult;
  int asd, ndim;
  int i;
  int per[7], per1[7];
  int nper1;

  for (i = 0; i < 7; i++)
    permute[i] = 0;

  /* find out for indirection array */
  asd = A_ASDG(result);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++) {
    subs[i] = ASD_SUBS(asd, i);
    if (is_vector_subscript(subs[i], list)) {
      compute_permute(array, subs[i], list, per);
      if (is_permuted(subs[i], per, per1, &nper1))
        permute[i] = put_data(per1, nper1);
      subs[i] = mk_isz_cval(1, astb.bnd.dtype);
    }
  }

  /* find out after eliminating indirections */

  newresult = mk_subscr(A_LOPG(result), subs, ndim, A_DTYPEG(result));
  compute_permute(array, newresult, list, per);

  for (i = 0; i < ndim; i++) {
    subs[i] = ASD_SUBS(asd, i);
    if (per[i] == 0)
      continue;
    if (is_vector_subscript(subs[i], list))
      continue;
    permute[i] = mk_isz_cval(per[i], astb.bnd.dtype);
  }
}

static void
init_pertbl(void)
{
  pertbl.size = 200;
  NEW(pertbl.base, TABLE, pertbl.size);
  pertbl.avl = 0;
}

static void
free_pertbl(void)
{
  FREE(pertbl.base);
  pertbl.base = NULL;
}

static int
get_pertbl(void)
{
  int nd;

  nd = pertbl.avl++;
  NEED(pertbl.avl, pertbl.base, TABLE, pertbl.size, pertbl.size + 100);
  if (nd > SPTR_MAX || pertbl.base == NULL)
    errfatal(7);
  return nd;
}

static int
put_data(int permute[7], int no)
{
  ADSC *ad;
  int dtype;
  int i, j;
  int arr;
  LOGICAL found;

  assert(no, "put_data: something is wrong", no, 2);

  /* find about whether same axis array created before */
  for (i = 0; i < pertbl.avl; i++) {
    if (pertbl.base[i].f2 == no) {
      found = TRUE;
      for (j = 0; j < no; j++) {
        if (permute[j] != pertbl.base[i].f4[j])
          found = FALSE;
      }
      if (found)
        return mk_id(pertbl.base[i].f1);
    }
  }

  arr = sym_get_array("axis", 0, DT_INT, 1);

  i = get_pertbl();
  pertbl.base[i].f1 = arr;
  pertbl.base[i].f2 = no;
  for (j = 0; j < no; j++)
    pertbl.base[i].f4[j] = permute[j];

  ALLOCP(arr, 0);
  dtype = DTYPEG(arr);
  ad = AD_DPTR(dtype);
  AD_LWAST(ad, 0) = AD_LWBD(ad, 0) = 0;
  AD_NUMELM(ad) = AD_UPBD(ad, 0) = AD_UPAST(ad, 0) = AD_EXTNTAST(ad, 0) =
      mk_isz_cval(no, astb.bnd.dtype);
  AD_DEFER(ad) = 0;
  AD_NOBOUNDS(ad) = 0;

  dinit_put(DINIT_LOC, (INT)arr);

  dtype = DDTG(DTYPEG(arr));

  for (i = 0; i < no; i++) {
    if (DTY(DT_INT) == TY_INT8) {
      INT val[2];
      val[0] = 0;
      val[1] = permute[i];
      dinit_put(dtype, getcon(val, DT_INT8));
    } else
      dinit_put(dtype, permute[i]);
  }
  dinit_put(DINIT_END, 0);
  DINITP(arr, 1);
  sym_is_refd(arr);

  return mk_id(arr);
}

/*This routine calculates permute of rhs based on lhs
 * for example, lhs(i,2, j,k) rhs(3,k,i,j) then
 * permute will be /0,3,1,2/
 */
static void
compute_permute(int lhs, int rhs, int list, int order[7])
{
  int asd, ndim;
  int i, j;
  int count, count1;
  int order1[7];
  LOGICAL found;
  int astli, nidx;
  int iloc;

  for (j = 0; j < 7; j++)
    order[j] = 0;

  assert(!is_duplicate(lhs, list), "compute_permute:something is wrong", lhs,
         3);

  /* rhs */
  asd = A_ASDG(rhs);
  ndim = ASD_NDIM(asd);
  count = 0;
  for (j = 0; j < ndim; ++j) {
    order[j] = 0;
    astli = 0;
    nidx = 0;
    search_forall_idx(ASD_SUBS(asd, j), list, &astli, &nidx);
    if (nidx == 1 && astli) {
      order[j] = ASTLI_SPTR(astli);
      count++;
    }
  }

  /* lhs */
  asd = A_ASDG(lhs);
  ndim = ASD_NDIM(asd);
  count1 = 0;
  for (j = 0; j < ndim; ++j) {
    astli = 0;
    nidx = 0;
    search_forall_idx(ASD_SUBS(asd, j), list, &astli, &nidx);
    if (nidx == 1 && astli) {
      order1[count1] = ASTLI_SPTR(astli);
      count1++;
    }
  }

  asd = A_ASDG(rhs);
  ndim = ASD_NDIM(asd);
  for (j = 0; j < ndim; j++) {
    if (order[j] == 0)
      continue;
    found = FALSE;
    for (i = 0; i < count1; i++) {
      if (order1[i] == order[j]) {
        found = TRUE;
        iloc = i + 1;
      }
    }
    assert(found, "compute_permute:something is wrong", lhs, 3);
    order[j] = iloc;
  }
}

static LOGICAL
is_permuted(int array, int per[7], int per1[7], int *nper1)
{
  int asd;
  int ndim;
  int count;
  int i;
  LOGICAL permuted;

  assert(A_TYPEG(array) == A_SUBSCR, "is_permuted: something is wrong", array,
         2);

  asd = A_ASDG(array);
  ndim = ASD_NDIM(asd);
  count = 0;
  for (i = 0; i < ndim; i++) {
    if (per[i]) {
      per1[count] = per[i];
      count++;
    }
  }

  permuted = FALSE;
  for (i = 0; i < count; i++) {
    if (per1[i] != (i + 1))
      permuted = TRUE;
  }

  *nper1 = count;
  return permuted;
}

static void
emit_sum_scatterx(int std)
{
  int sptr;
  int asd1;
  int ndim1;
  int ast1;
  int subs[7];
  int astnew;
  int tempast;
  int argt, nargs;
  int i, j;
  int forall;
  int list;
  int vflag, pflag;
  int vdim, pdim;
  int nvec;
  int secv;
  ADSC *ad;
  int glb, gub;
  int asn;
  int mask;
  int result_sec, base_sec, array_sec, mask_sec;
  int result, newresult;
  int base;
  int array;
  int func;
  int permute[7];
  int npermute;
  char name[40];
  int function, operator;
  int sectflag;

  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);

  sectflag = 0;

  mask = comminfo.scat.mask;
  result = comminfo.scat.result;
  base = comminfo.scat.base;
  array = comminfo.scat.array;
  operator= comminfo.scat.operator;
  function = comminfo.scat.function;
  if (!base)
    return;

  if (!comminfo.scat.array_simple) {
    int sptrtemp, newforall, asn, newstd, newarray;
    struct comminfo savecomminfo;
    sptrtemp = get_temp_forall(forall, base, std, std, 0, array);
    newarray = simple_reference_for_temp(sptrtemp, base, forall);
    /* assign temp from nonsimple array */
    newforall = mk_stmt(A_FORALL, 0);
    A_LISTP(newforall, A_LISTG(forall));
    A_SRCP(newforall, A_SRCG(forall));
    asn = mk_stmt(A_ASN, 0);
    A_DESTP(asn, newarray);
    A_SRCP(asn, array);
    A_IFSTMTP(newforall, asn);
    newstd = add_stmt_before(newforall, std);
    array = newarray;
    savecomminfo = comminfo;
    process_forall(newstd);
    transform_forall(newstd, newforall);
    comminfo = savecomminfo;
  }

  sptr = sptr_of_subscript(result);
  list = A_LISTG(forall);
  asd1 = A_ASDG(result);
  ndim1 = ASD_NDIM(asd1);

  vflag = 0;
  vdim = 0;
  nvec = 0;
  j = 0;
  for (i = 0; i < ndim1; i++) {
    subs[i] = ASD_SUBS(asd1, i);
    if (is_scalar(ASD_SUBS(asd1, i), list))
      continue;
    if (is_vector_subscript(ASD_SUBS(asd1, i), list)) {
      ad = AD_DPTR(DTYPEG(sptr));
      glb = AD_LWAST(ad, i);
      gub = AD_UPAST(ad, i);
      subs[i] = mk_isz_cval(1, astb.bnd.dtype);
      vflag |= 1 << j;
      vdim |= 1 << i;
      nvec++;
    }
    j++;
  }

  permute_axis(result, array, list, permute);

  npermute = 0;
  pflag = 0;
  pdim = 0;
  j = 0;
  for (i = 0; i < ndim1; i++) {
    if (is_scalar(ASD_SUBS(asd1, i), list))
      continue;
    if (permute[i]) {
      pflag |= 1 << j;
      pdim |= 1 << i;
      npermute++;
    }
    j++;
  }

  if (nvec == ndim1)
    result_sec = DESCRG(sptr);
  else {
    newresult = mk_subscr(A_LOPG(result), subs, ndim1, A_DTYPEG(result));
    astnew = forall_2_sec(newresult, forall);
    /* change astnew for vector dimension */
    ad = AD_DPTR(DTYPEG(sptr_of_subscript(astnew)));
    asd1 = A_ASDG(astnew);
    ndim1 = ASD_NDIM(asd1);
    for (i = 0; i < ndim1; i++) {
      subs[i] = ASD_SUBS(asd1, i);
      if (getbit(vdim, i)) {
        glb = AD_LWAST(ad, i);
        gub = AD_UPAST(ad, i);
        subs[i] = mk_triple(glb, gub, 0);
      }
    }
    astnew = mk_subscr(A_LOPG(astnew), subs, ndim1, A_DTYPEG(astnew));
    result_sec = make_sec_from_ast(astnew, std, std, 0, sectflag | NOTSECTFLAG);
  }

  base_sec = result_sec;

  tempast = forall_2_sec(array, forall);
  array_sec = make_sec_from_ast(tempast, std, std, 0, sectflag);

  if (mask) {
    mask = forall_2_sec(mask, forall);
    mask_sec = make_sec_from_ast(mask, std, std, 0, sectflag);
    mask = A_LOPG(mask);
    mask_sec = mk_id(mask_sec);
  } else {
    mask = mk_cval(1, DT_LOG);
    mask_sec = mk_cval(dtype_to_arg(A_DTYPEG(mask)), DT_INT);
  }

  nargs = 2 * 4 + 1 + 1 + 2 * nvec + npermute;
  argt = mk_argt(nargs);

  ARGT_ARG(argt, 0) = A_LOPG(result);
  DESCUSEDP(sptr, 1);
  ARGT_ARG(argt, 1) = A_LOPG(array);
  ARGT_ARG(argt, 2) = A_LOPG(base);
  ARGT_ARG(argt, 3) = mask;

  /* sections */
  ARGT_ARG(argt, 4) = check_member(result, mk_id(result_sec));
  ARGT_ARG(argt, 5) = check_member(array, mk_id(array_sec));
  ARGT_ARG(argt, 6) = check_member(base, mk_id(base_sec));
  ARGT_ARG(argt, 7) = mask_sec;

  ARGT_ARG(argt, 8) = mk_cval(vflag, DT_INT);
  ARGT_ARG(argt, 9) = mk_cval(pflag, DT_INT);
  j = 10;
  asd1 = A_ASDG(result);
  ndim1 = ASD_NDIM(asd1);
  for (i = 0; i < ndim1; i++) {
    if (!is_scalar(ASD_SUBS(asd1, i), list) &&
        is_vector_subscript(ASD_SUBS(asd1, i), list)) {
      astnew = forall_2_sec(ASD_SUBS(asd1, i), forall);
      secv = make_sec_from_ast(astnew, std, std, 0, sectflag);
      ARGT_ARG(argt, j) = A_LOPG(ASD_SUBS(asd1, i));
      j++;
      ARGT_ARG(argt, j) = mk_id(secv);
      j++;
    }
    if (permute[i]) {
      ARGT_ARG(argt, j) = permute[i];
      j++;
    }
  }
  ast1 = mk_stmt(A_CALL, 0);

  func = 0;
  strcpy(name, "");
  if (operator) {
    switch (operator) {
    case OP_ADD:
      strcpy(name, mkRteRtnNm(RTE_sum_scatterx));
      break;
    case OP_MUL:
      strcpy(name, mkRteRtnNm(RTE_product_scatterx));
      break;
    case OP_LOR:
      strcpy(name, mkRteRtnNm(RTE_any_scatterx));
      break;
    case OP_LAND:
      strcpy(name, mkRteRtnNm(RTE_all_scatterx));
      break;
    case OP_LNEQV:
      strcpy(name, mkRteRtnNm(RTE_parity_scatterx));
      break;
    }
  }
  if (function) {
    switch (function) {
    case I_MAX:
      strcpy(name, mkRteRtnNm(RTE_maxval_scatterx));
      break;
    case I_MIN:
      strcpy(name, mkRteRtnNm(RTE_minval_scatterx));
      break;
    case I_IAND:
      strcpy(name, mkRteRtnNm(RTE_iall_scatterx));
      break;
    case I_IOR:
      strcpy(name, mkRteRtnNm(RTE_iany_scatterx));
      break;
    case I_IEOR:
      strcpy(name, mkRteRtnNm(RTE_iparity_scatterx));
      break;
    }
  }

  assert(strcmp(name, ""), "emit_sum_scatterx: something is wrong", std, 2);
  func = mk_id(sym_mkfunc(name, DT_NONE));
  A_LOPP(ast1, func);
  A_ARGCNTP(ast1, nargs);
  A_ARGSP(ast1, argt);
  add_stmt_before(ast1, std);
  NODESCP(memsym_of_ast(A_LOPG(ast1)), 1);
  STD_DELETE(std) = 1;
}

static void
emit_scatterx(int std)
{
  int mask;
  int result;
  int array;
  int base;

  mask = comminfo.scat.mask;
  result = comminfo.scat.result;
  array = comminfo.scat.array;
  base = comminfo.scat.base;

  if (base)
    return;

  emit_scatterx_gatherx(std, result, array, mask, 0, 0, 0, A_HSCATTER);

  STD_DELETE(std) = 1;
}

static void
emit_scatterx_gatherx(int std, int result, int array, int mask, int allocstd,
                      int tempast0, int lhssec, int comm_type)
{
  int sptr, dest, lop;
  int asd1;
  int ndim1;
  int ast1;
  int subs[7];
  int astnew;
  int argt;
  int i, j;
  int forall;
  int list;
  int vflag, pflag;
  int pdim, vdim;
  int nvec;
  ADSC *ad;
  int glb, gub;
  int asn;
  int func;
  int permute[7];
  int npermute;
  int nd;
  int header;
  int vsub = 0, nvsub, newvsub;
  int commstd;
  int cp, xfer;
  int startstd;
  int ast;
  int v, sectvstd;
  int vsubstd, nvsubstd, maskstd;
  int lhs;
  int mask_id;
  int sectflag;
  INDEX_REUSE *irp;
  NEWVAR *nv;
  LOGICAL index_reuse;
  int index_reuse_condvar;
  int ifstd;

  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);
  lhs = A_DESTG(asn);
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);

  sectflag = 0;

  if (comm_type == A_HGATHER) {
    vsub = array;
    nvsub = result;
    func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_gatherx), DT_ADDR));
  } else if (comm_type == A_HSCATTER) {
    vsub = result;
    nvsub = array;
    func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_scatterx), DT_ADDR));
  }

  sptr = memsym_of_ast(vsub);
  list = A_LISTG(forall);
  asd1 = A_ASDG(vsub);
  ndim1 = ASD_NDIM(asd1);

  index_reuse = FALSE;
  open_dynpragma(std, STD_LINENO(std));
  for (irp = direct.index_reuse_list; irp; irp = irp->next) {
    for (nv = irp->reuse_list; nv; nv = nv->next) {
      if (sptr == nv->var) {
        index_reuse = TRUE;
        index_reuse_condvar = irp->condvar;
        goto found_index_reuse;
      }
    }
  }
found_index_reuse:
  close_pragma();

  vflag = 0;
  vdim = 0;
  nvec = 0;
  j = 0;
  for (i = 0; i < ndim1; i++) {
    subs[i] = ASD_SUBS(asd1, i);
    if (is_scalar(ASD_SUBS(asd1, i), list))
      continue;
    if (is_vector_subscript(ASD_SUBS(asd1, i), list)) {
      ad = AD_DPTR(DTYPEG(sptr));
      glb = AD_LWAST(ad, i);
      gub = AD_UPAST(ad, i);
      subs[i] = mk_isz_cval(1, astb.bnd.dtype);
      vflag |= 1 << j;
      vdim |= 1 << i;
      nvec++;
    }
    j++;
  }

  permute_axis(vsub, nvsub, list, permute);

  npermute = 0;
  pflag = 0;
  pdim = 0;
  j = 0;
  for (i = 0; i < ndim1; i++) {
    if (is_scalar(ASD_SUBS(asd1, i), list))
      continue;
    if (permute[i]) {
      pflag |= 1 << j;
      pdim |= 1 << i;
      npermute++;
    }
    j++;
  }

  newvsub = mk_subscr(A_LOPG(vsub), subs, ndim1, A_DTYPEG(vsub));
  astnew = forall_2_sec(newvsub, forall);
  /* change astnew for vector dimension */
  ad = AD_DPTR(DTYPEG(memsym_of_ast(astnew)));
  asd1 = A_ASDG(astnew);
  ndim1 = ASD_NDIM(asd1);
  for (i = 0; i < ndim1; i++) {
    subs[i] = ASD_SUBS(asd1, i);
    if (getbit(vdim, i)) {
      glb = AD_LWAST(ad, i);
      gub = AD_UPAST(ad, i);
      subs[i] = mk_triple(glb, gub, 0);
    }
  }
  newvsub = mk_subscr(A_LOPG(astnew), subs, ndim1, DTYPEG(sptr));
  vsubstd = make_sec_ast(newvsub, std, 0, sectflag | NOREINDEX);

  nvsub = forall_2_sec(nvsub, forall);
  nvsubstd = make_sec_ast(nvsub, std, allocstd, sectflag);

  if (mask && !comminfo.mask_phase) {
    mask = forall_2_sec(mask, forall);
    maskstd = make_sec_ast(mask, std, 0, sectflag);
    mask_id = mk_id(memsym_of_ast(mask));
  } else {
    mask = 0;
    mask_id = 0;
    maskstd = 0;
  }

  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  ast = new_node(A_HGATHER);
  A_SRCP(ast, A_LOPG(result));
  A_SDESCP(ast, 0);
  A_DESTP(ast, A_LOPG(array));
  A_DDESCP(ast, 0);
  A_MASKP(ast, mask_id);
  A_MDESCP(ast, 0);
  A_BVECTP(ast, 0);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_CGATHER_VSUB(nd) = newvsub;
  FT_CGATHER_NVSUB(nd) = nvsub;
  FT_CGATHER_MASK(nd) = mask;
  FT_CGATHER_SECTVSUB(nd) = vsubstd;
  FT_CGATHER_SECTNVSUB(nd) = nvsubstd;
  FT_CGATHER_SECTM(nd) = maskstd;
  FT_CGATHER_ALLOC(nd) = allocstd;
  FT_CGATHER_FREE(nd) = header;
  FT_CGATHER_REUSE(nd) = 0;
  FT_CGATHER_INDEXREUSE(nd) = index_reuse;
  FT_CGATHER_USELHS(nd) = 0;
  FT_CGATHER_LHS(nd) = lhs;
  FT_CGATHER_RHS(nd) = array;
  FT_CGATHER_SAME(nd) = 0;
  FT_CGATHER_VFLAG(nd) = vflag;
  FT_CGATHER_PFLAG(nd) = pflag;
  FT_CGATHER_VDIM(nd) = vdim;
  FT_CGATHER_PDIM(nd) = pdim;
  FT_CGATHER_NVEC(nd) = nvec;
  FT_CGATHER_NPER(nd) = npermute;
  FT_CGATHER_TYPE(nd) = comm_type;
  FT_CGATHER_LHSSEC(nd) = lhssec;
  FT_CGATHER_NOTLHS(nd) = (lhssec) ? 0 : 1;
  j = 8;
  asd1 = A_ASDG(vsub);
  ndim1 = ASD_NDIM(asd1);
  for (i = 0; i < ndim1; i++) {
    FT_CGATHER_SECTV(nd, i) = 0;
    FT_CGATHER_V(nd, i) = 0;
    FT_CGATHER_PERMUTE(nd, i) = 0;
    if (!is_scalar(ASD_SUBS(asd1, i), list) &&
        is_vector_subscript(ASD_SUBS(asd1, i), list)) {
      v = forall_2_sec(ASD_SUBS(asd1, i), forall);
      sectvstd = make_sec_ast(v, std, 0, sectflag);
      v = ASD_SUBS(asd1, i);
      FT_CGATHER_SECTV(nd, i) = sectvstd;
      assert(A_TYPEG(v) == A_SUBSCR,
             "emit_scatterx_gatherx: non-subscript in gather", A_TYPEG(v), 4);
      FT_CGATHER_V(nd, i) = A_LOPG(v);
    }
    if (permute[i]) {
      FT_CGATHER_PERMUTE(nd, i) = permute[i];
    }
  }

  A_OPT1P(ast, nd);
  cp = sym_get_cp();
  FT_CGATHER_OUT(nd) = cp;
  dest = mk_id(cp);
  A_DESTP(asn, dest);
  A_SRCP(asn, ast);

  if (index_reuse) {
    /*
     * 'vsub appeared in a JAHPF INDEX_REUSE directive:
     * 	!hpfj index_reuse [(<condition>)] vsub...
     *
     * Enclose the 'pghpf_comm_gatherx/scatterx' call in a
     * conditional as follows:
     * (i) if no <condition> is specified:
     *
     * 	if (cp == 0) then
     * 	  cp = pghpf_comm_gatherx/scatterx(...)
     * 	endif
     *
     * (ii) if <condition> is specified:
     *
     * 	if (cp == 0 .or. .not. <condition>) then
     * 	  if (cp /= 0) then
     * 	    call pghpf_comm_free(1,cp)
     * 	  endif
     * 	  cp = pghpf_comm_gatherx/scatterx(...)
     * 	endif
     */
    SAVEP(cp, 1);
    ast = mk_stmt(A_IFTHEN, 0);
    ast1 = mk_binop(OP_EQ, mk_id(cp), mk_convert(astb.i0, DT_ADDR), DT_LOG);
    if (index_reuse_condvar) {
      ast1 = mk_binop(OP_LOR, ast1,
                      mk_unop(OP_LNOT, mk_id(index_reuse_condvar), DT_LOG),
                      DT_LOG);
    }
    A_IFEXPRP(ast, ast1);
    ifstd = add_stmt_before(ast, header);
    A_STDP(ast, ifstd);

    if (index_reuse_condvar) {
      int predicate = mk_binop(OP_NE, mk_id(cp), mk_convert(astb.i0, DT_ADDR),
                               DT_LOG);
      int func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_free), DT_NONE));
      ast = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(ast, predicate);
      ifstd = add_stmt_before(ast, header);
      A_STDP(ast, ifstd);

      argt = mk_argt(2);
      ARGT_ARG(argt, 0) = astb.i1;
      ARGT_ARG(argt, 1) = mk_id(cp);
      ast = mk_stmt(A_CALL, 0);
      A_LOPP(ast, func);
      NODESCP(A_SPTRG(A_LOPG(ast)), 1);
      A_ARGCNTP(ast, 2);
      A_ARGSP(ast, argt);
      ifstd = add_stmt_before(ast, header);
      A_STDP(ast, ifstd);

      ast = mk_stmt(A_ENDIF, 0);
      ifstd = add_stmt_before(ast, header);
      A_STDP(ast, ifstd);
    }
  }

  commstd = add_stmt_before(asn, header);
  A_STDP(asn, commstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), commstd);
  FT_NRT(nd)++;

  if (index_reuse) {
    ast = mk_stmt(A_ENDIF, 0);
    ifstd = add_stmt_before(ast, header);
    A_STDP(ast, ifstd);
  }

  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  ast = new_node(A_HCSTART);
  lop = mk_id(cp);
  A_LOPP(ast, lop);
  A_SRCP(ast, array);
  A_DESTP(ast, result);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_CSTART_COMM(nd) = commstd;
  FT_CSTART_RHS(nd) = array;
  FT_CSTART_USEDSTD(nd) = comminfo.usedstd;
  xfer = sym_get_xfer();
  FT_CSTART_OUT(nd) = xfer;
  FT_CSTART_SECTL(nd) = vsubstd;
  FT_CSTART_SECTR(nd) = nvsubstd;
  FT_CSTART_ALLOC(nd) = allocstd;
  FT_CSTART_FREE(nd) = header;
  FT_CSTART_REF(nd) = tempast0;
  FT_CSTART_TYPE(nd) = comm_type;
  FT_CSTART_REUSE(nd) = 0;
  FT_CSTART_INVMVD(nd) = 0;
  FT_CSTART_USELHS(nd) = 0;
  FT_CSTART_SAME(nd) = 0;
  A_OPT1P(ast, nd);
  dest = mk_id(xfer);
  A_DESTP(asn, dest);
  A_SRCP(asn, ast);

  startstd = add_stmt_before(asn, header);
  A_STDP(asn, startstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), startstd);
  FT_NRT(nd)++;
}

static int
emit_gatherx(int a, int std, LOGICAL opt)
{
  int sptr;
  int asd1;
  int ndim1;
  int tempast, tempast0;
  int forall;
  int list;
  int lhs;
  int mask;
  int nd, header;
  int allocast, allocstd;
  int sectlstd;
  int lhssec;
  int sectflag;
  LOGICAL independent;

  forall = STD_AST(std);
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  lhs = comminfo.sub;
  mask = A_IFEXPRG(forall);
  list = A_LISTG(forall);
  asd1 = A_ASDG(a);
  ndim1 = ASD_NDIM(asd1);

  sectflag = 0;

  open_dynpragma(std, STD_LINENO(std));
  independent = (flg.x[19] & 0x100) != 0;
  close_pragma();

  sectlstd = 0;
  lhssec = 0;
  if (is_use_lhs(a, FALSE, independent, std)) {
    sptr = memsym_of_ast(comminfo.sub);
    tempast = lhs;
    lhssec = tempast = forall_2_sec(tempast, forall);
    sectlstd = make_sec_ast(tempast, std, 0, sectflag | NOREINDEX);
    nd = A_OPT1G(forall);
    FT_SECTL(nd) = sectlstd;
  }

  sptr = temp_gatherx(std, forall, lhs, lhs, DTY(DTYPEG(memsym_of_ast(a)) + 1),
                      &allocast);
  tempast0 = tempast = gatherx_temp_before(sptr, lhs, forall);

  allocstd = add_stmt_before(allocast, header);
  A_STDP(allocast, allocstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), allocstd);
  FT_NRT(nd)++;

  emit_scatterx_gatherx(std, tempast, a, mask, allocstd, tempast0, lhssec,
                        A_HGATHER);
  return a;
}

/* Algorithm:
 * This will choose the largest overlap shift at each dimension
 * among the same array in the set.
 * Store overlap_shift value in array symbol table.
 * mark the all OVERLAP as  NO_COMM but the first one.
 */
static void
opt_overlap(void)
{
  int i;
  int arr, arr1;
  int subinfo1, ndim;
  int subinfo;
  int align;
  int sptr, sptr1;

  /* Now compute the total overlap-shift for each separate array symbol */
  for (arr = trans.rhsbase; arr != 0; arr = ARREF_NEXT(arr)) {
    if (ARREF_CLASS(arr) != OVERLAP)
      continue;
    align = ALIGNG(ARREF_ARRSYM(arr));
    for (arr1 = arr; arr1 != 0; arr1 = ARREF_NEXT(arr1)) {
      sptr = ARREF_ARRSYM(arr);
      sptr1 = ARREF_ARRSYM(arr1);
      if (ARREF_ARRSYM(arr1) != ARREF_ARRSYM(arr))
        continue;
      /* find out shift values and store union of them into subinfo */
      subinfo = ARREF_SUB(arr);
      subinfo1 = ARREF_SUB(arr1);
      ndim = ARREF_NDIM(arr1);
      for (i = 0; i < ndim; ++i) {
        int v;
        if (SUBI_COMMT(subinfo1 + i) != COMMT_SHIFTC)
          continue;
        if ((v = SUBI_COMMV(subinfo1 + i)) < 0) {
          v = -v;
          if (v > SUBI_NOP(subinfo + i)) {
            SUBI_NOP(subinfo + i) = v;
            SUBI_NOP(subinfo1 + i) = v;
          }

        } else {
          if (v > SUBI_POP(subinfo + i)) {
            SUBI_POP(subinfo + i) = v;
            SUBI_POP(subinfo1 + i) = v;
          }
        }
      }

      if (flg.ipa) {
        /* allow common block overlap increase */
        if ((ARGG(sptr1) && SCG(sptr) != SC_CMBLK) || SCG(sptr1) == SC_DUMMY)
          continue;
      } else {
        if (ARGG(sptr1) || SCG(sptr1) == SC_DUMMY || SCG(sptr) == SC_CMBLK)
          continue;
      }

      ARREF_FLAG(arr1) = 2;
      subinfo = ARREF_SUB(arr);
      subinfo1 = ARREF_SUB(arr1);
      ndim = ARREF_NDIM(arr1);
    }
  }
}

static void
emit_overlap(int a)
{
  int align, sdesc, dest, lop;
  int arr;
  int asd, ndim;
  int astnew;
  int asn;
  int i;
  int startstd;
  int commstd;
  int cp, xfer;
  int nd;
  int sptr;
  int subs[7];
  int forall;
  int std;
  int ns, ps;
  int ast;
  int header;
  int subinfo;
  int arref;

  std = comminfo.std;
  forall = STD_AST(std);
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  /* put out the shift call for this symbol */
  arr = A_LOPG(a);
  sptr = memsym_of_ast(arr);
  align = ALIGNG(sptr);
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  arref = A_RFPTRG(a);
  subinfo = ARREF_SUB(arref);

  DESCUSEDP(sptr, 1);
  for (i = 0; i < ndim; ++i) {
    ns = mk_isz_cval(SUBI_NOP(subinfo + i), astb.bnd.dtype);
    ps = mk_isz_cval(SUBI_POP(subinfo + i), astb.bnd.dtype);
    subs[i] = mk_triple(ps, ns, 0);
  }
  astnew = mk_subscr(arr, subs, ndim, DTYPEG(sptr));

  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  ast = new_node(A_HOVLPSHIFT);
  A_SRCP(ast, astnew);
  sdesc = check_member(arr, mk_id(DESCRG(sptr)));
  A_SDESCP(ast, sdesc);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_SHIFT_RHS(nd) = a;
  FT_SHIFT_FREE(nd) = header;
  FT_SHIFT_REUSE(nd) = 0;
  FT_SHIFT_SAME(nd) = 0;
  FT_SHIFT_TYPE(nd) = 0;
  FT_SHIFT_BOUNDARY(nd) = 0;
  A_OPT1P(ast, nd);
  cp = sym_get_cp();
  FT_SHIFT_OUT(nd) = cp;
  dest = mk_id(cp);
  A_DESTP(asn, dest);
  A_SRCP(asn, ast);

  commstd = add_stmt_before(asn, header);
  A_STDP(asn, commstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), commstd);
  FT_NRT(nd)++;

  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  ast = new_node(A_HCSTART);
  lop = mk_id(cp);
  A_LOPP(ast, lop);
  A_SRCP(ast, astnew);
  A_DESTP(ast, astnew);
  nd = mk_ftb();
  FT_STD(nd) = std;
  FT_FORALL(nd) = forall;
  FT_CSTART_COMM(nd) = commstd;
  FT_CSTART_RHS(nd) = a;
  FT_CSTART_USEDSTD(nd) = comminfo.usedstd;
  xfer = sym_get_xfer();
  FT_CSTART_OUT(nd) = xfer;
  FT_CSTART_SECTL(nd) = 0;
  FT_CSTART_SECTR(nd) = 0;
  FT_CSTART_ALLOC(nd) = 0;

  FT_CSTART_FREE(nd) = header;
  FT_CSTART_REF(nd) = 0;
  FT_CSTART_TYPE(nd) = A_HOVLPSHIFT;
  FT_CSTART_REUSE(nd) = 0;
  FT_CSTART_INVMVD(nd) = 0;
  FT_CSTART_USELHS(nd) = 0;
  FT_CSTART_SAME(nd) = 0;
  A_OPT1P(ast, nd);
  dest = mk_id(xfer);
  A_DESTP(asn, dest);
  A_SRCP(asn, ast);

  startstd = add_stmt_before(asn, header);
  A_STDP(asn, startstd);
  nd = A_OPT1G(forall);
  plist(FT_RTL(nd), startstd);
  FT_NRT(nd)++;
}

static CTYPE *
getcyclic(void)
{
  int i;
  CTYPE *ct;
  ct = (CTYPE *)getitem(FORALL_AREA, sizeof(CTYPE));
  ct->lhs = 0;
  ct->ifast = 0;
  ct->endifast = 0;
  ct->inner_cyclic = clist();
  for (i = 0; i < 7; i++) {
    ct->c_lof[i] = 0;
    ct->c_dupl[i] = 0;
    ct->idx[i] = 0;
    ct->cb_init[i] = 0;
    ct->cb_do[i] = 0;
    ct->cb_block[i] = 0;
    ct->cb_inc[i] = 0;
    ct->cb_enddo[i] = 0;
    ct->c_init[i] = 0;
    ct->c_inc[i] = 0;
  }
  return ct;
}

static int
shape_comm_in_expr(int expr, int forall, int std, int nomask)
{
  int l, r, d, o;
  int i, nargs, argt;
  int lhs, sptr;

  if (expr == 0)
    return expr;
  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = shape_comm_in_expr(A_LOPG(expr), forall, std, nomask);
    r = shape_comm_in_expr(A_ROPG(expr), forall, std, nomask);
    if (l == A_LOPG(expr) && r == A_ROPG(expr))
      return expr;
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = shape_comm_in_expr(A_LOPG(expr), forall, std, nomask);
    if (l == A_LOPG(expr))
      return expr;
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(expr);
    l = shape_comm_in_expr(A_LOPG(expr), forall, std, nomask);
    if (l == A_LOPG(expr))
      return expr;
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(expr);
    l = shape_comm_in_expr(A_LOPG(expr), forall, std, nomask);
    if (l == A_LOPG(expr))
      return expr;
    return mk_paren(l, d);
  case A_SUBSTR:
    return expr;
  case A_INTR:
  case A_FUNC:
    /* size & present intrinsics do not need the array content,
     * no need to communicate
     */
    o = A_OPTYPEG(expr);
    if (o == I_SIZE || o == I_PRESENT)
      return expr;
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) =
          shape_comm_in_expr(ARGT_ARG(argt, i), forall, std, nomask);
    }
    return expr;
  case A_CNST:
  case A_CMPLXC:
    return expr;
  case A_MEM:
    if (!A_SHAPEG(expr))
      return expr;
    sptr = A_SPTRG(A_MEMG(expr));
    r = A_MEMG(expr);
    d = A_DTYPEG(r);
    l = shape_comm_in_expr(A_PARENTG(expr), forall, std, nomask);
    if (l == A_PARENTG(expr))
      return expr;
    return mk_member(l, r, d);
  case A_ID:
  case A_SUBSCR:
    if (!A_SHAPEG(expr))
      return expr;
    lhs = A_DESTG(A_IFSTMTG(forall));
    expr = convert_subscript(expr);
    return expr;
  default:
    interr("shape_comm_in_expr: unknown expression", expr, 2);
    return expr;
  }
}

static void
shape_communication(int std, int forall)
{
  int nd;
  int i;
  int cstd;
  int expr;
  int asn;
  int rhs;

  /* handle shape communication at forall first a(i) = pure_func(b) */
  expr = A_IFEXPRG(forall);
  asn = A_IFSTMTG(forall);
  rhs = A_SRCG(asn);
  rhs = shape_comm_in_expr(rhs, forall, std, 1);
  expr = shape_comm_in_expr(expr, forall, std, 1);
  A_SRCP(asn, rhs);
  A_IFEXPRP(forall, expr);

  /* handle shape communication at calls second */
  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    shape_comm(cstd, std, forall);
  }

  for (i = 0; i < FT_NSCALL(nd); i++) {
    cstd = glist(FT_SCALL(nd), i);
    shape_comm(cstd, std, forall);
  }
}

static void
shape_comm(int cstd, int fstd, int forall)
{
  int ast, ast1;
  int cstd1;
  int nd, nd1;
  int i;
  int nargs, argt;
  int lhs;
  int arg;

  ast = STD_AST(cstd);
  nd = A_OPT1G(ast);
  assert(nd, "call_comm: something is wrong", ast, 3);
  for (i = 0; i < FT_CALL_NCALL(nd); i++) {
    cstd1 = glist(FT_CALL_CALL(nd), i);
    ast1 = STD_AST(cstd1);
    nd1 = A_OPT1G(ast1);
    assert(nd1, "put_calls: something is wrong", ast1, 3);
    shape_comm(cstd1, fstd, forall);
  }
  nargs = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);
  for (i = 0; i < nargs; ++i) {
    arg = ARGT_ARG(argt, i);
    if (!A_SHAPEG(arg))
      continue;
    lhs = A_DESTG(A_IFSTMTG(forall));
    assert(A_TYPEG(arg) == A_SUBSCR || A_TYPEG(arg) == A_ID ||
               A_TYPEG(arg) == A_MEM,
           "shape_comm: array expression is not supported", arg, 3);
    arg = convert_subscript(arg);
  }
}

#ifdef FLANG_COMM_UNUSED
/* The function of this routine is to handle communication of arg.
 * This arg is from PURE function and it has shape. It will try to
 * bring to lhs of forall. Distribution of TMP will be based on LHS.
 * However, the size and shape of TMP will be based on both LHS and arg.
 * There are three rules for TMP:
 *        1-) heading dimensions size and distribution from LHS
 *        2-) trailing dimensions size from shape of arg with no distribution
 *        3-) remove idx from forall list if it does not appear at arg or mask
 * For example: (assume that a, b have different distributions.
 *      forall(i=1:n) a(i)= sum(b(i,iloc(i),:))
 * will be
 *      forall(i=1:n) tmp(i,:) =b(i,iloc(i),:)
 *      forall(i=1:n) a(i) = sum(tmp(i,:))
 * There will be no communication for tmp which becomes new arg of PURE.
 * is_pure_temp_too_large() decides whether tmp will have more dimension than
 * arg. if it is, tmp will be replication of arg.
 */
static int
gen_shape_comm(int arg, int forall, int std, int nomask)
{
  int newforall;
  int newstd;
  int sptr;
  int asn;
  int tmpast;
  int lhs;
  int mask;
  int list;
  int olist;
  int ast;
  int shape;
  int nd;
  int header;

  if (!A_SHAPEG(arg))
    return arg;
  lhs = A_DESTG(A_IFSTMTG(forall));
  olist = A_LISTG(forall);
  mask = A_IFEXPRG(forall);
  if (nomask)
    mask = 0;
  nd = A_OPT1G(forall);
  assert(nd, "gen_shape_comm: something is wrong", forall, 3);
  header = FT_HEADER(nd);
  list = construct_list_for_pure(arg, mask, olist);
  if (is_pure_temp_too_large(list, arg)) {
    tmpast = handle_pure_temp_too_large(arg, header);
    return tmpast;
  }
  /* put new list to forall for short time to trick
   * get_temp_pure() and reference_for_pure_temp()
   */
  A_LISTP(forall, list);
  /* create a pure temp */
  sptr = get_temp_pure(forall, lhs, arg, header, header, arg);
  tmpast = reference_for_pure_temp(sptr, lhs, arg, forall);
  /* put original list back to forall */
  A_LISTP(forall, olist);

  asn = mk_stmt(A_ASN, DTYPEG(sptr));
  A_DESTP(asn, tmpast);
  A_SRCP(asn, arg);

  if (list) {
    newforall = mk_stmt(A_FORALL, 0);
    A_LISTP(newforall, list);
    A_IFSTMTP(newforall, asn);
    A_IFEXPRP(newforall, mask);
  } else {
    shape = A_SHAPEG(tmpast);
    newforall = make_forall(shape, tmpast, 0, 0);
    ast = normalize_forall(newforall, asn, 0);
    A_IFSTMTP(newforall, ast);
    A_IFEXPRP(newforall, 0);
  }
  newforall = rename_forall_list(newforall);
  newstd = add_stmt_before(newforall, header);
  process_forall(newstd);

  newforall = STD_AST(newstd);
  transform_forall(newstd, newforall);
  return tmpast;
}

/* construct a new list based on old list
 * which must appear arg or mask expression
 */
static int
construct_list_for_pure(int arg, int mask, int list)
{
  int newlist;
  int isptr;
  int j;

  start_astli();
  for (j = list; j != 0; j = ASTLI_NEXT(j)) {
    isptr = ASTLI_SPTR(j);
    if (is_name_in_expr(arg, isptr) || is_name_in_expr(mask, isptr)) {
      /* include this one */
      newlist = add_astli();
      ASTLI_SPTR(newlist) = ASTLI_SPTR(j);
      ASTLI_TRIPLE(newlist) = ASTLI_TRIPLE(j);
    }
  }
  return ASTLI_HEAD;
}

/* This will find temp_reference for pure communication.
 *  lhs=a(i,j,2), arg=b(2,i,:) will be tmp=tmp(i,j,:)
 * heading dimension from lhs, trailing from arg.
 */
static int
reference_for_pure_temp(int sptr, int lhs, int arg, int forall)
{
  int subs[7];
  int list;
  int i, j;
  int asd;
  int ndim;
  int astnew;
  int shape;
  int sdim;

  list = A_LISTG(forall);
  asd = A_ASDG(lhs);
  ndim = ASD_NDIM(asd);
  j = 0;
  for (i = 0; i < ndim; i++) {
    if (search_forall_var(ASD_SUBS(asd, i), list)) {
      /* include this dimension */
      subs[j] = ASD_SUBS(asd, i);
      j++;
    }
  }

  shape = A_SHAPEG(arg);
  asd = A_ASDG(arg);
  ndim = ASD_NDIM(asd);
  sdim = 0;
  for (i = 0; i < ndim; i++) {
    if (A_TYPEG(ASD_SUBS(asd, i)) == A_TRIPLE || A_SHAPEG(ASD_SUBS(asd, i))) {
      /* include this dimension */
      subs[j] = ASD_SUBS(asd, i);
      j++;
      sdim++;
    }
  }
  assert(j == rank_of_sym(sptr), "reference_for_pure_temp: rank mismatched",
         sptr, 4);
  assert(shape, "reference_for_pure_temp: shape mismatched", sptr, 4);
  assert(SHD_NDIM(shape) == sdim, "reference_for_pure_temp: shape mismatched",
         sptr, 4);

  astnew = mk_subscr(mk_id(sptr), subs, j, DTYPEG(sptr));
  return astnew;
}

/* this will decide whether pure tmp will be larger than arg
 * if gen_shape_comm() choose to have distributed temp.
 * if it is, it will not choose the distributed temp.
 * it will choose to have replicated temp.
 */
static LOGICAL
is_pure_temp_too_large(int list, int arg)
{
  int count;
  int ndim;
  int asd;
  int i;
  int j;

  count = 0;
  for (j = list; j != 0; j = ASTLI_NEXT(j))
    count++;
  assert(A_TYPEG(arg) == A_SUBSCR, "is_pure_temp_too_large: not SUBSCR", arg,
         4);
  asd = A_ASDG(arg);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++) {
    if (A_TYPEG(ASD_SUBS(asd, i)) == A_TRIPLE || A_SHAPEG(ASD_SUBS(asd, i)))
      count++;
  }
  if (count > ndim)
    return TRUE;
  return FALSE;
}

/* this routine is to find distributed array in expr.
 * assign those array to the same size replicated temp
 * For example:   a(inx(i))
 * indx$temp = indx
 * a$temp = a
 * return a$temp(indx$temp(i))
 */
static int
handle_pure_temp_too_large(int expr, int std)
{
  int l, r, d, o;
  int l1, l2, l3;
  int i, nargs, argt;
  int tmp_sptr, tmp_ast;
  int forall, ast;
  int asd, ndim;
  int shape, std1;
  int sptr;
  int eledtype;
  int subs[7];
  int asn;

  if (expr == 0)
    return expr;
  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = handle_pure_temp_too_large(A_LOPG(expr), std);
    r = handle_pure_temp_too_large(A_ROPG(expr), std);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = handle_pure_temp_too_large(A_LOPG(expr), std);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(expr);
    l = handle_pure_temp_too_large(A_LOPG(expr), std);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(expr);
    l = handle_pure_temp_too_large(A_LOPG(expr), std);
    return mk_paren(l, d);
  case A_SUBSTR:
    return expr;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = handle_pure_temp_too_large(ARGT_ARG(argt, i), std);
    }
    return expr;
  case A_CNST:
  case A_CMPLXC:
    return expr;
  case A_MEM:
    sptr = A_SPTRG(A_MEMG(expr));
    if (DTY(DTYPEG(sptr)) != TY_ARRAY || !ALIGNG(sptr)) {
      r = A_MEMG(expr);
      d = A_DTYPEG(r);
      l = handle_pure_temp_too_large(A_PARENTG(expr), std);
      return mk_member(l, r, d);
    }
    goto replicate_temp;

  case A_ID:
    sptr = A_SPTRG(expr);
    if (STYPEG(sptr) != ST_ARRAY || !ALIGNG(sptr))
      return expr;
    eledtype = DTY(DTYPEG(sptr) + 1);

  replicate_temp:
    /* copy to replicate temp */
    tmp_sptr = get_temp_pure_replicated(sptr, std, std, expr);
    tmp_ast = mk_id(tmp_sptr);
    asn = mk_assn_stmt(tmp_ast, expr, eledtype);
    shape = A_SHAPEG(tmp_ast);
    forall = make_forall(shape, tmp_ast, 0, 0);
    A_ARRASNP(forall, TRUE);
    forall = rename_forall_list(forall);
    ast = normalize_forall(forall, asn, 0);
    A_IFSTMTP(forall, ast);
    A_IFEXPRP(forall, 0);
    std1 = add_stmt_before(forall, std);
    process_forall(std1);
    transform_forall(std1, forall);
    return mk_id(tmp_sptr);
  case A_SUBSCR:
    asd = A_ASDG(expr);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++) {
      subs[i] = handle_pure_temp_too_large(ASD_SUBS(asd, i), std);
    }
    l1 = handle_pure_temp_too_large(A_LOPG(expr), std);
    expr = mk_subscr(l1, subs, ndim, A_DTYPEG(expr));
    return expr;
  case A_TRIPLE:
    l1 = handle_pure_temp_too_large(A_LBDG(expr), std);
    l2 = handle_pure_temp_too_large(A_UPBDG(expr), std);
    l3 = handle_pure_temp_too_large(A_STRIDEG(expr), std);
    return mk_triple(l1, l2, l3);
  default:
    interr("handle_pure_temp_too_large: unknown expression", expr, 2);
    return expr;
  }
}
#endif

static void
insert_call_comm(int std, int forall)
{
  int nd;
  int i;
  int cstd;

  nd = A_OPT1G(forall);
  comminfo.mask_phase = 1;
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    put_call_comm(cstd, std, forall);
  }
  comminfo.mask_phase = 0;
  for (i = 0; i < FT_NSCALL(nd); i++) {
    cstd = glist(FT_SCALL(nd), i);
    put_call_comm(cstd, std, forall);
  }
}

static void
put_call_comm(int cstd, int fstd, int forall)
{
  int ast, ast1;
  int cstd1;
  int nd, nd1;
  int i;
  int nargs, argt;

  comminfo.usedstd = cstd;
  ast = STD_AST(cstd);
  nd = A_OPT1G(ast);
  assert(nd, "call_comm: something is wrong", ast, 3);
  for (i = 0; i < FT_CALL_NCALL(nd); i++) {
    cstd1 = glist(FT_CALL_CALL(nd), i);
    ast1 = STD_AST(cstd1);
    nd1 = A_OPT1G(ast1);
    assert(nd1, "put_calls: something is wrong", ast1, 3);
    put_call_comm(cstd1, fstd, forall);
  }
  nargs = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);
  for (i = 0; i < nargs; ++i) {
    ARGT_ARG(argt, i) = insert_forall_comm(ARGT_ARG(argt, i));
  }
}

static int
tag_call_comm(int std, int forall)
{
  int nd;
  int i;
  int cstd;

  nd = A_OPT1G(forall);
  comminfo.mask_phase = 1;
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    call_comm(cstd, std, forall);
  }
  comminfo.mask_phase = 0;
  for (i = 0; i < FT_NSCALL(nd); i++) {
    cstd = glist(FT_SCALL(nd), i);
    call_comm(cstd, std, forall);
  }
  return 1;
}

static void
call_comm(int cstd, int fstd, int forall)
{
  int ast, ast1;
  int cstd1;
  int nd, nd1;
  int i;
  int test;

  ast = STD_AST(cstd);
  nd = A_OPT1G(ast);
  assert(nd, "call_comm: something is wrong", ast, 3);
  for (i = 0; i < FT_CALL_NCALL(nd); i++) {
    cstd1 = glist(FT_CALL_CALL(nd), i);
    ast1 = STD_AST(cstd1);
    nd1 = A_OPT1G(ast1);
    assert(nd1, "put_calls: something is wrong", ast1, 3);
    call_comm(cstd1, fstd, forall);
  }
  test = tag_forall_comm(ast);
}

static int
sequentialize_mask_call(int forall, int stdnext)
{
  int nd;
  int i;
  int cstd;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NMCALL(nd); i++) {
    cstd = glist(FT_MCALL(nd), i);
    stdnext = sequentialize_call(cstd, stdnext, forall);
  }
  return stdnext;
}

static int
sequentialize_stmt_call(int forall, int stdnext)
{
  int nd;
  int i;
  int cstd;

  nd = A_OPT1G(forall);
  for (i = 0; i < FT_NSCALL(nd); i++) {
    cstd = glist(FT_SCALL(nd), i);
    stdnext = sequentialize_call(cstd, stdnext, forall);
  }
  return stdnext;
}

static int
sequentialize_call(int cstd, int stdnext, int forall)
{
  int ast, ast1;
  int cstd1;
  int nd, nd1;
  int i, lineno;
  int stdnext1;

  ast = STD_AST(cstd);
  nd = A_OPT1G(ast);
  assert(nd, "call_comm: something is wrong", ast, 3);
  for (i = 0; i < FT_CALL_NCALL(nd); i++) {
    cstd1 = glist(FT_CALL_CALL(nd), i);
    ast1 = STD_AST(cstd1);
    nd1 = A_OPT1G(ast1);
    assert(nd1, "put_calls: something is wrong", ast1, 3);
    stdnext = sequentialize_call(cstd1, stdnext, forall);
  }
  lineno = STD_LINENO(cstd);
  delete_stmt(cstd);
  stdnext = add_stmt_before(ast, stdnext);
  stdnext1 = STD_NEXT(stdnext);
  STD_LINENO(stdnext) = lineno;
  transform_ast(stdnext, ast);
  stdnext = stdnext1;
  return stdnext;
}

/* this routine will normalize forall triplet list,
 * It makes triple integer and
 * It eliminates distributed array from triplet.
 */
static int
normalize_forall_triplet(int std, int forall)
{
  int lb, ub, st;
  int triplet_list;
  int triplet;
  int list;
  int rhs_is_dist;
  int tmp_sptr;
  int newlist;
  int ast, dest;

  /* don't allow forall(i=1:n,j=istart(i):istop(i) */
  triplet_list = A_LISTG(forall);
  if (is_multiple_idx_in_list(triplet_list))
    return 0;

  /* It eliminates distributed array from triplet */
  triplet_list = A_LISTG(forall);
  start_astli();
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    triplet = ASTLI_TRIPLE(triplet_list);
    /* case forall(i=idx(1):n) */
    rhs_is_dist = FALSE;
    triplet = insert_comm_before(std, triplet, &rhs_is_dist, FALSE);
    newlist = add_astli();
    ASTLI_SPTR(newlist) = ASTLI_SPTR(triplet_list);
    ASTLI_TRIPLE(newlist) = triplet;
  }
  list = ASTLI_HEAD;
  A_LISTP(forall, list);

  /* make forall triple DT_INT if not */
  triplet_list = A_LISTG(forall);
  start_astli();
  for (; triplet_list; triplet_list = ASTLI_NEXT(triplet_list)) {
    triplet = ASTLI_TRIPLE(triplet_list);
    lb = A_LBDG(triplet);
    assert(lb, "normalize_forall_triplet: no lower bound at forall triplet",
           forall, 3);
    if (A_TYPEG(lb) == A_CONV)
      lb = A_LOPG(lb);
    if (!DT_ISINT(A_DTYPEG(lb))) {
      tmp_sptr = sym_get_scalar("lb", 0, astb.bnd.dtype);
      ast = mk_stmt(A_ASN, astb.bnd.dtype);
      dest = mk_id(tmp_sptr);
      A_DESTP(ast, dest);
      A_SRCP(ast, lb);
      add_stmt_before(ast, std);
      lb = mk_id(tmp_sptr);
    }
    ub = A_UPBDG(triplet);
    assert(ub, "normalize_forall_triplet: no lower bound at forall triplet",
           forall, 3);
    if (A_TYPEG(ub) == A_CONV)
      ub = A_LOPG(ub);
    if (!DT_ISINT(A_DTYPEG(ub))) {
      tmp_sptr = sym_get_scalar("ub", 0, astb.bnd.dtype);
      ast = mk_stmt(A_ASN, astb.bnd.dtype);
      dest = mk_id(tmp_sptr);
      A_DESTP(ast, dest);
      A_SRCP(ast, ub);
      add_stmt_before(ast, std);
      ub = mk_id(tmp_sptr);
    }
    st = A_STRIDEG(triplet);
    if (A_TYPEG(st) == A_CONV)
      st = A_LOPG(st);
    if (st)
      if (!DT_ISINT(A_DTYPEG(st))) {
        tmp_sptr = sym_get_scalar("st", 0, astb.bnd.dtype);
        ast = mk_stmt(A_ASN, astb.bnd.dtype);
        dest = mk_id(tmp_sptr);
        A_DESTP(ast, dest);
        A_SRCP(ast, st);
        add_stmt_before(ast, std);
        st = mk_id(tmp_sptr);
      }
    triplet = mk_triple(lb, ub, st);
    newlist = add_astli();
    ASTLI_SPTR(newlist) = ASTLI_SPTR(triplet_list);
    ASTLI_TRIPLE(newlist) = triplet;
  }
  list = ASTLI_HEAD;
  A_LISTP(forall, list);
  return 1;
}

/* This is a quick fix to move guard_forall after optimization.
 * guard_forall was inserting IF-THEN which was reducing
 * the optimization chance. guard_forall can be written
 * such that it will not need this fix. */
static void
fix_guard_forall(int std)
{
  CTYPE *ct;
  int ast;
  int asn;
  int subinfo;
  int lhs, lhsd;
  int ndim, asd;
  int i;
  int nd;

  ast = STD_AST(std);
  asn = A_IFSTMTG(ast);
  nd = A_OPT1G(ast);
  ct = FT_CYCLIC(nd);
  lhs = A_DESTG(asn);
  lhsd = left_subscript_ast(lhs);
  asd = A_ASDG(lhsd);
  ndim = ASD_NDIM(asd);
  subinfo = comminfo.subinfo;
  for (i = 0; i < ndim; ++i) {
    ct->c_dstt[i] = SUBI_DSTT(subinfo + i);
    ct->c_dupl[i] = SUBI_DUPL(subinfo + i);
    ct->c_idx[i] = SUBI_IDX(subinfo + i);
  }
  A_OPT1P(ast, nd);
}

/* This routine  is to check whether forall has dependency.
 * If it has, it creates temp which is shape array with lhs.
 * For example,
 *              forall(i=1:N) a(i) = a(i-1)+.....
 * will be rewritten
 *              forall(i=1:N) temp(i) = a(i-1)+.....
 *              forall(i=1:N) a(i) = temp(i)
 */
static void
forall_dependency_scalarize(int std, int *std1, int *std2)
{
  int lhs, rhs;
  int ast;
  int asn;
  int sptr;
  int temp_ast;
  int newforall, newasn;
  int expr;
  int lineno;
  LOGICAL bIndep;

  ast = STD_AST(std);
  asn = A_IFSTMTG(ast);
  if (A_TYPEG(asn) != A_ASN)
    return;
  lhs = A_DESTG(asn);
  rhs = A_SRCG(asn);
  expr = A_IFEXPRG(ast);

  /* forall-independent */
  lineno = STD_LINENO(std);
  open_pragma(lineno);
  bIndep = XBIT(19, 0x100) != 0;
  close_pragma();
  if (bIndep)
    return;

  /* take conditional expr, if there is dependency */
  if (is_dependent(lhs, expr, ast, std, std) && A_TYPEG(lhs) != A_SUBSTR) {
    sptr = get_temp_forall(ast, lhs, std, std, DT_LOG, 0);
    temp_ast = reference_for_temp(sptr, lhs, ast);
    A_IFEXPRP(ast, temp_ast);
    newforall = mk_stmt(A_FORALL, 0);
    A_LISTP(newforall, A_LISTG(ast));
    A_IFEXPRP(newforall, 0);
    newasn = mk_stmt(A_ASN, 0);
    A_DESTP(newasn, temp_ast);
    A_SRCP(newasn, expr);
    A_IFSTMTP(newforall, newasn);
    *std1 = add_stmt_before(newforall, std);
  }

  if (is_dependent(lhs, rhs, ast, std, std) && A_TYPEG(lhs) != A_SUBSTR) {
    sptr = get_temp_forall(ast, lhs, std, std, 0, lhs);
    temp_ast = reference_for_temp(sptr, lhs, ast);
    A_DESTP(asn, temp_ast);
    newforall = mk_stmt(A_FORALL, 0);
    A_LISTP(newforall, A_LISTG(ast));
    A_IFEXPRP(newforall, A_IFEXPRG(ast));
    newasn = mk_stmt(A_ASN, 0);
    A_DESTP(newasn, lhs);
    A_SRCP(newasn, temp_ast);
    A_IFSTMTP(newforall, newasn);
    *std2 = add_stmt_after(newforall, std);
  }
}

static int
fix_mem_ast(int astmem, int ast)
{

  int rslt;

  switch (A_TYPEG(ast)) {

  case A_BINOP:
    rslt = fix_mem_ast(astmem, A_LOPG(ast));
    if (rslt && rslt != A_LOPG(ast))
      A_LOPP(ast, rslt);
    rslt = fix_mem_ast(astmem, A_ROPG(ast));
    if (rslt && rslt != A_ROPG(ast))
      A_ROPP(ast, rslt);
    break;
  case A_UNOP:
    rslt = fix_mem_ast(astmem, A_LOPG(ast));
    if (rslt && rslt != A_LOPG(ast))
      A_LOPP(ast, rslt);
    break;
  case A_LABEL:
  case A_ENTRY:
  case A_ID:
    return check_member(astmem, ast);
  case A_SUBSCR:
  case A_SUBSTR:
    rslt = fix_mem_ast(astmem, A_LOPG(ast));
    if (rslt && rslt != A_LOPG(ast))
      A_LOPP(ast, rslt);
    break;
  case A_MEM:
    rslt = fix_mem_ast(astmem, A_PARENTG(ast));
    if (rslt && rslt != A_PARENTG(ast))
      A_PARENTP(ast, rslt);
    break;
  }
  return 0;
}

/* This routine will perform the following canonical conversion
 *
 * forall(i=l:u:s)  a(m*i+k) = ...i...
 *
 * will be converted into
 *
 * forall(i=m*l+k:m*u+k:m*s)  a(i) = ...(i-k)/m...
 */
/* ### rewrite this routine to handle members */
static int
canonical_conversion(int ast)
{
  int list;
  int asn;
  int astli;
  int base, stride;
  int expr;
  int newexpr;
  int l, u, s;
  int ll, uu, ss;
  int triple;
  int asd;
  int ndim;
  int isptr;
  int i, k;
  int zero = astb.bnd.zero;
  int ifexpr;
  int subs[7];
  int newdest;
  int nd, nd1;
  int ip, pstd, past;
  LITEMF *plist;
  int lhs, lhsd, sptr;
  int align;


  /* Don't replace the subscript if we intend it that way */
  if (!XBIT(58,0x1000000) && A_CONSTBNDG(ast))
    return 0;

  list = A_LISTG(ast);
  ifexpr = A_IFEXPRG(ast);
  asn = A_IFSTMTG(ast);
  expr = A_SRCG(asn);
  lhs = A_DESTG(asn);

  for (lhsd = lhs; A_TYPEG(lhsd) != A_ID;) {
    switch (A_TYPEG(lhsd)) {
    case A_SUBSCR:
      asd = A_ASDG(lhsd);
      ndim = ASD_NDIM(asd);

      /* don't let A(V(I)), where V is distributed, that is solved earlier */
      for (i = 0; i < ndim; ++i) {
        ss = ASD_SUBS(asd, i);
        if (is_dist_array_in_expr(ss)) {
          return 0;
        }
      }
      lhsd = A_LOPG(lhsd);
      break;
    case A_MEM:
      lhsd = A_PARENTG(lhsd);
      break;
    default:
      interr("canonical_conversion unexpected AST type on LHS", A_TYPEG(lhsd),
             3);
      break;
    }
  }
  lhsd = left_subscript_ast(lhs);
  asd = A_ASDG(lhsd);
  ndim = ASD_NDIM(asd);
  sptr = left_array_symbol(lhs);
  align = ALIGNG(sptr);

  /* don't let A(I+J), don't let A(I,I+1), let A(I,I) */
  for (i = 0; i < ndim; i++) {
    astli = 0;
    search_idx(ASD_SUBS(asd, i), list, &astli, &base, &stride);
    if (base == 0)
      return 0; /* i+j */
    if (astli == 0 && stride == zero)
      continue; /* only base */
    if (base == zero && stride == astb.bnd.one)
      continue; /* a(i) */
    isptr = ASTLI_SPTR(astli);
    for (k = 0; k < ndim; ++k) {
      if (k != i) {
        if (is_name_in_expr(ASD_SUBS(asd, k), isptr)) {
          return 0; /* A(i+1,i)  */
        }
      }
    }
  }


  for (i = 0; i < ndim; i++) {
    subs[i] = ASD_SUBS(asd, i);
    astli = 0;
    search_idx(ASD_SUBS(asd, i), list, &astli, &base, &stride);
    if (base == 0)
      return 0; /* i+j */
    if (astli == 0 && stride == zero)
      continue; /* only base */
    if (base == zero && stride == astb.bnd.one)
      continue; /* a(i) */
    ast_visit(1, 1);
    isptr = ASTLI_SPTR(astli);
    /* change the lhs subscript*/
    subs[i] = mk_id(isptr);

    /* calculate (i-k)/m   */
    newexpr = opt_binop(OP_SUB, mk_id(isptr), base, astb.bnd.dtype);
    newexpr = opt_binop(OP_DIV, newexpr, stride, astb.bnd.dtype);

    ast_replace(mk_id(isptr), newexpr);

    /* change the rhs expression*/
    expr = ast_rewrite(expr);

    /* change the ifexpr expression*/
    ifexpr = ast_rewrite(ifexpr);

    /* change also pcalls */
    nd = A_OPT1G(ast);
    plist = FT_PCALL(nd);
    for (ip = 0; ip < FT_NPCALL(nd); ip++) {
      pstd = plist->item;
      plist = plist->next;
      past = STD_AST(pstd);
      nd1 = A_OPT1G(past);
      past = ast_rewrite(past);
      A_OPT1P(past, nd1);
      STD_AST(pstd) = past;
      A_STDP(past, pstd);
    }

    ast_unvisit();

    /* change the forall list */
    triple = ASTLI_TRIPLE(astli);
    l = A_LBDG(triple);
    fix_mem_ast(l, base);
    u = A_UPBDG(triple);
    s = A_STRIDEG(triple);

    ll = opt_binop(OP_MUL, stride, l, astb.bnd.dtype);
    ll = opt_binop(OP_ADD, ll, base, astb.bnd.dtype);
    uu = opt_binop(OP_MUL, stride, u, astb.bnd.dtype);
    uu = opt_binop(OP_ADD, uu, base, astb.bnd.dtype);
    if (s == 0)
      ss = stride;
    else
      ss = opt_binop(OP_MUL, stride, s, astb.bnd.dtype);
    ASTLI_TRIPLE(astli) = mk_triple(ll, uu, ss);
  }
  newdest = mk_subscr(A_LOPG(lhsd), subs, ndim, A_DTYPEG(lhsd));
  newdest = replace_ast_subtree(lhs, lhsd, newdest);
  A_DESTP(asn, newdest);
  A_SRCP(asn, expr);
  A_IFEXPRP(ast, ifexpr);
  A_IFSTMTP(ast, asn);
  return 1;
}

/* this will find scalar communication at ast,
 * It expect that std is forall std.
 * It does not disturb other forall communication:
 * For example, forall(i=1:n) a(b(1),c(i)) = 1
 * Here, only perform communication for b(1).
 */
static int
scalar_communication(int ast, int std)
{
  int l, r, d, o;
  int l1, l2, l3;
  int a;
  int i, nargs, argt;
  int header;
  int forall;
  int rhs_is_dist;
  int asd, ndim;
  int subs[7];
  int nd;

  a = ast;
  if (!a)
    return a;
  forall = STD_AST(std);
  switch (A_TYPEG(ast)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = scalar_communication(A_LOPG(a), std);
    r = scalar_communication(A_ROPG(a), std);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = scalar_communication(A_LOPG(a), std);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(a);
    l = scalar_communication(A_LOPG(a), std);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(a);
    l = scalar_communication(A_LOPG(a), std);
    return mk_paren(l, d);
  case A_MEM:
    r = A_MEMG(a);
    d = A_DTYPEG(r);
    l = scalar_communication(A_PARENTG(a), std);
    return mk_member(l, r, d);
  case A_SUBSTR:
    return a;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = scalar_communication(ARGT_ARG(argt, i), std);
    }
    return a;
  case A_CNST:
  case A_CMPLXC:
    return a;
  case A_ID:
    return a;
  case A_SUBSCR:
    if (!A_SHAPEG(a) && is_array_element_in_forall(a, std)) {
      nd = A_OPT1G(forall);
      header = FT_HEADER(nd);
      rhs_is_dist = FALSE;
      a = insert_comm_before(header, a, &rhs_is_dist, FALSE);
      return a;
    }

    asd = A_ASDG(a);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++) {
      subs[i] = scalar_communication(ASD_SUBS(asd, i), std);
    }
    return mk_subscr(A_LOPG(a), subs, ndim, A_DTYPEG(a));
  case A_TRIPLE:
    l1 = scalar_communication(A_LBDG(a), std);
    l2 = scalar_communication(A_UPBDG(a), std);
    l3 = scalar_communication(A_STRIDEG(a), std);
    return mk_triple(l1, l2, l3);
  default:
    interr("scalar_communication: unknown expression", std, 2);
    return 0;
  }
}
