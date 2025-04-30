/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Detect communications for the communications module.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "gramtk.h"
#include "comm.h"
#include "extern.h"
#include "hpfutl.h"
#include "commopt.h"
#include "symutl.h"

static void process_sub(int, int);
static void tag_comms(int);

static void matched_dim(int);
static void no_comm_class(int);
static void overlap_class(int);
#ifdef FLANG_DETECT_UNUSED
static void copy_section_class(int);
static void gather_class(int, int);
#endif
static void convert_idx_scalar(int);
static int is_structured(int);
#ifdef FLANG_DETECT_UNUSED
static LOGICAL is_scatterx_gatherx_subscript(int, int);
static LOGICAL result_base_relation(int result, int base, int forall);
static LOGICAL mask_array_relation(int mask, int array, int forall);
static int generic_intrinsic_type(int);
static LOGICAL is_all_idx_appears(int, int);
#endif
static LOGICAL is_array_in_expr(int ast);
static LOGICAL is_nonscalar_array_in_expr(int ast);

/** \brief Go through and tag the communications for this statement.
    \return 0 if any unstructured communication
 */
int
tag_forall_comm(int ast)
{
  int std;
  int l, r, o;
  int a;
  int i, nargs, argt;
  int ndim, asd;
  int src;
  int arref;
  int sptr;

  a = ast;
  if (!a)
    return 1;
  std = comminfo.std;
  switch (A_TYPEG(ast)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(a);
    l = tag_forall_comm(A_LOPG(a));
    r = tag_forall_comm(A_ROPG(a));
    if (l == 0 || r == 0)
      return 0;
    return 1;
  case A_UNOP:
    o = A_OPTYPEG(a);
    l = tag_forall_comm(A_LOPG(a));
    if (l == 0)
      return 0;
    return 1;
  case A_CONV:
    l = tag_forall_comm(A_LOPG(a));
    if (l == 0)
      return 0;
    return 1;
  case A_PAREN:
    l = tag_forall_comm(A_LOPG(a));
    if (l == 0)
      return 0;
    return 1;
  case A_MEM:
    l = tag_forall_comm(A_PARENTG(a));
    if (l == 0)
      return 0;
    return 1;
  case A_SUBSTR:
    return 1;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      l = tag_forall_comm(ARGT_ARG(argt, i));
      if (l == 0)
        return 0;
    }
    if (A_OPTYPEG(a) == I_CSHIFT) {
      src = ARGT_ARG(argt, 0);
      arref = A_RFPTRG(src);
      assert(ARREF_CLASS(arref) == OVERLAP,
             "tag_forall_comm: CSHIFT must be overlap", a, 2);
    }
    if (A_OPTYPEG(a) == I_EOSHIFT) {
      src = ARGT_ARG(argt, 0);
      arref = A_RFPTRG(src);
      assert(ARREF_CLASS(arref) == OVERLAP,
             "tag_forall_comm: EOSHIFT must be overlap", a, 2);
    }

    return 1;
  case A_CNST:
  case A_CMPLXC:
    return 1;
  case A_ID:
    return 1;
  case A_SUBSCR:
    if (A_SHAPEG(a))
      return 1;
    sptr = sptr_of_subscript(a);
    if (!ALIGNG(sptr)) {
      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; i++) {
        l = tag_forall_comm(ASD_SUBS(asd, i));
        if (l == 0)
          return 0;
      }
      l = tag_forall_comm(A_LOPG(a));
      return l;
    }
    /* It is distributed */
    tag_comms(a);
    return 1;
  case A_CALL:
  case A_ICALL:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      if (!ARGT_ARG(argt, i))
        continue;
      l = tag_forall_comm(ARGT_ARG(argt, i));
      if (l == 0)
        return 0;
    }
    return 1;
  default:
    interr("tag_forall_comm: unknown expression", std, 2);
    return 0;
  }
}

void
process_rhs_sub(int a)
{
  int ndim1, sptr;
  int asd, ndim;
  int subinfo;
  int i;
  int arref;
  int align;

  assert(A_TYPEG(a) == A_SUBSCR, "process_rhs_sub: not SUBSCR", a, 4);
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  sptr = sptr_of_subscript(a);
  align = ALIGNG(sptr);
  ndim1 = ASD_NDIM(A_ASDG(comminfo.sub));

  arref = trans.arrb.stg_avail++;
  TRANS_NEED(trans.arrb, ARREF, trans.arrb.stg_size + 100);
  A_RFPTRP(a, arref);

  subinfo = trans.subb.stg_avail;
  trans.subb.stg_avail += ndim;
  assert(ndim < 100, "transform_forall: ndim huge?", 0, 4);
  TRANS_NEED(trans.subb, SUBINFO, trans.subb.stg_size + 1000);
  /* make list of rhs references */
  ARREF_SUB(arref) = subinfo;
  ARREF_NDIM(arref) = ndim;
  ARREF_NEXT(arref) = trans.rhsbase;
  trans.rhsbase = arref;
  ARREF_ARRSYM(arref) = sptr;
  ARREF_ARR(arref) = a;
  ARREF_TEMP(arref) = 0;
  ARREF_CLASS(arref) = NO_CLASS;
  for (i = 0; i < ndim; ++i) {
    SUBI_DUPL(subinfo + i) = 0;
    SUBI_NOP(subinfo + i) = 0;
    SUBI_POP(subinfo + i) = 0;
    SUBI_SUB(subinfo + i) = ASD_SUBS(asd, i);
    SUBI_DSTT(subinfo + i) = 0;
    SUBI_COMMT(subinfo + i) = COMMT_NOTAG;
    process_sub(subinfo + i, A_LISTG(comminfo.forall));
  }
  matched_dim(a);
}

/* try to find communication class for rhs array a
*  communication class can be NO_COMM,  OVERLAP,
*  COLLECTIVE, COPY_SECTION and IRREGULAR.
*  This is the beginning of communication detection algorithm.
*  It works for any alignment and distribution. Good luck!
*/
static void
tag_comms(int a)
{
  int ndim1, sptr;
  int asd, ndim;
  int arref;
  int align;
  int forall;
  int lhs;

  assert(A_TYPEG(a) == A_SUBSCR, "tag_comms: not SUBSCR", a, 4);
  sptr = sptr_of_subscript(a);
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  align = ALIGNG(sptr);
  ndim1 = ASD_NDIM(A_ASDG(comminfo.sub));
  forall = comminfo.forall;
  lhs = A_DESTG(A_IFSTMTG(forall));

  process_rhs_sub(a);
  arref = A_RFPTRG(a);

  ARREF_CLASS(arref) = NO_CLASS;
  if (is_structured(a)) {
    if (ARREF_CLASS(arref) == NO_CLASS) {
      no_comm_class(a);
    }
    if (ARREF_CLASS(arref) == NO_CLASS) {
      overlap_class(a);
    }
  }
  if (ARREF_CLASS(arref) == NO_CLASS) {
    ARREF_CLASS(arref) = NO_COMM;
  }
}

#if DEBUG
void
dumpsubinfo(int subinfo, int ndim)
{
  FILE *outfile;
  int i;
  if (gbl.dbgfil == NULL) {
    outfile = stderr;
  } else {
    outfile = gbl.dbgfil;
  }
  for (i = 0; i < ndim; ++i) {
    int s, idx, base, stride, sub, dstt, ldim, sptr;
    int commt, commv, cnst, diff, dupl, nop, pop;
    const char *class;
    s = subinfo + i;
    idx = SUBI_IDX(s);
    base = SUBI_BASE(s);
    stride = SUBI_STRIDE(s);
    sub = SUBI_SUB(s);
    dstt = SUBI_DSTT(s);
    ldim = SUBI_LDIM(s);
    commt = SUBI_COMMT(s);
    commv = SUBI_COMMV(s);
    cnst = SUBI_CNST(s);
    diff = SUBI_DIFF(s);
    dupl = SUBI_DUPL(s);
    nop = SUBI_NOP(s);
    pop = SUBI_POP(s);
    switch (commt) {
    case COMMT_NOTAG:
      class = "notag";
      break;
    case COMMT_NONE:
      class = "none";
      break;
    case COMMT_MULTI:
      class = "multi";
      break;
    case COMMT_SHIFTC:
      class = "shiftc";
      break;
    case COMMT_SHIFTV:
      class = "shiftv";
      break;
    case COMMT_TRANSFER:
      class = "transfer";
      break;
    case COMMT_REPLICATE:
      class = "replicate";
      break;
    case COMMT_CONST:
      class = "const";
      break;
    case COMMT_UNSTRUCT:
      class = "unstruct";
      break;
    default:
      class = "unknown?";
      break;
    }
    fprintf(outfile, " subinfo dim(%d)=%s", i, class);
    switch (commt) {
    case COMMT_SHIFTC:
    case COMMT_SHIFTV:
      fprintf(outfile, "(%d)", commv);
      break;
    }
    if (idx != 0) {
      fprintf(outfile, " idx(%d)", idx);
      if (idx > 0) {
        sptr = ASTLI_SPTR(idx);
        fprintf(outfile, "=sptr(%d)=", sptr);
        if (sptr <= 0 || sptr >= stb.stg_avail) {
          fprintf(outfile, "out-of-range");
        } else {
          fprintf(outfile, "%s", SYMNAME(sptr));
        }
      }
    }
    if (base != 0) {
      fprintf(outfile, " base(%d)=", base);
      if (base <= 0 || base >= astb.stg_avail) {
        fprintf(outfile, "out-of-range");
      } else {
        printast(base);
      }
    }
    if (stride != 0) {
      fprintf(outfile, " stride(%d)=", stride);
      if (stride <= 0 || stride >= astb.stg_avail) {
        fprintf(outfile, "out-of-range");
      } else {
        printast(stride);
      }
    }
    fprintf(outfile, "\n        ");
    if (sub != 0) {
      fprintf(outfile, " sub(%d)=", sub);
      if (sub <= 0 || sub >= astb.stg_avail) {
        fprintf(outfile, "out-of-range");
      } else {
        printast(sub);
      }
    }

    if (ldim != -1) {
      fprintf(outfile, " ldim(%d)", ldim);
    }
    if (cnst != -1) {
      fprintf(outfile, " cnst(%d)", cnst);
    }
    if (diff != 0) {
      fprintf(outfile, " diff(%d)=", diff);
      if (diff <= 0 || diff >= astb.stg_avail) {
        fprintf(outfile, "out-of-range");
      } else {
        printast(diff);
      }
    }
    if (dupl) {
      fprintf(outfile, " dupl");
    }
    if (nop || pop) {
      fprintf(outfile, " nop:pop(%d:%d)", nop, pop);
    }
    fprintf(outfile, "\n");
  }
} /* dumpsubinfo */

void
dumparref(int arref)
{
  int ndim, subinfo, sptr, ast, i;
  const char *class;
  FILE *outfile;
  if (gbl.dbgfil == NULL) {
    outfile = stderr;
  } else {
    outfile = gbl.dbgfil;
  }

  fprintf(outfile, "arref:%d", arref);
  if (trans.subb.stg_base == NULL) {
    fprintf(outfile, " arref not allocated\n");
    return;
  }
  if (arref <= 0 || arref >= trans.subb.stg_avail) {
    fprintf(outfile, " arref out of range [1:%d)\n", trans.subb.stg_avail);
    return;
  }
  ndim = ARREF_NDIM(arref);
  subinfo = ARREF_SUB(arref);
  switch (ARREF_CLASS(arref)) {
  case NO_CLASS:
    class = "NO_CLASS";
    break;
  case NO_COMM:
    class = "NO_COMM";
    break;
  case OVERLAP:
    class = "OVERLAP";
    break;
  case COLLECTIVE:
    class = "COLLECTIVE";
    break;
  case COPY_SECTION:
    class = "COPY_SECTION";
    break;
  case IRREGULAR:
    class = "IRREGULAR";
    break;
  default:
    class = "unknown?";
    break;
  }
  fprintf(outfile, " %s ndim(%d) subinfo(%d) next(%d) flag(%x) temp(%d)\n",
          class, ndim, subinfo, ARREF_NEXT(arref), ARREF_FLAG(arref),
          ARREF_TEMP(arref));
  sptr = ARREF_ARRSYM(arref);
  ast = ARREF_ARR(arref);
  if (sptr <= 0 || sptr > stb.stg_avail) {
    fprintf(outfile, "	sptr(%d)=out-of-range", sptr);
  } else {
    fprintf(outfile, "	sptr(%d)=%s", sptr, SYMNAME(sptr));
  }
  if (ast <= 0 || ast > astb.stg_avail) {
    fprintf(outfile, "  ast(%d)=out-of-range", ast);
  } else {
    fprintf(outfile, "  ast(%d)=", ast);
    printast(ast);
  }
  fprintf(outfile, " subinfo(");
  for (i = 0; i < ndim; ++i) {
    switch (SUBI_COMMT(subinfo + i)) {
    case COMMT_NOTAG:
      class = "notag";
      break;
    case COMMT_NONE:
      class = "none";
      break;
    case COMMT_MULTI:
      class = "multi";
      break;
    case COMMT_SHIFTC:
      class = "shiftc";
      break;
    case COMMT_SHIFTV:
      class = "shiftv";
      break;
    case COMMT_TRANSFER:
      class = "transfer";
      break;
    case COMMT_REPLICATE:
      class = "replicate";
      break;
    case COMMT_CONST:
      class = "const";
      break;
    case COMMT_UNSTRUCT:
      class = "unstruct";
      break;
    default:
      class = "unknown?";
      break;
    }
    fprintf(outfile, "%s", class);
    if (i < ndim - 1) {
      fprintf(outfile, ",");
    }
  }
  printf(")\n");
  dumpsubinfo(subinfo, ndim);
} /* dumparref */

void
printastref(int ast)
{
  int sptr;
  int arref;
  FILE *outfile;

  printast(ast);

  if (gbl.dbgfil == NULL) {
    outfile = stderr;
  } else {
    outfile = gbl.dbgfil;
  }

  sptr = memsym_of_ast(ast);
  arref = A_RFPTRG(ast);
  fprintf(outfile, ": symbol(%s) arref(%d)\n", SYMNAME(sptr), arref);

  dumparref(arref);
} /* printastref */

static void
printstdref_subscript(int ast, LOGICAL *junk)
{
  if (A_TYPEG(ast) == A_SUBSCR) {
    printastref(ast);
  }
} /* printstdref_subscript */

void
printstdref(int std)
{
  /* print the statement, then look for all subscript references,
   * print the astref for each */
  FILE *outfile;
  int ast;

  if (gbl.dbgfil == NULL) {
    outfile = stderr;
  } else {
    outfile = gbl.dbgfil;
  }
  fprintf(outfile, "std(%d) ", std);
  if (std < 0 || std >= astb.std.stg_avail) {
    fprintf(outfile, "out-of-range [0:%d)\n", astb.std.stg_avail);
    return;
  }
  ast = STD_AST(std);
  fprintf(outfile, "ast(%d): ", ast);
  if (ast < 0 || ast >= astb.stg_avail) {
    fprintf(outfile, "out-of-range [0:%d)\n", astb.stg_avail);
    return;
  }
  printast(ast);
  fprintf(outfile, "\n");
  ast_visit(1, 1);
  ast_traverse(ast, NULL, printstdref_subscript, NULL);
  ast_unvisit();
} /* printstdref */

#endif

/* This return check the sign of structured communication.
 * lhs and rhs has to be same template.
 * each rhs subscripts has to be linear.
 * matched dimension has to have same forall index not like lhs(j)=rhs(i)
 */
static int
is_structured(int a)
{
  int sptr;
  int asd, ndim;
  int subinfo;
  int i, j;
  int arref;

  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  sptr = sptr_of_subscript(a);
  arref = A_RFPTRG(a);
  subinfo = ARREF_SUB(arref);
  for (i = 0; i < ndim; ++i) {
    /* if rhs index is not linear */
    if (SUBI_BASE(subinfo + i) == 0)
      return 0;
    /* if matched, has to have same index */
    j = SUBI_LDIM(subinfo + i);
    if (j != -1) {
      if ((SUBI_IDX(subinfo + i) != 0) &&
          (SUBI_IDX(comminfo.subinfo + j) != 0) &&
          (ASTLI_SPTR(SUBI_IDX(subinfo + i)) !=
           ASTLI_SPTR(SUBI_IDX(comminfo.subinfo + j))))
        return 0;
    }
  }
  return 1;
}

/* Algorithm:
 * This routine will only work for BLOCK and GEN_BLOCK distribution.
 * It only mark array with OVERLAP,
 * This only marks dimension for COMMT_SHIFTC or COMMT_SHIFTV for BLOCK
 * If it marked before as COMMT_NONE, COMMT_CONST, it respect that.
 * iff each dimension is taged COMMT_SHIFTC, COMMT_NONE or COMMT_CONST.
 */
static void
overlap_class(int a)
{
  int ndim1, sptr;
  int asd, ndim, l;
  int subinfo;
  int i, j;
  int arref;
  int align;
  int count;
  int diff;
  int shdw;

  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  l = A_LOPG(a);
  sptr = sptr_of_subscript(a);
  if (POINTERG(sptr))
    return;
  align = 0;
  shdw = 0;
  ndim1 = ASD_NDIM(A_ASDG(comminfo.sub));
  arref = A_RFPTRG(a);
  subinfo = ARREF_SUB(arref);
  for (i = 0; i < ndim; ++i) {
    j = SUBI_LDIM(subinfo + i);
    if (j != -1) {
      if (SUBI_COMMT(subinfo + i) == COMMT_NONE ||
          SUBI_COMMT(subinfo + i) == COMMT_CONST)
        continue;
      if (SUBI_CNST(subinfo + i)) {
        diff = SUBI_DIFF(subinfo + i);
        if (A_TYPEG(diff) == A_CNST) {
          SUBI_COMMT(subinfo + i) = COMMT_SHIFTC;
          SUBI_COMMV(subinfo + i) = diff;
        }
      }
    }
  }
  count = 0;
  for (i = 0; i < ndim; ++i)
    if (SUBI_COMMT(subinfo + i) == COMMT_SHIFTC ||
        SUBI_COMMT(subinfo + i) == COMMT_NONE ||
        SUBI_COMMT(subinfo + i) == COMMT_CONST)
      count++;
  if (ndim == count)
    ARREF_CLASS(arref) = OVERLAP;
}

/* Algorithm:
 * This routine does not care about distribution types.
 * It only cares about alignment.
 * If array is not distributed, it will be marked NO_COMM class.
 * If lhs and rhs array is in different template,
 * this routine can not guarantee NO_COMM class.
 * If each dimension of rhs array marked as COMMT_NONE or COMMT_CONST.
 * then the rhs array is marked as NO_COMM class.
 */
static void
no_comm_class(int a)
{
  int sptr;
  int asd, ndim;
  int subinfo;
  int i;
  int arref;
  int align;
  int no_comm;
  int zero = astb.bnd.zero;
  int forall, lhs;
  int align1, sptr1;
  LOGICAL single_ok[7];

  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  sptr = sptr_of_subscript(a);
  align = ALIGNG(sptr);
  arref = A_RFPTRG(a);
  subinfo = ARREF_SUB(arref);
  /* simple case */
  forall = comminfo.forall;
  lhs = dist_ast(A_DESTG(A_IFSTMTG(forall)));
  sptr1 = sptr_of_subscript(lhs);
  align1 = ALIGNG(sptr1);
  if (lhs == a) {
    ARREF_CLASS(arref) = NO_COMM;
    return;
  }
  for (i = 0; i < 7; i++)
    single_ok[i] = FALSE;

  no_comm = 0;
  for (i = 0; i < ndim; ++i) {
    {
      if (SUBI_STRIDE(subinfo + i) == zero)
        SUBI_COMMT(subinfo + i) = COMMT_CONST;
      else
        SUBI_COMMT(subinfo + i) = COMMT_NONE;
      no_comm++;
      continue;
    }
  }

  if (ndim == no_comm)
    ARREF_CLASS(arref) = NO_COMM;
}

#ifdef FLANG_DETECT_UNUSED
/* Algorithm:
 * This routine does not cares about neither template nor distribution types.
 * It looks the subscripts of lhs and rhs:
 *  -All subscripts have to be linear.
 *  -No diagonal accesses.
 *  -No transpose accesses.
 *  -All array assignments are okay except indirection.
 */

static void
copy_section_class(int a)
{
  int ndim1;
  int asd, ndim;
  int subinfo, subinfo_lhs;
  int j;
  int arref;
  int forall;
  int list;
  int lhs;

  arref = A_RFPTRG(a);
  forall = comminfo.forall;
  list = A_LISTG(forall);
  lhs = comminfo.sub;

  /* if it was array assignment, It is okay */
  /* Except indirection array */

  if (A_ARRASNG(forall) && !is_indirection_in_it(a)) {
    ARREF_CLASS(arref) = COPY_SECTION;
    return;
  }

  /* rhs: has to have linear subscripts */
  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  subinfo = ARREF_SUB(arref);
  for (j = 0; j < ndim; ++j)
    if (SUBI_BASE(subinfo + j) == 0)
      return;

  /* lhs: has to have linear subscripts */
  ndim1 = ASD_NDIM(A_ASDG(comminfo.sub));
  subinfo_lhs = comminfo.subinfo;
  for (j = 0; j < ndim1; ++j)
    if (SUBI_BASE(subinfo_lhs + j) == 0)
      return;

  /* rhs: no diagonal access */
  if (is_duplicate(a, list))
    return;

  /* lhs: no diagonal access */
  if (is_duplicate(lhs, list))
    return;

  ARREF_CLASS(arref) = COPY_SECTION;
}

static void
gather_class(int rhs, int std)
{
  int forall;
  int asn;
  int lhs;
  int arref;
  int array;
  int mask;
  int list;

  arref = A_RFPTRG(rhs);
  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);
  list = A_LISTG(forall);
  array = lhs = A_DESTG(asn);
  mask = A_IFEXPRG(forall);
  if (!is_scatterx_gatherx_subscript(rhs, forall))
    return;
  if (is_duplicate(lhs, list))
    return;
  if (!comminfo.mask_phase)
    if (mask)
      if (!mask_array_relation(mask, array, forall))
        return;

  ARREF_CLASS(arref) = GATHER;
}

/* This routine checks whether array a has nice subscripts for
 *   scatterx and gatherx routines.
 * It can have duplicate.
 * It can have vector subscript
 *    but vector subscript has to has scalar or idx.
 * other subscript can have a*i+c
 */
static LOGICAL
is_scatterx_gatherx_subscript(int ast, int forall)
{
  int a, list;
  /* test that vector subscript must not be diagonal access */
  list = A_LISTG(forall);
  a = ast;
  while (A_TYPEG(a) != A_ID) {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      int i, ndim, sptr, dtype;
      int asd;
      sptr = sptr_of_subscript(a);
      dtype = DTYPEG(sptr);
      assert(DTY(dtype) == TY_ARRAY,
             "is_gatherx_scatterx_subscript: must be array", sptr, 4);

      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; ++i) {
        int ss;
        ss = ASD_SUBS(asd, i);
        if (is_scalar(ss, list) || is_idx(ss, list)) {
          /* ok */
        } else if (is_vector_subscript(ss, list)) {
          int lb;
          if (is_duplicate(ss, list))
            return FALSE;
          lb = ADD_LWBD(dtype, i);
          if (XBIT(58, 0x22) && lb != 0 && lb != astb.bnd.one)
            /* -Mhpf2, arrays are reindexed 1:extent
             * for gather/scatter to work, array must really be
             * indexed 1:extent */
            return FALSE;
        } else {
          /* not ok */
          return FALSE;
        }
      }
      a = A_LOPG(a);
    } else {
      interr("is_gatherx_scatterx_subscript: not member or subscript",
             A_TYPEG(a), 3);
      return FALSE;
    }
  }
  return TRUE;
}

static void
scatter_type(int std)
{
  int forall;
  int asn;
  int lhs;
  int src;
  int lop, rop;
  int op;
  int func;
  int n;
  int argt;

  comminfo.scat.result = 0;
  comminfo.scat.mask = 0;
  comminfo.scat.base = 0;
  comminfo.scat.array = 0;
  comminfo.scat.operator= 0;
  comminfo.scat.function = 0;
  comminfo.scat.array_simple = FALSE;

  forall = STD_AST(std);
  asn = A_IFSTMTG(forall);
  src = A_SRCG(asn);
  lhs = A_DESTG(asn);
  comminfo.scat.result = lhs;
  comminfo.scat.mask = A_IFEXPRG(forall);
  if (A_TYPEG(src) == A_SUBSCR) {
    comminfo.scat.array_simple = TRUE;
    comminfo.scat.array = src;
    return;
  }
  lop = 0;
  rop = 0;
  op = 0;
  func = 0;

  if (A_TYPEG(src) == A_BINOP) {
    lop = A_LOPG(src);
    rop = A_ROPG(src);
    op = A_OPTYPEG(src);
    func = 0;
  } else if (A_TYPEG(src) == A_INTR || A_TYPEG(src) == A_FUNC) {
    argt = A_ARGSG(src);
    n = A_ARGCNTG(src);
    if (n != 2)
      return;
    lop = ARGT_ARG(argt, 0);
    rop = ARGT_ARG(argt, 1);
    op = 0;
    func = A_OPTYPEG(src);
  }

  if (lop == 0)
    return;
  if (rop == 0)
    return;

  /* choose one of them as base, the other one array */
  if (lhs == rop) {
    comminfo.scat.base = rop;
    comminfo.scat.array = lop;
  } else if (lhs == lop) {
    comminfo.scat.base = lop;
    comminfo.scat.array = rop;
  } else
    return;

  comminfo.scat.operator= op;
  comminfo.scat.function = func;
  comminfo.scat.array_simple = TRUE;
}
#endif

LOGICAL
scatter_class(int std)
{
  return FALSE;
}

#ifdef FLANG_DETECT_UNUSED
static int
generic_intrinsic_type(int otype)
{

  switch (otype) {
  case I_MAX:
  case I_IMAX0:
  case I_MAX0:
  case I_AMAX1:
  case I_DMAX1:
  case I_JMAX0:
  case I_AIMAX0:
  case I_AMAX0:
  case I_MAX1:
  case I_IMAX1:
  case I_JMAX1:
  case I_AJMAX0:
    return I_MAX;

  case I_MIN:
  case I_IMIN0:
  case I_MIN0:
  case I_AMIN1:
  case I_DMIN1:
  case I_JMIN0:
  case I_AIMIN0:
  case I_AMIN0:
  case I_MIN1:
  case I_IMIN1:
  case I_JMIN1:
  case I_AJMIN0:
    return I_MIN;

  case I_IAND:
  case I_IIAND:
  case I_JIAND:
    return I_IAND;

  case I_IOR:
  case I_IIOR:
  case I_JIOR:
    return I_IOR;
  case I_IEOR:
  case I_IIEOR:
  case I_JIEOR:
    return I_IEOR;
  default:
    return otype;
  }
}

/* mask has to be array.
 * there should not be any communication between.
 */

static LOGICAL
mask_array_relation(int mask, int array, int forall)
{
  int list;
  int masksptr;

  list = A_LISTG(forall);

  if (!mask)
    return TRUE;
  if (A_TYPEG(mask) != A_SUBSCR)
    return FALSE;
  masksptr = memsym_of_ast(mask);
  if (!TY_ISLOG(DTY(DTYPEG(masksptr) + 1)))
    return FALSE;
  /*    if(!is_same_number_of_idx(mask, array, list)) return FALSE;  */

  if (DTY(DTYPEG(masksptr) + 1) != DT_LOG)
    return FALSE;

  return TRUE;
}

/* scatterx copies base into result by using copy_section.
 * normally forall does not allow that.
 * However, it does some case, For example,
 * A(V) = B(V) + C
 * where the extent of vector V covers extent of A.
 */

static LOGICAL
result_base_relation(int result, int base, int forall)
{

  if (!base)
    return FALSE;
  if (!result)
    return FALSE;

  if (result == base)
    return TRUE;
  return FALSE;
}
#endif

/** \brief Inquire whether array has indirection in its subscripts */
LOGICAL
is_indirection_in_it(int a)
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
        if (is_array_in_expr(ASD_SUBS(asd, i)))
          return TRUE;
      a = A_LOPG(a);
    } else if (A_TYPEG(a) == A_ID) {
      return FALSE;
    } else {
      interr("is_indirection_in_it: LHS not subscript or member", a, 4);
    }
  } while (1);
}

/** \brief Inquire whether an array reference has indirection in its subscripts
    and the indirection has nonscalar subscripts.
 */
LOGICAL
is_nonscalar_indirection_in_it(int a)
{
  do {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      int asd, ndim, i;
      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; ++i)
        if (is_nonscalar_array_in_expr(ASD_SUBS(asd, i)))
          return TRUE;
      a = A_LOPG(a);
    } else if (A_TYPEG(a) == A_ID) {
      return FALSE;
    } else {
      interr("is_indirection_in_it: LHS not subscript or member", a, 4);
    }
  } while (1);
} /* is_nonscalar_indirection_in_it */

/** \brief Inquire whether array has indirection in its subscripts */
LOGICAL
is_vector_indirection_in_it(int a, int list)
{
  do {
    if (A_TYPEG(a) == A_MEM) {
      a = A_PARENTG(a);
    } else if (A_TYPEG(a) == A_SUBSCR) {
      int asd;
      int ndim, i;
      int sub;

      asd = A_ASDG(a);
      ndim = ASD_NDIM(asd);
      for (i = 0; i < ndim; ++i) {
        sub = ASD_SUBS(asd, i);
        if (is_array_in_expr(sub) && is_vector_subscript(sub, list))
          return TRUE;
      }
      a = A_LOPG(a);
    } else if (A_TYPEG(a) == A_ID) {
      return FALSE;
    } else {
      interr("is_indirection_in_it: LHS not subscript or member", a, 4);
    }
  } while (1);
}

/* inquire whether expression has array */
static LOGICAL
is_array_in_expr(int ast)
{

  int argt, n, i;
  int sptr, lop;

  if (ast == 0)
    return FALSE;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    if (is_array_in_expr(A_LOPG(ast)))
      return TRUE;
    return is_array_in_expr(A_ROPG(ast));
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return is_array_in_expr(A_LOPG(ast));
  case A_CMPLXC:
  case A_CNST:
    return FALSE;
  case A_MEM:
    if (is_array_in_expr(A_MEMG(ast)))
      return TRUE;
    return is_array_in_expr(A_PARENTG(ast));
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      if (is_array_in_expr(ARGT_ARG(argt, i)))
        return TRUE;
    }
    return FALSE;

  case A_TRIPLE:
    if (is_array_in_expr(A_LBDG(ast)))
      return TRUE;
    if (is_array_in_expr(A_UPBDG(ast)))
      return TRUE;
    if (is_array_in_expr(A_STRIDEG(ast)))
      return TRUE;
    return FALSE;
  case A_SUBSCR:
    /* if this is a section descriptor array, we don't want to
     * treat this like an indexed subscript */
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
    if (STYPEG(sptr) == ST_DESCRIPTOR || DESCARRAYG(sptr)) /* set in rte.c */
      return FALSE;
    return TRUE;
  case A_ID:
    sptr = A_SPTRG(ast);
    if (DTY(DTYPEG(sptr)) == TY_ARRAY)
      return TRUE;
    return FALSE;
  default:
    interr("is_array_in_expr: bad opc", ast, 3);
    return TRUE;
  }
}

/*
 * inquire whether expression has array reference with nonscalar subscripts
 */
static LOGICAL
is_nonscalar_array_in_expr(int ast)
{
  int argt, n, i, sptr, lop, asd, ndim, sub;

  if (ast == 0)
    return FALSE;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    if (is_nonscalar_array_in_expr(A_LOPG(ast)))
      return TRUE;
    return is_nonscalar_array_in_expr(A_ROPG(ast));
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return is_nonscalar_array_in_expr(A_LOPG(ast));
  case A_CMPLXC:
  case A_CNST:
    return FALSE;
  case A_MEM:
    if (is_nonscalar_array_in_expr(A_MEMG(ast)))
      return TRUE;
    return is_nonscalar_array_in_expr(A_PARENTG(ast));
  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      if (is_nonscalar_array_in_expr(ARGT_ARG(argt, i)))
        return TRUE;
    }
    return FALSE;
  case A_TRIPLE:
    return FALSE;
  case A_SUBSCR:
    /* if this is a section descriptor array, we don't want to
     * treat this like an indexed subscript */
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
    if (STYPEG(sptr) == ST_DESCRIPTOR || DESCARRAYG(sptr)) /* set in rte.c */
      return FALSE;
    /* check the subscripts */
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      sub = ASD_SUBS(asd, i);
      if (A_SHAPEG(sub) || A_TYPEG(sub) == A_TRIPLE)
        return TRUE;
    }
    if (A_TYPEG(lop) == A_MEM) {
      return is_nonscalar_array_in_expr(A_PARENTG(lop));
    } else {
      return FALSE;
    }
  case A_ID:
    sptr = A_SPTRG(ast);
    if (DTY(DTYPEG(sptr)) == TY_ARRAY)
      return TRUE;
    return FALSE;
  default:
    interr("is_nonscalar_array_in_expr: bad opc", ast, 3);
    return TRUE;
  }
} /* is_nonscalar_array_in_expr */

/** \brief Inquire whether expression has distributed array */
LOGICAL
is_dist_array_in_expr(int ast)
{

  int argt, n, i;
  int sptr;

  if (ast == 0)
    return FALSE;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    if (is_dist_array_in_expr(A_LOPG(ast)))
      return TRUE;
    return is_dist_array_in_expr(A_ROPG(ast));
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    return is_dist_array_in_expr(A_LOPG(ast));
  case A_CMPLXC:
  case A_CNST:
    return FALSE;

  case A_MEM:
    if (is_dist_array_in_expr(A_MEMG(ast)))
      return TRUE;
    return is_dist_array_in_expr(A_PARENTG(ast));

  case A_INTR:
  case A_FUNC:
    argt = A_ARGSG(ast);
    n = A_ARGCNTG(ast);
    for (i = 0; i < n; ++i) {
      if (is_dist_array_in_expr(ARGT_ARG(argt, i)))
        return TRUE;
    }
    return FALSE;

  case A_TRIPLE:
    if (is_dist_array_in_expr(A_LBDG(ast)))
      return TRUE;
    if (is_dist_array_in_expr(A_UPBDG(ast)))
      return TRUE;
    if (is_dist_array_in_expr(A_STRIDEG(ast)))
      return TRUE;
    return FALSE;
  case A_SUBSCR:
    return is_dist_array_in_expr(A_LOPG(ast));
  case A_SUBSTR:
    if (is_dist_array_in_expr(A_LEFTG(ast)))
      return TRUE;
    if (is_dist_array_in_expr(A_RIGHTG(ast)))
      return TRUE;
    return is_dist_array_in_expr(A_LOPG(ast));
  case A_ID:
    sptr = A_SPTRG(ast);
    if (DTY(DTYPEG(sptr)) == TY_ARRAY && ALIGNG(sptr))
      return TRUE;
    return FALSE;
  default:
    interr("is_dist_array_in_expr: bad opc", ast, 3);
    return TRUE;
  }
}

/* Algorithm:
 * This routine does not care about distribution types.
 * It only cares about alignment.
 * If lhs and rhs are aligned into same template,
 * this routine finds out which dimension of rhs and lhs
 * matched on the same dim. of template.
 * if it finds, finds their differences according to their align functions.
 */

static void
matched_dim(int a)
{
  int asd, ndim;
  int subinfo;
  int i;
  int arref;

  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  arref = A_RFPTRG(a);
  subinfo = ARREF_SUB(arref);

  for (i = 0; i < ndim; ++i)
    SUBI_LDIM(subinfo + i) = -1;
}

void
search_idx(int ast, int list, int *astli, int *base, int *stride)
{
  int i;
  int base1, base2, stride1, stride2;
  int zero = astb.bnd.zero;
  int opc;
  int astli_a, nidx;

  astli_a = 0;
  nidx = 0;
  search_forall_idx(ast, list, &astli_a, &nidx);
  if (nidx == 0 && astli_a == 0) {
    *stride = zero;
    *base = ast;
    return;
  }

  switch (A_TYPEG(ast)) {
  case A_ID:
    for (i = list; i != 0; i = ASTLI_NEXT(i))
      if (ASTLI_SPTR(i) == A_SPTRG(ast))
        goto found;
    *base = ast;
    *stride = zero;
    return;
  found:
    if (*astli && *astli != i) {
      /* too many index variables */
      *base = 0;
      *stride = 0;
      return;
    }
    *astli = i;
    *stride = mk_isz_cval(1, astb.bnd.dtype);
    *base = zero;
    return;

  case A_BINOP:
    search_idx(A_LOPG(ast), list, astli, &base1, &stride1);
    if (base1 == 0) {
      *base = *stride = 0;
      return;
    }
    search_idx(A_ROPG(ast), list, astli, &base2, &stride2);
    if (base2 == 0) {
      *base = *stride = 0;
      return;
    }
    if (stride1 == zero && stride2 == 0) {
      *base = ast;
      *stride = zero;
      return;
    }
    switch (opc = A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
      *base = opt_binop(opc, base1, base2, A_DTYPEG(ast));
      *stride = opt_binop(opc, stride1, stride2, A_DTYPEG(ast));
      return;

    case OP_MUL:
      if (stride1 == zero) {
        /* invar * induc */
        *base = opt_binop(OP_MUL, base2, base1, A_DTYPEG(ast));
        *stride = opt_binop(OP_MUL, stride2, base1, A_DTYPEG(ast));
      } else if (stride2 == zero) {
        /* induc * invar */
        *base = opt_binop(OP_MUL, base1, base2, A_DTYPEG(ast));
        *stride = opt_binop(OP_MUL, stride1, base2, A_DTYPEG(ast));
      } else {
        /* classic nonlinear */
        *base = *stride = 0;
      }
      return;
    default:
      /* unknown binary op */
      *base = *stride = 0;
      return;
    }

  case A_UNOP:
    search_idx(A_LOPG(ast), list, astli, &base1, &stride1);
    if (base1 == 0) {
      *base = *stride = 0;
      return;
    }
    if (stride1 == zero) {
      *base = ast;
      *stride = zero;
      return;
    }
    switch (opc = A_OPTYPEG(ast)) {
    case OP_ADD:
      *base = base1;
      *stride = stride1;
      return;
    case OP_SUB:
      *base = opt_unop(OP_SUB, base1, A_DTYPEG(ast));
      *stride = opt_unop(OP_SUB, stride1, A_DTYPEG(ast));
      return;
    default:
      /* unknown binary op */
      *base = *stride = 0;
      return;
    }
  case A_CONV:
    search_idx(A_LOPG(ast), list, astli, &base1, &stride1);
    if (base1 == 0) {
      *base = *stride = 0;
      return;
    }
    *base = mk_convert(base1, A_DTYPEG(ast));
    *stride = mk_convert(stride1, A_DTYPEG(ast));
    return;

  case A_CMPLXC:
  case A_CNST:
    *stride = zero;
    *base = ast;
    return;

  case A_PAREN:
    search_idx(A_LOPG(ast), list, astli, &base1, &stride1);
    if (base1 == 0) {
      *base = *stride = 0;
      return;
    }
    *base = base1;
    *stride = stride1;
    return;

  case A_SUBSCR:
    /* see if this is one of the front-end's bounds arrays */
    /* really need invariant info, but ... */
    if (A_TYPEG(A_LOPG(ast)) == A_ID && NODESCG(A_SPTRG(A_LOPG(ast)))) {
      *base = ast;
      *stride = zero;
    } else
      *base = *stride = 0; /* special case indirection? */

    return;

  default:
    *base = *stride = 0;
  }
}

/* subscript index in SUBINFO and forall index list */
static void
process_sub(int sub, int list)
{
  /* Try to classify the type of subscript.  Currently just looks for
   * linear combination of forall index variables.  Set SUBI_IDX(sub)
   * to -1 if no luck.
   */
  int sub_ast;
  int astli;
  int base, stride;

  sub_ast = SUBI_SUB(sub);
  /* must look like: c2 +/- c1 * i where i is an index. */
  /* search for an index & do the recursion */
  astli = 0;
  search_idx(sub_ast, list, &astli, &base, &stride);
  if (base == 0) {
    /* hopeless */
    SUBI_IDX(sub) = -1;
    SUBI_STRIDE(sub) = 0;
    SUBI_BASE(sub) = 0;
    return;
  }
  SUBI_IDX(sub) = astli;
  SUBI_BASE(sub) = base;
  SUBI_STRIDE(sub) = stride;
}

int
process_lhs_sub(int std, int ast)
{
  int lhs, lhsd;
  int arref;
  int asn;
  int list;
  int asd;
  int subinfo;
  int i, numdim;
  int align;
  CTYPE *ct;
  int nd;
  int sptr;

  /* if the lhs is distributed, adjust the forall bounds; insert the
   * communication for the forall statement; adjust the rhs bounds
   */
  comminfo.unstruct = 0;
  nd = A_OPT1G(ast);
  ct = FT_CYCLIC(nd);
  asn = A_IFSTMTG(ast);
  /* get the array */
  lhs = A_DESTG(asn);
  ct->lhs = lhs;
  lhsd = left_subscript_ast(lhs);
  sptr = sptr_of_subscript(lhsd);

  align = ALIGNG(sptr);
  list = A_LISTG(ast); /* forall var list */
  asd = A_ASDG(lhsd);
  numdim = ASD_NDIM(asd);
  /* process the subscripts */
  arref = trans.arrb.stg_avail++;
  TRANS_NEED(trans.arrb, ARREF, trans.arrb.stg_size + 100);
  A_RFPTRP(lhsd, arref);
  trans.lhs = arref;

  subinfo = trans.subb.stg_avail;
  trans.subb.stg_avail += numdim;
  assert(numdim < 100, "transform_forall: numdim huge?", 0, 4);
  TRANS_NEED(trans.subb, SUBINFO, trans.subb.stg_size + 100);
  ARREF_SUB(arref) = subinfo;
  ARREF_NDIM(arref) = numdim;
  ARREF_TEMP(arref) = 0;
  ARREF_NEXT(arref) = 0;
  ARREF_CLASS(arref) = NO_CLASS;
  ARREF_ARRSYM(arref) = sptr;
  ARREF_ARR(arref) = lhsd;
  for (i = 0; i < numdim; ++i) {
    SUBI_SUB(subinfo + i) = ASD_SUBS(asd, i);
    SUBI_DSTT(subinfo + i) = 0;
    SUBI_NOP(subinfo + i) = 0;
    SUBI_POP(subinfo + i) = 0;
    SUBI_DUPL(subinfo + i) = 0;
    /* find out what kind of subscript this is */
    process_sub(subinfo + i, list);
  }
  /* if A(i,i), one of them will be SUBI_IDX and
     the other one will be SUBI_BASE */
  convert_idx_scalar(arref);

  for (i = 0; i < numdim; ++i)
    if (SUBI_BASE(subinfo + i) != 0)
      ct->idx[i] = SUBI_IDX(subinfo + i);

  /* Check the communications */
  comminfo.std = std;
  comminfo.subinfo = subinfo;
  comminfo.lhs = A_LOPG(lhsd);
  comminfo.sub = /* lhsd */ left_nonscalar_subscript_ast(lhs);
  comminfo.forall = ast;
  comminfo.mask_phase = 0;
  comminfo.ugly_mask = 0;

  return 1;
}

/* This routine is to scalarize SUBI_IDX if it appears more than one.
 * It chooses distributed dimension over non-distributed.
 * It chooses BLOCK over CYCLIC, CYCLIC(1) over CYCLIC(general).
 * If it finds duplicate, it makes everything SUBI_BASE.
 */

static void
convert_idx_scalar(int arref)
{
  int ndim;
  int i, j;
  int subinfo;
  int choice;
  int iprio, choiceprio;
  int zero = astb.bnd.zero;

  subinfo = ARREF_SUB(arref);
  ndim = ARREF_NDIM(arref);
  for (i = 0; i < ndim; ++i) {
    if (SUBI_IDX(subinfo + i) == 0)
      continue;
    if (SUBI_IDX(subinfo + i) == -1)
      continue;
    choice = i;
    choiceprio = 1;
    for (j = i + 1; j < ndim; j++) {
      if (SUBI_IDX(subinfo + choice) == SUBI_IDX(subinfo + j)) {
        /* same index, what is 'priority' of this distribution */
        iprio = 1;
        if (iprio < choiceprio) {
          choice = i;
          choiceprio = iprio;
        }
      }
    }

    for (j = 0; j < ndim; j++) {
      if (j == choice)
        continue;
      if (SUBI_IDX(subinfo + choice) == SUBI_IDX(subinfo + j)) {
        SUBI_DUPL(subinfo + j) = 1;
        SUBI_IDX(subinfo + j) = 0;
        SUBI_STRIDE(subinfo + j) = zero;
        SUBI_BASE(subinfo + j) = SUBI_SUB(subinfo + j);
      }
    }
  }
}

#ifdef FLANG_DETECT_UNUSED
/* This routine is to check
 * whether all ind from list appears at array subscript
 * For example, (i=, j=) a(i,j) true
 *  (i=, j=) a(1,j) false
 */
static LOGICAL
is_all_idx_appears(int a, int list)
{
  int j;
  for (j = list; j != 0; j = ASTLI_NEXT(j)) {
    int sptr, ast;
    LOGICAL found;
    sptr = ASTLI_SPTR(j);
    found = FALSE;
    ast = a;
    do {
      if (A_TYPEG(ast) == A_MEM) {
        ast = A_PARENTG(ast);
      } else if (A_TYPEG(ast) == A_SUBSCR) {
        int asd, ndim, i;
        asd = A_ASDG(ast);
        ndim = ASD_NDIM(asd);
        for (i = 0; i < ndim; ++i)
          if (is_name_in_expr(ASD_SUBS(asd, i), sptr))
            found = TRUE;
        ast = A_LOPG(ast);
      } else {
        interr("is_all_idx_appears: must be subscript or member", ast, 3);
      }
    } while (!found && A_TYPEG(ast) != A_ID);
    if (!found)
      return FALSE;
  }
  return TRUE;
}
#endif

