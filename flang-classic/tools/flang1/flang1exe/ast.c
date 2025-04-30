/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
  \brief Abstract syntax tree access module.

  This module contains the routines used to initialize, update, access, and
  dump the abstract syntax tree.

  <pre>
  q flags:
      -q  4  256   dump asts
      -q  4  512   include hash table of asts
  </pre>
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "scan.h"
#include "machar.h"
#include "state.h"
#include "ast.h"
#include "pragma.h"
#include "rte.h"
#include "extern.h"
#include "rtlRtns.h"
#include "semant.h" /* for get_temp */

#define HUGE_NUM2 0xffffffff
#define HUGE_NUM3 0xffffffff
#define BYTE_NUMBER16 16
static int reduce_iadd(int, INT);
static int reduce_i8add(int, int);
static int convert_cnst(int, int);
static SPTR sym_of_ast2(int);
static LOGICAL bounds_match(int, int, int);
static INT _fdiv(INT, INT);
static void _ddiv(INT *, INT *, INT *);
static int hex2char(INT *);
static int hex2nchar(INT *);
static void truncation_warning(int);
static void conversion_warning(void);

static int atemps; /* temp counter for bounds' temporaries */

#define MIN_INT64(n) \
  (((n[0] & 0xffffffff) == 0x80000000) && ((n[1] & 0xffffffff) == 0))

/** \brief Initialize AST table for new user program unit.
 */
void
ast_init(void)
{
  int i;

#if DEBUG
  assert(sizeof(AST) / sizeof(int) == 19, "bad AST size",
         sizeof(AST) / sizeof(int), 4);
#endif

  /* allocate AST and auxiliary structures: */

  if (astb.stg_size <= 0) {
    STG_ALLOC(astb, 2000);
#if DEBUG
    assert(astb.stg_base, "ast_init: no room for AST", astb.stg_size, 4);
#endif
  } else {
    STG_RESET(astb);
  }
  STG_NEXT(astb); /* reserve ast index 1 to terminate ast_traverse() */
  BZERO(astb.hshtb, int, HSHSZ + 1);

  if (astb.asd.stg_size <= 0) {
    astb.asd.stg_size = 200;
    NEW(astb.asd.stg_base, int, astb.asd.stg_size);
#if DEBUG
    assert(astb.asd.stg_base, "ast_init: no room for ASD", astb.asd.stg_size, 4);
#endif
  }
  BZERO(astb.asd.hash, int, 7);
  astb.asd.stg_base[0] = 0;
  astb.asd.stg_avail = 1;

  if (astb.shd.stg_size <= 0) {
    astb.shd.stg_size = 200;
    NEW(astb.shd.stg_base, SHD, astb.shd.stg_size);
#if DEBUG
    assert(astb.shd.stg_base, "ast_init: no room for SHD", astb.shd.stg_size, 4);
#endif
  } else
    BZERO(astb.shd.hash, int, 7);
  astb.shd.stg_base[0].lwb = 0;
  astb.shd.stg_base[0].upb = 0;
  astb.shd.stg_base[0].stride = 0;
  astb.shd.stg_avail = 1;

  if (astb.std.stg_size <= 0) {
    STG_ALLOC(astb.std, 200);
#if DEBUG
    assert(astb.std.stg_base, "ast_init: no room for STD", astb.std.stg_size, 4);
#endif
  } else {
    STG_RESET(astb.std);
  }

  STD_PREV(0) = STD_NEXT(0) = 0;
  STD_FLAGS(0) = 0;
  STD_LINENO(0) = 0;
  STD_FINDEX(0) = 0;

  if (astb.astli.stg_size <= 0) {
    astb.astli.stg_size = 200;
    NEW(astb.astli.stg_base, ASTLI, astb.astli.stg_size);
#if DEBUG
    assert(astb.astli.stg_base, "ast_init: no room for ASTLI", astb.astli.stg_size, 4);
#endif
  }
  astb.astli.stg_avail = 1;
  astb.astli.stg_base[0].h1 = 0;
  astb.astli.stg_base[0].h2 = 0;
  astb.astli.stg_base[0].flags = 0;
  astb.astli.stg_base[0].next = 0;

  if (astb.argt.stg_size <= 0) {
    astb.argt.stg_size = 200;
    NEW(astb.argt.stg_base, int, astb.argt.stg_size);
#if DEBUG
    assert(astb.argt.stg_base, "ast_init: no room for ARGT", astb.argt.stg_size, 4);
#endif
  }
  astb.argt.stg_avail = 1;
  astb.argt.stg_base[0] = 0;

  if (astb.comstr.stg_size <= 0) {
    astb.comstr.stg_size = 200;
    NEW(astb.comstr.stg_base, char, astb.comstr.stg_size);
#if DEBUG
    assert(astb.comstr.stg_base, "ast_init: no room for COMSTR", astb.comstr.stg_size,
           4);
#endif
  }
  astb.comstr.stg_avail = 0;
  astb.comstr.stg_base[0] = 0;

  BZERO(astb.implicit, char, sizeof(astb.implicit));

  BZERO(astb.stg_base + 0, AST, 2); /* initialize AST #0 and #1 */
                                /*
                                 * WARNING --- any changes/additions to the predeclared ASTs
                                 * need to be reflected in the interf/exterf module processing.
                                 * The ASTs before astb.firstuast are not written to the .mod
                                 * file and are used asis when encountered during the read processing.
                                 * NOTE that the current value of firstuast is 12 !!!
                                 */
  astb.i0 = mk_cval((INT)0, DT_INT);
  astb.i1 = mk_cval((INT)1, DT_INT);
/*
 * ensure that unique asts represent (void *)0, (void *)1, and the
 * character value indicating a non-present I/O character specifier.
 * Use %val with ID asts of a few predeclared symbol table pointers.
 * WARNING:  need to ensure that the ID ASTs have the same data type
 * as the %val ASTs.
 */
#define MKU(a, s, d)           \
  {                            \
    i = new_node(A_ID);        \
    A_SPTRP(i, s);             \
    A_DTYPEP(i, d);            \
    a = mk_unop(OP_VAL, i, d); \
  }

  MKU(astb.ptr0, 1, DT_INT);
  MKU(astb.ptr1, 2, DT_INT);
  MKU(astb.ptr0c, 3, DT_CHAR);

#undef MKU

  /*
   * astb.k0 & astb.k1 used to be created with astb.i0 & astb.i1, but
   * that caused compatibility problems with older modfiles.
   * the new predeclareds are added to the end of the predeclared
   * area, so that numbering of the older predeclareds remains
   * the same.
   */
  astb.k0 = mk_cval((INT)0, DT_INT8);
  astb.k1 = mk_cval((INT)1, DT_INT8);

  if (XBIT(68, 0x1)) {
    astb.bnd.dtype = DT_INT8;
    astb.bnd.zero = astb.k0;
    astb.bnd.one = astb.k1;
  } else {
    astb.bnd.dtype = DT_INT;
    astb.bnd.zero = astb.i0;
    astb.bnd.one = astb.i1;
  }

  /* fix length of DT_CHAR, DT_NCHAR */
  DTY(DT_CHAR + 1) = astb.bnd.one;
  DTY(DT_NCHAR + 1) = astb.bnd.one;

  atemps = 0;
  astb.firstuast = astb.stg_avail;
#if DEBUG
  assert(astb.firstuast == 12,
         "ast_init(): # of predeclared ASTs has changed -- fix interf or IVSN",
         astb.firstuast, 4);
#endif

  /* integer array(1) data type record */
  aux.dt_iarray = DT_IARRAY;

  DTY(DT_IARRAY + 1) = stb.user.dt_int;
  get_aux_arrdsc(DT_IARRAY, 1);
  ADD_LWAST(DT_IARRAY, 0) = 0;
  ADD_UPBD(DT_IARRAY, 0) = ADD_UPAST(DT_IARRAY, 0) =
      ADD_EXTNTAST(DT_IARRAY, 0) = astb.bnd.one;

  if (stb.user.dt_int == DT_INT) {
    aux.dt_iarray_int = aux.dt_iarray;
  } else {
    aux.dt_iarray_int = get_array_dtype(1, DT_INT);
    ADD_LWAST(aux.dt_iarray_int, 0) = 0;
    ADD_UPBD(aux.dt_iarray_int, 0) = ADD_UPAST(aux.dt_iarray_int, 0) =
        ADD_EXTNTAST(aux.dt_iarray_int, 0) = astb.bnd.one;
  }
}

void
ast_fini(void)
{
  if (astb.stg_base) {
    STG_DELETE(astb);
  }
  if (astb.asd.stg_base) {
    FREE(astb.asd.stg_base);
    astb.asd.stg_avail = astb.asd.stg_size = 0;
  }
  if (astb.shd.stg_base) {
    FREE(astb.shd.stg_base);
    astb.shd.stg_avail = astb.shd.stg_size = 0;
  }
  if (astb.std.stg_base) {
    STG_DELETE(astb.std);
  }
  if (astb.astli.stg_base) {
    FREE(astb.astli.stg_base);
    astb.astli.stg_avail = astb.astli.stg_size = 0;
  }
  if (astb.argt.stg_base) {
    FREE(astb.argt.stg_base);
    astb.argt.stg_avail = astb.argt.stg_size = 0;
  }
  if (astb.comstr.stg_base) {
    FREE(astb.comstr.stg_base);
    astb.comstr.stg_avail = astb.comstr.stg_size = 0;
  }
} /* ast_fini */

int
new_node(int type)
{
  int nd;

  nd = STG_NEXT(astb);
  if (nd > MAXAST || astb.stg_base == NULL)
    errfatal(7);
  A_TYPEP(nd, type);
  return nd;
}

#define ADD_NODE(nd, a, hashval)       \
  (nd) = new_node(a);                  \
  A_HSHLKP((nd), astb.hshtb[hashval]); \
  astb.hshtb[hashval] = (nd)

/* not used
#define HSH_0(a) hash_val(a, -1, -1, -1, -1)
#define HSH_1(a,o1) hash_val(a, o1, -1, -1, -1)
*/
#define HSH_2(a, o1, o2) hash_val(a, o1, o2, -1, -1)
#define HSH_3(a, o1, o2, o3) hash_val(a, o1, o2, o3, -1)
#define HSH_4(a, o1, o2, o3, o4) hash_val(a, o1, o2, o3, o4)

static INT
hash_val(int a, int hw3, int hw4, int hw5, int hw6)
{
  INT hashval;

  hashval = a;
  if (hw3 > 0)
    hashval ^= hw3 >> 4;
  if (hw4 > 0)
    hashval ^= hw4 << 8;
  if (hw5 > 0)
    hashval ^= hw5 >> 8;
  if (hw6 > 0)
    hashval ^= hw6 << 16;
  hashval &= 0x7fffffff;
  hashval %= HSHSZ;
  return hashval;
}

/* hash an ast with dtype & sptr (A_ID, A_CNST, A_LABEL) */
static int
hash_sym(int a, DTYPE dtype, int sptr)
{
  INT hashval;
  int nd;

  hashval = HSH_2(a, dtype, sptr);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && sptr == A_SPTRG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  if (dtype)
    A_DTYPEP(nd, dtype);
  A_SPTRP(nd, sptr);
  return nd;
}

/* hash an A_UNOP ast */
static int
hash_unop(int a, DTYPE dtype, int lop, int optype)
{
  INT hashval;
  int nd;

  hashval = HSH_3(a, dtype, lop, optype);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && lop == A_LOPG(nd) &&
        optype == A_OPTYPEG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_LOPP(nd, lop);
  A_OPTYPEP(nd, optype);
  return nd;
}

/* hash an A_BINOP op ast */
static int
hash_binop(int a, DTYPE dtype, int lop, int optype, int rop)
{
  INT hashval;
  int nd;

  hashval = HSH_4(a, dtype, lop, optype, rop);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && lop == A_LOPG(nd) &&
        optype == A_OPTYPEG(nd) && rop == A_ROPG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_LOPP(nd, lop);
  A_OPTYPEP(nd, optype);
  A_ROPP(nd, rop);
  return nd;
}

/* hash an A_PAREN ast */
static int
hash_paren(int a, DTYPE dtype, int lop)
{
  INT hashval;
  int nd;

  hashval = HSH_2(a, dtype, lop);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && lop == A_LOPG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_LOPP(nd, lop);
  return nd;
}

/* hash an A_CONV ast */
static int
hash_conv(int a, DTYPE dtype, int lop, int shd)
{
  INT hashval;
  int nd;

  hashval = HSH_3(a, dtype, lop, shd);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && lop == A_LOPG(nd) &&
        (!shd || shd == A_SHAPEG(nd)))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_LOPP(nd, lop);
  return nd;
}

/* hash an A_SUBSCR ast */
static int
hash_subscr(int a, DTYPE dtype, int lop, int asd)
{
  INT hashval;
  int nd;

  hashval = HSH_3(a, dtype, lop, asd);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && lop == A_LOPG(nd) &&
        asd == A_ASDG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_LOPP(nd, lop);
  A_ASDP(nd, asd);
  return nd;
}

/* hash an A_MEM ast */
static int
hash_mem(int a, DTYPE dtype, int parent, int mem)
{
  INT hashval;
  int nd;

  hashval = HSH_3(a, dtype, parent, mem);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && parent == A_PARENTG(nd) &&
        mem == A_MEMG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_PARENTP(nd, parent);
  A_MEMP(nd, mem);
  return nd;
}

/* hash an A_CMPLXC ast */
static int
hash_cmplxc(int a, DTYPE dtype, int lop, int rop)
{
  INT hashval;
  int nd;

  hashval = HSH_3(a, dtype, lop, rop);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && lop == A_LOPG(nd) &&
        rop == A_ROPG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_LOPP(nd, lop);
  A_ROPP(nd, rop);
  return nd;
}

/* hash an A_TRIPLE ast */
static int
hash_triple(int a, int lb, int ub, int stride)
{
  INT hashval;
  int nd;

  hashval = HSH_3(a, lb, ub, stride);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && lb == A_LBDG(nd) && ub == A_UPBDG(nd) &&
        stride == A_STRIDEG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_LBDP(nd, lb);
  A_UPBDP(nd, ub);
  A_STRIDEP(nd, stride);
  return nd;
}

/* hash an A_SUBSTR ast */
static int
hash_substr(int a, DTYPE dtype, int lop, int left, int right)
{
  INT hashval;
  int nd;

  hashval = HSH_4(a, dtype, lop, left, right);
  for (nd = astb.hshtb[hashval]; nd != 0; nd = A_HSHLKG(nd)) {
    if (a == A_TYPEG(nd) && dtype == A_DTYPEG(nd) && lop == A_LOPG(nd) &&
        left == A_LEFTG(nd) && right == A_RIGHTG(nd))
      return nd;
  }
  ADD_NODE(nd, a, hashval);
  A_DTYPEP(nd, dtype);
  A_LOPP(nd, lop);
  A_LEFTP(nd, left);
  A_RIGHTP(nd, right);
  return nd;
}

int
mk_id(int id)
{
  int ast = mk_id_noshape(id);
  if (A_SHAPEG(ast) == 0)
    A_SHAPEP(ast, mkshape(DTYPEG(id)));
  return ast;
}

int
mk_id_noshape(int id)
{
  if (id <= NOSYM || id >= stb.stg_avail) {
    interr("mk_id: invalid symbol table index", id, ERR_Severe);
  }
  return hash_sym(A_ID, DTYPEG(id), id); /* defer shape to later */
}

int
mk_init(int left, DTYPE dtype)
{
  int ast;
  ast = new_node(A_INIT);
  A_DTYPEP(ast, dtype);
  A_LEFTP(ast, left);
  return ast;
} /* mk_init */

int
mk_atomic(int stmt_type, int left, int right, DTYPE dtype)
{
  int ast;
  ast = new_node(stmt_type);
  A_DTYPEP(ast, dtype);
  A_LOPP(ast, left);
  A_ROPP(ast, right);
  return ast;
} /* mk_atomic */

/** \brief Make a constant AST given a constant symbol table pointer
 */
int
mk_cnst(int cnst)
{
  int ast;

  ast = hash_sym(A_CNST, DTYPEG(cnst), cnst);
  A_ALIASP(ast, ast);
  if (A_SHAPEG(ast) == 0 && DTY(DTYPEG(cnst)) == TY_ARRAY)
    A_SHAPEP(ast, mkshape((int)DTYPEG(cnst)));
  return ast;
}

int
mk_cval(INT v, DTYPE dtype)
{
  /* DT_INT may be 4 or 8 bytes, DT_LOG may be 4 or 8 bytes. This
   * function assumes that DT_INT and DT_LOG are always passed as a
   * 32-bit value, converts them appropriately if necessary, and
   * calls the 'real' mk_cval1.
   */
  DBLINT64 v1;

  if (DTY(dtype) == TY_INT8) {
    if (v < 0)
      v1[0] = -1;
    else
      v1[0] = 0;
    v1[1] = v;
    return mk_cval1(getcon(v1, DT_INT8), DT_INT8);
  }
  if (DTY(dtype) == TY_LOG8) {
    if (v < 0)
      v1[0] = -1;
    else
      v1[0] = 0;
    v1[1] = v;
    return mk_cval1(getcon(v1, DT_LOG8), DT_LOG8);
  }
  return mk_cval1(v, dtype);
}

int
mk_isz_cval(ISZ_T v, DTYPE dtype)
{
  if (dtype == DT_INT8) {
    DBLINT64 num;

    ISZ_2_INT64(v, num);
    return mk_cval1(getcon(num, DT_INT8), DT_INT8);
  }
  return mk_cval(v, dtype);
}

int
mk_fake_iostat()
{
  return mk_id(get_temp(astb.bnd.dtype));
}

/** \brief Make a constant AST given the actual (single word) value or
    a constant symbol table pointer; determined by data type.
 */
int
mk_cval1(INT v, DTYPE dtype)
{
  int cnst = 0;
  static INT val[2];
  int ast;

  switch (DTY(dtype)) {
  case TY_WORD:
  case TY_INT:
  case TY_LOG:
  case TY_REAL:
  case TY_SINT:
  case TY_BINT:
  case TY_SLOG:
  case TY_BLOG:
    if (v < 0)
      val[0] = -1;
    else
      val[0] = 0;
    val[1] = v;
    cnst = getcon(val, dtype);
    break;

  case TY_INT8:
  case TY_LOG8:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
  case TY_DWORD:
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
  case TY_NCHAR:
  case TY_HOLL:
  case TY_CHAR:
    cnst = v;
    break;

  case TY_PTR:
    interr("mk_cval1:ptr v", dtype, 3);
    break;

  default:
    interr("mk_cval1:baddtype", dtype, 1);
  }

  ast = hash_sym(A_CNST, dtype, cnst);
  A_ALIASP(ast, ast);

  if (A_SHAPEG(ast) == 0 && DTY(dtype) == TY_ARRAY)
    A_SHAPEP(ast, mkshape(dtype));
  return ast;
}

/** \brief Create an alias of ast if it isn't a constant AST.
    Its alias field will be set to the ast 'a_cnst'.
 */
void
mk_alias(int ast, int a_cnst)
{
  if (A_TYPEG(ast) != A_CNST)
    A_ALIASP(ast, a_cnst);
}

int
mk_label(int lab)
{
  return hash_sym(A_LABEL, 0, lab);
}

int
mk_binop(int optype, int lop, int rop, DTYPE dtype)
{
  int ast;
  int tmp;
  int ncons;
  LOGICAL commutable;
  INT v1, v2;
  int c1, c2;
  DBLINT64 inum1, inum2;

#if DEBUG
  if (A_TYPEG(lop) == A_TRIPLE || A_TYPEG(rop) == A_TRIPLE) {
    interr("mk_binop: trying to operate on a triplet", optype, 3);
  }
#endif
  switch (optype) {
  case OP_ADD:
  case OP_SUB:
  case OP_MUL:
  case OP_DIV:
    if (DTY(dtype) == TY_INT8 || DTY(dtype) == TY_LOG8) {
      lop = convert_int(lop, dtype);
      rop = convert_int(rop, dtype);
    }
    break;
  case OP_XTOI:
    if (DTY(dtype) == TY_INT8 || DTY(dtype) == TY_LOG8) {
      lop = convert_int(lop, dtype);
    }
    FLANG_FALLTHROUGH;
  default:
    break;
  }
  c1 = c2 = ncons = 0;
  commutable = FALSE;
  switch (optype) {
  case OP_MUL:
  case OP_ADD:
  case OP_LEQV:
  case OP_LNEQV:
  case OP_LOR:
  case OP_LAND:
    commutable = TRUE;
    FLANG_FALLTHROUGH;
  default:
    if (A_TYPEG(lop) == A_CNST) {
      ncons = 1;
      c1 = A_SPTRG(lop);
    }
    if (A_TYPEG(rop) == A_CNST) {
      ncons |= 2;
      c2 = A_SPTRG(rop);
    }
    if (commutable) {
      if (ncons == 1) {
        /*
         * make the left constant the right operand; note that for OP_LOR and
         * OP_LAND, 'folding' only examines the right operand.
         */
        tmp = lop;
        lop = rop;
        rop = tmp;
        c2 = c1;
        c1 = 0;
      } else if (ncons == 0 && lop > rop) {
        tmp = lop;
        lop = rop;
        rop = tmp;
      }
    }
    break;
  }

  if (ncons != 0 && DT_ISINT(dtype))
    switch (DTY(dtype)) {
    case TY_INT8:
    case TY_LOG8:
      switch (optype) {
      case OP_MUL:
        if (c2 == stb.k1)
          return lop;
        if (!A_CALLFGG(lop) && c2 == stb.k0)
          return mk_cnst(stb.k0);
        if (ncons == 3) {
          v1 = const_fold(OP_MUL, c1, c2, dtype);
          return mk_cnst(v1);
        }
        break;
      case OP_ADD:
        if (c2 == stb.k0)
          return lop;
        if (ncons & 2) {
          ast = reduce_i8add(lop, c2);
          if (ast)
            return ast;
          inum1[0] = CONVAL1G(c2);
          inum1[1] = CONVAL2G(c2);
          inum2[0] = 0;
          inum2[1] = 0;
          if (MIN_INT64(inum1))
            break;
          if (cmp64(inum1, inum2) < 0) {
            c2 = negate_const(c2, DT_INT8);
            rop = mk_cnst(c2);
            optype = OP_SUB;
          }
        }
        break;
      case OP_SUB:
        if (ncons == 1) {
          if (c1 == stb.k0)
            return mk_unop(OP_SUB, rop, dtype);
          break;
        }
        /* the second operand is a constant; the first operand may be a
         * constant.
         */
        if (c2 == stb.k0)
          return lop;
        inum1[0] = CONVAL1G(c2);
        inum1[1] = CONVAL2G(c2);
        if (MIN_INT64(inum1))
          break;
        tmp = negate_const(c2, DT_INT8);
        ast = reduce_i8add(lop, tmp);
        if (ast)
          return ast;
        inum2[0] = 0;
        inum2[1] = 0;
        if (cmp64(inum1, inum2) < 0) {
          c2 = negate_const(c2, DT_INT8);
          rop = mk_cnst(c2);
          optype = OP_ADD;
        }
        break;
      case OP_DIV:
        if (!A_CALLFGG(rop) && c1 == stb.k0)
          return mk_cnst(stb.k0);
        if (c2 == stb.k1)
          return lop;
        if (ncons == 3) {
          v1 = const_fold(OP_DIV, c1, c2, dtype);
          return mk_cnst(v1);
        }
        break;
      case OP_XTOI:
        if (c2 == stb.k1)
          return lop;
        if (!A_CALLFGG(lop) && c2 == stb.k0)
          return mk_cnst(stb.k1);
        if (!A_CALLFGG(rop)) {
          if (c1 == stb.k0)
            return mk_cnst(stb.k0);
          if (c1 == stb.k1)
            return mk_cnst(stb.k1);
        }
        break;
      default:
        break;
      }
      break;

    default:
      switch (optype) {
      case OP_MUL:
        if (rop == astb.i1)
          return lop;
        if (!A_CALLFGG(lop) && rop == astb.i0)
          return astb.i0;
        if (ncons == 3) {
          v1 = CONVAL2G(A_SPTRG(lop));
          v2 = CONVAL2G(A_SPTRG(rop));
          ast = mk_cval(v1 * v2, DT_INT);
          return ast;
        }
        break;
      case OP_ADD:
        v2 = CONVAL2G(A_SPTRG(rop));
        if (v2 == 0)
          return lop;
        if (ncons & 2) {
          ast = reduce_iadd(lop, v2);
          if (ast)
            return ast;
          if (v2 == (INT)0x80000000)
            break;
          if (v2 < 0) {
            rop = mk_cval(-v2, DT_INT);
            optype = OP_SUB;
          }
        }
        break;
      case OP_SUB:
        if (ncons == 1) {
          if (lop == astb.i0)
            return mk_unop(OP_SUB, rop, DT_INT);
          break;
        }
        /* the second operand is a constant; the first operand may be a
         * constant.
         */
        v2 = CONVAL2G(A_SPTRG(rop));
        if (v2 == 0)
          return lop;
        if (v2 == (INT)0x80000000)
          break;
        ast = reduce_iadd(lop, -v2);
        if (ast)
          return ast;
        if (v2 < 0) {
          rop = mk_cval(-v2, DT_INT);
          optype = OP_ADD;
        }
        break;
      case OP_DIV:
        if (!A_CALLFGG(rop) && lop == astb.i0)
          return astb.i0;
        if (rop == astb.i1)
          return lop;
        if (ncons == 3) {
          v1 = CONVAL2G(A_SPTRG(lop));
          v2 = CONVAL2G(A_SPTRG(rop));
          if (v2 == 0)
            break;
          ast = mk_cval(v1 / v2, DT_INT);
          return ast;
        }
        break;
      case OP_XTOI:
        if (rop == astb.i1)
          return lop;
        if (!A_CALLFGG(lop) && rop == astb.i0)
          return astb.i1;
        if (!A_CALLFGG(rop)) {
          if (lop == astb.i0)
            return astb.i0;
          if (lop == astb.i1)
            return astb.i1;
        }
        if (ncons == 3) {
          INT v;
          v1 = CONVAL2G(A_SPTRG(lop));
          v2 = CONVAL2G(A_SPTRG(rop));
          if (v2 < 0) {
            if (v1 == -1)
              /* if v2 is odd number, the result will be -1 */
              return (v2 & 1) ? mk_cval((INT)-1, DT_INT) : astb.i1;
            return astb.i0;
          }
          v = v1;
          while (--v2 > 0)
            v *= v1;
          ast = mk_cval(v, DT_INT);
          return ast;
        }
        break;
      case OP_LAND:
        v2 = CONVAL2G(A_SPTRG(rop));
        if (v2 == 0)
          return rop; /* something .and. .false. is .false */
        return lop;   /* something .and. .true. is something */
        break;
      case OP_LOR:
        v2 = CONVAL2G(A_SPTRG(rop));
        if (v2 != 0)
          return rop; /* something .or. .true. is .true */
        return lop;   /* something .or. .false. is something */
        break;
      default:
        break;
      }
      break;
    }

  if (DT_ISINT(dtype))
    switch (optype) {
    case OP_SUB:
      if (A_CALLFGG(rop))
        break;
      if (lop == rop) {
        switch (DTY(dtype)) {
        case TY_INT8:
        case TY_LOG8:
          return mk_cnst(stb.k0);
        default:
          return astb.i0;
        }
      } else if (A_DTYPEG(lop) == A_DTYPEG(rop) && A_TYPEG(lop) == A_BINOP &&
                 A_OPTYPEG(lop) == OP_ADD) {
        if (A_LOPG(lop) == rop) {
          return A_ROPG(lop);
        } else if (A_ROPG(lop) == rop) {
          return A_LOPG(lop);
        }
      }
      break;
    case OP_DIV:
      if (A_CALLFGG(rop))
        break;
      if (lop == rop)
        switch (DTY(dtype)) {
        case TY_INT8:
        case TY_LOG8:
          return mk_cnst(stb.k1);
        default:
          return astb.i1;
        }
      break;
    default:
      break;
    }

  ast = hash_binop(A_BINOP, dtype, lop, optype, rop);
  A_CALLFGP(ast, A_CALLFGG(lop) | A_CALLFGG(rop));
  A_SHAPEP(ast, A_SHAPEG(lop));
  return ast;
}

/* ast of left of '+' */
/* value of constant */
static int
reduce_iadd(int opnd, INT con)
{
  int new;
  INT v1;
  int lop, rop;
  int tmp;

#if DEBUG
  assert(opnd, "reduce_iadd:opnd is 0", con, 3);
#endif

  switch (A_TYPEG(opnd)) {
  case A_CNST:
    v1 = CONVAL2G(A_SPTRG(opnd));
    new = mk_cval(v1 + con, DT_INT);
    return new;

  case A_BINOP:
    switch (A_OPTYPEG(opnd)) {
    case OP_ADD:
      lop = A_LOPG(opnd);
      rop = A_ROPG(opnd);
      new = reduce_iadd(rop, con);
      if (new) {
        if (new == astb.i0)
          return lop;
        if (A_TYPEG(new) == A_CNST) {
          v1 = CONVAL2G(A_SPTRG(new));
          if (v1 < 0 && v1 != (INT)0x80000000) {
            new = mk_cval(-v1, DT_INT);
            new = hash_binop(A_BINOP, DT_INT, lop, OP_SUB, new);
            A_CALLFGP(new, A_CALLFGG(lop));
            A_SHAPEP(new, 0);
            return new;
          }
        } else if (lop > new) {
          tmp = lop;
          lop = new;
          new = tmp;
        }
        new = hash_binop(A_BINOP, DT_INT, lop, OP_ADD, new);
        A_CALLFGP(new, A_CALLFGG(lop) | A_CALLFGG(new));
        A_SHAPEP(new, 0);
        return new;
      }
      new = reduce_iadd(lop, con);
      if (new) {
        if (A_TYPEG(new) != A_CNST && (A_TYPEG(rop) == A_CNST || rop > new)) {
          tmp = rop;
          rop = new;
          new = tmp;
        }
        new = hash_binop(A_BINOP, DT_INT, rop, OP_ADD, new);
        A_CALLFGP(new, A_CALLFGG(rop) | A_CALLFGG(new));
        A_SHAPEP(new, 0);
        return new;
      }
      break;
    case OP_SUB:
      lop = A_LOPG(opnd);
      rop = A_ROPG(opnd);
      new = reduce_iadd(lop, con);
      if (new) {
        if (A_TYPEG(new) == A_CNST && new == astb.i0) {
          new = mk_unop(OP_SUB, rop, DT_INT);
          return new;
        }
        new = hash_binop(A_BINOP, DT_INT, new, OP_SUB, rop);
        A_CALLFGP(new, A_CALLFGG(new) | A_CALLFGG(rop));
        A_SHAPEP(new, 0);
        return new;
      }
      if (con == (INT)0x80000000)
        break;
      new = reduce_iadd(rop, -con);
      if (new) {
        if (new == astb.i0)
          return lop;
        if (A_TYPEG(new) == A_CNST) {
          v1 = CONVAL2G(A_SPTRG(new));
          if (v1 < 0 && v1 != (INT)0x80000000) {
            new = mk_cval(-v1, DT_INT);
            new = hash_binop(A_BINOP, DT_INT, lop, OP_ADD, new);
            A_CALLFGP(new, A_CALLFGG(lop));
            A_SHAPEP(new, 0);
            return new;
          }
        }
        new = hash_binop(A_BINOP, DT_INT, lop, OP_SUB, new);
        A_CALLFGP(new, A_CALLFGG(lop) | A_CALLFGG(new));
        A_SHAPEP(new, 0);
        return new;
      }
      break;
    }
    break;
  default:
    break;
  }

  return 0;
}

/* ast of left of '+' */
/* value of constant, a symbol table pointer */
static int
reduce_i8add(int opnd, int con_st)
{
  int new;
  int c1;
  int lop, rop;
  int tmp;
  DBLINT64 inum1, inum2;

#if DEBUG
  assert(opnd, "reduce_i8add:opnd is 0", con_st, 3);
#endif

  switch (A_TYPEG(opnd)) {
  case A_CNST:
    c1 = const_fold(OP_ADD, A_SPTRG(opnd), con_st, DT_INT8);
    new = mk_cnst(c1);
    return new;

  case A_BINOP:
    switch (A_OPTYPEG(opnd)) {
    case OP_ADD:
      lop = A_LOPG(opnd);
      rop = A_ROPG(opnd);
      new = reduce_i8add(rop, con_st);
      if (new) {
        if (A_TYPEG(new) == A_CNST) {
          c1 = A_SPTRG(new);
          if (c1 == stb.k0)
            return lop;
          inum1[0] = CONVAL1G(c1);
          inum1[1] = CONVAL2G(c1);
          inum2[0] = 0;
          inum2[1] = 0;
          if (!MIN_INT64(inum1) && cmp64(inum1, inum2) < 0) {
            new = negate_const(c1, DT_INT8);
            new = mk_cnst(new);
            new = hash_binop(A_BINOP, DT_INT8, lop, OP_SUB, new);
            A_CALLFGP(new, A_CALLFGG(lop));
            A_SHAPEP(new, 0);
            return new;
          }
        } else if (lop > new) {
          tmp = lop;
          lop = new;
          new = tmp;
        }
        new = hash_binop(A_BINOP, DT_INT8, lop, OP_ADD, new);
        A_CALLFGP(new, A_CALLFGG(lop) | A_CALLFGG(new));
        A_SHAPEP(new, 0);
        return new;
      }
      new = reduce_i8add(lop, con_st);
      if (new) {
        if (A_TYPEG(new) != A_CNST && (A_TYPEG(rop) == A_CNST || rop > new)) {
          tmp = rop;
          rop = new;
          new = tmp;
        }
        new = hash_binop(A_BINOP, DT_INT8, rop, OP_ADD, new);
        A_CALLFGP(new, A_CALLFGG(rop) | A_CALLFGG(new));
        A_SHAPEP(new, 0);
        return new;
      }
      break;
    case OP_SUB:
      lop = A_LOPG(opnd);
      rop = A_ROPG(opnd);
      new = reduce_i8add(lop, con_st);
      if (new) {
        if (A_TYPEG(new) == A_CNST && A_SPTRG(new) == stb.k0) {
          new = mk_unop(OP_SUB, rop, DT_INT8);
          return new;
        }
        new = hash_binop(A_BINOP, DT_INT8, new, OP_SUB, rop);
        A_CALLFGP(new, A_CALLFGG(new) | A_CALLFGG(rop));
        A_SHAPEP(new, 0);
        return new;
      }
      inum1[0] = CONVAL1G(con_st);
      inum1[1] = CONVAL2G(con_st);
      if (MIN_INT64(inum1))
        break;
      c1 = negate_const(con_st, DT_INT8);
      new = reduce_i8add(rop, c1);
      if (new) {
        if (A_TYPEG(new) == A_CNST) {
          c1 = A_SPTRG(new);
          if (c1 == stb.k0)
            return lop;
          inum1[0] = CONVAL1G(c1);
          inum1[1] = CONVAL2G(c1);
          inum2[0] = 0;
          inum2[1] = 0;
          if (!MIN_INT64(inum1) && cmp64(inum1, inum2) < 0) {
            c1 = negate_const(c1, DT_INT8);
            new = mk_cnst(c1);
            new = hash_binop(A_BINOP, DT_INT8, lop, OP_ADD, new);
            A_CALLFGP(new, A_CALLFGG(lop));
            A_SHAPEP(new, 0);
            return new;
          }
        }
        new = hash_binop(A_BINOP, DT_INT8, lop, OP_SUB, new);
        A_CALLFGP(new, A_CALLFGG(lop) | A_CALLFGG(new));
        A_SHAPEP(new, 0);
        return new;
      }
      break;
    }
    break;
  default:
    break;
  }

  return 0;
}

int
mk_unop(int optype, int lop, DTYPE dtype)
{
  int ast;
  INT conval;
  int shape;

#if DEBUG
  if (A_TYPEG(lop) == A_TRIPLE) {
    interr("mk_unop: trying to operate on a triplet", optype, 3);
  }
#endif
  switch (optype) {
  case OP_ADD:
  case OP_SUB:
  case OP_LNOT:
    if (DTY(dtype) == TY_INT8 || DTY(dtype) == TY_LOG8) {
      lop = convert_int(lop, dtype);
    }
    break;
  default:
    break;
  }

  shape = A_SHAPEG(lop);

  switch (optype) {
  case OP_ADD:
    return lop;

  case OP_SUB:
    if (A_TYPEG(lop) == A_CNST) {
      switch (DTY(dtype)) {
      case TY_BINT:
      case TY_SINT:
      case TY_INT:
      case TY_BLOG:
      case TY_SLOG:
      case TY_LOG:
        conval = CONVAL2G(A_SPTRG(lop));
        ast = mk_cval(-conval, DT_INT);
        break;

      case TY_REAL:
        conval = A_SPTRG(lop);
        if (NMPTRG(conval) != 0)
          goto noconstfold;
        conval = CONVAL2G(conval);
        conval = negate_const(conval, dtype);
        ast = mk_cval(conval, dtype);
        break;

      case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
#endif
      case TY_CMPLX:
      case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
#endif
      case TY_INT8:
      case TY_LOG8:
        conval = A_SPTRG(lop);
        if (NMPTRG(conval) != 0)
          goto noconstfold;
        conval = negate_const(conval, dtype);
        ast = mk_cnst((int)conval);
        break;

      default:
        interr("mk_unop-negate: bad dtype", dtype, 3);
        ast = astb.i0;
        break;
      }
      return ast;
    }
    break;

  case OP_LOC:
    shape = 0;
    break;

  default:
    break;
  }

noconstfold:
  ast = hash_unop(A_UNOP, dtype, lop, optype);
  A_CALLFGP(ast, A_CALLFGG(lop));
  A_SHAPEP(ast, shape);
  return ast;
}

int
mk_cmplxc(int lop, int rop, DTYPE dtype)
{
  int ast;

  ast = hash_cmplxc(A_CMPLXC, dtype, lop, rop);
  if (A_SHAPEG(ast) == 0 && DTY(dtype) == TY_ARRAY)
    A_SHAPEP(ast, mkshape(dtype));
  return ast;
}

int
mk_paren(int lop, DTYPE dtype)
{
  int ast;
  ast = hash_paren(A_PAREN, dtype, lop);
  A_CALLFGP(ast, A_CALLFGG(lop));
  A_SHAPEP(ast, A_SHAPEG(lop));

  return ast;
}

int
mk_convert(int lop, DTYPE dtype)
{
  int ast;

  if (A_TYPEG(lop) == A_CNST) {
    ast = convert_cnst(lop, dtype);
    if (ast != lop)
      return ast;
  }
  /* don't convert 'lop' */
  if (A_TYPEG(lop) == A_TRIPLE)
    return lop;
  ast = hash_conv(A_CONV, dtype, lop, 0);
  if (DTY(dtype) == TY_ARRAY && A_SHAPEG(ast) == 0) {
    if (A_SHAPEG(lop))
      A_SHAPEP(ast, A_SHAPEG(lop));
    else
      A_SHAPEP(ast, mkshape(dtype));
  }
  /* copy the ALIAS field for conversion between integer types */
  if (DT_ISINT(dtype) && DT_ISINT(A_DTYPEG(lop))) {
    A_ALIASP(ast, A_ALIASG(lop));
  }
  A_CALLFGP(ast, A_CALLFGG(lop));
  return ast;
}

/* Generate a convert of ast to dtype if it isn't the right type already. */
int
convert_int(int ast, DTYPE dtype)
{
  if (A_DTYPEG(ast) == dtype)
    return ast;
  return mk_convert(ast, dtype);
}

static int
convert_cnst(int cnst, int newtyp)
{
  INT oldval;
  int oldtyp;
  int to, from;
  int sptr;
  INT num[4], result;
  INT num1[8];
#ifdef TARGET_SUPPORTS_QUADFP
  INT num2[4];
#endif
  UINT unum[4];

  oldtyp = A_DTYPEG(cnst);
  if (newtyp == oldtyp)
    return cnst;
  to = DTY(newtyp);
  from = DTY(oldtyp);

  if (!TY_ISSCALAR(to) || !TY_ISSCALAR(from))
    return cnst;

  sptr = A_SPTRG(cnst);

  /* switch statement falls thru to call_mk_cval1 */
  switch (to) {
  default:
    /* TY_CHAR & TY_NCHAR: the lengths are not always precise */
    return cnst;
  case TY_WORD:
    result = CONVAL2G(sptr);
    break;
  case TY_DWORD:
    if (size_of(from) >= size_of(to)) {
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
    } else {
      num[1] = CONVAL2G(sptr);
      num[0] = (TY_ISINT(from) && num[1] < 0) ? -1 : 0;
    }
    result = getcon(num, newtyp);
    break;
  case TY_BLOG:
  case TY_BINT:
    switch (from) {
    case TY_WORD:
    case TY_DWORD:
      if (to == TY_BLOG)
        return cnst; /* don't convert typeless for now */
      FLANG_FALLTHROUGH;
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
    case TY_LOG8:
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_INT8:
      oldval = CONVAL2G(sptr);
      result = sign_extend(oldval, 8);
      break;
    default:
      goto other_int_cases;
    }
    break;
  case TY_SLOG:
  case TY_SINT:
    switch (from) {
    case TY_WORD:
    case TY_DWORD:
      if (to == TY_SLOG)
        return cnst; /* don't convert typeless for now */
      FLANG_FALLTHROUGH;
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_INT8:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
    case TY_LOG8:
      oldval = CONVAL2G(sptr);
      result = sign_extend(oldval, 16);
      break;
    default:
      goto other_int_cases;
    }
    break;
  case TY_LOG:
  case TY_INT:
    switch (from) {
    case TY_WORD:
    case TY_DWORD:
      if (to == TY_LOG)
        return cnst; /* don't convert typeless for now */
      FLANG_FALLTHROUGH;
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
      result = CONVAL2G(sptr);
      break;
    case TY_INT8:
    case TY_LOG8:
      result = sign_extend(CONVAL2G(sptr), 32);
      break;
    default:
      goto other_int_cases;
    }
    break;
  other_int_cases:
    switch (from) {
    case TY_CMPLX:
      oldval = CONVAL1G(sptr);
      xfix(oldval, &result);
      break;
    case TY_REAL:
      oldval = CONVAL2G(sptr);
      xfix(oldval, &result);
      break;
    case TY_DCMPLX:
      sptr = CONVAL1G(sptr);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      xdfix(num, &result);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      sptr = CONVAL1G(sptr);
      FLANG_FALLTHROUGH;
    case TY_QUAD:
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      num[2] = CONVAL3G(sptr);
      num[3] = CONVAL4G(sptr);
      xqfix(num, &result);
      break;
#endif
    default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
      return cnst;
    }
    break;

  case TY_LOG8:
  case TY_INT8:
    if (from == TY_DWORD || from == TY_INT8 || from == TY_LOG8) {
      if (to == TY_LOG8)
        return cnst; /* don't convert typeless for now */
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
    } else if (from == TY_WORD) {
      if (to == TY_LOG8)
        return cnst; /* don't convert typeless for now */
      num[0] = 0;
      unum[1] = CONVAL2G(sptr);
      num[1] = unum[1];
    } else if (TY_ISINT(from)) {
      oldval = CONVAL2G(sptr);
      if (oldval < 0) {
        num[0] = -1;
        num[1] = oldval;
      } else {
        num[0] = 0;
        num[1] = oldval;
      }
    } else {
      switch (from) {
      case TY_CMPLX:
        oldval = CONVAL1G(sptr);
        xfix64(oldval, num);
        break;
      case TY_REAL:
        oldval = CONVAL2G(sptr);
        xfix64(oldval, num);
        break;
      case TY_DCMPLX:
        sptr = CONVAL1G(sptr);
        FLANG_FALLTHROUGH;
      case TY_DBLE:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        xdfix64(num1, num);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        sptr = CONVAL1G(sptr);
        FLANG_FALLTHROUGH;
      case TY_QUAD:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        num1[2] = CONVAL3G(sptr);
        num1[3] = CONVAL4G(sptr);
        xqfix64(num1, num);
        break;
#endif
      default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
        return cnst;
      }
    }
    result = getcon(num, newtyp);
    break;

  case TY_REAL:
    if (from == TY_WORD || from == TY_DWORD)
      return cnst; /* don't convert typeless for now */
      /* result <- CONVAL2G(sptr) */
    else if (from == TY_INT8 || from == TY_LOG8) {
      num[0] = CONVAL1G(sptr);
      num[1] = CONVAL2G(sptr);
      xflt64(num, &result);
    } else if (TY_ISINT(from)) {
      oldval = CONVAL2G(sptr);
      xffloat(oldval, &result);
    } else {
      switch (from) {
      case TY_CMPLX:
        result = CONVAL1G(sptr);
        break;
      case TY_DCMPLX:
        sptr = CONVAL1G(sptr);
        FLANG_FALLTHROUGH;
      case TY_DBLE:
        num[0] = CONVAL1G(sptr);
        num[1] = CONVAL2G(sptr);
        xsngl(num, &result);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        sptr = CONVAL1G(sptr);
        FLANG_FALLTHROUGH;
      case TY_QUAD:
        num[0] = CONVAL1G(sptr);
        num[1] = CONVAL2G(sptr);
        num[2] = CONVAL3G(sptr);
        num[3] = CONVAL4G(sptr);
        xqtof(num, &result);
        break;
#endif
      default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
        return cnst;
      }
    }
    break;

  case TY_DBLE:
    if (from == TY_WORD) {
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <- 0
       * num[1] <- CONVAL2G(sptr)
       */
    } else if (from == TY_DWORD) {
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <-  CONVAL1G(sptr)
       * num[1] <-  CONVAL2G(sptr)
       */
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(sptr);
      num1[1] = CONVAL2G(sptr);
      xdflt64(num1, num);
    } else if (TY_ISINT(from))
      xdfloat(CONVAL2G(sptr), num);
    else {
      /* if a special 'named' constant, don't evaluate */
      if ((XBIT(49, 0x400000) || XBIT(51, 0x40)) && NMPTRG(sptr))
        return cnst;
      switch (from) {
      case TY_DCMPLX:
        result = CONVAL1G(sptr);
        goto call_mk_cval1;
      case TY_CMPLX:
        oldval = CONVAL1G(sptr);
        xdble(oldval, num);
        break;
      case TY_REAL:
        oldval = CONVAL2G(sptr);
        xdble(oldval, num);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        sptr = CONVAL1G(sptr);
        FLANG_FALLTHROUGH;
      case TY_QUAD:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        num1[2] = CONVAL3G(sptr);
        num1[3] = CONVAL4G(sptr);
        xqtod(num1, num);
        break;
#endif
      default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
        return cnst;
      }
    }
    result = getcon(num, DT_REAL8);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    if (from == TY_WORD) {
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <- 0
       * num[1] <- 0
       * num[2] <- 0
       * num[3] <- CONVAL2G(sptr)
       */
    } else if (from == TY_DWORD) {
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <- 0;
       * num[1] <- 0;
       * num[2] <-  CONVAL1G(sptr)
       * num[3] <-  CONVAL2G(sptr)
       */
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(sptr);
      num1[1] = CONVAL2G(sptr);
      xqflt64(num1, num);
    } else if (TY_ISINT(from))
      xqfloat(CONVAL2G(sptr), num);
    else {
      /* if a special 'named' constant, don't evaluate */
      if ((XBIT(49, 0x400000) || XBIT(51, 0x40)) && NMPTRG(sptr))
        return cnst;
      switch (from) {
      case TY_QCMPLX:
        sptr = CONVAL1G(sptr);
        num[0] = CONVAL1G(sptr);
        num[1] = CONVAL2G(sptr);
        num[2] = CONVAL3G(sptr);
        num[3] = CONVAL4G(sptr);
        break;
      case TY_DCMPLX:
        sptr = CONVAL1G(sptr);
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        xdtoq(num1, num);
        break;
      case TY_CMPLX:
        oldval = CONVAL1G(sptr);
        xftoq(oldval, num);
        break;
      case TY_REAL:
        oldval = CONVAL2G(sptr);
        xftoq(oldval, num);
        break;
      case TY_DBLE:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        xdtoq(num1, num);
        break;
      default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
        return cnst;
      }
    }
    result = getcon(num, DT_QUAD);
    break;
#endif
  case TY_CMPLX:
    /*  num[0] = real part
     *  num[1] = imaginary part
     */
    num[1] = 0;
    if (from == TY_WORD) {
      /* a la VMS */
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <- 0
       * num[1] <- CONVAL2G(sptr)
       */
    } else if (from == TY_DWORD) {
      /* a la VMS */
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <- CONVAL1G(sptr)
       * num[1] <- CONVAL2G(sptr)
       */
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(sptr);
      num1[1] = CONVAL2G(sptr);
      xflt64(num1, &num[0]);
    } else if (TY_ISINT(from))
      xffloat(CONVAL2G(sptr), &num[0]);
    else {
      switch (from) {
      case TY_REAL:
        num[0] = CONVAL2G(sptr);
        break;
      case TY_DBLE:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        xsngl(num1, &num[0]);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        num1[2] = CONVAL3G(sptr);
        num1[3] = CONVAL4G(sptr);
        xqtof(num1, &num[0]);
        break;
#endif
      case TY_DCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(sptr));
        num1[1] = CONVAL2G(CONVAL1G(sptr));
        xsngl(num1, &num[0]);
        num1[0] = CONVAL1G(CONVAL2G(sptr));
        num1[1] = CONVAL2G(CONVAL2G(sptr));
        xsngl(num1, &num[1]);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(sptr));
        num1[1] = CONVAL2G(CONVAL1G(sptr));
        num1[2] = CONVAL3G(CONVAL1G(sptr));
        num1[3] = CONVAL4G(CONVAL1G(sptr));
        xqtof(num1, &num[0]);
        num1[0] = CONVAL1G(CONVAL2G(sptr));
        num1[1] = CONVAL2G(CONVAL2G(sptr));
        num1[2] = CONVAL3G(CONVAL2G(sptr));
        num1[3] = CONVAL4G(CONVAL2G(sptr));
        xqtof(num1, &num[1]);
        break;
#endif
      default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
        return cnst;
      }
    }
    result = getcon(num, DT_CMPLX8);
    break;

  case TY_DCMPLX:
    if (from == TY_WORD) {
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <- 0
       * num[1] <- CONVAL2G(sptr)
       * num[0] <- getcon(num, DT_REAL8)
       * num[1] <- stb.dbl0
       */
    } else if (from == TY_DWORD) {
      return cnst; /* don't convert typeless for now */
      /*
       * num[0] <- CONVAL1G(sptr)
       * num[1] <- CONVAL2G(sptr)
       * num[0] <- getcon(num, DT_REAL8)
       * num[1] <- stb.dbl0;
       */
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(sptr);
      num1[1] = CONVAL2G(sptr);
      xdflt64(num1, num);
      num[0] = getcon(num, DT_REAL8);
      num[1] = stb.dbl0;
    } else if (TY_ISINT(from)) {
      xdfloat(CONVAL2G(sptr), num);
      num[0] = getcon(num, DT_REAL8);
      num[1] = stb.dbl0;
    } else {
      switch (from) {
      case TY_REAL:
        xdble(CONVAL2G(sptr), num);
        num[0] = getcon(num, DT_REAL8);
        num[1] = stb.dbl0;
        break;
      case TY_DBLE:
        num[0] = sptr;
        num[1] = stb.dbl0;
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        num1[2] = CONVAL3G(sptr);
        num1[3] = CONVAL4G(sptr);
        xqtod(num1, num);
        num[0] = getcon(num, DT_REAL8);
        num[1] = stb.dbl0;
        break;
#endif
      case TY_CMPLX:
        xdble(CONVAL1G(sptr), num1);
        num[0] = getcon(num1, DT_REAL8);
        xdble(CONVAL2G(sptr), num1);
        num[1] = getcon(num1, DT_REAL8);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(sptr));
        num1[1] = CONVAL2G(CONVAL1G(sptr));
        num1[2] = CONVAL3G(CONVAL1G(sptr));
        num1[3] = CONVAL4G(CONVAL1G(sptr));
        xqtod(num1, num2);
        num[0] = getcon(num2, DT_REAL8);
        num1[0] = CONVAL1G(CONVAL2G(sptr));
        num1[1] = CONVAL2G(CONVAL2G(sptr));
        num1[2] = CONVAL3G(CONVAL2G(sptr));
        num1[3] = CONVAL4G(CONVAL2G(sptr));
        xqtod(num1, num2);
        num[1] = getcon(num2, DT_REAL8);
        break;
#endif
      default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
        return cnst;
      }
    }
    result = getcon(num, DT_CMPLX16);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    if (from == TY_WORD) {
      return cnst; /* don't convert typeless for now */
    } else if (from == TY_DWORD) {
      return cnst; /* don't convert typeless for now */
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(sptr);
      num1[1] = CONVAL2G(sptr);
      xqflt64(num1, num);
      num[0] = getcon(num, DT_QUAD);
      num[1] = stb.quad0;
    } else if (TY_ISINT(from)) {
      xqfloat(CONVAL2G(sptr), num);
      num[0] = getcon(num, DT_QUAD);
      num[1] = stb.quad0;
    } else {
      switch (from) {
      case TY_REAL:
        xftoq(CONVAL2G(sptr), num);
        num[0] = getcon(num, DT_QUAD);
        num[1] = stb.quad0;
        break;
      case TY_DBLE:
        num1[0] = CONVAL1G(sptr);
        num1[1] = CONVAL2G(sptr);
        xdtoq(num1, num);
        num[0] = getcon(num, DT_QUAD);
        num[1] = stb.quad0;
        break;
      case TY_QUAD:
        num[0] = sptr;
        num[1] = stb.quad0;
        break;
      case TY_CMPLX:
        xftoq(CONVAL1G(sptr), num1);
        num[0] = getcon(num1, DT_QUAD);
        xftoq(CONVAL2G(sptr), num1);
        num[1] = getcon(num1, DT_QUAD);
        break;
      case TY_DCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(sptr));
        num1[1] = CONVAL2G(CONVAL1G(sptr));
        xdtoq(num1, num2);
        num[0] = getcon(num2, DT_QUAD);
        num1[0] = CONVAL1G(CONVAL2G(sptr));
        num1[1] = CONVAL2G(CONVAL2G(sptr));
        xdtoq(num1, num2);
        num[1] = getcon(num2, DT_QUAD);
        break;
      default: /* TY_HOLL, TY_CHAR, TY_NCHAR */
        return cnst;
      }
    }
    result = getcon(num, DT_QCMPLX);
    break;
#endif
  }

call_mk_cval1:
  cnst = mk_cval1(result, newtyp);
  return cnst;
}

int
mk_promote_scalar(int lop, DTYPE dtype, int shd)
{
  int ast = hash_conv(A_CONV, dtype, lop, shd);
  A_CALLFGP(ast, A_CALLFGG(lop));
  A_SHAPEP(ast, shd);
  return ast;
}

int
mk_subscr(int arr, int *subs, int numdim, DTYPE dtype)
{
  int asd = mk_asd(subs, numdim);
  return mk_subscr_copy(arr, asd, dtype);
}

int
mk_subscr_copy(int arr, int asd, DTYPE dtype)
{
  int i;
  int ast;
  int callfg;
  int shape;
  int numdim = ASD_NDIM(asd);

  assert(arr >= 0 && arr < astb.stg_avail, "mk_subscr_copy: invalid array ast", arr,
         ERR_Fatal);
  assert(asd >= 0 && asd < astb.asd.stg_avail, "mk_subscr_copy: invalid asd index",
         asd, ERR_Fatal);
  assert(dtype >= 0 && dtype < stb.dt.stg_avail,
         "mk_subscr_copy: invalid dtype index", dtype, ERR_Fatal);

  callfg = 0;
  for (i = 0; i < numdim; i++) {
    callfg |= A_CALLFGG(ASD_SUBS(asd, i));
  }

  shape = 0;
  if (A_TYPEG(arr) == A_MEM) {
    int shape_parent = A_SHAPEG(A_PARENTG(arr));
    int shape_mem = A_SHAPEG(A_MEMG(arr));
    if (shape_parent && shape_mem) {
      /* we are subscripting the member, need to use parent's shape */
      dtype = dtype_with_shape(dtype, shape_parent);
      shape = shape_parent;
    }
  }

  if (shape == 0) { /* not already chosen */
    /* see if there should be a shape */
    int shape_rank = 0;
    int arr_shape = A_SHAPEG(arr); /* shape of array */
    for (i = 0; i < numdim; ++i) {
      int sub = ASD_SUBS(asd, i);
      if (A_TYPEG(sub) == A_TRIPLE || A_SHAPEG(sub))
        ++shape_rank;
    }
    if (shape_rank > 0) {
      add_shape_rank(shape_rank);
      for (i = 0; i < numdim; ++i) {
        int sub = ASD_SUBS(asd, i);
        if (A_TYPEG(sub) == A_TRIPLE) {
          int lwb = A_LBDG(sub);
          int upb = A_UPBDG(sub);
          int stride = A_STRIDEG(sub);
          if (lwb == 0)
            lwb = astb.bnd.one;
          if (upb == 0 && arr_shape)
            upb = SHD_UPB(arr_shape, i);
          if (stride == 0)
            stride = astb.bnd.one;
          add_shape_spec(lwb, upb, stride);
        } else {
          int shp = A_SHAPEG(sub);
          if (shp != 0) {
            /* vector subscript */
            add_shape_spec(SHD_LWB(shp, 0), SHD_UPB(shp, 0),
                           SHD_STRIDE(shp, 0));
          }
        }
      }
      shape = mk_shape();
    }
  }
  if (shape == 0) {
    dtype = DDTG(dtype);
  }
  /* In the following case: a%b(i), where a and b are both arrays,
   * the input dtype is the type of b(i). It needs to be changed
   * to array of b(i). Also, the shape needs to be fixed.
   */
  ast = hash_subscr(A_SUBSCR, dtype, arr, asd);
  A_CALLFGP(ast, callfg | A_CALLFGG(arr));
  A_SHAPEP(ast, shape);
  if (DT_ISSCALAR(dtype)) {
    int al = complex_alias(ast);
    if (A_TYPEG(al) == A_INIT)
      A_ALIASP(ast, A_LEFTG(al));
  }
  return ast;
} /* mk_subscr_copy */

/* Find or create an ASD with these subscripts */
int
mk_asd(int *subs, int numdim)
{
  int i;
  int asd;
  assert(numdim > 0 && numdim <= MAXSUBS, "mk_subscr: bad numdim", numdim,
         ERR_Fatal);
  /* search the existing ASDs with the same number of dimensions */
  for (asd = astb.asd.hash[numdim - 1]; asd != 0; asd = ASD_NEXT(asd)) {
    for (i = 0; i < numdim; i++) {
      if (subs[i] != ASD_SUBS(asd, i))
        goto next_asd;
    }
    return asd;
  next_asd:;
  }

  /* allocate a new ASD; note that the type ASD allows for one subscript. */
  asd = astb.asd.stg_avail;
  astb.asd.stg_avail += sizeof(ASD) / sizeof(int) + numdim - 1;
  NEED(astb.asd.stg_avail, astb.asd.stg_base, int, astb.asd.stg_size, astb.asd.stg_avail + 240);
  ASD_NDIM(asd) = numdim;
  ASD_NEXT(asd) = astb.asd.hash[numdim - 1];
  astb.asd.hash[numdim - 1] = asd;
  for (i = 0; i < numdim; i++) {
    int sub = subs[i];
    assert(sub > 0, "mk_asd() bad subscript ast at dim", i + 1, ERR_Severe);
    ASD_SUBS(asd, i) = sub;
  }
  return asd;
}

/**
    \param lb ast of lower bound
    \param ub ast of upper bound
    \param stride ast of stride
    Any of these asts can be 0
 */
int
mk_triple(int lb, int ub, int stride)
{
  int ast;
  ast = hash_triple(A_TRIPLE, lb, ub, stride);
  A_CALLFGP(ast, (lb ? A_CALLFGG(lb) : 0) | (ub ? A_CALLFGG(ub) : 0) |
                     (stride ? A_CALLFGG(stride) : 0));
  return ast;
}

/**
    \param chr ast of character item being substring'd
    \param left position of leftmost character
    \param right position of rightmost character
    \param dtype dtype
 */
int
mk_substr(int chr, int left, int right, DTYPE dtype)
{
  int ast;

  ast = hash_substr(A_SUBSTR, dtype, chr, left, right);
  A_SHAPEP(ast, A_SHAPEG(chr));
  A_CALLFGP(ast, A_CALLFGG(chr) | (left ? A_CALLFGG(left) : 0) |
                     (right ? A_CALLFGG(right) : 0));
  return ast;
}

/** \brief For an AST tree with members and subscripts,
           if the base variable has the PARAMG bit set and all the subscripts
           are known constants, we can perhaps find the value the AST and set
           the A_ALIAS flag.
 */
int
complex_alias(int ast)
{
  int a, alias, sptr, asd, ndim, i, j, elem_offset, dtype;
  switch (A_TYPEG(ast)) {
  case A_SUBSCR:
    alias = complex_alias(A_LOPG(ast));
    if (alias == 0)
      return 0;
    dtype = A_DTYPEG(A_LOPG(ast));
    if (DTY(dtype) != TY_ARRAY)
      return 0;
    /* check the subscripts */
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    a = alias;
    alias = A_LEFTG(alias);
    if (alias == 0)
      return 0;
    if (A_TYPEG(alias) != A_INIT) {
      /*
       * presumably, this init is just a scalar promoted to an array.
       */
      return a;
    }
    elem_offset = 0;
    for (i = 0; i < ndim; ++i) {
      int ss, ssptr, ssval, lwbd, lwbdsptr, lwbdval, mplyr, mplyrsptr, mplyrval;
      ss = ASD_SUBS(asd, i);
      ss = A_ALIASG(ss);
      if (ss == 0)
        return 0;
      ssptr = A_SPTRG(ss);
      ssval = CONVAL2G(ssptr);
      /* lower bound of this dimension? */
      lwbd = ADD_LWAST(dtype, i);
      lwbd = A_ALIASG(lwbd);
      if (lwbd == 0)
        return 0;
      lwbdsptr = A_SPTRG(lwbd);
      lwbdval = CONVAL2G(lwbdsptr);
      mplyr = ADD_MLPYR(dtype, i);
      mplyr = A_ALIASG(mplyr);
      if (mplyr == 0)
        return 0;
      mplyrsptr = A_SPTRG(mplyr);
      mplyrval = CONVAL2G(mplyrsptr);

      elem_offset += (ssval - lwbdval) * mplyrval;
    }
    /* find this element of the named constant array */
    for (j = 0; j < elem_offset; ++j) {
      alias = A_RIGHTG(alias);
      if (alias == 0)
        return 0;
    }
    return alias;
    break;
  case A_MEM:
    alias = complex_alias(A_PARENTG(ast));
    if (alias == 0)
      return 0;
    /* find this member in the alias list */
    sptr = A_SPTRG(A_MEMG(ast));
    for (a = A_LEFTG(alias); a; a = A_RIGHTG(a)) {
      if (A_SPTRG(a) == sptr)
        return a;
    }
    return 0;
    break;
  case A_ID:
    /* is the symbol really a PARAMETER symbolic constant? */
    sptr = A_SPTRG(ast);
    if (!PARAMG(sptr))
      return 0;
    return PARAMVALG(sptr);
  default:
    return 0;
  }
} /* complex_alias */

int
mk_member(int parent, int mem, DTYPE dtype)
{
  int ast;
  int shape_parent, shape_mem;

  shape_parent = A_SHAPEG(parent);
  shape_mem = A_SHAPEG(mem);
  /* If both parent and member have a shape, there is really no
   * correct dtype for A_MEM. mk_subscr will have to check.
   */
  /* dtype is dtype of member */
  if (shape_mem) {
    int memsptr;
    /* if this member is a pointer, then we must modify the shape
     * descriptors to use the static descriptor which is in the
     * dtype */
    memsptr = A_SPTRG(mem);
    if ((POINTERG(memsptr) || ALLOCATTRG(memsptr)) && SDSCG(memsptr) &&
        STYPEG(SDSCG(memsptr)) == ST_MEMBER) {
      shape_mem = mk_mem_ptr_shape(parent, mem, A_DTYPEG(mem));
    }
    dtype = dtype_with_shape(dtype, shape_mem);
  } else if (shape_parent) {
    dtype = dtype_with_shape(DDTG(dtype), shape_parent);
  }
  ast = hash_mem(A_MEM, dtype, parent, mem);
  if (DTY(dtype) == TY_ARRAY) {
    if (shape_mem) {
      A_SHAPEP(ast, shape_mem);
    } else if (shape_parent) {
      A_SHAPEP(ast, shape_parent);
    } else {
      A_SHAPEP(ast, mkshape(dtype));
    }
  }
  A_CALLFGP(ast, A_CALLFGG(parent));
  if (DT_ISSCALAR(dtype)) {
    int al;
    al = complex_alias(ast);
    if (A_TYPEG(al) == A_INIT)
      A_ALIASP(ast, A_LEFTG(al));
  }
  return ast;
}

/*---------------------------------------------------------------------*/

/** \brief Make shape ilm(s) from an array descriptor.  Return the pointer to
   the
           the shape descriptor (SHD).
 */
int
mkshape(DTYPE dtype)
{
  int numdim, i;
  int lwb, upb, stride;

  if (DTY(dtype) != TY_ARRAY)
    return 0;
  numdim = ADD_NUMDIM(dtype);
  if (numdim > 7 || numdim < 1) {
    interr("mkshape: bad numdim", numdim, 3);
    numdim = 1;
    add_shape_rank(numdim);
    add_shape_spec(astb.bnd.one, astb.bnd.one, astb.bnd.one);
    return mk_shape();
  }

  add_shape_rank(numdim);
  for (i = 0; i < numdim; ++i) {
    lwb = lbound_of(dtype, i);
    upb = ADD_UPAST(dtype, i);
    stride = astb.bnd.one;
    add_shape_spec(lwb, upb, stride);
  }
  return mk_shape();
}

/** \brief Make shape ast(s) for an array reference off of a pointer in a
           derived type. Return the shape descriptor (SHD). Main difference
           is that the descriptor references need to be derived type
           components.
 */
int
mk_mem_ptr_shape(int parent, int mem, DTYPE dtype)
{
  int numdim, i;
  int lwb, upb, extnt, stride;
  int newlwb, newupb;
  int sdsc;
  int subs[1];
  int lwbds[MAXRANK];
  int upbds[MAXRANK];
  int asd;

  if (DTY(dtype) != TY_ARRAY)
    return 0;
  numdim = ADD_NUMDIM(dtype);
  if (numdim > 7 || numdim < 1) {
    interr("mkshape: bad numdim", numdim, 3);
    numdim = 1;
    add_shape_rank(numdim);
    add_shape_spec(astb.bnd.one, astb.bnd.one, astb.bnd.one);
    return mk_shape();
  }

  sdsc = SDSCG(A_SPTRG(mem));
  for (i = 0; i < numdim; ++i) {
    lwb = lbound_of(dtype, i);
    upb = ADD_UPAST(dtype, i);
    extnt = ADD_EXTNTAST(dtype, i);
    stride = astb.bnd.one;
    /* lwb, upb and extnt should look like x$sd(..) -- need to modify
     * them to be parent%x$sd(..)
     */
    assert(sdsc != 0, "mk_mem_ptr_shape: no static desc for pointer", mem, 4);
    assert(A_TYPEG(lwb) == A_SUBSCR, "mk_mem_ptr_shape: lwb not subs", lwb, 4);
    assert(memsym_of_ast(lwb) == sdsc, "mk_mem_ptr_shape: lwb not sdsc", lwb,
           4);
    assert(A_TYPEG(extnt) == A_SUBSCR, "mk_mem_ptr_shape: extnt not subs",
           extnt, 4);
    assert(memsym_of_ast(extnt) == sdsc, "mk_mem_ptr_shape: extnt not sdsc",
           extnt, 4);

    asd = A_ASDG(lwb);
    assert(ASD_NDIM(asd) == 1, "mk_mem_ptr_shape: lwb too many dims", lwb, 4);
    newlwb = mk_id(sdsc);
    newlwb = mk_member(parent, newlwb, A_DTYPEG(newlwb));
    subs[0] = ASD_SUBS(asd, 0);
    newlwb = mk_subscr(newlwb, subs, 1, astb.bnd.dtype);

    newupb = mk_id(sdsc);
    newupb = mk_member(parent, newupb, A_DTYPEG(newupb));
    asd = A_ASDG(extnt);
    assert(ASD_NDIM(asd) == 1, "mk_mem_ptr_shape: extnt too many dims", extnt,
           4);
    subs[0] = ASD_SUBS(asd, 0);
    newupb = mk_subscr(newupb, subs, 1, astb.bnd.dtype);
    newupb = mk_binop(OP_SUB, newupb, mk_isz_cval(1, A_DTYPEG(extnt)),
                      A_DTYPEG(extnt));
    newupb = mk_binop(OP_ADD, newlwb, newupb, A_DTYPEG(extnt));

    lwbds[i] = newlwb;
    upbds[i] = newupb;
  }
  stride = astb.bnd.one;
  add_shape_rank(numdim);
  for (i = 0; i < numdim; ++i)
    add_shape_spec(lwbds[i], upbds[i], stride);
  return mk_shape();
}

/*
 * define static structure used to represent the template for creating
 * a shape descriptor. A shape descriptor is called by the following calls:
 *
 * add_shape_rank(ndim)  -- begin by defining the shape's rank
 *
 * foreach dimension
 *     add_shape_spec(lwb, upb, stride) -- ASTs of lower and upper bounds and
 *                                         stride for dimension
 * mk_shape()            -- create shape descriptor in dynamic memory area
 *                          and return its pointer.
 *
 * reduc_shape()         -- create shape descriptor derived from an existing
 *                          shape descriptor excluding a given dimension.
 *
 */
static struct {
  short ndim; /* number of dimensions (rank) */
  short next; /* next dimension to be filled in */
  struct {
    int lwb;
    int upb;
    int stride;
  } spec[MAXRANK]; /* maximum number of dimensions */
} _shd;

int
mk_shape(void)
{
  int ndim;
  int shape;
  int i;

  ndim = _shd.ndim;
#if DEBUG
  assert(ndim && ndim == _shd.next, "mk_shape:inconsistent ndim,next",
         _shd.ndim, 4);
#endif

  /* search the existing SHDs with the same number of dimensions
   */
  for (shape = astb.shd.hash[ndim - 1]; shape; shape = SHD_NEXT(shape)) {
    for (i = 0; i < ndim; i++)
      if (SHD_LWB(shape, i) != _shd.spec[i].lwb ||
          SHD_UPB(shape, i) != _shd.spec[i].upb ||
          SHD_STRIDE(shape, i) != _shd.spec[i].stride)
        goto next_shape;
    goto found; /* return matching shape */
  next_shape:;
  }
  /*
   * allocate a new SHD; note that the type SHD allows for one
   * subscript.
   */
  shape = astb.shd.stg_avail;
  i = ndim + 1; /* WATCH declaration of SHD */
  astb.shd.stg_avail += i;
  NEED(astb.shd.stg_avail, astb.shd.stg_base, SHD, astb.shd.stg_size, astb.shd.stg_avail + 240);
  SHD_NDIM(shape) = ndim;
  SHD_NEXT(shape) = astb.shd.hash[ndim - 1];
  SHD_FILL(shape) = 0; /* avoid bogus UMR reports */
  astb.shd.hash[ndim - 1] = shape;
  for (i = 0; i < ndim; i++) {
    SHD_LWB(shape, i) = _shd.spec[i].lwb;
    SHD_UPB(shape, i) = _shd.spec[i].upb;
    SHD_STRIDE(shape, i) = _shd.spec[i].stride;
  }

found:
  return shape;
}

/** \brief Make an ast tree that computes the offset of the derived type or
           array element reference 'ast' from the start of the variable being
           referenced.
 */
int
mk_offset(int astx, int resdtype)
{
  int sptr, sptrdtype, offsetx, numdim, asd, i, sub, offx, ssoffx;
  switch (A_TYPEG(astx)) {
  case A_ID:
    return mk_isz_cval(0, resdtype);
  case A_SUBSTR:
    sptr = memsym_of_ast(astx);
    offsetx = mk_offset(A_PARENTG(astx), resdtype);
    offx = mk_binop(OP_SUB, A_LEFTG(astx), stb.i1, resdtype);
    offsetx = mk_binop(OP_ADD, offsetx, offx, resdtype);
    return offsetx;
  case A_SUBSCR:
    sptr = memsym_of_ast(astx);
    sub = A_ASDG(astx);
    sptrdtype = DTYPEG(sptr);
    asd = A_ASDG(astx);
    numdim = ADD_NUMDIM(sptrdtype);
    if (ASD_NDIM(asd) != numdim)
      interr("mk_offset: dimensions don't match", numdim, 3);
    offsetx = mk_offset(A_PARENTG(astx), resdtype);
    offx = 0;
    for (i = 0; i < numdim; ++i) {
      int ss = ASD_SUBS(sub, i);
      if (A_TYPEG(ss) == A_TRIPLE)
        ss = A_LBDG(ss);
      ssoffx = mk_binop(OP_SUB, ss, ADD_LWAST(sptrdtype, i), resdtype);
      ssoffx = mk_binop(OP_MUL, ssoffx, ADD_MLPYR(sptrdtype, i), resdtype);
      if (!offx) {
        offx = ssoffx;
      } else {
        offx = mk_binop(OP_ADD, offx, ssoffx, resdtype);
      }
    }
    offx = mk_binop(OP_MUL, offx, size_ast(sptr, DTY(sptrdtype + 1)), resdtype);
    offsetx = mk_binop(OP_ADD, offsetx, offx, resdtype);
    return offsetx;
  case A_MEM:
    sptr = A_SPTRG(A_MEMG(astx));
    offsetx = mk_offset(A_PARENTG(astx), resdtype);
    offsetx = mk_binop(OP_ADD, offsetx, mk_isz_cval(ADDRESSG(sptr), resdtype),
                       resdtype);
    return offsetx;
  default:
    interr("mk_offset: unexpected ast", astx, 3);
    return mk_isz_cval(0, resdtype);
  }
} /* mk_offset */

/** \brief Duplicate a shape descriptor excluding a given dimension.
    \param o_shape old shape
    \param astdim  ast of dimension to be excluded
    \param after   std after which code is produced to create the
                   bounds descriptor (if dim is not a constant)
 */
int
reduc_shape(int o_shape, int astdim, int after)
{
  int ndim;
  int o_ndim;
  int dim;
  int shape;
  int i;

  o_ndim = SHD_NDIM(o_shape);
  ndim = o_ndim - 1;

  if (A_ALIASG(astdim) == 0) {
    /* for non-constant dim, just create a dummy shape descriptor
     * of the correct rank for the intrinsic.  Each item in the descriptor
     * will reference a CCSYM symbol and will not appear in the output.
     */
    int sptr, a;

    if (ndim <= 0)
      return 0;

    sptr = getccsym('.', 0, ST_VAR);
    a = mk_id(sptr);
    DTYPEP(sptr, astb.bnd.dtype);
    add_shape_rank(ndim);
    for (i = 0; i < ndim; i++)
      add_shape_spec(a, a, a);
  } else {
    /* dim is a constant */

    dim = get_int_cval(A_SPTRG(A_ALIASG(astdim)));
    if (dim < 1 || dim > o_ndim) {
      error(423, 3, gbl.lineno, NULL, NULL);
      dim = 1;
    }
    if (ndim <= 0)
      return 0;

    add_shape_rank(ndim);
    for (i = 0; i < o_ndim; i++)
      if (i != dim - 1)
        add_shape_spec((int)SHD_LWB(o_shape, i), (int)SHD_UPB(o_shape, i),
                       (int)SHD_STRIDE(o_shape, i));
  }
  shape = mk_shape();
  return shape;
}

/** \brief Duplicate a shape descriptor increasing its rank at the given
   dimension.
    \param o_shape old shape
    \param astdim ast of dimension to add
    \param ub     ast of upper bound of dim at astdim
    \param after  std after which code is produced to create the
                  bounds descriptor (if dim is not a constant)
 */
int
increase_shape(int o_shape, int astdim, int ub, int after)
{
  int ndim;
  int o_ndim;
  int dim;
  int shape;
  int i;

  if (o_shape == 0) {
    /* scalar: create a rank 1 array */
    add_shape_rank(1);
    add_shape_spec(astb.bnd.one, ub, astb.bnd.one);
  } else {
    o_ndim = SHD_NDIM(o_shape);
    ndim = o_ndim + 1;

    if (A_ALIASG(astdim) == 0) {
      /* for non-constant dim, just create a dummy shape descriptor
       * of the correct rank for the intrinsic.  Each item in the
       * descriptor will reference a CCSYM symbol and will not appear in
       * the output.
       */
      int sptr, a;

      sptr = getccsym('.', 0, ST_VAR);
      a = mk_id(sptr);
      DTYPEP(sptr, astb.bnd.dtype);
      add_shape_rank(ndim);
      for (i = 0; i < ndim; i++)
        add_shape_spec(a, a, a);
    } else {
      /* dim is a constant */

      dim = get_int_cval(A_SPTRG(A_ALIASG(astdim)));
      if (dim < 1 || dim > ndim) {
        error(423, 3, gbl.lineno, NULL, NULL);
        dim = 1;
      }
      add_shape_rank(ndim);
      for (i = 0; i < o_ndim; i++) {
        if (i == dim - 1)
          add_shape_spec(astb.bnd.one, ub, astb.bnd.one);
        add_shape_spec((int)SHD_LWB(o_shape, i), (int)SHD_UPB(o_shape, i),
                       (int)SHD_STRIDE(o_shape, i));
      }
      if (o_ndim == dim - 1)
        add_shape_spec(astb.bnd.one, ub, astb.bnd.one);
    }
  }
  shape = mk_shape();
  return shape;
}

void
add_shape_rank(int ndim)
{
  _shd.ndim = ndim;
  _shd.next = 0;
}

void
add_shape_spec(int lwb, int upb, int stride)
{
  int i;

  i = _shd.next;
#if DEBUG
  assert(i < _shd.ndim, "add_shape_spec:exceed rank", i, 4);
#endif
  _shd.spec[i].lwb = lwb;
  _shd.spec[i].upb = upb;
  _shd.spec[i].stride = stride;
  _shd.next++;
}

/** \brief Check conformance of shape descriptors
    \return true if the data types for two shapes are conformable
            (have the same shape).  Shape is defined to be the rank and
            the extents of each dimension.
 */
LOGICAL
conform_shape(int shape1, int shape2)
{
  int ndim;
  int i;
  ISZ_T lb1, lb2; /* lower bounds if constants */
  ISZ_T ub1, ub2; /* upper bounds if constants */
  ISZ_T st1, st2; /* strides if constants */

  if (shape1 == shape2)
    return TRUE;
  ndim = SHD_NDIM(shape1);
  if (ndim != SHD_NDIM(shape2))
    return FALSE;

  for (i = 0; i < ndim; i++) {
    if ((lb1 = A_ALIASG(SHD_LWB(shape1, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    lb1 = get_isz_cval(A_SPTRG(lb1));

    if ((ub1 = A_ALIASG(SHD_UPB(shape1, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    ub1 = get_isz_cval(A_SPTRG(ub1));

    if ((st1 = A_ALIASG(SHD_STRIDE(shape1, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    st1 = get_isz_cval(A_SPTRG(st1));

    if ((lb2 = A_ALIASG(SHD_LWB(shape2, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    lb2 = get_isz_cval(A_SPTRG(lb2));

    if ((ub2 = A_ALIASG(SHD_UPB(shape2, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    ub2 = get_isz_cval(A_SPTRG(ub2));

    if ((st2 = A_ALIASG(SHD_STRIDE(shape2, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    st2 = get_isz_cval(A_SPTRG(st2));

    /* lower and upper bounds and stride are constants in this dimension*/

    if (!st1 || !st2 || (ub1 - lb1 + st1) / st1 != (ub2 - lb2 + st2) / st2)
      return FALSE;
  }

  return TRUE;
}

/** \brief Create an ast representing the extent of a dimension.
    \param shape  shape descriptor
    \param dim    which dimension (0 based)
 */
int
extent_of_shape(int shape, int dim)
{
  int a;
  int lb = SHD_LWB(shape, dim);
  int ub = SHD_UPB(shape, dim);
  int stride = SHD_STRIDE(shape, dim);

  a = mk_binop(OP_SUB, ub, lb, astb.bnd.dtype);
  a = mk_binop(OP_ADD, a, stride, astb.bnd.dtype);
  a = mk_binop(OP_DIV, a, stride, astb.bnd.dtype);

  if (A_ALIASG(a)) {
    int cv;
    cv = A_SPTRG(A_ALIASG(a)); /* constant ST entry */
    if (DTY(DT_INT) != TY_INT8 && !XBIT(68, 0x1)) {
      if (CONVAL2G(cv) < 0)
        /* zero-sized in the dimension */
        return astb.i0;
    } else {
      INT inum1[2], inum2[2];

      inum1[0] = CONVAL1G(cv);
      inum1[1] = CONVAL2G(cv);
      inum2[0] = 0;
      inum2[1] = 0;
      if (cmp64(inum1, inum2) < 0)
        /* zero-sized in the dimension */
        return astb.bnd.zero;
    }
  } else {
    /* 'a' is calculated as ((ub - lb + s) / s)
     * which works for negative strides as well.
     * Negative results are converted to zero.
     */
    int mask = mk_binop(OP_GE, a, astb.bnd.zero, astb.bnd.dtype);
    a = mk_merge(a, astb.bnd.zero, mask, astb.bnd.dtype);
  }

  return a;
}

/** \brief Get the lower bound of a shape descriptor.
    \param shape shape descriptor
    \param dim   which dimension (0 based)
    \return an ast if the lower bound is a constant; otherwise, return 0.
 */
int
lbound_of_shape(int shape, int dim)
{
  int lb = SHD_LWB(shape, dim);
  int ub = SHD_UPB(shape, dim);

  if (A_ALIASG(lb) && A_ALIASG(ub)) {
    if (get_isz_cval(A_SPTRG(A_ALIASG(lb))) >
        get_isz_cval(A_SPTRG(A_ALIASG(ub))))
      /* zero-sized in the dimension */
      return astb.bnd.zero;
    return lb;
  }
  return 0;
}

/** \brief Get the upper bound of a shape descriptor.
    \param shape shape descriptor
    \param dim   which dimension (0 based)
    \return an ast if the upper bound is a constant; otherwise, return 0.
 */
int
ubound_of_shape(int shape, int dim)
{
  int lb = SHD_LWB(shape, dim);
  int ub = SHD_UPB(shape, dim);

  if (A_ALIASG(lb) && A_ALIASG(ub)) {
    if (get_isz_cval(A_SPTRG(A_ALIASG(lb))) >
        get_isz_cval(A_SPTRG(A_ALIASG(ub))))
      /* zero-sized in the dimension */
      return astb.bnd.zero;
    return ub;
  }
  return 0;
}

int
rank_of_ast(int ast)
{
  int shape;

  shape = A_SHAPEG(ast);
  if (shape == 0)
    return 0;
  return SHD_NDIM(shape);
}

/** \brief Return the ast which computes the zero-base offset for an array.
 */
int
mk_zbase_expr(ADSC *ad)
{
  int i, numdim;
  int zbaseast = 0;

  numdim = AD_NUMDIM(ad);
  for (i = 0; i < numdim; i++) {
    if (i == 0) {
      zbaseast = AD_LWAST(ad, i);
    } else {
      int a;
      a = mk_binop(OP_MUL, AD_LWAST(ad, i), AD_MLPYR(ad, i), astb.bnd.dtype);
      zbaseast = mk_binop(OP_ADD, zbaseast, a, astb.bnd.dtype);
    }
  }
  return zbaseast;
}

/** \brief Return an ast that computes the multiplier from the multiplier,
           lower bound, and upper bound of the previous dimension.
 */
int
mk_mlpyr_expr(int lb, int ub, int mlpyr)
{
  int ast;

  if (lb == astb.bnd.one)
    ast = ub;
  else {
    ast = mk_binop(OP_SUB, ub, lb, astb.bnd.dtype);
    ast = mk_binop(OP_ADD, ast, astb.bnd.one, astb.bnd.dtype);
  }
  ast = mk_binop(OP_MUL, mlpyr, ast, astb.bnd.dtype);
  return ast;
}

/** \brief Return an ast that computes the extent (from the \a lb and \a ub).
 */
int
mk_extent_expr(int lb, int ub)
{
  INT extent_expr;

  if (A_ALIASG(lb) && ub && A_ALIASG(ub)) {
    extent_expr = mk_isz_cval(
        ad_val_of(A_SPTRG(ub)) - ad_val_of(A_SPTRG(lb)) + 1, astb.bnd.dtype);
  } else if (!ub) {
    extent_expr = mk_binop(OP_ADD, lb, astb.bnd.one, astb.bnd.dtype);
  } else if (lb == astb.bnd.one) {
    extent_expr = ub;
  } else {
    extent_expr = mk_binop(OP_ADD, mk_binop(OP_SUB, ub, lb, astb.bnd.dtype),
                           astb.bnd.one, astb.bnd.dtype);
  }

  return extent_expr;
}

/** \brief Return an ast to reference the extent.
 */
int
mk_extent(int lb, int ub, int dim)
{
  INT extent;

  if (lb && ub && A_ALIASG(lb) && A_ALIASG(ub)) {
    extent = mk_isz_cval(ad_val_of(A_SPTRG(ub)) - ad_val_of(A_SPTRG(lb)) + 1,
                         astb.bnd.dtype);
  } else if (lb && A_TYPEG(lb) == A_SUBSCR) {
    int sptr = memsym_of_ast(lb);
    if (STYPEG(sptr) == ST_DESCRIPTOR || STYPEG(sptr) == ST_ARRDSC ||
        (STYPEG(sptr) == ST_MEMBER && DESCARRAYG(sptr))) {
      extent = get_extent(sptr, dim);
    } else {
      /* extent <- call mk_extent_expr(lb, ub) */
      extent = mk_bnd_ast();
    }
  } else {
    if (lb == astb.bnd.one && ub) {
      extent = ub;
    } else {
      /* ub is probably an ID (for a temp var), allocate a temp for extent */
      extent = mk_bnd_ast();
    }
  }
  return extent;
}

int
mk_shared_extent(int lb, int ub, int dim)
{
  INT extent;

  if (lb && ub && A_ALIASG(lb) && A_ALIASG(ub)) {
    extent = mk_isz_cval(ad_val_of(A_SPTRG(ub)) - ad_val_of(A_SPTRG(lb)) + 1,
                         astb.bnd.dtype);
  } else if (lb && A_TYPEG(lb) == A_SUBSCR) {
    int sptr = memsym_of_ast(lb);
    if (STYPEG(sptr) == ST_DESCRIPTOR || STYPEG(sptr) == ST_ARRDSC ||
        (STYPEG(sptr) == ST_MEMBER && DESCARRAYG(sptr))) {
      extent = get_extent(sptr, dim);
    } else if (lb && ub) {
      extent = mk_extent_expr(lb, ub);
      extent = mk_shared_bnd_ast(extent);
    } else {
      extent = mk_bnd_ast();
    }
  } else {
    if (lb == astb.bnd.one && ub) {
      extent = ub;
    } else if (lb && ub) {
      /* ub is probably an ID (for a temp var), allocate a temp for extent */
      extent = mk_extent_expr(lb, ub);
      extent = mk_shared_bnd_ast(extent);
    } else {
      extent = mk_bnd_ast();
    }
  }
  return extent;
}

/* \brief returns TRUE if type of ast is a symbol or an object that can be
 * passed to sym_of_ast() or memsym_of_ast() functions.
 *
 * \param ast is the AST to test.
 *
 * \returns TRUE if ast is suitable for sym_of_ast(), etc. Otherwise FALSE.
 */
LOGICAL
ast_is_sym(int ast)
{
  return sym_of_ast2(ast) != 0;
}

/** \brief Like memsym_of_ast(), but for a member, returns the sptr of its
           parent, not the member.
 */
int
sym_of_ast(int ast)
{
  SPTR sptr = sym_of_ast2(ast);
  if (sptr == 0) {
    interr("sym_of_ast: unexpected ast", ast, 3);
    return stb.i0;
  }
  return sptr;
}

/* Like sym_of_ast() but return 0 if ast does not have a sym. */
static SPTR
sym_of_ast2(int ast)
{
  int alias = A_ALIASG(ast);
  if (alias)
    return A_SPTRG(alias);
  switch (A_TYPEG(ast)) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
    return A_SPTRG(ast);
  case A_SUBSCR:
  case A_SUBSTR:
  case A_CONV:
  case A_FUNC:
  case A_CALL:
    return sym_of_ast2(A_LOPG(ast));
  case A_MEM:
    return sym_of_ast2(A_PARENTG(ast));
  default:
    return 0;
  }
}

/** \brief Like sym_of_ast(), except for members it will return second to last
   parent
           member.

    For example, `pds%%data%%foo()` returns `data`, `pds%%data` returns
   `pds`.<br>
    This is used in computing the pass argument for a type-bound procedure
    expression.
 */
int
pass_sym_of_ast(int ast)
{
  int a;

  if ((a = A_ALIASG(ast)))
    return A_SPTRG(a);
  while (1) {
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
      return A_SPTRG(ast);
    case A_FUNC:
    case A_CALL:
    case A_SUBSCR:
    case A_SUBSTR:
      ast = A_LOPG(ast);
      if (A_TYPEG(ast) == A_MEM)
        ast = A_MEMG(ast);
      break;
    case A_MEM:
      ast = A_PARENTG(ast);
      if (A_TYPEG(ast) == A_MEM)
        return A_SPTRG(A_MEMG(ast));
      break;
    default:
      interr("pass_sym_of_ast: unexpected ast", ast, 3);
      return stb.i0;
    }
  }
}

/** \brief Like sym_of_ast(), but for a member, returns the sptr of the
    member itself, not its parent */
int
memsym_of_ast(int ast)
{
  int a;

  if ((a = A_ALIASG(ast)))
    return A_SPTRG(a);
  while (1) {
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
      return A_SPTRG(ast);
    case A_SUBSCR:
    case A_SUBSTR:
    case A_CONV:
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      ast = A_MEMG(ast);
      break;
    case A_FUNC:
    case A_CALL:
      ast = A_LOPG(ast);
      break;
    default:
      interr("memsym_of_ast:unexp.ast", ast, 3);
      return stb.i0;
    }
  }
}

/** \brief Replace sptr of ast, if member, replace the sptr of the member */
void
put_memsym_of_ast(int ast, int sptr)
{
  int a;

  if ((a = A_ALIASG(ast))) {
    A_SPTRP(a, sptr);
    return;
  }
  while (1) {
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
      A_SPTRP(ast, sptr);
      return;
    case A_SUBSCR:
    case A_SUBSTR:
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      ast = A_MEMG(ast);
      break;
    default:
      interr("put_memsym_of_ast:unexp.ast", ast, 3);
      return;
    }
  }
}

/** \brief Generate a replacement AST with a new sptr for certain AST types.
 *
 * This routine duplicates an AST and replaces its symbol table pointer with
 * the caller specified symbol table pointer. This routine is typically used
 * for replacing a generic type bound procedure with its resolved specific
 * type bound procedure. This routine currently works for A_ID and A_MEM AST 
 * types.
 *
 * \param ast is the original AST that we want to duplicate.
 * \param sptr is the new symbol table pointer for the new AST.
 *
 * \return the (new) replacement AST.
 */
int
replace_memsym_of_ast(int ast, SPTR sptr)
{
  switch (A_TYPEG(ast)) {
  case A_ID:
    return mk_id(sptr);
  case A_FUNC:
    return mk_func_node(A_FUNC, mk_id(sptr), A_ARGCNTG(ast), A_ARGSG(ast));
  case A_MEM:
    if (A_TYPEG(A_MEMG(ast)) == A_ID) {
      return mk_member(A_PARENTG(ast), mk_id(sptr), A_DTYPEG(ast)); 
    }
    FLANG_FALLTHROUGH;
  default:
    interr("replace_memsym_of_ast: unexpected ast", ast, 3);
  }
  return 0;
}

/** \brief Like memsym_of_ast(), but for looking for the sptr of a procedure
 * reference
 */
int
procsym_of_ast(int ast)
{
  while (1) {
    switch (A_TYPEG(ast)) {
    case A_ID:
      return A_SPTRG(ast);
    case A_SUBSCR:
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      ast = A_MEMG(ast);
      break;
    default:
      interr("procym_of_ast:unexp.ast", ast, 3);
      return stb.i0;
    }
  }
}

LOGICAL
pure_func_call(int func_ast)
{
  int entry;
  int iface;
  entry = procsym_of_ast(A_LOPG(func_ast));
  proc_arginfo(entry, NULL, NULL, &iface);
  if (iface && PUREG(iface))
    return TRUE;
  return FALSE;
}

LOGICAL
elemental_func_call(int func_ast)
{
  int entry;
  int iface;
  entry = procsym_of_ast(A_LOPG(func_ast));
  proc_arginfo(entry, NULL, NULL, &iface);
  if (iface && ELEMENTALG(iface))
    return TRUE;
  return FALSE;
}

/** \brief Return sptr of an A_SUBSCR */
int
sptr_of_subscript(int ast)
{
  int sptr;

  assert(A_TYPEG(ast) == A_SUBSCR, "sptr_of_subscript: not a subscript", ast,
         4);
  ast = A_LOPG(ast);
  sptr = 0;
  if (A_TYPEG(ast) == A_ID)
    sptr = A_SPTRG(ast);
  else if (A_TYPEG(ast) == A_MEM)
    sptr = A_SPTRG(A_MEMG(ast));
  else if (A_TYPEG(ast) == A_SUBSCR)
    sptr = sptr_of_subscript(ast);
  else if (A_TYPEG(ast) == A_CONV)
    sptr = memsym_of_ast(ast);
  else
    assert(0, "sptr_of_subscript: unknown type", ast, 4);
  return sptr;
} /* sptr_of_subscript */

/** \brief Return the leftmost array symbol.

    + for `a%%b(i)%%c%%d(j)%%e`, it will return `b`
    + for `a(i)%%d(j)`, it will return `a`
    + for `a(i)%%d`, it will return `a`
    + for `a%%b%%c%%d(i)`, it will return `d`
    + for scalar `a%%b`, it will return `a`
 */
int
left_array_symbol(int ast)
{
  int a, asym = 0;

  a = A_ALIASG(ast);
  if (a)
    return A_SPTRG(a);
  while (1) {
    int sptr;
    switch (A_TYPEG(ast)) {
    case A_ID:
      sptr = A_SPTRG(ast);
      if (DTY(DTYPEG(sptr)) == TY_ARRAY)
        return sptr;
      FLANG_FALLTHROUGH;
    case A_LABEL:
    case A_ENTRY:
      if (asym)
        return asym;
      return A_SPTRG(ast);
    case A_SUBSTR:
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      sptr = A_SPTRG(A_MEMG(ast));
      if (DTY(DTYPEG(sptr)) == TY_ARRAY)
        asym = sptr;
      ast = A_PARENTG(ast);
      break;
    case A_SUBSCR:
      ast = A_LOPG(ast);
      if (A_TYPEG(ast) == A_MEM) {
        asym = A_SPTRG(A_MEMG(ast));
        ast = A_PARENTG(ast);
      } else if (A_TYPEG(ast) == A_ID) {
        asym = A_SPTRG(ast);
        return asym;
      }
      break;
    default:
      interr("left_array_of_ast:unexpected ast type", ast, 3);
      break;
    }
  }
}

/** \brief Return the AST of the leftmost A_SUBSCR:

    + For `a%%b(i)%%c%%d(j)%%e`, it will return the AST of `a%%b(i)`
    + For `a(i)%%d(j)`, it will return the AST of `a(i)`
    + For `a(i)%%d`, it will return the AST of `a(i)`
    + For `a%%b%%c%%d(i)`, it will return `a%%b%%c%%d(i)`
    + For scalar `a%%b`, it will return `a`
 */
int
left_subscript_ast(int ast)
{
  int aleft = 0;
  while (1) {
    int sptr;
    switch (A_TYPEG(ast)) {
    case A_ID:
      sptr = A_SPTRG(ast);
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        interr("left_subscript_ast: found unsubscripted array ID", ast, 3);
      }
      FLANG_FALLTHROUGH;
    case A_LABEL:
    case A_ENTRY:
      if (aleft)
        return aleft;
      interr("left_subscript_ast: no subscripts", ast, 3);
      return ast;
    case A_SUBSTR:
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      sptr = A_SPTRG(A_MEMG(ast));
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        interr("left_subscript_ast: found unsubscripted array MEM", ast, 3);
      }
      ast = A_PARENTG(ast);
      break;
    case A_SUBSCR:
      aleft = ast;
      ast = A_LOPG(ast);
      /* skip over the 'parent' of a subscript, since its
       * symbol will be an array, and we want to save the A_SUBSCR,
         not the A_ID or A_MEM */
      if (A_TYPEG(ast) == A_MEM) {
        ast = A_PARENTG(ast);
      } else if (A_TYPEG(ast) == A_ID) {
        return aleft;
      }
      break;
    default:
      interr("left_subscript_ast:unexpected ast type", ast, 3);
      return aleft;
    }
  }
}

/** \brief This routine is similar to left_subscript_ast except it
           returns the leftmost non-scalar subscript.

    For `a(1)%%b(i)` return `b(i)`
 */
int
left_nonscalar_subscript_ast(int ast)
{
  int aleft = 0;
  int i, sub, ndim;
  while (1) {
    int sptr;
    switch (A_TYPEG(ast)) {
    case A_ID:
      sptr = A_SPTRG(ast);
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        interr("left_nonscalar_subscript_ast:"
               " found unsubscripted array ID",
               ast, 3);
      }
      FLANG_FALLTHROUGH;
    case A_LABEL:
    case A_ENTRY:
      if (aleft)
        return aleft;
      interr("left_nonscalar_subscript_ast: no subscripts", ast, 3);
      return ast;
    case A_SUBSTR:
      ast = A_LOPG(ast);
      break;
    case A_MEM:
      sptr = A_SPTRG(A_MEMG(ast));
      if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
        interr("left_nonscalr_subscript_ast:"
               " found unsubscripted array MEM",
               ast, 3);
      }
      ast = A_PARENTG(ast);
      break;
    case A_SUBSCR:
      /* check subscripts -- make sure they're not all constant */
      sub = A_ASDG(ast);
      ndim = ASD_NDIM(sub);
      for (i = 0; i < ndim; ++i) {
        if (A_TYPEG(ASD_SUBS(sub, i)) != A_CNST) {
          aleft = ast;
          break;
        }
      }
      ast = A_LOPG(ast);
      /* skip over the 'parent' of a subscript, since its
       * symbol will be an array, and we want to save the A_SUBSCR,
         not the A_ID or A_MEM */
      if (A_TYPEG(ast) == A_MEM) {
        ast = A_PARENTG(ast);
      } else if (A_TYPEG(ast) == A_ID) {
        return aleft;
      }
      break;
    default:
      interr("left_nonscalar_subscript_ast:unexpected ast type", ast, 3);
      return aleft;
    }
  }
}

/** \brief Return the AST of the leftmost A_SUBSCR or A_ID that is distributed
           or aligned.

    For `a%%b(i)%%c%%d(j)%%e`, it will return the AST of
    `%%e`, `d(j)`, `%%c`, `b(i)`, or `a`, depending on which is distributed.
 */
int
dist_ast(int ast)
{
  int nextast, sptr, aleft;
  for (; ast; ast = nextast) {
    nextast = sptr = 0;
    switch (A_TYPEG(ast)) {
    case A_ID:
      sptr = A_SPTRG(ast);
      break;
    case A_SUBSTR:
      nextast = A_LOPG(ast);
      break;
    case A_MEM:
      sptr = A_SPTRG(A_MEMG(ast));
      nextast = A_PARENTG(ast);
      break;
    case A_SUBSCR:
      aleft = A_LOPG(ast);
      /* skip over the 'parent' of a subscript, since its
       * symbol will be an array, and we want to save the A_SUBSCR,
         not the A_ID or A_MEM */
      if (A_TYPEG(aleft) == A_MEM) {
        sptr = A_SPTRG(A_MEMG(aleft));
        nextast = A_PARENTG(aleft);
      } else if (A_TYPEG(aleft) == A_ID) {
        sptr = A_SPTRG(aleft);
      } else {
        interr("dist_ast: found naked subscript", ast, 3);
        return 0;
      }
      break;
    default:
      interr("dist_ast:unexpected ast type", ast, 3);
    }
    if (sptr) {
      switch (STYPEG(sptr)) {
      case ST_VAR:
      case ST_ARRAY:
      case ST_MEMBER:
        if (DISTG(sptr) || ALIGNG(sptr))
          return ast;
        break;
      default:
        break;
      }
    }
  }
  return 0;
} /* dist_ast */

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

/* contiguous_array_section is a simple 3 state state machine (the 3rd state,
 *  FALSE, is implicit).
 *                      |inputs
 *state                |DIM_WHOLE| DIM_TRIPLE           | DIM_ELMNT
 *---------------------|--------------------------------------------------
 *START                | START   | TRIPLE_SNGL_ELEM_SEEN|
 *TRIPLE_SNGL_ELEM_SEEN| FALSE   | FALSE                | TRIPLE_SNGL_ELEM_SEEN
 */
static LOGICAL
contiguous_array_section(int subscr_ast)
{
  enum { START, TRIPLE_SNGL_ELEM_SEEN } state;
  enum {
    DIM_WHOLE,  /* ":"  */
    DIM_TRIPLE, /* "lb:ub:", no stride allowed */
    DIM_ELMNT,  /* "indx"   */
    DONT_CARE,
  } tkn;

  int asd;
  int ndims, dim;
  int ast;

  asd = A_ASDG(subscr_ast);
  ndims = ASD_NDIM(asd);

  state = START;
  for (dim = 0; dim < ndims; dim++) {
    ast = ASD_SUBS(asd, dim);
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_MEM:
    case A_SUBSCR:
    case A_FUNC:
      if (A_SHAPEG(ast))
        return FALSE;
      FLANG_FALLTHROUGH;
    case A_CNST:
    case A_BINOP:
    case A_UNOP:
      tkn = DIM_ELMNT;
      break;
    case A_TRIPLE:
      if (is_whole_dim(subscr_ast, dim))
        tkn = DIM_WHOLE;
      else if (stride1_triple(ast))
        tkn = DIM_TRIPLE;
      else
        return FALSE;
      break;
    case A_CONV:
      tkn = DONT_CARE;
      break;
    default:
      interr("contiguous_array_section: unexpected dimension type", 0, 3);
    }

    switch (state) {
    case START:
      if (tkn == DIM_TRIPLE || tkn == DIM_ELMNT)
        state = TRIPLE_SNGL_ELEM_SEEN;
      break;
    case TRIPLE_SNGL_ELEM_SEEN:
      if (tkn != DIM_ELMNT)
        return FALSE;
      break;
    }
  }
  return TRUE;
}

/** \brief Determine if array \a arr_ast covers all extent at dim i

    For example, on `a(1,:)` return true for second dim.
 */
LOGICAL
is_whole_dim(int arr_ast, int i)
{
  ADSC *ad;
  int asd;
  int sptr;
  int st, sub;
  int descr;
  int lb;
  int up;
  int ad_lwast;
  int ad_upast;

  assert(A_TYPEG(arr_ast) == A_SUBSCR, "is_whole_dim: must be SUBSCR", 2,
         arr_ast);
  asd = A_ASDG(arr_ast);
  sptr = memsym_of_ast(arr_ast);
  ad = AD_DPTR(DTYPEG(sptr));
  sub = ASD_SUBS(asd, i);
  if (A_TYPEG(sub) != A_TRIPLE)
    return FALSE;

  descr = SDSCG(sptr);
  lb = A_LBDG(sub);
  up = A_UPBDG(sub);
  ad_lwast = check_member(arr_ast, AD_LWAST(ad, i));
  ad_upast = check_member(arr_ast, AD_UPAST(ad, i));
  if (ASSUMSHPG(sptr) && ad_lwast != lb && lb != astb.i1 &&
      lb != astb.bnd.one) {
    return FALSE;
  } else if (STYPEG(sptr) == ST_MEMBER && (ad_lwast != lb || ad_upast != up)) {
    /* a member whole dim looks like
     *  lb = <descr>[i].lb
     *  up = <descr>[i].up - <descr>[i].lb + 1
     * look for these patterns (does the following look for enough
     * of these patterns?)
     */
    if (A_TYPEG(lb) != A_SUBSCR || memsym_of_ast(lb) != descr) {
      return FALSE;
    }
    if (A_TYPEG(up) == A_BINOP) {
      if (A_TYPEG(A_LOPG(up)) != A_SUBSCR ||
          memsym_of_ast(A_LOPG(up)) != descr) {
        return FALSE;
      }
      if (A_TYPEG(A_ROPG(up)) != A_BINOP ||
          A_TYPEG(A_LOPG(A_ROPG(up))) != A_SUBSCR ||
          memsym_of_ast(A_LOPG(A_ROPG(up))) != descr) {
        return FALSE;
      }
    } else {
      return FALSE;
    }
  } else if (ad_lwast != lb || ad_upast != up) {
    return FALSE;
  }

  st = A_STRIDEG(sub);
  if (st != 0 && st != astb.i1 && st != astb.bnd.one)
    return FALSE;
  return TRUE;
}

LOGICAL
is_whole_array(int arr_ast)
{
  int shape, lop, sptr, ndim, i, dtype;

  assert(A_TYPEG(arr_ast) == A_SUBSCR, "is_whole_array: must be SUBSCR",
         arr_ast, 2);
  if (A_TYPEG(arr_ast) == A_SUBSCR) {
    lop = A_LOPG(arr_ast);
  } else {
    lop = arr_ast;
  }
  switch (A_TYPEG(lop)) {
  case A_ID:
    sptr = A_SPTRG(lop);
    lop = 0;
    break;
  case A_MEM:
    sptr = A_SPTRG(A_MEMG(lop));
    lop = A_PARENTG(lop);
    break;
  default:
    interr("is_whole_array: subscript error", arr_ast, 4);
  }

  shape = A_SHAPEG(arr_ast);
  if (shape == 0)
    return FALSE;
  ndim = SHD_NDIM(shape);
  if (ndim != rank_of_sym(sptr))
    return FALSE;
  dtype = DTYPEG(sptr);
  for (i = 0; i < ndim; ++i) {
    int stride;
    stride = SHD_STRIDE(shape, i);
    if (stride != 0 && stride != astb.i1)
      return FALSE;
    /* some array expressions of the form a(:) will have ADD_LWBD==0
     * but ADD_LWAST wiil be  set */
    if (ADD_LWBD(dtype, i) != 0) {
      if (!bounds_match(ADD_LWBD(dtype, i), SHD_LWB(shape, i), lop))
        return FALSE;
    } else if (!bounds_match(ADD_LWAST(dtype, i), SHD_LWB(shape, i), lop)) {
      return FALSE;
    }
    if (!bounds_match(ADD_UPBD(dtype, i), SHD_UPB(shape, i), lop))
      return FALSE;
  }
  return TRUE;
} /* is_whole_array */

/* for normal array, lwdtype will be expression or section descriptor.
 * for derived type member, lwdtype will be LW$SD(29) or some such,
 * while lwshape will be X%LW$SD(29) or some such.  Make sure
 * the X% matches the parent, while the LW$SD(29) matches also
 */
static LOGICAL
bounds_match(int lwdtype, int lwshape, int parent)
{
  if (lwdtype == lwshape)
    return TRUE;
  if (A_TYPEG(lwdtype) == A_SUBSCR && A_TYPEG(lwshape) == A_SUBSCR) {
    /* see if these are section descriptor references */
    int adtype, ashape;
    adtype = A_LOPG(lwdtype);
    ashape = A_LOPG(lwshape);
    if (A_TYPEG(adtype) == A_ID && A_TYPEG(ashape) == A_MEM) {
      int asddtype, asdshape;
      if (A_PARENTG(ashape) != parent)
        return FALSE;
      if (A_SPTRG(adtype) != A_SPTRG(A_MEMG(ashape)))
        return FALSE;
      asddtype = A_ASDG(lwdtype);
      asdshape = A_ASDG(lwshape);
      if (ASD_NDIM(asddtype) != 1 || ASD_NDIM(asdshape) != 1)
        return FALSE;
      if (ASD_SUBS(asddtype, 0) != ASD_SUBS(asdshape, 0))
        return FALSE;
      /* yes, shape is X%P$SD(n) and dtype is P$SD(n) */
      return TRUE;
    }
  }
  return FALSE;
} /* bounds_match */

LOGICAL
simply_contiguous(int arr_ast)
{
  int sptr;

  switch (A_TYPEG(arr_ast)) {
  case A_ID:
    sptr = sym_of_ast(arr_ast);
    if (POINTERG(sptr) || ASSUMSHPG(sptr))
      return CONTIGATTRG(sptr);
    return TRUE;
  case A_FUNC:
    sptr = sym_of_ast(arr_ast);
    return !POINTERG(sptr);
  case A_SUBSTR:
    return FALSE;
  case A_MEM:
    sptr = sym_of_ast(arr_ast);
    if (!DT_ISCMPLX(STYPEG(sptr))) {
      sptr = memsym_of_ast(arr_ast);
      if (POINTERG(sptr) || ASSUMSHPG(sptr))
        return CONTIGATTRG(sptr);
    }
    break;
  case A_SUBSCR:
    sptr = memsym_of_ast(arr_ast);
    if (POINTERG(sptr)) {
      return CONTIGATTRG(sptr);
    }
    return contiguous_array_section(arr_ast);
  }

  return FALSE;
}

LOGICAL
bnds_remap_list(int subscr_ast)
{
  int asd;
  int ndims, dim;
  int ast;

  if (A_TYPEG(subscr_ast) != A_SUBSCR) {
    return FALSE;
  }

  asd = A_ASDG(subscr_ast);
  ndims = ASD_NDIM(asd);
  for (dim = 0; dim < ndims; dim++) {
    ast = ASD_SUBS(asd, dim);
    if (A_TYPEG(ast) == A_TRIPLE) {
      if (A_UPBDG(ast)) {
        return TRUE;
      }
    }
  }
  return FALSE;
}

/** \brief In \a original replace \a subtree with \a replacement.
    \param original    `a%%b(i)%%c%%d(j)%%e`
    \param subtree     `a%%b(i)`
    \param replacement `a%%b(1:2)`
    \return New ast: `a%%b(1:2)%%c%%d(j)%%e`
 */
int
replace_ast_subtree(int original, int subtree, int replacement)
{
  int p, ast, subs[MAXRANK], nsubs, i, asd, dtype;
  /* only A_ID, A_SUBSCR, A_SUBSTR, A_MEM allowed */
  if (subtree == replacement) /* in a%b(1)%j replace a%b(1) by a%b(1) */
    return original;
  if (subtree == original) /* in a%b(1) replace a%b(1) by a%b(i:j) */
    return replacement;
  switch (A_TYPEG(original)) {
  case A_SUBSTR:
    p = replace_ast_subtree(A_LOPG(original), subtree, replacement);
    ast =
        mk_substr(p, A_LEFTG(original), A_RIGHTG(original), A_DTYPEG(original));
    return ast;
  case A_SUBSCR:
    p = replace_ast_subtree(A_LOPG(original), subtree, replacement);
    asd = A_ASDG(original);
    nsubs = ASD_NDIM(asd);
    for (i = 0; i < nsubs; ++i)
      subs[i] = ASD_SUBS(asd, i);
    ast = mk_subscr(p, subs, nsubs, A_DTYPEG(original));
    return ast;
  case A_MEM:
    p = replace_ast_subtree(A_PARENTG(original), subtree, replacement);
    dtype = A_DTYPEG(original);
    if (A_SHAPEG(A_PARENTG(original)) && !A_SHAPEG(p))
      /*
       * the parent has shape, not the member, so the type of the new member
       * tree needs to be scalar.
       */
      dtype = DDTG(dtype);
    ast = mk_member(p, A_MEMG(original), dtype);
    return ast;
  case A_ID:
    /* should not get here, the replacement should have
     * replaced the original by now */
    interr("replace_ast_subtree: unexpected ID ast", original, 3);
    FLANG_FALLTHROUGH;
  default:
    interr("replace_ast_subtree: unexpected ast type", original, 3);
  }
  return replacement;
} /* replace_ast_subtree */

/** \brief Given an ast, return an ast with the element size */
int
elem_size_of_ast(int ast)
{
  DTYPE dtype;
  int bytes;
  int i;
  int is_arr = 0;

  dtype = A_DTYPEG(ast);

  if (DTY(dtype) == TY_ARRAY) {
    is_arr = 1;
    dtype = DTY(dtype + 1);
  }

  if (DTY(dtype) == TY_CHAR) {
    if (dtype != DT_ASSCHAR && dtype != DT_DEFERCHAR)
      bytes = mk_isz_cval(size_of(dtype), astb.bnd.dtype);
    else {
      if (!is_arr)
        i = sym_mkfunc_nodesc(mkRteRtnNm(RTE_lena), astb.bnd.dtype);
      else
        i = sym_mkfunc_nodesc_expst(mkRteRtnNm(RTE_lena), astb.bnd.dtype);
      bytes = begin_call(A_FUNC, i, 1);
      add_arg(ast);
    }
  }
  else if (DTY(dtype) == TY_NCHAR) {
    if (dtype != DT_ASSNCHAR && dtype != DT_DEFERNCHAR)
      bytes = mk_isz_cval(size_of(dtype), astb.bnd.dtype);
    else {
      if (!is_arr)
        i = sym_mkfunc_nodesc(mkRteRtnNm(RTE_nlena), astb.bnd.dtype);
      else
        i = sym_mkfunc_nodesc_expst(mkRteRtnNm(RTE_nlena), astb.bnd.dtype);
      bytes = begin_call(A_FUNC, i, 1);
      add_arg(ast);
    }
  }
  else {
    bytes = mk_isz_cval(size_of(dtype), astb.bnd.dtype);
  }

  return bytes;
}

int
size_of_ast(int ast)
{
  int shape;
  int ndim;
  int i;
  int sz;
  int tmp;

  shape = A_SHAPEG(ast);
  if (shape == 0)
    return astb.bnd.one;
  ndim = SHD_NDIM(shape);
  sz = astb.bnd.one;
  for (i = 0; i < ndim; i++) {
    int t;
    tmp = mk_binop(OP_SUB, check_member(ast, (int)SHD_UPB(shape, i)),
                   check_member(ast, (int)SHD_LWB(shape, i)), astb.bnd.dtype);
    t = check_member(ast, (int)SHD_STRIDE(shape, i));
    tmp = mk_binop(OP_ADD, tmp, t, astb.bnd.dtype);
    tmp = mk_binop(OP_DIV, tmp, t, astb.bnd.dtype);
    sz = mk_binop(OP_MUL, sz, tmp, astb.bnd.dtype);
  }
  return sz;
}

int
mk_bnd_ast(void)
{
  int bnd;

  if (XBIT(68, 0x1))
    bnd = getcctmp('b', atemps++, ST_VAR, DT_INT8);
  else
    bnd = getcctmp('b', atemps++, ST_VAR, DT_INT4);
  SCP(bnd, SC_LOCAL);
  CCSYMP(bnd, 1);
  return mk_id(bnd);
}

/** \brief Create a shared bounds temporary.
    \param ast the AST of the bounds expression which will be stored in the temp

    The same temp will be used for multiple uses of an expression.
 */
int
mk_shared_bnd_ast(int ast)
{
  int bnd;
  if (XBIT(68, 0x1))
    bnd = getcctmp('e', ast, ST_VAR, DT_INT8);
  else
    bnd = getcctmp('e', ast, ST_VAR, DT_INT4);
  SCP(bnd, SC_LOCAL);
  CCSYMP(bnd, 1);
  /*ADDRTKNP(bnd, 1); should be unnecessary since optutil.c considers
   * scalar temps (HCCSYM is set) as 'implicitly live'.
   */
  return mk_id(bnd);
}

int
mk_stmt(int stmt_type, DTYPE dtype)
{
  int ast;

  ast = new_node(stmt_type);
  if (dtype)
    A_DTYPEP(ast, dtype);
  return ast;
}

int
mk_std(int ast)
{
  int std;

  std = STG_NEXT(astb.std);
  if (std > MAXAST || astb.std.stg_base == NULL)
    errfatal(7);
  STD_AST(std) = ast; /* link std to ast */
  A_STDP(ast, std);   /* link ast to std */
  return std;
}

int
add_stmt(int ast)
{
  int std;

  std = mk_std(ast);

  insert_stmt_before(std, 0);
  if (gbl.in_include) {
    STD_LINENO(std) = gbl.lineno;
    STD_FINDEX(std) = gbl.findex;
    STD_ORIG(std) = 1;
  } else {
    STD_LINENO(std) = gbl.lineno;
    STD_FINDEX(std) = gbl.findex;
  }
  if (scn.currlab && !DEFDG(scn.currlab)) {
    STD_LABEL(std) = scn.currlab;
    DEFDP(scn.currlab, 1);
  } else
    STD_LABEL(std) = 0;

  return std;
}

static void
set_par(int std)
{
  int bef, aft;
  bef = STD_PREV(std);
  aft = STD_NEXT(std);
  if (bef && aft) {
    if (STD_PAR(bef) && STD_PAR(aft))
      STD_PAR(std) = 1;
    if (STD_TASK(bef) && STD_TASK(aft))
      STD_TASK(std) = 1;
  }
}

int
add_stmt_after(int ast, int stmt)
{
  int std;

  assert(ast, "add_stmt_after: sees ast of 0", ast, 2);

  std = mk_std(ast);
  insert_stmt_after(std, stmt);
  if (flg.smp) {
    set_par(std);
  }

  return std;
}

int
add_stmt_before(int ast, int stmt)
{
  int std;

  assert(ast, "add_stmt_before: sees ast of 0", ast, 2);

  std = mk_std(ast);

  insert_stmt_before(std, stmt);
  if (flg.smp) {
    set_par(std);
  }

  return std;
}

/* Insert std into STD list after stdafter; copy lineno and findex from stdafter
 * to std. */
void
insert_stmt_after(int std, int stdafter)
{
  STD_PREV(std) = stdafter;
  STD_NEXT(std) = STD_NEXT(stdafter);
  STD_PREV(STD_NEXT(stdafter)) = std;
  STD_NEXT(stdafter) = std;
  STD_LINENO(std) = STD_LINENO(stdafter);
  STD_FINDEX(std) = STD_FINDEX(stdafter);
}

/* Insert std into STD list before stdbefore; copy lineno and findex from
 * stdbefore
 * to std. */
void
insert_stmt_before(int std, int stdbefore)
{
  STD_NEXT(std) = stdbefore;
  STD_PREV(std) = STD_PREV(stdbefore);
  STD_NEXT(STD_PREV(stdbefore)) = std;
  STD_PREV(stdbefore) = std;
  STD_LINENO(std) = STD_LINENO(stdbefore);
  STD_FINDEX(std) = STD_FINDEX(stdbefore);
}

/* Remove std from the STD list. */
void
remove_stmt(int std)
{
  int prev = STD_PREV(std);
  int next = STD_NEXT(std);
#if DEBUG
  if (STD_NEXT(prev) != std || STD_PREV(next) != std) {
    interr("remove_stmt: corrupt STD or deleting statement twice", std,
           ERR_Severe);
    return;
  }
#endif
  STD_NEXT(prev) = next;
  STD_PREV(next) = prev;
  /* clear the pointers so we don't delete the statement twice */
  STD_NEXT(std) = 0;
  STD_PREV(std) = 0;
}

/* Move std(s) before stdbefore */
void
move_range_before(int sstd, int estd, int stdbefore)
{
  if (!(sstd && estd && stdbefore))
    return;

  STD_NEXT(STD_PREV(sstd)) = STD_NEXT(estd);
  STD_PREV(STD_NEXT(estd)) = STD_PREV(sstd);

  if (sstd == estd) {
    insert_stmt_before(sstd, stdbefore);
  } else {
    STD_NEXT(STD_PREV(stdbefore)) = sstd;
    STD_PREV(sstd) = STD_PREV(stdbefore);
    STD_PREV(stdbefore) = estd;
    STD_NEXT(estd) = stdbefore;
  }
}

/* Move std(s) after stdafter */
void
move_range_after(int sstd, int estd, int stdafter)
{
  if (!(sstd && estd && stdafter))
    return;

  STD_NEXT(STD_PREV(sstd)) = STD_NEXT(estd);
  STD_PREV(STD_NEXT(estd)) = STD_PREV(sstd);

  if (sstd == estd) {
    insert_stmt_after(sstd, stdafter);
  } else {
    STD_PREV(STD_NEXT(stdafter)) = estd;
    STD_NEXT(estd) = STD_NEXT(stdafter);
    STD_NEXT(stdafter) = sstd;
    STD_PREV(sstd) = stdafter;
  }
}

/* Move all STDs starting with std to before stdbefore */
void
move_stmts_before(int std, int stdbefore)
{
  int stdnext;
  for (; std != 0; std = stdnext) {
    stdnext = STD_NEXT(std);
    remove_stmt(std);
    insert_stmt_before(std, stdbefore);
    if (flg.smp) {
      set_par(std);
    }
  }
}

/* Move all STDs starting with std to after stdafter */
void
move_stmts_after(int std, int stdafter)
{
  int stdnext;
  for (; std != 0; std = stdnext) {
    stdnext = STD_NEXT(std);
    remove_stmt(std);
    insert_stmt_after(std, stdafter);
    if (flg.smp) {
      set_par(std);
    }
  }
}

void
ast_to_comment(int ast)
{
  int std = A_STDG(ast);
  int par = STD_PAR(std);
  int accel = STD_ACCEL(std);
  int newast = mk_stmt(A_COMMENT, 0);

  A_LOPP(newast, ast);
  STD_AST(std) = newast;
  A_STDP(newast, std);
  STD_FLAGS(std) = 0;
  STD_PAR(std) = par;
  STD_ACCEL(std) = accel;
}

int
mk_comstr(char *str)
{
  int newast;
  INT indx;

  newast = mk_stmt(A_COMSTR, 0);
  indx = astb.comstr.stg_avail;
  A_COMPTRP(newast, indx);
  astb.comstr.stg_avail += strlen(str) + 1;
  NEED(astb.comstr.stg_avail, astb.comstr.stg_base, char, astb.comstr.stg_size,
       astb.comstr.stg_avail + 200);
  strcpy(COMSTR(newast), str);
  astb.comstr.stg_base[indx] = '!';

  return newast;
}

/** \brief Create an ARGT
    \param cnt number of arguments in the ARGT
 */
int
mk_argt(int cnt)
{
  int argt;

  if (cnt == 0)
    return 0;
  argt = astb.argt.stg_avail;
  astb.argt.stg_avail += cnt + 1;
  NEED(astb.argt.stg_avail, astb.argt.stg_base, int, astb.argt.stg_size, astb.argt.stg_avail + 200);
  if (argt > MAX_NMPTR || astb.argt.stg_base == NULL)
    errfatal(7);
  ARGT_CNT(argt) = cnt;

  return argt;
}

/**
    \param cnt Number of arguments in the ARGT
 */
void
unmk_argt(int cnt)
{
  if (cnt == 0)
    return;
  astb.argt.stg_avail -= cnt + 1;
} /* unmk_argt */

/* AST List (ASTLI) Management */

static int tail_astli; /* tail of ast list */

/** \brief Initialize for a new ast list.

    The head of the list is stored in ast.astli.base[0].next
    and is accessed via the macro ASTLI_HEAD.

    Call add_astli() to add items to the end of the list.
 */
void
start_astli(void)
{
  tail_astli = 0; /* no elements in the list */
  ASTLI_HEAD = 0;
}

/** \brief Create and return an AST list item, adding it to the end of the
          current list.
 */
int
add_astli(void)
{
  int astli;

  astli = astb.astli.stg_avail++;
  NEED(astb.astli.stg_avail, astb.astli.stg_base, ASTLI, astb.astli.stg_size,
       astb.astli.stg_size + 200);
  if (astli > MAX_NMPTR || astb.astli.stg_base == NULL)
    errfatal(7);
  ASTLI_NEXT(tail_astli) = astli;
  ASTLI_NEXT(astli) = 0;
  tail_astli = astli;
  ASTLI_FLAGS(astli) = 0;

  return astli;
}

static void
reset_astli(void)
{
  if (ASTLI_HEAD) {
    astb.astli.stg_avail = ASTLI_HEAD;
    ASTLI_HEAD = 0;
  }
} /* reset_astli */

/**
    \param firstc first character in range
    \param lastc  last character in range
    \param dtype  implicit dtype pointer: 0 => NONE
 */
void
ast_implicit(int firstc, int lastc, DTYPE dtype)
{
  int i, j;

  if (dtype == 0)
    astb.implicit[54] = 1;
  else if (DTY(dtype) != TY_DERIVED) {
    i = IMPL_INDEX(firstc);
    j = IMPL_INDEX(lastc);
    for (; i <= j; i++)
      astb.implicit[i] = dtype;
  }
}

/*-----------------------------------------------------------------------*/

static struct {
  int argt;
  int ast;
  int arg_num;
  int ast_type;
  int arg_count;
} curr_call = {0, 0, 0, 0, 0};

/**
    \param ast_type A_FUNC, A_CALL, or A_INTR
    \param func     sptr of function to invoke
    \param count    number of arguments
 */
int
begin_call(int ast_type, int func, int count)
{
  int lop;
  /* make sure the previous call completed */
  if (curr_call.arg_num < curr_call.arg_count)
    interr("begin_call called before the previous procedure call completed",
           curr_call.arg_num, 3);
  curr_call.arg_count = count;
  curr_call.argt = mk_argt(count); /* mk_argt stuffs away count */
  curr_call.ast_type = ast_type;
  curr_call.ast = new_node(ast_type);
  lop = mk_id(func);
  A_LOPP(curr_call.ast, lop);
  A_ARGCNTP(curr_call.ast, count);
  A_ARGSP(curr_call.ast, curr_call.argt);
  if (ast_type == A_FUNC)
    A_CALLFGP(curr_call.ast, 1);

  curr_call.arg_num = 0;

  return curr_call.ast;
}

/** \brief Add an argument
    \param arg AST of argument to add.
 */
void
add_arg(int arg)
{
  if (curr_call.arg_num >= curr_call.arg_count)
    interr("add_arg called with too many arguments, or one begin_call mixed in "
           "with another",
           curr_call.arg_num, ERR_Severe);
  ARGT_ARG(curr_call.argt, curr_call.arg_num) = arg;
  curr_call.arg_num++;
  if (A_CALLFGG(arg))
    A_CALLFGP(curr_call.ast, 1);
}

/** \brief For an elemental intrinsic or function AST created by begin_call()
   and
    one or more calls to add_arg, fill in the result dtype and shape of the AST.
    \param dtype scalar dtype of the function/intrinsic
    \param promote if TRUE, promote the dtype to an array & create a shape
   descriptor
 */
void
finish_args(DTYPE dtype, LOGICAL promote)
{
  int shape;

  shape = 0;
  if (promote) {
    dtype = get_array_dtype(1, dtype);
    shape = A_SHAPEG(ARGT_ARG(curr_call.argt, 0));
  }
  A_DTYPEP(curr_call.ast, dtype);
  A_SHAPEP(curr_call.ast, shape);
}

int
mk_func_node(int ast_type, int func_ast, int paramct, int argt)
{
  int ast;

  ast = new_node(ast_type);
  A_LOPP(ast, func_ast);
  A_ARGCNTP(ast, paramct);
  A_ARGSP(ast, argt);
  if (ast_type == A_INTR || ast_type == A_ICALL) {
    int i;
    for (i = 0; i < paramct; i++)
      if (ARGT_ARG(argt, i) && A_CALLFGG(ARGT_ARG(argt, i))) {
        A_CALLFGP(ast, 1);
        break;
      }
  } else
    A_CALLFGP(ast, 1);

  return ast;
}

int
mk_assn_stmt(int dest, int source, DTYPE dtype)
{
  int ast;
  ast = mk_stmt(A_ASN, dtype);
  A_DESTP(ast, dest);
  A_SRCP(ast, source);
  return ast;
}

static int astMatch; /* AST # for matching */

/* This is the callback function for contains_ast(). */
static LOGICAL
_contains_ast(int astTarg, LOGICAL *pflag)
{
  if (astMatch == astTarg) {
    *pflag = TRUE;
    return TRUE;
  }
  return FALSE;
}

/** \brief Return TRUE if astSrc occurs somewhere within astTarg.

    WARNING: This routine may not produce correct results for non-leaf
    AST's -- correctness depends on hashing capabilities.
 */
LOGICAL
contains_ast(int astTarg, int astSrc)
{
  LOGICAL result = FALSE;

  if (!astTarg)
    return FALSE;

  astMatch = astSrc;
  ast_visit(1, 1);
  ast_traverse(astTarg, _contains_ast, NULL, &result);
  ast_unvisit();
  return result;
}

/* general ast rewrite functions:  uses a list to keep track of the ast nodes
 * which have been visited;  if a node is visited, the node's REPL field
 * is the ast which replaces the node.
 */

static int visit_list = 0;
static ast_visit_fn _visited;

int rewrite_opfields = 0;

#if DEBUG
static LOGICAL ast_visit_state = FALSE;
#endif
static LOGICAL ast_check_visited = TRUE;

/** \brief Add an AST to the visit list.

    An ast is added to the visit list during ast_rewrite() and ast_traverse().
 */
void
ast_visit(int old, int new)
{
#if DEBUG
  if (old == 0)
    interr("ast_visit sees ast of 0", 0, 2);
  if (old == 1 && new == 1) {
    if (ast_visit_state == TRUE && ast_check_visited) {
      interr("ast_visit without ast_unvisit", 0, 1);
    }
    ast_visit_state = TRUE;
  } else if (ast_visit_state == FALSE && ast_check_visited) {
    interr("ast_visit without ast_visit(1,1)", 0, 1);
  }
#endif
  if (A_VISITG(old) == 0) { /* allow multiple replacements */
    A_VISITP(old, visit_list);
    visit_list = old;
  }
}

/** \brief The \a old AST is to be replaced by the \a new AST.

    Set its REPL field and add to the visit list.  The caller of ast_rewrite()
    will have called ast_replace() one or more times to 'initialize' the
    rewriting process.
 */
void
ast_replace(int old, int new)
{
#if DEBUG
  if (old == 0)
    interr("ast_replace sees ast of 0", 0, 2);
  if (ast_visit_state == FALSE) {
    interr("ast_replace without ast_visit(1,1)", 0, 1);
  }
#endif
  A_REPLP(old, new);
  ast_visit(old, new);
}

/** \brief Traverse the visit list to clean up the nodes in the list.

    The caller must call ast_unvisit(). ast_unvisit() also clears the REPL
   field.
 */
void
ast_unvisit(void)
{
  int next;

#if DEBUG
  if (ast_visit_state == FALSE && ast_check_visited) {
    interr("ast_unvisit without ast_visit(1,1)", 0, 1);
  }
  ast_visit_state = FALSE;
#endif
  for (; visit_list; visit_list = next) {
    next = A_VISITG(visit_list);
    A_REPLP(visit_list, 0);
    A_VISITP(visit_list, 0);
  }
  _visited = NULL;
  rewrite_opfields = 0;
}

void
ast_unvisit_norepl(void)
{
  int next;

#if DEBUG
  if (ast_visit_state == FALSE) {
    interr("ast_unvisit_repl without ast_visit(1,1)", 0, 1);
  }
  ast_visit_state = FALSE;
#endif
  for (; visit_list; visit_list = next) {
    next = A_VISITG(visit_list);
    A_VISITP(visit_list, 0);
  }
  _visited = NULL;
  rewrite_opfields = 0;
}

/** \brief Visit the nodes on the 'visit_list' again, call \a proc on each one.
 */
void
ast_revisit(ast_visit_fn proc, int *extra_arg)
{
  if (visit_list) {
    int v;
    v = visit_list;
    (*proc)(v, extra_arg);
    for (v = A_VISITG(v); v && v != visit_list; v = A_VISITG(v))
      (*proc)(v, extra_arg);
  }
} /* ast_revisit */

int
ast_rewrite(int ast)
{
  int atype;
  int astnew;
  int parent, mem, left, right, lop, rop, l1, l2, l3, sub, lbd, upbd, stride,
      dest, src, ifexpr, ifstmt, dolab, dovar, m1, m2, m3, itriple, otriple,
      otriple1, dim, bvect, ddesc, sdesc, mdesc, vsub, chunk, npar, start,
      align, m4, stblk, lastvar, endlab, finalexpr, priorityexpr;
  DTYPE dtype;
  int devsrc;
  int asd;
  int numdim;
  int subs[MAXRANK];
  int argt;
  int argcnt;
  int argtnew;
  int anew;
  int i;
  LOGICAL changes;
  int astli, astlinew;
  int rank, rank1;
  int shape, procbind;

  if (ast == 0)
    return 0; /* watch for a 'null' argument */
  if (A_REPLG(ast))
    return A_REPLG(ast);
  shape = A_SHAPEG(ast);
  astnew = ast; /* default */
  changes = FALSE;
  switch (atype = A_TYPEG(ast)) {
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_LABEL:
    /* nothing changes */
    break;
  case A_MEM:
    parent = ast_rewrite((int)A_PARENTG(ast));
    mem = A_MEMG(ast);
    if (A_REPLG(mem)) {
      if (A_TYPEG(A_REPLG(mem)) == A_ID) {
        mem = A_REPLG(mem);
      }
    }
    if (parent != A_PARENTG(ast) || mem != A_MEMG(ast)) {
      astnew = mk_member(parent, mem, A_DTYPEG(ast));
    }
    break;
  case A_SUBSTR:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite((int)A_LOPG(ast));
    left = ast_rewrite((int)A_LEFTG(ast));
    right = ast_rewrite((int)A_RIGHTG(ast));
    if (left != A_LEFTG(ast) || right != A_RIGHTG(ast) || lop != A_LOPG(ast)) {
      astnew = mk_substr(lop, left, right, dtype);
    }
    break;
  case A_BINOP:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite((int)A_LOPG(ast));
    rop = ast_rewrite((int)A_ROPG(ast));
    if (lop != A_LOPG(ast) || rop != A_ROPG(ast)) {
      rank = (shape ? SHD_NDIM(shape) : 0);
      shape = A_SHAPEG(lop);
      rank1 = (shape ? SHD_NDIM(shape) : 0);
      if (rank != rank1) {
        if (rank == 0)
          rank = rank1;
        dtype = get_array_dtype(rank, DDTG(A_DTYPEG(lop)));
      }
      astnew = mk_binop((int)A_OPTYPEG(ast), lop, rop, dtype);
    }
    break;
  case A_UNOP:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite((int)A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      rank = (shape ? SHD_NDIM(shape) : 0);
      shape = A_SHAPEG(lop);
      rank1 = (shape ? SHD_NDIM(shape) : 0);
      if (rank != rank1) {
        if (rank == 0)
          rank = rank1;
        dtype = get_array_dtype(rank, DDTG(A_DTYPEG(lop)));
      }
      astnew = mk_unop((int)A_OPTYPEG(ast), lop, dtype);
    }
    break;
  case A_PAREN:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite((int)A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      rank = (shape ? SHD_NDIM(shape) : 0);
      shape = A_SHAPEG(lop);
      rank1 = (shape ? SHD_NDIM(shape) : 0);
      if (rank != rank1) {
        if (rank == 0)
          rank = rank1;
        dtype = get_array_dtype(rank, DDTG(A_DTYPEG(lop)));
      }
      astnew = mk_paren(lop, dtype);
    }
    break;
  case A_CONV:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite((int)A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      rank = (shape ? SHD_NDIM(shape) : 0);
      shape = A_SHAPEG(lop);
      rank1 = (shape ? SHD_NDIM(shape) : 0);
      if (rank != rank1) {
        if (rank == 0)
          rank = rank1;
        dtype = get_array_dtype(rank, DDTG(A_DTYPEG(ast)));
      }
      astnew = mk_convert(lop, dtype);
    }
    break;
  case A_SUBSCR:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite((int)A_LOPG(ast));
    if (lop != A_LOPG(ast))
      changes = TRUE;
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd);
    assert(numdim > 0 && numdim <= 7, "ast_rewrite: bad numdim", ast, 4);
    for (i = 0; i < numdim; ++i) {
      sub = ast_rewrite((int)ASD_SUBS(asd, i));
      if (sub != ASD_SUBS(asd, i))
        changes = TRUE;
      subs[i] = sub;
    }
    if (changes) {
      astnew = mk_subscr(lop, subs, numdim, dtype);
    }
    break;
  case A_INIT:
    dtype = A_DTYPEG(ast);
    left = ast_rewrite((int)A_LEFTG(ast));
    right = ast_rewrite((int)A_RIGHTG(ast));
    if (left != A_LEFTG(ast) || right != A_RIGHTG(ast)) {
      astnew = mk_init(left, dtype);
      A_RIGHTP(astnew, right);
      A_SPTRP(astnew, A_SPTRG(ast));
    }
    break;
  case A_TRIPLE:
    lbd = ast_rewrite((int)A_LBDG(ast));
    upbd = ast_rewrite((int)A_UPBDG(ast));
    stride = ast_rewrite((int)A_STRIDEG(ast));
    if (lbd != A_LBDG(ast) || upbd != A_UPBDG(ast) ||
        stride != A_STRIDEG(ast)) {
      astnew = mk_triple(lbd, upbd, stride);
    }
    break;
  case A_FUNC:
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast))
      changes = TRUE;
    argt = A_ARGSG(ast);
    argcnt = A_ARGCNTG(ast);
    argtnew = mk_argt(argcnt);
    for (i = 0; i < argcnt; i++) {
      anew = ast_rewrite(ARGT_ARG(argt, i));
      ARGT_ARG(argtnew, i) = anew;
      if (ARGT_ARG(argtnew, i) != ARGT_ARG(argt, i))
        changes = TRUE;
    }
    if (!changes) {
      unmk_argt(argcnt);
    } else {
      astnew = mk_func_node((int)A_TYPEG(ast), lop, argcnt, argtnew);
      A_SHAPEP(astnew, A_SHAPEG(ast));
      A_DTYPEP(astnew, A_DTYPEG(ast));
    }
    break;
  case A_INTR:
    lop = ast_rewrite((int)A_LOPG(ast));
    if (lop != A_LOPG(ast))
      changes = TRUE;
    argt = A_ARGSG(ast);
    argcnt = A_ARGCNTG(ast);
    argtnew = mk_argt(argcnt);
    for (i = 0; i < argcnt; i++) {
      anew = ast_rewrite(ARGT_ARG(argt, i));
      ARGT_ARG(argtnew, i) = anew;
      if (ARGT_ARG(argtnew, i) != ARGT_ARG(argt, i))
        changes = TRUE;
    }
    if (!changes) {
      unmk_argt(argcnt);
    } else {
      astnew = mk_func_node((int)A_TYPEG(ast), lop, argcnt, argtnew);
      A_OPTYPEP(astnew, A_OPTYPEG(ast));
      A_SHAPEP(astnew, A_SHAPEG(ast));
      A_DTYPEP(astnew, A_DTYPEG(ast));
    }
    switch (A_OPTYPEG(astnew)) {
    /* optimize a few intrinsics */
    case I_SIZE:
      /* is dim present and a constant ? */
      if (ARGT_ARG(argtnew, 1) && (i = A_ALIASG(ARGT_ARG(argtnew, 1)))) {
        int lwb, upb, stride;
        i = CONVAL2G(A_SPTRG(i)) - 1;
        shape = A_SHAPEG(ARGT_ARG(argtnew, 0));
        lwb = SHD_LWB(shape, i);
        upb = SHD_UPB(shape, i);
        stride = SHD_STRIDE(shape, i);
        if (stride == 0)
          stride = astb.bnd.one;
        if (lwb && A_ALIASG(lwb) && upb && A_ALIASG(upb) &&
            A_ALIASG(stride)) { /* stride is always nonzero here */
          astnew = upb;
          if (lwb != stride) {
            astnew = mk_binop(OP_SUB, astnew, lwb, astb.bnd.dtype);
            astnew = mk_binop(OP_ADD, astnew, stride, astb.bnd.dtype);
          }
          if (stride != astb.bnd.one) {
            astnew = mk_binop(OP_DIV, astnew, stride, astb.bnd.dtype);
          }
        }
      }
      break;
    default:
      break;
    }
    break;
  case A_ICALL:
  case A_CALL:
    lop = ast_rewrite((int)A_LOPG(ast));
    if (lop != A_LOPG(ast))
      changes = TRUE;
    argt = A_ARGSG(ast);
    argcnt = A_ARGCNTG(ast);
    argtnew = mk_argt(argcnt);
    for (i = 0; i < argcnt; i++) {
      anew = ast_rewrite(ARGT_ARG(argt, i));
      ARGT_ARG(argtnew, i) = anew;
      if (ARGT_ARG(argtnew, i) != ARGT_ARG(argt, i))
        changes = TRUE;
    }
    if (!changes) {
      unmk_argt(argcnt);
    } else {
      astnew = mk_func_node((int)A_TYPEG(ast), lop, argcnt, argtnew);
      A_OPTYPEP(astnew, A_OPTYPEG(ast));
      A_SHAPEP(astnew, A_SHAPEG(ast));
      if (atype == A_ICALL)
        A_DTYPEP(astnew, A_DTYPEG(ast));
      if (atype == A_CALL)
        A_INVOKING_DESCP(astnew, A_INVOKING_DESCG(ast));
    }
    break;
  case A_ASN:
    dtype = A_DTYPEG(ast);
    dest = ast_rewrite(A_DESTG(ast));
    src = ast_rewrite(A_SRCG(ast));
    if (dest != A_DESTG(ast) || src != A_SRCG(ast)) {
      shape = A_SHAPEG(A_DESTG(ast));
      rank = (shape ? SHD_NDIM(shape) : 0);
      shape = A_SHAPEG(dest);
      rank1 = (shape ? SHD_NDIM(shape) : 0);
      if (rank != rank1) {
        if (rank == 0)
          rank = rank1;
        dtype = get_array_dtype(rank, DDTG(A_DTYPEG(dest)));
      }
      astnew = mk_assn_stmt(dest, src, dtype);
    }
    break;
  case A_IF:
    ifexpr = ast_rewrite(A_IFEXPRG(ast));
    ifstmt = ast_rewrite(A_IFSTMTG(ast));
    if (ifexpr != A_IFEXPRG(ast) || ifstmt != A_IFSTMTG(ast)) {
      astnew = mk_stmt(A_IF, 0);
      A_IFEXPRP(astnew, ifexpr);
      A_IFSTMTP(astnew, ifstmt);
    }
    break;
  case A_IFTHEN:
  case A_ELSEIF:
    ifexpr = ast_rewrite(A_IFEXPRG(ast));
    if (ifexpr != A_IFEXPRG(ast)) {
      astnew = mk_stmt(A_TYPEG(ast), 0);
      A_IFEXPRP(astnew, ifexpr);
    }
    break;
  case A_AIF:
    ifexpr = ast_rewrite(A_IFEXPRG(ast));
    l1 = ast_rewrite(A_L1G(ast));
    l2 = ast_rewrite(A_L2G(ast));
    l3 = ast_rewrite(A_L3G(ast));
    if (ifexpr != A_IFEXPRG(ast) || l1 != A_L1G(ast) || l2 != A_L2G(ast) ||
        l3 != A_L3G(ast)) {
      astnew = mk_stmt(A_AIF, 0);
      A_IFEXPRP(astnew, ifexpr);
      A_L1P(astnew, l1);
      A_L2P(astnew, l2);
      A_L3P(astnew, l3);
    }
    break;
  case A_GOTO:
    l1 = ast_rewrite(A_L1G(ast));
    if (l1 != A_L1G(ast)) {
      astnew = mk_stmt(A_GOTO, 0);
      A_L1P(astnew, l1);
    }
    break;
  case A_CGOTO:
  case A_AGOTO:
    start_astli();
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast))
      changes = TRUE;
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      astlinew = add_astli();
      ASTLI_AST(astlinew) = ast_rewrite(ASTLI_AST(astli));
      if (ASTLI_AST(astlinew) != ASTLI_AST(astli))
        changes = TRUE;
    }
    if (!changes) {
      reset_astli();
    } else {
      astnew = mk_stmt(A_TYPEG(ast), 0);
      A_LISTP(astnew, ASTLI_HEAD);
      A_LOPP(astnew, lop);
    }
    break;
  case A_ASNGOTO:
#if DEBUG
    assert(A_TYPEG(A_SRCG(ast)) == A_LABEL,
           "_ast_trav, src A_ASNGOTO not label", A_SRCG(ast), 3);
#endif
    if (FMTPTG(A_SPTRG(A_SRCG(ast)))) {
      src = A_SRCG(ast);
      dest = ast_rewrite(A_DESTG(ast));
    } else {
      src = ast_rewrite(A_SRCG(ast));
      dest = ast_rewrite(A_DESTG(ast));
    }
    if (src != A_SRCG(ast) || dest != A_DESTG(ast)) {
      astnew = mk_stmt(A_ASNGOTO, 0);
      A_SRCP(astnew, src);
      A_DESTP(astnew, dest);
    }
    break;
  case A_DO:
    dolab = ast_rewrite(A_DOLABG(ast));
    dovar = ast_rewrite(A_DOVARG(ast));
    m1 = ast_rewrite(A_M1G(ast));
    m2 = ast_rewrite(A_M2G(ast));
    m3 = ast_rewrite(A_M3G(ast));
    m4 = ast_rewrite(A_M4G(ast));
    if (dolab != A_DOLABG(ast) || dovar != A_DOVARG(ast) || m1 != A_M1G(ast) ||
        m2 != A_M2G(ast) || m3 != A_M3G(ast) || m4 != A_M4G(ast)) {
      astnew = mk_stmt(A_DO, 0);
      A_DOLABP(astnew, dolab);
      A_DOVARP(astnew, dovar);
      A_M1P(astnew, m1);
      A_M2P(astnew, m2);
      A_M3P(astnew, m3);
      A_M4P(astnew, m4);
    }
    break;
  case A_DOWHILE:
    dolab = ast_rewrite(A_DOLABG(ast));
    ifexpr = ast_rewrite(A_IFEXPRG(ast));
    if (dolab != A_DOLABG(ast) || ifexpr != A_IFEXPRG(ast)) {
      astnew = mk_stmt(A_DOWHILE, 0);
      A_DOLABP(astnew, dolab);
      A_IFEXPRP(astnew, ifexpr);
    }
    break;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      astnew = mk_stmt(A_TYPEG(ast), 0);
      A_LOPP(astnew, lop);
    }
    break;
  case A_ALLOC:
    lop = ast_rewrite(A_LOPG(ast));
    src = ast_rewrite(A_SRCG(ast));
    dest = ast_rewrite(A_DESTG(ast));
    m3 = ast_rewrite(A_M3G(ast));
    start = ast_rewrite(A_STARTG(ast));
    dtype = A_DTYPEG(ast);
    devsrc = ast_rewrite(A_DEVSRCG(ast));
    align = ast_rewrite(A_ALIGNG(ast));
    if (lop != A_LOPG(ast) || src != A_SRCG(ast) || dest != A_DESTG(ast) ||
        m3 != A_M3G(ast) || start != A_STARTG(ast) ||
        devsrc != A_DEVSRCG(ast) || align != A_ALIGNG(ast)) {
      astnew = mk_stmt(A_ALLOC, 0);
      A_TKNP(astnew, A_TKNG(ast));
      A_DALLOCMEMP(astnew, A_DALLOCMEMG(ast));
      A_FIRSTALLOCP(astnew, A_FIRSTALLOCG(ast));
      A_LOPP(astnew, lop);
      A_SRCP(astnew, src);
      A_DESTP(astnew, dest);
      A_M3P(astnew, m3);
      A_STARTP(astnew, start);
      A_DTYPEP(astnew, dtype);
      A_DEVSRCP(astnew, devsrc);
      A_ALIGNP(astnew, align);
    }
    break;
  case A_WHERE:
    ifexpr = ast_rewrite(A_IFEXPRG(ast));
    ifstmt = ast_rewrite(A_IFSTMTG(ast));
    if (ifexpr != A_IFEXPRG(ast) || ifstmt != A_IFSTMTG(ast)) {
      astnew = mk_stmt(A_WHERE, 0);
      A_IFEXPRP(astnew, ifexpr);
      A_IFSTMTP(astnew, ifstmt);
    }
    break;
  case A_FORALL:
    ifexpr = ast_rewrite(A_IFEXPRG(ast));
    ifstmt = ast_rewrite(A_IFSTMTG(ast));
    if (ifexpr != A_IFEXPRG(ast) || ifstmt != A_IFSTMTG(ast))
      changes = TRUE;
    start_astli();
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      int s;
      astlinew = add_astli();
      ASTLI_TRIPLE(astlinew) = ast_rewrite(ASTLI_TRIPLE(astli));
      s = ast_rewrite(mk_id((int)ASTLI_SPTR(astli)));
      ASTLI_SPTR(astlinew) = A_SPTRG(s);
      if (ASTLI_TRIPLE(astlinew) != ASTLI_TRIPLE(astli) ||
          ASTLI_SPTR(astlinew) != ASTLI_SPTR(astli))
        changes = TRUE;
    }
    if (!changes) {
      reset_astli();
    } else {
      astnew = mk_stmt(A_FORALL, 0);
      A_LISTP(astnew, ASTLI_HEAD);
      A_IFEXPRP(astnew, ifexpr);
      A_IFSTMTP(astnew, ifstmt);
    }
    break;
  case A_REDIM:
    src = ast_rewrite(A_SRCG(ast));
    if (src != A_SRCG(ast)) {
      astnew = mk_stmt(A_REDIM, 0);
      A_SRCP(astnew, src);
    }
    break;
  case A_ENTRY:
  case A_COMMENT:
  case A_COMSTR:
  case A_ELSE:
  case A_ENDIF:
  case A_ELSEFORALL:
  case A_ELSEWHERE:
  case A_ENDWHERE:
  case A_ENDFORALL:
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
    break;
  case A_REALIGN:
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      astnew = mk_stmt(A_REALIGN, (int)A_DTYPEG(ast));
      A_LOPP(astnew, lop);
    }
    break;
  case A_REDISTRIBUTE:
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      astnew = mk_stmt(A_REDISTRIBUTE, (int)A_DTYPEG(ast));
      A_LOPP(astnew, lop);
    }
    break;
  case A_HLOCALIZEBNDS:
    lop = ast_rewrite(A_LOPG(ast));
    itriple = ast_rewrite(A_ITRIPLEG(ast));
    otriple = ast_rewrite(A_OTRIPLEG(ast));
    dim = ast_rewrite(A_DIMG(ast));
    if (lop != A_LOPG(ast) || itriple != A_ITRIPLEG(ast) ||
        otriple != A_OTRIPLEG(ast) || dim != A_DIMG(ast)) {
      astnew = mk_stmt(A_HLOCALIZEBNDS, 0);
      A_LOPP(astnew, lop);
      A_ITRIPLEP(astnew, itriple);
      A_OTRIPLEP(astnew, otriple);
      A_DIMP(astnew, dim);
    }
    break;
  case A_HALLOBNDS:
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      astnew = mk_stmt(A_HALLOBNDS, 0);
      A_LOPP(astnew, lop);
    }
    break;
  case A_HCYCLICLP:
    lop = ast_rewrite(A_LOPG(ast));
    itriple = ast_rewrite(A_ITRIPLEG(ast));
    otriple = ast_rewrite(A_OTRIPLEG(ast));
    otriple1 = ast_rewrite(A_OTRIPLE1G(ast));
    dim = ast_rewrite(A_DIMG(ast));
    if (lop != A_LOPG(ast) || itriple != A_ITRIPLEG(ast) ||
        otriple != A_OTRIPLEG(ast) || otriple1 != A_OTRIPLE1G(ast) ||
        dim != A_DIMG(ast)) {
      astnew = mk_stmt(A_HCYCLICLP, 0);
      A_LOPP(astnew, lop);
      A_ITRIPLEP(astnew, itriple);
      A_OTRIPLEP(astnew, otriple);
      A_OTRIPLE1P(astnew, otriple1);
      A_DIMP(astnew, dim);
    }
    break;
  case A_HOFFSET:
    dest = ast_rewrite(A_DESTG(ast));
    lop = ast_rewrite(A_LOPG(ast));
    rop = ast_rewrite(A_ROPG(ast));
    if (dest != A_DESTG(ast) || lop != A_LOPG(ast) || rop != A_ROPG(ast)) {
      astnew = mk_stmt(A_HOFFSET, 0);
      A_DESTP(astnew, dest);
      A_LOPP(astnew, lop);
      A_ROPP(astnew, rop);
    }
    break;
  case A_HSECT:
    lop = ast_rewrite(A_LOPG(ast));
    bvect = ast_rewrite(A_BVECTG(ast));
    if (lop != A_LOPG(ast) || bvect != A_BVECTG(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, DT_INT);
      A_LOPP(astnew, lop);
      A_BVECTP(astnew, bvect);
    }
    break;
  case A_HCOPYSECT:
    dest = ast_rewrite(A_DESTG(ast));
    src = ast_rewrite(A_SRCG(ast));
    ddesc = ast_rewrite(A_DDESCG(ast));
    sdesc = ast_rewrite(A_SDESCG(ast));
    if (dest != A_DESTG(ast) || src != A_SRCG(ast) || ddesc != A_DDESCG(ast) ||
        sdesc != A_SDESCG(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, DT_INT);
      A_DESTP(astnew, dest);
      A_SRCP(astnew, src);
      A_DDESCP(astnew, ddesc);
      A_SDESCP(astnew, sdesc);
    }
    break;
  case A_HPERMUTESECT:
    dest = ast_rewrite(A_DESTG(ast));
    src = ast_rewrite(A_SRCG(ast));
    ddesc = ast_rewrite(A_DDESCG(ast));
    sdesc = ast_rewrite(A_SDESCG(ast));
    bvect = ast_rewrite(A_BVECTG(ast));
    if (dest != A_DESTG(ast) || src != A_SRCG(ast) || ddesc != A_DDESCG(ast) ||
        sdesc != A_SDESCG(ast) || bvect != A_BVECTG(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, DT_INT);
      A_DESTP(astnew, dest);
      A_SRCP(astnew, src);
      A_DDESCP(astnew, ddesc);
      A_SDESCP(astnew, sdesc);
      A_BVECTP(astnew, bvect);
    }
    break;
  case A_HOVLPSHIFT:
    src = ast_rewrite(A_SRCG(ast));
    sdesc = ast_rewrite(A_SDESCG(ast));
    if (src != A_SRCG(ast) || sdesc != A_SDESCG(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, DT_INT);
      A_SRCP(astnew, src);
      A_SDESCP(astnew, sdesc);
    }
    break;
  case A_HGETSCLR:
    dest = ast_rewrite(A_DESTG(ast));
    src = ast_rewrite(A_SRCG(ast));
    lop = ast_rewrite(A_LOPG(ast));
    if (dest != A_DESTG(ast) || src != A_SRCG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_DESTP(astnew, dest);
      A_SRCP(astnew, src);
      A_LOPP(astnew, lop);
    }
    break;
  case A_HGATHER:
  case A_HSCATTER:
    vsub = ast_rewrite(A_VSUBG(ast));
    dest = ast_rewrite(A_DESTG(ast));
    src = ast_rewrite(A_SRCG(ast));
    ddesc = ast_rewrite(A_DDESCG(ast));
    sdesc = ast_rewrite(A_SDESCG(ast));
    mdesc = ast_rewrite(A_MDESCG(ast));
    bvect = ast_rewrite(A_BVECTG(ast));

    if (vsub != A_VSUBG(ast) || dest != A_DESTG(ast) || src != A_SRCG(ast) ||
        ddesc != A_DDESCG(ast) || sdesc != A_SDESCG(ast) ||
        mdesc != A_MDESCG(ast) || bvect != A_BVECTG(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, DT_INT);
      A_VSUBP(astnew, vsub);
      A_DESTP(astnew, dest);
      A_SRCP(astnew, src);
      A_DDESCP(astnew, ddesc);
      A_SDESCP(astnew, sdesc);
      A_MDESCP(astnew, mdesc);
      A_BVECTP(astnew, bvect);
    }
    break;
  case A_HCSTART:
    lop = ast_rewrite(A_LOPG(ast));
    dest = ast_rewrite(A_DESTG(ast));
    src = ast_rewrite(A_SRCG(ast));
    if (lop != A_LOPG(ast) || dest != A_DESTG(ast) || src != A_SRCG(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, DT_INT);
      A_LOPP(astnew, lop);
      A_DESTP(astnew, dest);
      A_SRCP(astnew, src);
    }
    break;
  case A_HCFINISH:
  case A_HCFREE:
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_LOPP(astnew, lop);
    }
    break;
  case A_HOWNERPROC:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite(A_LOPG(ast));
    dim = ast_rewrite(A_DIMG(ast));
    m1 = ast_rewrite(A_M1G(ast));
    m2 = ast_rewrite(A_M2G(ast));
    if (lop != A_LOPG(ast) || dim != A_DIMG(ast) || m1 != A_M1G(ast) ||
        m2 != A_M2G(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, dtype);
      A_LOPP(astnew, lop);
      A_DIMP(astnew, dim);
      A_M1P(astnew, m1);
      A_M2P(astnew, m2);
    }
    break;
  case A_HLOCALOFFSET:
    dtype = A_DTYPEG(ast);
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      astnew = new_node(atype);
      A_DTYPEP(astnew, dtype);
      A_LOPP(astnew, lop);
    }
    break;
  case A_CRITICAL:
  case A_ENDCRITICAL:
    break;
  case A_MASTER:
    break;
  case A_ENDMASTER:
    lop = A_LOPG(ast); /* its master */
    argcnt = A_ARGCNTG(ast);
    if (argcnt) {
      /* copy present */
      argt = A_ARGSG(ast);
      argtnew = mk_argt(argcnt);
      for (i = 0; i < argcnt; i++) {
        anew = ast_rewrite(ARGT_ARG(argt, i));
        ARGT_ARG(argtnew, i) = anew;
        if (ARGT_ARG(argtnew, i) != ARGT_ARG(argt, i))
          changes = TRUE;
      }
      if (!changes) {
        unmk_argt(argcnt);
      } else {
        astnew = mk_stmt(atype, 0);
        A_ARGSP(astnew, argtnew);
        A_ARGCNTP(astnew, argcnt);
        A_LOPP(astnew, lop);
        A_LOPP(lop, astnew); /* update reverse link */
      }
    }
    break;
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
  case A_ENDATOMIC:
  case A_BARRIER:
  case A_NOBARRIER:
    break;
  case A_MP_PARALLEL:
    ifexpr = ast_rewrite(A_IFPARG(ast));
    npar = ast_rewrite(A_NPARG(ast));
    endlab = ast_rewrite(A_ENDLABG(ast));
    procbind = ast_rewrite(A_PROCBINDG(ast));
    if (ifexpr != A_IFPARG(ast) || npar != A_NPARG(ast) ||
        endlab != A_ENDLABG(ast)) {
      astnew = mk_stmt(A_MP_PARALLEL, 0);
      A_IFPARP(astnew, ifexpr);
      A_NPARP(astnew, npar);
      A_LOPP(astnew,
             A_LOPG(ast)); /* A_MP_PARALLEL points to A_MP_ENDPARALLEL */
      A_LOPP(A_LOPG(ast), astnew);         /* and back */
      A_ENDLABP(A_ENDLABG(ast), astnew);   /* and back */
      A_PROCBINDP(A_ENDLABG(ast), astnew); /* and back */
    }
    break;
  case A_MP_TEAMS:
    ifexpr = ast_rewrite(A_NTEAMSG(ast));
    npar = ast_rewrite(A_THRLIMITG(ast));
    if (ifexpr != A_NTEAMSG(ast) || npar != A_THRLIMITG(ast)) {
      astnew = mk_stmt(A_MP_TEAMS, 0);
      A_NTEAMSP(astnew, ifexpr);
      A_THRLIMITP(astnew, npar);
      A_LOPP(astnew, A_LOPG(ast)); /* A_MP_TEAMS points to A_MP_ENDTEAMS */
      A_LOPP(A_LOPG(ast), astnew); /* and back */
    }
    break;
  case A_MP_TASK:
    ifexpr = ast_rewrite(A_IFPARG(ast));
    endlab = ast_rewrite(A_ENDLABG(ast));
    priorityexpr = ast_rewrite(A_PRIORITYG(ast));
    finalexpr = ast_rewrite(A_FINALPARG(ast));
    if (ifexpr != A_IFPARG(ast) || endlab != A_ENDLABG(ast) ||
        finalexpr != A_FINALPARG(ast) || priorityexpr != A_PRIORITYG(ast)) {
      astnew = mk_stmt(A_MP_TASK, 0);
      A_IFPARP(astnew, ifexpr);
      A_FINALPARP(astnew, finalexpr);
      A_ENDLABP(astnew, endlab);
      A_LOPP(astnew, A_LOPG(ast)); /* A_MP_TASK points to A_MP_ENDTASK */
      A_LOPP(A_LOPG(ast), astnew); /* and back */
    }
    break;
  case A_MP_TASKLOOP:
    ifexpr = ast_rewrite(A_IFPARG(ast));
    finalexpr = ast_rewrite(A_FINALPARG(ast));
    priorityexpr = ast_rewrite(A_PRIORITYG(ast));
    if (ifexpr != A_IFPARG(ast) || finalexpr != A_FINALPARG(ast) ||
        priorityexpr != A_PRIORITYG(ast)) {
      astnew = mk_stmt(A_MP_TASKLOOP, 0);
      A_IFPARP(astnew, ifexpr);
      A_FINALPARP(astnew, finalexpr);
      A_PRIORITYP(astnew, priorityexpr);
      A_LOPP(astnew, A_LOPG(ast)); /* A_MP_TASKLOOP points to A_MP_ETASKLOOP */
      A_LOPP(A_LOPG(ast), astnew); /* and back */
    }
    break;
  case A_MP_TARGET:
  case A_MP_TARGETDATA:
    ifexpr = ast_rewrite(A_IFPARG(ast));
    if (ifexpr != A_IFPARG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_IFPARP(astnew, ifexpr);
      A_LOPP(astnew,
             A_LOPG(ast)); /* A_MP_TARGETxx points to A_MP_ENDTARGETxx */
      A_LOPP(A_LOPG(ast), astnew); /* and back */
    }
    break;
  case A_MP_TARGETUPDATE:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
    ifexpr = ast_rewrite(A_IFPARG(ast));
    if (ifexpr != A_IFPARG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_IFPARP(astnew, ifexpr);
    }
    break;

  case A_MP_ENDTARGET:
  case A_MP_ENDTARGETDATA:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_TASKGROUP:
  case A_MP_ETASKGROUP:
  case A_MP_ETASKDUP:
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
  case A_MP_BCOPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_EMPSCOPE:
  case A_MP_FLUSH:
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_ETASKLOOPREG:
  case A_MP_ATOMICREAD:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
  case A_MP_MAP:
  case A_MP_EMAP:
  case A_MP_TARGETLOOPTRIPCOUNT:
  case A_MP_EREDUCTION:
  case A_MP_BREDUCTION:
  case A_MP_REDUCTIONITEM:
    break;
  case A_MP_ATOMICWRITE:
    rop = ast_rewrite(A_ROPG(ast));
    if (rop != A_ROPG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_LOPP(astnew, A_LOPG(ast));
      A_ROPP(astnew, rop);
      A_MEM_ORDERP(astnew, A_MEM_ORDERG(ast));
    }
    break;
  case A_MP_CANCELLATIONPOINT:
    rop = ast_rewrite(A_ENDLABG(ast));
    if (rop != A_ENDLABG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_ENDLABP(astnew, rop);
      A_CANCELKINDP(astnew, A_CANCELKINDG(ast));
    }
    break;
  case A_MP_CANCEL:
    rop = ast_rewrite(A_ENDLABG(ast));
    lop = ast_rewrite(A_IFPARG(ast));
    if (rop != A_ENDLABG(ast) || rop != A_IFPARG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_ENDLABP(astnew, rop);
      A_CANCELKINDP(astnew, A_CANCELKINDG(ast));
    }
    break;
  case A_MP_TASKFIRSTPRIV:
    rop = ast_rewrite(A_ROPG(ast));
    lop = ast_rewrite(A_LOPG(ast));
    if (rop != A_ROPG(ast) || lop != A_LOPG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_SPTRP(astnew, A_SPTRG(ast));
      A_ROPP(astnew, rop);
      A_LOPP(astnew, lop);
    }
    break;

  case A_MP_BMPSCOPE:
    stblk = ast_rewrite(A_STBLKG(ast));
    if (stblk != A_STBLKG(ast)) {
      astnew = mk_stmt(A_MP_BMPSCOPE, 0);
      A_STBLKP(astnew, stblk);
    }
    break;
  case A_MP_PRE_TLS_COPY:
  case A_MP_COPYIN:
  case A_MP_COPYPRIVATE:
    rop = ast_rewrite(A_ROPG(ast));
    if (rop != A_ROPG(ast)) {
      astnew = mk_stmt(atype, 0);
      A_SPTRP(astnew, A_SPTRG(ast));
      A_ROPP(astnew, rop);
    }
    break;
  case A_MP_TASKLOOPREG:
    m1 = ast_rewrite(A_M1G(ast));
    m2 = ast_rewrite(A_M2G(ast));
    m3 = ast_rewrite(A_M3G(ast));
    if (m1 != A_M1G(ast) || m2 != A_M2G(ast) || m3 != A_M3G(ast)) {
      astnew = mk_stmt(A_MP_TASKLOOPREG, 0);
      A_M1P(astnew, m1);
      A_M2P(astnew, m2);
      A_M3P(astnew, m3);
    }
    break;
  case A_MP_PDO:
    dolab = ast_rewrite(A_DOLABG(ast));
    dovar = ast_rewrite(A_DOVARG(ast));
    lastvar = ast_rewrite(A_LASTVALG(ast));

    /* don't rewrite bounds if this is distribute parallel do
     * unless we combine the distribute and parallel do in
     * a single loop.
     */
    if (A_DISTPARDOG(ast)) {
      m1 = A_M1G(ast);
      m2 = A_M2G(ast);
      m3 = A_M3G(ast);
    } else {
      m1 = ast_rewrite(A_M1G(ast));
      m2 = ast_rewrite(A_M2G(ast));
      m3 = ast_rewrite(A_M3G(ast));
    }
    chunk = ast_rewrite(A_CHUNKG(ast));
    if (dolab != A_DOLABG(ast) || dovar != A_DOVARG(ast) || m1 != A_M1G(ast) ||
        lastvar != A_LASTVALG(ast) || m2 != A_M2G(ast) || m3 != A_M3G(ast) ||
        chunk != A_CHUNKG(ast)) {
      astnew = mk_stmt(A_MP_PDO, 0);
      A_DOLABP(astnew, dolab);
      A_DOVARP(astnew, dovar);
      A_LASTVALP(astnew, lastvar);
      A_M1P(astnew, m1);
      A_M2P(astnew, m2);
      A_M3P(astnew, m3);
      A_CHUNKP(astnew, chunk);
      A_SCHED_TYPEP(astnew, A_SCHED_TYPEG(ast));
      A_ORDEREDP(astnew, A_ORDEREDG(ast));
      A_DISTRIBUTEP(astnew, A_DISTRIBUTEG(ast));
      A_DISTPARDOP(astnew, A_DISTPARDOG(ast));
      A_TASKLOOPP(astnew, A_TASKLOOPG(ast));
    }
    break;
  case A_MP_ENDPDO:
  case A_MP_ENDSECTIONS:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
    break;
  case A_PREFETCH:
    lop = ast_rewrite(A_LOPG(ast));
    if (lop != A_LOPG(ast)) {
      astnew = new_node(atype);
      A_LOPP(astnew, lop);
      A_OPTYPEP(astnew, A_OPTYPEG(ast));
    }
    break;
  case A_PRAGMA:
    lop = ast_rewrite(A_LOPG(ast));
    rop = ast_rewrite(A_ROPG(ast));
    if (lop != A_LOPG(ast) || rop != A_ROPG(ast)) {
      astnew = new_node(atype);
      A_LOPP(astnew, lop);
      A_ROPP(astnew, rop);
      A_PRAGMATYPEP(astnew, A_PRAGMATYPEG(ast));
      A_PRAGMASCOPEP(astnew, A_PRAGMASCOPEG(ast));
    }
    break;
  default:
    interr("ast_rewrite: unexpected ast", ast, 2);
    return ast;
  }

  ast_replace(ast, astnew);
  if (astnew != ast) {
    if (rewrite_opfields & 0x1)
      A_OPT1P(astnew, A_OPT1G(ast));
    if (rewrite_opfields & 0x2)
      A_OPT2P(astnew, A_OPT2G(ast));
  }
  return astnew;
}

/** \brief Only called by the semantic analyzer; if it needs to be used by all
   phases,
           many ASTs need to be added as cases.
 */
void
ast_clear_repl(int ast)
{
  int asd;
  int numdim;
  int arg;
  int argt;
  int argcnt;
  int i;

  if (ast == 0)
    return; /* watch for a 'null' argument */
  if (A_REPLG(ast) == 0)
    return;
  switch (A_TYPEG(ast)) {
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_LABEL:
    break;
  case A_MEM:
    ast_clear_repl((int)A_PARENTG(ast));
    break;
  case A_SUBSTR:
    ast_clear_repl((int)A_LOPG(ast));
    ast_clear_repl((int)A_LEFTG(ast));
    ast_clear_repl((int)A_RIGHTG(ast));
    break;
  case A_BINOP:
    ast_clear_repl((int)A_LOPG(ast));
    ast_clear_repl((int)A_ROPG(ast));
    break;
  case A_UNOP:
    ast_clear_repl((int)A_LOPG(ast));
    break;
  case A_PAREN:
    ast_clear_repl((int)A_LOPG(ast));
    break;
  case A_CONV:
    ast_clear_repl((int)A_LOPG(ast));
    break;
  case A_SUBSCR:
    ast_clear_repl((int)A_LOPG(ast));
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd);
    assert(numdim > 0 && numdim <= 7, "ast_clear_repl: bad numdim", ast, 4);
    for (i = 0; i < numdim; ++i)
      ast_clear_repl((int)ASD_SUBS(asd, i));
    break;
  case A_TRIPLE:
    ast_clear_repl((int)A_LBDG(ast));
    ast_clear_repl((int)A_UPBDG(ast));
    ast_clear_repl((int)A_STRIDEG(ast));
    break;
  case A_FUNC:
    ast_clear_repl((int)A_LOPG(ast));
    argt = A_ARGSG(ast);
    argcnt = A_ARGCNTG(ast);
    for (i = 0; i < argcnt; i++) {
      arg = ARGT_ARG(argt, i);
      (void)ast_clear_repl(arg);
    }
    break;
  case A_INTR:
  case A_ICALL:
    ast_clear_repl((int)A_LOPG(ast));
    argt = A_ARGSG(ast);
    argcnt = A_ARGCNTG(ast);
    for (i = 0; i < argcnt; i++) {
      arg = ARGT_ARG(argt, i);
      (void)ast_clear_repl(arg);
    }
    break;
  case A_REALIGN:
  case A_REDISTRIBUTE:
    ast_clear_repl((int)A_LOPG(ast));
    break;
  default:
    interr("ast_clear_repl: unexpected ast", ast, 2);
  }

  A_REPLP(ast, 0);
}

static ast_preorder_fn _preorder;
static ast_visit_fn _postorder;
static void _ast_trav(int ast, int *extra_arg);

/** \brief General ast traversal function: uses a list to keep track of the
           ast nodes which have been visited; if a node is visited, the node's
           REPL field is non-zero.
    \param ast       the ast to traverse
    \param preorder  called before visiting children; return TRUE to prevent
                     visiting ast's operands
    \param postorder called after visiting children
    \param extra_arg passed to preorder and postorder

    \a preorder and \a postorder can be NULL. If they are not, they are called
    with two arguments, an ast and a pointer. The pointer argument 'extra_arg'
    (possibly NULL) may be used by the caller to pass value(s) to visit
    routines, used by the visit routines to return values, or both.

    Visited asts are linked together using 'visit_list'; the caller must call
    ast_unvisit() to cleanup up the VISIT and REPL fields of the asts.  To begin
    the traverse, ast #1 must be marked visited by the caller; e.g.,
    <pre>
      ast_visit(1, 1);
    </pre>
 */
void
ast_traverse(int ast, ast_preorder_fn preorder, ast_visit_fn postorder,
             int *extra_arg)
{
  ast_preorder_fn save_preorder = _preorder;
  ast_visit_fn save_postorder = _postorder;
  LOGICAL save_ast_check_visited = ast_check_visited;
  ast_check_visited = TRUE;
  _preorder = preorder;
  _postorder = postorder;
  _ast_trav(ast, extra_arg);
  _preorder = save_preorder;
  _postorder = save_postorder;
  ast_check_visited = save_ast_check_visited;
}

/** \brief Recursively visit the ast operands of \a ast; useful if the caller
           needs check the 'result' (via extra_arg) of the visit function.

    See ast_traverse() for details about params.

    For the case where it's necessary to perform certain actions/checks when
    an ast has already been visited, ast_estab_visited(visit) may be called
    prior to ast_traverse() to establish such a function.  ast_unvisit()
    removes this function.
 */
void
ast_traverse_all(int ast, ast_preorder_fn preorder, ast_visit_fn postorder,
                 int *extra_arg)
{
  ast_preorder_fn save_preorder = _preorder;
  ast_visit_fn save_postorder = _postorder;
  LOGICAL save_ast_check_visited = ast_check_visited;
  ast_check_visited = FALSE;
  _preorder = preorder;
  _postorder = postorder;
  _ast_trav(ast, extra_arg);
  _preorder = save_preorder;
  _postorder = save_postorder;
  ast_check_visited = save_ast_check_visited;
}

/** \brief While in an ast_traverse recursion, continue on another subtree */
void
ast_traverse_more(int ast, int *extra_arg)
{
  _ast_trav(ast, extra_arg);
} /* ast_traverse_more */

static void
_ast_trav(int ast, int *extra_arg)
{
  if (ast_check_visited) {
    if (A_VISITG(ast)) {
      if (_visited != NULL)
        (*_visited)(ast, extra_arg);
      return;
    }
    ast_visit(ast, 1);
  }

  if (_preorder != NULL) {
    if ((*_preorder)(ast, extra_arg))
      return;
  }

  ast_trav_recurse(ast, extra_arg);

  if (_postorder != NULL)
    (*_postorder)(ast, extra_arg);
}

void
ast_trav_recurse(int ast, int *extra_arg)
{
  int atype;
  int i, asd;
  int astli;
  int argt;
  int cnt;

  switch (atype = A_TYPEG(ast)) {
  case A_NULL:
  case A_ID:
  case A_CNST:
  case A_LABEL:
    break;
  case A_BINOP:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    _ast_trav((int)A_ROPG(ast), extra_arg);
    break;
  case A_UNOP:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_CMPLXC:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    _ast_trav((int)A_ROPG(ast), extra_arg);
    break;
  case A_CONV:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_PAREN:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_MEM:
    _ast_trav((int)A_PARENTG(ast), extra_arg);
    _ast_trav((int)A_MEMG(ast), extra_arg);
    break;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    _ast_trav((int)A_LOPG(ast), extra_arg);
    for (i = 0; i < (int)ASD_NDIM(asd); i++)
      _ast_trav((int)ASD_SUBS(asd, i), extra_arg);
    break;
  case A_SUBSTR:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_LEFTG(ast))
      _ast_trav((int)A_LEFTG(ast), extra_arg);
    if (A_RIGHTG(ast))
      _ast_trav((int)A_RIGHTG(ast), extra_arg);
    break;
  case A_INIT:
    if (A_LEFTG(ast))
      _ast_trav((int)A_LEFTG(ast), extra_arg);
    if (A_RIGHTG(ast))
      _ast_trav((int)A_RIGHTG(ast), extra_arg);
    break;
  case A_TRIPLE:
    /* [lb]:[ub][:stride] */
    if (A_LBDG(ast))
      _ast_trav((int)A_LBDG(ast), extra_arg);
    if (A_UPBDG(ast))
      _ast_trav((int)A_UPBDG(ast), extra_arg);
    if (A_STRIDEG(ast))
      _ast_trav((int)A_STRIDEG(ast), extra_arg);
    break;
  case A_INTR:
  case A_CALL:
  case A_ICALL:
  case A_FUNC:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    cnt = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    for (i = 0; i < cnt; i++)
      /* watch for optional args */
      if (ARGT_ARG(argt, i) != 0)
        _ast_trav((int)ARGT_ARG(argt, i), extra_arg);
    break;
  case A_ASN:
    _ast_trav((int)A_DESTG(ast), extra_arg);
    _ast_trav((int)A_SRCG(ast), extra_arg);
    break;
  case A_IF:
    _ast_trav((int)A_IFEXPRG(ast), extra_arg);
    _ast_trav((int)A_IFSTMTG(ast), extra_arg);
    break;
  case A_IFTHEN:
    _ast_trav((int)A_IFEXPRG(ast), extra_arg);
    break;
  case A_ELSE:
    break;
  case A_ELSEIF:
    _ast_trav((int)A_IFEXPRG(ast), extra_arg);
    break;
  case A_AIF:
    _ast_trav((int)A_IFEXPRG(ast), extra_arg);
    _ast_trav((int)A_L1G(ast), extra_arg);
    _ast_trav((int)A_L2G(ast), extra_arg);
    _ast_trav((int)A_L3G(ast), extra_arg);
    break;
  case A_GOTO:
    _ast_trav((int)A_L1G(ast), extra_arg);
    break;
  case A_CGOTO:
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli))
      _ast_trav((int)ASTLI_AST(astli), extra_arg);
    _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_AGOTO:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli))
      _ast_trav((int)ASTLI_AST(astli), extra_arg);
    break;
  case A_ASNGOTO:
#if DEBUG
    assert(A_TYPEG(A_SRCG(ast)) == A_LABEL,
           "_ast_trav, src A_ASNGOTO not label", A_SRCG(ast), 3);
#endif
    if ((i = FMTPTG(A_SPTRG(A_SRCG(ast)))))
      _ast_trav((int)A_DESTG(ast), extra_arg);
    else {
      _ast_trav((int)A_SRCG(ast), extra_arg);
      _ast_trav((int)A_DESTG(ast), extra_arg);
    }
    break;
  case A_DO:
    if (A_DOLABG(ast))
      _ast_trav((int)A_DOLABG(ast), extra_arg);
    _ast_trav((int)A_DOVARG(ast), extra_arg);
    _ast_trav((int)A_M1G(ast), extra_arg);
    _ast_trav((int)A_M2G(ast), extra_arg);
    if (A_M3G(ast))
      _ast_trav((int)A_M3G(ast), extra_arg);
    if (A_M4G(ast))
      _ast_trav((int)A_M4G(ast), extra_arg);
    break;
  case A_DOWHILE:
    if (A_DOLABG(ast))
      _ast_trav((int)A_DOLABG(ast), extra_arg);
    _ast_trav((int)A_IFEXPRG(ast), extra_arg);
    break;
  case A_STOP:
  case A_PAUSE:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_RETURN:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_ALLOC:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_DESTG(ast))
      _ast_trav((int)A_DESTG(ast), extra_arg);
    if (A_M3G(ast))
      _ast_trav((int)A_M3G(ast), extra_arg);
    if (A_STARTG(ast))
      _ast_trav((int)A_STARTG(ast), extra_arg);
    if (A_DEVSRCG(ast))
      _ast_trav((int)A_DEVSRCG(ast), extra_arg);
    if (A_ALIGNG(ast))
      _ast_trav((int)A_ALIGNG(ast), extra_arg);
    _ast_trav((int)A_SRCG(ast), extra_arg);
    break;
  case A_WHERE:
    _ast_trav((int)A_IFEXPRG(ast), extra_arg);
    if (A_IFSTMTG(ast))
      _ast_trav((int)A_IFSTMTG(ast), extra_arg);
    break;
  case A_ELSEFORALL:
  case A_ELSEWHERE:
    break;
  case A_FORALL:
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli))
      _ast_trav((int)ASTLI_TRIPLE(astli), extra_arg);
    if (A_IFEXPRG(ast))
      _ast_trav((int)A_IFEXPRG(ast), extra_arg);
    if (A_IFSTMTG(ast))
      _ast_trav((int)A_IFSTMTG(ast), extra_arg);
    break;
  case A_REDIM:
    _ast_trav((int)A_SRCG(ast), extra_arg);
    break;
  case A_ENTRY:
  case A_COMMENT:
  case A_COMSTR:
  case A_ENDIF:
  case A_ENDWHERE:
  case A_ENDFORALL:
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
    break;
  case A_REALIGN:
  case A_REDISTRIBUTE:
    _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_HLOCALIZEBNDS:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_ITRIPLEG(ast))
      _ast_trav((int)A_ITRIPLEG(ast), extra_arg);
    if (A_OTRIPLEG(ast))
      _ast_trav((int)A_OTRIPLEG(ast), extra_arg);
    if (A_DIMG(ast))
      _ast_trav((int)A_DIMG(ast), extra_arg);
    break;
  case A_HALLOBNDS:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_HCYCLICLP:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_ITRIPLEG(ast))
      _ast_trav((int)A_ITRIPLEG(ast), extra_arg);
    if (A_OTRIPLEG(ast))
      _ast_trav((int)A_OTRIPLEG(ast), extra_arg);
    if (A_OTRIPLE1G(ast))
      _ast_trav((int)A_OTRIPLE1G(ast), extra_arg);
    if (A_DIMG(ast))
      _ast_trav((int)A_DIMG(ast), extra_arg);
    break;
  case A_HOFFSET:
    _ast_trav((int)A_DESTG(ast), extra_arg);
    _ast_trav((int)A_LOPG(ast), extra_arg);
    _ast_trav((int)A_ROPG(ast), extra_arg);
    break;
  case A_HSECT:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_BVECTG(ast))
      _ast_trav((int)A_BVECTG(ast), extra_arg);
    break;
  case A_HCOPYSECT:
    if (A_DESTG(ast))
      _ast_trav((int)A_DESTG(ast), extra_arg);
    if (A_SRCG(ast))
      _ast_trav((int)A_SRCG(ast), extra_arg);
    if (A_DDESCG(ast))
      _ast_trav((int)A_DDESCG(ast), extra_arg);
    if (A_SDESCG(ast))
      _ast_trav((int)A_SDESCG(ast), extra_arg);
    break;
  case A_HPERMUTESECT:
    if (A_DESTG(ast))
      _ast_trav((int)A_DESTG(ast), extra_arg);
    if (A_SRCG(ast))
      _ast_trav((int)A_SRCG(ast), extra_arg);
    if (A_DDESCG(ast))
      _ast_trav((int)A_DDESCG(ast), extra_arg);
    if (A_SDESCG(ast))
      _ast_trav((int)A_SDESCG(ast), extra_arg);
    if (A_BVECTG(ast))
      _ast_trav((int)A_BVECTG(ast), extra_arg);
    break;
  case A_HOVLPSHIFT:
    if (A_SRCG(ast))
      _ast_trav((int)A_SRCG(ast), extra_arg);
    if (A_SDESCG(ast))
      _ast_trav((int)A_SDESCG(ast), extra_arg);
    break;
  case A_HGETSCLR:
    if (A_DESTG(ast))
      _ast_trav((int)A_DESTG(ast), extra_arg);
    if (A_SRCG(ast))
      _ast_trav((int)A_SRCG(ast), extra_arg);
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_HGATHER:
  case A_HSCATTER:
    if (A_VSUBG(ast))
      _ast_trav((int)A_VSUBG(ast), extra_arg);
    if (A_DESTG(ast))
      _ast_trav((int)A_DESTG(ast), extra_arg);
    if (A_SRCG(ast))
      _ast_trav((int)A_SRCG(ast), extra_arg);
    if (A_DDESCG(ast))
      _ast_trav((int)A_DDESCG(ast), extra_arg);
    if (A_SDESCG(ast))
      _ast_trav((int)A_SDESCG(ast), extra_arg);
    if (A_MDESCG(ast))
      _ast_trav((int)A_MDESCG(ast), extra_arg);
    if (A_BVECTG(ast))
      _ast_trav((int)A_BVECTG(ast), extra_arg);
    break;
  case A_HCSTART:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_DESTG(ast))
      _ast_trav((int)A_DESTG(ast), extra_arg);
    if (A_SRCG(ast))
      _ast_trav((int)A_SRCG(ast), extra_arg);
    break;
  case A_HCFINISH:
  case A_HCFREE:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_HOWNERPROC:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_DIMG(ast))
      _ast_trav((int)A_DIMG(ast), extra_arg);
    if (A_M1G(ast))
      _ast_trav((int)A_M1G(ast), extra_arg);
    if (A_M2G(ast))
      _ast_trav((int)A_M2G(ast), extra_arg);
    break;
  case A_MASTER:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MASTER LOP field not set", ast, 2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_ENDMASTER:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_ENDMASTER LOP field not set", ast, 2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    cnt = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    for (i = 0; i < cnt; i++)
      _ast_trav((int)ARGT_ARG(argt, i), extra_arg);
    break;
  case A_CRITICAL:
  case A_ENDCRITICAL:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_[END]CRITICAL LOP field not set", ast, 2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
  case A_ENDATOMIC:
  case A_BARRIER:
  case A_NOBARRIER:
    break;
  case A_MP_PARALLEL:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_PARALLEL LOP field not set", ast, 2);
#endif
    if (A_IFPARG(ast))
      _ast_trav((int)A_IFPARG(ast), extra_arg);
    if (A_NPARG(ast))
      _ast_trav((int)A_NPARG(ast), extra_arg);
    if (A_ENDLABG(ast))
      _ast_trav((int)A_ENDLABG(ast), extra_arg);
    if (A_PROCBINDG(ast))
      _ast_trav((int)A_PROCBINDG(ast), extra_arg);
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_ENDPARALLEL:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_ENDPARALLEL LOP field not set", ast,
           2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_TEAMS:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_TEAMS LOP field not set", ast, 2);
#endif
    if (A_NTEAMSG(ast))
      _ast_trav((int)A_NTEAMSG(ast), extra_arg);
    if (A_THRLIMITG(ast))
      _ast_trav((int)A_THRLIMITG(ast), extra_arg);
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_TARGET:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_TARGET LOP field not set", ast, 2);
#endif
    if (A_IFPARG(ast))
      _ast_trav((int)A_IFPARG(ast), extra_arg);
    break;
  case A_MP_ENDTARGET:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_ENDTARGET LOP field not set", ast, 2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_TARGETDATA:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_TARGETDATA LOP field not set", ast, 2);
#endif
    if (A_IFPARG(ast))
      _ast_trav((int)A_IFPARG(ast), extra_arg);
    break;
  case A_MP_ENDTARGETDATA:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_ENDTARGETDATA LOP field not set", ast,
           2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;

  case A_MP_TARGETUPDATE:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
    if (A_IFPARG(ast))
      _ast_trav((int)A_IFPARG(ast), extra_arg);
    break;

  case A_MP_TASK:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_TASK LOP field not set", ast, 2);
#endif
    if (A_IFPARG(ast))
      _ast_trav((int)A_IFPARG(ast), extra_arg);
    if (A_ENDLABG(ast))
      _ast_trav((int)A_ENDLABG(ast), extra_arg);
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_ENDTASK:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_ENDTASK LOP field not set", ast, 2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_TASKLOOP:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_TASKLOOP LOP field not set", ast, 2);
#endif
    if (A_IFPARG(ast))
      _ast_trav((int)A_IFPARG(ast), extra_arg);
    if (A_FINALPARG(ast))
      _ast_trav((int)A_FINALPARG(ast), extra_arg);
    if (A_PRIORITYG(ast))
      _ast_trav((int)A_PRIORITYG(ast), extra_arg);
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_ETASKLOOP:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_ETASKLOOP LOP field not set", ast, 2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_[END]CRITICAL LOP field not set", ast,
           2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
    break;
  case A_MP_CANCEL:
    if (A_IFPARG(ast))
      _ast_trav((int)A_IFPARG(ast), extra_arg);
#if DEBUG
    assert(A_ENDLABG(ast), "_ast_trav, A_MP_CANCEL ENDLAB field not set", ast,
           2);
#endif
    if (A_ENDLABG(ast))
      _ast_trav((int)A_ENDLABG(ast), extra_arg);
    break;
  case A_MP_CANCELLATIONPOINT:
#if DEBUG
    assert(A_ENDLABG(ast),
           "_ast_trav, A_MP_CANCELLATIONPOINT ENDLAB field not set", ast, 2);
#endif
    if (A_ENDLABG(ast))
      _ast_trav((int)A_ENDLABG(ast), extra_arg);
    break;
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_[END]MASTER LOP field not set", ast,
           2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_[END]SINGLE LOP field not set", ast,
           2);
#endif
    /* call _ast_trav((int)A_LOPG(ast), extra_arg) */
    break;
  case A_MP_TASKFIRSTPRIV:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_MP_TASKFIRSTPRIV LOP field not set", ast,
           2);
    assert(A_ROPG(ast), "_ast_trav, A_MP_TASKFIRSTPRIV ROP field not set", ast,
           2);
#endif
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_ROPG(ast))
      _ast_trav((int)A_ROPG(ast), extra_arg);
    break;
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_TASKGROUP:
  case A_MP_ETASKGROUP:
  case A_MP_BARRIER:
  case A_MP_ETASKDUP:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_ENDPDO:
  case A_MP_PRE_TLS_COPY:
  case A_MP_BCOPYIN:
  case A_MP_COPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_COPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_EMPSCOPE:
  case A_MP_FLUSH:
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_ETASKLOOPREG:
  case A_MP_MAP:
  case A_MP_EMAP:
  case A_MP_TARGETLOOPTRIPCOUNT:
  case A_MP_EREDUCTION:
  case A_MP_BREDUCTION:
  case A_MP_REDUCTIONITEM:
    break;
  case A_MP_BMPSCOPE:
#if DEBUG
    assert(A_STBLKG(ast), "_ast_trav, A_MP_BMPSCOPE STBLK field not set", ast,
           2);
#endif
    if (A_STBLKG(ast))
      _ast_trav((int)A_STBLKG(ast), extra_arg);
    break;
  case A_MP_TASKLOOPREG:
    if (A_M1G(ast))
      _ast_trav((int)A_M1G(ast), extra_arg);
    if (A_M2G(ast))
      _ast_trav((int)A_M2G(ast), extra_arg);
    if (A_M3G(ast))
      _ast_trav((int)A_M3G(ast), extra_arg);
    break;
  case A_MP_PDO:
    if (A_DOLABG(ast))
      _ast_trav((int)A_DOLABG(ast), extra_arg);
    _ast_trav((int)A_DOVARG(ast), extra_arg);
    if (A_LASTVALG(ast))
      _ast_trav((int)A_LASTVALG(ast), extra_arg);
    _ast_trav((int)A_M1G(ast), extra_arg);
    _ast_trav((int)A_M2G(ast), extra_arg);
    if (A_M3G(ast))
      _ast_trav((int)A_M3G(ast), extra_arg);
    if (A_CHUNKG(ast))
      _ast_trav((int)A_CHUNKG(ast), extra_arg);
    if (A_ENDLABG(ast))
      _ast_trav((int)A_ENDLABG(ast), extra_arg);
    break;
  case A_MP_SECTIONS:
    if (A_ENDLABG(ast))
      _ast_trav((int)A_ENDLABG(ast), extra_arg);
    break;
  case A_MP_ATOMICREAD:
    if (A_SRCG(ast))
      _ast_trav((int)A_SRCG(ast), extra_arg);
    break;
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_ROPG(ast))
      _ast_trav((int)A_ROPG(ast), extra_arg);
    break;
  case A_MP_ENDSECTIONS:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
    break;
  case A_PREFETCH:
#if DEBUG
    assert(A_LOPG(ast), "_ast_trav, A_PREFETCH LOP field not set", ast, 2);
#endif
    _ast_trav((int)A_LOPG(ast), extra_arg);
    break;
  case A_PRAGMA:
    if (A_LOPG(ast))
      _ast_trav((int)A_LOPG(ast), extra_arg);
    if (A_ROPG(ast))
      _ast_trav((int)A_ROPG(ast), extra_arg);
    break;
  default:
    interr("ast_trav_recurse:bad astype", atype, 3);
  }
}

static int indent = 0;

/* routine must be externally visible */
void
_dump_shape(int shd, FILE *file)
{
  int l, nd, ii;

  if (file == NULL)
    file = stderr;
  for (l = 0; l < indent; ++l)
    fprintf(file, " ");
  fprintf(file, "  shape:%5d\n", shd);
  nd = SHD_NDIM(shd);
  for (ii = 0; ii < nd; ++ii) {
    for (l = 0; l < indent; ++l)
      fprintf(file, " ");
    fprintf(file, "  [%d].  lwb: %5d   upb: %5d  stride: %5d\n", ii,
            SHD_LWB(shd, ii), SHD_UPB(shd, ii), SHD_STRIDE(shd, ii));
  }
}

/* routine must be externally visible */
void
dump_shape(int shd)
{
  _dump_shape(shd, gbl.dbgfil);
}

/* routine must be externally visible */
void
_dump_one_ast(int i, FILE *file)
{
  int asd, j, k;
  char typeb[512];
  int l;

  if (i <= 0 || i > astb.stg_avail)
    return;
  if (file == NULL)
    file = stderr;
  for (l = 0; l < indent; ++l)
    fprintf(file, " ");
  fprintf(file, "%-10s  hshlk/std:%5d", astb.atypes[A_TYPEG(i)],
          (int)A_HSHLKG(i));
  switch (A_TYPEG(i)) {
  default:
    break;
  case A_ID:
  case A_CNST:
  case A_BINOP:
  case A_UNOP:
  case A_CMPLXC:
  case A_CONV:
  case A_PAREN:
  case A_MEM:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_FUNC:
  case A_INTR:
  case A_INIT:
  case A_ASN:
    getdtype(A_DTYPEG(i), typeb);
    fprintf(file, "  type:%s", typeb);
    break;
  }
  switch (A_TYPEG(i)) {
  default:
    break;
  case A_ID:
  case A_BINOP:
  case A_UNOP:
  case A_CMPLXC:
  case A_CONV:
  case A_PAREN:
  case A_SUBSTR:
  case A_FUNC:
  case A_INTR:
    fprintf(file, "  alias:%5d  callfg:%d", (int)A_ALIASG(i),
            (int)A_CALLFGG(i));
    break;
  }
  if (A_VISITG(i))
    fprintf(file, " visit=%d", A_VISITG(i));
  fprintf(file, " opt=(%d,%d)\n", (int)A_OPT1G(i), (int)A_OPT2G(i));
  for (l = 0; l < indent; ++l)
    fprintf(file, " ");
  fprintf(file, "aptr:%5d", i);
  switch (A_TYPEG(i)) {
  case A_NULL:
    fprintf(file, "  <null_ast>");
    break;
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
    fprintf(file, "  sptr:%5d (%s)", (int)A_SPTRG(i), SYMNAME(A_SPTRG(i)));
    break;
  case A_CNST:
#if DEBUG
    assert(i == A_ALIASG(i), "dump_one_ast, alias of cnst not self", i, 3);
#endif
    fprintf(file, "  sptr:%5d (%s)", (int)A_SPTRG(i),
            getprint((int)A_SPTRG(i)));
    break;
  case A_BINOP:
    fprintf(file, "  lop :%5d  rop:%5d  optype:%d", (int)A_LOPG(i),
            (int)A_ROPG(i), (int)A_OPTYPEG(i));
    break;
  case A_UNOP:
    fprintf(file, "  lop :%5d  optype:%d", (int)A_LOPG(i), (int)A_OPTYPEG(i));
    if (i == astb.ptr0)
      fprintf(file, "   ptr0");
    else if (i == astb.ptr1)
      fprintf(file, "   ptr1");
    else if (i == astb.ptr0c)
      fprintf(file, "   ptr0c");
    break;
  case A_CMPLXC:
    fprintf(file, "  lop :%5d  rop:%5d", (int)A_LOPG(i), (int)A_ROPG(i));
    break;
  case A_CONV:
    fprintf(file, "  opnd:%5d", (int)A_LOPG(i));
    break;
  case A_PAREN:
    fprintf(file, "  opnd:%5d", (int)A_LOPG(i));
    break;
  case A_MEM:
    fprintf(file, "  parent:%5d  mem:%5d", (int)A_PARENTG(i), (int)A_MEMG(i));
    if (A_ALIASG(i)) {
      fprintf(file, "  alias:%5d", (int)A_ALIASG(i));
    }
    break;
  case A_SUBSCR:
    asd = A_ASDG(i);
    fprintf(file, "  opnd:%5d  asd:%5d", (int)A_LOPG(i), asd);
    if (A_ALIASG(i)) {
      fprintf(file, "  alias:%5d", (int)A_ALIASG(i));
    }
    for (j = 0; j < (int)ASD_NDIM(asd); j++) {
      fprintf(file, "\n");
      for (l = 0; l < indent; ++l)
        fprintf(file, " ");
      fprintf(file, "     [%d]:%5d", j + 1, (int)ASD_SUBS(asd, j));
    }
    break;
  case A_SUBSTR:
    fprintf(file, "  opnd:%5d  left:%5d  right:%5d", (int)A_LOPG(i),
            (int)A_LEFTG(i), (int)A_RIGHTG(i));
    break;
  case A_TRIPLE:
    fprintf(file, "  lb:%5d,  ub:%5d,  stride:%5d", (int)A_LBDG(i),
            (int)A_UPBDG(i), (int)A_STRIDEG(i));
    break;
  case A_FUNC:
  case A_INTR:
  case A_CALL:
  case A_ICALL:
    j = A_ARGCNTG(i);
    fprintf(file, "  lop:%5d  argcnt:%5d  args:%5d", (int)A_LOPG(i), j,
            (int)A_ARGSG(i));
    if (A_TYPEG(i) == A_INTR || A_TYPEG(i) == A_ICALL || A_TYPEG(i) == A_INIT)
      fprintf(file, "  optype:%5d", (int)A_OPTYPEG(i));
    k = 0;
    while (j--) {
      fprintf(file, "\n");
      for (l = 0; l < indent; ++l)
        fprintf(file, " ");
      fprintf(file, "     (%5d):%5d", k, (int)ARGT_ARG(A_ARGSG(i), k));
      k++;
    }
    break;
  case A_ASN:
  case A_ASNGOTO:
    fprintf(file, "  dest:%5d  src:%5d", (int)A_DESTG(i), (int)A_SRCG(i));
    break;
  case A_IF:
    fprintf(file, "  ifexpr:%5d  ifstmt:%5d", (int)A_IFEXPRG(i),
            (int)A_IFSTMTG(i));
    break;
  case A_IFTHEN:
    fprintf(file, "  ifexpr:%5d", (int)A_IFEXPRG(i));
    break;
  case A_ELSE:
    break;
  case A_ELSEIF:
    fprintf(file, "  ifexpr:%5d", (int)A_IFEXPRG(i));
    break;
  case A_ENDIF:
    break;
  case A_AIF:
    fprintf(file, "  ifexpr:%5d,", (int)A_IFEXPRG(i));
    fprintf(file, "  l1:%5d,  l2:%5d,  l3:%5d", (int)A_L1G(i), (int)A_L2G(i),
            (int)A_L3G(i));
    break;
  case A_GOTO:
    fprintf(file, "  l1:%5d", A_L1G(i));
    break;
  case A_CGOTO:
  case A_AGOTO:
    fprintf(file, "  lop:%5d  list:%5d", A_LOPG(i), j = A_LISTG(i));
    dump_astli(j);
    break;
  case A_DO:
    fprintf(file, "  lab:%5d", (int)A_DOLABG(i));
    fprintf(file, "  var:%5d", (int)A_DOVARG(i));
    fprintf(file, "  m1:%5d", (int)A_M1G(i));
    fprintf(file, "  m2:%5d", (int)A_M2G(i));
    fprintf(file, "  m3:%5d", (int)A_M3G(i));
    fprintf(file, "  m4:%5d", (int)A_M4G(i));
    break;
  case A_DOWHILE:
    fprintf(file, "  lab:%5d", (int)A_DOLABG(i));
    fprintf(file, "  ifexpr:%5d", (int)A_IFEXPRG(i));
    break;
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
    break;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    fprintf(file, "  lop:%5d", (int)A_LOPG(i));
    break;
  case A_ALLOC:
    fprintf(file,
            "  tkn:%5d  lop:%5d  src:%5d  dest:%5d  m3:%5d"
            "start:%5d  dallocmem: %d  firstalloc: %d devsrc: %d align: %d",
            (int)A_TKNG(i), (int)A_LOPG(i), A_SRCG(i), A_DESTG(i), A_M3G(i),
            A_STARTG(i), A_DALLOCMEMG(i), A_FIRSTALLOCG(i), A_DEVSRCG(i),
            A_ALIGNG(i));
    break;
  case A_WHERE:
    fprintf(file, "  ifstmt:%5d  ifexpr:%5d", (int)A_IFSTMTG(i),
            (int)A_IFEXPRG(i));
    break;
  case A_FORALL:
    fprintf(file, "  ifstmt:%5d  ifexpr:%5d  src:%5d  list:%5d",
            (int)A_IFSTMTG(i), (int)A_IFEXPRG(i), A_SRCG(i),
            j = (int)A_LISTG(i));
    dump_astli(j);
    break;
  case A_ELSEWHERE:
  case A_ENDWHERE:
  case A_ENDFORALL:
  case A_ELSEFORALL:
    break;
  case A_REDIM:
    fprintf(file, "  src:%5d", (int)A_SRCG(i));
    break;
  case A_COMMENT:
    fprintf(file, "  lop:%5d", (int)A_LOPG(i));
    break;
  case A_INIT:
    fprintf(file, "  left:%5d  right:%5d  sptr:%5d (%s)", (int)A_LEFTG(i),
            (int)A_RIGHTG(i), (int)A_SPTRG(i), getprint((int)A_SPTRG(i)));
    break;
  case A_COMSTR:
    fprintf(file, "  comment:%s", COMSTR(i));
    break;
  case A_HALLOBNDS:
    fprintf(file, "  lop:%5d", A_LOPG(i));
    break;
  case A_HCYCLICLP:
    fprintf(file, "  lop:%5d", A_LOPG(i));
    fprintf(file, "  itriple:%5d", A_ITRIPLEG(i));
    fprintf(file, "  otriple:%5d", A_OTRIPLEG(i));
    fprintf(file, "  otriple1:%5d", A_OTRIPLE1G(i));
    fprintf(file, "  dim:%5d", A_DIMG(i));
    break;
  case A_HOFFSET:
    fprintf(file, " dest:%5d", A_DESTG(i));
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " rop:%5d", A_ROPG(i));
    break;
  case A_HSECT:
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " bvect:%5d", A_BVECTG(i));
    break;
  case A_HCOPYSECT:
    fprintf(file, " dest:%5d", A_DESTG(i));
    fprintf(file, " src:%5d", A_SRCG(i));
    fprintf(file, " ddesc:%5d", A_DDESCG(i));
    fprintf(file, " sdesc:%5d", A_SDESCG(i));
    break;
  case A_HPERMUTESECT:
    fprintf(file, " dest:%5d", A_DESTG(i));
    fprintf(file, " src:%5d", A_SRCG(i));
    fprintf(file, " ddesc:%5d", A_DDESCG(i));
    fprintf(file, " sdesc:%5d", A_SDESCG(i));
    fprintf(file, " bvect:%5d", A_BVECTG(i));
    break;
  case A_HOVLPSHIFT:
    fprintf(file, " src:%5d", A_SRCG(i));
    fprintf(file, " sdesc:%5d", A_SDESCG(i));
    break;
  case A_HGETSCLR:
    fprintf(file, " dest:%5d", A_DESTG(i));
    fprintf(file, " src:%5d\n", A_SRCG(i));
    if (A_LOPG(i)) {
      fprintf(file, " lop:%5d\n", A_LOPG(i));
    }
    break;
  case A_HGATHER:
  case A_HSCATTER:
    fprintf(file, " vsub:%5d", A_VSUBG(i));
    fprintf(file, " dest:%5d", A_DESTG(i));
    fprintf(file, " src:%5d\n", A_SRCG(i));
    fprintf(file, " ddesc:%5d", A_DDESCG(i));
    fprintf(file, " sdesc:%5d", A_SDESCG(i));
    fprintf(file, " mdesc:%5d", A_MDESCG(i));
    fprintf(file, " bvect:%5d", A_BVECTG(i));
    break;
  case A_HCSTART:
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " dest:%5d", A_DESTG(i));
    fprintf(file, " src:%5d\n", A_SRCG(i));
    break;
  case A_HCFINISH:
  case A_HCFREE:
    fprintf(file, " lop:%5d", A_LOPG(i));
    break;
  case A_MASTER:
    fprintf(file, " lop:%5d", A_LOPG(i));
    break;
  case A_ENDMASTER:
    fprintf(file, " lop:%5d", A_LOPG(i));
    j = A_ARGCNTG(i);
    fprintf(file, " argcnt:%5d", j);
    fprintf(file, " args:%5d\n", A_ARGSG(i));
    k = 0;
    while (j-- > 0) {
      fprintf(file, "\n");
      for (l = 0; l < indent; ++l)
        fprintf(file, " ");
      fprintf(file, "     (%5d):%5d", k, (int)ARGT_ARG(A_ARGSG(i), k));
      k++;
    }
    break;
  case A_CRITICAL:
  case A_ENDCRITICAL:
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
    fprintf(file, " lop:%5d", A_LOPG(i));
    break;
  case A_ENDATOMIC:
  case A_BARRIER:
  case A_NOBARRIER:
    break;
  case A_MP_BMPSCOPE:
    fprintf(file, " stblk:%5d", A_STBLKG(i));
    break;
  case A_MP_EMPSCOPE:
    break;
  case A_MP_PARALLEL:
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " ifpar:%5d", A_IFPARG(i));
    fprintf(file, " npar:%5d", A_NPARG(i));
    fprintf(file, " endlab:%5d", A_ENDLABG(i));
    fprintf(file, " procbind:%5d", A_PROCBINDG(i));
    break;
  case A_MP_ATOMICREAD:
    fprintf(file, " rhs/expr:%5d", A_SRCG(i));
    break;
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    fprintf(file, " lhs:%5d", A_LOPG(i));
    fprintf(file, " rhs/expr:%5d", A_ROPG(i));
    break;
  case A_MP_TEAMS:
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " nteams:%5d", A_NTEAMSG(i));
    fprintf(file, " thrlimit:%5d", A_THRLIMITG(i));
    break;
  case A_MP_TASKFIRSTPRIV:
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " rop:%5d", A_ROPG(i));
    break;
  case A_MP_TASK:
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " ifpar:%5d", A_IFPARG(i));
    fprintf(file, " final:%5d", A_FINALPARG(i));
    if (A_UNTIEDG(i))
      fprintf(file, "  untied");
    if (A_EXEIMMG(i))
      fprintf(file, "  exeimm");
    if (A_MERGEABLEG(i))
      fprintf(file, "  mergeable");
    if (A_ENDLABG(i))
      fprintf(file, " endlab:%5d", A_ENDLABG(i));
    break;
  case A_MP_TASKLOOP:
    fprintf(file, " lop:%5d", A_LOPG(i));
    fprintf(file, " ifpar:%5d", A_IFPARG(i));
    fprintf(file, " final:%5d", A_FINALPARG(i));
    fprintf(file, " priority:%5d", A_PRIORITYG(i));
    if (A_UNTIEDG(i))
      fprintf(file, "  untied");
    if (A_EXEIMMG(i))
      fprintf(file, "  exeimm");
    if (A_MERGEABLEG(i))
      fprintf(file, "  mergeable");
    if (A_NOGROUPG(i))
      fprintf(file, "  nogroup");
    if (A_GRAINSIZEG(i))
      fprintf(file, "  grainsize");
    if (A_NUM_TASKSG(i))
      fprintf(file, "  num_tasks");
    break;
  case A_MP_TARGET:
    fprintf(file, " iftarget:%5d", A_IFPARG(i));
    break;
  case A_MP_TARGETUPDATE:
    fprintf(file, " iftargetupdate:%5d", A_IFPARG(i));
    break;
  case A_MP_TARGETEXITDATA:
    fprintf(file, " iftargetexitdata:%5d", A_IFPARG(i));
    break;
  case A_MP_TARGETENTERDATA:
    fprintf(file, " iftargetenterdata:%5d", A_IFPARG(i));
    break;
  case A_MP_TARGETDATA:
    fprintf(file, " iftargetdata:%5d", A_IFPARG(i));
    break;

  case A_MP_ENDPARALLEL:
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_ENDSECTIONS:
  case A_MP_SECTIONS:
    fprintf(file, " endlab:%5d", (int)A_ENDLABG(i));
    break;
  case A_MP_ENDTASK:
    fprintf(file, " lop:%5d", A_LOPG(i));
    break;
  case A_MP_CANCEL:
    fprintf(file, " ifcancel:%5d", A_IFPARG(i));
    fprintf(file, " cancelkind:%5d", A_CANCELKINDG(i));
    fprintf(file, " endlab:%5d", (int)A_ENDLABG(i));
    break;
  case A_MP_CANCELLATIONPOINT:
    fprintf(file, " cancelkind:%5d", A_CANCELKINDG(i));
    fprintf(file, " endlab:%5d", (int)A_ENDLABG(i));
    break;
  case A_MP_PDO:
    fprintf(file, "  lab:%5d", (int)A_DOLABG(i));
    fprintf(file, "  var:%5d", (int)A_DOVARG(i));
    fprintf(file, "  lastvar:%5d", (int)A_LASTVALG(i));
    fprintf(file, "  m1:%5d", (int)A_M1G(i));
    fprintf(file, "  m2:%5d", (int)A_M2G(i));
    fprintf(file, "  m3:%5d\n", (int)A_M3G(i));
    fprintf(file, "  chunk:%5d", (int)A_CHUNKG(i));
    fprintf(file, "  sched_type:%5d", (int)A_SCHED_TYPEG(i));
    if (A_ORDEREDG(i))
      fprintf(file, "  ordered");
    if (A_DISTPARDOG(i))
      fprintf(file, "  distpardo");
    if (A_DISTRIBUTEG(i))
      fprintf(file, "  distribute");
    if (A_TASKLOOPG(i))
      fprintf(file, "  taskloop");
    if (A_ENDLABG(i))
      fprintf(file, "  endlab:%5d", (int)A_ENDLABG(i));
    break;
  case A_MP_TASKLOOPREG:
    fprintf(file, "  m1:%5d", (int)A_M1G(i));
    fprintf(file, "  m2:%5d", (int)A_M2G(i));
    fprintf(file, "  m3:%5d\n", (int)A_M3G(i));
    break;
  case A_MP_ETASKLOOPREG:
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_ENDTARGETDATA:
  case A_MP_ENDTARGET:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_TASKGROUP:
  case A_MP_ETASKGROUP:
  case A_MP_BARRIER:
  case A_MP_ETASKDUP:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_ENDPDO:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_BCOPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_FLUSH:
    break;
  case A_MP_PRE_TLS_COPY:
  case A_MP_COPYIN:
  case A_MP_COPYPRIVATE:
    fprintf(file, "  sptr:%5d (%s)", (int)A_SPTRG(i),
            getprint((int)A_SPTRG(i)));
    fprintf(file, "  size:%5d", (int)A_ROPG(i));
    break;
  case A_PREFETCH:
    fprintf(file, " lop:%5d  optype:%d", A_LOPG(i), A_OPTYPEG(i));
    break;
  case A_PRAGMA:
    fprintf(file, " lop:%5d rop:%5d  type:%d scope:%d", A_LOPG(i), A_ROPG(i),
            A_PRAGMATYPEG(i), A_PRAGMASCOPEG(i));
    if (A_PRAGMATYPEG(i) == PR_ACCTILE) {
      j = A_ARGCNTG(i);
      fprintf(file, "  argcnt:%5d  args:%5d", (int)A_LOPG(i), j);
      k = 0;
      while (j--) {
        fprintf(file, "\n");
        for (l = 0; l < indent; ++l)
          fprintf(file, " ");
        fprintf(file, "     (%5d):%5d", k, (int)ARGT_ARG(A_ARGSG(i), k));
        k++;
      }
    }
    break;
  default:
    fprintf(file, "NO DUMP AVL");
    break;
  }
  fprintf(file, "\n");
  if ((A_TYPEG(i) == A_ASN || A_ISEXPR(A_TYPEG(i))) && A_SHAPEG(i)) {
    dump_shape(A_SHAPEG(i));
  }
}

/* routine must be externally visible */
void
dump_one_ast(int i)
{
  _dump_one_ast(i, gbl.dbgfil);
}

/* routine must be externally visible */
void
dump_ast_tree(int i)
{
  int j, k;
  int asd;

  if (i <= 0 || i > astb.stg_avail)
    return;
  fprintf(gbl.dbgfil, "\n");
  dump_one_ast(i);
  switch (A_TYPEG(i)) {
  case A_NULL:
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_CNST:
  case A_CMPLXC:
  case A_GOTO:
  case A_CGOTO:
  case A_AGOTO:
    break;
  case A_BINOP:
    indent += 3;
    dump_ast_tree(A_LOPG(i));
    dump_ast_tree(A_ROPG(i));
    indent -= 3;
    break;
  case A_MEM:
    indent += 3;
    dump_ast_tree(A_MEMG(i));
    dump_ast_tree(A_PARENTG(i));
    indent -= 3;
    break;
  case A_CONV:
  case A_UNOP:
  case A_PAREN:
    indent += 3;
    dump_ast_tree(A_LOPG(i));
    indent -= 3;
    break;
  case A_SUBSCR:
    asd = A_ASDG(i);
    indent += 3;
    dump_ast_tree(A_LOPG(i));
    indent += 1;
    for (j = 0; j < (int)ASD_NDIM(asd); j++) {
      dump_ast_tree(ASD_SUBS(asd, j));
    }
    indent -= 4;
    break;
  case A_SUBSTR:
    indent += 3;
    dump_ast_tree(A_LEFTG(i));
    dump_ast_tree(A_RIGHTG(i));
    dump_ast_tree(A_LOPG(i));
    indent -= 3;
    break;
  case A_INIT:
    indent += 3;
    dump_ast_tree(A_LEFTG(i));
    indent -= 3;
    dump_ast_tree(A_RIGHTG(i));
    break;
  case A_TRIPLE:
    indent += 3;
    dump_ast_tree(A_LBDG(i));
    dump_ast_tree(A_UPBDG(i));
    dump_ast_tree(A_STRIDEG(i));
    indent -= 3;
    break;
  case A_FUNC:
  case A_INTR:
  case A_CALL:
  case A_ICALL:
    indent += 1;
    dump_ast_tree(A_LOPG(i));
    j = A_ARGCNTG(i);
    indent += 2;
    k = 0;
    while (j--) {
      dump_ast_tree(ARGT_ARG(A_ARGSG(i), k));
      k++;
    }
    indent -= 3;
    break;
  case A_ASN:
  case A_ASNGOTO:
    indent += 3;
    dump_ast_tree(A_DESTG(i));
    dump_ast_tree(A_SRCG(i));
    indent -= 3;
    break;
  case A_IF:
    indent += 3;
    dump_ast_tree(A_IFEXPRG(i));
    dump_ast_tree(A_IFSTMTG(i));
    indent -= 3;
    break;
  case A_IFTHEN:
    indent += 3;
    dump_ast_tree(A_IFEXPRG(i));
    indent -= 3;
    break;
  case A_ELSE:
    break;
  case A_ELSEIF:
    indent += 3;
    dump_ast_tree(A_IFEXPRG(i));
    indent -= 3;
    break;
  case A_ENDIF:
    break;
  case A_AIF:
    indent += 3;
    dump_ast_tree(A_IFEXPRG(i));
    indent -= 3;
    break;
  case A_DO:
    indent += 3;
    dump_ast_tree(A_M1G(i));
    dump_ast_tree(A_M2G(i));
    dump_ast_tree(A_M3G(i));
    dump_ast_tree(A_M4G(i));
    indent -= 3;
    break;
  case A_DOWHILE:
    indent += 3;
    dump_ast_tree(A_IFEXPRG(i));
    indent -= 3;
    break;
  case A_ENDDO:
  case A_CONTINUE:
  case A_END:
    break;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
    indent += 3;
    dump_ast_tree(A_LOPG(i));
    indent -= 3;
    break;
  case A_ALLOC:
    break;
  case A_WHERE:
    indent += 3;
    dump_ast_tree(A_IFEXPRG(i));
    dump_ast_tree(A_IFSTMTG(i));
    indent -= 3;
    break;
  case A_FORALL:
    break;
  case A_ELSEWHERE:
  case A_ENDWHERE:
  case A_ENDFORALL:
  case A_ELSEFORALL:
    break;
  case A_REDIM:
    break;
  case A_COMMENT:
  case A_COMSTR:
    break;
  case A_REALIGN:
  case A_REDISTRIBUTE:
    break;
  case A_HLOCALIZEBNDS:
    break;
  case A_HALLOBNDS:
    break;
  case A_HCYCLICLP:
    break;
  case A_HOFFSET:
    break;
  case A_HSECT:
    break;
  case A_HCOPYSECT:
    break;
  case A_HPERMUTESECT:
    break;
  case A_HOVLPSHIFT:
    break;
  case A_HGETSCLR:
    indent += 3;
    dump_ast_tree(A_DESTG(i));
    dump_ast_tree(A_SRCG(i));
    if (A_LOPG(i)) {
      dump_ast_tree(A_LOPG(i));
    }
    indent -= 3;
    break;
  case A_HGATHER:
  case A_HSCATTER:
    break;
  case A_HCSTART:
    break;
  case A_HCFINISH:
  case A_HCFREE:
    break;
  case A_MASTER:
    break;
  case A_ENDMASTER:
    j = A_ARGCNTG(i);
    indent += 3;
    k = 0;
    while (j-- > 0) {
      dump_ast_tree(ARGT_ARG(A_ARGSG(i), k));
      k++;
    }
    indent -= 3;
    break;
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
  case A_PREFETCH:
    indent += 3;
    dump_ast_tree(A_LOPG(i));
    indent -= 3;
    break;
  case A_PRAGMA:
    indent += 3;
    dump_ast_tree(A_LOPG(i));
    dump_ast_tree(A_ROPG(i));
    if (A_PRAGMATYPEG(i) == PR_ACCTILE) {
      j = A_ARGCNTG(i);
      k = 0;
      while (j-- > 0) {
        int a = ARGT_ARG(A_ARGSG(i), k);
        dump_ast_tree(a);
        k++;
      }
    }
    indent -= 3;
    break;
    indent -= 3;
    break;
  case A_CRITICAL:
  case A_ENDCRITICAL:
  case A_ENDATOMIC:
  case A_BARRIER:
  case A_NOBARRIER:
    break;
  case A_MP_PARALLEL:
    indent += 3;
    dump_ast_tree(A_IFPARG(i));
    dump_ast_tree(A_NPARG(i));
    dump_ast_tree(A_ENDLABG(i));
    dump_ast_tree(A_PROCBINDG(i));
    indent -= 3;
    break;
  case A_MP_TEAMS:
    indent += 3;
    dump_ast_tree(A_NTEAMSG(i));
    dump_ast_tree(A_THRLIMITG(i));
    indent -= 3;
    break;
  case A_MP_BMPSCOPE:
    indent += 3;
    dump_ast_tree(A_STBLKG(i));
    indent -= 3;
    break;
  case A_MP_TASK:
  case A_MP_TASKLOOP:
    indent += 3;
    dump_ast_tree(A_IFPARG(i));
    dump_ast_tree(A_FINALPARG(i));
    dump_ast_tree(A_PRIORITYG(i));
    indent -= 3;
    break;
  case A_MP_TASKFIRSTPRIV:
    indent += 3;
    dump_ast_tree(A_LOPG(i));
    dump_ast_tree(A_ROPG(i));
    indent -= 3;
    break;
  case A_MP_TARGET:
  case A_MP_TARGETDATA:
    indent += 3;
    dump_ast_tree(A_IFPARG(i));
    dump_ast_tree(A_LOPG(i));
    indent -= 3;
    break;
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_TARGETUPDATE:
    indent += 3;
    dump_ast_tree(A_IFPARG(i));
    indent -= 3;
    break;

  case A_MP_ENDTARGET:
  case A_MP_ENDTARGETDATA:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_TASKGROUP:
  case A_MP_ETASKGROUP:
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
  case A_MP_ETASKDUP:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_ENDTASK:
  case A_MP_EMPSCOPE:
  case A_MP_ETASKLOOPREG:
  case A_MP_TASKDUP:
    break;
  case A_MP_TASKREG:
    indent += 3;
    dump_ast_tree(A_ENDLABG(i));
    indent -= 3;
    break;
  case A_MP_CANCEL:
    indent += 3;
    dump_ast_tree(A_IFPARG(i));
    dump_ast_tree(A_ENDLABG(i));
    indent -= 3;
    break;
  case A_MP_SECTIONS:
  case A_MP_CANCELLATIONPOINT:
    indent += 3;
    dump_ast_tree(A_ENDLABG(i));
    indent -= 3;
    break;
  case A_MP_PDO:
    indent += 3;
    dump_ast_tree(A_M1G(i));
    dump_ast_tree(A_M2G(i));
    dump_ast_tree(A_M3G(i));
    dump_ast_tree(A_CHUNKG(i));
    indent -= 3;
    break;
  case A_MP_TASKLOOPREG:
    indent += 3;
    dump_ast_tree(A_M1G(i));
    dump_ast_tree(A_M2G(i));
    dump_ast_tree(A_M3G(i));
    indent -= 3;
    break;
  case A_MP_ATOMICREAD:
    dump_ast_tree(A_SRCG(i));
    indent -= 3;
    break;
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    dump_ast_tree(A_LOPG(i));
    dump_ast_tree(A_ROPG(i));
    indent -= 3;
    break;
  case A_MP_ENDPDO:
  case A_MP_ENDSECTIONS:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_PRE_TLS_COPY:
  case A_MP_BCOPYIN:
  case A_MP_COPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_COPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_FLUSH:
    break;
  default:
    fprintf(gbl.dbgfil, "NO DUMP AVL");
    break;
  }
}

/* routine must be externally visible */
void
dump_ast(void)
{
  unsigned int i;

  fprintf(gbl.dbgfil, "AST Table\n");
  for (i = 1; i < astb.stg_avail; i++) {
    fprintf(gbl.dbgfil, "\n");
    _dump_one_ast(i, gbl.dbgfil);
  }

  fprintf(gbl.dbgfil, "\n");
  if (DBGBIT(4, 512)) {
    fprintf(gbl.dbgfil, "HashIndex  First\n");
    for (i = 0; i <= HSHSZ; i++)
      if (astb.hshtb[i])
        fprintf(gbl.dbgfil, "  %5d    %5d\n", i, (int)astb.hshtb[i]);
  }
}

/* routine must be externally visible */
void
dump_astli(int astli)
{
  while (astli) {
    fprintf(gbl.dbgfil, "\n%5d.  h1:%-5d  h2:%-5d  flags:%04x", astli,
            (int)ASTLI_SPTR(astli), (int)ASTLI_TRIPLE(astli),
            (int)ASTLI_FLAGS(astli));
    astli = ASTLI_NEXT(astli);
  }
}

/* routine must be externally visible */
void
_dump_std(int std, FILE *fil)
{
  int ast;
  if (fil == NULL)
    fil = stderr;
  ast = STD_AST(std);
  fprintf(fil, "std:%5d.  lineno:%-5d  label:%-5d(%s)  ast:%-5d", std,
          STD_LINENO(std), STD_LABEL(std),
          STD_LABEL(std) ? SYMNAME(STD_LABEL(std)) : "", ast);
#undef _PFG
#define _PFG(cond, str) \
  if (cond)             \
  fprintf(fil, " %s", str)
  _PFG(A_CALLFGG(ast), "callfg");
  _PFG(STD_EX(std), "ex");
  _PFG(STD_ST(std), "st");
  _PFG(STD_BR(std), "br");
  _PFG(STD_DELETE(std), "delete");
  _PFG(STD_IGNORE(std), "ignore");
  _PFG(STD_SPLIT(std), "split");
  _PFG(STD_MINFO(std), "info");
  _PFG(STD_LOCAL(std), "local");
  _PFG(STD_PURE(std), "pure");
  _PFG(STD_PAR(std), "par");
  _PFG(STD_CS(std), "cs");
  _PFG(STD_PARSECT(std), "parsect");
  _PFG(STD_TASK(std), "task");
  fprintf(fil, "\n");
  if (STD_LABEL(std))
    fprintf(fil, "%s:\n", SYMNAME(STD_LABEL(std)));
  dbg_print_ast(ast, fil);
}

/* routine must be externally visible */
void
dump_std(void)
{
  int std;
  for (std = STD_NEXT(0); std; std = STD_NEXT(std)) {
    _dump_std(std, gbl.dbgfil);
  }
}

/* routine must be externally visible */
void
dump_stg_stat(const char *where)
{
  FILE *fil;
  if (gbl.dbgfil == NULL)
    fil = stderr;
  else
    fil = gbl.dbgfil;
  fprintf(fil, "  Storage Allocation %s\n", where);
  fprintf(fil, "  AST   :%8d\n", astb.stg_avail);
  fprintf(fil, "  ASD   :%8d\n", astb.asd.stg_avail);
  fprintf(fil, "  STD   :%8d\n", astb.std.stg_avail);
  fprintf(fil, "  ASTLI :%8d\n", astb.astli.stg_avail);
  fprintf(fil, "  ARGT  :%8d\n", astb.argt.stg_avail);
  fprintf(fil, "  SHD   :%8d\n", astb.shd.stg_avail);
  fprintf(fil, "  SYM   :%8d\n", stb.stg_avail);
  fprintf(fil, "  DT    :%8d\n", stb.dt.stg_avail);
}

#include <stdarg.h>

static int _huge(DTYPE);

int
ast_intr(int i_intr, DTYPE dtype, int cnt, ...)
{
  int ast = 0;
  int sptr, sptre;
  va_list vargs;
  int opnd;

  va_start(vargs, cnt);

  sptr = intast_sym[i_intr];
  if (STYPEG(sptr) == ST_PD) {
    /* allow only those predeclareds which are passed thru as intrinsics */
    if (i_intr == I_HUGE) {
      va_end(vargs);
      return _huge(dtype);
    }
    ast = begin_call(A_INTR, sptr, cnt);
    while (cnt--) {
      opnd = va_arg(vargs, int);
      (void)add_arg(opnd);
    }
    A_OPTYPEP(ast, i_intr);
  } else {
    sptre = sptr;
    if (STYPEG(sptr) == ST_GENERIC) {
      switch (DTY(dtype)) {
      case TY_SLOG:
      case TY_SINT:
        if ((sptr = GSINTG(sptr)))
          break;
        FLANG_FALLTHROUGH;
      case TY_WORD:
      case TY_DWORD:
      case TY_BLOG:
      case TY_BINT:
      case TY_LOG:
      case TY_INT:
        sptr = GINTG(sptr);
        break;
      case TY_LOG8:
      case TY_INT8:
        sptr = GINT8G(sptr);
        break;
      case TY_REAL:
        sptr = GREALG(sptr);
        break;
      case TY_DBLE:
        sptr = GDBLEG(sptr);
        break;
      case TY_CMPLX:
        sptr = GCMPLXG(sptr);
        break;
      case TY_DCMPLX:
        sptr = GDCMPLXG(sptr);
        break;
      default:
        sptr = 0;
        break;
      }
      assert(sptr != 0, "ast_intr: unknown generic", 0, 3);
    }
    if (STYPEG(sptre) == ST_INTRIN || STYPEG(sptre) == ST_GENERIC) {
      ast = begin_call(A_INTR, sptre, cnt);
      while (cnt--) {
        opnd = va_arg(vargs, int);
        (void)add_arg(opnd);
      }
      A_OPTYPEP(ast, INTASTG(sptr));
    } else if (i_intr == I_INT) {
      opnd = va_arg(vargs, int);
      sptre = sym_mkfunc_nodesc(mkRteRtnNm(RTE_int), DT_INT);
      ast = begin_call(A_FUNC, sptre, 2);
      (void)add_arg(opnd);
      (void)add_arg(mk_cval((INT)ty_to_lib[DTYG(A_TYPEG(opnd))], DT_INT));
    } else if (i_intr == I_REAL) {
      opnd = va_arg(vargs, int);
      sptre = sym_mkfunc_nodesc(mkRteRtnNm(RTE_real), DT_REAL4);
      ast = begin_call(A_FUNC, sptre, 2);
      (void)add_arg(opnd);
      (void)add_arg(mk_cval((INT)ty_to_lib[DTYG(A_TYPEG(opnd))], DT_INT));
    } else if (i_intr == I_DBLE) {
      opnd = va_arg(vargs, int);
      sptre = sym_mkfunc_nodesc(mkRteRtnNm(RTE_dble), DT_DBLE);
      ast = begin_call(A_FUNC, sptre, 2);
      (void)add_arg(opnd);
      (void)add_arg(mk_cval((INT)ty_to_lib[DTYG(A_TYPEG(opnd))], DT_INT));
    } else {
      assert(FALSE, "ast_intr: unknown predefined", i_intr, ERR_Fatal);
    }
  }
  A_DTYPEP(ast, dtype);
  A_SHAPEP(ast, 0);

  va_end(vargs);
  return ast;
}

static int
_huge(DTYPE dtype)
{
  INT val[4];
  int tmp, ast, sptr;
  const char *sname;

  switch (DTYG(dtype)) {
  case TY_BINT:
    val[0] = 0x7f;
    sname = "huge(1_1)";
    goto const_int_val;
  case TY_SINT:
    val[0] = 0x7fff;
    sname = "huge(1_2)";
    goto const_int_val;
  case TY_INT:
    val[0] = 0x7fffffff;
    sname = "huge(1_4)";
    goto const_int_val;
  case TY_INT8:
    val[0] = 0x7fffffff;
    val[1] = 0xffffffff;
    sname = "huge(1_8)";
    goto const_int8_val;
  case TY_REAL:
    /* 3.402823466E+38 */
    val[0] = 0x7f7fffff;
    sname = "huge(1.0_4)";
    goto const_real_val;
  case TY_DBLE:
    sname = "huge(1.0_8)";
    if (XBIT(49, 0x40000)) {               /* C90 */
#define C90_HUGE "0.136343516952426e+2466" /* 0577757777777777777776 */
      atoxd(C90_HUGE, &val[0], strlen(C90_HUGE));
    } else {
      /* 1.79769313486231571E+308 */
      val[0] = 0x7fefffff;
      val[1] = 0xffffffff;
    }
    goto const_dble_val;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    sname = "huge(1.0_16)";
    /* the largest number that is not an infinity in quad precision.
       Approximating 1.189731e+4932 */
    val[0] = 0x7ffeffff;
    val[1] = 0xffffffff;
    val[2] = HUGE_NUM2;
    val[3] = HUGE_NUM3;
    goto const_quad_val;
#endif
  default:
    return 0; /* caller must check */
  }

const_int_val:
  ast = mk_cval1(val[0], DT_INT4);
  return ast;

const_int8_val:
  tmp = getcon(val, DT_INT8);
  ast = mk_cval1(tmp, DT_INT8);
  return ast;

const_real_val:
  ast = mk_cval1(val[0], DT_REAL4);
  sptr = A_SPTRG(ast);
  /* just added? */
  if (NMPTRG(sptr) == 0 && (XBIT(49, 0x400000) || XBIT(51, 0x40)))
    NMPTRP(sptr, putsname(sname, strlen(sname)));
  return ast;

const_dble_val:
  tmp = getcon(val, DT_REAL8);
  ast = mk_cnst(tmp);
  sptr = A_SPTRG(ast);
  /* just added? */
  if (NMPTRG(sptr) == 0 && (XBIT(49, 0x400000) || XBIT(51, 0x40)))
    NMPTRP(sptr, putsname(sname, strlen(sname)));
  return ast;

#ifdef TARGET_SUPPORTS_QUADFP
const_quad_val:
  tmp = getcon(val, DT_QUAD);
  ast = mk_cnst(tmp);
  sptr = A_SPTRG(ast);
  if (NMPTRG(sptr) == 0 && (XBIT(49, 0x400000) || XBIT(51, 0x40)))
    NMPTRP(sptr, putsname(sname, strlen(sname)));
  return ast;
#endif
}

/* utility function to ensure that an expression has type dt_needed.
 * If expression needs to be converted, the 'int' intrinsic is used.
 */
static int
mk_int(int expr, DTYPE dt_needed)
{
  DTYPE dt;
  int inp;

  inp = expr;
  if (A_TYPEG(inp) == A_CONV)
    inp = A_LOPG(inp);
  dt = DDTG(A_DTYPEG(inp));
  if (dt != dt_needed) {
    if (A_TYPEG(inp) == A_CNST) {
      int new;
      new = convert_cnst(inp, dt_needed);
      if (new != inp)
        return new;
    }
    expr = ast_intr(I_INT, dt_needed, 1, inp);
  }
  return expr;
}

/** \brief Utility function to ensure that an expression has type DT_INT
           (default integer).
 */
int
mk_default_int(int expr)
{
  return mk_int(expr, DT_INT);
}

/** \brief Utility function to ensure that an expression has a type suitable for
           array bounds, DT_INT8 for -Mlarge_arrays, DT_INT otherwise.
 */
int
mk_bnd_int(int expr)
{
  return mk_int(expr, astb.bnd.dtype);
}

int
mk_smallest_val(DTYPE dtype)
{
  INT val[4];
  int tmp;

  switch (DTYG(dtype)) {
  case TY_BINT:
    val[0] = ~0x7f;
    if (XBIT(51, 0x1))
      val[0] |= 0x01;
    break;
  case TY_SINT:
    val[0] = ~0x7fff;
    if (XBIT(51, 0x2))
      val[0] |= 0x0001;
    break;
  case TY_INT:
    val[0] = ~0x7fffffff;
    if (XBIT(51, 0x4))
      val[0] |= 0x00000001;
    break;
  case TY_INT8:
    if (XBIT(49, 0x1040000)) {
      /* T3D/T3E or C90 Cray targets - workaround for cray compiler:
       * -9223372036854775808_8 (-huge()-1) is considered to be out of
       * range; just return -huge().
       */
      tmp = _huge(DT_INT8);
      tmp = mk_unop(OP_SUB, tmp, dtype);
      return tmp;
    }
    val[0] = ~0x7fffffff;
    val[1] = 0;
    if (XBIT(51, 0x8))
      val[1] |= 0x00000001;
    tmp = getcon(val, DT_INT8);
    return (mk_cval1(tmp, DT_INT8));
  case TY_REAL:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
    tmp = _huge(dtype);
    tmp = mk_unop(OP_SUB, tmp, dtype);
    return tmp;
  default:
    return 0; /* caller must check */
  }
  /* const_int_val */
  return (mk_cval1(val[0], DT_INT4));
}

int
mk_largest_val(DTYPE dtype)
{
  return ast_intr(I_HUGE, dtype, 0);
}

int
mk_merge(int tsource, int fsource, int mask, DTYPE resdt)
{
  int func;
  int newargt, newast;
  newargt = mk_argt(3);
  ARGT_ARG(newargt, 0) = tsource;
  ARGT_ARG(newargt, 1) = fsource;
  ARGT_ARG(newargt, 2) = mask;
  if (resdt == DT_INT8) {
    func = sym_mkfunc_nodesc(mkRteRtnNm(RTE_mergei8), DT_INT8);
  } else {
    func = sym_mkfunc_nodesc(mkRteRtnNm(RTE_mergei), DT_INT);
  }
  newast = mk_func_node(A_INTR, mk_id(func), 3, newargt);
  A_OPTYPEP(newast, I_MERGE);
  A_DTYPEP(newast, resdt);
  return newast;
}

void rw_ast_state(RW_ROUTINE, RW_FILE)
{
  int nw;

  RW_FD(astb.hshtb, astb.hshtb, 1);
  RW_SCALAR(astb.stg_avail);
  RW_SCALAR(astb.stg_cleared);
  RW_SCALAR(astb.stg_dtsize);
  RW_FD(astb.stg_base, AST, astb.stg_avail);

  RW_FD(astb.asd.hash, astb.asd.hash, 1);
  RW_SCALAR(astb.asd.stg_avail);
  RW_SCALAR(astb.asd.stg_cleared);
  RW_SCALAR(astb.asd.stg_dtsize);
  RW_FD(astb.asd.stg_base, int, astb.asd.stg_avail);

  RW_FD(astb.shd.hash, astb.shd.hash, 1);
  RW_SCALAR(astb.shd.stg_avail);
  RW_SCALAR(astb.shd.stg_cleared);
  RW_SCALAR(astb.shd.stg_dtsize);
  RW_FD(astb.shd.stg_base, SHD, astb.shd.stg_avail);

  RW_SCALAR(astb.astli.stg_avail);
  RW_SCALAR(astb.astli.stg_cleared);
  RW_SCALAR(astb.astli.stg_dtsize);
  RW_FD(astb.astli.stg_base, ASTLI, astb.astli.stg_avail);

  RW_SCALAR(astb.argt.stg_avail);
  RW_SCALAR(astb.argt.stg_cleared);
  RW_SCALAR(astb.argt.stg_dtsize);
  RW_FD(astb.argt.stg_base, int, astb.argt.stg_avail);

  RW_SCALAR(astb.comstr.stg_avail);
  RW_SCALAR(astb.comstr.stg_cleared);
  RW_SCALAR(astb.comstr.stg_dtsize);
  RW_FD(astb.comstr.stg_base, char, astb.comstr.stg_avail);

}

/*
 * remove std from link list of stds
 * On the other hand, if it is the ENTSTD of any entry, change to A_CONTINUE
 */
void
delete_stmt(int std)
{
  int entry;
  for (entry = gbl.entries; entry > NOSYM; entry = SYMLKG(entry)) {
    if (ENTSTDG(entry) == std) {
      /* change to A_CONTINUE instead */
      if (A_TYPEG(STD_AST(std)) != A_CONTINUE) {
        STD_AST(std) = mk_stmt(A_CONTINUE, 0);
      }
      return;
    }
  }
  if (STD_PTASGN(std)) {
    STD_AST(std) = mk_stmt(A_CONTINUE, 0);
    return;
  }

  remove_stmt(std);
  STD_DELETE(std) = 1;
  STD_LINENO(std) = -1;
  STD_FINDEX(std) = 1;
}

int
add_nullify_ast(int sptrast)
{
  int sptr;
  int ast;

  sptr = intast_sym[I_NULLIFY];
  ast = begin_call(A_ICALL, sptr, 1);
  A_OPTYPEP(ast, I_NULLIFY);
  add_arg(sptrast);
  return ast;
}

/** \brief Looks for an assumed shape expression in an AST.
    \param ast is the AST expression that we're examining.
*/
int
has_assumshp_expr(int ast)
{
  int sptr, rslt, i;
  switch (A_TYPEG(ast)) {
  case A_CONV:
    return has_assumshp_expr(A_LOPG(ast));
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return has_assumshp_expr(ARGT_ARG(i, 0));
    }
    break;
  case A_CNST:
    return 0;
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
    sptr = A_SPTRG(ast);
    if (DTY(DTYPEG(sptr)) != TY_ARRAY)
      return 0;
    return ASSUMSHPG(sptr);
  case A_SUBSCR:
  case A_SUBSTR:
    return has_assumshp_expr(A_LOPG(ast));
  case A_MEM:
    rslt = has_assumshp_expr(A_MEMG(ast));
    if (!rslt) {
      ast = A_PARENTG(ast);
      rslt = has_assumshp_expr(ast);
    }
    return rslt;
  case A_UNOP:
    return has_assumshp_expr(A_LOPG(ast));
  case A_BINOP:
    rslt = has_assumshp_expr(A_LOPG(ast));
    if (!rslt)
      rslt = has_assumshp_expr(A_ROPG(ast));
    return rslt;
  default:
    interr("has_assumshp_expr: unexpected ast type", A_TYPEG(ast), 3);
  }
  return 0;
}

/** \brief Looks for an adjustable array expression in an AST.
    \param ast is the AST expression that we're examining.
*/
int
has_adjustable_expr(int ast)
{
  int sptr, rslt, i;
  switch (A_TYPEG(ast)) {
  case A_CONV:
    return has_adjustable_expr(A_LOPG(ast));
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return has_adjustable_expr(ARGT_ARG(i, 0));
    }
    break;
  case A_CNST:
    return 0;
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
    sptr = A_SPTRG(ast);
    if (DTY(DTYPEG(sptr)) != TY_ARRAY)
      return 0;
    return ADJARRG(sptr);
  case A_SUBSCR:
  case A_SUBSTR:
    return has_adjustable_expr(A_LOPG(ast));
  case A_MEM:
    rslt = has_adjustable_expr(A_MEMG(ast));
    if (!rslt) {
      ast = A_PARENTG(ast);
      rslt = has_adjustable_expr(ast);
    }
    return rslt;
  case A_UNOP:
    return has_adjustable_expr(A_LOPG(ast));
  case A_BINOP:
    rslt = has_adjustable_expr(A_LOPG(ast));
    if (!rslt)
      rslt = has_adjustable_expr(A_ROPG(ast));
    return rslt;
  default:
    interr("has_adjustable_expr: unexpected ast type", A_TYPEG(ast), 3);
  }
  return 0;
}

/** \brief Looks for a pointer expression in an AST.
    \param ast is the AST expression that we're examining.
*/
int
has_pointer_expr(int ast)
{
  int sptr, rslt, i;
  switch (A_TYPEG(ast)) {
  case A_CONV:
    return has_pointer_expr(A_LOPG(ast));
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return has_pointer_expr(ARGT_ARG(i, 0));
    }
    break;
  case A_CNST:
    return 0;
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
    sptr = A_SPTRG(ast);
    return POINTERG(sptr);
  case A_SUBSCR:
  case A_SUBSTR:
    return has_pointer_expr(A_LOPG(ast));
  case A_MEM:
    rslt = has_pointer_expr(A_MEMG(ast));
    if (!rslt) {
      ast = A_PARENTG(ast);
      rslt = has_pointer_expr(ast);
    }
    return rslt;
  case A_UNOP:
    return has_pointer_expr(A_LOPG(ast));
  case A_BINOP:
    rslt = has_pointer_expr(A_LOPG(ast));
    if (!rslt)
      rslt = has_pointer_expr(A_ROPG(ast));
    return rslt;
  default:
    interr("has_pointer_expr: unexpected ast type", A_TYPEG(ast), 3);
  }
  return 0;
}

/** \brief Looks for an allocatable expression in an AST.
    \param ast is the AST expression that we're examining.
*/
int
has_allocatable_expr(int ast)
{
  int sptr, rslt, i;
  switch (A_TYPEG(ast)) {
  case A_CONV:
    return has_allocatable_expr(A_LOPG(ast));
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_INT1:
    case I_INT2:
    case I_INT4:
    case I_INT8:
    case I_INT:
      i = A_ARGSG(ast);
      return has_allocatable_expr(ARGT_ARG(i, 0));
    }
    break;
  case A_CNST:
    return 0;
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
    sptr = A_SPTRG(ast);
    return ALLOCATTRG(sptr);
  case A_SUBSCR:
  case A_SUBSTR:
    return has_allocatable_expr(A_LOPG(ast));
  case A_MEM:
    rslt = has_allocatable_expr(A_MEMG(ast));
    if (!rslt) {
      ast = A_PARENTG(ast);
      rslt = has_allocatable_expr(ast);
    }
    return rslt;
  case A_UNOP:
    return has_allocatable_expr(A_LOPG(ast));
  case A_BINOP:
    rslt = has_allocatable_expr(A_LOPG(ast));
    if (!rslt)
      rslt = has_allocatable_expr(A_ROPG(ast));
    return rslt;
  default:
    interr("has_allocatable_expr: unexpected ast type", A_TYPEG(ast), 3);
  }
  return 0;
}

/** \brief Check if the derived type tag is the iso_c_binding: c_ptr or
   c_funptr.
           These types are compatible with pointers.
    \return true if this AST is an intrinsic call to c_loc or c_funcloc
*/
int
is_iso_cloc(int ast)
{
  return is_iso_c_loc(ast) || is_iso_c_funloc(ast);
}

/** \brief Check if this AST is an intrinsic call to c_loc. */
int
is_iso_c_loc(int ast)
{
  return A_TYPEG(ast) == A_INTR && A_OPTYPEG(ast) == I_C_LOC;
}

/** \brief Check if this AST is an intrinsic call to c_funloc. */
int
is_iso_c_funloc(int ast)
{
  return A_TYPEG(ast) == A_INTR && A_OPTYPEG(ast) == I_C_FUNLOC;
}

/** \brief Find the symbol table entry of pointer variable from an ast
           representing a pointer object.
 */
int
find_pointer_variable(int ast)
{
  switch (A_TYPEG(ast)) {
  case A_ID:
    return (A_SPTRG(ast));
  case A_MEM:
    ast = A_MEMG(ast);
    if (A_TYPEG(ast) == A_ID)
      return (A_SPTRG(ast));
    FLANG_FALLTHROUGH;
  default:
    break;
  }
  return 0;
}

/** \brief Find the symbol table entry of the target from an ast representing
           the target in a pointer assignment.
 */
void
find_pointer_target(int ast, int *pbase, int *psym)
{
  int base, sym;

  sym = base = 0;
again:
  switch (A_TYPEG(ast)) {
  case A_ID:
    base = A_SPTRG(ast);
    break;
  case A_FUNC:
  case A_SUBSCR:
  case A_SUBSTR:
    ast = A_LOPG(ast);
    goto again;
  case A_MEM:
    if (sym == 0)
      sym = A_SPTRG(A_MEMG(ast));
    ast = A_PARENTG(ast);
    goto again;
  default:
    break;
  }
  if (STYPEG(base) == ST_ENTRY && FVALG(base)) {
    base = FVALG(base);
  }
  if (sym == 0)
    sym = base;
  *pbase = base;
  *psym = sym;
}

/** \brief Convert a hollerith constant to a numeric value.
    \param cp  character pointer to hollerith character string
    \param num result of conversion of hollerith to numeric
    \param bc  byte count of destination area i.e. *1, *2, *4, *8 or *16
 */
void
holtonum(char *cp, INT *num, int bc)
{
  unsigned char *p, buf[18];
  int sc, i;
  int lc;

  /*
   * There are 4 32-bit parcels.  Index 'i' starts at the parcel to begin
   * filling and moves upward.  For example, for a 8 byte quantity 'i' would
   * start at 2 and end at 3 thus the last two words of 'num' array contain
   * the 64-bit number.
   */
  num[0] = num[1] = num[2] = num[3] = 0;
  sprintf((char *)buf, "%-17.17s", cp); /* Need 1 xtra char to detect trunc */
  p = buf;
  /* Select the initial parcel based on size of destination area */
  i = 3;
  if (bc > 4)
    i = 2;
  if (bc > 8)
    i = 0;
  if (flg.endian) {
    /*
     * The big endian byte order simply shifts each new character left 8
     * bits FEWER than the previous shifted character producing the order
     * ABCDEF...
     */
    while (i <= 3) {
      sc = (bc < 4) ? bc : 4; /* Initial shift count */
      while (sc--)
        num[i] |= *p++ << (sc * 8);
      i++;
    }
  } else {
    /*
     * The little endian byte order simply shifts each new character left 8
     * bits MORE than the previous shifted character producing the order
     * ...FEDCBA
     */
    while (i <= 3) {
      sc = (bc < 4) ? bc : 4; /* Initial shift count */
      lc = sc - 1;
      while (sc--)
        num[i] |= *p++ << ((lc - sc) * 8);
      i++;
    }
  }

  if (*p != '\0' && *p != ' ')
    errwarn(24);
}

INT
negate_const(INT conval, DTYPE dtype)
{
  SNGL result, realrs, imagrs;
  DBLE dresult, drealrs, dimagrs;
#ifdef TARGET_SUPPORTS_QUADFP
  IEEE128 qresult, qrealrs, qimagrs;
#endif
  static INT num[4];

  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
    return (-conval);

  case TY_INT8:
  case TY_LOG8:
    return const_fold(OP_SUB, (INT)stb.k0, conval, dtype);

  case TY_REAL:
    xfneg(conval, &result);
    return (result);

  case TY_DBLE:
    num[0] = CONVAL1G(conval);
    num[1] = CONVAL2G(conval);
    xdneg(num, dresult);
    return getcon(dresult, DT_REAL8);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    num[0] = CONVAL1G(conval);
    num[1] = CONVAL2G(conval);
    num[2] = CONVAL3G(conval);
    num[3] = CONVAL4G(conval);
    xqneg(num, qresult);
    return getcon(qresult, DT_QUAD);
#endif

  case TY_CMPLX:
    xfneg(CONVAL1G(conval), &realrs);
    xfneg(CONVAL2G(conval), &imagrs);
    num[0] = realrs;
    num[1] = imagrs;
    return getcon(num, DT_CMPLX8);

  case TY_DCMPLX:
    dresult[0] = CONVAL1G(CONVAL1G(conval));
    dresult[1] = CONVAL2G(CONVAL1G(conval));
    xdneg(dresult, drealrs);
    dresult[0] = CONVAL1G(CONVAL2G(conval));
    dresult[1] = CONVAL2G(CONVAL2G(conval));
    xdneg(dresult, dimagrs);
    num[0] = getcon(drealrs, DT_REAL8);
    num[1] = getcon(dimagrs, DT_REAL8);
    return getcon(num, DT_CMPLX16);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    qresult[0] = CONVAL1G(CONVAL1G(conval));
    qresult[1] = CONVAL2G(CONVAL1G(conval));
    qresult[2] = CONVAL3G(CONVAL1G(conval));
    qresult[3] = CONVAL4G(CONVAL1G(conval));
    xqneg(qresult, qrealrs);
    qresult[0] = CONVAL1G(CONVAL2G(conval));
    qresult[1] = CONVAL2G(CONVAL2G(conval));
    qresult[2] = CONVAL3G(CONVAL2G(conval));
    qresult[3] = CONVAL4G(CONVAL2G(conval));
    xqneg(qresult, qimagrs);
    num[0] = getcon(qrealrs, DT_QUAD);
    num[1] = getcon(qimagrs, DT_QUAD);
    return getcon(num, DT_QCMPLX);
#endif

  default:
    interr("negate_const: bad dtype", dtype, 3);
    return (0);
  }
}

INT
const_fold(int opr, INT conval1, INT conval2, DTYPE dtype)
{
#ifdef TARGET_SUPPORTS_QUADFP
  IEEE128 qtemp, qresult, qnum1, qnum2;
  IEEE128 qreal1, qreal2, qrealrs, qimag1, qimag2, qimagrs;
  IEEE128 qtemp1, qtemp2;
#endif
  DBLE dtemp, dresult, num1, num2;
  DBLE dreal1, dreal2, drealrs, dimag1, dimag2, dimagrs;
  DBLE dtemp1, dtemp2;
  SNGL temp, result;
  SNGL real1, real2, realrs, imag1, imag2, imagrs;
  SNGL temp1;
  UINT val1, val2;
  DBLINT64 inum1, inum2, ires;
  int cvlen1, cvlen2, urs;
  char *p, *q;

  switch (DTY(dtype)) {
  case TY_WORD:
    if (opr != OP_CMP) {
      error(33, 3, gbl.lineno, " ", CNULL);
      return (0);
    }
    return (xucmp((UINT)conval1, (UINT)conval2));

  case TY_DWORD:
    /* only comparisons in 64-bits allowed */
    if (opr != OP_CMP) {
      error(33, 3, gbl.lineno, " ", CNULL);
      return (0);
    }
    val1 = (UINT)CONVAL1G(conval1);
    val2 = (UINT)CONVAL2G(conval2);
    urs = xucmp(val1, val2);
    if (urs == 0) {
      /* 1st words are equal, compare 2nd words */
      return (xucmp((UINT)CONVAL1G(conval1), (UINT)CONVAL2G(conval2)));
    }
    return (urs);
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    switch (opr) {
    case OP_ADD:
      return conval1 + conval2;
    case OP_CMP:
      /*
       *  the following doesn't work 'cause it could exceed the
       *  range of an int:
       *  return (conval1 - conval2);
       */
      if (conval1 < conval2)
        return (INT)-1;
      if (conval1 > conval2)
        return (INT)1;
      return (INT)0;
    case OP_SUB:
      return conval1 - conval2;
    case OP_MUL:
      return conval1 * conval2;
    case OP_DIV:
      if (conval2 == 0) {
        errsev(98);
        conval2 = 1;
      }
      return conval1 / conval2;
    case OP_XTOI:
      /*
       * we get here if we're trying to init a x**k in an array constructor
       * where x is the constant and k is the iterator; the actual evaluation
       * will occur in the backend
       */
      return 0;
    }
    break;

  case TY_INT8:
    inum1[0] = CONVAL1G(conval1);
    inum1[1] = CONVAL2G(conval1);
    inum2[0] = CONVAL1G(conval2);
    inum2[1] = CONVAL2G(conval2);
    switch (opr) {
    case OP_ADD:
      add64(inum1, inum2, ires);
      break;
    case OP_CMP:
      /*
       *  the following doesn't work 'cause it could exceed the
       *  range of an int:
       *  return (conval1 - conval2);
       */
      return cmp64(inum1, inum2);
    case OP_SUB:
      sub64(inum1, inum2, ires);
      break;
    case OP_MUL:
      mul64(inum1, inum2, ires);
      break;
    case OP_DIV:
      if (inum2[0] == 0 && inum2[1] == 0) {
        errsev(98);
        inum2[1] = 1;
      }
      div64(inum1, inum2, ires);
      break;
    case OP_XTOI:
      /*
       * we get here if we're trying to init a x**k in an array constructor
       * where x is the constant and k is the iterator; the actual evaluation
       * will occur in the backend
       */
      ires[0] = ires[1] = 0;
      break;
    }
    return getcon(ires, DT_INT8);

  case TY_REAL:
    switch (opr) {
    case OP_ADD:
      xfadd(conval1, conval2, &result);
      return result;
    case OP_SUB:
      xfsub(conval1, conval2, &result);
      return result;
    case OP_MUL:
      xfmul(conval1, conval2, &result);
      return result;
    case OP_DIV:
      result = _fdiv(conval1, conval2);
      return result;
    case OP_CMP:
      return xfcmp(conval1, conval2);
    case OP_XTOI:
    case OP_XTOX:
      xfpow(conval1, conval2, &result);
      return result;
    }
    break;

  case TY_DBLE:
    num1[0] = CONVAL1G(conval1);
    num1[1] = CONVAL2G(conval1);
    num2[0] = CONVAL1G(conval2);
    num2[1] = CONVAL2G(conval2);
    switch (opr) {
    case OP_ADD:
      xdadd(num1, num2, dresult);
      break;
    case OP_SUB:
      xdsub(num1, num2, dresult);
      break;
    case OP_MUL:
      xdmul(num1, num2, dresult);
      break;
    case OP_DIV:
      _ddiv(num1, num2, dresult);
      break;
    case OP_CMP:
      return xdcmp(num1, num2);
    case OP_XTOI:
    case OP_XTOX:
      xdpow(num1, num2, dresult);
      break;
    default:
      goto err_exit;
    }
    return getcon(dresult, DT_REAL8);

#ifdef TARGET_SUPPORTS_QUADFP
  /* support quad precision const fold */
  case TY_QUAD:
    qnum1[0] = CONVAL1G(conval1);
    qnum1[1] = CONVAL2G(conval1);
    qnum1[2] = CONVAL3G(conval1);
    qnum1[3] = CONVAL4G(conval1);
    qnum2[0] = CONVAL1G(conval2);
    qnum2[1] = CONVAL2G(conval2);
    qnum2[2] = CONVAL3G(conval2);
    qnum2[3] = CONVAL4G(conval2);
    switch (opr) {
    case OP_ADD:
      xqadd(qnum1, qnum2, qresult);
      break;
    case OP_SUB:
      xqsub(qnum1, qnum2, qresult);
      break;
    case OP_MUL:
      xqmul(qnum1, qnum2, qresult);
      break;
    case OP_DIV:
      xqdiv(qnum1, qnum2, qresult);
      break;
    case OP_CMP:
      return xqcmp(qnum1, qnum2);
    case OP_XTOI:
    case OP_XTOX:
      xqpow(qnum1, qnum2, qresult);
      break;
    default:
      goto err_exit;
    }
    return getcon(qresult, DT_QUAD);
#endif

  case TY_CMPLX:
    real1 = CONVAL1G(conval1);
    imag1 = CONVAL2G(conval1);
    real2 = CONVAL1G(conval2);
    imag2 = CONVAL2G(conval2);
    switch (opr) {
    case OP_ADD:
      xfadd(real1, real2, &realrs);
      xfadd(imag1, imag2, &imagrs);
      break;
    case OP_SUB:
      xfsub(real1, real2, &realrs);
      xfsub(imag1, imag2, &imagrs);
      break;
    case OP_MUL:
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      xfmul(real1, real2, &temp1);
      xfmul(imag1, imag2, &temp);
      xfsub(temp1, temp, &realrs);
      xfmul(real1, imag2, &temp1);
      xfmul(real2, imag1, &temp);
      xfadd(temp1, temp, &imagrs);
      break;
    case OP_DIV:
      /*
       *  realrs = real2;
       *  if (realrs < 0)
       *      realrs = -realrs;
       *  imagrs = imag2;
       *  if (imagrs < 0)
       *      imagrs = -imagrs;
       */
      if (xfcmp(real2, CONVAL2G(stb.flt0)) < 0)
        xfsub(CONVAL2G(stb.flt0), real2, &realrs);
      else
        realrs = real2;

      if (xfcmp(imag2, CONVAL2G(stb.flt0)) < 0)
        xfsub(CONVAL2G(stb.flt0), imag2, &imagrs);
      else
        imagrs = imag2;

      /* avoid overflow */

      if (xfcmp(realrs, imagrs) <= 0) {
        /*
         *  if (realrs <= imagrs) {
         *      temp = real2 / imag2;
         *      temp1 = 1.0f / (imag2 * (1 + temp * temp));
         *      realrs = (real1 * temp + imag1) * temp1;
         *      imagrs = (imag1 * temp - real1) * temp1;
         *  }
         */
        temp = _fdiv(real2, imag2);

        xfmul(temp, temp, &temp1);
        xfadd(CONVAL2G(stb.flt1), temp1, &temp1);
        xfmul(imag2, temp1, &temp1);
        temp1 = _fdiv(CONVAL2G(stb.flt1), temp1);

        xfmul(real1, temp, &realrs);
        xfadd(realrs, imag1, &realrs);
        xfmul(realrs, temp1, &realrs);

        xfmul(imag1, temp, &imagrs);
        xfsub(imagrs, real1, &imagrs);
        xfmul(imagrs, temp1, &imagrs);
      } else {
        /*
         *  else {
         *      temp = imag2 / real2;
         *      temp1 = 1.0f / (real2 * (1 + temp * temp));
         *      realrs = (real1 + imag1 * temp) * temp1;
         *      imagrs = (imag1 - real1 * temp) * temp1;
         *  }
         */
        temp = _fdiv(imag2, real2);

        xfmul(temp, temp, &temp1);
        xfadd(CONVAL2G(stb.flt1), temp1, &temp1);
        xfmul(real2, temp1, &temp1);
        temp1 = _fdiv(CONVAL2G(stb.flt1), temp1);

        xfmul(imag1, temp, &realrs);
        xfadd(real1, realrs, &realrs);
        xfmul(realrs, temp1, &realrs);

        xfmul(real1, temp, &imagrs);
        xfsub(imag1, imagrs, &imagrs);
        xfmul(imagrs, temp1, &imagrs);
      }
      break;
    case OP_CMP:
      /*
       * for complex, only EQ and NE comparisons are allowed, so return
       * 0 if the two constants are the same, else 1:
       */
      return (conval1 != conval2);
    case OP_XTOX:
      xcfpow(real1, imag1, real2, imag2, &realrs, &imagrs);
      break;
    default:
      goto err_exit;
    }
    num1[0] = realrs;
    num1[1] = imagrs;
    return getcon(num1, DT_CMPLX8);

  case TY_DCMPLX:
    dreal1[0] = CONVAL1G(CONVAL1G(conval1));
    dreal1[1] = CONVAL2G(CONVAL1G(conval1));
    dimag1[0] = CONVAL1G(CONVAL2G(conval1));
    dimag1[1] = CONVAL2G(CONVAL2G(conval1));
    dreal2[0] = CONVAL1G(CONVAL1G(conval2));
    dreal2[1] = CONVAL2G(CONVAL1G(conval2));
    dimag2[0] = CONVAL1G(CONVAL2G(conval2));
    dimag2[1] = CONVAL2G(CONVAL2G(conval2));
    switch (opr) {
    case OP_ADD:
      xdadd(dreal1, dreal2, drealrs);
      xdadd(dimag1, dimag2, dimagrs);
      break;
    case OP_SUB:
      xdsub(dreal1, dreal2, drealrs);
      xdsub(dimag1, dimag2, dimagrs);
      break;
    case OP_MUL:
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      xdmul(dreal1, dreal2, dtemp1);
      xdmul(dimag1, dimag2, dtemp);
      xdsub(dtemp1, dtemp, drealrs);
      xdmul(dreal1, dimag2, dtemp1);
      xdmul(dreal2, dimag1, dtemp);
      xdadd(dtemp1, dtemp, dimagrs);
      break;
    case OP_DIV:
      dtemp2[0] = CONVAL1G(stb.dbl0);
      dtemp2[1] = CONVAL2G(stb.dbl0);
      /*  drealrs = dreal2;
       *  if (drealrs < 0)
       *      drealrs = -drealrs;
       *  dimagrs = dimag2;
       *  if (dimagrs < 0)
       *      dimagrs = -dimagrs;
       */
      if (xdcmp(dreal2, dtemp2) < 0)
        xdsub(dtemp2, dreal2, drealrs);
      else {
        drealrs[0] = dreal2[0];
        drealrs[1] = dreal2[1];
      }
      if (xdcmp(dimag2, dtemp2) < 0)
        xdsub(dtemp2, dimag2, dimagrs);
      else {
        dimagrs[0] = dimag2[0];
        dimagrs[1] = dimag2[1];
      }

      /* avoid overflow */

      dtemp2[0] = CONVAL1G(stb.dbl1);
      dtemp2[1] = CONVAL2G(stb.dbl1);
      if (xdcmp(drealrs, dimagrs) <= 0) {
        /*  if (drealrs <= dimagrs) {
         *     dtemp = dreal2 / dimag2;
         *     dtemp1 = 1.0 / (dimag2 * (1 + dtemp * dtemp));
         *     drealrs = (dreal1 * dtemp + dimag1) * dtemp1;
         *     dimagrs = (dimag1 * dtemp - dreal1) * dtemp1;
         *  }
         */
        _ddiv(dreal2, dimag2, dtemp);

        xdmul(dtemp, dtemp, dtemp1);
        xdadd(dtemp2, dtemp1, dtemp1);
        xdmul(dimag2, dtemp1, dtemp1);
        _ddiv(dtemp2, dtemp1, dtemp1);

        xdmul(dreal1, dtemp, drealrs);
        xdadd(drealrs, dimag1, drealrs);
        xdmul(drealrs, dtemp1, drealrs);

        xdmul(dimag1, dtemp, dimagrs);
        xdsub(dimagrs, dreal1, dimagrs);
        xdmul(dimagrs, dtemp1, dimagrs);
      } else {
        /*  else {
         *  	dtemp = dimag2 / dreal2;
         *  	dtemp1 = 1.0 / (dreal2 * (1 + dtemp * dtemp));
         *  	drealrs = (dreal1 + dimag1 * dtemp) * dtemp1;
         *  	dimagrs = (dimag1 - dreal1 * dtemp) * dtemp1;
         *  }
         */
        _ddiv(dimag2, dreal2, dtemp);

        xdmul(dtemp, dtemp, dtemp1);
        xdadd(dtemp2, dtemp1, dtemp1);
        xdmul(dreal2, dtemp1, dtemp1);
        _ddiv(dtemp2, dtemp1, dtemp1);

        xdmul(dimag1, dtemp, drealrs);
        xdadd(dreal1, drealrs, drealrs);
        xdmul(drealrs, dtemp1, drealrs);

        xdmul(dreal1, dtemp, dimagrs);
        xdsub(dimag1, dimagrs, dimagrs);
        xdmul(dimagrs, dtemp1, dimagrs);
      }
      break;
    case OP_CMP:
      /*
       * for complex, only EQ and NE comparisons are allowed, so return
       * 0 if the two constants are the same, else 1:
       */
      return (conval1 != conval2);
    case OP_XTOX:
      xcdpow(dreal1, dimag1, dreal2, dimag2, drealrs, dimagrs);
      break;
    default:
      goto err_exit;
    }
    num1[0] = getcon(drealrs, DT_REAL8);
    num1[1] = getcon(dimagrs, DT_REAL8);
    return getcon(num1, DT_CMPLX16);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    qreal1[0] = CONVAL1G(CONVAL1G(conval1));
    qreal1[1] = CONVAL2G(CONVAL1G(conval1));
    qreal1[2] = CONVAL3G(CONVAL1G(conval1));
    qreal1[3] = CONVAL4G(CONVAL1G(conval1));
    qimag1[0] = CONVAL1G(CONVAL2G(conval1));
    qimag1[1] = CONVAL2G(CONVAL2G(conval1));
    qimag1[2] = CONVAL3G(CONVAL2G(conval1));
    qimag1[3] = CONVAL4G(CONVAL2G(conval1));
    qreal2[0] = CONVAL1G(CONVAL1G(conval2));
    qreal2[1] = CONVAL2G(CONVAL1G(conval2));
    qreal2[2] = CONVAL3G(CONVAL1G(conval2));
    qreal2[3] = CONVAL4G(CONVAL1G(conval2));
    qimag2[0] = CONVAL1G(CONVAL2G(conval2));
    qimag2[1] = CONVAL2G(CONVAL2G(conval2));
    qimag2[2] = CONVAL3G(CONVAL2G(conval2));
    qimag2[3] = CONVAL4G(CONVAL2G(conval2));
    switch (opr) {
    case OP_ADD:
      xqadd(qreal1, qreal2, qrealrs);
      xqadd(qimag1, qimag2, qimagrs);
      break;
    case OP_SUB:
      xqsub(qreal1, qreal2, qrealrs);
      xqsub(qimag1, qimag2, qimagrs);
      break;
    case OP_MUL:
      /* (a + bi) * (c + di) ==> (ac-bd) + (ad+cb)i */
      xqmul(qreal1, qreal2, qtemp1);
      xqmul(qimag1, qimag2, qtemp);
      xqsub(qtemp1, qtemp, qrealrs);
      xqmul(qreal1, qimag2, qtemp1);
      xqmul(qreal2, qimag1, qtemp);
      xqadd(qtemp1, qtemp, qimagrs);
      break;
    case OP_DIV:
      qtemp2[0] = CONVAL1G(stb.quad0);
      qtemp2[1] = CONVAL2G(stb.quad0);
      qtemp2[2] = CONVAL3G(stb.quad0);
      qtemp2[3] = CONVAL4G(stb.quad0);
      /*  qrealrs = qreal2;
       *  if (qrealrs < 0)
       *      qrealrs = -qrealrs;
       *  qimagrs = qimag2;
       *  if (qimagrs < 0)
       *      qimagrs = -qimagrs;
       */
      if (xqcmp(qreal2, qtemp2) < 0)
        xqsub(qtemp2, qreal2, qrealrs);
      else {
        qrealrs[0] = qreal2[0];
        qrealrs[1] = qreal2[1];
        qrealrs[2] = qreal2[2];
        qrealrs[3] = qreal2[3];
      }
      if (xqcmp(qimag2, qtemp2) < 0)
        xqsub(qtemp2, qimag2, qimagrs);
      else {
        qimagrs[0] = qimag2[0];
        qimagrs[1] = qimag2[1];
        qimagrs[2] = qimag2[2];
        qimagrs[3] = qimag2[3];
      }

      /* avoid overflow */

      qtemp2[0] = CONVAL1G(stb.quad1);
      qtemp2[1] = CONVAL2G(stb.quad1);
      qtemp2[2] = CONVAL3G(stb.quad1);
      qtemp2[3] = CONVAL4G(stb.quad1);
      if (xqcmp(qrealrs, qimagrs) <= 0) {
        /*  if (qrealrs <= qimagrs) {
         *     qtemp = qreal2 / qimag2;
         *     qtemp1 = 1.0 / (qimag2 * (1 + qtemp * qtemp));
         *     qrealrs = (qreal1 * qtemp + qimag1) * qtemp1;
         *     qimagrs = (qimag1 * qtemp - qreal1) * qtemp1;
         *  }
         */
        xqdiv(qreal2, qimag2, qtemp);

        xqmul(qtemp, qtemp, qtemp1);
        xqadd(qtemp2, qtemp1, qtemp1);
        xqmul(qimag2, qtemp1, qtemp1);
        xqdiv(qtemp2, qtemp1, qtemp1);

        xqmul(qreal1, qtemp, qrealrs);
        xqadd(qrealrs, qimag1, qrealrs);
        xqmul(qrealrs, qtemp1, qrealrs);

        xqmul(qimag1, qtemp, qimagrs);
        xqsub(qimagrs, qreal1, qimagrs);
        xqmul(qimagrs, qtemp1, qimagrs);
      } else {
        /*  else {
         *  	qtemp = qimag2 / qreal2;
         *  	qtemp1 = 1.0 / (qreal2 * (1 + qtemp * qtemp));
         *  	qrealrs = (qreal1 + qimag1 * qtemp) * qtemp1;
         *  	qimagrs = (qimag1 - qreal1 * qtemp) * qtemp1;
         *  }
         */
        xqdiv(qimag2, qreal2, qtemp);

        xqmul(qtemp, qtemp, qtemp1);
        xqadd(qtemp2, qtemp1, qtemp1);
        xqmul(qreal2, qtemp1, qtemp1);
        xqdiv(qtemp2, qtemp1, qtemp1);

        xqmul(qimag1, qtemp, qrealrs);
        xqadd(qreal1, qrealrs, qrealrs);
        xqmul(qrealrs, qtemp1, qrealrs);

        xqmul(qreal1, qtemp, qimagrs);
        xqsub(qimag1, qimagrs, qimagrs);
        xqmul(qimagrs, qtemp1, qimagrs);
      }
      break;
    case OP_CMP:
      /*
       * for complex, only EQ and NE comparisons are allowed, so return
       * 0 if the two constants are the same, else 1:
       */
      return (conval1 != conval2);
    case OP_XTOX:
      xcqpow(qreal1, qimag1, qreal2, qimag2, qrealrs, qimagrs);
      break;
    default:
      goto err_exit;
    }
    num1[0] = getcon(qrealrs, DT_QUAD);
    num1[1] = getcon(qimagrs, DT_QUAD);
    return getcon(num1, DT_QCMPLX);
#endif

  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
    if (opr != OP_CMP) {
      errsev(91);
      return 0;
    }
    /*
     * opr is assumed to be OP_CMP, only EQ and NE comparisons are
     * allowed so just return 0 if eq, else 1:
     */
    return (conval1 != conval2);

  case TY_NCHAR:
    if (opr != OP_CMP) {
      errsev(91);
      return 0;
    }
#define KANJI_BLANK 0xA1A1
    {
      int bytes, val1, val2;
      /* following if condition prevent seg fault from following example;
       * logical,parameter ::b=char(32,kind=2).eq.char(45,kind=2)
       */
      if (CONVAL1G(conval1) > stb.stg_avail ||
          CONVAL1G(conval2) > stb.stg_avail) {
        errsev(91);
        return 0;
      }
      cvlen1 = string_length(DTYPEG(CONVAL1G(conval1)));
      cvlen2 = string_length(DTYPEG(CONVAL1G(conval2)));
      p = stb.n_base + CONVAL1G(CONVAL1G(conval1));
      q = stb.n_base + CONVAL1G(CONVAL1G(conval2));

      while (cvlen1 > 0 && cvlen2 > 0) {
        val1 = kanji_char((unsigned char *)p, cvlen1, &bytes);
        p += bytes, cvlen1 -= bytes;
        val2 = kanji_char((unsigned char *)q, cvlen2, &bytes);
        q += bytes, cvlen2 -= bytes;
        if (val1 != val2)
          return (val1 - val2);
      }

      while (cvlen1 > 0) {
        val1 = kanji_char((unsigned char *)p, cvlen1, &bytes);
        p += bytes, cvlen1 -= bytes;
        if (val1 != KANJI_BLANK)
          return (val1 - KANJI_BLANK);
      }

      while (cvlen2 > 0) {
        val2 = kanji_char((unsigned char *)q, cvlen2, &bytes);
        q += bytes, cvlen2 -= bytes;
        if (val2 != KANJI_BLANK)
          return (KANJI_BLANK - val2);
      }
    }
    return 0;

  case TY_CHAR:
    if (opr != OP_CMP) {
      errsev(91);
      return 0;
    }
    /* opr is OP_CMP, return -1, 0, or 1:  */
    cvlen1 = string_length(DTYPEG(conval1));
    cvlen2 = string_length(DTYPEG(conval2));
    if (cvlen1 == 0 || cvlen2 == 0) {
      return cvlen1 - cvlen2;
    }
    /* change the shorter string to be of same length as the longer: */
    if (cvlen1 < cvlen2) {
      conval1 = cngcon(conval1, (int)DTYPEG(conval1), (int)DTYPEG(conval2));
      cvlen1 = cvlen2;
    } else
      conval2 = cngcon(conval2, (int)DTYPEG(conval2), (int)DTYPEG(conval1));

    p = stb.n_base + CONVAL1G(conval1);
    q = stb.n_base + CONVAL1G(conval2);
    do {
      if (*p != *q)
        return (*p - *q);
      ++p;
      ++q;
    } while (--cvlen1);
    return 0;
  }

err_exit:
  interr("const_fold: bad args", dtype, 3);
  return (0);
}

/** \brief Convert constant from oldtyp to newtyp.
    \return constant value for 32-bit constants, or symbol table pointer

   Issue error messages only for impossible conversions.<br>
   Can only be used for scalar constants.

   Remember: Non-decimal constants are octal, hexadecimal, or hollerith
   constants which are represented by DT_WORD, DT_DWORD and DT_HOLL.
   Non-decimal constants 'assume' data types rather than go thru a conversion.
   Hollerith constants have a data type of DT_HOLL in the semantic stack;
   the CONVAL1 field locates a constant of data type DT_CHAR and the
   CONVAL2 field indicates the kind of Hollerith ('h', 'l', or 'r').

   Hollerith constants are always treated as scalars while octal or
   hexadecimal constants can be promoted to vectors.
 */
INT
cngcon(INT oldval, int oldtyp, int newtyp)
{
  int to, from;
  char *cp;
  int newcvlen, oldcvlen, blnk;
  INT num[4], result;
  INT num1[8];
#ifdef TARGET_SUPPORTS_QUADFP
  INT num2[4];
#endif
  INT swap;
  UINT unum[4];

#define MASKH32(sptr) (CONVAL1G(sptr) & 0xFFFFFFFF)
  if (is_empty_typedef(newtyp) && oldtyp == DT_INT4) {
    /* Special case for empty typedef */
    newtyp = DT_INT4;
  }
  if (newtyp == oldtyp)
    return oldval;
  to = DTY(newtyp);
  from = DTY(oldtyp);

  if ((!TY_ISSCALAR(to) && to != TY_NUMERIC) || !TY_ISSCALAR(from))
    goto type_conv_error;

  if (F77OUTPUT) {
    if (TY_ISLOG(to) && (!TY_ISLOG(from)))
      /* "Illegal type conversion $" */
      error(432, 2, gbl.lineno, "to logical", CNULL);
    if (TY_ISLOG(from) && (!TY_ISLOG(to)))
      error(432, 2, gbl.lineno, "from logical", CNULL);
  }

  switch (to) {
  case TY_WORD:
    break;

  case TY_BLOG:
  case TY_BINT:
    /* decimal integer constants are 32-bits, BUT, PARAMETER
        may be TY_SLOG, TY_SINT, TY_BLOG, or TY_BINT.
     */
    switch (from) {
    case TY_WORD:
      if (oldval & 0xFFFFFF00)
        errwarn(15);
      return (sign_extend(oldval, 8));
    case TY_DWORD:
      result = CONVAL2G(oldval);
      if (CONVAL1G(oldval))
        errwarn(15);
      return (sign_extend(result, 8));
    case TY_INT8:
    case TY_LOG8:
      result = CONVAL2G(oldval);
      if ((((result & 0xFFFFFF80) != 0xFFFFFF80) && (result & 0xFFFFFF00)) ||
          (MASKH32(oldval) != 0 && MASKH32(oldval) != 0xFFFFFFFF))
        truncation_warning(result & 0xFF);
      return (sign_extend(result, 8));
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
      if (((oldval & 0xFFFFFF80) != 0xFFFFFF80) && (oldval & 0xFFFFFF00))
        truncation_warning(oldval & 0xFF);
      return (sign_extend(oldval, 8));
    default:
      break;
    }
    goto other_int_cases;
  case TY_SLOG:
  case TY_SINT:
    switch (from) {
    case TY_WORD:
      if (oldval & 0xFFFF0000)
        errwarn(15);
      return (sign_extend(oldval, 16));
    case TY_DWORD:
      result = CONVAL2G(oldval);
      if (CONVAL1G(oldval))
        errwarn(15);
      return (sign_extend(result, 16));
    case TY_INT8:
    case TY_LOG8:
      result = CONVAL2G(oldval);
      if ((((result & 0xFFFF8000) != 0xFFFF8000) && (result & 0xFFFF0000)) ||
          (MASKH32(oldval) != 0 && MASKH32(oldval) != 0xFFFFFFFF))
        truncation_warning(result & 0xFFFF);
      return (sign_extend(result, 16));
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
      if (((oldval & 0xFFFF8000) != 0xFFFF8000) && (oldval & 0xFFFF0000))
        truncation_warning(oldval & 0xFFFF);
      return (sign_extend(oldval, 16));
    default:
      break;
    }
    goto other_int_cases;
  case TY_LOG:
  case TY_INT:
    if (from == TY_DWORD) {
      result = CONVAL2G(oldval);
      if (CONVAL1G(oldval))
        errwarn(15);
      return (result);
    }
    if (from == TY_INT8) {
      result = CONVAL2G(oldval);
      if (MASKH32(oldval) != 0 && (MASKH32(oldval) != 0xFFFFFFFF))
        truncation_warning(CONVAL1G(oldval));
      return sign_extend(result, 32);
    }
    if (from == TY_LOG8) {
      result = CONVAL2G(oldval);
      return sign_extend(result, 32);
    }
    if (TY_ISLOG(to) && TY_ISLOG(from))
      /* -standard removes _TY_ISINT from logical types, so explicitly
       * check for logicals.
       */
      return oldval;
    if (from == TY_WORD || TY_ISINT(from))
      return oldval;
  other_int_cases:
    switch (from) {
    case TY_CMPLX:
      oldval = CONVAL1G(oldval);
      FLANG_FALLTHROUGH;
    case TY_REAL:
      xfix(oldval, &result);
      return result;
    case TY_DCMPLX:
      oldval = CONVAL1G(oldval);
      FLANG_FALLTHROUGH;
    case TY_DBLE:
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
      xdfix(num, &result);
      return result;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      oldval = CONVAL1G(oldval);
      FLANG_FALLTHROUGH;
    case TY_QUAD:
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
      num[2] = CONVAL3G(oldval);
      num[3] = CONVAL4G(oldval);
      xqfix(num, &result);
      return result;
#endif
    case TY_HOLL:
      cp = stb.n_base + CONVAL1G(CONVAL1G(oldval));
      goto char_to_int;
    case TY_CHAR:
      if (flg.standard)
        conversion_warning();
      cp = stb.n_base + CONVAL1G(oldval);
    char_to_int:
      oldcvlen = 4;
      if (to == TY_BLOG || to == TY_BINT)
        oldcvlen = 1;
      if (to == TY_SLOG || to == TY_SINT)
        oldcvlen = 2;
      if (to == TY_LOG8 || to == TY_INT8)
        oldcvlen = 8;
      holtonum(cp, num, oldcvlen);
      return num[3];
    default: /* TY_NCHAR comes here */
      break;
    }
    break;

  case TY_LOG8:
  case TY_INT8:
    if (from == TY_DWORD || from == TY_INT8 || from == TY_LOG8) {
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
      return getcon(num, newtyp);
    } else if (from == TY_WORD) {
      unum[0] = 0;
      unum[1] = oldval;
      return getcon((INT *)unum, newtyp);
    } else if (TY_ISINT(from) || (TY_ISLOG(to) && TY_ISLOG(from))) {
      if (oldval < 0) {
        num[0] = -1;
        num[1] = oldval;
      } else {
        num[0] = 0;
        num[1] = oldval;
      }
      return getcon(num, newtyp);
    } else {
      switch (from) {
      case TY_CMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_REAL:
        xfix64(oldval, num);
        return getcon(num, newtyp);
      case TY_DCMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_DBLE:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        xdfix64(num1, num);
        return getcon(num, newtyp);
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_QUAD:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        num1[2] = CONVAL3G(oldval);
        num1[3] = CONVAL4G(oldval);
        xqfix64(num1, num);
        return getcon(num, newtyp);
#endif
      case TY_HOLL:
        cp = stb.n_base + CONVAL1G(CONVAL1G(oldval));
        goto char_to_int8;
      case TY_CHAR:
        if (flg.standard)
          conversion_warning();
        cp = stb.n_base + CONVAL1G(oldval);
      char_to_int8:
        holtonum(cp, num, 8);
        if (flg.endian == 0) {
          /* for little endian, need to swap words in each double word
           * quantity.  Order of bytes in a word is okay, but not the
           * order of words.
           */
          swap = num[2];
          num[2] = num[3];
          num[3] = swap;
        }
        return getcon(&num[2], newtyp);
      default: /* TY_NCHAR comes here */
        break;
      }
    }
    break;

  case TY_REAL:
    if (from == TY_WORD)
      return oldval;
    else if (from == TY_DWORD) {
      result = CONVAL2G(oldval);
      if (CONVAL1G(oldval))
        errwarn(15);
      return result;
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
      xflt64(num, &result);
      return result;
    } else if (TY_ISINT(from)) {
      xffloat(oldval, &result);
      return result;
    } else {
      switch (from) {
      case TY_CMPLX:
        return CONVAL1G(oldval);
      case TY_DCMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_DBLE:
        num[0] = CONVAL1G(oldval);
        num[1] = CONVAL2G(oldval);
        xsngl(num, &result);
        return result;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_QUAD:
        num[0] = CONVAL1G(oldval);
        num[1] = CONVAL2G(oldval);
        num[2] = CONVAL3G(oldval);
        num[3] = CONVAL4G(oldval);
        xqtof(num, &result);
        return result;
#endif
      case TY_HOLL:
        cp = stb.n_base + CONVAL1G(CONVAL1G(oldval));
        goto char_to_real;
      case TY_CHAR:
        if (flg.standard)
          conversion_warning();
        cp = stb.n_base + CONVAL1G(oldval);
      char_to_real:
        holtonum(cp, num, 4);
        return num[3];
      default:
        break;
      }
    }
    break;

  case TY_DBLE:
    if (from == TY_WORD) {
      num[0] = 0;
      num[1] = oldval;
    } else if (from == TY_DWORD) {
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(oldval);
      num1[1] = CONVAL2G(oldval);
      xdflt64(num1, num);
    } else if (TY_ISINT(from))
      xdfloat(oldval, num);
    else {
      switch (from) {
      case TY_DCMPLX:
        return CONVAL1G(oldval);
      case TY_CMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_REAL:
        xdble(oldval, num);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_QUAD:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        num1[2] = CONVAL3G(oldval);
        num1[3] = CONVAL4G(oldval);
        xqtod(num1, num);
        break;
#endif
      case TY_HOLL:
        cp = stb.n_base + CONVAL1G(CONVAL1G(oldval));
        goto char_to_dble;
      case TY_CHAR:
        if (flg.standard)
          conversion_warning();
        cp = stb.n_base + CONVAL1G(oldval);
      char_to_dble:
        holtonum(cp, num, 8);
        if (flg.endian == 0) {
          /* for little endian, need to swap words in each double word
           * quantity.  Order of bytes in a word is okay, but not the
           * order of words.
           */
          swap = num[2];
          num[2] = num[3];
          num[3] = swap;
        }
        return getcon(&num[2], DT_REAL8);
      default:
        errsev(91);
        return (stb.dbl0);
      }
    }
    return getcon(num, DT_REAL8);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    if (from == TY_WORD) {
      num[0] = 0;
      num[1] = 0;
      num[2] = 0;
      num[3] = oldval;
    } else if (from == TY_DWORD) {
      num[0] = 0;
      num[1] = 0;
      num[2] = CONVAL1G(oldval);
      num[3] = CONVAL2G(oldval);
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(oldval);
      num1[1] = CONVAL2G(oldval);
      xqflt64(num1, num);
    } else if (TY_ISINT(from))
      xqfloat(oldval, num);
    else {
      switch (from) {
      case TY_QCMPLX:
        return CONVAL1G(oldval);
      case TY_CMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_REAL:
        xftoq(oldval, num);
        break;
      case TY_DCMPLX:
        oldval = CONVAL1G(oldval);
        FLANG_FALLTHROUGH;
      case TY_DBLE:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        xdtoq(num1, num);
        break;
      case TY_HOLL:
        cp = stb.n_base + CONVAL1G(CONVAL1G(oldval));
        goto char_to_quad;
      case TY_CHAR:
        if (flg.standard)
          conversion_warning();
        cp = stb.n_base + CONVAL1G(oldval);
      char_to_quad:
        holtonum(cp, num, BYTE_NUMBER16);
        if (flg.endian == 0) {
          /* for little endian, need to swap words in each double word
           * quantity.  Order of bytes in a word is okay, but not the
           * order of words.
           */
          swap = num[0];
          num[0] = num[3];
          num[3] = swap;
          swap = num[1];
          num[1] = num[2];
          num[2] = swap;
        }
        return getcon(num, DT_QUAD);
      default:
        errsev(S_0091_Constant_expression_of_wrong_data_type);
        return (stb.quad0);
      }
    }
    return getcon(num, DT_QUAD);
#endif

  case TY_CMPLX:
    /*  num[0] = real part
     *  num[1] = imaginary part
     */
    num[1] = 0;
    if (from == TY_WORD) {
      /* a la VMS */
      num[0] = 0;
      num[1] = oldval;
    } else if (from == TY_DWORD) {
      /* a la VMS */
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(oldval);
      num1[1] = CONVAL2G(oldval);
      xflt64(num1, &num[0]);
    } else if (TY_ISINT(from))
      xffloat(oldval, &num[0]);
    else {
      switch (from) {
      case TY_REAL:
        num[0] = oldval;
        break;
      case TY_DBLE:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        xsngl(num1, &num[0]);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        num1[2] = CONVAL3G(oldval);
        num1[3] = CONVAL4G(oldval);
        xqtof(num1, num);
        break;
#endif
      case TY_DCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(oldval));
        num1[1] = CONVAL2G(CONVAL1G(oldval));
        xsngl(num1, &num[0]);
        num1[0] = CONVAL1G(CONVAL2G(oldval));
        num1[1] = CONVAL2G(CONVAL2G(oldval));
        xsngl(num1, &num[1]);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(oldval));
        num1[1] = CONVAL2G(CONVAL1G(oldval));
        num1[2] = CONVAL3G(CONVAL1G(oldval));
        num1[3] = CONVAL4G(CONVAL1G(oldval));
        xqtof(num1, &num[0]);
        num1[0] = CONVAL1G(CONVAL2G(oldval));
        num1[1] = CONVAL2G(CONVAL2G(oldval));
        num1[2] = CONVAL3G(CONVAL2G(oldval));
        num1[3] = CONVAL4G(CONVAL2G(oldval));
        xqtof(num1, &num[1]);
        break;
#endif
      case TY_HOLL:
        cp = stb.n_base + CONVAL1G(CONVAL1G(oldval));
        goto char_to_cmplx;
      case TY_CHAR:
        if (flg.standard)
          conversion_warning();
        cp = stb.n_base + CONVAL1G(oldval);
      char_to_cmplx:
        holtonum(cp, num, 8);
        return getcon(&num[2], DT_CMPLX8);
      default:
        num[0] = 0;
        num[1] = 0;
        errsev(91);
      }
    }
    return getcon(num, DT_CMPLX8);

  case TY_DCMPLX:
    if (from == TY_WORD) {
      num[0] = 0;
      num[1] = oldval;
      num[0] = getcon(num, DT_REAL8);
      num[1] = stb.dbl0;
    } else if (from == TY_DWORD) {
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
      num[0] = getcon(num, DT_REAL8);
      num[1] = stb.dbl0;
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(oldval);
      num1[1] = CONVAL2G(oldval);
      xdflt64(num1, num);
      num[0] = getcon(num, DT_REAL8);
      num[1] = stb.dbl0;
    } else if (TY_ISINT(from)) {
      xdfloat(oldval, num);
      num[0] = getcon(num, DT_REAL8);
      num[1] = stb.dbl0;
    } else {
      switch (from) {
      case TY_REAL:
        xdble(oldval, num);
        num[0] = getcon(num, DT_REAL8);
        num[1] = stb.dbl0;
        break;
      case TY_DBLE:
        num[0] = oldval;
        num[1] = stb.dbl0;
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        num1[2] = CONVAL3G(oldval);
        num1[3] = CONVAL4G(oldval);
        xqtod(num1, num);
        num[0] = getcon(num, DT_REAL8);
        num[1] = stb.dbl0;
	break;
#endif
      case TY_CMPLX:
        xdble(CONVAL1G(oldval), num1);
        num[0] = getcon(num1, DT_REAL8);
        xdble(CONVAL2G(oldval), num1);
        num[1] = getcon(num1, DT_REAL8);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(oldval));
        num1[1] = CONVAL2G(CONVAL1G(oldval));
        num1[2] = CONVAL3G(CONVAL1G(oldval));
        num1[3] = CONVAL4G(CONVAL1G(oldval));
        xqtod(num1, num2);
        num[0] = getcon(num2, DT_REAL8);
        num1[0] = CONVAL1G(CONVAL2G(oldval));
        num1[1] = CONVAL2G(CONVAL2G(oldval));
        num1[2] = CONVAL3G(CONVAL2G(oldval));
        num1[3] = CONVAL4G(CONVAL2G(oldval));
        xqtod(num1, num2);
        num[1] = getcon(num2, DT_REAL8);
        break;
#endif
      case TY_HOLL:
        cp = stb.n_base + CONVAL1G(CONVAL1G(oldval));
        goto char_to_dcmplx;
      case TY_CHAR:
        if (flg.standard)
          conversion_warning();
        cp = stb.n_base + CONVAL1G(oldval);
      char_to_dcmplx:
        holtonum(cp, num1, 16);
        if (flg.endian == 0) {
          /* for little endian, need to swap words in each double word
           * quantity.  Order of bytes in a word is okay, but not the
           * order of words.
           */
          swap = num1[0];
          num1[0] = num1[1];
          num1[1] = swap;
          swap = num1[2];
          num1[2] = num1[3];
          num1[3] = swap;
        }
        num[0] = getcon(&num1[0], DT_REAL8);
        num[1] = getcon(&num1[2], DT_REAL8);
        break;
      default:
        num[0] = 0;
        num[1] = 0;
        errsev(91);
      }
    }
    return getcon(num, DT_CMPLX16);

#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    if (from == TY_WORD) {
      num[0] = 0;
      num[1] = 0;
      num[2] = 0;
      num[3] = oldval;
      num[0] = getcon(num, DT_QUAD);
      num[1] = stb.quad0;
    } else if (from == TY_DWORD) {
      num[0] = 0;
      num[1] = 0;
      num[2] = CONVAL1G(oldval);
      num[3] = CONVAL2G(oldval);
      num[0] = getcon(num, DT_QUAD);
      num[1] = stb.quad0;
    } else if (from == TY_INT8 || from == TY_LOG8) {
      num1[0] = CONVAL1G(oldval);
      num1[1] = CONVAL2G(oldval);
      xqflt64(num1, num);
      num[0] = getcon(num, DT_QUAD);
      num[1] = stb.quad0;
    } else if (TY_ISINT(from)) {
      xqfloat(oldval, num);
      num[0] = getcon(num, DT_QUAD);
      num[1] = stb.quad0;
    } else {
      switch (from) {
      case TY_REAL:
        xftoq(oldval, num);
        num[0] = getcon(num, DT_QUAD);
        num[1] = stb.quad0;
        break;
      case TY_DBLE:
        num1[0] = CONVAL1G(oldval);
        num1[1] = CONVAL2G(oldval);
        xdtoq(num1, num);
        num[0] = getcon(num, DT_QUAD);
        num[1] = stb.quad0;
	break;
      case TY_QUAD:
        num[0] = oldval;
        num[1] = stb.quad0;
        break;
      case TY_CMPLX:
        xftoq(CONVAL1G(oldval), num1);
        num[0] = getcon(num1, DT_QUAD);
        xftoq(CONVAL2G(oldval), num1);
        num[1] = getcon(num1, DT_QUAD);
        break;
      case TY_DCMPLX:
        num1[0] = CONVAL1G(CONVAL1G(oldval));
        num1[1] = CONVAL2G(CONVAL1G(oldval));
        xdtoq(num1, num2);
        num[0] = getcon(num2, DT_QUAD);
        num1[0] = CONVAL1G(CONVAL2G(oldval));
        num1[1] = CONVAL2G(CONVAL2G(oldval));
        xdtoq(num1, num2);
        num[1] = getcon(num2, DT_QUAD);
        break;
      default:
        num[0] = 0;
        num[1] = 0;
        errsev(91);
      }
    }
    return getcon(num, DT_QCMPLX);
#endif

  case TY_NCHAR:
    if (from == TY_WORD) {
      num[0] = 0;
      num[1] = oldval;
      oldval = hex2nchar(num);
      cp = stb.n_base + CONVAL1G(oldval);
      oldcvlen = kanji_len((unsigned char *)cp, string_length(DTYPEG(oldval)));
      oldtyp = get_type(2, TY_NCHAR, mk_cval(oldcvlen, DT_INT4));
      if (newtyp == oldtyp)
        return oldval;
    } else if (from == TY_DWORD) {
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
      oldval = hex2nchar(num);
      cp = stb.n_base + CONVAL1G(oldval);
      oldcvlen = kanji_len((unsigned char *)cp, string_length(DTYPEG(oldval)));
      oldtyp = get_type(2, TY_NCHAR, mk_cval(oldcvlen, DT_INT4));
      if (newtyp == oldtyp)
        return oldval;
    } else if (from != TY_NCHAR) {
      errsev(146);
      return getstring(" ", 1);
    }
    goto char_shared;

  case TY_CHAR:
    if (from == TY_WORD) {
      num[0] = 0;
      num[1] = oldval;
      oldval = hex2char(num);
      /* old value is now in character form; must changed oldtyp
       * and must check if lengths just happen to be equal.
       */
      oldtyp = DTYPEG(oldval);
      if (newtyp == oldtyp)
        return oldval;
    } else if (from == TY_DWORD) {
      num[0] = CONVAL1G(oldval);
      num[1] = CONVAL2G(oldval);
      oldval = hex2char(num);
      /* old value is now in character form; must changed oldtyp
       * and must check if lengths just happen to be equal.
       */
      oldtyp = DTYPEG(oldval);
      if (newtyp == oldtyp)
        return oldval;
    } else if (from != TY_CHAR && from != TY_HOLL) {
      errsev(146);
      return getstring(" ", 1);
    }

  char_shared:
    if (newtyp == DT_ASSCHAR || newtyp == DT_DEFERCHAR)
      return oldval;
    if (newtyp == DT_ASSNCHAR || newtyp == DT_DEFERNCHAR)
      return oldval;
    newcvlen = string_length(newtyp);
    if (from == TY_HOLL) {
      oldval = CONVAL1G(oldval); /* locate Hollerith's char constant */
      oldtyp = DTYPEG(oldval);
    }
    oldcvlen = string_length(oldtyp);
    if (oldcvlen > newcvlen) {
      /* truncate character string: */
      errinfo(122);
      if (from == TY_NCHAR) {
        /* oldval is kanji string, CONVAL1G(oldval) is char string */
        cp = local_sname(stb.n_base + CONVAL1G(CONVAL1G(oldval)));
      } else
        cp = local_sname(stb.n_base + CONVAL1G(oldval));
      if (from == TY_NCHAR ||
          (to == TY_NCHAR && (from == TY_WORD || from == TY_DWORD)))
        /* compute actual num bytes used to represent newcvlen chars:*/
        newcvlen = kanji_prefix((unsigned char *)cp, newcvlen,
                                DTY(DTYPEG(oldval) + 1));
      result = getstring(cp, newcvlen);
      if (to == TY_NCHAR) {
        num[0] = result;
        num[1] = 0;
        num[2] = 0;
        num[3] = 0;
        result = getcon(num, newtyp);
      }
      return result;
    }

    /* oldcvlen < newcvlen -    pad with blanks.  This works for regular
       and kanji strings.  Note (from == oldcvlen) unless type is TY_NCHAR
       and there are one or more Kanji(2 byte) characters in the string. */

    newcvlen -= oldcvlen; /* number of pad blanks */
    blnk = ' ';
    if (from == TY_NCHAR) /* double for NCHAR */
      newcvlen *= 2, blnk = 0xA1;
    from =
        string_length(DTYPEG(oldval)); /* number bytes in char string const */
    cp = getitem(0, from + newcvlen);
    BCOPY(cp, stb.n_base + CONVAL1G(oldval), char, (INT)from);
    if (newcvlen > 0) {
      do {
        cp[from++] = blnk;
      } while (--newcvlen > 0);
    }
    result = getstring(cp, from);
    if (to == TY_NCHAR) {
      num[0] = result;
      num[1] = 0;
      num[2] = 0;
      num[3] = 0;
      result = getcon(num, newtyp);
    }
    return result;

  case TY_NUMERIC:
    if (!TY_ISNUMERIC(from))
      goto type_conv_error;
    return oldval;

  default:
    break;
  }

type_conv_error:
  errsev(91);
  return 0;
}

static void
truncation_warning(int c)
{
  char buf[20];
  sprintf(buf, "%d", c);
  error(W_0128_Integer_constant_truncated_to_fit_data_type_OP1, ERR_Warning,
        gbl.lineno, buf, 0);
}

static void
conversion_warning(void)
{
  error(W_0170_PGI_Fortran_extension_OP1_OP2, ERR_Warning, gbl.lineno,
        "conversion of CHARACTER constant to numeric", 0);
}

static INT
_fdiv(INT dividend, INT divisor)
{
  INT quotient;
#ifdef TM_FRCP
  INT temp;

  if (!flg.ieee) {
    xfrcp(divisor, &temp);
    xfmul(dividend, temp, &quotient);
  } else
    xfdiv(dividend, divisor, &quotient);
#else
  xfdiv(dividend, divisor, &quotient);
#endif
  return quotient;
}

static void
_ddiv(INT *dividend, INT *divisor, INT *quotient)
{
#ifdef TM_DRCP
  INT temp[2];

  if (!flg.ieee) {
    xdrcp(divisor, temp);
    xdmul(dividend, temp, quotient);
  } else
    xddiv(dividend, divisor, quotient);
#else
  xddiv(dividend, divisor, quotient);
#endif
}

/** \brief Convert doubleword hex/octal value to a character.
    \param hexval two-element array of [0] msw, [1] lsw
    \return the symbol table entry of the character constant


    The conversion is performed by copying an 8-bit value (2 hex digits) to a
    character position which is endian-dependent.  The endian-dependency is
    handled as if the hex value is "equivalenced" with a character value of the
    same length.  The length of the character constant returned is determined
    by the magnitude of the hex values (leading 0's are not converted).  Note
    that this conversion returns the same character value in context of an
    assignment or data initialization.

    We may be incompatible with other implementations with respect to data
    initialization:
    1.  if the value is smaller than the char item being initialized, the
        conversion process results in appending blanks;  other systems may
        pad with 'nulls'
    2.  if the value is larger, truncation of the least significant characters
        ("rightmost") occurs; other systems truncate the most significant
        characters ("leftmost").
 */
static int
hex2char(INT *hexval)
{
  UINT val;
  int i;
  int len;
  char *p;
  char buf[8];

  len = 0;
  if (flg.endian) {
    /* big endian: rightmost 2 hex digits are in last byte position */
    p = buf + 7;
    i = -1;
  } else {
    /* little endian: rightmost 2 hex digits are in first byte position */
    p = buf;
    i = 1;
  }
  val = hexval[1];
  while (val) {
    *p = val & 0xff;
    p += i;
    len++;
    val >>= 8;
  }
  val = hexval[0];
  while (val) {
    *p = val & 0xff;
    p += i;
    len++;
    val >>= 8;
  }

  if (len == 0) {
    len = 1;
    *p = '\0';
  } else if (flg.endian)
    p++;
  else
    p = buf;

  return getstring(p, len);
}

/*
 * convert doubleword hex/octal value to an ncharacter.  Function return value
 * is the symbol table entry of the character constant.  The conversion is
 * performed by copying an 8-bit value (2 hex digits) to a character position
 * which is endian-dependent. The endian-dependency is handled as if
 * the hex value is "equivalenced" with a ncharacter value of the same length.
 * The length of the ncharacter constant returned is determined by the magnitude
 * of the hex values (leading 0's are not converted).  Note that this conversion
 * returns the same ncharacter value in context of an assignment or data
 * initialization.  We may be incompatible with other implementations
 * with respect to data initialization:
 * 1.  if the value is smaller than the nchar item being initialized, the
 *     conversion process results in appending blanks;  other systems may
 *     pad with 'nulls'
 * 2.  if the value is larger, truncation of the least significant characters
 *     ("rightmost") occurs; other systems truncate the most significant
 *     characters ("leftmost").
 *
 * hexval[0] is msw, hexval[1] is lsw
 */
static int
hex2nchar(INT *hexval)
{
  UINT val;
  int i;
  int len;
  unsigned short *p;
  unsigned short buf[4];

  len = 0;
  if (flg.endian) {
    /* big endian: rightmost 2 hex digits are in last byte position */
    p = buf + 3;
    i = -1;
  } else {
    /* little endian: rightmost 2 hex digits are in first byte position */
    p = buf;
    i = 1;
  }
  val = hexval[1];
  while (val) {
    *p = val & 0xffff;
    p += i;
    len += 2;
    val >>= 16;
  }
  val = hexval[0];
  while (val) {
    *p = val & 0xffff;
    p += i;
    len += 2;
    val >>= 16;
  }
  if (len == 0) {
    len = 1;
    *p = '\0';
  } else if (flg.endian)
    p++;
  else
    p = buf;

  return getstring((char *)p, len);
}

int
resolve_ast_alias(int ast)
{
  int alias;
  while (ast && (alias = A_ALIASG(ast)) > 0 &&
         alias != ast /* prevent looping on bogus A_CNST self-aliases */) {
    ast = alias;
  }
  return ast;
}

LOGICAL
is_array_ast(int ast)
{
  if ((ast = resolve_ast_alias(ast))) {
    if (is_array_dtype(get_ast_dtype(ast)))
      return TRUE;
    switch (A_TYPEG(ast)) {
    case A_ID:
      return is_array_sptr(A_SPTRG(ast));
    case A_SUBSTR:
      return is_array_ast(A_LOPG(ast));
    case A_MEM:
      return is_array_ast(A_MEMG(ast)) || is_array_ast(A_PARENTG(ast));
    case A_SUBSCR: {
      int asd = A_ASDG(ast);
      int dims = ASD_NDIM(asd);
      int j;
      for (j = 0; j < dims; ++j) {
        if (is_array_ast(ASD_SUBS(asd, j)))
          return TRUE;
      }
    }
      return is_array_ast(A_LOPG(ast));
    case A_TRIPLE:
      return TRUE;
    }
  }
  return FALSE;
}

LOGICAL
has_vector_subscript_ast(int ast)
{
  if ((ast = resolve_ast_alias(ast))) {
    switch (A_TYPEG(ast)) {
    case A_PAREN:
    case A_CONV:
    case A_SUBSTR:
      return has_vector_subscript_ast(A_LOPG(ast));
    case A_MEM:
      return has_vector_subscript_ast(A_PARENTG(ast));
    case A_SUBSCR: {
      int asd = A_ASDG(ast);
      int dims = ASD_NDIM(asd);
      int j;
      for (j = 0; j < dims; ++j) {
        int subs_ast = ASD_SUBS(asd, j);
        if (A_TYPEG(subs_ast) != A_TRIPLE && is_array_ast(subs_ast))
          return TRUE;
      }
    }
      return has_vector_subscript_ast(A_LOPG(ast));
    }
  }
  return FALSE;
}

LOGICAL
is_data_ast(int ast)
{
  if ((ast = resolve_ast_alias(ast))) {
    switch (A_TYPEG(ast)) {
    case A_ID:
      return !is_procedure_ptr(A_SPTRG(ast));
    case A_LABEL:
    case A_ENTRY:
      return FALSE;
    case A_CNST:
    case A_CMPLXC:
    case A_CONV:
    case A_UNOP:
    case A_BINOP:
    case A_PAREN:
      return TRUE;
    case A_FUNC: {
      DTYPE dtype = A_DTYPEG(ast);
      return dtype <= 0 || DTY(dtype) == TY_PROC;
    }
    case A_MEM:
      return is_data_ast(A_MEMG(ast));
    case A_SUBSTR:
    case A_SUBSCR:
      return TRUE;
    }
  }
  return FALSE;
}

LOGICAL
is_variable_ast(int ast)
{
  if ((ast = resolve_ast_alias(ast))) {
    switch (A_TYPEG(ast)) {
    case A_ID:
      return !is_procedure_ptr(A_SPTRG(ast));
    case A_MEM:
      return is_variable_ast(A_MEMG(ast)) && is_variable_ast(A_PARENTG(ast));
    case A_SUBSTR:
    case A_SUBSCR:
      return is_variable_ast(A_LOPG(ast));
    }
  }
  return FALSE;
}

int
get_ast_asd(int ast)
{
  if ((ast = resolve_ast_alias(ast)) && A_TYPEG(ast) == A_SUBSCR)
    return A_ASDG(ast);
  return 0;
}

DTYPE
get_ast_dtype(int ast)
{
  if ((ast = resolve_ast_alias(ast))) {
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_CNST:
    case A_LABEL:
    case A_BINOP:
    case A_UNOP:
    case A_CMPLXC:
    case A_CONV:
    case A_PAREN:
    case A_MEM:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_FUNC:
    case A_INTR:
    case A_INIT:
    case A_ASN:
    case A_ICALL:
      /* Only these AST types interpret A_DTYPEG's overloaded field
       * as containing a data type table index.
       */
      return A_DTYPEG(ast);
    }
  }
  return DT_NONE;
}

int
get_ast_rank(int ast)
{
  if ((ast = resolve_ast_alias(ast))) {
    int shd;
    DTYPE dtype;

    /* These tests of those representations are arranged
     * here in descending order of credibility.  When multiple
     * representations are present, We don't check their consistency
     * because there are indeed cases where they'll differ.
     */
    if ((shd = A_SHAPEG(ast)))
      return SHD_NDIM(shd); /* AST has explicit shape description */
    if (is_array_dtype(dtype = get_ast_dtype(ast)))
      return ADD_NUMDIM(dtype); /* Data type of AST is an array */
  }
  return 0;
}

/* This utility finds the most relevant symbol table reference in an AST,
 * preferring member symbols to their parents.  It's like memsym_of_ast()
 * but it fails gracefully and returns 0 when presented with an AST
 * that does not contain a symbol.
 */
int
get_ast_sptr(int ast)
{
  int sptr = 0;
  if ((ast = resolve_ast_alias(ast))) {
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
      sptr = A_SPTRG(ast);
      break;
    case A_SUBSCR:
    case A_SUBSTR:
    case A_CONV:
    case A_FUNC:
      sptr = get_ast_sptr(A_LOPG(ast));
      break;
    case A_MEM:
      sptr = get_ast_sptr(A_MEMG(ast));
      if (sptr <= NOSYM)
        sptr = get_ast_sptr(A_PARENTG(ast));
      break;
    }
  }
  return sptr;
}

/* Create a duplicate of an AST with a new data type. */
int
rewrite_ast_with_new_dtype(int ast, DTYPE dtype)
{
  if (A_DTYPEG(ast) != dtype) {
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_CNST:
    case A_LABEL: {
      int sptr = A_SPTRG(ast);
      int orig_sptr_dtype = DTYPEG(sptr);
      DTYPEP(sptr, dtype);
      ast = mk_id(sptr);
      DTYPEP(sptr, orig_sptr_dtype);
      return ast;
    }
    case A_MEM:
      return mk_member(A_PARENTG(ast), A_MEMG(ast), dtype);
    case A_SUBSCR: {
      int j, rank = get_ast_rank(ast), asd = A_ASDG(ast), subs[MAXRANK];
      for (j = 0; j < rank; ++j) {
        subs[j] = ASD_SUBS(asd, j);
      }
      return mk_subscr(A_LOPG(ast), subs, rank, dtype);
    }
    case A_ALLOC: /* and possibly others */
      /* not hashed, so it's okay to substitute dtype in situ */
      A_DTYPEP(ast, dtype);
      break;
    default:
      interr("rewrite_ast_with_new_dtype: can't replace dtype in A_TYPE",
             A_TYPEG(ast), 3);
    }
  }
  return ast;
}

/*
 * Create a duplicated AST
 */
int
mk_duplicate_ast(int ast)
{
  int newast;

  /*switch (A_TYPEG(ast)) {
  case A_PRAGMA:
    newast = mk_stmt(A_PRAGMA, 0);
    astb.stg_base[newast] = astb.stg_base[ast];
    break;
  default:
    interr("mk_duplicate_ast: A_TYPE is not supported yet",
           A_TYPEG(ast), ERR_Informational);
           }*/
  newast = mk_stmt(A_TYPEG(ast), 0);
  astb.stg_base[newast] = astb.stg_base[ast];

  return newast;
}

/* Get the most credible shape (rank and extents) of an AST from the various
 * sources of information that exist.  Returns the rank, which is also
 * the number of leading entries that have been filled in extent_asts[].
 */
int
get_ast_extents(int extent_asts[], int from_ast, DTYPE arr_dtype)
{
  int rank = get_ast_rank(from_ast);

  if (rank > 0) {
    int shape = A_SHAPEG(from_ast);
    int asd = A_TYPEG(from_ast) == A_SUBSCR ? A_ASDG(from_ast) : 0;
    int dim;

    for (dim = 0; dim < rank; ++dim) {
      int lb = 0, ub = 0, stride = 0, extent;
      if (shape) {
        lb = SHD_LWB(shape, dim);
        ub = SHD_UPB(shape, dim);
        stride = SHD_STRIDE(shape, dim);
      }
      if (!ub && asd) {
        int subscript = ASD_SUBS(asd, dim);
        if (A_TYPEG(subscript) == A_TRIPLE) {
          lb = A_LBDG(subscript);
          ub = A_UPBDG(subscript);
          stride = A_STRIDEG(subscript);
        } else {
          int subscr_shape = A_SHAPEG(subscript);
          if (subscr_shape > 0)
            ub = extent_of_shape(subscr_shape, 0);
        }
      }
      if (!ub && is_array_dtype(arr_dtype))
        ub = ADD_UPAST(arr_dtype, dim);
      if (!ub)
        ub = astb.bnd.one;
      if (!lb && is_array_dtype(arr_dtype))
        lb = ADD_LWAST(arr_dtype, dim);
      if (!lb)
        lb = astb.bnd.one;
      if (!stride)
        stride = astb.bnd.one;

      extent = ub;
      if (lb != stride) {
        extent = mk_binop(OP_SUB, extent, lb, astb.bnd.dtype);
        extent = mk_binop(OP_ADD, extent, stride, astb.bnd.dtype);
      }
      if (stride != astb.bnd.one)
        extent = mk_binop(OP_DIV, extent, stride, astb.bnd.dtype);
      extent_asts[dim] = extent;
    }
  }
  return rank;
}

/* Get the rank and lower/upper bounds on each dimension from an AST
 * and/or an array dtype, if possible.  When lower and upper bounds
 * cannot all be discerned, or when strides appear, then set the lower
 * bounds all to 1 and use extents as the upper bounds.
 */
int
get_ast_bounds(int lower_bound_asts[], int upper_bound_asts[], int from_ast,
               DTYPE arr_dtype)
{
  int rank = get_ast_rank(from_ast);

  if (rank > 0) {
    int shape = A_SHAPEG(from_ast);
    int asd = A_TYPEG(from_ast) == A_SUBSCR ? A_ASDG(from_ast) : 0;
    int dim = 0;

    for (dim = 0; dim < rank; ++dim) {
      int lb = 0, ub = 0;
      if (asd) {
        int subscript = ASD_SUBS(asd, dim);
        if (subscript > 0) {
          if (A_TYPEG(subscript) == A_TRIPLE ||
              A_SHAPEG(subscript) > 0 /* vector-valued subscript */) {
            break;
          }
        }
      }
      if (shape) {
        int stride = SHD_STRIDE(shape, dim);
        if (stride > 0 && stride != astb.bnd.one) {
          break;
        }
        lb = SHD_LWB(shape, dim);
        ub = SHD_UPB(shape, dim);
      }
      if (is_array_dtype(arr_dtype)) {
        if (!ub) {
          ub = ADD_UPAST(arr_dtype, dim);
        }
        if (!lb) {
          lb = ADD_LWAST(arr_dtype, dim);
        }
      }

      if (lb > 0 && ub > 0) {
        lower_bound_asts[dim] = lb;
        upper_bound_asts[dim] = ub;
      } else {
        break;
      }
    }

    if (dim < rank) {
      /* Could not get good lower and upper bounds on all dimensions,
       * or there's a subscript triplet or vector-valued subscript.
       * Set the lower bounds all to 1, then try to extract extents
       * for use as the upper bounds.
       */
      for (dim = 0; dim < rank; ++dim) {
        lower_bound_asts[dim] = astb.bnd.one;
      }
      return get_ast_extents(upper_bound_asts, from_ast, arr_dtype);
    }
  }
  return rank;
}

int
add_extent_subscripts(int to_ast, int rank, const int extent_asts[],
                      DTYPE elt_dtype)
{
  if (rank > 0) {
    int j, triple_asts[MAXRANK];
    for (j = 0; j < rank; ++j) {
      triple_asts[j] = mk_triple(astb.bnd.one, extent_asts[j], 0);
    }
    to_ast = mk_subscr(to_ast, triple_asts, rank, elt_dtype);
  }
  return to_ast;
}

int
add_bounds_subscripts(int to_ast, int rank, const int lower_bound_asts[],
                      const int upper_bound_asts[], DTYPE elt_dtype)
{
  if (rank > 0) {
    int j, triple_asts[MAXRANK];
    for (j = 0; j < rank; ++j) {
      triple_asts[j] = mk_triple(lower_bound_asts[j], upper_bound_asts[j], 0);
    }
    to_ast = mk_subscr(to_ast, triple_asts, rank, elt_dtype);
  }
  return to_ast;
}

/* Add subscript triples to an array-valued AST that span a shape
 * taken from another AST.
 */
int
add_shapely_subscripts(int to_ast, int from_ast, DTYPE arr_dtype,
                       DTYPE elt_dtype)
{
  int extent_asts[MAXRANK];
  int rank = get_ast_extents(extent_asts, from_ast, arr_dtype);
  return add_extent_subscripts(to_ast, rank, extent_asts, elt_dtype);
}

/* If an array AST is a whole array, return the SPTR of the array or the
 * structure component. */
SPTR
get_whole_array_sym(int arr_ast)
{
  if (A_TYPEG(arr_ast) == A_ID)
    return A_SPTRG(arr_ast);
  if (A_TYPEG(arr_ast) == A_MEM && !A_SHAPEG(A_PARENTG(arr_ast)))
    return A_SPTRG(A_MEMG(arr_ast));
  return SPTR_NULL;
}
