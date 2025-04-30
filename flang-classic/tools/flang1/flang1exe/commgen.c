/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief Fortran communications optimizer module
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "comm.h"
#include "dtypeutl.h"
#include "symutl.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "transfrm.h"
#include "extern.h"
#include "commopt.h"
#include "rte.h"
#include "fdirect.h"
#include "rtlRtns.h"
#include "ilidir.h" /* for open_pragma, close_pragma */

#ifdef FLANG_COMMGEN_UNUSED
static void generate_forall(int);
#endif
static void change_forall_triple(int, int, int, LOGICAL);
static int fill_cyclic_k(int);
static void fill_cyclic_1(int);
static void generate_hallobnds(int);
static void generate_sect(int);
static void generate_copy(int);
static void generate_gather(int);
#ifdef FLANG_COMMGEN_UNUSED
static void generate_shift(int);
#endif
static void generate_get_scalar(void);
static void eliminate_redundant(void);
#ifdef FLANG_COMMGEN_UNUSED
static int rewrite_expr(int, int, int);
#endif
static void pointer_changer(void);
static int pointer_squeezer(int);
static int cyclic_section(int, int, int, int, int);
#ifdef FLANG_COMMGEN_UNUSED
static LOGICAL is_same_lower_dim(int, int, int, int);
#endif
static int gen_minmax(int, int, int);
#ifdef FLANG_COMMGEN_UNUSED
static int rhs_cyclic(int, int, int);
static int cyclic_localize(int, int, int);
#endif

void
comm_generator(void)
{
  int std, stdnext;
  int ast;
  int type;

  rt_outvalue();
  eliminate_redundant();

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    gbl.lineno = STD_LINENO(std);
    ast = STD_AST(std);
    type = A_TYPEG(ast);
    if (type == A_ASN)
      type = A_TYPEG(A_SRCG(ast));
    switch (type) {
    case A_HALLOBNDS:
      generate_hallobnds(ast);
      break;
    case A_HSECT:
      generate_sect(ast);
      break;
    case A_HCOPYSECT:
      generate_copy(ast);
      break;
    case A_HGATHER:
      generate_gather(ast);
      break;
    default:
      break;
    }
  }
  generate_get_scalar();
  if (!XBIT(58, 0x8000000)) {
    pointer_changer();
  }
}

static void
generate_get_scalar(void)
{

  int l, lsym;
  int astnew;
  int i, j, nargs, argt;
  int asd;
  int ndim;
  int same;
  int dest, src, dest1;
  int commstd, rt;
  int std, ast;
  int nd, a;
  int asn, lop;

  init_gstbl();
  find_get_scalar();

  for (i = 0; i < gstbl.avl; i++) {
    commstd = gstbl.base[i].f1;
    rt = STD_AST(commstd);
    nd = A_OPT1G(rt);
    assert(A_TYPEG(rt) == A_HGETSCLR, "generate_get_scalar: wrong ast type", 2,
           rt);
    assert(nd, "generate_get_scalar: something is wrong", 2, rt);
    std = FT_STD(nd); /* this is where getscalar is used */
    ast = STD_AST(std);
    if (STD_DELETE(commstd)) {
      same = FT_GETSCLR_SAME(nd);
      src = A_SRCG(same);
      dest = A_DESTG(same);
      dest1 = A_DESTG(rt);
      asn = mk_stmt(A_ASN, 0);
      A_DESTP(asn, dest1);
      A_SRCP(asn, dest);
      add_stmt_after(asn, commstd);
      assert(src && dest, "generate_get_scalar: something is wrong", 2, rt);
      delete_stmt(commstd);
      continue;
    }

    a = A_SRCG(rt);
    asd = A_ASDG(a);
    ndim = ASD_NDIM(asd);
    l = A_LOPG(a);
    lsym = memsym_of_ast(l);
    dest = A_DESTG(rt);

    /* put out a call to fetch the value */
    /* call pghpf_get_scalar(temp, array_base, array, subscripts) */
    nargs = ndim + 3;
    argt = mk_argt(nargs);
    ARGT_ARG(argt, 0) = dest;
    ARGT_ARG(argt, 1) = l;
    if (A_LOPG(rt)) {
      ARGT_ARG(argt, 2) = A_LOPG(rt);
    } else {
      ARGT_ARG(argt, 2) = check_member(l, mk_id(DESCRG(lsym)));
    }
    DESCUSEDP(lsym, 1);
    for (j = 0; j < ndim; ++j) {
      a = mk_default_int(ASD_SUBS(asd, j));
      if (normalize_bounds(lsym))
        a = sub_lbnd(A_DTYPEG(l), j, a, 0);
      ARGT_ARG(argt, j + 3) = a;
    }
    astnew = mk_stmt(A_CALL, 0);
    lop = mk_id(sym_mkfunc(mkRteRtnNm(RTE_get_scalar), DT_NONE));
    A_LOPP(astnew, lop);
    A_ARGCNTP(astnew, nargs);
    A_ARGSP(astnew, argt);
    add_stmt_before(astnew, commstd);
    report_comm(commstd, GETSCALAR_CAUSE);
    delete_stmt(commstd);
    continue;
  }
  free_gstbl();
}

static void
generate_hallobnds(int ast)
{
  int newalloc, newdealloc, deallocstd;
  int i;
  int asd, ndim;
  int subs[7];
  int arr;
  int std;
  int nd;
  int same;

  assert(A_TYPEG(ast) == A_HALLOBNDS, "generate_hallobnds: wrong ast type", 2,
         ast);
  nd = A_OPT1G(ast);
  assert(nd, "generate_hallobnds: some thing is wrong", 2, ast);
  std = A_STDG(ast);
  if (STD_DELETE(std)) {
    same = FT_ALLOC_SAME(nd);
    delete_stmt(std);
    return;
    /*A_OTRIPLEP(ast, A_OTRIPLEG(same)); */
  }
  STD_DELETE(std) = 1;
  deallocstd = FT_ALLOC_FREE(nd);
  arr = A_LOPG(ast);
  asd = A_ASDG(arr);
  ndim = ASD_NDIM(asd);
  for (i = 0; i < ndim; i++)
    subs[i] = ASD_SUBS(asd, i);
  newalloc = mk_mem_allocate(A_LOPG(arr), subs, std, 0);
  STD_RESCOPE(newalloc) = 1;
  if (deallocstd) {
    newdealloc = mk_mem_deallocate(A_LOPG(arr), deallocstd);
    STD_RESCOPE(newdealloc) = 1;
  }
  delete_stmt(std);
}

static void
generate_sect(int ast)
{
  int freestd;
  int sptr;
  int i;
  int asd, ndim;
  int subs[7];
  int arr;
  int std;
  int nd;
  int same;
  int sec;
  int sect;
  int allocstd;
  int alloc;
  int nd1;
  int sectflag;

  sect = A_SRCG(ast);
  assert(A_TYPEG(sect) == A_HSECT, "generate_sect: wrong ast type", 2, ast);
  nd = A_OPT1G(sect);
  assert(nd, "generate_sect: some thing is wrong", 2, ast);
  std = A_STDG(ast);
  if (STD_DELETE(std)) {
    same = FT_SECT_SAME(nd);
    delete_stmt(std);
    return;
  }
  allocstd = FT_SECT_ALLOC(nd);
  arr = A_LOPG(sect);
  if (allocstd) {
    alloc = STD_AST(allocstd);
    nd1 = A_OPT1G(alloc);
    sptr = FT_ALLOC_OUT(nd1);
    asd = A_ASDG(arr);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++)
      subs[i] = ASD_SUBS(asd, i);
    arr = mk_subscr(mk_id(sptr), subs, ndim, DTYPEG(sptr));
  }
  freestd = FT_SECT_FREE(nd);
  sec = FT_SECT_OUT(nd);
  sectflag = FT_SECT_FLAG(nd);
  make_sec_from_ast(arr, std, freestd, mk_id(sec), sectflag);
  delete_stmt(std);
}

static void
generate_copy(int ast)
{
  int freestd;
  int sptr;
  int std;
  int nd, nd1;
  int same;
  int cp;
  int copy;
  int alloc_std, alloc;
  int nargs, argt, astnew;
  int sectl_std, sectl;
  int sectr_std, sectr;
  int src, dest;
  int asn, desc;
  int atp;

  copy = A_SRCG(ast);
  assert(A_TYPEG(copy) == A_HCOPYSECT, "generate_copy: wrong ast type", 2, ast);
  nd = A_OPT1G(copy);
  assert(nd, "generate_sect: some thing is wrong", 2, ast);
  std = A_STDG(ast);
  if (STD_DELETE(std)) {
    same = FT_CCOPY_SAME(nd);
    delete_stmt(std);
    return;
  }

  sectl_std = FT_CCOPY_SECTL(nd);
  sectl = A_SRCG(STD_AST(sectl_std));
  nd1 = A_OPT1G(sectl);
  sptr = FT_SECT_OUT(nd1);
  desc = check_member(FT_SECT_ARR(nd1), mk_id(sptr));
  A_DDESCP(copy, desc);

  sectr_std = FT_CCOPY_SECTR(nd);
  sectr = A_SRCG(STD_AST(sectr_std));
  nd1 = A_OPT1G(sectr);
  sptr = FT_SECT_OUT(nd1);
  desc = check_member(FT_SECT_ARR(nd1), mk_id(sptr));
  A_SDESCP(copy, desc);

  if (FT_CCOPY_USELHS(nd)) {
    dest = A_LOPG(FT_CCOPY_LHS(nd));
    A_DESTP(copy, dest);
  } else {
    alloc_std = FT_CCOPY_ALLOC(nd);
    alloc = STD_AST(alloc_std);
    nd1 = A_OPT1G(alloc);
    sptr = FT_ALLOC_OUT(nd1);
    dest = mk_id(sptr);
    A_DESTP(copy, dest);
  }

  src = A_SRCG(copy);
  A_SRCP(copy, A_LOPG(src));

  freestd = FT_CCOPY_FREE(nd);
  cp = FT_CCOPY_OUT(nd);

  /* cp =pghpf_comm_copy_(void *db, void *sb, section *sd, section *ss); */
  nargs = 4;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = A_DESTG(copy);
  ARGT_ARG(argt, 1) = A_SRCG(copy);
  ARGT_ARG(argt, 2) = A_DDESCG(copy);
  ARGT_ARG(argt, 3) = A_SDESCG(copy);

  astnew = mk_func_node(A_FUNC,
                        mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_copy), DT_ADDR)),
                        nargs, argt);
  A_DTYPEP(astnew, DT_INT);
  NODESCP(A_SPTRG(A_LOPG(astnew)), 1);

  asn = mk_stmt(A_ASN, 0);
  dest = mk_id(cp);
  A_DESTP(asn, dest);
  A_SRCP(asn, astnew);
  add_stmt_before(asn, std);

  /*	  pghpf_comm_free_(cp) */
  /* free communication schedules */

  argt = mk_argt(2);
  ARGT_ARG(argt, 0) = astb.i1;
  ARGT_ARG(argt, 1) = mk_id(cp);
  ast = mk_stmt(A_CALL, 0);
  atp = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_free), DT_NONE));
  A_LOPP(ast, atp);
  NODESCP(A_SPTRG(A_LOPG(ast)), 1);
  A_ARGCNTP(ast, 2);
  A_ARGSP(ast, argt);
  add_stmt_after(ast, freestd);
  delete_stmt(std);
}

static void
generate_gather(int ast)
{
  int freestd;
  int sptr;
  int i;
  int std;
  int nd, nd1;
  int same;
  int cp;
  int gather;
  int alloc_std, alloc;
  int nargs, argt, astnew;
  int sectm_std, sectm;
  int sectv_std, sectv;
  int dest;
  int asn;
  int v;
  int j;
  int mask, mask_sec_ast;
  int ndim1, asd1;
  int npermute;
  int vsub, nvsub, vsub_ast;
  int vsubstd, nvsubstd, sectvsub, sectnvsub;
  int nvsub_ast, vsub_sec_ast, nvsub_sec_ast;
  int vec[7], vecsec[7], permute[7];
  int array_sec_ast, result_sec_ast, result_ast, array_ast;
  int vflag, pflag, nvec;
  int vdim, pdim;
  int func;
  int commstd;
  int comm_type;
  int atp;

  gather = A_SRCG(ast);
  assert(A_TYPEG(gather) == A_HGATHER, "generate_gather: wrong ast type", 2,
         ast);
  nd = A_OPT1G(gather);
  assert(nd, "generate_sect: some thing is wrong", 2, ast);
  std = A_STDG(ast);
  if (STD_DELETE(std)) {
    same = FT_CGATHER_SAME(nd);
    delete_stmt(std);
    return;
  }

  vsub = FT_CGATHER_VSUB(nd);
  vsub_ast = A_LOPG(vsub);
  asd1 = A_ASDG(vsub);
  ndim1 = ASD_NDIM(asd1);

  vsubstd = FT_CGATHER_SECTVSUB(nd);
  if (vsubstd) {
    sectvsub = A_SRCG(STD_AST(vsubstd));
    nd1 = A_OPT1G(sectvsub);
    vsub_sec_ast = check_member(vsub_ast, mk_id(FT_SECT_OUT(nd1)));
  } else {
    sptr = memsym_of_ast(vsub);
    vsub_sec_ast = check_member(vsub_ast, DESCRG(sptr));
  }

  nvsub = FT_CGATHER_NVSUB(nd);
  nvsubstd = FT_CGATHER_SECTNVSUB(nd);

  if (nvsubstd) {
    sectnvsub = A_SRCG(STD_AST(nvsubstd));
    nd1 = A_OPT1G(sectnvsub);
    nvsub_sec_ast = check_member(A_LOPG(sectnvsub), mk_id(FT_SECT_OUT(nd1)));
  } else {
    sptr = memsym_of_ast(A_LOPG(nvsub));
    nvsub_sec_ast = check_member(nvsub, DESCRG(sptr));
  }

  comm_type = FT_CGATHER_TYPE(nd);

  if (comm_type == A_HGATHER) {
    if (FT_CGATHER_USELHS(nd)) {
      nvsub_ast = A_LOPG(FT_CGATHER_LHS(nd));
    } else {
      alloc_std = FT_CGATHER_ALLOC(nd);
      alloc = STD_AST(alloc_std);
      nd1 = A_OPT1G(alloc);
      nvsub_ast = mk_id(FT_ALLOC_OUT(nd1));
    }
  } else if (comm_type == A_HSCATTER) {
    nvsub_ast = A_LOPG(nvsub);
  }

  mask = A_MASKG(gather);
  if (mask) {
    sectm_std = FT_CGATHER_SECTM(nd);
    sectm = A_SRCG(STD_AST(sectm_std));
    nd1 = A_OPT1G(sectm);
    mask_sec_ast = mk_id(FT_SECT_OUT(nd1));
  } else {
    mask = mk_cval(1, DT_LOG);
    mask_sec_ast = mk_cval(dtype_to_arg(A_DTYPEG(mask)), DT_INT);
  }

  if (comm_type == A_HGATHER) {
    func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_gatherx), DT_ADDR));
    array_sec_ast = vsub_sec_ast;
    result_sec_ast = nvsub_sec_ast;
    result_ast = nvsub_ast;
    array_ast = vsub_ast;
  } else if (comm_type == A_HSCATTER) {
    func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_scatterx), DT_ADDR));
    result_sec_ast = vsub_sec_ast;
    array_sec_ast = nvsub_sec_ast;
    result_ast = vsub_ast;
    array_ast = nvsub_ast;
  }

  vflag = FT_CGATHER_VFLAG(nd);
  vdim = FT_CGATHER_VDIM(nd);
  pflag = FT_CGATHER_PFLAG(nd);
  pdim = FT_CGATHER_PDIM(nd);
  nvec = FT_CGATHER_NVEC(nd);
  npermute = FT_CGATHER_NPER(nd);

  asd1 = A_ASDG(vsub);
  ndim1 = ASD_NDIM(asd1);
  for (i = 0; i < ndim1; i++) {
    if (getbit(vdim, i)) {
      int vsptr;
      v = FT_CGATHER_V(nd, i);
      vsptr = memsym_of_ast(v);
      sectv_std = FT_CGATHER_SECTV(nd, i);
      if (sectv_std) {
        sectv_std = FT_CGATHER_SECTV(nd, i);
        sectv = A_SRCG(STD_AST(sectv_std));
        nd1 = A_OPT1G(sectv);
        sptr = FT_SECT_OUT(nd1);
      } else
        sptr = DESCRG(vsptr);

      vec[i] = v;
      vecsec[i] = check_member(v, mk_id(sptr));
    }
    if (getbit(pdim, i)) {
      permute[i] = FT_CGATHER_PERMUTE(nd, i);
    }
  }

  nargs = 2 * 3 + 1 + 1 + 2 * nvec + npermute;
  argt = mk_argt(nargs);

  ARGT_ARG(argt, 0) = result_ast;
  ARGT_ARG(argt, 1) = array_ast;
  ARGT_ARG(argt, 2) = mask;

  /* sections */
  ARGT_ARG(argt, 3) = result_sec_ast;
  ARGT_ARG(argt, 4) = array_sec_ast;
  ARGT_ARG(argt, 5) = mask_sec_ast;

  ARGT_ARG(argt, 6) = mk_cval(vflag, DT_INT);
  ARGT_ARG(argt, 7) = mk_cval(pflag, DT_INT);
  j = 8;
  asd1 = A_ASDG(vsub);
  ndim1 = ASD_NDIM(asd1);
  for (i = 0; i < ndim1; i++) {
    if (getbit(vdim, i)) {
      ARGT_ARG(argt, j) = vec[i];
      j++;
      ARGT_ARG(argt, j) = vecsec[i];
      j++;
    }
    if (getbit(pdim, i)) {
      ARGT_ARG(argt, j) = permute[i];
      j++;
    }
  }

  astnew = mk_func_node(A_FUNC, func, nargs, argt);
  A_DTYPEP(astnew, DT_INT);

  cp = FT_CGATHER_OUT(nd);

  asn = mk_stmt(A_ASN, 0);
  dest = mk_id(cp);
  A_DESTP(asn, dest);
  A_SRCP(asn, astnew);
  commstd = add_stmt_before(asn, std);
  NODESCP(A_SPTRG(A_LOPG(astnew)), 1);
  delete_stmt(std);

  if (!FT_CGATHER_INDEXREUSE(nd)) {
    /*
     * Free communication schedules:
     *	call pghpf_comm_free_(cp)
     */
    argt = mk_argt(2);
    ARGT_ARG(argt, 0) = astb.i1;
    ARGT_ARG(argt, 1) = mk_id(cp);
    ast = mk_stmt(A_CALL, 0);
    atp = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_free), DT_NONE));
    A_LOPP(ast, atp);
    NODESCP(A_SPTRG(A_LOPG(ast)), 1);
    A_ARGCNTP(ast, 2);
    A_ARGSP(ast, argt);

    freestd = FT_CGATHER_FREE(nd);
    add_stmt_after(ast, freestd);
  }
}

#ifdef FLANG_COMMGEN_UNUSED
static void
generate_shift(int ast)
{
  int freestd;
  int sptr;
  int i, j;
  int asd, ndim;
  int std;
  int nd;
  int same;
  int cp;
  int shift;
  int nargs, argt, astnew;
  int src, dest;
  int asn;
  int align;
  int type, boundary;
  int atp;

  shift = A_SRCG(ast);
  assert(A_TYPEG(shift) == A_HOVLPSHIFT, "generate_shift: wrong ast type", 2,
         ast);
  nd = A_OPT1G(shift);
  std = A_STDG(ast);
  if (STD_DELETE(std)) {
    assert(nd, "generate_shift: can't delete stmt", 2, ast);
    same = FT_SHIFT_SAME(nd);
    delete_stmt(std);
    return;
  }

  src = A_SRCG(shift);
  sptr = memsym_of_ast(src);
  align = ALIGNG(sptr);
  asd = A_ASDG(src);
  ndim = ASD_NDIM(asd);
  DESCUSEDP(sptr, 1);

  if (nd) {
    freestd = FT_SHIFT_FREE(nd);
    cp = FT_SHIFT_OUT(nd);
  } else {
    freestd = 0;
    cp = A_SPTRG(A_DESTG(ast));
  }

  /* create overlap_shift schedule
   * cp = pghpf_comm_shift_(void *b, section *g, ...);
   *		  ... = -shift, +shift for each dim
   */
  type = (nd ? FT_SHIFT_TYPE(nd) : 0);
  if (type == I_EOSHIFT)
    nargs = ndim * 2 + 2 + 1;
  else
    nargs = ndim * 2 + 2;

  argt = mk_argt(nargs);
  j = 0;
  ARGT_ARG(argt, j) = A_LOPG(src);
  j++;
  ARGT_ARG(argt, j) = check_member(A_LOPG(src), mk_id(DESCRG(sptr)));
  j++;
  DESCUSEDP(sptr, 1);
  if (type == I_EOSHIFT) {
    boundary = FT_SHIFT_BOUNDARY(nd);
    if (!boundary)
      boundary = astb.ptr0;
    ARGT_ARG(argt, j) = boundary;
    j++;
  }

  for (i = 0; i < ndim; ++i) {
    ARGT_ARG(argt, j) = mk_default_int(A_LBDG(ASD_SUBS(asd, i)));
    j++;
    ARGT_ARG(argt, j) = mk_default_int(A_UPBDG(ASD_SUBS(asd, i)));
    j++;
  }

  if (type == I_EOSHIFT)
    astnew = mk_func_node(
        A_FUNC, mk_id(sym_mkfunc(mkRteRtnNm(RTE_olap_eoshift), DT_ADDR)), nargs,
        argt);
  else if (type == I_CSHIFT)
    astnew = mk_func_node(
        A_FUNC, mk_id(sym_mkfunc(mkRteRtnNm(RTE_olap_cshift), DT_ADDR)), nargs,
        argt);
  else
    astnew = mk_func_node(
        A_FUNC, mk_id(sym_mkfunc(mkRteRtnNm(RTE_olap_shift), DT_ADDR)), nargs,
        argt);

  A_DTYPEP(astnew, DT_INT);
  NODESCP(A_SPTRG(A_LOPG(astnew)), 1);

  asn = mk_stmt(A_ASN, 0);
  dest = mk_id(cp);
  A_DESTP(asn, dest);
  A_SRCP(asn, astnew);
  add_stmt_before(asn, std);
  delete_stmt(std);

  /*	  pghpf_comm_free_(cp) */
  /* free communication schedules */
  if (!nd)
    return;

  argt = mk_argt(2);
  ARGT_ARG(argt, 0) = astb.i1;
  ARGT_ARG(argt, 1) = mk_id(cp);
  ast = mk_stmt(A_CALL, 0);
  atp = mk_id(sym_mkfunc(mkRteRtnNm(RTE_comm_free), DT_NONE));
  A_LOPP(ast, atp);
  NODESCP(A_SPTRG(A_LOPG(ast)), 1);
  A_ARGCNTP(ast, 2);
  A_ARGSP(ast, argt);
  add_stmt_after(ast, freestd);
}
#endif

/*  This routine will give outvalue   */

void
rt_outvalue(void)
{
  int std, stdnext;
  int ast;

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    if (STD_DELETE(std))
      continue;
    gbl.lineno = STD_LINENO(std);
    arg_gbl.std = std;
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_HLOCALIZEBNDS:
      outvalue_hlocalizebnds(ast);
      break;
    case A_HCYCLICLP:
      outvalue_hcycliclp(ast);
      break;
    default:
      break;
    }
  }
}

/* This part is used mostly to use lhs array for same case */
static void
eliminate_redundant(void)
{
  int std, stdnext;
  int ast;
  int type;
  int nd, nd1, nd2, nd3;
  int commstd, commasn, comm;
  int start;
  int allocstd, alloc;
  int sectlstd, sectlasn, sectl;
  LOGICAL lhs_used;
  int forall;
  int rt, rt_std;
  int lhs, rhs;
  int k;
  int lhs_sptr, alloc_sptr;
  LOGICAL independent;

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    gbl.lineno = STD_LINENO(std);
    arg_gbl.std = std;
    ast = STD_AST(std);
    lhs_used = FALSE;
    if (A_TYPEG(ast) == A_FORALL) {
      forall = ast;
      lhs = A_DESTG(A_IFSTMTG(forall));
      nd = A_OPT1G(forall);
      if (FT_NRT(nd) == 0)
        continue;
      for (k = 0; k < FT_NRT(nd); k++) {
        rt_std = glist(FT_RTL(nd), k);
        rt = STD_AST(rt_std);
        type = A_TYPEG(rt);
        if (type == A_ASN)
          type = A_TYPEG(A_SRCG(rt));
        if (type == A_HCSTART) {
          open_pragma(STD_LINENO(std));
          independent = (flg.x[19] & 0x100) != 0;
          close_pragma();
          start = A_SRCG(rt);
          nd1 = A_OPT1G(start);
          if (STD_DELETE(rt_std))
            continue;
          if (FT_CSTART_INVMVD(nd1))
            continue;

          /* try to use lhs for copy_section */
          if (FT_CSTART_REUSE(nd1) == 0 && FT_CSTART_TYPE(nd1) == A_HCOPYSECT) {
            commstd = FT_CSTART_COMM(nd1);
            if (lhs_used)
              continue;
            commasn = STD_AST(commstd);
            comm = A_SRCG(commasn);
            nd2 = A_OPT1G(comm);
            rhs = FT_CCOPY_RHS(nd2);
            if (!FT_SECTL(nd))
              continue;
            if (!is_use_lhs_final(rhs, forall, TRUE, independent, std))
              continue;
            if (FT_CCOPY_NOTLHS(nd2))
              continue;
            allocstd = FT_CCOPY_ALLOC(nd2);
            alloc = STD_AST(allocstd);
            alloc_sptr = A_SPTRG(A_LOPG(A_LOPG(alloc)));
            lhs_sptr = memsym_of_ast(lhs);
            if (rank_of_sym(alloc_sptr) != rank_of_sym(lhs_sptr))
              continue;
            nd3 = A_OPT1G(alloc);
            if (FT_ALLOC_REUSE(nd3) == 0) {
              STD_DELETE(allocstd) = 1;
              DESCUSEDP(FT_ALLOC_SPTR(nd3), 0);
              NODESCP(FT_ALLOC_SPTR(nd3), 1);
            }
            sectlstd = FT_CCOPY_SECTL(nd2);
            sectlasn = STD_AST(sectlstd);
            sectl = A_SRCG(sectlasn);
            nd3 = A_OPT1G(sectl);
            if (FT_SECT_REUSE(nd3) == 0)
              STD_DELETE(sectlstd) = 1;
            lhs_used = TRUE;
            FT_CCOPY_LHS(nd2) = lhs;
            FT_CCOPY_USELHS(nd2) = lhs;
            FT_CCOPY_SECTL(nd2) = FT_SECTL(nd);
            A_DESTP(start, lhs);
            FT_CSTART_USELHS(nd1) = lhs;
            FT_CSTART_SECTL(nd1) = FT_SECTL(nd);
          }

          /* try to lhs for gatherx */
          if (FT_CSTART_REUSE(nd1) == 0 && FT_CSTART_TYPE(nd1) == A_HGATHER) {
            commstd = FT_CSTART_COMM(nd1);
            if (lhs_used)
              continue;
            commasn = STD_AST(commstd);
            comm = A_SRCG(commasn);
            nd2 = A_OPT1G(comm);
            rhs = FT_CGATHER_RHS(nd2);
            if (!FT_SECTL(nd))
              continue;
            if (!is_use_lhs_final(rhs, forall, FALSE, independent, std))
              continue;
            if (FT_CGATHER_NOTLHS(nd2))
              continue;
            allocstd = FT_CGATHER_ALLOC(nd2);
            alloc = STD_AST(allocstd);
            alloc_sptr = A_SPTRG(A_LOPG(A_LOPG(alloc)));
            lhs_sptr = memsym_of_ast(lhs);
            if (rank_of_sym(alloc_sptr) != rank_of_sym(lhs_sptr))
              continue;
            nd3 = A_OPT1G(alloc);
            if (FT_ALLOC_REUSE(nd3) == 0) {
              STD_DELETE(allocstd) = 1;
              DESCUSEDP(FT_ALLOC_SPTR(nd3), 0);
              NODESCP(FT_ALLOC_SPTR(nd3), 1);
            }
            sectlstd = FT_CGATHER_SECTNVSUB(nd2);
            sectlasn = STD_AST(sectlstd);
            sectl = A_SRCG(sectlasn);
            nd3 = A_OPT1G(sectl);
            if (FT_SECT_REUSE(nd3) == 0)
              STD_DELETE(sectlstd) = 1;
            lhs_used = TRUE;
            FT_CGATHER_LHS(nd2) = lhs;
            FT_CGATHER_USELHS(nd2) = lhs;
            FT_CGATHER_SECTNVSUB(nd2) = FT_SECTL(nd);
            A_DESTP(start, lhs);
            FT_CSTART_USELHS(nd1) = lhs;
            FT_CSTART_SECTR(nd1) = FT_SECTL(nd);
          }
        }
      }

      if (lhs_used == FALSE) {
        sectlstd = FT_SECTL(nd);
        if (sectlstd == 0)
          continue;
        sectlasn = STD_AST(sectlstd);
        sectl = A_SRCG(sectlasn);
        nd1 = A_OPT1G(sectl);
        if (FT_SECT_REUSE(nd1) == 0)
          STD_DELETE(sectlstd) = 1;
      }
    }
  }
}

/* same_nidx: gatherx does not requires same number of idx */
LOGICAL
is_use_lhs_final(int a, int forall, LOGICAL same_nidx, LOGICAL independent,
                 int std)
{
  int lhs;
  int sptr, sptr_lhs;
  int list;
  int src;
  int asn;
  int forall1, fusedstd, asn1, header;
  int sptr_lhs1, i, nd, nd1;

  sptr = memsym_of_ast(a);
  asn = A_IFSTMTG(forall);
  lhs = A_DESTG(asn);
  list = A_LISTG(forall);
  src = A_SRCG(A_IFSTMTG(forall));
  sptr_lhs = memsym_of_ast(lhs);
  if (!independent && expr_dependent(src, lhs, std, std))
    return FALSE;
  if (A_IFEXPRG(forall))
    return FALSE;
  if (same_nidx)
    if (!is_same_number_of_idx(lhs, a, list))
      return FALSE;
  /* lhs should not used other lhs in the fused loop */
  /* lhs should not used other fused loops is used at rhs */
  nd = A_OPT1G(forall);
  header = FT_HEADER(nd);
  forall1 = STD_AST(header);
  if (forall != forall1) {
    asn1 = A_IFSTMTG(forall1);
    sptr_lhs1 = memsym_of_ast(A_DESTG(asn1));
    if (sptr_lhs == sptr_lhs1)
      return FALSE;
    if (expr_dependent(A_SRCG(asn1), lhs, header, std))
      return FALSE;
  }
  nd1 = A_OPT1G(forall1);
  for (i = 0; i < FT_NFUSE(nd1, 0); i++) {
    fusedstd = FT_FUSEDSTD(nd1, 0, i);
    forall1 = STD_AST(fusedstd);
    if (forall != forall1) {
      asn1 = A_IFSTMTG(forall1);
      sptr_lhs1 = memsym_of_ast(A_DESTG(asn1));
      if (sptr_lhs == sptr_lhs1)
        return FALSE;
      if (expr_dependent(A_SRCG(asn1), lhs, fusedstd, std))
        return FALSE;
    }
  }
  return TRUE;
}

void
outvalue_hlocalizebnds(int ast)
{
  int otriple;
  int lb, ub;

  assert(A_TYPEG(ast) == A_HLOCALIZEBNDS,
         "outvalue_hlocalizebnds: wrong ast type", 2, ast);
  otriple = A_OTRIPLEG(ast);
  if (otriple)
    return;
  lb = mk_id(trans_getbound(0, 0));
  ub = mk_id(trans_getbound(0, 1));
  otriple = mk_triple(lb, ub, 0);
  A_OTRIPLEP(ast, otriple);
}

void
outvalue_hcycliclp(int ast)
{
  int otriple;
  int cl, cu, cs;
  int lof0, los, lof;

  assert(A_TYPEG(ast) == A_HCYCLICLP, "outvalue_hcycliclp: wrong ast type", 2,
         ast);
  otriple = A_OTRIPLEG(ast);
  if (otriple)
    return;
  cl = mk_id(trans_getbound(0, 3));
  cu = mk_id(trans_getbound(0, 4));
  cs = mk_id(trans_getbound(0, 5));
  otriple = mk_triple(cl, cu, cs);
  A_OTRIPLEP(ast, otriple);

  lof0 = mk_id(trans_getbound(0, 6));
  los = mk_id(trans_getbound(0, 7));
  lof = mk_id(trans_getbound(0, 6));

  otriple = mk_triple(lof0, los, lof);
  A_OTRIPLE1P(ast, otriple);
}

/* put out the localize bounds call for this dimension
 * call pghpf_localize_bounds(
 *   aln, dim, lower, upper, stride, newlower, newupper)
 *		)
 */

void
generate_hlocalizebnds(int ast)
{
  int itriple, otriple;
  int argt, nargs;
  int forall, idx;
  int nd;
  int std, stdnew;
  int astnew, astmem, lop;
  int same;
  int st;
  int lb, ub, i1, i2, lhs, sptr_lhs, descr;
  int dim;

  assert(A_TYPEG(ast) == A_HLOCALIZEBNDS,
         "generate_hlocalizebnds: wrong ast type", 2, ast);
  nd = A_OPT1G(ast);
  /*	  assert(nd, "generate_hlocalizebnds: some thing is wrong", 2, ast);*/
  std = A_STDG(ast);
  if (STD_DELETE(std)) {
    assert(nd, "generate_hlocalizebnds: some thing is wrong", 2, ast);
    same = FT_BND_SAME(nd);
    A_OTRIPLEP(ast, A_OTRIPLEG(same));
  }
  itriple = A_ITRIPLEG(ast);
  otriple = A_OTRIPLEG(ast);
  if (nd) { /* for do-independent */
    forall = STD_AST(FT_STD(nd));
    assert(A_TYPEG(forall) == A_FORALL,
           "generate_hlocalizebnds: some thing is wrong", 2, ast);
    idx = FT_BND_IDX(nd);
    change_forall_triple(forall, idx, otriple, FALSE);
  }
  if (STD_DELETE(std)) {
    delete_stmt(A_STDG(ast));
    return;
  }

  st = A_STRIDEG(itriple);
  if (st == 0)
    st = mk_isz_cval(1, astb.bnd.dtype);
  lb = A_LBDG(itriple);
  ub = A_UPBDG(itriple);
  i1 = A_LBDG(otriple);
  i2 = A_UPBDG(otriple);

  dim = get_int_cval(A_SPTRG(A_DIMG(ast))) - 1;

  sptr_lhs = 0;
  if (nd) {
    lhs = FT_BND_LHS(nd);
    astmem = lhs;
    sptr_lhs = memsym_of_ast(lhs);
    descr = memsym_of_ast(A_LOPG(ast));
  } else {
    sptr_lhs = memsym_of_ast(A_LOPG(ast));
    astmem = A_LOPG(ast);
    descr = DESCRG(sptr_lhs);
    assert(STYPEG(descr) == ST_ARRDSC,
           "generate_hlocalizebnds: missing descriptor", ast, 4);
  }
  if (A_TYPEG(astmem) == A_SUBSCR)
    astmem = A_LOPG(astmem);
  if (st == astb.bnd.one && !XBIT(47, 0x80)) {
    inline_hlocalizebnds(i1, i2, lb, ub, sptr_lhs, descr, dim, std, astmem);
    delete_stmt(A_STDG(ast));
    return;
  }

  if (normalize_bounds(sptr_lhs)) {
    lb = mk_bnd_int(lb);
    lb = sub_lbnd(DTYPEG(sptr_lhs), dim, lb, astmem);
    ub = mk_bnd_int(ub);
    ub = sub_lbnd(DTYPEG(sptr_lhs), dim, ub, astmem);
  }

  nargs = 7;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = check_member(astmem, mk_id(descr));
  ARGT_ARG(argt, 1) = A_DIMG(ast);
  ARGT_ARG(argt, 2) = lb;
  ARGT_ARG(argt, 3) = ub;
  ARGT_ARG(argt, 4) = mk_bnd_int(st);
  otriple = A_OTRIPLEG(ast);
  ARGT_ARG(argt, 5) = mk_bnd_int(i1);
  ARGT_ARG(argt, 6) = mk_bnd_int(i2);
  astnew = mk_stmt(A_CALL, 0);
  lop = mk_id(sym_mkfunc(mkRteRtnNm(RTE_localize_bounds), DT_NONE));
  A_LOPP(astnew, lop);
  A_ARGCNTP(astnew, nargs);
  A_ARGSP(astnew, argt);
  stdnew = add_stmt_before(astnew, std);
  FT_BND_CALL(nd) = stdnew;

  if (normalize_bounds(sptr_lhs)) {
    astnew = add_lbnd(DTYPEG(sptr_lhs), dim, i1, astmem);
    astnew = mk_assn_stmt(i1, astnew, astb.bnd.dtype);
    add_stmt_before(astnew, std);

    astnew = add_lbnd(DTYPEG(sptr_lhs), dim, i2, astmem);
    astnew = mk_assn_stmt(i2, astnew, astb.bnd.dtype);
    add_stmt_before(astnew, std);
  }

  delete_stmt(A_STDG(ast));
}

/* Algorithm:
 *    call pghpf_cyclic_loop(d_a, dim, l, u, s, cl, cu, cs, lof0, los)
 */

void
generate_hcycliclp(int ast)
{
  int itriple, otriple, otriple1, l, u, lhs, lop;
  int argt, nargs;
  int forall, idx;
  int nd;
  int std, stdnew;
  int astnew, astmem;
  int otripleb;
  int same;
  int st;
  int sptr_lhs, descr, descrast;
  int dim;

  assert(A_TYPEG(ast) == A_HCYCLICLP, "generate_hcycliclp: wrong ast type", 2,
         ast);
  nd = A_OPT1G(ast);
  std = A_STDG(ast);
  if (STD_DELETE(std)) {
    assert(nd, "generate_hcycliclp: some thing is wrong", 2, ast);
    same = FT_BND_SAME(nd);
    A_OTRIPLEP(ast, A_OTRIPLEG(same));
    A_OTRIPLE1P(ast, A_OTRIPLE1G(same));
  }
  itriple = A_ITRIPLEG(ast);
  otriple = A_OTRIPLEG(ast);
  otriple1 = A_OTRIPLE1G(ast);

  if (nd) {
    /* HCYCLICLP generated from FORALL statement. */
    forall = STD_AST(FT_STD(nd));
    assert(A_TYPEG(forall) == A_FORALL,
           "generate_hlocalizebnds: some thing is wrong", 2, ast);
    idx = FT_BND_IDX(nd);
    if (A_CBLKG(ast) == 0) {
      fill_cyclic_1(ast);
      change_forall_triple(forall, idx, otriple, TRUE);
    } else {
      otripleb = fill_cyclic_k(ast);
      change_forall_triple(forall, idx, otripleb, FALSE);
    }

    if (STD_DELETE(std)) {
      delete_stmt(A_STDG(ast));
      return;
    }
    lhs = FT_BND_LHS(nd);
    sptr_lhs = memsym_of_ast(lhs);
    astmem = A_LOPG(lhs);
    descrast = A_LOPG(ast);
  } else {
    sptr_lhs = memsym_of_ast(A_LOPG(ast));
    astmem = A_LOPG(ast);
    assert(STYPEG(sptr_lhs) == ST_ARRAY, "generate_hcycliclp: missing array",
           ast, 4);
    descr = DESCRG(sptr_lhs);
    descrast = check_member(astmem, mk_id(descr));
    assert(STYPEG(descr) == ST_ARRDSC, "generate_hcycliclp: missing descriptor",
           ast, 4);
  }

  l = mk_default_int(A_LBDG(itriple));
  u = mk_default_int(A_UPBDG(itriple));
  if (normalize_bounds(sptr_lhs)) {
    dim = get_int_cval(A_SPTRG(A_DIMG(ast))) - 1;
    assert(0 <= dim && dim < 7, "generate_hcycliclp: bad dim", dim, 4);
    l = sub_lbnd(DTYPEG(sptr_lhs), dim, l, astmem);
    u = sub_lbnd(DTYPEG(sptr_lhs), dim, u, astmem);
  }

  nargs = 10;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = descrast;
  ARGT_ARG(argt, 1) = A_DIMG(ast);
  ARGT_ARG(argt, 2) = l;
  ARGT_ARG(argt, 3) = u;
  st = A_STRIDEG(itriple);
  if (st == 0)
    st = mk_isz_cval(1, astb.bnd.dtype);
  ARGT_ARG(argt, 4) = mk_bnd_int(st);
  otriple = A_OTRIPLEG(ast);
  ARGT_ARG(argt, 5) = mk_bnd_int(A_LBDG(otriple));
  ARGT_ARG(argt, 6) = mk_bnd_int(A_UPBDG(otriple));
  ARGT_ARG(argt, 7) = mk_bnd_int(A_STRIDEG(otriple));
  otriple1 = A_OTRIPLE1G(ast);
  ARGT_ARG(argt, 8) = mk_bnd_int(A_LBDG(otriple1));
  ARGT_ARG(argt, 9) = mk_bnd_int(A_UPBDG(otriple1));
  astnew = mk_stmt(A_CALL, 0);
  lop = mk_id(sym_mkfunc(mkRteRtnNm(RTE_cyclic_loop), DT_NONE));
  A_LOPP(astnew, lop);
  A_ARGCNTP(astnew, nargs);
  A_ARGSP(astnew, argt);
  stdnew = add_stmt_before(astnew, std);
  FT_BND_CALL(nd) = stdnew;

  if (normalize_bounds(sptr_lhs))
    if (nd && !A_CBLKG(ast)) {
      l = A_LBDG(otriple);
      astnew = add_lbnd(DTYPEG(sptr_lhs), dim, l, astmem);
      astnew = mk_assn_stmt(l, astnew, astb.bnd.dtype);
      add_stmt_before(astnew, std);

      u = A_UPBDG(otriple);
      astnew = add_lbnd(DTYPEG(sptr_lhs), dim, u, astmem);
      astnew = mk_assn_stmt(u, astnew, astb.bnd.dtype);
      add_stmt_before(astnew, std);
    }

  delete_stmt(A_STDG(ast));
}

static void
change_forall_triple(int forall, int idx, int otriple, LOGICAL is_cyclic)
{
  int list, listp, newlist;
  int lb, ub, st;
  int triple;

  list = A_LISTG(forall);

  /* go through & fixup the forall */
  start_astli();
  for (listp = list; listp != 0; listp = ASTLI_NEXT(listp)) {
    newlist = add_astli();
    if (ASTLI_SPTR(idx) == ASTLI_SPTR(listp)) {
      lb = A_LBDG(otriple);
      ub = A_UPBDG(otriple);
      if (is_cyclic)
        st = A_STRIDEG(otriple);
      else
        st = A_STRIDEG(ASTLI_TRIPLE(listp));
      triple = mk_triple(lb, ub, st);
      ASTLI_SPTR(newlist) = ASTLI_SPTR(listp);
      ASTLI_TRIPLE(newlist) = triple;
    } else {
      /* don't need to change this one */
      ASTLI_SPTR(newlist) = ASTLI_SPTR(listp);
      ASTLI_TRIPLE(newlist) = ASTLI_TRIPLE(listp);
    }
  }
  A_LISTP(forall, ASTLI_HEAD);
}

/* Algorithm:
 * cyclic(1) distribution with perfect alignment.
 *
 * chpf$ distribute t(cyclic) onto p
 * chpf$ align a(i) with t(i)
 *
 *    call pghpf_cyclic_loop(d_a, dim, l, u, s, cl, cu, cs, lof0, los)
 *     lof = lof0
 *     do i = cl, cu, cs
 *	  a(i-lof) = ... rhs(i) ...
 *	  lof = lof + los
 *     end do
 *
 * Variable names has the same meaning in the comment and code.
 */

static void
fill_cyclic_1(int ast)
{
  int forall;
  int nd, nd1;
  CTYPE *ct;
  int astnew;
  int itriple, otriple, otriple1;
  int lof, lof0, los;
  int dim, cdim, rhs;

  assert(A_TYPEG(ast) == A_HCYCLICLP, "fill_cyclic_1: wrong ast type", 2, ast);

  nd = A_OPT1G(ast);
  forall = STD_AST(FT_STD(nd));
  assert(A_TYPEG(forall) == A_FORALL, "fill_cyclic_1: some thing is wrong", 2,
         ast);
  nd1 = A_OPT1G(forall);
  ct = FT_CYCLIC(nd1);

  itriple = A_ITRIPLEG(ast);
  otriple = A_OTRIPLEG(ast);
  otriple1 = A_OTRIPLE1G(ast);
  lof0 = A_LBDG(otriple1);
  los = A_UPBDG(otriple1);
  lof = A_STRIDEG(otriple1);
  dim = A_DIMG(ast);
  cdim = get_int_cval(A_SPTRG(A_ALIASG(dim))) - 1;

  ct->c_lof[cdim] = lof;

  astnew = mk_stmt(A_ASN, DT_INT);
  A_DESTP(astnew, lof);
  A_SRCP(astnew, lof0);
  ct->c_init[cdim] = astnew;

  astnew = mk_stmt(A_ASN, DT_INT);
  A_DESTP(astnew, lof);
  rhs = mk_binop(OP_ADD, lof, los, DT_INT);
  A_SRCP(astnew, rhs);
  ct->c_inc[cdim] = astnew;
}

/* Algorithm:
 *    call pghpf_cyclic_loop(d_a, dim, l, u, s, cl, cu, cs, lof0, los)
 *    do ci = cl, cu, cs
 *	  lof = lof0
 *	  call pghpf_block_loop(d_a, dim, l, u, s, ci, bl, bu, bs)
 *	  do i = bl, bu, bs
 *	     a(i-lof) = ... rhs(i) ...
 *	  end do
 *	  lof = lof + los
 *     end do
 *
 *    Variable names has the same meaning in the comment and code.
 */

static int
fill_cyclic_k(int ast)
{
  int lb, ub;
  int l, u, s;
  int argt;
  int cl, cu, cs, ci;
  int lof, los, lof0;
  int astnew;
  int nargs;
  CTYPE *ct;
  int nd, nd1;
  int itriple, otriple, otriple1;
  int dim, cdim;
  int forall, lop, src;

  assert(A_TYPEG(ast) == A_HCYCLICLP, "fill_cyclic_k: wrong ast type", 2, ast);

  nd = A_OPT1G(ast);
  forall = STD_AST(FT_STD(nd));
  assert(A_TYPEG(forall) == A_FORALL, "fill_cyclic_k: some thing is wrong", 2,
         ast);
  nd1 = A_OPT1G(forall);
  ct = FT_CYCLIC(nd1);

  itriple = A_ITRIPLEG(ast);
  l = A_LBDG(itriple);
  u = A_UPBDG(itriple);
  s = A_STRIDEG(itriple);
  if (s == 0)
    s = mk_cval(1, DT_INT);
  otriple = A_OTRIPLEG(ast);
  cl = A_LBDG(otriple);
  cu = A_UPBDG(otriple);
  cs = A_STRIDEG(otriple);

  otriple1 = A_OTRIPLE1G(ast);
  lof0 = A_LBDG(otriple1);
  los = A_UPBDG(otriple1);
  lof = A_STRIDEG(otriple1);
  dim = A_DIMG(ast);
  cdim = get_int_cval(A_SPTRG(A_ALIASG(dim))) - 1;

  ct->c_lof[cdim] = lof;

  astnew = mk_stmt(A_ASN, DT_INT);
  A_DESTP(astnew, lof);
  A_SRCP(astnew, lof0);
  ct->cb_init[cdim] = astnew;

  astnew = mk_stmt(A_DO, 0);
  ci = mk_id(trans_getbound(0, 8));
  A_DOVARP(astnew, ci);
  A_M1P(astnew, cl);
  A_M2P(astnew, cu);
  A_M3P(astnew, cs);
  A_M4P(astnew, 0);
  ct->cb_do[cdim] = astnew;

  lb = mk_id(trans_getbound(0, 0));
  ub = mk_id(trans_getbound(0, 1));

  nargs = 8;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = A_LOPG(ast);
  /*    DESCUSEDP(lhs, 1); */

  ARGT_ARG(argt, 1) = mk_cval(cdim + 1, DT_INT);
  ARGT_ARG(argt, 2) = mk_default_int(l);
  ARGT_ARG(argt, 3) = mk_default_int(u);
  ARGT_ARG(argt, 4) = mk_default_int(s);
  ARGT_ARG(argt, 5) = mk_default_int(ci);
  ARGT_ARG(argt, 6) = mk_default_int(lb);
  ARGT_ARG(argt, 7) = ub;
  astnew = mk_stmt(A_CALL, 0);
  lop = mk_id(sym_mkfunc(mkRteRtnNm(RTE_block_loop), DT_NONE));
  A_LOPP(astnew, lop);
  A_ARGCNTP(astnew, nargs);
  A_ARGSP(astnew, argt);
  ct->cb_block[cdim] = astnew;

  astnew = mk_stmt(A_ASN, DT_INT);
  A_DESTP(astnew, lof);
  src = mk_binop(OP_ADD, lof, los, DT_INT);
  A_SRCP(astnew, src);
  ct->cb_inc[cdim] = astnew;

  astnew = mk_stmt(A_ENDDO, 0);
  ct->cb_enddo[cdim] = astnew;
  return mk_triple(lb, ub, 0);
}

#ifdef FLANG_COMMGEN_UNUSED
static void
generate_forall(int ast)
{
  int std;
  int lhs;
  int asn;
  int expr;
  int src;

  std = A_STDG(ast);
  asn = A_IFSTMTG(ast);
  src = A_SRCG(asn);
  src = rhs_cyclic(src, std, 0);
  A_SRCP(asn, src);
  expr = A_IFEXPRG(ast);
  if (expr)
    expr = rhs_cyclic(expr, std, 1);
  A_IFSTMTP(ast, asn);
  A_IFEXPRP(ast, expr);

  asn = A_IFSTMTG(ast);
  lhs = A_DESTG(asn);
  A_DESTP(asn, lhs);
  A_IFSTMTP(ast, asn);
}

static int
rhs_cyclic(int ast, int std, int ifexpr)
{
  int l, r, d, o;
  int l1, l2, l3;
  int a;
  int i, nargs, argt;
  int forall;
  int asd;
  int ndim;
  int subs[7];

  a = ast;
  if (!a)
    return a;
  forall = STD_AST(std);
  switch (A_TYPEG(ast)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = rhs_cyclic(A_LOPG(a), std, ifexpr);
    r = rhs_cyclic(A_ROPG(a), std, ifexpr);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(a);
    d = A_DTYPEG(a);
    l = rhs_cyclic(A_LOPG(a), std, ifexpr);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(a);
    l = rhs_cyclic(A_LOPG(a), std, ifexpr);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(a);
    l = rhs_cyclic(A_LOPG(a), std, ifexpr);
    return mk_paren(l, d);
  case A_MEM:
    l = rhs_cyclic(A_PARENTG(a), std, ifexpr);
    r = A_MEMG(a);
    d = A_DTYPEG(r);
    return mk_member(l, r, d);
  case A_SUBSTR:
    return a;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(a);
    argt = A_ARGSG(a);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = rhs_cyclic(ARGT_ARG(argt, i), std, ifexpr);
    }
    return a;
  case A_CNST:
  case A_CMPLXC:
    return a;
  case A_TRIPLE:
    l1 = rhs_cyclic(A_LBDG(a), std, ifexpr);
    l2 = rhs_cyclic(A_UPBDG(a), std, ifexpr);
    l3 = rhs_cyclic(A_STRIDEG(a), std, ifexpr);
    return mk_triple(l1, l2, l3);
  case A_ID:
    return a;
  case A_SUBSCR:
    asd = A_ASDG(a);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++)
      subs[i] = rhs_cyclic(ASD_SUBS(asd, i), std, ifexpr);
    a = mk_subscr(A_LOPG(a), subs, ndim, A_DTYPEG(a));
    return a;
  default:
    interr("rhs_cyclic: unknown expression", std, 2);
    return 0;
  }
}
#endif

int
gen_localize_index(int sptr, int dim, int subAst, int astmem)
{
  int nargs, argt;
  int astnew;

  /* localize this subscript */
  /* pghpf_localize_index(aln, dim, subAst) */
  nargs = 3;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = check_member(astmem, mk_id(DESCRG(sptr)));
  DESCUSEDP(sptr, 1);
  ARGT_ARG(argt, 1) = mk_isz_cval(dim + 1, astb.bnd.dtype);
  astnew = mk_bnd_int(subAst);
  if (normalize_bounds(sptr))
    astnew = sub_lbnd(DTYPEG(sptr), dim, astnew, astmem);
  ARGT_ARG(argt, 2) = astnew;
  astnew = mk_func_node(
      A_FUNC, mk_id(sym_mkfunc(mkRteRtnNm(RTE_localize_index), DT_INT)), nargs,
      argt);
  NODESCP(A_SPTRG(A_LOPG(astnew)), 1);
  A_DTYPEP(astnew, DT_INT);
  if (normalize_bounds(sptr))
    astnew = add_lbnd(DTYPEG(sptr), dim, astnew, astmem);
  return astnew;
}

/*
 * These values are only meaningful for cyclic and block-cyclic section
 * dimensions.	For whole arrays and block-distributed section dimensions,
 * sstride = 1 and soffset = 0.
 * You will need to change the code where you generate cyclic index
 * references to a section.  This would be for F90 pointers and maybe
 * dummy arguments.  You do not need to change cyclic index references to
 * whole arrays.
 * now you must generate
 *	  (i*sstride + soffset - clof)
 */
static int
cyclic_section(int sptr, int idxAst, int clofAst, int i, int memberast)
{
  int sdsc;
  int sstride, soffset;
  int ast, ast1, ast2;

  sdsc = DESCRG(sptr);
  assert(sdsc, "cyclic_section: no descriptor", sptr, 4);
  sstride = check_member(memberast, get_section_stride(sdsc, i));
  soffset = check_member(memberast, get_section_offset(sdsc, i));
  ast = mk_binop(OP_MUL, idxAst, sstride, astb.bnd.dtype);
  if (clofAst)
    ast1 = mk_binop(OP_SUB, soffset, clofAst, astb.bnd.dtype);
  else
    ast1 = soffset;
  ast2 = mk_binop(OP_ADD, ast, ast1, astb.bnd.dtype);
  return ast2;
}

#ifdef FLANG_COMMGEN_UNUSED
/* to check two sptr lower bounds are the same at
 * at the given dimension.
 */
static LOGICAL
is_same_lower_dim(int sptr, int dim, int sptr1, int dim1)
{
  ADSC *ad;
  ADSC *ad1;
  int lb, lb1;

  assert(DTY(DTYPEG(sptr)) == TY_ARRAY && DTY(DTYPEG(sptr1)) == TY_ARRAY,
         "is_same_lower_dim: must be array", sptr, 4);
  if (sptr != sptr1 && (POINTERG(sptr) || POINTERG(sptr1)))
    return FALSE;
  ad = AD_DPTR(DTYPEG(sptr));
  ad1 = AD_DPTR(DTYPEG(sptr1));

  lb = AD_LWAST(ad, dim);
  lb1 = AD_LWAST(ad1, dim1);
  if (lb == lb1)
    return TRUE;
  return FALSE;
}

static int
rewrite_expr(int expr, int a, int b)
{

  int l, r, d, o;
  int i, nargs, argt;

  if (expr == a)
    return b;
  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = rewrite_expr(A_LOPG(expr), a, b);
    r = rewrite_expr(A_ROPG(expr), a, b);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = rewrite_expr(A_LOPG(expr), a, b);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(expr);
    l = rewrite_expr(A_LOPG(expr), a, b);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(expr);
    l = rewrite_expr(A_LOPG(expr), a, b);
    return mk_paren(l, d);
  case A_MEM:
    l = rewrite_expr(A_PARENTG(expr), a, b);
    r = A_MEMG(expr);
    d = A_DTYPEG(r);
    return mk_member(l, r, d);
  case A_SUBSTR:
    interr("rewrite_expr: strings not supported", expr, 2);
    return 0;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) = rewrite_expr(ARGT_ARG(argt, i), a, b);
    }
    return expr;
  case A_CNST:
  case A_CMPLXC:
  case A_ID:
  case A_SUBSCR:
    return expr;
  default:
    interr("rewrite_expr: unknown expression", expr, 2);
    return expr;
  }
}
#endif

/* This routine is to emit bounds calculation for BLOCK distribution
 * stride==1
 *  i1= max(l,olb)
 *  i2= min(u,oub)
 */

int
inline_hlocalizebnds(int i1, int i2, int lb, int ub, int sptr_lhs, int descr,
                     int dim, int std, int astmem)
{
  int olb;
  int oub;
  int asn;
  int max, min;
  ADSC *ad;
  int l, u;
  int stdnext;

  assert(A_TYPEG(i1) == A_ID, "inline_hlocalizebnds: not A_ID", i1, 4);
  assert(A_TYPEG(i2) == A_ID, "inline_hlocalizebnds: not A_ID", i2, 4);
  assert(dim >= 0 && dim <= 7, "inline_hlocalizebnds: illegal dim", dim, 4);

  /* find array lower and upper bound */
  lb = mk_default_int(lb);
  ub = mk_default_int(ub);
  if (sptr_lhs) {
    DESCUSEDP(sptr_lhs, 1);
    assert(is_array_type(sptr_lhs), "inline_hlocalizebnds: not an array",
           sptr_lhs, 4);
    ad = AD_DPTR(DTYPEG(sptr_lhs));
    l = AD_LWAST(ad, dim);
    if (!l)
      l = astb.bnd.one;
    u = AD_UPAST(ad, dim);
  }

  olb = check_member(astmem, get_owner_lower(descr, dim));
  oub = check_member(astmem, get_owner_upper(descr, dim));
  if (sptr_lhs && normalize_bounds(sptr_lhs)) {
    olb = mk_binop(OP_ADD, olb, l, astb.bnd.dtype);
    olb = mk_binop(OP_SUB, olb, astb.bnd.one, astb.bnd.dtype);
    oub = mk_binop(OP_ADD, oub, l, astb.bnd.dtype);
    oub = mk_binop(OP_SUB, oub, astb.bnd.one, astb.bnd.dtype);
  }
  if (sptr_lhs && l == lb)
    max = olb;
  else
    max = gen_minmax(lb, olb, I_MAX);
  if (sptr_lhs && u == ub)
    min = oub;
  else
    min = gen_minmax(ub, oub, I_MIN);
  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  A_DESTP(asn, i1);
  A_SRCP(asn, max);
  stdnext = add_stmt_before(asn, std);
  STD_LINENO(stdnext) = STD_LINENO(std);
  STD_LOCAL(stdnext) = 1;
  stdnext = STD_NEXT(stdnext);
  asn = mk_stmt(A_ASN, astb.bnd.dtype);
  A_DESTP(asn, i2);
  A_SRCP(asn, min);
  stdnext = add_stmt_before(asn, stdnext);
  STD_LINENO(stdnext) = STD_LINENO(std);
  STD_LOCAL(stdnext) = 1;
  stdnext = STD_NEXT(stdnext);
  return stdnext;
}

/* type: I_MAX or I_MIN */
static int
gen_minmax(int astl, int astr, int type)
{
  int astminmax;
  int dl, dr;

  if (astl == astr)
    return astl;
  dl = A_DTYPEG(astl);
  dr = A_DTYPEG(astr);
  if (DT_ISINT(dl) && dl != dr) {
    if (DTY(dl) < DTY(dr))
      astl = ast_intr(I_INT, dr, 1, astl);
    else
      astr = ast_intr(I_INT, dl, 1, astr);
  }
  astminmax = ast_intr(type, A_DTYPEG(astl), 2, astl, astr);
  return astminmax;
}

static void
pointer_changer(void)
{
  int std, stdnext;
  int ast;

  for (std = STD_NEXT(0); std; std = stdnext) {
    stdnext = STD_NEXT(std);
    if (STD_DELETE(std))
      continue;
    gbl.lineno = STD_LINENO(std);
    arg_gbl.std = std;
    ast = STD_AST(std);
    ast = pointer_squeezer(ast);
    STD_AST(std) = ast;
    A_STDP(ast, std);
  }
}

static int
pointer_squeezer(int ast)
{
  int atype;
  int astnew;
  int opnd;
  int l, r, m1, m2, m3, m4, v;
  int dtype;
  int asd;
  int numdim;
  int subs[7];
  int argt;
  int argcnt;
  int argtnew;
  int i, any;
  int astli, astlinew;
  int sptr;
  int align;

  if (ast == 0)
    return 0; /* watch for a 'null' argument */
  dtype = A_DTYPEG(ast);
  switch (atype = A_TYPEG(ast)) {
  case A_CMPLXC:
  case A_CNST:
  case A_ID:
  case A_LABEL:
    astnew = ast;
    break;
  case A_MEM:
    astnew = pointer_squeezer((int)A_PARENTG(ast));
    if (astnew == A_PARENTG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_member(astnew, A_MEMG(ast), A_DTYPEG(ast));
    }
    break;
  case A_SUBSTR:
    opnd = pointer_squeezer((int)A_LOPG(ast));
    l = pointer_squeezer((int)A_LEFTG(ast));
    r = pointer_squeezer((int)A_RIGHTG(ast));
    if (opnd == A_LOPG(ast) && l == A_LEFTG(ast) && r == A_RIGHTG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_substr(opnd, l, r, dtype);
    }
    break;
  case A_BINOP:
    l = pointer_squeezer((int)A_LOPG(ast));
    r = pointer_squeezer((int)A_ROPG(ast));
    if (l == A_LOPG(ast) && r == A_ROPG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_binop((int)A_OPTYPEG(ast), l, r, dtype);
    }
    break;
  case A_UNOP:
    l = pointer_squeezer((int)A_LOPG(ast));
    if (l == A_LOPG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_unop((int)A_OPTYPEG(ast), l, dtype);
    }
    break;
  case A_PAREN:
    l = pointer_squeezer((int)A_LOPG(ast));
    if (l == A_LOPG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_paren(l, dtype);
    }
    break;
  case A_CONV:
    l = pointer_squeezer((int)A_LOPG(ast));
    if (l == A_LOPG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_convert(l, dtype);
    }
    break;
  case A_SUBSCR:
    any = 0;
    opnd = pointer_squeezer((int)A_LOPG(ast));
    if (opnd != A_LOPG(ast))
      ++any;
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd);
    sptr = sptr_of_subscript(ast);
    align = ALIGNG(sptr);
    if (!POINTERG(sptr) || A_SHAPEG(ast)) {
      for (i = 0; i < numdim; ++i) {
        l = pointer_squeezer((int)ASD_SUBS(asd, i));
        if (l != ASD_SUBS(asd, i))
          ++any;
        subs[i] = l;
      }
      if (any == 0) {
        astnew = ast;
      } else {
        astnew = mk_subscr(opnd, subs, numdim, dtype);
      }
      break;
    }
    assert(numdim > 0 && numdim <= 7, "pointer_squeezer: bad numdim", ast, 4);
    for (i = 0; i < numdim; ++i) {
      l = pointer_squeezer((int)ASD_SUBS(asd, i));
      if (l != ASD_SUBS(asd, i))
        ++any;
      l = cyclic_section(sptr, l, 0, i, A_LOPG(ast));
      ++any;
      subs[i] = l;
    }
    if (any == 0) {
      astnew = ast;
    } else {
      astnew = mk_subscr(opnd, subs, numdim, dtype);
    }
    break;
  case A_TRIPLE:
    l = pointer_squeezer((int)A_LBDG(ast));
    r = pointer_squeezer((int)A_UPBDG(ast));
    opnd = pointer_squeezer((int)A_STRIDEG(ast));
    if (opnd == A_STRIDEG(ast) && l == A_LBDG(ast) && r == A_UPBDG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_triple(l, r, opnd);
    }
    break;
  case A_FUNC:
  case A_ICALL:
  case A_CALL:
  case A_INTR:
    any = 0;
    l = A_LOPG(ast);
    opnd = pointer_squeezer(l);
    if (opnd != l)
      ++any;
    argt = A_ARGSG(ast);
    argcnt = A_ARGCNTG(ast);
    argtnew = mk_argt(argcnt);
    for (i = 0; i < argcnt; ++i) {
      ARGT_ARG(argtnew, i) = pointer_squeezer(ARGT_ARG(argt, i));
      if (ARGT_ARG(argtnew, i) != ARGT_ARG(argt, i))
        ++any;
    }
    if (any == 0) {
      astnew = ast;
      unmk_argt(argcnt);
    } else {
      astnew = mk_func_node((int)A_TYPEG(ast), opnd, argcnt, argtnew);
      A_OPTYPEP(astnew, A_OPTYPEG(ast));
      A_SHAPEP(astnew, A_SHAPEG(ast));
      A_DTYPEP(astnew, A_DTYPEG(ast));
      A_OPT1P(astnew, A_OPT1G(ast));
      A_OPT2P(astnew, A_OPT2G(ast));
    }
    break;
  case A_ASN:
    l = pointer_squeezer(A_DESTG(ast));
    r = pointer_squeezer(A_SRCG(ast));
    if (l == A_DESTG(ast) && r == A_SRCG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_assn_stmt(l, r, dtype);
    }
    break;
  case A_IF:
    l = pointer_squeezer(A_IFEXPRG(ast));
    r = pointer_squeezer(A_IFSTMTG(ast));
    if (l == A_IFEXPRG(ast) && r == A_IFSTMTG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_stmt(A_IF, 0);
      A_IFEXPRP(astnew, l);
      A_IFSTMTP(astnew, r);
    }
    break;
  case A_IFTHEN:
  case A_ELSEIF:
    l = pointer_squeezer(A_IFEXPRG(ast));
    if (l == A_IFEXPRG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_stmt(A_TYPEG(ast), 0);
      A_IFEXPRP(astnew, l);
    }
    break;
  case A_AIF:
    l = pointer_squeezer(A_IFEXPRG(ast));
    if (l == A_IFEXPRG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_stmt(A_AIF, 0);
      A_IFEXPRP(astnew, l);
      A_L1P(astnew, A_L1G(ast));
      A_L2P(astnew, A_L2G(ast));
      A_L3P(astnew, A_L3G(ast));
    }
    break;
  case A_GOTO:
    astnew = ast;
    break;
  case A_CGOTO:
  case A_AGOTO:
    l = pointer_squeezer(A_LOPG(ast));
    if (l == A_LOPG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_stmt(A_TYPEG(ast), 0);
      A_LOPP(astnew, l);
      A_LISTP(astnew, A_LISTG(ast));
    }
    break;
  case A_ASNGOTO:
    astnew = ast;
    break;
  case A_DO:
    l = pointer_squeezer(A_DOLABG(ast));
    v = pointer_squeezer(A_DOVARG(ast));
    m1 = pointer_squeezer(A_M1G(ast));
    m2 = pointer_squeezer(A_M2G(ast));
    m3 = pointer_squeezer(A_M3G(ast));
    m4 = pointer_squeezer(A_M4G(ast));
    if (l == A_DOLABG(ast) && v == A_DOVARG(ast) && m1 == A_M1G(ast) &&
        m2 == A_M2G(ast) && m3 == A_M3G(ast) && m4 == A_M4G(ast)) {
      astnew = ast;
    } else {
      astnew = mk_stmt(A_DO, 0);
      A_DOLABP(astnew, l);
      A_DOVARP(astnew, v);
      A_M1P(astnew, m1);
      A_M2P(astnew, m2);
      A_M3P(astnew, m3);
      A_M4P(astnew, m4);
    }
    break;
  case A_DOWHILE:
    l = pointer_squeezer(A_DOLABG(ast));
    r = pointer_squeezer(A_IFEXPRG(ast));
    if (l == A_DOLABG(ast) && r == A_IFEXPRG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_stmt(A_DOWHILE, 0);
      A_DOLABP(astnew, l);
      A_IFEXPRP(astnew, r);
    }
    break;
  case A_FORALL:
    any = 0;
    start_astli();
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      astlinew = add_astli();
      ASTLI_TRIPLE(astlinew) = pointer_squeezer(ASTLI_TRIPLE(astli));
      if (ASTLI_TRIPLE(astlinew) != ASTLI_TRIPLE(astli))
        ++any;
      l = pointer_squeezer(mk_id((int)ASTLI_SPTR(astli)));
      ASTLI_SPTR(astlinew) = A_SPTRG(l);
      if (ASTLI_SPTR(astlinew) != ASTLI_SPTR(astli))
        ++any;
    }
    l = pointer_squeezer(A_IFEXPRG(ast));
    r = pointer_squeezer(A_IFSTMTG(ast));
    if (any == 0 && l == A_IFEXPRG(ast) && r == A_IFSTMTG(ast)) {
      astnew = ast;
    } else {
      astnew = mk_stmt(A_FORALL, 0);
      A_LISTP(astnew, ASTLI_HEAD);
      A_IFEXPRP(astnew, l);
      A_IFSTMTP(astnew, r);
      A_OPT1P(astnew, A_OPT1G(ast));
      A_OPT2P(astnew, A_OPT2G(ast));
    }
    break;
  case A_STOP:
  case A_PAUSE:
  case A_RETURN:
  case A_ALLOC:
  case A_WHERE:
  case A_REDIM:
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
  case A_REALIGN:
  case A_REDISTRIBUTE:
  case A_HLOCALIZEBNDS:
  case A_HALLOBNDS:
  case A_HCYCLICLP:
  case A_HOFFSET:
  case A_HSECT:
  case A_HCOPYSECT:
  case A_HPERMUTESECT:
  case A_HOVLPSHIFT:
  case A_HGETSCLR:
  case A_HGATHER:
  case A_HSCATTER:
  case A_HCSTART:
  case A_HCFINISH:
  case A_HCFREE:
  case A_HOWNERPROC:
  case A_HLOCALOFFSET:
  case A_CRITICAL:
  case A_ENDCRITICAL:
  case A_MASTER:
  case A_ENDMASTER:
  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
  case A_BARRIER:
  case A_NOBARRIER:
  case A_MP_PARALLEL:
  case A_MP_ENDPARALLEL:
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_ATOMICREAD:
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_BARRIER:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_TASKFIRSTPRIV:
  case A_MP_PDO:
  case A_MP_ENDPDO:
  case A_MP_SECTIONS:
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
  case A_MP_TASK:
  case A_MP_TASKLOOP:
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_TASKLOOPREG:
  case A_MP_ETASKDUP:
  case A_MP_ETASKLOOPREG:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
  case A_MP_BMPSCOPE:
  case A_MP_EMPSCOPE:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_TARGET:
  case A_MP_ENDTARGET:
  case A_MP_TEAMS:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_FLUSH:
  case A_MP_TARGETUPDATE:
  case A_MP_TARGETDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_TARGETENTERDATA:
  case A_PREFETCH:
  case A_PRAGMA:
  case A_MP_CANCEL:
  case A_MP_CANCELLATIONPOINT:
    astnew = ast;
    break;
  default:
    interr("pointer_squeezer: unexpected ast", ast, 2);
    return ast;
  }
  return astnew;
}
