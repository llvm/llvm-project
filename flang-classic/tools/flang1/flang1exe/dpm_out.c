/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Fortran data partitioning module, output.
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
#include "gramtk.h"
#include "extern.h"
#include "dpm_out.h"
#include "rte.h"
#include "hpfutl.h"
#include "state.h"
#define RTE_C
#include "rte.h"
#include "pragma.h"
#include "lz.h"
#include "optimize.h"
#include "rtlRtns.h"

#define NO_PTR XBIT(49, 0x8000)
#define NO_CHARPTR XBIT(58, 0x1)
#define NO_DERIVEDPTR XBIT(58, 0x40000)

#ifdef FLANG_DPM_OUT_UNUSED
static int exist_test(int, int);
static void add_adjarr_bounds_extr_f77(int, int, int);
#endif
static bool allocate_one_auto(int);
static void component_init_allocd_auto(int, int);
#if defined(BND_ASSN_PRECEDES)
static int bnd_assn_precedes(int, int, int);
#endif
static void add_auto_bounds(int, int);
static void mk_allocate_scalar(int memberast, int sptr, int before);
static void mk_deallocate_scalar(int memberast, int sptr, int after);
static void dealloc_dt_auto(int, int, int);
#ifdef FLANG_DPM_OUT_UNUSED
static int find_actual(int, int, int);
static void set_actual(int, int, LOGICAL);
#endif

static void open_entry_guard(int);
static void close_entry_guard(void);

static void interface_for_entry(int, int);
static void reorder_args(int sptrEntry);
static void add_adjarr_bounds(int);
static void add_bound_assignments(int);
static void redimension(int, int);
static void declare_dummy_array(int);
static void declare_array_dummys(int);
static int get_array_pointer(int);
static int newargs_for_entry(int);
static void init_change_mk_id(void);
static void change_mk_id(int sptr, int sptr1);
static void do_change_mk_id(void);

static void finish_fl(void);
#ifdef FLANG_DPM_OUT_UNUSED
static void add_fl(int);
#endif
static bool emit_alnd(int sptr, int memberast, LOGICAL free_flag,
                      LOGICAL for_allocate, int allocbounds);
static void emit_secd(int sptr, int memberast, LOGICAL free_flag,
                      LOGICAL for_allocate);
#ifdef FLANG_DPM_OUT_UNUSED
static void construct_align_sc(int, int, int);
#endif
static void fix_sdsc_sc(int, int, int);
static void emit_redim(int arg);
static void emit_kopy_in(int, int, int);
#ifdef FLANG_DPM_OUT_UNUSED
static LOGICAL is_f77_adjustable(int sptr);
#endif
static void emit_scalar_kopy_in(int, int);
static int gen_ptr_in(int, int);
static int gen_ptr_out(int, int);
static int gen_copy_out(int, int, int, int);
static int gen_RTE_loc(int);
static LOGICAL is_set(int, int);
static int fill_argt_with_alnd(int sptr, int memberast, int argt, int alnd,
                               int j, int redist, int allocbounds);
static int getbits(int, int, int);
static void prepare_for_astout(void);
static void undouble_callee_args_f90(void);
static int get_scalar_in_expr(int expr, int std, LOGICAL astversion);
static int emit_get_scalar_sub(int, int);

#ifdef FLANG_DPM_OUT_UNUSED
static void update_with_actual(int);
static void update_bounds_with_actual(int);
static void emit_bcst_scalar(int sptr, int std);
#endif

static int get_arg_table(void);
static void put_arg_table(int);

/* FIXME - move these to header files */
LOGICAL has_overlap(int sptr);
int find_cc_symbols(int);

/* code insertion points at the beginning/end of a routine, block, or call */
static int EntryStd = 0, ExitStd = 0;
static int f77_local = 0;
static int redistribute = 0;
static int realign = 0;
static int allocatable_freeing = 0;
static int this_entry_g, new_dscptr_g;
static int this_entry_fval = 0; /* FVALG(interface_for_entry:this_entry) */
static char *currp;
static int *make_secd_flag;
static DTYPE typed_alloc = DT_NONE;

DTB dtb;
FL fl;

/** Data structures used to manage CUDA dynamic shared memory. */
typedef struct {
  int sptr;
  int elsz;
} DYNSH;

struct {
  DYNSH *stg_base;
  int stg_size, stg_avail;
} dynsh;

#define DYNSH_SPTR(i) (dynsh.stg_base[i].sptr)
#define DYNSH_ELSZ(i) (dynsh.stg_base[i].elsz)

typedef struct gbientry {
  int sptr, repl, lb, ub;
} gbientry;

static struct {
  gbientry *base;
  int avl, size, index, unconditional;
} gbitable = {NULL, 0, 0, 0, 1};

/* optimization table */

void
init_dtb(void)
{
  if (dtb.base == NULL) {
    dtb.size = 480;
    NEW(dtb.base, DTABLE, dtb.size);
  }
  dtb.avl = 1;
  BZERO(dtb.base + 0, DTABLE, 1);
}

void
free_dtb(void)
{
  FREE(dtb.base);
  dtb.avl = 0;
  dtb.size = 0;
}

static int
mk_dtb(int which)
{
  int nd;

  nd = dtb.avl++;
  NEED(dtb.avl, dtb.base, DTABLE, dtb.size, dtb.size + 480);
  if (nd > SPTR_MAX || dtb.base == NULL)
    errfatal(7);
  dtb.base[nd].which = which;
  return nd;
}

void
init_fl(void)
{
  fl.size = 200;
  NEW(fl.base, int, fl.size);
  fl.avl = 0;
}

static void
finish_fl(void)
{
  FREE(fl.base);
}

#ifdef FLANG_DPM_OUT_UNUSED
static void
add_fl(int a)
{
  int nd;
  int argt;
  int ast;
  int i;

  if (!allocatable_freeing) {
    int fr;
    /* pghpf_free(sec) */
    argt = mk_argt(1);
    ARGT_ARG(argt, 0) = mk_id(a);
    ast = mk_stmt(A_CALL, 0);
    fr = mk_id(sym_mkfunc(mkRteRtnNm(RTE_free), DT_NONE));
    A_LOPP(ast, fr);
    NODESCP(A_SPTRG(A_LOPG(ast)), 1);
    A_ARGCNTP(ast, 1);
    A_ARGSP(ast, argt);
    add_stmt_after(ast, ExitStd);
  } else {
    /* just in case, don't free more than once */
    for (i = 0; i < fl.avl; i++)
      if (fl.base[i] == a)
        return;
    nd = fl.avl++;
    NEED(fl.avl, fl.base, int, fl.size, fl.size + 100);
    if (nd > SPTR_MAX || fl.base == NULL)
      errfatal(7);
    fl.base[nd] = a;
  }
}
#endif

/**
   \brief Stub
 */
void
dpm_out_init(void)
{
  /* called from main() -- there should be a transform_init() */
}

static void
trans_mkproc(int sptr)
{
  int descr;
  int nargs, argt, astnew;
  int ndim, i;
  ADSC *ad;

  /* do procs descriptor */
  descr = DESCRG(sptr);

  if (VISITG(descr))
    return;
  VISITP(descr, 1);

  /* might be scalar */
  if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
    ndim = rank_of(DTYPEG(sptr));
    ad = AD_DPTR(DTYPEG(sptr));
  } else {
    ndim = 0;
    ad = 0;
  }

  nargs = ndim + 2;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = mk_id(descr);
  ARGT_ARG(argt, 1) = mk_cval(ndim, DT_INT);
  for (i = 0; i < ndim; ++i)
    ARGT_ARG(argt, i + 2) = mk_size(AD_LWBD(ad, i), AD_UPBD(ad, i));

  astnew = mk_func_node(A_CALL,
                        mk_id(sym_mkfunc(mkRteRtnNm(RTE_processors), DT_NONE)),
                        nargs, argt);

  add_stmt_before(astnew, EntryStd);
}

/**
   \brief Return a 'extent(array,dim)' call
 */
static int
extent(int array, int descriptor, int dimension)
{
  int nargs, argt, func, ast;
  int subs[1];

  if (DTYG(DTYPEG(array)) != TY_CHAR) {
    subs[0] = mk_isz_cval(get_global_extent_index(dimension), astb.bnd.dtype);
    ast = mk_subscr(descriptor, subs, 1, astb.bnd.dtype);
  } else {
    nargs = 2;

    argt = mk_argt(nargs);
    ARGT_ARG(argt, 0) = descriptor;
    ARGT_ARG(argt, 1) = mk_isz_cval(dimension + 1, astb.bnd.dtype);

    func = sym_mkfunc_nodesc(mkRteRtnNm(RTE_extent), astb.bnd.dtype);

    ast = mk_func_node(A_FUNC, mk_id(func), nargs, argt);
    A_DTYPEP(ast, astb.bnd.dtype);
  }
  return ast;
} /* extent */

static void
allocate_aligned(int sptr, int memberast, int basesptr)
{
  int dtype, mem;
  if (ALLOCG(sptr))
    return;
  dtype = DTYPEG(sptr);
  switch (DTY(dtype)) {
  case TY_ARRAY:
    /* may be used at internal subroutine */
    if (SCG(basesptr) == SC_NONE && gbl.internal == 1)
      SCP(basesptr, SC_LOCAL);
    break;
  case TY_DERIVED:
    /* if this is a derived type, look at members */
    if (POINTERG(sptr))
      break;
    mem = DTY(dtype + 1);
    if (mem <= NOSYM)
      return; /* empty derived type */
    if (memberast) {
      memberast = mk_member(A_PARENTG(memberast), mk_id(sptr), dtype);
    } else {
      memberast = mk_id(sptr);
    }
    /* make a A_MEM to pass to check_member */
    memberast = mk_member(memberast, mk_id(mem), DTYPEG(mem));
    for (; mem > NOSYM; mem = SYMLKG(mem)) {
      if (is_tbp_or_final(mem))
        continue; /* skip type bound procedures */
      if (!POINTERG(mem)) {
        allocate_aligned(mem, memberast, basesptr);
      }
    }
    break;
  }
} /* allocate_aligned */

/** Algorithm:
 * This routine puts allocate statement for each aligned arrays except
 * allocatable aligned arrays.
 * allocate(a(a$sd(33):a$sd(34)))
 * This routine is called after all pghpf_template and pghpf_instance
 * which set array descr fully.
 * At the end, it fixes declaration part of arrays
 * by making the deferred array real a(:,:)
 */
static void
allocate_for_aligned_array(void)
{
  int sptr;
  int align;
  int dtype;
  ADSC *ad;
  /* put barrier before any deallocate or copy_out if SMP */
  /* put out allocates for local arrays */
  for (sptr = stb.firstusym; sptr < stb.stg_avail; sptr++) {
    switch (STYPEG(sptr)) {
    case ST_VAR:
    case ST_ARRAY:
      break;
    default:
      continue;
    }
    if (IGNOREG(sptr))
      continue;
    allocate_aligned(sptr, 0, sptr);
  }

  /* this will fix the declaration of aligned arrays
   * real a(:,:) instead of real a(10,10)
   */
  /* fix up the local arrays */
  /* declare DYNAMIC array common */
  for (sptr = stb.firstusym; sptr < stb.stg_avail; sptr++) {
    int stype;
    stype = STYPEG(sptr);
    if (stype != ST_ARRAY)
      continue;

    /* declare DYNAMIC array pointer */

    align = ALIGNG(sptr);
    if (!align)
      continue;
    if (SCG(sptr) == SC_DUMMY)
      continue;
    dtype = DTYPEG(sptr);
    ad = AD_DPTR(dtype);
    AD_DEFER(ad) = 1;
  }
}

/** \brief check that distribution and alignment are mutually exclusive */
static void
check_flag(int flag)
{
  int flag1, flag2;

  flag1 =
      __PRESCRIPTIVE_ALIGN_TARGET | __DESCRIPTIVE_ALIGN_TARGET | __IDENTITY_MAP;
  flag2 = __OMITTED_DIST_TARGET | __PRESCRIPTIVE_DIST_TARGET |
          __DESCRIPTIVE_DIST_TARGET | __TRANSCRIPTIVE_DIST_TARGET |
          __OMITTED_DIST_FORMAT | __PRESCRIPTIVE_DIST_FORMAT |
          __DESCRIPTIVE_DIST_FORMAT | __TRANSCRIPTIVE_DIST_FORMAT;
  assert(!((flag & flag1) && (flag & flag2)),
         "check_flag: wrong flag for run-time", flag, 4);
}

static int
construct_flag(int sptr)
{
  int flag;
  flag = 0;
  if (ASUMSZG(sptr))
    flag |= __ASSUMED_SIZE;
  if (SEQG(sptr))
    flag |= __SEQUENTIAL;
  /* set assumed-shape only if copy-in is needed.  if sequential and
     not a pointer dummy, then copy-in isn't done, but template may
     be created */
  if (ASSUMSHPG(sptr) && (!SEQG(sptr) || POINTERG(sptr)))
    flag |= __ASSUMED_SHAPE;
  if (SAVEG(sptr))
    flag |= __SAVE;
  /* if it is pointer */
  if (POINTERG(sptr))
    flag |= __POINTER;
  if (!XBIT(47, 0x04)) {
    if (INTENTG(sptr) == INTENT_INOUT || INTENTG(sptr) == INTENT_DFLT)
      flag |= __INTENT_INOUT;
    if (INTENTG(sptr) == INTENT_IN)
      flag |= __INTENT_IN;
    if (INTENTG(sptr) == INTENT_OUT)
      flag |= __INTENT_OUT;
  } else
    flag |= __INTENT_INOUT;

  check_flag(flag);
  return flag;
}

static int
make_alnd(int sptr)
{
  int ndim;
  int i;
  ADSC *ad;
  int nd;
  int flag;
  int glb, gub;
  if (is_bad_dtype(DTYPEG(sptr)))
    return 0;
  if ((IGNOREG(sptr) || HCCSYMG(sptr)) && DESCRG(sptr) == 0)
    trans_mkdescr(sptr);
  assert(DESCRG(sptr), "make_alnd: descriptor does not exist", sptr, 3);
  if (ALNDG(DESCRG(sptr)))
    return ALNDG(DESCRG(sptr));

  nd = mk_dtb(1);
  BZERO(dtb.base + nd, DTABLE, 1);
  ndim = rank_of_sym(sptr);
  TMPL_RANK(nd) = ndim;
  TMPL_TYPE(nd) = REPLICATED;
  flag = construct_flag(sptr);
  TMPL_FLAG(nd) = flag;
  {
    ad = AD_DPTR(DTYPEG(sptr));

    for (i = 0; i < ndim; ++i) {
      /* glb */
      glb = AD_LWBD(ad, i);
      if (glb == 0 || A_TYPEG(glb) != A_ID || !HCCSYMG(A_SPTRG(glb)))
        glb = AD_LWAST(ad, i);
      if (glb == 0)
        glb = mk_cval(1, DT_INT);

      /* gub */
      gub = AD_UPBD(ad, i);
      if (gub == 0 || A_TYPEG(gub) != A_ID || !HCCSYMG(A_SPTRG(gub))) {
        gub = AD_UPAST(ad, i);
      }
      if (gub == 0)
        gub = mk_cval(1, DT_INT);

      TMPL_LB(nd, i) = glb;
      if (is_set(flag, __ASSUMED_SIZE) && i == ndim - 1)
        continue;
      if (is_set(flag, __ASSUMED_SHAPE))
        continue;
      TMPL_UB(nd, i) = gub;
    }
  }
  return nd;
}

#ifdef FLANG_DPM_OUT_UNUSED
static void
construct_align_sc(int alnd, int alignee, int target)
{
  int sptr;
  int sc, sc1;

  sc = NONE_SC;
  sptr = alignee;
  if (ALLOCG(sptr))
    sc = ALLOC_SC;
  else if (STYPEG(sptr) == ST_MEMBER)
    sc = STATIC_SC;
  else if (SCG(sptr) == SC_DUMMY)
    sc = DUMMY_SC;
  else if (SCG(sptr) == SC_CMBLK)
    sc = COMMON_SC;
  else if ((SCG(sptr) == SC_LOCAL || SCG(sptr) == SC_STATIC ||
            SCG(sptr) == SC_NONE) &&
           !ALLOCG(sptr))
    sc = STATIC_SC;

  TMPL_ALIGNEE_SC(alnd) = sc;

  sc = NONE_SC;
  sptr = target;
  if (ALLOCG(sptr))
    sc = ALLOC_SC;
  else if (STYPEG(sptr) == ST_MEMBER)
    sc = STATIC_SC;
  else if (SCG(sptr) == SC_DUMMY)
    sc = DUMMY_SC;
  else if (SCG(sptr) == SC_CMBLK)
    sc = COMMON_SC;
  else if ((SCG(sptr) == SC_LOCAL || SCG(sptr) == SC_STATIC ||
            SCG(sptr) == SC_NONE) &&
           !ALLOCG(sptr))
    sc = STATIC_SC;

  TMPL_TARGET_SC(alnd) = sc;

  sc = TMPL_ALIGNEE_SC(alnd);
  sc1 = TMPL_TARGET_SC(alnd);

  /* check correctness of alignment */
  if (sc == NONE_SC || sc1 == NONE_SC) {
    assert(0, "construct_align_sc: wrong alignment", alignee, 3);
    return;
  }

  if (sc == ALLOC_SC && sc1 == ALLOC_SC) {
    return;
  }
  if (sc == ALLOC_SC && sc1 == DUMMY_SC) {
    return;
  }

  if (sc == ALLOC_SC && sc1 == STATIC_SC) {
    return;
  }

  if (sc == ALLOC_SC && sc1 == COMMON_SC) {
    return;
  }

  if (sc == DUMMY_SC && sc1 == ALLOC_SC) {
    /* except module allocatable */
    if (!MDALLOCG(target))
      error(493, 4, gbl.lineno, "Dummy", SYMNAME(alignee));
    return;
  }

  if (sc == DUMMY_SC && sc1 == DUMMY_SC) {
    return;
  }

  if (sc == DUMMY_SC && sc1 == STATIC_SC) {
    return;
  }

  if (sc == DUMMY_SC && sc1 == COMMON_SC) {
    return;
  }

  if (sc == STATIC_SC && sc1 == ALLOC_SC) {
    /* except module allocatable */
    if (!MDALLOCG(target))
      error(493, 4, gbl.lineno, "Static", SYMNAME(alignee));
    return;
  }

  if (sc == STATIC_SC && sc1 == DUMMY_SC) {
    return;
  }

  if (sc == STATIC_SC && sc1 == STATIC_SC) {
    return;
  }

  if (sc == STATIC_SC && sc1 == COMMON_SC) {
    return;
  }

  if (sc == COMMON_SC && sc1 == ALLOC_SC) {
    error(493, 3, gbl.lineno, "COMMON", SYMNAME(alignee));
    return;
  }

  if (sc == COMMON_SC && sc1 == DUMMY_SC) {
    error(494, 4, gbl.lineno, SYMNAME(alignee), CNULL);
    return;
  }
  if (sc == COMMON_SC && sc1 == STATIC_SC) {
    return;
  }

  if (sc == COMMON_SC && sc1 == COMMON_SC) {
    return;
  }
}
#endif

static LOGICAL
is_set(int flag, int value)
{

  if (flag & value)
    return TRUE;
  else
    return FALSE;
}

/* type: ST_ARRAY or ST_TEMPLATE */
static void
share_alnd(int type)
{
  int sptr, sptr1;
  int arrdsc, arrdsc1;
  int alnd, alnd1;
  int descr;

  /* make alnd */
  for (sptr = aux.list[type]; sptr != NOSYM; sptr = SLNKG(sptr)) {
#if DEBUG
    /* aux.list[] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "share_alnd: corrupted aux.list[type]", sptr, 4);
#endif
    arrdsc = DESCRG(sptr);
    if (gbl.internal > 1 && !INTERNALG(sptr)) {
      /* in a contained subprogram */
      if (arrdsc && SDSCINITG(arrdsc) && SECDSCG(arrdsc) &&
          SCOPEG(SECDSCG(arrdsc)) == SCOPEG(sptr) &&
          STYPEG(SCOPEG(sptr)) != ST_MODULE)
        continue;
    }
    if (ALLOCG(sptr) && (ALNDG(arrdsc) || SECDSCG(arrdsc)))
      continue;
    if (F90POINTERG(sptr))
      continue;
    alnd = make_alnd(sptr);
    ALNDP(DESCRG(sptr), alnd);
  }

  for (sptr = aux.list[type]; sptr != NOSYM; sptr = SLNKG(sptr)) {
#if DEBUG
    /* aux.list[] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "share_alnd: corrupted aux.list[type]", sptr, 4);
#endif
    if (is_bad_dtype(DTYPEG(sptr)))
      continue;
    arrdsc = DESCRG(sptr);
    if (ALLOCG(sptr))
      continue;
    if (F90POINTERG(sptr))
      continue;
    if (gbl.internal > 1 && !INTERNALG(sptr)) {
      if (arrdsc && SDSCINITG(arrdsc) && SECDSCG(arrdsc) &&
          SCOPEG(SECDSCG(arrdsc)) == SCOPEG(sptr) &&
          STYPEG(SCOPEG(sptr)) != ST_MODULE)
        continue;
    }
    if (!VISITG(sptr)) {
      descr = SECDSCG(arrdsc);
      /* zeki descr = 0; */
      alnd = ALNDG(arrdsc);
      assert(alnd, "share_alnd:no alnd data structure", alnd, 3);
      if (TMPL_DESCR(alnd) == 0) {
        if (descr)
          TMPL_DESCR(alnd) = descr;
        else
          TMPL_DESCR(alnd) = sym_get_sdescr(sptr, -1);
      }
      if (STYPEG(sptr) == ST_MEMBER) {
        SECDSCP(arrdsc, TMPL_DESCR(alnd));
      }
      VISITP(sptr, 1);
    }
    if (XBIT(57, 0x400000))
      continue;
    if (CMBLKG(sptr) && (ALIGNG(sptr) || DISTG(sptr)))
      continue;
    for (sptr1 = SLNKG(sptr); sptr1 != NOSYM; sptr1 = SLNKG(sptr1)) {
      if (is_bad_dtype(DTYPEG(sptr1)))
        continue;
      if (SCG(sptr1) == SC_DUMMY)
        continue;
      if (ALLOCG(sptr1))
        continue;
      if (CMBLKG(sptr1) && (ALIGNG(sptr) || DISTG(sptr)))
        continue;
      if (STYPEG(sptr1) == ST_MEMBER && STYPEG(sptr) == ST_MEMBER) {
        if (ENCLDTYPEG(sptr1) != ENCLDTYPEG(sptr))
          continue;
        if (DDTG(DTYPEG(sptr1)) != DDTG(DTYPEG(sptr)))
          continue;
      } else if (STYPEG(sptr1) == ST_MEMBER || STYPEG(sptr) == ST_MEMBER) {
        continue;
      }
      if (!VISITG(sptr1)) {
        if (is_same_alnd(sptr, sptr1)) {
          arrdsc1 = DESCRG(sptr1);
          alnd1 = ALNDG(arrdsc1);
          if (alnd)
            TMPL_DESCR(alnd1) = TMPL_DESCR(alnd);
          if (STYPEG(sptr1) == ST_MEMBER) {
            SECDSCP(arrdsc1, TMPL_DESCR(alnd1));
          }
          VISITP(sptr1, 1);
        }
      }
    }
  }
  for (sptr = aux.list[type]; sptr != NOSYM; sptr = SLNKG(sptr)) {
#if DEBUG
    /* aux.list[] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "share_alnd: corrupted aux.list[type]", sptr, 4);
#endif
    VISITP(sptr, 0);
  }
}

LOGICAL
is_same_alnd(int sptr, int sptr1)
{
  int arrdsc, arrdsc1;
  int alnd, alnd1;
  int ndim, ndim1;
  int i;

  arrdsc = DESCRG(sptr);
  if (!arrdsc)
    return FALSE;
  arrdsc1 = DESCRG(sptr1);
  if (!arrdsc1)
    return FALSE;

  alnd = ALNDG(arrdsc);
  alnd1 = ALNDG(arrdsc1);
  if (alnd == 0 && alnd1 == 0)
    return TRUE;
  if (alnd == 0 || alnd1 == 0)
    return FALSE;

  ndim = TMPL_RANK(alnd);
  ndim1 = TMPL_RANK(alnd1);
  if (ndim != ndim1)
    return FALSE;

  if (TMPL_FLAG(alnd) != TMPL_FLAG(alnd1))
    return FALSE;
  if (TMPL_DIST_TARGET_DESCR(alnd) != TMPL_DIST_TARGET_DESCR(alnd1))
    return FALSE;
  if (TMPL_ISSTAR(alnd) != TMPL_ISSTAR(alnd1))
    return FALSE;
  if (TMPL_CONFORM(alnd) != TMPL_CONFORM(alnd1))
    return FALSE;
  if (TMPL_COLLAPSE(alnd) != TMPL_COLLAPSE(alnd1))
    return FALSE;
  if (TMPL_ALIGN_TARGET(alnd) != TMPL_ALIGN_TARGET(alnd1))
    return FALSE;
  if (TMPL_TARGET_DESCR(alnd) != TMPL_TARGET_DESCR(alnd1))
    return FALSE;

  for (i = 0; i < ndim; i++) {
    if (TMPL_LB(alnd, i) != TMPL_LB(alnd1, i))
      return FALSE;
    if (TMPL_UB(alnd, i) != TMPL_UB(alnd1, i))
      return FALSE;
  }

  return TRUE;
}

LOGICAL
is_same_secd(int sptr, int sptr1)
{
  int arrdsc, arrdsc1;
  int secd, secd1;
  int ndim, ndim1;

  arrdsc = DESCRG(sptr);
  if (!arrdsc)
    return FALSE;
  arrdsc1 = DESCRG(sptr1);
  if (!arrdsc1)
    return FALSE;

  secd = SECDG(arrdsc);
  if (!secd)
    return FALSE;
  secd1 = SECDG(arrdsc1);
  if (!secd1)
    return FALSE;

  ndim = INS_RANK(secd);
  ndim1 = INS_RANK(secd1);
  if (ndim != ndim1)
    return FALSE;
  if (INS_TEMPLATE(secd) != INS_TEMPLATE(secd1))
    return FALSE;
  if (dtype_to_arg(DTY(DTYPEG(sptr) + 1)) !=
      dtype_to_arg(DTY(DTYPEG(sptr1) + 1)))
    return FALSE;
  if (size_ast_of(mk_id(sptr), DTY(DTYPEG(sptr) + 1)) !=
      size_ast_of(mk_id(sptr), DTY(DTYPEG(sptr1) + 1)))
    return FALSE;
  return TRUE;
}

static int
make_secd(int sptr)
{
  int align;
  int ndim;
  int i;
  int nolap, polap;
  ADSC *ad;
  int secd;

  if (is_bad_dtype(DTYPEG(sptr)))
    return 0;
  assert(DESCRG(sptr), "make_secd: descriptor does not exist", sptr, 3);
  secd = mk_dtb(2);
  BZERO(dtb.base + secd, DTABLE, 1);
  ad = AD_DPTR(DTYPEG(sptr));
  align = ALIGNG(sptr);
  ndim = rank_of_sym(sptr);

  INS_DTYPE(secd) = DTYPEG(sptr);
  INS_DESCR(secd) = 0;
  INS_RANK(secd) = ndim;
  INS_TEMPLATE(secd) = TMPL_DESCR(ALNDG(DESCRG(sptr)));
  for (i = 0; i < ndim; ++i) {
    nolap = 0;
    polap = 0;
  }
  return secd;
}

static void
make_secd_for_members(int dtype)
{
  int mem, memdtype, descr, secdsc, secd, alnd;

  if (make_secd_flag[dtype])
    return;
  make_secd_flag[dtype] = 1;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
    memdtype = DTYPEG(mem);
    switch (DTY(memdtype)) {
    case TY_DERIVED:
      make_secd_for_members(memdtype);
      break;
    case TY_ARRAY:
      descr = DESCRG(mem);
      if (descr && STYPEG(descr) == ST_ARRDSC) {
        if (DESCUSEDG(mem) || XBIT(57, 0x40000) || ALLOCG(mem)) {
          alnd = ALNDG(descr);
          if (!alnd) {
            alnd = make_alnd(mem);
            ALNDP(descr, alnd);
          }
          secdsc = SECDSCG(descr);
          secd = SECDG(descr);
          if (!secd) {
            secd = make_secd(mem);
            SECDP(descr, secd);
          }
          secdsc = SECDSCG(descr);
          if (secdsc) {
            INS_DESCR(secd) = secdsc;
            INS_TEMPLATE(secd) = secdsc;
            TMPL_DESCR(alnd) = secdsc;
            change_mk_id(descr, secdsc);
          } else if (INS_DESCR(secd) == 0) {
            secdsc = sym_get_sdescr(mem, -1);
            INS_DESCR(secd) = secdsc;
            INS_TEMPLATE(secd) = secdsc;
            TMPL_DESCR(alnd) = secdsc;
            SECDSCP(descr, secdsc);
            change_mk_id(descr, secdsc);
          } else {
            secdsc = INS_DESCR(secd);
            INS_TEMPLATE(secd) = secdsc;
            TMPL_DESCR(alnd) = secdsc;
            SECDSCP(descr, secdsc);
          }
        }
      }
      memdtype = DTY(memdtype + 1);
      if (DTY(memdtype) == TY_DERIVED) {
        make_secd_for_members(memdtype);
      }
      break;
    }
  }
} /* make_secd_for_members */

/** \brief Return TRUE if we need to initialize the descriptor for this symbol
    because perhaps this is a host subroutine, and the internal subprograms
    will need the descriptor, or perhaps -g is set.
 */
LOGICAL
want_descriptor_anyway(int sptr)
{
  if (gbl.internal == 1) {
    int dtype;
    dtype = DTYPEG(sptr);
    if (DTY(dtype) != TY_ARRAY)
      return FALSE;
    if (!DESCRG(sptr))
      return FALSE;
    if (XBIT(57, 0x40000))
      return TRUE;
    /* descriptor for allocatable/pointer must be in host */
    if (ALLOCG(sptr))
      return TRUE;
  }
  if (flg.debug && !XBIT(123, 0x400) && !HCCSYMG(sptr) && !CCSYMG(sptr)) {
    /* only need non-fixed bounds */
    int dtype;
    dtype = DTYPEG(sptr);
    if (DTY(dtype) != TY_ARRAY)
      return FALSE;
    if (!DESCRG(sptr))
      return FALSE;
    if (ALIGNG(sptr) || DISTG(sptr) || ASSUMSHPG(sptr) || ALLOCG(sptr))
      return TRUE;
    if (ADD_DEFER(dtype) || ADD_ASSUMSHP(dtype))
      return TRUE;
  }
  return FALSE;
} /* want_descriptor_anyway */

static void
share_secd(void)
{
  int sptr, sptr1;
  int arrdsc, arrdsc1;
  int secd, secd1;
  int descr;

  /* make secd */
  for (sptr = aux.list[ST_ARRAY]; sptr != NOSYM; sptr = SLNKG(sptr)) {
    int secd;
#if DEBUG
    /* aux.list[] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "share_secd: corrupted aux.list[ST_ARRAY]", sptr, 4);
#endif
    arrdsc = DESCRG(sptr);
    if (gbl.internal > 1 && !INTERNALG(sptr)) {
      if (arrdsc && SDSCINITG(arrdsc) && SECDSCG(arrdsc) &&
          SCOPEG(SECDSCG(arrdsc)) == SCOPEG(sptr) &&
          STYPEG(SCOPEG(sptr)) != ST_MODULE)
        continue;
    }
    if (ALLOCG(sptr) && (SECDG(arrdsc) || SECDSCG(arrdsc)))
      continue;
    if (F90POINTERG(sptr))
      continue;
    secd = make_secd(sptr);
    SECDP(arrdsc, secd);
  }

  for (sptr = aux.list[ST_ARRAY]; sptr != NOSYM; sptr = SLNKG(sptr)) {
#if DEBUG
    /* aux.list[] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "share_secd: corrupted aux.list[ST_ARRAY]", sptr, 4);
#endif
    if (is_bad_dtype(DTYPEG(sptr)))
      continue;
    arrdsc = DESCRG(sptr);
    if (gbl.internal > 1 && !INTERNALG(sptr)) {
      if (arrdsc && SDSCINITG(arrdsc) && (descr = SECDSCG(arrdsc)) &&
          SCOPEG(descr) == SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) != ST_MODULE) {
        change_mk_id(arrdsc, descr);
        continue;
      }
    }
    if (ALLOCG(sptr) && (SECDG(arrdsc) == 0))
      continue;
    if (F90POINTERG(sptr))
      continue;
    if (!VISITG(sptr)) {
      descr = SECDSCG(arrdsc);
      secd = SECDG(arrdsc);
      if (INS_DESCR(secd) != 0)
        ;
      else if (descr)
        INS_DESCR(secd) = descr;
      else {
        INS_DESCR(secd) = sym_get_sdescr(sptr, -1);
        fix_sdsc_sc(sptr, INS_DESCR(secd), arrdsc);
      }
      change_mk_id(arrdsc, INS_DESCR(secd));
      VISITP(sptr, 1);
    }
    if (XBIT(57, 0x400000))
      continue;
    if (ALLOCG(sptr))
      continue;
    for (sptr1 = SLNKG(sptr); sptr1 != NOSYM; sptr1 = SLNKG(sptr1)) {
      if (is_bad_dtype(DTYPEG(sptr1)))
        continue;
      if (SCG(sptr1) == SC_DUMMY)
        continue;
      if (ALLOCG(sptr1))
        continue;
      if (CMBLKG(sptr1))
        continue;
      if (gbl.internal > 1 && !INTERNALG(sptr1))
        continue;
      if (!VISITG(sptr1)) {
        if (is_same_secd(sptr, sptr1)) {
          arrdsc1 = DESCRG(sptr1);
          secd1 = SECDG(arrdsc1);
          INS_DESCR(secd1) = INS_DESCR(secd);
          change_mk_id(arrdsc1, INS_DESCR(secd));
          VISITP(sptr1, 1);
        }
      }
    }
  }
  for (sptr = aux.list[ST_ARRAY]; sptr != NOSYM; sptr = SLNKG(sptr)) {
#if DEBUG
    /* aux.list[] must be terminated with NOSYM, not 0 */
    assert(sptr > 0, "share_secd: corrupted aux.list[ST_ARRAY]", sptr, 4);
#endif
    VISITP(sptr, 0);
  }

  NEW(make_secd_flag, int, stb.dt.stg_avail);
  BZERO(make_secd_flag, int, stb.dt.stg_avail);

  /* now handle array members in derived types */
  for (sptr = stb.firstosym; sptr < stb.stg_avail; sptr++) {
    int dtype;
    if (IGNOREG(sptr))
      continue;
    switch (STYPEG(sptr)) {
    case ST_VAR:
      dtype = DTYPEG(sptr);
      if (DTY(dtype) == TY_DERIVED) {
        make_secd_for_members(dtype);
      }
      break;
    case ST_ARRAY:
      dtype = DTYPEG(sptr);
      dtype = DTY(dtype + 1);
      if (DTY(dtype) == TY_DERIVED) {
        make_secd_for_members(dtype);
      }
      break;
    default:;
    }
  }
  if (flg.debug || (gbl.internal == 1 && XBIT(57, 0x40000))) {
    /* for hosts, or if debug set, initialize all members */
    int dtype;
    for (dtype = 1; dtype < stb.dt.stg_avail; dtype += dlen(DTY(dtype))) {
      if (DTY(dtype) == TY_DERIVED) {
        make_secd_for_members(dtype);
      }
    }
  }
  FREE(make_secd_flag);
}

/* MW: This used to change the A_SPTR field of the mk_id(sptr) to be sptr1.
 * Now it saves sptr1 in the symbol table of sptr, and puts sptr on a linked
 * list, so all the ASTs can be changed at once.  The problem was sometimes
 * the change was done too early, and a subsequent mk_id of the original
 * sptr would be added, but not changed. */

static int change_mk_id_list;

static void
init_change_mk_id(void)
{
  change_mk_id_list = NOSYM;
} /* init_change_mk_id */

static void
change_mk_id(int sptr, int sptr1)
{
  assert(STYPEG(sptr) == ST_ARRDSC, "change_mk_id: of non-arrdsc", STYPEG(sptr),
         3);
  if (RENAMEG(sptr) == 0) {
    SLNKP(sptr, change_mk_id_list);
    change_mk_id_list = sptr;
    RENAMEP(sptr, sptr1);
    if (flg.smp && PARREFG(sptr) && !PARREFG(sptr1)) {
      set_parref_flag2(sptr1, sptr, 0);
    }
  } else if (sptr1 != RENAMEG(sptr)) {
    assert(RENAMEG(sptr) == sptr1, "change_mk_id: arrdsc changed twice", sptr,
           3);
  }
} /* change_mk_id */

static void
do_change_mk_id(void)
{
  int ast, sptr, sptr1, nextsptr;
  for (sptr = change_mk_id_list; sptr > NOSYM; sptr = nextsptr) {
    nextsptr = SLNKG(sptr);
    SLNKP(sptr, 0);
    sptr1 = RENAMEG(sptr);
    RENAMEP(sptr, 0);
    ast = mk_id(sptr);
    A_SPTRP(ast, sptr1);
  }
} /* do_change_mk_id */

static void
desc_need_arrays(void)
{
  int sptr;

  for (sptr = stb.firstosym; sptr < stb.stg_avail; sptr++) {
    if (!is_array_type(sptr))
      continue;
    if (STYPEG(sptr) == ST_MEMBER) {
      if (ALIGNG(sptr) || RUNTIMEG(sptr) || ADJARRG(sptr)) {
        DESCUSEDP(sptr, 1);
      }
    } else {
      if (SCG(sptr) != SC_NONE && ALIGNG(sptr)) {
        DESCUSEDP(sptr, 1);
      }
      /* may be used at internal subroutine */
      if (want_descriptor_anyway(sptr))
        DESCUSEDP(sptr, 1);
    }
    if (!DESCUSEDG(sptr))
      continue;
    if (NODESCG(sptr))
      continue;
    if (IGNOREG(sptr))
      continue;
    /* add on the  list */
    if (SLNKG(sptr) == 0) {
      SLNKP(sptr, aux.list[ST_ARRAY]);
      aux.list[ST_ARRAY] = sptr;
    }
  }
}

static void
use_dummy_desc(void)
{
  int this_entry;
  int arg, narg;
  int dscptr;
  int i;

  for (this_entry = gbl.entries; this_entry != NOSYM;
       this_entry = SYMLKG(this_entry)) {
    narg = PARAMCTG(this_entry);
    dscptr = DPDSCG(this_entry);
    for (i = 0; i < narg; i++) {
      arg = aux.dpdsc_base[dscptr];
      if (is_kopy_in_needed(arg))
        DESCUSEDP(arg, 1);
      dscptr++;
    }
  }
}

LOGICAL
is_kopy_in_needed(int arg)
{
  switch (STYPEG(arg)) {
  default:
    /* procedures, labels, need no kopy in */
    return FALSE;
  case ST_VAR:
  case ST_ARRAY:
    /* only dummies, result variables passed like dummies */
    if (SCG(arg) != SC_DUMMY && !RESULTG(arg))
      return FALSE;
    /* pointer needs kopy-in, regardless of type */
    if (POINTERG(arg) || IS_PROC_DUMMYG(arg))
      return TRUE;
    /* other nonarrays need no kopy in */
    if (DTY(DTYPEG(arg)) != TY_ARRAY)
      return FALSE;
    /* sequential arrays need no kopy in */
    if (SEQG(arg)) {
      /* unless they WERE originally assumed-shape */
      int dtype;
      dtype = DTYPEG(arg);
      if (DTY(dtype) != TY_ARRAY || ADD_ASSUMSHP(dtype) != 2) {
        return FALSE;
      }
    }
    break;
  }
  /* default */
  return TRUE;
}

void
unvisit_every_sptr(void)
{
  int sptr;

  for (sptr = stb.firstosym; sptr < stb.stg_avail; sptr++) {
    VISITP(sptr, 0);
    VISIT2P(sptr, 0);
  }
}

/* call emit_alnd and emit_secd at subprogram entry */
static void
wrap_symbol(int sptr, int memberast, int basesptr)
{
  int mem, dtype, arrd, alloc;
  dtype = DTYPEG(sptr);
  alloc = 1;
  if (STYPEG(sptr) == ST_MEMBER && want_descriptor_anyway(sptr)) {
    /* create for host subprogram in case used in contained routine */
    DESCUSEDP(sptr, 1);
  }
  switch (SCG(basesptr)) {
  case SC_DUMMY:
    /* if a dummy for this routine, and not used, skip it */
    if (((gbl.internal <= 1 || INTERNALG(basesptr)) && !DESCUSEDG(sptr)) ||
        is_kopy_in_needed(basesptr)) {
      return;
    }
    break;
  case SC_NONE:
    if (!DESCUSEDG(sptr))
      return;
    alloc = 0;
    break; /* variable isn't used, descriptor is */
  default:;
  }
  switch (DTY(dtype)) {
  case TY_ARRAY:
    /* if an unused symbol from the containing routine, skip it */
    if (gbl.internal > 1 && !INTERNALG(sptr)) {
      if (DESCRG(sptr) && SDSCINITG(DESCRG(sptr)) &&
          (arrd = SECDSCG(DESCRG(sptr))) && SCOPEG(arrd) == SCOPEG(sptr) &&
          STYPEG(SCOPEG(sptr)) != ST_MODULE) {
        return;
      }
      /* FS 2001: module array, descriptor is in the host subprogram
       * don't fill section descriptor in contained subprogram;
       * check there is a DESCR, is has an SECDSC, and its scope
       * is the scope of the parent of the current subpgoram */
      if (DESCRG(sptr) && (arrd = SECDSCG(DESCRG(sptr))) &&
          SCOPEG(arrd) == SCOPEG(gbl.currsub) &&
          STYPEG(SCOPEG(sptr)) == ST_MODULE) {
        change_mk_id(DESCRG(sptr), arrd);
        return;
      }
    }
    /* if a variable or array, this was handled by allocate_one_auto */
    if (STYPEG(sptr) == ST_MEMBER && memberast) {
      if (ADJLENG(sptr) && alloc) {
        add_auto_len(sptr, EntryStd);
      }
      if (ADJARRG(sptr) || RUNTIMEG(sptr)) {
        (void)add_auto_bounds(sptr, EntryStd);
      }
      if (!POINTERG(sptr) && !ALLOCG(sptr) && alloc &&
          (ADJARRG(sptr) || RUNTIMEG(sptr) || ADJLENG(sptr))) {
        if (!ALIGNG(sptr) && !POINTERG(sptr)) {
          int ast, i, ndim, subscr[7];
          /* make the subscripts */
          ndim = ADD_NUMDIM(dtype);
          for (i = 0; i < ndim; ++i) {
            subscr[i] = mk_triple(ADD_LWAST(dtype, i), ADD_UPAST(dtype, i), 0);
          }
          ast = check_member(memberast, mk_id(sptr));
          mk_mem_allocate(ast, subscr, EntryStd, ast);
          mk_mem_deallocate(ast, ExitStd);
        }
      }
    }
    break;
  case TY_DERIVED:
    /* if this is a derived type, look at members */
    if (POINTERG(sptr) || is_tbp_or_final(sptr) /* skip tbp */) {
      return;
    }
    mem = DTY(dtype + 1);
    if (mem <= NOSYM)
      return; /* empty derived type */
    if (memberast) {
      memberast = mk_member(A_PARENTG(memberast), mk_id(sptr), dtype);
    } else {
      memberast = mk_id(sptr);
    }
    /* make a A_MEM to pass to check_member */
    memberast = mk_member(memberast, mk_id(mem), DTYPEG(mem));
    for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
      VISITP(mem, 0);
    }
    for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
      if (!POINTERG(mem) && !USELENG(mem)) /* TBD: use of length type params */
        wrap_symbol(mem, memberast, basesptr);
    }
    for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
      VISITP(mem, 0);
    }
    return;
  default:
    /* if an unused symbol from the containing routine, skip it */
    if (gbl.internal > 1 && !INTERNALG(sptr))
      return;
    if (memberast && ADJLENG(sptr) && alloc) {
      add_auto_len(sptr, EntryStd);
      if (!POINTERG(sptr) && !ALLOCG(sptr)) {
        /* scalar adjustable length char string */
        if (STYPEG(sptr) != ST_MEMBER &&
            (!CLASSG(sptr) || STYPEG(sptr) != ST_PROC) && /* skip tbp */
            EARLYSPECG(CVLENG(sptr))) {
          mk_allocate_scalar(memberast, sptr, STD_NEXT(EntryStd));
        } else {
          mk_allocate_scalar(memberast, sptr, EntryStd);
        }
        mk_deallocate_scalar(memberast, sptr, ExitStd);
      }
    }
    return;
  }

  /* predefined descriptor, case like MODULE */
  if (DESCRG(sptr) && (arrd = SECDSCG(DESCRG(sptr))) &&
      (ENCLFUNCG(basesptr) || ALLOCG(sptr))) {
    /* allocatable HACK */
    change_mk_id(DESCRG(sptr), arrd);
    /* dynamic from module */
    if (ENCLFUNCG(basesptr) && STYPEG(ENCLFUNCG(basesptr)) == ST_MODULE) {
      if (STYPEG(arrd) == ST_DESCRIPTOR && ENCLFUNCG(arrd) == 0) {
        /* section descriptor is local.
         * For derived type members, this is true.
         * For non-common base symbols, this is true.
         * For nonhoisted common base symbols, need to initialize it.
         * For nondistributed common base symbols, this is true.
         */
        emit_alnd(sptr, memberast, TRUE, FALSE, 0);
        emit_secd(sptr, memberast, TRUE, FALSE);
      }
    }
    if (ENCLFUNCG(basesptr) && STYPEG(ENCLFUNCG(basesptr)) == ST_BLOCK &&
        ADJARRG(sptr)) {
      /* let it falls through */
    } else

      return;
  }

  /* don't generate templates for RUNTIME arrays in common */
  if (!ALLOCG(sptr) && !is_kopy_in_needed(basesptr) &&
      SCG(basesptr) != SC_CMBLK && !CCSYMG(sptr)) {
    emit_alnd(sptr, memberast, TRUE, FALSE, 0);
    emit_secd(sptr, memberast, TRUE, FALSE);
    SDSCINITP(DESCRG(sptr), 1);
    return;
  }
  /* this part is added to allow unmapped common symbols
     to have a descriptor */
  if (!ALLOCG(sptr)) {
    if (SCG(basesptr) == SC_CMBLK) {
      int cmn_sptr;
      cmn_sptr = CMBLKG(basesptr);
      if (cmn_sptr) {
        /* Need a descriptor for this */
        emit_alnd(sptr, memberast, TRUE, FALSE, 0);
        emit_secd(sptr, memberast, TRUE, FALSE);
      }
    }
  }

} /* wrap_symbol */

void
transform_wrapup(void)
{
  SPTR sptr;
  int this_entry;
  int newdsc;
  int exitStdNext;
  int saveEntryStd, saveExitStd;
  SPTR *wraplist;
  int wrapcount, routinescope_wrapcount;
  bool need_init;
  int i, j;

  f77_local = 0;
  EntryStd = STD_FIRST;
  ExitStd = gbl.exitstd;
  exitStdNext = STD_NEXT(ExitStd);
  init_change_mk_id();
  use_dummy_desc();
  desc_need_arrays();
  share_alnd(ST_ARRAY);
  share_secd();

  // Add user symbols to wraplist.  This list is used to order calls to
  // wrap_symbol and related routines, which generate entry/exit code for
  // routine or block scope symbols.  wraplist symbol order, which needs
  // to account for various list and code insertion reversals, is:
  //  1. routine scope non-earlyspec syms
  //  2. routine scope earlyspec syms
  //  3. block scope non-earlyspec syms
  //  4. block scope earlyspec syms
  NEW(wraplist, SPTR, stb.stg_avail - stb.firstosym);
  wrapcount = 0;
  for (i = 0; i <= 1; ++i) {
    routinescope_wrapcount = wrapcount;
    for (j = 0; j <= 1; ++j) {
      for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr) {
        if (IGNOREG(sptr) || CONSTRUCTSYMG(sptr) != i || EARLYSPECG(sptr) != j)
          continue;
        switch (STYPEG(sptr)) {
        default:
          break;
        case ST_VAR:
        case ST_ARRAY:
          wraplist[wrapcount++] = sptr;
          break;
        case ST_MEMBER:
          if (DESCRG(sptr) && SECDSCG(DESCRG(sptr)) &&
              STYPEG(SECDSCG(DESCRG(sptr))) != ST_MEMBER)
            wraplist[wrapcount++] = sptr;
          break;
        }
      }
    }
  }

  /* Entry arg processing and copy in/out. */
  for (this_entry = gbl.entries; this_entry != NOSYM;
       this_entry = SYMLKG(this_entry)) {
    unvisit_every_sptr();
    EntryStd = STD_NEXT(ENTSTDG(this_entry));
    init_fl();
    close_entry_guard();
    newdsc = newargs_for_entry(this_entry);
    this_entry_g = this_entry;
    new_dscptr_g = newdsc;
    /* there is an ast_visit inside interface_for_entry;
     * the scope of that ast_visit/ast_unvisit continues until below the
     * loop below */
    interface_for_entry(this_entry, newdsc);
/* keep track of which temps used for automatic array bounds have
 * been assigned by putting them on the A_VISIT list */
    for (i = 0; i < wrapcount; ++i) {
      sptr = wraplist[i];
      saveEntryStd = EntryStd;
      saveExitStd = ExitStd;
      if (CONSTRUCTSYMG(sptr)) {
        EntryStd = BLOCK_ENDPROLOG_STD(sptr);
        ExitStd = STD_PREV(BLOCK_EXIT_STD(sptr));
      }
      need_init = false;
      if ((gbl.internal <= 1 || INTERNALG(sptr)) && AUTOBJG(sptr))
        need_init = allocate_one_auto(sptr);
      wrap_symbol(sptr, 0, sptr);
      if (need_init)
        component_init_allocd_auto(mk_id(sptr), EntryStd);

      /* Look for adjustable-length non-automatic character symbols. */
      if (STYPEG(sptr) != ST_MEMBER &&
          (gbl.internal <= 1 || INTERNALG(sptr)) && /* not host */
          (!AUTOBJG(sptr)) &&                       /* not automatic */
          (!ENCLFUNCG(sptr))) {                     /* not module */
        int dty = DTYG(DTYPEG(sptr));
        if ((dty == TY_CHAR || dty == TY_NCHAR) && ADJLENG(sptr) &&
            (POINTERG(sptr) ||
             (ALLOCG(sptr) && !HCCSYMG(sptr) && !CCSYMG(sptr))) &&
            SCG(sptr) != SC_DUMMY && SCG(sptr) != SC_CMBLK) {
          add_auto_len(sptr, EntryStd);
        }
      }

      /* Create MIDNUMG of dummy adjustable array here.  We cannot do it
       * too early because there is a check in semfin, and we can't do it
       * in lower as it is too late for uplevel reference. */
      if (flg.smp && PARREFG(sptr) && SCG(sptr) != SC_DUMMY &&
          (ADJLENG(sptr) || AUTOBJG(sptr))) {
        int midnum = MIDNUMG(sptr);
        if (midnum == 0) {
          SCP(sptr, SC_BASED);
          midnum = sym_get_ptr(sptr);
          MIDNUMP(sptr, midnum);
          set_parref_flag2(midnum, sptr, 0);
        }
      }
      EntryStd = saveEntryStd;
      ExitStd = saveExitStd;
    }

    // Don't repeat block symbol processing for secondary entry points.
    wrapcount = routinescope_wrapcount;

    /* There is an ast_visit inside interface_for_entry, called above.
     * The scope of that ast_visit/ast_unvisit continues until here. */
    ast_unvisit();
    allocate_for_aligned_array();
    emit_fl();
    finish_fl();
    open_entry_guard(this_entry);
    if (ENTSTDG(this_entry) != EntryStd) {
      int s;
      /* Rewrite any assignments added at the entry point. */
      for (s = STD_NEXT(ENTSTDG(this_entry)); s != EntryStd; s = STD_NEXT(s)) {
        int ast;
        arg_gbl.std = s;
        arg_gbl.lhs = 0;
        arg_gbl.used = FALSE;
        arg_gbl.inforall = FALSE;
        gbl.lineno = STD_LINENO(s);
        ast = STD_AST(s);
        if (A_TYPEG(ast) == A_ASN)
          rewrite_asn(ast, s, TRUE, 0);
      }
      /* reset LINENO for any statements added at the entry point.
       * this allows the debugger to set its breakpoints at the proper
       * point, which is after the prologue code */
      for (s = STD_NEXT(ENTSTDG(this_entry)); s != EntryStd; s = STD_NEXT(s)) {
        STD_LINENO(s) = 0;
      }
    }
  }
  unvisit_every_sptr();
  for (this_entry = gbl.entries; this_entry != NOSYM;
       this_entry = SYMLKG(this_entry)) {
    EntryStd = STD_NEXT(ENTSTDG(this_entry));
    declare_array_dummys(this_entry);
  }
  ExitStd = STD_PREV(exitStdNext); /* add gbi_array free stuff at end */
  prepare_for_astout();
  do_change_mk_id();
  free_dtb();
  ExitStd = EntryStd = 0;
  FREE(wraplist);
}

/*
 * astout.c does not use dtb data structures
 */
static void
prepare_for_astout(void)
{
  int sptr;
  int secd, arrdsc, sdsc;

  for (sptr = stb.firstusym; sptr < stb.stg_avail; sptr++) {
    if (DTY(DTYPEG(sptr)) != TY_ARRAY)
      continue;
    if (SDSCG(sptr)) {
      goto do_parref;
      continue;
    }

    arrdsc = DESCRG(sptr);
    if (!arrdsc)
      continue;
    {
      if (SECDSCG(arrdsc)) {
        SDSCP(sptr, SECDSCG(arrdsc));
      } else {
        secd = SECDG(arrdsc);
        if (secd) {
          sdsc = INS_DESCR(secd);
          change_mk_id(arrdsc, sdsc);
          SDSCP(sptr, sdsc);
          SECDSCP(arrdsc, sdsc);
        } else if (want_descriptor_anyway(sptr) &&
                   STYPEG(arrdsc) == ST_ARRDSC) {
          /* in host subprogram, or -g, make a section descriptor */
          sdsc = sym_get_sdescr(sptr, -1);
          change_mk_id(arrdsc, sdsc);
          SDSCP(sptr, sdsc);
          SECDSCP(arrdsc, sdsc);
          /*
          SHOULD CALL  fix_sdsc_sc(sptr, sdsc, arrdsc); ????
          */
        }
      }
    }
  do_parref:
    if (flg.smp && PARREFG(sptr)) {
      int sdsc = SDSCG(sptr);
      if (sdsc && !PARREFG(sdsc) && DESCUSEDG(sptr)) {
        set_parref_flag2(sptr, 0, 0);
      }
    }
  }
  undouble_callee_args_f90();
  for (sptr = stb.firstusym; sptr < stb.stg_avail; sptr++) {
    /* clear alndp, secdp fields for array descriptors */
    if (DTY(DTYPEG(sptr)) != TY_ARRAY)
      continue;
    arrdsc = DESCRG(sptr);
    if (!arrdsc)
      continue;
    ALNDP(arrdsc, 0);
    SECDP(arrdsc, 0);
  }
}

/*
 * The argument list was 'doubled', each argument getting its 'descriptor'
 * argument.
 * This is HPF heritage.  Now we remove the unneeded descriptor arguments,
 * and add in REFLECTED arguments.
 */

static LOGICAL arg_has_descriptor(int);

static LOGICAL
pass_reflected_arg_by_value(int arg)
{
  return FALSE;
}

static void
undouble_callee_args_f90(void)
{
  int this_entry;
  int dscptr, new_dscptr;
  int narg, orignarg, newnarg;
  int i;
  int arg, descr;
  int oldarg;

  for (this_entry = gbl.entries; this_entry != NOSYM;
       this_entry = SYMLKG(this_entry)) {
    int f_descr;
    int istart;
    narg = PARAMCTG(this_entry);
    if (!narg)
      continue;
    orignarg = narg / 2;
    newnarg = 0;
    dscptr = DPDSCG(this_entry);
    new_dscptr = get_arg_table();
    for (i = 0; i < orignarg; i++) {
      int arg = aux.dpdsc_base[dscptr + i];
      put_arg_table(arg);
      newnarg++;
      if (pass_reflected_arg_by_value(arg))
        newnarg++;
    }

    istart = 0;
    f_descr = 0;
    if (MVDESCG(this_entry)) {
      f_descr = FVALG(this_entry);
      if (f_descr && f_descr == aux.dpdsc_base[dscptr + 0]) {
        oldarg = NEWARGG(f_descr);
        if (arg_has_descriptor(oldarg)) {
          f_descr = aux.dpdsc_base[dscptr + orignarg + 0];
          istart = 1;
        }
      }
    }
    for (i = istart; i < orignarg; i++) {
      arg = aux.dpdsc_base[dscptr + i];
      oldarg = 0;
      if (arg)
        oldarg = NEWARGG(arg);
      descr = aux.dpdsc_base[dscptr + orignarg + i];
      if (arg_has_descriptor(oldarg)) {
        put_arg_table(descr);
        newnarg++;
      } else {
        /* change SC from DUMMY to LOCAL */
        if (XBIT(57, 0x10000)) {
          if (CLASSG(descr)) {
            if (STYPEG(SCOPEG(descr)) == ST_MODULE)
              SCP(descr, SC_EXTERN);
            else
              SCP(descr, SC_STATIC);
          } else
            SCP(descr, SC_LOCAL);
        }
      }
    }
    if (istart) {
      put_arg_table(f_descr);
      newnarg++;
    }
    PARAMCTP(this_entry, newnarg);
    DPDSCP(this_entry, new_dscptr);
  }
}

static LOGICAL
arg_has_descriptor(int oldarg)
{
  return oldarg > NOSYM &&
         (ASSUMSHPG(oldarg) || POINTERG(oldarg) || IS_PROC_DUMMYG(oldarg) ||
          ALLOCATTRG(oldarg) || is_kopy_in_needed(oldarg));
}

void
emit_fl(void)
{
  int nargs, argt;
  int ast;
  int i;

  if (fl.avl == 0)
    return;
  nargs = fl.avl + 1;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = mk_cval(fl.avl, DT_INT);
  for (i = 0; i < fl.avl; i++) {
    ARGT_ARG(argt, i + 1) = mk_id(fl.base[(fl.avl - 1) - i]);
  }
  ast = mk_func_node(A_CALL, mk_id(sym_mkfunc(mkRteRtnNm(RTE_freen), DT_NONE)),
                     nargs, argt);
  gbl.exitstd = add_stmt_after(ast, gbl.exitstd);
}

/*	 if (l) then
 *	      allocate (a(1:100))
 *	 else
 *	     allocate (a(2:101))
 *	 endif
 */

void
emit_alnd_secd(int sptr, int memberast, LOGICAL free_flag, int std,
               int allocbounds)
{
  int alnd, secd, ast;
  int old_desc, old_desc1;
  int savefreeing;
  int saveEntryStd, saveExitStd;
  bool tmplcall;

  if (free_flag) {
    init_change_mk_id();
    saveEntryStd = EntryStd;
    saveExitStd = ExitStd;
    EntryStd = std;
    ExitStd = std;
  }
  savefreeing = allocatable_freeing;
  allocatable_freeing = 1;
  if (ALNDG(DESCRG(sptr)))
    old_desc = TMPL_DESCR(ALNDG(DESCRG(sptr)));
  else
    old_desc = 0;
  alnd = make_alnd(sptr);
  ALNDP(DESCRG(sptr), alnd);
  if (alnd) {
    if (SECDSCG(DESCRG(sptr)))
      old_desc = SECDSCG(DESCRG(sptr));
    if (old_desc == 0)
      TMPL_DESCR(alnd) = sym_get_sdescr(sptr, -1);
    else {
      TMPL_DESCR(alnd) = old_desc;
      VISITP(old_desc, 0);
    }
    if (free_flag)
      ast_visit(1, 1);
    tmplcall = emit_alnd(sptr, memberast, free_flag, TRUE, allocbounds);
    /* Copy the type length of a string array pointer from a target string
       array during pointer association. */
    ast = STD_AST(std);
    if (XBIT(57, 0x200000) && A_TYPEG(ast) == A_ICALL &&
        A_OPTYPEG(ast) == I_PTR2_ASSIGN && POINTERG(sptr)) {
      int target_ast, dtype;
      target_ast = ARGT_ARG(A_ARGSG(ast), 1);
      dtype = A_DTYPEG(target_ast);
      if (DTY(dtype) == TY_ARRAY && DTY(DTY(dtype + 1)) == TY_CHAR) {
        /* The 5th argument to RTE_template is the type length. Pass the length
           if known, otherwise pass the result of a run-time LEN() call. */
        assert(tmplcall, "emit_alnd_secd: expected template call", 0, ERR_Fatal);
        if (string_length(DTY(dtype + 1)) != 0) {
          ARGT_ARG(A_ARGSG(STD_AST(STD_PREV(std))), 4) =
              mk_isz_cval(string_length(DTY(dtype + 1)), astb.bnd.dtype);
        } else {
          int sizeAst;
          sizeAst = sym_mkfunc_nodesc(mkRteRtnNm(RTE_lena), astb.bnd.dtype);
          sizeAst = begin_call(A_FUNC, sizeAst, 1);
          add_arg(target_ast);
          if (STYPEG(sptr) == ST_MEMBER)
            ARGT_ARG(A_ARGSG(STD_AST(STD_PREV(STD_PREV(std)))), 4) = sizeAst;
          else
            ARGT_ARG(A_ARGSG(STD_AST(STD_PREV(std))), 4) = sizeAst;
        }
      }
    }
    if (free_flag)
      ast_unvisit();
    /* Added second condition to `if' below.
     * We do not want to propagate SAVE to descriptor
     * if we're placing pointers in common blocks.
     * See also transfrm.c and astout.c.
     */
    if (SAVEG(sptr) && !POINTERG(sptr)) {
      SAVEP(TMPL_DESCR(ALNDG(DESCRG(sptr))), 1);
      if (NO_PTR || /* pointers not allowed in output */
          (NO_CHARPTR && DTYG(DTYPEG(sptr)) == TY_CHAR) ||
          (NO_DERIVEDPTR && DTYG(DTYPEG(sptr)) == TY_DERIVED))
        SAVEP(TMPL_DESCR(ALNDG(DESCRG(sptr))), 1);
    }
    if (GSCOPEG(sptr)) {
      GSCOPEP(TMPL_DESCR(ALNDG(DESCRG(sptr))), 1);
    }
  }

  if (SECDG(DESCRG(sptr)))
    old_desc1 = INS_DESCR(SECDG(DESCRG(sptr)));
  else
    old_desc1 = 0;
  secd = make_secd(sptr);
  SECDP(DESCRG(sptr), secd);
  /* predefined descriptor, case like MODULE */
  if (SECDSCG(DESCRG(sptr)))
    old_desc1 = SECDSCG(DESCRG(sptr));
  if (old_desc1) {
    INS_DESCR(SECDG(DESCRG(sptr))) = old_desc1;
    VISIT2P(INS_DESCR(SECDG(DESCRG(sptr))), 0);
    VISIT2P(old_desc1, 0);
  } else if (SDSCG(sptr) && HCCSYMG(sptr)) {
    // If there is already a (compiler-created) SDSC, use it.
    INS_DESCR(SECDG(DESCRG(sptr))) = SDSCG(sptr);
  } else {
    INS_DESCR(SECDG(DESCRG(sptr))) = sym_get_sdescr(sptr, -1);
  }
  change_mk_id(DESCRG(sptr), INS_DESCR(SECDG(DESCRG(sptr))));
  emit_secd(sptr, memberast, free_flag, TRUE);
  if (SAVEG(sptr) && !POINTERG(sptr)) {
    SAVEP(INS_DESCR(SECDG(DESCRG(sptr))), 1);
    if (NO_PTR || /* pointers not allowed in output */
        (NO_CHARPTR && DTYG(DTYPEG(sptr)) == TY_CHAR) ||
        (NO_DERIVEDPTR && DTYG(DTYPEG(sptr)) == TY_DERIVED))
      SAVEP(INS_DESCR(SECDG(DESCRG(sptr))), 1);
  }
  if (GSCOPEG(sptr)) {
    GSCOPEP(INS_DESCR(SECDG(DESCRG(sptr))), 1);
  }

  allocatable_freeing = savefreeing;
  if (free_flag) {
    do_change_mk_id();
    EntryStd = saveEntryStd;
    ExitStd = saveExitStd;
  }
}

static int
size_of_dtype(int dtype, int sptr, int memberast)
{
  int sizeAst;
  if (DTY(dtype) == TY_CHAR) {
    /* assumed length character */
    if (dtype == DT_ASSCHAR || dtype == DT_DEFERCHAR) {
      sizeAst = sym_mkfunc_nodesc(mkRteRtnNm(RTE_lena), astb.bnd.dtype);
      sizeAst = begin_call(A_FUNC, sizeAst, 1);
      add_arg(check_member(memberast, mk_id(sptr)));
    } else if (CVLENG(sptr) > NOSYM) {
      sizeAst = mk_bnd_int(mk_id(CVLENG(sptr)));
    } else {
      int clen;
      clen = DTY(dtype + 1);
      if (A_ALIASG(clen)) {
        sizeAst = A_ALIASG(clen);
      } else {
        sizeAst = clen;
      }
      sizeAst = mk_bnd_int(sizeAst);
    }
  } else {
    sizeAst = mk_isz_cval(size_of(dtype), astb.bnd.dtype);
  }
  return sizeAst;
}

static void
emit_secd(int sptr, int memberast, LOGICAL free_flag, LOGICAL for_allocate)
{
  int secd;
  int arrdsc, stype, descr;
  int nargs, argt, astnew;
  int ndim;
  int j;
  int collapse;
  int alnd;
  int func;
  int dtype;
  int entryStd = EntryStd;
  int sizeast;
  int descr_ast;

#ifdef IPA0
  ipa_ALN_INFO info[7];
#endif

  if (is_bad_dtype(DTYPEG(sptr)))
    return;
  if (NODESCG(sptr))
    return;
  if (!DESCUSEDG(sptr))
    return;
  if (normalize_bounds(sptr) && SCG(sptr) == SC_DUMMY && !SEQG(sptr)) {
    return;
  }

  arrdsc = DESCRG(sptr);
  assert(arrdsc > NOSYM, "emit_secd: descriptor does not exist", sptr, 3);
  alnd = ALNDG(arrdsc);
  secd = SECDG(arrdsc);
  /* case where array used as a template but never used as array */
  if (!secd)
    return;
  descr = INS_DESCR(secd);
  stype = STYPEG(descr);
  if (stype == ST_ARRDSC) {
    int secdsc;
    secdsc = SECDSCG(sptr);
    if (secdsc) {
      stype = STYPEG(secdsc);
    } else {
      stype = STYPEG(ARRAYG(sptr));
    }
  }
  /* don't initialize host subprogram symbols */
  if (!XBIT(57, 0x40000)) {
    if (SDSCINITG(arrdsc) && !realign && !redistribute && !for_allocate &&
        gbl.internal > 1 && !INTERNALG(descr) && stype != ST_MEMBER)
      return;
  } else {
    if (!realign && !redistribute && !for_allocate && gbl.internal > 1 &&
        !INTERNALG(arrdsc) && stype != ST_MEMBER)
      return;
  }
  ndim = INS_RANK(secd);

  if (stype != ST_MEMBER) {
    if (VISIT2G(descr))
      return;
    VISIT2P(descr, 1);
  }

/* void
 * ENTHPF(INSTANCE,instance)
 *  (F90_Desc *dd, F90_Desc *td,
 *   __INT_T *p_kind, __INT_T *p_len, __INT_T *p_collapse, ...)
 * ... = { [ __INT_T *no, __INT_T *po, ] }*
 */
  nargs = 5 + 2 * ndim + 1; /* Fix updated computation of nargs */
  argt = mk_argt(nargs);

  descr_ast = check_member(memberast, mk_id(INS_DESCR(secd)));
  ARGT_ARG(argt, 0) = descr_ast;
  assert(INS_TEMPLATE(secd), "emit_secd: TEMPLATE does not exist for", sptr, 3);
  ARGT_ARG(argt, 1) = check_member(memberast, mk_id(INS_TEMPLATE(secd)));
  ARGT_ARG(argt, 2) =
      mk_isz_cval(dtype_to_arg(DTY(INS_DTYPE(secd) + 1)), astb.bnd.dtype);

  dtype = DTY(INS_DTYPE(secd) + 1);
  sizeast = size_of_dtype(typed_alloc != DT_NONE ? typed_alloc : dtype, sptr,
                          memberast);
  ARGT_ARG(argt, 3) = sizeast;

  j = 4;

  collapse = TMPL_COLLAPSE(alnd) | TMPL_ISSTAR(alnd);
  ARGT_ARG(argt, j) = mk_isz_cval(collapse, astb.bnd.dtype);
  j++;
  func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_instance), DT_NONE));
  nargs = j;
  astnew = mk_func_node(A_CALL, func, nargs, argt);
  add_stmt_before(astnew, entryStd);

  set_type_in_descriptor(descr_ast, sptr, typed_alloc, 0 /* no parent AST */,
                         entryStd);
}

static void
emit_target_alnd(int alnd, int memberast, LOGICAL free_flag)
{
  int target;
  int sc, sc1;
  int alnd1;

  if (!TMPL_ALIGN_TARGET(alnd))
    return;

  target = TMPL_ALIGN_TARGET(alnd);
  DESCUSEDP(target, 1);
  NODESCP(target, 0);
  sc = TMPL_ALIGNEE_SC(alnd);
  sc1 = TMPL_TARGET_SC(alnd);

  /* if special case of inherited template, use the target */
  if (TMPL_TYPE(alnd) == INHERITED) {
    NODESCP(target, 1);
    return;
  }

  if (sc == ALLOC_SC && sc1 == ALLOC_SC) {
    alnd1 = ALNDG(DESCRG(target));
    assert(alnd, "emit_target_alnd: misplaced ALLOCATABLE alignment", target,
           3);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(alnd1);
    return;
  }
  if (sc == ALLOC_SC && sc1 == DUMMY_SC) {
    TMPL_TARGET_DESCR(alnd) = DESCRG(target);
    return;
  }

  if (sc == ALLOC_SC && sc1 == STATIC_SC) {
    /* static target (template) must be handled by
     * handle_nonalloc_template
     */
    TMPL_TARGET_DESCR(alnd) = DESCRG(target);
    return;
  }
  if (sc == ALLOC_SC && sc1 == COMMON_SC) {
    TMPL_TARGET_DESCR(alnd) = DESCRG(target);
    return;
  }
  if (sc == DUMMY_SC && sc1 == ALLOC_SC) {
    if (MDALLOCG(target))
      TMPL_TARGET_DESCR(alnd) = SECDSCG(DESCRG(target));
    else
      assert(0, "emit_target_alnd: wrong alignment", target, 3);
    return;
  }
  if (sc == DUMMY_SC && sc1 == DUMMY_SC) {
    /* if this is for a CALL statement, the dummy arguments
     * are processed in reverse order, don't need to recurse here */
    emit_kopy_in(target, this_entry_g, 0);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  if (sc == DUMMY_SC && sc1 == STATIC_SC) {
    emit_alnd(target, memberast, free_flag, FALSE, 0);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  if (sc == DUMMY_SC && sc1 == COMMON_SC) {
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  if (sc == STATIC_SC && sc1 == ALLOC_SC) {
    if (MDALLOCG(target))
      TMPL_TARGET_DESCR(alnd) = SECDSCG(DESCRG(target));
    else
      assert(0, "emit_target_alnd: wrong alignment", target, 3);
    return;
  }
  if (sc == STATIC_SC && sc1 == DUMMY_SC) {
    if (!TMPL_DESCR(ALNDG(DESCRG(target))))
      emit_kopy_in(target, this_entry_g, 0);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  if (sc == STATIC_SC && sc1 == STATIC_SC) {
    emit_alnd(target, memberast, free_flag, FALSE, 0);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  if (sc == STATIC_SC && sc1 == COMMON_SC) {
    emit_alnd(target, memberast, free_flag, FALSE, 0);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  if (sc == COMMON_SC && sc1 == ALLOC_SC) {
    assert(0, "emit_target_alnd: wrong alignment", target, 3);
    return;
  }
  if (sc == COMMON_SC && sc1 == DUMMY_SC) {
    assert(0, "emit_target_alnd: wrong alignment", target, 3);
    return;
  }
  if (sc == COMMON_SC && sc1 == STATIC_SC) {
    emit_alnd(target, memberast, free_flag, FALSE, 0);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  if (sc == COMMON_SC && sc1 == COMMON_SC) {
    emit_alnd(target, memberast, free_flag, FALSE, 0);
    TMPL_TARGET_DESCR(alnd) = TMPL_DESCR(ALNDG(DESCRG(target)));
    return;
  }
  assert(0, "emit_target_alnd: something is wrong with dist", target, 3);
}

/** \brief Scan an expression for compiler-created symbols.
    \return
    +  0 if the expression contains a compiler-created symbol
    +  1 if all variables are user declared
    + -1 if all symbols are constants (or something other than above)

    We use this routine when examining the distribution block factor
    expression.

 */
int
find_cc_symbols(int ast)
{
  int rslt = -1, k;

  k = A_TYPEG(ast);

  if (k) {

    if (k == A_UNOP || k == A_BINOP) {

      if (A_TYPEG(A_LOPG(ast))) {
        rslt = find_cc_symbols(A_LOPG(ast));
        if (!rslt)
          return 0;
      }

      if (A_TYPEG(A_ROPG(ast))) {
        int rslt_right;

        rslt_right = find_cc_symbols(A_ROPG(ast));

        if (!rslt_right)
          return 0;
        else if (rslt_right == 1)
          rslt = 1;
      }
    } else if (k == A_ID) {
      int j;

      j = A_SPTRG(ast);
      if (CCSYMG(j) || HCCSYMG(j))
        return 0;
      else
        return 1;
    }
  }

  return rslt;
}

void
set_typed_alloc(DTYPE t)
{
  /* used to pass the type into emit_alnd() when it's called via typed
   * allocation statement (e.g., allocate(type-spec::object) )
   */
  typed_alloc = t;
}

/*
 *  Set the type in a descriptor to a DTYPE, if known, or to the
 *  run-time type of a symbol.  The type is set by calling the
 *  run-time library routine RTE_set_type or RTE_set_intrin_type
 *  as appropriate.
 */
void
set_type_in_descriptor(int descriptor_ast, int sptr, DTYPE dtype0,
                       int parent_ast, int before_std)
{
  DTYPE dtype = dtype0;
  int tag_sptr, dtype_arg_ast, type_ast = 0;
  FtnRtlEnum func = RTE_no_rtn;

  if (dtype == DT_NONE && sptr > NOSYM)
    dtype = DTYPEG(sptr);
  if (is_array_dtype(dtype))
    dtype = array_element_dtype(dtype);

  tag_sptr = get_struct_tag_sptr(dtype);
  if (tag_sptr > NOSYM || is_unl_poly(sptr)) {
    /* polymorphic or derived type */
    func = RTE_set_type;
    if (tag_sptr > NOSYM && !UNLPOLYG(tag_sptr)) {
      /* known monomorphic derived type */
      int typedsc_sptr = get_static_type_descriptor(tag_sptr);
      if (typedsc_sptr > NOSYM)
        type_ast = mk_id(typedsc_sptr);
    }
    if (type_ast == 0 && sptr > NOSYM && (CLASSG(sptr) || FINALIZEDG(sptr))) {
      type_ast = find_descriptor_ast(sptr, parent_ast);
      if (type_ast == 0) {
        int typedsc_sptr = get_static_type_descriptor(sptr);
        if (typedsc_sptr > NOSYM)
          type_ast = mk_id(typedsc_sptr);
      }
    }
  } else if ((dtype_arg_ast = dtype_to_arg(dtype)) > 0) {
    /* intrinsic type */
    func = RTE_set_intrin_type;
    type_ast = mk_unop(OP_VAL, mk_cval1(dtype_arg_ast, DT_INT), DT_INT);
  }

  if (type_ast > 0 && type_ast != descriptor_ast) {
    int argt = mk_argt(2), astnew;
    int func_ast = mk_id(sym_mkfunc_nodesc(mkRteRtnNm(func), DT_NONE));
    ARGT_ARG(argt, 0) = descriptor_ast;
    ARGT_ARG(argt, 1) = (SCG(sptr) == SC_DUMMY && CLASSG(sptr)) ?
                        mk_id(get_type_descr_arg(gbl.currsub, sptr)) : type_ast;
    astnew = mk_func_node(A_CALL, func_ast, 2, argt);
    add_stmt_before(astnew, before_std);
  }
}

/*
 *   void
 *   pghpf_template
 *   (distr **template, __INT4_T *rank, __INT4_T *flags, ...)
 *   ... = [  [  proc *dist_target,  ]
 *	     __INT4_T *isstar,
 *	     {	__INT4_T *blocks,  }*  ]p
 *
 *	[  section *align_target, __INT4_T *conform,
 *	     [	__INT4_T *collapse,
 *	       {  __INT4_T *axis, __INT4_T *stride, __INT4_T *offset,  }*
 *	       __INT4_T *single,
 *	       {  __INT4_T *coordinate,  }*  ]	]
 *	  {  __INT4_T *lb,
 *	     [	__INT4_T *ub,  ]  }*
 */
static bool
emit_alnd(int sptr, int memberast, LOGICAL free_flag, LOGICAL for_allocate,
          int allocbounds)
{
  int alnd;
  int arrdsc, stype;
  int nargs, argt, astnew, cargs;
  int ndim;
  int func, descr;
  int realign1, redistribute1;
  int proc, proc_descr;
  int entryStd = EntryStd;

  if (is_bad_dtype(DTYPEG(sptr)))
    return false;
  if (NODESCG(sptr))
    return false;
  if (!DESCUSEDG(sptr))
    return false;
  if (normalize_bounds(sptr) && SCG(sptr) == SC_DUMMY && !SEQG(sptr)) {
    return false;
  }

  arrdsc = DESCRG(sptr);
  assert(arrdsc, "emit_alnd: descriptor does not exist", sptr, 3);
  alnd = ALNDG(arrdsc);
  assert(alnd, "emit_alnd: TEMPLATE does not exist", sptr, 3);
  ndim = TMPL_RANK(alnd);

  descr = TMPL_DESCR(alnd);
  stype = STYPEG(descr);
  if (stype == ST_ARRDSC) {
    int secdsc;
    secdsc = SECDSCG(sptr);
    if (secdsc) {
      stype = STYPEG(secdsc);
    } else {
      stype = STYPEG(ARRAYG(sptr));
    }
  }

  /* don't have to initialize descriptors for host subprogram symbols */
  if (SDSCINITG(arrdsc) && !realign && !redistribute && !for_allocate &&
      gbl.internal > 1 && !INTERNALG(descr) && stype != ST_MEMBER)
    return false;

  if (VISITG(descr))
    return false;
  VISITP(descr, 1);

  /* don't call recursively pghpf_realign and pghpf_redistribute */
  realign1 = realign;
  redistribute1 = redistribute;
  realign = 0;
  redistribute = 0;

  /* get target descriptor */
  if (TMPL_ALIGN_TARGET(alnd)) {
    emit_target_alnd(alnd, memberast, free_flag);
    assert(TMPL_TARGET_DESCR(alnd), "emit_alnd: no descriptor exist", sptr, 3);
  }

  proc = TMPL_DIST_TARGET(alnd);
  proc_descr = TMPL_DIST_TARGET_DESCR(alnd);
  if (proc_descr) {
    trans_mkproc(proc);
  }

  nargs = 8 + 6 * ndim + 9 + 7;
  cargs = 0;
  if (XBIT(57, 0x200000))
    nargs += 2; /* two more arguments for kind/len */
  argt = mk_argt(nargs);
  ARGT_ARG(argt, cargs++) = check_member(memberast, mk_id(TMPL_DESCR(alnd)));
  ARGT_ARG(argt, cargs++) = mk_isz_cval(TMPL_RANK(alnd), astb.bnd.dtype);
  ARGT_ARG(argt, cargs++) = mk_isz_cval(TMPL_FLAG(alnd), astb.bnd.dtype);
  if (XBIT(57, 0x200000)) { /* leave room for kind/len */
    ARGT_ARG(argt, cargs++) = mk_isz_cval(0, astb.bnd.dtype);
    ARGT_ARG(argt, cargs++) = mk_isz_cval(0, astb.bnd.dtype);
  }
  nargs = fill_argt_with_alnd(sptr, memberast, argt, alnd, cargs, redistribute1,
                              allocbounds);

  if (redistribute1)
    func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_redistribute), DT_NONE));
  else if (realign1)
    func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_realign), DT_NONE));
  else {

    /*
     * If the blocking factor is set by a user declared variable,
     * we need to generate a call to pghpf_check_block_size to
     * insure that the block size >= 1 ...
     *
     * We generate a call to pghpf_check_block_size if all variables
     * in the blocking factor expression are user declared. If there
     * are any compiler created variables, then we do not generate
     * the call. Also we do not generate the call if the expression
     * contains all constants.
     */

    if (XBIT(57, 0x200000)) {
      func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_template), DT_NONE));
    } else {
      func = mk_id(sym_mkfunc(mkRteRtnNm(RTE_templateDsc), DT_NONE));
    }
  }
  astnew = mk_func_node(A_CALL, func, nargs, argt);
  add_stmt_before(astnew, entryStd);

  /* Set the type in the descriptor for a derived type member. */
  if (STYPEG(sptr) == ST_MEMBER)
    set_type_in_descriptor(check_member(memberast, mk_id(TMPL_DESCR(alnd))),
                           sptr, typed_alloc, 0 /* no parent AST */, entryStd);

  return true;
}

void
make_temp_descriptor(int ast_ele, SPTR sptr_orig, SPTR sptr_tmp, int before_std)
{
    /* call pgf90_temp_desc(tmp desc, orig desc) */
    SPTR sptr_descr;
    int  ast;
    int nargs = 2;
    int argt = mk_argt(nargs);
    sptr_descr = DESCRG(sptr_tmp);
    assert(sptr_descr,"missing descriptor for tmp",(int)sptr_tmp,ERR_Fatal);
    ARGT_ARG(argt, 0) = mk_id(sptr_descr);
    sptr_descr = DESCRG(sptr_orig);
    assert(sptr_descr,"missing descriptor for orig",(int)sptr_orig,ERR_Fatal);
    ARGT_ARG(argt, 1) = check_member(ast_ele,mk_id(sptr_descr));

    ast =
        mk_func_node(A_CALL, 
                     mk_id(sym_mkfunc(mkRteRtnNm(RTE_tmp_desc), DT_NONE)),
                     nargs, argt);
    (void) add_stmt_before(ast, before_std);
}

void
init_sdsc_from_dtype(int sptr, DTYPE dtype, int before_std)
{
  init_sdsc(sptr, dtype, before_std, 0);
  set_type_in_descriptor(mk_id(SDSCG(sptr)), sptr, DT_NONE, 0, before_std);
}

static int
fill_argt_with_alnd(int sptr, int memberast, int argt, int alnd, int j,
                    int redist, int allocbounds)
{
  int i;
  int sptrTarg, tmpl_descr;
  int ndim;
  int flag;
  int asd;

  ndim = TMPL_RANK(alnd);

  if (memberast && A_TYPEG(memberast) == A_SUBSCR)
    memberast = A_LOPG(memberast);
  if (!memberast || A_TYPEG(memberast) != A_MEM)
    memberast = 0;

  if (memberast) {
    int s, a, dt, m;
    if (STYPEG(sptr) == ST_MEMBER && memberast) {
      s = MIDNUMG(sptr);
      if (s) {
        a = mk_id(s);
        ast_replace(a, check_member(memberast, a));
      }
      s = PTROFFG(sptr);
      if (s) {
        a = mk_id(s);
        ast_replace(a, check_member(memberast, a));
      }
    }
    /* any members of the same datatype might have SDSC fields */
    m = A_MEMG(memberast);
    dt = ENCLDTYPEG(A_SPTRG(m));
    for (m = DTY(dt + 1); m > NOSYM; m = SYMLKG(m)) {
      s = SDSCG(m);
      if (s) {
        a = mk_id(s);
        ast_replace(a, check_member(memberast, a));
      }
    }
  }

  if (TMPL_DIST_TARGET_DESCR(alnd)) {
    ARGT_ARG(argt, j) = check_member(
        memberast, ast_rewrite(mk_id(TMPL_DIST_TARGET_DESCR(alnd))));
    j++;
  }

  flag = TMPL_FLAG(alnd);
  if (TMPL_TYPE(alnd) == DISTRIBUTED &&
      ((getbits(flag, DIST_FORMAT_SHIFT, 2) == PRESCRIP) ||
       (getbits(flag, DIST_FORMAT_SHIFT, 2) == DESCRIP))) {
    ARGT_ARG(argt, j) = mk_isz_cval(TMPL_ISSTAR(alnd), astb.bnd.dtype);
    j++;
  }

  sptrTarg = TMPL_ALIGN_TARGET(alnd);
  if (sptrTarg) {
    tmpl_descr = TMPL_TARGET_DESCR(alnd);
    ARGT_ARG(argt, j) = check_member(memberast, ast_rewrite(mk_id(tmpl_descr)));
    j++;
  }

  if (TMPL_CONFORM(alnd)) {
    ARGT_ARG(argt, j) = TMPL_CONFORM(alnd);
    j++;
  }

  if (!is_set(TMPL_FLAG(alnd), __IDENTITY_MAP) && TMPL_TYPE(alnd) == ALIGNED) {
    ARGT_ARG(argt, j) = mk_isz_cval(TMPL_COLLAPSE(alnd), astb.bnd.dtype);
    j++;
  }

  if (allocbounds) {
    assert(A_TYPEG(allocbounds) == A_SUBSCR, "fill_argt: expecting subscript",
           A_TYPEG(allocbounds), 4);
    asd = A_ASDG(allocbounds);
    assert(ASD_NDIM(asd) == ndim, "fill_argt: dimensionality doesn't match",
           ASD_NDIM(asd), 4);
  }
  for (i = 0; i < ndim; ++i) {
    int ast;

    if (allocbounds) {
      /* this is the A_SUBSCR with the bounds */
      int triplet, stride;
      triplet = ASD_SUBS(asd, i);
      assert(A_TYPEG(triplet) == A_TRIPLE,
             "fill_argt: expecting triplet in allocate subscript",
             A_TYPEG(triplet), 4);

      /* Power: create lb and ub temps to avoid a gcc bug where a register was
       * not being updated after calls to mk_bnd_int, resulting in storing the
       * result in the old (now reallocated via mk_argt) astb.argt structure.
       * See: FS21905.
       */
      if ((stride = A_STRIDEG(triplet)) != 0 && A_TYPEG(stride) == A_CNST &&
          ad_val_of(A_SPTRG(stride)) < 0) {
        const int ub = mk_bnd_int(A_UPBDG(triplet));
        const int lb = mk_bnd_int(A_LBDG(triplet));
        ARGT_ARG(argt, j) = ub;
        j++;
        ARGT_ARG(argt, j) = lb;
        j++;
      } else {
        const int lb = mk_bnd_int(A_LBDG(triplet));
        const int ub = mk_bnd_int(A_UPBDG(triplet));
        ARGT_ARG(argt, j) = lb;
        j++;
        ARGT_ARG(argt, j) = ub;
        j++;
      }
    } else {
      if (TMPL_LB(alnd, i)) {
        ast =
            mk_bnd_int(check_member(memberast, ast_rewrite(TMPL_LB(alnd, i))));
        if (normalize_bounds(sptr))
          ast = astb.bnd.one;
        ARGT_ARG(argt, j) = ast;
        j++;
      }
      if (TMPL_UB(alnd, i)) {
        ast =
            mk_bnd_int(check_member(memberast, ast_rewrite(TMPL_UB(alnd, i))));
        if (normalize_bounds(sptr)) {
          ast = mk_binop(OP_SUB, ast,
                         check_member(memberast, ast_rewrite(TMPL_LB(alnd, i))),
                         astb.bnd.dtype);
          ast = mk_binop(OP_ADD, ast, astb.bnd.one, astb.bnd.dtype);
        }
        ARGT_ARG(argt, j) = ast;
        j++;
      }
    }
  }
  return j;
}

/* These two routines(get_arg_table and pur_arg_table) are
 * to get new arg list dscptr and put arg into list
 * is used to create new entry arg list such as:
 *  interface; function func(a,b)
 */
static int
get_arg_table(void)
{
  return aux.dpdsc_avl;
}

static void
put_arg_table(int arg)
{
  NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
       aux.dpdsc_size + 100);
  *(aux.dpdsc_base + (aux.dpdsc_avl++)) = arg;
}

/* This will create new variables for subroutine interface.
 * subroutine foo(array1, scalar1) will have new four new variables
 * subroutine foo(actual_array1, actual_scalar1, sec_array1, sec_scalar1)
 * This will be written at above format. The new variables is also stored
 * inti aux.dpdsc axuliary data structure. It stores first actual variables
 * and	then section descriptor. It will return the base address of this
 * new dpdsc which have 2*narg element. That will be stored into
 * ST_ENTRYs DPDSC and PARAMCT by interface_for_entry.
 */

static int
newargs_for_entry(int this_entry)
{
  int dscptr;
  int arg, narg;
  int i;
  int formal;
  int new_dscptr;
  int newarg, newdsc;

  narg = PARAMCTG(this_entry);
  dscptr = DPDSCG(this_entry);
  new_dscptr = get_arg_table();
  for (i = 0; i < narg; i++) {
    arg = aux.dpdsc_base[dscptr];
    if (arg == 0) {
      formal = 0;
    } else if (STYPEG(arg) != ST_ARRAY && STYPEG(arg) != ST_VAR &&
               !IS_PROC_DUMMYG(arg)) {
      formal = arg;
    } else {
      newarg = NEWARGG(arg);
      newdsc = NEWDSCG(arg);
      if (normalize_bounds(arg)) {
        if (newarg == arg) {
          if (needs_redim(arg))
            /* ...create new dummy symbol. */
            newarg = 0;
        } else {
          if (!needs_redim(arg))
            /* ...don't create new symbol. */
            newarg = arg;
        }
      }
      if (!F90POINTERG(arg) &&
          ((is_array_type(arg) && !is_bad_dtype(DTYPEG(arg))) ||
           POINTERG(arg) || ALLOCATTRG(arg) || IS_PROC_DUMMYG(arg))) {
        /* use the address field to hold new name for param/section */
        formal = newarg;
        if (XBIT(57, 0x80000) && (formal == arg || formal == 0) &&
            (POINTERG(arg) || ALLOCATTRG(arg) || IS_PROC_DUMMYG(arg))) {
          if (MIDNUMG(arg)) {
            SCP(MIDNUMG(arg), SC_DUMMY);
            OPTARGP(MIDNUMG(arg), OPTARGG(arg));
          }
          if (formal == 0)
            formal = arg;
        }
        if (!formal)
          formal = sym_get_formal(arg);
        newarg = formal;
        NEWARGP(newarg, arg);
        NEWDSCP(newarg, 0);
      } else
        formal = arg;
      NEWARGP(arg, newarg);
      NEWDSCP(arg, newdsc);
    }
    put_arg_table(formal);
    dscptr++;
  }
  dscptr = DPDSCG(this_entry);
  for (i = 0; i < narg; i++) {
    arg = aux.dpdsc_base[dscptr];
    if (arg == 0) {
      newdsc = sym_get_sec("alt", 1);
    } else {
      newdsc = NEWDSCG(arg);
      if (newdsc == 0) {
        set_preserve_descriptor(CLASSG(arg) || is_procedure_ptr(arg) ||
                                (sem.which_pass && IS_PROC_DUMMYG(arg)) ||
                                (ALLOCDESCG(arg) && RESULTG(arg)));
        newdsc = sym_get_arg_sec(arg);
        if (!ALLOCDESCG(arg) && RESULTG(arg)) { 
          /* Make sure the result has the updated descriptor in its SDSC
           * field. It's needed when setting up arguments for the function
           * callee. Also the ADDRESS field overloads NEWDSC which gets reset in
           * lower_visit_symbol() of lowersym.c for function results.
           */
          SDSCP(arg, newdsc);
        }
        set_preserve_descriptor(0);
        NEWDSCP(arg, newdsc);
      }
    }
    if (XBIT(54, 0x40) && CONTIGATTRG(arg)
        && STYPEG(newdsc) != ST_UNKNOWN
       ) { 
      /* Generate contiguity check on this argument. 
       * 
       * NOTE: For LLVM targets, this function gets called by
       * newargs_for_llvmiface() to set up placeholder descriptor
       * arguments in the interface. We do not want to 
       * generate contiguity checks in this case since an interface
       * block is non-executable code. The sym_get_arg_sec() function
       * above returns a newdsc without any STYPE when we're processing
       * an interface. Therefore, we check whether STYPEG(newdsc) != ST_UNKNOWN.
       */
      int ast = mk_id(arg);
      gen_contig_check(ast, ast, newdsc, FUNCLINEG(gbl.currsub), false,
                       EntryStd);
    }
    SCP(newdsc, SC_DUMMY);
    OPTARGP(newdsc, OPTARGG(arg));
    NEWARGP(newdsc, 0);
    NEWDSCP(newdsc, 0);
    NEED(aux.dpdsc_avl + 1, aux.dpdsc_base, int, aux.dpdsc_size,
         aux.dpdsc_size + 100);
    aux.dpdsc_base[aux.dpdsc_avl++] = newdsc;
    dscptr++;
  }
  INTERFACEP(this_entry, 1);
  return new_dscptr;
}

#ifdef FLANG_DPM_OUT_UNUSED
/* This routine generate IFTHEN to test static descriptor
 * initilaized. "if(a$sd(1) .eq. 0)
 */
static int
exist_test(int sdsc, int memberast)
{
  int subs[1];
  int astnew;
  int ifexpr;

  subs[0] = mk_cval(1, astb.bnd.dtype);
  astnew =
      mk_subscr(check_member(memberast, mk_id(sdsc)), subs, 1, astb.bnd.dtype);
  ifexpr = mk_binop(OP_EQ, astnew, astb.bnd.zero, DT_LOG);
  astnew = mk_stmt(A_IFTHEN, 0);
  A_IFEXPRP(astnew, ifexpr);
  return astnew;
}
#endif

static int *orderargs; /* List of arguments in dependence order. */

/*
 * Algorithm:
 * subroutine foo(a)
 * integer a(100)
 *  will be transformed.
 *
 *   subroutine foo(array_b, array_b_sec)  !  called actual in the code
 *   integer a(:)  ! called arg in the code
 *   integer array_b(1)
 *   integer array_b_sec(1)
 *   pointer (a_p, a)
 *   a_sect_p = pghpf_newsect(...
 *   ...
 *   a_p = pghpf_copy_in(array_b, a_sec, array_b_sec)
 *   redimension(a(..))
 *   ...
 *   pghpf_copy_out(array_b, a, array_b_sec, array_a_sec)
 *
 * non-sequential common blocks will be transformed as well.
 *
 * add_adjarr_bounds calls add_bound_assignments, which keeps track of
 * which temps have already been assigned by putting them on the A_VISIT list.
 */

static void
interface_for_entry(int this_entry, int new_dscptr)
{
  int arg, narg;
  int i;
  int argnum, dscptr;

  narg = PARAMCTG(this_entry);
  dscptr = DPDSCG(this_entry);
  this_entry_fval = FVALG(this_entry);

  if (narg) {
    NEW(orderargs, int, narg);
    reorder_args(this_entry);
  }
  /* the scope of this ast_visit/ast_unvisit continues past the return
   * from interface_for_entry to the caller, transform_wrapup */
  ast_visit(1, 1);

  for (i = 0; i < narg; i++) {
    argnum = orderargs[i];
    arg = aux.dpdsc_base[dscptr + argnum];
    if (STYPEG(arg) != ST_ARRAY && STYPEG(arg) != ST_VAR)
      continue;
    if (!f77_local && DTY(DTYPEG(arg)) == TY_ARRAY)
      add_adjarr_bounds(arg);
    if (ADJLENG(arg)) {
      add_auto_len(arg, EntryStd);
    }
    if (normalize_bounds(arg)) {
      if (needs_redim(arg))
        emit_redim(arg);
    } else
      emit_kopy_in(arg, this_entry, 0);
  }

  if (narg) {
    FREE(orderargs);
    PARAMCTP(this_entry, 2 * narg);
    DPDSCP(this_entry, new_dscptr);
  }
}

/* Return TRUE if the bounds of sptrA are dependent on sptrB, or the bounds
 * of sptrB. */
static LOGICAL
bounds_depends(int sptrA, int sptrB)
{
  int astB, astBnd;
  ADSC *adA, *adB;
  int ndimsA, dimA, ndimsB, dimB;
  int sptrSD, fval;

  if (DTY(DTYPEG(sptrA)) != TY_ARRAY)
    return FALSE;
  if (!ADJARRG(sptrA))
    return FALSE;
  astB = mk_id(sptrB);
  ndimsB = 0;
  if (DTY(DTYPEG(sptrB)) == TY_ARRAY) {
    adB = AD_DPTR(DTYPEG(sptrB));
    ndimsB = AD_NUMDIM(adB);
  }
  adA = AD_DPTR(DTYPEG(sptrA));
  ndimsA = AD_NUMDIM(adA);
  for (dimA = 0; dimA < ndimsA; dimA++) {
    if (contains_ast(AD_LWBD(adA, dimA), astB) ||
        contains_ast(AD_UPBD(adA, dimA), astB))
      return TRUE;
    for (dimB = 0; dimB < ndimsB; dimB++) {
      astBnd = AD_LWAST(adB, dimB);
      if (A_TYPEG(astBnd) == A_ID) {
        if (astBnd != AD_LWBD(adA, dimA) &&
            contains_ast(AD_LWBD(adA, dimA), astBnd))
          return TRUE;
        if (astBnd != AD_UPBD(adA, dimA) &&
            contains_ast(AD_UPBD(adA, dimA), astBnd))
          return TRUE;
      }
      astBnd = AD_UPAST(adB, dimB);
      if (A_TYPEG(astBnd) == A_ID) {
        if (astBnd != AD_LWBD(adA, dimA) &&
            contains_ast(AD_LWBD(adA, dimA), astBnd))
          return TRUE;
        if (astBnd != AD_UPBD(adA, dimA) &&
            contains_ast(AD_UPBD(adA, dimA), astBnd))
          return TRUE;
      }
    }
  }
  switch (STYPEG(sptrB)) {
  case ST_ARRAY:
    sptrSD = SDSCG(sptrB);
    if (sptrSD && bounds_depends(sptrA, sptrSD))
      return TRUE;
    break;
  default:;
  }
  switch (STYPEG(sptrA)) {
  case ST_ARRAY:
    return FALSE;
  case ST_PROC:
    fval = FVALG(sptrA);
    return fval && bounds_depends(fval, sptrB);
  default:
    return FALSE;
  }
}

/* return TRUE if the bounds or alignment of isptr depends on jsptr */
static LOGICAL
arg_depends(int isptr, int jsptr)
{
  int aln, dst;
  if (bounds_depends(isptr, jsptr)) {
    return TRUE;
  }
  aln = 0;
  dst = 0;
  return FALSE;
} /* arg_depends */

/* Reorder arguments to procedure sptrEntry, adding to the list beginning
 * at *orderargs. Argument A will come before argument B if B's bounds
 * are dependent on A's bounds, B is aligned with A, or A came originally
 * before B in the argument list. */
static void
reorder_args(int sptrEntry)
{
  int nargs, iargnum, jargnum, iarg, jarg, isptr, jsptr;
  int dscptr;
  int cycle;

  nargs = PARAMCTG(sptrEntry);
  dscptr = DPDSCG(sptrEntry);

  /* we are sorting a partial order;
   * the arguments may come in the order ( a b c d e f )
   * with dependences: (where c<a means c must precede a)
   *  c<a<e
   * Even a bubble sort can fail */
  for (iargnum = 0; iargnum < nargs; ++iargnum) {
    orderargs[iargnum] = iargnum;
  }
  cycle = 0;
  for (iargnum = 0; iargnum < nargs;) {
    /* we know 'iargnum' can follow all previous arguments. */
    /* see if it must follow any subsequent argument */
    iarg = orderargs[iargnum];
    isptr = aux.dpdsc_base[dscptr + iarg];
    for (jargnum = iargnum + 1; jargnum < nargs; ++jargnum) {
      jarg = orderargs[jargnum];
      jsptr = aux.dpdsc_base[dscptr + jarg];
      if (!jsptr)
        continue;
      if (arg_depends(isptr, jsptr)) {
        /* isptr must follow jsptr */
        /* swap iarg and jarg, restart for jarg */
        orderargs[iargnum] = jarg;
        orderargs[jargnum] = iarg;
        break;
      }
    }
    if (jargnum == nargs) {
      /* isptr is ok where it is */
      ++iargnum;
      cycle = 0; /* isptr is not involved in a cycle */
    } else {
      ++cycle; /* see if we have a dependence cycle */
      if (cycle > nargs) {
        /* quit */
        error(498, 3, gbl.lineno, SYMNAME(jsptr), CNULL);
        ++iargnum;
      }
    }
  }
} /* reorder_args */

/* Produce a REDIMENSION statement for the array whose symbol table pointer
 * is arg within the procedure given by this_entry. */
static void
emit_redim(int arg)
{
  int p_sptr;
  int newarg;
  int present;
  int astnew;

  newarg = NEWARGG(arg);
  assert(newarg, "emit_redim: no newarg", arg, 4);

  p_sptr = MIDNUMG(arg);
  if (!p_sptr)
    p_sptr = get_array_pointer(arg);

  if (OPTARGG(arg)) {
    present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(newarg));
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    add_stmt_before(astnew, EntryStd);
    gbitable.unconditional = 0;
  }

  if (!POINTERG(arg)) {
    redimension(arg, 0);

    /* Assign the location of the parameter to the array's pointer. */
    astnew = ast_intr(I_LOC, DT_INT, 1, mk_id(newarg));
    astnew = mk_assn_stmt(mk_id(p_sptr), astnew, DT_ADDR);
    add_stmt_before(astnew, EntryStd);
  }

  if (OPTARGG(arg)) {
    astnew = mk_stmt(A_ENDIF, 0);
    add_stmt_before(astnew, EntryStd);
    gbitable.unconditional = 1;
  }
}

/** \brief Return TRUE if the array with symbol table pointer arg requires a
    redimension statement. */
LOGICAL
needs_redim(int arg)
{
  int newarg;
  int dtyp;

  if (arg != 0)
    newarg = NEWARGG(arg);
  else
    newarg = 0;
  if (newarg == 0)
    return FALSE;
  if (SCG(arg) != SC_DUMMY)
    return FALSE;
  if (!is_array_type(arg))
    return FALSE;
  if (is_bad_dtype(DTYPEG(arg)))
    return FALSE;
  if (f77_local)
    return FALSE;
  if (!DESCRG(arg) || !SECDSCG(DESCRG(arg)))
    return FALSE;
  if (NO_PTR || (NO_CHARPTR && DTYG(DTYPEG(arg)) == TY_CHAR) ||
      (NO_DERIVEDPTR && DTYG(DTYPEG(arg)) == TY_DERIVED))
    /* ...Cray pointers not allowed. */
    return FALSE;
  dtyp = DDTG(DTYPEG(arg));
  if (dtyp == DT_ASSNCHAR || dtyp == DT_ASSCHAR || dtyp == DT_DEFERCHAR ||
      dtyp == DT_DEFERCHAR)
    /* ...can't redimension assumed length character arrays. */
    return FALSE;
  if (ASSUMSHPG(arg))
    return TRUE;
  if (SEQG(arg))
    return FALSE;
  return TRUE;
}

/*
 *  pghpf_kopy_in
 *  (char **dbase, section **dsect, char *abase, section *asect,
 *    __INT4_T *rank, __INT4_T *kind, __INT4_T *size, __INT4_T *flags, ...)
 * ... = [  [  proc *dist_target,  ]
 *	     __INT4_T *isstar,
 *	     {	__INT4_T *blocks,  }*  ]
 *	  [  section *align_target, __INT4_T *conform,
 *	     [	__INT4_T *collapse,
 *		{  __INT4_T *axis, __INT4_T *stride, __INT4_T *offset,	}*
 *		__INT4_T *single,
 *		{  __INT4_T *coordinate,  }*  ]  ]
 *	  {  __INT4_T *lb,
 *	     [	__INT4_T *ub,  ]
 *	     [	__INT4_T *no,  __INT4_T *po,  ]  }*
 */
static void
emit_kopy_in(int arg, int this_entry, int actual)
{

  int dscptr;
  int narg;
  int dummy_sec;
  int nargs, argt;
  int astnew, ast;
  int p_sptr, o_sptr;
  int asn;
  int newarg, newdsc;
  int arrdsc;
  int alnd, secd;
  int proc, proc_descr;
  int collapse;
  int ndim;
  int flag;
  int present;
  int pointerAst;
  int offsetAst;
  int baseAst, srcAst;
  int dtype;
  int is_kopy_out_needed;

  if (F90POINTERG(arg))
    return;
  if (POINTERG(arg) && XBIT(57, 0x80000))
    return;
  /* scalar pointer handling */
  if (POINTERG(arg) && DTY(DTYPEG(arg)) != TY_ARRAY) {
    emit_scalar_kopy_in(arg, this_entry);
    return;
  }

  narg = PARAMCTG(this_entry);
  dscptr = DPDSCG(this_entry);
  if (arg != 0) {
    newarg = NEWARGG(arg);
    newdsc = NEWDSCG(arg);
  } else
    newarg = newdsc = 0;
  if (newarg == 0)
    return;
  if (newdsc) {
    newdsc = check_member(actual, mk_id(newdsc));
  }
  if (!is_array_type(arg))
    return;
  if (is_bad_dtype(DTYPEG(arg)))
    return;

  if (!f77_local) {
    if (!DESCUSEDG(arg))
      return;
    if (!is_kopy_in_needed(arg))
      return;
    if (XBIT(57, 0x10000) && SCG(arg) == SC_DUMMY && !POINTERG(arg))
      return;
  }

  assert(DTY(DTYPEG(arg)) == TY_ARRAY && !is_bad_dtype(DTYPEG(arg)),
         "interface_for_entry: bad arg type", arg, 4);
  DESCUSEDP(arg, 1);
  dummy_sec = INS_DESCR(SECDG((DESCRG(arg))));
  arrdsc = DESCRG(arg);
  alnd = ALNDG(arrdsc);
  assert(alnd, "emit_kopy_in: TEMPLATE does not exist", arg, 3);
  secd = SECDG(arrdsc);
  assert(secd, "emit_kopy_in: not array", arg, 3);
  if (VISITG(TMPL_DESCR(alnd)))
    return;
  if (VISIT2G(INS_DESCR(secd)))
    return;
  VISITP(TMPL_DESCR(alnd), 1);
  VISIT2P(INS_DESCR(secd), 1);
  change_mk_id(DESCRG(arg), INS_DESCR(secd));
  is_kopy_out_needed = !SEQG(arg) || POINTERG(arg) || f77_local;

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    /* just to declare it */

    /*
     * Changed:
     *	present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present),DT_LOG);
     *	present = ast_intr(I_PRESENT, DT_LOG, 1, mk_id(newarg));
     */

    present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(newarg));
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    add_stmt_before(astnew, EntryStd);
    gbitable.unconditional = 0;
  }

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local && is_kopy_out_needed) {
    astnew = mk_stmt(A_ENDIF, 0);
    add_stmt_after(astnew, ExitStd);
  }

  /* make sure processor descriptor created */
  proc = TMPL_DIST_TARGET(alnd);
  proc_descr = TMPL_DIST_TARGET_DESCR(alnd);
  if (proc_descr) {
    trans_mkproc(proc);
  }

  /* pghpf_kopy_in creates only one descriptor */
  TMPL_DESCR(ALNDG((DESCRG(arg)))) = dummy_sec;

  /* make sure that, align_target is set before alignee */
  /* get target descriptor */
  if (TMPL_ALIGN_TARGET(alnd)) {
    emit_target_alnd(alnd, 0, TRUE);
    assert(TMPL_TARGET_DESCR(alnd), "emit_kopy_in: no descriptor", arg, 3);
  }

  ndim = TMPL_RANK(alnd);
  nargs = 8 + 2 + 9 * ndim + 9 + 7;
  argt = mk_argt(nargs);
  dtype = DTY(INS_DTYPE(secd) + 1);

  /* pointer to arg  */
  p_sptr = 0;
  if ((normalize_bounds(arg) || SEQG(arg)) && !f77_local)
    if (POINTERG(arg))
      p_sptr = get_array_pointer(arg);
    else
      p_sptr = 0;
  else {
    if (f77_local && MIDNUMG(arg))
      p_sptr = MIDNUMG(arg);
    else
      p_sptr = get_array_pointer(arg);
  }

  o_sptr = PTROFFG(arg);

  if (NO_PTR || (NO_CHARPTR && DTYG(DTYPEG(arg)) == TY_CHAR) ||
      (NO_DERIVEDPTR && DTYG(DTYPEG(arg)) == TY_DERIVED)) {
    if (!o_sptr)
      o_sptr = p_sptr;

    if (o_sptr) {
      int dest;
      asn = mk_stmt(A_ASN, 0);
      dest = mk_id(o_sptr);
      A_DESTP(asn, dest);
      A_SRCP(asn, astb.i1);
      add_stmt_before(asn, EntryStd);
    }
  }

  if (p_sptr)
    pointerAst = mk_id(p_sptr);
  else
    pointerAst = astb.ptr0;

  if (o_sptr)
    offsetAst = mk_id(o_sptr);
  else
    offsetAst = astb.ptr0;

  if (pointerAst == offsetAst)
    pointerAst = astb.ptr0;

  if (offsetAst == astb.ptr0 && pointerAst == astb.ptr0)
    baseAst = mk_id(arg);
  else if (offsetAst == astb.ptr0)
    baseAst = astb.ptr0;
  else
    baseAst = mk_id(arg);

  srcAst = check_member(actual, mk_id(newarg));

  flag = TMPL_FLAG(alnd);
  flag |= __NO_OVERLAPS;
  TMPL_FLAG(alnd) = flag;

  ARGT_ARG(argt, 0) = pointerAst;
  ARGT_ARG(argt, 1) = offsetAst;
  ARGT_ARG(argt, 2) = baseAst;
  ARGT_ARG(argt, 3) = mk_id(dummy_sec);
  ARGT_ARG(argt, 4) = srcAst;
  ARGT_ARG(argt, 5) = newdsc;
  ARGT_ARG(argt, 6) = mk_cval(TMPL_RANK(alnd), DT_INT);
  ARGT_ARG(argt, 7) = mk_cval(dtype_to_arg(dtype), DT_INT);
  ARGT_ARG(argt, 8) = size_of_dtype(dtype, arg, 0);
  ARGT_ARG(argt, 9) = mk_isz_cval(flag, astb.bnd.dtype);

  nargs = fill_argt_with_alnd(arg, 0, argt, alnd, 10, 0, 0);

  if (TMPL_TYPE(alnd) != REPLICATED && !is_set(flag, __NO_OVERLAPS)) {
    collapse = TMPL_COLLAPSE(alnd) | TMPL_ISSTAR(alnd);
  }

  if (POINTERG(arg))
    astnew = gen_ptr_in(arg, this_entry);
  else {
    astnew = mk_func_node(A_ICALL, mk_id(intast_sym[I_COPYIN]), nargs, argt);
    A_OPTYPEP(astnew, I_COPYIN);
  }

  add_stmt_before(astnew, EntryStd);

  if (p_sptr) {
    int dty;
    if (!f77_local) {
      astnew = 0;
      if (!POINTERG(arg)) {
        redimension(arg, 0);
      }
    }
    dty = DTYG(DTYPEG(arg));
    /* if it is optional dummy and no pointer*/
    if (OPTARGG(arg) && !f77_local &&
        (NO_PTR || (NO_CHARPTR && dty == TY_CHAR) ||
         (NO_DERIVEDPTR && dty == TY_DERIVED))) {
      astnew = mk_stmt(A_ELSE, 0);
      add_stmt_before(astnew, EntryStd);

      if (XBIT(57, 0x80) || dty == TY_CHAR || dty == TY_DERIVED) {
        int dest;
        /* can't use A_HOFFSET with character/derived types */
        /* don't want to when using $bs array as the base since
         * the $bs array is not present */
        ast = mk_stmt(A_ASN, 0);
        if (PTROFFG(arg)) {
          dest = mk_id(PTROFFG(arg));
        } else {
          dest = mk_id(p_sptr);
        }
        A_DESTP(ast, dest);
        A_SRCP(ast, astb.i0);
      } else {
        int dest, lop;
        ast = mk_stmt(A_HOFFSET, 0);
        lop = mk_id(arg);
        A_LOPP(ast, lop);
        A_ROPP(ast, astb.ptr0);

        if (PTROFFG(arg))
          dest = mk_id(PTROFFG(arg));
        else
          dest = mk_id(p_sptr);
        A_DESTP(ast, dest);
      }
      add_stmt_before(ast, EntryStd);
    } else if (OPTARGG(arg) && !f77_local) {
      int dest, src;
      astnew = mk_stmt(A_ELSE, 0);
      add_stmt_before(astnew, EntryStd);

      ast = mk_stmt(A_ASN, DT_INT);
      dest = mk_id(p_sptr);
      A_DESTP(ast, dest);
      if (DTY(DDTG(DTYPEG(arg))) == TY_CHAR) {
        src = gen_RTE_loc(astb.ptr0c);
      } else {
        src = gen_RTE_loc(astb.ptr0);
      }
      A_SRCP(ast, src);
      add_stmt_before(ast, EntryStd);
    }
  }

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    astnew = mk_stmt(A_ENDIF, 0);
    add_stmt_before(astnew, EntryStd);
  }

  if (!is_kopy_out_needed)
    return;
  if (POINTERG(arg))
    astnew = gen_ptr_out(arg, this_entry);
  else
    astnew = gen_copy_out(srcAst, arg, newdsc, dummy_sec);

  add_stmt_after(astnew, ExitStd);

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(newarg));
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    add_stmt_after(astnew, ExitStd);
  }
  gbitable.unconditional = 1;
} /* emit_kopy_in */

#ifdef FLANG_DPM_OUT_UNUSED
/* Return TRUE if the arry given by sptr is adjustable according to
 * Fortran-77. */
static LOGICAL
is_f77_adjustable(int sptr)
{
  ADSC *ad;
  int ndims, dim;
  int ast;
  int sptrBnd;

  if (STYPEG(sptr) != ST_ARRAY)
    return FALSE;
  ad = AD_DPTR(DTYPEG(sptr));
  if (!AD_ADJARR(ad))
    return FALSE;
  ndims = AD_NUMDIM(ad);
  for (dim = 0; dim < ndims; dim++) {
    ast = AD_LWBD(ad, dim);
    if (ast && !A_ALIASG(ast)) {
      if (A_TYPEG(ast) != A_ID)
        return FALSE;
      sptrBnd = A_SPTRG(ast);
      if (SCG(sptrBnd) != SC_DUMMY && SCG(sptrBnd) != SC_CMBLK)
        return FALSE;
    }
    ast = AD_UPBD(ad, dim);
    if (ast && !A_ALIASG(ast)) {
      if (A_TYPEG(ast) != A_ID)
        return FALSE;
      sptrBnd = A_SPTRG(ast);
      if (SCG(sptrBnd) != SC_DUMMY && SCG(sptrBnd) != SC_CMBLK)
        return FALSE;
    }
  }
  return TRUE;
}
#endif

/* pghpf_copy_out_(void *db, void *sb, section *ds,
 *			  section *ss, int intent);
 */
static int
gen_copy_out(int newarg, int arg, int newdsc, int dummy_sec)
{
  int nargs;
  int argt;
  int astnew, a;

  nargs = 5;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = newarg;
  ARGT_ARG(argt, 1) = mk_id(arg);
  ARGT_ARG(argt, 2) = newdsc;
  ARGT_ARG(argt, 3) = mk_id(dummy_sec);

/* If we have an F77_LOCAL intrinsic, we pass in a special flag into copy_out
 * to process the argument accordingly */
#ifndef PFGLANG
/* (Fixes NEC problem #95/tpr 2238) */
#endif

  ARGT_ARG(argt, 4) = mk_cval(INTENTG(arg), DT_INT);

  astnew = mk_stmt(A_ICALL, 0);
  a = mk_id(intast_sym[I_COPYOUT]);
  A_LOPP(astnew, a);
  A_ARGCNTP(astnew, nargs);
  A_ARGSP(astnew, argt);
  A_OPTYPEP(astnew, I_COPYOUT);
  return astnew;
}

/*  void
 *  ENTHPF(PTR_IN,ptr_in)
 *   (__INT_T *rank,		 ! dummy rank (0 == scalar)
 *    __INT_T *kind,		 ! dummy type code
 *    __INT_T *len,		 ! dummy element byte length
 *    char *db, 		 ! dummy array base address
 *    desc *dd, 		 ! dummy static descriptor
 *    char *ab, 		 ! actual array base address
 *    desc *ad);		 ! actual static descriptor
 */

static int
gen_ptr_in(int arg, int this_entry)
{

  int dscptr;
  int newarg, newdsc;
  int ndim;
  int base, static_desc;
  int actual_base, actual_desc;
  int dtype, dty;
  int sptr1;
  int nargs, newargt;
  int newast;
  int narg;
  int kind, len;

  if (arg == 0)
    return 0;
  assert(POINTERG(arg), "gen_ptr_in: must be pointer", arg, 4);

  narg = PARAMCTG(this_entry);
  dscptr = DPDSCG(this_entry);
  newarg = NEWARGG(arg);
  newdsc = NEWDSCG(arg);
  if (newarg == 0)
    return 0;
  if (is_bad_dtype(DTYPEG(arg)))
    return 0;

  ndim = rank_of_sym(arg);
  dtype = DTYPEG(arg);
  dty = DDTG(dtype);
  kind = dtype_to_arg(dty);
  /* check for assumed-length */
  if (dty == DT_ASSCHAR || dty == DT_ASSNCHAR || ADJLENG(arg)) {
    /* get 'cvlen' variable.
     * initialize 'cvlen' variable.
     */
    int cvlen, rhs, asn;
    cvlen = CVLENG(arg);
    if (cvlen == 0) {
      cvlen = sym_get_scalar(SYMNAME(arg), "len", astb.bnd.dtype);
      CVLENP(arg, cvlen);
      ADJLENP(arg, 1);
      if (SCG(arg) == SC_DUMMY)
        CCSYMP(cvlen, 1);
    }
    len = mk_id(cvlen);
    rhs = size_ast_of(mk_id(newarg), dty);
    if (ADJLENG(arg)) {
      rhs = mk_convert(rhs, DTYPEG(cvlen));
      rhs = ast_intr(I_MAX, DTYPEG(cvlen), 2, rhs, mk_cval(0, DTYPEG(cvlen)));
    }
    asn = mk_assn_stmt(len, rhs, DTYPEG(cvlen));
    add_stmt_before(asn, EntryStd);
  } else {
    len = size_ast_of(mk_id(arg), dty);
  }

  base = arg;
  static_desc = SDSCG(arg);
  actual_base = newarg;
  actual_desc = newdsc;

  assert(base, "gen_ptr_in: must be non-zero", base, 4);
  assert(static_desc, "gen_ptr_in: must be non-zero", static_desc, 4);
  assert(actual_base, "gen_ptr_in: must be non-zero", actual_base, 4);
  assert(actual_desc, "gen_ptr_in: must be non-zero", actual_desc, 4);

  nargs = 7;
  newargt = mk_argt(nargs);
  ARGT_ARG(newargt, 0) = mk_isz_cval(ndim, astb.bnd.dtype);
  ARGT_ARG(newargt, 1) = mk_isz_cval(kind, astb.bnd.dtype);
  ARGT_ARG(newargt, 2) = len;
  ARGT_ARG(newargt, 3) = mk_id(base);
  ARGT_ARG(newargt, 4) = mk_id(static_desc);
  ARGT_ARG(newargt, 5) = mk_id(actual_base);
  ARGT_ARG(newargt, 6) = mk_id(actual_desc);

  sptr1 = intast_sym[I_PTR_COPYIN];
  newast = mk_func_node(A_ICALL, mk_id(sptr1), nargs, newargt);
  A_OPTYPEP(newast, I_PTR_COPYIN);
  return newast;
}

/*  void
 *  ENTHPF(PTR_OUT,ptr_out)
 *   (char *db, 		 ! dummy array base address
 *    desc *dd, 		 ! dummy static descriptor
 *    char *ab, 		 ! actual array base address
 *    desc *ad);		 ! actual static descriptor
 */

static int
gen_ptr_out(int arg, int this_entry)
{

  int dscptr;
  int newarg, newdsc;
  int base, static_desc;
  int actual_base, actual_desc;
  int sptr1;
  int nargs, newargt;
  int newast;
  int narg;

  assert(POINTERG(arg), "gen_ptr_out: must be pointer", arg, 4);

  narg = PARAMCTG(this_entry);
  dscptr = DPDSCG(this_entry);
  if (arg != 0) {
    newarg = NEWARGG(arg);
    newdsc = NEWDSCG(arg);
  } else
    newarg = newdsc = 0;
  if (newarg == 0)
    return 0;
  if (is_bad_dtype(DTYPEG(arg)))
    return 0;

  base = arg;
  static_desc = SDSCG(arg);
  actual_base = newarg;
  actual_desc = newdsc;

  assert(base, "gen_ptr_out: must be non-zero", base, 4);
  assert(static_desc, "gen_ptr_out: must be non-zero", static_desc, 4);
  assert(actual_base, "gen_ptr_out: must be non-zero", actual_base, 4);
  assert(actual_desc, "gen_ptr_out: must be non-zero", actual_desc, 4);

  nargs = 4;
  newargt = mk_argt(nargs);
  ARGT_ARG(newargt, 0) = mk_id(actual_base);
  ARGT_ARG(newargt, 1) = mk_id(actual_desc);
  ARGT_ARG(newargt, 2) = mk_id(base);
  ARGT_ARG(newargt, 3) = mk_id(static_desc);

  sptr1 = intast_sym[I_PTR_COPYOUT];
  newast = mk_func_node(A_ICALL, mk_id(sptr1), nargs, newargt);
  A_OPTYPEP(newast, I_PTR_COPYOUT);
  return newast;
}

static void
emit_scalar_kopy_in(int arg, int this_entry)
{
  int astnew;
  int present;
  int p_sptr;
  int is_kopy_out_needed = !SEQG(arg) || POINTERG(arg) || f77_local;

  assert(POINTERG(arg), "emit_scalar_kopy_in: must be pointer", arg, 4);
  assert(STYPEG(arg) != ST_ARRAY, "emit_scalar_kopy_in: must be scalar", arg,
         4);

  if (F90POINTERG(arg))
    return;

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(arg));
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    add_stmt_before(astnew, EntryStd);
    gbitable.unconditional = 0;
  }

  astnew = gen_ptr_in(arg, this_entry);
  add_stmt_before(astnew, EntryStd);

  if (OPTARGG(arg) && !f77_local) {
    int dest, src;
    astnew = mk_stmt(A_ELSE, 0);
    add_stmt_before(astnew, EntryStd);

    astnew = mk_stmt(A_ASN, DT_INT);
    p_sptr = MIDNUMG(arg);
    dest = mk_id(p_sptr);
    A_DESTP(astnew, dest);
    if (DTY(DDTG(DTYPEG(arg))) == TY_CHAR) {
      src = gen_RTE_loc(astb.ptr0c);
    } else {
      src = gen_RTE_loc(astb.ptr0);
    }
    A_SRCP(astnew, src);
    add_stmt_before(astnew, EntryStd);
    src = gen_RTE_loc(astb.ptr0);

    astnew = mk_stmt(A_ENDIF, 0);
    add_stmt_before(astnew, EntryStd);
  }

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local && is_kopy_out_needed) {
    astnew = mk_stmt(A_ENDIF, 0);
    add_stmt_after(astnew, ExitStd);
  }

  astnew = gen_ptr_out(arg, this_entry);
  add_stmt_after(astnew, ExitStd);

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(arg));
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    add_stmt_after(astnew, ExitStd);
  }
}

/* This will open an if statement for each entry to free descriptors
 * which are defined for that entry.
 */
static void
open_entry_guard(int entry)
{
  int ast;
  int astnew;

  /* guard this with a check for this entry */
  if (gbl.ent_select != 0) {
    ast = mk_binop(OP_EQ, mk_id(gbl.ent_select),
                   mk_cval(ENTNUMG(entry), DT_INT), DT_LOG);
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, ast);
    add_stmt_after(astnew, ExitStd);
  }
}

static void
close_entry_guard(void)
{
  int astnew;

  if (gbl.ent_select != 0) {
    astnew = mk_stmt(A_ENDIF, 0);
    add_stmt_after(astnew, ExitStd);
  }
}

LOGICAL
getbit(int x, int p)
{
  return ((x >> p) & ~(~0u << 1));
}

static int
getbits(int x, int p, int n)
{
  return ((x >> (p)) & ~(~0 << n));
}

/*
 * subtract difference between actual and dummy lower bounds
 *  ast - locallower + globallower
 * special cases if ast == locallower or locallower == globallower
 */
static int
diff_lbnd(int dtyp, int dim, int ast, int descr)
{
  int ast1;
  int astBnd = ADD_LWAST(dtyp, dim);
  int astglb = get_global_lower(descr, dim);

  if (astBnd == astglb)
    return ast;

  if (astBnd == ast)
    return astglb;
  ast1 = ast;
  if (astBnd)
    astglb = mk_binop(OP_SUB, astglb, astBnd, astb.bnd.dtype);
  ast1 = mk_binop(OP_ADD, ast1, astglb, astb.bnd.dtype);
  return ast1;
} /* diff_lbnd */

/** \brief Create section descriptor and initialize prior to call.

    It initialize before std. And section information comes from AST arr.
    \p ast is A_SUBSCR with non-null shape.
    To find section boundary, it looks for AST. If AST is not enough
    (e.g, `A(:,1)`), it uses the shape.  Check for `A(:)%%B(5)` also.
 */
int
make_sec_from_ast(int ast, int std, int stdafter, int sec_ast, int sectflag)
{
  int sec;

  sec = make_sec_from_ast_chk(ast, std, stdafter, sec_ast, sectflag, 0);
  return sec;
}

int
make_sec_from_ast_chk(int ast, int std, int stdafter, int sec_ast, int sectflag,
                      int ignore_c)
{
  int a, arr, sec;
  int descr;
  int nargs, argt, astnew;
  int ndim, numdim;
  int i, j, dim_mask;
  int glb, gub, gst;
  int asd;
  int triple;
  int sptr, fsptr;
  int shape;
  int subs[7];
  LOGICAL rhs_is_dist;
  int bogus;
  int strd1_cnt;

  /* find the array that has the section */
  arr = 0;
  for (a = ast; a != 0;) {
    switch (A_TYPEG(a)) {
    case A_SUBSCR:
      /* it must have a shape; its parent
       * must be an A_ID, or an A_MEM whose parent has no shape */
      if (!A_SHAPEG(a)) {
        arr = a; /* probably an error */
        a = 0;
      } else {
        int lop;
        lop = A_LOPG(a);
        if (A_TYPEG(lop) == A_ID) {
          arr = a;
          a = 0;
        } else if (A_TYPEG(lop) == A_MEM) {
          int parent;
          parent = A_PARENTG(lop);
          if (A_SHAPEG(parent)) {
            a = parent;
          } else {
            arr = a;
            a = 0;
          }
        } else {
          interr("make_sec_from_ast: invalid A_SUBSCR parent", A_TYPEG(lop), 4);
        }
      }
      break;
    case A_MEM:
      a = A_PARENTG(a);
      break;
    case A_ID:
    default:
      a = 0;
      break;
    }
  }
  if (arr == 0 || A_TYPEG(arr) != A_SUBSCR) {
    interr("make_sec_from_ast: no subscript", ast, 2);
    return 0;
  }

  /* localize section, for example a(idx1(1), idx1(2):idx2(7)) */
  asd = A_ASDG(arr);
  numdim = ASD_NDIM(asd);
  if (!pure_gbl.local_mode) {
    int sptr, dtype;
    for (i = 0; i < numdim; ++i) {
      rhs_is_dist = FALSE;
      subs[i] = get_scalar_in_expr(ASD_SUBS(asd, i), std, TRUE);
      /*insert_comm_before(std, ASD_SUBS(asd, i), &rhs_is_dist, TRUE);*/
    }
    sptr = sptr_of_subscript(arr);
    dtype = DTYPEG(sptr);
    arr = mk_subscr(A_LOPG(arr), subs, numdim, dtype);
  }
  shape = A_SHAPEG(arr);
  assert(shape != 0, "make_sec_from_ast: shape null", 0, 4);
  /* create a section descriptor */
  sptr = sptr_of_subscript(arr);
  bogus = getbit(sectflag, 8);
  if (is_whole_array(arr) && !bogus) {
    DESCUSEDP(sptr, 1);
    return DESCRG(sptr);
  }
  if (sec_ast == 0) {
    sec = sym_get_sdescr(sptr, SHD_NDIM(shape)); /* ZB */
    sec_ast = mk_id(sec);
  } else {
    sec = NOSYM;
  }
  descr = DESCRG(sptr);
  ndim = rank_of(DTYPEG(sptr));
  nargs = 4 + 3 * ndim;
  /* RTE_sect(descr,olddesct,rank,l1,u2,s1,...,lN,uN,sN) */
  argt = mk_argt(nargs);
  nargs = 0;
  ARGT_ARG(argt, nargs++) = sec_ast;
  DESCUSEDP(sptr, 1);
  ARGT_ARG(argt, nargs++) = check_member(A_LOPG(arr), mk_id(descr));
  gst = mk_isz_cval(ndim, astb.bnd.dtype);
  ARGT_ARG(argt, nargs++) = mk_bnd_int(gst);
  asd = A_ASDG(arr);
  numdim = ASD_NDIM(asd);
  assert(numdim == ndim, "make_sec_from_ast: numdim from asd", sptr, 3);
  j = SHD_NDIM(shape) - 1;
  dim_mask = 0;
  strd1_cnt = 0;
  for (i = ndim - 1; i >= 0; --i) {
    dim_mask <<= 1;
    if (A_TYPEG(triple = ASD_SUBS(asd, i)) == A_TRIPLE) {
      assert(j >= 0, "make_sec_from_ast: SHD/ASD mismatch", arr, 4);
      glb = SHD_LWB(shape, j);
      gub = SHD_UPB(shape, j);
      gst = SHD_STRIDE(shape, j);
      --j;
      dim_mask |= 1; /* vector dimension */
    } else {
      glb = ASD_SUBS(asd, i);
      gub = glb;
      gst = mk_isz_cval(1, astb.bnd.dtype);
    }
    glb = mk_bnd_int(glb);
    gub = mk_bnd_int(gub);
    if (ASSUMSHPG(sptr) && XBIT(57, 0x10000)) {
      /* offset by difference between actual/dummy lower bounds */
      glb = diff_lbnd(DTYPEG(sptr), i, glb, descr);
      gub = diff_lbnd(DTYPEG(sptr), i, gub, descr);
    }
    ARGT_ARG(argt, 3 * i + nargs) = glb;
    ARGT_ARG(argt, 3 * i + nargs + 1) = gub;
    ARGT_ARG(argt, 3 * i + nargs + 2) = mk_bnd_int(gst);
    if (gst == astb.bnd.one)
      strd1_cnt++;
  }
  nargs += 3 * ndim;

  ARGT_ARG(argt, nargs++) = mk_isz_cval(sectflag | dim_mask, astb.bnd.dtype);
  if (size_of(DT_PTR) != size_of(DT_INT) && ndim <= 3) {
    /* for the hammer target, it's better to pass the
     * arguments to sect3 by ref ???? WHY ???  */
    switch (ndim) {
    case 3:
      fsptr = sym_mkfunc(mkRteRtnNm(RTE_sect3), DT_NONE);
      break;
    case 2:
      fsptr = sym_mkfunc(mkRteRtnNm(RTE_sect2), DT_NONE);
      break;
    case 1:
      fsptr = sym_mkfunc(mkRteRtnNm(RTE_sect1), DT_NONE);
      break;
    }
  } else if (ndim <= 3) {
    /* experiment with by value arguments */
    /* for the x86 target, it's better to pass the
     * arguments by value - makes sense!!  */
    switch (ndim) {
    case 3:
      fsptr = sym_mkfunc(mkRteRtnNm(RTE_sect3v), DT_NONE);
      break;
    case 2:
      fsptr = sym_mkfunc(mkRteRtnNm(RTE_sect2v), DT_NONE);
      break;
    case 1:
      fsptr = sym_mkfunc(mkRteRtnNm(RTE_sect1v), DT_NONE);
      break;
    }
    for (i = 2; i < nargs; i++) {
      ARGT_ARG(argt, i) = mk_unop(OP_VAL, ARGT_ARG(argt, i), DT_INT);
    }
  } else {
    fsptr = sym_mkfunc(mkRteRtnNm(RTE_sect), DT_NONE);
  }
  astnew = mk_func_node(A_CALL, mk_id(fsptr), nargs, argt);
  NODESCP(A_SPTRG(A_LOPG(astnew)), 1);

  std = add_stmt_before(astnew, std);

  return sec;
}

/** \brief Create a RTE_template call
    for an effective argument `a(1:n,3:m)`

    Create a template that looks like `(1:n,1:m-2)`.
    Return sptr of the template we create.
    \p ast must be an A_SUBSCR that has a section subscript
    RTE_template (template, rank, flags, type, kind, [lb, ub,]... )
 */
int
make_simple_template_from_ast(int ast, int std, LOGICAL need_type_in_descr)
{
  int asd, numdim, sptr;
  int shape, sec, shapedim, descr;
  int nargs, cargs, argt, astnew;
  int i, j;
  int fsptr, dtype;

  /* find the array that has the section */
  assert(ast > 0, "make_simple_template_from_ast: bad ast value", ast, 4);
  assert(A_TYPEG(ast) == A_SUBSCR,
         "make_simple_template_from_ast: expecting subscr", A_TYPEG(ast), 4);

  asd = A_ASDG(ast);
  numdim = ASD_NDIM(asd);
  sptr = sptr_of_subscript(ast);
  if (is_whole_array(ast)) {
    /* if this is the whole array, just use the descriptor we have */
    DESCUSEDP(sptr, 1);
    return DESCRG(sptr);
  }
  shape = A_SHAPEG(ast);
  assert(shape > 0, "make_simple_template_from_ast: null shape", 0, 4);
  /* create a section descriptor */
  shapedim = SHD_NDIM(shape);
  sec = sym_get_sdescr(sptr, shapedim);
  /* RTE_template (template, rank, kind, len, flags, [lb, ub,]... ) */
  if (ASSUMSHPG(sptr) && XBIT(57, 0x10000)) {
    DESCUSEDP(sptr, 1);
    descr = DESCRG(sptr);
  }
  if (flg.smp && SCG(sec) != SC_PRIVATE && !PARREFG(sec)) {
    set_parref_flag2(sec, 0, std);
  }
  nargs = 5 + 2 * shapedim;
  cargs = 0;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, cargs++) = mk_id(sec);
  ARGT_ARG(argt, cargs++) = mk_isz_cval(shapedim, astb.bnd.dtype);
  ARGT_ARG(argt, cargs++) = mk_isz_cval(0, astb.bnd.dtype);
  dtype = DTYPEG(sptr);
  dtype = DTY(dtype + 1);
  ARGT_ARG(argt, cargs++) = mk_isz_cval(dtype_to_arg(dtype), astb.bnd.dtype);
  ARGT_ARG(argt, cargs++) = size_of_dtype(dtype, sptr, ast);
  dtype = DTYPEG(sptr);
  assert(numdim == rank_of(dtype),
         "make_simple_template_from_ast: numdim/ndim mismatch asd", sptr, 4);
  j = 0;
  for (i = 0; i < numdim; ++i) {
    int ss, glb, gub, gst;
    ss = ASD_SUBS(asd, i);
    if (A_TYPEG(ss) == A_TRIPLE) {
      assert(j < shapedim, "make_simple_template_from_ast: SHD/ASD mismatch",
             shapedim, 4);
      glb = SHD_LWB(shape, j);
      gub = SHD_UPB(shape, j);
      gst = mk_bnd_int(SHD_STRIDE(shape, j));
      assert(gst == astb.bnd.one,
             "make_simple_template_from_ast: nonunit stride", gst, 4);
      glb = mk_bnd_int(glb);
      gub = mk_bnd_int(gub);
      if (ASSUMSHPG(sptr) && XBIT(57, 0x10000)) {
        /* offset by difference between actual/dummy lower bounds */
        glb = diff_lbnd(dtype, i, glb, descr);
        gub = diff_lbnd(dtype, i, gub, descr);
      }
      ARGT_ARG(argt, cargs++) = astb.bnd.one;
      ARGT_ARG(argt, cargs++) = mk_binop(OP_ADD, mk_binop(OP_SUB,
                                                 gub, glb, astb.bnd.dtype),
                                         astb.bnd.one, astb.bnd.dtype);
      ++j;
    }
  }

  fsptr = sym_mkfunc(mkRteRtnNm(RTE_template), DT_NONE);
  astnew = mk_func_node(A_CALL, mk_id(fsptr), nargs, argt);
  SDSCINITP(sec, 1);
  NODESCP(fsptr, 1);
  add_stmt_before(astnew, std);

  if (need_type_in_descr)
    set_type_in_descriptor(mk_id(sec), sptr, DT_NONE, ast, std);

  return sec;
}

void
set_assumsz_bound(int arg, int entry)
{
  ADSC *ad;
  int dtype;
  int i;
  int ast1, ast2;
  int std;
  int newarg, newdsc;
  int astnew, present;

#if DEBUG
  assert(STYPEG(arg) == ST_ARRAY, "set_assumed_dim: arg not array", arg, 4);
  assert(ASUMSZG(arg), "set_assumsz_bound: arg not assumed size", arg, 4);
#endif
  dtype = DTYPEG(arg);
  ad = AD_DPTR(dtype);
  std = ENTSTDG(entry);

  newarg = NEWARGG(arg);
  newdsc = NEWDSCG(arg);
  assert(newarg && newdsc, "set_assumsz_bounds: needs newarg", newarg, 3);

  /* if is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(newarg));

    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    std = add_stmt_after(astnew, std);
  }

  /* arg is assumed size, need to set its upper bound */
  i = AD_NUMDIM(ad) - 1;

  ast1 = extent(arg, mk_id(newdsc), i);

  /* ub = pghpf_extent(a, dim) */
  ast2 = mk_stmt(A_ASN, 0);
  A_SRCP(ast2, ast1);
  A_DESTP(ast2, AD_EXTNTAST(ad, i));
  std = add_stmt_after(ast2, std);
  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    astnew = mk_stmt(A_ENDIF, 0);
    std = add_stmt_after(astnew, std);
  }
  ENTSTDP(entry, std);
}

static bool
update_shape_info_expr(int arg, int ast)
{
  int i;
  int aptr, sptr, shd = 0, nd;

  switch (A_TYPEG(ast)) {
  case A_SUBSCR:
    aptr = (int)A_LOPG(ast);
    sptr = A_SPTRG(aptr);
    if (sptr == arg) {
      if ((shd = A_SHAPEG(aptr))) {
        nd = SHD_NDIM(shd);
        for (i = 0; i < nd; ++i)
          SHD_LWB(shd, i) = astb.bnd.one;
        return true;
      }
    }
    return false;
    break;
  case A_UNOP:
  case A_CONV:
  case A_PAREN:
    if (update_shape_info_expr(arg, A_LOPG(ast)))
      return true;
    break;
  case A_BINOP:
    if (update_shape_info_expr(arg, A_LOPG(ast)))
      return true;
    if (update_shape_info_expr(arg, A_ROPG(ast)))
      return true;
    break;
  default:
    break;
  }
  return false;
}

static void
update_shape_info(int arg)
{
  int std, ast, dst, aptr, sptr;
  int i, nd, shd = 0;

  for (std = STD_NEXT(0); std; std = STD_NEXT(std)) {
    ast = STD_AST(std);
    if (A_TYPEG(ast) != A_ASN && !A_ISEXPR(A_TYPEG(ast)))
      continue;
    dst = A_DESTG(ast);
    if (A_TYPEG(dst) != A_SUBSCR)
      continue;
    aptr = (int)A_LOPG(dst);
    sptr = A_SPTRG(aptr);
    if (sptr != arg) {
      if (update_shape_info_expr(arg, A_SRCG(ast)))
        return;
      continue;
    }

    if ((shd = A_SHAPEG(aptr))) {
      nd = SHD_NDIM(shd);
      for (i = 0; i < nd; ++i)
        SHD_LWB(shd, i) = astb.bnd.one;
      return; /* found match and adjustment made */
    }
  }
}

void
set_assumed_bounds(int arg, int entry, int actual)
{
  ADSC *ad;
  int dtype;
  int r;
  int i;
  int ast, ast1, ast2;
  int sav = 0;
  int tmp_lb, tmp_ub;
  int std;
  int newarg, newdsc;
  int astnew, present, zbaseast, prevmpyer;

  assert(is_array_type(arg), "set_assumed_bounds: arg not array", 0, 4);
  dtype = DTYPEG(arg);
  ad = AD_DPTR(dtype);
  assert(AD_DEFER(ad), "set_assumed_bounds: arg not deferred", arg, 4);
  std = ENTSTDG(entry);
  r = AD_NUMDIM(ad);

  newarg = NEWARGG(arg);
  newdsc = NEWDSCG(arg);
  /* OPTIONAL arg may not have newarg */
  if (OPTARGG(arg) && newarg == 0)
    return;
  assert(newarg && newdsc, "set_assumed_bounds: needs newarg", newarg, 3);

  /* if is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    present = sym_mkfunc_nodesc(mkRteRtnNm(RTE_present), stb.user.dt_log);
    present = ast_intr(I_PRESENT, stb.user.dt_log, 1,
                       check_member(actual, mk_id(newarg)));
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, present);
    std = add_stmt_after(astnew, std);
  }

  zbaseast = 0;
  prevmpyer = 0;

  /* did we not set lower bound to 1 in to_assumed_shape() or
   * mk_assumed_shape() because TARGET was not yet available
   * (still in parser) when this xbit was set?
   */
  if (XBIT(58, 0x400000) && !TARGETG(arg)) {
    for (i = 0; i < r; ++i) {
      if (AD_LWBD(ad, i) == AD_LWAST(ad, i)) {
        if (A_TYPEG(AD_LWBD(ad, i)) == A_ID) {
          /* add assignment std to set lb to 1 */
          ast = mk_stmt(A_ASN, 0);
          A_DESTP(ast, AD_LWBD(ad, i));
          A_SRCP(ast, astb.bnd.one);
          std = add_stmt_after(ast, std);
        }
        AD_LWBD(ad, i) = astb.bnd.one;
        AD_LWAST(ad, i) = astb.bnd.one;
      }
    }
    /* also, arg is assumed shape, and since !TARGET mark as stride 1 */
    SDSCS1P(arg, 1); /* see comment below regarding these xbits */
    if( XBIT(55,0x80) )
        update_shape_info(arg);
  }

  for (i = 0; i < r; ++i) {
    tmp_lb = AD_LWAST(ad, i); /* temp for lower bound */
    /* declare it by changing the  scope */
    if (A_TYPEG(tmp_lb) == A_ID) {
      IGNOREP(A_SPTRG(tmp_lb), 0);
    }

    if ((ast1 = AD_LWBD(ad, i)) == 0)
      /* lower bound not specified */
      ast1 = mk_isz_cval(1, astb.bnd.dtype);
    if (A_TYPEG(tmp_lb) == A_CNST) {
      sav = tmp_lb;
    } else if ((XBIT(54, 2) || (XBIT(58, 0x400000) && TARGETG(arg))) &&
        tmp_lb == ast1 && A_TYPEG(tmp_lb) == A_ID) {
      /*
      FIX ME: setting the descriptor bounds to 1 here does not work since
      there can be other references (such as loop bounds) which use the
      symbolic lower bounds for each dimension.
      ast1 = mk_isz_cval(1, astb.bnd.dtype);
      sav = AD_LWAST(ad, i) = AD_LWBD(ad, i) = ast1;
      */

      /* so we just assign the symbolic lower bound ID to 1 */
      ast1 = mk_isz_cval(1, astb.bnd.dtype);
      sav = ast1;
      ast2 = mk_stmt(A_ASN, 0);
      A_DESTP(ast2, tmp_lb);
      A_SRCP(ast2, ast1);
      std = add_stmt_after(ast2, std);
    } else if (tmp_lb != ast1) {
      /* output lower bound assignment */
      /* lb = <declared lower bound> */
      sav = ast1;
      ast2 = mk_stmt(A_ASN, 0);
      A_DESTP(ast2, tmp_lb);
      A_SRCP(ast2, ast1);
      std = add_stmt_after(ast2, std);
    }

    /* no need for upper bounds for pointer dummys */
    if (POINTERG(arg))
      continue;

    /* output upper bound assignment */
    ast2 = extent(arg, check_member(actual, mk_id(newdsc)), i);
    /* ub = lb - 1 + pghpf_extent(a, dim) */
    ast1 = mk_binop(
        OP_ADD,
        mk_binop(OP_SUB, ast1, mk_isz_cval(1, astb.bnd.dtype), astb.bnd.dtype),
        ast2, astb.bnd.dtype);
    ast2 = mk_stmt(A_ASN, 0);
    A_SRCP(ast2, ast1);
    tmp_ub = AD_UPAST(ad, i);
    A_DESTP(ast2, tmp_ub);
    /* declare it by changing the  scope */
    if (A_TYPEG(tmp_ub) == A_ID) {
      IGNOREP(A_SPTRG(tmp_ub), 0);
    }
    std = add_stmt_after(ast2, std);

    if (sav != astb.bnd.one) {
      int a;
      /* generate: if (ub < lb) then ub = lb - 1; */
      present = mk_binop(OP_LT, tmp_ub, tmp_lb, astb.bnd.dtype);
      astnew = mk_stmt(A_IFTHEN, 0);
      A_IFEXPRP(astnew, present);
      std = add_stmt_after(astnew, std);

      ast2 = mk_stmt(A_ASN, 0);
      A_DESTP(ast2, tmp_ub);
      a = mk_binop(OP_SUB, tmp_lb, astb.bnd.one, astb.bnd.dtype);
      A_SRCP(ast2, a);
      std = add_stmt_after(ast2, std);

      astnew = mk_stmt(A_ENDIF, 0);
      std = add_stmt_after(astnew, std);
    }
    {
      int tmp, nexttmp, ast;
      /* update ZBASE ast */
      if (XBIT(57, 0x10000)) {
        int astoff;
        int lb;
        /* account for difference between actual argument lower bound
         * and assumed-shape argument declared lower bound */
        ast = get_global_lower(newdsc, i);
        if (!XBIT(58, 0x40000000)) {
          astoff = get_section_offset(newdsc, i);
          ast = mk_binop(OP_ADD, astoff, ast, astb.bnd.dtype);
        }
        lb = ADD_LWBD(dtype, i);
        if (lb == 0 || A_ALIASG(lb)) {
          ISZ_T lbval;
          /* get the constant value, subtract one, subtract that from ss */
          if (lb == 0) {
            lbval = 1;
          } else {
            lb = mk_bnd_int(A_ALIASG(lb));
            lbval = get_isz_cval(A_SPTRG(lb));
          }
          if (lbval) {
            ast = mk_binop(OP_SUB, ast, mk_isz_cval(lbval, astb.bnd.dtype),
                           astb.bnd.dtype);
          }
        } else {
          int lwast;
          lwast = ADD_LWAST(dtype, i);
          ast = mk_binop(OP_SUB, ast, lwast, astb.bnd.dtype);
        }
        if (prevmpyer) {
          ast = mk_binop(OP_MUL, ast, prevmpyer, astb.bnd.dtype);
        }
      } else {
        if (AD_ZBASE(ad)) {
          ast = mk_binop(OP_MUL, AD_LWAST(ad, i), AD_MLPYR(ad, i),
                         astb.bnd.dtype);
        } else {
          ast = 0;
        }
      }
      if (i == 0 || zbaseast == 0) {
        zbaseast = ast;
      } else if (A_ALIASG(AD_ZBASE(ad)) == 0 && ast) {
        zbaseast = mk_binop(OP_ADD, zbaseast, ast, astb.bnd.dtype);
      }

      if (XBIT(57, 0x10000)) {
        if (i < r - 1) {
          /* add assignment to multiplier temp for next dimension */
          nexttmp = AD_MLPYR(ad, i + 1);
          if (nexttmp && A_ALIASG(nexttmp) == 0) {
            int sstride;
            prevmpyer = get_local_multiplier(newdsc, i + 1);
            if (XBIT(58, 0x40000000)) {
              /* no multiply-by-section-stride */
              ast = prevmpyer;
            } else {
              sstride = get_section_stride(newdsc, i + 1);
              ast = mk_binop(OP_MUL, sstride, prevmpyer, astb.bnd.dtype);
            }
            ast = mk_assn_stmt(nexttmp, ast, astb.bnd.dtype);
            std = add_stmt_after(ast, std);
          } else {
            prevmpyer = nexttmp;
          }
        }
      } else {
        /* add assignment to multiplier temp for next dimension */
        tmp = AD_MLPYR(ad, i);
        nexttmp = AD_MLPYR(ad, i + 1);
        if (tmp && nexttmp && A_ALIASG(nexttmp) == 0) {
          if (AD_LWBD(ad, i) == astb.bnd.one)
            ast = astb.bnd.one;
          else
            ast = AD_LWAST(ad, i);
          ast = mk_mlpyr_expr(ast, AD_UPAST(ad, i), tmp);
          ast = mk_assn_stmt(nexttmp, ast, astb.bnd.dtype);
          std = add_stmt_after(ast, std);
        }
        prevmpyer = nexttmp;
      }
    }
  }
  if (XBIT(57, 0x10000)) {
    int ast;
    ast = get_xbase(newdsc);
    zbaseast = mk_binop(OP_ADD, zbaseast, ast, astb.bnd.dtype);
  }
  if (zbaseast && A_ALIASG(AD_ZBASE(ad)) == 0) {
    int tmp, ast;
    /* add assignment to zbase temp */
    tmp = AD_ZBASE(ad);
    ast = mk_assn_stmt(tmp, zbaseast, astb.bnd.dtype);
    std = add_stmt_after(ast, std);
  }

  /* if it is optional dummy */
  if (OPTARGG(arg) && !f77_local) {
    astnew = mk_stmt(A_ENDIF, 0);
    std = add_stmt_after(astnew, std);
  }

  ENTSTDP(entry, std);
}

static void
component_init_allocd_auto(int ast, int std)
{
  SPTR sptr = memsym_of_ast(ast);
  DTYPE dtype = DTYPEG(sptr);
  DTYPE basedtype = DDTG(dtype);

  if (get_struct_initialization_tree(basedtype) != 0 && !CCSYMG(sptr)) {
    SPTR prototype = get_dtype_init_template(basedtype);
    if (prototype > NOSYM) {
      int tast = mk_id(prototype);
      int dest = mk_id(sptr);
      int j, ndim = 0;
      if (DTY(dtype) == TY_ARRAY) {
        int shape = A_SHAPEG(ast);
        int indx[MAXDIMS];
        ndim = SHD_NDIM(shape);
        for (j = 0; j < ndim; j++) {
          int astdo = mk_stmt(A_DO, 0);
          indx[j] = mk_id(get_temp(astb.bnd.dtype));
          A_DOVARP(astdo, indx[j]);
          A_M1P(astdo, SHD_LWB(shape, j));
          A_M2P(astdo, SHD_UPB(shape, j));
          A_M3P(astdo, astb.i1);
          A_M4P(astdo, 0);
          add_stmt_before(astdo, std);
        }
        dest = mk_subscr(dest, indx, ndim, basedtype);
      }
      add_stmt_before(mk_assn_stmt(dest, tast, basedtype), std);
      while (ndim-- > 0) {
        int astdo = mk_stmt(A_ENDDO, 0);
        add_stmt_before(astdo, std);
      }
    }
  }
}

/** \brief Insert memory allocation code for a symbol.
    \param sptr symbol table index for the symbol
    \return true if \sptr is an array of derived type, and the caller must
            initialize its components (by calling component_init_allocd_auto
            after calling wrap_symbol for sptr)
 */
static bool
allocate_one_auto(int sptr)
{
  bool need_init = false;

  if (ADJLENG(sptr))
    add_auto_len(sptr, EntryStd);
  if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
    if (ADJARRG(sptr) || RUNTIMEG(sptr)) {
      add_auto_bounds(sptr, EntryStd);
    }
    if (!ALIGNG(sptr)
    ) {
      ADSC *ad;
      int r, i, ast, subscr[7];
      ad = AD_DPTR(DTYPEG(sptr));
      /* make the subscripts */
      r = AD_NUMDIM(ad);
      for (i = 0; i < r; ++i)
        subscr[i] = mk_triple(AD_LWAST(ad, i), AD_UPAST(ad, i), 0);
      ast = mk_id(sptr);
      mk_mem_allocate(ast, subscr, EntryStd, ast);
      if (DTYG(DTYPEG(sptr)) == TY_DERIVED) {
        need_init = true; // caller must initialize components
        dealloc_dt_auto(ast, sptr, ExitStd);
      } else {
        mk_mem_deallocate(ast, ExitStd);
      }
    }
  } else if (!POINTERG(sptr)) {
    if (ADJLENG(sptr)) {
      /* scalar adjustable length char string */
      mk_allocate_scalar(0, sptr, EntryStd);
      mk_deallocate_scalar(0, sptr, ExitStd);
    }
  }
  return need_init;
} /* allocate_one_auto */

static int
get_array_pointer(int arg)
{
  int p_sptr;

  assert(SCG(arg) == SC_DUMMY || SCG(arg) == SC_CMBLK,
         "get_array_pointer: arg not dummy", arg, 4);
  p_sptr = MIDNUMG(arg);
  if (p_sptr == 0) {
    p_sptr = sym_get_ptr(arg);
    MIDNUMP(arg, p_sptr);
  }
  return p_sptr;
}

static void
declare_dummy_array(int arg)
{
  int dtype;
  ADSC *ad;

  /* if(SEQG(arg)&& !f77_local) return;*/
  if (!MIDNUMG(arg))
    return;
  if (SCG(arg) != SC_DUMMY && SCG(arg) != SC_CMBLK) {
    assert(SCG(arg) == SC_BASED,
           "declare_dummy_array: wrong SC for rewritten arg", arg, 4);
    assert(MIDNUMG(arg) != 0, "declare_dummy_arg: null pointer", arg, 4);
    return;
  }
  dtype = DTYPEG(arg);
  if (DTY(dtype) == TY_ARRAY) {
    ad = AD_DPTR(dtype);
    AD_DEFER(ad) = 1;
    ALLOCP(arg, 1);
  }
  SCP(arg, SC_BASED);
}

static void
declare_array_dummys(int this_entry)
{
  int dscptr;
  int arg, narg;
  int i;
  int oldarg;

  narg = PARAMCTG(this_entry);
  dscptr = DPDSCG(this_entry);
  for (i = 0; i < narg; i++) {
    arg = aux.dpdsc_base[dscptr + i];
    if (arg == 0)
      continue;
    if (STYPEG(arg) != ST_ARRAY && STYPEG(arg) != ST_VAR)
      continue;
    if (is_bad_dtype(DTYPEG(arg)))
      continue;
    if (!is_array_type(arg) && (!XBIT(58, 0x10000) || !POINTERG(arg)))
      continue;
    if (F90POINTERG(arg))
      continue;
    oldarg = NEWARGG(arg);
    if (oldarg != 0)
      declare_dummy_array(oldarg);
  }
}

int
get_allobnds(int sptr, int ast)
{
  int i;
  int ndim;
  int subs[7];
  int lb, ub;
  int arrdsc;
  int sdsc;
  int ins;
  int dtype;

  arrdsc = DESCRG(sptr);
  assert(arrdsc && STYPEG(arrdsc) == ST_ARRDSC,
         "get_allobnds: have to be descriptor", sptr, 3);
  dtype = DTYPEG(sptr);
  sdsc = 0;
  ins = SECDG(arrdsc);
  if (ins)
    sdsc = INS_DESCR(ins);
  if (!sdsc)
    sdsc = SECDSCG(arrdsc);
  assert(sdsc, "get_allobnds: no section descriptor", sptr, 3);
  ndim = rank_of_sym(sptr);
  for (i = 0; i < ndim; ++i) {
    lb = check_member(ast, get_local_lower(sdsc, i));
    if (normalize_bounds(sptr))
      lb = add_lbnd(dtype, i, lb, ast);
    ub = check_member(ast, get_local_upper(sdsc, i));
    if (normalize_bounds(sptr))
      ub = add_lbnd(dtype, i, ub, ast);
    subs[i] = mk_triple(lb, ub, 0);
  }
  return mk_subscr(ast, subs, ndim, DTY(DTYPEG(sptr)));
}

static void
redimension(int sptr, int memberast)
{
  int astnew;
  int tmpast;

  tmpast = get_allobnds(sptr, check_member(memberast, mk_id(sptr)));
  astnew = mk_stmt(A_REDIM, 0);
  /*A_TKNP(astnew, TK_REDIMENSION);*/
  interr("dmp_out.c:redimension()", sptr, 2);
  A_LOPP(astnew, 0);
  A_SRCP(astnew, tmpast);
  add_stmt_before(astnew, EntryStd);
}

#ifdef FLANG_DPM_OUT_UNUSED
static void
add_adjarr_bounds_extr_f77(int sym, int entry, int call_ast)
{
  int dtype;
  ADSC *ad;
  int numdim;
  int i;
  int bnd;
  int ast;
  int actual_bnd;
  int bnd_sptr;
  int bndast;

  dtype = DTYPEG(sym);
  ad = AD_DPTR(dtype);
  numdim = AD_NUMDIM(ad);
  /* NOTE: a bound is adjustable if its ast is non-zero and it is
   *	     not a constant or aliased constant.
   */
  for (i = 0; i < numdim; i++) {
    bnd = AD_LWBD(ad, i);
    actual_bnd = find_actual(bnd, entry, call_ast);
    bndast = AD_LWAST(ad, i);
    if (bndast && A_TYPEG(bndast) == A_ID) {
      bnd_sptr = A_SPTRG(bndast);
      IGNOREP(bnd_sptr, 0);
      if (actual_bnd && AD_LWAST(ad, i) != actual_bnd) {
        ast = mk_assn_stmt((int)AD_LWAST(ad, i), actual_bnd, DT_INT);
        add_stmt_before(ast, EntryStd);
      }
    }
    bnd = AD_UPBD(ad, i);
    actual_bnd = find_actual(bnd, entry, call_ast);
    bndast = AD_UPAST(ad, i);
    if (bndast && A_TYPEG(bndast) == A_ID) {
      bnd_sptr = A_SPTRG(bndast);
      IGNOREP(bnd_sptr, 0);
      if (actual_bnd && AD_UPAST(ad, i) != actual_bnd) {
        ast = mk_assn_stmt((int)AD_UPAST(ad, i), actual_bnd, DT_INT);
        add_stmt_before(ast, EntryStd);
      }
    }
  }
}

/* This will replace dummy with actual at distribute and alignment
 * data structure, for example
 * interface, extrinsic(f77_local) sub(a,m); distribute a(cyclic(m))
 */
static void
update_with_actual(int arg)
{
  int align;

  align = ALIGNG(arg);
  update_bounds_with_actual(arg);
}

static void
update_bounds_with_actual(int sptr)
{
  int ndim, i;
  ADSC *ad;
  int bnd, actual_bnd;

  if (DTY(DTYPEG(sptr)) != TY_ARRAY)
    return;

  ad = AD_DPTR(DTYPEG(sptr));
  ndim = AD_NUMDIM(ad);
  for (i = 0; i < ndim; i++) {
    bnd = AD_LWBD(ad, i);
    actual_bnd = ast_rewrite(bnd);
    AD_LWBD(ad, i) = actual_bnd;

    bnd = AD_LWAST(ad, i);
    actual_bnd = ast_rewrite(bnd);
    AD_LWAST(ad, i) = actual_bnd;

    bnd = AD_UPBD(ad, i);
    actual_bnd = ast_rewrite(bnd);
    AD_UPBD(ad, i) = actual_bnd;

    bnd = AD_UPAST(ad, i);
    actual_bnd = ast_rewrite(bnd);
    AD_UPAST(ad, i) = actual_bnd;

    bnd = AD_EXTNTAST(ad, i);
    actual_bnd = ast_rewrite(bnd);
    AD_EXTNTAST(ad, i) = actual_bnd;
  }
}

/* this routine will rewrite ast such that
 *  dummy will be replaced with actual
 */
static int
find_actual(int ast, int entry, int call_ast)
{
  int actual_ast;
  set_actual(entry, call_ast, FALSE);
  actual_ast = ast_rewrite(ast);
  return actual_ast;
}

/* This routine is to set actual with dummy
 * which later will be used for ast_rwrite
 */
static void
set_actual(int entry, int call_ast, LOGICAL arrays)
{
  int narg, dscptr;
  int argt;
  int i;
  int arg;
  int actual;

  narg = PARAMCTG(entry);
  dscptr = DPDSCG(entry);
  argt = A_ARGSG(call_ast);
  for (i = 0; i < narg; ++i) {
    arg = aux.dpdsc_base[dscptr + i];
    if (arg == 0)
      continue;
    if (STYPEG(arg) != ST_ARRAY && STYPEG(arg) != ST_VAR)
      continue;
    if (arrays || DTY(DTYPEG(arg)) != TY_ARRAY ||
        (DISTG(arg) == 0 && ALIGNG(arg) == 0)) {
      actual = ARGT_ARG(argt, i);
      if (actual && actual != astb.ptr0 && actual != astb.ptr0c) {
        ast_replace(mk_id(arg), actual);
      }
    }
  }
}
#endif

#undef BND_ASSN_PRECEDES

#if defined(BND_ASSN_PRECEDES)
static int
bnd_assn_precedes(int lhs, int entryStd, int wh)
{
  /*
   * We're at a point in the prologue where a bounds temp will be used
   * and  that the assignment to the temp has already been generated.
   * Determine if the assignment precedes this point -- if not, we'll
   * have to replicate the assignment (see f15414).
   *
   * 12/3/2008 NOTE:  Currently, we're not using this method to solve
   * f15414; scanning the STDs at this point is a bit risky, plus I
   * don't like the idea of generating redundant assignments that we
   * probably cannot delete.
   * The solution is to first look for those automatic arrays whose
   * whose bounds can be written 'early' in transform_wrapup (see
   * the first loop over the symbol table).  Nevertheless, I'm keeping
   * this function here (just in case) and using the macro BND_ASSN_PRECEDES
   * to guard its calls.
   */
  int aa, ss;
  int fnd = 0;
  for (ss = STD_PREV(entryStd); ss; ss = STD_PREV(ss)) {
    aa = STD_AST(ss);
    if (A_TYPEG(aa) == A_ASN && A_DESTG(aa) == lhs) {
      fnd = 1;
      break;
    }
  }
  fprintf(stderr, "%s %sFOUND%d\n", SYMNAME(A_SPTRG(lhs)), fnd ? "" : "NOT",
          wh);
  return fnd;
}
#endif

static void
add_auto_bounds(int sym, int entryStd)
{
  int dtype;
  ADSC *ad;
  int numdim;
  int i;
  int bnd;
  int ast;
  int tmp;
  int zbaseast, mlpyrast;

  dtype = DTYPEG(sym);
  ad = AD_DPTR(dtype);
  numdim = AD_NUMDIM(ad);
  zbaseast = 0;
  mlpyrast = astb.bnd.one;
  /* NOTE: a bound is adjustable if its ast is non-zero and it is
   *	     not a constant or aliased constant.
   */
  for (i = 0; i < numdim; i++) {
    bnd = AD_LWBD(ad, i);
    tmp = AD_LWAST(ad, i);
    if (A_TYPEG(bnd) == A_ID && EARLYSPECG(A_SPTRG(bnd))) {
      ;
    } else if (bnd && A_ALIASG(tmp) == 0 && bnd != tmp) {
      if (A_VISITG(tmp) == 0) {
        ast = mk_assn_stmt(tmp, bnd, DT_INT);
        bnd = get_scalar_in_expr(bnd, EntryStd, FALSE);
        add_stmt_before(ast, entryStd);
        A_SRCP(ast, bnd);
        ast_visit(tmp, tmp);
      }
#if defined(BND_ASSN_PRECEDES)
      else if (!bnd_assn_precedes(tmp, entryStd, 0)) {
      }
#endif
    }
    bnd = AD_UPBD(ad, i);
    tmp = AD_UPAST(ad, i);
    if (A_TYPEG(bnd) == A_ID && EARLYSPECG(A_SPTRG(bnd))) {
      ;
    } else if (bnd && A_ALIASG(tmp) == 0 && bnd != tmp) {
      if (A_VISITG(tmp) == 0) {
        ast = mk_assn_stmt(tmp, bnd, DT_INT);
        bnd = get_scalar_in_expr(bnd, EntryStd, FALSE);
        add_stmt_before(ast, entryStd);
        A_SRCP(ast, bnd);
        ast_visit(tmp, tmp);
      }
#if defined(BND_ASSN_PRECEDES)
      else if (!bnd_assn_precedes(tmp, entryStd, 1)) {
        /*
        ast = mk_assn_stmt(tmp, bnd, DT_INT);
        bnd = get_scalar_in_expr(bnd, entryStd, FALSE);
        add_stmt_before(ast, entryStd);
        A_SRCP(ast, bnd);
        */
      }
#endif
    }
    tmp = AD_EXTNTAST(ad, i);
    bnd = mk_extent_expr(AD_LWAST(ad, i), AD_UPAST(ad, i));
    if (tmp && A_ALIASG(tmp) == 0 && tmp != bnd) {
      if (A_VISITG(tmp) == 0) {
        ast = mk_assn_stmt(tmp, bnd, DT_INT);
        bnd = get_scalar_in_expr(bnd, EntryStd, FALSE);
        add_stmt_before(ast, entryStd);
        A_SRCP(ast, bnd);
        ast_visit(tmp, tmp);
      }
#if defined(BND_ASSN_PRECEDES)
      else if (!bnd_assn_precedes(tmp, entryStd, 2)) {
      }
#endif
    }
    tmp = AD_MLPYR(ad, i);
    if (tmp && !A_ALIASG(tmp) && tmp != mlpyrast && A_TYPEG(tmp) == A_ID) {
      if (A_VISITG(tmp) == 0) {
        ast = mk_assn_stmt(tmp, mlpyrast, DT_INT);
        add_stmt_before(ast, entryStd);
        ast_visit(tmp, tmp);
      }
#if defined(BND_ASSN_PRECEDES)
      else if (!bnd_assn_precedes(tmp, entryStd, 3)) {
      }
#endif
      mlpyrast = tmp;
    }
    mlpyrast = mk_binop(OP_MUL, mlpyrast, AD_EXTNTAST(ad, i), astb.bnd.dtype);
    if (zbaseast == 0) {
      zbaseast =
          mk_binop(OP_MUL, AD_LWAST(ad, i), AD_MLPYR(ad, i), astb.bnd.dtype);
    } else {
      zbaseast = mk_binop(
          OP_ADD, zbaseast,
          mk_binop(OP_MUL, AD_LWAST(ad, i), AD_MLPYR(ad, i), astb.bnd.dtype),
          astb.bnd.dtype);
    }
  }
  tmp = AD_NUMELM(ad);
  if (tmp && !A_ALIASG(tmp) && tmp != mlpyrast && A_TYPEG(tmp) == A_ID) {
    if (A_VISITG(tmp) == 0) {
      ast = mk_assn_stmt(tmp, mlpyrast, DT_INT);
      add_stmt_before(ast, entryStd);
      ast_visit(tmp, tmp);
    }
#if defined(BND_ASSN_PRECEDES)
    else if (!bnd_assn_precedes(tmp, entryStd, 4)) {
    }
#endif
  }
  tmp = AD_ZBASE(ad);
  if (tmp && A_ALIASG(tmp) == 0 && tmp != zbaseast && A_TYPEG(tmp) == A_ID) {
    if (A_VISITG(tmp) == 0) {
      ast = mk_assn_stmt(tmp, zbaseast, astb.bnd.dtype);
      add_stmt_before(ast, entryStd);
      ast_visit(tmp, tmp);
    }
#if defined(BND_ASSN_PRECEDES)
    else if (!bnd_assn_precedes(tmp, entryStd, 5)) {
    }
#endif
  }
} /* add_auto_bounds */

/* this is modified from symutl.c */
static void
mk_allocate_scalar(int memberast, int sptr, int before)
{
  /* build and insert the allocate statement */
  int ast, a;
  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_ALLOCATE);
  A_LOPP(ast, 0);
  a = check_member(memberast, mk_id(sptr));
  A_SRCP(ast, a);
  add_stmt_before(ast, before);
} /* mk_allocate_scalar */

static void
mk_deallocate_scalar(int memberast, int sptr, int after)
{
  /* build and insert the deallocate statement */
  int ast, a;
  ast = mk_stmt(A_ALLOC, 0);
  A_TKNP(ast, TK_DEALLOCATE);
  A_LOPP(ast, 0);
  a = check_member(memberast, mk_id(sptr));
  A_SRCP(ast, a);
  add_stmt_after(ast, after);
} /* mk_deallocate_scalar */

static void
dealloc_dt_auto(int ast, int sptr, int after)
{
  /*
   * 'deallocate' of an automatic array of derived type containing
   * allocatable components has already been handled --
   * see semutil2.c;sem_set_storage_class() and func.:rewrite_calls().
   */
  if (!has_allocattr(sptr))
    mk_mem_deallocate(ast, after);
}

static int
gen_RTE_loc(int arg_ast)
{
  return mk_unop(OP_LOC, arg_ast, DT_PTR);
}

static int
get_scalar_in_expr(int expr, int std, LOGICAL astversion)
{
  int l, r, d, o;
  int l1, l2, l3;
  int i, nargs, argt;

  if (expr == 0)
    return expr;
  switch (A_TYPEG(expr)) {
  /* expressions */
  case A_BINOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = get_scalar_in_expr(A_LOPG(expr), std, astversion);
    r = get_scalar_in_expr(A_ROPG(expr), std, astversion);
    return mk_binop(o, l, r, d);
  case A_UNOP:
    o = A_OPTYPEG(expr);
    d = A_DTYPEG(expr);
    l = get_scalar_in_expr(A_LOPG(expr), std, astversion);
    return mk_unop(o, l, d);
  case A_CONV:
    d = A_DTYPEG(expr);
    l = get_scalar_in_expr(A_LOPG(expr), std, astversion);
    return mk_convert(l, d);
  case A_PAREN:
    d = A_DTYPEG(expr);
    l = get_scalar_in_expr(A_LOPG(expr), std, astversion);
    return mk_paren(l, d);
  case A_MEM:
    l = get_scalar_in_expr(A_PARENTG(expr), std, astversion);
    r = A_MEMG(expr);
    d = A_DTYPEG(r);
    return mk_member(l, r, d);
  case A_SUBSTR:
    d = A_DTYPEG(expr);
    l1 = get_scalar_in_expr(A_LOPG(expr), std, astversion);
    l2 = l3 = 0;
    if (A_LEFTG(expr))
      l2 = get_scalar_in_expr(A_LEFTG(expr), std, astversion);
    if (A_RIGHTG(expr))
      l3 = get_scalar_in_expr(A_RIGHTG(expr), std, astversion);
    return mk_substr(l1, l2, l3, d);
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(expr);
    argt = A_ARGSG(expr);
    for (i = 0; i < nargs; ++i) {
      ARGT_ARG(argt, i) =
          get_scalar_in_expr(ARGT_ARG(argt, i), std, astversion);
    }
    return expr;
  case A_CNST:
  case A_CMPLXC:
  case A_ID:
    return expr;
  case A_SUBSCR:
    if (astversion)
      expr = emit_get_scalar(expr, std);
    else
      expr = emit_get_scalar_sub(expr, std);
    return expr;
  case A_TRIPLE:
    l1 = get_scalar_in_expr(A_LBDG(expr), std, astversion);
    l2 = get_scalar_in_expr(A_UPBDG(expr), std, astversion);
    l3 = get_scalar_in_expr(A_STRIDEG(expr), std, astversion);
    return mk_triple(l1, l2, l3);
  default:
    interr("get_scalar_in_expr: unknown expression", expr, 2);
    return expr;
  }
}

static int
emit_get_scalar_sub(int a, int std)
{
  int l;
  int astnew;
  int i, nargs, argt;
  int asd;
  int ndim;
  int sptr, sptr1;
  int arrdsc;
  int secd;
  int descr, lop;

  asd = A_ASDG(a);
  ndim = ASD_NDIM(asd);
  l = A_LOPG(a);
  sptr1 = sptr_of_subscript(a);
  if (!DISTG(sptr1) && !ALIGNG(sptr1))
    return a;
  /* It is distributed.  Create a temp to hold the value */
  sptr = sym_get_scalar(SYMNAME(sptr1), "s", DTY(DTYPEG(sptr1) + 1));
  /* put out a call to fetch the value */
  /* call pghpf_get_scalar(temp, array_base, array, subscripts) */

  arrdsc = DESCRG(sptr1);
  assert(arrdsc, "emit_get_scalar_sub: descriptor does not exist", sptr, 3);
  secd = SECDG(arrdsc);
  assert(secd, "emit_get_scalar_sub: descriptor does not exist", sptr, 3);
  descr = INS_DESCR(secd);
  assert(descr, "emit_get_scalar_sub: descriptor does not exist", sptr, 3);
  nargs = ndim + 3;
  argt = mk_argt(nargs);
  ARGT_ARG(argt, 0) = mk_id(sptr);
  ARGT_ARG(argt, 1) = mk_id(sptr1);
  ARGT_ARG(argt, 2) = mk_id(descr);
  DESCUSEDP(A_SPTRG(l), 1);
  for (i = 0; i < ndim; ++i) {
    astnew = mk_default_int(ASD_SUBS(asd, i));
    if (normalize_bounds(sptr1))
      astnew = sub_lbnd(DTYPEG(sptr1), i, astnew, a);
    ARGT_ARG(argt, i + 3) = astnew;
  }
  astnew = mk_stmt(A_CALL, 0);
  lop = mk_id(sym_mkfunc(mkRteRtnNm(RTE_get_scalar), DT_NONE));
  A_LOPP(astnew, lop);
  A_ARGCNTP(astnew, nargs);
  A_ARGSP(astnew, argt);
  add_stmt_before(astnew, std);
  report_comm(std, GETSCALAR_CAUSE);
  return mk_id(sptr);
}

/* Add assignments to bounds if sptr is an adjustable array. */
static void
add_adjarr_bounds(int sptr)
{
  if (DTY(DTYPEG(sptr)) != TY_ARRAY)
    return;

  if (ADJARRG(sptr)) {
    add_bound_assignments(sptr);
  }
}

/* look for A_ID for a OPTIONAL dummy parameter; make a
 * PRESENT(dummy) call; if there is more than one, .AND. the
 * tests together; I wrote this to handle the same AST types
 * handled in get_scalar_in_expr */
static void
find_presence(int ast, int *piftest)
{
  int nargs, argt, i, ndim, asd, sptr;
  if (ast == 0)
    return;
  switch (A_TYPEG(ast)) {
  /* expressions */
  case A_ID:
    sptr = A_SPTRG(ast);
    if (SCG(sptr) == SC_DUMMY && OPTARGG(sptr)) {
      int iftest;
      if (NEWARGG(sptr)) {
        iftest = ast_intr(I_PRESENT, stb.user.dt_log, 1, mk_id(NEWARGG(sptr)));
      } else {
        iftest = ast_intr(I_PRESENT, stb.user.dt_log, 1, ast);
      }
      if (*piftest) {
        iftest = mk_binop(OP_LAND, iftest, *piftest, stb.user.dt_log);
      }
      *piftest = iftest;
    }
    break;
  case A_CNST:
  case A_CMPLXC:
    break;
  case A_BINOP:
    find_presence(A_LOPG(ast), piftest);
    find_presence(A_ROPG(ast), piftest);
    break;
  case A_UNOP:
  case A_CONV:
  case A_PAREN:
    find_presence(A_LOPG(ast), piftest);
    break;
  case A_MEM:
    find_presence(A_PARENTG(ast), piftest);
    break;
  case A_SUBSTR:
    find_presence(A_LOPG(ast), piftest);
    if (A_LEFTG(ast))
      find_presence(A_LEFTG(ast), piftest);
    if (A_RIGHTG(ast))
      find_presence(A_RIGHTG(ast), piftest);
    break;
  case A_INTR:
  case A_FUNC:
    nargs = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    for (i = 0; i < nargs; ++i) {
      find_presence(ARGT_ARG(argt, i), piftest);
    }
    break;
  case A_SUBSCR:
    find_presence(A_LOPG(ast), piftest);
    asd = A_ASDG(ast);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; ++i) {
      find_presence(ASD_SUBS(asd, i), piftest);
    }
    break;
  case A_TRIPLE:
    find_presence(A_LBDG(ast), piftest);
    find_presence(A_UPBDG(ast), piftest);
    find_presence(A_STRIDEG(ast), piftest);
    break;
  default:
    interr("find_presence: unknown expression", ast, 2);
    break;
  }
} /* find_presence */

static int
add_presence(int ast, int std)
{
  int astpresent = 0;
  find_presence(ast, &astpresent);
  if (astpresent) {
    int astnew;
    astnew = mk_stmt(A_IFTHEN, 0);
    A_IFEXPRP(astnew, astpresent);
    add_stmt_before(astnew, std);
  }
  return astpresent;
} /* add_presence */

static void
add_end_presence(int astpresent, int std)
{
  if (astpresent) {
    int astnew;
    astnew = mk_stmt(A_ENDIF, 0);
    add_stmt_before(astnew, std);
  }
} /* add_end_presence */

static void
add_bound_assignments(int sym)
{
  int dtype;
  ADSC *ad;
  int numdim;
  int i;
  int bnd;
  int ast;
  int tmp;
  int zbaseast;
  int astpresent;
  int isfval;
  int std, adjstd;
  int entryStd = EntryStd;

  isfval = this_entry_fval == sym;

  dtype = DTYPEG(sym);
  ad = AD_DPTR(dtype);
  numdim = AD_NUMDIM(ad);

  adjstd = 0;
  zbaseast = 0;
  /* NOTE: a bound is adjustable if its ast is non-zero and it is
   *       not a constant or aliased constant.
   */
  for (i = 0; i < numdim; i++) {
    bnd = AD_LWBD(ad, i);
    tmp = AD_LWAST(ad, i);
    if (bnd && !A_ALIASG(tmp) && !EARLYSPECG(A_SPTRG(tmp)) && tmp != bnd) {
      if (A_VISITG(tmp) == 0) {
        astpresent = add_presence(bnd, entryStd);
        ast = mk_assn_stmt(tmp, bnd, astb.bnd.dtype);
        bnd = get_scalar_in_expr(bnd, entryStd, FALSE);
        A_SRCP(ast, bnd);
        std = add_stmt_before(ast, entryStd);
        add_end_presence(astpresent, entryStd);
        ast_visit(tmp, tmp); /* mark id ast as visited */
      }
    }
    bnd = AD_UPBD(ad, i);
    tmp = AD_UPAST(ad, i);
    if (bnd && !A_ALIASG(tmp) && !EARLYSPECG(A_SPTRG(tmp)) && tmp != bnd) {
      if (A_VISITG(tmp) == 0) {
        astpresent = add_presence(bnd, entryStd);
        ast = mk_assn_stmt(tmp, bnd, astb.bnd.dtype);
        bnd = get_scalar_in_expr(bnd, entryStd, FALSE);
        A_SRCP(ast, bnd);
        std = add_stmt_before(ast, entryStd);
        add_end_presence(astpresent, entryStd);
        ast_visit(tmp, tmp); /* mark id ast as visited */
      }
    }
    bnd = mk_extent_expr(AD_LWAST(ad, i), AD_UPAST(ad, i));
    tmp = AD_EXTNTAST(ad, i);
    if (bnd && A_ALIASG(tmp) == 0) {
      if (A_VISITG(tmp) == 0) {
        astpresent = add_presence(bnd, entryStd);
        ast = mk_assn_stmt(tmp, bnd, astb.bnd.dtype);
        bnd = get_scalar_in_expr(bnd, entryStd, FALSE);
        A_SRCP(ast, bnd);
        std = add_stmt_before(ast, entryStd);
        add_end_presence(astpresent, entryStd);
      }
      ast_visit(tmp, tmp); /* mark id ast as visited */
    }
    {
      /* update the ZBASE ast tree */
      int nexttmp, ast;
      if (i == 0) {
        zbaseast = AD_LWAST(ad, i);
      } else if (A_ALIASG(AD_ZBASE(ad)) == 0) {
        int a;
        a = mk_binop(OP_MUL, AD_LWAST(ad, i), AD_MLPYR(ad, i), astb.bnd.dtype);
        zbaseast = mk_binop(OP_ADD, zbaseast, a, astb.bnd.dtype);
      }
      /* add assignment to multiplier temp for next dimension */
      tmp = AD_MLPYR(ad, i);
      nexttmp = AD_MLPYR(ad, i + 1);
      if (tmp && nexttmp && A_ALIASG(nexttmp) == 0 && A_VISITG(nexttmp) == 0) {
        if (AD_LWBD(ad, i) == astb.bnd.one)
          ast = astb.bnd.one;
        else
          ast = AD_LWAST(ad, i);
        ast = mk_mlpyr_expr(ast, AD_UPAST(ad, i), tmp);
        ast = mk_assn_stmt(nexttmp, ast, astb.bnd.dtype);
        std = add_stmt_before(ast, entryStd);
        if (!adjstd)
          adjstd = std;
        ast_visit(nexttmp, nexttmp); /* mark id ast as visited */
      }
    }
  }
  if (A_ALIASG(AD_ZBASE(ad)) == 0) {
    /* add assignment to zbase temp */
    tmp = AD_ZBASE(ad);
    if (A_VISITG(tmp) == 0) {
      ast = mk_assn_stmt(tmp, zbaseast, astb.bnd.dtype);
      std = add_stmt_before(ast, entryStd);
      ast_visit(tmp, tmp); /* mark id ast as visited */
    }
  }
}

/**
 * If the symbol is pointer-based, need to make sure that the storage class
 * of its descriptor is consistent with its pointer; specifically (f15624),
 * if the pointer is SC_PRIVATE, the descriptor must be SC_PRIVATE.
 * Generally speaking, privatizing temps occurs at the time the temps are
 * created.  Context is set wrt being in a parallel region.  Unfortunately,
 * the parallel vs serial context is not available at transform_wrapup-time
 * where the only information available is what's in the symbol table.  The
 * routine used to create descriptors is via symutl.c:trans_mkdescr(). This
 * routine is called from various points in the compiler and creates ST_ARRDSC
 * entries which eventually become ...$sd temp arrays. The bug of f15624 has
 * to do with the descriptors of the temps created for copy-in/copy-out calls
 * at certain call sites.
 * REVISION - the private storage class cannot simply be copied to the
 * descriptor; a private allocatable array will have a shared descriptor.
 */
static void
fix_sdsc_sc(int sptr, int sdsc, int arrdsc)
{
  if (SCG(sptr) == SC_BASED) {
    if (SCG(arrdsc) == SC_PRIVATE) {
      SCP(sdsc, SC_PRIVATE);
    }
  }
}

/* FIXME - how are we planning to deal with these special macros?? */
void rw_dpmout_state(RW_ROUTINE, RW_FILE)
{
  int nw;

  RW_FD(&dtb.avl, dtb.avl, 1);
  if (dtb.avl) {
    RW_FD(dtb.base, DTABLE, dtb.avl);
  }
}

static int
get_num(void)
{
  char *p;
  INT val;

  while (*currp == ' ')
    ++currp;
  p = currp;
  while (*currp != ' ' && *currp != '\n')
    ++currp;
  (void)atoxi(p, &val, (int)(currp - p), 10);
  return val;
} /* get_num */

void
ipa_restore_dtb(char *line)
{
  int d, w;
  if (dtb.base == NULL) {
    init_dtb();
  }
  currp = line + 1;
  w = get_num();
  if (w == 1) {
    int i;
    d = get_num();
    if (d >= dtb.avl) {
      NEED(d + 1, dtb.base, DTABLE, dtb.size, d + 480);
      dtb.avl = d + 1;
    }
    TMPL_DESCR(d) = get_num();
    TMPL_RANK(d) = get_num();
    TMPL_FLAG(d) = get_num();
    TMPL_DIST_TARGET(d) = get_num();
    TMPL_DIST_TARGET_DESCR(d) = get_num();
    TMPL_ISSTAR(d) = get_num();
    TMPL_ALIGN_TARGET(d) = get_num();
    TMPL_TARGET_DESCR(d) = get_num();
    TMPL_CONFORM(d) = get_num();
    TMPL_COLLAPSE(d) = get_num();
    TMPL_TYPE(d) = get_num();
    TMPL_ALIGNEE_SC(d) = get_num();
    TMPL_TARGET_SC(d) = get_num();
    for (i = 0; i < TMPL_RANK(d); ++i) {
      TMPL_LB(d, i) = get_num();
      TMPL_UB(d, i) = get_num();
    }
  } else if (w == 2) {
    d = get_num();
    if (d >= dtb.avl) {
      NEED(d + 1, dtb.base, DTABLE, dtb.size, d + 480);
      dtb.avl = d + 1;
    }
    INS_DESCR(d) = get_num();
    INS_TEMPLATE(d) = get_num();
    INS_RANK(d) = get_num();
    INS_DTYPE(d) = get_num();
    INS_KIND(d) = get_num();
    INS_SIZE(d) = get_num();
  }
} /* ipa_restore_dtb */

int
newargs_for_llvmiface(int sptr)
{
  return newargs_for_entry(sptr);
}

void
interface_for_llvmiface(int this_entry, int new_dscptr)
{
  int arg, narg;
  int i;
  int argnum, dscptr;

  narg = PARAMCTG(this_entry);
  dscptr = DPDSCG(this_entry);
  this_entry_fval = FVALG(this_entry);

  if (narg) {
    NEW(orderargs, int, narg);
    reorder_args(this_entry);
  }

  for (i = 0; i < narg; i++) {
    argnum = orderargs[i];
    arg = aux.dpdsc_base[dscptr + argnum];
    if (STYPEG(arg) != ST_ARRAY && STYPEG(arg) != ST_VAR)
      continue;
    if (ADJLENG(arg)) {
      add_auto_len(arg, EntryStd);
    }
    if (normalize_bounds(arg)) {
      if (needs_redim(arg))
        emit_redim(arg);
    } else
      emit_kopy_in(arg, this_entry, 0);
  }

  if (narg) {
    FREE(orderargs);
    PARAMCTP(this_entry, 2 * narg);
    DPDSCP(this_entry, new_dscptr);
  }
}

void
undouble_callee_args_llvmf90(int iface)
{
  int this_entry = iface;
  int dscptr, new_dscptr;
  int narg, orignarg, newnarg;
  int i;
  int arg, descr;
  int oldarg;

  if (this_entry) {
    int f_descr;
    int istart;
    narg = PARAMCTG(this_entry);
    if (!narg)
      return;
    orignarg = narg / 2;
    newnarg = 0;
    dscptr = DPDSCG(this_entry);
    new_dscptr = get_arg_table();
    for (i = 0; i < orignarg; i++) {
      int arg = aux.dpdsc_base[dscptr + i];
      put_arg_table(arg);
      newnarg++;
      if (pass_reflected_arg_by_value(arg))
        newnarg++;
    }

    istart = 0;
    f_descr = 0;
    if (MVDESCG(this_entry)) {
      f_descr = FVALG(this_entry);
      if (f_descr && f_descr == aux.dpdsc_base[dscptr + 0]) {
        oldarg = NEWARGG(f_descr);
        if (arg_has_descriptor(oldarg)) {
          f_descr = aux.dpdsc_base[dscptr + orignarg + 0];
          istart = 1;
        }
      }
    }
    for (i = istart; i < orignarg; i++) {
      arg = aux.dpdsc_base[dscptr + i];
      oldarg = 0;
      if (arg)
        oldarg = NEWARGG(arg);
      descr = aux.dpdsc_base[dscptr + orignarg + i];
      if (arg_has_descriptor(oldarg)) {
        put_arg_table(descr);
        newnarg++;
      } else {
        /* change SC from DUMMY to LOCAL */
        if (XBIT(57, 0x10000)) {
          if (CLASSG(descr)) {
            if (STYPEG(SCOPEG(descr)) == ST_MODULE)
              SCP(descr, SC_EXTERN);
            else
              SCP(descr, SC_STATIC);
          } else
            SCP(descr, SC_LOCAL);
        }
      }
    }
    if (istart) {
      put_arg_table(f_descr);
      newnarg++;
    }
    PARAMCTP(this_entry, newnarg);
    DPDSCP(this_entry, new_dscptr);
  }
}

