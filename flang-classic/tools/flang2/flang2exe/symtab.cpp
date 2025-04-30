/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *  \brief Fortran symbol table access module
 *
 * FTN - symbol table access module.  This module contains the routines used
 * to initialize, update, access, and dump the symbol table.  Note that in
 * addition to being used by FTN, this module is used by the utility
 * program, symini, which processes intrinsic and generic definitions in
 * order to set up the initial symbol table for FTN.
 */

#include "gbldefs.h"
#include "error.h"
#include "symtab.h"
#include "machar.h"
#include "symtabdf.h"
#include "syminidf.h"
#include "soc.h"
#include "llmputil.h"
#include "llutil.h"
#include "llassem.h"
#include "dtypeutl.h"
#include "symfun.h"

/* implicit data types */
static struct {
  DTYPE dtype;
  bool set; /* True if set by IMPLICIT stmt */
} dtimplicit[26 + 26 + 2];

static void cng_generic(const char *, const char *);
static void cng_specific(const char *, const char *);
static void cng_inttyp(const char *, int);
static void clear_vc(void);

/* entry hack? */
static ENTRY onlyentry;

/*
 * Define macro which converts character into index into these arrays:
 */
#define IMPL_INDEX(uc)                          \
  (islower(uc) ? uc - 'a'                       \
               : (isupper(uc) ? 26 + (uc - 'A') \
                              : (uc == '$' ? 52 : (uc == '_' ? 53 : -1))))
/*--------------------------------------------------------------------------*/

/**
   \brief Initialize symbol table for new user program unit.
 */
void
sym_init(void)
{
  int i;
  INT tmp[2];
  DTYPE default_int;
  DTYPE default_real;
  extern void chkstruct();

  /* allocate symbol table and name table space:  */
  sym_init_first();

  init_chartab(); /* see dtypeutl.c */

  STG_RESET(stb.dt);
  STG_NEXT_SIZE(stb.dt, DT_MAX);
  for (i = 0; i <= DT_MAX; ++i)
    DTySet((DTYPE)i, pd_dtype[i]);

  /*
   * Set up initial implicit types.  All are real except for the letters i
   * thru n:
   */

  default_real = XBIT(124, 0x8) ? DT_DBLE : DT_REAL;
  for (i = 0; i < 54; i++) {
    dtimplicit[i].dtype = default_real;
    dtimplicit[i].set = false;
  }

  default_int = flg.i4 ? DT_INT : DT_SINT;
  if (XBIT(124, 0x10))
    default_int = DT_INT8;
  implicit_int(default_int);

  /*
   * now initialize symbol table. There are 2 cases: The first case occurs
   * within the utility symini - we start with a totally empty symbol
   * table. The second case occurs within FTN - the initial symbol table is
   * copied from some arrays set up by symini.
   */

  BCOPY(stb.stg_base, init_sym, SYM, INIT_SYMTAB_SIZE);
  stb.stg_avail = INIT_SYMTAB_SIZE;
  stb.stg_cleared = INIT_SYMTAB_SIZE;
  BCOPY(stb.n_base, init_names, char, INIT_NAMES_SIZE);
  stb.namavl = INIT_NAMES_SIZE;

  BCOPY(stb.hashtb, init_hashtb, int, HASHSIZE);

  if (XBIT(124, 0x20)) {
    cng_generic("real", "dble");
    cng_generic("cmplx", "dcmplx");
    cng_specific("alog", "dlog");
    cng_specific("alog10", "dlog10");
    cng_specific("amax1", "dmax1");
    cng_specific("amin1", "dmin1");
    cng_specific("amod", "dmod");
    cng_specific("cabs", "cdabs");
    cng_specific("csqrt", "cdsqrt");
    cng_specific("clog", "cdlog");
    cng_specific("cexp", "cdexp");
    cng_specific("csin", "cdsin");
    cng_specific("ccos", "cdcos");
    if (XBIT(124, 0x80000)) {
      cng_specific("floati", "dfloti");
      cng_specific("floatj", "dflotj");
      cng_specific("float", "dfloat");
      cng_specific(".floatk", ".dflotk");
    }
  }
  if (XBIT(124, 0x10)) {
    cng_generic("int", "int8");
    cng_specific("ifix", "kifix");
    cng_specific("idint", ".idint8");

    cng_generic("nint", "knint");
    cng_specific("idnint", "kidnnt");

    cng_specific("iabs", "kiabs");
    cng_specific("isign", "kisign");
    cng_specific("idim", "kidim");
    cng_specific("max0", ".kmax0");
    cng_specific("max1", "kmax1");
    cng_specific("min0", ".kmin0");
    cng_specific("min1", "kmin1");
    cng_specific("len", "klen");
    cng_specific("index", "kindex");

    cng_specific("lge", ".lge8");
    cng_specific("lgt", ".lgt8");
    cng_specific("lle", ".lle8");
    cng_specific("llt", ".llt8");

    cng_inttyp("ichar", DT_INT8);
    cng_inttyp("nlen", DT_INT8);
    cng_inttyp("nindex", DT_INT8);
  }

  /* enter constants into symbol table:  */

  /* int 0, 1 */
  tmp[0] = tmp[1] = 0;
  stb.i0 = getcon(tmp, DT_INT);
  stb.k0 = getcon(tmp, DT_INT8);
  tmp[1] = 1;
  stb.i1 = getcon(tmp, DT_INT);
  stb.k1 = getcon(tmp, DT_INT8);

  /* l.0, 2.0, 0.5 as floats(reals) */
  /* 1.0, 2.0, 0.5 as double */
  add_fp_constants();

  /* allocate space for auxiliary symtab structures: */

  if (aux.dpdsc_size <= 0) {
    aux.dpdsc_size = 100;
    NEW(aux.dpdsc_base, int, aux.dpdsc_size);
  }
  aux.dpdsc_avl = 0;

  if (aux.arrdsc_size <= 0) {
    aux.arrdsc_size = 200;
    NEW(aux.arrdsc_base, int, aux.arrdsc_size);
    aux.arrdsc_base[0] = 0;
  }
  aux.arrdsc_avl = 1;

  if (aux.nml_size <= 0) {
    aux.nml_size = 200;
    NEW(aux.nml_base, NMLDSC, aux.nml_size);
    aux.nml_base[0].sptr = 0;
    aux.nml_base[0].next = 0;
    aux.nml_base[0].lineno = 0;
  }
  aux.nml_avl = 1;

  if (aux.dvl_size <= 0) {
    aux.dvl_size = 32;
    NEW(aux.dvl_base, DVL, aux.dvl_size);
  }
  aux.dvl_avl = 0;

  if (aux.symi_size <= 0) {
    aux.symi_size = 100;
    NEW(aux.symi_base, SYMI, aux.symi_size);
    aux.symi_base[0].sptr = 0;
    aux.symi_base[0].next = 0;
  }
  aux.symi_avl = 1; /* 0 => end of list */

  llmp_reset_uplevel();

  BZERO(aux.vtypes, char, sizeof(aux.vtypes));
  clear_vc();

  aux.curr_entry = &onlyentry;
  stb.firstusym = (SPTR)stb.stg_avail;
  stb.lbavail = 0;
}

static void
cng_generic(const char *old, const char *New)
{
  int os, ns;

#undef COPYFIELD
#define COPYFIELD(f) stb.stg_base[os].f = stb.stg_base[ns].f
  os = getsym(old, strlen(old));
  ns = getsym(New, strlen(New));
#if DEBUG
  assert(STYPEG(os) == ST_GENERIC, "cng_generic not intr", os, ERR_Severe);
  assert(STYPEG(ns) == ST_GENERIC, "cng_generic not intr", ns, ERR_Severe);
#endif
  COPYFIELD(w9);
  COPYFIELD(w10);
  COPYFIELD(w11);
  COPYFIELD(w12);
  COPYFIELD(w13);
  COPYFIELD(w14);
  COPYFIELD(w15);
  COPYFIELD(w16);
#undef COPYFIELD
}

static void
cng_specific(const char *old, const char *New)
{
  int os, ns;

#define COPYFIELD(f) stb.stg_base[os].f = stb.stg_base[ns].f
  os = getsym(old, strlen(old));
  ns = getsym(New, strlen(New));
#if DEBUG
  assert(STYPEG(os) == ST_INTRIN, "cng_specific not intr", os, ERR_Severe);
  assert(STYPEG(ns) == ST_INTRIN, "cng_specific not intr", ns, ERR_Severe);
#endif
  DTYPEP(os, DTYPEG(ns));
  COPYFIELD(w9);
  COPYFIELD(w11);
  COPYFIELD(w12);
  COPYFIELD(w13);
  COPYFIELD(w14);
  COPYFIELD(w15);
#undef COPYFIELD
}

static void
cng_inttyp(const char *old, int dt)
{
  int ss;
  ss = getsym(old, strlen(old));
#if DEBUG
  assert(STYPEG(ss) == ST_INTRIN, "cng_inttyp not intr", ss, ERR_Severe);
#endif
  INTTYPP(ss, dt);
}

/**
   \brief Simple routine to reset the default integer type for
   implicitly typing integer variables.  Needed for compile-type
   processing of -i4/-noi4 options in OPTIONS statement.
 */
void
implicit_int(DTYPE default_int)
{
  int i;
  for (i = 8; i <= 13; i++) {
    dtimplicit[i].dtype = dtimplicit[i + 26].dtype = default_int;
  }
}

/**
   \brief Enter symbol with indicated name into symbol table,
   initialize the new entry, and return pointer to it.  If there is
   already such a symbol, just return pointer to the existing symbol
   table entry.

   \param name is the symbol name.

 */
SPTR
getsymbol(const char *name)
{
  return getsym(name, strlen(name));
}

/**
   \brief Like getsymbol, but accepts a string that is *not*
   null-terminated.

   \param name is the symbol name.
   \param olength is the number of characters in the symbol name.
 */
SPTR
getsym(const char *name, int olength)
{
  SPTR sptr; /* pointer to symbol table entry */

  sptr = installsym(name, olength);
  if (STYPEG(sptr) == ST_UNKNOWN)
    setimplicit(sptr);
  return sptr;
}

/* FIXME: getcon & get_acon are identical between C and Fortran;
   should be shared */

/**
   \brief Enter constant of given dtype and value into the symbol
   table and return pointer to it.  If an entry for the constant
   already exists, return pointer to the existing entry instead.

   \param value is the constant value (value[1] if 1 word).
   \param dtype - tbw.
 */
SPTR
getcon(INT *value, DTYPE dtype)
{
  SPTR sptr;   /* symbol table pointer */
  int hashval; /* index into hashtb */

  /*
   * First loop thru the appropriate hash link list to see if this constant
   * is already in the symbol table:
   */

  hashval = HASH_CON(value);
  if (hashval < 0)
    hashval = -hashval;
  for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
    if (DTY(dtype) == TY_128
#ifdef TARGET_SUPPORTS_QUADFP
        || DTY(dtype) == TY_QUAD
#endif
       ) {
      if (DTYPEG(sptr) != dtype || STYPEG(sptr) != ST_CONST ||
          CONVAL1G(sptr) != value[0] || CONVAL2G(sptr) != value[1] ||
          CONVAL3G(sptr) != value[2] || CONVAL4G(sptr) != value[3])
        continue;

      /* Matching entry has been found.  Return it:  */
      return sptr;
    }
    if (DTYPEG(sptr) != dtype || STYPEG(sptr) != ST_CONST ||
        CONVAL1G(sptr) != value[0] || CONVAL2G(sptr) != value[1])
      continue;

    /* Matching entry has been found.  Return it:  */

    return sptr;
  }

  /* Constant not found.  Create a new symbol table entry for it: */

  ADDSYM(sptr, hashval);
  CONVAL1P(sptr, value[0]);
  CONVAL2P(sptr, value[1]);
  if (DTY(dtype) == TY_128
#ifdef TARGET_SUPPORTS_QUADFP
      || DTY(dtype) == TY_QUAD
#endif
     ) {
    CONVAL3P(sptr, value[2]);
    CONVAL4P(sptr, value[3]);
  }
  STYPEP(sptr, ST_CONST);
  DTYPEP(sptr, dtype);

  return sptr;
}

SPTR
get_acon(SPTR sym, ISZ_T off)
{
  return get_acon3(sym, off, DT_CPTR);
}

/*
 * BIGOBJects are supported, need an acon-specific getcon
 */
SPTR
get_acon3(SPTR sym, ISZ_T off, DTYPE dtype)
{
  INT value[2];
  SPTR sptr;   /* symbol table pointer */
  int hashval; /* index into stb.hashtb */

  /*
   * First loop thru the appropriate hash link list to see if this constant
   * is already in the symbol table:
   */

  bgitoi64(off, value);
  value[0] = sym;
  hashval = HASH_CON(value);
  if (hashval < 0)
    hashval = -hashval;
  for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
    if (DTYPEG(sptr) != dtype || STYPEG(sptr) != ST_CONST ||
        CONVAL1G(sptr) != sym || ACONOFFG(sptr) != off)
      continue;

    /* Matching entry has been found.  Return it:  */

    return sptr;
  }

  /* Constant not found.  Create a new symbol table entry for it: */

  ADDSYM(sptr, hashval);
  CONVAL1P(sptr, sym);
  ACONOFFP(sptr, off);
  STYPEP(sptr, ST_CONST);
  DTYPEP(sptr, dtype);

  return sptr;
}

SPTR
get_vcon(INT *value, DTYPE dtype)
{
  SPTR sptr;   /* symbol table pointer */
  int hashval; /* index into stb.hashtb */
  int i, n;
  int vc;
  /*
   * First loop thru the appropriate hash link list to see if this constant
   * is already in the symbol table:
   */
  hashval = HASH_CON((&stb.dt.stg_base[dtype]));
  if (hashval < 0)
    hashval = -hashval;
  n = DTyVecLength(dtype);
  for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
    if (DTYPEG(sptr) != dtype || STYPEG(sptr) != ST_CONST)
      continue;
    vc = CONVAL1G(sptr);
    for (i = 0; i < n; i++)
      if (VCON_CONVAL(vc + i) != value[i])
        goto cont;
    /* Matching entry has been found.  Return it:  */
    return sptr;
  cont:;
  }

  /* Constant not found.  Create a new symbol table entry for it: */

  ADDSYM(sptr, hashval);

  vc = aux.vcon_avl;
  /*
   * Always add a 4th element to a  3-element vector constant
   */
  if (n != 3)
    aux.vcon_avl += n;
  else
    aux.vcon_avl += 4;
  NEED(aux.vcon_avl, aux.vcon_base, INT, aux.vcon_size, aux.vcon_size + 64);
  for (i = 0; i < n; i++)
    VCON_CONVAL(vc + i) = value[i];
  if (n == 3) {
    VCON_CONVAL(vc + 3) = 0;
  }
  CONVAL1P(sptr, vc);
  STYPEP(sptr, ST_CONST);
  DTYPEP(sptr, dtype);

  return sptr;
}

static int vc0[TY_MAX + 1][TY_VECT_MAXLEN];
static int vc1[TY_MAX + 1][TY_VECT_MAXLEN];
static int vcm0[TY_MAX + 1][TY_VECT_MAXLEN];
static int fltm0;
static int dblm0;

/* need to clear it per function */
static void
clear_vc(void)
{
  int arrsize = (TY_MAX + 1) * TY_VECT_MAXLEN;
  BZERO(vc0, int, arrsize);
  BZERO(vc1, int, arrsize);
  BZERO(vcm0, int, arrsize);
}

/** \brief Get a vector constant of a zero which suits the element type.
 */
int
get_vcon0(DTYPE dtype)
{
  int i, n, ty;
  INT zero;
  INT v[TY_VECT_MAXLEN];

  n = DTyVecLength(dtype);
#if DEBUG
  assert(sizeof(v) % sizeof(INT) <= (size_t)n, "get_vcon0 v[] not large enough",
         __LINE__, ERR_Severe);
#endif
  ty = DTY(DTySeqTyElement(dtype));
  if (vc0[ty][n - 1])
    return vc0[ty][n - 1];
  switch (ty) {
  case TY_INT8:
  case TY_LOG8:
    zero = stb.k0;
    break;
  case TY_FLOAT:
    zero = CONVAL2G(stb.flt0);
    break;
  case TY_DBLE:
    zero = stb.dbl0;
    break;
  default:
    zero = 0;
    break;
  }
  for (i = 0; i < n; i++)
    v[i] = zero;
  vc0[ty][n - 1] = get_vcon(v, dtype);
  return vc0[ty][n - 1];
}

/*
 * get a vector constant of a one which suits the element type.
 */
int
get_vcon1(DTYPE dtype)
{
  int i, n, ty;
  INT one, v[TY_VECT_MAXLEN];

  n = DTyVecLength(dtype);
#if DEBUG
  assert(sizeof(v) % sizeof(INT) <= (size_t)n, "get_vcon1 v[] not large enough",
         __LINE__, ERR_Severe);
#endif
  ty = DTY(DTySeqTyElement(dtype));
  if (vc1[ty][n - 1])
    return vc1[ty][n - 1];
  switch (ty) {
  case TY_INT8:
  case TY_LOG8:
    one = stb.k1;
    break;
  case TY_FLOAT:
    one = CONVAL2G(stb.flt1);
    break;
  case TY_DBLE:
    one = stb.dbl1;
    break;
  default:
    one = 1;
    break;
  }
  for (i = 0; i < n; i++)
    v[i] = one;
  vc1[ty][n - 1] = get_vcon(v, dtype);
  return vc1[ty][n - 1];
}

int
get_vconm0(DTYPE dtype)
{
  int i, n, ty;
  INT val[2], zero;
  INT v[TY_VECT_MAXLEN];

  n = DTyVecLength(dtype);
#if DEBUG
  assert(sizeof(v) % sizeof(INT) <= (size_t)n, "get_vconm0 v[] not large enough",
         __LINE__, ERR_Severe);
#endif
  ty = DTY(DTySeqTyElement(dtype));
  if (vcm0[ty][n - 1])
    return vcm0[ty][n - 1];
  switch (ty) {
  case TY_FLOAT:
    if (fltm0)
      zero = CONVAL2G(fltm0);
    else {
      val[0] = 0;
      val[1] = CONVAL2G(stb.flt0) | 0x80000000;
      fltm0 = getcon(val, DT_FLOAT);
      zero = val[1];
    }
    break;
  case TY_DBLE:
    if (!dblm0) {
      val[0] = CONVAL1G(stb.dbl0) | 0x80000000;
      val[1] = CONVAL2G(stb.dbl0);
      dblm0 = getcon(val, DT_DBLE);
    }
    zero = dblm0;
    break;
  default:
    vcm0[ty][n - 1] = get_vcon0(dtype);
    return vcm0[ty][n - 1];
  }
  for (i = 0; i < n; i++)
    v[i] = zero;
  vcm0[ty][n - 1] = (int)get_vcon(v, dtype); // ???
  return vcm0[ty][n - 1];
}

/*
 * get a vector constant by expanding a scalar
 */
SPTR
get_vcon_scalar(INT sclr, DTYPE dtype)
{
  int i, n;
  INT v[TY_VECT_MAXLEN];

  n = DTyVecLength(dtype);
#if DEBUG
  assert(sizeof(v) % sizeof(INT) <= (size_t)n, "get_vcon_scalar v[] not large enough",
         __LINE__, ERR_Severe);
#endif
  for (i = 0; i < n; i++)
    v[i] = sclr;
  return get_vcon(v, dtype);
}

ISZ_T
get_isz_cval(int con)
{
  INT num[2];
  ISZ_T v;
#if DEBUG
  assert(STYPEG(con) == ST_CONST, "get_isz_cval-not ST_CONST", con, ERR_unused);
  assert(DT_ISINT(DTYPEG(con)), "get_isz_cval-not 64-bit int const", con,
         ERR_unused);
#endif
  num[1] = CONVAL2G(con);
  if (size_of(DTYPEG(con)) > 4)
    num[0] = CONVAL1G(con);
  else if (num[1] < 0)
    num[0] = -1;
  else
    num[0] = 0;
  INT64_2_ISZ(num, v);
  return v;
}

/**
   \brief Sign extend an integer value of an indicated width (8, 16,
   32); value returned is sign extended with respect to the host's int
   type.
 */
INT
sign_extend(INT val, int width)
{
  /* 32-bit INT */
  int w;

  if (width == 32)
    return val;
  w = 32 - width;
  return ARSHIFT(LSHIFT(val, w), w);
}

SPTR
getstring(const char *value, int length)
{
  SPTR sptr;   /* symbol table pointer */
  int hashval; /* index into hashtb */
  char *np;    /* pointer to string characters */
  const char *p;
  int i;

  /*
   * first loop thru the appropriate hash link list to see if symbol is
   * already in the table:
   */
  HASH_STR(hashval, value, length);
  /* Ensure hash value is positive.  '\nnn' can cause negative hash values */
  if (hashval < 0)
    hashval = -hashval;
  for (sptr = stb.hashtb[hashval]; sptr != SPTR_NULL; sptr = HASHLKG(sptr)) {
    DTYPE dtype_;
    if (STYPEG(sptr) != ST_CONST)
      continue;

    i = dtype_ = DTYPEG(sptr);
    if (DTY(dtype_) == TY_CHAR && DTyCharLength(dtype_) == length) {
      /* now match the characters in the strings: */
      np = stb.n_base + CONVAL1G(sptr);
      p = value;
      i = length;
      while (i--)
        if (*np++ != *p++)
          goto Continue;

      /* Matching entry has been found in symtab.  Return it:  */
      return sptr;
    }
  Continue:;
  }

  /* String not found.  Create a new symtab entry for it:  */
  ADDSYM(sptr, hashval);
  CONVAL1P(sptr, putsname(value, length));
  STYPEP(sptr, ST_CONST);
  DTYPEP(sptr, get_type(2, TY_CHAR, length));

  return sptr;
}

SPTR
getntstring(const char *value)
{
  int len_string = strlen(value);

  if (len_string)
    len_string++;

  return getstring(value, len_string);
}

SPTR
getstringaddr(SPTR sptr)
{
  SPTR sptrx;

  for (sptrx = stb.firstusym; sptrx < stb.stg_avail; ++sptrx) {
    if ((STYPEG(sptrx) == ST_CONST) && (DTYPEG(sptr) == DT_ADDR) &&
        (CONVAL1G(sptrx) == sptr))
      return sptrx; /* found */
  }

  /* String not found.  Create a new symtab entry */
  NEWSYM(sptrx); /* can I use get_con here? */
  CONVAL1P(sptrx, sptr);
  STYPEP(sptrx, ST_CONST);
  DTYPEP(sptrx, DT_ADDR);

  return sptrx;
}

void
newimplicit(int firstc, int lastc, DTYPE dtype)
{
  int i, j; /* indices into implicit arrays */
  char temp[2];

  i = IMPL_INDEX(firstc);
  j = IMPL_INDEX(lastc);
  assert(i >= 0 & j >= 0 & i < 54 & j < 54, "newimplicit: bad impl range", i,
         ERR_Fatal);

  for (; i <= j; i++) {
    if (dtimplicit[i].set) {
      temp[0] = 'a' + i;
      temp[1] = 0;
      if (dtype == dtimplicit[i].dtype)
        error((error_code_t)54, ERR_Warning, gbl.lineno, temp, CNULL);
      else
        error((error_code_t)54, ERR_Severe, gbl.lineno, temp, CNULL);
    }
    dtimplicit[i].dtype = dtype;
    dtimplicit[i].set = true;
  }
}

/**
   \brief Assign to the indicated symbol table entry, the current
   implicit dtype.
 */
void
setimplicit(int sptr)
{
  int firstc; /* first character of symbol name */
  int i;      /* index into implicit tables defined by the
               * first character of the name of sptr.  */

  firstc = *SYMNAME(sptr);

  /*
   * determine index into implicit array.  Note that the value returned
   * will be -1 if this routine is being called from within the symini
   * utility for a symbol beginning with ".".
   */

  i = IMPL_INDEX(firstc);
  if (i != -1) {
    DTYPEP(sptr, dtimplicit[i].dtype);
  }
}

/**
   \brief Scan backwards in the symbol table to reapply the current
   implicit state to variables which have not been typed.  Invoked
   when an implicit statement follows specification statements
   (usually a severe error) and the option -x 125 0x80 is specified.
 */
void
reapply_implicit(void)
{
  int sptr;
  int firstc; /* first character of symbol name */
  int i;      /* index into implicit tables defined by the
               * first character of the name of sptr.  */

  for (sptr = stb.stg_avail - 1; sptr >= stb.firstusym; sptr--) {
    if (CCSYMG(sptr))
      continue;
    switch (STYPEG(sptr)) {
    case ST_VAR:
    case ST_PROC:
      if (!DCLDG(sptr)) {
        firstc = *SYMNAME(sptr);
        i = IMPL_INDEX(firstc);
        DTYPEP(sptr, dtimplicit[i].dtype);
      }
      break;
    case ST_ARRAY:
      if (!DCLDG(sptr)) {
        /* WARNING: believe it's safe to overwrite the dtype
         * in the dtype record; if not, need to 'duplicate' the
         * array dtype record.
         */
        firstc = *SYMNAME(sptr);
        i = IMPL_INDEX(firstc);
        DTySetFst(DTYPEG(sptr), dtimplicit[i].dtype);
      }
      break;
    default:
      break;
    }
  }
}

/** \brief Return ptr to printable representation of the indicated PARAMETER.
 *
 * \param sptr - symbol table pointer
 */
const char *
parmprint(int sptr)
{
  DTYPE dtype;
  char *buf;

  if (STYPEG(sptr) != ST_PARAM)
    return "";
  /*
   * Change the symbol table entry to an ST_CONST use getprint
   * to get the character representation.
   */
  STYPEP(sptr, ST_CONST);
  dtype = DTYPEG(sptr);
  if (DTY(dtype) == TY_SINT || DTY(dtype) == TY_BINT || DTY(dtype) == TY_HOLL ||
      DTY(dtype) == TY_WORD)
    DTYPEP(sptr, DT_INT);
  else if (DTY(dtype) == DT_SLOG || DTY(dtype) == DT_BLOG)
    DTYPEP(sptr, DT_LOG);
  if (TY_ISWORD(DTY(dtype))) {
    CONVAL2P(sptr, CONVAL1G(sptr));
    buf = getprint(sptr);
    CONVAL2P(sptr, 0);
  } else
    buf = getprint((int)CONVAL1G(sptr));
  STYPEP(sptr, ST_PARAM);
  DTYPEP(sptr, dtype);
  return buf;
}

/*---------------------------------------------------------------------*
 * getprint cannot be shared between FORTRAN and C                     *
 *---------------------------------------------------------------------*/

/** \brief Return ptr to printable representation of the indicated symbol.
 *
 * For symbols which are not constants, the name of the symbol is used.
 * Constants are converted into the appropriate character representation.
 *
 * \param sptr - symbol table pointer
 */
char *
getprint(int sptr)
{
  int len; /* length of character string */
  static char *b = NULL;
  char *from, *end, *to;
  int c;
  INT num[2];
  DTYPE dtype;

  if (STYPEG(sptr) != ST_CONST) {
    from = SYMNAME(sptr);
    if (*from == '\0') {
      static char bf[16];
      sprintf(bf, ".%d.", sptr);
      return bf;
    }
    return SYMNAME(sptr);
  }

  if (b == NULL) {
    NEW(b, char, 100);
  }
  dtype = DTYPEG(sptr);
  switch (DTY(dtype)) {
  case TY_WORD:
    sprintf(b, "%08X", CONVAL2G(sptr));
    break;
  case TY_DWORD:
    sprintf(b, "%08X%08X", CONVAL1G(sptr), CONVAL2G(sptr));
    break;
  case TY_INT8:
  case TY_LOG8:
    num[0] = CONVAL1G(sptr);
    num[1] = CONVAL2G(sptr);
    ui64toax(num, b, 22, 0, 10);
    break;
  case TY_INT:
  case TY_LOG:
    sprintf(b, "%d", CONVAL2G(sptr));
    break;
  case TY_REAL:
    num[0] = CONVAL2G(sptr);
    cprintf(b, "%17.10e", (INT *)((BIGINT)num[0]));
    break;

  case TY_DBLE:
    num[0] = CONVAL1G(sptr);
    num[1] = CONVAL2G(sptr);
    cprintf(b, "%24.17le", num);
    break;

  case TY_CMPLX:
    num[0] = CONVAL1G(sptr);
    cprintf(b, "%17.10e", (INT *)((BIGINT)num[0]));
    b[17] = ',';
    b[18] = ' ';
    num[0] = CONVAL2G(sptr);
    cprintf(&b[19], "%17.10e", (INT *)((BIGINT)num[0]));
    break;

  case TY_DCMPLX:
    num[0] = CONVAL1G(CONVAL1G(sptr));
    num[1] = CONVAL2G(CONVAL1G(sptr));
    cprintf(b, "%24.17le", num);
    b[24] = ',';
    b[25] = ' ';
    num[0] = CONVAL1G(CONVAL2G(sptr));
    num[1] = CONVAL2G(CONVAL2G(sptr));
    cprintf(&b[26], "%24.17le", num);
    break;

  case TY_NCHAR:
    sptr = CONVAL1G(sptr); /* sptr to char string constant */
    dtype = DTYPEG(sptr);
    FLANG_FALLTHROUGH;
  case TY_HOLL: /* Should be no holleriths in symbol table */
  case TY_CHAR:
    from = stb.n_base + CONVAL1G(sptr);
    len = DTyCharLength(dtype);
    end = b + 93;
    *b = '\"';
    for (to = b + 1; len-- && to < end;) {
      c = *from++ & 0xff;
      if (c == '\"' || c == '\'' || c == '\\') {
        *to++ = '\\';
        *to++ = c;
      } else if (c >= ' ' && c <= '~') {
        *to++ = c;
      } else if (c == '\n') {
        *to++ = '\\';
        *to++ = 'n';
      }
      else {
        *to++ = '\\';
        /* Mask off 8 bits worth of unprintable character */
        sprintf(to, "%03o", (c & 255));
        to += 3;
      }
    }
    *to++ = '\"';
    *to = '\0';
    break;

  case TY_128:
    sprintf(b, "%08x %08x %08x %08x", CONVAL1G(sptr), CONVAL2G(sptr),
            CONVAL3G(sptr), CONVAL4G(sptr));
    break;

  case TY_PTR:
    strcpy(b, "address constant");
    break;

  case TY_VECT:
    strcpy(b, "vector constant");
    break;

  default:
    interr("getprint:bad const dtype", sptr, ERR_Informational);
  }
  return b;
}

/*
 * dump symbol table information for symbol sptr.
 */
static void putaltname(FILE *, int, const char *);
static void putcuda(FILE *, int);

#undef _PFG
#define _PFG(cond, str) \
  if (cond)             \
  fprintf(dfil, "  %s", str)

#if DEBUG
/**
   \param file the file.
   \param sptr symbol currently being dumped.
 */
void
symdentry(FILE *file, int sptr)
{
  FILE *dfil;
  int dscptr;      /* ptr to dummy parameter descriptor list */
  char buff[110];  /* text buffer used to create output lines */
  char typeb[110]; /* buffer for text of dtype */
  int stype;       /* symbol type of sptr  */
  DTYPE dtype;     /* data type of sptr */
  int i;

  dfil = file ? file : stderr;
  strcpy(buff, getprint(sptr));
  stype = STYPEG(sptr);
  dtype = DTYPEG(sptr);

  /* write first line containing symbol name, dtype, and stype: */

  if (stype == ST_CMBLK || stype == ST_LABEL || stype == ST_GENERIC ||
      stype == ST_NML || stype == ST_PD)
    fprintf(dfil, "\n%-40.40s %s\n", buff, stb.stypes[stype]);
  else {
    *typeb = '\0';
    getdtype(dtype, typeb);
    fprintf(dfil, "\n%-40.40s %s %s\n", buff, typeb, stb.stypes[stype]);
  }

  /* write second line:  */

  fprintf(dfil, "sptr: %d  hashlk: %d  nmptr: %d  dtype: %d\n", sptr,
          HASHLKG(sptr), NMPTRG(sptr), dtype);

  switch (stype) {
  case ST_UNKNOWN:
  case ST_IDENT:
  case ST_VAR:
  case ST_ARRAY:
  case ST_STRUCT:
  case ST_UNION:
    fprintf(dfil, "dcld: %d  ccsym: %d  save: %d  ref: %d  dinit: %d  vol: %d",
            DCLDG(sptr), CCSYMG(sptr), SAVEG(sptr), REFG(sptr), DINITG(sptr),
            VOLG(sptr));
    fprintf(dfil, "  scope: %d  enclfunc: %d\n", SCOPEG(sptr), ENCLFUNCG(sptr));
    fprintf(dfil, "address: %" ISZ_PF "d  sc:%d(%s)  symlk: %d  midnum: %d",
            ADDRESSG(sptr), SCG(sptr),
            (SCG(sptr) <= SC_MAX) ? stb.scnames[SCG(sptr)] : "na", SYMLKG(sptr),
            MIDNUMG(sptr));
    if (CLENG(sptr))
      fprintf(dfil, "  clen: %d", CLENG(sptr));
    if (SOCPTRG(sptr))
      fprintf(dfil, "  socptr: %d", SOCPTRG(sptr));
#ifdef BASESYMG
    if (BASESYMG(sptr))
      fprintf(dfil, "  basesym: %d", BASESYMG(sptr));
#endif
    fprintf(dfil, "\n");
    fprintf(dfil, "addrtkn: %d", ADDRTKNG(sptr));
    _PFG(REGARGG(sptr), "regarg");
    _PFG(MEMARGG(sptr), "memarg");
    _PFG(COPYPRMSG(sptr), "copyprms");
    _PFG(ALLOCG(sptr), "alloc");
    _PFG(ASSNG(sptr), "assn");
    _PFG(THREADG(sptr), "thread");
    _PFG(QALNG(sptr), "qaln");
    _PFG(PASSBYVALG(sptr), "passbyval");
    _PFG(PASSBYREFG(sptr), "passbyref");
    _PFG(STDCALLG(sptr), "stdcall");
    _PFG(CFUNCG(sptr), "cfunc");
#ifdef CONTIGATTRG
    _PFG(CONTIGATTRG(sptr), "contigattr");
#endif
#ifdef TASKG
    _PFG(TASKG(sptr), "task");
#endif
#ifdef PARREFG
    _PFG(PARREFG(sptr), "parref");
#endif
#if defined(TARGET_WIN_X86)
    if (DLLG(sptr) == DLL_EXPORT)
      fprintf(dfil, "  dllexport");
    else if (DLLG(sptr) == DLL_IMPORT)
      fprintf(dfil, "  dllimport");
#endif
#ifdef INLNG
    _PFG(INLNG(sptr), "inln");
    if (INLNG(sptr) && SCG(sptr) == SC_BASED) {
      _PFG(UNSAFEG(sptr), "unsafe");
    }
#endif
    /*if (SCG(sptr) == SC_BASED)*/
    fprintf(dfil, "  noconflict:%d", NOCONFLICTG(sptr));
    if (stype == ST_ARRAY ||
        ((stype == ST_STRUCT || stype == ST_UNION) && dtype > 0 &&
         dtype < stb.dt.stg_avail && DTY(dtype) == TY_ARRAY)) {
      fprintf(dfil, " asumsz:%d adjarr:%d aftent:%d", (int)ASUMSZG(sptr),
              (int)ADJARRG(sptr), (int)AFTENTG(sptr));
      /* for fortran-90 */
      fprintf(dfil, " assumshp:%d", ASSUMSHPG(sptr));
      fprintf(dfil, " sdsc:%d origdim:%d sdscs1:%d", (int)SDSCG(sptr),
              (int)ORIGDIMG(sptr), SDSCS1G(sptr));
    }
    /* for fortran-90 */
    fprintf(dfil, "\n");
    fprintf(dfil, "pointer: %d", (int)POINTERG(sptr));
    fprintf(dfil, "  uplevel: %d", (int)UPLEVELG(sptr));
    fprintf(dfil, "  internref: %d", (int)INTERNREFG(sptr));
    fprintf(dfil, "  gscope: %d", (int)GSCOPEG(sptr));
    fprintf(dfil, "  origdummy: %d", (int)ORIGDUMMYG(sptr));
    _PFG(LSCOPEG(sptr), "lscope");
    _PFG(PTRSAFEG(sptr), "ptrsafe");
    _PFG(ALLOCATTRG(sptr), "allocattr");
    _PFG(F90POINTERG(sptr), "f90pointer");
    _PFG(REREFG(sptr), "reref");
    if (stype == ST_ARRAY) {
      _PFG(DESCARRAYG(sptr), "descarray");
    }
    if (SCG(sptr) == SC_DUMMY) {
      _PFG(OPTARGG(sptr), "optarg");
      _PFG(INTENTING(sptr), "intentin");
    }
    if (SCG(sptr) == SC_DUMMY) {
      _PFG(UNSAFEG(sptr), "unsafe");
      _PFG(HOMEDG(sptr), "homed");
    }
    fprintf(dfil, "\n");
    putaltname(dfil, sptr, "");
    if (SCG(sptr) != SC_DUMMY && SOCPTRG(sptr))
      dmp_socs(sptr, dfil);
    break;

  case ST_STAG:
  case ST_TYPEDEF:
    fprintf(dfil, "dcld: %d\n", DCLDG(sptr));
    _PFG(UNLPOLYG(sptr), "unlpoly");
    break;

  case ST_NML:
    fprintf(dfil,
            "symlk: %d   address: %" ISZ_PF
            "d   cmemf: %d   cmeml: %d   ref: %d\n",
            SYMLKG(sptr), ADDRESSG(sptr), CMEMFG(sptr), (int)CMEMLG(sptr),
            REFG(sptr));
    for (i = CMEMFG(sptr); i; i = NML_NEXT(i))
      fprintf(dfil, "    nml:%5d   sptr:%5d   %s\n", i, (int)NML_SPTR(i),
              SYMNAME(NML_SPTR(i)));
    break;

  case ST_MEMBER:
    fprintf(dfil, "address: %" ISZ_PF "d   symlk: %d   variant: %d   ccsym: %d",
            ADDRESSG(sptr), SYMLKG(sptr), VARIANTG(sptr), (int)CCSYMG(sptr));
    fprintf(dfil, " pointer: %d", (int)POINTERG(sptr));
    _PFG(LSCOPEG(sptr), "lscope");
    _PFG(PTRSAFEG(sptr), "ptrsafe");
#ifdef CONTIGATTRG
    _PFG(CONTIGATTRG(sptr), "contigattr");
#endif
    _PFG(CLASSG(sptr), "class");
    if (DTY(dtype) == TY_ARRAY) {
      fprintf(dfil, " sdscs1:%d", SDSCS1G(sptr));
    }
    fprintf(dfil, " vtable:%d", VTABLEG(sptr));
    fprintf(dfil, " iface:%d", IFACEG(sptr));
    fprintf(dfil, " tbplnk:%d", TBPLNKG(sptr));
    fprintf(dfil, "\n");
    break;

  case ST_CMBLK:
    fprintf(dfil,
            "save: %d   dinit: %d   size: %" ISZ_PF
            "d   vol:%d   alloc:%d   ccsym:%d",
            SAVEG(sptr), DINITG(sptr), SIZEG(sptr), VOLG(sptr), ALLOCG(sptr),
            CCSYMG(sptr));
    fprintf(dfil, "\n");
    fprintf(dfil, "  scope: %d  enclfunc: %d", SCOPEG(sptr), ENCLFUNCG(sptr));
#ifdef PDALNG
    fprintf(dfil, "  pdaln: %d", PDALNG(sptr));
#endif
    fprintf(dfil, "\n");
    fprintf(dfil, "midnum: %d   symlk: %d   cmemf: %d   cmeml: %d\n",
            MIDNUMG(sptr), SYMLKG(sptr), CMEMFG(sptr), (int)CMEMLG(sptr));
    putaltname(dfil, sptr, "");
    _PFG(THREADG(sptr), "thread");
    _PFG(QALNG(sptr), "qaln");
    _PFG(CFUNCG(sptr), "cfunc");
    _PFG(STDCALLG(sptr), "stdcall");
    _PFG(FROMMODG(sptr), "frommod");
    _PFG(MODCMNG(sptr), "modcmn");
#ifdef TLSG
    _PFG(TLSG(sptr), "tls");
#endif /* TLSG */
#ifdef USE_MPC
    if (ETLSG(sptr))
      fprintf(file, " etls: %d", ETLSG(sptr));
#endif /* USE_MPC */
#if defined(TARGET_WIN_X86)
    if (DLLG(sptr) == DLL_EXPORT)
      fprintf(dfil, "  dllexport");
    else if (DLLG(sptr) == DLL_IMPORT)
      fprintf(dfil, "  dllimport");
#endif
    fprintf(dfil, "\n");
    break;

  case ST_ENTRY:
    fprintf(dfil, "dcld: %d  ccsym: %d   address: %" ISZ_PF "d   midnum: %d   ",
            DCLDG(sptr), CCSYMG(sptr), ADDRESSG(sptr), MIDNUMG(sptr));
    fprintf(dfil, "symlk: %d   paramct: %d   dpdsc: %d\n", SYMLKG(sptr),
            PARAMCTG(sptr), DPDSCG(sptr));
    fprintf(dfil, "funcline: %d   copyprms: %d   bihnum: %d   fval: %d",
            (int)FUNCLINEG(sptr), COPYPRMSG(sptr), BIHNUMG(sptr), FVALG(sptr));
    fprintf(dfil, "   adjarr: %d   aftent: %d", ADJARRG(sptr), AFTENTG(sptr));
    _PFG(CONTAINEDG(sptr), "contained");
    fprintf(dfil, "\n");
    putaltname(dfil, sptr, "");
    _PFG(CFUNCG(sptr), "cfunc");
#ifdef CSTRUCTRETG
    _PFG(CSTRUCTRETG(sptr), "cstructret");
#endif
    _PFG(MSCALLG(sptr), "mscall");
    _PFG(CREFG(sptr), "cref");
    _PFG(NOMIXEDSTRLENG(sptr), "nomixedstrlen");
    _PFG(PASSBYVALG(sptr), "passbyval");
    _PFG(PASSBYREFG(sptr), "passbyref");
    _PFG(STDCALLG(sptr), "stdcall");
    _PFG(DECORATEG(sptr), "decorate");
#if defined(TARGET_WIN_X86)
    if (DLLG(sptr) == DLL_EXPORT)
      fprintf(dfil, "  dllexport");
    else if (DLLG(sptr) == DLL_IMPORT)
      fprintf(dfil, "  dllimport");
#endif
    if (WINNT_CALL)
      fprintf(dfil, " argsize:%d", ARGSIZEG(sptr));
    _PFG(ARETG(sptr), "aret");
    fprintf(dfil, "\n");
    putcuda(dfil, sptr);
    fprintf(dfil, "Parameter sptr's:\n");
    dscptr = DPDSCG(sptr);
    for (i = PARAMCTG(sptr); i > 0; dscptr++, i--)
      fprintf(dfil, "sptr = %d\n", *(aux.dpdsc_base + dscptr));
    break;

  case ST_PROC:
    fprintf(dfil, "dcld: %d   ref: %d  ccsym: %d  func: %d  midnum: %d   ",
            DCLDG(sptr), REFG(sptr), CCSYMG(sptr), FUNCG(sptr), MIDNUMG(sptr));
    fprintf(dfil, "sc:%d(%s)  symlk: %d", SCG(sptr),
            (SCG(sptr) <= SC_MAX) ? stb.scnames[SCG(sptr)] : "na",
            SYMLKG(sptr));
    fprintf(dfil, "\n");
    fprintf(dfil, "paramct: %d  dpdsc: %d  fval: %d", PARAMCTG(sptr),
            DPDSCG(sptr), FVALG(sptr));
    fprintf(dfil, "\n");
    if (SCG(sptr) == SC_DUMMY) {
      fprintf(dfil, "address: %" ISZ_PF "d", ADDRESSG(sptr));
      _PFG(UNSAFEG(sptr), "unsafe");
      fprintf(dfil, "  uplevel: %d", (int)UPLEVELG(sptr));
      fprintf(dfil, "  internref: %d", (int)INTERNREFG(sptr));
      fprintf(dfil, "  gscope: %d", (int)GSCOPEG(sptr));
    }
    _PFG(CONTAINEDG(sptr), "contained");
    _PFG(NEEDMODG(sptr), "needmod");
    _PFG(TYPDG(sptr), "typd");
    putaltname(dfil, sptr, "  ");
    _PFG(CFUNCG(sptr), "cfunc");
#ifdef CSTRUCTRETG
    _PFG(CSTRUCTRETG(sptr), "cstructret");
#endif
    _PFG(MSCALLG(sptr), "mscall");
    _PFG(CREFG(sptr), "cref");
    _PFG(NOMIXEDSTRLENG(sptr), "nomixedstrlen");
    _PFG(PASSBYVALG(sptr), "passbyval");
    _PFG(PASSBYREFG(sptr), "passbyref");
    _PFG(STDCALLG(sptr), "stdcall");
    _PFG(DECORATEG(sptr), "decorate");
#if defined(TARGET_WIN_X86)
    if (DLLG(sptr) == DLL_EXPORT)
      fprintf(dfil, "  dllexport");
    else if (DLLG(sptr) == DLL_IMPORT)
      fprintf(dfil, "  dllimport");
#endif
    _PFG(CNCALLG(sptr), "cncall");
#ifdef NOPADG
    _PFG(NOPADG(sptr), "nopad");
#endif
#ifdef ARG1PTRG
    _PFG(ARG1PTRG(sptr), "arg1ptr");
#endif
    _PFG(XMMSAFEG(sptr), "xmmsafe");
    if (WINNT_CALL)
      fprintf(dfil, " argsize:%d", ARGSIZEG(sptr));
    _PFG(ARETG(sptr), "aret");
    _PFG(VARARGG(sptr), "vararg");
    fprintf(dfil, "\n");
    putcuda(dfil, sptr);
    break;

  case ST_CONST:
    fprintf(dfil, "holl: %d   ", HOLLG(sptr));
    fprintf(dfil,
            "symlk: %d   address: %" ISZ_PF "d   conval1: %d   conval2: %d\n",
            SYMLKG(sptr), ADDRESSG(sptr), CONVAL1G(sptr), CONVAL2G(sptr));
    if (DTY(dtype) == TY_VECT) {
        int vc, n;
        vc = CONVAL1G(sptr);
        n = DTyVecLength(dtype);
        fprintf(dfil, "    vcon_base[%d]:\n", vc);
        for (i = 0; i < n; i += 4) {
            const char *f1, *f2;
            switch (DTySeqTyElement(dtype)) {
            case DT_FLOAT:
              f1 = "        %08x %08x";
              f2 = " %08x";
              break;
            case DT_DBLE:
            case DT_INT8:
              f1 = "        %8d %8d";
              f2 = " %8d";
              break;
            default:
              f1 = "        %08x %08x";
              f2 = " %08x";
         }
         fprintf(dfil, f1, VCON_CONVAL(vc + i), VCON_CONVAL(vc + i + 1));
         if (n > 2) {
             fprintf(dfil, f2, VCON_CONVAL(vc + i + 2));
             if (n != 3)
                 fprintf(dfil, f2, VCON_CONVAL(vc + i + 3));
         }
         fprintf(dfil, "\n");
      }
    }
    break;

  case ST_LABEL:
    fprintf(dfil,
            "rfcnt: %d  address: %" ISZ_PF
            "d  symlk: %d  iliblk: %d  fmtpt: %d  vol: %d\n",
            RFCNTG(sptr), ADDRESSG(sptr), SYMLKG(sptr), ILIBLKG(sptr),
            FMTPTG(sptr), VOLG(sptr));
    if (BEGINSCOPEG(sptr))
      fprintf(file, "beginscope ");
    if (ENDSCOPEG(sptr))
      fprintf(file, "endscope ");
    fprintf(file, "  in func: %d\n", ENCLFUNCG(sptr));
    break;

  case ST_STFUNC:
    fprintf(dfil, "sfdsc: %x   excvlen: %d\n", SFDSCG(sptr),
            DTyCharLength(DTYPEG(sptr)));
    break;

  case ST_PARAM:
    if (TY_ISWORD(DTY(dtype))) {
      fprintf(dfil, "conval1: 0x%x  (%s)\n", CONVAL1G(sptr), parmprint(sptr));
    } else {
      fprintf(dfil, "conval1: %d (sptr)\n", CONVAL1G(sptr));
    }
    break;

  case ST_INTRIN:
    fprintf(dfil, "dcld: %d   expst: %d\n", (int)DCLDG(sptr),
            (int)EXPSTG(sptr));
    *typeb = '\0';
    getdtype(ARGTYPG(sptr), typeb);
    fprintf(dfil, "pnmptr: %d   paramct: %d   ilm: %d   argtype: %s\n",
            PNMPTRG(sptr), PARAMCTG(sptr), (int)ILMG(sptr), typeb);
    *typeb = '\0';
    getdtype(INTTYPG(sptr), typeb);
    fprintf(dfil, "inttyp: %s\n", typeb);
    break;

  case ST_GENERIC:
    if (sptr >= stb.firstusym) {
      int dscptr;
      fprintf(dfil, "gsame: %d   gncnt:%d   gndsc:%d\n", (int)GSAMEG(sptr),
              GNCNTG(sptr), GNDSCG(sptr));
      fprintf(dfil, "Overloaded funcs:\n");
      for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr)) {
        fprintf(dfil, "sptr =%5d, %s\n", SYMI_SPTR(dscptr),
                SYMNAME(SYMI_SPTR(dscptr)));
      }
    } else
    {
      fprintf(dfil, "expst: %d   gsint: %d   gint: %d   greal: %d\n",
              EXPSTG(sptr), GSINTG(sptr), GINTG(sptr), (int)GREALG(sptr));
      fprintf(dfil,
              "gdble: %d   gcmplx: %d   gdcmplx: %d   gint8: %d   gsame: %d\n",
              GDBLEG(sptr), GCMPLXG(sptr), GDCMPLXG(sptr), GINT8G(sptr),
              (int)GSAMEG(sptr));
    }
    break;

  case ST_PD:
    fprintf(dfil, "pdnum: %d\n", PDNUMG(sptr));
    break;

  case ST_PLIST:
    fprintf(dfil, "ref: %d  dinit: %d\n", REFG(sptr), DINITG(sptr));
    fprintf(dfil, "address: %" ISZ_PF "d   pllen: %d", ADDRESSG(sptr),
            PLLENG(sptr));
    fprintf(dfil, "  uplevel: %d", (int)UPLEVELG(sptr));
    fprintf(dfil, "  internref: %d", (int)INTERNREFG(sptr));
    fprintf(dfil, "\n");
    if (SYMNAME(sptr)[1] == 'J')
      fprintf(dfil, "swel: %d, deflab: %d\n", SWELG(sptr), (int)DEFLABG(sptr));
#ifdef TLSG
    _PFG(TLSG(sptr), "tls");
#endif /* TLSG */
#ifdef USE_MPC
    if (ETLSG(sptr))
      fprintf(file, " etls: %d", ETLSG(sptr));
#endif /* USE_MPC */
    break;

  case ST_BLOCK:
    fprintf(dfil, "startline %d  endline %d  enclfunc %d\n", STARTLINEG(sptr),
            ENDLINEG(sptr), ENCLFUNCG(sptr));
    fprintf(file, "startlab %d  endlab %d  beginscopelab %d  endscopelab %d",
            STARTLABG(sptr), ENDLABG(sptr), BEGINSCOPELABG(sptr),
            ENDSCOPELABG(sptr));
    fprintf(dfil, " autobj: %d", AUTOBJG(sptr));
#ifdef PARUPLEVEL
    fprintf(dfil, " paruplevel: %d", PARUPLEVELG(sptr));
#endif
#ifdef PARSYMSG
    fprintf(dfil, " parsyms: %d", PARSYMSG(sptr));
    fprintf(dfil, " parsymsct: %d", PARSYMSCTG(sptr));
#endif
    fprintf(dfil, "\n");
    break;

  default:
    interr("symdmp: bad symbol type", stype, ERR_Informational);
  }
}
#endif

static void
putaltname(FILE *dfil, int sptr, const char *pref)
{
  int ss, len;
  char *np;
  ss = ALTNAMEG(sptr);
  if (!ss)
    return;
  fprintf(dfil, "%saltname:%d(", pref, ss);
  if (DECORATEG(sptr))
    fprintf(dfil, "_");
  len = DTyCharLength(DTYPEG(ss));
  np = stb.n_base + CONVAL1G(ss);
  while (true) {
    fprintf(dfil, "%c", *np);
    if (len <= 1)
      break;
    len--;
    np++;
  }
  fprintf(dfil, ")");
}

static void
putcuda(FILE *dfil, int sptr)
{
#ifdef CUDAG
  if (CUDAG(sptr)) {
    int cu;
    fprintf(dfil, "cuda: ");
    cu = CUDAG(sptr);
    if (cu & CUDA_HOST) {
      fprintf(dfil, "host");
      cu &= ~CUDA_HOST;
      if (cu)
        fprintf(dfil, "+");
    }
    if (cu & CUDA_DEVICE) {
      fprintf(dfil, "device");
      cu &= ~CUDA_DEVICE;
      if (cu)
        fprintf(dfil, "+");
    }
    if (cu & CUDA_GLOBAL) {
      fprintf(dfil, "global");
      cu &= ~CUDA_GLOBAL;
      if (cu)
        fprintf(dfil, "+");
    }
    if (cu & CUDA_GRID) {
      fprintf(dfil, "grid");
      cu &= ~CUDA_GRID;
      if (cu)
        fprintf(dfil, "+");
    }
    if (cu & CUDA_BUILTIN) {
      fprintf(dfil, "builtin");
      cu &= ~CUDA_BUILTIN;
      if (cu)
        fprintf(dfil, "+");
    }
    if (cu & CUDA_CONSTRUCTOR) {
      fprintf(dfil, "constructor");
      cu &= ~CUDA_CONSTRUCTOR;
      if (cu)
        fprintf(dfil, "+");
    }
#ifdef CUDA_STUB
    if (cu & CUDA_STUB) {
      fprintf(dfil, "stub");
      cu &= ~CUDA_STUB;
      if (cu)
        fprintf(dfil, "+");
    }
#endif
    fprintf(dfil, "\n");
  }
#endif
}

/**
   \brief Dump symbol table for debugging purposes.  If full == true,
   dump entire symbol table, otherwise dump symtab beginning with user
   symbols.
 */
void
symdmp(FILE *dfil, bool full)
{
#if DEBUG
  int sptr; /* symbol currently being dumped */

  for (sptr = (full ? 1 : stb.firstusym); sptr < stb.stg_avail; sptr++)
    symdentry(dfil, sptr);
#endif
}

#if DEBUG
void
dmp_socs(int sptr, FILE *file)
{
  int p;
  int q;

  fprintf(file, "dmp_socs(%d)\n", sptr);
  q = 0;
  for (p = SOCPTRG(sptr); p; p = SOC_NEXT(p)) {
    fprintf(file, " overlaps: %s\n", SYMNAME(SOC_SPTR(p)));
    if (q == p) {
      fprintf(file, ">>>>> soc loop\n");
      break;
    }
    q = p;
  }
}
#endif

/* FIXME: getccsym could be shared between C and Fortran */

static void
set_ccflags(int sptr, SYMTYPE stype)
{
  STYPEP(sptr, stype);
  CCSYMP(sptr, 1);
  LSCOPEP(sptr, 1);
#ifdef REFDP
  /* C+++ mark all compiler generated tmps as referenced */
  REFDP(sptr, 1);
#endif
}

SPTR
getccsym(char letter, int n, SYMTYPE stype)
{
  char name[16];
  SPTR sptr;

  sprintf(name, ".%c%04d", letter, n); /* at least 4, could be more */
  sptr = getsym(name, strlen(name));
  set_ccflags(sptr, stype);
  return sptr;
}

SPTR
getnewccsym(char letter, int n, SYMTYPE stype)
{
  char name[32];
  SPTR sptr;

  sprintf(name, ".%c%04d", letter, n); /* at least 4, could be more */
  NEWSYM(sptr);
  NMPTRP(sptr, putsname(name, strlen(name)));
  set_ccflags(sptr, stype);
  return sptr;
}

SPTR
getccsym_sc(char letter, int n, SYMTYPE stype, SC_KIND sc)
{
  SPTR sptr;

  if (sc != SC_PRIVATE) {
    sptr = getccsym(letter, n, stype);
    SCP(sptr, sc);
  } else {
    char name[16];
    sprintf(name, ".%c%04dp", letter, n); /* at least 4, could be more */
    sptr = getcctemp_sc(name, stype, sc);
    return sptr;
  }

  SCP(sptr, sc);
  return sptr;
}

SPTR
getcctemp_sc(const char *name, SYMTYPE stype, SC_KIND sc)
{
  SPTR sym;

  sym = getsym(name, strlen(name));
  set_ccflags(sym, stype);
  SCP(sym, sc);
  return sym;
}

SPTR
getccssym(const char *pfx, int n, SYMTYPE stype)
{
  char name[32];
  SPTR sptr;
  int i;

  sprintf(name, ".%s%04d", pfx, n); /* at least 4, could be more */
  i = 0;
  do {
    sptr = getsym(name, strlen(name));
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, stype);
      CCSYMP(sptr, 1);
      SCOPEP(sptr, stb.curr_scope);
      return sptr;
    }
    if (SCOPEG(sptr) == stb.curr_scope)
      return sptr;
    /* make up a new name */
    ++i;
    sprintf(&name[strlen(pfx) + 1], "%04d%03d", n, i);
  } while (1);
}

SPTR
getccssym_sc(const char *pfx, int n, SYMTYPE stype, SC_KIND sc)
{
  SPTR sptr;

  if (sc != SC_PRIVATE)
    sptr = getccssym(pfx, n, stype);
  else {
    char name[32];
    int i;
    sprintf(name, ".%s%04dp", pfx, n); /* at least 4, could be more */
    i = 0;
    do {
      sptr = getsym(name, strlen(name));
      if (STYPEG(sptr) == ST_UNKNOWN) {
        STYPEP(sptr, stype);
        CCSYMP(sptr, 1);
        SCOPEP(sptr, stb.curr_scope);
        break;
      }
      if (SCOPEG(sptr) == stb.curr_scope)
        break;
      /* make up a new name */
      ++i;
      sprintf(&name[strlen(pfx) + 1], "%04d%03dp", n, i);
    } while (1);
  }
  SCP(sptr, sc);
  return sptr;
}

/* FIXME: getccsym_copy is the same between C and Fortran */

/*
 * get a compiler symbol that is a 'copy' of the given symbol
 * append '.copy' to the name
 * copy the symbol type, data type, symbol class fields
 */
SPTR
getccsym_copy(SPTR oldsptr)
{
  SPTR sptr;
  int oldlen, len, i;
  char fname[39];
  char *name;

  if (STYPEG(oldsptr) != ST_VAR)
    return oldsptr;
  if (!DT_ISINT(DTYPEG(oldsptr)))
    return oldsptr;
  oldlen = strlen(SYMNAME(oldsptr));
  if (oldlen >= 32) {
    name = (char *)malloc(oldlen + 1);
  } else {
    name = fname;
  }
  strcpy(name, SYMNAME(oldsptr));
  strcat(name, ".copy");
  len = strlen(name);
  i = 0;
  do {
    sptr = getsym(name, strlen(name));
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, STYPEG(oldsptr));
      DTYPEP(sptr, DTYPEG(oldsptr));
      SCP(sptr, SCG(oldsptr));
      CCSYMP(sptr, 1);
      SCOPEP(sptr, SCOPEG(oldsptr));
      ENCLFUNCP(sptr, ENCLFUNCG(oldsptr));
      break;
    }
    if (SCOPEG(sptr) == stb.curr_scope)
      break;
    /* make up a new name */
    ++i;
    sprintf(name + len, "%d", i);
  } while (1);
#ifdef REFDP
  /* C+++ mark all compiler generated tmps as referenced */
  REFDP(sptr, 1);
#endif
  if (oldlen >= 32)
    free(name);
  return sptr;
}

/* FIXME: insert_sym is the same between C and Fortran */

/**
   \brief Create new symbol table entry and insert it in the hash list
   immediately in front of 'first':
 */
SPTR
insert_sym(SPTR first)
{
  SPTR sptr;
  int i, j;
  INT hashval;
  char *np;

  NEWSYM(sptr);
  NMPTRP(sptr, NMPTRG(first));
  /* link newly created symbol immediately in front of first: */
  np = SYMNAME(first);
  i = strlen(np);
  HASH_ID(hashval, np, i);
  HASHLKP(sptr, first);
  if (stb.hashtb[hashval] == first)
    stb.hashtb[hashval] = sptr;
  else {
    /* scan hash list to find immed. predecessor of first: */
    for (i = stb.hashtb[hashval]; (j = HASHLKG(i)) != first; i = j)
      assert(j != 0, "insert_sym: bad hash", first, ERR_Fatal);
    HASHLKP(i, sptr);
  }

  SYMLKP(sptr, NOSYM); /* installsym for ftn also sets SYMLK */
  setimplicit(sptr);
  return sptr;
}

/**
   \brief Create new symbol table entry and insert it in the hash list
   immediately in front of 'first':
 */
SPTR
insert_sym_first(SPTR first)
{
  SPTR sptr;
  int i;
  INT hashval;
  char *np;

  NEWSYM(sptr);
  NMPTRP(sptr, NMPTRG(first));
  /* link newly created symbol immediately in front of first: */
  np = SYMNAME(first);
  i = strlen(np);
  HASH_ID(hashval, np, i);
  HASHLKP(sptr, stb.hashtb[hashval]);
  stb.hashtb[hashval] = sptr;
  setimplicit(sptr);
  return sptr;
}

SPTR
getlab(void)
{
  return getccsym('B', stb.lbavail++, ST_LABEL);
}

int
get_entry_item(void)
{
  int ent = aux.entry_avail;
  if (aux.entry_avail++ == 0) {
    aux.entry_size = 10;
    NEW(aux.entry_base, ENTRY, aux.entry_size);
  } else {
    NEED(aux.entry_avail, aux.entry_base, ENTRY, aux.entry_size,
         aux.entry_size + 10);
  }
  return ent;
}

/**
   \brief Scan all hash lists and remove symbols whose scope is
   greater than or equal to the current scope:
 */
void
pop_scope(void)
{
  int i, j, sptr;

#if DEBUG
  if (DBGBIT(5, 1024))
    fprintf(gbl.dbgfil, "pop_scope(): scope %d\n", stb.curr_scope);
#endif
  for (i = 0; i < HASHSIZE; i++)
    for (sptr = stb.hashtb[i], j = 0; sptr; sptr = HASHLKG(sptr))
      if ((int)SCOPEG(sptr) >= stb.curr_scope) {
#if DEBUG
        if (DBGBIT(5, 1024))
          fprintf(gbl.dbgfil, "removing %s, sptr:%d\n", SYMNAME(sptr), sptr);
#endif
        if (j)
          HASHLKP(j, HASHLKG(sptr));
        else
          stb.hashtb[i] = HASHLKG(sptr);
      } else {
        j = sptr;
      }
}

/**
   \brief Scan all hash lists and remove specified symbol from scope.
 */
void
pop_sym(int sptr)
{
  char *name;
  INT hashval;
  int s, j;

#if DEBUG
  if (DBGBIT(5, 1024))
    fprintf(gbl.dbgfil, "pop_sym(): sym %d\n", sptr);
#endif
  if (NMPTRG(sptr) == 0)
    return;
  name = SYMNAME(sptr);
  HASH_ID(hashval, name, strlen(name));
  for (s = stb.hashtb[hashval], j = 0; s; s = HASHLKG(s)) {
    if (s == sptr) {
#if DEBUG
      if (DBGBIT(5, 1024))
        fprintf(gbl.dbgfil, "removing %s, sptr:%d\n", SYMNAME(sptr), sptr);
#endif
      if (j)
        HASHLKP(j, HASHLKG(sptr));
      else
        stb.hashtb[hashval] = HASHLKG(sptr);
      break;
    }
    j = s;
  }
  HASHLKP(sptr, SPTR_NULL);
}

/**
   \brief Create a function ST item given a name.
 */
SPTR
mkfunc(const char *nmptr)
{
  SPTR sptr;

  sptr = getsym(nmptr, strlen(nmptr));
  if (STYPEG(sptr) == ST_PROC) {
    if (!REFG(sptr) && !SYMLKG(sptr)) {
      SYMLKP(sptr, gbl.externs);
      gbl.externs = sptr;
    }
    sym_is_refd(sptr);
    return sptr;
  }
  STYPEP(sptr, ST_PROC);
  DTYPEP(sptr, DT_INT);
  SCP(sptr, SC_EXTERN);
  CCSYMP(sptr, 1);
#ifdef SDSCSAFEP
  SDSCSAFEP(sptr, 1);
#endif
  if (!REFG(sptr) && !SYMLKG(sptr)) {
    SYMLKP(sptr, gbl.externs);
    gbl.externs = sptr;
  }

  sym_is_refd(sptr);
  return sptr;
}

typedef enum LLVMCallBack_t { NO_LLVM_CALLBACK, LLVM_CALLBACK } LLVMCallBack_t;

static SPTR
vmk_prototype(LLVMCallBack_t llCallBack, const char *name, const char *attr,
              DTYPE resdt, int nargs, va_list vargs)
{
  DTYPE args[64];
  SPTR sptr;
  int i;
  unsigned flags = 0;

  if (nargs > 64) {
    interr("vmk_prototype: nargs exceeds", 64, ERR_Severe);
    nargs = 64;
  }
  sptr = getsym(name, strlen(name));
  for (i = 0; i < nargs; i++) {
    int argdt = va_arg(vargs, int);
    args[i] = (DTYPE) argdt;
  }
  sptr = mkfunc(name); /* NEED a mk_pfunc() */
  DTYPEP(sptr, resdt);
  /*
   * A string of blank separated words, only the first character of each word is
   * signficant
   */
  while (attr) {
    while (*attr <= ' ' && *attr)
      ++attr;
    if (*attr == '\0')
      break;
    switch (*attr++) {
    case 'f': /* fast */
      flags |= FAST_MATH_FLAG;
      break;
    case 'p': /* pure */
      PUREP(sptr, 1);
      break;
    default:
      break;
    }
    while (*attr > ' ')
      ++attr;
  }
  if (llCallBack == LLVM_CALLBACK)
    ll_add_func_proto(sptr, flags, nargs, args);
  return sptr;
}

/**
   \brief Make a prototype but do not register it

   Use when not using the LLVM backend or when the signature given is known to
   cause regressions in testing.
 */
SPTR
mk_prototype(const char *name, const char *attr, DTYPE resdt, int nargs, ...)
{
  va_list vargs;
  SPTR rv;
  va_start(vargs, nargs);
  rv = vmk_prototype(NO_LLVM_CALLBACK, name, attr, resdt, nargs, vargs);
  va_end(vargs);
  return rv;
}

/**
   \brief Make a prototype and register it with LLVM
 */
SPTR
mk_prototype_llvm(const char *name, const char *attr, DTYPE resdt, int nargs,
                  ...)
{
  va_list vargs;
  SPTR rv;
  va_start(vargs, nargs);
  rv = vmk_prototype(LLVM_CALLBACK, name, attr, resdt, nargs, vargs);
  va_end(vargs);
  return rv;
}

int
add_symitem(int sptr, int nxt)
{
  int i;
  i = aux.symi_avl++;
  NEED(aux.symi_avl, aux.symi_base, SYMI, aux.symi_size, aux.symi_avl + 100);
  SYMI_SPTR(i) = sptr;
  SYMI_NEXT(i) = nxt;
  return i;
}

#if DEBUG
int
dbg_symdentry(int sptr)
{
  symdentry(stderr, sptr);
  return 0;
}
#endif

SPTR
get_semaphore(void)
{
  SPTR sym;
  DTYPE dt;
  int ival[2];
  char name[10];
  ADSC *ad;
  static int semaphore_cnt = 0; /* counter for semaphore variables */

  strcpy(name, ".sem");
  sprintf(&name[4], "%05d", semaphore_cnt);
  semaphore_cnt++;
  sym = getsym(name, 9); /* semaphore variable, 1 per critical section */
  /*
   * kmpc requires a semaphore variable to be 32 bytes and
   * 8-byte aligned
   */
  dt = get_array_dtype(1, DT_INT8);
  STYPEP(sym, ST_ARRAY);
  DTYPEP(sym, dt);
  ad = AD_DPTR(dt);
  AD_NUMELM(ad) = AD_UPBD(ad, 0);
  AD_MLPYR(ad, 0) = stb.i1;
  AD_LWBD(ad, 0) = stb.i1;
  ival[0] = 0;
  ival[1] = 4;
  AD_UPBD(ad, 0) = getcon(ival, DT_INT);
  AD_NUMDIM(ad) = 1;
  AD_NUMELM(ad) = AD_UPBD(ad, 0);
  AD_SCHECK(ad) = 0;
  AD_ZBASE(ad) = stb.i1;
  ADDRTKNP(sym, 1);
  CCSYMP(sym, 1);
  SCP(sym, SC_STATIC);
  DCLDP(sym, 1);
  return sym;
}

#if DEBUG
int
tr_conval2g(char *fn, int ln, int s)
{
  if (DTYPEG(s) && DTY(DTYPEG(s)) == TY_PTR) {
    fprintf(stderr, "ACON CONVAL2G:%s:%d\n", fn, ln);
  }
  return stb.stg_base[s].w14;
}

int
tr_conval2p(char *fn, int ln, int s, int v)
{
  if (DTYPEG(s) && DTY(DTYPEG(s)) == TY_PTR) {
    fprintf(stderr, "ACON CONVAL2P:%s:%d\n", fn, ln);
  }
  stb.stg_base[s].w14 = v;
  return v;
}
#endif

SPTR
addnewsym(const char *name)
{
  SPTR sptr;
  NEWSYM(sptr);
  NMPTRP(sptr, putsname(name, strlen(name)));
  return sptr;
} /* addnewsym */

SPTR
adddupsym(SPTR oldsptr)
{
  SPTR sptr;
  NEWSYM(sptr);
  NMPTRP(sptr, NMPTRG(oldsptr));
  return sptr;
} /* adddupsym */
