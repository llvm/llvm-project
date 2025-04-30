/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 * \file
 * \brief Fortran - symbol table access module.

 * This module contains the routines used to initialize, update, access, and
 * dump the symbol table.  Note that in addition to being used by PGHPF, this
 * module is used by the utility program, symini, which processes intrinsic and
 * generic definitions in order to set up the initial symbol table for PGHPF.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "machar.h"
#include "symtab.h"
#include "symtabdf.h"
#include "symutl.h"
#include "syminidf.h"
#include "dtypeutl.h"
#include "soc.h"
#include "state.h"
#include "ast.h"
#include "semant.h"
#include "llmputil.h"
#include "rtlRtns.h"
#include "rte.h" /* for get_all_descriptors, get_static_descriptor */
#include <stdarg.h>


/* During IPA recompilations, compiler-generated temporary names
 * need to be distinct from the names that were used on the first
 * compile.
 */
#define IPA_RECOMPILATION_SUFFIX (XBIT(89, 0x40) ? "i" : "")

/* implicit data types */
typedef struct {
  int dtype;
  LOGICAL set, anyset; /* True if set by IMPLICIT stmt */
  LOGICAL typ8;        /* True if the type is altered by -r8 */
} DTIMPL[26 + 26 + 2];

typedef struct {
  SPTR dec_sptr;
  SPTR def_sptr; 
} DEC_DEF_MAP;

static DTIMPL dtimplicit;
static int dtimplicitsize = 0;
static DTIMPL *save_dtimplicit = NULL;
static int dtimplicitstack = 0;

static void cng_inttyp(int, int);
static void generate_type_mismatch_errors(SPTR s1, SPTR s2);
static void update_arrdsc(SPTR s, DEC_DEF_MAP *smap, int num_dummies);
/* entry hack? */
static ENTRY onlyentry;

/*--------------------------------------------------------------------------*/

/**
 * Initialize symbol table for new user program unit.
 */
void
sym_init(void)
{
  int i;
  INT tmp[2], res[2];
  static const char *npname = "hpf_np$";
  int sptr;

  /* allocate symbol table and name table space:  */
  sym_init_first();

  init_chartab(); /* see dtypeutl.c */

  STG_RESET(stb.dt);
  STG_NEXT_SIZE(stb.dt, DT_MAX);
  for (i = 0; i <= DT_MAX; ++i)
    DTY(i) = pd_dtype[i];

  if (XBIT(49, 0x800000)) {
    DT_REAL = DT_REAL8;
    DT_CMPLX = DT_CMPLX16;
    if (XBIT(49, 0x200)) {
      DT_DBLE = DT_REAL8;
      DT_DCMPLX = DT_CMPLX16;
    } else {
      DT_DBLE = DT_QUAD;
      DT_DCMPLX = DT_QCMPLX;
    }
  }
  if (XBIT(49, 0x80000000)) {
    DT_INT = DT_INT8;
    DT_LOG = DT_LOG8;
  }
  if (XBIT(49, 0x100) && !XBIT(125, 0x2000)) {
    DT_PTR = DT_INT8;
  }
  if (!XBIT(124, 0x10)) {
    stb.user.dt_int = DT_INT;
    stb.user.dt_log = DT_LOG;
  } else {
    /* -i8 */
    stb.user.dt_int = DT_INT8;
    stb.user.dt_log = DT_LOG8;
  }
  if (!XBIT(124, 0x8)) {
    stb.user.dt_real = DT_REAL;
    stb.user.dt_cmplx = DT_CMPLX;
  } else {
    /* -r8 */
    stb.user.dt_real = DT_DBLE;
    stb.user.dt_cmplx = DT_DCMPLX;
  }
  /*
   * Set up initial implicit types.  All are real except for the letters i
   * thru n:
   */
  init_implicit();

/*
 * now initialize symbol table. There are 2 cases: The first case occurs
 * within the utility symini - we start with a totally empty symbol
 * table. The second case occurs within PGHPF - the initial symbol table is
 * copied from some arrays set up by symini.
 */

#if DEBUG
  assert(stb.stg_size >= INIT_SYMTAB_SIZE, "sym_init:INIT_SYMTAB_SIZE",
         INIT_SYMTAB_SIZE, 0);
#endif
  BCOPY(stb.stg_base, init_sym, SYM, INIT_SYMTAB_SIZE);
  stb.stg_avail = INIT_SYMTAB_SIZE;
  stb.stg_cleared = INIT_SYMTAB_SIZE;
#if DEBUG
  assert(stb.n_size >= INIT_NAMES_SIZE, "sym_init:INIT_NAMES_SIZE",
         INIT_NAMES_SIZE, 0);
#endif
  BCOPY(stb.n_base, init_names, char, INIT_NAMES_SIZE);
  stb.namavl = INIT_NAMES_SIZE;

  BCOPY(stb.hashtb, init_hashtb, int, HASHSIZE);

  if (XBIT(124, 0x10)) {
    /* -i8 */
    cng_inttyp(intast_sym[I_ICHAR], DT_INT8);
  }

  /* change the specific intrinsics whose types are DT_QUAD and DT_QCMPLX
   * if the the '-dp' switch is present.  This could be within a #ifdef C90,
   * but we do build the T3E/C90 pghpfc on solaris.
   */
  if (XBIT(49, 0x200))
    for (i = 1; i < INIT_SYMTAB_SIZE; i++)
      if (STYPEG(i) == ST_INTRIN) {
        if (INTTYPG(i) == DT_QUAD)
          INTTYPP(i, DT_REAL8);
        else if (INTTYPG(i) == DT_QCMPLX)
          INTTYPP(i, DT_CMPLX16);
        if (ARGTYPG(i) == DT_QUAD)
          ARGTYPP(i, DT_REAL8);
        else if (ARGTYPG(i) == DT_QCMPLX)
          ARGTYPP(i, DT_CMPLX16);
      }

  /*
   * enter constants into symbol table:
   *
   * * * * * * * * * *  N O T E  * * * * * * * * *
   * * * * * * * * * *  N O T E  * * * * * * * * *
   * DO NOT CHANGE the order of entering these predefined symbols.
   * To add a predefined, insert it between the last getcon/getsym
   * and where stb.firstosym is set.
   */

  /* int 0, 1 */
  tmp[0] = tmp[1] = (INT)0;
  stb.i0 = getcon(tmp, DT_INT);
  if (DT_INT == DT_INT8)
    stb.k0 = stb.i0;
  else if (!XBIT(57, 0x2))
    stb.k0 = getcon(tmp, DT_INT8);
  else
    stb.k0 = 0;
  tmp[1] = (INT)1;
  stb.i1 = getcon(tmp, DT_INT);
  if (DT_INT == DT_INT8)
    stb.k1 = stb.i1;
  else if (!XBIT(57, 0x2))
    stb.k1 = getcon(tmp, DT_INT8);
  else
    stb.k1 = 0;

  add_fp_constants();
  /*
   * * * * * * * * * *  N O T E  * * * * * * * * *
   * NO MORE predefined contants until immediately before
   * stb.firstosym is set ...
   */

  /* create symbol hpf_np$ */
  sptr = getsymbol(npname);
  STYPEP(sptr, ST_UNKNOWN);
  DTYPEP(sptr, DT_INT);
  DCLDP(sptr, 1);
  SCP(sptr, SC_LOCAL);
  NODESCP(sptr, 1);
  gbl.sym_nproc = sptr;
  if (XBIT(70, 0x80000000)) {
    int bsym;
    SCP(sptr, SC_BASED);
    bsym = getsymf("%sp", npname); /* hpf_np$p */
    STYPEP(bsym, ST_UNKNOWN);
    DTYPEP(bsym, DT_PTR);
    DCLDP(bsym, 1);
    SCP(bsym, SC_LOCAL);
    NODESCP(bsym, 1);
    MIDNUMP(sptr, bsym);
  }

  /* allocate space for auxiliary symtab structures: */

  if (aux.dpdsc_size <= 0) {
    aux.dpdsc_size = 100;
    NEW(aux.dpdsc_base, int, aux.dpdsc_size);
  }
  aux.dpdsc_avl = 1; /* 0 => null DPDSC field */
  aux.dpdsc_base[0] = 0;

  if (aux.arrdsc_size <= 0) {
    aux.arrdsc_size = 200;
    NEW(aux.arrdsc_base, int, aux.arrdsc_size);
  }
  aux.arrdsc_base[0] = 0; /* numdim=0 if array descriptor is zero */
  aux.arrdsc_avl = 1;

  if (aux.nml_size <= 0) {
    aux.nml_size = 200;
    NEW(aux.nml_base, NMLDSC, aux.nml_size);
  }
  aux.nml_avl = 1;
  aux.nml_base[0].sptr = 0;
  aux.nml_base[0].next = 0;
  aux.nml_base[0].lineno = 0;

  if (!XBIT(7, 0x100000)) {
    if (aux.dvl_size <= 0) {
      aux.dvl_size = 32;
      NEW(aux.dvl_base, DVL, aux.dvl_size);
    }
    aux.dvl_avl = 0;
  }

  if (aux.symi_size <= 0) {
    aux.symi_size = 100;
    NEW(aux.symi_base, SYMI, aux.symi_size);
  }
  aux.symi_avl = 1; /* 0 => end of list */
  aux.symi_base[0].sptr = 0;
  aux.symi_base[0].next = 0;

  /*
   * * * * * * * * * *  N O T E  * * * * * * * * *
   * More predefined constants after ORIGINAL set; note the
   * value of NXTRA (number of post-original symbols.
   */
  tmp[0] = 0;
  res[0] = 0;
  tmp[1] = CONVAL2G(stb.flt0);
  xfneg(tmp[1], &res[1]);
  stb.fltm0 = getcon(res, DT_REAL4);
  tmp[0] = CONVAL1G(stb.dbl0);
  tmp[1] = CONVAL2G(stb.dbl0);
  xdneg(tmp, res);
  stb.dblm0 = getcon(res, DT_DBLE);
#define NXTRA 2

  aux.curr_entry = &onlyentry;
  stb.firstusym = stb.firstosym = stb.stg_avail;
  stb.lbavail = 99999;

  for (i = 0; i < ST_MAX; i++)
    aux.list[i] = NOSYM; /* 'empty' list for each stype */

  soc.avail = 1;
  if ((gbl.internal == 0) && (flg.ipa & 0x100) == 0) {
    /* clear at outer routines, but not for static$init routines */
    if (gbl.currmod == 0) {
      dtimplicitstack = 0;
    } else {
      dtimplicitstack = 1;
    }
  }
  symutl.sc = SC_LOCAL;

  llmp_reset_uplevel();
}

static void
cng_inttyp(int ss, int dt)
{
#if DEBUG
  assert(STYPEG(ss) == ST_INTRIN, "cng_inttyp not intr", ss, 3);
#endif
  INTTYPP(ss, dt);
}

/**
 * Set up initial implicit types.  All are real except for the letters i
 * thru n:
 */
void
init_implicit(void)
{
  int i;
  int default_int;

  for (i = 0; i < 54; i++) {
    dtimplicit[i].dtype = stb.user.dt_real;
    dtimplicit[i].set = FALSE;
    dtimplicit[i].anyset = FALSE; /* not explicit set anywhere */
    if (XBIT(124, 0x8) && !XBIT(49, 0x800000)) {
      dtimplicit[i].typ8 = TRUE;
    } else {
      dtimplicit[i].typ8 = FALSE;
    }
  }

  default_int = flg.i4 ? stb.user.dt_int : DT_SINT;
  implicit_int(default_int);
}

/**
 * simple routine to reset the default integer type for implicitly typing
 * integer variables.  Needed for compile-type processing of -i4/-noi4
 * options in OPTIONS statement.
 */
void
implicit_int(int default_int)
{
  int i;
  for (i = 8; i <= 13; i++) {
    dtimplicit[i].dtype = dtimplicit[i + 26].dtype = default_int;
    dtimplicit[i].typ8 = dtimplicit[i + 26].typ8 = FALSE;
  }
}

void
save_implicit(LOGICAL reset)
{
  if (save_dtimplicit == NULL) {
    dtimplicitsize = 4;
    NEW(save_dtimplicit, DTIMPL, dtimplicitsize);
    BZERO(save_dtimplicit, DTIMPL, dtimplicitsize);
  } else {
    NEED(dtimplicitstack + 1, save_dtimplicit, DTIMPL, dtimplicitsize,
         dtimplicitsize * 2);
  }
  BCOPY(save_dtimplicit[dtimplicitstack], dtimplicit, DTIMPL, 1);
  ++dtimplicitstack;
  if (reset) {
    int i;
    for (i = 0; i < 54; i++) {
      dtimplicit[i].set = FALSE; /* not explicitly set at this level */
    }
  }
}

void
restore_implicit(void)
{
  if (dtimplicitstack <= 0)
    interr("IMPLICIT stack too shallow", dtimplicitstack, 3);
  --dtimplicitstack;
  BCOPY(dtimplicit, save_dtimplicit[dtimplicitstack], DTIMPL, 1);
}

/**
 * Return the first & last sym pointers for the HPF_LIBRARY procedures.
 * Initially, the stype of the HPF library procedures is ST_HL;
 * when the USE statement is seen, the stype of these symbols is changed
 * to ST_PD.
 */
void
hpf_library_stat(int *firstp, int *lastp, int stype)
{
  if (stype == ST_CRAY) {
    *firstp = CRAY_FIRST;
    *lastp = CRAY_LAST;
  }
#if DEBUG
  else {
    interr("hpf_library_stat:illegal stype", stype, 0);
  }
#endif
}

/**
 * Return the first & last sym pointers for the ISO_C intrinsic
 * procedures. Initially, the stype of the ISO_C library procedures is ST_ISOC
 * when the USE statement is seen, the stype of these symbols is changed
 * to ST_INTRIN
 */
void
iso_c_lib_stat(int *firstp, int *lastp, int stype)
{

  if (stype == ST_ISOC) {
    *firstp = ISO_C_FIRST;
    *lastp = ISO_C_LAST;
  }
#if DEBUG
  else {
    interr("iso_c_lib_stat:illegal stype", stype, 0);
  }
#endif
}

extern int
get_ieee_arith_intrin(const char *nm)
{
  int i;

  for (i = IEEEARITH_FIRST; i <= IEEEARITH_LAST; i++) {
    if (strcmp(SYMNAME(i), nm) == 0)
      return i;
  }
  return 0;
}

/*
 * Enter symbol with indicated null-terminated name into symbol table,
 * initialize the new entry, and return pointer to it.  If there is already
 * such a symbol, just return pointer to the existing symbol table entry.
 */
int
getsymbol(const char *name)
{
  return getsym(name, strlen(name));
}

/** \brief Enter symbol with indicated name into symbol table, initialize
           the new entry.
    \param name symbol name
    \param olength number of characters in the symbol name
    \return pointer to new symbol

    If there is already such a symbol, just return pointer to the existing
    symbol table entry.
 */
int
getsym(const char *name, int olength)
{
  int sptr; /* pointer to symbol table entry */

  sptr = installsym(name, olength);
  switch (STYPEG(sptr)) {
  case ST_ISOC:
  case ST_IEEEARITH:
  case ST_IEEEEXCEPT:
  case ST_ISOFTNENV:
  case ST_CRAY:
    /* predefined symbol is the name of an HPF library procedure; since
     * the stype is ST_HL, this implies that a 'USE HPF_LIBRARY' statement
     * was not seen.  Consequently, this symbol is a user symbol and
     * a new (ST_UNKNOWN) symbol must be entered into the symbol table.
     */
    sptr = insert_sym(sptr);
    setimplicit(sptr);
    if (gbl.internal > 1)
      INTERNALP(sptr, 1);
    SCOPEP(sptr, stb.curr_scope);
    IGNOREP(sptr, 0);
    break;
  case ST_UNKNOWN:
    setimplicit(sptr);
    if (gbl.internal > 1)
      INTERNALP(sptr, 1);
    SCOPEP(sptr, stb.curr_scope);
    IGNOREP(sptr, 0);
    break;
  default:
    break;
  }
  return sptr;
}

/* Construct a name via vsnprintf(), then use it to look up or
 * create a symbol.
 */
int
getsymf(const char *fmt, ...)
{
  char buffer[MAXIDLEN + 1];
  va_list ap;

  va_start(ap, fmt);
  vsnprintf(buffer, sizeof buffer, fmt, ap);
  va_end(ap);
  buffer[sizeof buffer - 1] = '\0'; /* Windows vsnprintf bug work-around */
  return getsymbol(buffer);
}

/*--------------------------------------------------------------------------*
 * getcon & get_acon is identical between C and Fortran; should be shared   *
 *--------------------------------------------------------------------------*/

/** \brief Enter constant of given dtype and value into the symbol table and
   return
    pointer to it.

    If an entry for the constant already exists, return pointer to the existing
   entry instead.
 */
SPTR
getcon(INT *value, DTYPE dtype)
{
  int sptr;    /* symbol table pointer */
  int hashval; /* index into hashtb */

  /*
   * First loop thru the appropriate hash link list to see if this constant
   * is already in the symbol table:
   */

  hashval = HASH_CON(value);
  if (hashval < 0)
    hashval = -hashval;
  for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
    if (DTY(dtype) == TY_QUAD) {
      if (DTYPEG(sptr) != dtype || STYPEG(sptr) != ST_CONST ||
          CONVAL1G(sptr) != value[0] || CONVAL2G(sptr) != value[1] ||
          CONVAL3G(sptr) != value[2] || CONVAL4G(sptr) != value[3])
        continue;

      /* Matching entry has been found.  Return it:  */
      return (sptr);
    }
    if (DTYPEG(sptr) != dtype || STYPEG(sptr) != ST_CONST ||
        CONVAL1G(sptr) != value[0] || CONVAL2G(sptr) != value[1])
      continue;

    /* Matching entry has been found.  Return it:  */
    return (sptr);
  }

  /* Constant not found.  Create a new symbol table entry for it: */

  ADDSYM(sptr, hashval);
  STYPEP(sptr, ST_CONST);
  CONVAL1P(sptr, value[0]);
  CONVAL2P(sptr, value[1]);
  if (DTY(dtype) == TY_QUAD) {
    CONVAL3P(sptr, value[2]);
    CONVAL4P(sptr, value[3]);
  }
  DTYPEP(sptr, dtype);
  SCOPEP(sptr, 1);

  return (sptr);
}

/* constant value (value[1] if 1 word) */
int
hashcon(INT *value, int dtype, int sptr)
{
  int sptr1;   /* symbol table pointer */
  int hashval; /* index into hashtb */

  /*
   * First loop thru the appropriate hash link list to see if this constant
   * is already in the symbol table:
   */

  hashval = HASH_CON(value);
  if (hashval < 0)
    hashval = -hashval;
  for (sptr1 = stb.hashtb[hashval]; sptr1 != 0; sptr1 = HASHLKG(sptr1)) {

    if (sptr1 == sptr)
      return (sptr);
  }

  /* sptr not found.  */

  HASHLKP(sptr, stb.hashtb[hashval]);
  stb.hashtb[hashval] = sptr;

  return (sptr);
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
  int sptr;    /* symbol table pointer */
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

    return (sptr);
  }

  /* Constant not found.  Create a new symbol table entry for it: */

  ADDSYM(sptr, hashval);
  CONVAL1P(sptr, sym);
  ACONOFFP(sptr, off);
  STYPEP(sptr, ST_CONST);
  DTYPEP(sptr, dtype);
  SCOPEP(sptr, 1);

  return (sptr);
}

ISZ_T
get_isz_cval(int con)
{
  INT num[2];
  ISZ_T v;
#if DEBUG
  assert(STYPEG(con) == ST_CONST, "get_isz_cval-not ST_CONST", con, 0);
  assert(DT_ISINT(DTYPEG(con)) || DT_ISLOG(DTYPEG(con)),
         "get_int_cval-not int const", con, 0);
#endif
  if (XBIT(68, 0x1)) {
    if (size_of(DTYPEG(con)) <= 4)
      return get_int_cval(con);
  }
  num[0] = CONVAL1G(con);
  num[1] = CONVAL2G(con);
  INT64_2_ISZ(num, v);
  return v;
}

/**
 * Retrieve the value of an integer constant symbol table entry and
 * return as an 'INT'.  Coerce TY_INT8 values if necessary.
 */
INT
get_int_cval(int con)
{
  INT res;

#if DEBUG
  assert(STYPEG(con) == ST_CONST, "get_int_cval-not ST_CONST", con, 0);
  assert(DT_ISINT(DTYPEG(con)) || DT_ISLOG(DTYPEG(con)),
         "get_int_cval-not int const", con, 0);
#endif

  switch (DTY(DTYPEG(con))) {
  case TY_INT8:
  case TY_LOG8:
    res = CONVAL2G(con);
    break;
  default:
    res = CONVAL2G(con);
    break;
  }

  return res;
}

/**
 * Sign extend an integer value of an indicated width (8, 16, 32); value
 * returned is sign extended with respect to the host's int type.
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

/*--------------------------------------------------------------------------*/

/** \brief Enter character constant into symbol table and return
    pointer to it.
    \param value character string value
    \param length length of character string

    If the constant is already in the table,
    return pointer to the existing entry instead.
 */
int
getstring(const char *value, int length)
{
  int sptr;    /* symbol table pointer */
  int hashval; /* index into hashtb */
  char *np;    /* pointer to string characters */
  const char *p;
  int i, clen;
  int dtype;
  /*
   * first loop thru the appropriate hash link list to see if symbol is
   * already in the table:
   */
  HASH_STR(hashval, value, length);
  /* Ensure hash value is positive.  '\nnn' can cause negative hash values */
  if (hashval < 0)
    hashval = -hashval;
  for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
    if (STYPEG(sptr) != ST_CONST)
      continue;
    i = DTYPEG(sptr);
    if (DTY(i) == TY_CHAR) {
      clen = DTY(i + 1);
      if (clen > 0 && A_ALIASG(clen)) {
        clen = A_ALIASG(clen);
        clen = A_SPTRG(clen);
        clen = CONVAL2G(clen);
        if (clen == length) {
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
      }
    }
  Continue:;
  }

  /* String not found.  Create a new symtab entry for it:  */

  dtype = get_type(2, TY_CHAR, mk_cval(length, DT_INT4));
  ADDSYM(sptr, hashval);
  STYPEP(sptr, ST_CONST);
  CONVAL1P(sptr, putsname(value, length));
  DTYPEP(sptr, dtype);
  SCOPEP(sptr, 1);
  return (sptr);
}

/*--------------------------------------------------------------------------*/

/**
 * Create a 'kinded' hollerith constant from a string constant and return the
 * pointer to it.  If the constant is already in the table, return pointer to
 * the existing entry.  Possible kind values are:
 *    'h' - left justifed, blank filled.
 *    'l' - left justfied, zero filled.
 *    'r' - right justfied, zero filled.
 */
int
gethollerith(int strcon, int kind)
{
  INT val[2];
  int sptr;

  val[0] = strcon;
  val[1] = kind;
  sptr = getcon(val, DT_HOLL);
  HOLLP(sptr, 1);
  return sptr;
}

/*--------------------------------------------------------------------------*/

/** \brief Change the current settings for implicit variable types and character
           lengths.
    \param firstc characters delimiting range
    \param lastc new value assigned to range
    \param dtype data type
 */
void
newimplicit(int firstc, int lastc, int dtype)
{
  int i, j; /* indices into implicit arrays */
  char temp[2];

  i = IMPL_INDEX(firstc); /* IMPL_INDEX is defined in symtab.h */
  j = IMPL_INDEX(lastc);
#if DEBUG
  assert(i >= 0 && j >= 0 && i < 54 && j < 54, "newimplicit: bad impl range", i,
         4);
#endif

  for (; i <= j; i++) {
    if (dtimplicit[i].set) { /* already set */
      temp[0] = 'a' + i;
      temp[1] = 0;
      if (dtype == dtimplicit[i].dtype)
        error(54, 2, gbl.lineno, temp, CNULL);
      else
        error(54, 3, gbl.lineno, temp, CNULL);
    }
    dtimplicit[i].dtype = dtype;
    dtimplicit[i].set = TRUE;
    dtimplicit[i].anyset = TRUE; /* explicitly set */
    dtimplicit[i].typ8 = FALSE;
  }
}

void
newimplicitnone(void)
{
  int i;
  for (i = 0; i < 54; ++i) {
    dtimplicit[i].anyset = FALSE; /* explicit reset */
  }
} /* newimplicitnone */

/*--------------------------------------------------------------------------*/

/**
 * assign to the indicated symbol table entry, the current implicit dtype.
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

/** \brief Return FALSE if this symbol could not have been properly implicitly
 * typed
 */
LOGICAL
was_implicit(int sptr)
{
  int firstc, i;
  firstc = *SYMNAME(sptr);
  i = IMPL_INDEX(firstc);
  if (symutl.none_implicit && !dtimplicit[i].anyset)
    return FALSE;
  return TRUE;
} /* was_implicit */

/** \brief Return ptr to printable representation of the indicated PARAMETER.
    \param sptr symbol table pointer
 */
const char *
parmprint(int sptr)
{
  int dtype;
  const char *buf;
  INT save;

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
    save = CONVAL2G(sptr);
    CONVAL2P(sptr, CONVAL1G(sptr));
    buf = getprint(sptr);
    CONVAL2P(sptr, save);
  } else
    buf = getprint((int)CONVAL1G(sptr));
  STYPEP(sptr, ST_PARAM);
  DTYPEP(sptr, dtype);
  return buf;
}

/*---------------------------------------------------------------------*
 * getprint cannot be shared between FORTRAN and C                     *
 *---------------------------------------------------------------------*/

/**
 * Return ptr to printable representation of the indicated symbol.  For
 * symbols which are not constants, the name of the symbol is used.
 * Constants are converted into the appropriate character representation.
 */
static const char *
__log_print(INT val)
{
  if (val == 0)
    return ".FALSE.";
  return ".TRUE.";
}

/**
   \param sptr  symbol table pointer
 */
const char *
getprint(int sptr)
{
  int len; /* length of character string */
  static char *b = NULL;
  char *from, *end, *to;
  int c;
  INT num[4];
  int dtype;

  if (sptr == 0)
    return ".0.";
  if (sptr < 0)
    return ".neg.";
  if (sptr >= stb.stg_avail)
    return ".toobig.";
  if (STYPEG(sptr) != ST_CONST)
    return SYMNAME(sptr);

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
  case TY_BINT:
    sprintf(b, "%d_1", CONVAL2G(sptr));
    break;
  case TY_SINT:
    sprintf(b, "%d_2", CONVAL2G(sptr));
    break;
  case TY_INT:
    sprintf(b, "%d", CONVAL2G(sptr));
    break;
  case TY_INT8:
    num[0] = CONVAL1G(sptr);
    num[1] = CONVAL2G(sptr);
    ui64toax(num, b, 22, 0, 10);
    break;
  case TY_BLOG:
    sprintf(b, "%s_1", __log_print(CONVAL2G(sptr)));
    break;
  case TY_SLOG:
    sprintf(b, "%s_2", __log_print(CONVAL2G(sptr)));
    break;
  case TY_LOG:
  case TY_LOG8:
    sprintf(b, "%s", __log_print(CONVAL2G(sptr)));
    break;
  case TY_REAL:
    num[0] = CONVAL2G(sptr);
    cprintf(b, "%17.10e", (INT*)(size_t)(num[0]));
    break;

  case TY_DBLE:
    num[0] = CONVAL1G(sptr);
    num[1] = CONVAL2G(sptr);
    cprintf(b, "%24.17le", num);
    break;
  case TY_QUAD:
    num[0] = CONVAL1G(sptr);
    num[1] = CONVAL2G(sptr);
    num[2] = CONVAL3G(sptr);
    num[3] = CONVAL4G(sptr);
    cprintf(b, "%44.37qd", num);
    break;

  case TY_CMPLX:
    num[0] = CONVAL1G(sptr);
    cprintf(b, "%17.10e", (INT*)(size_t)(num[0]));
    b[17] = ',';
    b[18] = ' ';
    num[0] = CONVAL2G(sptr);
    cprintf(&b[19], "%17.10e", (INT*)(size_t)(num[0]));
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
    goto like_char;
  case TY_HOLL:
    sptr = CONVAL1G(sptr);
    dtype = DTYPEG(sptr);
    FLANG_FALLTHROUGH;
  case TY_CHAR:
  like_char:
    from = stb.n_base + CONVAL1G(sptr);
    len = string_length(dtype);
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

  case TY_PTR:
    strcpy(b, "address constant");
    break;

  default:
    interr("getprint:bad const dtype", sptr, 1);
  }
  return b;
}

/*--------------------------------------------------------------------------*/

#undef _PFG
#define _PFG(cond, str) \
  if (cond)             \
  fprintf(dfil, "  %s", str)

/**
   \param file the file
   \param sptr  symbol currently being dumped
*/
void
symdentry(FILE *file, int sptr)
{
  FILE *dfil;
  int dscptr;       /* ptr to dummy parameter descriptor list */
  char buff[210];   /* text buffer used to create output lines */
  char typeb[4096]; /* buffer for text of dtype */
  int stype;        /* symbol type of sptr  */
  int dtype;        /* data type of sptr */
  int i;

  dfil = file ? file : stderr;
  strcpy(buff, getprint(sptr));
  stype = STYPEG(sptr);
  dtype = DTYPEG(sptr);

  /* write first line containing symbol name, dtype, and stype: */

  if (stype == ST_CMBLK || stype == ST_LABEL || stype == ST_GENERIC ||
      stype == ST_NML || stype == ST_USERGENERIC || stype == ST_PD
  ) {
    fprintf(dfil, "\n%-40.40s %s\n", buff, stb.stypes[stype]);
  } else {
    *typeb = '\0';
    getdtype(dtype, typeb);
    fprintf(dfil, "\n%-40.40s %s %s\n", buff, typeb, stb.stypes[stype]);
  }
  if (UNAMEG(sptr)) {
    fprintf(dfil, "original uname:%s \n", stb.n_base + UNAMEG(sptr));
  }

  /* write second line:  */

  fprintf(dfil, "sptr: %d  hashlk: %d   nmptr: %d  dtype: %d  scope: %d", sptr,
          HASHLKG(sptr), NMPTRG(sptr), DTYPEG(sptr), SCOPEG(sptr));
  _PFG(INTERNALG(sptr), "internal");
  fprintf(dfil, "  lineno: %d", LINENOG(sptr));
  fprintf(dfil, "  enclfunc: %d\n", ENCLFUNCG(sptr));

  switch (stype) {
  case ST_UNKNOWN:
  case ST_IDENT:
  case ST_VAR:
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_STRUCT:
  case ST_UNION:
    fprintf(dfil, "dcld:%d   ccsym:%d   save:%d   ref:%d   dinit:%d",
            DCLDG(sptr), CCSYMG(sptr), SAVEG(sptr), REFG(sptr), DINITG(sptr));
    fprintf(dfil, "   vol:%d   ptrv:%d  cvlen:%d\n", VOLG(sptr), PTRVG(sptr),
            CVLENG(sptr));
    fprintf(dfil,
            "address: %" ISZ_PF "d   sc:%d(%s)   symlk: %d   midnum: %d   ",
            ADDRESSG(sptr), SCG(sptr),
            (SCG(sptr) <= SC_MAX) ? stb.scnames[SCG(sptr)] : "na", SYMLKG(sptr),
            MIDNUMG(sptr));
    fprintf(dfil, "socptr: %d   autobj: %d\n", SOCPTRG(sptr), AUTOBJG(sptr));
    fprintf(dfil, "addrtkn:%d  eqv:%d  hccsym:%d", ADDRTKNG(sptr), EQVG(sptr),
            HCCSYMG(sptr));
    fprintf(dfil, "  alloc:%d  arg:%d  seq:%d  nml:%d  assn:%d", ALLOCG(sptr),
            ARGG(sptr), SEQG(sptr), NMLG(sptr), ASSNG(sptr));
    fprintf(dfil, "\nprivate:%d", PRIVATEG(sptr));
    _PFG(MDALLOCG(sptr), "mdalloc");
#ifdef DYNAMICG
    _PFG(DYNAMICG(sptr), "dynamic");
#endif
    _PFG(POINTERG(sptr), "pointer");
    _PFG(F90POINTERG(sptr), "f90pointer");
    _PFG(TARGETG(sptr), "target");
    _PFG(NOMDCOMG(sptr), "nomdcom");
    _PFG(HIDDENG(sptr), "hidden");
    _PFG(IGNOREG(sptr), "ignore");
#ifdef LNRZDG
    _PFG(LNRZDG(sptr), "lnrzd");
#endif
    _PFG(TYP8G(sptr), "typ8");
    _PFG(ASSUMLENG(sptr), "assumlen");
    _PFG(ADJLENG(sptr), "adjlen");
    _PFG(PARAMG(sptr), "param");
    _PFG(PASSBYVALG(sptr), "passbyval");
    _PFG(PASSBYREFG(sptr), "passbyref");
    _PFG(CFUNCG(sptr), "cfunc");
    _PFG(STDCALLG(sptr), "stdcall");
    _PFG(ALLOCATTRG(sptr), "allocattr");
    _PFG(DESCARRAYG(sptr), "descarray");
    _PFG(CONTIGATTRG(sptr), "contigattr");
    _PFG(CLASSG(sptr), "class");
    _PFG(THREADG(sptr), "thread");
    _PFG(PROTECTEDG(sptr), "protected");
    if (stype == ST_VAR && CCSYMG(sptr))
      _PFG(EARLYSPECG(sptr), "earlyspec");
#ifdef PTRSTOREG
    _PFG(PTRSTOREG(sptr), "ptrstore");
#endif
    _PFG(DESCUSEDG(sptr), "descused");
    _PFG(ALLOCDESCG(sptr), "allocdesc");
    _PFG(RESHAPEDG(sptr), "reshaped");
    _PFG(INTERNREFG(sptr), "internref");

#ifdef TASKG
    _PFG(TASKG(sptr), "task");
#endif
#ifdef PARREFG
    _PFG(PARREFG(sptr), "parref");
#endif
#ifdef NOEXTENTG
    _PFG(NOEXTENTG(sptr), "noextent");
#endif
#if defined(TARGET_WIN)
    if (SCG(sptr) != SC_DUMMY) {
      if (DLLG(sptr) == DLL_EXPORT)
        fprintf(dfil, "  dllexport");
      else if (DLLG(sptr) == DLL_IMPORT)
        fprintf(dfil, "  dllimport");
    }
#endif
    fprintf(dfil, "  sdsc: %d", SDSCG(sptr));
    fprintf(dfil, "  ptroff: %d", PTROFFG(sptr));
    if (NEWARGG(sptr))
      fprintf(dfil, "  newarg: %d", NEWARGG(sptr));
    if (PARAMVALG(sptr))
      fprintf(dfil, "  paramval: %d", PARAMVALG(sptr));
    if (SCG(sptr) == SC_CMBLK)
      fprintf(dfil, "  cmblk: %d", CMBLKG(sptr));
    if (stype == ST_ARRAY) {
      fprintf(dfil, "\n");
      fprintf(dfil, "asumsz: %d   adjarr: %d   aftent: %d   assumshp: %d",
              (int)ASUMSZG(sptr), (int)ADJARRG(sptr), (int)AFTENTG(sptr),
              (int)ASSUMSHPG(sptr));
      fprintf(dfil, "  nodesc: %d", (int)NODESCG(sptr));
    }
    if (ADJARRG(sptr) || ADJLENG(sptr)) {
      fprintf(dfil, "\n");
      fprintf(dfil, "adjstrlk: %d", ADJSTRLKG(sptr));
    }
    fprintf(dfil, "  descr: %d\n", DESCRG(sptr));
    fprintf(dfil, "altname:%d\n", ALTNAMEG(sptr));
    if (SCG(sptr) == SC_DUMMY) {
      _PFG(RESULTG(sptr), "result   ");
      fprintf(dfil, "optarg:%d   intent:%s", OPTARGG(sptr),
              INTENTG(sptr) == INTENT_IN || INTENTG(sptr) == INTENT_OUT
                  ? (INTENTG(sptr) == INTENT_IN ? "IN" : "OUT")
                  : "INOUT");
      if (IGNORE_TKRG(sptr)) {
        fprintf(dfil, "   IGNORE_");
        if (IGNORE_TKRG(sptr) & IGNORE_T)
          fprintf(dfil, "T");
        if (IGNORE_TKRG(sptr) & IGNORE_K)
          fprintf(dfil, "K");
        if (IGNORE_TKRG(sptr) & IGNORE_R)
          fprintf(dfil, "R");
        if (IGNORE_TKRG(sptr) & IGNORE_D)
          fprintf(dfil, "D");
        if (IGNORE_TKRG(sptr) & IGNORE_M)
          fprintf(dfil, "M");
        if (IGNORE_TKRG(sptr) & IGNORE_C)
          fprintf(dfil, "C");
      }
      fprintf(dfil, "\n");
    }
    if (stype != ST_UNKNOWN && SCG(sptr) != SC_DUMMY && SOCPTRG(sptr))
      dmp_socs(sptr, dfil);

    if (DBGBIT(8, 4) && DTY(dtype) == TY_ARRAY) {
      /* print the declared array bounds */
      char comma = '(';
      ADSC *ad;
      int i;
      static char line[200], *p;
      ad = AD_DPTR(dtype);
      p = line;
      for (i = 0; i < AD_NUMDIM(ad); ++i) {
        *p++ = comma;
        comma = ',';
        if (AD_LWBD(ad, i)) {
          *p = '\0';
          getast(AD_LWBD(ad, i), p);
          p += strlen(p);
          *p++ = ':';
        }
        if (AD_UPBD(ad, i)) {
          *p = '\0';
          getast(AD_UPBD(ad, i), p);
          p += strlen(p);
        } else {
          *p++ = '*';
        }
      }
      *p++ = ')';
      *p = '\0';
      fprintf(dfil, "declared bounds %s\n", line);
    }
    break;

  case ST_STAG:
    fprintf(dfil, "dcld:%d   nest:%d\n", DCLDG(sptr), NESTG(sptr));
    break;

  case ST_NML:
    fprintf(dfil, "symlk: %d   address: %" ISZ_PF
                  "d   cmemf: %d   cmeml: %d   ref: %d\n",
            SYMLKG(sptr), ADDRESSG(sptr), CMEMFG(sptr), (int)CMEMLG(sptr),
            REFG(sptr));
    for (i = CMEMFG(sptr); i; i = NML_NEXT(i))
      fprintf(dfil, "    nml:%5d   sptr:%5d   %s\n", i, (int)NML_SPTR(i),
              SYMNAME(NML_SPTR(i)));
    break;

  case ST_MEMBER:
    fprintf(dfil,
            "address:%" ISZ_PF "d   symlk:%d   variant:%d   fnml:%d   ccsym:%d",
            ADDRESSG(sptr), SYMLKG(sptr), VARIANTG(sptr), (int)FNMLG(sptr),
            CCSYMG(sptr));
    fprintf(dfil, "\nencldtype:%d  sc:%d(%s)  private:%d", ENCLDTYPEG(sptr),
            SCG(sptr), (SCG(sptr) <= SC_MAX) ? stb.scnames[SCG(sptr)] : "na",
            PRIVATEG(sptr));
    _PFG(IGNOREG(sptr), "ignore");
    _PFG(POINTERG(sptr), "pointer");
#ifdef LNRZDG
    _PFG(LNRZDG(sptr), "lnrzd");
#endif
    _PFG(ALLOCG(sptr), "alloc");
    _PFG(SEQG(sptr), "seq");
    _PFG(ALLOCATTRG(sptr), "allocattr");
    _PFG(DESCARRAYG(sptr), "descarray");
    _PFG(NOPASSG(sptr), "nopass");
    _PFG(CONTIGATTRG(sptr), "contigattr");
    _PFG(CLASSG(sptr), "class");
    fprintf(dfil, "\n");
    fprintf(dfil, "ptroff:%d", PTROFFG(sptr));
    fprintf(dfil, "  midnum:%d", MIDNUMG(sptr));
    fprintf(dfil, "  sdsc:%d", SDSCG(sptr));
    fprintf(dfil, "  descr: %d", DESCRG(sptr));
    if (PASSG(sptr))
      fprintf(dfil, "  pass: %d", PASSG(sptr));
#ifdef IFACEG
    if (IFACEG(sptr))
      fprintf(dfil, "  iface: %d", IFACEG(sptr));
#endif
#ifdef VTABLEG
    if (VTABLEG(sptr))
      fprintf(dfil, "  vtable: %d", VTABLEG(sptr));
#endif
    fprintf(dfil, "\n");
    break;

  case ST_CMBLK:
    fprintf(dfil, "save:%d   dinit:%d   size:%" ISZ_PF "d   vol:%d   alloc:%d",
            SAVEG(sptr), DINITG(sptr), SIZEG(sptr), VOLG(sptr), ALLOCG(sptr));
    _PFG(THREADG(sptr), "thread");
    fprintf(dfil, "   seq:%d   private:%d\n", SEQG(sptr), PRIVATEG(sptr));
    fprintf(dfil, "hccsym:%d", HCCSYMG(sptr));
#ifdef PDALNG
    fprintf(dfil, "   pdaln:%d%s", PDALNG(sptr), PDALN_IS_DEFAULT(sptr) ? "(default)" : "");
#endif
    _PFG(HIDDENG(sptr), "hidden");
    _PFG(IGNOREG(sptr), "ignore");
    fprintf(dfil, "\n");
    fprintf(dfil, "midnum: %d   symlk: %d   cmemf: %d   cmeml: %d\n",
            MIDNUMG(sptr), SYMLKG(sptr), CMEMFG(sptr), (int)CMEMLG(sptr));
    fprintf(dfil, "altname: %d", ALTNAMEG(sptr));
    _PFG(MODCMNG(sptr), "modcmn");
    _PFG(QALNG(sptr), "qaln");
    _PFG(CFUNCG(sptr), "cfunc");
    _PFG(STDCALLG(sptr), "stdcall");
#ifdef TLSG
    _PFG(TLSG(sptr), "tls");
#endif /* TLSG */
#ifdef USE_MPC
    if (ETLSG(sptr))
      fprintf(file, " etls: %d", ETLSG(sptr));
#endif /* USE_MPC */
#if defined(TARGET_WIN)
    if (DLLG(sptr) == DLL_EXPORT)
      fprintf(dfil, "  dllexport");
    else if (DLLG(sptr) == DLL_IMPORT)
      fprintf(dfil, "  dllimport");
#endif
    fprintf(dfil, "\n");
    break;

  case ST_ENTRY:
    fprintf(dfil, "dcld: %d  ccsym: %d   entstd: %d   entnum: %d\n",
            DCLDG(sptr), CCSYMG(sptr), ENTSTDG(sptr), (int)ENTNUMG(sptr));
    fprintf(dfil, "endline: %d   symlk: %d   paramct: %d   dpdsc: %d\n",
            ENDLINEG(sptr), SYMLKG(sptr), PARAMCTG(sptr), DPDSCG(sptr));
    fprintf(dfil, "funcline: %d   bihnum: %d   fval: %d   pure: %d  impure: %d "
                  "  recur:%d\n",
            FUNCLINEG(sptr), BIHNUMG(sptr), FVALG(sptr), PUREG(sptr),
            IMPUREG(sptr), RECURG(sptr));
    fprintf(dfil, "adjarr:%d  aftent:%d  assumshp:%d", ADJARRG(sptr),
            AFTENTG(sptr), ASSUMSHPG(sptr));
    fprintf(dfil, "  private:%d", PRIVATEG(sptr));
    _PFG(ASSUMLENG(sptr), "assumlen");
    _PFG(ADJLENG(sptr), "adjlen");
    _PFG(POINTERG(sptr), "pointer");
    _PFG(PTRARGG(sptr), "ptrarg");
    _PFG(TYP8G(sptr), "typ8");
    _PFG(ELEMENTALG(sptr), "elemental");
    _PFG(DFLTG(sptr), "dflt");
    _PFG(ARETG(sptr), "aret");
    fprintf(dfil, "\n");
    fprintf(dfil, "   gsame: %d\n", (int)GSAMEG(sptr));
    fprintf(dfil, "altname: %d", ALTNAMEG(sptr));
#ifdef MVDESCG
    _PFG(MVDESCG(sptr), "mvdesc");
#endif
    _PFG(MSCALLG(sptr), "mscall");
#ifdef CREFP
    _PFG(CREFG(sptr), "cref");
    _PFG(NOMIXEDSTRLENG(sptr), "nomixedstrlen");
#endif
    _PFG(PASSBYVALG(sptr), "passbyval");
    _PFG(PASSBYREFG(sptr), "passbyref");
    _PFG(STDCALLG(sptr), "stdcall");
    _PFG(CFUNCG(sptr), "cfunc");
    _PFG(DECORATEG(sptr), "decorate");
#if defined(TARGET_WIN)
    if (DLLG(sptr) == DLL_EXPORT)
      fprintf(dfil, "  dllexport");
    else if (DLLG(sptr) == DLL_IMPORT)
      fprintf(dfil, "  dllimport");
#endif
    fprintf(dfil, "\n");
    fprintf(dfil, "Parameters:\n");
    dscptr = DPDSCG(sptr);
    for (i = PARAMCTG(sptr); i > 0; dscptr++, i--) {
      fprintf(dfil, "sptr =%5d", aux.dpdsc_base[dscptr]);
      if (aux.dpdsc_base[dscptr])
        fprintf(dfil, ", %s", SYMNAME(aux.dpdsc_base[dscptr]));
      fprintf(dfil, "\n");
    }
    break;

  case ST_PROC:
    fprintf(dfil, "dcld:%d   ref:%d   ccsym:%d   func:%d   typd:%d   pure:%d  "
                  "impure: %d\n",
            DCLDG(sptr), REFG(sptr), CCSYMG(sptr), FUNCG(sptr), TYPDG(sptr),
            PUREG(sptr), IMPUREG(sptr));
    fprintf(dfil, "fval:%d  recur:%d   private:%d   sc:%d(%s)\n", FVALG(sptr),
            RECURG(sptr), PRIVATEG(sptr), SCG(sptr),
            (SCG(sptr) <= SC_MAX) ? stb.scnames[SCG(sptr)] : "na");
    fprintf(dfil, "symlk: %d  paramct: %d  dpdsc: %d:", SYMLKG(sptr),
            PARAMCTG(sptr), DPDSCG(sptr));
    fprintf(dfil, "  private:%d  inmod:%d  fwdref:%d", PRIVATEG(sptr),
            INMODULEG(sptr), FWDREFG(sptr));
    _PFG(HCCSYMG(sptr), "hccsym");
    _PFG(ASSUMLENG(sptr), "assumlen");
    _PFG(ADJLENG(sptr), "adjlen");
    _PFG(POINTERG(sptr), "pointer");
    _PFG(PTRARGG(sptr), "ptrarg");
    _PFG(OPTARGG(sptr), "optarg");
    _PFG(TYP8G(sptr), "typ8");
    _PFG(ARETG(sptr), "aret");
    fprintf(dfil, "\n");
    fprintf(dfil, "altname: %d", ALTNAMEG(sptr));
#ifdef MVDESCG
    _PFG(MVDESCG(sptr), "mvdesc");
#endif
    _PFG(CFUNCG(sptr), "cfunc");
    _PFG(CSTRUCTRETG(sptr), "cstructret");
    _PFG(MSCALLG(sptr), "mscall");
#ifdef CREFP
    _PFG(CREFG(sptr), "cref");
    _PFG(NOMIXEDSTRLENG(sptr), "nomixedstrlen");
#endif
    _PFG(PASSBYVALG(sptr), "passbyval");
    _PFG(PASSBYREFG(sptr), "passbyref");
    _PFG(STDCALLG(sptr), "stdcall");
    _PFG(DECORATEG(sptr), "decorate");
#if defined(TARGET_WIN)
    if (SCG(sptr) != SC_DUMMY) {
      if (DLLG(sptr) == DLL_EXPORT)
        fprintf(dfil, "  dllexport");
      else if (DLLG(sptr) == DLL_IMPORT)
        fprintf(dfil, "  dllimport");
    }
#endif
    _PFG(ELEMENTALG(sptr), "elemental");
    _PFG(DFLTG(sptr), "dflt");
#ifdef NOCOG
    _PFG(NOCOG(sptr), "noco");
#endif
    _PFG(VARARGG(sptr), "vararg");
    fprintf(dfil, "\n");
    fprintf(dfil, "   gsame: %d", (int)GSAMEG(sptr));
    if (HCCSYMG(sptr) && INTENTG(sptr)) {
      fprintf(dfil, " intent:%s",
              INTENTG(sptr) == INTENT_IN || INTENTG(sptr) == INTENT_OUT
                  ? (INTENTG(sptr) == INTENT_IN ? "IN" : "OUT")
                  : "INOUT");
    }
    fprintf(dfil, "\n");
    if (DPDSCG(sptr) && PARAMCTG(sptr)) {
      fprintf(dfil, "Parameters:\n");
      dscptr = DPDSCG(sptr);
      for (i = PARAMCTG(sptr); i > 0; dscptr++, i--) {
        fprintf(dfil, "sptr =%5d", aux.dpdsc_base[dscptr]);
        if (aux.dpdsc_base[dscptr])
          fprintf(dfil, ", %s", SYMNAME(aux.dpdsc_base[dscptr]));
        fprintf(dfil, "\n");
      }
    }
    break;

  case ST_CONST:
    fprintf(dfil, "holl: %d   ", HOLLG(sptr));
    fprintf(dfil, "symlk: %d   address: %" ISZ_PF "d   conval1: %d   ",
            SYMLKG(sptr), ADDRESSG(sptr), CONVAL1G(sptr));
    if (DTYPEG(sptr) == DT_HOLL)
      fprintf(dfil, "conval2: %c\n", CONVAL2G(sptr));
    else
      fprintf(dfil, "conval2: %d\n", CONVAL2G(sptr));
    _PFG(PRIVATEG(sptr), "private");
    break;

  case ST_LABEL:
    fprintf(dfil, "rfcnt: %d  address: %" ISZ_PF
                  "d  symlk: %d  iliblk: %d  fmtpt: %d  agoto: %" ISZ_PF "d",
            RFCNTG(sptr), ADDRESSG(sptr), SYMLKG(sptr), ILIBLKG(sptr),
            FMTPTG(sptr), AGOTOG(sptr));
    _PFG(TARGETG(sptr), "target");
    _PFG(ASSNG(sptr), "assn");
    _PFG(VOLG(sptr), "vol");
    fprintf(dfil, "\n");
    break;

  case ST_STFUNC:
    fprintf(dfil, "symlk: %d   sfdsc: %x   excvlen: %d   sfast: %d\n",
            SYMLKG(sptr), (int)SFDSCG(sptr), (int)DTY(DTYPEG(sptr) + 1),
            SFASTG(sptr));
    break;
  case ST_PARAM:
    if (TY_ISWORD(DTY(dtype))) {
      /* fprintf(dfil, "conval1: 0x%lx\n", CONVAL1G(sptr)); */
      fprintf(dfil, "conval1: 0x%x  (%s)\n", CONVAL1G(sptr), parmprint(sptr));
    } else
      fprintf(dfil, "conval1: %d (sptr)\n", CONVAL1G(sptr));
    fprintf(dfil, "symlk:%d", SYMLKG(sptr));
    fprintf(dfil, "   private:%d", PRIVATEG(sptr));
    _PFG(DCLDG(sptr), "dcld");
    _PFG(TYPDG(sptr), "typd");
    _PFG(VAXG(sptr), "vax");
    _PFG(HIDDENG(sptr), "hidden");
    _PFG(IGNOREG(sptr), "ignore");
    _PFG(ENDG(sptr), "end");
    if (DTY(dtype) != TY_ARRAY)
      fprintf(dfil, "   conval2: %d(ast)\n", CONVAL2G(sptr));
    else
      fprintf(dfil, "   conval2: get_getitem_p(%d)\n", CONVAL2G(sptr));
    break;

  case ST_ISOC:
  case ST_ISOFTNENV:
  case ST_INTRIN:
    fprintf(dfil, "dcld:%d   expst:%d   typd:%d\n", (int)DCLDG(sptr),
            (int)EXPSTG(sptr), (int)TYPDG(sptr));
    *typeb = '\0';
    getdtype((int)ARGTYPG(sptr), typeb);
    fprintf(
        dfil, "pnmptr: %d   paramct: %d   ilm: %d   arrayf: %d   argtype: %s\n",
        PNMPTRG(sptr), PARAMCTG(sptr), (int)ILMG(sptr), ARRAYFG(sptr), typeb);
    *typeb = '\0';
    getdtype((int)INTTYPG(sptr), typeb);
    fprintf(dfil, "inttyp: %s   intast: %d", typeb, (int)INTASTG(sptr));
    _PFG(NATIVEG(sptr), "native");
    fprintf(dfil, "\n");
    break;

  case ST_USERGENERIC:
    fprintf(dfil, "gsame: %d  gncnt:%d  gndsc:%d  private:%d  gtype:%d\n",
            GSAMEG(sptr), GNCNTG(sptr), GNDSCG(sptr), PRIVATEG(sptr),
            GTYPEG(sptr));
    fprintf(dfil, "Overloaded funcs:\n");
    for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr))
      fprintf(dfil, "sptr =%5d, %s\n", SYMI_SPTR(dscptr),
              SYMNAME(SYMI_SPTR(dscptr)));
    break;

  case ST_GENERIC:
    fprintf(dfil,
            "expst:%d   typd:%d   gsame:%d  gsint:%d   gint:%d   gint8:%d\n",
            EXPSTG(sptr), TYPDG(sptr), GSAMEG(sptr), GSINTG(sptr), GINTG(sptr),
            GINT8G(sptr));
    fprintf(
        dfil,
        "greal:%d   gdble:%d   gquad:%d  gcmplx:%d   gdcmplx:%d   gqcmplx:%d\n",
        GREALG(sptr), GDBLEG(sptr), GQUADG(sptr), GCMPLXG(sptr), GDCMPLXG(sptr),
        GQCMPLXG(sptr));
    break;

  case ST_PD:
  case ST_IEEEARITH:
  case ST_IEEEEXCEPT:
  case ST_CRAY:
    fprintf(dfil, "pdnum: %d   intast: %d", (int)PDNUMG(sptr),
            (int)INTASTG(sptr));
    _PFG(DCLDG(sptr), "dcld");
    _PFG(NATIVEG(sptr), "native");
    fprintf(dfil, "\n");
    break;

  case ST_PLIST:
    fprintf(dfil, "ref: %d", REFG(sptr));
    if (SCG(sptr) == SC_CMBLK)
      fprintf(dfil, "  cmblk: %d", CMBLKG(sptr));
    _PFG(DINITG(sptr), "dinit");
    fprintf(dfil, "\n");
    fprintf(dfil, "address: %" ISZ_PF "d   pllen: %d   sc:%d(%s)   symlk: %d\n",
            ADDRESSG(sptr), PLLENG(sptr), SCG(sptr),
            (SCG(sptr) <= SC_MAX) ? stb.scnames[SCG(sptr)] : "na",
            SYMLKG(sptr));
    if (SYMNAME(sptr)[1] == 'J')
      fprintf(dfil, "swel: %d, deflab: %d\n", SWELG(sptr), (int)DEFLABG(sptr));
    break;

  case ST_ALIAS:
    fprintf(dfil, "symlk: %d  private: %d  ", SYMLKG(sptr), PRIVATEG(sptr));
    _PFG(IGNOREG(sptr), "ignore");
    fprintf(dfil, "\n");
    break;

  case ST_ARRDSC:
    fprintf(dfil, "descr: %d   secd: %d   secdsc: %d   slnk: %d  array %d",
            DESCRG(sptr), SECDG(sptr), SECDSCG(sptr), SLNKG(sptr),
            ARRAYG(sptr));
    fprintf(dfil, "   alnd: %d", ALNDG(sptr));
    fprintf(dfil, "\n");
    break;

  case ST_TYPEDEF:
    fprintf(dfil, "private:%d", PRIVATEG(sptr));
    _PFG(POINTERG(sptr), "pointer");
    _PFG(SEQG(sptr), "seq");
    _PFG(ALLOCFLDG(sptr), "allocfld");
    _PFG(CFUNCG(sptr), "bind(c)");
    _PFG(UNLPOLYG(sptr), "unlpoly");
    fprintf(dfil, "\n");
    break;
  case ST_MODULE:
    fprintf(dfil, "funcline: %d   ", FUNCLINEG(sptr));
    fprintf(dfil, "base: %d", CMEMFG(sptr));
    fprintf(dfil, "  private:%d", PRIVATEG(sptr));
    _PFG(DCLDG(sptr), "dcld");
    _PFG(DINITG(sptr), "dinit");
    _PFG(NEEDMODG(sptr), "needmod");
    _PFG(TYPDG(sptr), "typd");
#if defined(TARGET_WIN)
    if (DLLG(sptr) == DLL_EXPORT)
      fprintf(dfil, "  dllexport");
    else if (DLLG(sptr) == DLL_IMPORT)
      fprintf(dfil, "  dllimport");
#endif
    fprintf(dfil, "\n");
    break;
  case ST_OPERATOR:
    fprintf(dfil, "inkind:%d   pdnum:%d", INKINDG(sptr), PDNUMG(sptr));
    fprintf(dfil, "  gncnt:%d   gndsc:%d", GNCNTG(sptr), GNDSCG(sptr));
    fprintf(dfil, "  private:%d\n", PRIVATEG(sptr));
    fprintf(dfil, "Overloaded funcs:\n");
    for (dscptr = GNDSCG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr))
      fprintf(dfil, "sptr =%5d, %s\n", SYMI_SPTR(dscptr),
              SYMNAME(SYMI_SPTR(dscptr)));
    break;
  case ST_MODPROC:
    fprintf(dfil, "symlk: %d   symi: %d   gsame: %d private: %d\n",
            SYMLKG(sptr), SYMIG(sptr), (int)GSAMEG(sptr), PRIVATEG(sptr));
    fprintf(dfil, "Mapped from generics/operators:\n");
    for (dscptr = SYMIG(sptr); dscptr; dscptr = SYMI_NEXT(dscptr))
      fprintf(dfil, "sptr =%5d, %s\n", SYMI_SPTR(dscptr),
              SYMNAME(SYMI_SPTR(dscptr)));
    break;
  case ST_CONSTRUCT:
    fprintf(dfil, "funcline:%d\n", FUNCLINEG(sptr));
    break;

  case ST_BLOCK:
    fprintf(dfil, "startline %d  endline %d  enclfunc %d\n", STARTLINEG(sptr),
            ENDLINEG(sptr), ENCLFUNCG(sptr));
    fprintf(dfil, "startlab %d  endlab %d", STARTLABG(sptr), ENDLABG(sptr));
    fprintf(dfil, " autobj: %d", AUTOBJG(sptr));
    fprintf(dfil, "\n");
    break;

  default:
    interr("symdmp: bad symbol type", stype, 1);
  }
}

/**
 * dump symbol table for debugging purposes.  If full == TRUE, dump entire
 * symbol table, otherwise dump symtab beginning with user symbols.
 */
void
symdmp(FILE *dfil, LOGICAL full)
{
  int sptr; /* symbol currently being dumped */

  for (sptr = (full ? 1 : stb.firstusym); sptr < stb.stg_avail; sptr++)
    symdentry(dfil, sptr);
}

void
dmp_socs(int sptr, FILE *file)
{
  int p;
  int q;

  fprintf(file, "dmp_socs(%d)\n", sptr);
  if (!soc.base) {
    fprintf(file, "ERROR -  soc.base is null\n");
    return;
  }
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

/*--------------------------------------------------------------------------*
 * getccsym could be shared between C and Fortran                           *
 *--------------------------------------------------------------------------*/

/**
 * create (or possibly reuse) a compiler created symbol whose name is of the
 * form . 'letter' dddd where dddd is the decimal representation of n and
 * 'letter' is the character specified in the letter argument.
 */
int
getccsym(int letter, int n, SYMTYPE stype)
{
  char name[32];
  int sptr, i;
  const char *suffix = IPA_RECOMPILATION_SUFFIX;

  sprintf(name, ".%c%04d%s", letter, n, suffix); /* at least 4, could be more */
  i = 0;
  do {
    sptr = getsymbol(name);
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, stype);
      CCSYMP(sptr, 1);
      IGNOREP(sptr, 0);
      SCOPEP(sptr, stb.curr_scope);
      return sptr;
    }
    if (SCOPEG(sptr) == stb.curr_scope && STYPEG(sptr) == stype)
      return sptr;
    /* make up a new name */
    ++i;
    sprintf(&name[2], "%04d%03d%s", n, i, suffix);
  } while (1);
}

/**
 * create (never reuse) a compiler created symbol whose name is of the
 * form . 'letter' dddd where dddd is the decimal representation of n
 * and 'letter' is the character specified in the letter argument.
 */
int
getnewccsym(int letter, int n, int stype)
{
  char name[32];
  int sptr;

  sprintf(name, ".%c%04d", letter, n); /* at least 4, could be more */
  NEWSYM(sptr);
  NMPTRP(sptr, putsname(name, strlen(name)));
  STYPEP(sptr, stype);
  CCSYMP(sptr, 1);
  SCOPEP(sptr, 2);
  if (gbl.internal > 1)
    INTERNALP(sptr, 1);
  return (sptr);
} /* getnewccsym */

/**
 * create (never reuse) a compiler created symbol whose name is of the
 * form .
 */
static int
getnewccsym2(char *name, int n, int stype)
{
  int sptr;

  NEWSYM(sptr);
  NMPTRP(sptr, putsname(name, strlen(name)));
  STYPEP(sptr, stype);
  CCSYMP(sptr, 1);
  SCOPEP(sptr, 2);
  if (gbl.internal > 1)
    INTERNALP(sptr, 1);
  return (sptr);
} /* getnewccsym2 */

int
getnewccsymf(int stype, const char *fmt, ...)
{
  char buffer[MAXIDLEN + 1];
  va_list ap;

  va_start(ap, fmt);
  vsnprintf(buffer, sizeof buffer, fmt, ap);
  va_end(ap);
  buffer[sizeof buffer - 1] = '\0'; /* Windows vsnprintf bug work-around */

  return getnewccsym2(buffer, 0 /*unused*/, stype);
}

/**
 * similar to getccsym, but storage class is an argument. Calls getccsym
 * if the storage class is not private; if private, a 'p' is appended to
 * the name.
 */
int
getccsym_sc(int letter, int n, int stype, int sc)
{
  int sptr;

  if (sc != SC_PRIVATE)
    sptr = getccsym(letter, n, stype);
  else {
    char name[32];
    int i;
    const char *suffix = IPA_RECOMPILATION_SUFFIX;
    sprintf(name, ".%c%04dp%s", letter, n,
            suffix); /* at least 4, could be more */
    i = 0;
    do {
      sptr = getsymbol(name);
      if (STYPEG(sptr) == ST_UNKNOWN) {
        STYPEP(sptr, stype);
        CCSYMP(sptr, 1);
        IGNOREP(sptr, 0);
        SCOPEP(sptr, stb.curr_scope);
        break;
      }
      if (SCOPEG(sptr) == stb.curr_scope && STYPEG(sptr) == stype &&
          SCG(sptr) == sc)
        break;
      /* make up a new name */
      ++i;
      sprintf(&name[2], "%04d%03dp%s", n, i, suffix);
    } while (1);
  }
  SCP(sptr, sc);
  return (sptr);
}

/**
 * create (or possibly reuse) a compiler created symbol whose name is of the
 * form . "pfx" dddd where dddd is the decimal representation of n and
 * "pfx" is the prefix specified in the pfx argument.
 */
int
getccssym(const char *pfx, int n, int stype)
{
  char name[32];
  int sptr, i;
  const char *suffix = IPA_RECOMPILATION_SUFFIX;

  sprintf(name, ".%s%04d%s", pfx, n, suffix); /* at least 4, could be more */
  i = 0;
  do {
    sptr = getsymbol(name);
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, stype);
      CCSYMP(sptr, 1);
      IGNOREP(sptr, 0);
      SCOPEP(sptr, stb.curr_scope);
      return sptr;
    }
    if (SCOPEG(sptr) == stb.curr_scope && STYPEG(sptr) == stype)
      return sptr;
    /* make up a new name */
    ++i;
    sprintf(&name[strlen(pfx) + 1], "%04d%03d%s", n, i, suffix);
  } while (1);
}

/**
 * similar to getccssym, but storage class is an argument. Calls getccssym
 * if the storage class is not private; if private, a 'p' is appended to
 * the name.
 */
int
getccssym_sc(const char *pfx, int n, int stype, int sc)
{
  int sptr;

  if (sc != SC_PRIVATE)
    sptr = getccssym(pfx, n, stype);
  else {
    int i = 0;
    sptr = getsymf(".%s%04dp%s", pfx, n, IPA_RECOMPILATION_SUFFIX);
    do {
      if (STYPEG(sptr) == ST_UNKNOWN) {
        STYPEP(sptr, stype);
        CCSYMP(sptr, 1);
        IGNOREP(sptr, 0);
        SCOPEP(sptr, stb.curr_scope);
        break;
      }
      if (SCOPEG(sptr) == stb.curr_scope && STYPEG(sptr) == stype &&
          SCG(sptr) == sc)
        break;
      /* make up a new name */
      sptr = getsymf(".%s%04d%03dp%s", pfx, n, ++i, IPA_RECOMPILATION_SUFFIX);
    } while (1);
  }
  SCP(sptr, sc);
  return sptr;
}

/**
 * create (or possibly reuse) a compiler created symbol whose name is of the
 * form z_'letter'_'d' where 'd' is the decimal representation of n.
 * If the storage class is private, 'p' is appended to the name.
 */
int
getcctmp_sc(int letter, int n, int stype, int dtype, int sc)
{
  int i = 0;
  const char *scp = sc == SC_PRIVATE ? "p" : "";
  int sptr = getsymf("z_%c_%d%s%s", letter, n,
                     IPA_RECOMPILATION_SUFFIX, scp);

#if DEBUG
  assert(sc, "getcctmp_sc: SC_NONE", letter, 4);
#endif
  do {
    if (STYPEG(sptr) == ST_UNKNOWN) {
      STYPEP(sptr, stype);
      DTYPEP(sptr, dtype);
      DCLDP(sptr, 1);
      SCOPEP(sptr, stb.curr_scope);
      HCCSYMP(sptr, 1);
      SCP(sptr, sc);
      IGNOREP(sptr, 0);
#ifdef CUDAG
      if (CUDAG(gbl.currsub) & (CUDA_GLOBAL | CUDA_DEVICE)) {
        DEVICEP(sptr, 1);
      }
#endif
      return (sptr);
    }
    /* getcctmp_sc() is called from get_arr_tmp() in semutil2.c in a
     * search loop that checks dtypes for acceptable matches, so we'll
     * allow distinct dtypes here if both are arrays.
     */
    if (SCOPEG(sptr) == stb.curr_scope && STYPEG(sptr) == stype &&
        SCG(sptr) == sc &&
        (DTYPEG(sptr) == dtype ||
         (dtype > 0 && DTYPEG(sptr) > 0 && DTY(dtype) == TY_ARRAY &&
          DTY(DTYPEG(sptr)) == TY_ARRAY)))
      return sptr;
    /* make up a new name */
    sptr = getsymf("z_%c_%d_%d%s%s", letter, n, ++i,
                   IPA_RECOMPILATION_SUFFIX, scp);
  } while (1);
}

/**
 * Create a local compiler-created symbol - calls getcctmp_sc with the
 * storage class SC_LOCAL.
 */
int
getcctmp(int letter, int n, int stype, int dtype)
{
  int sptr;
  sptr = getcctmp_sc(letter, n, stype, dtype, SC_LOCAL);
  if (sem.block_scope) {
    sptr = block_local_sym(sptr);
    STYPEP(sptr, stype);
    DTYPEP(sptr, dtype);
  }
  return sptr;
}

/*--------------------------------------------------------------------------*
 * insert_sym is the same between C and Fortran                             *
 *--------------------------------------------------------------------------*/

/**
 * create new symbol table entry and insert it in the hash list immediately
 * in front of 'first':
 */
int
insert_sym(int first)
{
  int sptr, i, j;
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
      assert(j != 0, "insert_sym: bad hash", first, 4);
    HASHLKP(i, sptr);
  }

  SYMLKP(sptr, NOSYM); /* installsym for ftn also sets SYMLK */
  setimplicit(sptr);
  if (gbl.internal > 1)
    INTERNALP(sptr, 1);
  SCOPEP(sptr, stb.curr_scope);
  return sptr;
}

/**
 * create new symbol table entry and insert it in the hash list
 * in front of the head of the list containing 'first':
 */
int
insert_sym_first(int first)
{
  int sptr, i;
  INT hashval;
  char *np;

  NEWSYM(sptr);
  NMPTRP(sptr, NMPTRG(first));
  /* link newly created symbol in front of the hash list: */
  np = SYMNAME(first);
  i = strlen(np);
  HASH_ID(hashval, np, i);
  HASHLKP(sptr, stb.hashtb[hashval]);
  stb.hashtb[hashval] = sptr;
  SYMLKP(sptr, NOSYM); /* installsym for ftn also sets SYMLK */
  setimplicit(sptr);
  if (gbl.internal > 1)
    INTERNALP(sptr, 1);
  SCOPEP(sptr, stb.curr_scope);
  return sptr;
}

/**
 * return a compiler-created label -- user labels begin with '.', compiler-
 * created labels begin with '%'.  Compiler-created labels will be mapped
 * to fortran 77 labels by astout.
 */
int
getlab(void)
{
  int lab;
  while (TRUE) {
    lab = getsymf("%%L%05d", stb.lbavail--);
    if (STYPEG(lab) != ST_LABEL) {
#if DEBUG
      assert(STYPEG(lab) == ST_UNKNOWN, "getlab,sym not unk", lab, 3);
#endif
      STYPEP(lab, ST_LABEL);
      CCSYMP(lab, 1);
      SYMLKP(lab, 0);
      break;
    }
  }
  return lab;
}

/** \brief Return TRUE if sptr is in symi list represented by list.
 */
LOGICAL
sym_in_sym_list(int sptr, int list)
{
  for (; list != 0; list = SYMI_NEXT(list)) {
    if (SYMI_SPTR(list) == sptr) {
      return TRUE;
    }
  }
  return FALSE;
}

/** \brief Return TRUE if these two symi lists have the same sptrs in the same
 * order.
 */
LOGICAL
same_sym_list(int list1, int list2)
{
  for (;;) {
    if (list1 == 0) {
      return list2 == 0;
    }
    if (list2 == 0 || SYMI_SPTR(list1) != SYMI_SPTR(list2)) {
      return FALSE;
    }
    list1 = SYMI_NEXT(list1);
    list2 = SYMI_NEXT(list2);
  }
}

/**
 * \brief remove a symbol from its hash list
 */
void
pop_sym(int sptr)
{
  char *name;
  INT hashval;
  int s, j, l;

#if DEBUG
  if (DBGBIT(5, 1024))
    fprintf(gbl.dbgfil, "pop_sym(): sym %d\n", sptr);
#endif
  if (NMPTRG(sptr) == 0)
    return;
  name = SYMNAME(sptr);
  l = strlen(name);
  HASH_ID(hashval, name, l);
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
  HASHLKP(sptr, 0);
}

/**
 * \brief push a symbol onto a hash list
 */
void
push_sym(int sptr)
{
  char *name;
  int l;
  INT hashval;
#if DEBUG
  if (DBGBIT(5, 1024))
    fprintf(gbl.dbgfil, "push_sym(sym %d)\n", sptr);
#endif
  if (NMPTRG(sptr) == 0)
    return;
  name = SYMNAME(sptr);
  l = strlen(name);
  HASH_ID(hashval, name, l);
  HASHLKP(sptr, stb.hashtb[hashval]);
  stb.hashtb[hashval] = sptr;
} /* push_sym */

/** create a function ST item given a name */
SPTR
mkfunc(const char *nmptr)
{
  SPTR sptr;

  sptr = getsymbol(nmptr);
  STYPEP(sptr, ST_PROC);
  DTYPEP(sptr, DT_INT);
  SCP(sptr, SC_EXTERN);
  CCSYMP(sptr, 1);
  return (sptr);
}

/**
   \brief create a coercion function based on the data type.
 */
const char *
mk_coercion_func_name(int dtype)
{
  FtnRtlEnum rtlRtn;

  switch (DTY(dtype)) {
  case TY_BINT:
    rtlRtn = RTE_int1;
    break;
  case TY_SINT:
    rtlRtn = RTE_int2;
    break;
  case TY_INT:
    rtlRtn = RTE_int4;
    break;
  case TY_INT8:
    rtlRtn = RTE_int8;
    break;
  case TY_BLOG:
    rtlRtn = RTE_log1;
    break;
  case TY_SLOG:
    rtlRtn = RTE_log2;
    break;
  case TY_LOG:
    rtlRtn = RTE_log4;
    break;
  case TY_LOG8:
    rtlRtn = RTE_log8;
    break;
  case TY_REAL:
    rtlRtn = RTE_real4;
    break;
  case TY_DBLE:
    rtlRtn = RTE_real8;
    break;
  case TY_QUAD:
    rtlRtn = RTE_real16;
    break;
  case TY_CMPLX:
    rtlRtn = RTE_cmplx8;
    break;
  case TY_DCMPLX:
    rtlRtn = RTE_cmplx16;
    break;
  case TY_QCMPLX:
    rtlRtn = RTE_cmplx32;
    break;
  default:
    interr("mk_coercion_func_name: ty not allowed", DTY(dtype), 3);
    rtlRtn = RTE_no_rtn;
    break;
  }
  return (mkRteRtnNm(rtlRtn));
}

/** \brief Create a coercion function based on the data type.
 */
int
mk_coercion_func(int dtype)
{
  int sptr;

  sptr = sym_mkfunc_nodesc(mk_coercion_func_name(dtype), dtype);
  return sptr;
}

/**
 * create an external variable given a name and its data type.  A common block
 * of the same name is created and its member is the variable.  If the variable
 * already exists, just return it.
 */
int
mk_external_var(char *name, int dtype)
{
  int commonsptr = getsymbol(name);
  int sptr;

  if (STYPEG(commonsptr) != ST_UNKNOWN) {
#if DEBUG
    if (DTY(dtype) != TY_ARRAY)
      assert(STYPEG(commonsptr) == ST_VAR,
             "mk_external_var:scalar name conflict", commonsptr, 3);
    else
      assert(STYPEG(commonsptr) == ST_ARRAY,
             "mk_external_var:array name conflict", commonsptr, 3);
#endif
    return commonsptr;
  }

  STYPEP(commonsptr, ST_CMBLK);
  DCLDP(commonsptr, 1);
  HCCSYMP(commonsptr, 1);
  SCP(commonsptr, SC_CMBLK);
  pop_sym(commonsptr); /* hide common block from subsequent getsyms */

  sptr = getsymbol(name);
  if (DTY(dtype) != TY_ARRAY)
    STYPEP(sptr, ST_VAR);
  else
    STYPEP(sptr, ST_ARRAY);
  DTYPEP(sptr, dtype);
  DCLDP(sptr, 1);
  HCCSYMP(sptr, 1);
  SCP(sptr, SC_CMBLK);

  SYMLKP(commonsptr, gbl.cmblks); /* link into list of common blocks */
  gbl.cmblks = commonsptr;
  CMEMFP(commonsptr, sptr); /* add the variable to the common */
  CMEMLP(commonsptr, NOSYM);
  CMBLKP(sptr, commonsptr);
  SYMLKP(sptr, NOSYM);

  return sptr;
}

/**
   \brief determine if an argument is an argument to a given entry
   \param ent   entry sptr
   \param arg   argument sptr
 */
LOGICAL
is_arg_in_entry(int ent, int arg)
{
  int dscptr; /* ptr to dummy parameter descriptor list */
  int i;

#if DEBUG
  assert(STYPEG(ent) == ST_ENTRY, "is_arg_entry:need ST_ENTRY", ent, 3);
  assert(SCG(arg) == SC_DUMMY, "is_arg_entry:need SC_DUMMY", arg, 3);
#endif
  dscptr = DPDSCG(ent);
  for (i = PARAMCTG(ent); i > 0; dscptr++, i--)
    if (arg == *(aux.dpdsc_base + dscptr))
      return TRUE;

  return FALSE;
}

/**
   \brief determine if an argument $p is an $p of argument to a given entry
   \param ent  entry sptr
   \param arg  argument sptr
 */
LOGICAL
is_argp_in_entry(int ent, int arg)
{
  int dscptr; /* ptr to dummy parameter descriptor list */
  int i;

#if DEBUG
  assert(STYPEG(ent) == ST_ENTRY, "is_arg_entry:need ST_ENTRY", ent, 3);
  assert(SCG(arg) == SC_DUMMY, "is_arg_entry:need SC_DUMMY", arg, 3);
#endif
  dscptr = DPDSCG(ent);
  for (i = PARAMCTG(ent); i > 0; dscptr++, i--) {
    int sptr = *(aux.dpdsc_base + dscptr);
    if (arg == sptr)
      return TRUE;
    if (POINTERG(sptr) || ALLOCG(sptr)) {
      if (arg == MIDNUMG(sptr))
        return TRUE;
    }
  }

  return FALSE;
}

int
resolve_sym_aliases(int sptr)
{
  while (sptr > NOSYM && STYPEG(sptr) == ST_ALIAS) {
    sptr = SYMLKG(sptr);
  }
  return sptr;
}

LOGICAL
is_procedure_ptr(int sptr)
{
  sptr = resolve_sym_aliases(sptr);
  if (sptr > NOSYM && (POINTERG(sptr) || IS_PROC_DUMMYG(sptr))) {
    switch (STYPEG(sptr)) {
    case ST_PROC:
    case ST_ENTRY:
      /* subprograms aren't considered to be procedure pointers */
      break;
    default:
      return is_procedure_ptr_dtype(DTYPEG(sptr));
    }
  }
  return FALSE;
}

void
proc_arginfo(int sptr, int *paramct, int *dpdsc, int *iface)
{
  if (!is_procedure_ptr(sptr)) {
    if (STYPEG(sptr) == ST_GENERIC || STYPEG(sptr) == ST_INTRIN) {
      if (paramct)
        *paramct = 0;
      if (dpdsc)
        *dpdsc = 0;
      if (iface)
        *iface = sptr;
    } else if (IS_TBP(sptr)) {
      int mem, sptr2;
      mem = 0;
      sptr2 = get_implementation(TBPLNKG(sptr), sptr, 0, &mem);
      if (STYPEG(BINDG(mem)) == ST_OPERATOR ||
          STYPEG(BINDG(mem)) == ST_USERGENERIC) {
        mem = get_specific_member(TBPLNKG(sptr), sptr);
        sptr = VTABLEG(mem);
      } else
        sptr = sptr2;
      if (paramct)
        *paramct = PARAMCTG(sptr);
      if (dpdsc)
        *dpdsc = DPDSCG(sptr);
      if (iface)
        *iface = (IFACEG(mem)) ? IFACEG(mem) : sptr;
    } else if (STYPEG(sptr) == ST_MEMBER && CLASSG(sptr) && CCSYMG(sptr) &&
               VTABLEG(sptr) && BINDG(sptr)) {
      int mem;
      mem = sptr;
      sptr = VTABLEG(sptr);
      if (paramct)
        *paramct = PARAMCTG(sptr);
      if (dpdsc)
        *dpdsc = DPDSCG(sptr);
      if (iface)
        *iface = (IFACEG(mem)) ? IFACEG(mem) : sptr;
      return;
    } else if (STYPEG(sptr) == ST_PD) {
      if (paramct)
        *paramct = 0;
      if (dpdsc)
        *dpdsc = 0;
      if (iface)
        *iface = sptr;
    } else {
      if (paramct)
        *paramct = PARAMCTG(sptr);
      if (dpdsc)
        *dpdsc = DPDSCG(sptr);
      if (iface)
        *iface = sptr;
    }
  } else {
    int dtype, dtproc;
    dtype = DTYPEG(sptr);
#if DEBUG
    assert(DTY(dtype) == TY_PTR, "proc_arginfo, expected TY_PTR dtype", sptr,
           4);
#endif
    dtproc = DTY(dtype + 1);
#if DEBUG
    assert(DTY(dtproc) == TY_PROC, "proc_arginfo, expected TY_PROC dtype", sptr,
           4);
#endif
    if (paramct)
      *paramct = DTY(dtproc + 3);
    if (dpdsc)
      *dpdsc = DTY(dtproc + 4);
    if (iface)
      *iface = DTY(dtproc + 2);
  }
}

/**
 * \brief Compares two symbols by returning true if they both have equivalent
 * interfaces. Otherwise, return false. 
 *
 * If flag is set, then we also make sure that sym1 and sym2 have the same
 * symbol name.
 */
bool
cmp_interfaces(int sym1, int sym2, int flag)
{

  int i, paramct, paramct2, dpdsc, dpdsc2, psptr, psptr2;
  int iface1, iface2;

  if (sym1 <= NOSYM)
    return false;
 
  /* It's OK for the argument procedure pointer to point to NOSYM as long as 
   * the formal procedure pointer points to a valid symtab entry.
   * 
   * We assume the following:
   *
   * sym1 is the formal procedure pointer dummy argument
   * sym2 is the actual procedure pointer argument
   */
  if (sym2 <= NOSYM)
    return true;

  if (STYPEG(sym1) != ST_PROC) {
    int scope, alt_iface;
    int hash, hptr, len;
    char *symname;
    symname = SYMNAME(sym1);
    len = strlen(symname);
    HASH_ID(hash, symname, len);
    for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
      if (STYPEG(hptr) == ST_PROC && strcmp(symname, SYMNAME(hptr)) == 0) {
        alt_iface = hptr;
        if (alt_iface && (scope = test_scope(alt_iface))) {
          if (scope <= test_scope(sym1)) {
            sym1 = alt_iface;
            break;
          }
        }
      }
    }
  }
  if (STYPEG(sym2) != ST_PROC) {
    int scope, alt_iface;
    int hash, hptr, len;
    char *symname;
    symname = SYMNAME(sym2);
    len = strlen(symname);
    HASH_ID(hash, symname, len);
    for (hptr = stb.hashtb[hash]; hptr; hptr = HASHLKG(hptr)) {
      if (STYPEG(hptr) == ST_PROC && strcmp(symname, SYMNAME(hptr)) == 0) {
        alt_iface = hptr;
        if (alt_iface && (scope = test_scope(alt_iface))) {
          if (scope <= test_scope(sym2)) {
            sym2 = alt_iface;
            break;
          }
        }
      }
    }
  }

  iface1 = iface2 = paramct = paramct2 = dpdsc = dpdsc2 = 0;
  proc_arginfo(sym1, &paramct2, &dpdsc2, &iface1);
  proc_arginfo(sym2, &paramct, &dpdsc, &iface2);
  if (!iface1 || !iface2)
    return false;
  if (flag && strcmp(SYMNAME(iface1), SYMNAME(iface2)) != 0)
    return false;
  if (paramct != paramct2)
    return false;
  if (iface1 && iface1 == iface2)
    return true;
  if (!eq_dtype2(DTYPEG(FVALG(iface1)), DTYPEG(FVALG(iface2)), 0))
    return false; /* result types differ */
  for (i = 0; i < paramct; ++dpdsc, ++dpdsc2, ++i) {
    psptr2 = *(aux.dpdsc_base + dpdsc2);
    psptr = *(aux.dpdsc_base + dpdsc);
    if (!psptr || !psptr2 || STYPEG(psptr) != STYPEG(psptr2) ||
        strcmp(SYMNAME(psptr), SYMNAME(psptr2)) != 0)
      return false;
    if (STYPEG(psptr) == ST_PROC && STYPEG(psptr2) == ST_PROC) {
      if (!cmp_interfaces(psptr, psptr2, flag)) {
        return false;
      }
    } else if (!eq_dtype2(DTYPEG(psptr), DTYPEG(psptr2), 0)) {
      return false;
    }
  }
  return true;
}

/**
 * \brief Tests the characteristics between two interfaces.
 *
 * \param psptr is the first interface.
 *
 * \param pstr2 is the second interface.
 * 
 * \param flag is a bit mask that enforces/relaxes certain checks (see
 *        cmp_interface_flags enum in symtab.c). 
 *
 * \return true if the two characteristics are compatible, else false.
 */
bool 
compatible_characteristics(int psptr, int psptr2, cmp_interface_flags flag)
{

    if (!psptr || !psptr2) {
      return false;
    }

    if ( (((flag & RELAX_INTENT_CHK) == 0) && 
            INTENTG(psptr) != INTENTG(psptr2)) ||
        (((flag & CMP_OPTARG) != 0) && OPTARGG(psptr) != OPTARGG(psptr2)) ||
        ALLOCATTRG(psptr) != ALLOCATTRG(psptr2) ||
        PASSBYVALG(psptr) != PASSBYVALG(psptr2) ||
        ASYNCG(psptr) != ASYNCG(psptr2) || VOLG(psptr) != VOLG(psptr2) ||
        CLASSG(psptr) != CLASSG(psptr2) ||
        (((flag & RELAX_POINTER_CHK) == 0) && 
           POINTERG(psptr) != POINTERG(psptr2)) ||
        TARGETG(psptr) != TARGETG(psptr2) ||
        CONTIGATTRG(psptr) != CONTIGATTRG(psptr2)) {
        if (flag & CMP_SUBMOD_IFACE)
          generate_type_mismatch_errors(psptr, psptr2);

        return false;
    }

    if ((flag & RELAX_STYPE_CHK) == 0 && STYPEG(psptr) != STYPEG(psptr2)) {
      return false;
    }

    if (strcmp(SYMNAME(psptr), SYMNAME(psptr2)) != 0) {
      if (flag & CMP_SUBMOD_IFACE) {
        /* function may use itself name as a return variable, so no name 
           comparison for function return variables.
         */
        if (!RESULTG(psptr) && !RESULTG(psptr2))
          error(1057, ERR_Severe, gbl.lineno, SYMNAME(psptr2),SYMNAME(psptr));  
      }
      if ((flag & IGNORE_ARG_NAMES) == 0 && (flag & CMP_SUBMOD_IFACE) == 0)
        return false;
    }

    if (STYPEG(psptr) == ST_PROC && STYPEG(psptr2) == ST_PROC && 
        (flag & DEFER_IFACE_CHK) == 0) {
      if (!cmp_interfaces_strict(psptr, psptr2, (flag | CMP_OPTARG))) {
        return false;
      }
    } else if (DTY(DTYPEG(psptr)) == DTY(DTYPEG(psptr2)) && 
               (DTY(DTYPEG(psptr)) == TY_CHAR
               || DTY(DTYPEG(psptr)) == TY_NCHAR
               )) {
               /* check character objects only when they both 
                * have constant lengths or at least one is assumed shape.
                */
               int d1 = DTYPEG(psptr);
               int a1 = DTY(d1+1);
               int d2 = DTYPEG(psptr2);
               int a2 = DTY(d2+1);
               if ((a1 == 0 || a2 == 0 || 
                   (A_TYPEG(a1) == A_CNST && A_TYPEG(a2) == A_CNST)) &&
                   !eq_dtype2(d1, d2, 0)) {
                   return FALSE;
               }
    } else if (!eq_dtype2(DTYPEG(psptr), DTYPEG(psptr2), 0)) {
      if (flag & CMP_SUBMOD_IFACE) {
        /* check whether variable type matches for:
           1. argument type
           2. function return type
         */
        if ((DTY(DTYPEG(psptr)) != DTY(DTYPEG(psptr2))) || 
            (DDTG(DTYPEG(psptr)) != DDTG(DTYPEG(psptr2)) && 
             DTYG(DTYPEG(psptr)) != DTYG(DTYPEG(psptr2))))
          generate_type_mismatch_errors(psptr, psptr2); 
      }
      return FALSE;
    } else if (DTY(DTYPEG(psptr)) == TY_ARRAY && 
               DTY(DTYPEG(psptr2)) == TY_ARRAY) {
        /* Check extents of array dimensions. Note: the call to eq_dtype2()
         * above checks type and rank.
         */
        ADSC *ad, *ad2;
        int i, ast, ast2, numdim;

        ad = AD_DPTR(DTYPEG(psptr));
        numdim = AD_NUMDIM(ad);

        ad2 = AD_DPTR(DTYPEG(psptr2));

        for(i=0; i < numdim; ++i) {
          ast = AD_EXTNTAST(ad, i);
          ast2 = AD_EXTNTAST(ad2, i);
          if (A_TYPEG(ast) == A_CNST && A_TYPEG(ast2) == A_CNST &&
              CONVAL2G(A_SPTRG(ast)) != CONVAL2G(A_SPTRG(ast2))) {
              return false;
          }
        }
    }
   
    return true;
}

/**
 * \brief Same as cmp_interfaces() except we also compare the characteristics as
 * defined in "12.2 Characteristics of procedures" in F2003 Spec. 
 */
bool
cmp_interfaces_strict(SPTR sym1, SPTR sym2, cmp_interface_flags flag)
{
  int i, paramct, paramct2, dpdsc, dpdsc2, psptr, psptr2;
  int iface1, iface2, j;
  bool relax1, relax2; 

  iface1 = iface2 = paramct = paramct2 = dpdsc = dpdsc2 = 0;
  proc_arginfo(sym1, &paramct, &dpdsc, &iface1);
  proc_arginfo(sym2, &paramct2, &dpdsc2, &iface2);

  if (FVALG(sym1) && FVALG(sym2) && dpdsc > 0 && dpdsc2 > 0) {
    /* Check characteristics of results if applicable. We do this here
     * to handle the case where one symbol will have its result in argument
     * 1 and another symbol will not. This occurs when one symbol is a
     * function and another symbol is a function interface (i.e., we do not
     * put the function result into argument 1 for interfaces). We then
     * adjust parameter counts and argument descriptors when the result is
     * in an argument so parameter counts are consistent between the two
     * symbols.
     */
    if (paramct > 0) {
      psptr = aux.dpdsc_base[dpdsc];
      if (FVALG(sym1) == psptr) {
          paramct--;
          dpdsc++;
       }
    }
    if (paramct2 > 0) {
      psptr2 = aux.dpdsc_base[dpdsc2];
      if (FVALG(sym2) == psptr2) {
        paramct2--;
        dpdsc2++;
      }
     }
     psptr = FVALG(sym1);
     psptr2 = FVALG(sym2);
     if (!compatible_characteristics(psptr, psptr2, flag)) {
       return false;
     }
  }

  /* we may have added descriptors such as type descriptors to the
   * argument descriptor. Do not count them.
   */

  for (j = i = 0; i < paramct; ++i) {
    psptr = aux.dpdsc_base[dpdsc + i];
    if (CCSYMG(psptr) && CLASSG(psptr)) {
      ++j;
    }
  }
  paramct -= j;

  for (j = i = 0; i < paramct2; ++i) {
    psptr2 = aux.dpdsc_base[dpdsc2 + i];
    if (CCSYMG(psptr2) && CLASSG(psptr2)) {
      ++j;
    }
  }
  paramct2 -= j;

  if (PUREG(sym1) != PUREG(sym2) || IMPUREG(sym1) != IMPUREG(sym2)) {
    if (flag & CMP_SUBMOD_IFACE)
      error(1060, ERR_Severe, gbl.lineno, "PURE function prefix",SYMNAME(sym1));    
 
    relax1 = (flag & RELAX_PURE_CHK_1) != 0;
    relax2 = (flag & RELAX_PURE_CHK_2) != 0;

    if (!relax1 && !relax2) {
      /* both arguments must have matching pure/impure attributes */
      return false;
    }
    if (relax1 != relax2 && PUREG(sym1) != PUREG(sym2)) {
      if (!relax1 && PUREG(sym1)) {
        /* argument 1 has pure but argument 2 does not. */
        return false;
      }
      if (!relax2 && PUREG(sym2)) {
        /* argument 2 has pure but argument 1 does not */
        return false;
      }
    }  
  }
  if (paramct != paramct2) {
    if (flag & CMP_SUBMOD_IFACE)
      error(1059, ERR_Severe, gbl.lineno, SYMNAME(sym1), NULL);
    return false;
  }
  if (CFUNCG(sym1) != CFUNCG(sym2)){
    if (flag & CMP_SUBMOD_IFACE)
      error(1060, ERR_Severe, gbl.lineno, "BIND attribute", SYMNAME(sym1));
    return false;
  }
  if (ELEMENTALG(sym1) != ELEMENTALG(sym2)){
    if (flag & CMP_SUBMOD_IFACE)
      error(1060, ERR_Severe, gbl.lineno, "ELEMENTAL function prefix",SYMNAME(sym1));
    return false;
  }
 
  if ((FVALG(sym1) && !FVALG(sym2)) || (FVALG(sym2) && !FVALG(sym1))) {
    if (flag & CMP_SUBMOD_IFACE)
      error(1058, ERR_Severe, gbl.lineno, SYMNAME(sym1), NULL);
    return false;
  }

  if (!iface1 || !iface2)
    return false;
  if ( ((flag & CMP_IFACE_NAMES) != 0) && strcmp(SYMNAME(iface1), 
       SYMNAME(iface2)) != 0)
    return false;
  if (iface1 && iface1 == iface2)
    return true;

  for (i = 0; i < paramct; ++dpdsc, ++dpdsc2, ++i) {
    psptr2 = aux.dpdsc_base[dpdsc2];
    psptr = aux.dpdsc_base[dpdsc];

    if (!compatible_characteristics(psptr, psptr2, flag)) {
      return false;
    }
 
  }
  return true;
}

/** \brief Copy flags from one symbol to another symbol.
 *
 * This routine is the same as dup_sym() except it preserves the symbol's
 * name, hash link, scope, and name pointer. In other words, it copies all but 
 * 4 flags from one symbol to another. The 4 flags that are not copied are 
 * the hashlk, symlk, scope, and nmptr.
 *
 * \param dest is the receiving symbol table pointer of the flags.
 * \param src is the source symbol table pointer of the flags.
 */
void
copy_sym_flags(SPTR dest, SPTR src)
{

  SYM *destSym;
  SPTR hashlk;
  SPTR symlk;
  INT nmptr;
  INT scope;

  destSym = (stb.stg_base + dest);
  hashlk = destSym->hashlk;
  symlk = destSym->symlk;
  nmptr = destSym->nmptr;
  scope = destSym->scope;

  *destSym = *(stb.stg_base + src);

  destSym->hashlk = hashlk;
  destSym->symlk = symlk;
  destSym->nmptr = nmptr;
  destSym->scope = scope;

}
  
/**
 * replace contents of a symbol with values defining every field while ensuring
 * values necessary for the hashing function are saved and restored.
 */
void
dup_sym(int new, SYM *content)
{
  int hashlk, nmptr, scope;

  hashlk = HASHLKG(new);
  nmptr = NMPTRG(new);
  scope = SCOPEG(new);
  *(stb.stg_base + new) = *content;
  HASHLKP(new, hashlk);
  NMPTRP(new, nmptr);
  SCOPEP(new, scope);
}

/** \Brief Create a duplicate of this sym and return it */
int
insert_dup_sym(int sptr)
{
  int new_sptr = insert_sym(sptr);
  dup_sym(new_sptr, &stb.stg_base[sptr]);
  return new_sptr;
}

/** If mod is a submodule, return the module it is a submodule of.
 *  If it's a module, return mod. Otherwise 0.
 */
SPTR
get_ancestor_module(SPTR mod)
{
  if (mod == 0 || STYPEG(mod) != ST_MODULE)
    return 0;
  for (;;) {
    SPTR parent = PARENTG(mod);
    if (parent == 0)
      return mod;
    mod = parent;
  }
}

/** return the symbol of the explicit interface of the ST_PROC
 */
SPTR 
find_explicit_interface(SPTR s) {
  SPTR sptr;
  for (sptr = HASHLKG(s); sptr; sptr = HASHLKG(sptr)) {
    /* skip noise sptr with same string name*/
    if (NMPTRG(sptr) != NMPTRG(s))
      continue;

    if (!INMODULEG(sptr))
      break;
    if (SEPARATEMPG(sptr))
      return sptr;
  }

  return 0;
}

/** \brief Instantiate a copy of a separate module subprogram's 
           declared interface as part of the MODULE PROCEDURE's 
           definition (i.e., implement what would have taken place 
           had the subprogram been defined with a MODULE SUBROUTINE
           or MODULE FUNCTION with a compatible interface).
 */
SPTR
instantiate_interface(SPTR iface)
{
  int dummies;
  SPTR fval, proc;
  DEC_DEF_MAP *dec_def_map;
  proc = insert_dup_sym(iface);
  gbl.currsub = proc;

  SCOPEP(proc, SCOPEG(find_explicit_interface(proc)));
  dummies = PARAMCTG(iface);
  NEW(dec_def_map, DEC_DEF_MAP, dummies);
  fval = NOSYM;

  STYPEP(proc, ST_ENTRY);
  INMODULEP(proc, TRUE);

  if (FVALG(iface) > NOSYM) {
    fval = insert_sym_first(FVALG(iface));
    dup_sym(fval, &stb.stg_base[FVALG(iface)]);

    /* Needs to disable hidden attribute to enable proc to 
     * access derived type members
     */
    HIDDENP(fval, 0);
    IGNOREP(fval, 0);

    SCOPEP(fval, proc);
    if (ENCLFUNCG(FVALG(iface)) == iface) {
      ENCLFUNCP(fval, proc);
    }
    FVALP(proc, fval);
    ++aux.dpdsc_avl; /* always reserve one for fval */
  }

  if (dummies > 0 || fval > NOSYM) {
    int iface_dpdsc = DPDSCG(iface);
    int proc_dpdsc = aux.dpdsc_avl;
    int j;

    aux.dpdsc_avl += dummies;
    NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
         aux.dpdsc_size + dummies + 100);
    DPDSCP(proc, proc_dpdsc);
    if (fval > NOSYM) {
      aux.dpdsc_base[proc_dpdsc - 1] = fval;
    }
    for (j = 0; j < dummies; ++j) {
      SPTR arg = aux.dpdsc_base[iface_dpdsc + j];
      if (arg > NOSYM) {
        dec_def_map[j].dec_sptr = arg;
        arg = insert_dup_sym(arg);
        dec_def_map[j].def_sptr = arg;
        SCOPEP(arg, proc);
        if (DTY(DTYPEG(arg)) == TY_ARRAY && ASSUMSHPG(arg)) {
          DTYPE elem_dt = array_element_dtype(DTYPEG(arg));
          int arr_dsc = mk_arrdsc();
          DTY(arr_dsc + 1) = elem_dt;
          DTYPEP(arg, arr_dsc);
          trans_mkdescr(arg);
          ALLOCP(arg, TRUE);
          /* needs to tie the array descritor with the symbol arg here*/
          get_static_descriptor(arg);
        }
        if (ALLOCATTRG(arg) || POINTERG(arg)) {
          if (!SDSCG(arg))
            get_static_descriptor(arg);
          if (!PTROFFG(arg))
            get_all_descriptors(arg);
        }

        HIDDENP(arg, 0);
        IGNOREP(arg, 0);
        if (ENCLFUNCG(arg) == iface) {
          ENCLFUNCP(arg, proc);
        }
      }
      aux.dpdsc_base[proc_dpdsc + j] = arg;
    }
  }

  if (ADJARRG(fval)) {
    ADSC *ad;
    int arr_dsc;
    DTYPE elem_dt;
    ad = AD_DPTR(DTYPEG(FVALG(iface)));
    update_arrdsc(fval, dec_def_map, dummies);
    elem_dt = array_element_dtype(DTYPEG(iface));
    arr_dsc = mk_arrdsc();
    DTY(arr_dsc + 1) = elem_dt;
    DTYPEP(fval, arr_dsc); 
    trans_mkdescr(fval);
  }

  FREE(dec_def_map);
  
  return proc;
}

/** \brief Update array bound AST SPTRs (old_sptr) using newly created SPTRs 
           (new_sptr) by referring to DEC_DEF_MAP. The DEC_DEF_MAP is a struct 
           which contains mapping info from the old_sptr to new_sptr.
 */
static void 
update_arrdsc(SPTR s, DEC_DEF_MAP *smap, int num_dummies) {
  int i, j;
  SPTR dec_sptr_lwbd, dec_sptr_upbd;
  ADSC *ad;
  ad = AD_DPTR(DTYPEG(s));
  sem.arrdim.ndim = AD_NUMDIM(ad);
  sem.arrdim.ndefer = AD_DEFER(ad);
  for (i = 0; i < sem.arrdim.ndim; ++i) {
    /* restore arrdsc bound ast info from *ad */
    sem.bounds[i].lwast = AD_LWAST(ad, i);
    sem.bounds[i].upast = AD_UPAST(ad, i);

    /* update arrdsc bound ast info */
    dec_sptr_lwbd = A_SPTRG(AD_LWBD(ad, i));
    dec_sptr_upbd = A_SPTRG(AD_UPBD(ad, i));
    for (j = 0; j < num_dummies; ++j) {  
      if (dec_sptr_lwbd == smap[j].dec_sptr)
        sem.bounds[i].lwast = mk_id(smap[j].def_sptr);
      if (dec_sptr_upbd == smap[j].dec_sptr)
        sem.bounds[i].upast = mk_id(smap[j].def_sptr);
    }
  }
}

/**
 * reinitialize a symbol
 */
void
reinit_sym(int sptr)
{
  int nmptr, scope, hashlk;
  hashlk = HASHLKG(sptr);
  nmptr = NMPTRG(sptr);
  scope = SCOPEG(sptr);
  BZERO(stb.stg_base + sptr, char, sizeof(SYM));
  HASHLKP(sptr, hashlk);
  NMPTRP(sptr, nmptr);
  SCOPEP(sptr, scope);
} /* reinit_sym */

char *
sym_strsave(const char *s)
{
  int i;
  char *p;

  i = strlen(s);
  NEW(p, char, i + 1);
  strcpy(p, s);
  return p;
}

static int manglecount = 0;

/**
 * create a distinct mangled name
 */
char *
mangle_name(const char *basename, const char *purpose)
{
  int length, i, j;
  int sptr, hashval;
  int tail_index;
  static char name[MAXIDLEN + 1];
  int max_idlen = MAXIDLEN;

/* we use the convention: basename$purpose%d
 * if purpose is absent, just basename%d
 * to deal with the length issue:
 * +  if the length of the name exceeds the max allowed, truncate.
 * +  if a clash occurs, append to or replace the 'tail' of the name
 *    with a digit string (max 5 digits).
)    */

  if (flg.standard) {
    max_idlen = STANDARD_MAXIDLEN;
  }
  length = strlen(basename);
  if (length > max_idlen)
    length = max_idlen;
  memcpy(name, basename, length);
  if (purpose) {
    j = length + 1 + strlen(purpose); /* basname$purpose */
    if (j > max_idlen)
      j = max_idlen;
    j -= length; /* room for $purpose */
    if (j > 0) {
      name[length] = '$';
      memcpy(&name[length + 1], purpose, j - 1);
      length += j;
    }
  }
  name[length] = '\0';
  tail_index = length; /* append digit string */
  if ((max_idlen - length) < 5)
    tail_index = max_idlen - 5; /* no room replace last 5 characters */
  for (i = 0;;) {
    length = strlen(name);
    HASH_ID(hashval, name, length);
    for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
      if (IGNOREG(sptr) && stb.curr_scope == SCOPEG(sptr))
        continue;
      if (strcmp(name, SYMNAME(sptr)) == 0)
        break;
    }
    if (sptr == 0)
      break;
    ++i;
    ++manglecount;
    if (manglecount >= 10000)
      manglecount = 0;
    sprintf(&name[tail_index], "%d", manglecount);
    assert(i < 10000, "mangle_name: too many temps", 0, 4);
  }
  return name;
}

/**
   same as mangle_name, except only clash for members of the same derived type
 */
char *
mangle_name_dt(const char *basename, const char *purpose, int encldtype)
{
  int length, i, j;
  int sptr, hashval;
  int tail_index;
  static char name[MAXIDLEN + 1];
  int max_idlen = MAXIDLEN;

  if (flg.standard) {
    max_idlen = STANDARD_MAXIDLEN;
  }

  length = strlen(basename);
  if (length > max_idlen)
    length = max_idlen;
  memcpy(name, basename, length);
  if (purpose) {
    j = length + 1 + strlen(purpose); /* basname$purpose */
    if (j > max_idlen)
      j = max_idlen;
    j -= length; /* room for $purpose */
    if (j > 0) {
      name[length] = '$';
      memcpy(&name[length + 1], purpose, j - 1);
      length += j;
    }
  }
  name[length] = '\0';
  tail_index = length; /* append digit string */
  if ((max_idlen - length) < 5)
    tail_index = max_idlen - 5; /* no room replace last 5 characters */
  for (i = 0;;) {
    length = strlen(name);
    HASH_ID(hashval, name, length);
    for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
      if (STYPEG(sptr) != ST_MEMBER || ENCLDTYPEG(sptr) != encldtype)
        continue; /* no clash */
      if (strcmp(name, SYMNAME(sptr)) == 0)
        break;
    }
    if (sptr == 0)
      break;
    ++i;
    sprintf(&name[tail_index], "%d", i);
    assert(i < 10000, "mangle_name: too many temps", 0, 4);
  }
  return name;
}

/**
   can be called after name mangling if the original name needs to be saved in
   the symbol table, (for instance, for debug symbols.)
 */
void
save_uname(int newsptr, INT oldnmptr)
{
  if (!newsptr) {
    interr("save_uname bad sptr", newsptr, 3);
    return;
  }

  if (!UNAMEG(newsptr)) {
    UNAMEP(newsptr, oldnmptr); /* save original user name */
  } else {
    ; /* do nothing, name is being changed again. */
  }

#if DEBUG
  assert(oldnmptr <= stb.namavl, "save_uname: bad nmptr", oldnmptr, 3);
#endif
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

/**
 * switch from ST_CRAFT/ST_CRAY to ST_PD or vice versa
 */
void
change_predefineds(int stype, LOGICAL remove)
{
  int first, last;
  int s;

  hpf_library_stat(&first, &last, stype);
  if (first == 0)
    return;
  if (!remove) {
    for (s = first; s <= last; s++)
      if (STYPEG(s) == stype)
        STYPEP(s, ST_PD);
  } else {
    for (s = first; s <= last; s++)
      if (STYPEG(s) == ST_PD)
        BCOPY(stb.stg_base + s, init_sym + s, SYM, 1);
  }
}

#if DEBUG
void
dbg_symdentry(int sptr)
{
  symdentry(stderr, sptr);
}
#endif

void rw_sym_state(RW_ROUTINE, RW_FILE)
{
  int nw;

  RW_FD(stb.hashtb, stb.hashtb, 1);
  RW_SCALAR(stb.firstusym);
  RW_SCALAR(stb.stg_avail);
  RW_SCALAR(stb.stg_cleared);
  RW_FD(stb.stg_base, SYM, stb.stg_avail);

  RW_SCALAR(stb.namavl);
  RW_FD(stb.n_base, char, stb.namavl);

  RW_SCALAR(stb.lbavail);

  RW_SCALAR(aux.dpdsc_avl);
  RW_FD(aux.dpdsc_base, int, aux.dpdsc_avl);

  RW_SCALAR(aux.arrdsc_avl);
  RW_FD(aux.arrdsc_base, int, aux.arrdsc_avl);

  RW_SCALAR(aux.nml_avl);
  RW_FD(aux.nml_base, NMLDSC, aux.nml_avl);

  RW_SCALAR(aux.symi_avl);
  RW_FD(aux.symi_base, SYMI, aux.symi_avl);

  RW_FD(aux.list, int, ST_MAX + 1);

  RW_SCALAR(gbl.cmblks);

  RW_SCALAR(soc.avail);
  if (soc.avail > 1) {
    if (ISREAD()) {
      if (soc.size == 0) {
        soc.size = soc.avail + 100;
        NEW(soc.base, SOC_ITEM, soc.size);
      } else {
        NEED(soc.avail, soc.base, SOC_ITEM, soc.size, soc.avail + 1000);
      }
    }
    RW_FD(soc.base, SOC_ITEM, soc.avail);
  }
}

/**
 * Compilation is finished - deallocate storage, close files, etc.
 */
void
symtab_fini(void)
{
  FREE(stb.stg_base);
  stb.stg_size = 0;
  FREE(stb.n_base);
  stb.n_size = 0;
  STG_DELETE(stb.dt);
  FREE(stb.w_base);
  stb.w_size = 0;
  fini_chartab();
  if (aux.dpdsc_base) {
    FREE(aux.dpdsc_base);
    aux.dpdsc_avl = aux.dpdsc_size = 0;
  }
  if (aux.arrdsc_base) {
    FREE(aux.arrdsc_base);
    aux.arrdsc_avl = aux.arrdsc_size = 0;
  }
  if (aux.nml_base) {
    FREE(aux.nml_base);
    aux.nml_avl = aux.nml_size = 0;
  }
  if (aux.dvl_base) {
    FREE(aux.dvl_base);
    aux.dvl_avl = aux.dvl_size = 0;
  }
  if (aux.symi_base) {
    FREE(aux.symi_base);
    aux.symi_avl = aux.symi_size = 0;
  }
  if (soc.base) {
    FREE(soc.base);
    soc.avail = soc.size = 0;
  }
  if (save_dtimplicit) {
    FREE(save_dtimplicit);
    dtimplicitsize = dtimplicitstack = 0;
  }
} /* symtab_fini */

/**
 * call this when -standard is set
 */
void
symtab_standard(void)
{
  /* remove _TY_INT from TY_LOG types */
  dttypes[TY_BLOG] &= ~_TY_INT;
  dttypes[TY_SLOG] &= ~_TY_INT;
  dttypes[TY_LOG] &= ~_TY_INT;
  dttypes[TY_LOG8] &= ~_TY_INT;
} /* symtab_standard */

/**
 * call this when -nostandard is set; undo what symtab_standard does
 */
void
symtab_nostandard(void)
{
  /* add _TY_INT to TY_LOG types */
  dttypes[TY_BLOG] |= _TY_INT;
  dttypes[TY_SLOG] |= _TY_INT;
  dttypes[TY_LOG] |= _TY_INT;
  dttypes[TY_LOG8] |= _TY_INT;
} /* symtab_nostandard */

/** \brief Adding intrinsics, predeclareds, etc. to symini_ftn has the effect
           of rendering existng .mod files incompatible because the values of
           stb.firstosym will be different.

    stb.firstosym is computed from:
    1. the number of intrinsics, predeclareds defined by symini_ftn, and
    2. the number of predeclared constants/symbols created in
    symtab.c:sym_init() (e.g., 1.0, 0, 2, etc.)

    It should be possible to map the 'firstosym' symbols from the
    previous symini to the current symini by just subscripting a table
    using the old symbol as the index.  The purpose of oldsyms0() is to
    generate the table, map_init0[], based on the old (6.1) version of
    symini and the current symini which defines the mapping.  The 6.1
    information, represented  by init_sym0[] and init_names0[], are
    manually extracted from the 6.1-generated syminidf.h; these tables
    are just renamed versions of init_sym[] & init_names[].   oldsyms0()
    just scans these 'symbols' and looks for the symbols with the same
    names in the current symini.  In addition to generating the
    table, map_init0[], the size of the table, init_syms0_size, is
    generated.  If new intrinsics are added after today's symini is
    released, a new table & size, presumably named init_sym1 and
    init_syms1_size, will be generated from inputs init_sym1 &
    init_names1.  This process should be able to be repreated to create
    map_init2, ...

    Determine if a firstosym value read from a .mod file matches
    the current initial symtab or a previous initial symtab for
    which we have mapping information.
 */
int
can_map_initsym(int old_firstosym)
{
  int xtra;
  if (old_firstosym == stb.firstosym) {
    return 1;
  }
  if (old_firstosym == stb.firstosym - NXTRA) {
    return 1;
  }
  xtra = stb.firstosym - INIT_SYMTAB_SIZE - NXTRA; /*predefined consts, etc.*/
  if (old_firstosym == (init_syms0_size + xtra)) {
    return 1;
  }
  if (old_firstosym == (init_syms1_size + xtra)) {
    return 1;
  }
  if (old_firstosym == (init_syms2_size + xtra)) {
    return 1;
  }
  if (old_firstosym == (init_syms3_size + xtra)) {
    return 1;
  }
  return 0;
}

/** \brief Determine if oldsym, read from a .mod file, is a predefined
           (intrinsic, predeclared, constant, etc.).

    If it is, attempt to map the oldsym to
    the current set of predefineds; note that oldsym could also be from
    the current set.  The value of old_firstosym indicates to which set,
    current or older, oldsym belongs.
 */
int
map_initsym(int oldsym, int old_firstosym)
{
  int xtra;

  if (old_firstosym == stb.firstosym) {
    /* current set of predefineds */
    if (oldsym < stb.firstosym)
      return oldsym;
    return 0;
  }

  if (old_firstosym == (stb.firstosym - NXTRA)) {
    /* current set of predefineds */
    if (oldsym < (stb.firstosym - NXTRA))
      return oldsym;
    return 0;
  }

  if (oldsym >= old_firstosym)
    return 0;

  xtra = stb.firstosym - INIT_SYMTAB_SIZE - NXTRA;
  if (old_firstosym == (init_syms0_size + xtra)) {
    if (oldsym >= init_syms0_size) {
      xtra = oldsym - init_syms0_size;
      return INIT_SYMTAB_SIZE + xtra;
    }
    return map_init0[oldsym];
  }
  if (old_firstosym == (init_syms1_size + xtra)) {
    if (oldsym >= init_syms1_size) {
      xtra = oldsym - init_syms1_size;
      return INIT_SYMTAB_SIZE + xtra;
    }
    return map_init1[oldsym];
  }
  if (old_firstosym == (init_syms2_size + xtra)) {
    if (oldsym >= init_syms2_size) {
      xtra = oldsym - init_syms2_size;
      return INIT_SYMTAB_SIZE + xtra;
    }
    return map_init2[oldsym];
  }
  if (old_firstosym == (init_syms3_size + xtra)) {
    if (oldsym >= init_syms3_size) {
      xtra = oldsym - init_syms3_size;
      return INIT_SYMTAB_SIZE + xtra;
    }
    return map_init3[oldsym];
  }
  interr("map_initsym: bad osym", old_firstosym, 0);
  return 0;
}

/** \brief Convert two dollar signs to a hyphen. Especially used for the
           submodule *.mod file renaming:
           ancestor$$submod.mod -> ancestor-submod.mod
 */
void 
convert_2dollar_signs_to_hyphen(char *name) {
  char *p, *q;
  p = q = name;
  while (*q) {
    if (*q == '$' && *(q+1) == '$') {
      q = q + 2;
      *p++ = '-';
    }
    *p++ = *q++;
  }
  *p = *q;
}

/** \brief Used for check whether sym2 used inside the scope of sym1 is defined in 
           parent modules (SCOPEG(sym2)) and used by inherited submodules 
           ENCLFUNCG(sym1).
 */
bool 
is_used_by_submod(SPTR sym1, SPTR sym2) {
  if (SCOPEG(sym2) == sym1 && 
      STYPEG(ENCLFUNCG(sym1)) == ST_MODULE && 
      STYPEG(SCOPEG(sym2)) == ST_MODULE &&
      SCOPEG(sym2) == ANCESTORG(ENCLFUNCG(sym1)))
     return true;

  /* when sym2 is defined in the common block of parent module of submodule sym1 */
  if (SCG(sym2) == SC_CMBLK)
    return SCOPEG(CMBLKG(sym2)) == ANCESTORG(ENCLFUNCG(sym1));

  return false;
}

/** \brief Emit variable type mismatch errors for either subprogram argument variables 
           or function return type based on separate module subprogram's definition vs.
           its declaration.
 */
static void
generate_type_mismatch_errors(SPTR s1, SPTR s2) {
  if (RESULTG(s1) && RESULTG(s2))
    error(1061, ERR_Severe, gbl.lineno, SYMNAME(s1), NULL);
  else
    error(1058, ERR_Severe, gbl.lineno, SYMNAME(s1), NULL);
}

