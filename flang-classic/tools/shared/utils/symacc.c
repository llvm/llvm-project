/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/********************************************************
  FIXME: get rid of this "important notice" and proliferating copies.

             I M P O R T A N T   N O T I C E
       Do not modify this file if it resides in the
       directory src -- modify the copy which is in
       ../utils/symtab and then run copy.sh

********************************************************/
/**
   \file
   \brief Generic access module.  Used by all compilers and
   initialization utility.
 */

/* FIXME: This file is compiled with different gbldefs.h included
   depending on in which part of the build it is recompiled. */
#include "symacc.h"
#ifdef UTILSYMTAB
#include "universal.h"
#define ERR_Fatal 4
#else
#include "error.h"
#endif
#include <stdarg.h>

#ifndef STANDARD_MAXIDLEN
#define STANDARD_MAXIDLEN MAXIDLEN
#endif

#ifdef UTILSYMTAB
extern STB stb;
GBL gbl;
#else
extern STB stb;
GBL gbl;
#endif
static char buff[132];

void
sym_init_first(void)
{
  int i;

  int sizeof_SYM = sizeof(SYM) / sizeof(INT);

/* Disable checking sym size on Windows.
 * https://github.com/flang-compiler/flang/issues/1043
 */
#if !defined(_WIN64)
#if defined(PGHPF)
  assert(sizeof_SYM == 46, "bad SYM size", sizeof_SYM, ERR_Fatal);
#else
  assert(sizeof_SYM == 38, "bad SYM size", sizeof_SYM, ERR_Fatal);
#endif
#endif // _WIN64

  if (stb.stg_base == NULL) {
#ifdef UTILSYMTAB
    stb.stg_size = 1000;
    NEW(stb.stg_base, SYM, stb.stg_size);
    BZERO(stb.stg_base, SYM, stb.stg_size);
    stb.stg_avail = 1;
#else
    STG_ALLOC(stb, 1000);
#endif
    assert(stb.stg_base, "sym_init: no room for symtab", stb.stg_size,
           ERR_Fatal);
#if defined(PGHPF) && !defined(PGF90)
    stb.n_size = 7011;
#else
    stb.n_size = STB_NSIZE;
#endif
    NEW(stb.n_base, char, stb.n_size);
    assert(stb.n_base, "sym_init: no room for namtab", stb.n_size, ERR_Fatal);
    stb.n_base[0] = 0;
#ifdef UTILSYMTAB
    stb.dt.stg_size = 400;
    NEW(stb.dt.stg_base, ISZ_T, stb.dt.stg_size);
#else
    STG_ALLOC(stb.dt, 400);
#endif
    assert(stb.dt.stg_base, "sym_init: no room for dtypes", stb.dt.stg_size,
           ERR_Fatal);
    /* basically, this is sidecar of dt_base */

#ifdef PGCPLUS
    stb.w_size = 1280;
#else
    stb.w_size = 32;
#endif
    NEW(stb.w_base, INT, stb.w_size);
    assert(stb.w_base, "sym_init: no room for wtab", stb.w_size, ERR_Fatal);
  }
  /* allocate  deepcopy info */
#ifndef UTILSYMTAB
#endif

  stb.namavl = 1;
  stb.wrdavl = 0;
  for (i = 0; i <= HASHSIZE; i++)
    stb.hashtb[i] = SPTR_NULL;

#ifndef INIT
#ifdef PGHPF
  DT_INT = DT_INT4;
  DT_REAL = DT_REAL4;
  DT_CMPLX = DT_CMPLX8;
  DT_LOG = DT_LOG4;
  DT_DBLE = DT_REAL8;
  DT_DCMPLX = DT_CMPLX16;
  DT_PTR = DT_INT4;
#endif
#endif
}

/** \brief Expand symbol storage area when NEWSYM runs out of area.

    It is assumed that stb.stg_avail is 1 more than the index of the current
    symbol being created. */
void
realloc_sym_storage()
{
#ifdef UTILSYMTAB
  symini_errfatal(7);
#else
  DEBUG_ASSERT(stb.stg_avail > stb.stg_size,
               "realloc_sym_storage: call only if necessary");
  if (stb.stg_avail > SPTR_MAX + 1 || stb.stg_base == NULL)
    symini_errfatal(7);
  /* Use unsigned arithmetic to avoid risk of overflow. */
  DEBUG_ASSERT(stb.stg_size > 0,
               "realloc_sym_storage: symbol storage not initialized?");
  STG_NEED(stb);
  DEBUG_ASSERT(stb.stg_avail <= stb.stg_size,
               "realloc_sym_storage: internal error");
#endif
}

/**
   \brief Look up symbol with indicated name.
   \return If there is already such a symbol, the pointer to the
   existing symbol table entry; or 0 if a symbol doesn't exist.
   \param name is a symbol name.
   \param olength is the number of characters in the symbol name.
 */
SPTR
lookupsym(const char *name, int olength)
{
  int length;
  SPTR sptr;   /* pointer to symbol table entry */
  INT hashval; /* index into hashtb. */

  /*
   * Loop thru the appropriate hash link list to see if symbol is
   * already in the table:
   */

  length = olength;
  if (length > MAXIDLEN) {
    length = MAXIDLEN;
  }
  HASH_ID(hashval, name, length);
  for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
#if defined(PGHPF) && !defined(INIT)
    if (HIDDENG(sptr))
      continue;
#endif
    if (strncmp(name, SYMNAME(sptr), length) != 0 ||
        *(SYMNAME(sptr) + length) != '\0')
      continue;

    /* matching entry has been found in symbol table. return it: */

    return sptr;
  }
  return SPTR_NULL;
} /* lookupsym */

/** \brief Issue diagnostic for identifer that is too long.

    \param name - identifier (without terminating njull)
    \param olength - length of identifier
    \param max_idlen - maximum allowed length

    Though this routine has only one lexical call site, it is factored
    out to not clutter the common path in installsym_ex.
  */
static void
report_too_long_identifier(const char *name, int olength, int max_idlen)
{
  static char *ebuf;
  static int ebuf_sz = 0;
  char len_buf[12];
  if (ebuf_sz == 0) {
    ebuf_sz = olength + 1;
    NEW(ebuf, char, ebuf_sz);
  } else {
    int ii;
    NEED(olength + 1, ebuf, char, ebuf_sz, olength + 1);
    ii = strlen(ebuf);
    if (ii < olength)
      strcpy(ebuf + (ii - 2), "..."); /* there's room for at least 1 '.'*/
  }
  memcpy(ebuf, name, olength);
  ebuf[olength] = '\0';
  sprintf(len_buf, "%d", max_idlen);
  symini_error(16, 2, gbl.lineno, ebuf, len_buf);
}

/**
   \brief Get the symbol table index for a NUL-terminated name.
 */
SPTR
lookupsymbol(const char *name)
{
  return lookupsym(name, strlen(name));
}

/**
   \brief Construct a name via printf-style formatting and then
   look it up in the symbol table via lookupsymbol().
 */
SPTR
lookupsymf(const char *fmt, ...)
{
  char buffer[MAXIDLEN + 1];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buffer, sizeof buffer - 1, fmt, ap);
  va_end(ap);
  buffer[sizeof buffer - 1] = '\0'; /* Windows workaround */
  return lookupsymbol(buffer);
}

/**
   \brief Enter symbol with indicated name into symbol table,
   initialize the new entry, and return pointer to it.  If there is
   already such a symbol, just return pointer to the existing symbol
   table entry.

   \param name is the symbol name.
   \param olength is the number of characters in the symbol name.
 */
SPTR
installsym_ex(const char *name, int olength, IS_MODE mode)
{
  int length;
  SPTR sptr;   /* pointer to symbol table entry */
  INT hashval; /* index into hashtb. */
  bool toolong;
  int nmptr;
  static int max_idlen = MAXIDLEN;

  /*
   * Trim identifier if it is too long.
   */
  toolong = false;
  length = olength;
#if !defined(PGHPF) && defined(PGFTN) && !defined(UTILSYMTAB)
  if (flg.standard) {
    max_idlen = 31;
  }
#elif defined(PGF90) && !defined(UTILSYMTAB)
  if (flg.standard) {
    max_idlen = STANDARD_MAXIDLEN;
  }
#endif
  if (length > max_idlen) {
    length = max_idlen;
    toolong = true;
  }

  nmptr = 0;
  if (mode != IS_QUICK) {
    /*
     * Loop thru the appropriate hash link list to see if symbol is
     * already in the table.
     */
    HASH_ID(hashval, name, length);
    for (sptr = stb.hashtb[hashval]; sptr != 0; sptr = HASHLKG(sptr)) {
      const char *sname;
      int np = NMPTRG(sptr);
      if (np + length >= stb.namavl)
        continue;
      sname = stb.n_base + np;
      if (sname[0] != name[0] || sname[length] != '\0')
        continue;
      if (strncmp(name, sname, length) != 0)
        continue;
      nmptr = np;
#if defined(PGHPF) && !defined(INIT)
      if (HIDDENG(sptr))
        continue;
#endif

      /* Matching entry has been found in symbol table. Return it. */

      return sptr;
    }
  }

  /* Symbol not found.  Create a new symbol table entry. */

  NEWSYM(sptr);
  if (mode != IS_QUICK) {
    LINKSYM(sptr, hashval);
  }

  if (!nmptr)
    nmptr = putsname(name, length);
  NMPTRP(sptr, nmptr);
#if defined(PGFTN) && !defined(INIT)
  SYMLKP(sptr, NOSYM);
#endif
#if defined(PGHPF) && !defined(INIT)
  SCOPEP(sptr, 0);
#endif
#ifdef LINENOP
  LINENOP(sptr, gbl.lineno);
#endif

  if (toolong) {
    report_too_long_identifier(name, olength, max_idlen);
  }

  return sptr;
}

/**
   \brief Put a string of characters into the symbol names storage
   area and return pointer to the string (relative to
   stb.n_base). This routine is used to enter both symbol names and
   character string constants.

   \param name are the characters to be entered.
   \param length is the number of characters to be entered.
 */
int
putsname(const char *name, int length)
{
  int nptr; /* index into character storage area */
  char *np; /* pointer into character storage area */
  int i;    /* counter */

  nptr = stb.namavl;
  stb.namavl += (length + 1);
  while (stb.namavl > stb.n_size) {
    /* To avoid quadratic behavior, we increase the storage area size
       by a factor, not a constant.  Use unsigned arithmetic here
       to avoid risk of overflow. */
    unsigned n = 2u * stb.n_size;
    if (n > MAX_NMPTR) {
      n = MAX_NMPTR;
      if (stb.namavl > n)
        symini_errfatal(7); /* names table overflow */
    }
    NEED(stb.namavl, stb.n_base, char, stb.n_size, n);
  }
  np = stb.n_base + nptr;
  for (i = 0; i < length; i++)
    *np++ = *name++;
  *np = '\0';

  return nptr;
}

/**
   \brief Create a local copy of a name known to be stored in the 'stb.n_base'
   area.

   Used when a symbol needs to be created from a name stored in the
   area; a purify umr error could occur if the area is realloc'd.  The
   char pointer to the copy is returned.
 */
char *
local_sname(char *name)
{
  static char *safe_p;
  static int safe_sz = 0;
  int length;

  length = strlen(name) + 2 + 6; /* MW: add one more character,
                                    needed in semfunc2 */
  /* Hongyon: add 6 more for
     _cr and _nm for cref,nomixed */
  if (safe_sz == 0) {
    safe_sz = length + 100;
    NEW(safe_p, char, safe_sz);
  } else {
    NEED(length, safe_p, char, safe_sz, length + 100);
  }

  strcpy(safe_p, name);

  return safe_p;
}

#if !defined(UTILSYMTAB)
void
add_fp_constants(void)
{
  INT tmp[4];
  INT res[4];

  tmp[0] = 0;
#if defined(PGHPF) || defined(PGF90) || defined(PGFTN)
  atoxf("0.0", &tmp[1], 3);
#ifdef DT_REAL4
  /***** Only the f90/hpf front-ends know aobut the sized-real *****/
  /*
   * NOTE (A REMINDER):
   * 1. the order in which the following constants are created can
   *    never changed -- their sptr numbers as well what they denote
   *    is asseumed by the modfile processing.
   * 2. additional predefined constants CANNOT be added unless
   *    we propagate how many more to symtab.c:can_map_initsym() and
   *    symtab.c:map_initsym().
   *    For now, just create them, such as stb.fltm0 and stb.dblm0,
   *    in symtab.c after symtab.c:sym_init() defines stb.firstosym
   */

  stb.flt0 = getcon(tmp, DT_REAL4);
  atoxf("1.0", &tmp[1], 3);
  stb.flt1 = getcon(tmp, DT_REAL4);
  atoxf("2.0", &tmp[1], 3);
  stb.flt2 = getcon(tmp, DT_REAL4);
  atoxf("0.5", &tmp[1], 3);
  stb.flthalf = getcon(tmp, DT_REAL4);

  atoxd("0.0", &tmp[0], 3);
  stb.dbl0 = getcon(tmp, DT_REAL8);
  atoxd("1.0", &tmp[0], 3);
  stb.dbl1 = getcon(tmp, DT_REAL8);
  atoxd("2.0", &tmp[0], 3);
  stb.dbl2 = getcon(tmp, DT_REAL8);
  atoxd("0.5", &tmp[0], 3);
  stb.dblhalf = getcon(tmp, DT_REAL8);

#else
  /***** the f90 backend *****/
  stb.flt0 = getcon(tmp, DT_REAL);
  atoxf("1.0", &tmp[1], 3);
  stb.flt1 = getcon(tmp, DT_REAL);
  atoxf("2.0", &tmp[1], 3);
  stb.flt2 = getcon(tmp, DT_REAL);
  atoxf("0.5", &tmp[1], 3);
  stb.flthalf = getcon(tmp, DT_REAL);

  atoxd("0.0", &tmp[0], 3);
  stb.dbl0 = getcon(tmp, DT_DBLE);
  atoxd("1.0", &tmp[0], 3);
  stb.dbl1 = getcon(tmp, DT_DBLE);
  atoxd("2.0", &tmp[0], 3);
  stb.dbl2 = getcon(tmp, DT_DBLE);
  atoxd("0.5", &tmp[0], 3);
  stb.dblhalf = getcon(tmp, DT_DBLE);

  tmp[0] = 0;
  res[0] = 0;
  tmp[1] = CONVAL2G(stb.flt0);
  xfneg(tmp[1], &res[1]);
  stb.fltm0 = getcon(res, DT_REAL);
  tmp[0] = CONVAL1G(stb.dbl0);
  tmp[1] = CONVAL2G(stb.dbl0);
  xdneg(tmp, res);
  stb.dblm0 = getcon(res, DT_DBLE);
#endif

#else
  /***** the pgcpp backend & pgc *****/
  /* float 0, 1, 2, .5, -0.0 */
  atoxf("0.0", &tmp[1], 3);
  stb.flt0 = getcon(tmp, DT_FLOAT);
  atoxf("1.0", &tmp[1], 3);
  stb.flt1 = getcon(tmp, DT_FLOAT);
  atoxf("2.0", &tmp[1], 3);
  stb.flt2 = getcon(tmp, DT_FLOAT);
  atoxf("0.5", &tmp[1], 3);
  stb.flthalf = getcon(tmp, DT_FLOAT);

  /* 1.0, 2.0, 0.5, -0.0  as double, quad */
  atoxd("0.0", &tmp[0], 3);
  stb.dbl0 = getcon(tmp, DT_DBLE);
  stb.quad0 = getcon(tmp, DT_QUAD); /* quad currently the same as double */
  atoxd("1.0", &tmp[0], 3);
  stb.dbl1 = getcon(tmp, DT_DBLE);
  stb.quad1 = getcon(tmp, DT_QUAD); /* quad currently the same as double */
  atoxd("2.0", &tmp[0], 3);
  stb.dbl2 = getcon(tmp, DT_DBLE);
  stb.quad2 = getcon(tmp, DT_QUAD); /* quad currently the same as double */
  atoxd("0.5", &tmp[0], 3);
  stb.dblhalf = getcon(tmp, DT_DBLE);
  stb.quadhalf = getcon(tmp, DT_QUAD); /* quad currently the same as double */

  tmp[0] = 0;
  res[0] = 0;
  tmp[1] = CONVAL2G(stb.flt0);
  xfneg(tmp[1], &res[1]);
  stb.fltm0 = getcon(res, DT_FLOAT);
  tmp[0] = CONVAL1G(stb.dbl0);
  tmp[1] = CONVAL2G(stb.dbl0);
  xdneg(tmp, res);
  stb.dblm0 = getcon(res, DT_DBLE);
  stb.quadm0 = getcon(res, DT_QUAD); /* quad currently the same as double */
#endif

#ifdef LONG_DOUBLE_X87
  atoxe("0.0", tmp, 3);
  stb.x87_0 = getcon(tmp, DT_X87);
  xeneg(tmp, res);
  stb.x87_m0 = getcon(res, DT_X87);
  atoxe("1.0", tmp, 3);
  stb.x87_1 = getcon(tmp, DT_X87);
  atoxe("2.0", tmp, 3);
  stb.x87_2 = getcon(tmp, DT_X87);
#endif
#ifdef DOUBLE_DOUBLE
  tmp[0] = stb.dbl0;
  tmp[1] = stb.dbl0;
  stb.doubledouble_0 = getcon(tmp, DT_DOUBLEDOUBLE);
  tmp[0] = stb.dblm0;
  tmp[1] = stb.dblm0;
  stb.doubledouble_m0 = getcon(tmp, DT_DOUBLEDOUBLE);
  tmp[0] = stb.dbl1;
  tmp[1] = stb.dbl0;
  stb.doubledouble_1 = getcon(tmp, DT_DOUBLEDOUBLE);
#endif
#ifdef LONG_DOUBLE_FLOAT128
  atoxq("0.0", &tmp[0], 4);
  stb.float128_0 = getcon(tmp, DT_FLOAT128);
  xqneg(tmp, res);
  stb.float128_m0 = getcon(res, DT_FLOAT128);
  atoxq("1.0", &tmp[0], 4);
  stb.float128_1 = getcon(tmp, DT_FLOAT128);
  atoxq("0.5", &tmp[0], 4);
  stb.float128_half = getcon(tmp, DT_FLOAT128);
  atoxq("2.0", &tmp[0], 4);
  stb.float128_2 = getcon(tmp, DT_FLOAT128);
#endif
}

bool
is_flt0(SPTR sptr)
{
  if (sptr == stb.flt0 || sptr == stb.fltm0)
    return true;
  return false;
}

bool
is_dbl0(SPTR sptr)
{
  if (sptr == stb.dbl0 || sptr == stb.dblm0)
    return true;
  return false;
}

bool
is_quad0(SPTR sptr)
{
  if (sptr == stb.quad0 || sptr == stb.quadm0)
    return true;
  return false;
}

#ifdef LONG_DOUBLE_X87
bool
is_x87_0(SPTR sptr)
{
  return sptr == stb.x87_0 || sptr == stb.x87_m0;
}

bool
is_cmplx_x87_0(SPTR sptr)
{
  if (is_x87_0(CONVAL1G(sptr)) && is_x87_0(CONVAL2G(sptr)))
    return true;
  return false;
}
#endif /* LONG_DOUBLE_X87 */

#ifdef DOUBLE_DOUBLE
bool
is_doubledouble_0(SPTR sptr)
{
  return sptr == stb.doubledouble_0 || sptr == stb.doubledouble_m0;
}

bool
is_cmplx_doubledouble_0(SPTR sptr)
{
  return is_doubledouble_0(CONVAL1G(sptr)) && is_doubledouble_0(CONVAL2G(sptr));
}
#endif /* DOUBLE_DOUBLE */

#ifdef LONG_DOUBLE_FLOAT128
bool
is_float128_0(SPTR sptr)
{
  return sptr == stb.float128_0 || sptr == stb.float128_m0;
}
#endif /* LONG_DOUBLE_FLOAT128 */

bool
is_cmplx_flt0(SPTR sptr)
{
  if (CONVAL1G(sptr) == CONVAL2G(stb.flt0) ||
      CONVAL1G(sptr) == CONVAL2G(stb.fltm0)) {
    if (CONVAL2G(sptr) == CONVAL2G(stb.flt0) ||
        CONVAL2G(sptr) == CONVAL2G(stb.fltm0)) {
      return true;
    }
  }
  return false;
}

bool
is_creal_flt0(SPTR sptr)
{
  if (CONVAL1G(sptr) == CONVAL2G(stb.flt0) ||
      CONVAL1G(sptr) == CONVAL2G(stb.fltm0))
    return true;
  return false;
}

bool
is_cimag_flt0(SPTR sptr)
{
  if (CONVAL2G(sptr) == CONVAL2G(stb.flt0) ||
      CONVAL2G(sptr) == CONVAL2G(stb.fltm0))
    return true;
  return false;
}

bool
is_cmplx_dbl0(SPTR sptr)
{
  return is_dbl0(SymConval1(sptr)) && is_dbl0(SymConval2(sptr));
}

bool
is_cmplx_quad0(SPTR sptr)
{
  return is_quad0(SymConval1(sptr)) && is_quad0(SymConval2(sptr));
}

#endif

void
symini_errfatal(int n)
{
#ifdef UTILSYMTAB
  sprintf(buff, "Fatal error number %d", n);
  symini_error(n, 4, gbl.lineno, CNULL, CNULL);
#else
  errfatal((error_code_t)n);
#endif
}

void
symini_error(int n, int s, int l, const char *c1, const char *c2)
{
#ifdef UTILSYMTAB
  if (c1 == NULL)
    c1 = "";
  if (c2 == NULL)
    c2 = "";
  sprintf(buff, "Error number %d, line %d, params %s %s\n", n, l, c1, c2);
  fputs(buff, stderr);
  if (s == 4)
    exit(1);
#else
  error((error_code_t)n, (enum error_severity)s, l, c1, c2);
#endif
}

void
symini_interr(const char *txt, int val, int sev)
{
  char buff[8];

  sprintf(buff, "%7d", val);
  symini_error(0, sev, gbl.lineno, txt, buff);
}

