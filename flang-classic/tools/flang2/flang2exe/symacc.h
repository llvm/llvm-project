/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef SYMACC_H_
#define SYMACC_H_

#include "scutil.h"
#include "gbldefs.h"
#include "global.h"
struct SYM;
#include "symtab.h"
#include "sharedefs.h"

/* Keep in sync with INIT_NAMES_SIZE, in syminidf.h. */
#define STB_NSIZE 3289

/**
 * \file
 * \brief Various definitions for symacc.
 */

#ifdef __cplusplus
/** Must have same layout as a T* */
template <typename T, typename Index> class IndexBy
{
  T *rep;

public:
  T &operator[](Index index)
  {
    return rep[index];
  }
  /* Rest of operators are required to make macro NEED work.
     Their functionality is deliberately minimized with intent
     to minimize accidental use outside macro NEED.  It would
     be better that "operator void*" have the explicit keyword, but
     Microsoft 10.0 Open Tools does not support that C++11 feature. */
  operator char*() const
  {
    return reinterpret_cast<char*>(rep);
  }
  void operator=(T *ptr)
  {
    rep = ptr;
  }
  bool operator!() const
  {
    return !rep;
  }
  void *operator+(int offset) const
  {
    return reinterpret_cast<void *>(rep + offset);
  }
};

#define INDEX_BY(T, Index) IndexBy<T, Index>
#else
#define INDEX_BY(T, Index) T *
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* FIXME: down the file there are declarations that depend on ST_MAX
   etc. and not guarded by #ifdef INIT.  Either INIT is always
   defined or there exist alternative definitions for these values
   somewhere.  This needs to be unified and cleaned.  */

/* hashtab stuff */
#define HASHSIZE 9973
#define HASH_CON(p) ((p[0] ^ p[1]) % HASHSIZE)
#define HASH_ID(hv, p, len)                            \
  hv = p[(len)-1] | (*p << 16) | (p[(len) >> 1] << 8); \
  if ((int)(len) > 3)                                  \
    hv ^= (p[1] << 4);                                 \
  hv %= HASHSIZE;
#define HASH_STR(hv, p, len)     \
  if (len) {                     \
    /*hv =*/HASH_ID(hv, p, len); \
  } else                         \
    hv = 0;

/* limits */
#define MAX_NMPTR 134217728

/* for exclusive use by NEWSYM */
void realloc_sym_storage();

/* symbol creation macros */
#define NEWSYM(sptr)         \
  sptr = (SPTR)STG_NEXT(stb);

#define LINKSYM(sptr, hashval)        \
  HASHLKP(sptr, stb.hashtb[hashval]); \
  stb.hashtb[hashval] = sptr

#define ADDSYM(sptr, hashval) \
  NEWSYM(sptr);               \
  LINKSYM(sptr, hashval)

/*  symbol table typedef declarations */

typedef struct SYM {
  SYMTYPE stype : 8;
  SC_KIND sc : 8;
  unsigned b3 : 8;
  unsigned b4 : 8;
  DTYPE dtype;
  SPTR hashlk;
  SPTR symlk;
  INT scope;
  INT nmptr;
  unsigned f1 : 1, f2 : 1, f3 : 1, f4 : 1, f5 : 1, f6 : 1, f7 : 1, f8 : 1;
  unsigned f9 : 1, f10 : 1, f11 : 1, f12 : 1, f13 : 1, f14 : 1, f15 : 1, f16 : 1;
  unsigned f17 : 1, f18 : 1, f19 : 1, f20 : 1, f21 : 1, f22 : 1, f23 : 1, f24 : 1;
  unsigned f25 : 1, f26 : 1, f27 : 1, f28 : 1, f29 : 1, f30 : 1, f31 : 1, f32 : 1;
  INT w8;
  INT w9;
  ISZ_T w10;
  INT w11;
  INT w12;
  INT w13;
  ISZ_T w14;
  INT w15;
  INT w16;
  INT w17;
  INT w18;
  unsigned f33 : 1, f34 : 1, f35 : 1, f36 : 1, f37 : 1, f38 : 1, f39 : 1, f40 : 1;
  unsigned f41 : 1, f42 : 1, f43 : 1, f44 : 1, f45 : 1, f46 : 1, f47 : 1, f48 : 1;
  unsigned f49 : 1, f50 : 1, f51 : 1, f52 : 1, f53 : 1, f54 : 1, f55 : 1, f56 : 1;
  unsigned f57 : 1, f58 : 1, f59 : 1, f60 : 1, f61 : 1, f62 : 1, f63 : 1, f64 : 1;
  INT w20;
  unsigned f65 : 1, f66 : 1, f67 : 1, f68 : 1, f69 : 1, f70 : 1, f71 : 1, f72 : 1;
  unsigned f73 : 1, f74 : 1, f75 : 1, f76 : 1, f77 : 1, f78 : 1, f79 : 1, f80 : 1;
  unsigned f81 : 1, f82 : 1, f83 : 1, f84 : 1, f85 : 1, f86 : 1, f87 : 1, f88 : 1;
  unsigned f89 : 1, f90 : 1, f91 : 1, f92 : 1, f93 : 1, f94 : 1, f95 : 1, f96 : 1;
  INT w22;
  INT w23;
  INT w24;
  unsigned f97 : 1, f98 : 1, f99 : 1, f100 : 1, f101 : 1, f102 : 1, f103 : 1,
      f104 : 1;
  unsigned f105 : 1, f106 : 1, f107 : 1, f108 : 1, f109 : 1, f110 : 1, f111 : 1,
      f112 : 1;
  unsigned f113 : 1, f114 : 1, f115 : 1, f116 : 1, f117 : 1, f118 : 1, f119 : 1,
      f120 : 1;
  unsigned f121 : 1, f122 : 1, f123 : 1, f124 : 1, f125 : 1, f126 : 1, f127 : 1,
      f128 : 1;
  INT w26;
  INT w27;
  INT w28;
  INT w29;
  INT w30;
  INT w31;
  INT w32;
  INT palign;
} SYM;

/*   symbol table data declarations:  */
typedef struct {
  const char *stypes[ST_MAX + 1];
  OVCLASS ovclass[ST_MAX + 1];
  const char *ocnames[OC_MAX + 1];
  const char *scnames[SC_MAX + 1];
  const char *tynames[TY_MAX + 1];
  SPTR i0, i1;
  SPTR k0, k1;
  SPTR flt0, dbl0, quad0;
  SPTR fltm0, dblm0, quadm0; /* floating point minus 0 */
  SPTR flt1, dbl1, quad1;
  SPTR flt2, dbl2, quad2;
  SPTR flthalf, dblhalf, quadhalf;
  struct{
    STG_MEMBERS(ISZ_T);
  }dt;
  int curr_scope;
  SPTR hashtb[HASHSIZE + 1];
  SPTR firstusym, firstosym;
  STG_MEMBERS(SYM);
  char *n_base;
  int n_size;
  int namavl;
  int lbavail;
  int lb_string_avail;
  INT *w_base;
  int w_size;
  int wrdavl;
#ifdef LONG_DOUBLE_FLOAT128
  /* __float128 0.0, -0.0, 1.0, .5, and 2.0 */
  SPTR float128_0, float128_m0, float128_1;
  SPTR float128_half, float128_2;
#endif
} STB;

extern STB stb;

#ifdef __cplusplus
inline SPTR SymConval1(SPTR sptr) {
  return static_cast<SPTR>(CONVAL1G(sptr));
}
inline SPTR SymConval2(SPTR sptr) {
  return static_cast<SPTR>(CONVAL2G(sptr));
}
#else
#define SymConval1 CONVAL1G
#define SymConval2 CONVAL2G
#endif

/** mode parameter for installsym_ex. */
typedef enum IS_MODE {
  /* Create new symbol if it does not already exist. */
  IS_GET,
  /* Create new symbol always and do NOT insert it in the hash table. */
  IS_QUICK
} IS_MODE;

void sym_init_first(void);
SPTR lookupsym(const char *, int);
SPTR lookupsymbol(const char *);
SPTR lookupsymf(const char *, ...);
#define installsym(name, olength) installsym_ex(name, olength, IS_GET)
SPTR installsym_ex(const char *name, int olength, IS_MODE mode);
int putsname(const char *, int);
char *local_sname(char *);
void add_fp_constants(void);
bool is_flt0(SPTR sptr);
bool is_dbl0(SPTR sptr);
bool is_quad0(SPTR sptr);
bool is_x87_0(SPTR sptr);
bool is_doubledouble_0(SPTR sptr);
bool is_cmplx_flt0(SPTR sptr);
bool is_creal_flt0(SPTR sptr);
bool is_cimag_flt0(SPTR sptr);
bool is_cmplx_dbl0(SPTR sptr);
bool is_cmplx_quad0(SPTR sptr);
bool is_cmplx_x87_0(SPTR sptr);
bool is_cmplx_doubledouble_0(SPTR sptr);

void put_err(int sev, const char *txt);

void symini_errfatal(int n);
void symini_error(int n, int s, int l, const char *c1, const char *c2);
void symini_interr(const char *txt, int val, int sev);

#if defined(__cplusplus)
}
#endif

#endif // SYMACC_H_
