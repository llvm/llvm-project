/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Host dependent, version dependent, and miscellaneous macros used
   throughout PGFTN.
 */

#ifndef BE_GBLDEFS_H_
#define BE_GBLDEFS_H_

#include <stdint.h>
#include "universal.h"
#include "platform.h"
#include "pgifeat.h"
#include <scutil.h>

#define NEW_ARG_PARSER

/* enable negative zero */
#define USEFNEG 1

#define PGFTN
#define SCFTN
#define SCC_SCFTN
#define PGC_PGFTN
#define PGF90
#ifndef DEBUG
#define DEBUG 1
#endif
#define DBGBIT(n, m) (flg.dbg[n] & m)

#define XBIT(n, m) (flg.x[n] & m)
/* flag to test if using WINNT calling conventions: */
#define WINNT_CALL XBIT(121, 0x10000)
#define WINNT_CREF XBIT(121, 0x40000)
#define WINNT_NOMIXEDSTRLEN XBIT(121, 0x80000)
/* This x-bit controls the insertion of scope labels. On by default. */
#define XBIT_USE_SCOPE_LABELS !XBIT(198, 0x40000)

#define CNULL ((char *)0)
#define uf(s) error(V_0000_Internal_compiler_error_OP1_OP2, ERR_Informational, gbl.lineno, "Unimplemented feature", s)

/* Fortran Standard max identifier length (used with -Mstandard) */
#define STANDARD_MAXIDLEN 63

/* internal max identifier length - allow room for suffix like $sd and $td$ft */
#define MAXIDLEN 163

/*  should replace local MAX_FNAME_LENs with: */
#define MAX_FILENAME_LEN 2048

/* Max function/variable name length, 
 * Function/variable name in C++ can be really long 
 * The length of Fortran function/variable name definition 
 * is to align with C/C++ definition. 1024 may be not necessary for fortran */
#define MAX_FUNCTION_NAME_LEN	(1024)
#define MAX_VARIABLE_NAME_LEN	(1024)

typedef int8_t INT8;
typedef int16_t INT16;
typedef uint16_t UINT16;

typedef int ILM_T; /* defined here, rather than ilm.h, because of problems
                    * with the order of includes
                    */

/* define a host type which represents 'size_t' for array extents. */
#define ISZ_T BIGINT
#define UISZ_T BIGUINT
#define ISZ_PF BIGIPFSZ
#define ISZ_2_INT64(s, r) bgitoi64(s, r)
#define INT64_2_ISZ(s, r) r = i64tobgi(s)
#define ISZ_ABS labs

typedef BIGINT BV;

/* ETLS/TLS threadprivate features */

#undef TRUE
#undef FALSE
typedef bool LOGICAL;
#define TRUE true
#define FALSE false

/*
 * Define truth values for Fortran.  The negate operation is dependent
 * upon the values chosen.
 */
#define SCFTN_TRUE gbl.ftn_true
#define SCFTN_FALSE 0
#define SCFTN_NEGATE(n) ((~(n)) & SCFTN_TRUE)

#define BCOPY(p, q, dt, n) memcpy(p, q, (sizeof(dt) * (n)))
#define BZERO(p, dt, n) memset((p), 0, (sizeof(dt) * (n)))
#if DEBUG
#define FREE(p) sccfree((char *)p), p = NULL
#else
#define FREE(p) sccfree((char *)p)
#endif

#define NODWARF 1
#define NOELF64 1

#if DEBUG
void reportarea(int full);
void bjunk(void *p, BIGUINT64 n);

#define NEW(p, dt, n)                               \
  if (1) {                                          \
    p = (dt *)sccalloc((BIGUINT64)(sizeof(dt) * (n)));   \
    if (DBGBIT(7, 2))                               \
      bjunk((char *)(p), (BIGUINT64)(sizeof(dt) * (n))); \
  } else
#define NEED(n, p, dt, size, newsize)                                   \
  if (n > size) {                                                       \
    p = (dt *)sccrelal((char *)p, ((BIGUINT64)((newsize) * sizeof(dt))));    \
    if (DBGBIT(7, 2))                                                   \
      bjunk((char *)(p + size), (BIGUINT64)((newsize - size) * sizeof(dt))); \
    size = newsize;                                                     \
  } else

#else
#define NEW(p, dt, n) p = (dt *)sccalloc((BIGUINT64)(sizeof(dt) * (n)))
#define NEED(n, p, dt, size, newsize)                                \
  if (n > size) {                                                    \
    p = (dt *)sccrelal((char *)p, ((BIGUINT64)((newsize) * sizeof(dt)))); \
    size = newsize;                                                  \
  } else
#endif

#define NEEDB(n, p, dt, size, newsize)                               \
  if (n > size) {                                                    \
    p = (dt *)sccrelal((char *)p, ((BIGUINT64)((newsize) * sizeof(dt)))); \
    BZERO(p + size, dt, newsize - size);                             \
    size = newsize;                                                  \
  } else

#include "sharedefs.h"

typedef enum RUTYPE {
    RU_SUBR = 1,
    RU_FUNC = 2,
    RU_PROG = 3,
    RU_BDATA = 4
} RUTYPE;

#define CLRFPERR() (Fperr = FPE_NOERR)
/* NOTE :fperror prints an error message and then sets Fperr to FPE_NOERR    */
/*       it returns zero if Fperr was equal to FPE_NOERR , otherwise nonzero */
#define CHKFPERR() (Fperr != FPE_NOERR ? fperror() : 0)

/*  declare external functions which are used globally:  */

 char *sccalloc(BIGUINT64); /* from malloc.c: */
 void sccfree(char *);
 char *sccrelal(char *, BIGUINT64);

 char *getitem(int, int); /* from salloc.c: */
#define GETITEM(area, type) (type *) getitem(area, sizeof(type))
#define GETITEMS(area, type, n) (type *) getitem(area, (n) * sizeof(type))
 void freearea(int);
 int put_getitem_p(void *);
 void *get_getitem_p(int);
 void free_getitem_p(void);

 char *mkfname(const char *, const char *, const char *); /* from miscutil.c: */
 void set_xflag(int, INT);
 void set_yflag(int, INT);
#ifndef __cplusplus
 void bzero(void *, size_t);
#endif
void list_init(FILE*); /* from listing.c: */
void list_line(const char*);
void list_page(void);
void fprintf_str_esc_backslash(FILE *f, char *str);
void extractor_end(void);                          /* extractor.h */
void extractor_single_index_file(char *indexname); /* extractor.h */
void extractor_single_file(char *singlename);      /* extractor.h */
void ipasave_extractor_end(void);                  /* extractor.h */
void extractor(void);                              /* extractor.h */
void carry(void);                                  /* carry.c */
void xcarry(void);                                 /* carry.c */
#ifdef _WIN64
#define snprintf _snprintf
#define ALLOCA(type, number)  (type *) _alloca(sizeof(type) * (number))
#else
#define ALLOCA(type, number)  (type *) alloca(sizeof(type) * (number))
#endif

/* Enable LDSCMPLX/LDDCMPLX ili for byval arguments in Fortran */
#undef USE_LLVM_CMPLX
#define USE_LLVM_CMPLX 1

#endif /* BE_GBLDEFS_H_ */
