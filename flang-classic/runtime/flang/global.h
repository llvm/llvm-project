/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Global definitions and declarations for Fortran I/O library
 */

#ifndef FLANG_RUNTIME_GLOBAL_DEFS_H_
#define FLANG_RUNTIME_GLOBAL_DEFS_H_

#include "universal.h"
#include "fioMacros.h"
#include "stdioInterf.h" /* stubbed version of stdio.h */
#include "cnfg.h" /* declarations for configuration items */
#include <stdint.h>

#define GBL_SIZE_T_FORMAT "zu"

typedef int DBLINT64[2];
typedef unsigned int DBLUINT64[2];

/* declarations needed where integer*8 & logical*8 are supported and
 * the natural integer is integer*4 (__BIGINT is __INT4).
 */

#define I64_MSH(t) t[1]
#define I64_LSH(t) t[0]

extern int __ftn_32in64_;

#ifndef LOCAL_DEBUG
#define LOCAL_DEBUG 0
#endif

typedef unsigned short WCHAR;

/*  declare some external library functions required:  */

#define VOID void

#if defined(_WIN64)
WIN_MSVCRT_IMP char *WIN_CDECL getenv(const char *);
WIN_MSVCRT_IMP long WIN_CDECL strtol(const char *, char **, int);
WIN_MSVCRT_IMP char *WIN_CDECL strerror(int);
WIN_MSVCRT_IMP char *WIN_CDECL strstr(const char *, const char *);
#endif

typedef int32_t INT;       /* native integer at least 32 bits */
typedef uint32_t UINT; /* unsigned 32 bit native integer */
#define ISDIGIT(c) ((c) >= '0' && (c) <= '9')

/*
 * Because of bugs in AT&T SysV R4 fwrite, it is necessary to use
 * a special version of fwrite for line-buffered files.  This
 * is defined in the makefile as BROKEN_FWRITE.
 */
#undef FWRITE
#define FWRITE __io_fwrite

#define TRUE 1
#define FALSE 0
typedef int bool;
typedef char sbool; /* short boolean (for use in large structs) */

/*  true and false as represented in Fortran program at runtime:  */
#define FTN_TRUE GET_FIO_CNFG_FTN_TRUE
#define FTN_FALSE 0

#if LOCAL_DEBUG
#define assert(ex)                                                        \
  {                                                                       \
    if (!(ex)) {                                                          \
      (VOID) __io_fprintf(__io_stderr(),                                  \
                          "Fio-assertion failed: file \"%s\", line %d\n", \
                          __FILE__, __LINE__);                            \
    }                                                                     \
  }
#else
#define assert(ex)
#endif

#define STASH(str) (strcpy((char *)malloc(strlen(str) + 1), str))

/* defs used by __fortio_error */

#define ERR_FLAG 1
#define EOF_FLAG 2
#define EOR_FLAG 3

#define FIO_BITV_NONE 0x00
#define FIO_BITV_IOSTAT 0x01
#define FIO_BITV_ERR 0x02
#define FIO_BITV_EOF 0x04
#define FIO_BITV_EOR 0x08
#define FIO_BITV_IOMSG 0x10

#define FIO_STAT_INTERNAL_UNIT    \
  99 /* must be kept in sync with \
      * iso_fortran_env.f90:IOSTAT_INQUIRE_INTERNAL_UNIT */

/*
 * maximum filename length in bytes which should be sufficient for
 * most cases, including the names of scratch files.  open & inquire
 * will allow longer names, but must malloc/free temp space.
 */

#define MAX_NAMELEN 255

/* Fortran I/O error code definitions: */

#define FIO_ERROR_OFFSET 200 /* smallest error value */
                             /* 200 */
#define FIO_ESPEC 201
#define FIO_ECOMPAT 202
#define FIO_ERECLEN 203
#define FIO_EREADONLY 204
#define FIO_EDISPOSE 205
#define FIO_ESCRATCH 206
#define FIO_EOPENED 207
#define FIO_EEXIST 208
#define FIO_ENOEXIST 209
#define FIO_ENOMEM 210
#define FIO_ENAME 211
#define FIO_EUNIT 212
#define FIO_ERECL 213
#define FIO_EWRITEONLY 214
#define FIO_EFORM 215
/* 216 */
#define FIO_EEOF 217
#define FIO_EEOR 218
#define FIO_ETOOBIG 219
#define FIO_ETOOFAR 220
#define FIO_EFSYNTAX 221
#define FIO_EPAREN 222
#define FIO_EPT 223
#define FIO_ESTRING 224
#define FIO_ELEX 225
#define FIO_ELETTER 226
/* 227 */
#define FIO_ENOGROUP 228
#define FIO_ENMLEOF 229
#define FIO_ESCALEF 230
#define FIO_EERR_DATA_CONVERSION 231
/* 232 */
#define FIO_ETOOM 233
#define FIO_EEDITDSCR 234
#define FIO_EMISMATCH 235
#define FIO_EBIGREC 236
#define FIO_EQUAD 237
#define FIO_ETAB_VALUE_OUT_OF_RANGE 238
#define FIO_ENOTMEM 239
#define FIO_ELPAREN 240
#define FIO_EENDFMT 241
#define FIO_EDIRECT 242
#define FIO_EPNEST 243
#define FIO_ENONAME 244
#define FIO_ESYNTAX 245
#define FIO_EINFINITE_REVERSION 246
/* 247 */
#define FIO_ESUBSC 248
#define FIO_EFGD 249
#define FIO_EDOT 250
#define FIO_ECHAR 251
#define FIO_EEOFERR 252
#define FIO_EDREAD 253
#define FIO_EREPCNT 254
#define FIO_EASYNC 255
#define FIO_EPOS 256
#define FIO_EPOSV 257
#define FIO_ENEWUNIT 258

#define FIRST_NEWUNIT -13 /* newunits are less than or equal to  this  */
#define ILLEGAL_UNIT(u) \
  ((u) < 0 && ((u) > FIRST_NEWUNIT || (u) <= next_newunit))

/* Fortran I/O file control block struct */

typedef struct fcb {
  struct fcb *next; /* pointer to next fcb in avail or allocd
                     * list.
                     */
  FILE *fp;         /* UNIX file pointer from fopen().  Note that a
                     * non-NULL value for this field is what
                     * indicates that a particular FCB is in use.
                     */
  char *name;       /* file name */
  int unit;         /* unit number */
  __INT8_T reclen;  /* access record length in bytes or words for
                     * direct access files
                     */
  __INT8_T
  partial;     /* Flag/count of bytes in last record when the last record is
                * shorter than reclen.  Set and used only during a direct,
                * unformated read of last record . */
  int wordlen; /* length of words in bytes */
  __INT8_T nextrec; /* record number of next record */
  __INT8_T maxrec;  /* maximum record number (direct access only) */
  __INT8_T skip;    /* After a nonadvancing write statement, this
                     * field is the number of characters remaining
                     * in the buffer, i.e., it's possible that not
                     * all of data in the buffer is transferred to
                     * file. For example, the descriptors, T & TL
                     * could effect a record position before data
                     * which was already present in the buffer.
                     */
  char *skip_buff;  /* If skip is nonzero, this field is a pointer
                     * to an allocated temporary which contains the
                     * characters remaining in the buffer and not
                     * transferred to file.  Upon an ensuing write
                     * of the same file, the characters in the
                     * temporary buffer will be copied to the buffer
                     * used by fmtwrite.c.
                     */
  short status;     /* FIO_OLD or FIO_SCRATCH */
  short dispose;    /* KEEP, DELETE or SAVE */
  short acc;        /* FIO_DIRECT or FIO_SEQUENTIAL (never APPEND)*/
  short action;     /* READ, WRITE, or READWRITE */
  short blank;      /* FIO_NULL or ZERO */
  short form;       /* FIO_FORMATTED or FIO_UNFORMATTED */
  short pad;        /* YES or NO */
  short pos;        /* ASIS, REWIND, or APPEND */
  short delim;      /* APOSTROPHE, QUOTE, or NONE */
  short coherent;   /* coherency check for read & write (e.g. write
                     * followed by read needs a seek):
                     *   0 = no seek necessary for read/write
                     *   1 = coherent only if write.
                     *   2 = coherent only if read.
                     */
  short share;      /* bit vector of file sharing values TBD */
  short decimal;    /* COMMA, POINT, */
  short encoding;   /* UTF-8, UNKNOWN */
  short round;      /* UP, DOWN, ZERO, NEAREST, COMPATIBLE,
                     * PROCESSOR_DEFINED
                     */
  short sign;       /* PLUS, SUPPRESS, PROCESSOR_DEFINED */
  sbool eof_flag;   /* indicates that (imaginary) eof record has
                     * been read.  Initially FALSE, set by ENDFILE
                     * or read past endoffile; cleared by REWIND
                     * and BACKSPACE
                     */
  sbool named;      /* whether file is named or not */
  sbool stdunit;    /* FCB connected to stdin/stderr/stdout */
  sbool truncflag;  /* for sequential files only.  If write
                     * stmt occurs, file must be truncated if
                     * necessary
                     */
  sbool binary;     /* for unformatted files only, binary mode.
                     * if set, record length words are not present.
                     */
  sbool ispipe;     /* FCB connected to a tty or named pipe */
  sbool nonadvance; /* last fmt write had advance=no */
  sbool eor_flag;   /* nonadvancing unit is at the end-of-record;
                     * detected when the unit is a stdunit
                     */
  /*
   * byte_swap, native: two flags set when the CONVERT open specifier
   * is present.  The default value (the CONVERT specifer is absent)
   * for both flags is false.
   */
  sbool byte_swap;    /* unformatted data needs to be byte swapped */
  sbool native;       /* unformatted data is in native format */
  sbool asy_rw;       /* async read/write stmt active */
  struct asy *asyptr; /* pointer to asynch information,set by open */
  char *pread;        /* points to buffer of already read line
                       * this is currently used in namelist only
                       * record is read per line, we must point back
                       * to a position after '=' for a child io.
                       */
  char *pback;        /* need to keep track of the last line read
                       * used in nmlread too.
                       */
} FIO_FCB;

/*
 * FIO_FCB flags were moved to a separate header file because some low
 * level routines (rounding in particular) need to access them without
 * all the other global.h stuff
 */
#include "fio_fcb_flags.h"

/*
 * declare structure representing a value found during list-directed/namelist
 * read.  This value is stored by __fortio_assign()
 */
/* WARNING: assumes BIGINT can hold any BIGLOG size */
typedef struct atag {
  int dtype;       /* __BIGINT,__BIGLOG, __BIGREAL, __BIGCPLX, __(N)CHAR */
  union {          /* value: depends on dtype */
    __BIGINT_T i;  /* __BIGINT, __BIGLOG */
    __BIGREAL_T d; /* __BIGREAL */
    DBLINT64 i8; /* __INT8 */
    __INT8_T i8v;
    DBLUINT64 ui8; /* __LOG8 */
    __INT8_UT ui8v;
    struct {     /* __STR, __NCHAR */
      int len;   /* length of string */
      char *str; /* ptr to its characters */
    } c;
    struct atag *cmplx; /* __BIGCPLX: ptr to 2 element TKNVAL, */
    /* [0] - real, [1] - imag, both are __BIGREAL */
  } val;
} AVAL;

/*  declare global variables for Fortran I/O:  */

typedef struct {
  FIO_FCB *fcbs; /* pointer to list of allocated fcbs */
  INT *enctab;   /* pointer to buffer w encoded format */
  char *fname;   /* file name for OPEN error messages */
  int fnamelen;
  bool error;
  bool eof;
  bool pos_present;
  seekoffx_t pos;
} FIO_TBL;

/*  declare external variables/arrays used by Fortran I/O:  */

#include <errno.h>

extern FIO_TBL fioFcbTbls;
#define GET_FIO_FCBS fioFcbTbls.fcbs

extern int next_newunit; /* newunit counter */

/*extern short	__fortio_type_size[]; */

/* #define FIO_TYPE_SIZE(i) __fortio_type_size[i] */
#define FIO_TYPE_SIZE(i) (1 << GET_DIST_SHIFTS(i))

extern char *envar_fortranopt;

/*  declare external functions local to Fortran I/O:  */

extern int __fort_getpid();
__INT_T __fort_time(void);

/*****  assign.c  *****/
extern int __fortio_assign(char *, int, __CLEN_T, AVAL *);

/*****  fpcvt.c  *****/
extern char *__fortio_ecvt(__BIGREAL_T, int, int, int *, int *, int, int);
extern char *__fortio_fcvt(__BIGREAL_T, int, int, int, int *, int *, int, int);
WIN_MSVCRT_IMP double WIN_CDECL strtod(const char *, char **);
WIN_MSVCRT_IMP long double WIN_CDECL strtold(const char *, char **);
#define __fortio_strtod(x, y) strtod(x, y)
#define __fortio_strtold(x, y) strtold(x, y)

/*****  error.c  *****/
extern VOID set_gbl_newunit(bool newunit);
extern bool get_gbl_newunit();
extern VOID __fortio_errinit(__INT_T, __INT_T, __INT_T *, const char *);
extern VOID __fortio_errinit03(__INT_T unit, __INT_T bitv, __INT_T *iostat,
                               const char *str);
extern VOID __fortio_errend(void);
extern VOID __fortio_errend03(void);
extern int f90_old_huge_rec_fmt(void);
extern int __fortio_error(int);
extern int __fortio_eoferr(int);
extern int __fortio_eorerr(int);
extern const char *__fortio_errmsg(int);
extern int __fortio_check_format(void);
extern int __fortio_eor_crlf(void);
extern VOID __fortio_fmtinit(void);
extern VOID __fortio_fmtend(void);
#define EOR_CRLF __fortio_eor_crlf()
extern int __fortio_no_minus_zero(void);
int __fortio_new_fp_formatter(void);

/*****  hpfio.c  *****/
extern VOID __fort_status_init(__INT_T *, __INT_T *);
void __fortio_stat_init(__INT_T *bitv, __INT_T *iostat);
int __fortio_stat_bcst(int *stat);
#define DIST_STATUS_BCST(s) (s)
#define DIST_RBCSTL(a1, a2, a3, a4, a5, a6)
#define DIST_RBCST(a1, a2, a3, a4, a5)

/*****  utils.c  *****/
extern FIO_FCB *__fortio_alloc_fcb(void);
extern VOID __fortio_free_fcb(FIO_FCB *);
extern VOID __fortio_cleanup_fcb(void);
extern FIO_FCB *__fortio_rwinit(int, int, __INT_T *, int);
extern FIO_FCB *__fortio_find_unit(int);
extern int __fortio_zeropad(FILE *, long);
extern bool __fortio_eq_str(char *, __CLEN_T, const char *);
extern void *__fortio_fiofcb_asyptr(FIO_FCB *);
extern bool __fortio_fiofcb_asy_rw(FIO_FCB *);
extern void __fortio_set_asy_rw(FIO_FCB *, bool);
extern bool __fortio_fiofcb_stdunit(FIO_FCB *);
extern FILE *__fortio_fiofcb_fp(FIO_FCB *);
extern short __fortio_fiofcb_form(FIO_FCB *);
extern const char *__fortio_fiofcb_name(FIO_FCB *);
extern void *__fortio_fiofcb_next(FIO_FCB *);

extern bool __fio_eq_str(char *str, int len, char *pattern);
extern VOID __fortio_swap_bytes(char *, int, long);

#endif /* FLANG_RUNTIME_GLOBAL_DEFS_H_ */
