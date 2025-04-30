/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *  \brief accpp.c  -  PGC ansi source preprocessor module.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "machar.h"
#include "version.h"

#define ACC_DBG                                                         \
  fprintf(stderr, "%d: toktyp=%c(%d);  tokval=%s; line=%d\n", __LINE__, \
          toktyp, toktyp, tokval, cur_line);

/* structures and data local to this module : */

/* char classification macros */

#define _CS 1  /* c symbol */
#define _DI 2  /* digit */
#define _BL 4  /* blank */
#define _HD 8  /* hex digit */
#define _WS 16 /* white space (' ' or '\t') */

#define MASK(c) ((int)c & 0xFF)
#undef iscsym
#define iscsym(c) (ctable[MASK(c)] & _CS)
#define iswhite(c) (ctable[MASK(c)] & _WS)
#undef isblank
#define isblank(c) (ctable[MASK(c)] & _BL)
#define ishex(c) (ctable[MASK(c)] & (_HD | _DI))
#define isident(c) (ctable[MASK(c)] & (_CS | _DI))
#define isdig(c) (ctable[MASK(c)] & _DI)

static char ctable[256] = {
    0,         /* nul */
    0,         /* soh */
    0,         /* stx */
    0,         /* etx */
    0,         /* eot */
    0,         /* enq */
    0,         /* ack */
    0,         /* bel */
    0,         /* bs  */
    _BL | _WS, /* ht  */
    0,         /* nl  */
    _BL,       /* vt  */
    _BL,       /* ff  */
    _BL,       /* cr  */
    0,         /* so  */
    0,         /* si */
    0,         /* dle */
    0,         /* dc1 */
    0,         /* dc2 */
    0,         /* dc3 */
    0,         /* dc4 */
    0,         /* nak */
    0,         /* syn */
    0,         /* etb */
    0,         /* can */
    0,         /* em  */
    0,         /* sub */
    0,         /* esc */
    0,         /* fs  */
    0,         /* gs  */
    0,         /* rs  */
    0,         /* us  */
    _BL | _WS, /* sp  */
    0,         /* !  */
    0,         /* "  */
    0,         /* #  */
    _CS,       /* $  */
    0,         /* %  */
    0,         /* &  */
    0,         /* '  */
    0,         /* (  */
    0,         /* )  */
    0,         /* *  */
    0,         /* +  */
    0,         /* ,  */
    0,         /* -  */
    0,         /* .  */
    0,         /* /  */
    _DI,       /* 0  */
    _DI,       /* 1  */
    _DI,       /* 2  */
    _DI,       /* 3  */
    _DI,       /* 4  */
    _DI,       /* 5  */
    _DI,       /* 6  */
    _DI,       /* 7  */
    _DI,       /* 8  */
    _DI,       /* 9  */
    0,         /* :  */
    0,         /* ;  */
    0,         /* <  */
    0,         /* =  */
    0,         /* >  */
    0,         /* ?  */
    0,         /* @  */
    _CS | _HD, /* A  */
    _CS | _HD, /* B  */
    _CS | _HD, /* C  */
    _CS | _HD, /* D  */
    _CS | _HD, /* E  */
    _CS | _HD, /* F  */
    _CS,       /* G  */
    _CS,       /* H  */
    _CS,       /* I  */
    _CS,       /* J  */
    _CS,       /* K  */
    _CS,       /* L  */
    _CS,       /* M  */
    _CS,       /* N  */
    _CS,       /* O  */
    _CS,       /* P  */
    _CS,       /* Q  */
    _CS,       /* R  */
    _CS,       /* S  */
    _CS,       /* T  */
    _CS,       /* U  */
    _CS,       /* V  */
    _CS,       /* W  */
    _CS,       /* X  */
    _CS,       /* Y  */
    _CS,       /* Z  */
    0,         /* [  */
    0,         /* \  */
    0,         /* ]  */
    0,         /* ^  */
    _CS,       /* _  */
    0,         /* `  */
    _CS | _HD, /* a  */
    _CS | _HD, /* b  */
    _CS | _HD, /* c  */
    _CS | _HD, /* d  */
    _CS | _HD, /* e  */
    _CS | _HD, /* f  */
    _CS,       /* g  */
    _CS,       /* h  */
    _CS,       /* i  */
    _CS,       /* j  */
    _CS,       /* k  */
    _CS,       /* l  */
    _CS,       /* m  */
    _CS,       /* n  */
    _CS,       /* o  */
    _CS,       /* p  */
    _CS,       /* q  */
    _CS,       /* r  */
    _CS,       /* s  */
    _CS,       /* t  */
    _CS,       /* u  */
    _CS,       /* v  */
    _CS,       /* w  */
    _CS,       /* x  */
    _CS,       /* y  */
    _CS,       /* z  */
    0,         /* {  */
    0,         /* |  */
    0,         /* }  */
    0,         /* ~  */
    0,         /* del */
};

/* Fortran error indices do not match SCC, and this file will always use error
 * numbers specified by our C compiler.
 */
#define ERR_OFFSET 700

/* Error routine.
 * Preprocessor error codes less than 200 are assumed to be identical in both
 * Fortran and C compilers, except for error 23.
 */
#define pperror(n, x, sev)                               \
  do {                                                   \
    if (ERR_OFFSET && (n) == 23)                         \
      error(236, sev, startline, x, CNULL);              \
    else if ((n) < 200)                                  \
      error((n), sev, startline, x, CNULL);              \
    else                                                 \
      error((n) + ERR_OFFSET, sev, startline, x, CNULL); \
  } while (0)

#define pperr(n, sev) pperror(n, CNULL, sev)

#define LINELEN 12000
#define HASHSIZ 1031
#define TOKMAX 8192
#define IFMAX 20
#define MACMAX 6000
#define FORMALMAX 127 /* FS#14308 - set max formals to 127 */
#define ARGMAX 16384
#define TARGMAX 32768
#define MAX_PATHNAME_LEN 1024
#define MAXINC 20
#define MACSTK_MAX 100

/* Funny chars */
#define ARGST 0xFE
#define SENTINEL 0xFD
#define NOSUBST 0xFC
#define NOFUNC 0xFB
#define STRINGIZE 0xFA
#define CONCAT 0xF9
#define ENDTOKEN 0xF8
#define CONCAT1 0xF7
#define WHITESPACE 0xF6
#define PRAGMAOP 0xF5
#define VA_ARGS 0xF4
#define VA_REST 0xF3

/* Normal tokens */
#define T_IDENT 'a'
#define T_NSIDENT 'b'
#define T_NOFIDENT 'f'
#define T_PPNUM 'p'
#define T_POUND (-2)
#define T_STRING 's'
#define T_WSTRING 'w'
#define T_CHAR 'c'
#define T_WCHAR 'z'
#define T_OP 'o'
#define T_CONCAT (-5)
#define T_SENTINEL (-3)

#define D_IF 1
#define D_ELIF 2
#define D_ELSE 3
#define D_ENDIF 4
#define D_IFDEF 5
#define D_IFNDEF 6
#define D_INCLUDE 7
#define D_DEFINE 8
#define D_UNDEF 9
#define D_LINE 10
#define D_PRAGMA 11
#define D_ERROR 12
/* NON-ANSI directives go here -- D_MODULE is first NON-ANSI dir. */
#define D_MODULE 13
#define D_LIST 14
#define D_NOLIST 15
#define D_IDENT 16
#define D_INCLUDE_NEXT 17
#define D_WARNING 18

static FILE *ifp;

/* record for #if stack */
typedef struct {
  char truth;
  char true_seen;
  char else_seen;
} IFREC;

/* symbol table entry -- chained hash table */
typedef struct {
  short flags;
  short nformals;
  INT formals[FORMALMAX];
  INT name;
  INT value;
  INT next;
} PPSYM;

/* include stack */
typedef struct {
  char fname[MAX_PATHNAME_LEN];
  char dirname[MAX_PATHNAME_LEN];
  int lineno;
  int count;           /* Number of times include called recursively */
  int path_idx;        /* Where (its INCL_PATH index) the include file
                        * was found.
                        */
  LOGICAL from_stdinc; /* is a system header file */
  FILE *ifp;
} INCLSTACK;

#define F_PREDEF 1
#define F_FUNCLIKE 2
#define F_CHAR 4
#define F_VA_ARGS 8
#define F_VA_REST 16

/* if stack */
static int iftop;
static int ifsize;
static IFREC *_ifs;

/* hash table */
static INT hashtab[HASHSIZ];
static PPSYM *hashrec;
static INT nhash, next_hash;

/* string table */
static char *deftab;
static INT ndef, next_def;

/* suffix for dependent file, usually .o */
#ifdef TARGET_WIN
static const char *suffix = ".obj";
#else
static const char *suffix = ".o";
#endif

static const char *prevfile = NULL;
static int prevline = -1;

/* True if dodef() is parsing a #define string (nextok needs to know this) */
static LOGICAL in_dodef;

/* True if processing a fortran comment */
static LOGICAL in_ftn_comment;

/*
 * Tokens for parser.
 * Preprocessor constants must be interpreted as long (Standard 3.8.1).
 * When a long is the same size as an int (32 bits), it's sufficient to use
 * tobinary() and to represent constant expressions as ints.  When a long is
 * larger than an int, such as a 64-bit int, all constant expressions are
 * represented as a 64-bit value; tobinary64() is used to convert decimal
 * values into the 64-bit representation.
 *
 * There will be two sets of definitions for the parser token value and
 * macros representing the operations on a token value - the selection
 * is based on whether or not PPNUM_IS_32 is defined.
 */

#if LONG_IS_64
#undef PPNUM_IS_32
#else
#undef PPNUM_IS_32
#define PPNUM_IS_32
#endif

/*
 * c99 apparently defines a preprocessor number to be
 * any C number and any preprocessing integer arithmetic is defined to
 * be computed as intmax_t/uintmax_t values.  These types are typically
 * 64-bit types, so since pgcc supports 'long long':
 */
#undef PPNUM_IS_32

#if defined(PPNUM_IS_32)

#define FIXUL(a)

typedef struct {
  INT val;
  int isuns;
} PTOK;
#define PTOK_V(t) t.val
#define PTOK_VUNS(a) (UINT) PTOK_V(a)
#define PTOK_ISUNS(t) t.isuns
#else
typedef struct {
  DBLINT64 val;
  int isuns;
} PTOK;
#define PTOK_V(t) (t).val
#define PTOK_VUNS(a) (UINT *) PTOK_V(a)
#define PTOK_ISUNS(t) (t).isuns
#endif
/*
 * Define macros to perform misc. operations, generally of the form:
 *    a = a <op> b
 * Two sets of definitions are possible:
 * 1.  sizeof(long) == sizeof(int)
 * 2.  sizeof(long) >  sizeof(int)
 *
 * Note that for #2 on the ST100, only the rightmost 40 bits are significant.
 */
#if defined(PPNUM_IS_32)
#define PTOK_MUL(a, b) PTOK_V(a) *= PTOK_V(b)
#define PTOK_DIV(a, b) PTOK_V(a) /= PTOK_V(b)
#define PTOK_UDIV(a, b)                   \
  {                                       \
    UINT ui;                              \
    udiv(PTOK_VUNS(a), PTOK_VUNS(b), ui); \
    PTOK_V(a) = ui;                       \
  }

#define PTOK_MOD(a, b) PTOK_V(a) %= PTOK_V(b)
#define PTOK_UMOD(a, b)                   \
  {                                       \
    UINT ui;                              \
    umod(PTOK_VUNS(a), PTOK_VUNS(b), ui); \
    PTOK_V(a) = ui;                       \
  }

#define PTOK_ADD(a, b) PTOK_V(a) += PTOK_V(b)
#define PTOK_SUB(a, b) PTOK_V(a) -= PTOK_V(b)
#define PTOK_NEG(a, b) PTOK_V(a) = -PTOK_V(b)

#define PTOK_AND(a, b) PTOK_V(a) &= PTOK_V(b)
#define PTOK_OR(a, b) PTOK_V(a) |= PTOK_V(b)
#define PTOK_XOR(a, b) PTOK_V(a) ^= PTOK_V(b)
#define PTOK_BNOT(a, b) PTOK_V(a) = ~PTOK_V(b)
#define PTOK_LNOT(a, b) PTOK_V(a) = !PTOK_V(b)
#define PTOK_ANDAND(a, b) PTOK_V(a) = (PTOK_V(a) && PTOK_V(b))
#define PTOK_OROR(a, b) PTOK_V(a) = (PTOK_V(a) || PTOK_V(b))

#define PTOK_LSHIFT(a, b) PTOK_V(a) = LSHIFT(PTOK_V(a), PTOK_V(b))
#define PTOK_URSHIFT(a, b) PTOK_V(a) = URSHIFT(PTOK_V(a), PTOK_V(b))
#define PTOK_CRSHIFT(a, b) PTOK_V(a) = CRSHIFT(PTOK_V(a), PTOK_V(b))

#define PTOK_LOGEQ(a, b) PTOK_V(a) = PTOK_V(a) == PTOK_V(b)
#define PTOK_LOGNE(a, b) PTOK_V(a) = PTOK_V(a) != PTOK_V(b)
#define PTOK_LOGLT(a, b) PTOK_V(a) = PTOK_V(a) < PTOK_V(b)
#define PTOK_LOGLE(a, b) PTOK_V(a) = PTOK_V(a) <= PTOK_V(b)
#define PTOK_LOGGT(a, b) PTOK_V(a) = PTOK_V(a) > PTOK_V(b)
#define PTOK_LOGGE(a, b) PTOK_V(a) = PTOK_V(a) >= PTOK_V(b)

#define PTOK_UCMP(a, b) ucmp(PTOK_VUNS(a), PTOK_VUNS(b))
#define PTOK_LOGULT(a, b) PTOK_V(a) = PTOK_UCMP(a, b) < 0
#define PTOK_LOGULE(a, b) PTOK_V(a) = PTOK_UCMP(a, b) <= 0
#define PTOK_LOGUGT(a, b) PTOK_V(a) = PTOK_UCMP(a, b) > 0
#define PTOK_LOGUGE(a, b) PTOK_V(a) = PTOK_UCMP(a, b) >= 0

#define PTOK_ISNEG(a) (PTOK_V(a) < 0)
#define PTOK_ISNZ(a) (PTOK_V(a) != 0)
#define PTOK_ASSN(a, b) PTOK_V(a) = PTOK_V(b)

/* Misc. operations on the lsp (least significant portion) of a token; e.g.,
 *     ASSN32    - lsp = int; clear the most significant portion
 *     LSHIFT32  - lsp <<= int
 *     OR32      - lsp |=  int
 *     BTEST32   - test the bits of the lsp
 *     SEXTEND32 - signextend the lsp into the most significant portion
 */
#define PTOK_ASSN32(a, i) PTOK_V(a) = (i)
#define PTOK_LSHIFT32(a, i) PTOK_V(a) <<= (i)
#define PTOK_OR32(a, i) PTOK_V(a) |= (i)
#define PTOK_BTEST32(a, i) (PTOK_V(a) & (i))
#define PTOK_SEXTEND32(a)
#else
/*  LONG_IS_64 PTOK macro defs */

#define FIXUL(a)
#define MKXUL(a)

static DBLINT64 zero64 = {0, 0};
static void mod64(DBLINT64, DBLINT64, DBLINT64);
static void umod64(DBLUINT64, DBLUINT64, DBLUINT64);

#define PTOK_MUL(a, b) mul64(PTOK_V(a), PTOK_V(b), PTOK_V(a)) MKXUL(a)
#define PTOK_DIV(a, b) div64(PTOK_V(a), PTOK_V(b), PTOK_V(a))
#define PTOK_UDIV(a, b)                     \
  {                                         \
    DBLUINT64 ui;                              \
    udiv64(PTOK_VUNS(a), PTOK_VUNS(b), ui); \
    PTOK_V(a)[0] = ui[0];                   \
    PTOK_V(a)[1] = ui[1];                   \
    FIXUL(a);                               \
  }

#define PTOK_MOD(a, b) mod64(PTOK_V(a), PTOK_V(b), PTOK_V(a))
#define PTOK_UMOD(a, b) \
  umod64(PTOK_VUNS(a), PTOK_VUNS(b), PTOK_VUNS(a)) MKXUL(a)

#define PTOK_ADD(a, b) add64(PTOK_V(a), PTOK_V(b), PTOK_V(a)) MKXUL(a)
#define PTOK_SUB(a, b) sub64(PTOK_V(a), PTOK_V(b), PTOK_V(a)) MKXUL(a)
#define PTOK_NEG(a, b) neg64(PTOK_V(b), PTOK_V(a)) MKXUL(a)

#define PTOK_AND(a, b) and64(PTOK_V(a), PTOK_V(b), PTOK_V(a)) MKXUL(a)
#define PTOK_OR(a, b) or64(PTOK_V(a), PTOK_V(b), PTOK_V(a)) MKXUL(a)
#define PTOK_XOR(a, b) xor64(PTOK_V(a), PTOK_V(b), PTOK_V(a)) MKXUL(a)
#define PTOK_BNOT(a, b) not64(PTOK_V(b), PTOK_V(a)) MKXUL(a)
#define PTOK_LNOT(a, b) PTOK_ASSN32(a, cmp64(PTOK_V(b), zero64) == 0)
#define PTOK_ANDAND(a, b) \
  PTOK_ASSN32(a, cmp64(PTOK_V(a), zero64) && cmp64(PTOK_V(b), zero64))
#define PTOK_OROR(a, b) \
  PTOK_ASSN32(a, cmp64(PTOK_V(a), zero64) || cmp64(PTOK_V(b), zero64))

#define PTOK_LSHIFT(a, b) shf64(PTOK_V(a), PTOK_V(b)[1], PTOK_V(a)) MKXUL(a)
#define PTOK_URSHIFT(a, b) \
  ushf64(PTOK_VUNS(a), -PTOK_V(b)[1], PTOK_VUNS(a)) MKXUL(a)
#define PTOK_CRSHIFT(a, b) shf64(PTOK_V(a), -PTOK_V(b)[1], PTOK_V(a)) MKXUL(a)

#define PTOK_CMP(a, b) cmp64(PTOK_V(a), PTOK_V(b))
#define PTOK_LOGEQ(a, b) PTOK_ASSN32(a, PTOK_CMP(a, b) == 0)
#define PTOK_LOGNE(a, b) PTOK_ASSN32(a, PTOK_CMP(a, b) != 0)
#define PTOK_LOGLT(a, b) PTOK_ASSN32(a, PTOK_CMP(a, b) < 0)
#define PTOK_LOGLE(a, b) PTOK_ASSN32(a, PTOK_CMP(a, b) <= 0)
#define PTOK_LOGGT(a, b) PTOK_ASSN32(a, PTOK_CMP(a, b) > 0)
#define PTOK_LOGGE(a, b) PTOK_ASSN32(a, PTOK_CMP(a, b) >= 0)

#define PTOK_UCMP(a, b) ucmp64(PTOK_VUNS(a), PTOK_VUNS(b))
#define PTOK_LOGULT(a, b) PTOK_ASSN32(a, PTOK_UCMP(a, b) < 0)
#define PTOK_LOGULE(a, b) PTOK_ASSN32(a, PTOK_UCMP(a, b) <= 0)
#define PTOK_LOGUGT(a, b) PTOK_ASSN32(a, PTOK_UCMP(a, b) > 0)
#define PTOK_LOGUGE(a, b) PTOK_ASSN32(a, PTOK_UCMP(a, b) >= 0)

#define PTOK_ISNEG(a) (cmp64(PTOK_V(a), zero64) < 0)
#define PTOK_ISNZ(a) (cmp64(PTOK_V(a), zero64) != 0)
#define PTOK_ASSN(a, b) \
  (void)(PTOK_V(a)[0] = PTOK_V(b)[0], PTOK_V(a)[1] = PTOK_V(b)[1])

/* Misc. operations on the lsp (least significant portion) of a token; e.g.,
 *     ASSN32    - lsp = int; clear the most significant portion
 *     LSHIFT32  - lsp <<= int
 *     OR32      - lsp |=  int
 *     BTEST32   - test the bits of the lsp
 *     SEXTEND32 - signextend the lsp into the most significant portion
 */
#define PTOK_ASSN32(a, i) (void)(PTOK_V(a)[1] = (i), PTOK_V(a)[0] = 0)
#define PTOK_LSHIFT32(a, i) PTOK_V(a)[1] <<= (i)
#define PTOK_OR32(a, i) PTOK_V(a)[1] |= (i)
#define PTOK_BTEST32(a, i) (PTOK_V(a)[1] & (i))
#define PTOK_SEXTEND32(a)          \
  {                                \
    if (PTOK_V(a)[1] & 0x80000000) \
      PTOK_V(a)[0] = 0xffffffff;   \
    else                           \
      PTOK_V(a)[0] = 0;            \
  }
#endif

static int token;
static PTOK tokenval;
static int syntaxerr;

/* include stack */
static INCLSTACK *inclstack;
static INT incsize;
static INT inclev = 0;

/* List of include files processed */
typedef char INCLENTRY[MAX_PATHNAME_LEN];
static INCLENTRY *incllist;
static int incfiles = 0;
static int inclsize;

/* buffer for nextok */
static char *lineptr, *lineend, *savlptr;
static int linelen;
static INT linesize;
static char *linebuf;

/* __FILE__ and __LINE__ table offsets in hashrec */
static int lineloc, fileloc;

#define ifstack(name) _ifs[iftop].name

/* current file name, line number and system header flag*/
static char cur_fname[MAX_PATHNAME_LEN];
static int cur_line;
static LOGICAL cur_from_stdinc;
static char filename[MAX_PATHNAME_LEN] = {'\0'};
/* name of source file on #line statement */

/* directory containing current file */
static char dirwork[MAX_PATHNAME_LEN];

/* start line for error messages and #line */
static int startline;

static int pflag;     /* # number file output? */
static int list_flag; /* listing? */

/* macro substitution stack to detect macro recursion */
struct macstk_type {
  INT msptr;
  char *sav_lptr;
};
static struct macstk_type macstk[MACSTK_MAX];
static int macstk_top = -1;

static int toktyp;
static int nifile; /* position in flg.ifile */

static struct {
  /*  table of include directory paths, one per element */
  int *path;
  int cnt;
  int sz;
  /* string space for the paths */
  char *b;
  int b_avl;
  int b_sz;
  /* where last include was found */
  int last;
} idir;
#define INCLPATH(i) (idir.b + idir.path[i])

static LOGICAL cpp_comments; /* allow C++ style comments "//" to end-of-line */
static LOGICAL asm_mode = FALSE; /* preprocessing for assembler file */

#define GROW_ARGBUF(argbuf, argp, fstart, argbufsize, newsize) \
  {                                                            \
    int poffset = argp - argbuf;                               \
    int foffset = fstart - argbuf;                             \
    NEED(argbufsize + 1, argbuf, char, argbufsize, newsize);   \
    argp = argbuf + poffset;                                   \
    fstart = argbuf + foffset;                                 \
  }

/*  functions defined in this file:  */
extern void accpp(void);
extern void setasmmode(void);
static void delete(const char *);
static void pbchar(int);
static void pbstr(const char *);
static void clreol(int);
static int dlookup(char *);
static void dodef(int);
static void doincl(LOGICAL);
static void _doincl(char *, int, LOGICAL);
static void domodule(void);
static INT doparse(void);
static void doundef(int);
static int gtok(char *tokval, int expflag);
static void ifpush(void);
static PPSYM *lookup(const char *, int);
static void ptok(const char *);
static int subst(PPSYM *);
static INT strstore(const char *);
static INT tobinary(char *, int *, INT *);
static INT tobinary64(char *, int *, DBLINT64);
static int gettoken(void);
static void parse(int, PTOK *);
static void pr_line(const char *, int, LOGICAL);
static void doline(int, char *);
static int dopragma(void);
static void doerror(void);
static void dowarning(void);
static void doident(void);
static int findtok(char *, int);
static int macro_recur_check(PPSYM *);
static int predarg(char *);
static void predicate(char *);
#ifdef ATT_PREDICATE
static void add_predicate(char *);
#endif
static ptrdiff_t realloc_linebuf(ptrdiff_t);
static int nextok(char *, int);
static int _nextline(void);
static int mac_push(PPSYM *, char *);
static void popstack(void);
static void stash_paths(const char *);
static void dumpval(char *, FILE *);
#ifdef DUMPTAB
static void dumptab(void);
#endif
static void dumpmac(PPSYM *sp);
static void putmac(PPSYM *);
static void putunmac(char *);
static void update_actuals(char **, int, char *, char *);
static int skipbl(char *tokval, int flag);

static int
skipbl(char *tokval, int flag)
{
  int toktyp;

  for (toktyp = gtok(tokval, flag); iswhite(toktyp);
       toktyp = gtok(tokval, flag))
    ;

  return toktyp;
}

/** \brief
 * Replaces fprintf (ff, format, str)  ,
 * - checks for spaces in the string,
 * - and windows ':'
 * - and adds backspaces so that the output can be included in makefiles
 */
void
print_and_check(FILE *ff, char *str, char end_c)
{
  char *cp;
  char ch;

  for (cp = str; *cp; ++cp) {
    ch = *cp;
    if (ch == ' ')
      fputc('\\', ff);
#if defined(TARGET_WIN)
    /* c: changed to \c */
    if (isalpha(ch) && *(cp + 1) && *(cp + 1) == ':') {
      fprintf(ff, "/%c", ch);
      ++cp;
    } else
#endif

      fputc(ch, ff);
  }
  if (end_c)
    fputc(end_c, ff);
}

static void acc_display_macros()
{
  unsigned int i;
  for (i = 1U; i < next_hash; ++i) {
    char *name = &deftab[hashrec[i].name];
    char *value = &deftab[hashrec[i].value];
    if ((strncmp(name, "__LINE__", 8) == 0) ||
        (strncmp(name, "__DATE__", 8) == 0) ||
        (strncmp(name, "__FILE__", 8) == 0) ||
        (strncmp(name, "__TIME__", 8) == 0)) continue;
    fprintf(gbl.outfil, "%s : %s\n", name, value);
  }
}

void
accpp(void)
{
  int toktyp;
  char *p;
  PPSYM *sp;
  char **cp;
  int done;
  int i;
  INT mon;
  char tokval[TOKMAX];
  char **dirp;
  static char adate[] = "\"Mmm dd yyyy\"";
  static const char *months[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  static char atime[] = "\"hh:mm:ss\"";

  cpp_comments = FALSE;

/* -B or -c90x options:  Ignore C++ style comments "//" to end-of-line
 * 'cpp_comments' can only be true for C++, as "//" is the concatenation
 * operator in Fortran and we pass that on to the lexer in the Fortran case.
 */

  if (XBIT(123, 0x100000))
    ctable['$'] = 0;

  startline = -1;
  list_flag = flg.list;
  pflag = flg.es && !flg.p;

  NEW(inclstack, INCLSTACK, MAXINC);
  incsize = MAXINC;

  NEW(_ifs, IFREC, IFMAX);
  ifsize = IFMAX;

  strcpy(inclstack[0].fname, gbl.file_name);
  dirnam(inclstack[0].fname, inclstack[0].dirname);
  gbl.curr_file = gbl.file_name;
  inclstack[0].ifp = gbl.srcfil;
  inclstack[0].lineno = 0;
  inclstack[0].count = 1;
  inclstack[0].from_stdinc = FALSE;
  ifp = inclstack[0].ifp;

  linelen = LINELEN;
  linesize = (linelen + linelen);
  NEW(linebuf, char, linesize);

  lineptr = lineend = linebuf + linelen;

/* process def and undef command line options */

  for (cp = flg.def; cp && *cp; ++cp) {
    /* enter it */
    for (p = *cp; *p && *p != '='; ++p)
      ;
    if (!*p) {
      pbstr("1\n");
      pbchar(' ');
      pbstr(*cp);
      pbchar(' ');
      dodef(1);
    } else {
      char c;
      c = *p;
      *p = 0;
      pbchar('\n');
      if (p[1] != 0)
        pbstr(p + 1);
      pbchar(' ');
      pbstr(*cp);
      *p = c;
      pbchar(' ');
      dodef(1);
    }
  }

/* add ANSI std predefined macros  */
#define chkdef(name, val)        \
  do {                           \
    sp = lookup(name, 1);        \
    sp->flags |= F_PREDEF;       \
    sp->nformals = 0;            \
    if (sp->value != 0)          \
      pperror(245, name, 1);     \
    else {                       \
      sp->value = strstore(val); \
      if (XBIT(122, 0x40000))    \
        putmac(sp);              \
    }                            \
  } while(0)

  chkdef("__FLANG", "1");

  chkdef("__LINE__", "0");
  lineloc = sp - hashrec;

  chkdef("__FILE__", "1");
  fileloc = sp - hashrec;

  strncpy(&adate[2], gbl.datetime, 10);
  adate[4] = adate[7] = ' '; /* delete '/' */
  (void)atoxi(&adate[2], &mon, 2, 10);
  if (adate[5] == '0')
    adate[5] = ' ';
  strncpy(&adate[1], months[mon - 1], 3);
  strncpy(&atime[1], &gbl.datetime[12], 8);

  chkdef("__DATE__", adate);
  chkdef("__TIME__", atime);

  if (!XBIT(123, 0x400000)) {
    if (XBIT(123, 0x10) || XBIT(123, 0x80)) { /* -Xa or -Xt */
      chkdef("__STDC__", "0");
    } else if (XBIT(122, 2)) { /* -Xs */
      if ((sp = lookup("__STDC__", 0)) != 0) {
        if (sp->flags & F_PREDEF) {
          pperror(246, "__STDC__", 1);
          sp->flags &= ~F_PREDEF;
        }
        delete("__STDC__");
      }
    } else {
#ifdef TARGET_WIN
      /* Windows uses default -D__STDC__=0, matching Microsoft CL */
      chkdef("__STDC__", "0");
#else
      /* Linux uses default -D__STDC__=1, matching gcc */
      chkdef("__STDC__", "1");
                              /* Other targets:
                               *  SUA ??
                               *  OSX ??
                               */
#endif
    }
  }

  if (XBIT(123, 0x80000000)) {
    if (XBIT(122, 0x400000)) {
      chkdef("__STDC_VERSION__", "201112L");
    } else {
      chkdef("__STDC_VERSION__", "199901L");
    }
  }

/* Introduce macros for Fortran (if we are preprocessing Fortran code) */

#  define PP_MAJOR "__FLANG_MAJOR__"
#  define PP_MINOR "__FLANG_MINOR__"
#  define PP_PATCHLEVEL "__FLANG_PATCHLEVEL__"
    if (XBIT(124, 0x200000)) {
      chkdef("pgi", "1");
    }

  p = version.vsn;
  for (i = 0; i < TOKMAX - 1 && isdig(*p); i++)
    tokval[i] = *p++;
  tokval[i] = 0;
  if (i == 0) { /* no digits */
    chkdef(PP_MAJOR, "99");
    chkdef(PP_MINOR, "99");
  } else {
    while (*p != 0 && !isdig(*p)) /* skip non-digits */
        p++;
    chkdef(PP_MAJOR, tokval);
    for (i = 0; i < TOKMAX - 1 && isdig(*p); i++) tokval[i] = *p++;
    tokval[i] = 0;
    if (i == 0) { /* no digits */
      chkdef(PP_MINOR, "99");
    } else {
      chkdef(PP_MINOR, tokval);
    }
  }
  p = version.bld;
  while (*p != 0 && !isdig(*p)) /* skip non-digits */
    p++;
  for (i = 0; i < TOKMAX - 1 && isdig(*p); i++)
    tokval[i] = *p++;
  tokval[i] = 0;
  if (i == 0) { /* no digits */
    chkdef(PP_PATCHLEVEL, "99");
  } else {
    chkdef(PP_PATCHLEVEL, tokval);
  }

  /* value of _OPENMP is 201307, July, 2013 - version 4.0 */
  if (flg.smp && !XBIT(69, 1))
    chkdef("_OPENMP", "201307");

  sp = lookup("defined", 1);
  sp->flags |= F_PREDEF;
  sp->nformals = -1;

#ifdef TARGET_SUPPORTS_QUADFP
  chkdef("__flang_quadfp__", "1");
#endif

#undef chkdef

/* now undef from command line */
  for (cp = flg.undef; cp && *cp; ++cp) {
    /* undef it */
    if ((sp = lookup(*cp, 0)) != 0) {
      if (sp->flags & F_PREDEF) {
        pperror(246, *cp, 1);
        sp->flags &= ~F_PREDEF;
      }
      delete(*cp); /* should check if IDENT */
    }
  }

  if (flg.list_macros) {
    acc_display_macros();

    FREE(deftab);
    FREE(hashrec);
    /* FREE(inclstack); Do not free this, gbl.curr_file points to it */
    FREE(_ifs);
    FREE(linebuf);

    if (incllist != NULL) {
      FREE(incllist);
    }

    flg.es = TRUE;
    finish();
  }

  idir.b_avl = 0;
  idir.b_sz = 2048;
  NEW(idir.b, char, idir.b_sz);

  idir.cnt = 0;
  idir.sz = 64;
  NEW(idir.path, int, idir.sz);

  for (dirp = flg.idir; dirp && *dirp; dirp++)
    stash_paths(*dirp);
  if (XBIT(123, 0x1000)) {
    p = getenv("CPATH");
    if (p != NULL)
      stash_paths(p);
    p = getenv("C_INCLUDE_PATH");
    if (p != NULL)
      stash_paths(p);
  }
  if (flg.stdinc == 0)
    stash_paths(DIRSINCS);
  else if (flg.stdinc != (char *)1)
    stash_paths(flg.stdinc);

  inclstack[0].path_idx = idir.last = -1;

  /* at this point no predefined macros can be #undef'd */
  iftop = -1;
  ifpush();
  ifstack(truth) = 1;
  strcpy(cur_fname, inclstack[0].fname);
  cur_line = 1;
  cur_from_stdinc = inclstack[0].from_stdinc;
  nifile = 0;
  pr_line(cur_fname, cur_line, cur_from_stdinc);

  /* main processing loop */
  for (;;) {
    char *s_bl;
    toktyp = findtok(tokval, ifstack(truth));
    if (toktyp == EOF)
      break;
    s_bl = lineptr;
    toktyp =
        skipbl(tokval, 0); /* # JUNK -- JUNK is not subject to macro repl. */
    done = -1;             /* clreol with error message */
    if (toktyp == '\n') {  /* ANSI NULL directive */
      done = 1;            /* no clreol */
    } else if (toktyp == T_PPNUM) {
      done = 0;
      if (ifstack(truth)) {
        doline(0, tokval);
        done = 1;
      }
    } else if (toktyp != T_IDENT) {
      if (!XBIT(123, 0x200000)) {
        /* ansi sez this is an error */
        pperr(214, 2);
        done = 0; /* clreol with no error message */
      } else {
        ptok("#");
        if (iswhite(*s_bl))
          ptok(" "); /* put back one whitespace if any present */
        ptok(tokval);
        while ((toktyp = gtok(tokval, 1)) != EOF && toktyp != '\n')
          ptok(tokval);
        done = 1;
      }
    } else {
      switch (dlookup(tokval)) {
      case D_IF:
        done = 0;
        ifpush();
        if (_ifs[iftop - 1].truth) {
          done = 1;
          if (doparse()) {
            ifstack(truth) = 1;
            ifstack(true_seen) = 1;
          } else
            ifstack(truth) = 0;
        } else
          ifstack(truth) = 0;
        break;
      case D_ELIF:
        done = 0;
        if (iftop == 0)
          pperr(238, 3);
        else if (ifstack(else_seen))
          pperr(201, 3);
        else if (ifstack(true_seen))
          ifstack(truth) = 0;
        else if (!ifstack(truth) && !ifstack(true_seen) &&
                 _ifs[iftop - 1].truth) {
          done = 1;
          if (doparse()) {
            ifstack(true_seen) = 1;
            ifstack(truth) = 1;
          } else
            ifstack(truth) = 0;
        }
        break;
      case D_ELSE:
        done = 0;
        if (iftop == 0)
          pperr(239, 3);
        else if (ifstack(else_seen))
          pperr(202, 3);
        else if (ifstack(true_seen))
          ifstack(truth) = 0;
        else if (!ifstack(truth) && !ifstack(true_seen) &&
                 _ifs[iftop - 1].truth) {
          ifstack(true_seen) = 1;
          ifstack(truth) = 1;
        }
        if (iftop > 0)
          ifstack(else_seen) = 1;
        break;
      case D_ENDIF:
        done = 0;
        if (iftop == 0)
          pperr(240, 3);
        else
          --iftop;
        break;
      case D_IFDEF:
        done = -1;
        /*  ANS X3.159-1989: #ifdef have only one identifier
                        done = 0;
        */
        ifpush(); /* sets truth value to old truth value */
        toktyp = skipbl(tokval, 0);
        if (toktyp != T_IDENT) {
          pperr(231, 2);
          ifstack(truth) = 0;
          if (toktyp == '\n')
            done = 1;
        } else if (lookup(tokval, 0) == 0)
          ifstack(truth) = 0;
        else {
          ifstack(truth) &= 1;
          ifstack(true_seen) = 1;
        }
        break;
      case D_IFNDEF:
        done = -1;
        /*  ANS X3.159-1989: #ifndef have only one identifier
                        done = 0;
        */
        ifpush(); /* sets truth value to old truth value */
        toktyp = skipbl(tokval, 0);
        if (toktyp != T_IDENT) {
          pperr(232, 2);
          ifstack(truth) = 0;
          if (toktyp == '\n')
            done = 1;
        } else if (lookup(tokval, 0) != 0)
          ifstack(truth) = 0;
        else {
          ifstack(truth) &= 1;
          ifstack(true_seen) = 1;
        }
        break;
      case D_INCLUDE:
        done = 0;
        if (ifstack(truth)) {
          doincl(FALSE);
          done = 2;
        }
        break;
      case D_INCLUDE_NEXT:
        done = 0;
        if (ifstack(truth)) {
          doincl(TRUE);
          done = 2;
        }
        break;
      case D_DEFINE:
        done = 0;
        if (ifstack(truth)) {
          dodef(0);
          done = 1;
        }
        break;
      case D_UNDEF:
        done = 0;
        if (ifstack(truth)) {
          doundef(0);
          done = 1;
        }
        break;
      case D_LINE:
        done = 0;
        if (ifstack(truth)) {
          doline(1, (char *)0);
          done = 1;
        }
        break;
      case D_PRAGMA:
        done = 0;
        if (ifstack(truth)) {
          if (dopragma())
            /* returns non-zero if macro replacement is to occur
             * in the pragma line.
             */
            continue;
          done = 1;
        }
        break;
      case D_MODULE:
        done = 0;
        if (ifstack(truth)) {
          domodule();
          done = 1;
        }
        break;
      case D_LIST:
        list_flag = flg.list;
        break;
      case D_NOLIST:
        list_flag = 0;
        break;
      case D_IDENT:
        done = 0;
        if (ifstack(truth)) {
          doident();
          done = 1;
        }
        break;
      case D_ERROR:
        done = 0;
        if (ifstack(truth)) {
          doerror();
          done = 1;
        }
        break;
      case D_WARNING:
        done = 0;
        if (ifstack(truth)) {
          dowarning();
          done = 1;
        }
        break;
      default:
        if (!XBIT(123, 0x200000)) {
          if (ifstack(truth))
            pperror(236, tokval, 3);
          done = 0;
        } else {
          ptok("#");
          if (iswhite(*s_bl))
            ptok(" "); /* put back one whitespace if any present */
          /* toktyp == T_IDENT */
          if ((sp = lookup(tokval, 0)) != 0) {
            if (!macro_recur_check(sp)) {
              /* scan macro and substitute */
              int i;
              if ((i = subst(sp)) == EOF || i == T_SENTINEL) {
                toktyp = i;
                break;
              }
            }
            toktyp = T_NSIDENT;
          } else
            ptok(tokval);
          while ((toktyp = gtok(tokval, 1)) != EOF && toktyp != '\n')
            ptok(tokval);
          done = 1;
        }
        break;
      } /* switch */
    }   /* else */
    if (done < 1)
      clreol(-done); /* clreol resets in_ftn_comment */
    else if (done == 1) {
      ptok("\n");
      in_ftn_comment = FALSE;
    }
  } /* for */

  if (iftop != 0)
    pperr(218, 3);

  if (!flg.es)
    (void)fseek(gbl.cppfil, 0L, 0);

/* -M option:  Print list of include files to stdout */
/* -MD option:  Print list of include files to file <program>.d */
  if (XBIT(123, 2) || XBIT(123, 8)) {
    int count;
    if (gbl.dependfil == NULL) {
      if ((gbl.dependfil = tmpfile()) == NULL)
        errfatal(5);
    }
    count = strlen(gbl.module) + strlen(gbl.src_file) + 6;
    for (i = 0; i < incfiles; i++) {
      if ((count += (int)strlen(incllist[i])) >= 80) {
        fprintf(gbl.dependfil, "\\\n  ");
        count = strlen(incllist[i]) + 3;
      }
      if (!XBIT(123, 0x40000))
        fprintf(gbl.dependfil, "%s ", incllist[i]);
      else
        fprintf(gbl.dependfil, "\"%s\" ", incllist[i]);
    }
  }

  FREE(deftab);
  FREE(hashrec);
  /* FREE(inclstack); Do not free this, gbl.curr_file points to it */
  FREE(_ifs);
  FREE(linebuf);

  if (incllist != 0) {
    FREE(incllist);
  }
}

/** \brief
 * Format of pr_line output:
 * \# line "file_full_path" "from_stdinc"
 *
 * NB: -"form_stdinc" field is omitted in Fortran
 *     -If you modify "System_Header", make sure
 *     you modify get_line() in scan.c
 *     (both the string and its length)
 */
static void
pr_line(const char *name, int line, LOGICAL from_stdinc)
{
  static INT last_inclev = -1;

  /* -M option:  Print list of include files to stdout */
  if (XBIT(123, 2) || XBIT(123, 0x20000000) || XBIT(123, 0x40000000))
    return;

  if (pflag == 0) {
    if (inclev > 0 && inclev > last_inclev) {
      fprintf(gbl.cppfil, "# %d \"%s\"\n", inclstack[inclev - 1].lineno - 1,
              inclstack[inclev - 1].fname);
    }

    fprintf(gbl.cppfil, "# %d \"%s\"\n", line, name);
    last_inclev = inclev;
  }
}

static void
doline(int inflg, char *tokv)
{
  int toktyp;
  INT line;
  int isuns;
  char tokval[TOKMAX];

  if (inflg) {
    toktyp = skipbl(tokval, 1);
    if (toktyp != T_PPNUM || tobinary(tokval, &isuns, &line) != 0) {
      pperr(228, 2);
      clreol(0);
      return;
    }
  } else {
    if (tobinary(tokv, &isuns, &line) != 0) {
      pperr(228, 2);
      clreol(0);
      return;
    }
  }
  toktyp = skipbl(tokval, 1);
  if (toktyp == '\n') {
    /* increments next time */
    inclstack[inclev].lineno = line - 1;
    return;
  } else if (toktyp == T_STRING) {
    tokval[0] = 0;
    tokval[strlen(tokval + 1)] = 0;
    strcpy(inclstack[inclev].fname, tokval + 1);
    if (!filename[0]) {
      strcpy(filename, inclstack[inclev].fname);
      gbl.file_name = filename;
    }
    /* DON'T CHANGE dirnam */
    /* increments next time */
    inclstack[inclev].lineno = line - 1;
  } else if (toktyp == T_IDENT) {
    strcpy(inclstack[inclev].fname, tokval);
    /* DON'T CHANGE dirnam */
    /* increments next time */
    inclstack[inclev].lineno = line - 1;
  } else
    pperr(228, 2);
  clreol(XBIT(123, 0x1000000) ? 0 : 1);
}

static int
dopragma(void)
{
  char tokval[TOKMAX];
  int macro_repl;

  if (!XBIT(123, 0x2000))
    macro_repl = 0;
  else
    macro_repl = 1;
  ptok("\n"); /* pragma in col 1 */
  ptok("#pragma");
  if (!XBIT(123, 0x8000)) {
    for (toktyp = gtok(tokval, 0); iswhite(toktyp); toktyp = gtok(tokval, 0)) {
      ptok(tokval);
    }
    if (toktyp != EOF && toktyp != '\n') {
      ptok(tokval);
      if (toktyp == T_IDENT) {
        if (strcmp(tokval, "omp") == 0 || strcmp(tokval, "acc") == 0 ||
            strcmp(tokval, "cuda") == 0 || strcmp(tokval, "ident") == 0 ||
            strcmp(tokval, "pgi") == 0) {
          macro_repl = 1;
        } else if (strcmp(tokval, "STDC") == 0)
          macro_repl = 0;
      }
    } else
      ptok("\n");
  }
  if (!macro_repl) {
    /* Pass pragmas through to compiler */
    while ((toktyp = gtok(tokval, 0)) != EOF && toktyp != '\n')
      ptok(tokval);

    ptok("\n");
  }
  return macro_repl;
}

static void
doident(void)
{
  int toktyp;
  char tokval[TOKMAX];

  /* syntax is #ident "string" */
  /* intent is for "string" to appear in the object file */
  toktyp = skipbl(tokval, 1);
  if (toktyp != T_STRING) {
    pperr(250, 2);
    tokval[0] = 0;
    tokval[strlen(tokval + 1)] = 0;
    /* tokval+1 is the string to put in the object file */
  }
  if (toktyp != '\n')
    clreol(1);
}

static void
doerror(void)
{
  int toktyp;
  char tokval[TOKMAX];
  char buffer[TOKMAX];

  *buffer = 0;
  buffer[TOKMAX - 1] = 0;
  while ((toktyp = gtok(tokval, 0)) != EOF && toktyp != '\n') {
    strncat(buffer, tokval, TOKMAX - 1);
  }
  pperror(249, buffer, 4);
}

static void
dowarning(void)
{
  int toktyp;
  char tokval[TOKMAX];
  char buffer[TOKMAX];

  *buffer = 0;
  buffer[TOKMAX - 1] = 0;
  while ((toktyp = gtok(tokval, 0)) != EOF && toktyp != '\n') {
    strncat(buffer, tokval, TOKMAX - 1);
  }
  pperror(267, buffer, 2);
}

/** \param  erf
 * <pre>
 *- 0: do NOT print msg 251;
 *  -1/1: print msg 251(extraneous tokens ignored)
 * </pre>
 */
static void
clreol(int erf)
{
  int toktyp;
  char tokval[TOKMAX];
  int first = 0;
  LOGICAL is_fixed_form;

  is_fixed_form = flg.freeform;

  /* Scan to the end of the line (issue warnings if necessary) */
  while ((toktyp = gtok(tokval, 0)) != EOF && toktyp != '\n') {
    /* If fixed-form (72 column fortran) and in comment column, don't warn */
    if (is_fixed_form) {
      const int col = (lineptr - (linebuf + LINELEN + 1));
      if (col >= 73)
        first = 1; /* Set to one to prevent emitting a warning */
    }
    /* Only emit the error once (i.e., first has not been changed) */
    if (erf && first == 0 && !iswhite(toktyp)) {
      first = 1;
      pperr(251, 2);
    }
  }

  if (toktyp != '\n')
    pperr(252, 4);

  if (!XBIT(123, 0x80000))
    ptok("\n");

  in_ftn_comment = FALSE;
}

static int
dlookup(char *name)
{
  static struct {
    const char *name;
    int val;
  } directives[] = {{"define", D_DEFINE},
                    {"elif", D_ELIF},
                    {"else", D_ELSE},
                    {"endif", D_ENDIF},
                    {"error", D_ERROR},
                    {"ident", D_IDENT},
                    {"if", D_IF},
                    {"ifdef", D_IFDEF},
                    {"ifndef", D_IFNDEF},
                    {"include", D_INCLUDE},
                    {"include_next", D_INCLUDE_NEXT},
                    {"line", D_LINE},
                    {"list", D_LIST},
                    {"module", D_MODULE},
                    {"nolist", D_NOLIST},
                    {"pragma", D_PRAGMA},
                    {"undef", D_UNDEF},
                    {"warning", D_WARNING},
                    {"DEFINE", D_DEFINE},
                    {"ELIF", D_ELIF},
                    {"ELSE", D_ELSE},
                    {"ENDIF", D_ENDIF},
                    {"ERROR", D_ERROR},
                    {"IDENT", D_IDENT},
                    {"IF", D_IF},
                    {"IFDEF", D_IFDEF},
                    {"IFNDEF", D_IFNDEF},
                    {"INCLUDE", D_INCLUDE},
                    {"INCLUDE_NEXT", D_INCLUDE_NEXT},
                    {"LINE", D_LINE},
                    {"LIST", D_LIST},
                    {"MODULE", D_MODULE},
                    {"NOLIST", D_NOLIST},
                    {"PRAGMA", D_PRAGMA},
                    {"UNDEF", D_UNDEF},
                    {"WARNING", D_WARNING},
                    {0, 0}};
  int i;

  for (i = 0; directives[i].name; ++i)
    if (strcmp(directives[i].name, name) == 0)
      return (directives[i].val);
  return (0);
}

static int
inlist(int *list, int nl, char *value, char *macbuf)
{
  int i;

  for (i = 0; i < nl; ++i)
    if (strcmp(macbuf + list[i], value) == 0)
      return (i + 1);
  if (strcmp(value, "__VA_ARGS__") == 0)
    return -1;
  return (0);
}

/** \brief
 Default behavior:
         Add whitespace at the beginning and end of the macro body, and
         between tokens in the macro body.

 Behavior Modified by:
 - -Mx,123,x20:
         Do NOT add whitespace at the beginning, between and end of the
         macro body.
 - -Mx,123,x200:
         Add whitespace at beginning, end and between tokens in the macro
         body.  Do NOT add whitespace between tokens when OUTSIDE of a
         macro body.
 - -Mx,123,x800:
         Do NOT collapse consecutive whitespace ('cpp' mode).
 */
static void
dodef(int cmdline)
{
  int toktyp;
  PPSYM *sp;
  int nformals;
  char *macbuf;
  int macsize;
  int formals[FORMALMAX];
  char *p, *q, *r, *s;
  int defp;
  int i, n;
  int funclike;
  char tokval[TOKMAX];
  int prevtok;
  char tmpbuf[TOKMAX];
  int needspace = 1;

  NEW(macbuf, char, MACMAX);
  macsize = MACMAX;

  /* Tell nextok() that we are processing a #define */
  in_dodef = TRUE;

  /* fetch symbol after #define */
  p = macbuf;
  toktyp = skipbl(tokval, 0);
  if (toktyp != T_IDENT) {
    pperr(215, 2); /* illegal macro name is warning in cpp */
    clreol(0);
    goto ret;
  }
  /* look up the name */
  if ((sp = lookup(tokval, 1)) == 0) {
    /* must be predefined macro */
    clreol(0);
    goto ret;
  }
  /* here we need to read the definition */
  nformals = 0;
  funclike = 0;
  toktyp = gtok(tokval, 0);
  if (toktyp == '(') {
    int delim; /* just lex'd a delimitter, ( or , */
    delim = 1;
    funclike = F_FUNCLIKE;
    toktyp = skipbl(tokval, 0);
    for (;;) {
      if (toktyp == ')')
        break;
      if (toktyp != T_IDENT) {
        if (toktyp == T_OP && strcmp(tokval, "...") == 0) {
          /* next token better be ')' */
          toktyp = skipbl(tokval, 0);
          if (toktyp != ')') {
            pperror(224, "...", 3);
            clreol(0);
            goto nulldef;
          }
          if (delim) {
            funclike |= F_VA_ARGS;
            break;
          }
          if (XBIT(123, 0x1000) && !delim && nformals) {
            funclike |= F_VA_REST;
            break;
          }
          pperror(224, "...", 3);
          clreol(0);
          goto nulldef;
        } else {
          pperror(224, tokval, 3);
          clreol(0);
          goto nulldef;
        }
      }
      delim = 0;
      /* save formal */
      if (nformals + 1 > FORMALMAX)
        pperror(234, &deftab[sp->name], 3);
      else {

#define GROW_MACBUF(amt, p)                        \
  if (1) {                                         \
    int _n, _i, _offs;                             \
    _n = amt;                                      \
    if (_n < MACMAX)                               \
      _n = MACMAX;                                 \
    _offs = p - macbuf;                            \
    _i = _offs + amt;                              \
    NEED(_i, macbuf, char, macsize, macsize + _n); \
    p = macbuf + _offs;                            \
  } else

        n = strlen(tokval) + 1;
        /* Check for duplicate formal parameter */
        i = inlist(formals, nformals, tokval, macbuf);
        if (i) {
          if (i < 0)
            pperr(235, 3);
          else
            pperror(257, &deftab[sp->name], 2);
        }
        GROW_MACBUF(n, p);
        strcpy(p, tokval);
        formals[nformals++] = p - macbuf;
        p += n;
      }
      toktyp = skipbl(tokval, 0);
      if (toktyp == ',') {
        toktyp = skipbl(tokval, 0);
        delim = 1;
        continue;
      }
      if (toktyp == '\n') {
        pperror(242, &deftab[sp->name], 3);
        goto nulldef;
      }
    }
    toktyp = gtok(tokval, 0);
  }

  /*
   * now nformals has # formals, formals has formals, and token is first
   * after macro def
   */

  if (iswhite(toktyp)) {
    toktyp = skipbl(tokval, 0);
    needspace = 0;
  }
  if (toktyp == '\n') {
  nulldef:
    if (sp->value &&
        (sp->nformals != nformals || ((sp->flags & F_FUNCLIKE) != funclike) ||
         deftab[sp->value]))
      pperror(221, &deftab[sp->name], 2); /* redef is warning in cpp */
    else if (sp->value)
      pperror(222, &deftab[sp->name], 1);
    sp->value = strstore("");
    sp->flags |= funclike;
    sp->nformals = nformals;
    if ((XBIT(122, 0x10000) && !cmdline) || (XBIT(122, 0x40000) && cmdline))
      putmac(sp);
    goto ret;
  }
  /* collect definition */
  defp = p - macbuf;
  prevtok = 0;

  /* Add whitespace to the beginning of the macro body */
  if (!iswhite(toktyp) && !XBIT(123, 0x20)) {
    GROW_MACBUF(2, p);
    *p++ = WHITESPACE;
    needspace = 0;
  }

  for (;;) {
    if (toktyp == EOF) {
      pperror(209, &deftab[sp->name], 3);
      goto ret;
    }
    /* replace multiple white space by single space */
    if (iswhite(toktyp)) {
      toktyp = skipbl(tokval, 0);
      GROW_MACBUF(2, p);
      *p++ = ' ';
      needspace = 0;
      goto cont;
    }

    tmpbuf[0] = tokval[1]; /* character if T_CHAR toktyp */
    tmpbuf[1] = '\0';

    if (toktyp == T_IDENT &&
        (i = inlist(formals, nformals, tokval, macbuf)) == 0 &&
        prevtok == '#') {
      pperror(260, tokval, 3);
      clreol(0);
      goto ret;
    } else if (toktyp == T_IDENT &&
               (i = inlist(formals, nformals, tokval, macbuf)) != 0) {
      /* we found a formal */
      GROW_MACBUF(2, p);
      if (i == -1) { /* __VA_ARGS__ */
        if ((funclike & F_VA_ARGS) == 0) {
          pperr(235, 3);
          clreol(0);
          goto ret;
        }
        i = VA_ARGS;
      } else if ((funclike & F_VA_REST) && i == nformals) {
        i = VA_REST;
      }
      /* ident preceeded by '#' is string-ized */
      if (prevtok == '#') {
        while (iswhite(*--p)) /* delete '#' */
          ;
        --p; /* ENDTOKEN */
        *p++ = i;
        *p++ = STRINGIZE;
      } else if (prevtok == T_CONCAT) {
        /* delete white space before this token */
        while (iswhite(*--p))
          ;
        if (MASK(*p) == ENDTOKEN)
          --p;
        ++p;
        *p++ = i;
        *p++ = CONCAT1;
        /* delete white space after this token */
        toktyp = skipbl(tokval, 0);
        /*prevtok = T_CONCAT;*/
        prevtok = 0;
        goto cont;
      } else {

        /* Add whitespace between tokens in the macro body */
        if (!XBIT(123, 0x20) && needspace) {
          GROW_MACBUF(2, p);
          *p++ = WHITESPACE;
          needspace = 0;
        } else
          needspace = 1;

        /* make space for the next 3 */
        GROW_MACBUF(3, p);
        *p++ = i;
        /* comes last since subst scans backwards */
        *p++ = ARGST;
        *p++ = ENDTOKEN;
      }
    } else if (toktyp == T_CONCAT) {
      /* concatenation */
      needspace = 0;
      while (p > macbuf + defp && iswhite(*--p))
        ;
      /* ## directive placed at beginning of replacement list */
      if (p == macbuf + defp) {
        pperr(259, 3);
        clreol(0);
        goto ret;
      }
      if (MASK(*p) == ENDTOKEN)
        --p;
      if (MASK(*p) == ARGST || MASK(*p) == CONCAT1)
        *p = CONCAT;
      ++p;
      toktyp = skipbl(tokval, 0);
      /* ## directive placed at end of replacement list */
      if (toktyp == '\n') {
        pperr(259, 3);
        goto ret;
      }
      prevtok = T_CONCAT;
      goto cont;
    } else if (toktyp == T_CHAR &&
               (i = inlist(formals, nformals, tmpbuf, macbuf)) != 0) {

      if (!XBIT(123, 0x40)) { /* -Xs */
        /* "no macro replacement within a character constant" */
        pperror(261, CNULL, 2);

        n = strlen(tokval);
        GROW_MACBUF(n + 1, p);
        strcpy(p, tokval);
        p += n;
        *p++ = ENDTOKEN;
      } else {
        /* "macro replacement within a character constant" */
        pperror(262, CNULL, 2);

        GROW_MACBUF(2 + 3, p);
        *p++ = '\'';
        *p++ = i;
        /* comes last since subst scans backwards */
        *p++ = ARGST;
        *p++ = '\'';
        *p++ = ENDTOKEN;
        sp->flags |= F_CHAR;
      }
    } else if (toktyp == T_STRING) {
      int delim;
      n = strlen(tokval);
      GROW_MACBUF(n + 1, p);
      q = p;

      s = &tokval[1];
      *p++ = tokval[0];
      delim = tokval[0];
      while (*s && *s != delim) {

        while (*s == ' ')
          *p++ = *s++;
        r = &tmpbuf[0];
        while (*s && *s != ' ' && *s != delim) {
          *r++ = *s++;
          if (*(s - 1) == '\\')
            *r++ = *s++;
        }
        *r = '\0';

        i = inlist(formals, nformals, tmpbuf, macbuf);
        if (i > 0) {
          if (!XBIT(123, 0x40)) { /* -Xs, -Xt */
            /* "no macro replacement within a string literal" */
            p = q;
            strcpy(p, tokval);
            p += n;
            *p++ = ENDTOKEN;

            sp->flags &= ~F_CHAR;
            prevtok = toktyp;
            toktyp = gtok(tokval, 0);
            goto cont;
          }

          if (XBIT(123, 0x80)) { /* -Xt */
            if (!(sp->flags & F_CHAR))
              /* "macro replacement within a string literal" */
              pperror(264, CNULL, 2);
          }

          GROW_MACBUF(2, p);
          *p++ = i;
          /* comes last since subst scans backwards */
          *p++ = ARGST;
          *p++ = ENDTOKEN;
          sp->flags |= F_CHAR;
        } else {
          strcpy(p, tmpbuf);
          p += strlen(tmpbuf);
        }
      }
      if (*s)
        *p++ = *s;
      *p++ = ENDTOKEN;
    } else if (/*toktyp != '#' &&*/ toktyp != T_CONCAT) {

      /* Add whitespace between tokens in the macro body */
      if (!XBIT(123, 0x20) && needspace) {
        GROW_MACBUF(2, p);
        *p++ = WHITESPACE;
        needspace = 0;
      } else
        needspace = 1;

      n = strlen(tokval);
      GROW_MACBUF(n + 1, p);
      strcpy(p, tokval);
      p += n;
      *p++ = ENDTOKEN;
    }
    prevtok = toktyp;
    toktyp = gtok(tokval, 0);
  cont:
    if (toktyp == '\n')
      break;
  }
  /* strip trailing space */
  --p;
  while (iswhite(*p))
    --p;
  *++p = 0;
  /* install definition */
  /* compare definitions */
  if (sp->value &&
      (((sp->flags & F_FUNCLIKE) != funclike) || sp->nformals != nformals ||
       strcmp(macbuf + defp, &deftab[sp->value])))
    pperror(221, &deftab[sp->name], 2); /* warning in cpp */

  /* Test if the names of the formal parameters have been changed */
  if (sp->value && nformals == sp->nformals &&
      (sp->flags & F_FUNCLIKE) == funclike) {
    for (i = 0; i < nformals; i++) {
      char argbuf[16 + 1];
      if (strcmp(&macbuf[formals[i]], &deftab[sp->formals[i]])) {
        sprintf(argbuf, "%d", i + 1);
        error(258, 2, startline, argbuf, &deftab[sp->name]);
      }
    }
  }
  /* Save names of formal parameters */
  for (i = 0; i < nformals; i++) {
    sp->formals[i] = strstore(&macbuf[formals[i]]);
  }
  sp->value = strstore(macbuf + defp);
  sp->nformals = nformals;
  sp->flags |= funclike;
  if ((XBIT(122, 0x10000) && !cmdline) || (XBIT(122, 0x40000) && cmdline))
    putmac(sp);
  if (DBGBIT(2, 0x80))
    dumpmac(sp);

ret:
  FREE(macbuf);
#ifdef DUMPTAB
  dumptab();
#endif

  in_dodef = FALSE;
}

static void
doincl(LOGICAL include_next)
{
  int toktyp;
  char buff[MAX_PATHNAME_LEN];
  char *p;
  int type;
  char tokval[TOKMAX];
  int missing = 0;

  /* parse file name */
  toktyp = skipbl(tokval, 1);
  /* ### there is a problem here; "" filename not really a string */
  if (toktyp != T_STRING && toktyp != '<') {
    pperr(227, 3);
    return;
  }
  if (toktyp == T_STRING) {
    char delim;
    delim = *tokval;
    strncpy(buff, tokval + 1, MAX_PATHNAME_LEN);
    if ((p = strchr(buff, delim)) != NULL)
      *p = 0;
    buff[MAX_PATHNAME_LEN - 1] = 0;
    type = 0;
  } else {
    /* toktyp == '<' */
    buff[0] = 0;
    for (;;) {
      toktyp = gtok(tokval, 0);
      if (toktyp == '>')
        break;
      if (toktyp == '\n') {
        missing = 1;
        pperr(256, 2);
        break;
      }
      if (toktyp == EOF) {
        pperr(237, 3);
        break;
      }
      strncat(buff, tokval, MAX_PATHNAME_LEN);
    }
    buff[MAX_PATHNAME_LEN - 1] = 0;
    type = 1;
  }
  if (!missing)
    clreol(1);

  _doincl(buff, type, include_next);

}

#ifdef FLANG1_ACCPP_UNUSED
/** \brief
 * Check if fullname is a file in a standard include directory.
 */
static LOGICAL
is_in_stdinc(const char *fullname)
{
  const char *dirs = flg.stdinc;

  /* This x-bit disables our distinguishing system header files from normal
   * includes. */
  if (XBIT(123, 0x800000))
    return FALSE;

  /* Was stdinc even set? */
  if (dirs == NULL || dirs == (char *)1)
    return FALSE;

  /* The stdinc path contains directories separated by DIRSEP. */
  while (TRUE) {
    const char *path;

    /* Skip leading separators. */
    if (*dirs == DIRSEP) {
      dirs++;
      continue;
    }

    /* End of path reached. */
    if (*dirs == '\0')
      break;

    /* Remember beginning of path and advance dirs to the end. */
    path = dirs;
    while ((*dirs != '\0') && (*dirs != DIRSEP))
      dirs++;

    /* Check for a prefix match. */
    if (strncmp(fullname, path, (dirs - path)) == 0)
      return TRUE;
  }
  return FALSE;
}
#endif

/** \brief
 * Add fullname to the list of include files in incllist.
 *
 * This list is used to generate build system dependencies with the -M, -MM,
 * -MD, and -MMD options.
 *
 * The fullname buffer will be modified by this function.
 *
 * The file should be added to inclstack with a correct sys bit before calling
 * this function. The inclstack is inspected to implement -MM and -MMD
 * correctly.
 */
static void
add_to_incllist(char *fullname)
{
  INT i;

  /* Bail early if dependency generation is not enabled. */
  if (!XBIT(123, 2) &&          /* -M / -MM    */
      !XBIT(123, 8) &&          /* -MD / -MMD  */
      !XBIT(123, 0x20000000) && /* -MT         */
      !XBIT(123, 0x40000000))   /* -MQ         */
    return;

  /* When -MM or -MMD are active, skip system headers and any file included
   * from a system header directly or indirectly.
   *
   * This means we check the entire include stack for system headers. This
   * matches the behavior of GCC 3.1+ and Clang.
   */
  if (XBIT(123, 0x4000))
    for (i = inclev; i >= 0; i--)
      if (inclstack[i].from_stdinc)
        return;

  /* Remove duplicates. */
  for (i = 0; i < incfiles; i++)
    if (strcmp(incllist[i], fullname) == 0)
      return;

  /* Finally copy fullname to the end of incllist. */
  if (incllist == 0) {
    inclsize = 20;
    NEW(incllist, INCLENTRY, inclsize);
  } else {
    NEED(incfiles + 1, incllist, INCLENTRY, inclsize, inclsize + 20);
  }
  strcpy(incllist[incfiles], fullname);
  incfiles++;
}

/** \param type:
 * <pre>
 * -0  --  "file_name"
 * -1  --  <file_name>
 * </pre>
 */
static void
_doincl(char *name, int type, LOGICAL include_next)
{
  FILE *tmpfp;
  char fullname[MAX_PATHNAME_LEN];
  int i;

  NEED(inclev + 2, inclstack, INCLSTACK, incsize, incsize + MAXINC);

  /* look for included file:  */

  if (type == 0 && (idir.last == -1 || !include_next)) {
    /* look first in directory containing current included or base file */
    strcpy(dirwork, inclstack[inclev].dirname);
    i = strlen(dirwork) - 1;
    if (i >= 0 && dirwork[i] == '/')
      dirwork[i] = 0;
    if (fndpath(name, fullname, MAX_PATHNAME_LEN, dirwork) == 0) {
      idir.last = 0;
      goto found;
    }
  }
  i = 1;
  if (include_next && idir.last >= 0)
    i = idir.last + 1;
  for (; i <= idir.cnt; i++)
    if (fndpath(name, fullname, MAX_PATHNAME_LEN, INCLPATH(i)) == 0) {
      idir.last = i;
      goto found;
    }

  if (type == 0) { /* could be absolute path, check where it leads to */
    tmpfp = fopen(name, "r");
    if (tmpfp) {
      snprintf(fullname, MAX_PATHNAME_LEN, "%s", name);
      idir.last = 0;
      if (fclose(tmpfp) == 0)
        goto found;
    }
  }

  pperror(206, name, 4); /* cpp just continues, but why?? */
  return;

found:
  /* we need to increment the line # for this level */
  ++inclstack[inclev].lineno;
  ++inclev;
  if ((ifp = inclstack[inclev].ifp = fopen(fullname, "r")) == NULL) {
    --inclev; /* failed to open file so retract changes */
    --inclstack[inclev].lineno;
    error(2, 4, 0, fullname, CNULL);
    return;
  }

  /* Test for recursive includes */
  for (i = 0; i < inclev - 1; i++) {
    if (!strcmp(inclstack[i].fname, fullname)) {
      if (++inclstack[i].count > 37) {
        pperror(265, fullname, 4);
      }
      break;
    }
  }
  if (XBIT(122, 0x20000)) {
    if (gbl.curr_file != prevfile || startline != prevline + 1)
      printf("\n// %s:%d\n", gbl.curr_file, startline);
    prevfile = gbl.curr_file;
    prevline = startline;
    printf("#include%s %s\n", include_next ? "_next" : "", fullname);
  }

  strcpy(inclstack[inclev].fname, fullname);
  dirnam(inclstack[inclev].fname, inclstack[inclev].dirname);
  inclstack[inclev].count = 1;
  gbl.curr_file = inclstack[inclev].fname;
  inclstack[inclev].lineno = 0;
  inclstack[inclev].from_stdinc = FALSE;
  inclstack[inclev].path_idx = idir.last;

  pr_line(fullname, 1, inclstack[inclev].from_stdinc);
  add_to_incllist(fullname);
}

#ifdef FLANG1_ACCPP_UNUSED
static void
preincl(char *prename)
{
  int toktyp;
  char buff[MAX_PATHNAME_LEN];
  char fullname[MAX_PATHNAME_LEN];
  char *p;
  int type;
  char tokval[TOKMAX];
  int i;
  int missing = 0;

  /* parse file name */
  type = 0;

  NEED(inclev + 2, inclstack, INCLSTACK, incsize, incsize + MAXINC);

  /* look for included file:  */

  /* look first in directory containing current included or base file */
  strcpy(dirwork, inclstack[inclev].dirname);
  i = strlen(dirwork) - 1;
  if (i >= 0 && dirwork[i] == '/')
    dirwork[i] = 0;
  if (fndpath(prename, fullname, MAX_PATHNAME_LEN, dirwork) == 0) {
    idir.last = 0;
    goto found;
  }
  i = 1;
  for (; i <= idir.cnt; i++)
    if (fndpath(prename, fullname, MAX_PATHNAME_LEN, INCLPATH(i)) == 0) {
      idir.last = i;
      goto found;
    }

  pperror(206, prename, 4); /* cpp just continues, but why?? */
  return;

found:
  /* we need to increment the line # for this level */
  ++inclev;
  if ((ifp = inclstack[inclev].ifp = fopen(fullname, "r")) == NULL) {
    --inclev; /* failed to open file so retract changes */
    --inclstack[inclev].lineno;
    error(2, 4, 0, fullname, CNULL);
    return;
  }

  /* Test for recursive includes */
  for (i = 0; i < inclev - 1; i++) {
    if (!strcmp(inclstack[i].fname, fullname)) {
      if (++inclstack[i].count > 37) {
        pperror(265, fullname, 4);
      }
      break;
    }
  }
  if (XBIT(122, 0x20000)) {
    if (gbl.curr_file != prevfile || startline != prevline + 1)
      printf("\n// %s:%d\n", gbl.curr_file, startline);
    prevfile = gbl.curr_file;
    prevline = startline;
    printf("#include %s\n", fullname);
  }

  strcpy(inclstack[inclev].fname, fullname);
  dirnam(inclstack[inclev].fname, inclstack[inclev].dirname);
  inclstack[inclev].count = 1;
  gbl.curr_file = inclstack[inclev].fname;
  inclstack[inclev].lineno = 0;
  inclstack[inclev].from_stdinc = FALSE;
  inclstack[inclev].path_idx = idir.last;

  pr_line(fullname, 1, inclstack[inclev].from_stdinc);
  add_to_incllist(fullname);
} /* preincl */
#endif

static void
domodule(void)
{
  int toktyp;
  char *cp;
  int i;
  char tokval[TOKMAX];

  toktyp = skipbl(tokval, 1);
  if (toktyp != T_IDENT) {
    pperr(229, 2);
    if (toktyp != '\n')
      clreol(0);
    return;
  }
  NEW(cp, char, MAXIDLEN + 1);
  gbl.module = cp;
  for (i = 0; i < MAXIDLEN; ++i) {
    if (tokval[i] == 0)
      break;
    *cp++ = tokval[i];
  }
  *cp = 0;
  clreol(1);
}

static void
doundef(int cmdline)
{
  int toktyp;
  char tokval[TOKMAX];

  toktyp = skipbl(tokval, 0);
  if (toktyp != T_IDENT) {
    pperr(230, 2);
    if (toktyp != '\n')
      clreol(0);
  } else {
    delete(tokval);
    if ((XBIT(122, 0x10000) && !cmdline) || (XBIT(122, 0x40000) && cmdline))
      putunmac(tokval);
    clreol(1);
  }
}

static int
subst(PPSYM *sp)
{
  char *p, *q, *r, *r1;
  int nformals;
  int toktyp;
  char *actuals[FORMALMAX];
  int nactuals;
  char *argbuf;
  char *oldargbuf;
  char *targbuf;
  char *blank;
  char *argp;
  char *fstart;
  char tmp;
  int nlpar;
  int i, waswhite, instr;
  char tokval[TOKMAX];
  int argbufsize, targbufsize, newsize;
  int nl;
  int savech;
  size_t len;

#ifdef SUBST
  fprintf(stderr, "subst(%s)\n", deftab + sp->name);
#endif
  NEW(argbuf, char, ARGMAX + 1);
  argbufsize = ARGMAX + 1;

  NEW(targbuf, char, TARGMAX + 1);
  targbufsize = TARGMAX + 1;

  blank = strdup(""); /* required for assigning to actuals */

  /* Do not replace 'C' or 'c' in first column, yes, some people might define
   * a macro named 'C' or 'c'... people like myself.  --ignoramous
   */
  if (lineptr == (linebuf + LINELEN + 1)) {
    tmp = *(linebuf + LINELEN); /* First char in col 1 */
    if (tmp == 'c' || tmp == 'C') {
      if (tmp == 'c') /* Use ptok to skip to next token */
        ptok("c");
      else
        ptok("C");
      FREE(argbuf);
      FREE(targbuf);
      FREE(blank);
      return 0;
    }
  }

  if (sp == hashrec + lineloc) {
    sprintf(argbuf, "%d", inclstack[inclev].lineno);
    pbchar(ENDTOKEN);
    pbstr(argbuf);
    FREE(argbuf);
    FREE(targbuf);
    FREE(blank);
    return 0;
  } else if (sp == hashrec + fileloc) {
    /* escape backslashes in file name */
    argbuf[0] = '\"';
    for (p = inclstack[inclev].fname, i = 1; p && *p; p++, i++) {
      if (*p == '\\')
        argbuf[i++] = '\\';
      argbuf[i] = *p;
    }
    argbuf[i++] = '\"';
    argbuf[i] = '\0';
    pbchar(ENDTOKEN);
    pbstr(argbuf);
#ifdef SUBST
    fprintf(stderr, "subst(%s) returns\n", deftab + sp->name);
#endif
    FREE(argbuf);
    FREE(targbuf);
    FREE(blank);
    return 0;
  }
  argbuf[argbufsize - 1] = 0;
  p = &deftab[sp->value];
  nformals = sp->nformals;
  if (!(sp->flags & F_FUNCLIKE)) {
    /* macro has no args; just substitute */
    mac_push(sp, lineptr);
    pbstr(p);
#ifdef SUBST
    fprintf(stderr, "subst(%s) returns\n", deftab + sp->name);
#endif
    FREE(argbuf);
    FREE(targbuf);
    FREE(blank);
    return 0;
  }
  /* scan arguments */
  i = 0;
  nl = 0;
  for (toktyp = gtok(tokval, 0); isblank(toktyp) || toktyp == '\n';
       toktyp = gtok(tokval, 0)) {
    ++i;
    if (toktyp == '\n')
      ++nl;
  }
  if (toktyp != '(') { /* funclike not recognized */
    if (toktyp == EOF) {
      ptok(&deftab[sp->name]);
      ptok("\n");
      FREE(argbuf);
      FREE(targbuf);
      FREE(blank);
      return EOF;
    }

    if (toktyp == '#') {
      /* macro has no args; do NOT substitute */
      ptok(&deftab[sp->name]);
      FREE(argbuf);
      FREE(targbuf);
      FREE(blank);
      return (T_SENTINEL);
    }
    pbstr(tokval);
    if (nl) { /* FS#14715 need to preserve newline */
      pbchar('\n');
      cur_line++;
    }
    if (toktyp == T_NSIDENT)
      pbchar(NOSUBST);
    if (i)
      pbchar(' ');
    pbstr(&deftab[sp->name]);
    pbchar(NOFUNC);
#ifdef SUBST
    fprintf(stderr, "subst(%s) returns\n", deftab + sp->name);
#endif
    FREE(argbuf);
    FREE(targbuf);
    FREE(blank);
    return 0;
  }
  argp = argbuf;
  nactuals = 0;
  nlpar = 1;
  fstart = argp;
  for (;;) {
    toktyp = gtok(tokval, 0);
    if (toktyp == ')') {
      --nlpar;
      if (nlpar == 0)
        break;
    } else if (toktyp == '(')
      ++nlpar;
    else if (toktyp == ',') {
      if (nlpar == 1) {
        /* save actual */
        if (nactuals + 1 > FORMALMAX)
          pperror(233, &deftab[sp->name], 3);
        else if (argp - argbuf >= argbufsize - 1) {
          oldargbuf = argbuf;
          newsize = argp - argbuf + ARGMAX;
          GROW_ARGBUF(argbuf, argp, fstart, argbufsize, newsize);
          update_actuals(actuals, nactuals, oldargbuf, argbuf);
        }
        if (argp == fstart) {
          /* macro(,) case */
          *fstart = 0;
          ++argp;
        } else {
          while (iswhite(*fstart))
            ++fstart;
          --argp;
          while (iswhite(*argp) && argp > fstart)
            --argp;
          *++argp = 0;
          ++argp;
        }
        actuals[nactuals++] = fstart;
        fstart = argp;
        continue;
      }
    } else if (toktyp == EOF || toktyp == T_SENTINEL) {
      pperror(209, &deftab[sp->name], 3);
      FREE(argbuf);
      FREE(targbuf);
      FREE(blank);
      return toktyp;
    }
    if (argp - argbuf + (int)strlen(tokval) + 2 > argbufsize - 1) {
      newsize = argbufsize + (int)strlen(tokval) + 2 + ARGMAX;
      oldargbuf = argbuf;
      GROW_ARGBUF(argbuf, argp, fstart, argbufsize, newsize);
      update_actuals(actuals, nactuals, oldargbuf, argbuf);
    }
    if (iswhite(toktyp) || toktyp == '\n') {
      if ((argp > fstart) && *(argp - 1) != ' ') {
        if (argp - argbuf + 1 > argbufsize - 1) {
          newsize = argp - argbuf + 1 + ARGMAX;
          oldargbuf = argbuf;
          GROW_ARGBUF(argbuf, argp, fstart, argbufsize, newsize);
          update_actuals(actuals, nactuals, oldargbuf, argbuf);
        }
        *argp++ = ' ';
      }
      continue;
    }
    if (toktyp == T_NSIDENT)
      *argp++ = NOSUBST;
    strcpy(argp, tokval);
    argp += strlen(tokval);
    *argp++ = ENDTOKEN;
  }
  /* save last actual */
  if (nactuals + 1 > FORMALMAX)
    pperror(233, &deftab[sp->name], 3);
  else if (argp - argbuf >= argbufsize - 1) {
    oldargbuf = argbuf;
    newsize = argp - argbuf + 1 + ARGMAX;
    GROW_ARGBUF(argbuf, argp, fstart, argbufsize, newsize);
    update_actuals(actuals, nactuals, oldargbuf, argbuf);
  }
  if (argp == fstart) {
    *fstart = 0;
  } else {
    while (iswhite(*fstart))
      ++fstart;
    --argp;
    while (iswhite(*argp))
      --argp;
    *++argp = 0;
  }
  actuals[nactuals++] = fstart;

  /* substitute */
  if (nformals != nactuals) {
    if (!(nformals == 0 && nactuals == 1 && *actuals[0] == '\0')) {
      if ((sp->flags & (F_VA_ARGS | F_VA_REST))) {
        ; /* ok */
      } else
        pperror(205, &deftab[sp->name], 2);
    }
    for (i = nactuals; i < nformals; i++)
      actuals[i] = blank;
  }
  q = p;
  p += strlen(p); /* end of macro text */
  savlptr = lineptr;
  for (;;) {
    /* look for "funny" characters */
    while ((*--p & 0x80) == 0 && p >= q)
      pbchar(*p);
    if (p < q)
      break;
    switch (MASK(*p)) {
      int itmp;
      int iend;

    case STRINGIZE:
      --p;
      if (MASK(*p) == VA_ARGS) {
        if (nactuals <= nformals) {
          pbchar('"');
          pbchar('"');
          break;
        }
        i = nactuals - 1;
        iend = nformals;
      } else if (MASK(*p) == VA_REST) {
        /* NOTE: nformals includes the 'rest' argument */
        if (nactuals < nformals) {
          pbchar('"');
          pbchar('"');
          break;
        }
        i = nactuals - 1;
        iend = nformals - 1;
      } else
        iend = i = *p - 1;
      pbchar('"');
      for (; i >= iend; i--) {
        r = r1 = actuals[i];
        r += strlen(r);
        waswhite = 0;
        instr = 0;
        --r;
        while (iswhite(*r) && r >= r1)
          --r;
        while (r >= r1) {
          if (iswhite(*r) && !instr)
            waswhite = 1;
          else {
            if (waswhite) {
              waswhite = 0;
              pbchar(' ');
            }
            if (instr) {
              if (*r == '\\') {
                pbchar(*r);
                pbchar('\\');
              } else if (*r == '"') {
                pbchar(*r);
                pbchar('\\');
                if (r > r1 && r[-1] != '\\')
                  instr = 0;
              } else if ((*r & 0x80) == 0)
                pbchar(*r);
            } else if (*r == '"') {
              instr = 1;
              pbchar(*r);
              pbchar('\\');
            } else if ((*r & 0x80) == 0)
              pbchar(*r);
          }
          --r;
        }
        if (i != iend)
          pbstr(", ");
      }
      pbchar('"');
      break;

    case CONCAT:
      /* delete trailing ENDTOKEN */
      --p;
      if (MASK(*p) == VA_ARGS) {
        if (nactuals <= nformals)
          continue;
        i = nactuals - 1;
      } else if (MASK(*p) == VA_REST) {
        /* NOTE: nformals includes the 'rest' argument */
        if (nactuals <= nformals)
          continue;
        i = nactuals - 1;
      } else
        i = *p - 1;
      itmp = strlen(actuals[i]);
      if (itmp && MASK(actuals[i][itmp - 1]) == ENDTOKEN) {
        savech = actuals[i][itmp - 1];
        actuals[i][itmp - 1] = 0;
      } else
        savech = 0;
      goto concat_shared;

    case CONCAT1:
      /* delete trailing ENDTOKEN */
      --p;
      savech = 0;
    concat_shared:
      if (MASK(*p) == VA_ARGS) {
        if (nactuals <= nformals) {
          if (XBIT(123, 0x1000)) {
            /* `##' before __VA_ARGS__ discards the preceding
             * sequence of non-whitespace characters and ','
             * from the macro definition iff '...' args are not
             * present.
             */
            char *qq;
            for (qq = p - 1; qq >= q; qq--) {
              switch (MASK(*qq)) {
              case WHITESPACE:
                continue;
              case ',':
                p = qq;
                break;
              default:
                break;
              }
              break;
            }
          }
          continue;
        }
        for (i = nactuals - 1; i >= nformals; i--) {
          pbstr(actuals[i]);
          if (i != nformals) {
            pbchar(ENDTOKEN);
            pbchar(',');
          }
        }
      } else if (MASK(*p) == VA_REST) {
        /* NOTE: nformals includes the 'rest' argument */
        if (nactuals < nformals) {
          /* `##' before a rest argument that is empty discards the
           * preceding sequence of non-whitespace characters from
           * the macro definition. (If another macro argument
           * precedes, none of it is discarded.)
           */
          char *qq;
          for (qq = p - 1; qq >= q; qq--) {
            switch (MASK(*qq)) {
            case WHITESPACE:
              continue;
            case CONCAT:
              *qq = ARGST;
              FLANG_FALLTHROUGH;
            case ARGST:
              qq++;
              break;
            default:
              break;
            }
            break;
          }
          p = qq;
          continue;
        }
        for (i = nactuals - 1; i >= nformals - 1; i--) {
          pbstr(actuals[i]);
          if (i != nformals - 1) {
            pbchar(ENDTOKEN);
            pbchar(',');
          }
        }
      } else if (*p <= nactuals) {
        if (MASK(*actuals[*p - 1]) == NOSUBST) {
          pbstr(actuals[*p - 1] + 1);
        } else {
          pbstr(actuals[*p - 1]);
        }
        if (savech)
          actuals[i][itmp - 1] = savech;
      }
      break;

    case WHITESPACE:
    case ENDTOKEN:
      break;

    case PRAGMAOP:
      if (nactuals != 1) {
        pperror(269, "requires one argument which must be a string", 3);
        break;
      }
      r = r1 = actuals[0];
      if (*r1 == 'L')
        r1++;

      /* Special case: The first (and only) argument to _Pragma does not start
       * with a leading '"'.  Try to expand and see if we get a '"', if not,
       * fail.
       */
      if (*r1 != '"') {
        /* Expand the argument (can only be one argument) */
        pbchar(SENTINEL);
        pbstr(actuals[0]);
        toktyp = gtok(tokval, 1);
        if (toktyp != T_STRING)
          goto pragmaop_err;

        /* Save the string to a buffer, and terminate with ENDTOKEN */
        argp = targbuf;
        fstart = argp;
        if (argp - targbuf + (int)strlen(tokval) + 2 > targbufsize - 1) {
          newsize = argp - targbuf + (int)strlen(tokval) + 2 + TARGMAX;
          GROW_ARGBUF(targbuf, argp, fstart, targbufsize, newsize);
        }
        r = r1 = strcpy(argp, tokval);
        len = strlen(r);
        r[len + 0] = ENDTOKEN; /* " ENDTOKEN \0 */
        r[len + 1] = '\0';

        /* Sanity test: Better be the SENTINEL we pushed */
        toktyp = gtok(tokval, 0);

        /* Remove any literal prefixes or the leading quote */
        if (*r1 == 'L')
          ++r1;
        if ((toktyp != T_SENTINEL) || *r1 != '"') {
        pragmaop_err:
          while (toktyp != T_SENTINEL) /* Erase the sentinel that we added */
            toktyp = gtok(tokval, 0);
          pperror(269, "requires one argument which must be a string", 3);
          break;
        }
      }
      if (!XBIT(123, 0x2000000))
        pbchar('\n');
      r += strlen(r);
      r -= 3; /* last character - backup before " ENDTOKEN */
      r1++;   /* skip " */
      while (r >= r1) {
        pbchar(*r);
        if (*r == '"' && r >= r1 && *(r - 1) == '\\')
          /* replace the escape sequence \" with " */
          r--;
        else if (*r == '\\' && r >= r1 && *(r - 1) == '\\')
          /* replace the escape sequence \\ with \ */
          r--;
        r--;
      }
      pbstr("#pragma ");
      pbchar('\n'); /* pragma in col 1 */
      break;

    case VA_ARGS:
    case VA_REST:
      pperr(235, 3);
      break;

    case ARGST:
      --p;
      if (MASK(*p) == VA_ARGS) {
        if (nactuals <= nformals)
          continue;
        argp = targbuf;
        fstart = argp;
        pbchar(SENTINEL);
        for (i = nactuals - 1; i >= nformals; i--) {
          pbstr(actuals[i]);
          if (i != nformals) {
            pbchar(ENDTOKEN);
            pbchar(',');
          }
        }
        goto expand_actual;
      } else if (MASK(*p) == VA_REST) {
        /* NOTE: nformals includes the 'rest' argument */
        if (nactuals < nformals)
          continue;
        argp = targbuf;
        fstart = argp;
        pbchar(SENTINEL);
        for (i = nactuals - 1; i >= nformals - 1; i--) {
          pbstr(actuals[i]);
          if (i != nformals - 1) {
            pbchar(ENDTOKEN);
            pbchar(',');
          }
        }
        goto expand_actual;
      }
      if (*p > nactuals)
        continue;
      /* we have to expand this actual */
      argp = targbuf;
      fstart = argp;
      pbchar(SENTINEL);
      pbstr(actuals[*p - 1]);
    expand_actual:
      while ((toktyp = gtok(tokval, 1)) != T_SENTINEL) {
        /* save actual */
        if (toktyp == EOF) {
          pperror(209, &deftab[sp->name], 3);
          FREE(argbuf);
          FREE(targbuf);
          FREE(blank);
          return EOF;
        }
        if (argp - targbuf + (int)strlen(tokval) + 1 > targbufsize - 1) {
          newsize = argp - targbuf + (int)strlen(tokval) + 1 + TARGMAX;
          GROW_ARGBUF(targbuf, argp, fstart, targbufsize, newsize);
        }
        if (toktyp == T_NSIDENT)
          *argp++ = NOSUBST;
        strcpy(argp, tokval);
        argp += strlen(tokval);
        if (!(sp->flags & F_CHAR))
          *argp++ = ENDTOKEN;
      }
      if (argp != targbuf) {
        while (iswhite(*fstart))
          ++fstart;
        --argp;
        while (iswhite(*argp))
          --argp;
        *++argp = 0;
        pbstr(fstart);
      }
      break;
    default:
      interr("accpp: unexpected marker", startline, 3);
      break;
    }
  }

  mac_push(sp, savlptr);
#ifdef SUBST
  fprintf(stderr, "subst(%s) returns\n", deftab + sp->name);
#endif
  FREE(argbuf);
  FREE(targbuf);
  FREE(blank);
  return (0);
}

static void
ifpush(void)
{
  NEED(iftop + 2, _ifs, IFREC, ifsize, ifsize + IFMAX);

  ++iftop;
  ifstack(true_seen) = 0;
  ifstack(else_seen) = 0;
  if (iftop > 0)
    ifstack(truth) = _ifs[iftop - 1].truth;
}

static INT
strstore(const char *name)
{
  int i;
  int j;

  i = strlen(name) + 1;
  if (deftab == 0) {
    ndef = 256 < i ? i : 256;
    NEW(deftab, char, ndef);
    next_def = i + 1;
    j = 1;
  } else {
    j = next_def;
    next_def += i;
    if (next_def > ndef + 256)
      NEED(next_def, deftab, char, ndef, next_def);
    else
      NEED(next_def, deftab, char, ndef, ndef + 256);
  }
  strcpy(&deftab[j], name);
  return j;
}

static void
dumpval(char *p, FILE *ff)
{
  FILE *f;
  f = ff ? ff : stderr;
  for (; *p; ++p) {
    if ((*p & 0xFF) < 0x7F && (*p & 0xFF) > 31)
      fputc(*p, f);
    else {
      switch (MASK(*p)) {
      case ARGST:
        fprintf(f, "ARGST");
        break;
      case SENTINEL:
        fprintf(f, "SENTINEL");
        break;
      case NOSUBST:
        fprintf(f, "NOSUBST");
        break;
      case NOFUNC:
        fprintf(f, "NOFUNC");
        break;
      case STRINGIZE:
        fprintf(f, "STRINGIZE");
        break;
      case CONCAT:
        fprintf(f, "CONCAT");
        break;
      case ENDTOKEN:
        fprintf(f, "ENDTOKEN");
        break;
      case CONCAT1:
        fprintf(f, "CONCAT1");
        break;
      case WHITESPACE:
        fprintf(f, "WHITESPACE");
        break;
      case PRAGMAOP:
        fprintf(f, "PRAGMAOP");
        break;
      case VA_ARGS:
        fprintf(f, "VA_ARGS");
        break;
      case VA_REST:
        fprintf(f, "VA_REST");
        break;
      default:
        fprintf(f, "%d", *p & 0xFF);
      }
    }
  }
  fputc('\n', stderr);
}

static void
dumpmac(PPSYM *sp)
{
  fprintf(stderr, "%s (%d args, %x flags, %d next): def: ", &deftab[sp->name],
          sp->nformals, sp->flags, sp->next);
  dumpval(&deftab[sp->value], stderr);
}

#ifdef DUMPTAB
static void
dumptab(void)
{
  int i;
  for (i = 1; i < next_hash; ++i) {
    fprintf(stderr, "%d: ", i);
    dumpmac(hashrec + i);
  }
}
#endif

/*
 * output macro value in human readable form
 */
static void
putmac(PPSYM *sp)
{
  FILE *f;
  char *p;
  int i;
  f = stdout;
  if (startline >= 0) {
    if (gbl.curr_file != prevfile || startline != prevline + 1)
      fprintf(f, "\n// %s:%d\n", gbl.curr_file, startline);
    prevfile = gbl.curr_file;
    prevline = startline;
  }
  fprintf(f, "#define %s", deftab + sp->name);
  if (!XBIT(122, 0x80000)) {
    if (sp->nformals) {
      fprintf(f, "(");
      for (i = 0; i < sp->nformals; ++i) {
        if (i)
          fprintf(f, ",");
        fprintf(f, "%s", deftab + sp->formals[i]);
      }
      fprintf(f, ")");
    }
    fprintf(f, "  ");
    for (p = deftab + sp->value; *p; ++p) {
      char ch, nextch;
      ch = *p;
      nextch = *(p + 1);
      if (MASK(nextch) == ARGST) {
        fprintf(f, "%s", deftab + sp->formals[MASK(ch) - 1]);
        ++p;
      } else if (MASK(nextch) == STRINGIZE) {
        fprintf(f, "#%s", deftab + sp->formals[MASK(ch) - 1]);
        ++p;
      } else if (MASK(nextch) == CONCAT) {
        fprintf(f, "%s", deftab + sp->formals[MASK(ch) - 1]);
      } else if (MASK(nextch) == CONCAT1) {
        fprintf(f, "%s", deftab + sp->formals[MASK(ch) - 1]);
        ++p;
      } else if (MASK(ch) < 0x7F && MASK(ch) > 31) {
        fputc(ch, f);
      } else {
        switch (MASK(ch)) {
        case ARGST:
          break;
        case WHITESPACE:
          fprintf(f, " ");
          break;
        case STRINGIZE:
        case CONCAT1:
          break;
        case CONCAT:
          fprintf(f, "##");
          break;
        /* ignore these: */
        case SENTINEL:
        case NOSUBST:
        case NOFUNC:
        case ENDTOKEN:
        case PRAGMAOP:
        case VA_ARGS:
        case VA_REST:
        default:
          break;
        }
      }
    }
  }
  fputc('\n', f);
}

/** \brief
 * output undefine macro
 */
static void
putunmac(char *mac)
{
  FILE *f;
  f = stdout;
  if (startline >= 0) {
    if (gbl.curr_file != prevfile || startline != prevline + 1)
      fprintf(f, "\n// %s:%d\n", gbl.curr_file, startline);
    prevfile = gbl.curr_file;
    prevline = startline;
  }
  fprintf(f, "#undef %s\n", mac);
}

/* insflg -
 * 0: lookup only, do NOT add if not found
 *    Returns: NULL  - not previously def'd
 *             !NULL - previously def'd
 * 1: Add macro if not already defined
 *    Returns: NULL  - already def'd
 *             !NULL - added ok
 */
static PPSYM *
lookup(const char *name, int insflg)
{
  int i;
  char *cp;
  INT p, q;
  INT l, m;
  char buff[MAXIDLEN + 1];

  /* need to make sure not defining a predef macro */
  i = 0;
  l = m = (INT)0;

  strncpy(buff, name, MAXIDLEN);
  buff[MAXIDLEN] = 0;
  /*
   * break name into 3 byte chunks, sum them together, take remainder
   * modulo table size
   */
  cp = buff;
  while (*cp != '\0') {
    if (i == 3) {
      i = 0;
      m += l;
      l = (INT)0;
    }
    ++i;
    l <<= 8;
    l |= MASK(*cp);
    cp++;
  }
  m += l;
  i = m % HASHSIZ;
  for (q = 0, p = hashtab[i]; p != 0; q = p, p = hashrec[p].next) {
    if (strcmp(buff, &deftab[hashrec[p].name]) == 0) {
      if (insflg) {
        if (hashrec[p].flags & F_PREDEF) {
          /* can't def */
          pperror(247, name, 2);
          return NULL;
        }
      }
      if (hashrec[p].nformals != -1) /* is it 'defined'? */
        return &hashrec[p];
      return NULL;
    }
  }
  if (!insflg)
    return (NULL);
  /* allocate new record */
  if (hashrec == NULL) {
    nhash = 100;
    NEW(hashrec, PPSYM, nhash);
    next_hash = 2;
    p = 1;
  } else {
    p = next_hash;
    ++next_hash;
    NEED(next_hash, hashrec, PPSYM, nhash, nhash + 100);
  }
  hashrec[p].value = 0;
  hashrec[p].name = strstore(buff);
  hashrec[p].flags = 0;
  hashrec[p].nformals = 0;
  if (q == 0) {
    hashrec[p].next = hashtab[i];
    hashtab[i] = p;
  } else {
    hashrec[q].next = p;
    hashrec[p].next = 0;
  }
  return &hashrec[p];
}

static void
delete(const char *name)
{
  int i, j;
  INT l, m;
  INT p, q;
  const char *cp;

  i = j = 0;
  l = m = (INT)0;

  /*
   * break name into 3 byte chunks, sum them together, take remainder
   * modulo table size
   */
  cp = name;
  while (j < MAXIDLEN && *cp != '\0') {
    if (i == 3) {
      i = 0;
      m += l;
      l = (INT)0;
    }
    ++i;
    ++j;
    l <<= 8;
    l |= *cp++;
  }
  m += l;
  i = m % HASHSIZ;
  for (q = 0, p = hashtab[i]; p != 0; q = p, p = hashrec[p].next)
    if (strncmp(name, &deftab[hashrec[p].name], MAXIDLEN) == 0)
      goto found;
  return;
found:
  if (hashrec[p].flags & F_PREDEF) {
    /* can't undef */
    pperror(248, name, 2);
    return;
  }
  /* delete record */
  if (q == 0) /* first in list */
    hashtab[i] = hashrec[p].next;
  else
    hashrec[q].next = hashrec[p].next;
}

static void
ptok(const char *tok)
{
  FILE *fp;
  static int state = 1;
  static int nchars;
  static int needspace = 0;
  static int leading = 1;
  char *tokchr = strdup(tok);
  char *tokptr = tokchr;

  /* -M option:  Print list of include files to stdout */
  if (XBIT(123, 2) || XBIT(123, 0x20000000) || XBIT(123, 0x40000000))
    return;

  /* keep track of where compiler thinks we are */
  fp = gbl.cppfil;
  if (*tokchr == '\n') {
    if (state == 1 && !XBIT(123, 0x80000))
      return;
    state = 1; /* next token starts a new line */
    nchars = 0;
    ++cur_line;
    needspace = 0;
  }
  /* if starting a new line, make sure line # and file are correct */
  else if (state) {
    state = 0;
    leading = 1;
    /* make sure next line and file are correct */
    if (strcmp(cur_fname, inclstack[inclev].fname) || cur_line != startline) {
      strcpy(cur_fname, inclstack[inclev].fname);
      cur_line = startline;
      cur_from_stdinc = inclstack[inclev].from_stdinc;
      if (startline != 1)
        pr_line(cur_fname, startline, cur_from_stdinc);
    }
  }
  if (!XBIT(123, 0x20) && !XBIT(123, 0x200)) {
    if (needspace && !iswhite(*tokchr) && MASK(*tokchr) != WHITESPACE &&
        *tokchr != '\n') {
      (void)putc(' ', fp);
      ++nchars;
      needspace = 0;
    }
  }

  if (MASK(*tokchr) == WHITESPACE) {
    *tokchr = ' ';
  }

  /* Suppress duplicate whitespaces added by macro expansion */
  if (!XBIT(123, 0x800) && !leading && !needspace && iswhite(*tokchr)) {
    ++nchars;
    if (!(*(++tokchr)))
      return;
  }

  if (leading && !iswhite(*tokchr))
    leading = 0;

  needspace = !(iswhite(*tokchr) || *tokchr == '\n');
  while (*tokchr) {
    (void)putc(*tokchr++, fp);
    ++nchars;
  }
  FREE(tokptr);
}

#define Number 1
#define Lparen 2
#define Rparen 3
#define Eoln 4
#define Defined 5
#define Mult 6
#define Divide 7
#define Mod 8
#define Plus 9
#define Minus 10
#define Lshift 11
#define Rshift 12
#define Lt 13
#define Gt 14
#define Le 15
#define Ge 16
#define Eq 17
#define Ne 18
#define And 19
#define Xor 20
#define Or 21
#define Andand 22
#define Oror 23
#define Question 24
#define Colon 25
#define Comma 26
#define Minusx 27 /* unary minus operation */
#define Not 28
#define Compl 29 /* ~ */
#define Predicate 30
#define Plusx 31 /* unary plus */

/** \brief
 *  define parse table, PT, which has an entry for each of the
 *  tokens just defined:  */
static struct {
  INT8 led;  /* TRUE if this token is a legal infix
              * operator  */
  INT8 nud;  /* 0 if this token isn't a legal prefix
              * operator, else token to use in switch
              * stmt.  */
  short lbp; /* left binding power  */
  INT8 rbp;  /* right binding power  */
} PT[] = {
    {0, 0, 0, 0},         {0, Number, 0, 0}, /* Number */
    {0, Lparen, 0, 0},                       /* Lparen */
    {0, 0, 0, 0},                            /* Rparen */
    {0, 0, -1, 0},                           /* Eoln   */
    {0, Defined, 0, 0},                      /* Defined */
    {1, 0, 60, 60},                          /* Mult   */
    {1, 0, 60, 60},                          /* Divide */
    {1, 0, 60, 60},                          /* Mod    */
    {1, Plusx, 50, 50},                      /* Plus   */
    {1, Minusx, 50, 50},                     /* Minus  */
    {1, 0, 40, 40},                          /* Lshift */
    {1, 0, 40, 40},                          /* Rshift */
    {1, 0, 35, 35},                          /* Lt     */
    {1, 0, 35, 35},                          /* Gt     */
    {1, 0, 35, 35},                          /* Le     */
    {1, 0, 35, 35},                          /* Ge     */
    {1, 0, 30, 30},                          /* Eq     */
    {1, 0, 30, 30},                          /* Ne     */
    {1, 0, 27, 27},                          /* And    */
    {1, 0, 24, 24},                          /* Xor    */
    {1, 0, 21, 21},                          /* Or     */
    {1, 0, 18, 18},                          /* Andand */
    {1, 0, 15, 15},                          /* Oror   */
    {1, 0, 10, 9},                           /* Question */
    {0, 0, 0, 0},                            /* Colon  */
    {1, 0, 5, 5},                            /* Comma  */
    {0, 0, 0, 100},                          /* Minusx */
    {0, Not, 0, 100},                        /* Not    */
    {0, Compl, 0, 100},                      /* Compl  */
    {0, Predicate, 0, 0},                    /* Defined */
    {0, 0, 0, 100},                          /* Plusx */
};

#define CHECK(t)         \
  {                      \
    if (token != t)      \
      goto syntax_error; \
    token = gettoken();  \
  }

static INT
doparse(void)
{
  INT i;
  PTOK tmp;

  token = gettoken();
  if (token == Eoln) {
    pperr(226, 3);
    return (1);
  }
  syntaxerr = 0;
  parse(-1, &tmp);
  i = PTOK_ISNZ(tmp);
  if (syntaxerr) {
    if (token != Eoln)
      clreol(0);
  } else
    ptok("\n");
  return (i);
}

#define UCONV                                \
  if (PTOK_ISUNS(left) || PTOK_ISUNS(right)) \
    PTOK_ISUNS(left) = 1;                    \
  else                                       \
  PTOK_ISUNS(left) = 0

/** \brief
 *  parse "current" expression and return value computed
 *  by constant folding it.
 */
static void
parse(int rbp, PTOK *tk)
{
  int op;
  LOGICAL valid_left = FALSE;
  int t;
  PTOK right, tmp, left;
  do {
    op = token;
    tmp = tokenval;
    token = gettoken();

    if (!valid_left) {
      valid_left = TRUE;
      op = PT[op].nud;
      if (op == 0)
        goto syntax_error;
    } else {
      if (PT[op].led == 0)
        goto syntax_error;
    }

    if (op != Number && op != Defined && op != Predicate)
      parse(PT[op].rbp, &right);

    switch (op) {
    case Number:
      left = tmp;
      break;
    case Lparen:
      left = right;
      CHECK(Rparen);
      break;
    case Defined:
    case Predicate:
      t = token;
      if (token == Lparen)
        token = gettoken();
      left = tokenval;
      /* smt: according to ansi, "the argument to defined() shall be an
       * identifier" */
      if (toktyp != T_IDENT)
        goto syntax_error;
      token = gettoken();

      if (t == Lparen)
        CHECK(Rparen);
      break;
    case Mult:
      UCONV;
      PTOK_MUL(left, right); /* left *= right */
      break;
    case Divide:
      UCONV;
      if (!PTOK_ISUNS(left))
        PTOK_DIV(left, right); /* left /= right */
      else
        PTOK_UDIV(left, right);
      break;
    case Mod:
      UCONV;
      if (!PTOK_ISUNS(left))
        PTOK_MOD(left, right); /* left %= right */
      else
        PTOK_UMOD(left, right);
      break;
    case Plus:
      UCONV;
      PTOK_ADD(left, right); /* left += right */
      break;
    case Minus:
      UCONV;
      PTOK_SUB(left, right); /* left -= right */
      break;
    case Lshift:
    case Rshift:
      UCONV;
#ifdef TM_REVERSE_SHIFT
      if (PTOK_ISNEG(right)) {
        PTOK_NEG(right, right);
        op = (Lshift + Rshift) - op;
      }
#endif
      if (op == Lshift)
        PTOK_LSHIFT(left, right);
      else if (PTOK_ISUNS(left))
        PTOK_URSHIFT(left, right);
      else
        PTOK_CRSHIFT(left, right);
      break;
    case Lt:
      UCONV;
      if (!PTOK_ISUNS(left))
        PTOK_LOGLT(left, right);
      else
        PTOK_LOGULT(left, right);
      break;
    case Gt:
      UCONV;
      if (!PTOK_ISUNS(left))
        PTOK_LOGGT(left, right);
      else
        PTOK_LOGUGT(left, right);
      break;
    case Le:
      UCONV;
      if (!PTOK_ISUNS(left))
        PTOK_LOGLE(left, right);
      else
        PTOK_LOGULE(left, right);
      break;
    case Ge:
      UCONV;
      if (!PTOK_ISUNS(left))
        PTOK_LOGGE(left, right);
      else
        PTOK_LOGUGE(left, right);
      break;
    case Eq:
      UCONV;
      PTOK_LOGEQ(left, right);
      break;
    case Ne:
      UCONV;
      PTOK_LOGNE(left, right);
      break;
    case And:
      UCONV;
      PTOK_AND(left, right);
      break;
    case Xor:
      UCONV;
      PTOK_XOR(left, right);
      break;
    case Or:
      UCONV;
      PTOK_OR(left, right);
      break;
    case Comma:
      UCONV;
/* Ansi restriction, comma not allowed in directive,
   Section 3.4 */
      PTOK_ASSN(left, right);
      break;
    case Minusx:
      PTOK_ISUNS(left) = PTOK_ISUNS(right);
      PTOK_NEG(left, right);
      break;
    case Not:
      PTOK_LNOT(left, right);
      PTOK_ISUNS(left) = 0;
      break;
    case Compl:
      PTOK_ISUNS(left) = PTOK_ISUNS(right);
      PTOK_BNOT(left, right);
      break;
    case Plusx:
      PTOK_ISUNS(left) = PTOK_ISUNS(right);
      PTOK_ASSN(left, right);
      break;
    case Andand:
      PTOK_ANDAND(left, right);
      PTOK_ISUNS(left) = 0;
      break;
    case Oror:
      PTOK_OROR(left, right);
      PTOK_ISUNS(left) = 0;
      break;
    case Question:
      CHECK(Colon);
      parse(PT[Question].rbp, &tmp);
      if (PTOK_ISUNS(left) || PTOK_ISUNS(tmp))
        PTOK_ISUNS(left) = 1;
      if (PTOK_ISNZ(left))
        PTOK_ASSN(left, right);
      else
        PTOK_ASSN(left, tmp);
      break;
    default:
      goto syntax_error;
    }
  } while (PT[token].lbp > rbp);

  *tk = left;
  return;

syntax_error:
  pperr(226, 3);
  syntaxerr = 1;
  token = Eoln;
  PTOK_ASSN32(tmp, 1);
  PTOK_ISUNS(tmp) = 0;
  *tk = tmp;
}

static int
gettoken(void)
{
  static int ifdef = 0;
  char *s;
  PPSYM *sp;
  char tokval[TOKMAX];
  int i, c;
  INT t;

  for (;;) {
    toktyp = skipbl(tokval, !ifdef);
    if (toktyp == '\n')
      return Eoln; /* end of #if */
  again:
    switch (toktyp) {
    case T_OP:
      switch (*tokval) {
      case '|':
        if (tokval[1] == '|')
          return Oror;
        goto illch;
      case '&':
        if (tokval[1] == '&')
          return Andand;
        goto illch;
      case '>':
        if (tokval[1] == '>' && tokval[2] != '=')
          return Rshift;
        if (tokval[1] == '=')
          return Ge;
        goto illch;
      case '<':
        if (tokval[1] == '<' && tokval[2] != '=')
          return Lshift;
        if (tokval[1] == '=')
          return Le;
        goto illch;
      case '=':
        if (tokval[1] == '=')
          return Eq;
        goto illch;
      case '!':
        if (tokval[1] == '=')
          return Ne;
        goto illch;
      default:
        goto illch;
      }
    case '#':
      /* could be an AT&T predicate */
      toktyp = skipbl(tokval, 0);
      if (toktyp == '\n') {
        pperr(253, 2);
        return Eoln;
      }
      if (toktyp != T_IDENT) {
        pperr(253, 2);
        goto again;
      }
      predicate(tokval);
      ifdef = 2;
      return Predicate;
    case '|':
      return (Or);
    case '&':
      return (And);
    case '>':
      return (Gt);
    case '<':
      return (Lt);
    case '!':
      return (Not);
    case '+':
      return (Plus);
    case '-':
      return (Minus);
    case '*':
      return (Mult);
    case '/':
      return (Divide);
    case '%':
      return (Mod);
    case '^':
      return (Xor);
    case '?':
      return (Question);
    case ':':
      return (Colon);
    case '~':
      return (Compl);
    case '(':
      return (Lparen);
    case ')':
      return (Rparen);
    case ',':
      return (Comma);
    case T_PPNUM:
/* parse number */
#if defined(PPNUM_IS_32)
      if (tobinary(tokval, &PTOK_ISUNS(tokenval), &PTOK_V(tokenval)) != 0)
#else
      if (tobinary64(tokval, &PTOK_ISUNS(tokenval), PTOK_V(tokenval)))
#endif
      {
        pperr(254, 3);
        PTOK_ASSN32(tokenval, 0);
      }
      return Number;
    case T_IDENT:
      if (0 == strcmp(tokval, "defined")) {
        ifdef = 1;
        return (Defined);
      } else if (ifdef == 1) {
        sp = lookup(tokval, 0);
        ifdef = 0;
        PTOK_ASSN32(tokenval, sp && sp->value != 0);
        PTOK_ISUNS(tokenval) = 0;
        return (Number);
      } else {
        ifdef = 0;
        PTOK_ASSN32(tokenval, predarg(tokval));
        PTOK_ISUNS(tokenval) = 0;
        return Number;
      }
      break;
    case T_WCHAR:
      s = tokval + 2;
      goto commchar;
    case T_CHAR:
      s = tokval + 1;
    commchar:
      /* multi-char const? */
      i = 0;
      PTOK_ASSN32(tokenval, 0);
      PTOK_ISUNS(tokenval) = 0;
      while ((c = *s) != '\'' && i < 4) {
        ++s;
        ++i;
        PTOK_LSHIFT32(tokenval, 8);
        if (c == '\\') {
          switch (c = *s++) {
          case 'a':
            t = '\007';
            break;
          case 'b':
            t = '\b';
            break;
          case 'f':
            t = '\f';
            break;
          case 'n':
            t = '\n';
            break;
          case 'r':
            t = '\r';
            break;
          case 't':
            t = '\t';
            break;
          case 'v':
            t = '\v';
            break;
          case 'x':
            t = 0;
            while (ishex(*s)) {
              t *= 16;
              c = *s++;
              if (c >= 'a' && c <= 'f')
                t += c - 'a' + 10;
              else if (c >= 'A' && c <= 'F')
                t += c - 'A' + 10;
              else
                t += c - '0';
            }
            break;
          default:
            if (c >= '0' && c <= '7') {
              t = c - '0';
              if (*s >= '0' && *s <= '7') {
                t = t * 8 + (*s++ - '0');
                if (*s >= '0' && *s <= '7') {
                  t = t * 8 + (*s++ - '0');
                }
              }
            } else {
              t = c;
            }
            break;
          }
          PTOK_OR32(tokenval, t);
        } else
          PTOK_OR32(tokenval, c);
      }
      if (i == 0 || i > 4) {
        pperr(24, 3);
      }
#ifndef CHAR_IS_UNSIGNED
      if (i == 1) {
        if (PTOK_BTEST32(tokenval, 0x80)) {
          PTOK_OR32(tokenval, 0xffffff00);
          PTOK_SEXTEND32(tokenval);
        }
      } else if (i == 2) {
        if (PTOK_BTEST32(tokenval, 0x8000)) {
          PTOK_OR32(tokenval, 0xffff0000);
          PTOK_SEXTEND32(tokenval);
        }
      }
#endif
      PTOK_ISUNS(tokenval) = 0;
      return (Number);
    case T_STRING:
    case T_WSTRING:
      pperr(255, 3);
      PTOK_ASSN32(tokenval, 0);
      PTOK_ISUNS(tokenval) = 0;
      return Number;

    default:
    illch:
      pperr(255, 3);
      continue;
    }
  }
}

static INT
tobinary(char *st, int *isuns, INT *value)
{
  int radix;
  int r;
  char c;
  char *cp;

  c = *(cp = st);
  radix = 10;
  if (c == '0') {
    if (cp[1] == 'x' || cp[1] == 'X') {
      radix = 16;
      cp += 2;
      while (ishex(*cp))
        ++cp;
    } else {
      radix = 8;
      while (isdigit(*cp))
        ++cp;
    }
  } else
    while (isdigit(*cp))
      ++cp;
  *isuns = 0;
  if (radix == 16)
    st += 2;
  r = atoxi(st, value, (int)(cp - st), radix);
  if (r == -1) {
    if (radix == 16)
      st -= 2;
    c = *cp;
    *cp = 0;
    pperror(27, st, 3);
    *cp = c;
    st = cp;
    return -1;
  } else if (r != 0) { /* overflow */
    if (radix == 16)
      st -= 2;
    c = *cp;
    *cp = 0;
    pperror(23, st, 2);
    *cp = c;
    if (radix == 16)
      st += 2;
  }
  if (*cp == 'l' || *cp == 'L' || *cp == 'u' || *cp == 'U') {
    if (*cp == 'u' || *cp == 'U')
      *isuns = 1;
    ++cp;
    if (*cp == 'l' || *cp == 'L' || *cp == 'u' || *cp == 'U') {
      if (*cp == 'u' || *cp == 'U')
        *isuns = 1;
      ++cp;
    }
  }

  /* smt */
  /* Check illegal integer */
  if (cp != st + strlen(st)) {
    pperror(27, st, 3);
    return -1;
  }
  if (r == -1) {
    if (radix == 16)
      st -= 2;
    c = *cp;
    *cp = 0;
    pperror(27, st, 3);
    *cp = c;
    st = cp;
    return -1;
  }
  if ((*value & 0x80000000) || r == -2)
    *isuns = 1;
  return 0;
}

static INT
tobinary64(char *st, int *isuns, DBLINT64 value)
{
  int radix;
  int r;
  char c;
  char *cp;

  c = *(cp = st);
  radix = 10;
  if (c == '0') {
    if (cp[1] == 'x' || cp[1] == 'X') {
      radix = 16;
      cp += 2;
      while (ishex(*cp))
        ++cp;
    } else {
      radix = 8;
      while (isdigit(*cp))
        ++cp;
    }
  } else
    while (isdigit(*cp))
      ++cp;
  *isuns = 0;
  if (radix == 16)
    st += 2;
  r = atoxi64(st, value, (int)(cp - st), radix);
  if (r == -1) {
    if (radix == 16)
      st -= 2;
    c = *cp;
    *cp = 0;
    pperror(27, st, 3);
    *cp = c;
    st = cp;
    return -1;
  } else if (r != 0) { /* overflow */
    if (radix == 16)
      st -= 2;
    c = *cp;
    *cp = 0;
    pperror(23, st, 2);
    *cp = c;
    if (radix == 16)
      st += 2;
  }
  /* Process any suffix which may follow the constant -- can be
   * followed by L and/or U in any order or case
   */
  switch (*cp) {
  case 'l':
  case 'L':
    /*   suffix begins with L  */
    ++cp;
    switch (*cp) {
    case 'u':
    case 'U':
      /*****  LU  *****/
      *isuns = 1;
      ++cp;
      if (*cp == 'l' || *cp == 'L') {
        /*****  LUL *****/
        ++cp;
      }
      break;
    case 'l':
    case 'L':
      /*****  LL  *****/
      ++cp;
      if (*cp == 'u' || *cp == 'U') {
        /*****  LLU *****/
        *isuns = 1;
        ++cp;
      }
      break;
    }
    break;

  case 'u':
  case 'U':
    /*   suffix begins with U  */
    *isuns = 1;
    ++cp;
    if (*cp == 'l' || *cp == 'L') {
      /*****  UL  *****/
      ++cp;
      if (*cp == 'l' || *cp == 'L') {
        /*****  ULL *****/
        ++cp;
      }
    }
    break;
  }

  /* smt */
  /* Check illegal integer */
  if (cp != st + strlen(st)) {
    pperror(27, st, 3);
    return -1;
  }
  if (r == -1) {
    if (radix == 16)
      st -= 2;
    c = *cp;
    *cp = 0;
    pperror(27, st, 3);
    *cp = c;
    st = cp;
    return -1;
  }
  if ((value[0] & 0x80000000) || r == -2)
    *isuns = 1;
  return 0;
}

/* expflag == 1: expand token from macro */
static int
gtok(char *tokval, int expflag)
{
  PPSYM *sp;
  int toktyp;
  int i;
  int comment = 0;

again:
  /* -C option:  Pass comments through the preprocessor */
  comment = 0;
  if (XBIT(123, 1) && flg.es == TRUE) {
    flg.es = FALSE; /* Remove comments from directives */
    comment = 1;
  }
  toktyp = nextok(tokval, 0);

  if (comment)
    flg.es = TRUE;

  if (toktyp == T_NOFIDENT)
    toktyp = T_IDENT;
  else if (toktyp == T_IDENT && expflag && (sp = lookup(tokval, 0)) != 0) {
    if (!macro_recur_check(sp)) {
      /* scan macro and substitute */
      if ((i = subst(sp)) == EOF || i == T_SENTINEL)
        return i;
      goto again;
    }
    toktyp = T_NSIDENT;
  }
  return (toktyp);
}

static int
findtok(char *tokval, int truth)
{
  int i;
  int state;
  PPSYM *sp;
  int toktyp;

  state = 1;

  if (truth) {
    for (;;) {
      toktyp = nextok(tokval, 1);
      if (toktyp == EOF)
        return (EOF);
      if (toktyp == '#' && state) {
        return (T_POUND);
      }
      if (!isblank(toktyp))
        state = 0; /* allow whitespace before '#' directives */
      if (toktyp == T_NOFIDENT)
        toktyp = T_IDENT;
      else if (toktyp == T_IDENT && (sp = lookup(tokval, 0)) != 0) {
        if (!macro_recur_check(sp)) {
          /* scan macro and substitute */
          if ((i = subst(sp)) == EOF || i == T_SENTINEL)
            return i;
          continue;
        }
        toktyp = T_NSIDENT;
      }
      if (toktyp == '\n') {
        state = 1;
        in_ftn_comment = FALSE;
      }
      ptok(tokval);
    }
  } else {
    for (;;) {
      toktyp = nextok(tokval, 0);
      if (toktyp == EOF)
        return (EOF);
      if (toktyp == '#' && state) {
        return (T_POUND);
      }
      if (!isblank(toktyp))
        state = 0; /* allow whitespace before '#' directives */
      if (toktyp == '\n') {
        state = 1;
        in_ftn_comment = FALSE;
      }
    }
  }
}

#define inchar() (lineptr < lineend ? *lineptr++ : _nextline())

static int
nextok(char *tokval, int truth)
{
  int i;
  char *p, tmp;
  int delim;
  int c;
  int toktyp;
  int retval;
  char *savtokval = tokval;
  char *comment_ptr;
  int dot_seen;

again:
  if ((c = inchar()) == EOF) {
    retval = EOF;
    goto nextok_ret;
  }
  *tokval++ = c;
  switch (c) {
  case '$':
    if (XBIT(123, 0x100000))
      goto defret;
    FLANG_FALLTHROUGH;

  /* Identifier or Fortran 'C' comment.
   * If this is Fortran and a 'C', it might be an identifier or comment.
   * The 'c' or 'C' must be in the first column (fixed-form)
   * to be recognized here as Fortran comment.
   */
  case 'C':
  case 'c':
    in_ftn_comment = TRUE;
    if (!flg.freeform && lineptr == linebuf + LINELEN + 1)
    {
      /* c in column 1, start of old-style comment */
      /* everything to end of line is a comment */
      if (XBIT(124, 0x100000)) { /* If skip comments... */
        *tokval = 0;
        p = lineptr;
        while (*p != '\n')
          ++p;
        if ((i = p - lineptr) >= TOKMAX - 1)
          i = TOKMAX - 1;
        strncpy(tokval, lineptr, i);
        tokval[i] = 0;
        lineptr = p;
        retval = c;
        goto nextok_ret;
      }
      /* Fall thru if we do not have a C or c in the first column */
    }
    FLANG_FALLTHROUGH;

  case 'A':
  case 'B': /* case 'C': */
  case 'D':
  case 'E':
  case 'F':
  case 'G':
  case 'H':
  case 'I':
  case 'J':
  case 'K': /* case 'L': */
  case 'M':
  case 'N':
  case 'O':
  case 'P':
  case 'Q':
  case 'R':
  case 'S':
  case 'T':
  case 'U':
  case 'V':
  case 'W':
  case 'X':
  case 'Y':
  case 'Z':
  case '_':
  case 'a':
  case 'b':
  /* case 'c': */ case 'd':
  case 'e':
  case 'f':
  case 'g':
  case 'h':
  case 'i':
  case 'j':
  case 'k':
  case 'l':
  case 'm':
  case 'n':
  case 'o':
  case 'p':
  case 'q':
  case 'r':
  case 's':
  case 't':
  case 'u':
  case 'v':
  case 'w':
  case 'x':
  case 'y':
  case 'z':
    /* we know the whole identifier must be in the buffer */
    p = lineptr;
    retval = T_IDENT;
  ident:
    while (isident(*p++))
      /* NULL */;
    --p;
    if ((i = p - lineptr) >= TOKMAX - 1) {
      pperr(212, 3);
      i = TOKMAX - 1;
    }
    strncpy(tokval, lineptr, i);
    tokval[i] = 0;
    lineptr = p;
    goto nextok_ret;

  case 'L':
    p = lineptr;
    if (*p != '"' && *p != '\'') {
      retval = T_IDENT;
      goto ident;
    }
    delim = *p;
    i = 2;
    ++p;
    *tokval++ = delim;
    ++lineptr;
    toktyp = (delim == '"') ? T_WSTRING : T_WCHAR;
    goto wchar;

  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
    /* a number */
    /* must all be in buffer */
    p = lineptr;
  dotnum:
    dot_seen=0;
    for (;;) {
      if (*p == 'E' || *p == 'e') {
        ++p;
        if (*p == '+' || *p == '-')
          ++p;
      } else if (*p == 'P' || *p == 'p') {
        ++p;
        if (*p == '+' || *p == '-')
          ++p;
      } else if (*p == '.' && !dot_seen) {
        ++p;
        dot_seen=1;
      } else if (isident(*p))
        ++p;
      else
        break;
    }
    if ((i = p - lineptr) >= TOKMAX - 1) {
      pperr(220, 3);
      i = TOKMAX - 1;
    }

    strncpy(tokval, lineptr, i);
    tokval[i] = 0;
    toktyp = T_PPNUM;
    lineptr = p;
    retval = toktyp;
    goto nextok_ret;

  case '.':
    p = lineptr;
    if (asm_mode) {
      retval = T_IDENT;
      goto ident;
    }
    if (p >= lineend) {
      goto dotdone;
    }
    if (isdig(*p)) {
      goto dotnum;
    }
    if (*p == '.') {
      ++p;
      if (p >= lineend) {
        goto dotdone;
      }
      if (*p != '.')
        goto dotdone;
      lineptr = ++p;
      *tokval++ = '.';
      *tokval++ = '.';
      *tokval = 0;
      retval = T_OP;
      goto nextok_ret;
    }
  dotdone:
    *tokval = 0;
    retval = '.';
    goto nextok_ret;

  case '>': /* >=, >>, >>= */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;
    if (*lineptr == '>') {
      if (lineptr + 1 >= lineend)
        goto twocop;
      if (*++lineptr == '=') {
        *tokval++ = '>';
        *tokval++ = '=';
        ++lineptr;
        *tokval = 0;
        retval = T_OP;
        goto nextok_ret;
      } else {
        --lineptr;
        goto twocop;
      }
    }
    goto defret;
  case '<': /* <=, <<, <<= */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;
    if (*lineptr == '<') {
      if (lineptr + 1 >= lineend)
        goto twocop;
      if (*++lineptr == '=') {
        *tokval++ = '<';
        *tokval++ = '=';
        ++lineptr;
        *tokval = 0;
        retval = T_OP;
        goto nextok_ret;
      } else {
        --lineptr;
        goto twocop;
      }
    }
    goto defret;
  case '!': /* != */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;
    if (asm_mode) {
      if (*lineptr == '+' || *lineptr == '-')
        goto twocop;
    }
    in_ftn_comment = TRUE;
    if (XBIT(124, 0x100000)) { /* If skip comments is set */
      p = lineptr;
      while (*p != '\n')
        ++p;
      if ((i = p - lineptr) >= TOKMAX - 1) {
        i = TOKMAX - 1;
      }
      strncpy(tokval, lineptr, i);
      tokval[i] = 0;
      lineptr = p;
      retval = c;
      goto nextok_ret;
    }
    goto defret;
  case '+': /* ++, += */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=' || *lineptr == '+')
      goto twocop;
    if (asm_mode) {
      if (*lineptr == '!')
        goto twocop; /* +! */
    }
    goto defret;
  case '-': /* --, -=, -> */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=' || *lineptr == '-' || *lineptr == '>')
      goto twocop;
    if (asm_mode) {
      if (*lineptr == '!')
        goto twocop; /* -! */
    }
    goto defret;
  case '@':
    if (asm_mode) {
      if (lineptr < lineend && *lineptr == '(')
        goto twocop; /* @( */
      p = lineptr;
      retval = T_IDENT;
      goto ident;
    }
    goto defret;
  case '%': /* %= */
    if (asm_mode) {
      p = lineptr;
      retval = T_IDENT;
      goto ident;
    }
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;
    goto defret;
  case '&': /* &&, &= */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=' || *lineptr == '&')
      goto twocop;
    goto defret;
  case '*': /* *= */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;
    if (!flg.freeform && lineptr == linebuf + LINELEN + 1)
    {
      /* * in column 1, start of old-style comment */
      /* everything to end of line is a comment */
      if (XBIT(124, 0x100000)) { /* If skip comments... */
        *tokval = 0;
        p = lineptr;
        while (*p != '\n')
          ++p;
        if ((i = p - lineptr) >= TOKMAX - 1)
          i = TOKMAX - 1;
        strncpy(tokval, lineptr, i);
        tokval[i] = 0;
        lineptr = p;
        retval = c;
        goto nextok_ret;
      }
    }
    goto defret;
  case '=': /* == */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;
    goto defret;
  case '^': /* ^= */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;
    goto defret;
  case '|': /* ||, |= */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=' || *lineptr == '|')
      goto twocop;
    goto defret;
  case '#':
    if (!in_dodef && !flg.freeform && (lineptr != linebuf + LINELEN + 1))
    {
      /* Fixed form and not initial/first column, then skip.
       * This means that all preprocessor macro '#' tokens
       * in fixed-form must begin in column 1.
       * The line is passed onto the backend.
       */
      *tokval = 0;
      p = lineptr;
      while (*p != '\n')
        ++p; /* Advance to end of line */

      /* Skip the line if in a false branch of a macro conditional */
      if (ifstack(truth) == FALSE) {
        lineptr = p;
        retval = '\n';
        goto nextok_ret;
      }

      if ((i = p - lineptr) >= TOKMAX - 1)
        i = TOKMAX - 1;
      strncpy(tokval, lineptr, i);
      tokval[i] = 0;

      /* If not in a fixed-form comment, then pass line to rest of
       * compiler.  Else we are in a fixed-form comment and might be
       * passing the data following # twice.
       */
      tmp = *(linebuf + LINELEN); /* First character in column 1  */
      if (!(tmp == '*' || tmp == 'c' || tmp == 'C' || tmp == '!'))
        ptok(&tokval[-1]);
      lineptr = p;
      retval = c;
      goto nextok_ret;
    }
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '#') {
      ++lineptr;
      *tokval++ = '#';
      *tokval++ = 0;
      retval = T_CONCAT;
      goto nextok_ret;
    }
    goto defret;
  twocop:
    *tokval++ = *lineptr++;
    *tokval = 0;
    retval = T_OP;
    goto nextok_ret;

  case '"':
  case '\'':
    delim = c;
    i = 1;
    p = lineptr;
    toktyp = T_STRING;

  wchar:
    while (p < lineend && *p != '\n') {
      if (*p == delim)
        break;
      if (*p == (char)ENDTOKEN) {
        --p;
        break;
      }
      if (*p == '\\') {
        ++p;
        if (p >= lineend || *p == '\n')
          break;
      }
      ++p;
    }

    /* If the line is unterminated (no closing quote)
     * Note: Fortran, we do not error here as this might be a Fortran
     * comment we are scanning.  Instead, the Fortran lexer will error if
     * necessary.
     */
    if (p >= lineend || *p == '\n') {
      if (XBIT(122, 2) && flg.es) /* -Xs and -E/-P */
        ;
    } else
      ++p;
    /* copy to tokval */
    if ((i += p - lineptr) >= TOKMAX) {
      pperr(223, 4);
      *tokval = 0;
    } else {
      strncpy(tokval, lineptr, p - lineptr);
      tokval[p - lineptr] = 0;
      lineptr = p;
    }
    retval = toktyp;
    goto nextok_ret;
    /* NOTREACHED */;

  case '/': /* /*, //, /= */
    if (lineptr >= lineend)
      goto defret;
    if (*lineptr == '=')
      goto twocop;

    /* -B or -c9x options:  Ignore C++ style comments "//" to end-of-line */
    p = lineptr;
    if (cpp_comments && *lineptr == '/') {
      while (lineptr < lineend && *lineptr != '\n')
        ++lineptr;

      /* -C option:  Pass comments through the preprocessor */
      if (XBIT(123, 1) && flg.es == TRUE && ifstack(truth)) {
        i = lineptr - p;
        if (i > TOKMAX) /* Truncate comment if too long */
          i = TOKMAX;
        strncpy(tokval, p, i);
        tokval[i] = 0;
        ptok(&tokval[-1]);
      }

      /* comment is a space */
      --tokval;
      *tokval++ = ' ';
      *tokval = 0;
      retval = ' ';
      if (XBIT(122, 2)) {
        --tokval;
        goto again;
      }
      goto nextok_ret;
    }

    if (*lineptr != '*')
      goto defret;
    --tokval;

    /* Save start of comment (in case of -C option) */
    comment_ptr = lineptr - 1;

    lineptr++; /* Bypass the '*' */
    for (;;) {
      p = lineptr;
      /* scan to end of line or '*' */
      while (p < lineend && *p != '*')
        ++p;

      if (p >= lineend) {
        /* -C option:  Pass comments through the preprocessor */
        if (XBIT(123, 1) && flg.es == TRUE && ifstack(truth)) {
          i = p - comment_ptr;
          if (i > TOKMAX) /* Truncate comment if too long */
            i = TOKMAX;
          strncpy(tokval, comment_ptr, i);
          tokval[i - 1] = 0;
          tokval = savtokval;
          ptok(tokval);
        }
        if (p == lineend && in_ftn_comment) {
          goto defret;
        }
        ptok("\n");
        popstack();
        if (_nextline() == EOF) {
          pperr(208, 3);
          *tokval = 0;
          retval = EOF;
          goto nextok_ret;
        }
        --lineptr;
        comment_ptr = lineptr;
        continue;
      }
      if (p > lineptr && p[-1] == '/') /* possible nested comment */
        pperr(244, 1);
      if (*++p == '/') {
        /* -C option:  Pass comments through the preprocessor */
        if (XBIT(123, 1) && flg.es == TRUE && ifstack(truth)) {
          i = p - comment_ptr + 1;
          if (i > TOKMAX) /* Truncate comment if too long */
            i = TOKMAX;
          strncpy(tokval, comment_ptr, i);
          tokval[i] = 0;
          tokval = savtokval;
          ptok(tokval);
        }

        lineptr = p + 1;
        /* comment is a space */
        retval = ' ';
        *tokval++ = ' ';
        *tokval = 0;
        if (XBIT(122, 2)) {
          --tokval;
          goto again;
        }
        goto nextok_ret;
      }

      lineptr = p;
    }
    /* NOTREACHED */;

  case '\r':
    if (XBIT(50, 0x20) || *lineptr != '\n') {
      /* not DOS end-of-line */
      *tokval++ = ' ';
    }
    retval = ' ';
    *tokval = 0;
    goto nextok_ret;

  default:
    switch (MASK(c)) {
    case SENTINEL:
      retval = T_SENTINEL;
      *tokval = 0;
      goto nextok_ret;
    case NOFUNC:
      retval = T_NOFIDENT;
      --tokval;
      p = lineptr;
      goto ident;
    case NOSUBST:
      retval = T_NSIDENT;
      --tokval;
      p = lineptr;
      goto ident;
    case ENDTOKEN:
      --tokval;
      goto again;
    case WHITESPACE:
      --tokval;
      goto again;
    default:
      break;
    }
  /*****  fall thru  *****/
  defret:
    *tokval = 0;
    retval = c;
    goto nextok_ret;
  } /* switch */

nextok_ret:

  popstack();
  return (retval);
}

static int
_nextline(void)
{
  static int lastinc = 0;
  FILE *fp;
  char *p;
  int i;
  int c;
  int j;
  int firstime;
  int dojoin;
  char *start;
  char *savestart;
  ptrdiff_t diff;
  int isquest;
  char buffer[133];
  char *fn;

again:
  fp = ifp;
  lineptr = linebuf + linelen;
  startline = inclstack[inclev].lineno;
  if ((c = getc(fp)) == EOF) {
    if (inclev <= 0)
      return (EOF);
    fn = inclstack[inclev].fname;
    for (i = 0; i < inclev - 1; i++) {
      if (!strcmp(inclstack[i].fname, fn)) {
        --inclstack[i].count;
        break;
      }
    }
    fclose(ifp);
    --inclev;
      gbl.curr_file = inclstack[inclev].fname;
      ifp = inclstack[inclev].ifp;
      idir.last = inclstack[inclev].path_idx;
      strcpy(cur_fname, inclstack[inclev].fname);
      cur_from_stdinc = inclstack[inclev].from_stdinc;
      pr_line(gbl.curr_file, inclstack[inclev].lineno, cur_from_stdinc);
    goto again;
  }

  i = 0;
  firstime = 1;
  savestart = start = lineptr;

joinlines:
  p = start;
  dojoin = 0;
  isquest = 0;

  while (c != EOF && c != '\n') {
    if (p - linebuf > linesize - 2) {
      diff = realloc_linebuf(0);
      start += diff;
      savestart += diff;
      p += diff;
    }
    *p++ = c;
    if ((c = getc(fp)) == '?')
      isquest = 1;
    ++i;
  }

  /* Decide to join lines or not */
  if (c == '\n') {
    /* tpr 1359 - don't join lines for fortran */
    /* tpr 1405 - need to join lines if a preprocessor directive */
    if (p > lineptr && lineptr[0] == '#' && c == '\n' && p[-1] == '\\') {
      /* join \<return> lines */
      start = p - 1;
      dojoin = 1;
    }
  }

  *p++ = '\n';
  *p = '\0';
  lineend = p;

  if (lastinc <= inclev) {
    ++inclstack[inclev].lineno;
    if (firstime)
      ++startline;
  }
  lastinc = inclev;
  if (list_flag && (flg.include || inclev == 0)) {
    if (inclev == 0)
      sprintf(buffer, "(%5d) ", inclstack[inclev].lineno);
    else
      sprintf(buffer, "(*%04d) ", inclstack[inclev].lineno);
    j = (p - savestart) - 1;
    if (j >= 132 - 8) {
      strncpy(&buffer[8], savestart, 132 - 8);
      buffer[132] = 0;
    } else {
      strcpy(&buffer[8], savestart);
      buffer[j + 8] = 0;
    }
    list_line(buffer);
  }
  if (dojoin) {
    firstime = 0;
    savestart = start;
    c = getc(fp);
    goto joinlines;
  }
  return *lineptr++;
}

static ptrdiff_t
realloc_linebuf(ptrdiff_t extra)
{
  char *old_linebuf;
  int old_linesize;
  int i;
  ptrdiff_t diff;

  if (extra < LINELEN)
    extra = LINELEN;
  old_linebuf = linebuf;
  old_linesize = linesize;

  /* Aquire a new line buffer */
  linelen += extra;
  linesize = (2 * linelen);
  NEW(linebuf, char, linesize);

  /* Allocate HALF of the new memory at the beginning and HALF at the end */
  memcpy(linebuf + extra, old_linebuf, old_linesize);

  /* Calculate the difference in location of linebuf's. */
  diff = linebuf + extra - old_linebuf;

  /* Adjust linebuf pointers in macro stack area */
  for (i = 0; i <= macstk_top; ++i)
    macstk[i].sav_lptr += diff;

  /* Adjust linebuf pointers to new area */
  lineptr += diff;
  lineend += diff;
  savlptr += diff;

  FREE(old_linebuf);

  return (diff);
}

static void
pbchar(int c)
{
  if (c == EOF)
    return;
  if (lineptr <= linebuf)
    realloc_linebuf(0);
  *--lineptr = c;
}

static void
pbstr(const char *s)
{
  register char *p;
  p = lineptr - strlen(s);
  if (p < linebuf) {
    realloc_linebuf(linebuf - p);
    p = lineptr - strlen(s);
  }
  lineptr = p;
  while (*s)
    *p++ = *s++;
}

static int
mac_push(PPSYM *sp, char *lptr)
{
  /* push macro info. on macstk */
  macstk_top += 1;
  if (macstk_top <= (MACSTK_MAX - 1)) {
    macstk[macstk_top].msptr = sp - hashrec;
    macstk[macstk_top].sav_lptr = lptr;
  } else {
    interr("Macro recursion stack size exceeded", startline, 4);
  }
  return 0;
}

static void
popstack(void)
{
  while (macstk_top > -1 && lineptr > macstk[macstk_top].sav_lptr)
    --macstk_top;
  /*prstk();*/
}

static int
macro_recur_check(PPSYM *sp)
{
  int i;

  for (i = 0; i <= macstk_top; ++i)
    if (macstk[i].msptr == sp - hashrec)
      return 1;
  return 0;
}

/* predicate support
 * we only allow #predicate(ident)
 * this is only for at&t compatibility
 * we allow at most NPRED predicates; more are ignored
 */

#define NPRED 20
/* each entry is the predicate name followed by ' ' followed by 1 argument */
static char *predtab[NPRED];
static int prednum = 1;
static int pred = 0;

#ifdef ATT_PREDICATE
static void
add_predicate(char *str)
{
  char *p, *q;

  p = str;
  for (;;) {
    if (*p == '\0')
      return;
    while (*p == ' ')
      ++p;
    if (*p != '#')
      return;
    if (prednum >= NPRED)
      return;
    ++p;
    for (q = p; *q && *q != '('; ++q)
      ;
    *q = ' ';
    while (*q && *q != ')')
      ++q;
    predtab[prednum++] = p;
    if (*q != 0)
      p = q + 1;
    else
      p = q;
    *q = 0;
  }
}
#endif

static void
predicate(char *tokval)
{
  int i, j;

  j = strlen(tokval);
  pred = 0;
  for (i = 1; i < prednum; ++i)
    if (strncmp(tokval, predtab[i], j) == 0 && predtab[i][j] == ' ') {
      pred = i;
      break;
    }
}

static int
predarg(char *tokval)
{
  int val;
  char *p;
  int j;

  val = 0;
  if (pred != 0) {
    for (p = predtab[pred]; *p != ' ' && *p; ++p)
      ;
    while (*p == ' ')
      ++p;
    j = strlen(tokval);
    if (strncmp(p, tokval, j) == 0 && (p[j] == 0 || p[j] == ' '))
      val = 1;
  }
  pred = 0;
  return val;
}

static void
stash_paths(const char *dirs)
{
  const char *path;
  int n;

  if (dirs == NULL)
    return;
  while (TRUE) {           /* loop over dirs */
    if (*dirs == DIRSEP) { /* skip separater */
      dirs++;
      continue;
    }
    if (*dirs == '\0') /* end of dirs */
      break;
    path = dirs;
    while ((*dirs != '\0') && (*dirs != DIRSEP))
      dirs++;

    n = dirs - path + 1;
    NEED(idir.b_avl + n, idir.b, char, idir.b_sz, idir.b_sz + n + 128);
    strncpy(idir.b + idir.b_avl, path, n);

    ++idir.cnt;
    NEED(idir.cnt + 1, idir.path, int, idir.sz, idir.sz + 64);
    idir.path[idir.cnt] = idir.b_avl;

    idir.b_avl += n;
    idir.b[idir.b_avl - 1] = '\0';
  }
}

static void
mod64(DBLINT64 x, DBLINT64 y, DBLINT64 res)
{
  DBLINT64 tmp1, tmp2;

  div64(x, y, tmp1);
  mul64(y, tmp1, tmp2);
  sub64(x, tmp2, res);
}

static void
umod64(DBLUINT64 x, DBLUINT64 y, DBLUINT64 res)
{
  DBLUINT64 tmp1, tmp2;

  udiv64(x, y, tmp1);
  umul64(y, tmp1, tmp2);
  usub64(x, tmp2, res);
}

void
setasmmode(void)
{
  asm_mode = TRUE;
  ctable['.'] |= _CS;
  ctable['%'] |= _CS;
  ctable['@'] |= _CS;
  ctable[':'] |= _CS;
} /* setasmmode */

void
setsuffix(char *newsuffix)
{
  suffix = newsuffix;
} /* setsuffix */

/* actuals contains pointers to arguments of a macro call.
   these should be updated whenever argbuf is resized */
static void
update_actuals(char **actuals, int nactuals, char *oldargbuf, char *newargbuf)
{
  int i;
  int offset;

  for (i = 0; i < nactuals; i++) {
    offset = actuals[i] - oldargbuf;
    actuals[i] = newargbuf + offset;
  }
}
