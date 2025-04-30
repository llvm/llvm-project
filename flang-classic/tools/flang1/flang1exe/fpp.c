/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    Fortran source preprocessor module (approximates "old-style" cpp).
*/

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "machar.h"
#include "version.h"

/* structures and data local to this module : */

/* char classification macros */

#define _CS 1 /* c symbol */
#define _DI 2 /* digit */
#define _BL 4 /* blank */
#define _HD 8 /* hex digit */

#define MASK(c) ((int)c & 0xFF)
#undef iscsym
#undef isblank
#undef ishex
#undef isident
#undef isdig
#define iscsym(c) (ctable[MASK(c)] & _CS)
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
    _BL,       /* ht  */
    0,         /* nl  */
    _BL,       /* vt  */
    _BL,       /* np  */
    _BL,       /* cr  */
    0,         /* so  */
    0,         /* si  */
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
    _BL,       /* sp  */
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
    /* the values of the remaining 128 chars are 0 -- escape sequences
     * may yield a char > 127.
     */
};

#define pperror(n, x, sev) error(n, sev, startline, x, CNULL)
#define pperr(n, sev) pperror(n, CNULL, sev)

#define PP_MAXIDLEN 1023
#define LINELEN 2048
#define HASHSIZ 1031
#define TOKMAX 2048
#define IFMAX 20
#define MACMAX 2048
#define FORMALMAX 31
#define ARGST 0xFE
/* * * * * MAX_FNAME_LEN MUST be less than scan.c:CARDB_SIZE * * * * */
#define MAX_FNAME_LEN 2050
#define MAXINC 20
#define MACSTK_MAX 100

/* Funny chars */
#define NOFUNC 0xFB

#define T_IDENT 'a'
#define T_NOFIDENT 'f'
#define T_REAL 'e'
#define T_INTEGER '0'
#define T_POUND (-2)
#define T_STRING 's'
#define T_COMMENT 'c'

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
/*  NON-ANSI directives go here -- D_MODULE is first NON-ANSI dir. */
#define D_MODULE 12
#define D_LIST 13
#define D_NOLIST 14
#define D_IDENT 15
#define D_ERROR 16
#define D_WARNING 17
#define D_INCLUDE_NEXT 18

static FILE *ifp;

/* record for #if stack */
struct ifrec {
  char truth;
  char true_seen;
  char else_seen;
};

/* symbol table entry -- chained hash table */
typedef struct {
  INT name;
  INT value;
  INT next;
} PPSYM;

/* token value */
static char tokval[TOKMAX];

/* if stack */
static int iftop;
static struct ifrec _ifs[IFMAX];

/* hash table */
static INT hashtab[HASHSIZ];
static PPSYM *hashrec;
static INT nhash, next_hash;

/* string table */
static char *deftab;
static INT ndef, next_def;

/* tokens for parser */
static int token;
static INT tokenval;
static int syntaxerr;

/* include stack */
static struct {
  char fname[MAX_FNAME_LEN];
  char dirname[MAX_FNAME_LEN];
  int lineno;
  int path_idx; /* Where (its INCL_PATH index) the include file
                 * was found.
                 */
  FILE *ifp;
} inclstack[MAXINC];
static int inclev = 0;

/* List of include files processed */
typedef char INCLENTRY[MAX_FNAME_LEN];
static INCLENTRY *incllist;
static int incfiles = 0;
static int inclsize;

/* buffer for nextok */
static char *lineptr, *lineend;
static char linebuf[LINELEN + LINELEN];
static int have_comment;
static int look_for_comments;
static int save_look_for_comments;
#define SKIP_COMMENTS XBIT(124, 0x100000)

/* __FILE__ and __LINE__ table offsets in hashrec */
static int lineloc, fileloc;

#define ifstack(name) _ifs[iftop].name

/* current file name and line number */
static char cur_fname[MAX_FNAME_LEN];
static int cur_line;

/* directory containing current file */
static char dirwork[MAX_FNAME_LEN];

/* start line for error messages and #line */
static int startline;

static int list_flag; /* listing? */

/* macro substitution stack to detect macro recursion */
struct macstk_type {
  INT msptr;
  char *sav_lptr;
};
static struct macstk_type macstk[MACSTK_MAX];
static int macstk_top = -1;

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

static char *argbuf;
static size_t ARGMAX = 2048;

void
accpp(void); /* FIXME this is defined in accpp.c, needs to be in a header */

static int skipbl(char *tokval, int flag);
static void pr_line(const char *name, int line);
static void doline(void);
static void dopragma(void);
static void doident(void);
static void doerror(void);
static void dowarning(void);
static void clreol(void);
static int dlookup(char *name);
static int inlist(char **list, int nl, char *value);
static void dodef(void);
static void doincl(LOGICAL include_next);
static void domodule(void);
static void doundef(void);
static int subst(PPSYM *sp);
static void ifpush(void);
static INT strstore(const char *name);
static PPSYM *lookup(const char *name, int insflg);
static void delete (char *name);
static void ptok(const char *tok);
static INT doparse(void);
static INT parse(int rbp);
static int gettoken(void);
static INT tobinary(char *st, int b);
static int gtok(char *tokval, int expflag);
static int findtok(char *tokval, int truth);
static int nextok(char *tokval);
static int _nextline(void);
static void pbchar(int c);
static void pbstr(char *s);
static void mac_push(PPSYM *sp, char *lptr);
static void popstack(void);
static void macro_recur_check(PPSYM *sp);
static void stash_paths(const char *dirs);

static int
skipbl(char *tokval, int flag)
{
  int toktyp;

  for (toktyp = gtok(tokval, flag); isblank(toktyp);
       toktyp = gtok(tokval, flag))
    ;

  return toktyp;
}

static void fpp_display_macros()
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
fpp(void)
{
  int toktyp;
  char *p;
  PPSYM *sp;
  char **cp;
  int done;
  int i;
  INT mon;
  char **dirp;
  static char adate[] = "\377\"Mmm dd yyyy\"";
  static const char *months[12] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  static char atime[] = "\377\"hh:mm:ss\"";
  static char a99[] = "\37799";

  /* By default always preprocess using ANSI-C99 semantics (accpp)
   * 0x20:     Preprocessor does not add whitespace
   * 0x800:    Preprocessor does not collapse whitespace
   */
  if (XBIT(123, 0x4000000) == 0) { /* If legacy (fpp) mode is off... */
    flg.x[123] |= 0x20 | 0x800;
    accpp();
    return;
  }

  list_flag = flg.list;

  if (strlen(gbl.file_name) < MAX_FNAME_LEN) {
    strcpy(inclstack[0].fname, gbl.file_name);
  } else {
    strncpy(inclstack[0].fname, gbl.file_name, MAX_FNAME_LEN);
    inclstack[0].fname[MAX_FNAME_LEN - 1] = 0;
  }
  dirnam(inclstack[0].fname, inclstack[0].dirname);
  gbl.curr_file = gbl.file_name;
  inclstack[0].ifp = gbl.srcfil;
  inclstack[0].lineno = 0;
  ifp = inclstack[0].ifp;

/* process def command line options */

  for (cp = flg.def; cp && *cp; ++cp) {
    /* enter it */
    for (p = *cp; *p && *p != '='; ++p)
      ;
    if (!*p) {
      sp = lookup(*cp, 1);
      sp->value = strstore("\3771"); /* define to be 1 */
    } else {
      int savec;
      savec = *p;
      *p = 0;
      sp = lookup(*cp, 1);
      *p = '\377';
      sp->value = strstore(p);
      *p = savec;
    }
  }
  strcpy(cur_fname, inclstack[0].fname);
  cur_line = 1;
  pr_line(cur_fname, cur_line);

  iftop = -1;
  ifpush();
  ifstack(truth) = 1;

/*  add std predefined macros  */
#define chkdef(name, val)        \
  {                              \
    sp = lookup(name, 1);        \
    if (sp->value != 0)          \
      pperror(220, name, 1);     \
    else                         \
      sp->value = strstore(val); \
  }

#define PGIF "__PGIF90__"
#define PGIF_MINOR "__PGIF90_MINOR__"
#define PGIF_PATCHLEVEL "__PGIF90_PATCHLEVEL__"

#ifdef TARGET_SUPPORTS_QUADFP
  chkdef("__flang_quadfp__", "1");
#endif

  if (XBIT(124, 0x200000)) {
    chkdef("pgi", "\3771"); /* define to be 1 */
  }
  chkdef("__PGI", "\3771"); /* define to be 1 */

  chkdef("__LINE__", "0");
  lineloc = sp - hashrec;
  chkdef("__FILE__", "1");
  fileloc = sp - hashrec;

  strncpy(&adate[3], gbl.datetime, 10);
  adate[5] = adate[8] = ' '; /* delete '/' */
  (void)atoxi(&adate[3], &mon, 2, 10);
  strncpy(&adate[2], months[mon - 1], 3);
  strncpy(&atime[2], &gbl.datetime[12], 8);

  chkdef("__DATE__", adate);
  chkdef("__TIME__", atime);

  chkdef("__STDC__", "\3771"); /* define to be 1 */

  p = version.vsn;
  tokval[0] = '\377';
  for (i = 0; i < TOKMAX - 1 && isdig(*p); i++)
    tokval[i + 1] = *p++;
  tokval[i + 1] = 0;
  if (i == 0) { /* no digits */
    chkdef("__PGIC__", a99) chkdef("__PGIC_MINOR__", a99) chkdef(PGIF, a99)
        chkdef(PGIF_MINOR, a99)
  } else {
    chkdef("__PGIC__",
           tokval) while (*p != 0 && !isdig(*p)) /* skip non-digits */
        p++;
    for (i = 0; i < TOKMAX - 1 && isdig(*p); i++)
      tokval[i + 1] = *p++;
    tokval[i + 1] = 0;
    if (i == 0) { /* no digits */
      chkdef("__PGIC_MINOR__", a99) chkdef(PGIF_MINOR, a99)
    } else {
      chkdef("__PGIC_MINOR__", tokval) chkdef(PGIF_MINOR, tokval)
    }
  }
  p = version.bld;
  tokval[0] = '\377';
  while (*p != 0 && !isdig(*p)) /* skip non-digits */
    p++;
  for (i = 0; i < TOKMAX - 1 && isdig(*p); i++)
    tokval[i + 1] = *p++;
  tokval[i + 1] = 0;
  if (i == 0) { /* no digits */
    chkdef("__PGIC_PATCHLEVEL__", a99) chkdef(PGIF_PATCHLEVEL, a99)
  } else {
    chkdef("__PGIC_PATCHLEVEL__", tokval) chkdef(PGIF_PATCHLEVEL, tokval)
  }

  /* value of _OPENMP is 201307, July, 2013 - version 4.0 */
  if (flg.smp && !XBIT(69, 1))
    chkdef("_OPENMP", "\377201307") /* NO SEMI */

  argbuf = sccalloc(ARGMAX + 1);
  if (argbuf == NULL)
    error(7, 4, 0, CNULL, CNULL);

#undef chkdef

/* process undef command line options */

  for (cp = flg.undef; cp && *cp; ++cp) {
    /* undef it */
    delete (*cp); /* should check if IDENT */
  }
  
  if (flg.list_macros) {
    fpp_display_macros();

    FREE(argbuf);
    FREE(deftab);
    FREE(hashrec);
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
  if (flg.stdinc == 0)
    stash_paths(DIRSINCS);
  else if (flg.stdinc != (char *)1)
    stash_paths(flg.stdinc);

  inclstack[0].path_idx = idir.last = -1;

  for (;;) {
    toktyp = findtok(tokval, ifstack(truth));
    if (toktyp == EOF)
      break;
    done = 0;
    toktyp = skipbl(tokval, 1);
    if (toktyp == '\n') { /* do nothing, ANSI 'NULL' directive */
      done = 1;           /* no clreol */
    } else if (toktyp != T_IDENT) {
      /* check for # <lineno> <file>; just ignore it */
      if (!isdig(*tokval))
        pperr(234, 3);
    } else {
      switch (dlookup(tokval)) {
      case D_IF:
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
        if (iftop == 0)
          pperr(258, 3);
        else if (ifstack(else_seen))
          pperr(221, 3);
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
        if (iftop == 0)
          pperr(259, 3);
        else if (ifstack(else_seen))
          pperr(222, 3);
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
        if (iftop == 0)
          pperr(260, 3);
        else
          --iftop;
        break;
      case D_IFDEF:
        ifpush(); /* sets truth value to old truth value */
        toktyp = skipbl(tokval, 0);
        if (toktyp != T_IDENT) {
          pperr(251, 2);
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
        ifpush(); /* sets truth value to old truth value */
        toktyp = skipbl(tokval, 0);
        if (toktyp != T_IDENT) {
          pperr(252, 2);
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
        if (ifstack(truth)) {
          doincl(FALSE);
          done = 1;
        }
        break;
      case D_INCLUDE_NEXT:
        done = 0;
        if (ifstack(truth)) {
          doincl(TRUE);
          done = 1;
        }
        break;
      case D_DEFINE:
        if (ifstack(truth)) {
          dodef();
          done = 1;
        }
        break;
      case D_UNDEF:
        if (ifstack(truth)) {
          doundef();
          done = 1;
        }
        break;
      case D_LINE:
        if (ifstack(truth)) {
          doline();
          done = 1;
        }
        break;
      case D_PRAGMA:
        if (ifstack(truth)) {
          dopragma();
          done = 1;
        }
        break;
      case D_MODULE:
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
        if (ifstack(truth))
          pperror(256, tokval, 3);
        break;
      } /* switch */
    }   /* else */
    if (!done)
      clreol();
  }

  if (iftop != 0)
    pperr(238, 3);

  if (!flg.es)
    (void)fseek(gbl.cppfil, 0L, 0);

  FREE(argbuf);
  FREE(deftab);
  FREE(hashrec);

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
  if (incllist != 0) {
    FREE(incllist);
  }
}

static void
pr_line(const char *name, int line)
{
  if (!XBIT(123, 0x100)) {
    /* if compilation doesn't stop after preprocessing (the conditions
     * for checking this must be the same as in main()), communicate
     * to the scanner that ensuing lines are/aren't from an include
     * file.  The convention used is if the character after the '#'
     * character is:
     *     '-'  -- lines are not from an include file
     *     '+'  -- lines are from an include file
     */
    if (name)
      fprintf(gbl.cppfil, "#%c%d \"%s\"\n",
              flg.es ? ' ' : inclev == 0 ? '-' : '+', line, name);
    else
      fprintf(gbl.cppfil, "#%c%d\n", flg.es ? ' ' : inclev == 0 ? '-' : '+',
              line);
  }
}

static void
doline(void)
{
  int toktyp;
  INT line;

  toktyp = skipbl(tokval, 1);
  if (toktyp != T_INTEGER) {
    pperr(248, 2);
  } else {
    line = tobinary(tokval, 10);
    toktyp = skipbl(tokval, 1);
    if (toktyp == '\n') {
      /* increments next time */
      inclstack[inclev].lineno = line - 1;
      return;
    } else if (toktyp == T_STRING) {
      tokval[0] = 0;
      tokval[strlen(tokval + 1)] = 0;
      strcpy(inclstack[inclev].fname, tokval + 1);
      /* increments next time */
      inclstack[inclev].lineno = line - 1;
    } else if (toktyp == T_IDENT) {
      strcpy(inclstack[inclev].fname, tokval);
      /* increments next time */
      inclstack[inclev].lineno = line - 1;
    } else
      pperr(248, 2);
  }
  clreol();
}

static void
dopragma(void)
{
  clreol();
}

static void
doident(void)
{
  clreol();
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
  error(282, 4, startline, buffer, CNULL);
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
  error(283, 2, startline, buffer, CNULL);
}

static void
clreol(void)
{
  int toktyp;
  while ((toktyp = gtok(tokval, 0)) != EOF && toktyp != '\n')
    ;
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
inlist(char **list, int nl, char *value)
{
  int i;

  for (i = 0; i < nl; ++i)
    if (strcmp(list[i], value) == 0)
      return (i + 1);
  return (0);
}

static void
dodef(void)
{
  int toktyp;
  PPSYM *sp;
  int nformals;
  char macbuf[MACMAX + 1];
  char *formals[FORMALMAX];
  char *p;
  char *defp;
  int i;
  char *q, *q1;
  char savec;

  /* fetch symbol after #define */
  p = macbuf;
  toktyp = skipbl(tokval, 0);
  if (toktyp != T_IDENT) {
    pperr(235, 2); /* illegal macro name is warning in cpp */
    clreol();
    return;
  }
  /* look up the name */
  sp = lookup(tokval, 1);
  /* here we need to read the definition */
  nformals = -1;
  toktyp = gtok(tokval, 0);
  if (toktyp == '(') {
    nformals = 0;
    toktyp = skipbl(tokval, 0);
    for (;;) {
      if (toktyp == ')')
        break;
      if (toktyp != T_IDENT) {
        pperror(244, tokval, 3);
        clreol();
        goto nulldef;
      }
      /* save formal */
      if (nformals + 1 > FORMALMAX)
        pperror(254, &deftab[sp->name], 3);
      else if (p - macbuf + (int)strlen(tokval) + 1 > MACMAX)
        pperror(231, &deftab[sp->name], 3);
      else {
        strcpy(p, tokval);
        formals[nformals++] = p;
        p += strlen(tokval) + 1;
      }
      toktyp = skipbl(tokval, 0);
      if (toktyp == ',') {
        toktyp = skipbl(tokval, 0);
        continue;
      }
      if (toktyp == '\n') {
        pperror(262, &deftab[sp->name], 3);
        goto nulldef;
      }
    }
    toktyp = gtok(tokval, 0);
  }
  /*
   * now nformals has # formals, formals has formals, and token is first
   * after macro def
   */

  if (toktyp == '\n') {
  nulldef:
    if (sp->value && (((int)deftab[sp->value] & 0xFF) != (nformals & 0xFF) ||
                      deftab[sp->value + 1])) {
      pperror(241, &deftab[sp->name], 2); /* redef is warning in cpp */
    } else if (sp->value)
      pperror(242, &deftab[sp->name], 1);
    sp->value = strstore("\377");
    deftab[sp->value] = nformals;
    return;
  }
  if (nformals == -1 && !isblank(toktyp))
    pperr(245, 1); /* this maybe should be removed or made inform */
  if (isblank(toktyp)) {
    /* never add a blank for macros - the blank is a problem if the
     * macro expands to a directive.
     */
    toktyp = skipbl(tokval, 0);
  }
  /* collect definition */
  defp = p;
  *p++ = nformals;
  for (;;) {
    if (toktyp == T_IDENT) {
      if ((i = inlist(formals, nformals, tokval)) != 0) {
        /* we found a formal */
        if (p - macbuf + 2 > MACMAX) {
          goto lenerr;
        }
        *p++ = i;
        *(unsigned char *)p =
            ARGST; /* comes last since subst scans backwards */
        p++;
      } else
        goto literal;
    } else if (toktyp == T_STRING) {
      /* must scan the string looking for an ident in formal */
      q = tokval;
      for (;;) {
        while (*q && !iscsym(*q)) {
          if (p - macbuf + 1 > MACMAX)
            goto lenerr;
          *p++ = *q++;
        }
        if (*q == 0)
          break;
        q1 = q;
        while (iscsym(*q))
          ++q;
        savec = *q;
        *q = 0;
        if ((i = inlist(formals, nformals, q1)) != 0) {
          /* we found a formal */
          if (p - macbuf + 2 > MACMAX) {
          lenerr:
            pperror(227, &deftab[sp->name], 3);
            goto nulldef;
          }
          *p++ = i;
          *(unsigned char *)p = ARGST;
          p++;
        } else {
          if (p - macbuf + (int)strlen(q1) > MACMAX)
            goto lenerr;
          strcpy(p, q1);
          p += strlen(q1);
        }
        *q = savec;
      } /* for(;;) */
    } else {
    literal:
      if (p - macbuf + (int)strlen(tokval) > MACMAX)
        goto lenerr;
      /* if only a space and a comment after macro name arrive here,
         define macro but it has no body */
      if (toktyp == '\n')
        goto nulldef;
      strcpy(p, tokval);
      p += strlen(tokval);
    }
    toktyp = gtok(tokval, 0);
    if (toktyp == '\n')
      break;
  }
  *p = 0;
  /* install definition */
  if (nformals == 0)
    *defp = 1;
  /* compare definitions */
  if (sp->value && (strcmp(defp + 1, &deftab[sp->value + 1]) ||
                    ((int)deftab[sp->value] & 0xFF) != (nformals & 0xFF))) {
    pperror(241, &deftab[sp->name], 2); /* warning in cpp */
  } else if (sp->value)
    pperror(242, &deftab[sp->name], 1); /* maybe should be removed */
  sp->value = strstore(defp);
  if (nformals == 0)
    deftab[sp->value] = nformals;
}

static void
doincl(LOGICAL include_next)
{
  FILE *tmpfp;
  int toktyp;
  char buff[MAX_FNAME_LEN];
  char fullname[MAX_FNAME_LEN];
  char *p;
  int type;
  int i;

  /* parse file name */
  toktyp = skipbl(tokval, 1);
  /* ### there is a problem here; "" filename not really a string */
  if ((toktyp != T_STRING && toktyp != '<') ||
      (toktyp == T_STRING && *tokval == '\'')) {
    pperr(247, 3);
    return;
  } else if (toktyp == T_STRING) {
    strncpy(buff, tokval + 1, MAX_FNAME_LEN);
    if ((p = strchr(buff, '"')) != NULL)
      *p = 0;
    buff[MAX_FNAME_LEN - 1] = 0;
    type = 0;
  } else {
    /* toktyp == '<' */
    buff[0] = 0;
    for (;;) {
      toktyp = gtok(tokval, 0);
      if (toktyp == '>')
        break;
      if (toktyp == EOF) {
        pperr(257, 3);
        break;
      }
      strncat(buff, tokval, MAX_FNAME_LEN);
    }
    buff[MAX_FNAME_LEN - 1] = 0;
    type = 1;
  }
  clreol();
  if (inclev >= MAXINC - 1) {
    pperr(261, 3);
    return;
  }
  /* look for included file:  */

  if (type == 0 && (idir.last == -1 || !include_next)) {
    /* look first in directory containing current included or base file */
    strcpy(dirwork, inclstack[inclev].dirname);
    i = strlen(dirwork) - 1;
    if (i >= 0 && dirwork[i] == '/')
      dirwork[i] = 0;
    if (fndpath(buff, fullname, MAX_FNAME_LEN, dirwork) == 0) {
      idir.last = 0;
      goto found;
    }
  }
  i = 1;
  if (include_next && idir.last >= 0)
    i = idir.last + 1;
  for (; i <= idir.cnt; i++)
    if (fndpath(buff, fullname, MAX_FNAME_LEN, INCLPATH(i)) == 0) {
      idir.last = i;
      goto found;
    }

  if (type == 0) { /* could be absolute path, check where it leads to */
    tmpfp = fopen(buff, "r");
    if (tmpfp) {
      snprintf(fullname, MAX_FNAME_LEN, "%s", buff);
      idir.last = 0;
      if (fclose(tmpfp) == 0)
        goto found;
    }
  }

  pperror(226, buff, 4); /* cpp just continues, but why?? */
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
  strcpy(inclstack[inclev].fname, fullname);
  dirnam(inclstack[inclev].fname, inclstack[inclev].dirname);
  gbl.curr_file = inclstack[inclev].fname;
  inclstack[inclev].lineno = 0;
  inclstack[inclev].path_idx = idir.last;

  /* -M and -MD option:  Save list of include files processed
   * don't save if this is an include file and -x 123 0x4000 is
   * set (-MM/-MMD)
   */
  if ((XBIT(123, 2) || XBIT(123, 8)) && (type == 0 || !XBIT(123, 0x4000))) {
    if (incllist == 0) {
      inclsize = 20;
      NEW(incllist, INCLENTRY, inclsize);
    } else {
      NEED(incfiles + 1, incllist, INCLENTRY, inclsize, inclsize + 20);
    }
    strcpy(incllist[incfiles], fullname);
    incfiles++;
    for (i = 0; i < incfiles - 1; i++)
      if (!strcmp(incllist[i], fullname)) {
        incfiles--; /* Ignore duplicate */
        break;
      }
  }
  pr_line(fullname, 1);
}

static void
domodule(void)
{
  int toktyp;
  char *cp;
  int i;

  toktyp = skipbl(tokval, 1);
  if (toktyp != T_IDENT) {
    pperr(249, 2);
    return;
  }
  NEW(cp, char, PP_MAXIDLEN + 1);
  gbl.module = cp;
  for (i = 0; i < PP_MAXIDLEN; ++i) {
    if (tokval[i] == 0)
      break;
    *cp++ = tokval[i];
  }
  *cp = 0;
  clreol();
}

static void
doundef(void)
{
  int toktyp;

  toktyp = skipbl(tokval, 0);
  if (toktyp != T_IDENT) {
    pperr(250, 2);
  } else {
    delete (tokval);
  }
  clreol();
}

static int
subst(PPSYM *sp)
{
  char *p, *q;
  int nformals;
  int toktyp;
  char *actuals[FORMALMAX];
  int nactuals;
  char *argp;
  char *fstart;
  int nlpar;
  int i;
  int nl;
  size_t d;

  if (sp == hashrec + lineloc) {
    sprintf(argbuf, "%d", inclstack[inclev].lineno);
    pbstr(argbuf);
    return (0);
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
    pbstr(argbuf);
    return (0);
  }
  argbuf[ARGMAX] = 0;
  p = &deftab[sp->value];
  nformals = (int)*p & 0xFF;
  if (nformals == 0xFF) {
    /* macro has no args; just substitute */
    mac_push(sp, lineptr);
    pbstr(p + 1);
    return (0);
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
  if (toktyp != '(') { /* this is probably wrong (funclike vs ..) */
    if (XBIT(124, 0x10000)) {
      /* cpp mode */
      pperror(239, &deftab[sp->name], 2);
      /*pbstr(p + 1);*/
      nactuals = 0;
      goto substitute;
    }
    if (toktyp == EOF) {
      ptok(&deftab[sp->name]);
      ptok("\n");
      return EOF;
    }
    pbstr(tokval);
    if (nl) { /* FS#14715 need to preserve newline */
      pbchar('\n');
      cur_line++;
    }
    if (i)
      pbchar(' ');
    pbstr(&deftab[sp->name]);
    pbchar(NOFUNC); /* to prevent rescanning */
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
          pperror(253, &deftab[sp->name], 3);
        else {
          d = argp - argbuf;
          if (d >= ARGMAX) {
            ARGMAX = d + 2048;
            argbuf = sccrelal(argbuf, ARGMAX + 1);
            if (argbuf == NULL) {
              pperror(224, &deftab[sp->name], 3);
              error(7, 4, 0, CNULL, CNULL);
            }
            argbuf[ARGMAX] = 0;
            argp = argbuf + d;
          }
          actuals[nactuals++] = fstart;
          *argp++ = 0;
          fstart = argp;
        }
        continue;
      }
    } else if (toktyp == EOF) {
      pperror(229, &deftab[sp->name], 3);
      return (-1);
    }
    d = argp - argbuf;
    if (d + (int)strlen(tokval) > ARGMAX) {
      ARGMAX = d + strlen(tokval) + 2048;
      argbuf = sccrelal(argbuf, ARGMAX + 1);
      if (argbuf == NULL) {
        pperror(224, &deftab[sp->name], 3);
        error(7, 4, 0, CNULL, CNULL);
      }
      argbuf[ARGMAX] = 0;
      argp = argbuf + d;
    }
    strcpy(argp, tokval);
    argp += strlen(tokval);
  }
  /* save last actual */
  if (nactuals + 1 > FORMALMAX)
    pperror(253, &deftab[sp->name], 3);
  else {
    d = argp - argbuf;
    if (d >= ARGMAX) {
      ARGMAX = d + 2048;
      argbuf = sccrelal(argbuf, ARGMAX + 1);
      if (argbuf == NULL) {
        pperror(224, &deftab[sp->name], 3);
        error(7, 4, 0, CNULL, CNULL);
      }
      argbuf[ARGMAX] = 0;
      argp = argbuf + d;
    }
    actuals[nactuals++] = fstart;
    *argp++ = 0;
  }

  /* substitute */
  if (nformals != nactuals)
    if (!(nformals == 0 && nactuals == 1 && *actuals[0] == '\0'))
      pperror(225, &deftab[sp->name], 2); /* cpp is warning */

substitute:
  q = p;
  ++p;
  p += strlen(p); /* end of macro text */
  mac_push(sp, lineptr);
  for (;;) {
    while ((*--p & 0xFF) != ARGST && p > q)
      pbchar(*p);
    if (p <= q)
      break;
    --p;
    if (*p <= nactuals)
      pbstr(actuals[*p - 1]);
  }
  return 0;
}

static void
ifpush(void)
{
  if (iftop >= IFMAX - 1) {
    pperr(223, 3);
  } else {
    ++iftop;
    ifstack(true_seen) = 0;
    ifstack(else_seen) = 0;
    if (iftop > 0)
      ifstack(truth) = _ifs[iftop - 1].truth;
  }
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
  return (j);
}

static PPSYM *
lookup(const char *name, int insflg)
{
  int i;
  char *cp;
  INT p, q;
  INT l, m;
  char buff[PP_MAXIDLEN + 1];

  i = 0;
  l = m = 0l;

  strncpy(buff, name, PP_MAXIDLEN);
  buff[PP_MAXIDLEN] = 0;
  /*
   * break name into 3 byte chunks, sum them together, take remainder
   * modulo table size
   */
  cp = buff;
  while (*cp != '\0') {
    if (i == 3) {
      i = 0;
      m += l;
      l = 0l;
    }
    ++i;
    l <<= 8;
    l |= MASK(*cp);
    cp++;
  }
  m += l;
  i = m % HASHSIZ;
  for (q = 0, p = hashtab[i]; p != 0; q = p, p = hashrec[p].next)
    if (strcmp(buff, &deftab[hashrec[p].name]) == 0)
      return (&hashrec[p]);
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
  if (q == 0) {
    hashrec[p].next = hashtab[i];
    hashtab[i] = p;
  } else {
    hashrec[q].next = p;
    hashrec[p].next = 0;
  }
  return (&hashrec[p]);
}

static void delete (char *name)
{
  int i;
  INT l, m;
  INT p, q;
  char *cp;

  i = 0;
  l = m = 0l;

  if ((int)strlen(name) > PP_MAXIDLEN)
    name[PP_MAXIDLEN] = 0;
  /*
   * break name into 3 byte chunks, sum them together, take remainder
   * modulo table size
   */
  cp = name;
  while (*cp != '\0') {
    if (i == 3) {
      i = 0;
      m += l;
      l = 0l;
    }
    ++i;
    l <<= 8;
    l |= *cp++;
  }
  m += l;
  i = m % HASHSIZ;
  for (q = 0, p = hashtab[i]; p != 0; q = p, p = hashrec[p].next)
    if (strcmp(name, &deftab[hashrec[p].name]) == 0)
      goto found;
  return;
found:
  /* delete record */
  if (q == 0) { /* first in list */
    hashtab[i] = hashrec[p].next;
  } else {
    hashrec[q].next = hashrec[p].next;
  }
}

static void
ptok(const char *tok)
{
  FILE *fp;
  static int state = 1;

  /* keep track of where compiler thinks we are */
  if (*tok == '\n') {
    state = 1; /* next token starts a new line */
    ++cur_line;
  }
  /* if starting a new line, make sure line # and file are correct */
  else if (state) {
    state = 0;
    /* make sure next line and file are correct */
    if (strcmp(cur_fname, inclstack[inclev].fname)) {
      strcpy(cur_fname, inclstack[inclev].fname);
      cur_line = startline;
      if (cur_line != 1)
        pr_line(cur_fname, cur_line);
    } else if (cur_line != startline) {
      cur_line = startline;
      pr_line(NULL, cur_line);
    }
  }
  fp = gbl.cppfil;
  while (*tok)
    (void)putc(*tok++, fp);
}

/*---------------------------------------------------------------*/

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

/*  define parse table, PT, which has an entry for each of the
    tokens just defined:  */
static struct {
  INT8 led;  /* TRUE if this token is a legal infix
              * operator  */
  INT8 nud;  /* 0 if this token isn't a legal prefix
              * operator, else token to use in switch
              * stmt.  */
  short lbp; /* left binding power  */
  INT8 rbp;  /* right binding power  */
} PT[] = {
    {0, 0, 0, 0},        {0, Number, 0, 0}, /* Number */
    {0, Lparen, 0, 0},                      /* Lparen */
    {0, 0, 0, 0},                           /* Rparen */
    {0, 0, -1, 0},                          /* Eoln   */
    {0, Defined, 0, 0},                     /* Defined */
    {1, 0, 60, 60},                         /* Mult   */
    {1, 0, 60, 60},                         /* Divide */
    {1, 0, 60, 60},                         /* Mod    */
    {1, 0, 50, 50},                         /* Plus   */
    {1, Minusx, 50, 50},                    /* Minus  */
    {1, 0, 40, 40},                         /* Lshift */
    {1, 0, 40, 40},                         /* Rshift */
    {1, 0, 35, 35},                         /* Lt     */
    {1, 0, 35, 35},                         /* Gt     */
    {1, 0, 35, 35},                         /* Le     */
    {1, 0, 35, 35},                         /* Ge     */
    {1, 0, 30, 30},                         /* Eq     */
    {1, 0, 30, 30},                         /* Ne     */
    {1, 0, 27, 27},                         /* And    */
    {1, 0, 24, 24},                         /* Xor    */
    {1, 0, 21, 21},                         /* Or     */
    {1, 0, 18, 18},                         /* Andand */
    {1, 0, 15, 15},                         /* Oror   */
    {1, 0, 10, 9},                          /* Question */
    {0, 0, 0, 0},                           /* Colon  */
    {1, 0, 5, 5},                           /* Comma  */
    {0, 0, 0, 100},                         /* Minusx */
    {0, Not, 0, 100},                       /* Not    */
    {0, Compl, 0, 100},                     /* Compl  */
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

  token = gettoken();
  if (token == Eoln) {
    pperr(246, 3);
    return (1);
  }
  syntaxerr = 0;
  i = parse(-1);
  if (syntaxerr)
    if (token != Eoln)
      clreol();
  return (i);
}

/*  parse "current" expression and return value computed
    by constant folding it. */
static INT
parse(int rbp)
{
  INT right, tmp, left;
  int op;
  LOGICAL valid_left = FALSE;
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

    if (op != Number && op != Defined)
      right = parse(PT[op].rbp);

    switch (op) {
    case Number:
      left = tmp;
      break;
    case Lparen:
      left = right;
      CHECK(Rparen);
      break;
    case Defined:
      tmp = token;
      if (token == Lparen)
        token = gettoken();
      left = tokenval;
      CHECK(Number);
      if (tmp == Lparen)
        CHECK(Rparen);
      break;
    case Mult:
      left *= right;
      break;
    case Divide:
      left /= right;
      break;
    case Mod:
      left %= right;
      break;
    case Plus:
      left += right;
      break;
    case Minus:
      left -= right;
      break;
    case Lshift:
    case Rshift:
#ifdef TM_REVERSE_SHIFT
      if (right < 0) {
        right = -right;
        op = (Lshift + Rshift) - op;
      }
#endif
      if (op == Lshift)
        left = LSHIFT(left, right);
      else /* for now, can't tell if it's signed */
        left = URSHIFT(left, right);
      break;
    case Lt:
      left = left < right;
      break;
    case Gt:
      left = left > right;
      break;
    case Le:
      left = left <= right;
      break;
    case Ge:
      left = left >= right;
      break;
    case Eq:
      left = left == right;
      break;
    case Ne:
      left = left != right;
      break;
    case And:
      left &= right;
      break;
    case Xor:
      left ^= right;
      break;
    case Or:
      left |= right;
      break;
    case Andand:
      left = (left && right);
      break;
    case Oror:
      left = (left || right);
      break;
    case Question:
      CHECK(Colon);
      tmp = parse(PT[Question].rbp);
      left = left ? right : tmp;
      break;
    case Comma:
      left = right;
      break;
    case Minusx:
      left = -right;
      break;
    case Not:
      left = !right;
      break;
    case Compl:
      left = ~right;
      break;
    default:
      goto syntax_error;
    }
  } while (PT[token].lbp > rbp);

  return left;

syntax_error:
  pperr(246, 3);
  syntaxerr = 1;
  token = Eoln;
  return 1;
}

static int
gettoken(void)
{
  static int ifdef = 0;
  char *s;
  PPSYM *sp;
  int toktyp;

  for (;;) {
    save_look_for_comments = look_for_comments;
    look_for_comments = 0;
    toktyp = skipbl(tokval, !ifdef);
    look_for_comments = save_look_for_comments;
    if (toktyp == '\n')
      return Eoln; /* end of #if */
    switch (toktyp) {
    case '|':
      if ((toktyp = gtok(tokval, 1)) == '|')
        return (Oror);
      pbstr(tokval);
      return (Or);
    case '&':
      if ((toktyp = gtok(tokval, 1)) == '&')
        return (Andand);
      pbstr(tokval);
      return (And);
    case '>':
      if ((toktyp = gtok(tokval, 1)) == '>')
        return (Rshift);
      else if (toktyp == '=')
        return (Ge);
      pbstr(tokval);
      return (Gt);
    case '<':
      if ((toktyp = gtok(tokval, 1)) == '<')
        return (Lshift);
      else if (toktyp == '=')
        return (Le);
      pbstr(tokval);
      return (Lt);
    case '!':
      if ((toktyp = gtok(tokval, 1)) == '=')
        return (Ne);
      pbstr(tokval);
      return (Not);
    case '=':
      if ((toktyp = gtok(tokval, 1)) == '=')
        return (Eq);
      pbstr(tokval);
      goto illch;
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
    case T_INTEGER:
      /* parse number */
      if (tokval[0] == '0')
        tokenval = (tokval[1] == 'x' || tokval[1] == 'X')
                       ? tobinary(tokval + 2, 16)
                       : tobinary(tokval + 1, 8);
      else
        tokenval = tobinary(tokval, 10);
      return (Number);
    case T_IDENT:
      if (0 == strcmp(tokval, "defined")) {
        ifdef = 1;
        return (Defined);
      } else {
        sp = lookup(tokval, 0);
        if (ifdef != 0)
          ifdef = 0;
        tokenval = sp && sp->value != 0;
        return (Number);
      }
    case T_STRING:
      for (s = tokval + 1; *s != 0 && *s != '\''; ++s)
        if (*s == '\\') {
          ++s;
          if (*s == 0)
            break;
        }
      if (*s == '\'')
        *s = 0;

      if (tokval[1] == '\\') { /* escaped */
        switch (tokval[2]) {
        case 'a':
          tokenval = '\007';
          break;
        case 'b':
          tokenval = '\b';
          break;
        case 'f':
          tokenval = '\f';
          break;
        case 'n':
          tokenval = '\n';
          break;
        case 'r':
          tokenval = '\r';
          break;
        case 't':
          tokenval = '\t';
          break;
        case 'v':
          tokenval = '\v';
          break;
        case 'x':
          tokenval = tobinary(tokval + 3, 16) & 0xFF;
#ifndef CHAR_IS_UNSIGNED
          if (tokenval & 0x80)
            tokenval |= 0xffffff00;
#endif
          break;
        default:
          if (tokval[2] <= '7' && tokval[2] >= '0') {
            tokenval = tobinary(tokval + 2, 8) & 0xFF;
#ifndef CHAR_IS_UNSIGNED
            if (tokenval & 0x80)
              tokenval |= 0xffffff00;
#endif
          } else
            tokenval = tokval[2];
          break;
        }
      } else
        tokenval = tokval[1];
      return (Number);

    default:
    illch:
      pperr(246, 3);
      continue;
    }
  }
}

static INT
tobinary(char *st, int b)
{
  INT n, c, t;
  char *s;

  n = 0;
  s = st;
  c = 10;
  while ((c = *(s++))) {
    switch (c) {
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
      t = c - '0';
      break;
    case 'a':
    case 'b':
    case 'c':
    case 'd':
    case 'e':
    case 'f':
      t = c - 'a' + 10;
      if (b > 10)
        break;
      FLANG_FALLTHROUGH;
    case 'A':
    case 'B':
    case 'C':
    case 'D':
    case 'E':
    case 'F':
      t = c - 'A' + 10;
      if (b > 10)
        break;
      FLANG_FALLTHROUGH;
    default:
      t = -1;
      if (c == 'l' || c == 'L' || c == 'u' || c == 'U')
        if (*s == '\0' || *s == 'l' || *s == 'L' || *s == 'u' || *s == 'U')
          break;
      /* note that we don't handle unsigned arithmetic */
      pperror(236, st, 3);
    }
    if (t < 0)
      break;
    n = n * b + t;
  }
  return n;
}

static int
gtok(char *tokval, int expflag)
{
  PPSYM *sp;
  int toktyp;

again:
  toktyp = nextok(tokval);
  if (toktyp == T_NOFIDENT)
    toktyp = T_IDENT;
  else if (toktyp == T_IDENT && expflag && (sp = lookup(tokval, 0)) != 0) {
    /*macro_recur_check(sp);*/
    /* scan macro and substitute */
    if (subst(sp) == EOF)
      return (EOF);
    goto again;
  }
  return (toktyp);
}

static int
findtok(char *tokval, int truth)
{
  int state;
  PPSYM *sp;
  int toktyp;

  state = 1;

  if (truth) {
    for (;;) {
      toktyp = nextok(tokval);
      if (toktyp == EOF)
        return (EOF);
      if (toktyp == '#' && state) {
        return (T_POUND);
      }
      if (!isblank(toktyp) || !flg.freeform)
        state = 0; /* allow whitespace before '#' directives */
      if (toktyp == T_NOFIDENT)
        toktyp = T_IDENT;
      else if (toktyp == T_IDENT && (sp = lookup(tokval, 0)) != 0) {
        /*macro_recur_check(sp);*/
        /* scan macro and substitute */
        if (subst(sp) == EOF)
          return (EOF);
        continue;
      }
      if (toktyp == '\n')
        state = 1;
      ptok(tokval);
    }
  } else {
    for (;;) {
      toktyp = nextok(tokval);
      if (toktyp == EOF)
        return (EOF);
      if (toktyp == '#' && state) {
        return (T_POUND);
      }
      if (!isblank(toktyp) || !flg.freeform)
        state = 0; /* allow whitespace before '#' directives */
      if (toktyp == '\n')
        state = 1;
    }
  }
}

#define inchar() (lineptr < lineend ? *lineptr++ : _nextline())

static int
nextok(char *tokval)
{
  int i;
  char *p;
  int delim;
  int c;
  int toktyp;
  int retval;

again:
  if ((c = inchar()) == EOF) {
    retval = EOF;
    goto nextok_ret;
  }
  *tokval++ = c;
  switch (c) {
  case 'C':
  case 'c':
    if (!flg.freeform && look_for_comments && lineptr == linebuf + LINELEN + 1)
    {
      /* C in column 1, start of old-style comment */
      /* everything to end of line is a comment */
      if (!SKIP_COMMENTS)
        have_comment = 1;
      else {
        *tokval = 0;
        retval = T_COMMENT;
        p = lineptr;
        while (*p != '\n')
          ++p;
        if ((i = p - lineptr) >= TOKMAX - 1) {
          i = TOKMAX - 1;
        }
        strncpy(tokval, lineptr, i);
        tokval[i] = 0;
        lineptr = p;
        goto nextok_ret;
      }
    }
    FLANG_FALLTHROUGH;
  case 'A':
  case 'B':
  case 'D':
  case 'E':
  case 'F':
  case 'G':
  case 'H':
  case 'I':
  case 'J':
  case 'K':
  case 'L':
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
  case '$':
  case 'a':
  case 'b':
  case 'd':
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
      pperr(232, 3);
      i = TOKMAX - 1;
    }
    strncpy(tokval, lineptr, i);
    tokval[i] = 0;
    lineptr = p;
    goto nextok_ret;

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
    if (c == '0' && (*p == 'x' || *p == 'X')) {
      /* HEX number */
      ++p;
      while (ishex(*p++))
        /* NULL */;
      --p;
      if (*p == 'l' || *p == 'L')
        ++p;
      if ((i = p - lineptr) >= TOKMAX - 1) {
        pperr(240, 3);
        i = TOKMAX - 1;
      }
      strncpy(tokval, lineptr, i);
      tokval[i] = 0;
      lineptr = p;
      retval = T_INTEGER;
      goto nextok_ret;
    }
    while (isdig(*p++))
      /* NULL */;
    --p;
    toktyp = T_INTEGER;
    if (*p == 'l' || *p == 'L')
      ++p;
    else {
      /* might be floating point */
      if (*p == '.') {
      dotnum:
        toktyp = T_REAL;
        ++p;
        while (isdig(*p++))
          /* NULL */;
        --p;
      }
      if (*p == 'e' || *p == 'E') {
        toktyp = T_REAL;
        ++p;
        if (*p == '+' || *p == '-')
          ++p;
        while (isdig(*p++))
          /* NULL */;
        --p;
      }
    }
    if ((i = p - lineptr) >= TOKMAX - 1) {
      pperr(240, 3);
      i = TOKMAX - 1;
    }
    strncpy(tokval, lineptr, i);
    tokval[i] = 0;
    lineptr = p;
    retval = toktyp;
    goto nextok_ret;

  case '.':
    c = inchar();
    pbchar(c);
    if (isdig(c)) {
      p = lineptr; /* points to digit */
      /* tokval already contains '.' */
      goto dotnum;
    }
    *tokval = 0;
    retval = '.';
    goto nextok_ret;

  case '"':
  case '\'':
#define DOCOPY                               \
  do {                                       \
    if ((i += p - lineptr) >= TOKMAX) {      \
      pperr(243, 4);                         \
    } else {                                 \
      strncpy(tokval, lineptr, p - lineptr); \
      *(tokval += p - lineptr) = 0;          \
      lineptr = p;                           \
    }                                        \
  } while (0);
    delim = c;
    i = 1;
    p = lineptr;
    toktyp = T_STRING;
    for (;;) {
      while (p < lineend && *p != '\\' && *p != delim && *p != '\n')
        ++p;
      if (p >= lineend || *p == '\n') {
      unterm:
        /* don't issue 'Unterminated string' message, since it
           may be in a comment -- rely on the scanner to detect
           the error.
        pperr(263, 1);
        */
        /* copy to tokval */
        DOCOPY;
        retval = toktyp;
        goto nextok_ret;
      } else if (*p == delim) {
        ++p;
        /* copy to tokval */
        DOCOPY;
        retval = toktyp;
        goto nextok_ret;
      } else {
        /* backslash */
        ++p;
        if (p >= lineend)
          goto unterm;
        if (*p != '\n') {
          ++p;
        } else {
          --p;
          DOCOPY;
          p += 2;
          lineptr = p;
          if (p >= lineend) {
            if (lineptr < p) {
              DOCOPY;
            }
            if (_nextline() == EOF) {
              pperr(230, 3);
              retval = EOF;
              goto nextok_ret;
            }
            --lineptr;
            p = lineptr;
          }
        }
      }
    }
    /* NOTREACHED */;

  case '/':
    c = inchar();
    if (c != '*' || have_comment) {
      pbchar(c);
      *tokval = 0;
      retval = '/';
      goto nextok_ret;
    } else {
      *--tokval = 0;
      for (;;) {
        /* scan to end of line or '*' */
        p = lineptr;
        while (p < lineend && *p != '*')
          ++p;
        if (p >= lineend) {
          if (_nextline() == EOF) {
            pperr(228, 3);
            *tokval = 0;
            retval = EOF;
            goto nextok_ret;
          }
          --lineptr;
          continue;
        }
        if (p > lineptr && p[-1] == '/') /* possible nested comment */
          pperr(264, 1);
        if (*++p == '/') {
          lineptr = p + 1;
          goto again;
        }
        lineptr = p;
      }
    }
    /* NOTREACHED */;

  case '*':
    if (!flg.freeform && look_for_comments && lineptr == linebuf + LINELEN + 1)
    {
      /* * in column 1, start of old-style comment */
      /* everything to end of line is a comment */
      if (!SKIP_COMMENTS)
        have_comment = 1;
      else {
        *tokval = 0;
        retval = T_COMMENT;
        p = lineptr;
        while (*p != '\n')
          ++p;
        if ((i = p - lineptr) >= TOKMAX - 1) {
          i = TOKMAX - 1;
        }
        strncpy(tokval, lineptr, i);
        tokval[i] = 0;
        lineptr = p;
        goto nextok_ret;
      }
    }
    /* FALL THROUGH */
    *tokval = 0;
    retval = c;
    goto nextok_ret;

  case '!':
    if (look_for_comments) {
      /* everything to end of line is a comment */
      if (!SKIP_COMMENTS)
        have_comment = 1;
      else {
        *tokval = 0;
        retval = T_COMMENT;
        p = lineptr;
        while (*p != '\n')
          ++p;
        if ((i = p - lineptr) >= TOKMAX - 1) {
          i = TOKMAX - 1;
        }
        strncpy(tokval, lineptr, i);
        tokval[i] = 0;
        lineptr = p;
        goto nextok_ret;
      }
    }
    FLANG_FALLTHROUGH;
  default:
    switch (MASK(c)) {
    case NOFUNC:
      retval = T_NOFIDENT;
      --tokval;
      p = lineptr;
      goto ident;
    default:
      break;
    }
    *tokval = 0;
    retval = c;
    goto nextok_ret;
  } /* switch */

nextok_ret:

  /* pop macro stack */
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
  int isquest;
  char buffer[133];

again:
  fp = ifp;
  lineptr = linebuf + LINELEN;
  startline = inclstack[inclev].lineno;
  if ((c = getc(fp)) == EOF) {
    if (inclev <= 0)
      return (EOF);
    fclose(ifp);
    gbl.curr_file = inclstack[--inclev].fname;
    ifp = inclstack[inclev].ifp;
    idir.last = inclstack[inclev].path_idx;
    pr_line(gbl.curr_file, inclstack[inclev].lineno);
    strcpy(cur_fname, inclstack[inclev].fname);
    goto again;
  }

  i = 0;
  firstime = 1;
  savestart = start = lineptr;
  look_for_comments = 1;
  have_comment = 0;

joinlines:
  p = start;
  dojoin = 0;
  isquest = 0;
  while (i < LINELEN - 2 && c != EOF && c != '\n') {
    *p++ = c;
    c = getc(fp);
    ++i;
  }
  if (c != '\n' && c != EOF && i >= LINELEN - 2) {
    pperr(237, 4);
    /* NOTREACHED */;
  }
  /* tpr 1359 - don't join lines for fortran */
  /* tpr 1405 - need to join lines if a preprocessor directive */
  if (p > lineptr && lineptr[0] == '#' && c == '\n' && p[-1] == '\\') {
    /* join \<return> lines */
    start = p - 1;
    dojoin = 1;
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
    if (j > 132 - 8) {
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

static void
pbchar(int c)
{
  if (c == EOF)
    return;
  look_for_comments = 0;
  if (lineptr <= linebuf)
    pperr(255, 4);
  *--lineptr = c;
}

static void
pbstr(char *s)
{
  char *p;
  look_for_comments = 0;
  if ((p = lineptr - strlen(s)) < linebuf)
    pperr(255, 4);
  lineptr = p;
  while (*s)
    *p++ = *s++;
}

static void
mac_push(PPSYM *sp, char *lptr)
{
  /* push macro info. on macstk */
  macstk_top++;
  if (macstk_top <= (MACSTK_MAX - 1)) {
    macstk[macstk_top].msptr = sp - hashrec;
    macstk[macstk_top].sav_lptr = lptr;
  } else {
    macro_recur_check(sp);
    interr("Macro recursion stack size exceeded", startline, 4);
  }
}

static void
popstack(void)
{
  while (macstk_top > -1 && lineptr > macstk[macstk_top].sav_lptr)
    --macstk_top;
}

static void
macro_recur_check(PPSYM *sp)
{
  int i;
  for (i = macstk_top; i >= 0; i--)
    if (macstk[i].msptr == sp - hashrec) {
      pperror(233, &deftab[hashrec[macstk[i].msptr].name], 4);
    }
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
