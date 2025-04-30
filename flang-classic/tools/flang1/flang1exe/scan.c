/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file scan.c

    \brief This file contains the compiler's scanner (lexical analyzer) module.
*/

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "gramtk.h"
#include "error.h"
#include "semant.h"
#include "semstk.h"
#include "scan.h"
#include "ast.h"
#include "ccffinfo.h"
#include "fih.h"
#include "dinit.h"

/*  functions defined in this file:  */

extern void fe_restore_state(void);

static void set_input_form(LOGICAL);
static void get_stmt(void);
static void line_directive(void);
static void get_fn(void);
static int otodi(char *, int, INT *);
static int htodi(char *, int, INT *);
static int hex_to_i(int);
static int btodi(char *, int, INT *);
static int read_card(void);
static void write_card(void);
static int check_pgi_pragma(char *);
static int check_pragma(char *);
static int ic_strncmp(const char *, const char *);
static LOGICAL is_lineblank(char *);
static void crunch(void);
static int classify_smp(void);
static int classify_dec(void);
static int classify_pragma(void);
static int classify_pgi_pragma(void);
static int classify_ac_type(void);
static int classify_pgi_dir(void);
static int classify_kernel_pragma(void);
static void alpha(void);
static int get_id_name(char *, int);
static LOGICAL is_keyword(char *, int, const char *);
static int cmp(const char *, const char *, int *);
static int is_ident(char *);
static int is_digit_string(char *);
static int get_kind_id(char *);
static INT get_kind_value(int);
static int get_cstring(char *, int *);
static int get_fmtstr(char *);
static void get_number(int);
static void get_nondec(int);
static int get_prefixed_int(int);
static void check_ccon(void);
static void do_dot(void);
static void push_include(char *);
static void put_astfil(int, char *, LOGICAL);
static void put_lineno(int);
static void put_include(int, int);
static void realloc_stmtb(void);
static void ff_check_stmtb(void);
static void check_continuation(int);
static LOGICAL is_next_char(char *, int);
static int double_type(char *, int *);
void add_headerfile(const char *, int, int);

/*  external declarations  */

extern void parse_init(void);
extern void p_pragma(char *, int); /* directives */
extern LOGICAL fpp_;

#undef OPENMP
#undef SGIMP
#define OPENMP (flg.smp && !XBIT(69, 1))
#define SGIMP (flg.smp && !XBIT(69, 2))

/*   define various scanner maximum values:
     - maximum number of lines per statement (number of continuations + 1)
     - maximum number of columns allowed in a card
     - maximum include nesting depth
     - longest free form line allowed (fixed form is usually 72 or 132)
     - maximum length allowed for include file pathname
*/

/* for KANJI, the 2 following limits are effectively halved */
#define MAX_COLS 1000
#define CARDB_SIZE (MAX_COLS + 6)

#define INIT_LPS 21
#define MAX_IDEPTH 20
#define MAX_PATHNAME_LEN 4096

/*   define pseudo-characters - integer values equivalent
     to non-printing characters are used:  */

#define CH_STRING 31
#define CH_HOLLERITH 30
#define CH_O 29
#define CH_X 28
#define CH_IMPLP 27
#define CH_NULLSTR 26
#define CH_KSTRING 25 /* kanji string - NC'...' */
#define CH_PRAGMA 24
#define CH_IOLP 23
#define CH_B 22
#define CH_IMPRP 21
#define CH_UNDERSCORE 20
#define CH_FMTSTR 19

/*   define card types returned by read_card: */

#define CT_NONE 0
#define CT_INITIAL 1
#define CT_END 2
#define CT_CONTINUATION 3
#define CT_SMP 4
#define CT_DEC 5
#define CT_COMMENT 6
#define CT_EOF 7
#define CT_DIRECTIVE 8
#define CT_LINE 9
#define CT_PRAGMA 10
#define CT_FIXED 11
#define CT_FREE 12
#define CT_MEM 13
/* parsed pragma: */
#define CT_PPRAGMA 14
#define CT_ACC 15
#define CT_KERNEL 16
#define CT_PGI 17

/*   define sentinel types returned by read_card: */

#define SL_NONE 0
#define SL_HPF 1
#define SL_OMP 2
#define SL_SGI 3
#define SL_MEM 4
#define SL_PGI 6
#define SL_KERNEL 7

/* BIND keyword allowed in function declaration: use these states
   to allow for this
 */
#define B_NONE 0
#define B_FUNC_FOUND 1
#define B_RPAREN_FOUND 2
#define B_RESULT_FOUND 3
#define B_RESULT_RPAREN_FOUND 4

#define ERROR_STOP()       \
  {                        \
    tkntyp = TK_ERRORSTOP; \
    idlen += 5;            \
  }

static int bind_state = B_NONE;

/*   define data local to Scanner module:  */

static FILE *curr_fd; /* file descriptor for current input file */

static int incl_level;   /* current include level. starts at 0.  */
static int incl_stacksz; /* current size of include stack */

typedef struct { /* include-stack contents: */
  FILE *fd;
  int lineno;
  int findex;
  const char *fname;
  LOGICAL list_now;
  int card_type;          /* type of "look-ahead" card, CT_NONE if NA */
  int sentinel;           /* sentinel of 'card_type'.  Value is one of
                           * SL_NONE, SL_OMP, etc.  */
  char *first_char;       /* pointer to first character in look ahead */
  LOGICAL eof_flag;       /* current eof flag */
  LOGICAL is_freeform;    /* current fixed-/free- form state */
  char cardb[CARDB_SIZE]; /* contents of look ahead */
} ISTACK;

static ISTACK *incl_stack = NULL;

static int hdr_level;   /* current include level. starts at 0.  */
static int hdr_stacksz; /* current size of include stack */

typedef struct { /* include-stack contents: */
  int lineno;
  int findex;
  const char *fname;
} HDRSTACK;

static HDRSTACK *hdr_stack = NULL;

static int curr_line;          /* current source input file line number */
static LOGICAL first_line;     /* line is first line read */
static char cardb[CARDB_SIZE]; /* buffer containing last card read
                                * in. text terminated by newline
                                * character. */
static char save_extend_ch;    /* char overwritten @ pos flg.extend_source */
static LOGICAL list_now;       /* TRUE if source lines currently being
                                * written */
static char printbuff[CARDB_SIZE]; /* holds card if generating listing */
static char *first_char;           /* pointer into cardb to first character of
                                    * the card following label field.  Typically
                                    * points to 7th char, but may point earlier
                                    * if tabs or "&" in column 1 used. */
static int card_type;              /* type of card currently in cardb.  Value is
                                    * one of CT_INITIAL, CT_END, etc. */
static int sentinel;               /* sentinel of 'card_type'.  Value is one of
                                    * SL_NONE, SL_OMP, etc.  */
static char *stmtb;                /* buffer containing current Fortran stmt
                                    * terminated  by NULL. */
static char *stmtbefore = NULL;    /* 'stmtb' before crunch */
static char *stmtbafter = NULL;    /* 'stmtb' after crunch */
static short *last_char = NULL;    /* position in stmb of the last char for each
                                  * line */
static int card_count; /* number of cards making up the current stmt */
static int max_card;   /* maximum number of cards read in for any
                        * Fortran stmt */
static char *currc;    /* pointer into stmtb to current position */
static char *eos;      /* pointer into stmtb of last character */
static int leadCount;  /* number of leading spaces in current statement */
static int currCol;    /* If > 0, represents source column of current token */

static char *tkbuf = NULL; /* buffer used when tokens are read from file
                            * during _read_token(). */
static int tkbuf_sz;       /* size of tkbuf */

static int directive_sz; /* size of the directive string buffer */
static int options_sz;   /* size of the options string buffer */

static LOGICAL scnerrfg, /* lexical error found in this statement */
    exp_comma,           /* exposed (unparenthesized) comma in stmt */
    exp_equal,           /* exposed equal sign in current stmt */
    exp_ptr_assign,      /* exposed pointer assign ('=>') in stmt */
    exp_attr,            /* exposed attribute syntax (::) in stmt */
    follow_attr,         /* this is after the attribute (::) in stmt */
    exp_ac,              /* this is after the attribute ((/) in stmt */
    exp_dtvlist,         /* this is after the attribute DT( or DT'..'( */
    integer_only;        /* integer constants only - no real */

static LOGICAL par1_attr; /* '::' enclosed within single level parens */

static LOGICAL is_smp;     /* current statement is an SMP directive */
static LOGICAL is_sgi;     /* current statement is an sgi SMP directive
                            * (is_smp is set as well).
                            */
static LOGICAL is_dec;     /* current statement is a DEC directive */
static LOGICAL is_mem;     /* current statement is a mem directive */
static LOGICAL is_ppragma; /* current statement is a parsed pragma/directive */
static LOGICAL is_pgi; /* current statement is a pgi directive */
static bool is_doconcurrent; /* current statement is a do concurrent stmt */
static LOGICAL is_kernel; /* current statement is a parsed kernel directive */
static LOGICAL long_pragma_candidate; /* current statement may be a
                                       * long directive/pragma */
static int scmode;        /* scan mode - used to interpret alpha tokens
                           * Possible states and values are: */
#define SCM_FIRST 1
#define SCM_IDENT 2
#define SCM_FORMAT 3
#define SCM_IMPLICIT 4
#define SCM_FUNCTION 5
#define SCM_IO 6
#define SCM_TO 7
#define SCM_IF 8
#define SCM_DOLAB 9
#define SCM_GOTO 10
#define SCM_DONEXT 11
#define SCM_LOCALITY 12
#define SCM_ALLOC 13
#define SCM_ID_ATTR 14
#define SCM_NEXTIDENT 15 /* next exposed id is as if it begins a statement */
#define SCM_INTERFACE 16
#define SCM_OPERATOR 17 /* next id (presumably enclosed in '.'s) is a
                         * user-defined or named intrinsic operator */
#define SCM_LOOKFOR_OPERATOR 18 /* next id may be word 'operator' */
#define SCM_PAR 19
#define SCM_ACCEL 20
#define SCM_BIND 21 /* next id is keyword bind */
#define SCM_PROCEDURE 22
#define SCM_KERNEL 23
#define SCM_GENERIC 24
#define SCM_TYPEIS 25
#define SCM_DEFINED_IO 26
#define SCM_CHEVRON 27

static int par_depth;            /* current parentheses nesting depth */
static LOGICAL past_equal;       /* set if past the equal sign */
static LOGICAL reset_past_equal; /* set if past_equal must be reset */
static int acb_depth;            /* current 'array constructor begin' depth */
static int tkntyp;               /* token to be returned by get_token.  Value
                                  * is one of the terminal symbols (TK_XXX) of
                                  * the grammar.  */
static INT tknval;               /* additional info on token returned also to
                                  * parser. */
static int body_len;             /* number of characters in statement portion
                                  * of card - either 67 or 127
                                  * (flg.extend_source - 5) */
static LOGICAL no_crunch;        /* compiler directives (pragmas) can't be
                                  * crunched */
static LOGICAL in_include;       /* TRUE when statement in get_stmt() is read
                                  * from an include file. */
static int kind_id;              /* ST_PARAM if kind parammeter found by
                                  * get_kind_id() */
static LOGICAL seen_implp;       /* TRUE if the left paren surrounding the
                                  * IMPLICIT range was seen */
static LOGICAL is_freeform;      /* source form is freeform */
static LOGICAL sig_blanks;       /* blanks are significant */

/*  free form handling  */

static void ff_get_stmt(void);
static void ff_prescan(void);
static void ff_chk_pragma(char *);
static void ff_get_noncomment(char *);
static int ff_read_card(void);
static int ff_get_label(char *);

static struct {
  char *cavail;
  char *outptr; /* last previous char put into crunched stmt */
  char *amper_ptr;
  int last_char; /* last char position of current card */
} ff_state;

/*  switchable fixed or free form handling  */

static void (*p_get_stmt)(void) = get_stmt;
static int (*p_read_card)(void) = read_card;

/*  switchable get_token */

static int _get_token(INT *);
static void ill_char(int);
static void _write_token(int, INT);
static int _read_token(INT *);
static int (*p_get_token[])(INT *) = {_get_token, _read_token};

#include "kwddf.h"

static void init_ktable(KTABLE *);
static int keyword(char *, KTABLE *, int *, LOGICAL);
static int keyword_idx; /* index of KWORD entry found by keyword() */

/* Macro to NULL terminate a substring to error module */

#define CERROR(a, b, c, d, e, f) \
  {                              \
    char tmp;                    \
    tmp = *e;                    \
    *e = 0;                      \
    error(a, b, c, d, f);        \
    *e = tmp;                    \
  }

/** \brief Initialize Scanner.  This routine is called once at the beginning
      of execution.
    \param fd file descriptor for main input source file
 */
void
scan_init(FILE *fd)
{
  /*
   * for each set of keywords, determine the first and last keywords
   * beginning with a given letter.
   */
  init_ktable(&normalkw);
  init_ktable(&logicalkw);
  init_ktable(&iokw);
  init_ktable(&formatkw);
  init_ktable(&parallelkw);
  init_ktable(&parbegkw);
  init_ktable(&deckw);
  init_ktable(&pragma_kw);
  init_ktable(&ppragma_kw);
  init_ktable(&kernel_kw);
  init_ktable(&pgi_kw);

  if (XBIT(49, 0x1040000)) {
    /* T3D/T3E or C90 Cray targets */
    ctable['@'] |= _CS; /* allowed in an identifier */
    ctable['L'] |= _HO; /* left-justified, zero-filled Hollerith */
    ctable['R'] |= _HO; /* right-justified, zero-filled Hollerith */
    ctable['l'] |= _HO; /* left-justified, zero-filled Hollerith */
    ctable['r'] |= _HO; /* right-justified, zero-filled Hollerith */
  }
  curr_fd = fd;
  curr_line = 0;
  first_line = TRUE;
  in_include = FALSE;
  gbl.eof_flag = FALSE;
  body_len = flg.extend_source - 5;
  /*
   * Initially, create enough space for 21 lines (1 for the initial card,
   * 19 continuations, and an extra card to delay first reallocation until
   * after the 20th card is read).  Note that for each line, we never copy
   * more than 'MAX_COLS-1' characters into the statement -- read_card()
   * always locates a position after the first (cardb[0]) position.
   * read_card() also terminates a line with respect to the number of columns
   * allowed in a line (flg.extend_source)
   * More space is created as needed in get_stmt.
   */
  max_card = INIT_LPS;
  stmtbefore = sccalloc((BIGUINT64)(max_card * (MAX_COLS - 1) + 1));
  if (stmtbefore == NULL)
    error(7, 4, 0, CNULL, CNULL);
  stmtbafter = sccalloc((BIGUINT64)(max_card * (MAX_COLS - 1) + 1));
  if (stmtbafter == NULL)
    error(7, 4, 0, CNULL, CNULL);
  stmtb = stmtbefore;
  last_char = (short *)sccalloc((BIGUINT64)(max_card * sizeof(short)));
  if (last_char == NULL)
    error(7, 4, 0, CNULL, CNULL);

  incl_level = 0;
  hdr_level = 0;
  incl_stacksz = 2;
  hdr_stacksz = 2;
  NEW(incl_stack, ISTACK, incl_stacksz);
  NEW(hdr_stack, HDRSTACK, hdr_stacksz);

  /* trigger get_stmt call first time get_token is called: */

  currc = NULL;
  leadCount = 0;
  currCol = 0;

#if DEBUG
  if (DBGBIT(4, 1024)) {
    if ((astb.astfil = fopen("foobar", "w+")) == NULL)
      errfatal(5);
  } else
#endif
      if ((astb.astfil = tmpfile()) == NULL)
    errfatal(5);

  put_astfil(FR_SRC, gbl.file_name, TRUE);

  set_input_form(flg.freeform); /* sets is_freeform */

  /* initiate one-card look ahead: */

  list_now = flg.list;
  card_type = (*p_read_card)();
  directive_sz = CARDB_SIZE;
  NEW(scn.directive, char, directive_sz);
  options_sz = CARDB_SIZE;
  NEW(scn.options, char, options_sz);
  tkbuf_sz = CARDB_SIZE << 3;
  NEW(tkbuf, char, tkbuf_sz);
  scn.id.size = 1024;
  NEW(scn.id.name, char, scn.id.size);
  scn.id.avl = 0;
}

/** \brief Lexical or syntactical error found, so re-initialize the Scanner
    so that next Fortran statement will be processed.
 */
void
scan_reset(void)
{
  currc = NULL;
  leadCount = 0;
  currCol = 0;
  scn.end_program_unit = FALSE;
}

/** \brief Compilation is finished - deallocate storage, close files, etc.
 */
void
scan_fini(void)
{
  if (stmtbefore)
    FREE(stmtbefore);
  if (stmtbafter)
    FREE(stmtbafter);
  if (last_char)
    FREE(last_char);
  if (incl_stack)
    FREE(incl_stack);
  FREE(scn.directive);
  FREE(scn.options);
  if (tkbuf)
    FREE(tkbuf);
  FREE(scn.id.name);
  if (astb.astfil)
    fclose(astb.astfil);
}

/*
 * Dynamically switch the form of the input
 */
static void
set_input_form(LOGICAL free)
{
  if (free) {
    /*printf("switching to free @ %d\n", curr_line);*/
    p_get_stmt = ff_get_stmt;
    p_read_card = ff_read_card;
    is_freeform = TRUE;
  } else {
    /*printf("switching to fixed @ %d\n", curr_line);*/
    p_get_stmt = get_stmt;
    p_read_card = read_card;
    is_freeform = FALSE;
  }
}

int
get_token(INT *tknv)
{
  return p_get_token[sem.which_pass](tknv);
}


/*
 * Extracts next token and returns it to Parser.  Reads in new
 * Fortran statement if necessary.  Does some syntactic checking.
 */
static int
_get_token(INT *tknv)
{
  static int lparen;
  tknval = 0;

retry:
  if (currc == NULL) {
    scnerrfg = FALSE;
    put_astfil(FR_STMT, NULL, FALSE);
    stmtb = stmtbefore;
    if (!scn.multiple_stmts) {
      (*p_get_stmt)();
    }
    if (no_crunch) {
      no_crunch = FALSE;
      currc = stmtb;
      sig_blanks = FALSE;
    } else {
      crunch();
      stmtb = stmtbafter;
      if (scnerrfg) {
        parse_init();
        goto retry;
      }
      scn.id.avl = 0;
      currc = stmtb;
      scmode = SCM_FIRST;
      integer_only = FALSE;
      par_depth = 0;
      past_equal = FALSE;
      reset_past_equal = TRUE;
      acb_depth = 0;
      bind_state = B_NONE;
      if (is_smp) {
        if (classify_smp() == 0) {
          currc = NULL;
          goto retry;
        }
        goto ret_token;
      }
      if (is_dec) {
        if (classify_dec() == 0) {
          currc = NULL;
          goto retry;
        }
        goto ret_token;
      }
      if (is_mem) {
        if (classify_pragma() == 0) {
          currc = NULL;
          goto retry;
        }
        goto ret_token;
      }
      if (is_ppragma) {
        if (classify_pgi_pragma() == 0) {
          currc = NULL;
          goto retry;
        }
        goto ret_token;
      }
      if (is_pgi) {
        if (classify_pgi_dir() == 0) {
          currc = NULL;
          goto retry;
        }
        goto ret_token;
      }
      if (is_kernel) {
        if (classify_kernel_pragma() == 0) {
          currc = NULL;
          goto retry;
        }
        goto ret_token;
      }
    }
  }

again:
  switch (*currc++) {

  case ' ':
    goto again;

  case ';': /* statement separator; set flag and ... */
    scn.multiple_stmts = TRUE;
    FLANG_FALLTHROUGH;
  case '!':  /* inline comment character .. treat like end
              * of line character .......   */
  case '\n': /* return end of statement token: */
    currc = NULL;
    tkntyp = TK_EOL;
    goto ret_token;

  case 'a':
  case 'b':
  case 'c':
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
  case 'A':
  case 'B':
  case 'C':
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
    if ((scmode == SCM_IDENT) && (bind_state == B_RESULT_RPAREN_FOUND)) {
      scmode = SCM_FIRST;
    }
    alpha();
    break;

  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
  case '0':
    currc--;
    get_number(0);
    break;

  case CH_X:
    get_nondec(16);
    break;

  case CH_O:
    get_nondec(8);
    break;

  case CH_B:
    get_nondec(2);
    break;

  case '#':
    if (get_prefixed_int(16)) {
      ill_char('#');
      goto retry;
    }
    break;

  case '%': /* %loc or %fill tokens expected,
               else error */
    if (is_ident(currc) == 3 && strncmp(currc, "loc", 3) == 0) {
      currc += 3;
      tkntyp = TK_LOC;
      goto ret_token;
    }
    if (INSIDE_STRUCT && is_ident(currc) == 4 &&
        strncmp(currc, "fill", 4) == 0) {
      currc += 4;
      tkntyp = TK_FILL;
      goto ret_token;
    }
    tkntyp = TK_PERCENT;
    goto ret_token;

  case '&':
    tkntyp = TK_AMPERSAND;
    goto ret_token;

  case '(': /* check for complex constant, else just
             * paren: */
    if (*currc == '/') {
      if (par_depth == 0 && !past_equal && scmode != SCM_FORMAT &&
          scmode != SCM_PAR) {
        /* (/.../) can only occur inside () or on RHS */
        /* scmode = SCM_OPERATOR; */
      } else if (scmode != SCM_FORMAT && scmode != SCM_OPERATOR &&
                 scmode != SCM_PAR) {
        par_depth++;
        acb_depth++;
        currc++;
        tkntyp = TK_ACB;
        if (classify_ac_type()) {
          exp_ac = 1;
          lparen = 0;
        }
        goto ret_token;
      }
      lparen = 0;
    } else {
      if (tkntyp == TK_DT || exp_dtvlist == 1) {
        exp_dtvlist = 1;
        tkntyp = TK_DLP;
        par_depth++;
        goto ret_token;
      }
      lparen = 1;
    }
    check_ccon();
    break;

  case '[':
    tkntyp = TK_ACB;
    if (classify_ac_type()) {
      exp_ac = 1;
    }
    lparen = 0;
    acb_depth++;
    goto ret_token;
  case ']':
    tkntyp = TK_ACE;
    acb_depth--;
    goto ret_token;

  case ')': /* return right paren */
    par_depth--;
    if (par_depth == 0) {
      if (scmode == SCM_IO)
        scmode = SCM_IDENT;
      else if (scmode == SCM_NEXTIDENT)
        scmode = SCM_FIRST;
      else if (follow_attr)
        scmode = SCM_LOOKFOR_OPERATOR;
      else if (is_doconcurrent)
        scmode = SCM_LOCALITY;
    }
    tkntyp = TK_RPAREN;
    if (bind_state == B_FUNC_FOUND) {
      bind_state = B_RPAREN_FOUND;
    } else if (bind_state == B_RESULT_FOUND) {
      bind_state = B_RESULT_RPAREN_FOUND;
    }
    if (exp_dtvlist) {
      exp_dtvlist = 0;
    } else {
      lparen = 0;
    }
    goto ret_token;

  case '*': /* return * or ** token: */
    if (*currc == '*') {
      currc++;
      tkntyp = TK_EXPON;
    } else
      tkntyp = TK_STAR;
    goto ret_token;

  case '+': /* return plus token: */
    tkntyp = TK_PLUS;
    goto ret_token;
  case ',': /* return comma token: */
    if (par_depth == 0) {
      /* exposed comma seen */
      if (reset_past_equal)
        past_equal = FALSE;
      if (scmode == SCM_ID_ATTR)
        /* if expecting an id within the context of ::, next id is
         * an attributed keyword
         */
        scmode = SCM_FIRST;
      else if (follow_attr)
        scmode = SCM_LOOKFOR_OPERATOR;
    }
    if ((scmode == SCM_IDENT) && (bind_state == B_RPAREN_FOUND)) {
      scmode = SCM_FIRST;
    }
    tkntyp = TK_COMMA;
    goto ret_token;

  case '-': /* return minus sign token: */
    tkntyp = TK_MINUS;
    goto ret_token;

  case '.': /* return keyword enclosed in dots, real
             * constant, or just dot token:        */
    do_dot();
    break;
  case '/': /* slash, concatenate, or not equal token: */
    if (scmode != SCM_FORMAT) {
      if (*currc == '/') {
        currc++;
        tkntyp = TK_CONCAT;
        goto ret_token;
      }
      if (acb_depth && *currc == ')' && scmode != SCM_OPERATOR) {
        par_depth--;
        acb_depth--;
        currc++;
        tkntyp = TK_ACE;
        goto ret_token;
      }
    }
    if (*currc == '=') {
      currc++;
      tkntyp = TK_NE;
      tknval = ('/' << 8) | '=';
    } else
      tkntyp = TK_SLASH;
    goto ret_token;

  case ':': /* return colon or coloncolon token: */
    if (*currc == ':') {
      if (acb_depth > 0 && exp_ac && !lparen) {
        currc++;
        tkntyp = TK_COLONCOLON;
        exp_ac = 0;
        integer_only = false;
        goto ret_token;
      }
      if (par_depth == 0 && exp_attr) {
        currc++;
        tkntyp = TK_COLONCOLON;
        exp_attr = false;
        follow_attr = true;
        if (scmode != SCM_GENERIC)
          scmode = SCM_LOOKFOR_OPERATOR;
        goto ret_token;
      }
      if (par1_attr && par_depth == 1 &&
          (scmode == SCM_ALLOC || is_doconcurrent || scn.stmtyp == TK_FORALL)) {
        currc++;
        tkntyp = TK_COLONCOLON;
        integer_only = false;
        par1_attr = false;
        goto ret_token;
      }
    }
    tkntyp = TK_COLON;
    if (scn.stmtyp == TK_USE) {
      scmode = SCM_LOOKFOR_OPERATOR;
      follow_attr = true;
    }
    goto ret_token;

  case '=': /* return equals sign or eq compar. token */
    if (*currc == '=') {
      currc++;
      tkntyp = TK_EQ;
      tknval = ('=' << 8) | '=';
    } else if (*currc == '>') {
      currc++;
      tkntyp = TK_RENAME;
      past_equal = TRUE;
    } else {
      tkntyp = TK_EQUALS;
      past_equal = TRUE;
    }
    goto ret_token;

  case '<': /* less than or less than or equal */
    if (*currc == '=') {
      currc++;
      tkntyp = TK_LE;
      tknval = ('<' << 8) | '=';
    } else if (*currc == '>') {
      if (flg.standard)
        error(170, 2, gbl.lineno, "<> should be /=", CNULL);
      currc++;
      tkntyp = TK_NE;
      tknval = ('<' << 8) | '>';
    }
    else {
      tkntyp = TK_LT;
      tknval = '<';
    }
    goto ret_token;

  case '>': /* greater than or greater than or equal */
    if (*currc == '=') {
      currc++;
      tkntyp = TK_GE;
      tknval = ('>' << 8) | '=';
    }
    else {
      tkntyp = TK_GT;
      tknval = '>';
    }
    goto ret_token;

#undef MKC
#undef MERGE
#define MKC(cp) ((*(cp)) & 0xFF)
#define MERGE(cp) \
  ((MKC(cp) << 24) | (MKC(cp + 1) << 16) | (MKC(cp + 2) << 8) | (MKC(cp + 3)))
  case CH_HOLLERITH:
    tknval = MERGE(currc);
    currc += 4;
    tkntyp = TK_HOLLERITH;
    goto ret_token;
  case CH_STRING:
    tknval = MERGE(currc);
    currc += 4;
    if (*currc == '(' && tkntyp == TK_DT) {
      exp_dtvlist = 1;
    }
    tkntyp = TK_STRING;
    goto ret_token;
  case CH_NULLSTR:
    tknval = getstring("", 0);
    tkntyp = TK_STRING;
    goto ret_token;
  case CH_KSTRING:
    tknval = MERGE(currc);
    currc += 4;
    tkntyp = TK_KSTRING;
    goto ret_token;
  case CH_FMTSTR:
    tknval = MERGE(currc);
    currc += 4;
    tkntyp = TK_FMTSTR;
    goto ret_token;

  case CH_PRAGMA:
    /* go off and finish the processing of the compiler directive.
     * when done, need to get the next statement.
     */
    {
      /* null terminate line */
      char *p;
      p = currc;
      while (*p != '\n')
        p++;
      *p = '\0';
    }
    put_astfil(FR_PRAGMA, currc, TRUE);
    if (XBIT(49, 0x1040000) && strncmp(currc, "cray", 4) == 0) {
      int len;
      /* T3D/T3E or C90 Cray targets */
      len = strlen(currc + 4) + 6;
      NEED(len, scn.directive, char, directive_sz, len + CARDB_SIZE);
      strncpy(scn.directive, "CDIR$", 5);
      strcpy(scn.directive + 5, currc + 4);
      *--currc = '\n'; /* next token is TK_EOL */
      tkntyp = TK_DIRECTIVE;
      goto ret_token;
    }
    currc = NULL;
    goto retry;

  case CH_IOLP:
    par_depth++;
    tkntyp = TK_IOLP; /* add as case in _rd_token() */
    goto ret_token;

  case CH_IMPLP:
    par_depth++;
    seen_implp = TRUE;
    tkntyp = TK_IMPLP; /* add as case in _rd_token() */
    goto ret_token;

  case CH_IMPRP:
    par_depth--;
    seen_implp = FALSE;
    tkntyp = TK_RPAREN; /* add as case in _rd_token() */
    if (bind_state == B_FUNC_FOUND) {
      bind_state = B_RPAREN_FOUND;
    }
    goto ret_token;

  case CH_UNDERSCORE:
    tkntyp = TK_UNDERSCORE;
    goto ret_token;

  default: /* illegal character - ignore it: */
    ill_char(*(currc - 1));
    goto retry;
  }

  if (scnerrfg) {
    parse_init();
    goto retry;
  }
ret_token:
  *tknv = tknval;
  _write_token(tkntyp, tknval);
  return (tkntyp);
}

static void
ill_char(int ch)
{
  char ctmp[5];
  int c = ch & 0xFF;

  if (c < ' ' || c > '~')
    sprintf(ctmp, "%2X", c);
  else {
    ctmp[0] = c;
    ctmp[1] = '\0';
  }
  error(25, 2, gbl.lineno, ctmp, CNULL);
}

/*  read one Fortran statement, including up continuations
    into stmtb.  Process directive lines if encountered.  Skip past
    comment lines.  Handle end of files.  Extract labels from initial lines.
    Write lines to source listing.
*/
static void
get_stmt(void)
{
  char *p;
  char *cavail; /* available space ptr into stmtb  */
  int c, outp;
  LOGICAL endflg; /* this stmt is an END statement */
  char lbuff[8];  /* temporarily holds label name */
  int lineno;

  endflg = FALSE;
  card_count = 0;
  cavail = &stmtb[0];
  scn.is_hpf = FALSE;
  is_smp = FALSE;
  is_sgi = FALSE;
  is_dec = FALSE;
  is_mem = FALSE;
  is_ppragma = FALSE;
  is_kernel = FALSE;
  is_doconcurrent = false;
  is_pgi = FALSE;

  do {
  again:
    switch (card_type) {
    case CT_END:
      endflg = TRUE;
      goto initial_card;
    case CT_DEC:
      is_dec = TRUE;
      goto initial_card;
    case CT_MEM:
      is_mem = TRUE;
      goto initial_card;
    case CT_PPRAGMA:
      is_ppragma = TRUE;
      goto initial_card;
    case CT_PGI:
      is_pgi = TRUE;
      goto initial_card;
    case CT_KERNEL:
      is_kernel = TRUE;
      goto initial_card;
    case CT_SMP:
      is_smp = TRUE;
      is_sgi = sentinel == SL_SGI;
      FLANG_FALLTHROUGH;
    case CT_INITIAL:
    initial_card:
      gbl.in_include = in_include;
      put_astfil(curr_line, &printbuff[8], TRUE);
      if (card_count == 0) {
        if (hdr_level == 0)
          fihb.currfindex = gbl.findex = 1;
        else
          fihb.currfindex = gbl.findex = hdr_stack[hdr_level - 1].findex;
        gbl.curr_file = FIH_FULLNAME(gbl.findex);
      }
      card_count = 1;
      put_lineno(curr_line);

      p = first_char;
      while ((*cavail++ = *p++) != '\n')
        ;
      cavail--; /* delete trailing '\n' */
      /* may need to pad line out to 72 chars (less 1st 6): */
      c = p - first_char;
      if (XBIT(125, 4))
        c = kanji_len((unsigned char *)first_char, c);
      while (c < body_len) {
        *cavail++ = ' ';
        c++;
      }
      last_char[0] = cavail - stmtb - 1; /* locate last character of
                                          * this line
                                          */

      /* process label field: */

      outp = 2;
      if (sentinel != SL_SGI) {
        for (p = cardb; p < first_char - 1; p++) {
          c = *p & 0xff;
          if (!isblank(c)) {
            if (isdig(c))
              lbuff[outp++] = c;
            else {
              error(18, 3, curr_line, "field", CNULL);
              break;
            }
          }
        }
      }

      scn.currlab = 0;
      if (outp != 2) {
        atoxi(&lbuff[2], &scn.labno, outp - 2, 10);
        if (scn.labno == 0)
          error(18, 2, curr_line, "0", "- label field ignored");
        else {
          int lab_sptr = getsymf(".L%05ld", (long)scn.labno);
          scn.currlab = declref(lab_sptr, ST_LABEL, 'd');
          if (DEFDG(scn.currlab))
            errlabel(97, 3, curr_line, SYMNAME(lab_sptr), CNULL);
          /* linked list of labels for internal subprograms */
          if (sem.which_pass == 0 && gbl.internal > 1 &&
              SYMLKG(scn.currlab) == NOSYM) {
            SYMLKP(scn.currlab, sem.flabels);
            sem.flabels = scn.currlab;
          }
        }
      }
      break;

    case CT_CONTINUATION:
      check_continuation(curr_line);
      put_astfil(curr_line, &printbuff[8], TRUE);
      if (card_count == 0) {
        error(19, 3, curr_line, CNULL, CNULL);
        break;
      }
      if (++card_count >= max_card) {
        int chars_used;
        chars_used = cavail - stmtb;
        realloc_stmtb();
        cavail = stmtb + chars_used;
        if (stmtb == NULL)
          error(7, 4, 0, CNULL, CNULL);
      }
      if (flg.standard && card_count == 257)
        error(170, 2, curr_line, "more than 255 continuations", CNULL);

      p = first_char;
      while ((*cavail++ = *p++) != '\n')
        ;
      --cavail; /* delete trailing '\n' */
      /* may need to pad line out to 72 chars (less 1st 6): */
      c = p - first_char;
      if (XBIT(125, 4))
        c = kanji_len((unsigned char *)first_char, c);
      while (c < body_len) {
        *cavail++ = ' ';
        c++;
      }

      /* locate the last character position for this line */

      last_char[card_count - 1] = cavail - stmtb - 1;

      break;

    case CT_COMMENT:
      put_astfil(curr_line, &printbuff[8], TRUE);
      break;

    case CT_EOF:
      /* pop include  */
      if (incl_level > 0) {
        const char *save_filenm;

        incl_level--;
        if (incl_stack[incl_level].is_freeform) {
          set_input_form(TRUE);
          incl_level++;
          ff_get_stmt();
          return;
        }
        save_filenm = gbl.curr_file;
        curr_fd = incl_stack[incl_level].fd;
        gbl.findex = incl_stack[incl_level].findex;
        curr_line = incl_stack[incl_level].lineno;
        gbl.curr_file = incl_stack[incl_level].fname;
        list_now = incl_stack[incl_level].list_now;
        gbl.eof_flag = incl_stack[incl_level].eof_flag;
        if (curr_line == 1)
          add_headerfile(gbl.curr_file, curr_line + 1, 0);
        else
          add_headerfile(gbl.curr_file, curr_line, 0);

        put_include(FR_E_INCL, gbl.findex);

        card_type = incl_stack[incl_level].card_type;
        sentinel = incl_stack[incl_level].sentinel;
        if (card_type != CT_NONE) {
          first_char = incl_stack[incl_level].first_char;
          BCOPY(cardb, incl_stack[incl_level].cardb, char, CARDB_SIZE);
          if (card_type != CT_DIRECTIVE)
            write_card();
          if (card_type == CT_EOF && incl_level == 0) {
            if (gbl.currsub || sem.mod_sym) {
              gbl.curr_file = save_filenm;
              sem.mod_cnt = 0;
              sem.mod_sym = 0;
              sem.submod_sym = 0;
              errsev(22);
            }
            finish();
          }
        } else
          card_type = read_card();
        if (incl_level == 0)
          in_include = FALSE;
        if (card_type == CT_EOF && incl_level <= 0)
          errsev(22);
        else
          goto again;
      }
      /* terminate compilation:  */
      if (sem.mod_sym) {
        errsev(22);
        sem.mod_cnt = 0;
        sem.mod_sym = 0;
        sem.submod_sym = 0;
      }
      finish();
      FLANG_FALLTHROUGH;
    case CT_DIRECTIVE:
      put_astfil(curr_line, &printbuff[8], TRUE);
      put_lineno(curr_line);
      /* convert upper case letters to lower:  */
      for (p = &cardb[1]; (c = *p) != ' ' && c != '\n'; ++p)
        if (c >= 'A' && c <= 'Z')
          *p = tolower(c);
      if (strncmp(&cardb[1], "list", 4) == 0)
        list_now = flg.list;
      else if (strncmp(&cardb[1], "nolist", 6) == 0)
        list_now = FALSE;
      else if (strncmp(&cardb[1], "eject", 5) == 0) {
        if (list_now)
          list_page();
      } else if (strncmp(&cardb[1], "insert", 6) == 0)
        push_include(&cardb[8]);
      else /* unrecognized directive:  */
        errsev(20);
      break;

    case CT_LINE:
      lineno = gbl.lineno;
      line_directive();
      card_type = CT_COMMENT;

      break;

    case CT_PRAGMA:
      scn.currlab = 0;
      put_astfil(curr_line, &printbuff[8], TRUE);
      no_crunch = TRUE;
      if (card_count == 0) {
        if (hdr_level == 0)
          fihb.currfindex = gbl.findex = 1;
        else
          fihb.currfindex = gbl.findex = hdr_stack[hdr_level - 1].findex;
        gbl.curr_file = FIH_FULLNAME(gbl.findex);
      }
      card_count = 1;
      put_lineno(curr_line);
      p = first_char;
      *cavail++ = CH_PRAGMA;
      while ((*cavail++ = *p++) != '\n')
        ;
      cavail--;                          /* delete trailing '\n' */
      last_char[0] = cavail - stmtb - 1; /* locate last character of
                                          * this line
                                          */
      card_type = CT_INITIAL;            /* trick rest of processing */
      break;

    case CT_FIXED:
      put_astfil(curr_line, &printbuff[8], TRUE);
      set_input_form(FALSE);
      card_type = CT_COMMENT;
      break;

    case CT_FREE:
      set_input_form(TRUE);
      card_type = CT_COMMENT;
      ff_get_stmt();
      return;

    default:
      interr("get_stmt: bad ctype", card_type, 4);
    }
    /* start new listing page if at END, then read new card: */

    if (flg.list && card_type <= CT_COMMENT) {
      if (list_now)
        list_line(printbuff);
    }
#if DEBUG
    if (DBGBIT(4, 2))
      fprintf(gbl.dbgfil, "line(%4d) %s", curr_line, cardb);
#endif
    card_type = read_card();
    if (endflg) {
      if (card_type == CT_CONTINUATION)
        endflg = FALSE;
      else if (flg.list)
        list_page();
    }

  } while (!endflg &&
           (cavail == stmtb || card_type == CT_CONTINUATION ||
            card_type == CT_COMMENT || card_type == CT_LINE /* tpr 533 */
            ));
  *cavail = '\n';
  if (scn.currlab) {
    put_astfil(FR_LABEL, NULL, FALSE);
    put_astfil(scn.labno, NULL, FALSE);
  }
}

void
add_headerfile(const char *fname_buff, int cl, int includedir)
{
  if (!XBIT(120, 0x4000000)) {
    if (hdr_level == 0) {
      NEED(hdr_level + 1, hdr_stack, HDRSTACK, hdr_stacksz, hdr_level + 3);

      /* need to add original source file first */
      if (in_include) {
        gbl.findex = hdr_stack[hdr_level].findex = 1;
        fihb.currfindex = gbl.findex;
        FIH_PARENT(hdr_stack[hdr_level].findex) = 0;
        hdr_stack[hdr_level].lineno = 1;
        hdr_stack[hdr_level].fname = FIH_FULLNAME(gbl.findex);
        hdr_level++;
      } else {
        /* original source file */
        if (strcmp(FIH_FULLNAME(1), fname_buff) == 0) {
          gbl.findex = hdr_stack[hdr_level].findex = 1;
        } else {
          gbl.findex = hdr_stack[hdr_level].findex =
              addfile(fname_buff, 0, 0, 1, gbl.lineno, 1, hdr_level);
        }
        fihb.currfindex = gbl.findex;
        FIH_PARENT(hdr_stack[hdr_level].findex) = 0;
        hdr_stack[hdr_level].lineno = 1;
        hdr_stack[hdr_level].fname = FIH_FULLNAME(gbl.findex);
        put_include(FR_B_HDR, hdr_stack[hdr_level].findex);
        hdr_level++;
        return;
      }
    }
    if (strcmp(fname_buff, hdr_stack[hdr_level - 1].fname) == 0) {

      ; /* same file */
    } else if (hdr_level > 1 &&
               strcmp(fname_buff, hdr_stack[hdr_level - 2].fname) == 0 &&
               cl != 1) {
      /* must not do hdr_level == 0, it is original source file. */
      hdr_level--;
      put_include(FR_E_HDR, hdr_stack[hdr_level - 1].findex);
      /* make sure hdr_level never gets back down to 0 */
      if (hdr_level == 0)
        ++hdr_level;
    } else {
      NEED(hdr_level + 1, hdr_stack, HDRSTACK, hdr_stacksz, hdr_level + 3);
      hdr_stack[hdr_level].findex =
          addfile(fname_buff, 0, 0, 1, includedir ? curr_line : gbl.lineno,
                  cl - 1, hdr_level);
      hdr_stack[hdr_level].lineno = cl;
      hdr_stack[hdr_level].fname = FIH_FULLNAME(hdr_stack[hdr_level].findex);
      FIH_PARENT(hdr_stack[hdr_level].findex) = hdr_stack[hdr_level - 1].findex;
      put_include(FR_B_HDR, hdr_stack[hdr_level].findex);
      hdr_level++;
    }
  }
}

static void
line_directive(void)
{
  static char fname_buff[CARDB_SIZE];
  char *p;
  char *to;
  int cl;
  const char *tmp_ptr;

  /*
   * The syntax of a line directive is:
   * #<c><line #>[<d>][<file name>]
   * where:
   * #           appears in column 1,
   * <c>         is one or more blank characters, a '-', or '+',
   * <d>         is one or more blank characters,
   * <file name> is a quoted string.
   * If <file name> is not quoted string, it's assumed to be commentary.
   */

  /* preprocessor communicates to the scanner where or not the
   * ensuing line(s) came from an include file.
   */
  if (cardb[1] == '-') {
    in_include = FALSE;
    cardb[1] = ' ';
  } else if (cardb[1] == '+') {
    in_include = TRUE;
    cardb[1] = ' ';
  }
  write_card();
  put_astfil(curr_line, &printbuff[8], TRUE);
  if (!isblank(cardb[1]))
    goto ill_line;
  p = cardb + 2;
  while (isblank(*p)) /* skip blank characters */
    ++p;
  if (!isdig(*p))
    goto ill_line;
  cl = 0;
  for (; isdig(*p); ++p)
    cl = (10 * cl) + (*p - '0');
  while (isblank(*p)) /* skip blank characters */
    ++p;
  if (*p == '"') {
    cardb[CARDB_SIZE - 1] = '"'; /* limit length of file name */
    to = fname_buff;
    while (*++p != '"') {
      if (*p == '\n')
        goto ill_line;
      *to++ = *p;
    }
    if (to == fname_buff) /* check for empty string */
      *to++ = ' ';
    *to = '\0';
    add_headerfile(fname_buff, cl, 1);
  }

  curr_line = cl - 1;
  return;
ill_line:
  tmp_ptr = gbl.curr_file;
  if (hdr_level)
    gbl.curr_file = hdr_stack[hdr_level - 1].fname;
  error(21, 3, curr_line, CNULL, CNULL);
  gbl.curr_file = tmp_ptr;
}

static void
get_fn(void)
{
  /*
   * This is a hack to get the name of the file that was preprocessed
   * and saved to a file which is now being compiled.  We want the name
   * of the original file to show up as the file being debugged.
   *
   * The expected syntax of the line directive in this situation is:
   * #<c><line #>[<d>][<file name>]
   * where:
   * #           appears in column 1,
   * <c>         a blank
   * <line #>    1
   * <d>         is a blank.
   * <file name> is a quoted string.
   */
  char *p;
  int len;

  if (XBIT(120, 0x40000))
    return;
  if (cardb[1] != ' ' || cardb[2] != '1' || cardb[3] != ' ')
    return;
  p = &cardb[4];
  if (*p == '"') {
    while (*++p != '"') {
      if (*p == '\0' || *p == '\n')
        return;
    }
    len = p - &cardb[5];
    if (len <= 0)
      return;
    p = (char *)getitem(8, len + 1);
    strncpy(p, &cardb[5], len);
    p[len] = '\0';
    gbl.fn = p;
  }
}

static char *
_readln(int mx_len, LOGICAL len_err)
{
  int c;
  int i;
  char *p, *q;

  long_pragma_candidate = FALSE;
  if ((c = getc(curr_fd)) == EOF) {
    if (incl_level == 0) {
      gbl.eof_flag = TRUE;
    }
    fclose(curr_fd);
    return NULL;
  }
  curr_line++;
  i = 0;
  p = cardb - 1;
  while (c != '\n') {
    i++;
    if (i > mx_len) {
      if (len_err) {
        for (q = cardb; isblank(*q) && q < p; ++q)
          ;
        if (flg.standard || *q != '!')
          /* Flag non-comments; flag any statement under -Mstandard. */
          error(285, 3, curr_line, CNULL, CNULL);
        else
          /* Comments might be pragmas; set up to check those later. */
          long_pragma_candidate = TRUE;
      }
      /* skip to the end-of-line */
      while (1) {
        c = getc(curr_fd);
        if (c == '\n' || c == EOF)
          break;
      }
      break;
    }
    *++p = c;
    c = getc(curr_fd);
    if (c == EOF) {
      /* this can't be the first character of the line; this case
       * is detected as end-of-file (see above).
       */
      break;
    }
  }
  p[1] = '\n';
  p[2] = '\0';
  return p; /* last position */
}

/*  read one input line into cardb, and determine its type
    (card_type) and determine first character following the
    label field (first_char).
*/
static int
read_card(void)
{
  int c;
  int i;
  char *p; /* pointer into cardb */
  LOGICAL tab_seen;
  int ct_init;
  const char *tmp_ptr;

  assert(!gbl.eof_flag, "read_card:err", gbl.eof_flag, 4);
  sentinel = SL_NONE;

  p = _readln(CARDB_SIZE - 2, FALSE);
  if (p == NULL)
    return CT_EOF;

  ct_init = CT_INITIAL; /* initial card type */
  if (*cardb == '#') {
    if (first_line && !fpp_) {
      get_fn();
    }
    first_line = FALSE;
    return CT_LINE;
  }
  first_line = FALSE;
  save_extend_ch = cardb[flg.extend_source]; /* just in case it's needed */
  cardb[flg.extend_source] = '\n'; /* ensure that newline char marks end
                                    * of buff */
  c = cardb[0];
  if (c == '%')
    return CT_DIRECTIVE;
  if (c == '$') /* APFTN64 style of directives */
    return CT_DIRECTIVE;
  write_card();
  first_char = &cardb[6]; /* default first character of stmt */
  if (c == 'c' || c == 'C' || c == '*' || c == '!') {
/* possible compiler directive. these directives begin with (upper
 * or lower case):
 *     c$pragma
 *     cpgi$  cvd$  cdir$ !cdir
 * to check for a directive, all that's done is to copy at most N
 * characters after the leading 'c', where N is the max length of
 * the allowable prefixes, converting to lower case if necessary.
 * if the prefix matches one of the above, a special card type
 * is returned.   NOTE: can't process the directive now since
 * this card represents the read-ahead ---- NEED to ensure that
 * semantic actions are performed.
 */
#define MAX_DIRLEN 4
    char b[MAX_DIRLEN + 1], cc;

    /* sun's c$pragma is separate from those whose prefixes end with $ */
    if (cardb[1] == '$' && (cardb[2] == 'P' || cardb[2] == 'p') &&
        (cardb[3] == 'R' || cardb[3] == 'r') &&
        (cardb[4] == 'A' || cardb[4] == 'a') &&
        (cardb[5] == 'G' || cardb[5] == 'g') &&
        (cardb[6] == 'M' || cardb[6] == 'm') &&
        (cardb[7] == 'A' || cardb[7] == 'a')) {
      /*
       * communicate to p_pragma() that this is a sun directive.
       * do so by prepending the substring beginning with the
       * first character after "pragma"  with "sun".
       */
      first_char = &cardb[5];
      strncpy(first_char, "sun", 3);
      return (CT_PRAGMA);
    }

    if (OPENMP && /* c$smp, c$omp - smp directive sentinel */
        cardb[1] == '$' && (cardb[2] == 'S' || cardb[2] == 's' ||
                            cardb[2] == 'O' || cardb[2] == 'o') &&
        (cardb[3] == 'M' || cardb[3] == 'm') &&
        (cardb[4] == 'P' || cardb[4] == 'p')) {
      strncpy(cardb, "     ", 5);
      ct_init = CT_SMP; /* change initial card type */
      sentinel = SL_OMP;
      goto bl_firstchar;
    }
    /* SGI c$doacross, c$& */
    if (SGIMP && cardb[1] == '$' && (cardb[2] == 'D' || cardb[2] == 'd') &&
        (cardb[3] == 'O' || cardb[3] == 'o') &&
        (cardb[4] == 'A' || cardb[4] == 'a') &&
        (cardb[5] == 'C' || cardb[5] == 'c') &&
        (cardb[6] == 'R' || cardb[6] == 'r') &&
        (cardb[7] == 'O' || cardb[7] == 'o') &&
        (cardb[8] == 'S' || cardb[8] == 's') &&
        (cardb[9] == 'S' || cardb[9] == 's')) {
      sentinel = SL_SGI;
      first_char = &cardb[2];
      return CT_SMP;
    }
    if (SGIMP && cardb[1] == '$' && cardb[2] == '&') {
      if (!is_sgi)
        /* current statement is not an SGI smp statement; just
         * treat as a comment.
         */
        return CT_COMMENT;
      sentinel = SL_SGI;
      first_char = &cardb[3];
      return CT_CONTINUATION;
    }
    /* OpenMP conditional compilation sentinels */
    if (OPENMP && cardb[1] == '$' && (iswhite(cardb[2]) || isdigit(cardb[2]))) {
      c = cardb[0] = cardb[1] = ' ';
      goto bl_firstchar;
    }
    /* Miscellaneous directives which are parsed */
    if (XBIT(59, 0x4) && /* c$mem - mem directive sentinel */
        cardb[1] == '$' && (cardb[2] == 'M' || cardb[2] == 'm') &&
        (cardb[3] == 'E' || cardb[3] == 'e') &&
        (cardb[4] == 'M' || cardb[4] == 'm')) {
      strncpy(cardb, "     ", 5);
      ct_init = CT_MEM; /* change initial card type */
      sentinel = SL_MEM;
      goto bl_firstchar;
    }
    if (XBIT_PCAST && /* c$pgi - alternate pgi accelerator directive sentinel */
        cardb[1] == '$' && (cardb[2] == 'P' || cardb[2] == 'p') &&
        (cardb[3] == 'G' || cardb[3] == 'g') &&
        (cardb[4] == 'I' || cardb[4] == 'i')) {
      strncpy(cardb, "     ", 5);
      ct_init = CT_PGI; /* change initial card type */
      sentinel = SL_PGI;
      goto bl_firstchar;
    }
    if (XBIT(137, 1) && /* c$cuf - cuda kernel directive sentinel */
        cardb[1] == '$' && (cardb[2] == 'C' || cardb[2] == 'c') &&
        (cardb[3] == 'U' || cardb[3] == 'u') &&
        (cardb[4] == 'F' || cardb[4] == 'f')) {
      strncpy(cardb, "     ", 5);
      ct_init = CT_KERNEL; /* change initial card type */
      sentinel = SL_KERNEL;
      goto bl_firstchar;
    }
    if (XBIT(137, 1) && /* !@cuf - cuda kernel conditional compilation */
        cardb[1] == '@' && (cardb[2] == 'C' || cardb[2] == 'c') &&
        (cardb[3] == 'U' || cardb[3] == 'u') &&
        (cardb[4] == 'F' || cardb[4] == 'f') && iswhite(cardb[5])) {
      strncpy(cardb, "     ", 5);
      goto bl_firstchar;
    }

    i = 1;
    p = b;
    while (TRUE) {
      cc = cardb[i];
      if (cc >= 'A' && cc <= 'Z')
        *p = tolower(cc);
      else
        *p = cc;
      p++;
      if (i >= MAX_DIRLEN || cc == '$' || cc == '\n')
        break;
      i++;
    }
    if (cc == '$') {
      *p = '\0';
      if (strncmp(b, "pgi$", 4) == 0 || strncmp(b, "vd$", 3) == 0) {
        /* for these directives, point to first character after the
         * '$'.
         */
        first_char = &cardb[i + 1];
        if (check_pgi_pragma(first_char) == CT_PPRAGMA) {
          strncpy(cardb, "    ", 4);
          if (b[0] == 'p')
            cardb[4] = ' ';
          return CT_PPRAGMA;
        }
        return CT_PRAGMA;
      }
      if (strncmp(b, "dir$", 4) == 0) {
        /*
         * communicate to p_pragma() that this is a cray directive.
         * do so by prepending the substring beginning with the
         * first character after the '$' with "cray".
         */
        first_char = &cardb[1];
        strncpy(first_char, "cray", 4);
        i = check_pragma(first_char + 4);
        if (i == CT_PPRAGMA) {
          strncpy(cardb, "     ", 5);
        }
        return i;
      }
      if (XBIT(124, 0x100) && strncmp(b, "exe$", 4) == 0) {
        c = cardb[0] = cardb[1] = cardb[2] = cardb[3] = cardb[4] = ' ';
        goto bl_firstchar;
      }
#if defined(TARGET_WIN)
      if (strncmp(b, "dec$", 4) == 0) {
        c = cardb[0] = cardb[1] = cardb[2] = cardb[3] = cardb[4] = ' ';
        ct_init = CT_DEC; /* change initial card type */
        goto bl_firstchar;
      }
      if (strncmp(b, "ms$", 3) == 0) {
        /* in fixed-form, !ms$ cannot be continued, so just immediately
         * return the card type.
         */
        c = cardb[0] = cardb[1] = cardb[2] = cardb[3] = ' ';
        first_char = &cardb[4];
        return CT_DEC;
      }
#endif
    }
    return (CT_COMMENT);
  }
  if (c == '\n')
    return (CT_COMMENT);

  if (c == 'd' || c == 'D') {
    if (!flg.dlines)
      return (CT_COMMENT);
    c = cardb[0] = ' ';
  }
  if (c == '&') {
    first_char = &cardb[1];
    cardb[0] = ' ';
    return (CT_CONTINUATION);
  }
bl_firstchar:

  /* check for a totally empty line or a line with just a comment */
  tab_seen = FALSE;
  for (p = cardb; isblank(*p); p++)
    if (*p == '\t')
      tab_seen = TRUE;
  ;
  if (*p == '\n')
    return (CT_COMMENT);
  /*
   * When the first non-white character is a ! then it is a comment
   * if a tab has been seen or if no tab seen then the ! must not
   * be in column 6, the continuation column.
   */
  if (*p == '!' && (tab_seen || p != &cardb[5])) {
    return (CT_COMMENT);
  }

  /* check first 6 character positions for tab or newline char: */

  for (i = 0; i < 6; i++) {
    if (cardb[i] == '\t') {
      first_char = &cardb[i + 1];
      cardb[i] = ' ';
      if ((c = *first_char) >= '1' && c <= '9') {
        first_char = &cardb[i + 2]; /* vms tab-digit continuation */
        return (CT_CONTINUATION);
      }
      break;
    } else if (cardb[i] == '\n') {
      first_char = &cardb[i + 1];
      cardb[i + 1] = '\n';
      return (ct_init);
    }
  }

  /* check for normal type of continuation card: */
  /* We currently check error for the one we are scanning
     which is fihb.nextfindex */
  c = *(first_char - 1);
  if (c != ' ' && c != '0') {
    tmp_ptr = gbl.curr_file;
    for (p = cardb; p < first_char - 1; p++)
      if (*p != ' ') {
        if (hdr_level)
          gbl.curr_file = hdr_stack[hdr_level - 1].fname;
        error(21, 3, curr_line, CNULL, CNULL);
        gbl.curr_file = tmp_ptr;
        return (CT_COMMENT);
      }
    return (CT_CONTINUATION);
  }

  if (ct_init != CT_INITIAL)
    return (ct_init);

  /* finally- have it narrowed down to initial or end line. */

  /* scan to first non-blank character in stmt part: */
  if (p < first_char)
    for (p = first_char; isblank(*p); p++)
      ;

  if (*p != 'e' && *p != 'E')
    return (ct_init);
  if (*++p != 'n' && *p != 'N')
    return (ct_init);
  if (*++p != 'd' && *p != 'D')
    return (ct_init);
  if (*++p != 'q' && *p != 'Q')
    return (ct_init);

  /*  have a statement which begins with END -- this is the END statement
   *  if what follows are zero or more blanks and/or tabs followed by the
   *  end of line character or ! (inline comment)
   */
  for (++p; isblank(*p); ++p)
    ;
  if (*p != '\n' && *p != '!')
    return (ct_init);

  return (CT_END);
}

/* Construct single source listing line using cardb and curr_line. */
static void
write_card(void)
{
  char *from, *to;
  int max_len;
  int len;

  max_len = sizeof(printbuff) - 1; /* leave room for newline & null */
  sprintf(printbuff, "(%5d)  ", curr_line);
  if (incl_level > 0)
    printbuff[7] = '*';
  len = 8;
  for (from = cardb, to = &printbuff[8]; *from != '\n';) {
    if (++len >= max_len) {
      *to++ = '\0';
      break;
    }
    *to++ = *from++;
  }
  *to = '\0';
}

/*
 * Check whether this is a parsed !pgi$ pragma
 * Return CT_PRAGMA if not; CT_PPRAGMA if so.
 */
static int
check_pgi_pragma(char *beg)
{
  int c;
  int len;
  char *bbeg;
  int tkntyp;

  bbeg = beg;

  while (TRUE) {
    c = *beg;
    if (!iswhite(c))
      break;
    if (c == '\n')
      return CT_PRAGMA;
    beg++;
  }
  len = is_ident(beg);
  if (len == 0)
    return CT_PRAGMA;
  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  tkntyp = keyword(beg, &ppragma_kw, &len, TRUE);
  if (tkntyp)
    return CT_PPRAGMA;

  return CT_PRAGMA;
} /* check_pgi_pragma */

/* Certain directives affect the state of the scanner and need to be
 * processed now rather than by p_pragma().  Examples are the cray
 * directives, fixed and free.
 */
static int
check_pragma(char *beg)
{
  int c;
  int len;
  char *bbeg;

  bbeg = beg;

  while (TRUE) {
    c = *beg;
    if (!iswhite(c))
      break;
    if (c == '\n')
      return CT_PRAGMA;
    beg++;
  }
  len = is_ident(beg);
  if (len == 4) {
    if (ic_strncmp(beg, "free") == 0 && is_lineblank(beg + 4))
      return CT_FREE;
    return CT_PRAGMA;
  }
  if (len == 5) {
    if (ic_strncmp(beg, "fixed") == 0 && is_lineblank(beg + 5))
      return CT_FIXED;
    return CT_PRAGMA;
  }
  if (len == 10) {
    if (ic_strncmp(beg, "distribute") == 0) {
      beg += 10;
      while (TRUE) {
        c = *beg;
        if (!iswhite(c))
          break;
        if (c == '\n')
          return CT_PRAGMA;
        beg++;
      }
      len = is_ident(beg);
      if (len == 5) {
        if (ic_strncmp(beg, "point") == 0 && is_lineblank(beg + 5)) {
          strcpy(bbeg, "distributepoint\n");
          return CT_DEC;
        }
      }
    }
    if (ic_strncmp(beg, "ignore_tkr") == 0) {
      return CT_PPRAGMA;
    }
    return CT_PRAGMA;
  }
  if (len == 15) {
    if (ic_strncmp(beg, "distributepoint") == 0 && is_lineblank(beg + 15)) {
      strcpy(bbeg, "distributepoint\n");
      return CT_DEC;
    }
    return CT_PRAGMA;
  }

  return CT_PRAGMA;
}

/* simple strncmp, ignore case:  str may contain uppercase letters and may
 * not be null-terminated; pat contains only lower case characters and is
 * null-terminated.  length of str is at least the length of pattern.
 */
static int
ic_strncmp(const char *str, const char *pattern)
{
  int n;
  int ch;
  int i;

  n = strlen(pattern);
  for (i = 0; i < n; i++) {
    ch = str[i];
    if (ch >= 'A' && ch <= 'Z')
      ch += ('a' - 'A'); /* to lower case */
    if (ch != pattern[i])
      return (ch - pattern[i]);
  }
  return 0;
}

static LOGICAL
is_lineblank(char *start)
{
  char *p;
  int c;

  for (p = start; (c = *p) != '\n'; p++) {
    if (!iswhite(c))
      return FALSE;
  }
  return TRUE;
}

/*  Prepare one Fortran stmt for scanning.  The contents of the statement
    buffer are crunched from stmtbefore to stmtbafter.
   The following is done:
    1. blanks and tabs are eliminated.
    2. upper case letters converted to lower case, unless -upcase flag.
    3. Hollerith and character constants are extracted and entered into
       the symbol table.  A special marker is put into the crunched
       buffer to indicate the type of token, and the two locations which
       follow contain a symbol table pointer for the constant, which has
       been split.
    4  Non-decimal constants are marked in the crunch buffer.  A special
       marker is put into the buffer and the digits and the ending quote
       are copied to the buffer.
    5. Inline comments are stripped - this is done by moving the input
       pointer to the end of the line containing the '!'.
    6. correct balancing of parentheses is checked.
    7. correct balancing of brackets is checked.
    8. exposed equal sign, exposed comma, and exposed attribute (::) flags
       are set.
    9. extract the label, if present and the input is freeform.

*/
static void
crunch(void)
{
  int c, ctmp;        /* BOBT - better type int than char */
  static char *inptr; /* next char to be processed */
  char *outptr;       /* last previous char put into crunched stmt */
  char *outlim;       /* limit reverse scan for Holleriths */
  int parlev;         /* current parenthesis nesting level */
  int upper_to_lower; /* amount to add to convert u.c. to
                       * l.c. */
  int slshlv;         /* current slash nesting level, = 0 or 1 */
  char *p;            /* pointer into statement buffer */
  int bracket;        /* current bracket nesting level */
  int len;            /* length of character or Hollerith string */
  int sptr;           /* symbol table pointer. */
  int delim;          /* string delimiter ' or " */
  LOGICAL last_char_is_blank;
  int holl_kind; /* kind of Hollerith - 'h', 'l', 'r' */
  char *startid; /* where the first identifer starts */
  LOGICAL in_format;
  LOGICAL last_is_pseudo; /* last char copied is a pseudo char CH_... */

  parlev = slshlv = bracket = 0;
  exp_equal = exp_comma = FALSE;
  exp_ptr_assign = exp_attr = follow_attr = par1_attr = FALSE;
  outptr = outlim = stmtbafter - 1;
  upper_to_lower = 32;
  if (flg.ucase)
    upper_to_lower = 0;
  sig_blanks = scn.is_hpf || is_freeform;
  if (scn.multiple_stmts) {
    scn.multiple_stmts = FALSE;
    len = ff_get_label(inptr);
    inptr += len;
  } else if (is_freeform) {
    len = ff_get_label(stmtb);
    inptr = stmtb + len;
  } else
    inptr = stmtb;

  last_char_is_blank = FALSE;
  last_is_pseudo = FALSE;

  /* pick up the first identifier of the statement */
  for (; (c = *inptr) != '\n'; inptr++) {
    c &= 0xFF; /* only needed because of kanji support */
               /*
                * skip over any blank and non-printing (value less than blank)
                * characters.
                */
    if (!iswhite(c))
      break;
  }
  startid = outptr + 1;
  while (isident(c)) {
    if (isupper(c))
      c += upper_to_lower;
    *++outptr = c;
    c = *++inptr;
    c &= 0xFF;
  }
  in_format = FALSE;
  if (outptr - startid == 5 && strncmp(stmtb, "format", 6) == 0)
    /* 5 may appear to be incorrect, but outptr locates the last character
     * of the identifier.
     */
    in_format = TRUE;
  for (; (c = *inptr) != '\n'; inptr++) {
    last_char_is_blank = FALSE;
    c &= 0xFF; /* only needed because of kanji support */
               /*
                * ignore the blank character, and all non-printing characters whose
                * integer value is less than blank.  This includes tabs.
                */
    if (iswhite(c)) {
      if (sig_blanks) {
        *++outptr = ' '; /* blanks are significant */
        while ((c = *++inptr) != '\n') {
          c &= 0xFF;
          if (c > ' ')
            break;
        }
        inptr--;
        last_char_is_blank = TRUE;
        last_is_pseudo = FALSE;
      }
      continue;
    }
    if (isupper(c))
      c += upper_to_lower;

    if (c == '\'' || c == '"') {
      char *eostr;
      delim = c;
      len = 0;
      for (p = inptr + 1; (c = *p++) != delim;) {
        if (c == '\n')
          goto do_string;
        len++;
      }
      if ((c = *p) == delim) /* watch out for two consecutive */
        goto do_string;      /* quotes */
                             /*
                              * get the first printable character after the quote.
                              */
      eostr = p;
      while (iswhite(c)) {
        if (c == '\n')
          goto chk_mil_const;
        c = *++p;
      }
      if (c == 'x' || c == 'X') {
        *++outptr = CH_X;
        goto copynondec;
      }
      if (c == 'o' || c == 'O') {
        *++outptr = CH_O;
        goto copynondec;
      }
      if (c == 'c' || c == 'C') {
        /* this notation creates a null-terminated (ala C)
         * string. Cleat the 'c', insert the null character.
         */
        *p = ' ';
        *(eostr - 1) = '\0';
        *eostr = delim;
        if (flg.standard)
          error(170, 2, curr_line,
                "c-style termination of character constants ", CNULL);
        goto do_string;
      }
    chk_mil_const:
      /*
       * Check for f90 style of hex, octal, and binary constants:
       *    Z'ddddd', z'ddddd', O'ddddd', o'dddddd', B'ddddd', B"ddddd"
       * Begin by looking at the character immediately preceding the
       * first quote.
       */
      if (outptr == outlim)
        goto do_string;
      c = *outptr;
      if (c == 'z' || c == 'Z') {
        *outptr = CH_X; /* overwrite the 'z' */
        p--;            /* leave the pointer at the ' */
        goto copynondec;
      }
      if (!in_format && (c == 'x' || c == 'X')) {
        *outptr = CH_X; /* overwrite the 'x' */
        p--;            /* leave the pointer at the ' */
        if (flg.standard)
          error(170, 2, gbl.lineno, "X'...' hexadecimal constant notation",
                CNULL);
        goto copynondec;
      }
      if (c == 'o' || c == 'O') {
        *outptr = CH_O; /* overwrite the 'o' */
        p--;
        goto copynondec;
      }
      if (c == 'b' || c == 'B') {
        *outptr = CH_B; /* overwrite the 'b' */
        p--;
        goto copynondec;
      }
      goto do_string;

    copynondec:
      while (len-- >= 0) { /* copy all digits plus end quote -
                            * ignore white spaces */
        c = *++inptr;
        if (iswhite(c))
          continue;
        *++outptr = c;
      }
      inptr = p;
      last_is_pseudo = FALSE;
      continue;
    }
    if (isholl(c)) {
      /* possible Hollerith constant has been found: */

      if (c != 'h' && in_format)
        goto copychar;
      holl_kind = c;
      /* `outlim` is used so that `p` doesn't reverse scan into our
       * tokenized string built thus far.
       */
      for (p = outptr; p > outlim && isdig(*p); p--)
        ;
      if (p == outptr)
        goto copychar;
      ctmp = *p;
      if (iscsym(ctmp))
        goto copychar;
      if (parlev == 0 && slshlv == 0 && !exp_equal && ctmp != ',' &&
          ctmp != ')')
        goto copychar;
      *(outptr + 1) = '\0'; /* limit scan */
      sscanf(p + 1, "%d", &len);
      if (XBIT(125, 4)) {
        int ilen = (stmtb + last_char[card_count - 1]) - inptr;
        /* compute #bytes */
        len = kanji_prefix((unsigned char *)inptr + 1, len, ilen);
      }
      /* check for sufficient chars in line: */
      if ((inptr - stmtb) + len > last_char[card_count - 1]) {
        errwarn(123);
        len = (stmtb + last_char[card_count - 1]) - inptr;
      }
      sptr = getstring(inptr + 1, len);
      sptr = gethollerith(sptr, holl_kind);
      outptr = p + 1; /* point back to first digit */
      *outptr = CH_HOLLERITH;
      *++outptr = (char)((sptr >> 24) & 0xFF);
      *++outptr = (char)((sptr >> 16) & 0xFF);
      *++outptr = (char)((sptr >> 8) & 0xFF);
      *++outptr = (char)(sptr & 0xFF);
      outlim = outptr;
      inptr += len; /* point to last char of Hollerith constant */
      /* (should check that we haven't moved past end of line - '\n') */
      last_is_pseudo = TRUE;
      continue;
    }
    if (c == '!') { /*  strip inline comment  */
      int pos, i;
      char *ppp; /* ptr to pragma stuff if present */

      pos = inptr - stmtb; /* position of '!'	*/

      /* step thru the last character positions for all the lines in
       * this statement beginning with the first line; the line
       * containing '!' will be the first line whose last char position
       * is > or = to the position of '!'.
       * After the line is found, move the input pointer to the end
       * of the line -- inptr is incremented by the main crunch loop.
      */
      for (i = 0; i < card_count; i++) {
        if (last_char[i] >= pos)
          break;
      }
      /*
       * check for sun's inline form of c$pragma.
       *
       */
      ppp = inptr;
      inptr = stmtb + last_char[i];
      if (ppp[1] == '$' && (ppp[2] == 'P' || ppp[2] == 'p') &&
          (ppp[3] == 'R' || ppp[3] == 'r') &&
          (ppp[4] == 'A' || ppp[4] == 'a') &&
          (ppp[5] == 'G' || ppp[5] == 'g') &&
          (ppp[6] == 'M' || ppp[6] == 'm') &&
          (ppp[7] == 'A' || ppp[7] == 'a')) {
        /*
         * communicate to p_pragma() that this is a sun directive.
         * do so by prepending the substring beginning with the
         * first character after "pragma"  with "sun".
         * NOTE: p_pragma expects a terminated line; inptr locates
         *       the end of the line containing "!c$pragma".  In
         *       the next position store a newline.
         */
        char save_ch;
        save_ch = inptr[1];
        inptr[1] = '\n';
        strncpy(&ppp[5], "sun", 3);
        p_pragma(&ppp[5], gbl.lineno);
        inptr[1] = save_ch;
      }
      continue;
    }
    if (c == ';') {
      /*
       * Multiple statements:  save position of next character,
       * terminate current statement with the ';', and check
       * for errors in the current statement.
       */
      inptr++; /* locate the character after ';' */
      /* If line ends with a blank, delete it */
      if (outptr < outlim && !last_is_pseudo && *outptr == ' ')
        outptr--;
      /* WARNING: if the last line in multiple statements may be all
       * blanks, eos will point to the position one before the
       * beginning of stmtb and the first character will be '\n'.
       * This should be ok since the '\n' will cause get_token to
       * move on without the chance of eos being examined.
       */
      eos = outptr;
      *++outptr = ';';
      goto end_of_stmt_checks;
    }

    /* just copy the character after a few state checks */

    if (isalpha(c) || isdig(c))
      goto copychar;
    if (c == '(')
      parlev++;
    else if (c == ')') {
      if (--parlev < 0)
        break;
    } else if (c == '[')
      bracket++;
    else if (c == ']') {
      if (--bracket < 0)
        break;
    } else if (c == '/' && *(inptr + 1) != '=')
      slshlv = 1 - slshlv;
    else if (c == '=' && *(inptr + 1) == '=') {
      *++outptr = '='; /* == token */
      inptr++;
    } else if (c == '=' && *(inptr + 1) == '>') {
      *++outptr = '='; /* => token */
      inptr++;
      c = '>';
      exp_ptr_assign = TRUE;
    } else if (parlev == 0) {
      if (c == '=') {
        if (outptr > stmtb)
          switch (*outptr) { /* check char before '=' */
#if DEBUG
          case '=':
            interr("crunch ==", 3, 2);
            FLANG_FALLTHROUGH;
#endif
          case '<':
          case '>':
          case '/':
            break;
          default:
            exp_equal = TRUE;
          }
        else
          exp_equal = TRUE;
      } else if (c == ',')
        exp_comma = TRUE;
      else if (c == ':' && *(inptr + 1) == ':') {
        exp_attr = TRUE;
      }
    } else if (parlev == 1 && c == ':' && *(inptr + 1) == ':') {
      par1_attr = TRUE;
    }
  copychar:
    *++outptr = c;
    last_is_pseudo = FALSE;
    continue;

  do_string:
    /* a character string constant has been found: */
    sptr = get_cstring(inptr, &len);

    if (len) {
      /* the character after the quoted string may indicate that the
       * string is actually a Hollerith constant. However, this syntatical
       * form is not allowed in FORMAT statements.
       */
      if (!in_format && isholl(inptr[len + 2])) {
        *++outptr = CH_HOLLERITH;
        sptr = gethollerith(sptr, inptr[len + 2]);
        len++; /* so the Hollerith indicator is skipped */
      } else {
        *++outptr = CH_STRING;
        if (outptr > outlim + 1) {
          c = *(outptr - 1);
          /* check for a possible kind value preceding the string */
          if (c == '_') {
            c = *(outptr - 2);
            if (isident(c))
              *(outptr - 1) = CH_UNDERSCORE;
          }
          /*  check for NC preceding quoted string: */
          else if (c == 'c' || c == 'C') {
            c = *(outptr - 2);
            if (c == 'n' || c == 'N') { /* this is kanji string */
              outptr -= 2;              /* erase NC */
              *outptr = CH_KSTRING;
            }
          }
        }
      }
      *++outptr = (char)((sptr >> 24) & 0xFF);
      *++outptr = (char)((sptr >> 16) & 0xFF);
      *++outptr = (char)((sptr >> 8) & 0xFF);
      *++outptr = (char)(sptr & 0xFF);
    } else {

      /* Unterminated string can cause parser confusion:
         len is zero, and get_cstring returned error  */
      if (scnerrfg)
        return;

      /* use a special marker to denote the null string; necessary
       * since using CH_STRING requires >2  bytes
       */
      *++outptr = CH_NULLSTR;
      if (outptr > outlim + 1) {
        c = *(outptr - 1);
        /* check for a possible kind value preceding the string */
        if (c == '_') {
          c = *(outptr - 2);
          if (isident(c))
            *(outptr - 1) = CH_UNDERSCORE;
        }
      }
    }
    outlim = outptr;
    inptr += (len + 1);
    last_is_pseudo = TRUE;
  }

  /* If line ends with a blank, delete it */
  if (last_char_is_blank)
    outptr--;
  eos = outptr;
  *(++outptr) = '\n'; /* mark end of stmtb contents */

end_of_stmt_checks:
  if ((parlev || bracket) && !scnerrfg) {
    if (parlev)
      error(23, 3, gbl.lineno, "parentheses", CNULL);
    if (bracket)
      error(23, 3, gbl.lineno, "brackets", CNULL);
    scnerrfg = TRUE;
  }
}

/* ensure that the first identifier after '!$omp' is an SMP keyword. */
static int
classify_smp(void)
{
  char *cp;
  int idlen; /* number of characters in id string; becomes
              * the length of a keyword.
              */
  int c, savec;
  char *ip;
  int k;

  /* skip any leading white space */

  cp = currc;
  c = *cp;
  while (iswhite(c)) {
    if (c == '\n')
      goto no_identifier;
    c = *++cp;
  }

  /* extract maximal potential id string: */

  idlen = is_ident(cp);
  if (idlen == 0)
    goto no_identifier;

  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  tkntyp = keyword(cp, &parbegkw, &idlen, TRUE);
  ip = cp;
  cp += idlen;

  switch (scn.stmtyp = tkntyp) {
  case 0:
    goto ill_smp;

  case TK_ENDSTMT:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      switch (*++cp) {
      case 'a':
        if (k == 6 && strncmp(cp, "atomic", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_ENDATOMIC;
          goto end_shared;
        }
        break;
      case 'c':
        if (k == 8 && strncmp(cp, "critical", 8) == 0) {
          cp += 8;
          scn.stmtyp = tkntyp = TK_MP_ENDCRITICAL;
          goto end_shared;
        }
        break;
      case 'd':
        if (k == 2 && cp[1] == 'o') {
          scn.stmtyp = tkntyp = TK_MP_ENDPDO;
          cp += 2;
          if (*cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
              strncmp(cp + 1, "simd", 4) == 0) {
            cp += 4 + 1;
            scn.stmtyp = tkntyp = TK_MP_ENDDOSIMD;
          }
          goto end_shared_nowait;
        }
        if (k == 6 && strncmp(cp, "dosimd", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_ENDDOSIMD;
          goto end_shared_nowait;
        }
        if (strncmp(cp, "distribute", 10) == 0) {
          cp += 10;
          scn.stmtyp = tkntyp = TK_MP_ENDDISTRIBUTE;
          if ((is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) ||
              is_ident(cp)) {
            if (*cp == ' ')
              ++cp;
            switch (*cp) {
            case 'p':
              if (strncmp(cp, "parallel", 8) == 0) {
                cp += 8;
                if ((*cp == ' ' && is_ident(cp + 1) == 2 &&
                     strncmp(cp + 1, "do", 2) == 0) ||
                    (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
                  if (*cp == ' ')
                    ++cp;
                  cp += 2;
                  if ((*cp == ' ' && is_ident(cp + 1) &&
                       strncmp(cp + 1, "simd", 4) == 0) ||
                      (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                    if (*cp == ' ')
                      ++cp;
                    cp += 4;
                    scn.stmtyp = tkntyp = TK_MP_ENDDISTPARDOSIMD;
                    goto end_shared;
                  }
                  scn.stmtyp = tkntyp = TK_MP_ENDDISTPARDO;
                  goto end_shared;
                }
                cp -= 8;
                break;
              }
              goto end_shared;
            case 's':
              if (strncmp(cp, "simd", 4) == 0) {
                scn.stmtyp = tkntyp = TK_MP_ENDDISTSIMD;
                goto end_shared;
              }
            }
          }
          goto end_shared;
        }
        break;
      case 'm':
        if (k == 6 && strncmp(cp, "master", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_ENDMASTER;
          goto end_shared;
        }
        break;
      case 'o':
        if (k == 7 && strncmp(cp, "ordered", 7) == 0) {
          cp += 7;
          scn.stmtyp = tkntyp = TK_MP_ENDORDERED;
          goto end_shared;
        }
        break;
      case 'p':
        if (k == 16 && strncmp(cp, "parallelsections", 16) == 0) {
          cp += 16;
          scn.stmtyp = tkntyp = TK_MP_ENDPARSECTIONS;
          goto end_shared;
        }
        if (k == 17 && strncmp(cp, "parallelworkshare", 17) == 0) {
          cp += 17;
          scn.stmtyp = tkntyp = TK_MP_ENDPARWORKSHR;
          goto end_shared;
        }
        if (k == 10 && strncmp(cp, "paralleldo", 10) == 0) {
          cp += 10;
          scn.stmtyp = tkntyp = TK_MP_ENDPARDO;
          if (*cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
              strncmp(cp + 1, "simd", 4) == 0) {
            cp += 4 + 1;
            scn.stmtyp = tkntyp = TK_MP_ENDPARDOSIMD;
          }
          goto end_shared;
        }
        if (k == 14 && strncmp(cp, "paralleldosimd", 14) == 0) {
          cp += 14;
          scn.stmtyp = tkntyp = TK_MP_ENDPARDOSIMD;
          goto end_shared;
        }
        if (strncmp(cp, "parallel", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' && is_ident(cp + 1) == 2 &&
               strncmp(cp + 1, "do", 2) == 0) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_ENDPARDOSIMD;
              goto end_shared;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDPARDO;
            goto end_shared;
          } else if (*cp == ' ' && (k = is_ident(cp + 1)) == 9 &&
                     strncmp(cp + 1, "workshare", 9) == 0) {
            cp += 9 + 1;
            scn.stmtyp = tkntyp = TK_MP_ENDPARWORKSHR;
            goto end_shared;
          } else if (*cp == ' ' && (k = is_ident(cp + 1)) == 8 &&
                     strncmp(cp + 1, "sections", 8) == 0) {
            cp += 8 + 1;
            scn.stmtyp = tkntyp = TK_MP_ENDPARSECTIONS;
            goto end_shared;
          }
          scn.stmtyp = tkntyp = TK_MP_ENDPARALLEL;
          goto end_shared;
        }
        break;
      case 's':
        if (k == 6 && strncmp(cp, "single", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_ENDSINGLE;
          goto end_shared_nowait;
        }
        if (k == 8 && strncmp(cp, "sections", 8) == 0) {
          cp += 8;
          scn.stmtyp = tkntyp = TK_MP_ENDSECTIONS;
          goto end_shared_nowait;
        }
        if (k == 4 && strncmp(cp, "simd", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_ENDSIMD;
          goto end_shared_nowait;
        }
        break;
      case 't':
        if (k == 8 && strncmp(cp, "taskloop", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "simd", 4) == 0) ||
               (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 4;
            scn.stmtyp = tkntyp = TK_MP_ENDTASKLOOPSIMD;
            goto end_shared_nowait;
          }
          scn.stmtyp = tkntyp = TK_MP_ENDTASKLOOP;
          goto end_shared_nowait;
        }
        if (k == 4 && strncmp(cp, "task", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_ENDTASK;
          goto end_shared_nowait;
        }
        if ((*cp == ' ' && is_ident(cp + 1) &&
             strncmp(cp + 1, "teams", 5) == 0) ||
            (is_ident(cp) && strncmp(cp, "teams", 5) == 0)) {
          if (*cp == ' ')
            ++cp;
          cp += 5;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "distribute", 10) == 0) ||
              (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 10;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDISTSIMD;
              goto end_shared_nowait;
            }
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "parallel", 8) == 0) ||
                (is_ident(cp) && strncmp(cp, "parallel", 8) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 8;
              if ((*cp == ' ' && is_ident(cp + 1) == 2 &&
                   strncmp(cp + 1, "do", 2) == 0) ||
                  (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
                if (*cp == ' ')
                  ++cp;
                cp += 2;
                if ((*cp == ' ' && is_ident(cp + 1) &&
                     strncmp(cp + 1, "simd", 4) == 0) ||
                    (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                  if (*cp == ' ')
                    ++cp;
                  cp += 4;
                  scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDISTPARDOSIMD;
                  goto end_shared_nowait;
                }
                scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDISTPARDO;
                goto end_shared_nowait;
              }
              cp -= 8;
              break;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDIST;
            goto end_shared_nowait;
          }
          scn.stmtyp = tkntyp = TK_MP_ENDTEAMS;
          goto end_shared_nowait;
        }
        if (strncmp(cp, "target", 6) == 0) {
          cp += 6;
          if (*cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
              strncmp(cp + 1, "data", 4) == 0) {
            cp += 4 + 1;
            scn.stmtyp = tkntyp = TK_MP_ENDTARGETDATA;
            goto end_shared;
          }
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "teams", 5) == 0) ||
              (is_ident(cp) && strncmp(cp, "teams", 5) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 5;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "distribute", 10) == 0) ||
                (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 10;
              if ((*cp == ' ' && is_ident(cp + 1) &&
                   strncmp(cp + 1, "simd", 4) == 0) ||
                  (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                if (*cp == ' ')
                  ++cp;
                cp += 4;
                scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTSIMD;
                goto end_shared;
              }
              if ((*cp == ' ' && is_ident(cp + 1) &&
                   strncmp(cp + 1, "parallel", 8) == 0) ||
                  (is_ident(cp) && strncmp(cp, "parallel", 8) == 0)) {
                if (*cp == ' ')
                  ++cp;
                cp += 8;
                if ((*cp == ' ' && is_ident(cp + 1) &&
                     strncmp(cp + 1, "do", 2) == 0) ||
                    (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
                  if (*cp == ' ')
                    ++cp;
                  cp += 2;
                  if ((*cp == ' ' && is_ident(cp + 1) &&
                       strncmp(cp + 1, "simd", 4) == 0) ||
                      (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                    if (*cp == ' ')
                      ++cp;
                    cp += 4;
                    scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDOSIMD;
                    goto end_shared;
                  }
                  scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDO;
                  goto end_shared;
                } else {
                  cp -= 8;
                  goto end_shared;
                }
                scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDO;
                goto end_shared;
              }

              scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDIST;
              goto end_shared;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMS;
            goto end_shared;
          }
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "simd", 4) == 0) ||
              (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 4;
            scn.stmtyp = tkntyp = TK_MP_ENDTARGSIMD;
            goto end_shared;
          }
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "parallel", 8) == 0) ||
              (is_ident(cp) && strncmp(cp, "parallel", 8) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 8;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "do", 2) == 0) ||
                (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 2;
              if ((*cp == ' ' && is_ident(cp + 1) &&
                   strncmp(cp + 1, "simd", 4) == 0) ||
                  (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                if (*cp == ' ')
                  ++cp;
                cp += 4;
                scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDOSIMD;
                goto end_shared;
              }
              scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDO;
              goto end_shared;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDTARGPAR;
            goto end_shared;
          }
          scn.stmtyp = tkntyp = TK_MP_ENDTARGET;
          goto end_shared;
        }
        if (k == 9 && strncmp(cp, "taskgroup", 9) == 0) {
          cp += 9;
          scn.stmtyp = tkntyp = TK_MP_ENDTASKGROUP;
          goto end_shared_nowait;
        }
        if (k == 35 &&
            strncmp(cp, "targetteamsdistributeparalleldosimd", 35) == 0) {
          cp += 35;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDOSIMD;
          goto end_shared;
        }
        if (k == 25 && strncmp(cp, "targetteamsdistributesimd", 25) == 0) {
          cp += 25;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTSIMD;
          goto end_shared;
        }
        if (k == 21 && strncmp(cp, "targetteamsdistribute", 21) == 0) {
          cp += 21;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDIST;
          goto end_shared;
        }
        if (k == 10 && strncmp(cp, "targetdata", 10) == 0) {
          cp += 10;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGETDATA;
          goto end_shared;
        }
        if (k == 11 && strncmp(cp, "targetteams", 11) == 0) {
          cp += 11;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMS;
          goto end_shared;
        }
        if (k == 14 && strncmp(cp, "targetparallel", 14) == 0) {
          cp += 14;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGPAR;
          goto end_shared;
        }
        if (k == 16 && strncmp(cp, "targetparalleldo", 16) == 0) {
          cp += 16;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDO;
          goto end_shared;
        }
        if (k == 20 && strncmp(cp, "targetparalleldosimd", 20) == 0) {
          cp += 20;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDOSIMD;
          goto end_shared;
        }

        break;
      case 'w':
        if (k == 9 && strncmp(cp, "workshare", 9) == 0) {
          cp += 9;
          scn.stmtyp = tkntyp = TK_MP_ENDWORKSHARE;
          goto end_shared_nowait;
        }
        break;
      default:
        break;
      }
    }
    goto ill_smp;

  case TK_MP_ENDPDO:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      switch (*++cp) {
      case 'd':
        if (k == 6 && strncmp(cp, "dosimd", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_ENDDOSIMD;
        }
        break;
      case 's':
        if (k == 4 && strncmp(cp, "simd", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_ENDDOSIMD;
        }
      }
    }
    FLANG_FALLTHROUGH;
  case TK_MP_ENDSECTIONS:
  case TK_MP_ENDSIMD:
  case TK_MP_ENDSINGLE:
  case TK_MP_ENDWORKSHARE:
  case TK_MP_ENDTASK:
  case TK_MP_ENDTASKGROUP:
  end_shared_nowait:
    scmode = SCM_PAR;
    break;

  case TK_MP_ENDTARGET:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      switch (*++cp) {
      case 'd':
        if (k == 4 && strncmp(cp, "data", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGETDATA;
        }
        break;
      case 't':
        if (k == 5 && strncmp(cp, "teams", 5) == 0) {
          cp += 5;
          scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMS;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "distribute", 10) == 0) ||
              (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 10;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTSIMD;
              goto end_shared;
            }
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "parallel", 8) == 0) ||
                (is_ident(cp) && strncmp(cp, "parallel", 8) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 8;
              if ((*cp == ' ' && is_ident(cp + 1) &&
                   strncmp(cp + 1, "do", 2) == 0) ||
                  (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
                if (*cp == ' ')
                  ++cp;
                cp += 2;
                if ((*cp == ' ' && is_ident(cp + 1) &&
                     strncmp(cp + 1, "simd", 4) == 0) ||
                    (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                  if (*cp == ' ')
                    ++cp;
                  cp += 4;
                  scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDOSIMD;
                  goto end_shared;
                }
                scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDO;
                goto end_shared;
              } else {
                cp -= 8;
                goto end_shared;
              }
              scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDO;
              goto end_shared;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDIST;
            goto end_shared;
          }
        }
        break;
      case 'p':
        if ((*cp == ' ' && is_ident(cp + 1) &&
             strncmp(cp + 1, "parallel", 8) == 0) ||
            (is_ident(cp) && strncmp(cp, "parallel", 8) == 0)) {
          if (*cp == ' ')
            ++cp;
          cp += 8;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "do", 2) == 0) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDOSIMD;
              goto end_shared;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDO;
            goto end_shared;
          }
          scn.stmtyp = tkntyp = TK_MP_ENDTARGPAR;
          goto end_shared;
        }
        break;
      }
    }
    goto end_shared;

  case TK_MP_ENDPARDO:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
        strncmp(cp + 1, "simd", 4) == 0) {
      cp += 4 + 1;
      scn.stmtyp = tkntyp = TK_MP_ENDPARDOSIMD;
    }
    goto end_shared;

  case TK_MP_ENDDISTRIBUTE:
    if ((is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) ||
        is_ident(cp)) {
      if (*cp == ' ')
        ++cp;
      switch (*cp) {
      case 'p':
        if (strncmp(cp, "parallel", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' && is_ident(cp + 1) == 2 &&
               strncmp(cp + 1, "do", 2) == 0) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_ENDDISTPARDOSIMD;
              goto end_shared;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDDISTPARDO;
            goto end_shared;
          }
          cp -= 8;
          break;
        }
        FLANG_FALLTHROUGH;
      case 's':
        if (strncmp(cp, "simd", 4) == 0) {
          scn.stmtyp = tkntyp = TK_MP_ENDDISTSIMD;
          goto end_shared;
        }
      }
    }
    break;

  case TKF_ENDDISTPAR:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) == 2 &&
        strncmp(cp + 1, "do", 2) == 0) {
      cp += 2 + 1;
      scn.stmtyp = tkntyp = TK_MP_ENDDISTPARDO;
      scmode = SCM_PAR;
      break;
    }
    goto ill_smp;

  case TK_MP_ENDPARALLEL:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      switch (*++cp) {
      case 'd':
        if (k == 2 && cp[1] == 'o') {
          cp += 2;
          scn.stmtyp = tkntyp = TK_MP_ENDPARDO;
        } else if (k == 6 && strncmp(cp, "dosimd", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_ENDPARDOSIMD;
        }
        break;
      }
    }
    FLANG_FALLTHROUGH;
  case TK_MP_ENDCRITICAL:
  case TK_MP_ENDMASTER:
  case TK_MP_ENDORDERED:
  case TK_MP_ENDPARSECTIONS:
  case TK_MP_ENDPARWORKSHR:
  case TK_MP_ENDTARGPARDOSIMD:
  end_shared:
    break;

  case TK_MP_ENDTARGTEAMS:
    if ((*cp == ' ' && is_ident(cp + 1) &&
         strncmp(cp + 1, "distribute", 10) == 0) ||
        (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 10;
      if ((*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "simd", 4) == 0) ||
          (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
        if (*cp == ' ')
          ++cp;
        cp += 4;
        scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTSIMD;
        goto end_shared;
      }
      if ((*cp == ' ' && is_ident(cp + 1) &&
           strncmp(cp + 1, "parallel", 8) == 0) ||
          (is_ident(cp) && strncmp(cp, "parallel", 8) == 0)) {
        if (*cp == ' ')
          ++cp;
        cp += 8;
        if ((*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "do", 2) == 0) ||
            (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
          if (*cp == ' ')
            ++cp;
          cp += 2;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "simd", 4) == 0) ||
              (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 4;
            scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDOSIMD;
            goto end_shared;
          }
          scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDO;
          goto end_shared;
        } else {
          cp -= 8;
          goto end_shared;
        }
        scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDO;
        goto end_shared;
      }
      scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDIST;
      goto end_shared;
    }

    break;
  case TK_MP_ENDTARGTEAMSDIST:
    if ((is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) ||
        is_ident(cp)) {
      if (*cp == ' ')
        ++cp;
      switch (*cp) {
      case 'p':
        if (strncmp(cp, "parallel", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' && is_ident(cp + 1) == 2 &&
               strncmp(cp + 1, "do", 2) == 0) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDOSIMD;
              goto end_shared;
            }
            scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDO;
            goto end_shared;
          }
          cp -= 8;
          break;
        }
        FLANG_FALLTHROUGH;
      case 's':
        if (strncmp(cp, "simd", 4) == 0) {
          scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTSIMD;
          goto end_shared;
        }
      }
    }
    break;

  case TK_MP_ENDTARGPAR:
    if (is_freeform &&
        (*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "do", 2) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 2;
      if ((*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "simd", 4) == 0) ||
          (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
        if (*cp == ' ')
          ++cp;
        cp += 4;
        scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDOSIMD;
        goto end_shared;
      }
      scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDO;
      goto end_shared;
    }
    break;
  case TK_MP_ENDTARGPARDO:
    if (is_freeform &&
        (*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "simd", 4) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 4;
      scn.stmtyp = tkntyp = TK_MP_ENDTARGPARDOSIMD;
      goto end_shared;
    }
    break;
  case TK_MP_ENDDISTPARDO:
    if (is_freeform &&
        (*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "simd", 4) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 4;
      scn.stmtyp = tkntyp = TK_MP_ENDDISTPARDOSIMD;
      goto end_shared;
    }
    break;

  case TK_MP_ENDTARGTEAMSDISTPARDO:
    if (is_freeform &&
        (*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "simd", 4) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 4;
      scn.stmtyp = tkntyp = TK_MP_ENDTARGTEAMSDISTPARDOSIMD;
      goto end_shared;
    }
    break;

  case TK_MP_ENDTEAMS:
    if ((*cp == ' ' && is_ident(cp + 1) &&
         strncmp(cp + 1, "distribute", 10) == 0) ||
        (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 10;
      if ((*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "simd", 4) == 0) ||
          (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
        if (*cp == ' ')
          ++cp;
        cp += 4;
        scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDISTSIMD;
        goto end_shared;
      }
      if ((*cp == ' ' && is_ident(cp + 1) &&
           strncmp(cp + 1, "parallel", 8) == 0) ||
          (is_ident(cp) && strncmp(cp, "parallel", 8) == 0)) {
        if (*cp == ' ')
          ++cp;
        cp += 8;
        if ((*cp == ' ' && is_ident(cp + 1) && strncmp(cp + 1, "do", 2) == 0) ||
            (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
          if (*cp == ' ')
            ++cp;
          cp += 2;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "simd", 4) == 0) ||
              (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 4;
            scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDISTPARDOSIMD;
            goto end_shared;
          }
          scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDISTPARDO;
          goto end_shared;
        } else {
          cp -= 8;
          goto end_shared;
        }
        scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDISTPARDO;
        goto end_shared;
      }
      scn.stmtyp = tkntyp = TK_MP_ENDTEAMSDIST;
      goto end_shared;
    }

    break;

  case TK_MP_PARALLEL:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      switch (*++cp) {
      case 'd':
        if (k == 2 && cp[1] == 'o') {
          cp += 2;
          scn.stmtyp = tkntyp = TK_MP_PARDO;
          if (*cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
              strncmp(cp + 1, "simd", 4) == 0) {
            cp += 4 + 1;
            scn.stmtyp = tkntyp = TK_MP_PARDOSIMD;
          }
        } else if (k == 6 && strncmp(cp, "dosimd", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_PARDOSIMD;
        }
        break;
      case 's':
        if (k == 8 && strncmp(cp, "sections", 8) == 0) {
          cp += 8;
          scn.stmtyp = tkntyp = TK_MP_PARSECTIONS;
        }
        break;
      case 'w':
        if (k == 9 && strncmp(cp, "workshare", 9) == 0) {
          cp += 9;
          scn.stmtyp = tkntyp = TK_MP_PARWORKSHR;
        }
        break;
      default:
        break;
      }
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_PARDO:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
        strncmp(cp + 1, "simd", 4) == 0) {
      cp += 4 + 1;
      scn.stmtyp = tkntyp = TK_MP_PARDOSIMD;
    }
    FLANG_FALLTHROUGH;
  case TK_MP_PARSECTIONS:
  case TK_MP_PARWORKSHR:
  case TK_MP_PARDOSIMD:
    scmode = SCM_PAR;
    break;

  case TK_MP_PDO:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
        strncmp(cp + 1, "simd", 4) == 0) {
      cp += 4 + 1;
      scn.stmtyp = tkntyp = TK_MP_DOSIMD;
    }
    scmode = SCM_PAR;
    break;

  case TK_DECLARE:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      if (k == 9 && strncmp(cp + 1, "reduction", 9) == 0) {
        cp += 9 + 1;
        scn.stmtyp = tkntyp = TK_MP_DECLAREREDUCTION;
        break;
      }
      if (k == 4 && strncmp(cp + 1, "simd", 4) == 0) {
        cp += 4 + 1;
        scn.stmtyp = tkntyp = TK_MP_DECLARESIMD;
        scmode = SCM_PAR;
        break;
      }
      if (k == 6 && strncmp(cp + 1, "target", 6) == 0) {
        cp += 6 + 1;
        scn.stmtyp = tkntyp = TK_MP_DECLARETARGET;
        scmode = SCM_PAR;
        break;
      }
    }
    goto ill_smp;

  case TK_MP_DECLARESIMD:
  case TK_MP_DECLARETARGET:
    scmode = SCM_PAR;
    break;

  case TK_MP_DISTRIBUTE:
    if ((is_freeform && *cp == ' ' && is_ident(cp + 1)) || is_ident(cp)) {
      if (*cp == ' ')
        ++cp;
      switch (*cp) {
      case 's':
        if (strncmp(cp, "simd", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_DISTSIMD;
          break;
        }
        break;
      case 'p':
        if (strncmp(cp, "parallel", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' &&
               (is_ident(cp + 1) && strncmp(cp + 1, "do", 2) == 0)) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;

            if ((*cp == ' ' &&
                 (is_ident(cp + 1) && strncmp(cp + 1, "simd", 4) == 0)) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_DISTPARDOSIMD;
            } else {
              scn.stmtyp = tkntyp = TK_MP_DISTPARDO;
            }
            break;
          } else {
            cp -= 8;
            goto ill_smp;
          }
        }
        if (strncmp(cp, "paralleldo", 10) == 0) {
          scn.stmtyp = tkntyp = TK_MP_DISTPARDO;
        }
        if (strncmp(cp, "paralleldosimd", 14) == 0) {
          scn.stmtyp = tkntyp = TK_MP_DISTPARDOSIMD;
        }
        break;
      default:
        break;
      }
    }
    scmode = SCM_PAR;
    break;
  case TKF_DISTPAR:
    if ((*cp == ' ' && (is_ident(cp + 1) && strncmp(cp + 1, "do", 2) == 0)) ||
        (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 2;
      if (is_ident(cp) && strncmp(cp, "simd", 4) == 0) {
        cp += 4;
        scn.stmtyp = tkntyp = TK_MP_DISTPARDOSIMD;
      } else {
        scn.stmtyp = tkntyp = TK_MP_DISTPARDO;
      }
    } else {
      cp -= 8;
      goto ill_smp;
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_DOACROSS:
  case TK_MP_SECTIONS:
  case TK_MP_SINGLE:
  case TK_MP_WORKSHARE:
  case TK_MP_TASKLOOPSIMD:
  case TK_MP_ATOMIC:
  case TK_MP_DOSIMD:
  case TK_MP_SIMD:
  case TK_MP_TARGETDATA:
  case TK_MP_TARGETENTERDATA:
  case TK_MP_TARGETEXITDATA:
  case TK_MP_TARGETUPDATE:
  case TK_MP_TARGTEAMSDISTPARDOSIMD:
  case TK_MP_TARGTEAMSDISTSIMD:
  case TK_MP_TARGSIMD:
  case TK_MP_TEAMSDISTPARDOSIMD:
  case TK_MP_TEAMSDISTSIMD:
  case TK_MP_DISTPARDOSIMD:
  case TK_MP_DISTSIMD:
  case TK_MP_CANCEL:
    scmode = SCM_PAR;
    break;
  case TK_MP_TASK:
    if (is_ident(cp) && strncmp(cp, "loop", 4) == 0) {
      cp += 4;
      goto taskloop;
    } else {
      scn.stmtyp = tkntyp = TK_MP_TASK;
      scmode = SCM_PAR;
      break;
    }
  case TK_MP_TASKLOOP:
taskloop:
    if ((*cp == ' ' && (is_ident(cp + 1)) &&
         strncmp(cp + 1, "simd", 4) == 0) ||
        (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 4;
      scn.stmtyp = tkntyp = TK_MP_TASKLOOPSIMD;
    }
    scn.stmtyp = tkntyp = TK_MP_TASKLOOP;
    scmode = SCM_PAR;
    break;

  case TK_MP_TARGTEAMS:
    if ((*cp == ' ' && (is_ident(cp + 1)) &&
         strncmp(cp + 1, "distribute", 10) == 0) ||
        (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 10;
      if ((*cp == ' ' && is_ident(cp + 1)) || is_ident(cp)) {
        if (*cp == ' ')
          ++cp;
        switch (*cp) {
        case 's':
          if (strncmp(cp, "simd", 4) == 0) {
            cp += 4;
            scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTSIMD;
            break;
          }
          break;
        case 'p':
          if (strncmp(cp, "parallel", 8) == 0) {
            cp += 8;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "do", 2) == 0) ||
                (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 2;
              if ((*cp == ' ' && is_ident(cp + 1) &&
                   strncmp(cp + 1, "simd", 4) == 0) ||
                  (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                if (*cp == ' ')
                  ++cp;
                cp += 4;
                scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDOSIMD;
              } else {
                scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDO;
              }
              break;
            } else {
              cp -= 8;
              goto ill_smp;
            }
          }
          if (strncmp(cp, "paralleldo", 10) == 0) {
            cp += 10;
            scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDO;
          }
          if (strncmp(cp, "paralleldosimd", 14) == 0) {
            cp += 14;
            scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDOSIMD;
          }
          break;
        default:
          scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDIST;
          break;
        }
      } else {
        scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDIST;
      }
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_TEAMSDISTPARDO:
    if (is_freeform && *cp == ' ' && is_ident(cp + 1) &&
        strncmp(cp + 1, "simd", 4) == 0) {
      cp += 4 + 1;
      scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDOSIMD;
    }

    scmode = SCM_PAR;
    break;
  case TK_MP_TEAMSDIST:
    if ((is_freeform && *cp == ' ' && is_ident(cp + 1)) || is_ident(cp)) {
      if (*cp == ' ')
        ++cp;
      switch (*cp) {
      case 's':
        if (strncmp(cp, "simd", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_TEAMSDISTSIMD;
          break;
        }
        break;
      case 'p':
        if (strncmp(cp, "parallel", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' &&
               (is_ident(cp + 1) && strncmp(cp + 1, "do", 2) == 0)) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;
            if ((is_ident(cp) && strncmp(cp, "simd", 4) == 0) ||
                (*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDOSIMD;
            } else {
              scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDO;
            }
            break;
          } else {
            cp -= 8;
            goto ill_smp;
          }
        }
        if (strncmp(cp, "paralleldo", 10) == 0) {
          scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDO;
        }
        if (strncmp(cp, "paralleldosimd", 14) == 0) {
          scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDOSIMD;
        }
        break;
      default:
        break;
      }
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_DISTPARDO:
    if (is_freeform && *cp == ' ' && is_ident(cp + 1) &&
        strncmp(cp + 1, "simd", 4) == 0) {
      cp += 4 + 1;
      scn.stmtyp = tkntyp = TK_MP_DISTPARDOSIMD;
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_TARGTEAMSDISTPARDO:
    if ((*cp == ' ' && is_ident(cp + 1)) || is_ident(cp)) {
      if (*cp == ' ')
        ++cp;
      if (strncmp(cp, "simd", 4) == 0) {
        cp += 4;
        scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDOSIMD;
      }
    }
    scmode = SCM_PAR;
    break;
  case TK_MP_TARGTEAMSDIST:
    if ((*cp == ' ' && is_ident(cp + 1)) || is_ident(cp)) {
      if (*cp == ' ')
        ++cp;
      switch (*cp) {
      case 's':
        if (strncmp(cp, "simd", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTSIMD;
          break;
        }
        break;
      case 'p':
        if (strncmp(cp, "parallel", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "do", 2) == 0) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDOSIMD;
            } else {
              scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDO;
            }
            break;
          } else {
            cp -= 8;
            goto ill_smp;
          }
        }
        if (strncmp(cp, "paralleldo", 10) == 0) {
          cp += 10;
          scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDO;
        }
        if (strncmp(cp, "paralleldosimd", 14) == 0) {
          cp += 14;
          scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDOSIMD;
        }
        break;
      default:
        break;
      }
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_TEAMS:
    if ((*cp == ' ' && (is_ident(cp + 1)) &&
         strncmp(cp + 1, "distribute", 10) == 0) ||
        (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
      if (*cp == ' ')
        ++cp;
      cp += 10;
      if ((*cp == ' ' && is_ident(cp + 1)) || is_ident(cp)) {
        if (*cp == ' ')
          ++cp;
        switch (*cp) {
        case 's':
          if (strncmp(cp, "simd", 4) == 0) {
            cp += 4;
            scn.stmtyp = tkntyp = TK_MP_TEAMSDISTSIMD;
            break;
          }
          break;
        case 'p':
          if (strncmp(cp, "parallel", 8) == 0) {
            cp += 8;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "do", 2) == 0) ||
                (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 2;
              if ((*cp == ' ' && is_ident(cp + 1) &&
                   strncmp(cp + 1, "simd", 4) == 0) ||
                  (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                if (*cp == ' ')
                  ++cp;
                cp += 4;
                scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDOSIMD;
              } else {
                scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDO;
              }
              break;
            } else {
              cp -= 8;
              goto ill_smp;
            }
          }
          if (strncmp(cp, "paralleldo", 10) == 0) {
            cp += 10;
            scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDO;
          }
          if (strncmp(cp, "paralleldosimd", 14) == 0) {
            cp += 14;
            scn.stmtyp = tkntyp = TK_MP_TEAMSDISTPARDOSIMD;
          }
          break;
        default:
          scn.stmtyp = tkntyp = TK_MP_TEAMSDIST;
          break;
        }
      } else {
        scn.stmtyp = tkntyp = TK_MP_TEAMSDIST;
      }
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_TARGET:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      switch (*++cp) {
      case 'd':
        if (k == 4 && strncmp(cp, "data", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_TARGETDATA;
        }
        break;
      case 'e':
        if (k == 4 && strncmp(cp, "exit", 4) == 0) {
          cp += 4;
          if (*cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
              strncmp(cp + 1, "data", 4) == 0) {
            cp += 4 + 1;
            scn.stmtyp = tkntyp = TK_MP_TARGETEXITDATA;
          } else
            cp -= 4;
        } else if (k == 5 && strncmp(cp, "enter", 5) == 0) {
          cp += 5;
          if (*cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
              strncmp(cp + 1, "data", 4) == 0) {
            cp += 4 + 1;
            scn.stmtyp = tkntyp = TK_MP_TARGETENTERDATA;
          } else
            cp -= 5;
        } else if (k == 8 && strncmp(cp, "exitdata", 8) == 0) {
          cp += 8;
          scn.stmtyp = tkntyp = TK_MP_TARGETEXITDATA;
        } else if (k == 9 && strncmp(cp, "enterdata", 9) == 0) {
          cp += 9;
          scn.stmtyp = tkntyp = TK_MP_TARGETENTERDATA;
        }
        break;
      case 'p':
        if (strncmp(cp, "parallel", 8) == 0) {
          cp += 8;
          if ((*cp == ' ' && is_ident(cp + 1) &&
               strncmp(cp + 1, "do", 2) == 0) ||
              (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
            if (*cp == ' ')
              ++cp;
            cp += 2;
            if ((*cp == ' ' && is_ident(cp + 1) &&
                 strncmp(cp + 1, "simd", 4) == 0) ||
                (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
              if (*cp == ' ')
                ++cp;
              cp += 4;
              scn.stmtyp = tkntyp = TK_MP_TARGPARDOSIMD;
            } else {
              scn.stmtyp = tkntyp = TK_MP_TARGPARDO;
            }
            break;
          } else {
            scn.stmtyp = tkntyp = TK_MP_TARGPAR;
          }
        }
        if (is_ident(cp) && strncmp(cp, "paralleldo", 10) == 0) {
          cp += 10;
          scn.stmtyp = tkntyp = TK_MP_TARGPARDO;
        }
        if (is_ident(cp) && strncmp(cp, "paralleldosimd", 14) == 0) {
          cp += 14;
          scn.stmtyp = tkntyp = TK_MP_TARGPARDOSIMD;
        }
        break;

      case 't':
        if (k == 5 && strncmp(cp, "teams", 5) == 0) {
          cp += 5;
          scn.stmtyp = tkntyp = TK_MP_TARGTEAMS;
          if ((*cp == ' ' && (is_ident(cp + 1)) &&
               strncmp(cp + 1, "distribute", 10) == 0) ||
              (is_ident(cp) && strncmp(cp, "distribute", 10) == 0)) {
            scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDIST;
            if (*cp == ' ')
              ++cp;
            cp += 10;
            if ((*cp == ' ' && is_ident(cp + 1)) || is_ident(cp)) {
              if (*cp == ' ')
                ++cp;
              switch (*cp) {
              case 's':
                if (strncmp(cp, "simd", 4) == 0) {
                  cp += 4;
                  scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTSIMD;
                  break;
                }
                break;
              case 'p':
                if (strncmp(cp, "parallel", 8) == 0) {
                  cp += 8;
                  if ((*cp == ' ' && is_ident(cp + 1) &&
                       strncmp(cp + 1, "do", 2) == 0) ||
                      (is_ident(cp) && strncmp(cp, "do", 2) == 0)) {
                    if (*cp == ' ')
                      ++cp;
                    cp += 2;
                    if ((*cp == ' ' && is_ident(cp + 1) &&
                         strncmp(cp + 1, "simd", 4) == 0) ||
                        (is_ident(cp) && strncmp(cp, "simd", 4) == 0)) {
                      if (*cp == ' ')
                        ++cp;
                      cp += 4;
                      scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDOSIMD;
                    } else {
                      scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDO;
                    }
                    break;
                  } else {
                    cp -= 8;
                    goto ill_smp;
                  }
                }
                if (strncmp(cp, "paralleldo", 10) == 0) {
                  cp += 10;
                  scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDO;
                }
                if (strncmp(cp, "paralleldosimd", 14) == 0) {
                  cp += 14;
                  scn.stmtyp = tkntyp = TK_MP_TARGTEAMSDISTPARDOSIMD;
                }
                break;
              default:
                break;
              }
            }
          }
        }
        break;
      case 'u':
        if (k == 6 && strncmp(cp, "update", 6) == 0) {
          cp += 6;
          scn.stmtyp = tkntyp = TK_MP_TARGETUPDATE;
        }
        break;
      case 's':
        if (k == 4 && strncmp(cp, "simd", 4) == 0) {
          cp += 4;
          scn.stmtyp = tkntyp = TK_MP_TARGSIMD;
        }
        break;
      default:
        break;
      }
    }
    scmode = SCM_PAR;
    break;

  case TK_MP_BARRIER:
  case TK_MP_CRITICAL:
  case TK_MP_FLUSH:
  case TK_MP_MASTER:
  case TK_MP_SECTION:
  case TK_MP_THREADPRIVATE:
  case TK_MP_TASKWAIT:
  case TK_MP_TASKGROUP:
    break;

  case TK_MP_ORDERED:
    scn.stmtyp = tkntyp = TK_MP_ORDERED;
    scmode = SCM_PAR;
    break;


  case TKF_TARGETENTER:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
        strncmp(cp + 1, "data", 4) == 0) {
      cp += 4 + 1;
      scn.stmtyp = tkntyp = TK_MP_TARGETENTERDATA;
      scmode = SCM_PAR;
      break;
    }
    goto ill_smp;

  case TKF_TARGETEXIT:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) == 4 &&
        strncmp(cp + 1, "data", 4) == 0) {
      cp += 4 + 1;
      scn.stmtyp = tkntyp = TK_MP_TARGETEXITDATA;
      scmode = SCM_PAR;
      break;
    }
    goto ill_smp;

  case TKF_CANCELLATION:
    if (is_freeform && *cp == ' ' && (k = is_ident(cp + 1)) == 5 &&
        strncmp(cp + 1, "point", 5) == 0) {
      cp += 5 + 1;
      scn.stmtyp = tkntyp = TK_MP_CANCELLATIONPOINT;
      break;
    }
    goto ill_smp;

  default:
    break;
  }

  currc = cp;
  return tkntyp;

ill_smp:
  savec = *cp;
  *cp = 0;
  error(287, 2, gbl.lineno, "OpenMP", ip);
  *cp = savec;
  return 0;

no_identifier:
  error(288, 2, gbl.lineno, "OpenMP", CNULL);
  return 0;
}

/* ensure that the first identifier after '!dec$' is a DEC keyword. */
static int
classify_dec(void)
{
  char *cp;
  int idlen; /* number of characters in id string; becomes
              * the length of a keyword.
              */
  int c, savec;
  char *ip;
  int k;

  /* skip any leading white space */

  cp = currc;
  c = *cp;
  while (iswhite(c)) {
    if (c == '\n')
      goto no_identifier;
    c = *++cp;
  }

  /* extract maximal potential id string: */

  idlen = is_ident(cp);
  if (idlen == 0)
    goto no_identifier;

  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  tkntyp = keyword(cp, &deckw, &idlen, TRUE);
  ip = cp;
  cp += idlen;

  switch (scn.stmtyp = tkntyp) {
  case 0:
    goto ill_dec;
  case TK_DISTRIBUTE:
    if (*cp == ' ' && (k = is_ident(cp + 1)) != 0) {
      if (k == 5 && strncmp(cp + 1, "point", 5) == 0) {
        tkntyp = TK_DISTRIBUTEPOINT;
        cp += 5 + 1;
        break;
      }
      goto ill_dec;
    }
    FLANG_FALLTHROUGH;
  default:
    break;
  }

  currc = cp;
  return tkntyp;

ill_dec:
  savec = *cp;
  *cp = 0;
  error(287, 2, gbl.lineno, "DEC", ip);
  *cp = savec;
  return 0;

no_identifier:
  error(288, 2, gbl.lineno, "DEC", CNULL);
  return 0;
}

/* ensure that the first identifier after a misc. sentinel is
 * pragma keyword.
 */
static int
classify_pragma(void)
{
  char *cp;
  int idlen; /* number of characters in id string; becomes
              * the length of a keyword.
              */
  int c, savec;
  char *ip;

  /* skip any leading white space */

  cp = currc;
  c = *cp;
  while (iswhite(c)) {
    if (c == '\n')
      goto no_identifier;
    c = *++cp;
  }

  /* extract maximal potential id string: */

  idlen = is_ident(cp);
  if (idlen == 0)
    goto no_identifier;

  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  tkntyp = keyword(cp, &pragma_kw, &idlen, TRUE);
  ip = cp;
  cp += idlen;

  switch (scn.stmtyp = tkntyp) {
  case 0:
    goto ill_dir;
  default:
    break;
  }

  currc = cp;
  return tkntyp;

ill_dir:
  savec = *cp;
  *cp = 0;
  error(287, 2, gbl.lineno, "MEM", ip);
  *cp = savec;
  return 0;

no_identifier:
  error(288, 2, gbl.lineno, "MEM", CNULL);
  return 0;
}

/*
 * ensure that the first identifier after a misc. sentinel is
 * parsed PGI pragma keyword.
 */
static int
classify_pgi_pragma(void)
{
  char *cp;
  int idlen; /* number of characters in id string; becomes
              * the length of a keyword. */
  int c, savec;
  char *ip;

  /* skip any leading white space */
  cp = currc;
  c = *cp;
  while (iswhite(c)) {
    if (c == '\n')
      goto no_identifier;
    c = *++cp;
  }

  /* extract maximal potential id string: */

  idlen = is_ident(cp);
  if (idlen == 0)
    goto no_identifier;

  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  tkntyp = keyword(cp, &ppragma_kw, &idlen, TRUE);
  ip = cp;
  cp += idlen;

  if (tkntyp == 0)
    goto ill_dir;
  scn.stmtyp = tkntyp;

  currc = cp;
  return tkntyp;

ill_dir:
  savec = *cp;
  *cp = 0;
  error(287, 2, gbl.lineno, "PGI", ip);
  *cp = savec;
  return 0;

no_identifier:
  error(288, 2, gbl.lineno, "PGI", CNULL);
  return 0;
}

static int
classify_pgi_dir(void)
{
  char *cp;
  int idlen; /* number of characters in id string; becomes
              * the length of a keyword. */
  int c, savec;
  char *ip;

  /* skip any leading white space */
  cp = currc;
  c = *cp;
  while (iswhite(c)) {
    if (c == '\n')
      goto no_identifier;
    c = *++cp;
  }

  /* extract maximal potential id string: */

  idlen = is_ident(cp);
  if (idlen == 0)
    goto no_identifier;

  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  tkntyp = keyword(cp, &pgi_kw, &idlen, TRUE);
  ip = cp;
  cp += idlen;
  
  scmode = SCM_ACCEL;

  if (tkntyp == 0)
    goto ill_dir;
  scn.stmtyp = tkntyp;

  currc = cp;
  return tkntyp;

ill_dir:
  savec = *cp;
  *cp = 0;
  error(287, 2, gbl.lineno, "pgi", ip);
  *cp = savec;
  return 0;

no_identifier:
  error(288, 2, gbl.lineno, "pgi", CNULL);
  return 0;
}

/*
 * ensure that the first identifier after a misc. sentinel is
 * parsed cuda kernel directive keyword.
 */
static int
classify_kernel_pragma(void)
{
  char *cp;
  int idlen; /* number of characters in id string; becomes
              * the length of a keyword. */
  int c, savec;
  char *ip;

  /* skip any leading white space */
  cp = currc;
  c = *cp;
  while (iswhite(c)) {
    if (c == '\n')
      goto no_identifier;
    c = *++cp;
  }

  /* extract maximal potential id string: */

  idlen = is_ident(cp);
  if (idlen == 0)
    goto no_identifier;

  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  tkntyp = keyword(cp, &kernel_kw, &idlen, TRUE);
  ip = cp;
  cp += idlen;
  scmode = SCM_KERNEL;

  if (tkntyp == 0)
    goto ill_dir;
  scn.stmtyp = tkntyp;

  currc = cp;
  return tkntyp;

ill_dir:
  savec = *cp;
  *cp = 0;
  error(287, 2, gbl.lineno, "CUF", ip);
  *cp = savec;
  return 0;

no_identifier:
  error(288, 2, gbl.lineno, "CUF", CNULL);
  return 0;
}

/*
 * ensure that the first identifier after (/  is
 * parsed as type keyword for array constructor.
 */
static int
classify_ac_type(void)
{
  char *cp, *ip;
  int c, idlen;
  int paren = 0;

  /* skip any leading white space */
  cp = currc;
  c = *cp;

  cp = currc;
  c = *cp;
  while (iswhite(c)) {
    if (c == '\n')
      goto no_ac_type;
    c = *++cp;
  }

  idlen = is_ident(cp);
  if (idlen == 0)
    goto no_ac_type;

  ip = cp;
  cp += idlen;

  if (*cp == ' ')
    ++cp;

  if (*cp == '*') {
    ++cp;
    if (*cp == ' ')
      ++cp;
    idlen = is_digit_string(cp);
    cp += idlen;

  } else if (*cp == '(') {
    ++cp;
    ++paren;
    if (*cp == ' ')
      ++cp;
    c = *cp;
    while (paren) {
      if (c == '(')
        ++paren;
      if (c == ')')
        --paren;
      if (c == '\n')
        goto no_ac_type;
      c = *++cp;
    }
  }
  if (*cp == ' ')
    ++cp;

  /* now search for :: */
  if (*cp == ':' && *(cp + 1) == ':') {
    return 1;
  }

no_ac_type:
  return 0;
}

/*  extract token which begins with an alphabetic token -
    either a keyword or identifier.
    Call keyword look-up routine if necessary.
*/
static void
alpha(void)
{
  int idlen;             /* number of characters in id string; becomes
                          * the length of a keyword.
                          */
  int o_idlen;           /* length of original id string */
  char *cp;              /* pointer into stmtb. */
  char id[MAXIDLEN * 4]; /* temp buffer to hold id; larger
                          * for 'identifier too long' message */
  int c, count;
  char *ip;
  int k;

  /* step 1: extract maximal potential id string: */

  ip = id;
  cp = --currc;
  count = MAXIDLEN * 4;
  do {
    c = *cp++;
    if (--count >= 0)
      *ip++ = c;
  } while (isident(c));
  if (ip != id)
    --ip;
  *ip = '\0';
  --cp; /* point to first char after identifier
         * string */
  o_idlen = idlen = cp - currc;

  /* step 2 - check scan mode to determine further processing */

  switch (scmode) {
  case SCM_FIRST: /* first token of a statement is to be
                   * processed */
    if ((cp[0] == ':' && cp[1] != ':') ||
        (is_freeform && cp[0] == ' ' && cp[1] == ':' && cp[2] != ':')) {
      tknval = get_id_name(id, idlen);
      tkntyp = TK_NAMED_CONSTRUCT;
      scn.stmtyp = 0;
      goto check_name;
    }
    if (idlen == 5 && strncmp(id, "error", 5) == 0 && 
        *cp == ' ' && strncmp((cp+1), "stop", 4) == 0) {
      ERROR_STOP();
      goto alpha_exit;
    }
    if (exp_comma && idlen == 5 && strncmp(id, "quiet", 5) == 0) {
       tkntyp = TK_QUIET;
       goto alpha_exit;
    }
    if (exp_attr && exp_comma && idlen == 4 && strncmp(id, "kind", 4) == 0) {
      tkntyp = TK_KIND;
      goto alpha_exit;
    }
    if (exp_attr && exp_comma && idlen == 3 && strncmp(id, "len", 3) == 0) {
      tkntyp = TK_LEN;
      goto alpha_exit;
    }
    break;
  case SCM_GENERIC:
    if (follow_attr) {
      if (idlen == 8 && strncmp(id, "operator", 8) == 0) {
        scmode = SCM_OPERATOR;
        if (sem.type_mode == 2) {
          /* generic type bound procedure case */
          tkntyp = TK_OPERATOR;
          goto alpha_exit;
        }
      }
      if (idlen == 10 && strncmp(id, "assignment", 10) == 0 &&
          sem.type_mode == 2) {
        /* generic type bound procedure case */
        tkntyp = TK_ASSIGNMENT;
        goto alpha_exit;
      }
    }
    goto fall_thru_scm_ident;
  case SCM_LOOKFOR_OPERATOR: /* look for 'operator' followed by '(' */
    if (idlen == 8 && *cp == '(' && strncmp(id, "operator", 8) == 0) {
      scmode = SCM_OPERATOR;
    }
    FLANG_FALLTHROUGH;
  case SCM_IDENT: /* look only for identifiers */
  fall_thru_scm_ident:
    if (bind_state == B_RPAREN_FOUND) {
      if ((strncmp(id, "result", 6) == 0))
        bind_state = B_RESULT_FOUND;
      if ((strncmp(id, "bind", 6) == 0)) {
        bind_state = B_NONE;
        goto get_keyword;
      }
    }

    goto return_identifier;

  case SCM_FORMAT: /* look for keywords inside FORMAT
                    * statements: */
    if (*id == '$') {
      tkntyp = TK_DOLLAR;
      idlen = 1;
      goto alpha_exit;
    }
    tkntyp = keyword(id, &formatkw, &idlen, FALSE);
    if (tkntyp <= 0)
      goto return_identifier;

    /*
     * special case for edit descriptors not followed by a digit
     */
    if (!isdig(id[1])) {
      switch (tkntyp) {
      case TK_A:
        tkntyp = TK_AFORMAT;
        break;
      case TK_N:
        tkntyp = TK_NFORMAT;
        break;
      case TK_L:
        tkntyp = TK_LFORMAT;
        break;
      case TK_I:
        tkntyp = TK_IFORMAT;
        break;
      case TK_O:
        tkntyp = TK_OFORMAT;
        break;
      case TK_Z:
        tkntyp = TK_ZFORMAT;
        break;
      case TK_F:
        tkntyp = TK_FFORMAT;
        break;
      case TK_E:
        tkntyp = TK_EFORMAT;
        break;
      case TK_G:
        tkntyp = TK_GFORMAT;
        break;
      case TK_D:
        tkntyp = TK_DFORMAT;
        break;
      case TK_DT:
        if (*cp == ' ' || *cp == ',' || *cp == ')') {
          tkntyp = TK_DTFORMAT;
        } else if (*cp == '(') {
        } else {
        }
        break;
      default:
        break;
      }
    } else if (id[1] == '0' && id[2] == 0) {
      if (*cp == ' ' || *cp == ',' || *cp == ')') {
        tkntyp = TK_G0FORMAT; /* G0 */
        idlen += 1;
      }
    }
    goto alpha_exit;

  case SCM_IMPLICIT: /* look for letters, NONE, or keywords in
                      * IMPLICIT stmt */
    if (seen_implp) {
      if (idlen > 1)
        goto return_identifier;
      tkntyp = TK_LETTER;
      tknval = *currc;
    } else if (par_depth)
      goto return_identifier;
    else if (idlen == 4 && strncmp(id, "none", 4) == 0)
      tkntyp = TK_NONE;
    else if ((tkntyp = keyword(id, &normalkw, &idlen, sig_blanks)) == 0)
      goto return_identifier;
    else if (tkntyp == TKF_DOUBLE) {
      tkntyp = double_type(&currc[idlen], &idlen);
      if (tkntyp) {
        goto alpha_exit;
      }
      goto return_identifier;
    } else if (tkntyp < 0) /* one of other TKF_ values */
      goto return_identifier;
    goto alpha_exit;

  case SCM_PROCEDURE:
    /*
     * First identifer after TK_PROCEDURE.
     * For procedures, procedure pointers, and procedure component
     * pointers, parens must follow PROCEDURE and enclose the
     * procedure interface.  Also, if attributes are present, this
     * declaration must be entity style.
     */
    if (par_depth) {
      /*
       * looking at <proc interf>, so it could be an identifier or
       * a type; in any case, keep the same scmode.
       */
      if ((tkntyp = keyword(id, &normalkw, &idlen, sig_blanks)) == 0)
        goto return_identifier;
      switch (tkntyp) {
      /* should only look for type keywords */
      case TK_REAL:
      case TK_INTEGER:
      case TK_DBLEPREC:
      case TK_LOGICAL:
      case TK_COMPLEX:
      case TK_CHARACTER:
      case TK_NCHARACTER:
      case TK_DBLECMPLX:
      case TK_BYTE:
        if (o_idlen == idlen)
          goto alpha_exit;
        break;
      case TK_TYPE:
        /* check for TYPE or CLASS */

        /* "type (...)" or  class (...)"  */
        if (o_idlen == idlen)
          goto alpha_exit;

        /* Possible type(...) or class(...) */
        if (idlen == 4 && strncmp(id, "type", 4) == 0) {
          if (id[4] != '(') {
            idlen = is_ident(id);
            if (idlen == o_idlen) { 
              goto return_identifier;
            }
          }
        } else if (idlen == 5 && strncmp(id, "class", 5) == 0) {
          if (id[5] != '(') {
            idlen = is_ident(id);
            if (idlen == o_idlen) {
              goto return_identifier;
            }
          }
        }
        goto alpha_exit;
      case TKF_DOUBLE:
        tkntyp = double_type(&currc[idlen], &idlen);
        if (tkntyp) {
          goto alpha_exit;
        }
        break;
      }
      idlen = o_idlen;
      goto return_identifier;
    }
    if (exp_attr) {
      scmode = SCM_FIRST;
      goto get_keyword;
    }
    scmode = SCM_IDENT;
    goto return_identifier;

  case SCM_FUNCTION: /* look for the keyword FUNCTION after a type */
    if (par_depth == 0) {
      k = idlen;
      tkntyp = keyword(id, &normalkw, &k, sig_blanks);
      switch (tkntyp) {
      case TK_FUNCTION:
        scmode = SCM_IDENT;
        bind_state = B_FUNC_FOUND;
        idlen = k;
        goto alpha_exit;
      case TK_ELEMENTAL:
      case TK_RECURSIVE:
      case TK_PURE:
      case TK_IMPURE:
      case TK_MODULE:
        scmode = SCM_FIRST;
        idlen = k;
        goto alpha_exit;
      case TK_ATTRIBUTES:
        scmode = SCM_NEXTIDENT;
        idlen = k;
        goto alpha_exit;
      }
      scmode = SCM_IDENT;
    }
    goto return_identifier;

  case SCM_IO: /* check for keywords in I/O statements: */
    if (par_depth == 0) {
      scmode = SCM_IDENT;
      goto return_identifier;
    }
    if (*cp == ' ')
      cp++;
    if (*cp != '=') {
      if (strcmp(id, "readonly") == 0) {
        /* should we allow readonly= for better error checking? */
        tkntyp = TK_READONLY;
        goto alpha_exit;
      }
      if (strcmp(id, "shared") == 0) {
        /* should we allow shared= for better error checking? */
        tkntyp = TK_SHARED;
        goto alpha_exit;
      }
      goto return_identifier;
    }
    tkntyp = keyword(id, &iokw, &idlen, sig_blanks);
    if (tkntyp <= 0)
      goto return_identifier;
    goto alpha_exit;

  case SCM_TO: /* look for the "TO" keyword in a goto
                * assignment: */
    scmode = SCM_IDENT;
    if (id[0] == 't' && id[1] == 'o') {
      tkntyp = TK_TO;
      idlen = 2;
      goto alpha_exit;
    }
    goto return_identifier;

  case SCM_IF: /* alphabetic token in an IF or ELSEIF stmt: */
    if (par_depth)
      goto return_identifier;

    /*
     * have eaten the expression enclosed in parens following the IF or
     * ELSEIF. Now, treat the statement as if scmode = SCM_FIRST;
     */
    break;

  case SCM_DOLAB: /* have finished processing DO <label> : */
    scmode = SCM_IDENT;
    goto return_identifier;

  case SCM_GOTO: /* currently in a GOTO stmt? */
    goto return_identifier;

  case SCM_DONEXT: /* statement is a DO <label> WHILE or CONCURRENT. This mode
                    * is entered when the DO keyword is seen.  What follows
                    * should be the WHILE or CONCURRENT keyword.
                    */
    tkntyp = keyword(id, &normalkw, &idlen, sig_blanks);
    if (tkntyp == TK_WHILE || tkntyp == TK_CONCURRENT) {
      is_doconcurrent = tkntyp == TK_CONCURRENT;
      scmode = SCM_IDENT;
      goto alpha_exit;
    }
    /*
     * Could give an error message indicating that the WHILE/CONCURRENT
     * keyword is expected.
     */
    goto return_identifier;

  case SCM_LOCALITY:
    tkntyp = keyword(id, &normalkw, &idlen, sig_blanks);
    switch (tkntyp) {
    case TK_LOCAL:
    case TK_LOCAL_INIT:
    case TK_SHARED:
    case TK_NONE:
      scmode = SCM_IDENT;
      goto alpha_exit;
    case TK_DEFAULT:
      /* Remain in SCM_LOCALITY mode to look for NONE. */
      goto alpha_exit;
    }
    break;

  case SCM_TYPEIS:
    /* In the context of "type is", check to see if these
     * are a part of an identifier, or if they really are intrinsic
     * type tokens.
     */
    if (par_depth) {
      tkntyp = keyword(id, &normalkw, &idlen, sig_blanks);
      switch (tkntyp) {
      /* should only look for type keywords */
      case TK_REAL:
      case TK_INTEGER:
      case TK_DBLEPREC:
      case TK_LOGICAL:
      case TK_COMPLEX:
      case TK_CHARACTER:
      case TK_NCHARACTER:
      case TK_DBLECMPLX:
      case TK_BYTE:
        if (o_idlen == idlen) {
          if (*(currc + idlen) == '*')
            integer_only = TRUE;
          goto alpha_exit;
        }
        break;
      case TKF_DOUBLE:
        tkntyp = double_type(&currc[idlen], &idlen);
        if (tkntyp) {
          goto alpha_exit;
        }
        break;
      }
    }
    idlen = o_idlen;
    goto return_identifier;
  case SCM_ALLOC: /* keywords in allocate/deallocate stmts: */
    if (*cp == ' ')
      cp++;
    if (par1_attr) {
      tkntyp = keyword(id, &normalkw, &idlen, sig_blanks);
      switch (tkntyp) {
      case TK_REAL:
      case TK_INTEGER:
      case TK_DBLEPREC:
      case TK_LOGICAL:
      case TK_COMPLEX:
      case TK_CHARACTER:
      case TK_NCHARACTER:
      case TK_DBLECMPLX:
      case TK_BYTE:
        if (o_idlen == idlen) {
          if (*(currc + idlen) == '*')
            integer_only = TRUE;
          goto alpha_exit;
        }
        FLANG_FALLTHROUGH;
      case TKF_DOUBLE:
        tkntyp = double_type(&currc[idlen], &idlen);
        if (tkntyp) {
          goto alpha_exit;
        }
        break;
      }
      idlen = o_idlen;
      goto return_identifier;
    }
    if (*cp == '=') {
      tkntyp = keyword(id, &iokw, &idlen, sig_blanks);
      if (tkntyp > 0)
        goto alpha_exit;
    }
    goto return_identifier;

  case SCM_ID_ATTR:
    /* exposed comma not seen; just return an identifier */
    goto return_identifier;

  case SCM_NEXTIDENT:
    /* par_depth should be 1; return an identifer; set scan mode so
     * that the next exposed id is as if the next ident begins a
     * statement
     */
    goto return_identifier;

  case SCM_INTERFACE:
    if (idlen == 8 && strncmp(id, "operator", 8) == 0) {
      tkntyp = TK_OPERATOR;
      scmode = SCM_OPERATOR;
      goto alpha_exit;
    }
    if (idlen == 10 && strncmp(id, "assignment", 10) == 0) {
      tkntyp = TK_ASSIGNMENT;
      goto alpha_exit;
    }
    if (strncmp(id, "read", 4) == 0 || strncmp(id, "write", 5) == 0) {
      scmode = SCM_DEFINED_IO;
      goto return_identifier;
    }
    FLANG_FALLTHROUGH;
  case SCM_OPERATOR:
    scmode = SCM_FIRST;
    if (scn.stmtyp == TK_USE) {
      scmode = SCM_LOOKFOR_OPERATOR;
    }
    goto return_identifier;

  case SCM_PAR:
    if (par_depth)
      goto return_identifier;
    tkntyp = keyword(id, &parallelkw, &idlen, sig_blanks);
    if (tkntyp == 0)
      goto return_identifier;
    goto alpha_exit;

  case SCM_KERNEL:
    if (par_depth)
      goto return_identifier;
    tkntyp = keyword(id, &kernel_kw, &idlen, sig_blanks);
    if (tkntyp == 0)
      goto return_identifier;
    goto alpha_exit;

  case SCM_BIND:
    if (idlen == 4 && strncmp(id, "bind", 4) == 0) {
      tkntyp = TK_BIND;
      scmode = SCM_IDENT;
      goto alpha_exit;
    }
    break;
  case SCM_DEFINED_IO:
    if ((idlen == 9 && strncmp(id, "formatted", 9) == 0) ||
        (idlen == 11 && strncmp(id, "unformatted", 11) == 0)) {
      goto return_identifier;
    }
    break;

  default:
    interr("alpha: bad scan mode", scmode, 4);
  }

  /* step 3 - process the first token of a statement: */

  scmode = SCM_IDENT;
  scn.stmtyp = 0;
  if (idlen == 5 &&  strncmp(id, "error", 5) == 0 && 
      *cp == ' ' && strncmp((cp+1), "stop", 4) == 0) {
    ERROR_STOP();
    goto alpha_exit;
  }
  if (idlen == 9 && strncmp(id, "associate", 9) == 0) {
    char *cp2 = cp;
    if (*cp == ' ')
      cp++;
    if (*cp == '(') {
      tkntyp = TK_ASSOCIATE;
      goto alpha_exit;
    }
    cp = cp2;
  }
  if (idlen == 6 && strncmp(id, "select", 6) == 0) {
    char *cp2 = cp;
    if (*cp == ' ')
      cp++;
    if (strncmp(cp, "type", 4) == 0) {
      cp += 4;
      if (*cp == ' ')
        cp++;
      if (*cp == '(') {
        tkntyp = TK_SELECTTYPE;
        idlen += 4 + 1;
        goto alpha_exit;
      }
    }
    cp = cp2;
  }
  if (idlen == 10 && strncmp(id, "selecttype", 10) == 0) {
    char *cp2 = cp;
    if (*cp == ' ')
      cp++;
    if (*cp == '(') {
      tkntyp = TK_SELECTTYPE;
      goto alpha_exit;
    }
    cp = cp2;
  }
  /*
   * if this stmt contains an exposed equals sign, this indicates that id
   * is an identifier on the left hand side of an assignment stmt except
   * for:
   *		1. IF(exp) a = b
   *		2. DO [<label>] var = m1, m2 [, m3]
   *		3. WHERE (expr) a = b
   *             FORALL (stuff) a = b
   *		4. PARAMETER a = const, if no executable statement
   *		   has been seen.
   *          5. OPTIONS/...[/]=...
   *          6. ENUMERATOR ... a = 2
   */
  if (exp_equal && !exp_attr) {
    if (idlen < 2)
      goto return_identifier;
    if (*cp == ' ')
      cp++;
    if (idlen == 2 && id[0] == 'i' && id[1] == 'f' && *cp == '(') {

      /* possible IF stmt. Scan to matching ')'.  */

      count = 1;
      do {
        ++cp;
        if (*cp == CH_STRING || *cp == CH_HOLLERITH)
          cp += 5; /* next four chars sym pointer */
        if (*cp == ')')
          count--;
        else if (*cp == '(')
          count++;
      } while (count > 0);
      if (cp[1] == ' ')
        ++cp;
      if (cp[1] == '=' || cp[1] == '(' || cp[1] == '.')
        /*  if(...) =,  if(...)(...) =, if(10).t */
        goto return_identifier;
      goto get_keyword;
    }
    if (exp_comma && (!is_freeform || idlen == 2) && id[0] == 'd' &&
        id[1] == 'o') {
      /* return keyword "DO" since a comma is exposed */

      tkntyp = TK_DO;
      idlen = 2;
      scmode = SCM_DOLAB;
      scn.stmtyp = TK_DO;
      goto alpha_exit;
    }
    if (*cp == '(' && ((idlen == 5 && strncmp(id, "where", 5) == 0) ||
                       (idlen == 6 && strncmp(id, "forall", 6) == 0))) {
      /* possible WHERE/FORALL stmt. Scan to matching ')'.  */

      count = 1;
      do {
        cp++;
        if (*cp == CH_STRING || *cp == CH_HOLLERITH)
          cp += 5; /* next four chars sym pointer */
        if (*cp == ')')
          count--;
        else if (*cp == '(')
          count++;
      } while (count > 0);
      if (cp[1] == ' ')
        ++cp;
      if (cp[1] == '=')
        goto return_identifier; /* where(...) = ... */
      goto get_keyword;
    }
    if (sem.pgphase <= PHASE_SPEC && is_keyword(id, idlen, "parameter")) {
      if (idlen == 9) {
        if (!is_freeform)
          /* in fixed form, 'parameter' is the only identifier */
          goto return_identifier;
        if (!iscsym(*cp))
          /* in freeform, 'parameter' is not followed by another
           * identifier.
           */
          goto return_identifier;
      }
      if (!is_freeform && *cp != '=')
        goto return_identifier;
      scn.stmtyp = tkntyp = TK_PARAMETER;
      idlen = 9;
      goto alpha_exit;
    }
    if (*cp == '/' && is_keyword(id, idlen, "options"))
      goto get_keyword;
    if (sem.in_enum && is_keyword(id, idlen, "enumerator")) {
      scn.stmtyp = tkntyp = TK_ENUMERATOR;
      idlen = 10;
      goto alpha_exit;
    }
    goto return_identifier;
  }

  /*  USE , INTRINSIC  :: mod_name */
  if (exp_attr && is_keyword(id, idlen, "use")) {
    goto get_keyword;
  }
  /*
   * if this stmt contains an exposed pointer assign, this indicates that id
   * is an identifier on the left hand side of an assignment stmt except
   * for:
   *          1. USE <ident> , ... => ...
   *		2. IF(exp) a = b
   *		3. FORALL (stuff) a = b
   *          4. a type bound procedure, e.g.,
   *               PROCEDURE ... <ident> => <ident>
   */
  if (exp_ptr_assign && !exp_attr) {
    if (idlen < 2)
      goto return_identifier;
    if (sem.type_mode == 2 && idlen >= strlen("procedure"))
      goto get_keyword;
    if (*cp == ' ')
      cp++;
    if (exp_comma && is_keyword(id, idlen, "use"))
      goto get_keyword;
    if (idlen == 2 && id[0] == 'i' && id[1] == 'f' && *cp == '(') {

      /* possible IF stmt. Scan to matching ')'.  */

      count = 1;
      do {
        ++cp;
        if (*cp == CH_STRING || *cp == CH_HOLLERITH)
          cp += 5; /* next four chars sym pointer */
        if (*cp == ')')
          count--;
        else if (*cp == '(')
          count++;
      } while (count > 0);
      if (cp[1] == ' ')
        ++cp;
      if (cp[1] == '=' || cp[1] == '(' || cp[1] == '.')
        /*  if(...) =,  if(...)(...) =, if(5).t */
        goto return_identifier;
      goto get_keyword;
    }
    if (*cp == '(' && idlen == 6 && strncmp(id, "forall", 6) == 0) {
      /* possible FORALL stmt. Scan to matching ')'.  */

      count = 1;
      do {
        cp++;
        if (*cp == CH_STRING || *cp == CH_HOLLERITH)
          cp += 5; /* next four chars sym pointer */
        if (*cp == ')')
          count--;
        else if (*cp == '(')
          count++;
      } while (count > 0);
      if (cp[1] == ' ')
        ++cp;
      if (cp[1] == '=')
        goto return_identifier; /* forall(...) = ... */
      goto get_keyword;
    }
    goto return_identifier;
  }
get_keyword:
  tkntyp = keyword(id, &normalkw, &idlen, sig_blanks);
  if (tkntyp == 0)
    goto return_identifier;
  bind_state = B_NONE;
  switch (scn.stmtyp = tkntyp) {
  case TK_FUNCTION:
  case TK_SUBROUTINE:
  case TK_ENTRY:
    bind_state = B_FUNC_FOUND;
    break;
  case TKF_GO:
    if (!is_freeform)
      goto return_identifier;
    ip = &currc[idlen];
    if (*ip != ' ' || is_ident(ip + 1) != 2 || ip[1] != 't' || ip[2] != 'o')
      goto return_identifier;
    scn.stmtyp = tkntyp = TK_GOTO;
    ip += 3;
    if (*ip == ' ')
      ip++;
    if (!isdig(*ip))
      tkntyp = TK_GOTOX;
    idlen += 2 + 1;
    scmode = SCM_GOTO;
    break;
  case TKF_SELECT:
    if (!is_freeform)
      goto return_identifier;
    ip = &currc[idlen];
    if (*ip != ' ' || is_ident(ip + 1) != 4 || strncmp(ip + 1, "case", 4) != 0)
      goto return_identifier;
    scn.stmtyp = tkntyp = TK_SELECTCASE;
    idlen += 4 + 1;
    break;
  case TKF_NO:
    ip = &currc[idlen];
    if (*ip != ' ' || is_ident(ip + 1) != 8 ||
        strncmp(ip + 1, "sequence", 8) != 0)
      goto return_identifier;
    scn.stmtyp = tkntyp = TK_NOSEQUENCE;
    idlen += 8 + 1;
    break;
  case TK_BLOCK:
    if (!is_freeform)
      break;
    ip = &currc[idlen];
    if (*ip != ' ' || is_ident(ip + 1) != 4 || strncmp(ip + 1, "data", 4) != 0)
      break;
    scn.stmtyp = tkntyp = TK_BLOCKDATA;
    idlen += 4 + 1;
    break;
  case TK_CASE:
    if (is_freeform) {
      ip = &currc[idlen];
      if (*ip != ' ' || is_ident(ip + 1) != 7 ||
          strncmp(ip + 1, "default", 7) != 0)
        break; /* just return the token for case */
      tkntyp = TK_CASEDEFAULT;
      idlen += 7 + 1;
    }
    break;
  case TK_CASEDEFAULT:
    if (is_freeform)
      /* blank is required between 'case' and 'default' */
      goto return_identifier;
    scn.stmtyp = TK_CASE;
    tkntyp = TK_CASEDEFAULT;
    break;

  case TK_GOTO:
    if (is_freeform) {
      ip = &currc[idlen];
      if (*ip == ' ')
        ip++;
      if (!isdig(*ip))
        tkntyp = TK_GOTOX;
    } else if (!isdig(id[4]))
      tkntyp = TK_GOTOX;
    scmode = SCM_GOTO;
    break;

  case TK_IF:
    scmode = SCM_IF;
    break;

  case TK_ELSEIF:
    /* Ensure that ELSEIF is followed by a left paren; otherwise,
     * assume that the token is ELSE and the IF is a construct name.
     * Need to do this for both free- & fixed- form; the standard
     * says that in freeform, the blank between ELSE & IF is optional.
     */
    if (!is_next_char(cp, '(')) {
      scn.stmtyp = tkntyp = TK_ELSE;
      idlen = 4;
      goto alpha_exit;
    }
    scmode = SCM_IF;
    break;

  case TK_ELSE:
    ip = &currc[idlen];
    /*
     * In freeform, the blank between ELSE & IF is optional.
     */
    if (is_freeform && *ip == ' ') {
      switch (is_ident(ip + 1)) {
      case 2:
        if (ip[1] == 'i' && ip[2] == 'f') {
          /* Ensure that ELSE IF is followed by a left paren;
           * otherwise, assume that the token is ELSE and the IF
           * is a contruct name.
           */
          if (!is_next_char(ip + 3, '(')) {
            scn.stmtyp = tkntyp = TK_ELSE;
            idlen = 5; /* 4 for the 'else' + 1 for the ' ' */
            goto alpha_exit;
          }
          idlen += 2 + 1; /* length of "if" + 1 for the ' ' */
          scn.stmtyp = tkntyp = TK_ELSEIF;
          scmode = SCM_IF;
        }
        break;
      case 5:
        if (strncmp(ip + 1, "where", 5) == 0) {
          idlen += 5 + 1; /* length of "where" + 1 for the ' ' */
          scn.stmtyp = tkntyp = TK_ELSEWHERE;
        }
        break;
      default:
        break;
      }
    }
    break;

  case TK_ALLOCATABLE:
    if (exp_attr)
      scmode = SCM_ID_ATTR;
    break;

  case TK_COMMON:
    if (*(currc + idlen) == ',') {
      /*
       * COMMON , <attr> [, <attr>]...  <common stuff>
       * common attributes are expected -- treat next alpha as a
       * keyword.
       */
      scmode = SCM_FIRST;
      break;
    }
    FLANG_FALLTHROUGH;
  case TK_RECORD:
  case TK_STRUCTURE:
    scmode = SCM_IDENT;
    break;
  case TK_SAVE:
    if (exp_attr)
      scmode = SCM_ID_ATTR;
    else
      scmode = SCM_IDENT;
    break;

  case TK_TYPE:
    if (sem.pgphase != PHASE_EXEC)
      goto begins_with_type;
    if (is_freeform) {
      if (strncmp(currc + idlen, " is", 3) == 0) {
        currc += idlen + 3;
        scmode = SCM_TYPEIS;
        tkntyp = TK_TYPEIS;
        return;
      }
    } else if (strncmp(currc, "typeis", 6) == 0) {
      tkntyp = TK_TYPEIS;
      currc += 6;
      scmode = SCM_TYPEIS;
      return;
    }
    goto begins_with_type;
  case TK_CLASS:
    if (sem.pgphase != PHASE_EXEC)
      goto begins_with_type;
    if (is_freeform) {
      if (strncmp(currc + idlen, " is", 3) == 0) {
        currc += idlen + 3;
        tkntyp = TK_CLASSIS;
        return;
      }
      if (strncmp(currc + idlen, " default", 8) == 0) {
        tkntyp = TK_CLASSDEFAULT;
        currc += idlen + 8;
        return;
      }
    } else {
      if (strncmp(currc, "classis", 7) == 0) {
        tkntyp = TK_CLASSIS;
        currc += 7;
        return;
      }
      if (strncmp(currc, "classdefault", 12) == 0) {
        tkntyp = TK_CLASSDEFAULT;
        currc += 12;
        return;
      }
    }
    goto begins_with_type;
  case TK_REAL:
  case TK_INTEGER:
  case TK_LOGICAL:
  case TK_COMPLEX:
  case TK_CHARACTER:
  case TK_NCHARACTER:
    if (*(currc + idlen) == '*')
      integer_only = TRUE;
    goto begins_with_type;
  case TKF_DOUBLE:
    scn.stmtyp = tkntyp = double_type(&currc[idlen], &idlen);
    if (tkntyp == 0)
      goto return_identifier;
    FLANG_FALLTHROUGH;
  case TK_DBLEPREC:
  case TK_DBLECMPLX:
  case TK_BYTE:
  begins_with_type:
    if (!exp_comma && sem.pgphase == PHASE_INIT)
      scmode = SCM_FUNCTION;
    if (exp_attr)
      scmode = SCM_ID_ATTR;
    break;

  case TK_FORMAT:
    scmode = SCM_FORMAT;
    /* may want to create a character string for the edit list. */
    k = get_fmtstr(currc + 6);
    if (k) {
      /* format string has been created */
      currc[0] = CH_FMTSTR;
      currc[1] = (char)((k >> 24) & 0xFF);
      currc[2] = (char)((k >> 16) & 0xFF);
      currc[3] = (char)((k >> 8) & 0xFF);
      currc[4] = (char)(k & 0xFF);
      currc[5] = '\n';
      return;
    }
    /* otherwise, lex and parse the edit list */
    break;

  case TK_IMPLICIT:
    scmode = SCM_IMPLICIT;
    seen_implp = FALSE;
    /* if last character is a right paren, scan backwards to find the
     * matching left paren and replace with a special marker so that
     * a context sensitive left paren can be returned for IMPLICIT
     * statements.  This is due to the ambiguity in
     *     IMPLICIT CHARACTER (n) (a-z)
     * i.e., don't know if the first left paren begins a length
     * specifier or a range-list.
     * NOTE: To get here, parens must be balanced.
     */
    if (*eos == ')') {
      ip = eos;
    imp_lp:
      *ip = CH_IMPRP; /* mark matching right paren */
      while (*--ip != '(')
        ;
      *ip = CH_IMPLP; /* mark matching left paren */
                      /*
                       * since implicit specifiers may be a list (separated by commas),
                       * need to scan backwards to find other cases of a range enclosed
                       * by parens.  By definition, if a comma is found which is not
                       * enclosed by parens, the right paren which immediately precedes
                       * the comma is the right paren of the range.
                       */
      k = 0;          /* paren depth */
      while (cp < ip) {
        c = *--ip;
        if (c == ')')
          k++;
        else if (c == '(')
          k--;
        else if (c == ',' && k == 0) {
          /*
           * found comma not enclosed by parens.  if the syntax of
           * the statement is correct, the comma will be preceded
           * by a right paren.
           */
          while (*--ip != ')')
            if (cp > ip)
              break;
          if (*ip == ')')
            goto imp_lp;
        }
      }
    }
    break;

  case TK_ASSIGN:
    scmode = SCM_TO;
    break;

  case TK_PRINT:
    past_equal = TRUE;
    reset_past_equal = FALSE;
    break;

  case TK_WRITE:
  case TK_ENCODE:
  case TK_INQUIRE:
    past_equal = TRUE;
    reset_past_equal = FALSE;
    FLANG_FALLTHROUGH;
  case TK_BACKSPACE:
  case TK_CLOSE:
  case TK_DECODE:
  case TK_ENDFILE:
  case TK_FLUSH:
  case TK_OPEN:
  case TK_READ:
  case TK_REWIND:
  case TK_WAIT:
  any_io:
    scmode = SCM_IO;
    if (*(currc + idlen) == '(')
      *(currc + idlen) = CH_IOLP;
    else if (*(currc + idlen) == ' ' && *(currc + idlen + 1) == '(') {
      idlen++; /* eat space */
      *(currc + idlen) = CH_IOLP;
    }
    break;

  case TK_DO:
    /*
     * this DO statement is not of the 'DO iteration' variety since this
     * type of DO is found in step 3 when there is an exposed equal and
     * comma. The next identifier should be the keyword WHILE or CONCURRENT
     */
    scmode = SCM_DONEXT;
    break;

  case TKF_DOWHILE:
    /*
     * 'dowhile' seen as a single keyword; this is only legal if the input
     * source form is fixed.
     */
    if (!is_freeform) {
      scn.stmtyp = tkntyp = TK_DO;
      scmode = SCM_DONEXT;
      idlen = 2;
      goto alpha_exit;
    }
    goto return_identifier;

  case TKF_DOCONCURRENT:
    /*
     * 'doconcurrent' seen as a single keyword; this is only legal if the input
     * source form is fixed.
     */
    if (!is_freeform) {
      scn.stmtyp = tkntyp = TK_DO;
      scmode = SCM_DONEXT;
      idlen = 2;
      goto alpha_exit;
    }
    goto return_identifier;

  case TK_ALLOCATE:
  case TK_DEALLOCATE:
    scmode = SCM_ALLOC;
    break;

  case TK_OPTIONS:
    /*
     * overwrite '/' which follows "options" with '\n' to force end
     * of statement token the next time get_token is called.  Also,
     * save location of stuff after "options/".
     */
    currc += idlen;
    *currc = '\n';
    ip = currc + 1;
    k = 1; /* for null-terminating character */
    while (*ip++ != '\n')
      k++;
    NEED(k, scn.options, char, options_sz, k + CARDB_SIZE);
    strncpy(scn.options, currc + 1, k - 1);
    scn.options[k - 1] = '\0';
    return;

  /*
   * Need to determine if special scan mode is needed for any of the
   * following.
   */
  case TK_ELEMENTAL:
  case TK_PURE:
  case TK_IMPURE:
  case TK_RECURSIVE:
    scmode = SCM_FIRST;
    break;
  case TK_ABSTRACT:
    if (exp_attr)
      scmode = SCM_ID_ATTR;
    else
      scmode = SCM_FIRST;
    break;
  case TK_ATTRIBUTES:
  case TK_LAUNCH_BOUNDS:
    scmode = SCM_NEXTIDENT;
    break;
  case TK_SEQUENCE:
    break;
  case TK_DIMENSION:
    if (exp_attr) {
      tkntyp = TK_DIMATTR;
      scmode = SCM_ID_ATTR;
    }
    break;
  case TKF_ARRAY:
    if (exp_attr) {
      tkntyp = TK_DIMATTR;
      scmode = SCM_ID_ATTR;
      if (flg.standard)
        error(170, 2, gbl.lineno, "ARRAY attribute should be DIMENSION", CNULL);
      break;
    }
    goto return_identifier;
  /* SAVE and ALLOCATABLE handled separately */
  case TK_ASYNCHRONOUS:
  case TK_AUTOMATIC:
  case TK_EXTERNAL:
  case TK_INTENT:
  case TK_INTRINSIC:
  case TK_OPTIONAL:
  case TK_NON_INTRINSIC:
  case TK_PARAMETER:
  case TK_POINTER:
  case TK_PROTECTED:
  case TK_STATIC:
  case TK_TARGET:
  case TK_BIND:
  case TK_VALUE:
  case TK_VOLATILE:
  case TK_PASS:
  case TK_NOPASS:
  case TK_EXTENDS:
  case TK_CONTIGUOUS:
    if (exp_attr)
      scmode = SCM_ID_ATTR;
    break;
  case TK_PRIVATE:
  case TK_PUBLIC:
    if (exp_attr)
      scmode = SCM_ID_ATTR;
    else {
      scmode = SCM_LOOKFOR_OPERATOR;
      follow_attr = TRUE;
    }
    break;

  case TK_PROCEDURE:
    if (sem.type_mode == 2) {
      tkntyp = TK_TPROCEDURE;
      break;
    }
    scmode = SCM_PROCEDURE;
    break;

  case TK_MODULE:
    scmode =
        sem.pgphase == PHASE_INIT && sem.mod_cnt == 0 && !sem.interface ? SCM_IDENT
                                                                        : SCM_FIRST;
    break;

  case TK_SUBMODULE:
    scmode = SCM_IDENT;
    break;

  case TK_ENDBLOCK:
    ip = &currc[idlen];
    if (is_freeform && *ip == ' ' && is_ident(ip + 1) == 4 &&
        strncmp(ip + 1, "data", 4) == 0 && !sem.block_scope) {
      idlen += 4 + 1;
      scn.stmtyp = tkntyp = TK_ENDBLOCKDATA;
      goto end_program_unit;
    }
    break;

  case TK_ENDSTMT:
    ip = &currc[idlen];
    if (is_freeform && *ip == ' ' && (k = is_ident(ip + 1)) != 0) {
      switch (*++ip) {
      case 'a':
        if (k == 9 && strncmp(ip, "associate", 9) == 0) {
          idlen += 9 + 1;
          scn.stmtyp = tkntyp = TK_ENDASSOCIATE;
          goto alpha_exit;
        }
        break;
      case 'b':
        if (k == 9 && strncmp(ip, "blockdata", 9) == 0 &&
            !sem.block_scope) {
          idlen += 9 + 1;
          scn.stmtyp = tkntyp = TK_ENDBLOCKDATA;
          goto end_program_unit;
        }
        if (k == 5 && strncmp(ip, "block", 5) == 0) {
          ip += 5;
          if (*ip == ' ' && (k = is_ident(ip + 1)) == 4 &&
              strncmp(ip + 1, "data", 4) == 0 && !sem.block_scope) {
            idlen += 10 + 1;
            scn.stmtyp = tkntyp = TK_ENDBLOCKDATA;
            goto end_program_unit;
          }
          idlen += 5 + 1;
          scn.stmtyp = tkntyp = TK_ENDBLOCK;
          goto alpha_exit;
        }
        break;
      case 'd':
        if (k == 2 && ip[1] == 'o') {
          idlen += 2 + 1;
          scn.stmtyp = tkntyp = TK_ENDDO;
          goto alpha_exit;
        }
        break;
      case 'e':
        if (k == 4 && strncmp(ip, "enum", 4) == 0) {
          idlen += 4 + 1;
          if (!sem.in_enum) {
            goto return_identifier;
          }
          scn.stmtyp = tkntyp = TK_ENDENUM;
          goto alpha_exit;
        }
        break;
      case 'f':
        if (k == 4 && strncmp(ip, "file", 4) == 0) {
          idlen += 4 + 1;
          scn.stmtyp = tkntyp = TK_ENDFILE;
          goto any_io;
        }
        if (k == 6 && strncmp(ip, "forall", 6) == 0) {
          idlen += 6 + 1;
          scn.stmtyp = tkntyp = TK_ENDFORALL;
          goto alpha_exit;
        }
        if (k == 8 && strncmp(ip, "function", 8) == 0) {
          idlen += 8 + 1;
          scn.stmtyp = tkntyp = TK_ENDFUNCTION;
          goto end_program_unit;
        }
        break;
      case 'i':
        if (k == 2 && ip[1] == 'f') {
          idlen += 2 + 1;
          scn.stmtyp = tkntyp = TK_ENDIF;
          goto alpha_exit;
        }
        if (k == 9 && strncmp(ip, "interface", 9) == 0) {
          idlen += 9 + 1;
          scn.stmtyp = tkntyp = TK_ENDINTERFACE;
          scmode = SCM_INTERFACE;
          goto alpha_exit;
        }
        break;
      case 'm':
        if (k == 6 && strncmp(ip, "module", 6) == 0) {
          idlen += 6 + 1;
          scn.stmtyp = tkntyp = TK_ENDMODULE;
          goto end_module;
        }
        if (k == 3 && strncmp(ip, "map", 3) == 0) {
          idlen += 3 + 1;
          scn.stmtyp = tkntyp = TK_ENDMAP;
          goto alpha_exit;
        }
        break;
      case 'p':
        if (k == 9 && strncmp(ip, "procedure", k) == 0) {
          idlen += k + 1;
          scn.stmtyp = tkntyp = TK_ENDPROCEDURE;
          goto end_program_unit;
        }
        if (k == 7 && strncmp(ip, "program", 7) == 0) {
          idlen += 7 + 1;
          scn.stmtyp = tkntyp = TK_ENDPROGRAM;
          goto end_program_unit;
        }
        break;
      case 's':
        if (k == 6 && strncmp(ip, "select", 6) == 0) {
          idlen += 6 + 1;
          scn.stmtyp = tkntyp = TK_ENDSELECT;
          goto alpha_exit;
        }
        if (k == 9 && strncmp(ip, "structure", 9) == 0) {
          idlen += 9 + 1;
          scn.stmtyp = tkntyp = TK_ENDSTRUCTURE;
          goto alpha_exit;
        }
        if (k == 9 && strncmp(ip, "submodule", k) == 0) {
          idlen += k + 1;
          scn.stmtyp = tkntyp = TK_ENDSUBMODULE;
          goto end_module;
        }
        if (k == 10 && strncmp(ip, "subroutine", 10) == 0) {
          idlen += 10 + 1;
          scn.stmtyp = tkntyp = TK_ENDSUBROUTINE;
          goto end_program_unit;
        }
        break;
      case 't':
        if (k == 4 && strncmp(ip, "type", 4) == 0) {
          idlen += 4 + 1;
          scn.stmtyp = tkntyp = TK_ENDTYPE;
          goto alpha_exit;
        }
        break;
      case 'u':
        if (k == 5 && strncmp(ip, "union", 5) == 0) {
          idlen += 5 + 1;
          scn.stmtyp = tkntyp = TK_ENDUNION;
          goto alpha_exit;
        }
        break;
      case 'w':
        if (k == 5 && strncmp(ip, "where", 5) == 0) {
          idlen += 5 + 1;
          scn.stmtyp = tkntyp = TK_ENDWHERE;
          goto alpha_exit;
        }
        break;

      default:
        break;
      }
      goto return_identifier;
    }
    goto end_program_unit;
  case TK_CONTAINS:
    if (sem.type_mode == 1) {
      tkntyp = TK_TCONTAINS;
      break;
    }
    if (gbl.currsub == 0 && gbl.currmod > NOSYM) {
      /* CONTAINS within a module - statement will be treated as the
       * END of a blockdata
       */
      goto end_program_unit;
    }
    /*
     * CONTAINS of an internal procedure.  Don't treat as an end of
     * subprogram yet.
     */
    break;
  case TK_ENDBLOCKDATA:
    if (sem.block_scope) {
      scn.stmtyp = tkntyp = TK_ENDBLOCK;
      idlen -= 4;
      break;
    }
    FLANG_FALLTHROUGH;
  case TK_ENDFUNCTION:
  case TK_ENDPROCEDURE:
  case TK_ENDPROGRAM:
  case TK_ENDSUBROUTINE:
  end_program_unit:
    if (sem.interface)
      break;
    if (gbl.internal && gbl.currsub)
      break;
    scn.end_program_unit = TRUE;
    break;
  case TK_ENDMODULE:
  case TK_ENDSUBMODULE:
  end_module:
    scn.end_program_unit = TRUE;
    break;

  case TK_INTERFACE:
  case TK_ENDINTERFACE:
    scmode = SCM_INTERFACE;
    break;

  case TK_GENERIC:
    if (sem.type_mode == 2)
      scmode = SCM_GENERIC; /* generic type bound procedure */
    break;

  case TK_USE:
    if (exp_attr)
      scmode = SCM_ID_ATTR;
    break;
  case TK_ENUM:
    scmode = SCM_BIND;
    break;
  case TK_ENUMERATOR:
  case TK_ENDENUM:
    if (!sem.in_enum) {
      goto return_identifier;
    }
    FLANG_FALLTHROUGH;
  default:
    break;
  }
  goto alpha_exit;

/* step 4 - enter identifier into symtab and return it: */

return_identifier:
  if (par1_attr == 1 && par_depth == 1 &&
      (is_doconcurrent || scn.stmtyp == TK_FORALL)) {
    tkntyp = keyword(id, &normalkw, &idlen, sig_blanks);
    if (tkntyp == TK_INTEGER && o_idlen == idlen)
      goto alpha_exit;
  }
  if (exp_ac) {
    if (*cp == ' ')
      cp++;
    if (acb_depth) {
      tkntyp = keyword(id, &normalkw, &idlen, sig_blanks);
      switch (tkntyp) {
      case TK_REAL:
      case TK_INTEGER:
      case TK_DBLEPREC:
      case TK_LOGICAL:
      case TK_COMPLEX:
      case TK_CHARACTER:
      case TK_NCHARACTER:
      case TK_DBLECMPLX:
      case TK_BYTE:
        if (o_idlen == idlen) {
          if (*(currc + idlen) == '*')
            integer_only = TRUE;
          goto alpha_exit;
        }
      }
      idlen = o_idlen;
      goto do_idtoken;
    }
    if (acb_depth && *cp == '=') {
      tkntyp = keyword(id, &iokw, &idlen, sig_blanks);
      if (tkntyp > 0)
        goto alpha_exit;
    }
  }

do_idtoken:
  tkntyp = TK_IDENT;
  tknval = get_id_name(id, idlen);
check_name:
  if (flg.standard || !XBIT(57, 0x40)) {
    /* these errors are severe, since at least one compiler (cf90)
     * will report severe errors also.
     */
    char *nm = scn.id.name + tknval;
    if (id[0] == '_') {
      if (!XBIT(57, 0x40))
        error(34, 3, gbl.lineno, nm, CNULL);
      else
        error(170, 2, gbl.lineno, "Identifier begins with '_' -", nm);
    } else if (id[0] == '$') {
      if (!XBIT(57, 0x40))
        error(34, 3, gbl.lineno, nm, CNULL);
      else
        error(170, 2, gbl.lineno, "Identifier begins with '$' -", nm);
    }
    if (flg.standard) {
      if (strchr(nm + 1, (int)'$'))
        error(170, 2, gbl.lineno, "Identifier contains '$' -", nm);
    }
  }
alpha_exit:
  currc += idlen;
}

static void
init_ktable(KTABLE *ktable)
{
  int nkwds;
  KWORD *base;
  int i;
  int ch;

  nkwds = ktable->kcount;
  base = ktable->kwds;
  /*
   * Scan the keyword table (KTABLE) to determine the keywords which begin
   * with each lowercase letter.  When completed, the first and last members
   * of the keyword table will be used by keyword() to inclusively search
   * the first and last keywords beginning with the first letter of an
   * identifer.  Note that first[ch-'a'] is zero if there does not exist a
   * keyword which begins with 'ch'.  A nonzero value, i, represents the
   * index, i, into the KWORD table.
   */
  for (i = 1; i < nkwds; i++) {
    ch = *(base[i].keytext) - 'a';
#if DEBUG
    /* ensure keywords begin with a lowercase letter */
    if ((ch + 'a') < 'a' || (ch + 'a') > 'z') {
      interrf(ERR_Fatal, "Illegal keyword, %s, for init_ktable",
              base[i].keytext);
    }
#endif
    if (ktable->first[ch] == 0)
      ktable->first[ch] = i;
    ktable->last[ch] = i;
  }
}

static int
get_id_name(char *id, int idlen)
{
  int tkv;
  char *p;

  tkv = scn.id.avl;
  scn.id.avl += idlen + 1; /* 1 extra for '\0' */
  NEED(scn.id.avl, scn.id.name, char, scn.id.size, scn.id.avl + 256);
  p = scn.id.name + tkv;
  while (idlen-- > 0)
    *p++ = *id++;
  *p = '\0';
  return tkv;
}

/*
 * Determine if the identifier is the given keyword.  If the source input
 * form is fixed, the keyword need only be a prefix of the identifier.
 * If the source form is free, the keyword and identifier need to be an
 * exact match.
 */
static LOGICAL
is_keyword(char *id, int idlen, const char *kwd)
{
  int len;

  len = strlen(kwd);
  if (strncmp(id, kwd, len) == 0) {
    if (is_freeform)
      return idlen == len;
    return TRUE;
  }
  return FALSE;
}

/*
 * Determine if the 'token' beginning at ip is an identifer.
 * Return the length of the identifer; 0 is returned if it
 * isn't an identifier.
 */
static int
is_ident(char *ip)
{
  int count;
  char c;

  c = *ip++;
  if (iscsym(c)) {
    count = 1;
    while (TRUE) {
      c = *ip++;
      if (isident(c))
        ;
      else
        break;
      count++;
    }
  } else
    count = 0;

  return count;
}

/*
 * Determine if the 'token' beginning at ip is a string of digits.
 * Return the length of the digits; 0 is returned if it isn't a digit string.
 */
static int
is_digit_string(char *ip)
{
  int count;
  char c;

  count = 0;
  while (TRUE) {
    c = *ip++;
    if (isdig(c))
      ;
    else
      break;
    count++;
  }

  return count;
}

/*
 * expecting a kind parameter; return values:
 * o  if an identifier is found, its length is returned.
 *    'kind_id' is the sptr of the ST_PARAM if the identifier is a legal
 *    KIND parameter; otherwise, 'kind_id' is 0.
 * o  value of the function is 0 if an identfier is not found.
 */
static int
get_kind_id(char *ip)
{
  int kind_id_len;
  int alias_id;
  if ((kind_id_len = is_ident(ip))) {

    kind_id = getsym(ip, kind_id_len);
    while (kind_id) {
      alias_id = kind_id;
      if (STYPEG(kind_id) == ST_ALIAS)
        kind_id = SYMLKG(kind_id);
      if (STYPEG(kind_id) == ST_PARAM && DT_ISINT(DTYPEG(kind_id))) {
        return kind_id_len;
      }
      kind_id = alias_id;
      for (kind_id = HASHLKG(kind_id); kind_id != 0;
           kind_id = HASHLKG(kind_id)) {
        if (HIDDENG(kind_id))
          continue;
        if (strncmp(ip, SYMNAME(kind_id), kind_id_len) != 0 ||
            *(SYMNAME(kind_id) + kind_id_len) != '\0')
          continue;

        if (stb.ovclass[STYPEG(kind_id)] != OC_OTHER)
          continue;

        if (STYPEG(kind_id) == ST_ALIAS)
          kind_id = SYMLKG(kind_id);

        break;
      }
    }
    if (!kind_id) {
      kind_id = getsym(ip, kind_id_len);
      error(84, 3, gbl.lineno, SYMNAME(kind_id), "- KIND parameter");
      kind_id = 0;
    }
  } else if ((kind_id_len = is_digit_string(ip))) {
    INT num[2];
    num[0] = 0;
    (void)atoxi(ip, &num[1], kind_id_len, 10);
    kind_id = getcon(num, DT_INT4);
  } else
    kind_id = 0;
  return kind_id_len;
}

static INT
get_kind_value(int knd)
{
  int dtype;

  if (STYPEG(knd) == ST_PARAM) {
    dtype = DTYPEG(knd);
    if (!DT_ISINT(dtype)) {
      error(84, 3, gbl.lineno, SYMNAME(knd), "- KIND parameter");
      return 0;
    }
    return cngcon(CONVAL1G(knd), dtype, DT_INT4);
  }
#if DEBUG
  assert(STYPEG(knd) == ST_CONST, "get_kind_value:unexp.stype", STYPEG(knd), 3);
#endif
  if (STYPEG(knd) != ST_CONST) {
    interr("get_kind_value:unexp.stype", STYPEG(knd), 3);
    return 0;
  }
  dtype = DTYPEG(knd);
  if (!DT_ISINT(dtype)) {
    error(33, 3, gbl.lineno, "- KIND parameter", CNULL);
    return 0;
  }
  return cngcon(CONVAL2G(knd), dtype, DT_INT4);
}

/*  return token id for the longest keyword in keyword table
 *  'ktype', which is a prefix of the id string.
 *  Set 'keylen' to the length of the keyword found.
 *  Possible return values:
 *	>  0 - keyword found (corresponds to a TK_ value).
 *	== 0 - keyword not found.
 *	<  0 - keyword 'prefix' found (corresponds to a TKF_ value).
 *  If a match is found, keyword_idx is set to the index of the KWORD
 *  entry matching the keyword.
 */
static int
keyword(char *id, KTABLE *ktable, int *keylen, LOGICAL exact)
{
  int chi, low, high, p, cond;
  KWORD *base;

  /* convert first character (a letter) of an identifier into a subscript */
  chi = *id - 'a';
  if (chi < 0 || chi > 25)
    return 0; /* not a letter 'a' .. 'z' */
  low = ktable->first[chi];
  if (low == 0)
    return 0; /* a keyword does not begin with the letter */
  high = ktable->last[chi];
  base = ktable->kwds;
  /*
   * Searching for the longest keyword which is a prefix of the identifier.
   */
  p = 0;
  for (; low <= high; low++) {
    cond = cmp(id, base[low].keytext, keylen);
    if (cond < 0)
      break;
    if (cond == 0)
      p = low;
  }
  if (p) {
    keyword_idx = p;
    return base[p].toktyp;
  }
  return 0;
}

/*  Return 0 if kw is a prefix of id.  Otherwise find
 *  the first differing character position and return
 *  a negative number if id < kw and a positive number
 *  if id > kw.
 *  If id is a prefix of kw, return -1.
 *  When 0 is returned, keylen is set to length of keyword.
 */
static int
cmp(const char *id, const char *kw, int *keylen)
{
  char c;
  const char *first = kw;

  do {
    if ((c = *(id++)) != *kw) {
      if (!c)
        return (-1); /* end of id reached */
      return (c - *kw);
    }
  } while (*++kw);

  *keylen = kw - first;
  return (0);
}

static int
get_cstring(char *p, int *len)
{
  int c, n;
  char *outp, delimc;
  char *begin;

  delimc = *p++;
  outp = begin = p;
  do {
    c = *p++;
    if (c == delimc) {
      if (*p != delimc)
        break;
      c = *p++;
    } else if (c == '\\') {
      if (!flg.standard && !XBIT(124, 0x40)) {
        c = *p++;
        if (c == 'n')
          c = '\n';
        else if (c == 't')
          c = '\t';
        else if (c == 'v')
          c = '\v';
        else if (c == 'b')
          c = '\b';
        else if (c == 'r')
          c = '\r';
        else if (c == 'f')
          c = '\f';
        else if (isodigit(c)) {
          n = c - '0';
          if (isodigit(*p)) {
            n = (n * 8) + ((*p++) - '0');
            if (isodigit(*p))
              n = (n * 8) + ((*p++) - '0');
          }
          c = n;
        } else if (c == '\n') {
          goto err;
        } else if (c == delimc && *p == '\n') {
          errsev(601);
          scnerrfg = TRUE;
          *len = 0;
          return 0;
        }
      }
    } else if (c == '\n') {
    err: /* missing end quote */
      errsev(26);
      scnerrfg = TRUE;
      *len = 0;
      return 0;
    }
    *outp++ = c;
  } while (TRUE);

  *len = (p - begin) - 1;
  if (*len <= 0) { /* error if zero length string  */
    return getstring("", 0);
  }
  return getstring(begin, outp - begin);
}

static void fmt_putchar(int);
static void fmt_putstr(const char *);
static void char_to_text(int);

static struct {
  char *str;
  int len;
  int sz;
} fmtstr;

static int
get_fmtstr(char *p)
{
  char c;
  int sptr, sptr2;
  char *from;
  int dtype;
  int len;
  char b[64];

  /* Only create the format string if requested */
  if (!XBIT(58, 0x200)) { /* similar check exists in semantio.c:chk_fmtid() */
    if (is_freeform) {
      /* blanks are not significant in a format list; need to 'crunch' */
      char *avl = p;
      char *q = p;
      while ((c = *q++) != '\n') {
        switch (c) {
        case CH_X:
        case CH_O:
        case CH_B:
        case CH_UNDERSCORE:
          *avl++ = c;
          break;
        case CH_HOLLERITH:
        case CH_KSTRING:
        case CH_STRING:
          *avl++ = c;
          *avl++ = *q++;
          *avl++ = *q++;
          *avl++ = *q++;
          *avl++ = *q++;
          break;
        case CH_NULLSTR:
          *avl++ = c;
          break;
        default:
          if (!iswhite(c))
            *avl++ = c;
          break;
        }
      }
      *avl = '\n';
    }
    return 0;
  }
  while (TRUE) {
    c = *p;
    if (c == '\0' || c == '\n')
      return 0;
    if (c == '(')
      break;
    ++p;
  }
  fmtstr.sz = 4096;
  fmtstr.len = 0;
  NEW(fmtstr.str, char, fmtstr.sz);
  /*
   * crunch() has already determined that the edit list is properly
   * parenthesized.
   */
  while ((c = *p++) != '\n') {
    switch (c) {
    case CH_X:
    case CH_O:
    case CH_B:
    case CH_UNDERSCORE:
      goto err;
    case CH_HOLLERITH:
      sptr = MERGE(p);
      p += 4;
      sptr2 = CONVAL1G(sptr);
      dtype = DTYPEG(sptr2);
      from = stb.n_base + CONVAL1G(sptr2);
      len = string_length(dtype);
      sprintf(b, "%d", len);
      fmt_putstr(b);
      /* kind of hollerith - 'h', 'l', or 'r' */
      fmt_putchar(CONVAL2G(sptr));
      while (len--) {
        c = *from++ & 0xff;
        fmt_putchar(c);
      }
      break;
    case CH_KSTRING:
      sptr = MERGE(p);
      p += 4;
      fmt_putstr("nc");
      sptr = CONVAL1G(sptr);
      goto do_str;
    case CH_STRING:
      sptr = MERGE(p);
      p += 4;
    do_str:
      fmt_putchar('\'');
      dtype = DTYPEG(sptr);
      from = stb.n_base + CONVAL1G(sptr);
      len = string_length(dtype);
      while (len--)
        char_to_text(*from++);
      fmt_putchar('\'');
      break;
    case CH_NULLSTR:
      fmt_putstr("\'\'");
      break;
    default:
      fmt_putchar(c);
      break;
    }
  }
  fmt_putchar('\0');
  sptr = getstring(fmtstr.str, fmtstr.len - 1);
  FREE(fmtstr.str);
  return sptr;
err:
  FREE(fmtstr.str);
  return 0;
}

static void
fmt_putchar(int ch)
{
  int pos;

  pos = fmtstr.len++;
  NEED(fmtstr.len, fmtstr.str, char, fmtstr.sz, fmtstr.sz + 4096);
  fmtstr.str[pos] = ch;
}

static void
fmt_putstr(const char *str)
{
  int ch;

  while ((ch = *str++))
    fmt_putchar(ch);
}

/*
 * emit a character with consideration given to the ', escape sequences,
 * unprintable characters, etc.
 */
static void
char_to_text(int ch)
{
  int c;
  char b[8];

  c = ch & 0xff;
  if (c == '\\' && !flg.standard && !XBIT(124, 0x40)) {
    fmt_putchar('\\');
    fmt_putchar('\\');
  } else if (c == '\'') {
    fmt_putchar('\'');
    fmt_putchar('\'');
  } else if (c >= ' ' && c <= '~')
    fmt_putchar(c);
  else if (c == '\n') {
    fmt_putchar('\\');
    fmt_putchar('n');
  } else if (c == '\t') {
    fmt_putchar('\\');
    fmt_putchar('t');
  } else if (c == '\v') {
    fmt_putchar('\\');
    fmt_putchar('v');
  } else if (c == '\b') {
    fmt_putchar('\\');
    fmt_putchar('b');
  } else if (c == '\r') {
    fmt_putchar('\\');
    fmt_putchar('r');
  } else if (c == '\f') {
    fmt_putchar('\\');
    fmt_putchar('f');
  } else {
    /* Mask off 8 bits worth of unprintable character */
    sprintf(b, "\\%03o", c);
    fmt_putstr(b);
  }
}

/*  extract integer, real, or double precision constant token
    and convert into SC form:
*/
/* == 0 normal case. == 1 this is first part
 * of complex constant. == 2 this is 2nd part
 * of complex constant. */
static void
get_number(int cplxno)
{
  char c;
  char *cp;
  INT num[4];
  int sptr;
  LOGICAL d_exp;
  LOGICAL q_exp;
  int kind_id_len;
  int errcode;
  int nmptr;
  int kind;
  LOGICAL chk_octal;

  chk_octal = TRUE; /* Attempt to recognize Cray's octal extension */
  kind_id_len = 0;
  d_exp = FALSE;
  q_exp = FALSE;
  nmptr = 0;
  c = *(cp = currc);
  if (c == '-' || c == '+')
    c = *++cp;
  if (c == '.')
    goto state2;
  assert(isdig(c), "get_number: bad start", (int)c, 3);

  do {
    c = *++cp;
  } while (isdig(c));
  if (scmode == SCM_FORMAT || scmode == SCM_DOLAB || integer_only) {
    /* Cray's octal extension is not allowed in this context; problematic
     * cases:
     *     format (... 10b ...)
     *     do 10 b= ...
     *     real*1 b
     */
    chk_octal = FALSE;
    goto return_integer;
  }
  if (c == '.') {
    /*
     * watch out for: digits . eq|digits|e|d|E|D
     */
    if (cp[1] == 'e') {
      if (cp[2] == 'q')
        goto return_integer; /* digits .eq */
      goto state2;           /* digits .e */
    }
    if (isdig(cp[1]) || cp[1] == 'd' || cp[1] == 'q')
      goto state2; /* digits . digits|d|q */
    if (islower(cp[1]))
      goto return_integer; /* digits . <lowercase letter> */
    goto state2;           /* could still be digits . E|D */
  }
  if (c == 'e' || c == 'E' || c == 'd' || c == 'D' ||
      c == 'q' || c == 'Q')
    goto state3;
  goto return_integer;
state2: /* digits . [ digits ]  */
  do {
    c = *++cp;
  } while (isdig(c));
  assert(cp > currc + 1, "get_number:single dot", (int)c, 3);
  if (c == 'e' || c == 'E' || c == 'd' || c == 'D' ||
      c == 'q' || c == 'Q')
    goto state3;
  goto return_real;

state3: /* digits [ . [ digits ] ] { e | d | q }  */
  if (c == 'q') {
    q_exp = TRUE;
  } else if (c == 'd') {
    d_exp = TRUE;
  }
  c = *++cp;
  if (isdig(c))
    goto state5;
  if (c == '+' || c == '-')
    goto state4;
  goto syntax_error;

state4: /* digits [ . [ digits ] ] { e | d } { + | - }  */
  c = *++cp;
  if (!isdig(c))
    goto syntax_error;

state5: /* digits [ . [ digits ] ] { e | d } [ + | - ] digits  */
  c = *++cp;
  if (isdig(c))
    goto state5;
  goto return_real;

syntax_error:
  errsev(28);
  scnerrfg = 1;
  return;

return_integer:
  integer_only = FALSE; /* Can now allow real numbers */
  if (cplxno) {
    if (*cp == '_' && (kind_id_len = get_kind_id(cp + 1))) {
      kind_id_len++; /* account for '_' */
      if (kind_id) { /* kind_id is non-zero if ident is a KIND parameter*/
        int dtype;
        dtype = select_kind(DT_INT, TY_INT, get_kind_value(kind_id));
        c = *(cp + kind_id_len);
        if ((cplxno == 1 && c != ',') || (cplxno == 2 && c != ')'))
          return;
        num[0] = 0;
        if (dtype == DT_INT8)
          errcode = atoxi64(currc, num, (int)(cp - currc), 10);
        else
          errcode = atosi32(currc, &num[1], (int)(cp - currc), 10);
        tkntyp = TK_K_ICON;
        tknval = getcon(num, dtype);
        goto chk_err;
      }
    }
    goto return_real;
  }
  tkntyp = TK_ICON;
  if (*cp == '_' && (kind_id_len = get_kind_id(cp + 1))) {
    kind_id_len++; /* account for '_' */
    if (kind_id) { /* kind_id is non-zero if ident is a KIND parameter */
      int dtype;
      dtype = select_kind(DT_INT, TY_INT, get_kind_value(kind_id));
      num[0] = 0;
      errcode = atosi32(currc, &num[1], (int)(cp - currc), 10);
      if (dtype == DT_INT8 && (errcode == -2 || errcode == -3))
        errcode = atoxi64(currc, num, (int)(cp - currc), 10);
      tkntyp = TK_K_ICON;
      tknval = getcon(num, dtype);
      goto chk_err;
    }
  } else if (chk_octal && (*cp == 'b' || *cp == 'B')) {
    /* Possible Cray extension for octal constants - octal digits followed
     * by 'b'.
     */
    int len;
    char *p;

    len = cp - currc; /* length of digit string */
    p = currc;
    while (len > 0) { /* determine if all are octal digits */
      if (*p > '7')
        break;
      p++;
      len--;
    }
    if (len == 0) {
      /* Have a cray octal number. Overwrite the 'b' with '"' thus
       * terminating the octal constant for get_nondec().
       */
      if (flg.standard)
        error(170, 2, gbl.lineno,
              "octal constant composed of octal digits followed by 'b'", CNULL);
      *cp = '"';
      get_nondec(8);
      return;
    }
  }
  errcode = atosi32(currc, &tknval, (int)(cp - currc), 10);
  if (kind_id_len == 0 && errcode == 0 && *cp == '#') {
    switch (tknval) {
      char *savep;
    case 2:
    case 8:
    case 10:
    case 16:
      savep = currc;
      currc = cp + 1;
      if (!get_prefixed_int(tknval))
        return;
      currc = savep;
      break;
    }
    tkntyp = TK_ICON;
    error(34, 3, gbl.lineno, "#", CNULL);
    currc = cp + 1; /* skip over # */
    return;
  }
  if (!XBIT(57, 0x2) && (errcode == -2 || errcode == -3)) {
    errcode = atoxi64(currc, num, (int)(cp - currc), 10);
    tkntyp = TK_K_ICON;
    tknval = getcon(num, DT_INT8);
  }
chk_err:
  if (errcode == -1 || errcode == -2)
    CERROR(27, 3, gbl.lineno, currc, cp, CNULL);
  currc = cp + kind_id_len;
  return;

return_real:
  kind = stb.user.dt_real;
  if (*cp == '_' && (kind_id_len = get_kind_id(cp + 1))) {
    if (kind_id) { /* kind_id is non-zero if ident is a KIND parameter */
      kind = select_kind(DT_REAL, TY_REAL, get_kind_value(kind_id));
      if (q_exp || d_exp) /* can't say 'D' or 'Q' and kind */
        error(84, 3, gbl.lineno, SYMNAME(kind_id), "- KIND parameter");
    }
    kind_id_len++; /* account for '_' */
    /* save the string */
    nmptr = putsname(currc, cp - currc + kind_id_len);
    if (XBIT(57, 0x10) && DTY(kind) == TY_QUAD) {
      error(437, 2, gbl.lineno, "Constant with kind type 16 ", "REAL(8)");
      kind = DT_REAL8;
    }
  } else {
    /* constant was not explicitly kinded */
    if (q_exp) {
      kind = DT_QUAD;
      if (XBIT(57, 0x10) && DTY(kind) == TY_QUAD) {
        error(437, 2, gbl.lineno, "Constant with kind type 16 ", "REAL(8)");
        kind = DT_REAL8;
      }
    } else if (d_exp) {
      kind = DT_DBLE;
      if (!XBIT(49, 0x200))
        /* not -dp */
        nmptr = putsname(currc, cp - currc);
      if (XBIT(57, 0x10) && DTY(kind) == TY_QUAD) {
        error(437, 2, gbl.lineno, "DOUBLE PRECISION constant", "REAL");
        kind = DT_REAL;
      }
    } else {
      kind = stb.user.dt_real;
      nmptr = putsname(currc, cp - currc);
    }
  }
  if (cplxno) {
    c = *(cp + kind_id_len);
    if ((cplxno == 1 && c != ',') || (cplxno == 2 && c != ')'))
      return;
  }
  switch (DTY(kind)) {
#ifdef TARGET_SUPPORTS_QUADFP
  /* deal with the quad precision literal constant */
  case TY_QUAD:
    tkntyp = TK_QCON;
    errcode = atoxq(currc, num, (int)(cp - currc));
    switch (errcode) {
    case 0:
      break;
    case -1:
    default:
      CERROR(28, 3, gbl.lineno, currc, cp, CNULL);
      break;
    case -2:
      CERROR(112, 1, gbl.lineno, currc, cp, CNULL);
      break;
    case -3:
      CERROR(111, 1, gbl.lineno, currc, cp, CNULL);
      break;
    }
    sptr = tknval = getcon(num, DT_QUAD);
    break;
#endif
  case TY_DBLE:
    tkntyp = TK_DCON;
    errcode = atoxd(currc, num, (int)(cp - currc));
    switch (errcode) {
    case 0:
      break;
    case -1:
    default:
      CERROR(28, 3, gbl.lineno, currc, cp, CNULL);
      break;
    case -2:
      CERROR(112, 1, gbl.lineno, currc, cp, CNULL);
      break;
    case -3:
      CERROR(111, 1, gbl.lineno, currc, cp, CNULL);
      break;
    }
    sptr = tknval = getcon(num, DT_REAL8);
    break;
  case TY_REAL:
    tkntyp = TK_RCON;
    errcode = atoxf(currc, &tknval, (int)(cp - currc));
    switch (errcode) {
    case 0:
      break;
    case -1:
    default:
      CERROR(28, 3, gbl.lineno, currc, cp, CNULL);
      break;
    case -2:
      CERROR(112, 1, gbl.lineno, currc, cp, CNULL);
      break;
    case -3:
      CERROR(111, 1, gbl.lineno, currc, cp, CNULL);
      break;
    }
    num[0] = 0;
    num[1] = tknval;
    sptr = getcon(num, DT_REAL4);
    break;
  default:
    interr("get_number: can't happen", 0, 4);
    break;
  }
  /* NMPTRP(sptr, nmptr); */
  currc = cp + kind_id_len;
}

static void
get_nondec(int radix)
{
  char *cp;
  int rtn;
  INT val[2];
  int ndig;

  for (cp = currc; *cp != '\'' && *cp != '"'; cp++)
    ;

  tkntyp = TK_NONDEC;
  ndig = cp - currc;

  if (RADIX2_FMT_ERR(radix, ndig) || RADIX8_FMT_ERR(radix, ndig, currc) ||
      RADIX16_FMT_ERR(radix, ndig))
    CERROR(1219, 3, gbl.lineno, "boz feature", cp, currc);

  if ((rtn = atoxi(currc, &tknval, ndig, radix)) < 0) {
    if ((rtn == -1) && (radix == 8)) {
      /* illegal digit */
      tknval = 0;
      CERROR(29, 3, gbl.lineno, "octal", cp, currc);
    } else if ((rtn == -1) && (radix == 16)) {
      /* illegal digit */
      tknval = 0;
      CERROR(29, 3, gbl.lineno, "hexadecimal", cp, currc);
    } else if ((rtn == -1) && (radix == 2)) {
      /* illegal digit */
      tknval = 0;
      CERROR(29, 3, gbl.lineno, "binary", cp, currc);
    } else {
      /* 64-bit nondecimal constant */
      tkntyp = TK_NONDDEC;
      if (radix == 8) {
        rtn = otodi(currc, ndig, val);
        if (rtn == -1) { /* illegal digit */
          val[0] = val[1] = 0;
          CERROR(29, 3, gbl.lineno, "octal", cp, currc);
        }
        if (rtn == -2)
          CERROR(109, 1, gbl.lineno, "octal", cp, currc);
      } else if (radix == 2) {
        rtn = btodi(currc, ndig, val);
        if (rtn == -1) { /* illegal digit */
          val[0] = val[1] = 0;
          CERROR(29, 3, gbl.lineno, "binary", cp, currc);
        }
        if (rtn == -2)
          CERROR(109, 1, gbl.lineno, "binary", cp, currc);
      } else { /* hex digits */
        rtn = htodi(currc, ndig, val);
        if (rtn == -1) { /* illegal digit */
          val[0] = val[1] = 0;
          CERROR(29, 3, gbl.lineno, "hexadecimal", cp, currc);
        }
        if (rtn == -2)
          CERROR(109, 1, gbl.lineno, "hexadecimal", cp, currc);
      }
      tknval = getcon(val, DT_DWORD);
    }
  } else {
    switch (radix) {
    case 2:
      if (ndig > 32)
        goto mk_dw;
      break;
    case 8:
      if (ndig > 10)
        goto mk_dw;
      break;
    case 16:
      if (ndig <= 8)
        break;
    mk_dw:
      val[0] = 0;
      val[1] = tknval;
      tknval = getcon(val, DT_DWORD);
      tkntyp = TK_NONDDEC;
      break;
    }
  }
  currc = cp + 1;
}

/*
 * Extracts radix-prefixed integer constants; unlike get_nondec(), the
 * type of the constants is integer, not 'typeless'.
 */
static int
get_prefixed_int(int radix)
{
  char *cp;
  int rtn;
  INT val[2];
  int errcode;

  switch (radix) {
  case 2:
    for (cp = currc; *cp == '0' || *cp == '1'; cp++)
      ;
    break;
  case 8:
    for (cp = currc; isodigit(*cp); cp++)
      ;
    break;
  case 10:
    for (cp = currc; isdig(*cp); cp++)
      ;
    break;
  case 16:
    for (cp = currc; ishex(*cp); cp++)
      ;
    break;
  }
  if (cp == currc)
    return 1; /* error */

  tkntyp = TK_ICON;
  if (radix == 10) {
    errcode = atosi32(currc, &tknval, (int)(cp - currc), 10);
    if (!XBIT(57, 0x2) && (errcode == -2 || errcode == -3)) {
      errcode = atoxi64(currc, val, (int)(cp - currc), 10);
      tkntyp = TK_K_ICON;
      tknval = getcon(val, DT_INT8);
    }
    if (errcode == -1 || errcode == -2)
      CERROR(27, 3, gbl.lineno, currc, cp, CNULL);
  } else if ((rtn = atoxi(currc, &tknval, (int)(cp - currc), radix)) < 0) {
    if (rtn == -1) {
      interr("get_prefixed_int: bad digit for radix", radix, 4);
    } else {
      /* 64-bit nondecimal constant */
      tkntyp = TK_K_ICON;
      if (radix == 8) {
        rtn = otodi(currc, (int)(cp - currc), val);
        if (rtn == -2)
          CERROR(109, 1, gbl.lineno, "octal", cp, currc);
      } else if (radix == 2) {
        rtn = btodi(currc, (int)(cp - currc), val);
        if (rtn == -2)
          CERROR(109, 1, gbl.lineno, "binary", cp, currc);
      } else { /* hex digits */
        rtn = htodi(currc, (int)(cp - currc), val);
        if (rtn == -2)
          CERROR(109, 1, gbl.lineno, "hexadecimal", cp, currc);
      }
      tknval = getcon(val, DT_INT8);
    }
  }
  if (flg.standard)
    error(170, 2, gbl.lineno, "radix prefixed constant", CNULL);
  currc = cp;
  return 0;
}

/* Convert hex digits to a 64-bit constant.  Hex digits must be larger
 * than 32-bits. Returns -2 for overflow, -1 for an illegal digit.
 * On overflow truncation occurs at left, with most significant digits.
 */
static int
htodi(char *s, int l, INT *num)
{
  int i, j, k;
  char *end;
  int v;

  i = k = 0;
  j = 1;
  num[1] = num[0] = 0;
  end = s + l;

  while (i < l) {
    if ((num[0] & 0xF0000000L) != 0)
      return (-2);

    v = hex_to_i((int)*(end - i - 1));
    if (v == -1)
      return (-1);
    num[j] |= v << (4 * k);
    i++;
    k++;
    if (i == 8) {
      j--;
      k = 0;
    }
  } /* while */
  return (0);
}

static int
hex_to_i(int c)
{
  int v;

  if (c >= '0' && c <= '9')
    v = (c - '0');
  else if (c >= 'A' && c <= 'F')
    v = (c - 'A' + 10);
  else if (c >= 'a' && c <= 'f')
    v = (c - 'a' + 10);
  else
    v = -1;
  return v;
}

/* Convert octal digits to a 64-bit constant.  Octal digits must be larger
 * than 32-bits. Returns -2 for overflow, -1 for an illegal digit.
 * On overflow truncation occurs at left, with most significant digits.
 */
static int
otodi(char *s, int l, INT *num)
{
  int i, save;
  char *end;

  for (i = 0; i < l; i++)
    if (*(s + i) < '0' || *(s + i) > '7')
      return (-1);

  end = s + l;
  num[1] = num[0] = 0;

  /* low-order 10 digits go to low-order 60 bits of low-order word */
  for (i = 0; i < 10 && l > 0; ++i, --l) {
    --end;
    num[1] |= ((*end) - '0') << (3 * i);
  }

  if (l > 0) {
    /* 11-th digit is split: 2 bits to low-order word, 1 bit to high
     * order word */
    --end;
    save = (*end) - '0';
    num[1] |= (save & 03) << 30;
    --l;
    num[0] = save >> 2;
    for (i = 0; i < 10 && l > 0; ++i, --l) {
      --end;
      num[0] |= ((*end) - '0') << (3 * i + 1);
    }
    if (l > 0) {
      /* if there is still another digit, it can only be a '1' */
      --end;
      --l;
      save = (*end) - '0';
      num[0] |= (save & 01) << 31;
      if (save > 1)
        return -2;
    }
    /* can't handle any more digits, have handled all digits
     * to this point correctly, though */
    if (l > 0)
      return -2;
  }
  return 0;
}

/* Convert binary digits to a 64-bit constant.  Binary digits must be larger
 * than 32-bits. Returns -2 for overflow, -1 for an illegal digit.
 * On overflow truncation occurs at left, with most significant digits.
 */
static int
btodi(char *s, int l, INT *num)
{
  int i, j, k;
  char *end;
  char c;

  i = k = 0;
  j = 1;
  num[1] = num[0] = 0;
  end = s + l;

  while (i < l) {
    if ((num[0] & 0x80000000L) != 0)
      return (-2);

    c = *(end - i - 1);
    if (c != '0' && c != '1')
      return (-1);
    if (c == '1')
      num[j] |= 1 << k;
    i++;
    k++;
    if (i == 32) {
      j--;
      k = 0;
    }
  } /* while */
  return (0);
}

/*  A left paren has been reached.  Scan ahead to determine if it is
    the beginning of a complex constant.  If it is a complex constant,
    enter it into symtab and return complex constant token, else just
    return left parenthesis token.
*/
static void
check_ccon(void)
{
  char c, *save_currc, *nextc;
  INT num[4], val[4];
#ifdef TARGET_SUPPORTS_QUADFP
  INT val1[2];
#endif
  int tok1;
  int rdx, idx;
  INT swp;

  save_currc = currc;
  switch (scmode) {
  case SCM_IDENT:
  case SCM_IF:
  case SCM_LOOKFOR_OPERATOR:
    break;
  case SCM_CHEVRON:
    scmode = SCM_IDENT;
    goto return_paren;
  default:
    goto return_paren;
  }
  c = *(currc - 2);
  if (isident(c))
    goto return_paren;
  if (c == ' ' && (is_freeform || scn.is_hpf)) {
    char *p;
    p = currc - 2;
    while (p > stmtb) {
      c = *--p;
      if (isident(c))
        goto return_paren;
      if (c != ' ')
        break;
    }
  }

  c = *currc;
  if (c == ' ')
    c = *++currc;
  nextc = currc + 1;
  if (c == '+' || c == '-') {
    c = *nextc;
    nextc++;
  }
  if (!isdig(c) && c != '.')
    goto return_paren;
  if (c == '.' && !isdig(*nextc))
    goto return_paren;
  tkntyp = 0;
  get_number(1);
  if (scnerrfg)
    return;
  if (tkntyp == 0)
    goto return_paren;
  num[0] = tknval;
  tok1 = tkntyp; /* remember first token type - real, double, or TK_K_ICON */
  assert(*currc == ',', "check_ccon: not comma", (int)*currc, 3);
  c = *++currc;
  if (c == ' ')
    c = *++currc;
  nextc = currc + 1;
  if (c == '+' || c == '-') {
    c = *nextc;
    nextc++;
  }
  if (!isdig(c) && c != '.')
    goto return_paren;
  if (c == '.' && !isdig(*nextc))
    goto return_paren;
  tkntyp = 0;
  get_number(2);
  if (scnerrfg)
    return;
  if (tkntyp == 0)
    goto return_paren;
  num[1] = tknval;
  currc++;

  /* check real and imaginary parts to ensure identical types  */

  /* Note: we don't try to save the string for the whole constant */
  rdx = 0;
  idx = 1;
  if (tok1 == TK_K_ICON || tkntyp == TK_K_ICON) {
    if (tok1 != TK_K_ICON) {
      /*  swap to put in form  k_icon, xxx */
      swp = num[0];
      num[0] = num[1];
      num[1] = swp;
      swp = tok1;
      tok1 = tkntyp;
      tkntyp = swp;
      rdx = 1;
      idx = 0;
    }
    if (DTYPEG(num[0]) != DT_INT8)
      val[0] = CONVAL2G(num[0]);
    else
      val[0] = num[0];
    switch (tkntyp) {
    case TK_K_ICON:
      if (DTYPEG(num[1]) != DT_INT8)
        val[1] = CONVAL2G(num[1]);
      else
        val[1] = num[1];
      num[0] = cngcon(val[0], DTYPEG(num[0]), stb.user.dt_real);
      num[1] = cngcon(val[1], DTYPEG(num[1]), stb.user.dt_real);
      switch (DTY(stb.user.dt_real)) {
      case TY_DBLE:
        tkntyp = TK_DCON;
        break;
      default:
        tkntyp = TK_RCON;
        break;
      }
      break;
    case TK_RCON:
      num[0] = cngcon(val[0], DTYPEG(num[0]), DT_REAL4);
      break;
    case TK_DCON:
      num[0] = cngcon(val[0], DTYPEG(num[0]), DT_REAL8);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TK_QCON:
      num[0] = cngcon(val[0], DTYPEG(num[0]), DT_QUAD);
      break;
#endif
    default:
      interr("check_ccon: unexp.constant", tkntyp, 3);
      tkntyp = TK_RCON;
      break;
    }
    val[0] = num[rdx];
    val[1] = num[idx];
    switch (tkntyp) {
    case TK_RCON:
      tkntyp = TK_CCON;
      tknval = getcon(val, DT_CMPLX);
      break;
    case TK_DCON:
      tkntyp = TK_DCCON;
      tknval = getcon(val, DT_CMPLX16);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TK_QCON:
      tkntyp = TK_QCCON;
      tknval = getcon(val, DT_QCMPLX);
      break;
#endif
    }
  } else {
#ifdef TARGET_SUPPORTS_QUADFP
    if (tok1 == TK_QCON) {
      if (tkntyp == TK_RCON) { /* (quad, real)  */
        xftoq(num[1], val);
        num[1] = getcon(val, DT_QUAD);
      } else if (tkntyp == TK_DCON) { /* (quad, double)  */
        val1[0] = CONVAL1G(num[1]);
        val1[1] = CONVAL2G(num[1]);
        xdtoq(val1, val);
        num[1] = getcon(val, DT_QUAD);
      }
      tkntyp = TK_QCCON;
      tknval = getcon(num, DT_QCMPLX);
      return;
    } else
#endif
    if (tok1 == TK_DCON) {
      if (tkntyp == TK_RCON) { /* (double, real)  */
        xdble(num[1], val);
        num[1] = getcon(val, DT_DBLE);
#ifdef TARGET_SUPPORTS_QUADFP
      } else if (tkntyp == TK_QCON) { /* (double, quad)  */
        val1[0] = CONVAL1G(num[0]);
        val1[1] = CONVAL2G(num[0]);
        xdtoq(val1, val);
        num[0] = getcon(val, DT_QUAD);
        tkntyp = TK_QCCON;
        tknval = getcon(num, DT_QCMPLX);
        return;
#endif
      }
    } else if (tkntyp == TK_RCON) { /* (real, real)  */
      tkntyp = TK_CCON;
      tknval = getcon(num, DT_CMPLX);
      /**  NOTE:  "name" includes parens **/
      NMPTRP(tknval, putsname(save_currc - 1, currc - save_currc + 1));
      return;
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (tkntyp == TK_QCON) { /* (real, quad) */
      xftoq(num[0], val);
      num[0] = getcon(val, DT_QUAD);
      tkntyp = TK_QCCON;
      tknval = getcon(num, DT_QCMPLX);
      return;
#endif
    } else { /* (real, double)  */
      xdble(num[0], val);
      num[0] = getcon(val, DT_DBLE);
    }
    tkntyp = TK_DCCON;
    tknval = getcon(num, DT_CMPLX16);
  }
  return;

return_paren:
  currc = save_currc;
  tkntyp = TK_LPAREN; /* add as case in _rd_token() */
  par_depth++;
}

/*  A dot (.) has been reached.  The token is either a keyword
    enclosed in dots, a real constant, or just a dot.
*/
static void
do_dot(void)
{
  int len;
  char c = *currc;
  INT val[2];

  if (scmode != SCM_FORMAT) {
    if (isalpha(c)) {
      len = is_ident(currc);
      /*
       * Check for user defined operators first.  If one of the
       * non-standard 'logical keywords' is declared as a user
       * defined operator, we do not want to return it as an intrinsic
       * operator or logical constant.
       */
      if (currc[len] == '.') {
        int sptr = lookupsym(currc, len);
        if (sptr) {
          sptr = refocsym(sptr, OC_OPERATOR);
          if (sptr && STYPEG(sptr) == ST_OPERATOR) {
            currc += (len + 1);
            tkntyp = TK_DEFINED_OP;
            tknval = sptr;
            return;
          }
        }
      }
      tkntyp = keyword(currc, &logicalkw, &len, TRUE);
      if (tkntyp != 0 && currc[len] == '.') {
        /*
         * If this identifier is expected to be defined as an OPERATOR,
         * any nonstandard 'logical' keyword must be returned as the
         * sequence of tokens '.' <ident> '.';  the logical keyword
         * must not be returned as an intrinsic operator or logical
         * constant.
         */
        if (scmode == SCM_OPERATOR && t2[keyword_idx].nonstandard) {
          tkntyp = TK_DOT;
          return;
        }
        if (tkntyp == TK_LOGCONST) {
          char *cp;
          int kind_id_len;
          cp = currc + (len + 1);
          tknval = (c == 't' ? SCFTN_TRUE : SCFTN_FALSE);
          if (*cp == '_' && (kind_id_len = get_kind_id(cp + 1))) {
            /* kind_id is non-zero if ident is a KIND parameter */
            if (kind_id) {
              int dtype;
              dtype = select_kind(DT_LOG, TY_LOG, get_kind_value(kind_id));
              if (dtype != DT_LOG8) {
                val[0] = 0;
                val[1] = tknval;
              } else {
                if (c != 't')
                  val[0] = val[1] = 0;
                else if (gbl.ftn_true == -1)
                  val[0] = val[1] = -1;
                else {
                  val[0] = 0;
                  val[1] = 1;
                }
              }
              tknval = getcon(val, dtype);
              tkntyp = TK_K_LOGCONST;
            }
            kind_id_len++; /* account for '_' */
            len += kind_id_len;
          }
        }
        currc += (len + 1);
        return;
      }
    } else if (isdig(c)) {
      currc--;
      get_number(0);
      return;
    } else if (c == '.') {
      currc++;
      tkntyp = TK_DOTDOT;
      return;
    }
  }
  tkntyp = TK_DOT;
}

/* for supporting file $INSERT directive */
static void
push_include(char *p)
{
  char *fullname, *begin;
  char c;
  FILE *tmp_fd;

  while (*p == ' ' || *p == '\t')
    ++p;
  c = *p++; /* delimiter character */
  if (c != '\'' && c != '"') {
    c = '\n';
    p--;
  }
  begin = p;
  while (*p != c && *p != '\n')
    ++p;
  *p = '\0';

  fullname = getitem(8, MAX_PATHNAME_LEN + 1);
  if (incl_level < MAX_IDEPTH) {
    if (flg.idir) {
      for (c = 0; (p = flg.idir[c]); ++c)
        if (fndpath(begin, fullname, MAX_PATHNAME_LEN, p) == 0)
          goto found;
    }
    if (fndpath(begin, fullname, MAX_PATHNAME_LEN, DIRWORK) == 0)
      goto found;
    if (flg.stdinc == 0) {
      if (fndpath(begin, fullname, MAX_PATHNAME_LEN, DIRSINCS) == 0)
        goto found;
    } else if (flg.stdinc != (char *)1) {
      if (fndpath(begin, fullname, MAX_PATHNAME_LEN, flg.stdinc) == 0)
        goto found;
    }
    goto not_found;
  found:
    in_include = TRUE;
    NEED(incl_level + 1, incl_stack, ISTACK, incl_stacksz, incl_level + 3);
    incl_stack[incl_level].fd = curr_fd;
    incl_stack[incl_level].findex = gbl.findex;
    incl_stack[incl_level].lineno = curr_line;
    incl_stack[incl_level].fname = gbl.curr_file;
    incl_stack[incl_level].list_now = list_now;
    incl_stack[incl_level].card_type = CT_NONE; /* no look ahead */
    incl_stack[incl_level].sentinel = SL_NONE;  /* no look ahead */
    incl_stack[incl_level].eof_flag = gbl.eof_flag;
    incl_stack[incl_level].is_freeform = is_freeform;
    gbl.eof_flag = FALSE;

    tmp_fd = fopen(fullname, "r");
    if (tmp_fd != NULL) {
      curr_fd = tmp_fd;
      ++incl_level;
      add_headerfile(fullname, 1, 0);
      if (!XBIT(120, 0x4000000)) {
        gbl.findex = hdr_stack[hdr_level - 1].findex;
        gbl.curr_file = hdr_stack[hdr_level - 1].fname;
      } else {
        gbl.curr_file = fullname;
      }
      curr_line = 0;
      if (flg.list)
        list_line(fullname);
      list_now = flg.list;
      put_include(FR_B_INCL, gbl.findex);
      /* -M option:  Print list of include files to stdout */
      /* -MD option:  Print list of include files to file <program>.d */
      if (XBIT(123, 2) || XBIT(123, 8)) {
        if (gbl.dependfil == NULL) {
          if ((gbl.dependfil = tmpfile()) == NULL)
            errfatal(5);
        } else
          fprintf(gbl.dependfil, "\\\n  ");
        if (!XBIT(123, 0x40000))
          fprintf(gbl.dependfil, "%s ", fullname);
        else
          fprintf(gbl.dependfil, "\"%s\" ", fullname);
      }
      return;
    }
  }
not_found:
  /* file not found, nesting depth exceeded, unable to open: */
  error(17, 3, curr_line, begin, CNULL);
}

/* for supporting VAX-style include statement */
void
scan_include(char *str)
{
  char *p;
  char *fullname, *dirname;
  char c;
  LOGICAL new_list_now;
  FILE *tmp_fd;
  char *q;

  if (sem.which_pass != 0 || sem.mod_cnt > 1)
    return;
  /*
   * str locates string as stored by symtab (get_string); presumably
   * it's ok to use the same space to compress the string.
   */
  p = q = str;
  while ((c = *p++)) {
    if (c == ' ' || c == '\t')
      continue;
    *q++ = c;
  }
  *q = '\0';
  new_list_now = list_now;

  /* look for /list or /nolist at the end - sets new_list_now */
  q--;
  if ((*q == 't' || *q == 'T') && (*--q == 's' || *q == 'S') &&
      (*--q == 'i' || *q == 'I') && (*--q == 'l' || *q == 'L')) {
    if (*--q == '/') {
      new_list_now = flg.list;
      *q = '\0';
    } else if ((*q == 'o' || *q == 'O') && (*--q == 'n' || *q == 'N') &&
               (*--q == '/')) {
      new_list_now = FALSE;
      *q = '\0';
    }
  }
  fullname = getitem(8, MAX_PATHNAME_LEN + 1);
  dirname = getitem(8, MAX_PATHNAME_LEN + 1);
  if (incl_level < MAX_IDEPTH) {
    if (flg.idir) {
      for (c = 0; (p = flg.idir[c]); ++c)
        if (fndpath(str, fullname, MAX_PATHNAME_LEN, p) == 0)
          goto found;
    }
    dirnam(gbl.curr_file, dirname);
    if (fndpath(str, fullname, MAX_PATHNAME_LEN, dirname) == 0)
      goto found;
    if (fndpath(str, fullname, MAX_PATHNAME_LEN, DIRWORK) == 0)
      goto found;
    if (flg.stdinc == 0) {
      if (fndpath(str, fullname, MAX_PATHNAME_LEN, DIRSINCS) == 0)
        goto found;
    } else if (flg.stdinc != (char *)1) {
      if (fndpath(str, fullname, MAX_PATHNAME_LEN, flg.stdinc) == 0)
        goto found;
    }
    goto not_found;
  found:
    in_include = TRUE;
    NEED(incl_level + 1, incl_stack, ISTACK, incl_stacksz, incl_level + 3);
    incl_stack[incl_level].fd = curr_fd;
    incl_stack[incl_level].findex = gbl.findex;
    incl_stack[incl_level].lineno = curr_line;
    incl_stack[incl_level].fname = gbl.curr_file;
    incl_stack[incl_level].list_now = list_now;
    incl_stack[incl_level].card_type = card_type;
    incl_stack[incl_level].sentinel = sentinel;
    incl_stack[incl_level].first_char = first_char;
    incl_stack[incl_level].eof_flag = gbl.eof_flag;
    incl_stack[incl_level].is_freeform = is_freeform;
    gbl.eof_flag = FALSE;
    BCOPY(incl_stack[incl_level].cardb, cardb, char, CARDB_SIZE);
    tmp_fd = fopen(fullname, "r");
    if (tmp_fd != NULL) {
      curr_fd = tmp_fd;
      ++incl_level;
      add_headerfile(fullname, 1, 0);
      if (!XBIT(120, 0x4000000)) {
        gbl.findex = hdr_stack[hdr_level - 1].findex;
        gbl.curr_file = hdr_stack[hdr_level - 1].fname;
      } else {
        gbl.curr_file = fullname;
      }
      curr_line = 0;
      if (flg.list)
        list_line(fullname);
      list_now = new_list_now;
      put_include(FR_B_INCL, gbl.findex);
      card_type = (*p_read_card)(); /* initiate one card look ahead */
      /* -M option:  Print list of include files to stdout */
      /* -MD option:  Print list of include files to file <program>.d */
      if (XBIT(123, 2) || XBIT(123, 8)) {
        if (gbl.dependfil == NULL) {
          if ((gbl.dependfil = tmpfile()) == NULL)
            errfatal(5);
        } else
          fprintf(gbl.dependfil, "\\\n  ");
        if (!XBIT(123, 0x40000))
          fprintf(gbl.dependfil, "%s ", fullname);
        else
          fprintf(gbl.dependfil, "\"%s\" ", fullname);
      }
      return;
    }
  }
not_found:
  /* file not found, nesting depth exceeded, unable to open: */
  error(17, 3, gbl.lineno, str, CNULL);
}

/* Define structures and macros for OPTIONS processing: */

static int options_seen = FALSE; /* TRUE if OPTIONS seen in prev. subpg. */
struct c {
  const char *cmd;
  INT caselabel;
};
static int getindex(struct c *, int, char *);

#define SW_CHECK 1
#define SW_NOCHECK 2
#define SW_EXTEND 3
#define SW_NOEXTEND 4
#define SW_GFLOAT 5
#define SW_NOGFLOAT 6
#define SW_I4 8
#define SW_NOI4 9
#define SW_RECURS 10
#define SW_NORECURS 11
#define SW_STANDARD 12
#define SW_NOSTANDARD 13
#define SW_REENTR 14
#define SW_NOREENTR 15

/* cmd field (switch names) must be in alphabetical order for search */
static struct c swtchtab[] = {
    {"check", SW_CHECK},
    {"extend_source", SW_EXTEND},
    {"f77", SW_STANDARD},
    {"g_floating", SW_GFLOAT},
    {"i4", SW_I4},
    {"nocheck", SW_NOCHECK},
    {"noextend_source", SW_NOEXTEND},
    {"nof77", SW_NOSTANDARD},
    {"nog_floating", SW_NOGFLOAT},
    {"noi4", SW_NOI4},
    {"norecursive", SW_NORECURS},
    {"noreentrant", SW_NOREENTR},
    {"nostandard", SW_NOSTANDARD},
    {"recursive", SW_RECURS},
    {"reentrant", SW_REENTR},
    {"standard", SW_STANDARD},
};
#define NSWDS (sizeof(swtchtab) / sizeof(struct c))

static struct { /* flg values which can appear after OPTIONS */
  int extend_source;
  LOGICAL i4;
  LOGICAL standard;
  LOGICAL recursive;
  int x7;
  int x70;
} save_flg;

void
scan_opt_restore(void)
{
  if (options_seen) {
    flg.extend_source = save_flg.extend_source;
    flg.i4 = save_flg.i4;
    flg.standard = save_flg.standard;
    flg.recursive = save_flg.recursive;
    flg.x[7] = save_flg.x7;
    flg.x[70] = save_flg.x70;
    body_len = flg.extend_source - 5;
    options_seen = FALSE;
  }
}

/* for supporting VAX-style options statement */
void
scan_options(void)
{
  char *p;
  char *argstring;
  int indice;
  char savec;

  if (DBGBIT(1, 1)) {
    fprintf(gbl.dbgfil, "%%");
    fprintf(gbl.dbgfil, "%s", scn.options);
    fprintf(gbl.dbgfil, "%%\n");
  }
  options_seen = TRUE;
  save_flg.extend_source = flg.extend_source;
  save_flg.i4 = flg.i4;
  save_flg.standard = flg.standard;
  save_flg.recursive = flg.recursive;
  save_flg.x7 = flg.x[7];
  save_flg.x70 = flg.x[70];

  /* loop thru OPTIONS flags: */
  savec = *scn.options;
  for (argstring = scn.options; savec; argstring = p + 1) {
    for (p = argstring + 1;; p++)
      if (*p == '/' || *p == '\0' || *p == '=') {
        savec = *p;
        *p = '\0';
        break;
      }

    indice = getindex(swtchtab, NSWDS, argstring);

    if (indice < 0)
      goto opt_error;

    switch (swtchtab[indice].caselabel) {
    case SW_CHECK:
      if (savec == '=') {
        do {
          argstring = ++p;
          while (TRUE) {
            p++;
            if (*p == ',' || *p == '/' || *p == '\0') {
              savec = *p;
              *p = '\0';
              break;
            }
          }
          if (strcmp(argstring, "all") == 0)
            flg.x[70] |= 0x2;
          else if (strcmp(argstring, "bounds") == 0)
            flg.x[70] |= 0x2;
          else if (strcmp(argstring, "nobounds") == 0)
            flg.x[70] &= (~0x2);
          else if (strcmp(argstring, "none") == 0)
            flg.x[70] &= (~0x2);
          else if (strcmp(argstring, "nooverflow") == 0)
            ; /* NO-OP */
          else if (strcmp(argstring, "nounderflow") == 0)
            ; /* NO-OP */
          else if (strcmp(argstring, "overflow") == 0)
            ; /* NO-OP */
          else if (strcmp(argstring, "underflow") == 0)
            ; /* NO-OP */
          else {
            error(197, 3, 0, "check", CNULL);
            break;
          }
        } while (savec == ',');
        break;
      }
      /* /CHECK no value */
      flg.x[70] |= 0x2;
      break;
    case SW_NOCHECK:
      flg.x[70] &= (~0x2);
      break;
    case SW_EXTEND:
      /*
       * since line is being extended, as a precaution the character
       * overwritten to define the absolute end-of-line by read_card()
       * is restored.
       * Also, the print buffer is filled after the current card/line
       * is in the extended state.
       */
      cardb[save_flg.extend_source] = save_extend_ch;
      flg.extend_source = 132;
      body_len = flg.extend_source - 5;
      cardb[132] = '\n';
      if (card_type != CT_DIRECTIVE)
        write_card();
      break;
    case SW_NOEXTEND:
      /*
       * since the length of the line is reduced, don't care about
       * restoring the character overwritten (it's either at the
       * same position or later in the line).  Do care about
       * filling the print buffer.
       */
      flg.extend_source = 72;
      body_len = flg.extend_source - 5;
      cardb[72] = '\n';
      if (card_type != CT_DIRECTIVE)
        write_card();
      break;
    case SW_GFLOAT: /* NO-OP */
    case SW_NOGFLOAT:
      break;
    case SW_I4:
      flg.i4 = TRUE;
      implicit_int(DT_INT); /* call routine in symtab.c */
      break;
    case SW_NOI4:
      flg.i4 = FALSE;
      implicit_int(DT_SINT); /* call routine in symtab.c */
      break;
    case SW_RECURS:
      flg.recursive = TRUE;
      break;
    case SW_NORECURS:
      flg.recursive = FALSE;
      break;
    case SW_STANDARD:
      flg.standard = TRUE;
      symtab_standard();
      break;
    case SW_NOSTANDARD:
      flg.standard = FALSE;
      symtab_nostandard();
      break;
    case SW_REENTR:
      flg.x[7] |= 0x2;      /* inhibit terminal func optz. */
      flg.recursive = TRUE; /* no static locals */
      break;
    case SW_NOREENTR:
      flg.x[7] &= ~(0x2);
      flg.recursive = FALSE;
      break;
    default:
      goto opt_error;
    } /* end switch */
    if (savec == '=')
      goto opt_error;
    continue;

  opt_error:
    error(197, 3, 0, argstring, CNULL);
    if (savec != '=')
      continue;

    while (TRUE) {
      p++;
      if (*p == '/' || *p == '\0') {
        savec = *p;
        *p = '\0';
        break;
      }
    }
  }
}

/*
 *  getindex()
 *     Sequentially searches table[].cmd for elements with prefix string.
 *     Returns   index if found,  -1 if not found , -2 if matches  >1 elements
 *     NOTE: table must be in lexic. order to find duplicate prefix matches
 */
static int
getindex(struct c *table, int num_elem, char *string)
{
  int i;
  int l;
  int fnd;
  int len;

  l = -1;
  fnd = -1;
  i = 0;
  len = strlen(string);
  while ((i < num_elem) && ((l = strncmp(string, table[i].cmd, len)) > 0)) {
    i++;
  }
  if (!l) {
    if (len == strlen(table[i].cmd))
      fnd = i;
    /* check next value to see if it matches, too */
    else if ((++i < num_elem) &&
             ((l = strncmp(string, table[i].cmd, len)) == 0))
      fnd = -2;
    else /* found unique match */
      fnd = --i;
  }

  return (fnd);
}

static void
put_astfil(int type, char *line, LOGICAL newline)
{
  int nw;

  nw = fwrite((char *)&type, sizeof(int), 1, astb.astfil);
  if (nw != 1)
    error(10, 4, 0, "(AST file)", CNULL);
  if (line != NULL)
    fputs(line, astb.astfil);
  if (newline)
    fputc('\n', astb.astfil);
}

static void
put_lineno(int lineno)
{
  static int type = FR_LINENO;
  int nw;

  gbl.lineno = lineno;
  nw = fwrite((char *)&type, sizeof(int), 1, astb.astfil);
  if (nw != 1)
    error(10, 4, 0, "(AST file)", CNULL);
  fprintf(astb.astfil, "%d\n", lineno);
}

static void
put_include(int type, int findex)
{
  int nw;

  nw = fwrite((char *)&type, sizeof(int), 1, astb.astfil);
  if (nw != 1)
    error(10, 4, 0, "(AST file)", CNULL);
  nw = fwrite((char *)&findex, sizeof(int), 1, astb.astfil);
  if (nw != 1)
    error(10, 4, 0, "(AST file)", CNULL);
}

/*  read one Fortran statement, including continuations
    into stmtb.  Process directive lines if encountered.  Skip past
    comment lines.  Handle end of files.  Extract labels from initial lines.
    Write lines to source listing.
*/
static void
ff_get_stmt(void)
{
  char *p;
  int c;

  card_count = 0;
  ff_state.cavail = &stmtb[0];
  scn.is_hpf = FALSE;
  is_smp = FALSE;
  is_sgi = FALSE;
  is_dec = FALSE;
  is_mem = FALSE;
  is_ppragma = FALSE;
  is_pgi = FALSE;
  is_kernel = FALSE;
  is_doconcurrent = false;

  for (p = printbuff + 8; *p != '\0' && (isblank(*p));) {
    ++p;
  }
  leadCount = p - (printbuff + 8);

  do {
  again:
    switch (card_type) {
    case CT_COMMENT:
      put_astfil(curr_line, &printbuff[8], TRUE);
      break;

    case CT_EOF:
      /* pop include  */
      if (incl_level > 0) {
        const char *save_filenm;

        incl_level--;
        if (!incl_stack[incl_level].is_freeform) {
          set_input_form(FALSE);
          incl_level++;
          get_stmt();
          return;
        }
        save_filenm = gbl.curr_file;
        curr_fd = incl_stack[incl_level].fd;
        gbl.findex = incl_stack[incl_level].findex;
        curr_line = incl_stack[incl_level].lineno;
        gbl.curr_file = incl_stack[incl_level].fname;
        list_now = incl_stack[incl_level].list_now;
        gbl.eof_flag = incl_stack[incl_level].eof_flag;
        if (curr_line == 1)
          add_headerfile(gbl.curr_file, curr_line + 1, 0);
        else
          add_headerfile(gbl.curr_file, curr_line, 0);

        put_include(FR_E_INCL, gbl.findex);

        card_type = incl_stack[incl_level].card_type;
        sentinel = incl_stack[incl_level].sentinel;
        if (card_type != CT_NONE) {
          first_char = incl_stack[incl_level].first_char;
          BCOPY(cardb, incl_stack[incl_level].cardb, char, CARDB_SIZE);
          if (card_type != CT_DIRECTIVE)
            write_card();
          if (card_type == CT_EOF && incl_level == 0) {
            if (gbl.currsub || sem.mod_sym) {
              gbl.curr_file = save_filenm;
              sem.mod_cnt = 0;
              sem.mod_sym = 0;
              sem.submod_sym = 0;
              errsev(22);
            }
            finish();
          }
        } else
          card_type = ff_read_card();
        if (incl_level == 0)
          in_include = FALSE;
        if (card_type == CT_EOF && incl_level <= 0)
          errsev(22);
        else
          goto again;
      }
      /* terminate compilation:  */
      if (sem.mod_sym) {
        errsev(22);
        sem.mod_cnt = 0;
        sem.mod_sym = 0;
        sem.submod_sym = 0;
      }
      finish();
      FLANG_FALLTHROUGH;
    case CT_DIRECTIVE:
      put_astfil(curr_line, &printbuff[8], TRUE);
      put_lineno(curr_line);
      /* convert upper case letters to lower:  */
      for (p = &cardb[1]; (c = *p) != ' ' && c != '\n'; ++p)
        if (c >= 'A' && c <= 'Z')
          *p = tolower(c);
      if (strncmp(&cardb[1], "list", 4) == 0)
        list_now = flg.list;
      else if (strncmp(&cardb[1], "nolist", 6) == 0)
        list_now = FALSE;
      else if (strncmp(&cardb[1], "eject", 5) == 0) {
        if (list_now)
          list_page();
      } else if (strncmp(&cardb[1], "insert", 6) == 0)
        push_include(&cardb[8]);
      else /* unrecognized directive:  */
        errsev(20);
      break;

    case CT_LINE:
      line_directive();
      card_type = CT_COMMENT;
      break;

    case CT_PRAGMA:
      put_astfil(curr_line, &printbuff[8], TRUE);
      no_crunch = TRUE;
      if (card_count == 0) {
        if (hdr_level == 0)
          fihb.currfindex = gbl.findex = 1;
        else
          fihb.currfindex = gbl.findex = hdr_stack[hdr_level - 1].findex;
        gbl.curr_file = FIH_FULLNAME(gbl.findex);
      }
      card_count = 1;
      put_lineno(curr_line);
      p = first_char;
      *ff_state.cavail++ = CH_PRAGMA;
      while ((*ff_state.cavail++ = *p++) != '\n')
        ;
      card_type = CT_INITIAL; /* trick rest of processing */
      break;

    case CT_FIXED:
      set_input_form(FALSE);
      card_type = CT_COMMENT;
      get_stmt();
      return;

    case CT_FREE:
      put_astfil(curr_line, &printbuff[8], TRUE);
      set_input_form(TRUE);
      card_type = CT_COMMENT;
      break;

    case CT_DEC:
      is_dec = TRUE;
      goto initial_card;
    case CT_MEM:
      is_mem = TRUE;
      goto initial_card;
    case CT_PPRAGMA:
      is_ppragma = TRUE;
      goto initial_card;
    case CT_PGI:
      is_pgi = TRUE;
      goto initial_card;
    case CT_KERNEL:
      is_kernel = TRUE;
      goto initial_card;
    case CT_SMP:
      is_smp = TRUE;
      is_sgi = sentinel == SL_SGI;
      FLANG_FALLTHROUGH;
    case CT_INITIAL:
    initial_card:
      gbl.in_include = in_include;
      put_astfil(curr_line, &printbuff[8], TRUE);
      if (card_count == 0) {
        if (hdr_level == 0)
          fihb.currfindex = gbl.findex = 1;
        else
          fihb.currfindex = gbl.findex = hdr_stack[hdr_level - 1].findex;
        gbl.curr_file = FIH_FULLNAME(gbl.findex);
      }
      card_count = 1;
      put_lineno(curr_line);

      ff_prescan();
      card_type = CT_INITIAL;
      break;

    case CT_CONTINUATION:
      if (sentinel == SL_SGI) {
        /* sgi continuation - we reach this point if the previous
         * statement isn't continued in usual f90 manner, i.e., the
         * last nonblank character isn't '&'.  If the f90 style is
         * used, the above call to ff_prescan() processes the
         * continuation.
         */
        put_astfil(curr_line, &printbuff[8], TRUE);
        card_count++;
        ff_check_stmtb();
        ff_prescan();
      } else
        error(290, 3, curr_line, CNULL, CNULL);
      break;

    default:
      interr("ff_get_stmt: bad ctype", card_type, 4);
    }
    /* start new listing page if at END, then read new card: */

    if (flg.list && card_type <= CT_COMMENT) {
      if (list_now)
        list_line(printbuff);
    }
#if DEBUG
    if (DBGBIT(4, 2))
      fprintf(gbl.dbgfil, "line(%4d) %s", curr_line, cardb);
#endif
    card_type = ff_read_card();
  } while (ff_state.cavail == stmtb || card_type == CT_CONTINUATION ||
           card_type == CT_COMMENT || card_type == CT_LINE /* tpr 533 */
           );
}

/*  read one input line into cardb, and determine its type
    (card_type) and determine first character following the
    label field (first_char).
*/
static int
ff_read_card(void)
{
  int c;
  int i;
  char *p;      /* pointer into cardb */
  char *firstp; /* pointer which locates first nonblank char */
  int ct;

  assert(!gbl.eof_flag, "ff_read_card:err", gbl.eof_flag, 4);
  sentinel = SL_NONE;

  p = _readln(MAX_COLS, TRUE);
  if (p == NULL)
    return CT_EOF;
  first_char = cardb;
  ff_state.last_char = (p - cardb);

  if (*cardb == '#') {
    if (first_line && !fpp_) {
      get_fn();
    }
    first_line = FALSE;
    return CT_LINE;
  }
  first_line = FALSE;
  c = cardb[0];
  if (c == '%')
    return CT_DIRECTIVE;
  if (c == '$') /* APFTN64 style of directives */
    return CT_DIRECTIVE;
  write_card();
  for (p = cardb; isblank(*p); p++)
    ;
  first_char = firstp = p; /* first non-blank character in stmt */
  c = *p;
  if (c == '\n')
    return CT_COMMENT;
  ct = CT_INITIAL;
  if (c == '!') {
/* possible compiler directive. these directives begin with (upper
 * or lower case):
 *     c$pragma
 *     cpgi$  cvd$  cdir$
 * to check for a directive, all that's done is to copy at most N
 * characters after the leading 'c', where N is the max length of
 * the allowable prefixes, converting to lower case if necessary.
 * if the prefix matches one of the above, a special card type
 * is returned.   NOTE: can't process the directive now since
 * this card represents the read-ahead ---- NEED to ensure that
 * semantic actions are performed.
 */
#define MAX_DIRLEN 4
    char b[MAX_DIRLEN + 1], cc;

    /* sun's c$pragma is separate from those whose prefixes end with $ */
    if (p[1] == '$' && (p[2] == 'P' || p[2] == 'p') &&
        (p[3] == 'R' || p[3] == 'r') && (p[4] == 'A' || p[4] == 'a') &&
        (p[5] == 'G' || p[5] == 'g') && (p[6] == 'M' || p[6] == 'm') &&
        (p[7] == 'A' || p[7] == 'a')) {
      /*
       * communicate to p_pragma() that this is a sun directive.
       * do so by prepending the substring beginning with the
       * first character after "pragma"  with "sun".
       */
      first_char = firstp + 5;
      strncpy(first_char, "sun", 3);
      if (long_pragma_candidate)
        error(285, 3, curr_line, CNULL, CNULL);
      return CT_PRAGMA;
    }

    if (OPENMP && /* c$smp, c$omp - smp directive sentinel */
        p[1] == '$' &&
        (p[2] == 'S' || p[2] == 's' || p[2] == 'O' || p[2] == 'o') &&
        (p[3] == 'M' || p[3] == 'm') && (p[4] == 'P' || p[4] == 'p')) {
      firstp += 5;
      for (; isblank(*firstp); firstp++)
        ;
      first_char = firstp; /* first non-blank character in stmt */
      c = *firstp;
      ct = CT_SMP;
      sentinel = SL_OMP;
      goto bl_firstchar;
    }
    /* SGI c$doacross, c$& */
    if (SGIMP && p[1] == '$' && (p[2] == 'D' || p[2] == 'd') &&
        (p[3] == 'O' || p[3] == 'o') && (p[4] == 'A' || p[4] == 'a') &&
        (p[5] == 'C' || p[5] == 'c') && (p[6] == 'R' || p[6] == 'r') &&
        (p[7] == 'O' || p[7] == 'o') && (p[8] == 'S' || p[8] == 's') &&
        (p[9] == 'S' || p[9] == 's')) {
      sentinel = SL_SGI;
      first_char = &p[2];
      if (long_pragma_candidate)
        error(285, 3, curr_line, CNULL, CNULL);
      return CT_SMP;
    }
    /* OpenMP conditional compilation sentinels */
    if (OPENMP && p[1] == '$' && (iswhite(p[2]) || isdigit(p[2]))) {
      firstp += 2;
      for (; isblank(*firstp); firstp++)
        ;
      first_char = firstp; /* first non-blank character in stmt */
      c = *firstp;
      goto bl_firstchar;
    }
    /* sgi's continuation convention ('!$&') will be detected here */
    if (SGIMP && p[1] == '$' && p[2] == '&') {
      if (!is_sgi)
        /* current statement is not an SGI smp statement; just
         * treat as a comment.
         */
        return CT_COMMENT;
      sentinel = SL_SGI;
      first_char = firstp + 3; /* first character after '&' */
      if (long_pragma_candidate)
        error(285, 3, curr_line, CNULL, CNULL);
      return CT_CONTINUATION;
    }
    /* Miscellaneous directives which are parsed */
    if (XBIT(59, 0x4) && /* c$mem - mem directive sentinel */
        p[1] == '$' && (p[2] == 'M' || p[2] == 'm') &&
        (p[3] == 'E' || p[3] == 'e') && (p[4] == 'M' || p[4] == 'm')) {
      firstp += 5;
      for (; isblank(*firstp); firstp++)
        ;
      first_char = firstp; /* first non-blank character in stmt */
      c = *firstp;
      ct = CT_MEM; /* change initial card type */
      sentinel = SL_MEM;
      goto bl_firstchar;
    }
    if (XBIT_PCAST && /* c$pgi - alternate pgi accelerator directive sentinel */
        p[1] == '$' && (p[2] == 'P' || p[2] == 'p') &&
        (p[3] == 'G' || p[3] == 'g') && (p[4] == 'I' || p[4] == 'i')) {
      firstp += 5;
      for (; isblank(*firstp); firstp++)
        ;
      first_char = firstp; /* first non-blank character in stmt */
      c = *firstp;
      ct = CT_PGI; /* change initial card type */
      sentinel = SL_PGI;
      goto bl_firstchar;
    }
    if (XBIT(137, 1) && /* c$cuf - cuda kernel directive sentinel */
        p[1] == '$' && (p[2] == 'C' || p[2] == 'c') &&
        (p[3] == 'U' || p[3] == 'u') && (p[4] == 'F' || p[4] == 'f')) {
      firstp += 5;
      for (; isblank(*firstp); firstp++)
        ;
      first_char = firstp; /* first non-blank character in stmt */
      c = *firstp;
      ct = CT_KERNEL; /* change initial card type */
      sentinel = SL_KERNEL;
      goto bl_firstchar;
    }
    if (XBIT(137, 1) && /* !@cuf - cuda kernel conditional compilation */
        p[1] == '@' && (p[2] == 'C' || p[2] == 'c') &&
        (p[3] == 'U' || p[3] == 'u') && (p[4] == 'F' || p[4] == 'f') &&
        iswhite(p[5])) {
      firstp += 5;
      for (; isblank(*firstp); firstp++)
        ;
      first_char = firstp; /* first non-blank character in stmt */
      c = *firstp;
      goto bl_firstchar;
    }

    i = 1;
    p = b;
    while (TRUE) {
      cc = firstp[i];
      if (cc >= 'A' && cc <= 'Z')
        *p = tolower(cc);
      else
        *p = cc;
      p++;
      if (i >= MAX_DIRLEN || cc == '$' || cc == '\n')
        break;
      i++;
    }
    if (cc == '$') {
      *p = '\0';
      if (strncmp(b, "pgi$", 4) == 0 || strncmp(b, "vd$", 3) == 0) {
        /* for these directives, point to first character after the
         * '$'.
         */
        first_char = &firstp[i + 1];
        if (long_pragma_candidate)
          error(285, 3, curr_line, CNULL, CNULL);
        return check_pgi_pragma(first_char);
      }
      if (strncmp(b, "dir$", 4) == 0) {
        /*
         * communicate to p_pragma() that this is a cray directive.
         * do so by prepending the substring beginning with the
         * first character after the '$' with "cray".
         */
        first_char = &firstp[1];
        strncpy(first_char, "cray", 4);
        i = check_pragma(first_char + 4);
        if (i == CT_PPRAGMA) {
          strncpy(firstp, "     ", 5);
        }
        if (long_pragma_candidate)
          error(285, 3, curr_line, CNULL, CNULL);
        return i;
      }
      if (XBIT(124, 0x100) && strncmp(b, "exe$", 4) == 0) {
        firstp += 5;
        first_char = firstp;
        c = *firstp;
        goto bl_firstchar;
      }
#if defined(TARGET_WIN)
      if (strncmp(b, "dec$", 4) == 0) {
        firstp += 5;
        first_char = firstp;
        c = *firstp;
        ct = CT_DEC;
        goto bl_firstchar;
      }
      if (strncmp(b, "ms$", 3) == 0) {
        firstp += 4;
        first_char = firstp;
        c = *firstp;
        ct = CT_DEC;
        goto bl_firstchar;
      }
#endif
    }
    return CT_COMMENT;
  }
bl_firstchar:
  if (long_pragma_candidate)
    error(285, 3, curr_line, CNULL, CNULL);
  if (c == '&') {
    first_char = firstp + 1;
    return CT_CONTINUATION;
  }

  return ct;
}

/*  Prepare one Fortran stmt for crunching.  Copy the current card to
 *  the statement buffer.  Need to watch for the current card containing
 *  the continuation character.
 */
static void
ff_prescan(void)
{
  int c;
  char *inptr; /* next char to be processed */
  char *p;     /* pointer into statement buffer */
  char quote;
  char *amp; /* pointer to '&' in cardb */

  ff_state.outptr = ff_state.cavail - 1;
  ff_state.amper_ptr = NULL;

  for (inptr = first_char; (c = *inptr) != '\n'; inptr++) {
    *++ff_state.outptr = c;
    switch (c) {
    default:
      break;

    case 'a':
    case 'b':
    case 'c':
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
    case 'A':
    case 'B':
    case 'C':
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
    /* have the start of an identifier; eat all characters allowed
     * to make up an identifier - facilitates checking for a Hollerith
     * constant.
     */
    again_id:
      do {
        c = *++inptr;
        *++ff_state.outptr = c;
      } while (isident(c));
      if (c == '&') {
        last_char[card_count - 1] = ff_state.outptr - stmtb - 1;
        ff_get_noncomment(inptr + 1);
        ff_state.outptr--;
        if (card_type != CT_NONE) {
          inptr = first_char - 1;
          goto again_id;
        }
        goto exit_ff_prescan;
      }
      ff_state.outptr--;
      inptr--;
      break;

    case '&':
      last_char[card_count - 1] = ff_state.outptr - stmtb - 1;
      ff_get_noncomment(inptr + 1);
      ff_state.outptr--;
      inptr = first_char - 1;
      if (card_type == CT_NONE)
        goto exit_ff_prescan;
      break;

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
      p = ff_state.outptr;
    again_number:
      do {
        c = *++inptr;
        *++ff_state.outptr = c;
      } while (isdig(c));
      if (c == '&') {
        last_char[card_count - 1] = ff_state.outptr - stmtb - 1;
        ff_get_noncomment(inptr + 1);
        ff_state.outptr--;
        if (card_type != CT_NONE) {
          inptr = first_char - 1;
          goto again_number;
        }
        goto exit_ff_prescan;
      }
      if (isholl(c)) {
        int len;

        /* Hollerith constant has been found: */

        sscanf(p, "%d", &len);
        if (XBIT(125, 4)) {
          int ilen = (cardb + ff_state.last_char) - inptr;
          /* compute #bytes */
          len = kanji_prefix((unsigned char *)inptr + 1, len, ilen);
        }
      again_hollerith:
        ff_state.amper_ptr = NULL;
        while (len-- > 0) {
          *++ff_state.outptr = c = *++inptr;
          if (c == '\n') {
            len++;
            ff_state.outptr--;
            inptr--;
            break;
          }
          if (c == '&') {
            ff_state.amper_ptr = ff_state.outptr;
            amp = inptr;
          } else if (c != ' ')
            ff_state.amper_ptr = NULL;
        }
        if (len >= 0) { /* len == -1 ==> found all characters */
          if (ff_state.amper_ptr == NULL)
            goto exit_ff_prescan;
#if DEBUG
          assert(ff_state.amper_ptr == ff_state.outptr, "ff_prescan:Hollerith&",
                 inptr - ff_state.amper_ptr + 1, 3);
#endif
          last_char[card_count - 1] = ff_state.amper_ptr - stmtb - 1;
          ff_get_noncomment(amp + 1);
          if (card_type != CT_NONE) {
            len++;
            inptr = first_char - 1;
            ff_state.outptr = ff_state.amper_ptr - 1;
            goto again_hollerith;
          }
          goto exit_ff_prescan;
        }
      }
      ff_state.outptr--;
      inptr--;
      break;

    case '"':
    case '\'':
      quote = c;
      ff_state.amper_ptr = NULL;
      while (TRUE) {
        c = *++inptr;
        if (c == '\n') {
          if (ff_state.amper_ptr == NULL)
            goto exit_ff_prescan;
          last_char[card_count - 1] = ff_state.amper_ptr - stmtb - 1;
          ff_get_noncomment(amp + 1);
          if (card_type != CT_NONE) {
            inptr = first_char - 1;
            if (flg.standard && *inptr != '&') {
              error(170, 2, curr_line, "'&' required as the first character of "
                                       "a continued string literal",
                    CNULL);
            }
            ff_state.outptr = ff_state.amper_ptr - 1;
            ff_state.amper_ptr = NULL;
            continue;
          }
          goto exit_ff_prescan;
        }
        *++ff_state.outptr = c;
        if (c == '&') {
          ff_state.amper_ptr = ff_state.outptr;
          amp = inptr;
        } else if (c == quote) {
          if (*(inptr + 1) == quote) {
            *++ff_state.outptr = quote;
            inptr++;
          } else
            break;
        } else if (c == '\\' && !flg.standard && !XBIT(124, 0x40) &&
                   inptr[1] != '\n') {
          /* backslash escape in character constant */
          *++ff_state.outptr = *++inptr;
        }
      }
      break;

    case '!':
      ff_chk_pragma(inptr + 1);
      ff_state.outptr--;
      goto exit_for;
    }
  }

exit_for:;

exit_ff_prescan:
  *++ff_state.outptr = '\n'; /* mark end of stmtb contents */
  ff_state.cavail = ff_state.outptr;
  last_char[card_count - 1] = ff_state.cavail - stmtb - 1;
}

/* pointer to character after '!' */
static void
ff_chk_pragma(char *ppp)
{
  if (ppp[0] == '$' && (ppp[1] == 'P' || ppp[1] == 'p') &&
      (ppp[2] == 'R' || ppp[2] == 'r') && (ppp[3] == 'A' || ppp[3] == 'a') &&
      (ppp[4] == 'G' || ppp[4] == 'g') && (ppp[5] == 'M' || ppp[5] == 'm') &&
      (ppp[6] == 'A' || ppp[6] == 'a')) {
    /*
     * communicate to p_pragma() that this is a sun directive.
     * do so by prepending the substring beginning with the
     * first character after "pragma"  with "sun".
     * NOTE: p_pragma expects a terminated line; inptr locates
     *       the end of the line containing "!c$pragma".  In
     *       the next position store a newline.
     */
    strncpy(&ppp[4], "sun", 3);
    p_pragma(&ppp[4], gbl.lineno);
  }
}

/* locates character after '&' */
/*
 * while processing a number or identifier, found a '&' (expect
 * continuation):
 * 1.  if this is the last non-blank character before, if any,
 *     inline comment, get a noncomment line.
 * 2.  if not, then an error.
 */
static void
ff_get_noncomment(char *inptr)
{
  char *p;
  char c;

  for (p = inptr; (c = *p) != '\n'; p++) {
    if (c == '!') {
      ff_chk_pragma(p + 1);
      c = '\n';
      break;
    }
    if (!iswhite(c))
      break;
  }
  if (c == '\n') {
    while (TRUE) {
      if (flg.list && card_type <= CT_COMMENT) {
        if (list_now)
          list_line(printbuff);
      }
#if DEBUG
      if (DBGBIT(4, 2))
        fprintf(gbl.dbgfil, "line(%4d) %s", curr_line, cardb);
#endif
      card_type = ff_read_card();
      if (card_type == CT_LINE) {
        line_directive();
      }

      if (card_type != CT_COMMENT && card_type != CT_LINE)
        break;
    }
  } else
    card_type = CT_NONE;

  switch (card_type) {
  case CT_SMP:
  case CT_MEM:
  case CT_PGI:
  case CT_KERNEL:
  case CT_DIRECTIVE:
    /* In free source form, OpenMP, 'mem', %, and $ don't require the
     * '&' appended to their respective sentinels; e.g.
     *   !$omp ...openmp...  &
     *   !$omp ...continuation...
     *
     *   !$mem ...mem...  &
     *   !$mem ...continuation...
     *
     *      parentstruct &
     *   %member = ...       !!! parentstruct%member =
     *
     *      namecontains &
     *   $sign = ...         !!! namecontains$sign =
     *
     */
    card_type = CT_CONTINUATION;
    FLANG_FALLTHROUGH;
  case CT_INITIAL:
  case CT_CONTINUATION:
    check_continuation(curr_line);
    put_astfil(curr_line, &printbuff[8], TRUE);
    if (card_count == 0) {
      error(19, 3, curr_line, CNULL, CNULL);
      card_type = CT_NONE;
      return;
    }
    card_count++;
    ff_check_stmtb();
    return;
  default:
    error(295, 3, gbl.lineno, CNULL, CNULL);
    break;
  }

  /* error */
  card_type = CT_NONE;
}

static int
ff_get_label(char *inp)
{
  int c;
  int cnt;       /* number of characters processed */
  char *labp;

  scn.currlab = 0;
  cnt = 0;

  /* skip any leading white space */

  c = *inp;
  while (iswhite(c)) {
    if (c == '\n')
      goto ret;
    c = *++inp;
    cnt++;
  }

  labp = inp;
  while (isdig(c)) {
    cnt++;
    c = *++inp;
  }
  if (labp != inp) {
    atoxi(labp, &scn.labno, inp - labp, 10);
    if (scn.labno == 0)
      error(18, 3, gbl.lineno, "0", CNULL);
    else if (scn.labno > 99999)
      error(18, 3, gbl.lineno, "- length exceeds 5 digits", CNULL);
    else {
      int lab_sptr = getsymf(".L%05ld", (long)scn.labno);
      if (!iswhite(c))
        errlabel(18, 3, curr_line, SYMNAME(lab_sptr),
                 "- must be followed by one or more blanks");
      scn.currlab = declref(lab_sptr, ST_LABEL, 'd');
      if (DEFDG(scn.currlab))
        errlabel(97, 3, gbl.lineno, SYMNAME(lab_sptr), CNULL);
      /* linked list of labels for internal subprograms */
      if (sem.which_pass == 0 && gbl.internal > 1 &&
          SYMLKG(scn.currlab) == NOSYM) {
        SYMLKP(scn.currlab, sem.flabels);
        sem.flabels = scn.currlab;
      }
      put_astfil(FR_LABEL, NULL, FALSE);
      put_astfil(scn.labno, NULL, FALSE);
    }
  }

ret:
  return cnt;
}

static struct {
  long file_pos;
} fe_state;

void
fe_init(void)
{
  int nw;

  fe_state.file_pos = ftell(astb.astfil);
  /* fflush(astb.astfil);  fflush() doesn't seem to be always sufficient as an
   * intervening operation for read followed by write. So, replace with an
   * fseek().
   */
  nw = fseek(astb.astfil, fe_state.file_pos, 0);
#if DEBUG
  assert(nw == 0, "fe_init:bad rewind", nw, 4);
#endif
}

void
fe_save_state(void)
{
  fe_state.file_pos = ftell(astb.astfil);
}

void
fe_restart(void)
{
  int nw;

  nw = fseek(astb.astfil, fe_state.file_pos, 0);
#if DEBUG
  if (nw == -1)
    perror("fe_restart - fseek on astb.astfil");
  assert(nw == 0, "fe_restart:bad rewind", nw, 4);
  if (DBGBIT(4, 1024))
    fprintf(gbl.dbgfil, "----- begin file -----\n");
#endif
  /* close dinit files */
  dinit_end();
  gbl.eof_flag = FALSE;
  gbl.nowarn = TRUE; /*disable warnings for second parse; semfin enables*/
}

/*
 * restore when the second parse sees an end-of-file or FR_END record
 * or when an end-statement is seen which was the last statement in
 * the file.
 */
static void
_restore_state(void)
{
}

/*
 * restore only when the second parse sees an end-of-file or FR_END record.
 */
void
fe_restore_state(void)
{
  int nw;

  sem.which_pass = 0;
  scmode = SCM_FIRST;
  _restore_state();
  nw = fseek(astb.astfil, 0L, 0);
#if DEBUG
  assert(nw == 0, "_restore_state:bad rewind", nw, 4);
#endif
}

#include "tokdf.h"
static void
_write_token(int tk, INT ctkv)
{
  static int type = FR_TOKEN;
  int nw;
  int s1, s2;
  int len;
  char *p;

  nw = fwrite((char *)&type, sizeof(int), 1, astb.astfil);
  if (nw != 1)
    error(10, 4, 0, "(AST file)", CNULL);
  fprintf(astb.astfil, "%d", tk);
  fprintf(astb.astfil, " %d", ctkv); /* default token value */

  currCol = ((int)(currc - stmtb)) + leadCount;

  switch (tk) {
  case TK_IDENT:
  case TK_NAMED_CONSTRUCT:
    p = scn.id.name + ctkv;
    len = strlen(p);
    currCol = (currCol - len) + 1;
    fprintf(astb.astfil, " %d %d %s", currCol, len, p);
    fprintf(astb.astfil, " %d %s", (int)strlen(p), p);
    break;
  case TK_DEFINED_OP:
    p = SYMNAME(ctkv);
    len = strlen(SYMNAME(ctkv));
    fprintf(astb.astfil, " %d %d %s", currCol, len, p);
    break;
  case TK_ICON:
  case TK_RCON:
  case TK_NONDEC:
  case TK_LOGCONST:
    fprintf(astb.astfil, " %d %x", currCol, ctkv);
    break;
  case TK_DCON:
  case TK_CCON:
  case TK_NONDDEC:
    fprintf(astb.astfil, " %d %x %x", currCol, CONVAL1G(ctkv), CONVAL2G(ctkv));
    break;
  case TK_K_ICON:
  case TK_K_LOGCONST:
    fprintf(astb.astfil, " %d %x %x %d", currCol, CONVAL1G(ctkv),
            CONVAL2G(ctkv), DTYPEG(ctkv));
    break;
  case TK_QCON:
    fprintf(astb.astfil, " %d %x %x %x %x", currCol, CONVAL1G(ctkv),
            CONVAL2G(ctkv), CONVAL3G(ctkv), CONVAL4G(ctkv));
    break;
  case TK_DCCON:
    s1 = CONVAL1G(ctkv);
    s2 = CONVAL2G(ctkv);
    fprintf(astb.astfil, " %d %x %x %x %x", currCol, CONVAL1G(s1), CONVAL2G(s1),
            CONVAL1G(s2), CONVAL2G(s2));
    break;
  case TK_QCCON:
    s1 = CONVAL1G(ctkv);
    s2 = CONVAL2G(ctkv);
    fprintf(astb.astfil, " %d %x %x %x %x %x %x %x %x", currCol, CONVAL1G(s1),
            CONVAL2G(s1), CONVAL3G(s1), CONVAL4G(s1), CONVAL1G(s2),
            CONVAL2G(s2), CONVAL3G(s2), CONVAL4G(s2));
    break;
  case TK_HOLLERITH:
    fprintf(astb.astfil, " %d %d", currCol,
            CONVAL2G(ctkv)); /* kind of hollerith */
    ctkv = CONVAL1G(ctkv);   /* auxiliary char constant */
    goto common_str;         /* fall thru */
  case TK_FMTSTR:
  case TK_STRING:
  case TK_KSTRING:
    fprintf(astb.astfil, " %d", currCol);
  common_str:
    len = string_length(DTYPEG(ctkv));
    fprintf(astb.astfil, " %d ", len);
    p = stb.n_base + CONVAL1G(ctkv);
    while (len-- > 0)
      fprintf(astb.astfil, "%02x", (*p++) & 0xff);
    break;
  case TK_DIRECTIVE:
    len = (int)strlen(scn.directive);
    fprintf(astb.astfil, " %d %d", currCol, len);
    fprintf(astb.astfil, " %s", scn.directive);
    break;
  case TK_OPTIONS:
    len = (int)strlen(scn.options);
    fprintf(astb.astfil, " %d %d", currCol, (int)strlen(scn.options));
    fprintf(astb.astfil, " %s", scn.options);
    break;
  case TK_ENDSTMT:
  case TK_ENDBLOCKDATA:
  case TK_ENDFUNCTION:
  case TK_ENDPROCEDURE:
  case TK_ENDPROGRAM:
  case TK_ENDSUBROUTINE:
  case TK_ENDMODULE:
  case TK_ENDSUBMODULE:
  case TK_CONTAINS:
    fprintf(astb.astfil, " %d %d", currCol, gbl.eof_flag);
    break;
  case TK_EOL:
    currCol = 0;
    FLANG_FALLTHROUGH;
  default:
    fprintf(astb.astfil, " %d", currCol);
    break;
  }
  fprintf(astb.astfil, " %s", tokname[tk]);
  fprintf(astb.astfil, "\n");
}

static char *tkp;
static void _rd_tkline(char **tkbuf, int *tkbuf_sz);
static int _rd_token(INT *);
static INT get_num(int);
#ifdef FLANG_SCAN_UNUSED
static void get_string(char *);
#endif

/** \brief trim white space of source line that has continuations and return
 * the index of the last character in the source line.
 *
 * This function is called by contIndex().
 *
 * \param line is the source line we are processing.
 *
 * \return the index (an integer) of the last character in the source line.
 */
static int
trimContIdx(char *line)
{
  int len;
  char *p;

  if (line == NULL)
    return 0;

  len = strlen(line);
  if (len == 0)
    return 0;

  for (p = (line + len) - 1; p > line && isspace(*p); --p)
    ;

  return (int)(p - line);
}

static int
numLeadingSpaces(char *line)
{
  int i;

  if (line == NULL)
    return 0;

  for (i = 0; *line != '\0'; ++line, ++i) {
    if (!isspace(*line) && *line != '&')
      break;
  }

  return i;
}

/** \brief Parse a source line with comments/continuations/string literals
 *  and return the index of the last non-white space character of the
 *  source line.
 *
 * \param line is the source line that we are processing.
 *
 * \return the index (an integer) of the last character in source line.
 */
static int
contIndex(char *line)
{
  int i;
  bool seenText = false;
  int seenQuote = 0;
  bool seenFin = false;
  int len;

  if (line == NULL)
    return 0;

  len = strlen(line);

  for (i = 0; i < len; ++i) {
    if (!seenText && !isspace(line[i]) && line[i] != '&') {
      seenText = TRUE;
    }
    if (!seenText) {
      continue;
    }
    if (seenFin) {
      return i;
    }
    if (seenQuote == 0 && (line[i] == '\'' || line[i] == '"')) {
      seenQuote = line[i];
    } else if (seenQuote != 0 && line[i] == seenQuote) {
      seenQuote = 0;
    } else if (seenQuote == 0 && (line[i] == '!' || line[i] == '&')) {
      seenFin = true;
      return i + 1;
    }
  }

  return trimContIdx(line);
}

/** \brief get the post-processed source line in the current source file.
  *
  * \param line is the desired source line number.
  *
  * \param src_file is used to store a copy of the source filename that the
  * line number is associated with. It could be different from gbl.curr_file.
  * If src_file is NULL, then this parameter is ignored. Caller is responsible
  * to free the memory that src_file points to.
  *
  * \param col is the column number associated with the token in the source
  * line. It is usually needed if line has continuators.
  *
  * \param srcCol is the adjusted column number for the current source line. It
  * may be different than col when the column is associated with a continued
  * source line. This parameter is ignored if it is 0.
  *
  * \param contNo is greater than zero when source line is a continuation of
  * the source line specified in line.
  *
  * \return the source line associated with line. Result is NULL if line not
  * found in source file. Caller is responsible to free the memory allocated
  * for the result.
  */
char *
get_src_line(int line, char **src_file, int col, int *srcCol, int *contNo)
{
  int fr_type, i, scratch_sz = 0, line_sz = 0, srcfile_sz = 0;
  char *scratch_buf = NULL;
  char *line_buf = NULL;
  char *srcfile_buf = NULL;
  long offset;
  int curr_line = 0, len;
  int line_len = 0;
  int adjCol = 0;
  int is_cont = 0;
  int adjSrcLine = 0;
  int saveCol = currCol;

  offset = ftell(astb.astfil);
  rewind(astb.astfil);
  while (TRUE) {
    i = fread((char *)&fr_type, sizeof(int), 1, astb.astfil);
    if (feof(astb.astfil) || i != 1) {
      /* EOF */
      break;
    }
    switch (fr_type) {
    case FR_LINENO:
      _rd_tkline(&scratch_buf, &scratch_sz);
      sscanf(scratch_buf, "%d", &curr_line);
      break;
    case FR_SRC:
      _rd_tkline(&srcfile_buf, &srcfile_sz);
      if (src_file) {
        *src_file = srcfile_buf;
      }
      break;
    case FR_STMT:
      i = fread((char *)&curr_line, sizeof(int), 1, astb.astfil);
      if (feof(astb.astfil) || i != 1) {
        interr("get_src_line: truncated ast file", 0, 4);
        break;
      }
    next_stmt:
      _rd_tkline(&line_buf, &line_sz);
      if (curr_line == line) {

        adjCol = line_len;

        i = contIndex(line_buf);
        line_len += (i > 0) ? i : strlen(line_buf);
        line_len -= (is_cont) ? numLeadingSpaces(line_buf) : 0;

        if (col < line_len) {
          if (is_cont) {
            col = ((col + numLeadingSpaces(line_buf)) - adjCol) + is_cont;
          }
          goto fin;
        } else {
          ++is_cont;
          continue;
        }
      } else if (line_buf) {
        i = sizeof(int);
        len = (line_sz > i) ? i : line_sz;
        for (i = 0; i < len; ++i)
          line_buf[i] = '\0';
        line_len = 0;
        is_cont = 0;
      }
      if (curr_line > line) {
        goto fin;
      }
      break;
    default:
      if (fr_type > 0 && is_cont > 0) {
        /* got a line continuation */
        adjSrcLine++;
        goto next_stmt;
      }
      _rd_tkline(&scratch_buf, &scratch_sz);
    }
  }
fin:
  currCol = saveCol;
  fseek(astb.astfil, offset, SEEK_SET);
  FREE(scratch_buf);
  if (srcCol) {
    *srcCol = col;
  }
  if (contNo) {
    *contNo = adjSrcLine;
  }
  return line_buf;
}

static int
_read_token(INT *tknv)
{
  int i;
  int fr_type;
  static int prev_lineno = 0;
  static int incl_level = 0;
  static int lineno = 0;
  int findex;
  int lab_sptr;

  while (TRUE) {
    i = fread((char *)&fr_type, sizeof(int), 1, astb.astfil);
    if (i < 1) {
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "----- end of file -----\n");
#endif
      fe_restore_state();
      return get_token(tknv);
    }
    switch (fr_type) {
    case FR_END:
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "----- end of file: FR_END -----\n");
#endif
      fe_restore_state();
      return get_token(tknv);
    case FR_SRC:
      goto read_line;
    case FR_B_INCL:
      incl_level++;
      (void)fread((char *)&findex, sizeof(int), 1, astb.astfil);
      gbl.findex = findex;
      gbl.curr_file = FIH_FULLNAME(gbl.findex);
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "Include level %d: %s", incl_level, tkp);
#endif
      break;
    case FR_B_HDR:
      (void)fread((char *)&findex, sizeof(int), 1, astb.astfil);
      gbl.findex = findex;
      gbl.curr_file = FIH_FULLNAME(gbl.findex);
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "All include level %d: %s", hdr_level, tkp);
#endif
      break;
    case FR_E_INCL:
      (void)fread((char *)&findex, sizeof(int), 1, astb.astfil);
      gbl.findex = findex;
      gbl.curr_file = FIH_FULLNAME(gbl.findex);
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "End of include level %d: %s", incl_level, tkp);
#endif
      incl_level--;
      break;
    case FR_E_HDR:
      (void)fread((char *)&findex, sizeof(int), 1, astb.astfil);
      gbl.findex = findex;
      gbl.curr_file = FIH_FULLNAME(gbl.findex);
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "End of all include level %d: %s", hdr_level, tkp);
#endif
      break;
    case 0:
      lineno = gbl.lineno = prev_lineno + 1;
      goto read_line;
    case FR_LABEL:
      (void)fread((char *)&fr_type, sizeof(int), 1, astb.astfil);
      lab_sptr = getsymf(".L%05ld", (long)fr_type);
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "Label %d\n", fr_type);
#endif
      scn.currlab = declref(lab_sptr, ST_LABEL, 'd');
      /** HACK -- don't check for multiple defs if the label was moved
       ** from a host's end statement to its CONTAINS statement.  SEE
       ** semant.c:/sem.end_host_labno/
       **/
      if (DEFDG(scn.currlab) && !L3FG(scn.currlab))
        errlabel(97, 3, curr_line, SYMNAME(lab_sptr), CNULL);
      /* linked list of labels for internal subprograms */
      if (sem.which_pass == 0 && gbl.internal > 1 &&
          SYMLKG(scn.currlab) == NOSYM) {
        SYMLKP(scn.currlab, sem.flabels);
        sem.flabels = scn.currlab;
      }
      break;
    case FR_LINENO:
      _rd_tkline(&tkbuf, &tkbuf_sz);
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "  Lineno: %s", tkp);
#endif
      gbl.lineno = get_num(10);
      break;
    case FR_PRAGMA:
      _rd_tkline(&tkbuf, &tkbuf_sz);
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "  Pragma: %s", tkp);
#endif
      p_pragma(tkp, gbl.lineno);
      break;
    default:
      lineno = fr_type;
    read_line:
      _rd_tkline(&tkbuf, &tkbuf_sz);
      switch (fr_type) {
      case FR_SRC:
#if DEBUG
        if (DBGBIT(4, 1024))
          fprintf(gbl.dbgfil, "Source File: %s", tkbuf);
#endif
        break;
      default:
#if DEBUG
        if (DBGBIT(4, 1024))
          fprintf(gbl.dbgfil, "%5d: %s", lineno, tkbuf);
#endif
        break;
      }
      break;
    case FR_STMT:
#if DEBUG
      if (DBGBIT(4, 1024))
        fprintf(gbl.dbgfil, "----- new stmt: FR_STMT -----\n");
#endif
      scmode = SCM_FIRST;
      scn.stmtyp = 0;
      scn.currlab = 0;
      scn.is_hpf = FALSE;
      par_depth = 0;
      break;
    case FR_TOKEN:
      return _rd_token(tknv);
    }
    prev_lineno = lineno;
  }
}

static void
_rd_tkline(char **tkbuf, int *tkbuf_sz)
{
  int i;
  int ch;
  char *p;

  i = 0;
  while (TRUE) {
    ch = getc(astb.astfil);
    if (i + 1 >= *tkbuf_sz) {
      *tkbuf_sz += CARDB_SIZE << 3;
      *tkbuf = sccrelal(*tkbuf, *tkbuf_sz);
    }
    (*tkbuf)[i++] = ch;
    if (ch == '\n')
      break;
  }
  (*tkbuf)[i] = '\0';
  p = tkp = *tkbuf;
  /* Process #include files */
  if (*p == '#') {
    ++p;
    while (isblank(*p)) /* skip blank characters */
      ++p;
    if (!isdig(*p)) {
      const char *tmp_ptr;
      tmp_ptr = gbl.curr_file;
      if (hdr_level)
        gbl.curr_file = hdr_stack[hdr_level - 1].fname;
      error(21, 3, curr_line, CNULL, CNULL);
      gbl.curr_file = tmp_ptr;
      return;
    }
    while (isdig(*p)) {
      ++p;
    }
    ++p;
    while (isblank(*p)) {
      ++p;
    }
    if (*p == '"') {
      *(p + CARDB_SIZE) = '"'; /* limit length of file name */
    }
  }
}

int
getCurrColumn(void)
{
  return currCol;
}

static int
_rd_token(INT *tknv)
{
  int i;
  int len;
  INT val[2], num[4];
  char *p, *q;
  int kind;
  int dtype;

  _rd_tkline(&tkbuf, &tkbuf_sz);
#if DEBUG
  if (DBGBIT(4, 1024))
    fprintf(gbl.dbgfil, "  TOKEN: %s", tkbuf);
#endif
  tkntyp = get_num(10);
  tknval = get_num(10);  /* default token value */
  currCol = get_num(10); /* get column number */

  switch (tkntyp) {
  case TK_IDENT:
  case TK_NAMED_CONSTRUCT:
    len = get_num(10);
    p = scn.id.name + tknval;
    while (len-- > 0)
      *p++ = *++tkp;
    *p = '\0';
    break;
  case TK_DEFINED_OP:
    len = get_num(10);
    tknval = getsym(++tkp, len);
    break;
  case TK_ICON:
  case TK_RCON:
  case TK_NONDEC:
  case TK_LOGCONST:
    tknval = get_num(16);
    break;
  case TK_DCON:
    num[0] = get_num(16);
    num[1] = get_num(16);
    tknval = getcon(num, DT_REAL8);
    break;
  case TK_QCON:
    num[0] = get_num(16);
    num[1] = get_num(16);
    num[2] = get_num(16);
    num[3] = get_num(16);
    tknval = getcon(num, DT_QUAD);
    break;
  case TK_CCON:
    num[0] = get_num(16);
    num[1] = get_num(16);
    tknval = getcon(num, DT_CMPLX8);
    break;
  case TK_DCCON:
    num[0] = get_num(16);
    num[1] = get_num(16);
    val[0] = getcon(num, DT_REAL8);
    num[0] = get_num(16);
    num[1] = get_num(16);
    val[1] = getcon(num, DT_REAL8);
    tknval = getcon(val, DT_CMPLX16);
    break;
  case TK_QCCON:
    num[0] = get_num(16);
    num[1] = get_num(16);
    num[2] = get_num(16);
    num[3] = get_num(16);
    val[0] = getcon(num, DT_QUAD);
    num[0] = get_num(16);
    num[1] = get_num(16);
    num[2] = get_num(16);
    num[3] = get_num(16);
    val[1] = getcon(num, DT_QUAD);
    tknval = getcon(val, DT_QCMPLX);
    break;
  case TK_NONDDEC:
    num[0] = get_num(16);
    num[1] = get_num(16);
    tknval = getcon(num, DT_DWORD);
    break;
  case TK_K_ICON:
  case TK_K_LOGCONST:
    num[0] = get_num(16);
    num[1] = get_num(16);
    dtype = get_num(10);
    tknval = getcon(num, dtype);
    break;
  case TK_HOLLERITH:
    kind = get_num(10);
    FLANG_FALLTHROUGH;
  case TK_FMTSTR:
  case TK_STRING:
  case TK_KSTRING:
    i = len = get_num(10);
    q = p = ++tkp;
    while (i-- > 0) {
      num[0] = hex_to_i((int)p[0]) << 4;
      num[0] |= hex_to_i((int)p[1]);
      *q++ = num[0];
      p += 2;
    }
    tknval = getstring(tkp, len);
    if (tkntyp == TK_HOLLERITH)
      tknval = gethollerith(tknval, kind);
    break;
  case TK_DIRECTIVE:
    len = get_num(10);
    p = scn.directive;
    while (len-- > 0)
      *p++ = *++tkp;
    *p = '\0';
    break;
  case TK_OPTIONS:
    len = get_num(10);
    p = scn.options;
    while (len-- > 0)
      *p++ = *++tkp;
    *p = '\0';
    break;
  case TK_LPAREN:
  case TK_IOLP:
  case TK_IMPLP:
    par_depth++;
    break;
  case TK_RPAREN:
    par_depth--;
    if (bind_state == B_FUNC_FOUND) {
      bind_state = B_RPAREN_FOUND;
    }
    if (par_depth == 0 && scmode == SCM_IF)
      scmode = SCM_FIRST;
    break;
  default:
    if (scmode == SCM_FIRST) {
      scn.stmtyp = tkntyp;
      scmode = SCM_IDENT;
      switch (scn.stmtyp) {
      case TK_ENDSTMT:
      case TK_ENDBLOCKDATA:
      case TK_ENDFUNCTION:
      case TK_ENDPROCEDURE:
      case TK_ENDPROGRAM:
      case TK_ENDSUBROUTINE:
      case TK_CONTAINS: /* CONTAINS statement will be treated as the
                         * END of of a blockdata */
        if (sem.interface == 0)
          scn.end_program_unit = TRUE;
        gbl.eof_flag = get_num(10);
        if (gbl.eof_flag)
          _restore_state();
        break;
      case TK_ENDMODULE:
      case TK_ENDSUBMODULE:
        scn.end_program_unit = TRUE;
        gbl.eof_flag = get_num(10);
        if (gbl.eof_flag)
          _restore_state();
        break;
      case TK_IF:
        scmode = SCM_IF;
        break;
      default:
        break;
      }
    }
    break;
  }
  *tknv = tknval;
  return tkntyp;
}

int
get_named_stmtyp(void)
{
  long file_pos;
  int nw;
  int fr_type;
  int i;
  int tkn;
  /*
   * The token which names a control construct (TK_NAMED_CONSTRUCT) was
   * just seen by semant.  Now need to determine the actual statement type.
   * Recall that the syntax is
   *   named_construct : <token>
   */
  file_pos = ftell(astb.astfil);

  /* : */
  i = fread((char *)&fr_type, sizeof(int), 1, astb.astfil);
  _rd_tkline(&tkbuf, &tkbuf_sz);
#if DEBUG
  assert(i >= 1, "get_named_stmtyp:bad read 1", i, 4);
  assert(fr_type == FR_TOKEN, "get_named_stmtyp:expected FR_TOKEN 1, got",
         fr_type, 4);
  tkn = get_num(10);
  assert(tkn == TK_COLON, "get_named_stmtyp:expected :, got", tkn, 4);
#endif

  /* <token> */
  i = fread((char *)&fr_type, sizeof(int), 1, astb.astfil);
  _rd_tkline(&tkbuf, &tkbuf_sz);
#if DEBUG
  assert(i >= 1, "get_named_stmtyp:bad read 2", i, 4);
  assert(fr_type == FR_TOKEN, "get_named_stmtyp:expected FR_TOKEN 2, got",
         fr_type, 4);
#endif
  tkn = get_num(10);

  nw = fseek(astb.astfil, file_pos, 0);
#if DEBUG
  assert(nw == 0, "get_named_stmtyp:bad week", nw, 4);
#endif

  return tkn;
}

static int
get_num(int radix)
{
  char *p;
  INT val;

  while (*tkp == ' ')
    tkp++;
  p = tkp;
  while (*tkp != ' ' && *tkp != '\n')
    tkp++;
  (void)atoxi(p, &val, (int)(tkp - p), radix);
  return val;
}

#ifdef FLANG_SCAN_UNUSED
static void
get_string(char *dest)
{
  int i;
  char ch;

  while (*tkp == ' ')
    tkp++;
  i = 0;
  while ((ch = *tkp) != ' ' && ch != '\n') {
    dest[i++] = ch;
    tkp++;
  }
  dest[i] = '\0';
}
#endif

static void
realloc_stmtb(void)
{
  int which = 0;
  if (stmtb == stmtbefore)
    which = 1;
  max_card += 20;
  stmtbefore = sccrelal(stmtbefore, (BIGUINT64)(max_card * (MAX_COLS - 1) + 1));
  if (stmtbefore == NULL)
    error(7, 4, 0, CNULL, CNULL);
  stmtbafter = sccrelal(stmtbafter, (BIGUINT64)(max_card * (MAX_COLS - 1) + 1));
  if (stmtbafter == NULL)
    error(7, 4, 0, CNULL, CNULL);
  if (which)
    stmtb = stmtbefore;
  else
    stmtb = stmtbafter;
  last_char =
      (short *)sccrelal((char *)last_char, (BIGUINT64)(max_card * sizeof(short)));
  if (last_char == NULL)
    error(7, 4, 0, CNULL, CNULL);
}

static void
ff_check_stmtb(void)
{
  if (card_count >= max_card) {
    char *oldp;

    oldp = stmtb;
    realloc_stmtb();
    if (stmtb == NULL)
      error(7, 4, 0, CNULL, CNULL);
    ff_state.cavail = stmtb + (ff_state.cavail - oldp);
    ff_state.outptr = stmtb + (ff_state.outptr - oldp);
    if (ff_state.amper_ptr)
      ff_state.amper_ptr = stmtb + (ff_state.amper_ptr - oldp);
  }
  if (flg.standard && card_count == 257)
    error(170, 2, curr_line, "more than 255 continuations", CNULL);
}

static void
check_continuation(int lineno)
{
  switch (sentinel) {
  case SL_OMP:
    if (!is_smp)
      goto cont_error;
    break;
  case SL_SGI:
    if (!is_sgi)
      goto cont_error;
    break;
  case SL_MEM:
    if (!is_mem)
      goto cont_error;
    break;
  case SL_PGI:
    if (!is_pgi)
      goto cont_error;
    break;
  case SL_KERNEL:
    if (!is_kernel)
      goto cont_error;
    break;
  default:
    if (scn.is_hpf || is_smp || is_sgi || is_mem || is_ppragma || is_kernel || is_pgi
        )
      goto cont_error;
    break;
  }
  return;

cont_error:
  error(292, 3, lineno, CNULL, CNULL);
}

static LOGICAL
is_next_char(char *s, int ch)
{
  while (*s != ch) {
    if (*s == ' ') {
      s++;
      continue;
    }
    return FALSE;
  }
  return TRUE;
}

static int
double_type(char *ip, int *p_idlen)
{
  if (!is_freeform)
    return 0;
  if (*ip == ' ') {
    int k;
    k = is_ident(ip + 1);
    if (k == 7 && strncmp(ip + 1, "complex", 7) == 0) {
      *p_idlen += 7 + 1;
      return TK_DBLECMPLX;
    }
    if (k == 9 && strncmp(ip + 1, "precision", 9) == 0) {
      *p_idlen += 9 + 1;
      return TK_DBLEPREC;
    }
  }
  return 0;
}
