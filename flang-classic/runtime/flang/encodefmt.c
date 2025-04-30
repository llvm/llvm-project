/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*  encodefmt.c - translate format string into encoded form at runtime. */

#include "global.h"
#include "feddesc.h"

#define STACK_SIZE 20 /* max nesting depth of parens */
#define is_digit(c) ((c) >= '0' && (c) <= '9')

typedef int ERRCODE;

static long numval; /* numeric value computed by ef_getnum */
static char *firstchar, *lastchar;
static int curpos; /* current avail posn in output buffer */
static int paren_stack[STACK_SIZE];
static bool enclosing_parens;
static INT *buff = NULL;
static int buffsize = 0;
static char quote;

static ERRCODE check_outer_parens(char *, __CLEN_T);
static bool ef_getnum(char *, int *);
static int ef_nextchar(char *, int *);
static void ef_put(INT);
static void ef_putnum(INT);
static ERRCODE ef_putstring(char *, INT, int);
static void ef_alloc(int);
static INT ef_error(ERRCODE);
static void ef_putvlist(char *, int *);
static void ef_putdt();
static void ef_putiotype(char *, int *);

/* ----------------------------------------------------------------------- */

static __INT_T
_f90io_encode_fmt(char *str,      /* unencoded format string */
                 __INT_T *nelem, /* number of elements if array */
                 __CLEN_T str_siz)
{
  char *p;
  char c, cnext;
  ERRCODE i;
  int numlen = 0;
  int code1, code2, code3, code4;
  bool j;
  bool rep_count;        /* TRUE if integer token has just been processed */
  int reversion_loc = 0; /* position in output buffer */
  int paren_level = 0;   /* current paren nesting level */
  int k, n;
  bool unlimited_repeat_count = FALSE;

  /* the following call is just to ensure __fortio_init() has been called: */
  __fortio_errinit03(0, 0, NULL, "encode format string");
  __fortio_fmtinit();

  /*  make basic checks for legal input and determine if this format has the
      outer set of parentheses present or not.  Set global variables
      enclosing_parens, firstchar and lastchar:  */

  n = 1;
  if (*nelem)
    n = *nelem;
  i = check_outer_parens(str, str_siz * n);
  if (i != 0)
    return ef_error(i);

  curpos = 0; /* current available output buffer position */
  rep_count = FALSE;
  p = firstchar;

  while (p <= lastchar) {
    c = *p++;
    switch (c) {
    case ',':
      goto check_no_repeat_count;

    case ' ': /* ignore blanks */
    case '\t':
    case '\r':
      break;

    case '(':
      paren_level++;
      if (paren_level > STACK_SIZE)
        return ef_error(FIO_EFSYNTAX /*"stack overflow"*/);
      ef_put(FED_LPAREN);
      paren_stack[paren_level - 1] = curpos;
      if (paren_level == 1) {
        if (rep_count)
          reversion_loc = curpos - 3; /* point to repeat count */
        else
          reversion_loc = curpos - 1; /* point to left paren */
      }
      rep_count = FALSE;
      break;

    case ')':
      if (paren_level < 1) {
        if (enclosing_parens) {
          enclosing_parens = 0;
          goto end_of_format; /* ignore remaining part of input */
        }
        return ef_error(FIO_EPAREN /*"unbal parens"*/);
      }
      while (p <= lastchar && *p == ' ') /* skip spaces */
        ++p;
      if (unlimited_repeat_count &&
          paren_level == 1 &&
          p <= lastchar &&
          *p != ')') {
        /* F'08 unlimited repeat count must be used on the only or last
         * constituent of the top-level format list.  Compile-time
         * FORMAT statement parsing allows unlimited repetition to
         * precede other parenthesized lists, with a warning or
         * (with -Mstandard) a severe error.
         */
        return ef_error(FIO_EFSYNTAX);
      }
      paren_level--;

      if (paren_stack[paren_level] == curpos)
        return ef_error(FIO_EFSYNTAX /*"syntax - empty parens"*/);
      ef_put(FED_RPAREN);
      ef_put(paren_stack[paren_level]);
      break;

    case 'p':
    case 'P': /*  scale factor;  preceding integer required. */
      if (!rep_count)
        return ef_error(FIO_EPT /*"illegal P descriptor"*/);
      rep_count = FALSE;
      ef_put(FED_P);
      ef_putnum(numval);
      break;
    case '\'':
    case '\"':
      quote = c;
      n = 0; /* count number of quote characters in string */
      for (k = 0; p + k <= lastchar; k++) {
        if (p[k] == quote) {
          if (p + k < lastchar && p[k + 1] == quote)
            n++, k++; /* two quotes in a row */
          else
            break;
        }
      }

      if (p + k > lastchar)
        return ef_error(FIO_ESTRING /*"unterminated char string"*/);

      i = ef_putstring(p, k, n);
      if (i != 0)
        return ef_error(i);
      p += (k + 1);
      goto check_no_repeat_count;

    case 'h':
    case 'H':
      if (!rep_count)
        return ef_error(FIO_ESTRING /*"illegal Hollerith constant"*/);
      rep_count = FALSE;
      quote = '\'';
      i = ef_putstring(p, numval, 0);
      if (i != 0)
        return ef_error(i);
      p += numval;
      break;

    case 't':
    case 'T': /*  check for TL, TR or T edit descriptors:  */
      c = ef_nextchar(p, &numlen);
      p += numlen;
      if (c == 'L')
        ef_put(FED_TL);
      else if (c == 'R')
        ef_put(FED_TR);
      else {
        ef_put(FED_T);
        p--;
      }
      j = ef_getnum(p, &numlen);
      if (!j) /* number is required */
        return ef_error(FIO_EPT /*"T descriptor missing value"*/);
      p += numlen;
      ef_putnum(numval);
      goto check_no_repeat_count;

    case 'x':
    case 'X':
      ef_put(FED_X);
      if (rep_count) {
        ef_putnum(numval);
        rep_count = FALSE;
      } else
        ef_putnum(1L); /* default repeat count == 1 */
      break;

    case 'r':
    case 'R': /*  check for RU, RD, RZ, RN, RC, or RP descriptor:  */
      c = ef_nextchar(p, &numlen);
      switch (c) {
      case 'U':
        ef_put(FED_RU);
        break;
      case 'D':
        ef_put(FED_RD);
        break;
      case 'Z':
        ef_put(FED_RZ);
        break;
      case 'N':
        ef_put(FED_RN);
        break;
      case 'C':
        ef_put(FED_RC);
        break;
      case 'P':
        ef_put(FED_RP);
        break;
      default:
        return ef_error(FIO_ELETTER /*"unrecognized format code"*/);
      }
      p += numlen;
      goto check_no_repeat_count;

    case 's':
    case 'S': /*  check for SP, SS, or S descriptor:  */
      c = ef_nextchar(p, &numlen);
      p += numlen;
      if (c == 'P')
        ef_put(FED_SP);
      else if (c == 'S')
        ef_put(FED_SS);
      else {
        ef_put(FED_S);
        p--;
      }
      goto check_no_repeat_count;

    case 'b':
    case 'B': /*  check for BN or BZ edit descriptor: */
      c = ef_nextchar(p, &numlen);
      if (c == 'N') {
        ef_put(FED_BN);
        p += numlen;
      } else if (c == 'Z') {
        ef_put(FED_BZ);
        p += numlen;
      } else { /*  process B edit descriptor:  */
        code1 = FED_Bw_m;
        rep_count = FALSE;
        j = ef_getnum(p, &numlen);
        if (j == FALSE)
          return ef_error(FIO_EPT /*"illegal B descriptor"*/);
        p += numlen;
        ef_put(code1);
        ef_putnum(numval);
        c = ef_nextchar(p, &numlen);
        if (c != '.')
          ef_putnum(1L); /* default value for 'm' field */
        else {
          p += numlen;
          j = ef_getnum(p, &numlen);
          if (!j)
            return ef_error(FIO_EDOT /*"num expected after '.'"*/);
          ef_putnum(numval);
          p += numlen;
        }
        break;
      }
      goto check_no_repeat_count;

    case '/':
      rep_count = FALSE;
      ef_put(FED_SLASH);
      break;

    case ':':
      ef_put(FED_COLON);
      goto check_no_repeat_count;

    case 'q':
    case 'Q':
      ef_put(FED_Q);
      goto check_no_repeat_count;

    case '$':
      ef_put(FED_DOLLAR);
      goto check_no_repeat_count;

    case 'a':
    case 'A':
      code1 = FED_Aw;
      code2 = FED_A;
      goto A_shared;

    case 'l':
    case 'L':
      code1 = FED_Lw;
      code2 = FED_L;
    A_shared: /* process A or L edit descriptor */
      rep_count = FALSE;
      j = ef_getnum(p, &numlen);
      if (j == FALSE)
        ef_put(code2);
      else {
        p += numlen;
        ef_put(code1);
        ef_putnum(numval);
      }
      break;

    case 'F':
    case 'f':
      code1 = FED_Fw_d;
      code2 = FED_F;
      goto F_shared;

    case 'E':
    case 'e':
      c = ef_nextchar(p, &numlen);
      if (c == 'N') {
        code1 = FED_ENw_d;
        p += numlen;
        goto EN_shared;
      }
      if (c == 'S') {
        code1 = FED_ESw_d;
        p += numlen;
        goto EN_shared;
      }
      code1 = FED_Ew_d;
      code2 = FED_E;
      goto F_shared;

    EN_shared: /* process EN or ES edit descriptor */
      rep_count = FALSE;
      j = ef_getnum(p, &numlen);
      if (j == FALSE)
        return ef_error(FIO_EFGD /*"syntax, width expected"*/);
      p += numlen;
      ef_put(code1);
      ef_putnum(numval);
      c = ef_nextchar(p, &numlen);
      if (c != '.')
        return ef_error(FIO_EFGD /*"syntax, '.' expected"*/);
      else {
        p += numlen;
        j = ef_getnum(p, &numlen);
        if (!j)
          return ef_error(FIO_EDOT /*"number expd after '.'"*/);
        ef_putnum(numval);
        p += numlen;

        /*  check for E<numval> which optionally follows
            ENw.d or ESw.d edit descriptors:  */

        c = ef_nextchar(p, &numlen);
        if (c == 'E') {
          p += numlen;
          ef_put(FED_Ee);
          j = ef_getnum(p, &numlen);
          if (!j)
            return ef_error(FIO_EFGD);
          p += numlen;
          ef_putnum(numval);
        }
      }
      break;

    case 'G':
    case 'g':
      code1 = FED_Gw_d;
      code2 = FED_G;
      code3 = FED_G0;
      code4 = FED_G0_d;
      goto F_shared;

    case 'D':
    case 'd':
      /*  check for DC or DP edit descriptor and DT too: */
      c = ef_nextchar(p, &numlen);
      if (c == 'C') {
        ef_put(FED_DC);
        p += numlen;
        goto check_no_repeat_count;
      }
      if (c == 'P') {
        ef_put(FED_DP);
        p += numlen;
        goto check_no_repeat_count;
      }
      if (c == 'T') {
        ef_put(FED_DT);
        p += numlen;
        c = *(p);
        if (c == '(') {
          ef_putnum(2L);
          ef_putdt();
          p++;
          ef_putvlist(p, &numlen);
          p += numlen;
        } else if (c == '\'' || c == '\"') {
          p++;
          ef_putiotype(p, &numlen);
          p += numlen;
        } else {
          ef_putnum(1L);
          ef_putdt();
        }
        rep_count = FALSE;
        goto check_no_repeat_count;
      }
      /*  process D edit descriptor:  */
      code1 = FED_Dw_d;
      code2 = FED_D;
    F_shared: /*  process F, E, G or D edit descriptor  */
      rep_count = FALSE;
      j = ef_getnum(p, &numlen);
      p += numlen;
      cnext = ef_nextchar(p, &numlen);
      p -= numlen;
      if (j == FALSE) {
        ef_put(code2);
      } else if ((c == 'g' || c == 'G') && numval == 0
                  && cnext != '.') {
        p += numlen;
        j = ef_getnum(p, &numlen);
        if (j == FALSE) {
          /* G0 */
          ef_put(code3);
        } else {
          return ef_error(FIO_EFGD);
        }
      } else {
        p += numlen;
        if ((c == 'g' || c == 'G') && numval == 0) {
          /* G0.d */
          ef_put(code4);
        } else {
          ef_put(code1);
        }
        ef_putnum(numval);
        c = ef_nextchar(p, &numlen);
        if (c != '.')
          return ef_error(FIO_EFGD /*"syntax, '.' expected"*/);
        else {
          p += numlen;
          j = ef_getnum(p, &numlen);
          if (!j)
            return ef_error(FIO_EDOT /*"number expd after '.'"*/);
          ef_putnum(numval);
          p += numlen;

          /*  check for E<numval> which optionally follows
              Ew.d or Gw.d edit descriptors:  */

          if (code1 == FED_Ew_d || code1 == FED_Gw_d) {
            c = ef_nextchar(p, &numlen);
            if (c == 'E') {
              p += numlen;
              ef_put(FED_Ee);
              j = ef_getnum(p, &numlen);
              if (!j)
                return ef_error(FIO_EFGD);
              p += numlen;
              ef_putnum(numval);
            }
          }
        }
      }
      break;

    case 'I':
    case 'i':
      code1 = FED_Iw_m;
      code2 = FED_I;
      goto I_shared;

    case 'O':
    case 'o':
      code1 = FED_Ow_m;
      code2 = FED_O;
      goto I_shared;

    case 'Z':
    case 'z':
      code1 = FED_Zw_m;
      code2 = FED_Z;
    I_shared: /*  process I, O or Z edit descriptor:  */
      rep_count = FALSE;
      j = ef_getnum(p, &numlen);
      if (j == FALSE)
        ef_put(code2);
      else {
        p += numlen;
        ef_put(code1);
        ef_putnum(numval);
        c = ef_nextchar(p, &numlen);
        if (c != '.')
          ef_putnum(1L); /* default value for 'm' field */
        else {
          p += numlen;
          j = ef_getnum(p, &numlen);
          if (!j)
            return ef_error(FIO_EDOT /*"num expected after '.'"*/);
          ef_putnum(numval);
          p += numlen;
        }
      }
      break;

    case '+':
    case '-': /*  number must follow '+' or '-' token:  */
      j = ef_getnum(p, &numlen);
      if (j == FALSE || rep_count)
        return ef_error(FIO_EDOT /*"syntax error (+/-)"*/);
      p += numlen;
      if (c == '-')
        numval = -numval;
      rep_count = TRUE;
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
      p--;
      (void) ef_getnum(p, &numlen);
      p += numlen;
      rep_count = TRUE;

      /*  except for certain edit descriptors, put out 'repeat count'
          now:  */
      c = ef_nextchar(p, &numlen);
      if (c != 'X' && c != 'P' && c != 'H')
        ef_putnum(numval);
      break;

    case 'c':
    case 'j':
    case 'k':
    case 'm': /* case 'n': */
    case 'u':
    case 'v':
    case 'w':
    case 'y':
    case 'C':
    case 'J':
    case 'K':
    case 'M': /* case 'N': */
    case 'U':
    case 'V':
    case 'W':
    case 'Y':
    case 'n':
    case 'N':
      return ef_error(FIO_ELETTER /*"unrecognized format code"*/);

    case '*':
      if (paren_level != 0 ||
          p > lastchar ||
          *p != '(') {
        /* A F'08 unlimited repeat count can appear only before a
         * parenthesized list.
         */
        return ef_error(FIO_EFSYNTAX);
      }
      rep_count = TRUE;
      unlimited_repeat_count = TRUE;
      ef_putnum(0x7fffffff);
      break;

    default:
      return ef_error(FIO_ECHAR /*"illegal char"*/);

    check_no_repeat_count:
      if (rep_count)
        return ef_error(FIO_EFSYNTAX /*"syntax (repcount)"*/);
    }

  } /* while (p <= lastchar); */

  assert(p == lastchar + 1); /* run off end of format (?)  */

end_of_format:
  if (paren_level != 0)
    return ef_error(FIO_EPAREN /*"unbal parens"*/);
  if (rep_count)
    return ef_error(FIO_EFSYNTAX /*"syntax - number"*/);

  if (envar_fortranopt != NULL && strstr(envar_fortranopt, "vaxio") != NULL)
    ;
  else if (enclosing_parens) {
    return ef_error(FIO_EENDFMT /*"unexpected end of format"*/);
  }

  ef_put(FED_END); /* end of format */
  ef_put(reversion_loc);

  return 0; /* no error */
}

/* ------------------------------------------------------------------- */

static ERRCODE
check_outer_parens(char *p, /* ptr to format string to be encoded */
                   __CLEN_T len)
{
  char *q;

  if (len < 1 || p == 0)
    return FIO_ELPAREN; /*"no starting '('" */

  q = p + (len - 1);

  /*  scan past leading blanks:   */

  for (; *p == ' ' && p <= q; p++)
    ;

  if (q < p)            /*"illegal, empty format"*/
    return FIO_ELPAREN; /*"no starting '('" */

  enclosing_parens = FALSE;
  if (*p == '(') {
    enclosing_parens = TRUE;
    p++; /* point to first character following '(':  */
  }
  if (envar_fortranopt != NULL && strstr(envar_fortranopt, "vaxio") != NULL)
    ;
  else if (!enclosing_parens) {
    return FIO_ELPAREN; /*"no starting '('" */
  }

  firstchar = p;
  lastchar = q;
  return 0;
}

/* ------------------------------------------------------------------- */

static bool ef_getnum(
    /*  if first token, beginning at point p, is a number, assign its
        value to numval and return TRUE:  */
    char *p, int *len) /* return number of characters scanned */
{
  char *begin = p;
  int c;
  int retlen;

  while (p <= lastchar && *p == ' ')
    p++;
  if (p > lastchar)
    return FALSE;

  c = *p++;
  if (!is_digit(c))
    return FALSE;

  numval = 0;

  do {
    numval = 10 * numval + (c - '0');
    c = ef_nextchar(p, &retlen);
    p += retlen;
  } while (is_digit(c));

  *len = p - begin - 1;
  return TRUE; /* number was present */
}

/* ---------------------------------------------------------------- */

static int ef_nextchar(char *p, int *len)
{
  char *begin = p, c;

  while (p <= lastchar && *p == ' ')
    p++;
  *len = p - begin + 1;
  if (p > lastchar)
    return '\0';

  c = *p;
  if (c >= 'a' && c <= 'z') /* convert to u.c. for convenience: */
    c = c + ('A' - 'a');
  return c;
}

/* ---------------------------------------------------------------- */

#ifdef FLANG_ENCODEFMT_UNUSED
/* call after encounter DT */
static int ef_nextdtchar(char *p, int *len)
{
  char *begin = p, c;

  if (p <= lastchar && (*p == '\'' || *p == '('))
    p++;
  *len = p - begin + 1;
  if (p > lastchar)
    return '\0';

  c = *p;
  if (c >= 'a' && c <= 'z') /* convert to u.c. for convenience: */
    c = c + ('A' - 'a');
  return c;
}
#endif

/* -------------------------------------------------------------------- */

static void
ef_put(INT val)
{
  if (curpos >= buffsize)
    ef_alloc(0);
  buff[curpos] = val;
  curpos++;
}

/* ------------------------------------------------------------------ */

static void
ef_putnum(INT val)
{
  if (curpos + 1 >= buffsize)
    ef_alloc(0);
  buff[curpos++] = 0;
  buff[curpos++] = val;
}

/* ----------------------------------------------------------------- */
static void
ef_putvlist(char *p, int *len)
/* always put vlist as INT8,
 * ENTF90IO(DTS_FMTR,dts_fmtr)/ENTF90IO(DTS_FMTW,dts_fmtw)
 * will handle it if it were i4 */
{
  char *begin = p;
  char *op = p;
  INT i, j, cnt;

  cnt = 1;
  while (op <= lastchar && *op != ')') {
    if (*op == ',') {
      ++cnt;
    }
    ++op;
  }

  if (cnt) {
    ef_putnum(cnt);
  }

  i = 0;

  if (curpos + 1 >= buffsize)
    ef_alloc(0);

  /* this value will be change to non-zero in 
   * ENTF90IO(DTS_FMTR,dts_fmtr)/ENTF90IO(DTS_FMTW,dts_fmtw)
   * when this particular vlist first encounter
   */

  ef_putnum(0L);

  while (p <= lastchar && *p == ' ')
    ++p;

  while (p <= lastchar && *p != ')') {
    int negate = *p == '-';
    if (*p == '+' || *p == '-')
      ++p;
    j = ef_getnum(p, len);

    if (!j) {
      break;
    } else {
      ++i;
    }
    if (curpos + 1 >= buffsize)
      ef_alloc(0);
    if (negate)
      numval = -numval;
    buff[curpos++] = (__INT8_T)numval;
    curpos++; /* make sure the size of numval is 8 */

    if (curpos + 1 >= buffsize)
      ef_alloc(0);
    p += *len;
    while ((*p == ',' || *p == ' ') && p <= lastchar)
      ++p;
  }
#if DEBUG
  if (i != cnt) {
    printf("in cnt:%d is not the same as out cnt:%d\n", cnt, i);
  }
#endif

  *len = p - begin + 1;
}

/* ----------------------------------------------------------------- */

static ERRCODE ef_putstring(
    char *p, INT len,
    int quote_count) /* always 0 for Hollerith; number of '''s in string */
{
  char *q;

  if (len - quote_count < 0 || p + (len - 1) > lastchar)
    return FIO_ESTRING /*"illegal Hollerith or character string"*/;

  len -= quote_count;
  ef_put((INT)FED_STR);
  ef_put(len);
  if (curpos + len > buffsize)
    ef_alloc(len);

  q = (char *)&buff[curpos];
  curpos += (len + 3) >> 2;

  while (len--) {
    if (*p == quote && quote_count > 0)
      quote_count--, p++;
    *q++ = *p;
    p++;
  }

  return 0;
}

#define DT_LEN 2
static void
ef_putdt()
{
  char *q;

  ef_putnum(2L);
  if (curpos + DT_LEN + 16 > buffsize)
    ef_alloc(DT_LEN + 16);

  q = (char *)&buff[curpos];
  *q++ = 'D';
  *q++ = 'T';
  curpos += (DT_LEN + 3) >> 2;
}

static void
ef_putiotype(char *p, int *numlen)
/* also check if vlist is present */
{
  char *q = p;
  char *tptr, *fptr;
  int n = 0;
  int vlist_ispresent = 0;
  int len = 0;
  *numlen = 0;
  while (q <= lastchar && *q != '\'' && *q != '\"') {
    ++n;
    ++len;
    ++q;
  }
  n++; /* ' */
  ++q;
  if (*q == '(') {
    n++;
    vlist_ispresent = 1;
  }

  if (vlist_ispresent)
    ef_putnum(2L);
  else
    ef_putnum(1L);

  ef_putnum(DT_LEN + len);

  if (curpos + DT_LEN + len + 16 > buffsize)
    ef_alloc(DT_LEN + len + 16);

  tptr = (char *)&buff[curpos];
  *tptr++ = 'D';
  *tptr++ = 'T';
  fptr = p;
  while (fptr != q) {
    *tptr++ = *fptr++;
  }
  curpos += (DT_LEN + len + 3) >> 2;

  len = 0;
  p = p + n;
  if (vlist_ispresent) {
    ef_putvlist(p, &len);
    n += len;
  }
  *numlen = n;
}

/* ------------------------------------------------------------------ */

static void
ef_alloc(int len)
{
  buffsize += (300 + len);
  if (buff == NULL)
    buff = (INT *)malloc(buffsize * sizeof(INT));
  else
    buff = (INT *)realloc(buff, buffsize * sizeof(INT));
  fioFcbTbls.enctab = buff;
  assert(buff != NULL);
}

/* ------------------------------------------------------------------ */

static INT
ef_error(ERRCODE code)
/*  store error code indication at beginning of fmt output buffer: */
{
  curpos = 0;
  ef_put((INT)FED_ERROR);
  ef_put((INT)code);
  return 1;
}

/* handle either character or non-character format string */

__INT_T
ENTF90IO(ENCODE_FMTA, encode_fmta)
(__INT_T *kind,  /* type of data containing format string */
 __INT_T *nelem, /* number of elements if array */
 DCHAR(str)      /* unencoded format string */
 DCLEN64(str))
{
  __CLEN_T len;
  int s = 0;
  len = (*kind == __STR) ? CLEN(str) : GET_DIST_SIZE_OF(*kind);
  buff = NULL;
  buffsize = 0;

  if (LOCAL_MODE) {
    s = _f90io_encode_fmt(CADR(str), nelem, len);
    __fortio_errend03();
    return s;
  }

  if (GET_DIST_LCPU == GET_DIST_IOPROC)
    (void)_f90io_encode_fmt(CADR(str), nelem, len);
  __fortio_errend03();
  return 0;
}
/* 32 bit CLEN version */
__INT_T
ENTF90IO(ENCODE_FMT, encode_fmt)
(__INT_T *kind,  /* type of data containing format string */
 __INT_T *nelem, /* number of elements if array */
 DCHAR(str)      /* unencoded format string */
 DCLEN(str))
{
  return ENTF90IO(ENCODE_FMTA, encode_fmta) (kind, nelem, CADR(str),
           (__CLEN_T)CLEN(str));
}

__INT_T
ENTCRF90IO(ENCODE_FMTA, encode_fmta)
(__INT_T *kind,  /* type of data containing format string */
 __INT_T *nelem, /* number of elements if array */
 DCHAR(str)      /* unencoded format string */
 DCLEN64(str))
{
  __CLEN_T len;
  int s = 0;
  buff = NULL;
  buffsize = 0;
  len = (*kind == __STR) ? CLEN(str) : GET_DIST_SIZE_OF(*kind);
  s = _f90io_encode_fmt(CADR(str), nelem, len);
  __fortio_errend03();
  return s;
}
/* 32 bit CLEN version */
__INT_T
ENTCRF90IO(ENCODE_FMT, encode_fmt)
(__INT_T *kind,  /* type of data containing format string */
 __INT_T *nelem, /* number of elements if array */
 DCHAR(str)      /* unencoded format string */
 DCLEN(str))
{
  return ENTCRF90IO(ENCODE_FMTA, encode_fmta) (kind, nelem, CADR(str),
                            (__CLEN_T)CLEN(str));
}

/* address of character format string is passed in an integer variable */

__INT_T
ENTF90IO(ENCODE_FMTV, encode_fmtv)
(char **str) /* address of ptr to unencoded format string */
{
  int len = 999999; /* no restriction on length */
  int s = 0;
  __INT_T nelem = 1;
  buff = NULL;
  buffsize = 0;
  if (LOCAL_MODE) {
    s = _f90io_encode_fmt(*str, &nelem, len);
    __fortio_errend03();
    return s;
  }

  if (GET_DIST_LCPU == GET_DIST_IOPROC)
    (void)_f90io_encode_fmt(*str, &nelem, len);
  __fortio_errend03();
  return 0;
}

__INT_T
ENTCRF90IO(ENCODE_FMTV, encode_fmtv)
(char **str) /* address of ptr to unencoded format string */
{
  int len = 999999; /* no restriction on length */
  int s = 0;
  __INT_T nelem = 1;
  buff = NULL;
  buffsize = 0;
  s = _f90io_encode_fmt(*str, &nelem, len);
  __fortio_errend03();
  return s;
}
