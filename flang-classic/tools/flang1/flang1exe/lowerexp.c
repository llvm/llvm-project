/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
  \file
  \brief Routines used by lower.c for lowering to ILMs
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "comm.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "ast.h"
#include "semant.h"
#include "dinit.h"
#include "soc.h"
#include "gramtk.h"
#include "rte.h"
#include "extern.h"
#include "rtlRtns.h"

#define INSIDE_LOWER
#include "lower.h"

static LOGICAL lower_check_ast(int ast, int *unused);

void
ast_error(const char *s, int ast)
{
  lerror("%s [ast=%d,asttype=%d,datatype=%d]", s, ast, A_TYPEG(ast),
         A_REPLG(ast));
  if (gbl.dbgfil) {
    if (gbl.dbgfil != stderr) {
      fprintf(gbl.dbgfil,
              "---------------------------\n"
              "%s [ast=%d,asttype=%d,datatype=%d]\n",
              s, ast, A_TYPEG(ast), A_REPLG(ast));
    }
#if DEBUG
    dump_one_ast(ast);
    dbg_print_ast(ast, gbl.dbgfil);
#endif
  }
} /* ast_error */

/* convert whatever type ilm is to BINT */
static int
conv_bint_ilm(int ast, int ilm, int dtype)
{
  int s;
  char *cp;
  int n[4];
  switch (DTYG(dtype)) {
  case TY_BLOG:
  case TY_BINT:
    break;
  case TY_SLOG:
  case TY_SINT:
    ilm = plower("oi", "STOI", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "ITOSC", ilm);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "I8TOI", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
  case TY_REAL:
    ilm = plower("oi", "FIX", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
  case TY_DBLE:
    ilm = plower("oi", "DFIX", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "QFIX", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "REAL", ilm);
    ilm = plower("oi", "FIX", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "DREAL", ilm);
    ilm = plower("oi", "DFIX", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "QREAL", ilm);
    ilm = plower("oi", "QFIX", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
#endif
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", s);
      ilm = plower("oi", "UITOSC", ilm);
    } else {
      ilm = plower("oi", "UITOSC", ilm);
    }
    break;
  case TY_DWORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", s);
    } else {
      ilm = plower("oi", "K2I", ilm);
      ilm = plower("oi", "ITOSC", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_BINT));
      ilm = plower("oS", "ICON", s);
    } else {
      ast_error("unknown hollerith type for conversion to integer", ast);
    }
    break;
  case TY_CHAR:
    if (!ast || A_TYPEG(ast) != A_CNST) {
      ast_error("cannot convert string to integer", ast);
    } else {
      int sptr;
      sptr = A_SPTRG(ast);
      cp = stb.n_base + CONVAL1G(sptr);
      holtonum(cp, n, 1);
      s = lower_getintcon(n[3]);
      ilm = plower("oS", "ICON", s);
    }
    break;
  default:
    ast_error("unknown source type for conversion to integer", ast);
    break;
  }
  return ilm;
} /* conv_bint_ilm */

/* convert whatever type ast is to BINT */
static int
conv_bint(int ast)
{
  return conv_bint_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_bint */

/* convert whatever type ilm is to SINT */
static int
conv_sint_ilm(int ast, int ilm, int dtype)
{
  int s;
  char *cp;
  int n[4];
  switch (DTYG(dtype)) {
  case TY_BLOG:
  case TY_BINT:
    ilm = plower("oi", "SCTOI", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_SLOG:
  case TY_SINT:
    break;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "I8TOI", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_REAL:
    ilm = plower("oi", "FIX", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_DBLE:
    ilm = plower("oi", "DFIX", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "QFIX", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "REAL", ilm);
    ilm = plower("oi", "FIX", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "DREAL", ilm);
    ilm = plower("oi", "DFIX", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "QREAL", ilm);
    ilm = plower("oi", "QFIX", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
#endif
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", s);
      ilm = plower("oi", "UITOS", ilm);
    } else {
      ilm = plower("oi", "UITOS", ilm);
    }
    break;
  case TY_DWORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", s);
      ilm = plower("oi", "UITOS", ilm);
    } else {
      ilm = plower("oi", "K2I", ilm);
      ilm = plower("oi", "ITOS", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_SINT));
      ilm = plower("oS", "ICON", s);
    } else {
      ast_error("unknown hollerith type for conversion to integer", ast);
    }
    break;
  case TY_CHAR:
    if (!ast || A_TYPEG(ast) != A_CNST) {
      ast_error("cannot convert string to integer", ast);
    } else {
      int sptr;
      sptr = A_SPTRG(ast);
      cp = stb.n_base + CONVAL1G(sptr);
      holtonum(cp, n, 2);
      s = lower_getintcon(n[3]);
      ilm = plower("oS", "ICON", s);
      ilm = plower("oi", "UITOS", ilm);
    }
    break;
  default:
    ast_error("unknown source type for conversion to integer", ast);
    break;
  }
  return ilm;
} /* conv_sint_ilm */

/* convert whatever type ast is to SINT */
static int
conv_sint(int ast)
{
  return conv_sint_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_sint */

/* convert whatever type ilm is to INT */
static int
conv_int_ilm(int ast, int ilm, int dtype)
{
  int s;
  char *cp;
  int n[4];
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
    ilm = plower("oi", "SCTOI", ilm);
    break;
  case TY_SINT:
  case TY_SLOG:
    ilm = plower("oi", "STOI", ilm);
    break;
  case TY_INT:
  case TY_LOG:
    break;
  case TY_PTR:
    if (XBIT(49, 0x100)) { /* 64-bit pointers */
      ilm = plower("oi", "I8TOI", ilm);
    }
    break;
  case TY_INT8:
  case TY_LOG8:
    ilm = plower("oi", "I8TOI", ilm);
    break;
  case TY_REAL:
    ilm = plower("oi", "FIX", ilm);
    break;
  case TY_DBLE:
    ilm = plower("oi", "DFIX", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "QFIX", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "REAL", ilm);
    ilm = plower("oi", "FIX", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "DREAL", ilm);
    ilm = plower("oi", "DFIX", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "QREAL", ilm);
    ilm = plower("oi", "QFIX", ilm);
    break;
#endif
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", s);
    } else {
      ilm = plower("oi", "UITOI", ilm);
    }
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", s);
    } else {
      ilm = plower("oi", "K2I", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", s);
    } else {
      ast_error("unknown hollerith type for conversion to integer", ast);
    }
    break;
  case TY_CHAR:
    if (!ast || A_TYPEG(ast) != A_CNST) {
      ast_error("cannot convert string to integer", ast);
    } else {
      int sptr;
      sptr = A_SPTRG(ast);
      cp = stb.n_base + CONVAL1G(sptr);
      holtonum(cp, n, 4);
      s = lower_getintcon(n[3]);
      ilm = plower("oS", "ICON", s);
    }
    break;
  default:
    ast_error("unknown source type for conversion to integer", ast);
    break;
  }
  return ilm;
} /* conv_int_ilm */

/* convert whatever type ast is to INT */
static int
conv_int(int ast)
{
  return conv_int_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_int */

/* convert whatever type ilm is to INT8 */
static int
conv_int8_ilm(int ast, int ilm, int dtype)
{
  int s;
  char *cp;
  int n[4];
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
  case TY_INT:
  case TY_LOG:
    ilm = conv_int_ilm(ast, ilm, dtype);
    ilm = plower("oi", "ITOI8", ilm);
    break;
  case TY_PTR:
    if (!XBIT(49, 0x100)) { /* not 64-bit pointers */
      ilm = plower("oi", "ITOI8", ilm);
    }
    break;
  case TY_INT8:
  case TY_LOG8:
    break;
  case TY_REAL:
    ilm = plower("oi", "KFIX", ilm);
    break;
  case TY_DBLE:
    ilm = plower("oi", "KDFIX", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "KQFIX", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "REAL", ilm);
    ilm = plower("oi", "KFIX", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "DREAL", ilm);
    ilm = plower("oi", "KDFIX", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "QREAL", ilm);
    ilm = plower("oi", "KQFIX", ilm);
    break;
#endif
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_INT8);
      ilm = plower("oS", "KCON", s);
    } else {
      ilm = plower("oi", "UI2K", ilm);
    }
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_INT8);
      ilm = plower("oS", "KCON", s);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_INT8);
      ilm = plower("oS", "KCON", s);
    } else {
      ast_error("unknown hollerith type for conversion to integer*8", ast);
    }
    break;
  case TY_CHAR:
    if (!ast || A_TYPEG(ast) != A_CNST) {
      ast_error("cannot convert string to integer", ast);
    } else {
      int sptr;
      sptr = A_SPTRG(ast);
      cp = stb.n_base + CONVAL1G(sptr);
      holtonum(cp, n, 8);
      if (flg.endian == 0) {
        int swap;
        /* for little endian, need to swap words in each double word
         * quantity.  Order of bytes in a word is okay, but not the
         * order of words.
         */
        swap = n[2];
        n[2] = n[3];
        n[3] = swap;
      }
      s = getcon(n + 2, DT_INT8);
      VISITP(s, 1);
      lower_use_datatype(DT_INT8, 1);
      ilm = plower("oS", "ICON", s);
    }
    break;
  default:
    ast_error("unknown source type for conversion to integer*8", ast);
    break;
  }
  return ilm;
} /* conv_int8_ilm */

/* convert whatever type ast is to INT8 */
static int
conv_int8(int ast)
{
  return conv_int8_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_int8 */

/* convert whatever type ilm is to WORD */
static int
conv_word_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
    ilm = plower("oi", "SCTOUI", ilm);
    break;
  case TY_SINT:
  case TY_SLOG:
    ilm = plower("oi", "STOUI", ilm);
    break;
  case TY_INT:
  case TY_LOG:
    ilm = plower("oi", "ITOUI", ilm);
    break;
  case TY_INT8:
  case TY_LOG8:
    ilm = plower("oi", "K2I", ilm);
    break;
  case TY_REAL:
    ilm = plower("oi", "RTOUI", ilm);
    break;
  case TY_DBLE:
    ilm = plower("oi", "DTOUI", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "QTOUI", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "CTOUDI", ilm);
    ilm = plower("oi", "UDITOUI", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "CDTOUDI", ilm);
    ilm = plower("oi", "UDITOUI", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "CQTOUDI", ilm);
    ilm = plower("oi", "UDITOUI", ilm);
    break;
#endif
  case TY_WORD:
    break;
  case TY_DWORD:
    ilm = plower("oi", "K2I", ilm);
    ilm = plower("oi", "ITOUI", ilm);
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_WORD));
      ilm = plower("oS", "ICON", s);
    } else {
      ast_error("unknown hollerith type for conversion to word", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to word", ast);
    break;
  }
  return ilm;
} /* conv_word_ilm */

/* convert whatever type ast is to WORD */
static int
conv_word(int ast)
{
  return conv_word_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_word */

/* convert whatever type ilm is to DWORD */
static int
conv_dword_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
    ilm = plower("oi", "SCTOI", ilm);
    ilm = plower("oi", "I2K", ilm);
    break;
  case TY_SINT:
  case TY_SLOG:
    ilm = plower("oi", "STOI", ilm);
    ilm = plower("oi", "I2K", ilm);
    break;
  case TY_INT:
  case TY_LOG:
    ilm = plower("oi", "I2K", ilm);
    break;
  case TY_INT8:
  case TY_LOG8:
    break;
  case TY_REAL:
    ilm = plower("oi", "RTOUI", ilm);
    ilm = plower("oi", "UI2K", ilm);
    break;
  case TY_DBLE:
    ilm = plower("oi", "D2K", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "Q2K", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "CTOUDI", ilm);
    ilm = plower("oi", "UDITOD", ilm);
    ilm = plower("oi", "D2K", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "CDTOUDI", ilm);
    ilm = plower("oi", "UDITOD", ilm);
    ilm = plower("oi", "D2K", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "CQTOUDI", ilm);
    ilm = plower("oi", "UDITOD", ilm);
    ilm = plower("oi", "D2K", ilm);
    break;
#endif
  case TY_WORD:
    ilm = plower("oi", "UI2K", ilm);
    break;
  case TY_DWORD:
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_INT8);
      ilm = plower("oS", "KCON", s);
    } else {
      ast_error("unknown hollerith type for conversion to integer*8", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to integer*8", ast);
    break;
  }
  return ilm;
} /* conv_dword_ilm */

/* convert whatever type ast is to DWORD */
static int
conv_dword(int ast)
{
  return conv_dword_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_dword */

/* convert whatever type ilm is to BLOG */
static int
conv_blog_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BLOG:
  case TY_BINT:
    break;
  case TY_SLOG:
  case TY_SINT:
    ilm = plower("oi", "STOI", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    break;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "ITOSC", ilm);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "I8TOI", ilm);
    ilm = plower("oi", "ITOSC", ilm);
    FLANG_FALLTHROUGH;
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_SLOG));
      ilm = plower("oS", "LCON", s);
    } else {
      ilm = plower("oi", "UITOSC", ilm);
    }
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_SLOG));
      ilm = plower("oS", "LCON", s);
    } else {
      ilm = plower("oi", "K2I", ilm);
      ilm = plower("oi", "ITOSC", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_LOG4));
      ilm = plower("oS", "LCON", s);
    } else {
      ast_error("unknown type for conversion to logical", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to logical", ast);
    break;
  }
  return ilm;
} /* conv_blog_ilm */

/* convert whatever type ast is to BLOG */
static int
conv_blog(int ast)
{
  return conv_blog_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_blog */

/* convert whatever type ilm is to SLOG */
static int
conv_slog_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BLOG:
  case TY_BINT:
    ilm = plower("oi", "SCTOI", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_SLOG:
  case TY_SINT:
    break;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_SLOG));
      ilm = plower("oS", "LCON", s);
    } else {
      ilm = plower("oi", "UITOS", ilm);
    }
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "I8TOI", ilm);
    ilm = plower("oi", "ITOS", ilm);
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_SLOG));
      ilm = plower("oS", "LCON", s);
    } else {
      ilm = plower("oi", "K2I", ilm);
      ilm = plower("oi", "ITOS", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_LOG4));
      ilm = plower("oS", "LCON", s);
    } else {
      ast_error("unknown type for conversion to logical", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to logical", ast);
    break;
  }
  return ilm;
} /* conv_slog_ilm */

/* convert whatever type ast is to SLOG */
static int
conv_slog(int ast)
{
  return conv_slog_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_slog */

/* convert whatever type ilm is to LOG */
static int
conv_log_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BLOG:
  case TY_BINT:
    ilm = plower("oi", "SCTOI", ilm);
    break;
  case TY_SLOG:
  case TY_SINT:
    ilm = plower("oi", "STOI", ilm);
    break;
  case TY_LOG:
    break;
  case TY_INT:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getlogcon(cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_LOG4));
      ilm = plower("oS", "LCON", s);
    } else {
      return ilm;
    }
    break;
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getlogcon(cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_LOG4));
      ilm = plower("oS", "LCON", s);
    } else {
      ilm = plower("oi", "UITOI", ilm);
    }
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getlogcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_LOG4));
      ilm = plower("oS", "LCON", s);
    } else {
      ilm = plower("oi", "K2I", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getintcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_LOG4));
      ilm = plower("oS", "LCON", s);
    } else {
      ast_error("unknown type for conversion to logical", ast);
    }
    break;
  case TY_REAL:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
  case TY_LOG8:
  case TY_INT8:
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
    ilm = conv_int_ilm(ast, ilm, dtype);
    break;
  case TY_CHAR:
    if (DTY(dtype + 1) == astb.i1 && ast && A_TYPEG(ast) == A_CNST) {
      int sptr = A_SPTRG(ast);
      /* create an integer with the value of the character */
      s = (int)(stb.n_base[CONVAL1G(sptr)]);
      s = lower_getintcon(s);
      ilm = plower("oS", "ICON", s);
    } else {
      ast_error("cannot convert string to logical", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to logical", ast);
    break;
  }
  return ilm;
} /* conv_log_ilm */

/* convert whatever type ast is to LOG */
static int
conv_log(int ast)
{
  return conv_log_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_log */

/* convert whatever type ilm is to LOG8 */
static int
conv_log8_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BLOG:
  case TY_BINT:
  case TY_SLOG:
  case TY_SINT:
  case TY_LOG:
  case TY_INT:
  case TY_REAL:
    ilm = conv_log_ilm(ast, ilm, dtype);
    ilm = plower("oi", "ITOI8", ilm);
    break;
  case TY_WORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = A_SPTRG(ast);
      s = cngcon(CONVAL2G(s), DTYG(dtype), DT_LOG8);
      ilm = plower("oS", "KCON", s);
    } else {
      ilm = plower("oi", "UI2K", ilm);
    }
    break;
  case TY_DWORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_LOG8);
      ilm = plower("oS", "KCON", s);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_LOG8);
      ilm = plower("oS", "KCON", s);
    } else {
      ast_error("unknown type for conversion to logical", ast);
    }
    break;
  case TY_LOG8:
  case TY_INT8:
    break;
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
    ilm = conv_int8_ilm(ast, ilm, dtype);
    break;
  case TY_CHAR:
    if (DTY(dtype + 1) == astb.i1 && ast && A_TYPEG(ast) == A_CNST) {
      int sptr = A_SPTRG(ast);
      /* create an integer with the value of the character */
      s = (int)(stb.n_base[CONVAL1G(sptr)]);
      s = lower_getintcon(s);
      ilm = plower("oS", "ICON", s);
      ilm = plower("oi", "ITOI8", ilm);
    } else {
      ast_error("cannot convert string to logical", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to logical*8", ast);
    break;
  }
  return ilm;
} /* conv_log8_ilm */

/* convert whatever type ast is to LOG8 */
static int
conv_log8(int ast)
{
  return conv_log8_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_log8 */

/* convert whatever type ilm is to REAL */
static int
conv_real_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
  case TY_LOG:
  case TY_INT:
    ilm = conv_int_ilm(ast, ilm, dtype);
    ilm = plower("oi", "FLOAT", ilm);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "FLOATK", ilm);
    break;
  case TY_REAL:
    break;
  case TY_DBLE:
    ilm = plower("oi", "SNGL", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "SNGQ", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "REAL", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "DREAL", ilm);
    ilm = plower("oi", "SNGL", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "QREAL", ilm);
    ilm = plower("oi", "SNGQ", ilm);
    break;
#endif
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getrealcon(
          cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_REAL4));
      ilm = plower("oS", "RCON", s);
    } else {
      ilm = plower("oi", "UITOR", ilm);
    }
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getrealcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_REAL4));
      ilm = plower("oS", "RCON", s);
    } else {
      ilm = plower("oi", "K2R", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getrealcon(cngcon(A_SPTRG(ast), DTYG(dtype), DT_REAL4));
      ilm = plower("oS", "RCON", s);
    } else {
      ast_error("unknown type for conversion to real", ast);
    }
    break;
  case TY_PTR:
    dtype = DTY(dtype + 1);
    if (DTY(dtype) == TY_PROC)
      dtype = DTY(dtype + 1);
    return conv_real_ilm(ast, ilm, dtype);
  default:
    ast_error("unknown source type for conversion to real", ast);
    break;
  }
  return ilm;
} /* conv_real_ilm */

/* convert whatever type ast is to REAL */
static int
conv_real(int ast)
{
  return conv_real_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_real */

/* convert whatever type ilm is to DBLE */
static int
conv_dble_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
    ilm = conv_int_ilm(ast, ilm, dtype);
    FLANG_FALLTHROUGH;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "DFLOAT", ilm);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "DFLOATK", ilm);
    break;
  case TY_REAL:
    ilm = plower("oi", "DBLE", ilm);
    break;
  case TY_DBLE:
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "DBLEQ", ilm);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "REAL", ilm);
    ilm = plower("oi", "DBLE", ilm);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "DREAL", ilm);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "QREAL", ilm);
    ilm = plower("oi", "DBLEQ", ilm);
    break;
#endif
  case TY_WORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_REAL8);
      ilm = plower("oS", "DCON", s);
    } else {
      ilm = plower("oi", "UITOD", ilm);
    }
    break;
  case TY_DWORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_REAL8);
      ilm = plower("oS", "DCON", s);
    } else {
      ilm = plower("oi", "K2D", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_REAL8);
      ilm = plower("oS", "DCON", s);
    } else {
      ast_error("unknown hollerith type for conversion to real*8", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to double precision", ast);
    break;
  }
  return ilm;
} /* conv_dble_ilm */

/* convert whatever type ast is to DBLE */
static int
conv_dble(int ast)
{
  return conv_dble_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_dble */

#ifdef TARGET_SUPPORTS_QUADFP
/* convert whatever type ilm is to QUAD */
static int conv_quad_ilm(int ast, int ilm, int dtype)
{
  int s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
    ilm = conv_int_ilm(ast, ilm, dtype);
    FLANG_FALLTHROUGH;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "QFLOAT", ilm);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "QFLOATK", ilm);
    break;
  case TY_REAL:
    ilm = plower("oi", "RQUAD", ilm);
    break;
  case TY_DBLE:
    ilm = plower("oi", "DQUAD", ilm);
    break;
  case TY_QUAD:
    break;
  case TY_CMPLX:
    /* get the real part from the complex */
    ilm = plower("oi", "REAL", ilm);
    /* convert the float to quad precision */
    ilm = plower("oi", "RQUAD", ilm);
    break;
  case TY_DCMPLX:
    /* get the real part from the complex */
    ilm = plower("oi", "DREAL", ilm);
    /* convert the double to quad precision */
    ilm = plower("oi", "DQUAD", ilm);
    break;
  case TY_QCMPLX:
    /* get the real part from the complex */
    ilm = plower("oi", "QREAL", ilm);
    break;
  case TY_WORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_QUAD);
      ilm = plower("oS", "QCON", s);
    } else {
      ilm = plower("oi", "UITOQ", ilm);
    }
    break;
  case TY_DWORD:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_QUAD);
      ilm = plower("oS", "QCON", s);
    } else {
      ilm = plower("oi", "K2Q", ilm);
    }
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_QUAD);
      ilm = plower("oS", "QCON", s);
    } else {
      ast_error("unknown hollerith type for conversion to real*16", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to quad precision", ast);
    break;
  }
  return ilm;
} /* conv_quad_ilm */

/* convert whatever type ast is to QUAD */
static int conv_quad(int ast)
{
  return conv_quad_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_dble */
#endif

/* convert whatever type ilm is to CMPLX */
static int
conv_cmplx_ilm(int ast, int ilm, int dtype)
{
  int ilmimag, ilmreal, s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
    ilm = conv_int_ilm(ast, ilm, dtype);
    FLANG_FALLTHROUGH;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "FLOAT", ilm);
    ilmimag = plower("oS", "RCON", lowersym.realzero);
    ilm = plower("oii", "CMPLX", ilm, ilmimag);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "I8TOI", ilm);
    ilm = plower("oi", "FLOAT", ilm);
    ilmimag = plower("oS", "RCON", lowersym.realzero);
    ilm = plower("oii", "CMPLX", ilm, ilmimag);
    break;
  case TY_REAL:
    ilmimag = plower("oS", "RCON", lowersym.realzero);
    ilm = plower("oii", "CMPLX", ilm, ilmimag);
    break;
  case TY_DBLE:
    ilm = plower("oi", "SNGL", ilm);
    ilmimag = plower("oS", "RCON", lowersym.realzero);
    ilm = plower("oii", "CMPLX", ilm, ilmimag);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "SNGQ", ilm);
    ilmimag = plower("oS", "RCON", lowersym.realzero);
    ilm = plower("oii", "CMPLX", ilm, ilmimag);
    break;
#endif
  case TY_CMPLX:
    break;
  case TY_DCMPLX:
    ilmimag = plower("oi", "DIMAG", ilm);
    ilmimag = plower("oi", "SNGL", ilmimag);
    ilmreal = plower("oi", "DREAL", ilm);
    ilmreal = plower("oi", "SNGL", ilmreal);
    ilm = plower("oii", "CMPLX", ilmreal, ilmimag);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilmimag = plower("oi", "QIMAG", ilm);
    ilmimag = plower("oi", "SNGQ", ilmimag);
    ilmreal = plower("oi", "QREAL", ilm);
    ilmreal = plower("oi", "SNGQ", ilmreal);
    ilm = plower("oii", "CMPLX", ilmreal, ilmimag);
    break;
#endif
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = lower_getrealcon(
          cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_REAL4));
      ilmreal = plower("oS", "RCON", s);
    } else {
      ilmreal = plower("oi", "UITOR", ilm);
    }
    ilmimag = plower("oS", "RCON", lowersym.realzero);
    ilm = plower("oii", "CMPLX", ilmreal, ilmimag);
    break;
  case TY_DWORD:
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_CMPLX8);
      ilm = plower("oS", "CCON", s);
    } else {
      ast_error("unknown type for conversion to complex", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to complex", ast);
    break;
  }
  return ilm;
} /* conv_cmplx_ilm */

/* convert whatever type ast is to CMPLX */
static int
conv_cmplx(int ast)
{
  return conv_cmplx_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_cmplx */

/* convert whatever type ilm is to DCMPLX */
static int
conv_dcmplx_ilm(int ast, int ilm, int dtype)
{
  int ilmimag, ilmreal, s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
    ilm = conv_int_ilm(ast, ilm, dtype);
    FLANG_FALLTHROUGH;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "DFLOAT", ilm);
    ilmimag = plower("oS", "DCON", lowersym.dblezero);
    ilm = plower("oii", "DCMPLX", ilm, ilmimag);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "DFLOATK", ilm);
    ilmimag = plower("oS", "DCON", lowersym.dblezero);
    ilm = plower("oii", "DCMPLX", ilm, ilmimag);
    break;
  case TY_REAL:
    ilm = plower("oi", "DBLE", ilm);
    ilmimag = plower("oS", "DCON", lowersym.dblezero);
    ilm = plower("oii", "DCMPLX", ilm, ilmimag);
    break;
  case TY_DBLE:
    ilmimag = plower("oS", "DCON", lowersym.dblezero);
    ilm = plower("oii", "DCMPLX", ilm, ilmimag);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oi", "DBLEQ", ilm);
    ilmimag = plower("oS", "DCON", lowersym.realzero);
    ilm = plower("oii", "DCMPLX", ilm, ilmimag);
    break;
#endif
  case TY_CMPLX:
    ilmimag = plower("oi", "IMAG", ilm);
    ilmimag = plower("oi", "DBLE", ilmimag);
    ilmreal = plower("oi", "REAL", ilm);
    ilmreal = plower("oi", "DBLE", ilmreal);
    ilm = plower("oii", "DCMPLX", ilmreal, ilmimag);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilmimag = plower("oi", "QIMAG", ilm);
    ilmimag = plower("oi", "DBLEQ", ilmimag);
    ilmreal = plower("oi", "QREAL", ilm);
    ilmreal = plower("oi", "DBLEQ", ilmreal);
    ilm = plower("oii", "DCMPLX", ilmreal, ilmimag);
    break;
#endif
  case TY_DCMPLX:
    break;
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_REAL8);
      ilmreal = plower("oS", "DCON", s);
    } else {
      ilmreal = plower("oi", "UITOD", ilm);
    }
    ilmimag = plower("oS", "DCON", lowersym.dblezero);
    ilm = plower("oii", "DCMPLX", ilmreal, ilmimag);
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_REAL8);
      ilmreal = plower("oS", "DCON", s);
    } else {
      ilmreal = plower("oi", "K2D", ilm);
    }
    ilmimag = plower("oS", "DCON", lowersym.dblezero);
    ilm = plower("oii", "DCMPLX", ilmreal, ilmimag);
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_CMPLX16);
      ilm = plower("oS", "CDCON", s);
    } else {
      ast_error("unknown hollerith type for conversion to complex*16", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to complex*16", ast);
    break;
  }
  return ilm;
} /* conv_dcmplx_ilm */

/* convert whatever type ast is to DCMPLX */
static int
conv_dcmplx(int ast)
{
  return conv_dcmplx_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_dcmplx */

#ifdef TARGET_SUPPORTS_QUADFP
/* convert whatever type ilm is to QCMPLX */
static int conv_qcmplx_ilm(int ast, int ilm, int dtype)
{
  int ilmimag, ilmreal, s;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
    ilm = conv_int_ilm(ast, ilm, dtype);
    FLANG_FALLTHROUGH;
  case TY_LOG:
  case TY_INT:
    ilm = plower("oi", "QFLOAT", ilm);
    ilmimag = plower("oS", "QCON", lowersym.quadzero);
    ilm = plower("oii", "QCMPLX", ilm, ilmimag);
    break;
  case TY_LOG8:
  case TY_INT8:
    ilm = plower("oi", "QFLOATK", ilm);
    ilmimag = plower("oS", "QCON", lowersym.quadzero);
    ilm = plower("oii", "QCMPLX", ilm, ilmimag);
    break;
  case TY_REAL:
    ilm = plower("oi", "RQUAD", ilm);
    ilmimag = plower("oS", "QCON", lowersym.quadzero);
    ilm = plower("oii", "QCMPLX", ilm, ilmimag);
    break;
  case TY_DBLE:
    ilm = plower("oi", "DQUAD", ilm);
    ilmimag = plower("oS", "QCON", lowersym.quadzero);
    ilm = plower("oii", "QCMPLX", ilm, ilmimag);
    break;
  case TY_QUAD:
    ilmimag = plower("oS", "QCON", lowersym.quadzero);
    ilm = plower("oii", "QCMPLX", ilm, ilmimag);
    break;
  case TY_CMPLX:
    ilmimag = plower("oi", "IMAG", ilm);
    ilmimag = plower("oi", "RQUAD", ilmimag);
    ilmreal = plower("oi", "REAL", ilm);
    ilmreal = plower("oi", "RQUAD", ilmreal);
    ilm = plower("oii", "QCMPLX", ilmreal, ilmimag);
    break;
  case TY_DCMPLX:
    ilmimag = plower("oi", "DIMAG", ilm);
    ilmimag = plower("oi", "DQUAD", ilmimag);
    ilmreal = plower("oi", "DREAL", ilm);
    ilmreal = plower("oi", "DQUAD", ilmreal);
    ilm = plower("oii", "QCMPLX", ilmreal, ilmimag);
    break;
  case TY_QCMPLX:
    break;
  case TY_WORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(CONVAL2G(A_SPTRG(ast)), DTYG(dtype), DT_QUAD);
      ilmreal = plower("oS", "QCON", s);
    } else {
      ilmreal = plower("oi", "UITOQ", ilm);
    }
    ilmimag = plower("oS", "QCON", lowersym.quadzero);
    ilm = plower("oii", "QCMPLX", ilmreal, ilmimag);
    break;
  case TY_DWORD:
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_QUAD);
      ilmreal = plower("oS", "QCON", s);
    } else {
      ilmreal = plower("oi", "K2Q", ilm);
    }
    ilmimag = plower("oS", "QCON", lowersym.quadzero);
    ilm = plower("oii", "QCMPLX", ilmreal, ilmimag);
    break;
  case TY_HOLL:
    /* convert by padding with blanks or truncating */
    if (ast && A_TYPEG(ast) == A_CNST) {
      s = cngcon(A_SPTRG(ast), DTYG(dtype), DT_QCMPLX);
      ilm = plower("oS", "CQCON", s);
    } else {
      ast_error("unknown hollerith type for conversion to complex*32", ast);
    }
    break;
  default:
    ast_error("unknown source type for conversion to complex*32", ast);
    break;
  }
  return ilm;
} /* conv_qcmplx_ilm */

/* convert whatever type ast is to QCMPLX */
static int conv_qcmplx(int ast)
{
  return conv_qcmplx_ilm(ast, lower_ilm(ast), A_NDTYPEG(ast));
} /* conv_qcmplx */
#endif

int
lower_conv_ilm(int ast, int ilm, int fromdtype, int todtype)
{
  if (DTYG(fromdtype) == DTYG(todtype))
    return ilm;

  switch (DTYG(todtype)) {
  case TY_BINT:
    ilm = conv_bint_ilm(ast, ilm, fromdtype);
    break;
  case TY_SINT:
    ilm = conv_sint_ilm(ast, ilm, fromdtype);
    break;
  case TY_INT:
    ilm = conv_int_ilm(ast, ilm, fromdtype);
    break;
  case TY_BLOG:
    ilm = conv_blog_ilm(ast, ilm, fromdtype);
    break;
  case TY_SLOG:
    ilm = conv_slog_ilm(ast, ilm, fromdtype);
    break;
  case TY_LOG:
    ilm = conv_log_ilm(ast, ilm, fromdtype);
    break;
  case TY_INT8:
    ilm = conv_int8_ilm(ast, ilm, fromdtype);
    break;
  case TY_REAL:
    ilm = conv_real_ilm(ast, ilm, fromdtype);
    break;
  case TY_DBLE:
    ilm = conv_dble_ilm(ast, ilm, fromdtype);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = conv_quad_ilm(ast, ilm, fromdtype);
    break;
#endif
  case TY_CMPLX:
    ilm = conv_cmplx_ilm(ast, ilm, fromdtype);
    break;
  case TY_DCMPLX:
    ilm = conv_dcmplx_ilm(ast, ilm, fromdtype);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = conv_qcmplx_ilm(ast, ilm, fromdtype);
    break;
#endif
  case TY_WORD:
    ilm = conv_word_ilm(ast, ilm, fromdtype);
    break;
  case TY_DWORD:
    ilm = conv_dword_ilm(ast, ilm, fromdtype);
    break;
  default:
    ast_error("unknown target type for ilm conversion", ast);
    lerror("target type was %d", todtype);
    break;
  }
  return ilm;
} /* lower_conv_ilm */

int
lower_conv(int ast, int dtype)
{
  int ilm, adtype;
  adtype = A_NDTYPEG(ast);
  ilm = lower_ilm(ast);
  if (adtype <= 0 || eq_dtype(DTYG(adtype), DTYG(dtype)))
    return ilm;

  switch (DTYG(dtype)) {
  case TY_BINT:
    ilm = conv_bint(ast);
    break;
  case TY_SINT:
    ilm = conv_sint(ast);
    break;
  case TY_INT:
    ilm = conv_int(ast);
    break;
  case TY_INT8:
    ilm = conv_int8(ast);
    break;
  case TY_BLOG:
    ilm = conv_blog(ast);
    break;
  case TY_SLOG:
    ilm = conv_slog(ast);
    break;
  case TY_LOG:
    ilm = conv_log(ast);
    break;
  case TY_LOG8:
    ilm = conv_log8(ast);
    break;
  case TY_REAL:
    ilm = conv_real(ast);
    break;
  case TY_DBLE:
    ilm = conv_dble(ast);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  /* to support the type conversion */
  case TY_QUAD:
    ilm = conv_quad(ast);
    break;
#endif
  case TY_CMPLX:
    ilm = conv_cmplx(ast);
    break;
  case TY_DCMPLX:
    ilm = conv_dcmplx(ast);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = conv_qcmplx(ast);
    break;
#endif
  case TY_WORD:
    ilm = conv_word(ast);
    break;
  case TY_DWORD:
    ilm = conv_dword(ast);
    break;
  case TY_PTR:
    /* convert to the pointee type */
    return lower_conv(ast, DTY(dtype + 1));
  default:
    ast_error("unknown target type for ast conversion", ast);
    lerror("target type was %d", dtype);
    break;
  }
  return ilm;
} /* lower_conv */

char *
ltyped(const char *opname, int dtype)
{
  static char OP[100];
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_WORD:
    strcpy(OP, "I");
    break;
  case TY_PTR:
    if (XBIT(49, 0x100)) { /* 64-bit pointers */
      strcpy(OP, "K");
    } else {
      strcpy(OP, "I");
    }
    break;
  case TY_INT8:
  case TY_DWORD:
  case TY_LOG8:
    strcpy(OP, "K");
    break;
  case TY_REAL:
    strcpy(OP, "R");
    break;
  case TY_DBLE:
    strcpy(OP, "D");
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    strcpy(OP, "Q");
    break;
#endif
  case TY_CMPLX:
    strcpy(OP, "C");
    break;
  case TY_DCMPLX:
    strcpy(OP, "CD");
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    strcpy(OP, "CQ");
    break;
#endif
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
    strcpy(OP, "L");
    break;
  case TY_CHAR:
    strcpy(OP, "CH");
    break;
  case TY_NCHAR:
    strcpy(OP, "NCH");
    break;
  default:
    strcpy(OP, "");
    lerror("untyped operation %s (type %d)", opname, dtype);
    break;
  }
  strcat(OP, opname);
  return OP;
} /* typed */

static char *
styped(const char *opname, int dtype)
{
  static char OP[100];
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    strcpy(OP, "I");
    break;
  case TY_INT8:
  case TY_LOG8:
    strcpy(OP, "K");
    break;
  case TY_REAL:
    strcpy(OP, "R");
    break;
  case TY_DBLE:
    strcpy(OP, "D");
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    strcpy(OP, "Q");
    break;
#endif
  case TY_CMPLX:
    strcpy(OP, "C");
    break;
  case TY_DCMPLX:
    strcpy(OP, "CD");
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    strcpy(OP, "CQ");
    break;
#endif
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
    strcpy(OP, "L");
    break;
  case TY_CHAR:
    strcpy(OP, "S");
    break;
  case TY_NCHAR:
    strcpy(OP, "NS");
    break;
  default:
    strcpy(OP, "");
    lerror("untyped s-operation %s (type %d)", opname, dtype);
    break;
  }
  strcat(OP, opname);
  return OP;
} /* styped */

/* generate the ILM for a simple arithmetic binary operator.
 * the prefix for the operator name depends on the expression type */
static int
lower_bin_arith(int ast, const char *opname, int ldtype, int rdtype)
{
  int dtype, ilm, lilm, rilm;
  dtype = A_NDTYPEG(ast);
  if (dtype <= 0) {
    ast_error("unrecognized data type in lower_bin_arith", ast);
    return 0;
  }
  lilm = lower_conv(A_LOPG(ast), ldtype);
  rilm = lower_conv(A_ROPG(ast), rdtype);
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_INT8:
  case TY_REAL:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
  case TY_WORD:
  case TY_DWORD:
    /* OK */
    break;
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
    ast_error("logical result for arithmetic operation", ast);
    return 0;
  case TY_CHAR:
  case TY_NCHAR:
    ast_error("character result for arithmetic operation", ast);
    return 0;
  default:
    ast_error("unknown result for arithmetic operation", ast);
    return 0;
  }
  ilm = plower("oii", ltyped(opname, dtype), lilm, rilm);
  return ilm;
} /* lower_bin_arith */

/* generate the ILM for a simple arithmetic unary operator.
 * the prefix for the operator name depends on the expression type */
static int
lower_un_arith(int ast, const char *opname, int ldtype)
{
  int dtype, ilm, lilm;
  dtype = A_NDTYPEG(ast);
  if (dtype <= 0) {
    ast_error("unrecognized data type in lower_un_arith", ast);
    return 0;
  }
  lilm = lower_conv(A_LOPG(ast), ldtype);
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_INT8:
  case TY_REAL:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
  case TY_QCMPLX:
#endif
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_WORD:
  case TY_DWORD:
    break;
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
    ast_error("logical result for arithmetic operation", ast);
    return 0;
  case TY_CHAR:
  case TY_NCHAR:
    ast_error("character result for arithmetic operation", ast);
    return 0;
  default:
    ast_error("unknown result for arithmetic operation", ast);
    return 0;
  }
  ilm = plower("oi", ltyped(opname, dtype), lilm);
  return ilm;
} /* lower_un_arith */

/* generate the ILM for a simple comparison operator.
 * the prefix for the operator name depends on the expression type */
static int
lower_bin_comparison(int ast, const char *op)
{
  int dtype, ilm, lilm, rilm, base;
  char opname[15];

  dtype = A_NDTYPEG(ast);
  if (dtype <= 0) {
    ast_error("unrecognized data type in lower_bin_comparison", ast);
    return 0;
  }
  strcpy(opname, op);
  switch (DTYG(dtype)) {
  case TY_LOG:
  case TY_BLOG:
  case TY_SLOG:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_WORD:
    break;
  case TY_LOG8:
  case TY_INT8:
  case TY_DWORD:
    strcat(opname, "8");
    break;

  case TY_REAL:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
    ast_error("arithmetic result for comparison operation", ast);
    return 0;
  case TY_CHAR:
  case TY_NCHAR:
    ast_error("character result for comparison operation", ast);
    return 0;
  default:
    ast_error("unknown result for comparison operation", ast);
    return 0;
  }
  dtype = A_NDTYPEG(A_LOPG(ast));
  if (dtype <= 0) {
    ast_error("unrecognized data type in lower_bin_comparison", ast);
    return 0;
  }
  base = 0;
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_INT8:
  case TY_REAL:
  case TY_DBLE:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
    break;
  case TY_BLOG:
    dtype = DT_BINT;
    break;
  case TY_SLOG:
    dtype = DT_SINT;
    break;
  case TY_WORD:
  case TY_LOG:
    dtype = DT_INT4;
    break;
  case TY_DWORD:
  case TY_LOG8:
    dtype = DT_INT8;
    break;
  case TY_CHAR:
  case TY_NCHAR:
    base = 1;
    break;
#ifndef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
  case TY_QCMPLX:
#endif
  default:
    ast_error("unknown operand type for comparison operation", ast);
    return 0;
  }
  if (base) {
    lilm = lower_base(A_LOPG(ast));
    rilm = lower_base(A_ROPG(ast));
  } else {
    lilm = lower_ilm(A_LOPG(ast));
    rilm = lower_conv(A_ROPG(ast), dtype);
  }
  ilm = plower("oii", styped("CMP", dtype), lilm, rilm);
  ilm = plower("oi", opname, ilm);
  return ilm;
} /* lower_bin_comparison */

/* for a logical operation (and,or,not) if the operand
 * is not another logical operation, add an lnop */
static int
add_lnop(int ilm, int ast, int dtype)
{
  const char *opc;
  switch (A_TYPEG(ast)) {
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_LNEQV:
    case OP_LEQV:
    case OP_LOR:
    case OP_LAND:
    case OP_SCAND:
      return ilm;
    }
    break;
  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_LNOT:
      return ilm;
    }
    break;
  }
  /* otherwise, add LNOP */
  switch (DTYG(dtype)) {
  case TY_LOG8:
  case TY_INT8:
  case TY_DWORD:
    opc = "LNOP8";
    break;
  default:
    opc = "LNOP";
    break;
  }
  ilm = plower("oi", opc, ilm);
  plower("o", "NOP");
  return ilm;
} /* add_lnop */

/* generate the ILM for a simple logical binary operator.
 * the suffix for the operator name depends on the expression type */
static int
lower_bin_logical(int ast, const char *op)
{
  int dtype, ilm, lilm, rilm;
  char opname[15];
  dtype = A_NDTYPEG(ast);
  if (dtype <= 0) {
    ast_error("unrecognized data type in lower_bin_logical", ast);
    return 0;
  }
  strcpy(opname, op);
  switch (DTYG(dtype)) {
  case TY_LOG:
  case TY_BLOG:
  case TY_SLOG:
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_WORD:
    break;
  case TY_LOG8:
  case TY_INT8:
  case TY_DWORD:
    strcat(opname, "8");
    break;

  case TY_REAL:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
    ast_error("arithmetic result for logical operation", ast);
    return 0;
  case TY_CHAR:
  case TY_NCHAR:
    ast_error("character result for logical operation", ast);
    return 0;
  default:
    ast_error("unknown result for logical operation", ast);
    return 0;
  }
  lilm = lower_conv(A_LOPG(ast), dtype);
  rilm = lower_conv(A_ROPG(ast), dtype);
  lilm = add_lnop(lilm, A_LOPG(ast), dtype);
  rilm = add_lnop(rilm, A_ROPG(ast), dtype);
  ilm = plower("oii", opname, lilm, rilm);
  return ilm;
} /* lower_bin_logical */

/* generate the ILM for a simple logical unary operator.
 * the suffix for the operator name depends on the expression type */
static int
lower_un_logical(int ast, const char *op)
{
  int dtype, ilm, lilm;
  char opname[15];
  dtype = A_NDTYPEG(ast);
  if (dtype <= 0) {
    ast_error("unrecognized data type in lower_un_logical", ast);
    return 0;
  }
  strcpy(opname, op);
  switch (DTYG(dtype)) {
  case TY_SLOG:
  case TY_BLOG:
  case TY_LOG:
    break;
  case TY_LOG8:
    strcat(opname, "8");
    break;

  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_WORD:
    break;
  case TY_INT8:
  case TY_DWORD:
    strcat(opname, "8");
    break;
  case TY_REAL:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
    ast_error("arithmetic result for logical operation", ast);
    return 0;
  case TY_CHAR:
  case TY_NCHAR:
    ast_error("character result for logical operation", ast);
    return 0;
  default:
    ast_error("unknown result for logical operation", ast);
    return 0;
  }
  lilm = lower_conv(A_LOPG(ast), dtype);
  lilm = add_lnop(lilm, A_LOPG(ast), dtype);
  ilm = plower("oi", opname, lilm);
  return ilm;
} /* lower_un_logical */

int
lower_parenthesize_expression(int ast)
{
  int adtype;
  adtype = A_NDTYPEG(ast);
  if (ast == astb.ptr0 || ast == astb.ptr1 || ast == astb.ptr0c)
    return A_ILMG(ast);
  if (A_TYPEG(ast) == A_PAREN && (DT_ISNUMERIC(adtype) || DT_ISLOG(adtype))) {
    int a = A_LOPG(ast);
    if (A_TYPEG(a) == A_ID || A_TYPEG(a) == A_CNST) {
      int temp, lilm, rilm, ilm;
      temp = lower_scalar_temp(adtype);
      lilm = plower("oS", "BASE", temp);
      rilm = A_ILMG(ast);
      lower_typestore(adtype, lilm, rilm);
      ilm = plower("oS", "BASE", temp);
      return ilm;
    }
  }
  return A_ILMG(ast);
} /* parenthesize_expression */

/* Return true for RTE functions that permit null pointers as args.
 * Don't insert null pointer check, even if -Mchkptr is set.
 */
static bool
function_null_allowed(SPTR sptr)
{
  static FtnRtlEnum rtl_functions_null_allowed[] = {
      RTE_associated,
      RTE_associated_chara,
      RTE_associated_t,
      RTE_associated_tchara,
      RTE_conformable_11v,
      RTE_conformable_1dv,
      RTE_conformable_22v,
      RTE_conformable_2dv,
      RTE_conformable_33v,
      RTE_conformable_3dv,
      RTE_conformable_d1v,
      RTE_conformable_d2v,
      RTE_conformable_d3v,
      RTE_conformable_dd,
      RTE_conformable_dnv,
      RTE_conformable_ndv,
      RTE_conformable_nnv,
      RTE_extends_type_of,
      RTE_lena,
      RTE_lentrima,
      RTE_same_type_as,
      RTE_no_rtn /* marks end of list */
  };
  int i;
  for (i = 0;; i += 1) {
    FtnRtlEnum rtn = rtl_functions_null_allowed[i];
    if (rtn == RTE_no_rtn)
      return false;
    if (strcmp(SYMNAME(sptr), mkRteRtnNm(rtn)) == 0)
      return true;
  }
}

int get_byval(int, int);

static int
lower_function(int ast)
{
  int count, realcount, args, symfunc, dtype, i, ilm, ilm2;
  int dtproc, iface = 0, sptr, prevsptr, paramc;
  int callee;
  int functmp, functmpilm, functmpinc, funcusetmp, funcusecall;
  int paramcount, params, save_disable_ptr_chk;
  static int functmpcount;
  int is_procsym = 0;
  const char *UCALL;
  const char *PUFUNC;
  const char *UFUNC;
  int is_tbp, tbp_nopass_arg, tbp_nopass_sdsc, tbp_mem;
  int tbp_bind, tbp_inv;
  int unlpoly; /* CLASS(*) */
  int retdesc;
  int bindC_structret = 0;
  bool procDummyNeedsDesc;

  /* symfunc <- A_SPTRG(A_LOPG(ast )) */
  symfunc = procsym_of_ast(A_LOPG(ast));
  if (STYPEG(symfunc) == ST_MEMBER && CLASSG(symfunc) && CCSYMG(symfunc) &&
      VTABLEG(symfunc)) {
    symfunc = (IFACEG(symfunc)) ? IFACEG(symfunc) : VTABLEG(symfunc);
  }

  procDummyNeedsDesc = proc_arg_needs_proc_desc(symfunc);

  switch (A_TYPEG(A_LOPG(ast))) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    tbp_mem = memsym_of_ast(A_LOPG(ast));
    if (STYPEG(tbp_mem) == ST_PROC && CLASSG(tbp_mem) && IS_TBP(tbp_mem)) {
      i = 0;
      get_implementation(TBPLNKG(tbp_mem), tbp_mem, 0, &i);
      if (STYPEG(BINDG(i)) == ST_OPERATOR ||
          STYPEG(BINDG(i)) == ST_USERGENERIC) {
        i = get_specific_member(TBPLNKG(tbp_mem), VTABLEG(i));
      }

      tbp_mem = i;
    }

    break;
  default:
    tbp_mem = 0;
  }
  tbp_nopass_arg = is_tbp = tbp_nopass_sdsc = tbp_bind = tbp_inv = 0;
  if (tbp_mem && CLASSG(tbp_mem) && CCSYMG(tbp_mem) &&
      STYPEG(tbp_mem) == ST_MEMBER) {
    tbp_bind = BINDG(tbp_mem);
    is_procsym = 1;
    is_tbp = 1;
    UCALL = "UVCALLA";
    PUFUNC = "PUVFUNCA";
    UFUNC = "UVFUNCA";
#if DEBUG
    assert(!tbp_bind || STYPEG(tbp_bind) == ST_PROC,
           "lower_function: invalid stype for type bound procedure",
           STYPEG(tbp_bind), 4);
#endif
    if (!INVOBJG(tbp_bind) && !NOPASSG(tbp_mem)) {
      /* Try to resolve INVOBJ. INVOBJ may be 0 here due to a
       * late attempt to resolve a generic routine/operator (e.g.,
       * a call to queue_tbp(0,0,0,0,TBP_COMPLETE_GENERIC) in
       * is_intrinsic_opr() of semgnr.c).
       * When we call queue_tbp(0,0,0,0,TBP_COMPLETE_GENERIC), we might
       * generate one or more tbp symbols with the same name. This can occur
       * if a tbp symbol and/or implementation is used in different
       * contexts. Therefore, tbp_bind and its INVOBJ field may not get
       * fully resolved until later.
       */
      get_tbp_argno(tbp_bind, ENCLDTYPEG(tbp_mem));
    }
#if DEBUG
    assert(!INVOBJG(tbp_bind) != !NOPASSG(tbp_mem),
           "lower_function: either invobj or nopass must be set; not "
           "none or both",
           symfunc, 4);
#endif
    if (NOPASSG(tbp_mem)) {
      tbp_nopass_arg = pass_sym_of_ast(A_LOPG(ast));
      tbp_nopass_sdsc =
          A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast)) : 0;
      if (!tbp_nopass_sdsc)
        tbp_nopass_sdsc = get_type_descr_arg(gbl.currsub, tbp_nopass_arg);
      lower_expression(A_LOPG(ast));
      tbp_nopass_arg = lower_base(A_LOPG(ast));
    } else {
      tbp_inv = find_dummy_position(symfunc, PASSG(tbp_mem));
      if (tbp_inv == 0)
        tbp_inv = max_binding_invobj(symfunc, INVOBJG(tbp_bind));
    }
  } else if (!is_procedure_ptr(symfunc) && !procDummyNeedsDesc) {
    is_procsym = 1;
    UCALL = "UCALL";
    PUFUNC = "PUFUNC";
    UFUNC = "UFUNC";
  } else if (procDummyNeedsDesc || is_procedure_ptr(symfunc)) {
    is_procsym = STYPEG(symfunc) == ST_PROC;
    UCALL = "UPCALLA";
    PUFUNC = "PUFUNC";
    UFUNC = "PUFUNCA";
  } else {
    is_procsym = 0;
    UCALL = "UCALLA";
    PUFUNC = "PUFUNCA";
    UFUNC = "UFUNCA";
  }
  count = A_ARGCNTG(ast);
  NEED(count, lower_argument, int, lower_argument_size, count + 10);
  args = A_ARGSG(ast);
  save_disable_ptr_chk = lower_disable_ptr_chk;
  if (is_procsym) {
    if (function_null_allowed(symfunc)) {
      lower_disable_ptr_chk = 1;
    }

    callee = (procDummyNeedsDesc || is_procedure_ptr(symfunc))
                 ? lower_base(A_LOPG(ast))
                 : symfunc;
    paramcount = PARAMCTG(symfunc);
    params = DPDSCG(symfunc);
    /* get result datatype from function name */
    if (is_tbp != 1)
      dtype = A_NDTYPEG(A_LOPG(ast));
    else
      dtype = DTYPEG(callee);
  } else {
    dtype = DTYPEG(symfunc);
#if DEBUG
    assert(DTY(dtype) == TY_PTR, "lower_ptrfunction, expected TY_PTR dtype",
           symfunc, 4);
#endif
    dtproc = DTY(dtype + 1);
#if DEBUG
    assert(DTY(dtproc) == TY_PROC, "lower_ptrfunction, expected TY_PROC dtype",
           symfunc, 4);
#endif
    if (DTY(dtproc + 2) > NOSYM) {
      /* The procedure pointer has an interface.  Get the function result
       * type from that interface, since the result type in the procedure
       * pointer's DTYPE record can be wrong and I don't know how to fix them.
       */
      dtype = DTYPEG(DTY(dtproc + 2));
    } else {
      dtype = DTY(dtproc + 1); /* result type */
    }
    lower_expression(A_LOPG(ast));
    callee = lower_base(A_LOPG(ast));
    iface = DTY(dtproc + 2);
    paramcount = DTY(dtproc + 3);
    params = DTY(dtproc + 4);
  }
  A_NDTYPEP(ast, dtype);
  functmp = 0;
  functmpinc = 0;
  funcusetmp = 0;
  funcusecall = 0;
  switch (DTYG(dtype)) {
  case TY_CMPLX:
  case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
    functmpinc = 1; /* count the function temp as an extra argument */
    ++functmpcount;
    functmp = lower_scalar_temp(dtype);
    break;
  case TY_CHAR:
  case TY_NCHAR:
    ++functmpcount;
    functmp = lower_scalar_temp(dtype);
    funcusetmp = 1;
    break;
  case TY_DERIVED:
  case TY_STRUCT:
    if (CFUNCG(symfunc)) {
      retdesc = check_return(DTYPEG(FVALG(symfunc)));
      if (retdesc != CLASS_MEM && retdesc != CLASS_PTR) {
        bindC_structret = 1;
      } else {
        funcusecall = 1;
      }
    } else {
      funcusecall = 1;
    }
    if (CFUNCG(symfunc) || (iface && CFUNCG(iface))) {
      CSTRUCTRETP(symfunc, 1);
    }
    functmpinc = 1; /* count the function temp as an extra argument */
    ++functmpcount;
    functmp = lower_scalar_temp(dtype);
    ARGP(functmp, 1);
    funcusetmp = 1;
    break;
  default:
    break;
  }
  realcount = 0;
  for (i = 0; i < count; ++i) {
    int a;
    a = ARGT_ARG(args, i);
    if (a > 0) {
      if (A_TYPEG(a) != A_LABEL)
        lower_expression(a);
      switch (A_TYPEG(a)) {
      case A_ID:
      case A_MEM:
      case A_SUBSCR:
      case A_CNST:
        break;
      default:
        lower_ilm(a);
      }
    }
  }
  paramc = 0;
  sptr = 0;
  for (i = 0; i < count; ++i) {
    int a, param, byval;
    prevsptr = sptr;
    sptr = 0;
    a = ARGT_ARG(args, i);
    lower_argument[i] = 0;
    param = 0;
    if (paramc < paramcount) {
      param = aux.dpdsc_base[params + paramc];
      ++paramc;
      if (symfunc == gbl.currsub) {
        /* argument list was rewritten; use original argument */
        int nparam = NEWARGG(param);
        if (nparam)
          param = nparam;
      }
    }
    if (a == 0)
      continue;
    byval = 0;
    ++realcount;
    if (byval) {
      switch (A_TYPEG(a)) {
        int dt;

      case A_ID:
        /* for nonscalar identifiers, just pass by reference */
        sptr = A_SPTRG(a);

        switch (STYPEG(sptr)) {
        case ST_VAR:
        case ST_IDENT:
          if (param && POINTERG(param) && POINTERG(sptr))
            goto by_reference;
          break;
        default:
          goto by_reference;
        }
        goto by_value;
      case A_UNOP:
        if (A_OPTYPEG(a) == OP_BYVAL)
          goto by_reference;
        goto by_value;
      case A_MEM:
        /* if the formal is a pointer, pass the pointer address,
         * otherwise pass the data base address */
        sptr = A_SPTRG(A_MEMG(a));
        if (param && POINTERG(param) && POINTERG(sptr))
          goto by_reference;
        FLANG_FALLTHROUGH;
      case A_INTR:
        if (is_iso_cloc(a)) {
          /* byval C_LOC(x) == regular pass by reference (X),
             no type checking
           */
          a = ARGT_ARG(A_ARGSG(a), 0);
          goto by_reference;
        }
        FLANG_FALLTHROUGH;
      default:
      /* expressions & scalar variables -- always emit BYVAL.
       * expand will take do the right thing for nonscalar
       * expressions.
       */
      by_value:
        dt = A_DTYPEG(a);
        ilm = lower_ilm(a);
        if (DTY(dt) == TY_CHAR || DTY(dt) == TY_NCHAR) {
          if (DTY(dt) == TY_CHAR)
            ilm = plower("oi", "ICHAR", ilm);
          else
            ilm = plower("oi", "INCHAR", ilm);
          if (DTY(stb.user.dt_int) == TY_INT8)
            ilm = plower("oi", "ITOI8", ilm);
          dt = stb.user.dt_int;
        }
        lower_argument[i] = plower("oid", "BYVAL", ilm, dt);
        break;
      }
      continue;
    }
  by_reference:
    unlpoly = 0;
    if (param && is_unl_poly(param)) {
      unlpoly = 1;
    }
    switch (A_TYPEG(a)) {
    case A_ID:
      /* if the formal is a pointer, pass the pointer address,
       * otherwise pass the data base address */
      sptr = A_SPTRG(a);
      if (param && ((POINTERG(param) && POINTERG(sptr)) ||
                    (ALLOCATTRG(param) && ALLOCATTRG(sptr)))) {
        lower_disable_ptr_chk = 1;
        if (DTY(DTYPEG(sptr)) == TY_ARRAY && !XBIT(57, 0x80000)) {
          lower_argument[i] = lower_base(a);
        } else {
          ilm = lower_target(a);
          ilm2 = plower("oS", "BASE", sptr);
          lower_argument[i] = plower("oii", "PARG", ilm, ilm2);
        }
        lower_disable_ptr_chk = 0;
      } else {
        lower_argument[i] = lower_base(a);
      }
      switch (STYPEG(sptr)) {
      case ST_PROC:
      case ST_ENTRY:
      case ST_MODPROC:
        break;
      default:
        if (DTYPEG(sptr)) {
          lower_argument[i] =
              plower_arg("oid", lower_argument[i], DTYPEG(sptr), unlpoly);
        }
      }
      break;
    case A_MEM:
      /* if the formal is a pointer, pass the pointer address,
       * otherwise pass the data base address */
      sptr = A_SPTRG(A_MEMG(a));
      if (param && ((POINTERG(param) && POINTERG(sptr)) ||
                    (ALLOCATTRG(param) && ALLOCATTRG(sptr)))) {
        lower_disable_ptr_chk = 1;
        if (DTY(DTYPEG(sptr)) == TY_ARRAY && !XBIT(57, 0x80000)) {
          lower_argument[i] = lower_base(a);
        } else {
          ilm = lower_target(a);
          ilm2 = plower("oS", "BASE", sptr);
          lower_argument[i] = plower("oii", "PARG", ilm, ilm2);
        }
        lower_disable_ptr_chk = 0;
      } else {
        lower_argument[i] = lower_base(a);
      }
      lower_argument[i] =
          plower_arg("oid", lower_argument[i], DTYPEG(sptr), unlpoly);
      break;
    case A_SUBSCR:
    case A_CNST:
      lower_argument[i] = lower_base(a);
      if (A_DTYPEG(a)) {
        lower_argument[i] =
            plower_arg("oid", lower_argument[i], A_DTYPEG(a), unlpoly);
      }
      break;
    default:
      lower_argument[i] = lower_parenthesize_expression(a);
      if (A_DTYPEG(a)) {
        lower_argument[i] =
            plower_arg("oid", lower_argument[i], A_DTYPEG(a), unlpoly);
      }
      break;
    }
  }
  if (functmp) {
    functmpilm = plower("oS", "BASE", functmp);
    functmpilm = plower_arg("oid", functmpilm, DTYPEG(functmp), 0);
  }
  if (funcusecall) {
    ilm = plower("om", UCALL);
  } else {
    if (bindC_structret) {
      int retdesc = check_return(DTYPEG(FVALG(symfunc)));
      if (retdesc != CLASS_MEM && retdesc != CLASS_PTR) {
        ilm = plower("om", "SFUNC");
      }
    } else {
      if (procDummyNeedsDesc || is_procedure_ptr(symfunc)) {
        char *l;
        char op[100] = {'P', '\0'};
        int dtype2 = DTY(dtype + 1);
        if (DTY(dtype2) == TY_PROC) {
          if (DTY(dtype2 + 2)) {
            dtype2 = DTYPEG(DTY(dtype2 + 2));
            if (DTY(dtype2) == TY_ARRAY)
              dtype2 = DTY(dtype2 + 1);
          } else {
            dtype2 = DTY(dtype2 + 1);
          }
          l = ltyped(UFUNC + 1, dtype2);
        } else {
          l = ltyped(UFUNC + 1, dtype);
        }
        strcat(op, l);
        ilm = plower("om", op);
      } else {
        ilm = plower("om", ltyped(UFUNC, dtype));
      }
    }
  }

  if (is_tbp) {
    int is_cfunc = (CFUNCG(symfunc) || (iface && CFUNCG(iface)));
    VTABLEP(tbp_mem, symfunc);
    plower("nnsm", realcount + functmpinc, is_cfunc, tbp_mem);
  } else if (procDummyNeedsDesc || is_procedure_ptr(symfunc)) {
    int sdsc = A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast))
                                     : SDSCG(memsym_of_ast(ast));
    int is_cfunc = (CFUNCG(symfunc) || (iface && CFUNCG(iface)));
    plower("nnsim", realcount + functmpinc, is_cfunc, sdsc, callee);
  } else if (is_procsym) {
    plower("nsm", realcount + functmpinc, callee);
  } else {
    int is_cfunc = (CFUNCG(symfunc) || (iface && CFUNCG(iface)));
    plower("nnim", realcount + functmpinc, is_cfunc, callee);
  }

  if (is_tbp) {
    if (tbp_nopass_arg) {
      plower("im", tbp_nopass_arg);
      plower("sm", tbp_nopass_sdsc);
    } else {
      int a, sdsc, a_sptr, a_dtype;
      i = tbp_inv - 1;
      plower("im", lower_argument[i]);
      a = ARGT_ARG(args, i);
      a_sptr = memsym_of_ast(a);
      a_dtype = DTYPEG(a_sptr);
      if (DTY(a_dtype) == TY_ARRAY)
        a_dtype = DTY(a_dtype + 1);
      sdsc = A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast)) : 0;
      if (!sdsc) {
        if (!CLASSG(a_sptr) && DTY(a_dtype) == TY_DERIVED) {
          sdsc = get_static_type_descriptor(DTY(a_dtype + 3));
        } else {
          sdsc = get_type_descr_arg(gbl.currsub, a_sptr);
        }
      }
      plower("sm", sdsc);
    }
  }

  if (functmp) {
    plower("am", functmpilm, dtype);
  }

  for (i = 0; i < count; ++i) {
    int a;
    a = ARGT_ARG(args, i);
    if (a > 0) {
      plower("am", lower_argument[i], A_NDTYPEG(a));
    }
  }
  plower("C", symfunc);
  if (funcusetmp && !bindC_structret) {
    /* don't use the function return value, use the temp */
    ilm = plower("oS", "BASE", functmp);
  }
  A_ILMP(ast, ilm);
  lower_disable_ptr_chk = save_disable_ptr_chk;
  return ilm;
} /* lower_function */

/* options argument to intrin_name: */
/* I_K_r_D_C_CD means           int, int8, real, real8, cmplx, cmpl16 char */
/*                       prefix:  I    K   none    D       C     CD        */
#define in_I_K_r_D_C_CD 0x0331333
/*                       prefix:  I    K    R      D       C     CD        */
#define in_I_K_R_D_C_CD 0x0333333
/*                       prefix:  I    K    R      D       C     CD        */
#define in_Il_K_R_D_C_CD 0x0b33333
/*                       prefix:  I    K    R      D                       */
/*                       prefix:  log                                      */
#define in_I_K_R_D 0x0333300
/*                       prefix:  I    K   none    D                       */
#define in_Il_K_R_D 0x0b33300
/*                       prefix:  I    K   none    D                       */
/*                       prefix:  log                                      */
#define in_I_K_r_D 0x0331300
/*                       prefix:  I    K    R      D       Q               */
#define in_I_K_r_D_Q 0x03313c0
/*                       prefix:  I    K    R      D                       */
#define in_i_K_A_D 0x0135300
/*                       prefix:  I    K    R      D                       */
#define in_i_K_A_D_Q 0x01353c0
/*                       prefix:  none K                                   */
#define in_i_K 0x0130000
/*                       prefix:  log  K                                   */
#define in_il_K 0x0930000
/*                       prefix:  I    K                                   */
#define in_I_K 0x0330000
/*                       prefix:  none 64                                  */
#define in_i_64 0x0150000
/*                       prefix:  none none                                */
#define in_i_k 0x0110000
/*                       prefix:  none                                     */
#define in_i 0x0100000
/*                       prefix:  J    K                                   */
#define in_J_K 0x0530000
/*                       prefix:  none      A      D                       */
#define in_R_D 0x0003300
/*                       prefix:            R      D        Q              */
#define in_R_D_Q 0x00033c0
/*                       prefix:            R      D                       */
#define in_r_D 0x0001300
/*                       prefix:            R      D        Q              */
#define in_r_D_Q 0x00013c0
/*                       prefix:            R      D       C     CD        */
#define in_R_D_C_CD 0x0001333
/*                       prefix:           none    D       C     CD        */
#define in_r_D_C_CD 0x0001333
/*                       prefix:            R      D       C     CD   CQ   */
#define in_R_D_Q_C_CD_CQ 0xc0033f3
/*                       prefix:           none    D       C     CD   CQ   */
#define in_r_D_Q_C_CD_CQ 0xc0013f3
/*                       prefix:                        none      D        */
#define in_c_cD 0x0000015
/*                       prefix: none       A      D                       */
#define in_A_D 0x0005300
/*                       prefix: none       A      D              Q        */
#define in_A_D_Q 0x00053c0
/*                       prefix:                   D                       */
#define in_d 0x0000100
/*                       prefix:                                           */
#define in_c 0x1000000
#define in_nc 0x2000000
#define in_c_nc 0x3000000
#define in_c_cD_cQ 0x400001d

#define IARGS 100
static int intr_argbf[IARGS];
static int *intrinsic_args = intr_argbf;
static int intr_argsz = IARGS;

static int *
need_intr_argbf(int nargs)
{
  if (nargs > intr_argsz) {
    if (intr_argsz == IARGS) {
      intr_argsz = nargs + IARGS;
      NEW(intrinsic_args, int, nargs);
    } else {
      NEED(nargs, intrinsic_args, int, intr_argsz, nargs + IARGS);
    }
  }
  return intrinsic_args;
}

static int
intrin_name(const char *name, int ast, int options)
{
#define allowI 0x0100000
#define prefixI 0x0200000
#define prefixJ 0x0400000
#define allowL 0x0800000
#define allowK 0x0010000
#define prefixK 0x0020000
#define suffix64 0x0040000
#define allowR 0x0001000
#define prefixR 0x0002000
#define prefixA 0x0004000
#define allowD 0x0000100
#define prefixD 0x0000200
#define allowQ 0x0000040
#define prefixQ 0x0000080
#define allowC 0x0000010
#define prefixC 0x0000020
#define allowCD 0x0000001
#define prefixCD 0x0000002
#define prefixcD 0x0000004
#define allowchar 0x1000000
#define allownchar 0x2000000
#define allowCQ 0x4000000
#define prefixCQ 0x8000000
#define prefixcQ 0x0000008

  int dtype, ok, ilm;
  const char *prefix;
  const char *suffix;
  char intrname[50];
  dtype = A_NDTYPEG(ast);
  prefix = "";
  suffix = "";
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
  case TY_WORD:
    ok = options & allowI;
    if (options & prefixI) {
      prefix = "I";
    } else if (options & prefixJ) {
      prefix = "J";
    }
    break;
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
    ok = options & allowL;
    if (options & prefixI) {
      prefix = "I";
    } else if (options & prefixJ) {
      prefix = "J";
    }
    break;
  case TY_DWORD:
  case TY_INT8:
  case TY_LOG8:
    ok = options & allowK;
    if (options & prefixK) {
      prefix = "K";
    } else if (options & suffix64) {
      suffix = "64";
    }
    break;
  case TY_REAL:
    ok = options & allowR;
    if (options & prefixR) {
      prefix = "R";
    } else if (options & prefixA) {
      prefix = "A";
    }
    break;
  case TY_DBLE:
    ok = options & allowD;
    if (options & prefixD) {
      prefix = "D";
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ok = options & allowQ;
    if (options & prefixQ) {
      prefix = "Q";
    }
    break;
#endif
  case TY_CMPLX:
    ok = options & allowC;
    if (options & prefixC) {
      prefix = "C";
    }
    break;
  case TY_DCMPLX:
    ok = options & allowCD;
    if (options & prefixCD) {
      prefix = "CD";
    } else if (options & prefixcD) {
      prefix = "D";
    }
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ok = options & allowCQ;
    if (options & prefixCQ) {
      prefix = "CQ";
    } else if (options & prefixcQ) {
      prefix = "Q";
    }
    break;
#endif
  case TY_CHAR:
    ok = options & allowchar;
    break;
  case TY_NCHAR:
    ok = options & allownchar;
    break;
  default:
    ast_error("unexpected type for intrinsic function", ast);
    ok = 1;
    break;
  }
  if (!ok) {
    ast_error("unexpected result type for intrinsic function", ast);
  }
  strcpy(intrname, prefix);
  strcat(intrname, name);
  strcat(intrname, suffix);
  ilm = plower("om", intrname);
  return ilm;
} /* intrin_name */

static int
intrin_name_bsik(const char *name, int ast)
{
  int dtype, ilm;
  const char *prefix;
  char intrname[50];
  dtype = A_NDTYPEG(ast);
  prefix = "";
  switch (DTYG(dtype)) {
  case TY_BINT:
  case TY_BLOG:
    prefix = "B";
    break;
  case TY_SINT:
  case TY_SLOG:
    prefix = "S";
    break;
  case TY_INT:
  case TY_WORD:
  case TY_LOG:
    prefix = "I";
    break;
  case TY_DWORD:
  case TY_INT8:
  case TY_LOG8:
    prefix = "K";
    break;
  default:
    ast_error("unexpected type for intrinsic function", ast);
    prefix = "I";
    break;
  }
  strcpy(intrname, prefix);
  strcat(intrname, name);
  ilm = plower("om", intrname);
  return ilm;
}

/* return the 'REAL' type nearest in length to dtype */
static int
nearest_real_type(int dtype)
{
  switch (DTY(dtype)) {
  case TY_DWORD:
#ifndef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
#endif
  case TY_INT8:
  case TY_DBLE:
  case TY_DCMPLX:
#ifndef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
#endif
  case TY_LOG8:
    return DT_DBLE;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
  case TY_QCMPLX:
    return DT_QUAD;
#endif
  default:
    return DT_REAL;
  }
} /* nearest_real_type */

/*
 * return TRUE if this is a function which can have a NULL pointer argument
 * if so, we don't insert a null pointer check, even if -Mchkptr is set
 */
static int
intrinsic_null_allowed(int intr)
{
  switch (intr) {
  case I_ALLOCATED:
  case I_ASSOCIATED:
  case I_PRESENT:
  case I_LEN:
  case I_IS_CONTIGUOUS:
  case I_C_ASSOCIATED:
    return TRUE;
  default:
    return FALSE;
  }
} /* intrinsic_null_allowed */

static int
intrinsic_arg_dtype(int intr, int ast, int args, int nargs)
{
  switch (intr) {
  /* the first set of intrinsics do no type conversion;
   * they appear in the order they are listed in symini_ftn.n for the
   * f90 back end. */
  case I_SQRT:
  case I_DSQRT:
  case I_CSQRT:
  case I_CDSQRT:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSQRT:
#endif

  case I_LOG:
  case I_ALOG:
  case I_DLOG:
  case I_CLOG:
  case I_CDLOG:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QLOG:
#endif

  case I_LOG10:
  case I_ALOG10:
  case I_DLOG10:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QLOG10:
#endif

  case I_EXP:
  case I_DEXP:
  case I_CEXP:
  case I_CDEXP:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QEXP:
#endif

  case I_SIN:
  case I_DSIN:
  case I_CSIN:
  case I_CDSIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_CQSIN:
  case I_QSIN:
#endif

  case I_SIND:
  case I_DSIND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSIND:
#endif

  case I_COS:
  case I_DCOS:
  case I_CCOS:
  case I_CDCOS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCOS:
#endif

  case I_COSD:
  case I_DCOSD:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCOSD:
#endif

  case I_TAN:
  case I_DTAN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QTAN:
#endif

  case I_TAND:
  case I_DTAND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QTAND:
#endif

  case I_ASIN:
  case I_DASIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QASIN:
#endif

  case I_ASIND:
  case I_DASIND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QASIND:
#endif

  case I_ACOS:
  case I_DACOS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QACOS:
#endif

  case I_ACOSD:
  case I_DACOSD:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QACOSD:
#endif

  case I_ATAN:
  case I_DATAN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN:
#endif

  case I_ATAND:
  case I_DATAND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAND:
#endif

  case I_ATAN2:
  case I_DATAN2:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN2:
#endif

  case I_ATAN2D:
  case I_DATAN2D:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN2D:
#endif

  case I_SINH:
  case I_DSINH:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSINH:
#endif

  case I_COSH:
  case I_DCOSH:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCOSH:
#endif

  case I_TANH:
  case I_DTANH:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QTANH:
#endif

  case I_ERF:
  case I_ERFC:
  case I_ERFC_SCALED:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QERF:
  case I_QERFC:
  case I_QERFC_SCALED:
#endif
  case I_GAMMA:
  case I_LOG_GAMMA:
  case I_HYPOT:
  case I_ACOSH:
  case I_ASINH:
  case I_ATANH:
  case I_BESSEL_J0:
  case I_BESSEL_J1:
  case I_BESSEL_Y0:
  case I_BESSEL_Y1:

  case I_IABS:
  case I_IIABS:
  case I_JIABS:
  case I_KIABS:

  case I_AINT:
  case I_DINT:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QINT:
#endif

  case I_ANINT:
  case I_DNINT:

  case I_CEILING:
  case I_FLOOR:

  case I_CONJG:
  case I_DCONJG:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCONJG:
#endif

  case I_IIDIM:
  case I_JIDIM:
  case I_KIDIM:
  case I_IDIM:
  case I_DIM:
  case I_DDIM:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QDIM:
#endif

  case I_IMOD:
  case I_JMOD:
  case I_KMOD:
  case I_MOD:
  case I_AMOD:
  case I_DMOD:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMOD:
#endif

  case I_IISIGN:
  case I_JISIGN:
  case I_KISIGN:
  case I_ISIGN:
  case I_SIGN:
  case I_DSIGN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSIGN:
#endif

  case I_IIAND:
  case I_JIAND:
  case I_IIOR:
  case I_JIOR:
  case I_IIEOR:
  case I_JIEOR:
  case I_INOT:
  case I_JNOT:
  case I_IISHFT:
  case I_JISHFT:
  case I_KISHFT:

  case I_IBITS:
  case I_IIBITS:
  case I_JIBITS:
  case I_KIBITS:
  case I_IBSET:
  case I_IIBSET:
  case I_JIBSET:
  case I_KIBSET:
  case I_BTEST:
  case I_BITEST:
  case I_BJTEST:
  case I_BKTEST:
  case I_IBCLR:
  case I_IIBCLR:
  case I_JIBCLR:
  case I_KIBCLR:
  case I_ISHFTC:
  case I_IISHFTC:
  case I_JISHFTC:
  case I_KISHFTC:
  case I_LSHIFT:
  case I_RSHIFT:

  case I_IAND:
  case I_IOR:
  case I_IEOR:
  case I_XOR:
  case I_NOT:
  case I_ISHFT:
  case I_MAX:
  case I_MIN:

  case I_AND:
  case I_OR:
  case I_EQV:
  case I_NEQV:
  case I_COMPL:

  case I_LEADZ:
  case I_TRAILZ:
  case I_POPCNT:
  case I_POPPAR:
    return A_NDTYPEG(ast);

  case I_ABS:
  case I_DABS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QABS:
#endif
  case I_CABS:
  case I_CDABS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_CQABS:
#endif
  case I_BESSEL_JN:
  case I_BESSEL_YN:
    /* don't coerce */
    return -1;

  /* MAX, MIN */
  case I_MAX1:
  case I_MIN1:
  case I_IMAX1:
  case I_KMAX1:
  case I_IMIN1:
  case I_KMIN1:
  case I_JMAX1:
  case I_JMIN1:
  case I_AMAX1: /* r*4,r*4 -> r*4 */
  case I_AMIN1:
    return DT_REAL4;
  case I_DMAX1:
  case I_DMIN1:
    return DT_REAL8;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMAX:
  case I_QMIN:
    return DT_QUAD;
#endif

  case I_MAX0: /* i*4,i*4 -> i*4 */
  case I_MIN0:
  case I_JMAX0: /* i*4,i*4 -> i*4 */
  case I_JMIN0:
  case I_AMAX0:
  case I_AMIN0:
  case I_AJMAX0:
  case I_AJMIN0:
    return DT_INT4;
  case I_IMAX0: /* i*2,i*2 -> i*2 */
  case I_IMIN0:
  case I_AIMAX0:
  case I_AIMIN0:
    return DT_SINT;
  case I_KMAX0:
  case I_KMIN0:
    return DT_INT8;

  /* type conversion to integer */
  case I_IFIX:
  case I_IIFIX:
  case I_JIFIX:
  case I_IDINT:
  case I_IIDINT:
  case I_JIDINT:
  case I_IINT:
  case I_INT:
  case I_JINT:
  case I_INT1:
  case I_INT2:
  case I_INT4:
  case I_INT8:
    return -1;

  /* conversion real to nearest integer */
  case I_ININT:
  case I_JNINT:
  case I_KNINT:
    return DT_REAL4;

  /* conversion double to nearest integer */
  case I_IDNINT:
  case I_IIDNNT:
  case I_JIDNNT:
  case I_KIDNNT:
    return DT_REAL8;

  /* generic, conversion to nearest integer */
  case I_NINT:
    return nearest_real_type(A_NDTYPEG(ARGT_ARG(args, 0)));

  /* type conversion to real */
  case I_FLOATI:
  case I_FLOATJ:
  case I_FLOAT:
  case I_SNGL:
  case I_REAL:
    return -1;

  /* type conversion to double */
  case I_DFLOTI:
  case I_DFLOAT:
  case I_DFLOTJ:
  case I_DREAL:
  case I_DBLE:
    return -1;

#ifdef TARGET_SUPPORTS_QUADFP
  case I_QIMAG:
#endif
  case I_DIMAG:
  case I_AIMAG:
  case I_IMAG:
    /* return imaginary part */
#ifdef TARGET_SUPPORTS_QUADFP
    if (A_NDTYPEG(ast) == DT_QUAD)
      return DT_QCMPLX;
#endif
    if (A_NDTYPEG(ast) == DT_REAL8)
      return DT_CMPLX16;
    return DT_CMPLX8;

  /* double precision product of reals */
  case I_DPROD:
    return DT_REAL4;

  case I_CMPLX:
  case I_DCMPLX:
    return -1;

  /* ichar family */
  case I_ICHAR:
  case I_IACHAR:
    /* just get base address of argument */
    intrinsic_args[0] = lower_base(ARGT_ARG(args, 0));
    return -1;

  case I_LEN:
  case I_KLEN:
  case I_LEN_TRIM:
    intrinsic_args[0] = lower_base(ARGT_ARG(args, 0));
    return -1;

  case I_INDEX:
  case I_KINDEX:
    return -1;

  case I_LGE:
  case I_LGT:
  case I_LLE:
  case I_LLT:
    return -1;

  case I_LOC:
  case I_C_FUNLOC:
  case I_C_LOC:
    intrinsic_args[0] = lower_base(ARGT_ARG(args, 0));
    return -1;

  /* shift family */
  case I_SHIFT:
    intrinsic_args[0] = lower_conv(ARGT_ARG(args, 0), A_NDTYPEG(ast));
    intrinsic_args[1] = lower_conv(ARGT_ARG(args, 1), DT_INT4);
    return -1;

  /* type conversion to char */
  case I_CHAR:
    return DT_INT4;
  case I_ACHAR:
    return DT_INT4;
  case I_NCHAR:
    return DT_INT4;
  case I_NLEN:
    intrinsic_args[0] = lower_base(ARGT_ARG(args, 0));
    return -1;
  case I_NINDEX:
    return -1;

  case I_ALLOCATED:
  case I_ASSOCIATED:
  case I_PRESENT:
  case I_MERGE:
  case I_ILEN:
  case I_IS_CONTIGUOUS:
  case I_C_ASSOCIATED:
    return -1;

  case I_SIZE:
  case I_LBOUND:
  case I_UBOUND:
  case I_MODULO:
  case I_EXPONENT:
  case I_FRACTION:
  case I_RRSPACING:
  case I_SPACING:
  case I_NEAREST:
  case I_SCALE:
  case I_SET_EXPONENT:
  case I_VERIFY:
  case I_RAN:
  case I_ISNAN:
    return -1;

  case I_ZEXT:
  case I_IZEXT:
  case I_JZEXT:
    return DT_INT4;

  case NEW_INTRIN:
    return A_DTYPEG(ast);
    /*------------------*/

  case I_DATE:
  case I_EXIT:
  case I_IDATE:
  case I_TIME:
  case I_MVBITS:

  case I_SECNDS:
  case I_DATE_AND_TIME:
  case I_RANDOM_NUMBER:
  case I_RANDOM_SEED:
  case I_SYSTEM_CLOCK:
  case I_KIND:
  case I_SELECTED_INT_KIND:
  case I_SELECTED_REAL_KIND:
  case I_EPSILON:
  case I_HUGE:
  case I_TINY:
  case I_NULLIFY:
  case I_RANF:
  case I_RANGET:
  case I_RANSET:
  case I_INT_MULT_UPPER:

  case I_ALL:
  case I_ANY:
  case I_COUNT:
  case I_DOT_PRODUCT:
  case I_NORM2:
  case I_MATMUL:
  case I_MATMUL_TRANSPOSE:
  case I_MAXLOC:
  case I_MAXVAL:
  case I_MINLOC:
  case I_MINVAL:
  case I_FINDLOC:
  case I_PACK:
  case I_PRODUCT:
  case I_SUM:
  case I_SPREAD:
  case I_TRANSPOSE:
  case I_UNPACK:
  case I_NUMBER_OF_PROCESSORS:
  case I_CSHIFT:
  case I_EOSHIFT:
  case I_RESHAPE:
  case I_SHAPE:
  case I_ADJUSTL:
  case I_ADJUSTR:
  case I_BIT_SIZE:
  case I_DIGITS:
  case I_LOGICAL:
  case I_MAXEXPONENT:
  case I_MINEXPONENT:
  case I_PRECISION:
  case I_RADIX:
  case I_RANGE:
  case I_REPEAT:
  case I_TRANSFER:
  case I_TRIM:
  case I_SCAN:
  case I_DOTPRODUCT:
  case I_PROCESSORS_SHAPE:
  case I_LASTVAL:
  case I_REDUCE_SUM:
  case I_REDUCE_PRODUCT:
  case I_REDUCE_ANY:
  case I_REDUCE_ALL:
  case I_REDUCE_PARITY:
  case I_REDUCE_IANY:
  case I_REDUCE_IALL:
  case I_REDUCE_IPARITY:
  case I_REDUCE_MINVAL:
  case I_REDUCE_MAXVAL:
  case I_PTR2_ASSIGN:
  case I_PTR_COPYIN:
  case I_PTR_COPYOUT:
  case I_UNIT:
  case I_LENGTH:
  case I_COT:
  case I_DCOT:
  case I_SHIFTL:
  case I_SHIFTR:
  case I_DSHIFTL:
  case I_DSHIFTR:
  default:
    return -1;
  }
} /* intrinsic_arg_dtype */

static int
f90_function(const char *name, int dtype, int args, int nargs)
{
  int i, symfunc, ilm;
  need_intr_argbf(nargs);
  symfunc = lower_makefunc(name, dtype, FALSE);
  for (i = 0; i < nargs; ++i) {
    intrinsic_args[i] = lower_base(ARGT_ARG(args, i));
  }
  ilm = plower("onsm", ltyped("FUNC", dtype), nargs, symfunc);
  return ilm;
} /* f90_function */

static int
f90_value_function(const char *name, int dtype, int args, int nargs)
{
  int i, symfunc, ilm;
  need_intr_argbf(nargs);
  symfunc = lower_makefunc(name, dtype, FALSE);
  for (i = 0; i < nargs; ++i) {
    ilm = lower_ilm(ARGT_ARG(args, i));
    ilm = plower("oi", "DPVAL", ilm);
    intrinsic_args[i] = ilm;
  }
  ilm = plower("onsm", ltyped("FUNC", dtype), nargs, symfunc);
  return ilm;
} /* f90_value_function */

/* 2nd argument must be int */
static int
f90_value_function_I2(const char *name, int dtype, int args, int nargs)
{
  int i, symfunc, ilm;
  need_intr_argbf(nargs);
  symfunc = lower_makefunc(name, dtype, FALSE);
  for (i = 0; i < nargs; ++i) {
    int ast = ARGT_ARG(args, i);
    ilm = lower_ilm(ast);
    if (i == 1) {
      ilm = lower_conv_ilm(ast, ilm, A_NDTYPEG(ast), DT_INT);
    }
    ilm = plower("oi", "DPVAL", ilm);
    intrinsic_args[i] = ilm;
  }
  ilm = plower("onsm", ltyped("FUNC", dtype), nargs, symfunc);
  return ilm;
} /* f90_value_function_I2 */

static int
new_intrin_sym(int ast)
{
  int ast_spec = 0;
  int sptr = A_SPTRG(ast);

  switch (DTY(A_DTYPEG(ast))) {
  case TY_DCMPLX:
    ast_spec = GDCMPLXG(sptr);
    break;
  case TY_CMPLX:
    ast_spec = GCMPLXG(sptr);
    break;
  }
  return ast_spec;
}

static int
lower_intrinsic(int ast)
{
  int intr, ilm = 0, ilm1, ilm2, args, nargs, i, arg0, argdtype, dty, dtype,
      symfunc, input_ast;
  int shape, cnt, num, arg, arg1, arg2, fromdtype;
  int sptr;
  int pairwise = 0, argsdone = 0, save_disable_ptr_chk;
  const char *rtn_name;
  FtnRtlEnum rtlRtn = RTE_no_rtn;
  int retDtype;
  char *nm;

  if (is_iso_cloc(ast)) {
    /*
     * semant may type cloc() as the derived type, c_ptr
     */
    A_NDTYPEP(ast, DT_PTR);
  }
  nargs = A_ARGCNTG(ast);
  args = A_ARGSG(ast);
  intr = A_OPTYPEG(ast);

  if (intr != NEW_INTRIN) {
    symfunc = EXTSYMG(intast_sym[intr]);
  } else {
    symfunc = new_intrin_sym(A_LOPG(ast));
  }
  save_disable_ptr_chk = lower_disable_ptr_chk;
  if (intrinsic_null_allowed(intr)) {
    lower_disable_ptr_chk = 1;
  }
  need_intr_argbf(nargs);
  if (symfunc) {
    dtype = A_DTYPEG(ast);
    for (i = 0; i < nargs; ++i) {
      intrinsic_args[i] = lower_base(ARGT_ARG(args, i));
    }
    ilm = plower("onsm", ltyped("FUNC", dtype), nargs, symfunc);
    for (i = 0; i < nargs; ++i) {
      plower("im", intrinsic_args[i]);
    }
    plower("e");
    return ilm;
  }
  argdtype = intrinsic_arg_dtype(intr, ast, args, nargs);
  /* some intrinsics look only at one argument */
  switch (intr) {
  case I_IDNINT:
  case I_ININT:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QNINT:
#endif
  case I_JNINT:
  case I_KNINT:
  case I_NINT:
  case I_IIDNNT:
  case I_JIDNNT:
  case I_KIDNNT:
  case I_AINT:
  case I_DINT:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QINT:
#endif
  case I_ANINT:
  case I_DNINT:
  case I_FLOOR:
  case I_CEILING:
    nargs = 1;
  }
  if (argdtype >= 0) {
    for (i = 0; i < nargs; ++i) {
      intrinsic_args[i] = lower_conv(ARGT_ARG(args, i), argdtype);
    }
  }
  switch (intr) {
  /* abs family */
  case I_IABS:
  case I_IIABS:
  case I_JIABS:
  case I_KIABS:
    ilm = intrin_name("ABS", ast, in_I_K_r_D_C_CD);
    break;

  case I_ABS:
  case I_DABS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QABS:
#endif
  case I_CABS:
  case I_CDABS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_CQABS:
#endif
    /* use datatype of argument */
    arg1 = ARGT_ARG(args, 0);
    lower_expression(arg1);
    intrinsic_args[0] = lower_ilm(arg1);
#ifdef TARGET_SUPPORTS_QUADFP
    if (intr == I_QABS)
      ilm = intrin_name("ABS", arg1, in_r_D_Q);
    else if (intr == I_CQABS)
      ilm = intrin_name("ABS", arg1, in_r_D_Q_C_CD_CQ);
    else
#endif
      ilm = intrin_name("ABS", arg1, in_I_K_r_D_C_CD);
    break;

  /* acos family */
  case I_ACOS:
  case I_DACOS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QACOS:
#endif
    ilm = intrin_name("ACOS", ast, in_r_D_Q);
    break;
  case I_ACOSD:
  case I_DACOSD:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QACOSD:
#endif
    ilm = intrin_name("ACOSD", ast, in_r_D_Q);
    break;

  /* and family */
  case I_IIAND:
  case I_JIAND:
  case I_IAND:
    ilm = intrin_name("AND", ast, in_i_K);
    break;

  case I_AND:
    ilm = intrin_name("AND", ast, in_i_K);
    break;

  /* asin family */
  case I_ASIN:
  case I_DASIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QASIN:
#endif
    ilm = intrin_name("ASIN", ast, in_r_D_Q);
    break;
  case I_ASIND:
  case I_DASIND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QASIND:
#endif
    ilm = intrin_name("ASIND", ast, in_r_D_Q);
    break;

  /* atan family */
  case I_ATAN:
  case I_DATAN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN:
#endif
    ilm = intrin_name("ATAN", ast, in_r_D_Q);
    break;
  case I_ATAND:
  case I_DATAND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAND:
#endif
    ilm = intrin_name("ATAND", ast, in_r_D_Q);
    break;

  case I_ERF:
    ilm = intrin_name("ERF", ast, in_r_D_Q);
    break;
  case I_ERFC:
    ilm = intrin_name("ERFC", ast, in_r_D_Q);
    break;
  case I_ERFC_SCALED:
    ilm = intrin_name("ERFC_SCALED", ast, in_r_D_Q);
    break;
  case I_GAMMA:
    ilm = intrin_name("GAMMA", ast, in_r_D_Q);
    break;
  case I_LOG_GAMMA:
    ilm = intrin_name("LOG_GAMMA", ast, in_r_D_Q);
    break;
  case I_HYPOT:
    ilm = intrin_name("HYPOT", ast, in_r_D_Q);
    break;
  case I_ACOSH:
    ilm = intrin_name("ACOSH", ast, in_r_D_Q);
    break;
  case I_ASINH:
    ilm = intrin_name("ASINH", ast, in_r_D_Q);
    break;
  case I_ATANH:
    ilm = intrin_name("ATANH", ast, in_r_D_Q);
    break;
  case I_BESSEL_J0:
    ilm = intrin_name("BESSEL_J0", ast, in_r_D);
    break;
  case I_BESSEL_J1:
    ilm = intrin_name("BESSEL_J1", ast, in_r_D);
    break;
  case I_BESSEL_Y0:
    ilm = intrin_name("BESSEL_Y0", ast, in_r_D);
    break;
  case I_BESSEL_Y1:
    ilm = intrin_name("BESSEL_Y1", ast, in_r_D);
    break;
  case I_BESSEL_YN:
  case I_BESSEL_JN:
    args = A_ARGSG(ast);
    arg1 = ARGT_ARG(args, 0);
    lower_expression(arg1);
    intrinsic_args[0] = lower_ilm(arg1);
    arg1 = ARGT_ARG(args, 1);
    lower_expression(arg1);
    intrinsic_args[1] = lower_ilm(arg1);
    if (intr == I_BESSEL_YN)
      ilm = intrin_name("BESSEL_YN", ast, in_r_D);
    else
      ilm = intrin_name("BESSEL_JN", ast, in_r_D);
    break;

  /* atan2 family */
  case I_ATAN2:
  case I_DATAN2:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN2:
#endif
    ilm = intrin_name("ATAN2", ast, in_r_D_Q);
    break;
  case I_ATAN2D:
  case I_DATAN2D:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN2D:
#endif
    ilm = intrin_name("ATAN2D", ast, in_r_D_Q);
    break;

  /* char family */
  case I_CHAR:
    ilm = intrin_name("CHAR", ast, in_c);
    break;
  case I_ACHAR:
    ilm = intrin_name("CHAR", ast, in_c);
    break;
  case I_NCHAR:
    ilm = intrin_name("NCHAR", ast, in_nc);
    break;

  /* cmplx */
  case I_CMPLX:
  case I_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCMPLX:
#endif
    arg1 = ARGT_ARG(args, 0);
    arg2 = 0;
    if (nargs >= 2)
      arg2 = ARGT_ARG(args, 1);
    if (arg2 == 0) {
      switch (DTYG(A_NDTYPEG(ast))) {
      case TY_CMPLX:
        ilm = lower_conv(arg1, DT_CMPLX8);
        break;
      case TY_DCMPLX:
        ilm = lower_conv(arg1, DT_CMPLX16);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        ilm = lower_conv(arg1, DT_QCMPLX);
        break;
#endif
      default:
        break;
      }
      A_ILMP(ast, ilm);
      return ilm;
    } else {
      switch (DTYG(A_NDTYPEG(ast))) {
      case TY_CMPLX:
        ilm = lower_conv(arg1, DT_REAL4);
        ilm2 = lower_conv(arg2, DT_REAL4);
        ilm = plower("oii", "CMPLX", ilm, ilm2);
        break;
      case TY_DCMPLX:
        ilm = lower_conv(arg1, DT_REAL8);
        ilm2 = lower_conv(arg2, DT_REAL8);
        ilm = plower("oii", "DCMPLX", ilm, ilm2);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        ilm = lower_conv(arg1, DT_QUAD);
        ilm2 = lower_conv(arg2, DT_QUAD);
        ilm = plower("oii", "QCMPLX", ilm, ilm2);
        break;
#endif
      default:
        break;
      }
      A_ILMP(ast, ilm);
      return ilm;
    }

  /* conjg family */
  case I_CONJG:
  case I_DCONJG:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCONJG:
#endif
    ilm = intrin_name("CONJG", ast, in_c_cD_cQ);
    break;

  /* cos family */
  case I_COS:
  case I_DCOS:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCOS:
#endif
  case I_CCOS:
  case I_CDCOS:
    ilm = intrin_name("COS", ast, in_r_D_Q_C_CD_CQ);
    break;
  case I_COSD:
  case I_DCOSD:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCOSD:
#endif
    ilm = intrin_name("COSD", ast, in_r_D_Q);
    break;

  /* cosh family */
  case I_COSH:
  case I_DCOSH:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QCOSH:
#endif
    ilm = intrin_name("COSH", ast, in_r_D_Q);
    break;

  /* dble family */
  case I_DFLOTI:
  case I_DFLOAT:
  case I_DFLOTJ:
  case I_DBLE:
    ilm = conv_dble(ARGT_ARG(args, 0));
    A_ILMP(ast, ilm);
    return ilm;

  /* dprod */
  case I_DPROD:
    ilm = intrin_name("DPROD", ast, in_d);
    break;

  /* dim family */
  case I_IIDIM:
  case I_JIDIM:
  case I_KIDIM:
  case I_IDIM:
  case I_DDIM:
  case I_DIM:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QDIM:
#endif
    ilm = intrin_name("DIM", ast, in_I_K_r_D_Q);
    break;

  /* exp family */
  case I_EXP:
  case I_DEXP:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QEXP:
#endif
  case I_CEXP:
  case I_CDEXP:
    ilm = intrin_name("EXP", ast, in_r_D_Q_C_CD_CQ);
    break;

  /* ibclr family */
  case I_IIBCLR:
  case I_JIBCLR:
  case I_KIBCLR:
  case I_IBCLR:
    ilm = intrin_name("BCLR", ast, in_I_K);
    break;

  /* ibits family */
  case I_IIBITS:
  case I_JIBITS:
  case I_KIBITS:
  case I_IBITS:
    ilm = intrin_name("BITS", ast, in_I_K);
    break;

  /* ibset family */
  case I_IIBSET:
  case I_JIBSET:
  case I_KIBSET:
  case I_IBSET:
    ilm = intrin_name("BSET", ast, in_I_K);
    break;

  /* ibtest family */
  case I_BITEST:
  case I_BJTEST:
  case I_BTEST:
    ilm = intrin_name("BTEST", ast, in_il_K);
    break;

  case I_BKTEST:
    if (argdtype != DT_LOG8) {
      /*
       * Need to have a special case for BTEST of 64-bit integers whose
       * result dtype is logical*4. Compute the BTEST in 64-bit and create
       * a logical*8 result and then convert the result to logical*4.
       * Without the special case, the arguments are converted to
       * integer*4 value and then a 32-bit BTEST is performed.
       */
      ilm1 = lower_conv(ARGT_ARG(args, 0), DT_INT8);
      ilm2 = lower_conv(ARGT_ARG(args, 1), DT_INT8);
      ilm = plower("om", "KBTEST");
      plower("im", ilm1);
      plower("im", ilm2);
      plower("e");
      ilm = plower("oi", "I8TOI", ilm);
      A_ILMP(ast, ilm);
      return ilm;
    }
    ilm = plower("om", "KBTEST");
    break;

  /* ichar family */
  case I_ICHAR:
  case I_IACHAR:
    arg1 = ARGT_ARG(args, 0);
    fromdtype = A_NDTYPEG(arg1);
    if (DTY(fromdtype) == TY_NCHAR) {
      ilm = intrin_name("INCHAR", ast, in_i_k);
    } else {
      ilm = intrin_name("ICHAR", ast, in_i_k);
    }
    plower("im", intrinsic_args[0]);
    plower("e");
    dtype = A_DTYPEG(ast);
    if (dtype == DT_INT8) {
      /* convert to int8 */
      ilm = plower("oi", "ITOI8", ilm);
    }
    argsdone = 1;
    break;

  /* imag family */
  case I_AIMAG:
  case I_DIMAG:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QIMAG:
#endif
    ilm = intrin_name("IMAG", ast, in_r_D_Q);
    break;

  /* int family */
  case I_IDINT:
  case I_JIDINT:
  case I_IIDINT:
  case I_IINT:
  case I_JINT:
  case I_INT:
  case I_IFIX:
  case I_JIFIX:
  case I_IIFIX:
  case I_INT1:
  case I_INT2:
  case I_INT4:
  case I_INT8:
    dty = DTYG(A_NDTYPEG(ast));
    if (dty == TY_INT8) {
      ilm = conv_int8(ARGT_ARG(args, 0));
    } else if (dty == TY_INT) {
      ilm = conv_int(ARGT_ARG(args, 0));
    } else {
      ilm = lower_base(ARGT_ARG(args, 0));
      ilm2 = plower(
          "oS", "ICON",
          lower_getintcon(ty_to_lib[DTYG(A_NDTYPEG(ARGT_ARG(args, 0)))]));
      symfunc =
          lower_makefunc(mk_coercion_func_name(dty), A_NDTYPEG(ast), FALSE);
      ilm = plower("onsiiC", "IFUNC", 2, symfunc, ilm, ilm2, symfunc);
    }
    A_ILMP(ast, ilm);
    return ilm;

  case I_C_LOC:
  case I_C_FUNLOC:
  case I_LOC:
    ilm = plower("om", "LOC");
    break;

  case I_LOGICAL:
    arg1 = ARGT_ARG(args, 0);
    ilm = lower_conv(arg1, A_NDTYPEG(ast));
    argsdone = 1;
    break;

  /* log family */
  case I_ALOG:
  case I_DLOG:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QLOG:
#endif
  case I_CLOG:
  case I_CDLOG:
  case I_LOG:
    ilm = intrin_name("LOG", ast, in_r_D_Q_C_CD_CQ);
    break;

  case I_LOG10:
  case I_ALOG10:
  case I_DLOG10:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QLOG10:
#endif
    ilm = intrin_name("LOG10", ast, in_r_D_Q);
    break;

  /* max family */
  case I_MAX:
  case I_MAX0:  /* i*4,i*4 -> i*4 */
  case I_IMAX0: /* i*2,i*2 -> i*2 */
  case I_JMAX0: /* i*4,i*4 -> i*4 */
  case I_KMAX0: /* i*8,i*8 -> i*8 */
  case I_AMAX0:
  case I_AIMAX0:
  case I_AJMAX0:
  case I_MAX1:
  case I_IMAX1:
  case I_JMAX1:
  case I_KMAX1:
  case I_AMAX1: /* r*4,r*4 -> r*4 */
  case I_DMAX1:
    /*
    i0: BOS l0 n1 n0
    i4: BASE s37944 ;specstring$len
    i6: ICON s656   ;4
    i8: BASE s37931 ;speclist$len
    i10: KLD i8
    i12: I8TOI i10
    i14: KMAX i12 i6   ---> Should be "i14: IMAX i12 i6"
    i17: IMUL i14 i6
    i20: ITOI8 i17
    i22: KCON s610  ;0
    i24: KMAX i20 i22
    i27: KST i4 i24
    For intrinsic function, compiler will convert operands dtype to the same as
    the intrinsic, e.g. like "i20 ITOI8 i17" shows here. But when generating the
    MAX instruction, it checks operands dtype to decide which types of MAX to be
    generated. When we converting operands initially, symtab is not changed, so,
    MAX instruction just needs to use the same dtype as intrinsic function. e.g.
    the first  KMAX is incorrect here, as operands type is integer not
    integer*8. To fix the issue, we check whether operands have the same dtype,
    if yes we just user the first operand dtype, otherwise use the
    intrinsic-func dtype as the operands have been converted the same as the one
    of intrinsic-func.
    */
    arg0 = ARGT_ARG(args, 0);
    arg1 = ARGT_ARG(args, 1);
    input_ast = A_NDTYPEG(arg0) == A_NDTYPEG(arg1) ? arg0 : ast;
    ilm = intrin_name("MAX", input_ast, in_I_K_R_D);
    pairwise = 1;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMAX:
    arg0 = ARGT_ARG(args, 0);
    arg1 = ARGT_ARG(args, 1);
    input_ast = A_NDTYPEG(arg0) == A_NDTYPEG(arg1) ? arg0 : ast;
    ilm = intrin_name("MAX", input_ast, in_r_D_Q);
    pairwise = 1;
    break;
#endif

  /* min family */
  case I_MIN:
  case I_MIN0:
  case I_IMIN0:
  case I_JMIN0:
  case I_KMIN0:
  case I_AMIN0:
  case I_AIMIN0:
  case I_AJMIN0:
  case I_MIN1:
  case I_IMIN1:
  case I_JMIN1:
  case I_KMIN1:
  case I_AMIN1:
  case I_DMIN1:
    arg0 = ARGT_ARG(args, 0);
    arg1 = ARGT_ARG(args, 1);
    input_ast = A_NDTYPEG(arg0) == A_NDTYPEG(arg1) ? arg0 : ast;
    ilm = intrin_name("MIN", input_ast, in_I_K_R_D);
    pairwise = 1;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMIN:
    arg0 = ARGT_ARG(args, 0);
    arg1 = ARGT_ARG(args, 1);
    input_ast = A_NDTYPEG(arg0) == A_NDTYPEG(arg1) ? arg0 : ast;
    ilm = intrin_name("MIN", input_ast, in_r_D_Q);
    pairwise = 1;
    break;
#endif

  /* mod family */
  case I_IMOD:
  case I_JMOD:
  case I_KMOD:
  case I_AMOD:
  case I_DMOD:
  case I_MOD:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMOD:
#endif
    ilm = intrin_name("MOD", ast, in_i_K_A_D_Q);
    break;

  /* nint family */
  case I_IDNINT:
  case I_ININT:
  case I_JNINT:
  case I_KNINT:
  case I_NINT:
  case I_IIDNNT:
  case I_JIDNNT:
  case I_KIDNNT:
    dty = DTYG(A_NDTYPEG(ast));
#ifdef TARGET_SUPPORTS_QUADFP
    if (DTYG(A_NDTYPEG(ARGT_ARG(args, 0))) == TY_QUAD) {
      ilm = intrin_name("QNINT", ast, in_I_K);
    } else
#endif
    if (DTYG(A_NDTYPEG(ARGT_ARG(args, 0))) == TY_DBLE) {
      ilm = intrin_name("DNINT", ast, in_I_K);
    } else {
      ilm = intrin_name("NINT", ast, in_i_K);
    }
    if (dty != TY_INT && dty != TY_INT8) {
      plower("im", intrinsic_args[0]);
      plower("e");
      ilm2 = plower("oS", "ICON", lower_getintcon(ty_to_lib[TY_INT]));
      symfunc =
          lower_makefunc(mk_coercion_func_name(dty), A_NDTYPEG(ast), FALSE);
      ilm = plower("onsiiC", "IFUNC", 2, symfunc, ilm, ilm2, symfunc);
      A_ILMP(ast, ilm);
      return ilm;
    }
    break;

  /* not family */
  case I_INOT:
  case I_JNOT:
  case I_NOT:
  case I_COMPL:
    ilm = intrin_name("NOT", ast, in_i_K);
    break;

  /* or family */
  case I_IIOR:
  case I_JIOR:
  case I_IOR:
  case I_OR:
    ilm = intrin_name("OR", ast, in_i_K);
    break;

#ifdef TARGET_SUPPORTS_QUADFP
  case I_QREAL:
#endif
  case I_DREAL:
  case I_REAL:
    arg = ARGT_ARG(args, 0);
    argdtype = A_NDTYPEG(arg);
    ilm = lower_ilm(arg);
    switch (DTYG(argdtype)) {
    case TY_CMPLX:
      ilm = plower("oi", "REAL", ilm);
      argdtype = DT_REAL4;
      break;
    case TY_DCMPLX:
      ilm = plower("oi", "DREAL", ilm);
      argdtype = DT_REAL8;
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      ilm = plower("oi", "QREAL", ilm);
      argdtype = DT_QUAD;
      break;
#endif
    default:
      break;
    }
    ilm = lower_conv_ilm(ast, ilm, argdtype, A_NDTYPEG(ast));
    return ilm;

  /* real family */
  case I_FLOATI:
  case I_FLOATJ:
  case I_FLOAT:
  case I_SNGL:
    ilm = conv_real(ARGT_ARG(args, 0));
    A_ILMP(ast, ilm);
    return ilm;

  /* sin family */
  case I_SIN:
  case I_DSIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSIN:
#endif
  case I_CSIN:
  case I_CDSIN:
  case I_CQSIN:
    ilm = intrin_name("SIN", ast, in_r_D_Q_C_CD_CQ);
    break;
  case I_SIND:
  case I_DSIND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSIND:
#endif
    ilm = intrin_name("SIND", ast, in_r_D_Q);
    break;

  /* sinh family */
  case I_SINH:
  case I_DSINH:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSINH:
#endif
    ilm = intrin_name("SINH", ast, in_r_D_Q);
    break;

  /* sqrt family */
  case I_SQRT:
  case I_DSQRT:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSQRT:
#endif
  case I_CSQRT:
  case I_CDSQRT:
    ilm = intrin_name("SQRT", ast, in_r_D_Q_C_CD_CQ);
    break;

  /* tan family */
  case I_TAN:
  case I_DTAN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QTAN:
#endif
    ilm = intrin_name("TAN", ast, in_r_D_Q);
    break;
  case I_TAND:
  case I_DTAND:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QTAND:
#endif
    ilm = intrin_name("TAND", ast, in_r_D_Q);
    break;

  /* tanh family */
  case I_TANH:
  case I_DTANH:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QTANH:
#endif
    ilm = intrin_name("TANH", ast, in_r_D_Q);
    break;

  /* shift family */
  case I_JISHFT:
  case I_KISHFT:
  case I_ISHFT:
    ilm = intrin_name("ISHFT", ast, in_J_K);
    break;
  case I_IISHFT:
    ilm = plower("om", "IISHFT");
    break;
  case I_IISHFTC:
    ilm = plower("om", "IISHFTC");
    break;
  case I_JISHFTC:
  case I_KISHFTC:
  case I_ISHFTC:
    ilm = intrin_name("SHFTC", ast, in_I_K);
    break;
  case I_SHIFT:
    ilm = intrin_name("SHIFT", ast, in_i_K);
    break;
  case I_LSHIFT:
    ilm = intrin_name("ULSHIFT", ast, in_i_K);
    break;
  case I_RSHIFT:
    ilm = intrin_name("URSHIFT", ast, in_i_K);
    break;

  /* sign family */
  case I_IISIGN:
  case I_JISIGN:
  case I_KISIGN:
  case I_ISIGN:
  case I_DSIGN:
  case I_SIGN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSIGN:
#endif
    ilm = intrin_name("SIGN", ast, in_I_K_r_D_Q);
    break;

  /* xor family */
  case I_IIEOR:
  case I_JIEOR:
  case I_IEOR:
  case I_XOR:
  case I_NEQV:
    ilm = intrin_name("XOR", ast, in_i_K);
    break;

  case I_EQV:
    ilm = intrin_name("XNOR", ast, in_i_K);
    break;

  case I_LEN:
  case I_KLEN:
    arg1 = ARGT_ARG(args, 0);
    fromdtype = A_NDTYPEG(arg1);
    if (DTY(fromdtype) == TY_NCHAR) {
      ilm = intrin_name("NLEN", ast, in_i);
    } else {
      ilm = intrin_name("LEN", ast, in_i_K);
    }
    break;

  case I_LEN_TRIM:
    dtype = A_DTYPEG(ast);
    symfunc = lower_makefunc(mkRteRtnNm(RTE_lentrima), dtype, FALSE);
    if (dtype == DT_INT8) {
      ilm = plower("onsm", "KFUNC", nargs, symfunc);
    } else {
      ilm = plower("onsm", "IFUNC", nargs, symfunc);
    }
    break;

  case I_CEILING:
    dtype = A_NDTYPEG(ast);
    ilm = intrin_name("CEIL", ast, in_R_D_Q);
    break;
  case I_FLOOR:
    dtype = A_NDTYPEG(ast);
    ilm = intrin_name("FLOOR", ast, in_R_D_Q);
    break;

  case I_AINT:
  case I_DINT:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QINT:
#endif
    dtype = A_NDTYPEG(ast);
    ilm = intrin_name("INT", ast, in_A_D_Q);
    break;

  case I_ANINT:
  case I_DNINT:
    dtype = A_NDTYPEG(ast);
#ifdef TARGET_SUPPORTS_QUADFP
    if (DTYG(A_NDTYPEG(ARGT_ARG(args, 0))) == TY_QUAD) {
      ilm = intrin_name("QANINT", ast, in_A_D_Q);
    } else {
      ilm = intrin_name("NINT", ast, in_A_D);
    }
#else
    ilm = intrin_name("NINT", ast, in_A_D);
#endif
    break;

  case I_INDEX:
  case I_KINDEX:
    if (nargs == 4) {
      int a3, s3;
      /* check for .false. third argument */
      a3 = ARGT_ARG(args, 2);
      if (A_TYPEG(a3) == A_CNST) {
        s3 = A_SPTRG(a3);
        if (DTYPEG(s3) == DT_LOG && CONVAL2G(s3) == 0) {
          nargs = 2;
        }
      }
    }
    if (nargs == 2) {
      for (i = 0; i < nargs; ++i) {
        intrinsic_args[i] = lower_base(ARGT_ARG(args, i));
      }
      if (DTY(A_DTYPEG(ARGT_ARG(args, 0))) == TY_NCHAR) {
        ilm = plower("om", "NINDEX");
        fromdtype = DT_INT4;
      } else if (A_NDTYPEG(ast) == DT_INT4) {
        ilm = plower("om", "INDEX");
        fromdtype = DT_INT4;
      } else if (A_NDTYPEG(ast) == DT_INT8) {
        ilm = plower("om", "KINDEX");
        fromdtype = DT_INT8;
      } else {
        ilm = plower("om", "INDEX");
        fromdtype = DT_INT4;
      }
    } else {
      dtype = A_DTYPEG(ast);
      if (dtype == DT_INT8) {
        fromdtype = DT_INT8;
      } else {
        fromdtype = DT_INT4;
      }
      ilm = f90_function(mkRteRtnNm(RTE_indexa), fromdtype, args, nargs);
    }
    break;

  case I_NINDEX:
    for (i = 0; i < 2; ++i) {
      intrinsic_args[i] = lower_base(ARGT_ARG(args, i));
    }
    ilm = plower("om", "NINDEX");
    break;

  case I_ALLOCATED:
    rtlRtn = RTE_allocated;
    ilm = f90_function(mkRteRtnNm(rtlRtn), stb.user.dt_log, args, nargs);
    break;

  case I_PRESENT:
    /* single argument */
    arg0 = ARGT_ARG(args, 0);
    if (!XBIT(57, 0x20000000) && !XBIT(57, 0x8000)) {
      /* streamlined present -- 'absent' is just NULL */
      if (A_TYPEG(arg0) == A_ID) {
        int sym;
        sym = A_SPTRG(arg0);
        if (SCG(sym) == SC_BASED && MIDNUMG(sym) && XBIT(57, 0x80000)) {
          ilm1 = lower_base_sptr(MIDNUMG(sym));
        } else if (POINTERG(sym) && NEWARGG(sym)) {
          /* special case for optional pointer arguments */
          ilm1 = lower_base_sptr(NEWARGG(sym));
        } else
          ilm1 = lower_base(arg0);
        ilm2 = plower("oS", "ACON", lowersym.ptrnull);
        ilm = plower("oii", "PCMP", ilm1, ilm2);
        ilm = plower("oi", "NE", ilm);
      } else {
        /* something interesting created by IPA inlining or arg
         * propagation
         */
        sptr = lower_getlogcon(SCFTN_TRUE);
        ilm = plower("oS", "LCON", sptr);
      }
    } else {
      dtype = A_DTYPEG(ast);
      argdtype = A_NDTYPEG(arg0);
      if (DTYG(argdtype) == TY_CHAR || DTY(argdtype) == TY_NCHAR) {
        rtlRtn = RTE_presentc;
      } else {
        if (A_TYPEG(arg0) == A_ID && POINTERG(A_SPTRG(arg0)) &&
            !XBIT(57, 0x80000))
          rtlRtn = RTE_present_ptr;
        else {
          rtlRtn = RTE_present;
        }
      }
      symfunc = lower_makefunc(mkRteRtnNm(rtlRtn), dtype, FALSE);
      ilm1 = 0;
      if (A_TYPEG(arg0) == A_ID) {
        int sym;
        sym = A_SPTRG(arg0);
        if (SCG(sym) == SC_BASED && MIDNUMG(sym) && XBIT(57, 0x80000)) {
          ilm1 = lower_base_sptr(MIDNUMG(sym));
        } else if (POINTERG(sym) && NEWARGG(sym)) {
          /* special case for optional pointer arguments */
          ilm1 = lower_base_sptr(NEWARGG(sym));
        }
      }
      if (!ilm1)
        ilm1 = lower_base(arg0);
      ilm = plower("onsi", ltyped("FUNC", dtype), 1, symfunc, ilm1);
    }
    argsdone = 1;
    break;

  case I_LGE:
  case I_LGT:
  case I_LLE:
  case I_LLT:
    if (nargs != 2) {
      lerror("wrong number of arguments for L[LG][ET] comparison intrinsic");
      return 0;
    }
    intrinsic_args[0] = lower_base(ARGT_ARG(args, 0));
    intrinsic_args[1] = lower_base(ARGT_ARG(args, 1));
    ilm = plower("oii", styped("CMP", A_DTYPEG(ARGT_ARG(args, 0))),
                 intrinsic_args[0], intrinsic_args[1]);
    switch (intr) {
    case I_LGE:
      ilm = plower("oi", "GE", ilm);
      break;
    case I_LGT:
      ilm = plower("oi", "GT", ilm);
      break;
    case I_LLE:
      ilm = plower("oi", "LE", ilm);
      break;
    case I_LLT:
      ilm = plower("oi", "LT", ilm);
      break;
    }
    A_ILMP(ast, ilm);
    return ilm;

  case I_MERGE:
    switch (DTY(A_DTYPEG(ast))) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_INT8:
    case TY_FLOAT:
    case TY_DBLE:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
    case TY_LOG8:
      for (i = 0; i < 2; i++) {
        arg = ARGT_ARG(args, i);
        lower_expression(arg);
        intrinsic_args[i] = lower_ilm(arg);
      }
      intrinsic_args[2] = lower_conv(ARGT_ARG(args, 2), DT_LOG4);
      ilm = intrin_name("MERGE", ast, in_Il_K_R_D_C_CD);
      nargs = 3;
      break;
    default:
      /* just treat like a function call */

      if ((DTY(A_DTYPEG(ast)) == TY_CMPLX || DTY(A_DTYPEG(ast)) == TY_DCMPLX
#ifdef TARGET_SUPPORTS_QUADFP
           || DTY(A_DTYPEG(ast)) == TY_QCMPLX
#endif
           ) && (XBIT(70, 0x40000000))) {
        for (i = 0; i < 2; i++) {
          arg = ARGT_ARG(args, i);
          lower_expression(arg);
          intrinsic_args[i] = lower_ilm(arg);
        }
        intrinsic_args[2] = lower_conv(ARGT_ARG(args, 2), DT_LOG4);
        ilm = intrin_name("MERGE", ast, in_R_D_Q_C_CD_CQ);
        nargs = 3;
      } else {
        ilm = lower_function(ast);
        A_ILMP(ast, ilm);
        return ilm;
      }
      break;
    }
    break;

  case I_ADJUSTL:
  case I_ADJUSTR:
  case I_TRIM:
    ilm = lower_function(ast);
    A_ILMP(ast, ilm);
    return ilm;

  case I_ILEN:
    /* just treat like a function call, with pghpf prefix */
    ilm = lower_function(ast);
    A_ILMP(ast, ilm);
    return ilm;

  case I_ISNAN:
    arg = ARGT_ARG(args,0);
    lower_expression(arg);
    ilm = plower("oi", styped("ISNAN", A_DTYPEG(arg)),
                 lower_ilm(arg));
    A_ILMP(ast, ilm);
    return ilm;

  case I_NLEN:
    ilm = intrin_name("NLEN", ast, in_i);
    break;

  case I_SIZE:
    arg = ARGT_ARG(args, 0);
    shape = A_SHAPEG(arg);
    cnt = SHD_NDIM(shape);
    lower_expression(ARGT_ARG(args, 1));
    for (i = 0; i < cnt; ++i) {
      lower_expression(check_member(arg, SHD_LWB(shape, i)));
      if (SHD_UPB(shape, i))
        lower_expression(check_member(arg, SHD_UPB(shape, i)));
      lower_expression(check_member(arg, SHD_STRIDE(shape, i)));
    }
    num = 0;
    intrinsic_args[num++] =
        plower("oS", lowersym.bnd.con, lower_getiszcon(cnt));
    intrinsic_args[num++] = lower_base(ARGT_ARG(args, 1));
    for (i = 0; i < cnt; ++i) {
      argdtype = A_DTYPEG(arg);
      if (ASSUMSHPG(find_array(arg, 0)) &&
          A_TYPEG(ADD_LWBD(argdtype, i)) == A_CNST &&
          ADD_LWBD(argdtype, i) != astb.bnd.one && ADD_LWAST(argdtype, i)) {
        /* if the argument is an assumed shape array with constant lbound
         * that is not 1, the actual lower bound is in a compiler generated
         * temp that is set by code generated in dpm_out.c:set_assumed_bounds.
         */
        int lb = ADD_LWAST(argdtype, i);
        lower_expression(lb);
        intrinsic_args[num++] = lower_base(lb);
      } else {
        intrinsic_args[num++] =
            lower_base(check_member(arg, SHD_LWB(shape, i)));
      }
      if (SHD_UPB(shape, i)) {
        intrinsic_args[num++] =
            lower_base(check_member(arg, SHD_UPB(shape, i)));
      } else {
        intrinsic_args[num++] = lower_null_arg();
      }
      intrinsic_args[num++] =
          lower_base(check_member(arg, SHD_STRIDE(shape, i)));
    }
    dtype = A_DTYPEG(ast);
    symfunc = lower_makefunc(mkRteRtnNm(RTE_size), DT_INT8, FALSE);
    if (dtype == DT_INT8) {
      ilm = plower("onsm", "KFUNC", num, symfunc);
    } else {
      ilm = plower("onsm", "IFUNC", num, symfunc);
    }
    for (i = 0; i < num; ++i) {
      plower("im", intrinsic_args[i]);
    }
    plower("C", symfunc);
    A_ILMP(ast, ilm);
    return ilm;

  case I_LBOUND:
  case I_UBOUND:
    arg = ARGT_ARG(args, 0);
    shape = A_SHAPEG(arg);
    cnt = SHD_NDIM(shape);
    lower_expression(ARGT_ARG(args, 1));
    for (i = 0; i < cnt; ++i) {
      lower_expression(check_member(arg, SHD_LWB(shape, i)));
      if (SHD_UPB(shape, i))
        lower_expression(check_member(arg, SHD_UPB(shape, i)));
    }
    num = 0;
    intrinsic_args[num++] = plower("oS", "ICON", lower_getintcon(cnt));
    intrinsic_args[num++] = lower_base(ARGT_ARG(args, 1));
    for (i = 0; i < cnt; ++i) {
      argdtype = A_DTYPEG(arg);
      if (ASSUMSHPG(find_array(arg, 0)) &&
          A_TYPEG(ADD_LWBD(argdtype, i)) == A_CNST &&
          ADD_LWBD(argdtype, i) != astb.bnd.one && ADD_LWAST(argdtype, i)) {
        /* if the argument is an assumed shape array with constant lbound
         * that is not 1, the actual lower bound is in a compiler generated
         * temp that is set by code generated in dpm_out.c:set_assumed_bounds.
         */
        int lb = ADD_LWAST(argdtype, i);
        lower_expression(lb);
        intrinsic_args[num++] = lower_base(lb);
      } else {
        intrinsic_args[num++] =
            lower_base(check_member(arg, SHD_LWB(shape, i)));
      }
      if (SHD_UPB(shape, i)) {
        intrinsic_args[num++] =
            lower_base(check_member(arg, SHD_UPB(shape, i)));
      } else {
        intrinsic_args[num++] = lower_null_arg();
      }
    }
    if (intr == I_LBOUND) {
      symfunc = lower_makefunc(mkRteRtnNm(RTE_lb), DT_INT4, FALSE);
    } else {
      symfunc = lower_makefunc(mkRteRtnNm(RTE_ub), DT_INT4, FALSE);
    }
    ilm = plower("onsm", "IFUNC", num, symfunc);
    for (i = 0; i < num; ++i) {
      plower("im", intrinsic_args[i]);
    }
    plower("C", symfunc);
    A_ILMP(ast, ilm);
    return ilm;

  case I_MODULO:
    /*
     * see semfunc.c for the spelling of the function name.
     */
    dtype = A_NDTYPEG(ast);
    symfunc = A_SPTRG(A_LOPG(ast));
    for (i = 0; i < nargs; ++i) {
      ilm = lower_ilm(ARGT_ARG(args, i));
      ilm = plower("oi", "DPVAL", ilm);
      intrinsic_args[i] = ilm;
    }
    ilm = plower("onsm", ltyped("FUNC", dtype), nargs, symfunc);
    break;

  case I_EXPONENT:
    dtype = A_DTYPEG(ast);
    switch (DTY(DDTG(A_NDTYPEG(ARGT_ARG(args, 0))))) {
      case TY_REAL:
        rtlRtn = RTE_exponx;
        break;
      case TY_DBLE:
        rtlRtn = RTE_expondx;
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        rtlRtn = RTE_exponqx;
        break;
#endif
      default:
        ast_error("unexpected argument type for exponent", ast);
        break;
    }
    rtn_name = mkRteRtnNm(rtlRtn);
    retDtype = (dtype == DT_INT8) ? DT_INT8 : DT_INT4;
    ilm = f90_value_function(rtn_name, retDtype, args, nargs);
    break;

  case I_FRACTION:
    if (DTY(DDTG(A_NDTYPEG(ARGT_ARG(args, 0)))) == TY_REAL) {
      ilm = f90_value_function(mkRteRtnNm(RTE_fracx), DT_REAL4, args, nargs);
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (DTY(DDTG(A_NDTYPEG(ARGT_ARG(args, 0)))) == TY_QUAD) {
      ilm = f90_value_function(mkRteRtnNm(RTE_fracqx), DT_QUAD, args, nargs);
#endif
    } else {
      ilm = f90_value_function(mkRteRtnNm(RTE_fracdx), DT_REAL8, args, nargs);
    }
    break;

  case I_RRSPACING:
    if (DTY(DDTG(A_NDTYPEG(ast))) == TY_REAL) {
      ilm =
          f90_value_function(mkRteRtnNm(RTE_rrspacingx), DT_REAL4, args, nargs);
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (DTY(DDTG(A_NDTYPEG(ast))) == TY_QUAD) {
      ilm =
          f90_value_function(mkRteRtnNm(RTE_rrspacingqx), DT_QUAD, args, nargs);
#endif
    } else {
      ilm = f90_value_function(mkRteRtnNm(RTE_rrspacingdx), DT_REAL8, args,
                               nargs);
    }
    break;
  case I_SPACING:
    if (DTY(DDTG(A_NDTYPEG(ast))) == TY_REAL) {
      ilm = f90_value_function(mkRteRtnNm(RTE_spacingx), DT_REAL4, args, nargs);
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (DTY(DDTG(A_NDTYPEG(ast))) == TY_QUAD) {
      ilm = f90_value_function(mkRteRtnNm(RTE_spacingqx), DT_QUAD, args, nargs);
#endif
    } else {
      ilm =
          f90_value_function(mkRteRtnNm(RTE_spacingdx), DT_REAL8, args, nargs);
    }
    break;
  case I_NEAREST:
    if (DTY(DDTG(A_NDTYPEG(ast))) == TY_REAL) {
      ilm = f90_value_function(mkRteRtnNm(RTE_nearestx), DT_REAL4, args, nargs);
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (DTY(DDTG(A_NDTYPEG(ast))) == TY_QUAD) {
      ilm = f90_value_function(mkRteRtnNm(RTE_nearestqx), DT_QUAD, args, nargs);
#endif
    } else {
      ilm =
          f90_value_function(mkRteRtnNm(RTE_nearestdx), DT_REAL8, args, nargs);
    }
    break;
  case I_SCALE:
    if (DTY(DDTG(A_NDTYPEG(ast))) == TY_REAL) {
      ilm =
          f90_value_function_I2(mkRteRtnNm(RTE_scalex), DT_REAL4, args, nargs);
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (DTY(DDTG(A_NDTYPEG(ast))) == TY_QUAD) {
      ilm =
          f90_value_function_I2(mkRteRtnNm(RTE_scaleqx), DT_QUAD, args, nargs);
#endif
    } else {
      ilm =
          f90_value_function_I2(mkRteRtnNm(RTE_scaledx), DT_REAL8, args, nargs);
    }
    break;
  case I_SET_EXPONENT:
    if (DTY(DDTG(A_NDTYPEG(ast))) == TY_REAL) {
      ilm =
          f90_value_function_I2(mkRteRtnNm(RTE_setexpx), DT_REAL4, args, nargs);
#ifdef TARGET_SUPPORTS_QUADFP
    } else if (DTY(DDTG(A_NDTYPEG(ast))) == TY_QUAD) {
      ilm =
          f90_value_function_I2(mkRteRtnNm(RTE_setexpqx), DT_QUAD, args, nargs);
#endif
    } else {
      ilm = f90_value_function_I2(mkRteRtnNm(RTE_setexpdx), DT_REAL8, args,
                                  nargs);
    }
    break;
  case I_VERIFY:
    dtype = A_DTYPEG(ast);
    rtlRtn = (DTY(DDTG(A_NDTYPEG(ARGT_ARG(args, 0)))) == TY_CHAR) ? RTE_verifya
                                                                  : RTE_nverify;
    retDtype = (dtype == DT_INT8) ? DT_INT8 : DT_INT4;
    ilm = f90_function(mkRteRtnNm(rtlRtn), retDtype, args, nargs);
    break;
  case I_SCAN:
    dtype = A_DTYPEG(ast);
    rtlRtn = (DTY(DDTG(A_NDTYPEG(ARGT_ARG(args, 0)))) == TY_CHAR) ? RTE_scana
                                                                  : RTE_scana;
    retDtype = (dtype == DT_INT8) ? DT_INT8 : DT_INT4;
    ilm = f90_function(mkRteRtnNm(rtlRtn), retDtype, args, nargs);
    break;
  case I_ASSOCIATED:
    /* determine whether the first argument is NULL or not,
     * and, if the second argument is present, whether the first
     * argument points to the second */
    arg1 = ARGT_ARG(args, 0);
    arg2 = ARGT_ARG(args, 1);
    ilm1 = lower_target(arg1);
    sptr = 0;
    switch (A_TYPEG(arg1)) {
    case A_ID:
      sptr = A_SPTRG(arg1);
      if (!XBIT(49, 0x20000000)) {
        ilm1 = lower_typeload(DT_ADDR, ilm1);
      } else {
        ilm1 = lower_typeload(DT_PTR, ilm1);
      }
      break;
    case A_MEM:
      sptr = A_SPTRG(A_MEMG(arg1));
      if (!XBIT(49, 0x20000000)) {
        ilm1 = lower_typeload(DT_ADDR, ilm1);
      } else {
        ilm1 = lower_typeload(DT_PTR, ilm1);
      }
      break;
    default:
      ilm1 = lower_typeload(DT_PTR, ilm1);
      break;
    }
    if (sptr && !XBIT(49, 0x20000000)) {
      ilm2 = plower("oS", "ACON", lowersym.ptrnull);
      ilm = plower("oii", "PCMP", ilm1, ilm2);
    } else if (DT_PTR == DT_INT || DT_PTR != DT_INT8) {
      ilm2 = plower("oS", "ICON", lowersym.intzero);
      ilm = plower("oii", "ICMP", ilm1, ilm2);
    } else {
      ilm2 = plower("oS", "KCON", lowersym.intzero);
      ilm = plower("oii", "KCMP", ilm1, ilm2);
    }
    ilm = plower("oi", "NE", ilm);
    if (arg2) {
      ilm2 = lower_address(arg2);
      if (sptr && !XBIT(49, 0x20000000)) {
        ilm2 = plower("oii", "PCMP", ilm1, ilm2);
      } else if (DT_PTR == DT_INT || DT_PTR != DT_INT8) {
        ilm2 = plower("oii", "ICMP", ilm1, ilm2);
      } else {
        ilm2 = plower("oii", "KCMP", ilm1, ilm2);
      }
      ilm2 = plower("oi", "EQ", ilm2);
      ilm = plower("oii", "LAND", ilm, ilm2);
    }
    argsdone = 1;
    break;

  case I_C_ASSOCIATED:
    /* determine whether the first argument is NULL or not,
     * and, if the second argument is present, whether the first
     * argument points to the second */
    arg1 = ARGT_ARG(args, 0);
    arg2 = ARGT_ARG(args, 1);
    ilm1 = lower_base(arg1);
    ilm1 = lower_typeload(A_DTYPEG(arg1), ilm1);
    if (A_DTYPEG(arg1) == DT_INT4) {
      ilm2 = plower("oS", "ICON", lowersym.intzero);
      ilm = plower("oii", "ICMP", ilm1, ilm2);
    } else {
      ilm2 = plower("oS", "KCON", lowersym.intzero);
      ilm = plower("oii", "KCMP", ilm1, ilm2);
    }
    ilm = plower("oi", "NE", ilm);
    if (arg2) {
      ilm2 = lower_base(arg2);
      ilm2 = lower_typeload(A_DTYPEG(arg1), ilm2);
      if (A_DTYPEG(arg1) == DT_INT4) {
        ilm2 = plower("oii", "ICMP", ilm1, ilm2);
      } else {
        ilm2 = plower("oii", "KCMP", ilm1, ilm2);
      }
      ilm2 = plower("oi", "EQ", ilm2);
      ilm = plower("oii", "LAND", ilm, ilm2);
    }
    argsdone = 1;
    break;

  case I_IS_CONTIGUOUS:
    ilm = f90_function(mkRteRtnNm(RTE_is_contiguous), stb.user.dt_log, args,
                       nargs);
    break;

  case I_RAN:
    for (i = 0; i < nargs; ++i) {
      intrinsic_args[i] = lower_base(ARGT_ARG(args, i));
    }
    if (A_DTYPEG(ast) != DT_REAL8) {
      symfunc = lower_makefunc("ftn_ran", DT_REAL4, FALSE);
      ilm = plower("onsm", "RFUNC", nargs, symfunc);
    } else {
      /* -r8 */
      symfunc = lower_makefunc("ftn_dran", DT_REAL8, FALSE);
      ilm = plower("onsm", "DFUNC", nargs, symfunc);
    }
    break;

  case I_ZEXT:
  case I_IZEXT:
    symfunc = lower_makefunc("ftn_izext", DT_INT, TRUE);
    intrinsic_args[0] = plower("oi", "DPVAL", intrinsic_args[0]);
    intrinsic_args[1] = plower("on", "DPSCON", 4);
    nargs = 2;
    ilm = plower("onsm", "IFUNC", nargs, symfunc);
    break;

  case I_JZEXT:
    symfunc = lower_makefunc("ftn_jzext", DT_INT, TRUE);
    intrinsic_args[0] = plower("oi", "DPVAL", intrinsic_args[0]);
    intrinsic_args[1] = plower("on", "DPSCON", 4);
    nargs = 2;
    ilm = plower("onsm", "IFUNC", nargs, symfunc);
    break;

  case I_NUMBER_OF_PROCESSORS:
    symfunc = A_SPTRG(A_LOPG(ast));
    dtype = A_DTYPEG(ast);
    for (i = 0; i < nargs; ++i) {
      intrinsic_args[i] = lower_base(ARGT_ARG(args, i));
    }
    ilm = plower("onsm", ltyped("FUNC", dtype), nargs, symfunc);
    for (i = 0; i < nargs; ++i) {
      plower("im", intrinsic_args[i]);
    }
    plower("e");
    return ilm;

  case I_LEADZ:
    ilm = intrin_name_bsik("LEADZ", ast);
    break;
  case I_TRAILZ:
    ilm = intrin_name_bsik("TRAILZ", ast);
    break;
  case I_POPCNT:
    ilm = intrin_name_bsik("POPCNT", ast);
    break;
  case I_POPPAR:
    ilm = intrin_name_bsik("POPPAR", ast);
    break;

  case NEW_INTRIN:
    nm = SYMNAME(A_LOPG(A_SPTRG(ast)));
    if (strcmp(nm, "acos") == 0)
      ilm = intrin_name("ACOS", ast, in_r_D_C_CD);
    else if (strcmp(nm, "asin") == 0)
      ilm = intrin_name("ASIN", ast, in_r_D_C_CD);
    else if (strcmp(nm, "atan") == 0)
      ilm = intrin_name("ATAN", ast, in_r_D_C_CD);
    else if (strcmp(nm, "cosh") == 0)
      ilm = intrin_name("COSH", ast, in_r_D_C_CD);
    else if (strcmp(nm, "sinh") == 0)
      ilm = intrin_name("SINH", ast, in_r_D_C_CD);
    else if (strcmp(nm, "tanh") == 0)
      ilm = intrin_name("TANH", ast, in_r_D_C_CD);
    else if (strcmp(nm, "tan") == 0)
      ilm = intrin_name("TAN", ast, in_r_D_C_CD);
    else {
      ast_error("unrecognized NEW INTRINSIC", ast);
      break;
    }
    A_ILMP(ast, ilm);
    break;

    /*------------------*/

  case I_DATE:
  case I_EXIT:
  case I_IDATE:
  case I_TIME:
  case I_MVBITS:

  case I_SECNDS:
  case I_DATE_AND_TIME:
  case I_RANDOM_NUMBER:
  case I_RANDOM_SEED:
  case I_CPU_TIME:
  case I_SYSTEM_CLOCK:
  case I_KIND:
  case I_SELECTED_INT_KIND:
  case I_SELECTED_REAL_KIND:
  case I_EPSILON:
  case I_HUGE:
  case I_TINY:
  case I_NULLIFY:
  case I_RANF:
  case I_RANGET:
  case I_RANSET:
  case I_INT_MULT_UPPER:

  case I_ALL:
  case I_ANY:
  case I_COUNT:
  case I_DOT_PRODUCT:
  case I_MATMUL:
  case I_MATMUL_TRANSPOSE:
  case I_FINDLOC:
  case I_MAXLOC:
  case I_MAXVAL:
  case I_MINLOC:
  case I_MINVAL:
  case I_PACK:
  case I_PRODUCT:
  case I_SUM:
  case I_SPREAD:
  case I_TRANSPOSE:
  case I_UNPACK:
  case I_CSHIFT:
  case I_EOSHIFT:
  case I_RESHAPE:
  case I_SHAPE:
  case I_BIT_SIZE:
  case I_DIGITS:
  case I_MAXEXPONENT:
  case I_MINEXPONENT:
  case I_PRECISION:
  case I_RADIX:
  case I_RANGE:
  case I_REPEAT:
  case I_TRANSFER:
  case I_DOTPRODUCT:
  case I_PROCESSORS_SHAPE:
  case I_LASTVAL:
  case I_REDUCE_SUM:
  case I_REDUCE_PRODUCT:
  case I_REDUCE_ANY:
  case I_REDUCE_ALL:
  case I_REDUCE_PARITY:
  case I_REDUCE_IANY:
  case I_REDUCE_IALL:
  case I_REDUCE_IPARITY:
  case I_REDUCE_MINVAL:
  case I_REDUCE_MAXVAL:
  case I_PTR2_ASSIGN:
  case I_PTR_COPYIN:
  case I_PTR_COPYOUT:
  case I_UNIT:
  case I_LENGTH:
  case I_COT:
  case I_DCOT:
  case I_SHIFTL:
  case I_SHIFTR:
  case I_DSHIFTL:
  case I_DSHIFTR:
  case I_C_F_POINTER:
  case I_C_F_PROCPOINTER:

  default:
    ast_error("unknown intrinsic function", ast);
    return 0;
  }

  if (!argsdone) {
    if (pairwise && nargs > 2) {
      plower("ii", intrinsic_args[0], intrinsic_args[1]);
      for (i = 2; i < nargs; ++i) {
        ilm = plower("Oii", ilm, intrinsic_args[i]);
      }
    } else {
      for (i = 0; i < nargs; ++i) {
        plower("im", intrinsic_args[i]);
      }
      plower("e");
    }
  }

  /* convert output type? */
  switch (intr) {
  /* max/min family */
  case I_MAX:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMAX:
#endif
  case I_MIN:
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QMIN:
#endif
  case I_MAX0: /* i*4,i*4 -> i*4 */
  case I_MIN0:
  case I_IMAX0: /* i*2,i*2 -> i*2 */
  case I_IMIN0:
  case I_JMAX0: /* i*4,i*4 -> i*4 */
  case I_JMIN0:
  case I_AMAX0:
  case I_AMIN0:
  case I_AIMAX0:
  case I_AIMIN0:
  case I_AJMAX0:
  case I_AJMIN0:
  case I_MAX1:
  case I_KMAX1:
  case I_MIN1:
  case I_KMIN1:
  case I_AMAX1: /* r*4,r*4 -> r*4 */
  case I_AMIN1:
  case I_DMAX1:
  case I_DMIN1:
  case I_IMAX1:
  case I_IMIN1:
  case I_JMAX1:
  case I_JMIN1:
    ilm = lower_conv_ilm(ast, ilm, argdtype, A_NDTYPEG(ast));
    break;
  case I_ANINT:
  case I_DNINT:
    dtype = DDTG(A_NDTYPEG(ast));
    if (dtype != DDTG(argdtype)) {
      ilm2 = plower("oS", "ICON", lower_getintcon(ty_to_lib[DTYG(argdtype)]));
      symfunc = lower_makefunc(mk_coercion_func_name(DTYG(dtype)), dtype, TRUE);
      ilm = plower("onsiiC", ltyped("FUNC", dtype), 2, symfunc, ilm, ilm2,
                   symfunc);
      A_ILMP(ast, ilm);
    }
    break;
  case I_INDEX:
  case I_KINDEX:
    ilm = lower_conv_ilm(ast, ilm, fromdtype, A_NDTYPEG(ast));
    FLANG_FALLTHROUGH;
  default:
    break;
  }
  lower_disable_ptr_chk = save_disable_ptr_chk;

  return ilm;
} /* lower_intrinsic */

#if AST_MAX != 165
#error "Need to edit lowerexp.c to add or delete A_... AST types"
#endif

static int _xtoi(int, int, char *);

static void
lower_ast(int ast, int *unused)
{
  int dtype, rdtype, lop, rop, lilm, rilm, ilm = 0, base = 0;
  int ss, ndim, i, sptr, checksubscr, pointersubscr;
  int subscriptilm[10], subscriptilmx[10], lowerboundilm[10], upperboundilm[10];
  LOGICAL norm;

  dtype = A_DTYPEG(ast);
  A_NDTYPEP(ast, dtype);
  switch (A_TYPEG(ast)) {
  case A_NULL:
    break;
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
      ilm = lower_bin_arith(ast, "ADD", dtype, dtype);
      break;
    case OP_CMP:
      ilm = lower_bin_arith(ast, "CMP", dtype, dtype);
      break;
    case OP_DIV:
      ilm = lower_bin_arith(ast, "DIV", dtype, dtype);
      break;
    case OP_LAND:
    case OP_SCAND:
      if (XBIT(125, 0x8))
        ilm = lower_bin_logical(ast, "UXLAND");
      else
        ilm = lower_bin_logical(ast, "LAND");
      break;
    case OP_LEQV:
      if (XBIT(125, 0x8))
        ilm = lower_bin_logical(ast, "UXLEQV");
      else
        ilm = lower_bin_logical(ast, "LEQV");
      break;
    case OP_LNEQV:
      if (XBIT(125, 0x8))
        ilm = lower_bin_logical(ast, "UXLNEQV");
      else
        ilm = lower_bin_logical(ast, "XOR");
      break;
    case OP_LOR:
      ilm = lower_bin_logical(ast, "LOR");
      break;
    case OP_MUL:
      ilm = lower_bin_arith(ast, "MUL", dtype, dtype);
      break;
    case OP_SUB:
      ilm = lower_bin_arith(ast, "SUB", dtype, dtype);
      break;
    case OP_XTOI:
    case OP_XTOX:
      rop = A_ROPG(ast);
      rdtype = A_NDTYPEG(rop);
      if (rdtype <= 0) {
        ast_error("unknown type in exponential power", ast);
        break;
      }
      switch (DTYG(rdtype)) {
      case TY_BINT:
      case TY_SINT:
      case TY_INT:
#define __MAXPOW 10
        if (A_ALIASG(rop)) {
          int csym, cval;
          rop = A_ALIASG(rop);
          csym = A_SPTRG(rop);
          cval = CONVAL2G(csym);
          if ((flg.ieee && cval != 1 && cval != 2) || 
               XBIT(124, 0x200) || cval < 1 || cval > __MAXPOW) {
            /* don't replace ** with a sequence of multiplies */
            ilm = lower_bin_arith(ast, "TOI", dtype, DT_INT4);
          } else {
            ilm = lower_ilm(A_LOPG(ast));
            ilm = _xtoi(ilm, cval, ltyped("MUL", dtype));
          }
        } else {
          ilm = lower_bin_arith(ast, "TOI", dtype, DT_INT4);
        }
        break;
      case TY_INT8:
        if (A_ALIASG(rop)) {
          int csym, cval;
          rop = A_ALIASG(rop);
          csym = A_SPTRG(rop);
          cval = CONVAL2G(csym);
          if ((flg.ieee && cval != 1 && cval != 2) || CONVAL1G(csym) ||
               XBIT(124, 0x200) || cval < 1 || cval > __MAXPOW) {
            /* don't replace ** with a sequence of multiplies */
            ilm = lower_bin_arith(ast, "TOK", dtype, DT_INT8);
          } else {
            ilm = lower_ilm(A_LOPG(ast));
            ilm = _xtoi(ilm, cval, ltyped("MUL", dtype));
          }
        } else
          ilm = lower_bin_arith(ast, "TOK", dtype, DT_INT8);
        break;
      case TY_CMPLX:
        ilm = lower_bin_arith(ast, "TOC", dtype, dtype);
        break;
      case TY_DCMPLX:
        ilm = lower_bin_arith(ast, "TOCD", dtype, dtype);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
        ilm = lower_bin_arith(ast, "TOCQ", dtype, dtype);
        break;
#endif
      case TY_REAL:
        ilm = lower_bin_arith(ast, "TOR", dtype, dtype);
        break;
      case TY_DBLE:
        ilm = lower_bin_arith(ast, "TOD", dtype, dtype);
        break;
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QUAD:
        ilm = lower_bin_arith(ast, "TOQ", dtype, dtype);
        break;
#endif
      default:
        ast_error("unexpected exponent type", ast);
        break;
      }
      break;
    case OP_EQ:
      ilm = lower_bin_comparison(ast, "EQ");
      break;
    case OP_GE:
      ilm = lower_bin_comparison(ast, "GE");
      break;
    case OP_GT:
      ilm = lower_bin_comparison(ast, "GT");
      break;
    case OP_LE:
      ilm = lower_bin_comparison(ast, "LE");
      break;
    case OP_LT:
      ilm = lower_bin_comparison(ast, "LT");
      break;
    case OP_NE:
      ilm = lower_bin_comparison(ast, "NE");
      break;
    case OP_CAT:
      lilm = lower_base(A_LOPG(ast));
      rilm = lower_base(A_ROPG(ast));
      if (DTY(A_NDTYPEG(ast)) == TY_NCHAR) {
        ilm = plower("oii", "NSCAT", lilm, rilm);
      } else {
        ilm = plower("oii", "SCAT", lilm, rilm);
      }
      break;
    case OP_AIF:
    case OP_CON:
    case OP_FUNC:
    case OP_LD:
    case OP_LOG:
    case OP_ST:
    default:
      ast_error("don't know how to handle type binary operator", ast);
      break;
    }
    base = ilm;
    break;

  case A_CMPLXC:
    switch (DTYG(dtype)) {
    case TY_CMPLX:
      lilm = lower_conv(A_LOPG(ast), DT_REAL4);
      rilm = lower_conv(A_ROPG(ast), DT_REAL4);
      ilm = plower("oii", "CMPLX", lilm, rilm);
      FLANG_FALLTHROUGH;
    case TY_DCMPLX:
      lilm = lower_conv(A_LOPG(ast), DT_REAL8);
      rilm = lower_conv(A_ROPG(ast), DT_REAL8);
      ilm = plower("oii", "DCMPLX", lilm, rilm);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      lilm = lower_conv(A_LOPG(ast), DT_QUAD);
      rilm = lower_conv(A_ROPG(ast), DT_QUAD);
      ilm = plower("oii", "QCMPLX", lilm, rilm);
      break;
#endif
    default:
      ast_error("unknown operand type for (real,imag)", ast);
      break;
    }
    base = ilm;
    break;

  case A_CNST:
    if (dtype <= 0) {
      ast_error("unrecognized data type", ast);
      break;
    }
    sptr = A_SPTRG(ast);
    lower_visit_symbol(sptr);
    switch (DTYG(dtype)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
      ilm = plower("oS", "ICON", sptr);
      base = ilm;
      break;
    case TY_INT8:
      ilm = plower("oS", "KCON", sptr);
      base = ilm;
      break;
    case TY_LOG8:
      sptr = cngcon(sptr, DTYG(dtype), DT_INT8);
      ilm = plower("oS", "KCON", sptr);
      base = ilm;
      break;
    case TY_REAL:
      ilm = plower("oS", "RCON", sptr);
      base = ilm;
      break;
    case TY_DBLE:
      ilm = plower("oS", "DCON", sptr);
      base = ilm;
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      ilm = plower("oS", "QCON", sptr);
      base = ilm;
      break;
#endif
    case TY_CMPLX:
      ilm = plower("oS", "CCON", sptr);
      base = ilm;
      break;
    case TY_DCMPLX:
      ilm = plower("oS", "CDCON", sptr);
      base = ilm;
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      ilm = plower("oS", "CQCON", sptr);
      base = ilm;
      break;
#endif
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
      sptr = lower_getintcon(cngcon(CONVAL2G(sptr), DTYG(dtype), DT_LOG4));
      ilm = plower("oS", "LCON", sptr);
      base = ilm;
      break;
    case TY_CHAR:
    case TY_NCHAR:
      ilm = plower("os", "BASE", sptr);
      base = ilm;
      break;
    case TY_HOLL:
      ilm = plower("os", "BASE", sptr);
      break;
    case TY_WORD:
      sptr = lower_getintcon(cngcon(CONVAL2G(sptr), DTYG(dtype), DT_INT4));
      ilm = plower("oS", "ICON", sptr);
      base = ilm;
      break;
    case TY_DWORD:
      sptr = cngcon(sptr, DTYG(dtype), DT_INT8);
      ilm = plower("oS", "KCON", sptr);
      base = ilm;
      break;
    default:
      ast_error("unrecognized constant type", ast);
      break;
    }
    break;

  case A_CONV:
    /* see if no conversion is required */
    lop = A_LOPG(ast);
    if (eq_dtype(dtype, A_NDTYPEG(lop))) {
      /* no conversion, copy the ALIAS field, so
       * spurious converted constants appear constant */
      A_ALIASP(ast, A_ALIASG(lop));
      ilm = A_ILMG(lop);
      base = A_BASEG(lop);
    } else {
      int alias;
      alias = A_ALIASG(ast);
      if (alias && alias != ast && eq_dtype(dtype, A_DTYPEG(alias))) {
        /* put out the constant */
        lower_ast(alias, unused);
        ilm = A_ILMG(alias);
        base = A_BASEG(alias);
      } else {
        switch (DTYG(dtype)) {
        case TY_BINT:
          ilm = conv_bint(lop);
          break;
        case TY_SINT:
          ilm = conv_sint(lop);
          break;
        case TY_INT:
          ilm = conv_int(lop);
          break;
        case TY_PTR:
          if (XBIT(49, 0x100)) { /* 64-bit pointers */
            ilm = conv_int8(lop);
          } else {
            ilm = conv_int(lop);
          }
          break;
        case TY_INT8:
          ilm = conv_int8(lop);
          break;
        case TY_WORD:
          ilm = conv_word(lop);
          break;
        case TY_DWORD:
          ilm = conv_dword(lop);
          break;
        case TY_BLOG:
          ilm = conv_blog(lop);
          break;
        case TY_SLOG:
          ilm = conv_slog(lop);
          break;
        case TY_LOG:
          ilm = conv_log(lop);
          break;
        case TY_LOG8:
          ilm = conv_log8(lop);
          break;
        case TY_REAL:
          ilm = conv_real(lop);
          break;
        case TY_DBLE:
          ilm = conv_dble(lop);
          break;
#ifdef TARGET_SUPPORTS_QUADFP
        case TY_QUAD:
          ilm = conv_quad(lop);
          break;
#endif
        case TY_CMPLX:
          ilm = conv_cmplx(lop);
          break;
        case TY_DCMPLX:
          ilm = conv_dcmplx(lop);
          break;
#ifdef TARGET_SUPPORTS_QUADFP
        case TY_QCMPLX:
          ilm = conv_qcmplx(lop);
          break;
#endif
        case TY_CHAR:
        case TY_NCHAR:
          ilm = lower_ilm(lop);
          break;
        default:
          ast_error("unknown target type for conversion", ast);
          lerror("target type was %d", dtype);
          break;
        }
        base = ilm;
      }
    }
    break;

  case A_FUNC:
    /* function call */
    if (is_iso_cptr(A_DTYPEG(A_LOPG(ast))) && CFUNCG(A_SPTRG(A_LOPG(ast)))) {
      /* functions with BIND(c) and returning iso_cptrs should be treated as
         functions returning integers (pointers),  for pass
         by value and all processing
       */
      A_NDTYPEP(A_LOPG(ast), DT_PTR);
    }
    ilm = lower_function(ast);
    base = ilm;
    break;

  case A_ID:
    sptr = A_SPTRG(ast);
    dtype = DTYPEG(sptr);
    A_NDTYPEP(ast, dtype);
    break;

  case A_INTR:
    ilm = lower_intrinsic(ast);
    base = ilm;
    break;

  case A_INIT:
    ast_error("unexpected AST type", ast);
    break;

  case A_LABEL:
    sptr = A_SPTRG(ast);
    if (FMTPTG(sptr)) {
      /* FORMAT label */
      if (lowersym.loc == 0) {
        lowersym.loc = lower_makefunc(mkRteRtnNm(RTE_loc), DT_PTR, FALSE);
      }
      ilm = plower("oS", "BASE", FMTPTG(sptr));
      if (XBIT(49, 0x100)) {
        ilm = plower("onsiC", "KFUNC", 1, lowersym.loc, ilm, lowersym.loc);
      } else {
        ilm = plower("onsiC", "IFUNC", 1, lowersym.loc, ilm, lowersym.loc);
      }
    } else {
      /* GOTO label */
      lower_visit_symbol(sptr);
      ilm = plower("oS", "ACON", sptr);
    }
    base = ilm;
    break;

  case A_MEM: /* member */
    dtype = DTYPEG(A_SPTRG(A_MEMG(ast)));

    if (DTY(dtype) == TY_PTR && DTY(DTY(dtype + 1)) != TY_PROC)
      dtype = DTY(dtype + 1);
    A_NDTYPEP(ast, dtype);
    break;

  case A_PAREN:
    ilm = lower_ilm(A_LOPG(ast));
    base = ilm;
    break;

  case A_SUBSCR:
    base = lower_base(A_LOPG(ast));
    sptr = 0;
    lop = A_LOPG(ast);
    if (A_TYPEG(lop) == A_ID) {
      sptr = A_SPTRG(lop);
    } else if (A_TYPEG(lop) == A_MEM) {
      sptr = A_SPTRG(A_MEMG(lop));
    }
    ss = A_ASDG(ast);
    ndim = ASD_NDIM(ss);
    for (i = 0; i < ndim; ++i) {
      lower_expression(ASD_SUBS(ss, i));
      subscriptilm[i] = lower_ilm(ASD_SUBS(ss, i));
      if (XBIT(68, 0x1)) {
        if (A_DTYPEG(ASD_SUBS(ss, i)) != DT_INT8)
          subscriptilm[i] = plower("oi", "ITOI8", subscriptilm[i]);
        subscriptilmx[i] = subscriptilm[i];
      } else {
        if (A_DTYPEG(ASD_SUBS(ss, i)) == DT_INT8)
          subscriptilmx[i] = plower("oi", "I8TOI", subscriptilm[i]);
        else
          subscriptilmx[i] = subscriptilm[i];
      }
    }
    norm = FALSE;
    if (XBIT(58, 0x22)) {
      norm = normalize_bounds(sptr);
    }
    checksubscr = 0;
    if (XBIT(70, 2) && !DESCARRAYG(sptr)) {
      /* -Mbounds set, no descriptor array */
      checksubscr = !lower_disable_subscr_chk;
#ifdef CUDAG
      if (CUDAG(GBL_CURRFUNC) & (CUDA_DEVICE | CUDA_GLOBAL)) {
        checksubscr = 0;
      }
#endif
      pointersubscr = 0;
      if (POINTERG(sptr))
        pointersubscr = 1;
    }
    /* need to linearize subscripts for HPF */
    if (sptr && !HCCSYMG(sptr) && LNRZDG(sptr) && XBIT(52, 4)) {
      int descilm, idxilm, linilm, desc;
      int dtype;
      desc = SDSCG(sptr);
      if (desc == 0 || NODESCG(sptr) || !DESCUSEDG(sptr) ||
          STYPEG(desc) == ST_PARAM) {
        /* linearized, -x 52 4, no descriptor.
         * actual bounds in datatype, don't need to normalize bounds */
        dtype = DTYPEG(sptr);
        /* dtype here is the linearized dtype.
         * get the old datatype */
        dtype = DTY(dtype - 1);
        if (dtype > 0) {
          lerror("unknown linearized datatype");
          return;
        }
        dtype = -dtype;
        linilm = 0;
        /* for reference A(i,j,k), dims A(i0:i1,j0:j1,k0:k1) */
        /* compute '((k-k0)*(j1-j0+1) + j-j0)*(i1-i0+1) + i-i0 + 1 */
        for (i = ndim - 1; i >= 0; --i) {
          int lw, up;
          int ssilm, lwilm, upilm, strideilm, oneilm;
          ssilm = subscriptilm[i];
          lw = ADD_LWBD(dtype, i);
          if (lw == 0)
            lw = astb.bnd.one;
          if ((lw == astb.bnd.one && i == 0) ||
              (lw == astb.bnd.zero && !checksubscr)) {
            lwilm = 0;
          } else {
            lw = check_member(ast, lw);
            lower_expression(lw);
            lwilm = lower_ilm(lw);
          }
          lowerboundilm[i] = lwilm;
          if (linilm == 0) {
            /* for rightmost dimension, get 'lin = k' */
            linilm = ssilm;
            if (checksubscr) {
              /* need upperboundilm for checks */
              up = ADD_UPAST(dtype, i);
              if (up == 0) {
                upilm = 0;
              } else {
                up = check_member(ast, up);
                lower_expression(up);
                upilm = lower_ilm(up);
              }
              upperboundilm[i] = upilm;
            }
          } else {
            /* compute '(UP-LO+1)*lin + j' */
            up = ADD_UPAST(dtype, i);
            if (up == 0)
              up = astb.bnd.one;
            up = check_member(ast, up);
            lower_expression(up);
            upilm = lower_ilm(up);
            upperboundilm[i] = upilm;
            if (lw == astb.bnd.one) {
              strideilm = upilm;
            } else {
              if (lw == astb.bnd.zero) {
                strideilm = upilm;
              } else {
                strideilm = plower("oii", lowersym.bnd.sub, upilm, lwilm);
              }
              oneilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
              strideilm = plower("oii", lowersym.bnd.add, strideilm, oneilm);
            }
            linilm = plower("oii", lowersym.bnd.mul, linilm, strideilm);
            linilm = plower("oii", lowersym.bnd.add, linilm, ssilm);
          }
          /* compute 'lin-j0' */
          if (lw != astb.bnd.one || i > 0) {
            if (lw != astb.bnd.zero) {
              linilm = plower("oii", lowersym.bnd.sub, linilm, lwilm);
            }
            if (i == 0) {
              oneilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
              linilm = plower("oii", lowersym.bnd.add, linilm, oneilm);
            }
          }
        }
      } else {
        int descdtype, descddty;
        /* linearized, -x 52 4, do have a descriptor.
         * array bounds and offset may be normalized.
         * multipliers are ok */
        dtype = DTYPEG(sptr);
        dtype = DTY(dtype - 1);
        if (dtype > 0) {
          lerror("unknown linearized datatype");
          return;
        }
        dtype = -dtype;
        descilm = lower_replacement(lop, desc);
        idxilm =
            plower("oS", lowersym.bnd.con, lower_getiszcon(get_xbase_index()));
        descdtype = DTYPEG(desc);
        descddty = DDTG(descdtype);
        linilm = plower("onidi", "ELEMENT", 1, descilm, descdtype, idxilm);
        linilm = lower_typeload(descddty, linilm);
        for (i = 0; i < ndim; ++i) {
          int silm;
          /* (subscript_i)*multiplier_i */
          /* lower_i and multiplier_i are in the descriptor */
          silm = subscriptilm[i];
          if (norm) {
            int lw, lwilm, oneilm;
            /* 2:10 is now 1:9, add original lower bound */
            lw = ADD_LWBD(dtype, i);
            if (lw != 0 && lw != astb.i1) {
              if (lw != astb.bnd.zero) {
                lw = check_member(ast, lw);
                lower_expression(lw);
                lwilm = lower_ilm(lw);
                silm = plower("oii", lowersym.bnd.sub, silm, lwilm);
              }
              oneilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
              silm = plower("oii", lowersym.bnd.add, silm, oneilm);
            }
          }
          if (i == 0 && !POINTERG(sptr)) {
            /* subscript '0' needs no multiplier */
            rilm = silm;
            if (checksubscr) {
              /* needed for bounds checking */
              descilm = lower_replacement(lop, desc);
            }
          } else {
            descilm = lower_replacement(lop, desc);
            idxilm = plower("oS", lowersym.bnd.con,
                            lower_getiszcon(get_multiplier_index(i)));
            rilm = plower("onidi", "ELEMENT", 1, descilm, descdtype, idxilm);
            rilm = lower_typeload(descddty, rilm);
            rilm = plower("oii", lowersym.bnd.mul, silm, rilm);
          }
          linilm = plower("oii", lowersym.bnd.add, linilm, rilm);
          if (checksubscr) {
            int lw, lwilm, oneilm;
            lwilm = 0;
            if (norm) {
              /* 2:10 is now 1:9, add original lower bound */
              lw = ADD_LWBD(dtype, i);
              if (lw != 0 && lw != astb.i1) {
                lw = check_member(ast, lw);
                lower_expression(lw);
                lwilm = lower_ilm(lw);
                oneilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
                lwilm = plower("oii", lowersym.bnd.sub, lwilm, oneilm);
              }
            }
            idxilm = plower("oS", lowersym.bnd.con,
                            lower_getiszcon(get_global_lower_index(i)));
            descdtype = DTYPEG(desc);
            rilm = plower("onidi", "ELEMENT", 1, descilm, descdtype, idxilm);
            rilm = lower_typeload(descddty, rilm);
            if (lwilm) {
              rilm = plower("oii", lowersym.bnd.add, rilm, lwilm);
            }
            lowerboundilm[i] = rilm;
            idxilm = plower("oS", lowersym.bnd.con,
                            lower_getiszcon(get_global_extent_index(i)));
            rilm = plower("onidi", "ELEMENT", 1, descilm, descdtype, idxilm);
            rilm = lower_typeload(descddty, rilm);
            if (lwilm) {
              rilm = plower("oii", lowersym.bnd.add, rilm, lwilm);
            }
            idxilm = plower("oS", lowersym.bnd.con, lower_getiszcon(1));
            rilm = plower("oii", lowersym.bnd.sub, rilm, idxilm);
            rilm = plower("oii", lowersym.bnd.add, rilm, lowerboundilm[i]);
            upperboundilm[i] = rilm;
            if (pointersubscr && !XBIT(58, 0x8000000)) {
              /* undo effects of 'cyclic_section' in commgen.
               * subtract section offset,
               * divide by section stride */
              int strilm, offilm;
              strilm = plower("oS", lowersym.bnd.con,
                              lower_getiszcon(get_section_stride_index(i)));
              strilm =
                  plower("onidi", "ELEMENT", 1, descilm, descdtype, strilm);
              strilm = lower_typeload(descddty, strilm);
              strilm = plower("oii", lowersym.bnd.div, subscriptilm[i], strilm);
              offilm = plower("oS", lowersym.bnd.con,
                              lower_getiszcon(get_section_offset_index(i)));
              offilm =
                  plower("onidi", "ELEMENT", 1, descilm, descdtype, offilm);
              offilm = lower_typeload(descddty, offilm);
              offilm = plower("oii", lowersym.bnd.sub, strilm, offilm);
              subscriptilmx[i] = offilm;
            }
          }
        }
      }
      if (checksubscr) {
        lower_check_subscript(0, ast, ndim, subscriptilmx, lowerboundilm,
                              upperboundilm);
      }
      ndim = 1;
      subscriptilm[0] = linilm;
    } else {
      int desc;
      int arrparam; /* array parameter */
      int checkit;
      arrparam = 0;
      checkit = 0;
      desc = SDSCG(sptr);
      if (sptr && checksubscr) {
        if (!HCCSYMG(sptr))
          checkit = 1;
        else if (PARAMG(sptr)) {
          arrparam = A_SPTRG(PARAMVALG(sptr));
          if (arrparam)
            checkit = 1;
        }
      }
#ifdef CUDAG
      if (CUDAG(GBL_CURRFUNC) & (CUDA_DEVICE | CUDA_GLOBAL)) {
        checkit = 0;
      }
#endif
      if (checkit) {
        /* fill in the bounds for checking */
        if (desc == 0 || NODESCG(sptr) || !DESCUSEDG(sptr) || ASUMSZG(sptr) ||
            STYPEG(desc) == ST_PARAM || ASSUMSHPG(sptr)) {
          int dtype;
          dtype = DTYPEG(sptr);
          for (i = ndim - 1; i >= 0; --i) {
            int lw, up, lwilm, upilm;
            lw = ADD_LWAST(dtype, i);
            if (lw == 0)
              lw = astb.bnd.one;
            if (lw == astb.bnd.one) {
              lwilm = 0;
            } else {
              lw = check_member(ast, lw);
              lower_expression(lw);
              lwilm = lower_ilm(lw);
            }
            lowerboundilm[i] = lwilm;
            /* need upperboundilm for checks */
            up = ADD_UPAST(dtype, i);
            if (up == 0) {
              upilm = 0;
            } else {
              up = check_member(ast, up);
              lower_expression(up);
              upilm = lower_ilm(up);
            }
            upperboundilm[i] = upilm;
          }
        } else {
          int descilm, idxilm, dtype, rilm, descdtype, descddty;
          /* array bounds in descriptor may be normalized */
          dtype = DTYPEG(sptr);
          descilm = lower_replacement(lop, desc);
          for (i = 0; i < ndim; ++i) {
            int lw, lwilm, oneilm;
            lwilm = 0;
            if (norm) {
              /* 2:10 is now 1:9, add original lower bound */
              lw = ADD_LWBD(dtype, i);
              if (lw != 0 && lw != astb.i1) {
                lw = check_member(ast, lw);
                lower_expression(lw);
                lwilm = lower_ilm(lw);
                oneilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
                lwilm = plower("oii", lowersym.bnd.sub, lwilm, oneilm);
              }
            }
            idxilm = plower("oS", lowersym.bnd.con,
                            lower_getiszcon(get_global_lower_index(i)));
            descdtype = DTYPEG(desc);
            descddty = DDTG(descdtype);
            rilm = plower("onidi", "ELEMENT", 1, descilm, descdtype, idxilm);
            rilm = lower_typeload(descddty, rilm);
            if (lwilm) {
              rilm = plower("oii", lowersym.bnd.add, rilm, lwilm);
            }
            lowerboundilm[i] = rilm;
            idxilm = plower("oS", lowersym.bnd.con,
                            lower_getiszcon(get_global_extent_index(i)));
            rilm = plower("onidi", "ELEMENT", 1, descilm, descdtype, idxilm);
            rilm = lower_typeload(descddty, rilm);
            if (lwilm) {
              rilm = plower("oii", lowersym.bnd.add, rilm, lwilm);
            }
            idxilm = plower("oS", lowersym.bnd.con, lower_getiszcon(1));
            rilm = plower("oii", lowersym.bnd.sub, rilm, idxilm);
            rilm = plower("oii", lowersym.bnd.add, rilm, lowerboundilm[i]);
            upperboundilm[i] = rilm;
            if (pointersubscr && !XBIT(58, 0x8000000)) {
              /* undo effects of 'cyclic_section' in commgen.
               * subtract section offset,
               * divide by section stride */
              int strilm, offilm;
              strilm = plower("oS", lowersym.bnd.con,
                              lower_getiszcon(get_section_stride_index(i)));
              strilm =
                  plower("onidi", "ELEMENT", 1, descilm, descdtype, strilm);
              strilm = lower_typeload(descddty, strilm);
              strilm = plower("oii", lowersym.bnd.div, subscriptilm[i], strilm);
              offilm = plower("oS", lowersym.bnd.con,
                              lower_getiszcon(get_section_offset_index(i)));
              offilm =
                  plower("onidi", "ELEMENT", 1, descilm, descdtype, offilm);
              offilm = lower_typeload(descddty, offilm);
              offilm = plower("oii", lowersym.bnd.sub, strilm, offilm);
              subscriptilmx[i] = offilm;
            }
          }
        }
        lower_check_subscript(arrparam, ast, ndim, subscriptilmx, lowerboundilm,
                              upperboundilm);
      }
      if (norm && desc != 0 && !NODESCG(sptr) && DESCUSEDG(sptr) &&
          STYPEG(desc) != ST_PARAM) {
        int dtype;
        /* subtract off original lower bound
         * 2:10 is now 1:9, subtract original lower bound */
        dtype = DTYPEG(sptr);
        for (i = 0; i < ndim; ++i) {
          int lw, lwilm, oneilm;
          lw = ADD_LWBD(dtype, i);
          rilm = subscriptilm[i];
          if (lw != 0 && lw != astb.i1) {
            if (lw != astb.bnd.zero) {
              lw = check_member(ast, lw);
              lower_expression(lw);
              lwilm = lower_ilm(lw);
              rilm = plower("oii", lowersym.bnd.sub, rilm, lwilm);
            }
            oneilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
            rilm = plower("oii", lowersym.bnd.add, rilm, oneilm);
          }
          subscriptilm[i] = rilm;
        }
      }
    }
    base = plower("onidm", "ELEMENT", ndim, base, A_NDTYPEG(A_LOPG(ast)));
    for (i = 0; i < ndim; ++i) {
      plower("im", subscriptilm[i]);
    }
    plower("e");
    break;

  case A_SUBSTR:
    ilm = lower_base(A_LOPG(ast));
    if (A_LEFTG(ast)) {
      lilm = lower_ilm(A_LEFTG(ast));
    } else {
      lilm = plower("oS", "ICON", lowersym.intone);
    }
    if (A_RIGHTG(ast)) {
      rilm = lower_ilm(A_RIGHTG(ast));
    } else {
      int len;
      int lop = A_LOPG(ast);
      len = DTY(A_NDTYPEG(lop) + 1); /* char string length */
      if (len && A_ALIASG(len)) {
        len = A_ALIASG(len);
        len = A_SPTRG(len);
        rilm = plower("oS", "ICON", len); /* ilm */
      } else {
        /* assumed length string, use 'len' function */
        rilm = plower("oi", "LEN", ilm);
      }
    }
    if (DTY(A_NDTYPEG(ast)) == TY_NCHAR) {
      ilm = plower("oiii", "NSUBS", ilm, lilm, rilm);
    } else {
      ilm = plower("oiii", "SUBS", ilm, lilm, rilm);
    }
    base = ilm;
    break;

  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_NEG:
    case OP_SUB:
      ilm = lower_un_arith(ast, "NEG", dtype);
      base = ilm;
      break;
    case OP_LNOT:
      if (XBIT(125, 0x8))
        ilm = lower_un_logical(ast, "UXLNOT");
      else
        ilm = lower_un_logical(ast, "LNOT");
      base = ilm;
      break;
    case OP_ADD:
      ilm = lower_ilm(A_LOPG(ast));
      base = ilm;
      break;
    case OP_LOC: {
      /* use LOC operator */
      if (A_LOPG(ast) == astb.ptr0) {
        ilm = lower_null();
      } else if (A_LOPG(ast) == astb.ptr0c) {
        ilm = lower_null();
      } else {
        ilm = lower_base(A_LOPG(ast));
        ilm = plower("oi", "LOC", ilm);
        ilm = lower_conv_ilm(ast, ilm, DT_PTR, A_NDTYPEG(ast));
      }
      base = ilm;
    } break;
    case OP_VAL:
      if (ast == astb.ptr0) {
        ilm = base = lower_null_arg();
      } else if (ast == astb.ptr0c) {
        ilm = base = lower_nullc_arg();
      } else {
        ilm = lower_ilm(A_LOPG(ast));
        ilm = plower("oi", "DPVAL", ilm);
        base = ilm;
      }
      break;
    case OP_BYVAL:
      dtype = A_DTYPEG(A_LOPG(ast));
      ilm = lower_ilm(A_LOPG(ast));
      ilm = plower("oid", "BYVAL", ilm, dtype);
      base = ilm;
      break;
    case OP_REF:
      ilm = lower_ilm(A_LOPG(ast));
      ilm = plower("oi", "DPREF", ilm);
      base = ilm;
      break;
    default:
      ast_error("don't know how to handle type unary operator", ast);
      break;
    }
    break;

  case A_COMMENT:
  case A_COMSTR:
    /* ignore comments */
    break;
  case A_MP_ATOMICREAD:
    ilm = lower_base(A_SRCG(ast));
    i = 0;
    ilm = plower("oin", "MP_ATOMICREAD", ilm, A_MEM_ORDERG(ast));
    base = ilm;
    break;
    /* ------------- unsupported AST types ------------- */

  case A_ATOMIC:
  case A_ATOMICCAPTURE:
  case A_ATOMICREAD:
  case A_ATOMICWRITE:
  case A_BARRIER:
  case A_CRITICAL:
  case A_ELSEFORALL:
  case A_ELSEWHERE:
  case A_ENDATOMIC:
  case A_ENDCRITICAL:
  case A_ENDFORALL:
  case A_ENDMASTER:
  case A_ENDWHERE:
  case A_FORALL:
  case A_MASTER:
  case A_NOBARRIER:
  case A_TRIPLE:
  case A_WHERE:
  case A_MP_PARALLEL:
  case A_MP_ENDPARALLEL:
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_BARRIER:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_PDO:
  case A_MP_ENDPDO:
  case A_MP_SECTIONS:
  case A_MP_ENDSECTIONS:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_PRE_TLS_COPY:
  case A_MP_BCOPYIN:
  case A_MP_COPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_COPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
  case A_MP_TASK:
  case A_MP_TASKLOOP:
  case A_MP_TASKFIRSTPRIV:
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_ETASKDUP:
  case A_MP_TASKLOOPREG:
  case A_MP_ETASKLOOPREG:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
  case A_MP_BMPSCOPE:
  case A_MP_EMPSCOPE:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_PREFETCH:
  case A_PRAGMA:
  case A_MP_TARGET:
  case A_MP_ENDTARGET:
  case A_MP_TEAMS:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_TARGETUPDATE:
  case A_MP_TARGETDATA:
  case A_MP_ENDTARGETDATA:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_CANCEL:
  case A_MP_CANCELLATIONPOINT:
  case A_MP_ATOMICWRITE:
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
  default:
    ast_error("bad ast optype in expression", ast);
    break;
  }
  A_ILMP(ast, ilm);
  A_BASEP(ast, base);
} /* lower_ast */

int
lower_null(void)
{
  int ilm;

  if (!XBIT(49, 0x20000000)) {
    ilm = plower("oS", "ACON", lowersym.ptrnull);
  } else if (XBIT(49, 0x100)) {
    ilm = plower("oS", "KCON", lowersym.intzero);
  } else {
    ilm = plower("oS", "ICON", lowersym.intzero);
  }
  return ilm;
}

int
lower_null_arg(void)
{
  int ilm;
  ilm = lowersym_pghpf_cmem(&lowersym.ptr0);
  if (!XBIT(57, 0x8000))
    ilm = plower("oi", "DPVAL", ilm);
  return ilm;
}

int
lower_nullc_arg(void)
{
  int ilm;
  ilm = lowersym_pghpf_cmem(&lowersym.ptr0c);
  if (!XBIT(57, 0x8000))
    ilm = plower("o", "DPNULL");
  return ilm;
}

/*
 *  raising an operand to a constant power >= 1.  generate ILMs which
 *  maximize cse's (i.e., generate a balanced tree).
 *  opn -  operand (ILM) raised to power 'pwr'
 *  pwr -  power (constant)
 *  opc -  mult ILM opcode
 */
static int
_xtoi(int opn, int pwr, char *opc)
{
  int res;
  int p2; /* largest power of 2 such that 2**p2 <= opn**pwr */
  int n;

  if (pwr >= 2) {
    p2 = 0;
    n = pwr;
    while ((n >>= 1) > 0)
      p2++;

    n = 1 << p2; /* 2**p2 */
    res = opn;
    /* generate a balanced multiply tree whose height is p2 */
    while (p2-- > 0)
      res = plower("oii", opc, res, res);

    /* residual */
    n = pwr - n;
    if (n > 0) {
      int right;
      right = _xtoi(opn, n, opc);
      res = plower("oii", opc, res, right);
    }

    return res;
  }
  return opn;
}

static int
lower_logical_expr(int ast)
{
  int ilm;
  switch (A_NDTYPEG(ast)) {
  case TY_BINT:
  case TY_BLOG:
  case TY_SINT:
  case TY_SLOG:
  case TY_INT:
  case TY_LOG:
  case TY_INT8:
  case TY_LOG8:
    ilm = lower_ilm(ast);
    break;
  default:
    ilm = conv_int(ast);
    break;
  }
  return ilm;
} /* lower_logical_expr */

void
lower_logical(int ast, iflabeltype *iflabp)
{
  int dtype, lop, ilm = 0, ilm2;
  int sptr;
  iflabeltype nlab;

  dtype = A_DTYPEG(ast);
  A_NDTYPEP(ast, dtype);

  switch (A_TYPEG(ast)) {
  case A_NULL:
    break;
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_LAND:
    case OP_SCAND:
      if (iflabp->thenlabel == 0) {
        /* The incoming fall-through case is 'then'.
         *  brfalse(left) elselabel
         *  brfalse(right) elselabel */
        lower_logical(A_LOPG(ast), iflabp);
        lower_logical(A_ROPG(ast), iflabp);
      } else {
        /* The incoming fall-through case is 'else'.
         *  brfalse(left) newlabel
         *  brtrue(right) thenlabel
         *  newlabel: */
        nlab.thenlabel = 0;
        nlab.elselabel = lower_lab();
        lower_logical(A_LOPG(ast), &nlab);
        /* second operand can fall through if true, branch around if false */
        lower_logical(A_ROPG(ast), iflabp);
        plower("oL", "LABEL", nlab.elselabel);
        lower_reinit();
      }
      break;
    case OP_LOR:
      if (iflabp->thenlabel == 0) {
        /* The incoming fall-through case is 'then'.
         *  brtrue(left) newlabel
         *  brfalse(right) elselabel
         *  newlabel: */
        nlab.thenlabel = lower_lab();
        nlab.elselabel = 0;
        lower_logical(A_LOPG(ast), &nlab);
        /* second operand can fall through if true, branch around if false */
        lower_logical(A_ROPG(ast), iflabp);
        plower("oL", "LABEL", nlab.thenlabel);
        lower_reinit();
      } else {
        /* The incoming fall-through case is 'else'.
         *  brtrue(left) thenlabel
         *  brtrue(right) thenlabel */
        lower_logical(A_LOPG(ast), iflabp);
        lower_logical(A_ROPG(ast), iflabp);
      }
      break;
    case OP_LEQV:
      lower_expression(A_LOPG(ast));
      lower_expression(A_ROPG(ast));
      if (XBIT(125, 0x8))
        ilm = lower_bin_logical(ast, "UXLEQV");
      else
        ilm = lower_bin_logical(ast, "LEQV");
      if (iflabp->thenlabel) {
        plower("oiS", "BRT", ilm, iflabp->thenlabel);
      } else {
        plower("oiS", "BRF", ilm, iflabp->elselabel);
      }
      break;
    case OP_LNEQV:
      lower_expression(A_LOPG(ast));
      lower_expression(A_ROPG(ast));
      if (XBIT(125, 0x8))
        ilm = lower_bin_logical(ast, "UXLNEQV");
      else
        ilm = lower_bin_logical(ast, "LNEQV");
      if (iflabp->thenlabel) {
        plower("oiS", "BRT", ilm, iflabp->thenlabel);
      } else {
        plower("oiS", "BRF", ilm, iflabp->elselabel);
      }
      break;
    case OP_EQ:
    case OP_GE:
    case OP_GT:
    case OP_LE:
    case OP_LT:
    case OP_NE:
      lower_expression(ast);
      ilm = A_ILMG(ast);
      if (iflabp->thenlabel) {
        plower("oiS", "BRT", ilm, iflabp->thenlabel);
      } else {
        plower("oiS", "BRF", ilm, iflabp->elselabel);
      }
      break;
    default:
      lower_expression(ast);
      ilm = lower_logical_expr(ast);
      if (iflabp->thenlabel) {
        plower("oiS", "BRT", ilm, iflabp->thenlabel);
      } else {
        plower("oiS", "BRF", ilm, iflabp->elselabel);
      }
      break;
    }
    break;

  case A_CMPLXC:
    lower_expression(ast);
    ilm = A_ILMG(ast);
    ilm2 = plower("oS", "ICON", lowersym.intzero);
    ilm2 = lower_conv_ilm(ast, ilm, DT_INT4, A_NDTYPEG(ast));
    ilm = plower("oii", ltyped("CMP", A_NDTYPEG(ast)), ilm, ilm2);
    if (iflabp->thenlabel) {
      plower("oiS", "BRT", ilm, iflabp->thenlabel);
    } else {
      plower("oiS", "BRF", ilm, iflabp->elselabel);
    }
    break;

  case A_CNST:
    if (dtype <= 0) {
      ast_error("unrecognized data type", ast);
      break;
    }
    sptr = A_SPTRG(ast);
    lower_visit_symbol(sptr);
    switch (DTYG(dtype)) {
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_INT8:
    case TY_REAL:
    case TY_DBLE:
    case TY_CMPLX:
    case TY_DCMPLX:
      lower_expression(ast);
      ilm = A_ILMG(ast);
      ilm2 = plower("oS", "ICON", lowersym.intzero);
      ilm2 = lower_conv_ilm(ast, ilm, DT_INT4, A_NDTYPEG(ast));
      ilm = plower("oii", ltyped("CMP", A_NDTYPEG(ast)), ilm, ilm2);
      if (iflabp->thenlabel) {
        plower("oiS", "BRT", ilm, iflabp->thenlabel);
      } else {
        plower("oiS", "BRF", ilm, iflabp->elselabel);
      }
      break;
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
    case TY_LOG8:
      /* is it true or false? */
      if (CONVAL2G(A_SPTRG(ast)) == 0) {
        /* False: branch to false label or fall through */
        if (iflabp->elselabel) {
          plower("oS", "BR", iflabp->elselabel);
        }
      } else {
        /* True: branch to true label or fall through */
        if (iflabp->thenlabel) {
          plower("oS", "BR", iflabp->thenlabel);
        }
      }
      break;
    case TY_CHAR:
    case TY_NCHAR:
      ast_error("unrecognized char", ast);
      break;
    }
    break;

  case A_CONV:
    lop = A_LOPG(ast);
    switch (A_DTYPEG(lop)) {
    case DT_LOG4:
    case DT_LOG8:
      lower_logical(lop, iflabp);
      break;
    default:
      lower_expression(ast);
      ilm = lower_logical_expr(ast);
      if (iflabp->thenlabel) {
        plower("oiS", "BRT", ilm, iflabp->thenlabel);
      } else {
        plower("oiS", "BRF", ilm, iflabp->elselabel);
      }
    }
    break;

  case A_INTR:
    lower_expression(ast);
    ilm = lower_logical_expr(ast);
    if (iflabp->thenlabel) {
      plower("oiS", "BRT", ilm, iflabp->thenlabel);
    } else {
      plower("oiS", "BRF", ilm, iflabp->elselabel);
    }
    break;

  case A_INIT:
    ast_error("unexpected AST type", ast);
    break;

  case A_ID:
  case A_MEM: /* member */
  case A_SUBSCR:
  case A_FUNC:
    lower_expression(ast);
    ilm = lower_logical_expr(ast);
    if (iflabp->thenlabel) {
      plower("oiS", "BRT", ilm, iflabp->thenlabel);
    } else {
      plower("oiS", "BRF", ilm, iflabp->elselabel);
    }
    break;

  case A_LABEL:
    lower_expression(ast);
    break;

  case A_PAREN:
    lower_logical(A_LOPG(ast), iflabp);
    break;

  case A_SUBSTR:
    lower_expression(ast);
    break;

  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_LNOT:
      nlab.thenlabel = iflabp->elselabel;
      nlab.elselabel = iflabp->thenlabel;
      lower_logical(A_LOPG(ast), &nlab);
      break;
    default:
      lower_expression(ast);
      ilm = lower_logical_expr(ast);
      if (iflabp->thenlabel) {
        plower("oiS", "BRT", ilm, iflabp->thenlabel);
      } else {
        plower("oiS", "BRF", ilm, iflabp->elselabel);
      }
      break;
    }
    break;

    /* ------------- unsupported AST types ------------- */

  case A_ATOMIC:
  case A_BARRIER:
  case A_COMMENT:
  case A_COMSTR:
  case A_CRITICAL:
  case A_ELSEFORALL:
  case A_ELSEWHERE:
  case A_ENDATOMIC:
  case A_ENDCRITICAL:
  case A_ENDFORALL:
  case A_ENDMASTER:
  case A_ENDWHERE:
  case A_FORALL:
  case A_MASTER:
  case A_NOBARRIER:
  case A_TRIPLE:
  case A_WHERE:
  default:
    ast_error("bad ast optype in logical expression", ast);
    break;
  }
} /* lower_logical */

/* Called for A_FUNC or A_INTR when no subscript checking should be done
 * on the arguments.  Must be called during preorder traversal so we can
 * set lower_disable_subscr_chk before subscripting is evaluated.
 */
static void
lower_no_subscr_chk(int ast, int *unused)
{
  int cnt, argt, i;
  int save_disable_subscr_chk;

  save_disable_subscr_chk = lower_disable_subscr_chk;
  lower_disable_subscr_chk = 1;
  ast_traverse((int)A_LOPG(ast), lower_check_ast, lower_ast, NULL);
  cnt = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);
  for (i = 0; i < cnt; i++)
    /* watch for optional args */
    if (ARGT_ARG(argt, i) != 0)
      ast_traverse(ARGT_ARG(argt, i), lower_check_ast, lower_ast, NULL);
  lower_ast(ast, unused);
  lower_disable_subscr_chk = save_disable_subscr_chk;
}

static LOGICAL
lower_check_ast(int ast, int *unused)
{
  int argt, shape, i, ndim;
  int symfunc;

  /* return TRUE to not recurse below here */
  switch (A_TYPEG(ast)) {
  case A_FUNC:
    symfunc = memsym_of_ast(A_LOPG(ast));
    if (strcmp(SYMNAME(symfunc), mkRteRtnNm(RTE_lena)) == 0) {
      /* Disable subscript checking for LEN */
      lower_no_subscr_chk(ast, unused);
      return TRUE;
    }
    break;
  case A_INTR:
    switch (A_OPTYPEG(ast)) {
    case I_LEN:
      /* Disable subscript checking for LEN */
      lower_no_subscr_chk(ast, unused);
      return TRUE;
    case I_SIZE:
      /* for the 'size' intrinsic, we use the shape, not the
       * arguments */
      lower_ast(ast, unused);
      return TRUE;
    case I_UBOUND:
    case I_LBOUND:
      /* use shape of first argument */
      argt = A_ARGSG(ast);
      ast_traverse(ARGT_ARG(argt, 1), lower_check_ast, lower_ast, NULL);
      shape = A_SHAPEG(ARGT_ARG(argt, 0));
      ndim = SHD_NDIM(shape);
      for (i = 0; i < ndim; ++i) {
        if (SHD_LWB(shape, i)) {
          ast_traverse(SHD_LWB(shape, i), lower_check_ast, lower_ast, NULL);
        }
        if (SHD_UPB(shape, i)) {
          ast_traverse(SHD_UPB(shape, i), lower_check_ast, lower_ast, NULL);
        }
      }
      lower_ast(ast, unused);
      return TRUE;
    }
    break;
  }
  return FALSE;
} /* lower_check_ast */

/** \brief Use ast_traverse to lower the expression asts. */
void
lower_expression(int ast)
{
  ast_traverse(ast, lower_check_ast, lower_ast, NULL);
} /* lower_expression */

void
lower_reinit(void)
{
  ast_revisit(lower_clear_opt, 0);
  ast_unvisit_norepl();
  ast_visit(1, 1);
} /* lower_reinit */

void
lower_exp_finish(void)
{
  if (intr_argsz > IARGS) {
    FREE(intrinsic_args);
    intrinsic_args = intr_argbf;
    intr_argsz = IARGS;
  }
}
