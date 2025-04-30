/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief Add array bounds and null pointer checks.
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "symutl.h"
#include "ast.h"

#define INSIDE_LOWER
#include "lower.h"

#define LEN 128
static char Name[LEN + 2];
static int Len;

static int fill_name(int ast);

static void
fill_recurse(int ast)
{
  char *s;
  int l;
  switch (A_TYPEG(ast)) {
  case A_ID:
    s = SYMNAME(A_SPTRG(ast));
    l = strlen(s);
    if (Len + l < LEN) {
      strcpy(Name + Len, s);
      Len += l;
    }
    break;
  case A_MEM:
    fill_name(A_LOPG(ast));
    s = SYMNAME(A_SPTRG(A_MEMG(ast)));
    l = strlen(s);
    if (Len + l + 1 < LEN) {
      strcpy(Name + Len, "%");
      strcpy(Name + Len + 1, s);
      Len += l + 1;
    }
    break;
  case A_SUBSCR:
    fill_name(A_LOPG(ast));
    break;
  case A_SUBSTR:
    fill_name(A_LOPG(ast));
    break;
  default:
    ast_error("unknown ast type for checks", ast);
    break;
  }
} /* fill_recurse */

static int
fill_name(int ast)
{
  int sname;
  Len = 0;
  Name[0] = '\0';
  fill_recurse(ast);
  Name[Len] = '\0';
  Name[Len + 1] = '\0';
  sname = getstring(Name, Len + 1);
  return sname;
} /* fill_name */

/** \brief Generate a pointer check.

    Generate a call:
    <pre>
        ftn_ptrchk(%val(pointer),%val(lineno),%ref("varname"),%ref("filename"))
    </pre>
 */
void
lower_check_pointer(int ast, int ilm)
{
  int sname, ilm1, ilmline, ilmname, ilmfile;

  /* make up the symbol name, lineno, filename */
  sname = fill_name(ast);
  ilmline = plower("oS", "ICON", lower_getintcon(lower_line));
  ilmline = plower("oi", "DPVAL", ilmline);
  ilmname = plower("oS", "BASE", sname);
  ilmfile = plower("oS", "BASE", lowersym.sym_chkfile);

  /* make up the call */
  ilm1 = lower_typeload(DT_ADDR, ilm);
  ilm1 = plower("oi", "DPVAL", ilm1);
  plower("onsiiiiC", "CALL", 4, lowersym.sym_ptrchk, ilm1, ilmline, ilmname,
         ilmfile, lowersym.sym_ptrchk);
} /* lower_check_pointer */

/** \brief Generate a subscript check.

    Generate a call for each subscript
    <pre>
        ftn_ptrchk(%val(subscript),%val(lower),%val(upper),%val(dimension),
                   %val(lineno),%ref("varname"),%ref("filename"))
    </pre>

    \a ast is a subscript ast

    If \a sym is zero,  derive the name of the array from the subscript ast;
    otherwise, use the name of sym.
 */

void
lower_check_subscript(int sym, int ast, int ndim, int *ilm, int *lower,
                      int *upper)
{
  int i, sname, ilm1, ilm2, ilm3, ilm4, ilmline, ilmname, ilmfile;

  /* make up the symbol name, lineno, filename */
  if (sym == 0)
    sname = fill_name(ast);
  else {
    char *s;
    s = SYMNAME(sym);
    Len = strlen(s);
    if (Len > LEN)
      Len = LEN;
    strncpy(Name, s, Len);
    Name[Len] = '\0';
    Name[Len + 1] = '\0';
    sname = getstring(Name, Len + 1);
  }
  ilmline = plower("oS", "ICON", lower_getintcon(lower_line));
  ilmline = plower("oi", "DPVAL", ilmline);
  ilmname = plower("oS", "BASE", sname);
  ilmfile = plower("oS", "BASE", lowersym.sym_chkfile);

  for (i = 0; i < ndim; ++i) {
    /* make up the call */
    ilm1 = plower("oi", "DPVAL", ilm[i]);
    if (lower[i] == 0) {
      lower[i] = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
    }
    ilm2 = plower("oi", "DPVAL", lower[i]);
    if (upper[i] == 0) {
      upper[i] = plower("oS", lowersym.bnd.con, lowersym.bnd.max);
    }
    ilm3 = plower("oi", "DPVAL", upper[i]);
    ilm4 = plower("oS", "ICON", lower_getintcon(i + 1));
    ilm4 = plower("oi", "DPVAL", ilm4);
    plower("onsiiiiiiiC", "CALL", 7, lowersym.sym_subchk, ilm1, ilm2, ilm3,
           ilm4, ilmline, ilmname, ilmfile, lowersym.sym_subchk);
  }
} /* lower_check_subscript */
