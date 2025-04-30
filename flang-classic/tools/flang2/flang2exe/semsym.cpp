/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief FTN Semantic action routines to resolve symbol references as
   to overloading class.  This module hides the walking of hash chains
   and overloading class checks.
 */

#include "semsym.h"
#include "error.h"
#include "global.h"
#include "semant.h"
#include "xref.h"

SPTR
declref(SPTR sptr, SYMTYPE stype, int def)
{
  SYMTYPE st;
  SPTR sptr1;
  SPTR first;

  first = sptr;
  do {
    st = STYPEG(sptr);
    if (st == ST_UNKNOWN)
      goto return1; /* stype not set yet, set it */
    if (stb.ovclass[st] == stb.ovclass[stype]) {
      if (stype != st) {
        if (def == 'd') {
          /* Redeclare of intrinsic symbol is okay unless frozen */
          if (IS_INTRINSIC(st)) {
            sptr1 = newsym(sptr);
            if (sptr1 != SPTR_NULL)
              sptr = sptr1;
            goto return1;
          }
          /* multiple declaration */
          error(S_0044_Multiple_declaration_for_symbol_OP1, ERR_Severe,
                gbl.lineno, SYMNAME(first), CNULL);
        } else
          /* illegal use of symbol */
          error(S_0084_Illegal_use_of_symbol_OP1_OP2, ERR_Severe, gbl.lineno,
                SYMNAME(first), CNULL);
        break;
      }
      goto return2; /* found, return it */
    }
    sptr = HASHLKG(sptr);
  } while (sptr && NMPTRG(sptr) == NMPTRG(first));

  /* create new one if def or illegal use */
  sptr = insert_sym(first);
return1:
  STYPEP(sptr, stype);
return2:
  if (flg.xref)
    xrefput(sptr, def);
  return sptr;
}

SPTR
declsym(SPTR sptr, SYMTYPE stype, bool errflg)
{
  SYMTYPE st;
  SPTR sptr1, first;

  first = sptr;
  do {
    st = STYPEG(sptr);
    if (st == ST_UNKNOWN)
      goto return1; /* Brand new symbol, return it. */
    if (st == ST_IDENT && stb.ovclass[st] == stb.ovclass[stype])
      goto return1; /* Found sym in same overloading class */
    if (stb.ovclass[st] == stb.ovclass[stype]) {
      if (stype == st) {
        /* Possible attempt to multiply define symbol */
        if (errflg) {
          error(S_0044_Multiple_declaration_for_symbol_OP1, ERR_Severe,
                gbl.lineno, SYMNAME(first), CNULL);
          break;
        } else
          goto return2;
      } else {
        /* Redeclare of intrinsic symbol is okay unless frozen */
        if (IS_INTRINSIC(st)) {
          if ((sptr1 = newsym(sptr)) != 0)
            sptr = sptr1;
          goto return1;
        } else {
          error(S_0043_Illegal_attempt_to_redefine_OP1_OP2, ERR_Severe,
                gbl.lineno, "symbol", SYMNAME(first));
          break;
        }
      }
    }
    sptr = HASHLKG(sptr);
  } while (sptr && NMPTRG(sptr) == NMPTRG(first));

  /* create new one if def or illegal use */
  sptr = insert_sym(first);
return1:
  STYPEP(sptr, stype);
return2:
  if (flg.xref)
    xrefput(sptr, 'd');
  return sptr;
}

SPTR
refsym(SPTR sptr, int oclass)
{
  int st;
  SPTR first;

  first = sptr;
  do {
    st = STYPEG(sptr);
    if (st == ST_UNKNOWN)
      goto return1;
    if (stb.ovclass[st] == oclass)
      goto returnit;
    sptr = HASHLKG(sptr);
  } while (sptr && NMPTRG(sptr) == NMPTRG(first));

  /* Symbol in given overloading class not found, create new one */
  sptr = insert_sym(first);
return1:
  if (flg.xref)
    xrefput(sptr, 'd');
returnit:
  return sptr;
}

SPTR
getocsym(SPTR sptr, int oclass)
{
  int st;
  SPTR first;

  first = sptr;
  do {
    st = STYPEG(sptr);
    if (st == ST_UNKNOWN)
      goto return1;
    if (stb.ovclass[st] == oclass)
      goto returnit; /* found it! */
    sptr = HASHLKG(sptr);
  } while (sptr && NMPTRG(sptr) == NMPTRG(first));

  /* create new symbol if undefined or illegal use */
  sptr = insert_sym(first);
return1:
  if (flg.xref)
    xrefput(sptr, 'd');
returnit:
  return sptr;
}

SPTR
newsym(SPTR sptr)
{
  SPTR sp2;

  if (EXPSTG(sptr)) {
    /* Symbol previously frozen as an intrinsic */
    error(S_0043_Illegal_attempt_to_redefine_OP1_OP2, ERR_Severe, gbl.lineno,
          "intrinsic", SYMNAME(sptr));
    return SPTR_NULL;
  }
  /*
   * try to find another sym in the same overloading class; we need to
   * try this first since there could be multiple occurrences of an
   * intrinsic and therefore the sptr appears more than once in the
   * semantic stack.  E.g.,
   *    call sub (sin, sin)
   * NOTE that in order for this to work we need to perform another getsym
   * to start at the beginning of the hash links for symbols whose names
   * are the same.
   */
  sp2 = getsym(LOCAL_SYMNAME(sptr), strlen(SYMNAME(sptr)));
  sp2 = getocsym(sp2, OC_OTHER);
  if (sp2 != sptr)
    return sp2;
  /*
   * create a new symbol with the same name:
   */
  error(I_0035_Predefined_intrinsic_OP1_loses_intrinsic_property,
        ERR_Informational, gbl.lineno, SYMNAME(sptr), CNULL);
  sp2 = insert_sym(sptr);

  /* transfer dtype if it was explicitly declared for sptr:  */

  if (DCLDG(sptr)) {
    DTYPEP(sp2, DTYPEG(sptr));
    DCLDP(sp2, 1);
  }

  return sp2;
}

SPTR
ref_ident(SPTR sptr)
{
  SPTR sym;

  sym = refsym(sptr, OC_OTHER);
  if (IS_INTRINSIC(STYPEG(sym))) {
    sym = newsym(sym);
    if (sym == 0)
      sym = insert_sym(sptr);
  }
  if (STYPEG(sym) == ST_UNKNOWN)
    STYPEP(sym, ST_IDENT);

  return sym;
}

