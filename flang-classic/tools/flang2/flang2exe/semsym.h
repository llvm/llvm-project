/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef SEMSYM_H_
#define SEMSYM_H_

#include "gbldefs.h"
#include "symtab.h"

/**
   \brief Look up symbol having a specific symbol type.

   If a symbol is found in the same overloading class and has the same
   symbol type, it is returned to the caller.  If a symbol is found in
   the same overloading class, the action of declref depends on the
   stype of the existing symbol and value of the argument def:
   1.  if symbol is an unfrozen intrinsic and def is 'd' (define), its
       intrinsic property is removed and a new symbol is declared,
   2.  if def is 'd', a multiple declaration error occurs, or
   3.  if def is not 'd', an 'illegal use' error occurs.

   If an error occurs or a matching symbol is not found, one is
   created and its symbol type is initialized.
 */
SPTR declref(SPTR sptr, SYMTYPE stype, int def);

/**
   \brief Declare a new symbol.

   An error can occur if the symbol is already in the symbol table:
   - if the symbol types match treat as an error if 'errflg' is true
     otherwise its okay and return symbol to caller
   - else if symbol is an intrinsic attempt to remove symbol's
     intrinsic property otherwise it is an error
 */
SPTR declsym(SPTR sptr, SYMTYPE stype, bool errflg);

/**
   \brief Look up symbol matching overloading class of given symbol
   type.

   The sptr is returned for the symbol whose overloading class matches
   the overloading class of the symbol type given.  If no symbol is
   found in the given overloading class one is created.  (If scoping
   becomes a Fortran feature, this routine will not use it)
 */
SPTR getocsym(SPTR sptr, int oclass);

/**
   \brief Reset fields of intrinsic or generic symbol, sptr, to zero
   in preparation for changing its symbol type by the Semantic
   Analyzer. If the symbol type of the symbol has been 'frozen', issue
   an error message and notify the caller by returning a zero symbol
   pointer.
 */
SPTR newsym(SPTR sptr);

/**
   \brief Reference a symbol when it's known the context requires an
   identifier.  If an error occurs (e.g., symbol which is frozen as an
   intrinsic), a new symbol is created so that processing can
   continue.  If the symbol found is ST_UNKNOWN, its stype is changed
   to ST_IDENT.
 */
SPTR ref_ident(SPTR sptr);

/**
   \brief Look up a symbol having the given overloading class.

   If the symbol with the overloading class is found its sptr is
   returned.  If no symbol with the given overloading class is found,
   a new sptr is returned.  (If scoping becomes a Fortran feature,
   this routine will implement it)
 */
SPTR refsym(SPTR sptr, int oclass);

#endif // SEMSYM_H_
