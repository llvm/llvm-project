/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef NMEUTIL_H_
#define NMEUTIL_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "nme.h"
#include <stdio.h>

/**
   \brief ...
 */
bool basenme_is_static(int nme);

/**
   \brief ...
 */
bool is_presym(int nme);

/**
   \brief ...
 */
bool is_smove_member(int nme);

/**
   \brief ...
 */
DTYPE dt_nme(int nm);

/**
   \brief Main add NME routine

   Enter nme into the NME area.

<pre>
  the type NT_INDARR is used to distinguish the cases:
    typedef float* fp;
    typedef float f10[10];
    fp* ppf;		// pointer to pointer to float
    fp a10pf[10];	// array of 10 pointers to float
    f10* pa10f;	// pointer to array of 10 floats
   ppf[i][j]		// appears as IM_ELEMENT -> IM_PLD -> IM_ELEMENT ->
  IM_PLD -> ppf
   a10pf[i][j]		// appears as IM_ELEMENT -> IM_PLD -> IM_ELEMENT ->
  a10pf
   pa10f[i][j]		// appears as IM_ELEMENT -> IM_ELEMENT -> IM_PLD ->
  pa10f
  NT_INDARR is used when an IM_ELEMENT -> IM_PLD appears in the ILM file.
</pre>
 */
int add_arrnme(NT_KIND type, SPTR insym, int nm, ISZ_T cnst, int sub,
               bool inlarr);

/**
   \brief Add NME routine with no subscripts

   enter nme into the NME area; use add_arrnme but add a subscript field of 0
   and an inlarr field of false.
 */
SPTR addnme(NT_KIND type, SPTR insym, int nm, ISZ_T cnst);

/**
   \brief ...
 */
int add_nme_with_pte(int nm, int ptex);

/**
   \brief ...
 */
int addpte(int type, SPTR sptr, int val, int next);

/**
   \brief ...
 */
int add_rpct_nme(int orig_nme, int rpct_loop);

/**
   \brief ...
 */
int basenme_of(int nme);

/**
   \brief ...
 */
SPTR basesym_of(int nme);

/**
   \brief ...
 */
int _build_sym_nme(DTYPE dt, int curr_off, int offset, int nme);

/**
   \brief ...
 */
int build_sym_nme(SPTR sym, int offset, bool ptr_mem_op);

/**
   \brief ...
 */
int _conflict(int nm1, int nm2);

/**
   \brief ...
 */
int conflict(int nm1, int nm2);

/**
   \brief ...
 */
int hlconflict(int nm1, int nm2);

/**
   \brief ...
 */
int lookupnme(NT_KIND type, int insym, int nm, ISZ_T cnst);

/**
   \brief ...
 */
int __print_nme(FILE *ff, int nme);

/**
   \brief Print the symbol reference associated with a names entry.
 
   prints the symbol reference represented by a names entry and returns the base
   symbol of a reference given its names entry -- this is for scalar and
   structure references only
 */
int print_nme(int nme);

/**
   \brief ...
 */
int usersym_of(int nme);

/**
   \brief ...
 */
int zbasenme_of(int nme);

/**
   \brief ...
 */
void add_rpct(int rpct_nme1, int rpct_nme2);

/**
   \brief ...
 */
void count_conflict(void);

/**
   \brief ...
 */
void __dmpnme(FILE *f, int i, int flag);

/// \brief Dump names table
void dmpnme(void);

/// \brief Pretty print a nme
void dumpname(int opn);

/**
   \brief ...
 */
void __dumpname(FILE *f, int opn);

/**
   \brief ...
 */
void loc_of(int nme);

/**
   \brief ...
 */
void loc_of_vol(int nme);

#ifdef PTRSTOREP
/**
   \brief ...
 */
void ptrstore_of(int nme);
#endif

/**
   \brief ...
 */
void nme_end(void);

/**
   \brief ...
 */
void nme_init(void);

/**
   \brief ...
 */
void PrintTopHash(void);

/**
   \brief ...
 */
void PrintTopNMEHash(void);

#endif // NMEUTIL_H_
