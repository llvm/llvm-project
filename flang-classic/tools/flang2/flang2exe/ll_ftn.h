/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef LL_FTN_H_
#define LL_FTN_H_

#include "gbldefs.h"
#include "symtab.h"
#include "ll_structure.h"

extern SPTR master_sptr;

/**
   \brief ...
 */
ISZ_T get_socptr_offset(int sptr);

/**
   \brief ...
 */
bool has_multiple_entries(int sptr);

/**
   \brief ...
 */
bool is_fastcall(int ilix);

/**
   \brief ...
 */
char *get_entret_arg_name(void);

/**
   \brief ...
 */
char *get_llvm_ifacenm(SPTR sptr);

/**
   \brief ...
 */
char *get_local_overlap_var(void);

/// This function collect all arguments from all Entry include main routine,
/// removing the duplicates, put into new dpdsc
/// Returns the master_sptr value
int get_entries_argnum(void);

/**
   \brief ...
 */
SPTR get_iface_sptr(SPTR sptr);

/**
   \brief ...
 */
SPTR get_master_sptr(void);

/**
   \brief ...
 */
DTYPE get_return_type(SPTR func_sptr);

/**
   \brief ...
 */
int is_iso_cptr(DTYPE d_dtype);

/**
   \brief ...
 */
int mk_charlen_address(int sptr);

/**
   \brief ...
 */
LL_Type *get_ftn_lltype(SPTR sptr);

/**
   \brief ...
 */
LL_Type *get_local_overlap_vartype(void);

/**
   \brief ...
 */
void assign_array_lltype(DTYPE dtype, int size, int sptr);

/**
   \brief ...
 */
void fix_llvm_fptriface(void);

/**
   \brief ...
 */
void get_local_overlap_size(void);

/**
   \brief ...
 */
void ll_process_routine_parameters(SPTR func_sptr);

/**
   \brief Write out all Entry's as a separate routine

   Each entry will call a master/common routine (MCR).  The first argument to
   the MCR will determine which label(Entry) control will jump to upon entry
   into the MCR.  If the MCR is a function, the next argument will be the
   function's return value.  The next argument(s) will be all non-duplicate
   aggregate arguments for all entries.  The MCR will always effectively be a
   subroutine.
 */
void print_entry_subroutine(LL_Module *module);

/**
   \brief ...
 */
void reset_equiv_var(void);

/**
   \brief ...
 */
void reset_master_sptr(void);

/**
   \brief ...
 */
void stb_process_routine_parameters(void);

/// Store interface function name in fptr_local table.  This table is done per
/// routine.  It stores the name that will be used to search for function
/// signature of what it points to.  The interface name is in the form of
/// <getname(gbl.currsub)>_$_<getname(iface)>, which is done in
/// get_llvm_ifacenm().
void store_llvm_localfptr(void);

/**
   \brief ...
 */
void write_llvm_lltype(int sptr);

/**
   \brief ...
 */
void write_local_overlap(void);

/**
   \brief ...
 */
void write_master_entry_routine(void);

/**
   \brief ...
 */
bool need_charlen(DTYPE);

#endif // LL_FTN_H_
