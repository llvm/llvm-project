/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ILMUTIL_H_
#define ILMUTIL_H_

#include "gbldefs.h"
#include "symtab.h"
#include "ilmtp.h"

/**
   \brief ...
 */
ILM_T *save_ilms0(void *area);

/**
   \brief ...
 */
ILM_T *save_ilms(int area);

/**
   \brief ...
 */
int ad1ilm(int opc);

/**
   \brief ...
 */
int ad2ilm(int opc, int opr1);

/**
   \brief ...
 */
int ad3ilm(int opc, int opr1, int opr2);

/**
   \brief ...
 */
int ad4ilm(int opc, int opr1, int opr2, int opr3);

/**
   \brief ...
 */
int ad5ilm(int opc, int opr1, int opr2, int opr3, int opr4);

/**
   \brief ...
 */
int adNilm(int n, int opc, ...);

/**
   \brief ...
 */
int count_ilms(void);

/**
   \brief ...
 */
int _dumponeilm(ILM_T *ilm_base, int i, int check);

/**
   \brief ...
 */
int get_entry(void);

/**
   \brief FIXME
   \return index of callee for given operation, which must have type IMTY_PROC.
   index+k is the index for the kth argument.  index-1 is the index of the dtype
   for the function signature if there is one.
 */
int ilm_callee_index(ILM_OP opc);

/**
   \brief Determine if the ILM performs a call and the call has a pointer to a
   return slot.  If so, return the operand index of the slot.  Otherwise return
   0.  Returns 0 for non-call ILMs.
 */
int ilm_return_slot_index(ILM_T *ilmp);

/**
   \brief ...
 */
int rdgilms(int mode);

/**
   \brief ...
 */
int rdilms(void);

/**
   \brief ...
 */
long get_ilmpos(void);

/**
   \brief ...
 */
long get_ilmstart(void);

#ifdef ST_UNKNOWN /* Use ST_UNKNOWN to detect if SYMTYPE is defined. */
/**
   \brief If a function returning a value of type ret_type needs to have a
   pointer to a temporary for possible use as as return slot, return the SYMTYPE
   for that temporary.  Otherwise return ST_UNKNOWN.

   The result is a property of ILM, not the ABI.
 */
SYMTYPE ilm_symtype_of_return_slot(DTYPE ret_type);
#endif

/**
   \brief ...
 */
void add_ilms(ILM_T *p);

/**
   \brief ...
 */
void addlabel(int sptr);

/**
   \brief ...
 */
void dmpilms(void);

/**
   \brief ...
 */
void _dumpilms(ILM_T *ilm_base, int check);

/**
   \brief ...
 */
void dumpilms(void);

/**
   \brief ...
 */
void dumpilmtree(int ilmptr);

/**
   \brief ...
 */
void dumpilmtrees(void);

/**
   \brief ...
 */
void dumpsingleilm(ILM_T *ilm_base, int i);

/**
   \brief ...
 */
void fini_ilm(void);

/**
   \brief ...
 */
void fini_next_gilm(void);

/**
   \brief ...
 */
void gwrilms(int nilms);

/**
   \brief ...
 */
void init_global_ilm_mode(void);

/**
   \brief ...
 */
void init_global_ilm_position(void);

/**
   \brief ...
 */
void init_ilm(int ilmsize);

/**
   \brief ...
 */
void init_next_gilm(void);

/**
   \brief ...
 */
void mkbranch(int ilmptr, int truelb, int flag);

/**
   \brief ...
 */
void reloc_ilms(ILM_T *p);

/**
   \brief ...
 */
void reset_global_ilm_position(void);

/**
   \brief ...
 */
void restartilms(void);

/**
   \brief ...
 */
void RestoreGilms(FILE *fil);

/**
   \brief ...
 */
void rewindilms(void);

/**
   \brief ...
 */
void SaveGilms(FILE *fil);

/**
   \brief ...
 */
void set_gilmb_mode(int mode);

/**
   \brief ...
 */
void set_ilmpos(long pos);

/**
   \brief ...
 */
void set_ilmstart(int start);

/**
   \brief ...
 */
void swap_next_gilm(void);

/**
   \brief ...
 */
void wrilms(int linenum);

#endif // ILMUTIL_H_
