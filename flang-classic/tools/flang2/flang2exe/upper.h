/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UPPER_H_
#define UPPER_H_

/** \file
 * \brief Header file for upper - import the lowered F90/HPF code
 */
/*
 * Compatibility History:
 * before 6.2  -- 1.9
 * 6.2         -- 1.10
 *                Includes all of 1.9 + PASSBYVAL & PASSBYREF
 * 7.0         -- 1.11
 *                Includes all of 1.10 + CFUNC for variables
 * 7.1         -- 1.12
 *                Includes all of 1.11 + DECORATE
 * 7.2         -- 1.13
 *                Includes all of 1.12 + CREF & NOMIXEDSTRLEN
 * 8.0         -- 1.14
 *                Includes all of 1.13 + FILE INDEX ENTRIES
 * 8.1         -- 1.15
 *                Includes all of 1.14 + new types + cuda flags
 * 9.0-3       -- 1.16
 *                Includes all of 1.15 + cudaemu value
 * 10.6        -- 1.17
 *                Includes all of 1.16 + sptr for Constant ID data init + denorm
 * 10.9        -- 1.18
 *                Includes all of 1.17 + reflected/mirrored/devcopy flags and
 * devcopy field
 * 11.0        -- 1.19
 *                Includes all of 1.18 + mscall & cref for vars & members
 * 11.4        -- 1.20
 *                Includes all of 1.19 + libm & libc for functions
 * 12.7        -- 1.21
 *                Includes all of 1.20 + TASK for variables
 * 12.7        -- 1.22
 *                Includes all of 1.21 + cuda texture flag
 * 12.7        -- 1.23
 *                Includes all of 1.21 + INTENTIN flag
 * 13.0        -- 1.24
 *                Includes all of 1.23 + DATACNST flag
 * 13.5        -- 1.25
 *                Includes all of 1.24 + MODCMN flag
 * 13.8        -- 1.26
 *                Includes all of 1.25 + DOBEGNZ & DOENDNZ
 * 13.9        -- 1.27
 *                Includes all of 1.26 + symbol ACCCREATE and ACCRESIDENT
 * 14.0        -- 1.28
 *                Includes all of 1.27 + ACCROUT
 * 14.4        -- 1.29
 *                Includes all of 1.28 + CUDAMODULE
 * 14.4        -- 1.30
 *                Includes all of 1.29 + MANAGED + additionsl ILM operand
 *                    for the call ILMs via a procedure ptr, e.g., CALLA,
 *                    CDFUNCA, etc.
 * 14.7        -- 1.31
 *                All of 1.30 + ACCCREATE + ACCRESIDENT for common blocks,
 *                    +ACCLINK +ACCCOPYIN symbol flags
 * 15.0        -- 1.32
 *                All of 1.31 + new FARGF ILM
 * 15.3        -- 1.33
 *                All of 1.32 + FWDREF flag + INTERNREF flag + AGOTO field
 * 15.4        -- 1.34
 *                All of 1.33 + ST_MEMBER IFACE field
 * 15.7        -- 1.35
 *                All of 1.34 + ST_ENTRY/ST_PROC ARET field
 * 15.9        -- 1.36
 *                All of 1.35 + PARREF, PARSYMS, & PARSYMSCT
 * 15.10       -- 1.37
 *                All of 1.36 + IM_BMPSCOPE/IM_EMPSCOPE
 * 16.1        -- 1.38
 *                All of 1.37 + IM_MPLOOP/IM_MPSCHED and
 *                    IM_MPBORDERED/IM_MPEORDERED + TPALLOC + IM_FLUSH flag
 * 16.4        -- 1.39
 *                All of 1.38 + IM_ETASK and IM_TASKFIRSPRIV
 * 16.5        -- 1.40
 *                All of 1.39 + ISOTYPE flag + update MP_SCHED and MPLOOP ilm
 * 16.6        -- 1.41
 *                All of 1.40 + IM_LSECTION
 * 16.6        -- 1.42
 *                All of 1.41 + VARARG
 * 16.8        -- 1.43
 *                All of 1.42 + ALLOCATTR + F90POINTER
 * 16.10       -- 1.44
 *                All of 1.43 +
 * IM_TASKGROUP/ETASKGROUP/TARGET/TARGETDATA/TARGETUPDATE/
 *                TARGETEXITDATA/TARGETENTERDATA/DISTRIBUTE/TEAMS and their
 * combinations
 *                constructs including TARGET/TEAMS/DISTRIBUTE/PARALLEL
 * DO/CANCEL/
 *                CANCELLATIONPOINT.
 * 17.0        -- 1.45
 *                All of 1.44 + INVOBJINC + PARREF for ST_PROC
 * 17.2        -- 1.46
 *                All of 1.45 + etls + tls, irrspective of target
 * 17.7        -- 1.47
 *                All of 1.46 + BPARA + PROC_BIND + MP_ATOMIC..
 * 17.10        -- 1.48 
 *                All of 1.47 + ETASKFIRSTPRIV, MP_[E]TASKLOOP, 
 *                MP_[E]TASKLOOPREG
 * 18.1         -- 1.49 
 *                All of 1.48 , MP_TASKLOOPVARS, [B/E]TASKDUP
 *
 * 18.4
 *              --1.50
 *                All of 1.49 +
 *                Internal procedures passed as arguments and pointer targets
 * 18.7         -- 1.51
 *                All of 1.50 +
 *                remove parsyms field and add parent for ST_BLOCK,
 *                receive "no_opts" (no optional arguments) flag for ST_ENTRY
 *                and ST_PROC symbols.
 * 18.10        -- 1.52
 *                All of 1.51 +
 *                add IS_INTERFACE flag for ST_PROC, and for ST_MODULE when
 *                emitting as ST_PROC
 * 19.3         -- 1.53
 *                All of 1.52 +
 *                Add has_alias bit, and length and name of the alias for Fortran
 *                module variable when it is on the ONLY list of a USE statement.
 *                This is for Fortran LLVM compiler only.
 *
 * 19.10        -- 1.54
 *              All of 1.53 +
 *              pass allocptr and ptrtarget values for default initialization
 *              of standalone pointers.
 *
 * 20.1         -- 1.55
 *              All of 1.54 +
 *              pass elemental field for subprogram when emitting ST_ENTRY.
 *
 *              For ST_PROC, receive IS_PROC_PTR_IFACE flag.
 *
 * 23.12        -- 1.56
 *              All of 1.55 + PALIGN
 */

#include "gbldefs.h"
#include "semant.h"

#define VersionMajor 1
#define VersionMinor 56

/**
   \brief ...
 */
char *getexnamestring(char *string, int sptr, int stype, int scg,
                      int extraunderscore);

/**
   \brief ...
 */
int F90_nme_conflict(int nme1, int nme2);

/**
   \brief Detect Fortran 90 structure member name conflicts.
 * Return 0 if they point to different addresses;
 * Return 1 otherwise.
 */
int F90_struct_mbr_nme_conflict(int nme1, int nme2);

/**
   \brief ...
 */
int getswel(int sz);

/**
   \brief ...
 */
int IPA_allcall_safe(int sptr);

/**
   \brief ...
 */
int IPA_call_safe(int funcsptr, int sptr);

/**
   \brief ...
 */
int IPA_func_almostpure(int sptr);

/**
   \brief ...
 */
int IPA_func_pure(int sptr);

/**
   \brief ...
 */
int IPA_nme_conflict(int nme1, int nme2);

/**
   \brief ...
 */
int IPA_noaddr(int sptr);

/**
   \brief ...
 */
int IPA_NoFree(void);

/**
   \brief ...
 */
int IPA_pointer_safe(int nme);

/**
   \brief ...
 */
int IPA_range(int sptr, int *plo, int *phi);

/**
   \brief ...
 */
int IPA_safe(int sptr);

/**
   \brief ...
 */
SPTR llvm_get_uplevel_newsptr(int oldsptr);

/**
   \brief ...
 */
long IPA_pstride(int sptr);

/**
   \brief ...
 */
long IPA_sstride(int sptr);

/**
   \brief ...
 */
void cuda_emu_end(void);

/**
   \brief ...
 */
void cuda_emu_start(void);

/**
   \brief ...
 */
void dmp_const(CONST *acl, int indent);

/**
   \brief ...
 */
void stb_upper_init(void);

/**
   \brief ...
 */
void upper_assign_addresses(void);

/**
   \brief ...
 */
void upper_init(void);

/**
   \brief ...
 */
void upper(int stb_processing);

/**
   \brief ...
 */
void upper_save_syminfo(void);

/**
   \brief Search for Module variable alias name saved by upper()
   \param sptr   The sptr of a Module variable
 */
const char *lookup_modvar_alias(SPTR sptr);

/**
   \brief return start symbol SPTR
 */
SPTR get_symbol_start(void);

#endif // UPPER_H_
