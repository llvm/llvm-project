/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef DINIT_H_
#define DINIT_H_

/** \file
 * \brief (Fortran) declarations needed to use dinitutil.c module.
 */

#include "gbldefs.h"
#include "symtab.h"
#include <stdio.h>

struct CONST;
struct VAR;

typedef struct DREC {/* dinit file record */
  DTYPE dtype;  /*  also sptr  */
  ISZ_T conval; /*  also offset */
} DREC;

#define DINIT_ENDFILE  ((DTYPE)-96)
#define DINIT_STARTARY ((DTYPE)-95)
#define DINIT_ENDARY   ((DTYPE)-94)
#define DINIT_TYPEDEF  ((DTYPE)-93)
#define DINIT_ENDTYPE  ((DTYPE)-92)
#define DINIT_LOC      ((DTYPE)-99)
#define DINIT_SLOC     ((DTYPE)-91)
#define DINIT_REPEAT   ((DTYPE)-88)
#define DINIT_SECT     ((DTYPE)-87) /* conval field is sptr to string with section name */
#define DINIT_DATASECT ((DTYPE)-86) /* return to data section */
#define DINIT_OFFSET   ((DTYPE)-77)
#define DINIT_LABEL    ((DTYPE)-33)
#define DINIT_ZEROES   ((DTYPE)-66)
#define DINIT_VPUINSTR ((DTYPE)-55) /* sparc/VPU compiler only */
#define DINIT_COMMENT  ((DTYPE)-44) /* comment string for asm file - the
                                     * DREC.conval field is an int index into
                                     * the getitem_p table (salloc.c) which
                                     * contains the pointer to the string.
                                     */
#define DINIT_FILL     ((DTYPE)-59)
#define DINIT_MASK     ((DTYPE)-60)
#define DINIT_ZEROINIT ((DTYPE)-61) /* llvm : use zeroinitializer */

#define DINIT_FUNCCOUNT ((DTYPE)-31) /* gbl.func_count value */
#define DINIT_STRING   ((DTYPE)-30) // holds string initialization, length given
#define DINIT_PROC     ((DTYPE)-42) /* procedure symbol value  */

/**
   \brief ...
 */
bool dinit_ok(int sptr);

/**
   \brief ...
 */
int mk_largest_val(DTYPE dtype);

/**
   \brief ...
 */
int mk_smallest_val(DTYPE dtype);

/**
   \brief ...
 */
int mk_unop(int optype, int lop, DTYPE dtype);

/**
   \brief ...
 */
void dinit(struct VAR *ivl, struct CONST *ict);

/**
   \brief ...
 */
void dmp_ict(struct CONST *ict, FILE *f);

/**
   \brief ...
 */
void dmp_ivl(struct VAR *ivl, FILE *f);

/**
   \brief ...
 */
void do_dinit(void);


#endif
