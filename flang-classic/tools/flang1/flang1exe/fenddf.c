/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Data definitions for Fortran front-end data structures.
 */

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "semant.h"
#include "soc.h"
#include "scan.h"
#include "semstk.h"
#include "flgdf.h"

GBL gbl;

SEM sem;

SST *sst = NULL;

SCN scn;

SOC soc;

SWEL *switch_base;

AUX aux;

/** \brief We only allow casting to and from TY_WORD and TY_DWORD.  Therefore,
   you
    index into the following table as follows
    cast_types[dtype][{DT_WORD or DT_DWORD} - 1][0 for from or 1 for to]
    Entries are:
    -  0  -  casting unnecessary
    -  1  -  casting necessary
    - -1  -  casting not allowed
 */
INT cast_types[NTYPE][2][2] = {
    /* DT_NONE */ {{0, 0}, {0, 0}},
    /* DT_WORD */ {{0, 0}, {1, 1}},
    /* DT_DWORD */ {{1, 1}, {0, 0}},
    /* DT_HOLL */ {{1, -1}, {1, -1}},
    /* DT_BINT */ {{1, 1}, {1, 1}},
    /* DT_SINT */ {{1, 1}, {1, 1}},
    /* DT_INT */ {{1, 1}, {1, 1}},
    /* DT_INT8 */ {{1, 1}, {1, 1}},
    /* DT_REAL2 */ {{1, 1}, {1, 1}},
    /* DT_REAL */ {{1, 1}, {1, 1}},
    /* DT_DBLE */ {{1, 1}, {1, 1}},
    /* DT_QUAD */ {{-1, -1}, {-1, -1}},
    /* DT_CMPLX4 */ {{-1, -1}, {1, -1}},
    /* DT_CMPLX */ {{-1, -1}, {1, -1}},
    /* DT_DCMPLX */ {{-1, -1}, {1, -1}},
    /* DT_QCMPLX */ {{-1, -1}, {-1, -1}},
    /* DT_BLOG */ {{1, 1}, {1, 1}},
    /* DT_SLOG */ {{1, 1}, {1, 1}},
    /* DT_LOG */ {{1, 1}, {1, 1}},
    /* DT_LOG8 */ {{1, 1}, {1, 1}},
    /* DT_ADDR */ {{-1, -1}, {-1, -1}},
    /* DT_CHAR */ {{-1, -1}, {-1, -1}},
    /* DT_NCHAR */ {{-1, -1}, {-1, -1}},
};
