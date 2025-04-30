/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* defs.c -- definitions of globals that were originally in files that
 *           had to be split into file.c and file_i8.c versions.
 */

#include "stdioInterf.h"
#include "fioMacros.h"

#include "fort_vars.h"

/* dist.c */
__INT_T f90DummyGenBlock = 0;
__INT_T *f90DummyGenBlockPtr = &f90DummyGenBlock;

/* dbug.c */
/*
 * explicitly initialize for OSX since the object of this file will not be
 * linked in from the library.  The link will not see the .comm for
 * __fort_test (i.e., its definition!).
 */
__fort_vars_t   __fort_vars __attribute__((aligned(128))) = {
    .debug      = 0,
    .zmem       = 0,
    .debugn     = 0,
    .ioproc     = 0,
    .lcpu       = 0,
    .np2        = 0,
    .pario      = 0,
    .quiet      = 0,
    .tcpus      = 0,
    .test       = 0,
    .heapz      = 0,
    .heap_block = 0,
    .tids       = NULL,
    .red_what   = NULL,
};
