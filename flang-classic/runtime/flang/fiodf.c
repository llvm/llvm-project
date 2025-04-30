/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* fiodf.h - define global data for Fortran I/O */

#include "global.h"

/* define global variables for fortran I/O (members of struct fioFcbTbls): */

FIO_TBL fioFcbTbls = {0};

#if defined(_WIN64)
FIO_FCB *
__get_hpfio_fcbs(void)
{
  return fioFcbTbls.fcbs;
}
#endif

/* define array giving sizes in bytes of the different data types: */

short __fortio_type_size[] = {
    1,  /* (byte) */
    2,  /* signed short */
    2,  /* unsigned short */
    4,  /* signed int */
    4,  /* unsigned int */
    4,  /* signed long int */
    4,  /* unsigned long int */
    4,  /* float */
    8,  /* double */
    8,  /* (float complex) */
    16, /* (double complex) */
    1,  /* signed char */
    1,  /* unsigned char */
    16, /* long double */
    1,  /* (string) */
    8,  /* long long */
    8,  /* unsigned long long */
    1,  /* byte logical */
    2,  /* short logical */
    4,  /* logical */
    8,  /* logical*8 */
    4,  /* typeless */
    8,  /* double typeless */
    2,  /* ncharacter - kanji */
};
