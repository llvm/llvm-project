/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Definitions for FORTRAN edit descriptor symbolics
 *
 * If contents are changed, changes must also be applied to the run-time
 * document file (fio.n) and this file must copied to the io source rte
 * directory.
 */

typedef enum {
  __BYTE = 0,   /* (byte) */
  __WORD = 1,   /* typeless */
  __DWORD = 2,  /* double typeless */
  __HOLL = 3,   /* hollerith */
  __BINT = 4,   /* byte integer */
  __SINT = 5,   /* short integer */
  __INT = 6,    /* integer*4 */
  __REAL = 7,   /* real */
  __DBLE = 8,   /* real*8 */
  __QUAD = 9,   /* real*16*/
  __CPLX = 10,  /* (real complex) */
  __DCPLX = 11, /* (real*8 complex) */
  __BLOG = 12,  /* byte logical */
  __SLOG = 13,  /* short logical */
  __LOG = 14,   /* logical */
  __CHAR = 15,  /* signed char */
  __NCHAR = 16, /* ncharacter - kanji */
  __INT8 = 17,  /* integer*8 */
  __LOG8 = 18   /* logical*8 */
} _pgfio_type;
