/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*   io3f.h - interface to the I/O support of f90 and f77 */

#include "global.h"

/*
 * 3f code uses f77 functions/globals - just define macros to access
 * the f90 equivalents.
 */
#define __fio_close(f, s) __fortio_close(f, s)
#define __fio_find_unit(u) __fortio_find_unit(u)
#define fio_fileno(f) __fort_getfd(f)
#if !defined(_WIN64)
#define pgi_fio fioFcbTbl
#endif

#if defined(_WIN64)
#define __PC_DOS 1
#else
#define __PC_DOS 0
#endif

#define FIO_FCB_ASYPTR(f) __fortio_fiofcb_asyptr(f)
#define FIO_FCB_ASY_RW(f) __fortio_fiofcb_asy_rw(f)
#define FIO_FCB_SET_ASY_RW(f, a) __fortio_set_asy_rw(f, a)
#define FIO_FCB_STDUNIT(f) __fortio_fiofcb_stdunit(f)
#define FIO_FCB_FP(f) __fortio_fiofcb_fp(f)
#define FIO_FCB_FORM(f) __fortio_fiofcb_form(f)
#define FIO_FCB_NAME(f) __fortio_fiofcb_name(f)
#define FIO_FCB_NEXT(f) __fortio_fiofcb_next(f)
