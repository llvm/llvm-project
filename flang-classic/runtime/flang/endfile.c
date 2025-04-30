/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief
 * Implement Fortran ENDFILE statement.
 */

#include "global.h"

static int
_f90io_endfile(__INT_T *unit, __INT_T *bitv, __INT_T *iostat)
{
  FIO_FCB *f;

  __fortio_errinit03(*unit, *bitv, iostat, "ENDFILE");
  if (ILLEGAL_UNIT(*unit))
    return __fortio_error(FIO_EUNIT);

  /*	call rwinit to get FCB pointer, do error checking, and truncate
      file if necessary:  */

  f = __fortio_rwinit(*unit, FIO_UNFORMATTED, 0L, 2 /*endfile*/);
  if (f == NULL)
    return ERR_FLAG;

  f->eof_flag = TRUE;
  return 0; /* no error occurred */
}

__INT_T
ENTF90IO(ENDFILE, endfile)(__INT_T *unit, __INT_T *bitv, __INT_T *iostat)
{
  int s = 0;

  __fort_status_init(bitv, iostat);
  if (LOCAL_MODE || GET_DIST_LCPU == GET_DIST_IOPROC)
    s = _f90io_endfile(unit, bitv, iostat);
  __fortio_errend03();
  return DIST_STATUS_BCST(s);
}

__INT_T
ENTCRF90IO(ENDFILE, endfile)(__INT_T *unit, __INT_T *bitv, __INT_T *iostat)
{
  int s = 0;
  s = _f90io_endfile(unit, bitv, iostat);
  __fortio_errend03();
  return s;
}
