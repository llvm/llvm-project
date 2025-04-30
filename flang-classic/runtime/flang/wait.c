/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements Fortran WAIT statement.
 */

#include "global.h"
#include "async.h"

#if defined(_WIN64)
#define access _access
#endif

/* ------------------------------------------------------------------ */

__INT_T
ENTF90IO(WAIT, wait)(__INT_T *unit, __INT_T *bitv, __INT_T *iostat, __INT_T *id)
{
  FIO_FCB *f;
  int s = 0;

  __fort_status_init(bitv, iostat);
  __fortio_errinit03(*unit, *bitv, iostat, "WAIT");
  if (ILLEGAL_UNIT(*unit)) {
    s = __fortio_error(FIO_EUNIT);
    __fortio_errend03();
    return s;
  }

  f = __fortio_find_unit(*unit);
  if (f == NULL) {
    __fortio_errend03();
    return (0);
  }

  /* check for outstanding async i/o */

  if (f->asy_rw) {/* stop any async i/o */
    f->asy_rw = 0;
    if (Fio_asy_disable(f->asyptr) == -1) {
      s = (__fortio_error(__io_errno()));
      __fortio_errend03();
      return s;
    }
  }

  __fortio_errend03();
  return 0; /* no error occurred */
}
