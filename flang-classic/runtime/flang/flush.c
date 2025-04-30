/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements Fortran FLUSH statement.
 */

#include "global.h"
#include "async.h"

int ENTF90IO(FLUSH, flush)(__INT_T *unit, __INT_T *bitv, __INT_T *iostat)
{
  FIO_FCB *f;
  int s = 0;

  __fort_status_init(bitv, iostat);
  __fortio_errinit03(*unit, *bitv, iostat, "FLUSH");
  if (ILLEGAL_UNIT(*unit)) {/* check for illegal unit number */
    s = __fortio_error(FIO_EUNIT);
    __fortio_errend03();
    return s;
  }

  f = __fortio_find_unit(*unit);

  if (f) {

    /* check for outstanding async i/o */

    if (f->asy_rw) {/* stop any async i/o */
      f->asy_rw = 0;
      if (Fio_asy_disable(f->asyptr) == -1) {
        s = (__fortio_error(__io_errno()));
        __fortio_errend03();
        return s;
      }
    }

    if (__io_fflush(f->fp) != 0) {
      s = __fortio_error(__io_errno());
      __fortio_errend03();
      return s;
    }
  }

  __fortio_errend03();
  return 0; /* no error occurred */
}
