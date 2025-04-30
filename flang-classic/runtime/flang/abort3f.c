/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
  * \brief Implements LIB3F abort subprogram.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "open_close.h"

void ENT3F(ABORT, abort)()
{
  void *f, *q;

  for (f = GET_FIO_FCBS; f != NULL; f = q) {
    q = FIO_FCB_NEXT(f);
    (void) __fio_close(f, 0 /*dispose = default*/);
  }
  abort();
}
