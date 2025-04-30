/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements LIB3F 64-bit fseek subroutine.  */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"

int ENT3F(FSEEK64, fseek64)(int *lu, long long *offset, int *from)
{
  FILE *f;

  /* DON'T issue any error messages */

  f = __getfile3f(*lu);
  if (f) {
    int fr;

    switch (*from) {
    case 0:
      fr = SEEK_SET;
      break;
    case 1:
      fr = SEEK_CUR;
      break;
    case 2:
      fr = SEEK_END;
      break;
    default:
      /* ERROR */
      fprintf(__io_stderr(), "Illegal fseek value %d\n", *from);
      return 0;
    }
    if (__io_fseek64(f, *offset, fr) == 0)
      return 0;
    return __io_errno();
  }

  return 0;
}
