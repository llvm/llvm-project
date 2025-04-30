/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief Implements LIB3F flush subroutine. In this case, a wrapper function */

/* must include ent3f.h AFTER io3f.h */
#include "io3f.h"
#include "ent3f.h"
#include "utils3f.h"
#include "stdioInterf.h"

void ENT3F(FLUSH, flush)(int *lu)
{
  FILE *f;

  /* DON'T issue any error messages */

  f = __getfile3f(*lu);
  if (f)
    fflush(f);

  return;
}
