/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/*	date3f.c - Implements LIB3F date subprogram.  */

#include "ent3f.h"
#include "enames.h"
#include "ftnmiscsup.h"

void ENT3F(DATE, date)(DCHAR(buf) DCLEN(buf))
{
  Ftn_date(CADR(buf), CLEN(buf));
}
