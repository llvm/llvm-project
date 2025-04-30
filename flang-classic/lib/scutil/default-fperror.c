/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file
 * \brief Default fperror() stub
 *
 *  Default fperror() routine invoked by the scutil compile-time
 *  evaluation routines.  It resides here in its own source file
 *  so that it can be overridden, which all compiler clients do.
 */

#include "legacy-folding-api.h"
#include <stdio.h>

void
fperror(int fpe)
{
  switch (fpe) {
  case FPE_NOERR:
    break;
  case FPE_INVOP:
    fprintf(stderr, "illegal input or NaN error\n");
    break;
  case FPE_FPOVF: /* == FPE_IOVF, FPE_DIVZ */
    fprintf(stderr, "overflow error\n");
    break;
  case FPE_FPUNF:
    fprintf(stderr, "underflow error\n");
    break;
  default:
    fprintf(stderr, "unknown floating-point error (%d)\n", fpe);
  }
}
