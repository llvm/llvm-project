/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief X flag handling routines
 */

#include "flang/ArgParser/xflag.h"
#include <stdbool.h>

/** Does this index correspond to a bitvector (true) or plain value (false) */
bool
is_xflag_bitvector(int index)
{
  switch (index) {
  case 9:   /* max cnt for unroller */
  case 10:  /* # of times to unroll */
  case 16:  /* lower bound loop iter count for vectorizer */
  case 27:  /* default/max overlap size */
  case 30:  /* lower limit on iteration count */
  case 31:  /* lower limit on strip size */
  case 32:  /* cache size */
  case 33:  /* upper limit on strip size in complex-vector loops */
  case 35:  /* max iteration count */
  case 38:  /* MAXVPUS */
  case 40:  /* # array loads/stores */
  case 41:  /* # fp operations */
  case 43:  /* # CPUs */
  case 44:  /* min parallel loop count */
  case 64:  /* code straightening optimizations */
  case 79:  /* CSE of DP loads: distance threshold */
  case 100: /* break blocks */
  case 101: /* ST processor SI information */
  case 105: /* upper limit on unrolling with unroll & jam */
  case 106: /* set unroll factor for scalar  unroll & jam */
  case 130: /* Levels of VLIW scheduling */
  case 131: /* Levels of predication */
  case 133: /* VLIW density */
  case 134: /* register stall limit */
  case 138: /* Limit of the #of prefetches for a loop */
  case 139: /* Limit of #vili for vectorizing single precision loop */
  case 140: /* Limit of #vili for vectorizing double precision loop */
  case 157: /* count for unroller for multiblock loops */
  case 160: /* nest level for intensity count */
  case 181: /* 3D tile size */
  case 188: /* openacc default vector length */
  case 199: /* max blocks in loop to be fused */
  case 249: /* LLVM version number */
    return false;
  default:
    break;
  }
  return true;
}

/** Set a value for and x-flag */
void
set_xflag_value(int *xflags, int index, int value)
{
  if (is_xflag_bitvector(index))
    xflags[index] |= value;
  else
    xflags[index] = value;
}

/** Clear value for and x-flag */
void
unset_xflag_value(int *xflags, int index, int value)
{
  if (is_xflag_bitvector(index))
    xflags[index] &= ~value;
  else
    xflags[index] = 0;
}
