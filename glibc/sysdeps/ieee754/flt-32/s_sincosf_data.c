/* Compute sine and cosine of argument.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdint.h>
#include <math.h>
#include "math_config.h"
#include "s_sincosf.h"

/* The constants and polynomials for sine and cosine.  The 2nd entry
   computes -cos (x) rather than cos (x) to get negation for free.  */
const sincos_t __sincosf_table[2] =
{
  {
    { 1.0, -1.0, -1.0, 1.0 },
#if TOINT_INTRINSICS
    0x1.45F306DC9C883p-1,
#else
    0x1.45F306DC9C883p+23,
#endif
    0x1.921FB54442D18p0,
    0x1p0,
    -0x1.ffffffd0c621cp-2,
    0x1.55553e1068f19p-5,
    -0x1.6c087e89a359dp-10,
    0x1.99343027bf8c3p-16,
    -0x1.555545995a603p-3,
    0x1.1107605230bc4p-7,
    -0x1.994eb3774cf24p-13
  },
  {
    { 1.0, -1.0, -1.0, 1.0 },
#if TOINT_INTRINSICS
    0x1.45F306DC9C883p-1,
#else
    0x1.45F306DC9C883p+23,
#endif
    0x1.921FB54442D18p0,
    -0x1p0,
    0x1.ffffffd0c621cp-2,
    -0x1.55553e1068f19p-5,
    0x1.6c087e89a359dp-10,
    -0x1.99343027bf8c3p-16,
    -0x1.555545995a603p-3,
    0x1.1107605230bc4p-7,
    -0x1.994eb3774cf24p-13
  }
};

/* Table with 4/PI to 192 bit precision.  To avoid unaligned accesses
   only 8 new bits are added per entry, making the table 4 times larger.  */
const uint32_t __inv_pio4[24] =
{
  0xa2,       0xa2f9,	  0xa2f983,   0xa2f9836e,
  0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529,
  0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
  0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0,
  0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599,
  0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041
};
