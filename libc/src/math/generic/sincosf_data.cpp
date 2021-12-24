//===-- sinf/cosf data tables ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math_utils.h"
#include "sincosf_utils.h"

#include <stdint.h>

namespace __llvm_libc {

// The constants and polynomials for sine and cosine.  The 2nd entry
// computes -cos (x) rather than cos (x) to get negation for free.
const sincos_t __SINCOSF_TABLE[2] = {
    {{1.0, -1.0, -1.0, 1.0},
     as_double(0x41645f306dc9c883),
     as_double(0x3ff921fb54442d18),
     as_double(0x3ff0000000000000),
     as_double(0xbfdffffffd0c621c),
     as_double(0x3fa55553e1068f19),
     as_double(0xbf56c087e89a359d),
     as_double(0x3ef99343027bf8c3),
     as_double(0xbfc555545995a603),
     as_double(0x3f81107605230bc4),
     as_double(0xbf2994eb3774cf24)},
    {{1.0, -1.0, -1.0, 1.0},
     as_double(0x41645f306dc9c883),
     as_double(0x3ff921fb54442d18),
     as_double(0xbff0000000000000),
     as_double(0x3fdffffffd0c621c),
     as_double(0xbfa55553e1068f19),
     as_double(0x3f56c087e89a359d),
     as_double(0xbef99343027bf8c3),
     as_double(0xbfc555545995a603),
     as_double(0x3f81107605230bc4),
     as_double(0xbf2994eb3774cf24)},
};

// Table with 4/PI to 192 bit precision.  To avoid unaligned accesses
// only 8 new bits are added per entry, making the table 4 times larger.
const uint32_t __INV_PIO4[24] = {
    0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e, 0x836e4e44,
    0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
    0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, 0x34ddc0db, 0xddc0db62,
    0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041};

} // namespace __llvm_libc
