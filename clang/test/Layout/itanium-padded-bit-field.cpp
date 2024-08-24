// RUN: %clang_cc1 -DOLD_ABI=false -std=c++17 -fsyntax-only -triple %itanium_abi_triple -fdump-record-layouts -Wno-bitfield-width %s | FileCheck %s --check-prefixes=CHECK,NEW
// RUN: %clang_cc1 -fclang-abi-compat=19.0 -DOLD_ABI=true -std=c++17 -fsyntax-only -triple %itanium_abi_triple -fdump-record-layouts -Wno-bitfield-width %s | FileCheck %s --check-prefixes=CHECK,OLD

using i4 = __INT32_TYPE__;
using i1 = __INT8_TYPE__;

struct OversizedBase {
  i4 i : 33;
};

static_assert(sizeof(OversizedBase) == 8);
// CHECK:*** Dumping AST Record Layout
// CHECK:           0 | struct OversizedBase
// CHECK-NEXT: 0:0-32 |   i4 i

// NEW-NEXT:          | [sizeof=8, dsize=5, align=4,
// NEW-NEXT:          |  nvsize=5, nvalign=4]

// OLD-NEXT:          | [sizeof=8, dsize=8, align=4,
// OLD-NEXT:          |  nvsize=8, nvalign=4]

struct X : OversizedBase {
  i1 in_padding;
};
static_assert(sizeof(X) == (OLD_ABI ? 12 : 8));
// CHECK:*** Dumping AST Record Layout
// CHECK:           0 | struct X
// CHECK-NEXT:      0 |   struct OversizedBase (base)
// CHECK-NEXT: 0:0-32 |     i4 i

// NEW-NEXT:        5 |   i1 in_padding
// NEW-NEXT:          | [sizeof=8, dsize=6, align=4,
// NEW-NEXT:          |  nvsize=6, nvalign=4]

// OLD-NEXT:        8 |   i1 in_padding
// OLD-NEXT:          | [sizeof=12, dsize=9, align=4,
// OLD-NEXT:          |  nvsize=9, nvalign=4]

struct Y : OversizedBase {
  i1 in_padding[3];
};

static_assert(sizeof(Y) == (OLD_ABI ? 12 : 8));
// CHECK:*** Dumping AST Record Layout
// CHECK:           0 | struct Y
// CHECK-NEXT:      0 |   struct OversizedBase (base)
// CHECK-NEXT: 0:0-32 |     i4 i

// NEW-NEXT:        5 |   i1[3] in_padding
// NEW-NEXT:          | [sizeof=8, dsize=8, align=4,
// NEW-NEXT:          |  nvsize=8, nvalign=4]

// OLD-NEXT:        8 |   i1[3] in_padding
// OLD-NEXT:          | [sizeof=12, dsize=11, align=4,
// OLD-NEXT:          |  nvsize=11, nvalign=4]

struct Z : OversizedBase {
  i1 in_padding[4];
};

static_assert(sizeof(Z) == 12);
// CHECK:*** Dumping AST Record Layout
// CHECK:           0 | struct Z
// CHECK-NEXT:      0 |   struct OversizedBase (base)
// CHECK-NEXT: 0:0-32 |     i4 i

// NEW-NEXT:        5 |   i1[4] in_padding
// NEW-NEXT:          | [sizeof=12, dsize=9, align=4,
// NEW-NEXT:          |  nvsize=9, nvalign=4]

// OLD-NEXT:        8 |   i1[4] in_padding
// OLD-NEXT:          | [sizeof=12, dsize=12, align=4,
// OLD-NEXT:          |  nvsize=12, nvalign=4]

namespace BitInt {
struct alignas(4) OversizedBase {
  _BitInt(9) i : 10;
};
static_assert(sizeof(OversizedBase) == 4);
// CHECK:*** Dumping AST Record Layout
// CHECK:           0 | struct BitInt::OversizedBase
// CHECK-NEXT: 0:0-9  |   _BitInt(9) i

// NEW-NEXT:        | [sizeof=4, dsize=2, align=4,
// NEW-NEXT:        |  nvsize=2, nvalign=4]

// OLD-NEXT:        | [sizeof=4, dsize=4, align=4,
// OLD-NEXT:        |  nvsize=4, nvalign=4]

struct X : OversizedBase {
  i1 in_padding;
};
static_assert(sizeof(X) == (OLD_ABI ? 8 : 4));
// CHECK:*** Dumping AST Record Layout
// CHECK:           0 | struct BitInt::X
// CHECK-NEXT:      0 |   struct BitInt::OversizedBase (base)
// CHECK-NEXT:  0:0-9 |     _BitInt(9) i

// NEW-NEXT:        2 |   i1 in_padding
// NEW-NEXT:          | [sizeof=4, dsize=3, align=4,
// NEW-NEXT:          |  nvsize=3, nvalign=4]

// OLD-NEXT:        4 |   i1 in_padding
// OLD-NEXT:          | [sizeof=8, dsize=5, align=4,
// OLD-NEXT:          |  nvsize=5, nvalign=4]
}
