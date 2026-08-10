// RUN: %clang_cc1 -fsyntax-only -fclang-abi-compat=6 -triple x86_64-linux-gnu -fdump-record-layouts %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V6
// RUN: %clang_cc1 -fsyntax-only -fclang-abi-compat=7 -triple x86_64-linux-gnu -fdump-record-layouts %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-V7

// In Clang 6 and before, we determined that Nonempty was empty, so we
// applied EBO to it.
struct Nonempty { int : 4; };
struct A : Nonempty { int n; };
int k = sizeof(A);

// CHECK:*** Dumping AST Record Layout
// CHECK:             0 | struct A
// CHECK-V6-NEXT:     0 |   struct Nonempty (base) (empty)
// CHECK-V7-NEXT:     0 |   struct Nonempty (base){{$}}
// CHECK-NEXT:    0:0-3 |     int
// CHECK-V6-NEXT:     0 |   int n
// CHECK-V7-NEXT:     4 |   int n
// CHECK-V6-NEXT:       | [sizeof=4, dsize=4, align=4,
// CHECK-V6-NEXT:       |  nvsize=4, nvalign=4]
// CHECK-V7-NEXT:       | [sizeof=8, dsize=8, align=4,
// CHECK-V7-NEXT:       |  nvsize=8, nvalign=4]
