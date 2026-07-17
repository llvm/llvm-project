// RUN: %clang_cc1 -fno-rtti -triple aarch64-windows-msvc -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-ARM64
// RUN: %clang_cc1 -fno-rtti -triple arm64ec-windows-msvc -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64
// RUN: %clang_cc1 -fno-rtti -triple x86_64-windows-msvc -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

// On Arm64 (but not Arm64EC or x64), MSVC reuses the tail padding of an
// over-aligned base class for a subsequent base class.

struct alignas(16) Type {
  long long a, b, c;
};
struct Empty {};
struct Mid : Type {};
struct Node { void *next; };

// The over-aligned base's tail padding is reused for the following base only on
// Arm64.
struct Derived : Mid, Node {};

int a = sizeof(Derived);

// CHECK-ARM64-LABEL:  0 | struct Derived
// CHECK-ARM64-NEXT:   0 |   struct Mid (base)
// CHECK-ARM64-NEXT:   0 |     struct Type (base)
// CHECK-ARM64:       24 |   struct Node (base)
// CHECK-ARM64-NEXT:  24 |     void * next
// CHECK-ARM64-NEXT:     | [sizeof=32, align=16,
// CHECK-ARM64-NEXT:     |  nvsize=32, nvalign=16]

// CHECK-X64-LABEL:  0 | struct Derived
// CHECK-X64-NEXT:   0 |   struct Mid (base)
// CHECK-X64-NEXT:   0 |     struct Type (base)
// CHECK-X64:       32 |   struct Node (base)
// CHECK-X64-NEXT:  32 |     void * next
// CHECK-X64-NEXT:     | [sizeof=48, align=16,
// CHECK-X64-NEXT:     |  nvsize=48, nvalign=16]
