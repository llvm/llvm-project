// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK

// Before PR45420, we would only find the alignment on this record. Afterwards,
// we can see the alignment on the typedef through the array type.
// FIXME: What about other type sugar, like _Atomic? This would only matter in a
// packed struct context.
struct __declspec(align(16)) AlignedStruct { int x; };
typedef int  __declspec(align(16)) AlignedInt;

#define CHECK_SIZE(X, Align) \
  _Static_assert(__alignof(struct X) == Align, "should be aligned");

#pragma pack(push, 2)

struct A {
  struct AlignedStruct a[1];
};
CHECK_SIZE(A, 16);

struct B {
  char b;
  AlignedInt a[1];
};
CHECK_SIZE(B, 16);

struct C {
  char b;
  AlignedInt a[];
};
CHECK_SIZE(C, 16);

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct AlignedStruct
// CHECK-NEXT:          0 |   int x
// CHECK-NEXT:            | [sizeof=16, align=16]
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct A
// CHECK-NEXT:          0 |   struct AlignedStruct [1] a
// CHECK-NEXT:            | [sizeof=16, align=16]
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct B
// CHECK-NEXT:          0 |   char b
// CHECK-NEXT:         16 |   AlignedInt [1] a
// CHECK-NEXT:            | [sizeof=32, align=16]
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:          0 | struct C
// CHECK-NEXT:          0 |   char b
// CHECK-NEXT:         16 |   AlignedInt [] a
// CHECK-NEXT:            | [sizeof=16, align=16]

#pragma pack(pop)

