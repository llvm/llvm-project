// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple i686-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fms-extensions -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s

struct EmptyIntMemb {
  int FlexArrayMemb[0];
};
// CHECK:       *** Dumping AST Record Layout
// CHECK-NEXT:  0 | struct EmptyIntMemb
// CHECK-NEXT:  0 | int[0] FlexArrayMemb
// CHECK-NEXT:    | [sizeof=4, align=4

struct EmptyLongLongMemb {
  long long FlexArrayMemb[0];
};
// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:  0 | struct EmptyLongLongMemb
// CHECK-NEXT:  0 | long long[0] FlexArrayMemb
// CHECK-NEXT:    | [sizeof=4, align=8

struct EmptyAligned2LongLongMemb {
  long long __declspec(align(2)) FlexArrayMemb[0];
};

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:  0 | struct EmptyAligned2LongLongMemb
// CHECK-NEXT:  0 | long long[0] FlexArrayMemb
// CHECK-NEXT:    | [sizeof=4, align=8

struct EmptyAligned8LongLongMemb {
  long long __declspec(align(8)) FlexArrayMemb[0];
};

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:  0 | struct EmptyAligned8LongLongMemb
// CHECK-NEXT:  0 | long long[0] FlexArrayMemb
// CHECK-NEXT:    | [sizeof=8, align=8

#pragma pack(1)
struct __declspec(align(4)) EmptyPackedAligned4LongLongMemb {
  long long FlexArrayMemb[0];
};
#pragma pack()

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:  0 | struct EmptyPackedAligned4LongLongMemb
// CHECK-NEXT:  0 | long long[0] FlexArrayMemb
// CHECK-NEXT:    | [sizeof=4, align=4

#pragma pack(1)
struct EmptyPackedAligned8LongLongMemb {
  long long __declspec(align(8)) FlexArrayMemb[0];
};
#pragma pack()

// CHECK: *** Dumping AST Record Layout
// CHECK-NEXT:  0 | struct EmptyPackedAligned8LongLongMemb
// CHECK-NEXT:  0 | long long[0] FlexArrayMemb
// CHECK-NEXT:    | [sizeof=8, align=8


int a[
sizeof(struct EmptyIntMemb)+
sizeof(struct EmptyLongLongMemb)+
sizeof(struct EmptyAligned2LongLongMemb)+
sizeof(struct EmptyAligned8LongLongMemb)+
sizeof(struct EmptyPackedAligned4LongLongMemb)+
sizeof(struct EmptyPackedAligned8LongLongMemb)+
0];
