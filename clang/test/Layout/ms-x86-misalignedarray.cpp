// RUN: %clang_cc1 -fno-rtti -triple i686-pc-win32 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fno-rtti -emit-llvm-only -triple x86_64-pc-win32 -fdump-record-layouts -fsyntax-only %s 2>/dev/null \
// RUN:            | FileCheck %s -check-prefix CHECK-X64

struct T0 { char c; };
struct T2 : virtual T0 { };
struct T3 { T2 a[1]; char c; };

#ifdef _ILP32
// expected-error@-3 {{size of array element}}
#endif

// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64: *** Dumping AST Record Layout
// CHECK-X64-NEXT:    0 | struct T3
// CHECK-X64-NEXT:    0 |   T2[1] a
// CHECK-X64-NEXT:   16 |   char c
// CHECK-X64-NEXT:      | [sizeof=24, align=8
// CHECK-X64-NEXT:      |  nvsize=24, nvalign=8]

int a[sizeof(T3)];
