// RUN: %clang_cc1 -emit-llvm-only -triple armv7a-none-windows-eabi -fdump-record-layouts %s 2>/dev/null | FileCheck %s
// RUN: %clang_cc1 -emit-llvm-only -triple thumbv7-none-windows-eabihf -fdump-record-layouts %s 2>/dev/null | FileCheck %s

// Verify that ARM Windows with eabi environment correctly uses Microsoft
// record layout for classes with virtual inheritance.
// This is a regression test for a crash caused by incorrect record layout
// selection when TheCXXABI is Microsoft but HasMicrosoftRecordLayout was false.

class A {};
class B : virtual A {};
B b;

// CHECK: *** Dumping AST Record Layout
// CHECK: *** Dumping AST Record Layout
// CHECK:          0 | class B
// CHECK-NEXT:     0 |   (B vbtable pointer)
// CHECK-NEXT:     4 |   class A (virtual base) (empty)
// CHECK-NEXT:       | [sizeof=4, align=4,
// CHECK-NEXT:       |  nvsize=4, nvalign=4]

// Microsoft record layout should NOT print dsize= (only Itanium layout does)
// CHECK-NOT: dsize=
