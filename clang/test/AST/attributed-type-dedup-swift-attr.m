// RUN: %clang_cc1 -fobjc-arc -emit-pch -o %t.pch %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t.pch | FileCheck %s

// Checks that swift_attr deduplication works correctly.
// The following test should generate two attributed types, not three.

#define ATTR_A __attribute__((swift_attr("@A")))
#define ATTR_B __attribute__((swift_attr("@B")))

void f1(int * ATTR_A p);
void f2(int * ATTR_A p);
void f3(int * ATTR_B p);

// CHECK-COUNT-2: <TYPE_ATTRIBUTED
// CHECK-NOT:     <TYPE_ATTRIBUTED
