// RUN: %clang_cc1 -no-opaque-pointers %s -triple %itanium_abi_triple -std=c++20 -emit-llvm -O2 -o - | FileCheck %s

// p0388 conversions to unbounded array
// dcl.init.list/3

namespace One {
int ga[1];

// CHECK-LABEL: @_ZN3One5frob1Ev
// CHECK-NEXT: entry:
// CHECK-NEXT: ret [0 x i32]* bitcast ([1 x i32]* @_ZN3One2gaE to [0 x i32]*)
auto &frob1() {
  int(&r1)[] = ga;

  return r1;
}

// CHECK-LABEL: @_ZN3One5frob2ERA1_i
// CHECK-NEXT: entry:
// CHECK-NEXT: %0 = bitcast [1 x i32]* %arp to [0 x i32]*
// CHECK-NEXT: ret [0 x i32]* %0
auto &frob2(int (&arp)[1]) {
  int(&r2)[] = arp;

  return r2;
}
} // namespace One
