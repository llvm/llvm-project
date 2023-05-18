// RUN: %clang_cc1 %s -triple %itanium_abi_triple -std=c++20 -emit-llvm -O2 -o - | FileCheck %s

// p0388 conversions to unbounded array
// dcl.init.list/3

namespace One {
int ga[1];

// CHECK-LABEL: @_ZN3One5frob1Ev
// CHECK-NEXT: entry:
// CHECK-NEXT: ret ptr @_ZN3One2gaE
auto &frob1() {
  int(&r1)[] = ga;

  return r1;
}

// CHECK-LABEL: @_ZN3One5frob2ERA1_i
// CHECK-NEXT: entry:
// CHECK-NEXT: ret ptr %arp
auto &frob2(int (&arp)[1]) {
  int(&r2)[] = arp;

  return r2;
}

// CHECK-LABEL: @_ZN3One3fooEi
// CHECK-NEXT: entry:
// CHECK-NEXT: ret void
void foo(int a) {
  auto f = [](int(&&)[]) {};
  f({a});
}

} // namespace One
