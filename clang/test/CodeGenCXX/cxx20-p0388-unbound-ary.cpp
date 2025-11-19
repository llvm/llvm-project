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

constexpr int gh151716() {
  int(&&g)[]{0,1,2};
  return g[2];
}
// CHECK-LABEL: @_ZN3One10gh151716_fEv
// CHECK-NEXT: entry:
// CHECK-NEXT:   %v = alloca i32, align 4
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(ptr nonnull %v)
// CHECK-NEXT:   store volatile i32 2, ptr %v, align 4
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(ptr nonnull %v)
// CHECK-NEXT:   ret void
void gh151716_f() {
  volatile const int v = gh151716();
}

} // namespace One
