// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

namespace test0 {
  extern id x;

  struct A {
    id x;
    A();
  };
  A::A() : x(test0::x) {}

// CHECK-LABEL:    define{{.*}} void @_ZN5test01AC2Ev(
// CHECK:      [[THIS:%.*]] = alloca ptr, align 8
// CHECK-NEXT: store 
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[THIS]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[TEST0:%.*]], ptr [[T0]], i32 0, i32 0
// CHECK-NEXT: [[T2:%.*]] = load ptr, ptr @_ZN5test01xE
// CHECK-NEXT: call ptr @objc_assign_strongCast(ptr [[T2]], ptr [[T1]])
// CHECK-NEXT: ret void
}
