// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s | FileCheck %s

namespace test0 {
  extern id x;

  struct A {
    id x;
    A();
  };
  A::A() : x(test0::x) {}

// CHECK-LABEL:    define{{.*}} void @_ZN5test01AC2Ev(
// CHECK:      [[THIS:%.*]] = alloca [[TEST0:%.*]]*, align 8
// CHECK-NEXT: store 
// CHECK-NEXT: [[T0:%.*]] = load [[TEST0]]*, [[TEST0]]** [[THIS]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [[TEST0]], [[TEST0]]* [[T0]], i32 0, i32 0
// CHECK-NEXT: [[T2:%.*]] = load i8*, i8** @_ZN5test01xE
// CHECK-NEXT: call i8* @objc_assign_strongCast(i8* [[T2]], i8** [[T1]])
// CHECK-NEXT: ret void
}
