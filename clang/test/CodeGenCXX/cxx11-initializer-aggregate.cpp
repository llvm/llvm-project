// RUN: %clang_cc1 -std=c++11 -S -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck %s

struct A { int a, b; int f(); };

// CHECK: define {{.*}}@_Z3fn1i(
int fn1(int x) {
  // CHECK: %[[INITLIST:.*]] = alloca %struct.A
  // CHECK: %[[A:.*]] = getelementptr inbounds %struct.A* %[[INITLIST]], i32 0, i32 0
  // CHECK: store i32 %{{.*}}, i32* %[[A]], align 4
  // CHECK: %[[B:.*]] = getelementptr inbounds %struct.A* %[[INITLIST]], i32 0, i32 1
  // CHECK: store i32 5, i32* %[[B]], align 4
  // CHECK: call i32 @_ZN1A1fEv(%struct.A* %[[INITLIST]])
  return A{x, 5}.f();
}

struct B { int &r; int &f() { return r; } };

// CHECK: define {{.*}}@_Z3fn2Ri(
int &fn2(int &v) {
  // CHECK: %[[INITLIST2:.*]] = alloca %struct.B, align 8
  // CHECK: %[[R:.*]] = getelementptr inbounds %struct.B* %[[INITLIST2:.*]], i32 0, i32 0
  // CHECK: store i32* %{{.*}}, i32** %[[R]], align 8
  // CHECK: call i32* @_ZN1B1fEv(%struct.B* %[[INITLIST2:.*]])
  return B{v}.f();
}
