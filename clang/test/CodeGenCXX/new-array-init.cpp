// RUN: %clang_cc1 -std=c++11 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

// CHECK: define void @_Z2fni
void fn(int n) {
  // CHECK: icmp ult i{{32|64}} %{{[^ ]+}}, 3
  // CHECK: store i32 1
  // CHECK: store i32 2
  // CHECK: store i32 3
  // CHECK: icmp eq i32*
  // CHECK-NEXT: br i1
  new int[n] { 1, 2, 3 };
}

// CHECK: define void @_Z15const_underflowv
void const_underflow() {
  // CHECK-NOT: icmp ult i{{32|64}} %{{[^ ]+}}, 3
  // CHECK: call noalias i8* @_Zna{{.}}(i{{32|64}} -1)
  new int[2] { 1, 2, 3 };
}

// CHECK: define void @_Z11const_exactv
void const_exact() {
  // CHECK-NOT: icmp ult i{{32|64}} %{{[^ ]+}}, 3
  // CHECK-NOT: icmp eq i32*
  new int[3] { 1, 2, 3 };
}

// CHECK: define void @_Z16const_sufficientv
void const_sufficient() {
  // CHECK-NOT: icmp ult i{{32|64}} %{{[^ ]+}}, 3
  new int[4] { 1, 2, 3 };
  // CHECK: ret void
}
