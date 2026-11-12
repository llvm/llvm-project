// RUN: %clang_cc1 -x c++ -triple x86_64-pc-linux-gnu -emit-llvm < %s | FileCheck %s

struct A { A(int); ~A(); };
int f(const A &);
// CHECK: call void @_ZN1AC1Ei
// CHECK-NEXT: call noundef i32 @_Z1fRK1A
// CHECK-NEXT: call void @_ZN1AD1Ev
// CHECK: call void @_ZN1AC1Ei
// CHECK-NEXT: call noundef i32 @_Z1fRK1A
// CHECK-NEXT: call void @_ZN1AD1Ev
template<typename T> void g() {
  int a[f(3)];
  int b[f(3)];
}
int main() { g<int>(); }
