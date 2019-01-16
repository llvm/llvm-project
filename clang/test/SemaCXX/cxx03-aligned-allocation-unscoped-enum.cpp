// RUN: %clang_cc1 -std=c++03 -triple x86_64-pc-linux-gnu %s \
// RUN:   -faligned-allocation -emit-llvm -o - -Wno-c++11-extensions | FileCheck %s

// Ensure Clang doesn't confuse std::align_val_t with the sized deallocation
// parameter when the enum type is unscoped. Libc++ does this in C++03 in order
// to support aligned allocation in that dialect.

using size_t = __decltype(sizeof(0));

namespace std {
enum align_val_t : size_t {};
}
_Static_assert(__is_same(__underlying_type(std::align_val_t), size_t), "");

// CHECK-LABEL: define void @_Z1fPi(
void f(int *p) {
  // CHECK-NOT: call void @_ZdlPvSt11align_val_t(
  // CHECK: call void @_ZdlPv(
  // CHECK: ret void
  delete p;
}
