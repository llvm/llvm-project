// RUN: %clang_cc1 -std=c++23 -emit-llvm -triple x86_64 %s -o - | FileCheck %s

void func1([[clang::readnone]] int*) {
  // CHECK: @_Z5func1Pi(ptr noundef readnone %0)
}
void func2([[clang::readnone]] int&) {
  // CHECK: @_Z5func2Ri(ptr noundef nonnull readnone align 4 dereferenceable(4) %0)
}

void func3([[clang::readonly]] int*) {
  // CHECK: @_Z5func3Pi(ptr noundef readonly %0)
}
void func4([[clang::readonly]] int&) {
  // CHECK: @_Z5func4Ri(ptr noundef nonnull readonly align 4 dereferenceable(4) %0)
}

void func5([[clang::writeonly]] int*) {
  // CHECK: @_Z5func5Pi(ptr noundef writeonly %0)
}
void func6([[clang::writeonly]] int&) {
  // CHECK: @_Z5func6Ri(ptr noundef nonnull writeonly align 4 dereferenceable(4) %0)
}

struct S {
  void func1() [[clang::readnone]];
  void func2() [[clang::readonly]];
  void func3() [[clang::writeonly]];
};

void S::func1() [[clang::readnone]] {
  // CHECK: @_ZN1S5func1Ev(ptr noundef nonnull readnone align 1 dereferenceable(1) %this)
}
void S::func2() [[clang::readonly]] {
  // CHECK: @_ZN1S5func2Ev(ptr noundef nonnull readonly align 1 dereferenceable(1) %this)
}
void S::func3() [[clang::writeonly]] {
  // CHECK: @_ZN1S5func3Ev(ptr noundef nonnull writeonly align 1 dereferenceable(1) %this)
}
