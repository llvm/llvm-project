// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -Wno-dynamic-exception-spec | FileCheck %s --check-prefix CHECK --check-prefix CHECK-CXX11
// RUN: %clang_cc1 -std=c++1z -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -Wno-dynamic-exception-spec | FileCheck %s --check-prefix CHECK --check-prefix CHECK-CXX17

// CHECK: define {{.*}} @_Z1aPFivE(
void a(int() throw(int, float)) {}
// CHECK-CXX11: define {{.*}} @_Z1bPFivE(
// CHECK-CXX17: define {{.*}} @_Z1bPDoFivE(
void b(int() noexcept) {}
// CHECK-CXX11: define {{.*}} @_Z1cPFivE(
// CHECK-CXX17: define {{.*}} @_Z1cPDoFivE(
void c(int() throw()) {}
// CHECK: define {{.*}} @_Z1dPFivE(
void d(int() noexcept(false)) {}
// CHECK-CXX11: define {{.*}} @_Z1ePFivE(
// CHECK-CXX17: define {{.*}} @_Z1ePDoFivE(
void e(int() noexcept(true)) {}

template<bool B> void f(int() noexcept(B)) {}
// CHECK: define {{.*}} @_Z1fILb0EEvPDOT_EFivE(
template void f<false>(int());
// CHECK: define {{.*}} @_Z1fILb1EEvPDOT_EFivE(
template void f<true>(int() noexcept);

template<typename...T> void g(int() throw(T...)) {}
// CHECK: define {{.*}} @_Z1gIJEEvPDwDpT_EFivE(
template void g<>(int() noexcept);
// CHECK: define {{.*}} @_Z1gIJfEEvPDwDpT_EFivE(
template void g<float>(int());

// We consider the exception specifications in parameter and return type here
// to be different.
template<typename...T> auto h(int() throw(int, T...)) -> int (*)() throw(T..., int) { return nullptr; }
// CHECK: define {{.*}} @_Z1hIJEEPDwDpT_iEFivEPDwiS1_EFivE(
template auto h<>(int()) -> int (*)();
// CHECK: define {{.*}} @_Z1hIJfEEPDwDpT_iEFivEPDwiS1_EFivE(
template auto h<float>(int()) -> int (*)();

template<typename...T> auto i(int() throw(int, T...)) -> int (*)() throw(int, T...) { return nullptr; }
// CHECK: define {{.*}} @_Z1iIJEEPDwiDpT_EFivES3_(
template auto i<>(int()) -> int (*)();
// CHECK: define {{.*}} @_Z1iIJfEEPDwiDpT_EFivES3_(
template auto i<float>(int()) -> int (*)();
