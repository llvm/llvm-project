// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s --check-prefix CHECK --check-prefix CHECK-CXX11
// RUN: %clang_cc1 -std=c++1z -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s --check-prefix CHECK --check-prefix CHECK-CXX17

// CHECK: define {{.*}} @_Z1aPFivE(
void a(int() throw(int, float)) {}
// CHECK-CXX11: define {{.*}} @_Z1bPFivE(
// CHECK-CXX17: define {{.*}} @_Z1bPnxFivE(
void b(int() noexcept) {}
// CHECK-CXX11: define {{.*}} @_Z1cPFivE(
// CHECK-CXX17: define {{.*}} @_Z1cPnxFivE(
void c(int() throw()) {}
// CHECK: define {{.*}} @_Z1dPFivE(
void d(int() noexcept(false)) {}
// CHECK-CXX11: define {{.*}} @_Z1ePFivE(
// CHECK-CXX17: define {{.*}} @_Z1ePnxFivE(
void e(int() noexcept(true)) {}

template<bool B> void f(int() noexcept(B)) {}
// CHECK: define {{.*}} @_Z1fILb0EEvPnXT_EFivE(
template void f<false>(int());
// CHECK: define {{.*}} @_Z1fILb1EEvPnXT_EFivE(
template void f<true>(int() noexcept);

template<typename...T> void g(int() throw(T...)) {}
// CHECK: define {{.*}} @_Z1gIJEEvPtwDpT_EFivE(
template void g<>(int() noexcept);
// CHECK: define {{.*}} @_Z1gIJfEEvPtwDpT_EFivE(
template void g<float>(int());

// We consider the exception specifications in parameter and return type here
// to be different.
template<typename...T> auto h(int() throw(int, T...)) -> int (*)() throw(T..., int) { return nullptr; }
// CHECK: define {{.*}} @_Z1hIJEEPtwDpT_iEFivEPtwiS1_EFivE(
template auto h<>(int()) -> int (*)();
// CHECK: define {{.*}} @_Z1hIJfEEPtwDpT_iEFivEPtwiS1_EFivE(
template auto h<float>(int()) -> int (*)();

// FIXME: The C++11 manglings here are wrong; they should be the same as the
// C++17 manglings.
// The mangler mishandles substitutions for instantiation-dependent types that
// differ only in type sugar that is not relevant for mangling. (In this case,
// the types differ in presence/absence of ParenType nodes under the pointer.)
template<typename...T> auto i(int() throw(int, T...)) -> int (*)() throw(int, T...) { return nullptr; }
// CHECK-CXX11: define {{.*}} @_Z1iIJEEPtwiDpT_EFivEPS2_(
// CHECK-CXX17: define {{.*}} @_Z1iIJEEPtwiDpT_EFivES3_(
template auto i<>(int()) -> int (*)();
// CHECK-CXX11: define {{.*}} @_Z1iIJfEEPtwiDpT_EFivEPS2_(
// CHECK-CXX17: define {{.*}} @_Z1iIJfEEPtwiDpT_EFivES3_(
template auto i<float>(int()) -> int (*)();
