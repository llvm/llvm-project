// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fclang-abi-compat=23 -emit-llvm -o - %s | FileCheck %s

// No namespace gets new tags in this file, so the mangled names must not
// depend on -fclang-abi-compat. In particular the implicit tags of a function
// depend on the substitutions that are available where its encoding is
// embedded: in two<> the namespace was already mangled as part of the first
// template argument.

namespace std2 {
inline namespace __cxx11 __attribute__((abi_tag("cxx11"))) { struct string {}; }
}
template <class X, class Y> void two() {}
template <class Y> void one() {}

std2::string ret_only() {
  struct L {};
  two<std2::string, L>();
  one<L>();
  return {};
}

inline std2::string ret_inline() {
  auto l = [] {};
  two<std2::string, decltype(l)>();
  one<decltype(l)>();
  return {};
}
void use() { ret_inline(); }

// CHECK-DAG: define {{.*}} @_Z8ret_onlyB5cxx11v(
// CHECK-DAG: define {{.*}} @_Z3twoIN4std27__cxx116stringEZ8ret_onlyvE1LEvv(
// CHECK-DAG: define {{.*}} @_Z3oneIZ8ret_onlyB5cxx11vE1LEvv(
// CHECK-DAG: define {{.*}} @_Z10ret_inlineB5cxx11v(
// CHECK-DAG: define {{.*}} @_Z3twoIN4std27__cxx116stringEZ10ret_inlinevEUlvE_Evv(
// CHECK-DAG: define {{.*}} @_Z3oneIZ10ret_inlineB5cxx11vEUlvE_Evv(
