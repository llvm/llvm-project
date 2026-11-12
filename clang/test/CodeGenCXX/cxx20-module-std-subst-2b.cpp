// RUN: %clang_cc1 -std=c++20 %S/Inputs/cxx20-module-std-subst-2a.cpp -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s

export module Foo;
import RenameString;

namespace std {
template <typename T> struct char_traits {};
} // namespace std

// use Sb mangling, not Ss as this is not global-module std::char_traits
// std::char_traits.
// CHECK-DAG: void @_ZW3Foo1fRSbIcStS_11char_traitsIcESaIcEE(
void f(str<char, std::char_traits<char>> &s) {
}
