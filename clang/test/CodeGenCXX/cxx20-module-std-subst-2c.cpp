// RUN: %clang_cc1 -std=c++20 %S/Inputs/cxx20-module-std-subst-2a.cpp -triple %itanium_abi_triple -emit-module-interface -o %t
// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -fmodule-file=%t -emit-llvm -o - | FileCheck %s
module;
# 5 __FILE__ 1
namespace std {
template <typename A> struct char_traits {};
} // namespace std
# 9 "" 2
export module Bar;
import RenameString;

// Use Ss as this is global-module std::char_traits
// CHECK-DAG: void @_ZW3Bar1gRSs(
void g(str<char, std::char_traits<char>> &s) {
}
