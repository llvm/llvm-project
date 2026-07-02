// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir

namespace std {
inline namespace __1 {
template <class Iter, class T> Iter find(Iter first, Iter last, const T &value);
}
}

char8_t *char8_call(char8_t *first, char8_t *last, const char8_t &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPDuDuEET_S2_S2_RKT0_{{.*}} func_info<#cir.func_identity<"std::find">>
