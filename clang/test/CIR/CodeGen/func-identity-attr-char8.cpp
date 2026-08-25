// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --implicit-check-not=narrow_char_params

namespace std {
inline namespace __1 {
template <class Iter, class T> Iter find(Iter first, Iter last, const T &value);
}
}

char8_t *char8_call(char8_t *first, char8_t *last, const char8_t &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPDuDuEET_S2_S2_RKT0_{{.*}} func_info<#cir.func_identity<"std::find">>

wchar_t *wchar_call(wchar_t *first, wchar_t *last, const wchar_t &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPwwEET_S2_S2_RKT0_{{.*}} func_info<#cir.func_identity<"std::find">>

char16_t *char16_call(char16_t *first, char16_t *last, const char16_t &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPDsDsEET_S2_S2_RKT0_{{.*}} func_info<#cir.func_identity<"std::find">>

char32_t *char32_call(char32_t *first, char32_t *last, const char32_t &value) {
  return std::find(first, last, value);
}
// CHECK: cir.func{{.*}} @_ZNSt3__14findIPDiDiEET_S2_S2_RKT0_{{.*}} func_info<#cir.func_identity<"std::find">>
