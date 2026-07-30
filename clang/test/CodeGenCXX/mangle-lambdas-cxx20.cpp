// RUN: %clang_cc1 -std=c++20 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s -w | FileCheck %s

template <typename T>
auto ft1() {
  return []<typename U = T>(T p1 = [] { return T{}; } (),
                            U p2 = [] { return U{}; } ()) { return p1+p2; };
}
void test_ft1() {
  // CHECK: call noundef i32 @_ZZZ3ft1IiEDavENKUlTyiT_E_clIiEEDaiS0_Ed0_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZZ3ft1IiEDavENKUlTyiT_E_clIiEEDaiS0_Ed_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZ3ft1IiEDavENKUlTyiT_E_clIiEEDaiS0_
  ft1<int>()();
}
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ3ft1IiEDavENKUlTyiT_E_clIiEEDaiS0_
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZ3ft1IiEDavENKUlTyiT_E_clIiEEDaiS0_Ed0_NKUlvE_clEv
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZ3ft1IiEDavENKUlTyiT_E_clIiEEDaiS0_Ed_NKUlvE_clEv

template <typename T>
auto vt1 = []<typename U = T>(T p1 = [] { return T{}; } (),
                              U p2 = [] { return U{}; } ()) { return p1+p2; };
void test_vt1() {
  // CHECK: call noundef i32 @_ZZNK3vt1IiEMUlTyiT_E_clIiEEDaiS1_Ed0_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZNK3vt1IiEMUlTyiT_E_clIiEEDaiS1_Ed_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZNK3vt1IiEMUlTyiT_E_clIiEEDaiS1_
  vt1<int>();
}
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK3vt1IiEMUlTyiT_E_clIiEEDaiS1_
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZNK3vt1IiEMUlTyiT_E_clIiEEDaiS1_Ed0_NKUlvE_clEv
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZNK3vt1IiEMUlTyiT_E_clIiEEDaiS1_Ed_NKUlvE_clEv
