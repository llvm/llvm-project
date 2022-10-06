// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s -w | FileCheck %s

template<typename T = int>
auto ft1() {
  return [](int p = [] { return 0; } ()) { return p; };
}
void test_ft1() {
  // CHECK: call noundef i32 @_ZZZ3ft1IiEDavENKUliE_clEiEd_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZ3ft1IiEDavENKUliE_clEi
  ft1<>()();
}
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ3ft1IiEDavENKUliE_clEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZ3ft1IiEDavENKUliE_clEiEd_NKUlvE_clEv

template <typename T>
auto ft2() {
  struct S {
    T operator()(T p = []{ return 0; }()) const { return p; }
  };
  return S{};
}
void test_ft2() {
  // CHECK: call noundef i32 @_ZZZ3ft2IiEDavENK1SclEiEd_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZZ3ft2IiEDavENK1SclEi
  ft2<int>()();
}
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZ3ft2IiEDavENK1SclEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZZ3ft2IiEDavENK1SclEiEd_NKUlvE_clEv

template <typename>
auto vt1 = [](int p = [] { return 0; } ()) { return p; };
void test_vt1() {
  // CHECK: call noundef i32 @_ZZNK3vt1IiEMUliE_clEiEd_NKUlvE_clEv
  // CHECK: call noundef i32 @_ZNK3vt1IiEMUliE_clEi
  vt1<int>();
}
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZNK3vt1IiEMUliE_clEi
// CHECK-LABEL: define linkonce_odr noundef i32 @_ZZNK3vt1IiEMUliE_clEiEd_NKUlvE_clEv
