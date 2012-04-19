// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -verify -fexceptions -fcxx-exceptions -triple x86_64-linux-gnu | FileCheck %s

void h();

template<typename T> void f() noexcept(sizeof(T) == 4) { h(); }
template<typename T> void g() noexcept(sizeof(T) == 4);

template<typename T> struct S {
  static void f() noexcept(sizeof(T) == 4) { h(); }
  static void g() noexcept(sizeof(T) == 4);
};

// CHECK: define {{.*}} @_Z1fIsEvv() {
template<> void f<short>() { h(); }
// CHECK: define {{.*}} @_Z1fIA2_sEvv() nounwind {
template<> void f<short[2]>() noexcept { h(); }

// CHECK: define {{.*}} @_ZN1SIsE1fEv()
// CHECK-NOT: nounwind
template<> void S<short>::f() { h(); }
// CHECK: define {{.*}} @_ZN1SIA2_sE1fEv() nounwind
template<> void S<short[2]>::f() noexcept { h(); }

// CHECK: define {{.*}} @_Z1fIDsEvv() {
template void f<char16_t>();
// CHECK: define {{.*}} @_Z1fIA2_DsEvv() nounwind {
template void f<char16_t[2]>();

// CHECK: define {{.*}} @_ZN1SIDsE1fEv()
// CHECK-NOT: nounwind
template void S<char16_t>::f();
// CHECK: define {{.*}} @_ZN1SIA2_DsE1fEv() nounwind
template void S<char16_t[2]>::f();

void h() {
  // CHECK: define {{.*}} @_Z1fIiEvv() nounwind {
  f<int>();
  // CHECK: define {{.*}} @_Z1fIA2_iEvv() {
  f<int[2]>();

  // CHECK: define {{.*}} @_ZN1SIiE1fEv() nounwind
  S<int>::f();
  // CHECK: define {{.*}} @_ZN1SIA2_iE1fEv()
  // CHECK-NOT: nounwind
  S<int[2]>::f();

  // CHECK: define {{.*}} @_Z1fIfEvv() nounwind {
  void (*f1)() = &f<float>;
  // CHECK: define {{.*}} @_Z1fIdEvv() {
  void (*f2)() = &f<double>;

  // CHECK: define {{.*}} @_ZN1SIfE1fEv() nounwind
  void (*f3)() = &S<float>::f;
  // CHECK: define {{.*}} @_ZN1SIdE1fEv()
  // CHECK-NOT: nounwind
  void (*f4)() = &S<double>::f;

  // CHECK: define {{.*}} @_Z1fIA4_cEvv() nounwind {
  (void)&f<char[4]>;
  // CHECK: define {{.*}} @_Z1fIcEvv() {
  (void)&f<char>;

  // CHECK: define {{.*}} @_ZN1SIA4_cE1fEv() nounwind
  (void)&S<char[4]>::f;
  // CHECK: define {{.*}} @_ZN1SIcE1fEv()
  // CHECK-NOT: nounwind
  (void)&S<char>::f;
}

// CHECK: define {{.*}} @_Z1iv
void i() {
  // CHECK: declare {{.*}} @_Z1gIiEvv() nounwind
  g<int>();
  // CHECK: declare {{.*}} @_Z1gIA2_iEvv()
  // CHECK-NOT: nounwind
  g<int[2]>();

  // CHECK: declare {{.*}} @_ZN1SIiE1gEv() nounwind
  S<int>::g();
  // CHECK: declare {{.*}} @_ZN1SIA2_iE1gEv()
  // CHECK-NOT: nounwind
  S<int[2]>::g();

  // CHECK: declare {{.*}} @_Z1gIfEvv() nounwind
  void (*g1)() = &g<float>;
  // CHECK: declare {{.*}} @_Z1gIdEvv()
  // CHECK-NOT: nounwind
  void (*g2)() = &g<double>;

  // CHECK: declare {{.*}} @_ZN1SIfE1gEv() nounwind
  void (*g3)() = &S<float>::g;
  // CHECK: declare {{.*}} @_ZN1SIdE1gEv()
  // CHECK-NOT: nounwind
  void (*g4)() = &S<double>::g;

  // CHECK: declare {{.*}} @_Z1gIA4_cEvv() nounwind
  (void)&g<char[4]>;
  // CHECK: declare {{.*}} @_Z1gIcEvv()
  // CHECK-NOT: nounwind
  (void)&g<char>;

  // CHECK: declare {{.*}} @_ZN1SIA4_cE1gEv() nounwind
  (void)&S<char[4]>::g;
  // CHECK: declare {{.*}} @_ZN1SIcE1gEv()
  // CHECK-NOT: nounwind
  (void)&S<char>::g;
}

template<typename T> struct Nested {
  template<bool b, typename U> void f() noexcept(sizeof(T) == sizeof(U));
};

// CHECK: define {{.*}} @_Z1jv
void j() {
  // CHECK: declare {{.*}} @_ZN6NestedIiE1fILb1EcEEvv(
  // CHECK-NOT: nounwind
  Nested<int>().f<true, char>();
  // CHECK: declare {{.*}} @_ZN6NestedIlE1fILb0ElEEvv({{.*}}) nounwind
  Nested<long>().f<false, long>();
}
