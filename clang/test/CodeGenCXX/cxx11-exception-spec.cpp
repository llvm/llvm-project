// RUN: %clang_cc1 -no-opaque-pointers -std=c++11 -emit-llvm %s -o - -verify -fexceptions -fcxx-exceptions -triple x86_64-linux-gnu | FileCheck %s
// expected-no-diagnostics

void h();

template<typename T> void f() noexcept(sizeof(T) == 4) { h(); }
template<typename T> void g() noexcept(sizeof(T) == 4);

template<typename T> struct S {
  static void f() noexcept(sizeof(T) == 4) { h(); }
  static void g() noexcept(sizeof(T) == 4);
};

// CHECK: define {{.*}} @_Z1fIsEvv() [[NONE:#[0-9]+]] {
template<> void f<short>() { h(); }
// CHECK: define {{.*}} @_Z1fIA2_sEvv() [[NUW:#[0-9]+]] personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
template<> void f<short[2]>() noexcept { h(); }

// CHECK: define {{.*}} @_ZN1SIsE1fEv()
// CHECK-NOT: [[NUW]]
template<> void S<short>::f() { h(); }
// CHECK: define {{.*}} @_ZN1SIA2_sE1fEv() [[NUW]]
template<> void S<short[2]>::f() noexcept { h(); }

// CHECK: define {{.*}} @_Z1fIDsEvv() [[NONE]] comdat {
template void f<char16_t>();
// CHECK: define {{.*}} @_Z1fIA2_DsEvv() [[NUW]] comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
template void f<char16_t[2]>();

// CHECK: define {{.*}} @_ZN1SIDsE1fEv()
// CHECK-NOT: [[NUW]]
template void S<char16_t>::f();
// CHECK: define {{.*}} @_ZN1SIA2_DsE1fEv() [[NUW]]
template void S<char16_t[2]>::f();

void h() {
  // CHECK: define {{.*}} @_Z1fIiEvv() [[NUW]] comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  f<int>();
  // CHECK: define {{.*}} @_Z1fIA2_iEvv() [[NONE]] comdat {
  f<int[2]>();

  // CHECK: define {{.*}} @_ZN1SIiE1fEv() [[NUW]]
  S<int>::f();
  // CHECK: define {{.*}} @_ZN1SIA2_iE1fEv()
  // CHECK-NOT: [[NUW]]
  S<int[2]>::f();

  // CHECK: define {{.*}} @_Z1fIfEvv() [[NUW]] comdat  personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  void (*f1)() = &f<float>;
  // CHECK: define {{.*}} @_Z1fIdEvv() [[NONE]] comdat {
  void (*f2)() = &f<double>;

  // CHECK: define {{.*}} @_ZN1SIfE1fEv() [[NUW]]
  void (*f3)() = &S<float>::f;
  // CHECK: define {{.*}} @_ZN1SIdE1fEv()
  // CHECK-NOT: [[NUW]]
  void (*f4)() = &S<double>::f;

  // CHECK: define {{.*}} @_Z1fIA4_cEvv() [[NUW]] comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  (void)&f<char[4]>;
  // CHECK: define {{.*}} @_Z1fIcEvv() [[NONE]] comdat {
  (void)&f<char>;

  // CHECK: define {{.*}} @_ZN1SIA4_cE1fEv() [[NUW]]
  (void)&S<char[4]>::f;
  // CHECK: define {{.*}} @_ZN1SIcE1fEv()
  // CHECK-NOT: [[NUW]]
  (void)&S<char>::f;
}

// CHECK: define {{.*}} @_Z1iv
void i() {
  // CHECK: declare {{.*}} @_Z1gIiEvv() [[NUW2:#[0-9]+]]
  g<int>();
  // CHECK: declare {{.*}} @_Z1gIA2_iEvv()
  // CHECK-NOT: [[NUW]]
  g<int[2]>();

  // CHECK: declare {{.*}} @_ZN1SIiE1gEv() [[NUW2]]
  S<int>::g();
  // CHECK: declare {{.*}} @_ZN1SIA2_iE1gEv()
  // CHECK-NOT: [[NUW]]
  S<int[2]>::g();

  // CHECK: declare {{.*}} @_Z1gIfEvv() [[NUW2]]
  void (*g1)() = &g<float>;
  // CHECK: declare {{.*}} @_Z1gIdEvv()
  // CHECK-NOT: [[NUW]]
  void (*g2)() = &g<double>;

  // CHECK: declare {{.*}} @_ZN1SIfE1gEv() [[NUW2]]
  void (*g3)() = &S<float>::g;
  // CHECK: declare {{.*}} @_ZN1SIdE1gEv()
  // CHECK-NOT: [[NUW]]
  void (*g4)() = &S<double>::g;

  // CHECK: declare {{.*}} @_Z1gIA4_cEvv() [[NUW2]]
  (void)&g<char[4]>;
  // CHECK: declare {{.*}} @_Z1gIcEvv()
  // CHECK-NOT: [[NUW]]
  (void)&g<char>;

  // CHECK: declare {{.*}} @_ZN1SIA4_cE1gEv() [[NUW2]]
  (void)&S<char[4]>::g;
  // CHECK: declare {{.*}} @_ZN1SIcE1gEv()
  // CHECK-NOT: [[NUW]]
  (void)&S<char>::g;
}

template<typename T> struct Nested {
  template<bool b, typename U> void f() noexcept(sizeof(T) == sizeof(U));
};

// CHECK: define {{.*}} @_Z1jv
void j() {
  // CHECK: declare {{.*}} @_ZN6NestedIiE1fILb1EcEEvv(
  // CHECK-NOT: [[NUW]]
  Nested<int>().f<true, char>();
  // CHECK: declare {{.*}} @_ZN6NestedIlE1fILb0ElEEvv({{.*}}) [[NUW2]]
  Nested<long>().f<false, long>();
}

// CHECK: attributes [[NONE]] = { {{.*}} }
// CHECK: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
// CHECK: attributes [[NUW2]] = { nounwind{{.*}} }



namespace PR19190 {
template <class T> struct DWFIterator { virtual void get() throw(int) = 0; };
void foo(DWFIterator<int> *foo) { foo->get(); }
}
