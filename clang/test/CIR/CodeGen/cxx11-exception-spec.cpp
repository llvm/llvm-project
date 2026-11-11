// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fexceptions -fcxx-exceptions -verify -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fexceptions -fcxx-exceptions -verify -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fexceptions -fcxx-exceptions -verify -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Without exceptions every definition is nounwind, whatever its exception
// specification.
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fclangir -emit-llvm %s -o %t-noeh-cir.ll
// RUN: FileCheck --input-file=%t-noeh-cir.ll %s -check-prefix=LLVM-NOEH
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o %t-noeh.ll
// RUN: FileCheck --input-file=%t-noeh.ll %s -check-prefix=LLVM-NOEH

// LLVM-NOEH-NOT: attributes #{{[0-9]+}} = { mustprogress noinline optnone
// LLVM-NOEH: attributes #{{[0-9]+}} = { mustprogress noinline nounwind optnone
// LLVM-NOEH-NOT: attributes #{{[0-9]+}} = { mustprogress noinline optnone
// expected-no-diagnostics

void h();

template<typename T> void f() noexcept(sizeof(T) == 4) { h(); }
template<typename T> void g() noexcept(sizeof(T) == 4);

template<typename T> struct S {
  static void f() noexcept(sizeof(T) == 4) { h(); }
  static void g() noexcept(sizeof(T) == 4);
};

// LLVM: define {{.*}} @_Z1fIsEvv() [[NONE:#[0-9]+]] {
// CIR: cir.func{{.*}} @_Z1fIsEvv() attributes {{{.*}}mustprogress, optnone} {
template<> void f<short>() { h(); }
// LLVM: define {{.*}} @_Z1fIA2_sEvv() [[NUW:#[0-9]+]] personality ptr @__gxx_personality_v0 {
// CIR: cir.func{{.*}} @_Z1fIA2_sEvv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
template<> void f<short[2]>() noexcept { h(); }

// LLVM: define {{.*}} @_ZN1SIsE1fEv()
// CIR: cir.func{{.*}} @_ZN1SIsE1fEv() attributes {{{.*}}mustprogress, optnone} {
// LLVM-NOT: [[NUW]]
template<> void S<short>::f() { h(); }
// LLVM: define {{.*}} @_ZN1SIA2_sE1fEv() [[NUW]]
// CIR: cir.func{{.*}} @_ZN1SIA2_sE1fEv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
template<> void S<short[2]>::f() noexcept { h(); }

// LLVM: define {{.*}} @_Z1fIDsEvv() [[NONE]] comdat {
// CIR: cir.func{{.*}} @_Z1fIDsEvv() attributes {{{.*}}mustprogress, optnone} {
template void f<char16_t>();
// LLVM: define {{.*}} @_Z1fIA2_DsEvv() [[NUW]] comdat personality ptr @__gxx_personality_v0 {
// CIR: cir.func{{.*}} @_Z1fIA2_DsEvv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
template void f<char16_t[2]>();

// LLVM: define {{.*}} @_ZN1SIDsE1fEv()
// CIR: cir.func{{.*}} @_ZN1SIDsE1fEv() attributes {{{.*}}mustprogress, optnone} {
// LLVM-NOT: [[NUW]]
template void S<char16_t>::f();
// LLVM: define {{.*}} @_ZN1SIA2_DsE1fEv() [[NUW]]
// CIR: cir.func{{.*}} @_ZN1SIA2_DsE1fEv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
template void S<char16_t[2]>::f();

void h() {
  // LLVM: define {{.*}} @_Z1fIiEvv() [[NUW]] comdat personality ptr @__gxx_personality_v0 {
  // CIR: cir.func{{.*}} @_Z1fIiEvv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
  f<int>();
  // LLVM: define {{.*}} @_Z1fIA2_iEvv() [[NONE]] comdat {
  // CIR: cir.func{{.*}} @_Z1fIA2_iEvv() attributes {{{.*}}mustprogress, optnone} {
  f<int[2]>();

  // LLVM: define {{.*}} @_ZN1SIiE1fEv() [[NUW]]
  // CIR: cir.func{{.*}} @_ZN1SIiE1fEv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
  S<int>::f();
  // LLVM: define {{.*}} @_ZN1SIA2_iE1fEv()
  // CIR: cir.func{{.*}} @_ZN1SIA2_iE1fEv() attributes {{{.*}}mustprogress, optnone} {
  // LLVM-NOT: [[NUW]]
  S<int[2]>::f();

  // LLVM: define {{.*}} @_Z1fIfEvv() [[NUW]] comdat  personality ptr @__gxx_personality_v0 {
  // CIR: cir.func{{.*}} @_Z1fIfEvv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
  void (*f1)() = &f<float>;
  // LLVM: define {{.*}} @_Z1fIdEvv() [[NONE]] comdat {
  // CIR: cir.func{{.*}} @_Z1fIdEvv() attributes {{{.*}}mustprogress, optnone} {
  void (*f2)() = &f<double>;

  // LLVM: define {{.*}} @_ZN1SIfE1fEv() [[NUW]]
  // CIR: cir.func{{.*}} @_ZN1SIfE1fEv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
  void (*f3)() = &S<float>::f;
  // LLVM: define {{.*}} @_ZN1SIdE1fEv()
  // CIR: cir.func{{.*}} @_ZN1SIdE1fEv() attributes {{{.*}}mustprogress, optnone} {
  // LLVM-NOT: [[NUW]]
  void (*f4)() = &S<double>::f;

  // LLVM: define {{.*}} @_Z1fIA4_cEvv() [[NUW]] comdat personality ptr @__gxx_personality_v0 {
  // CIR: cir.func{{.*}} @_Z1fIA4_cEvv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
  (void)&f<char[4]>;
  // LLVM: define {{.*}} @_Z1fIcEvv() [[NONE]] comdat {
  // CIR: cir.func{{.*}} @_Z1fIcEvv() attributes {{{.*}}mustprogress, optnone} {
  (void)&f<char>;

  // LLVM: define {{.*}} @_ZN1SIA4_cE1fEv() [[NUW]]
  // CIR: cir.func{{.*}} @_ZN1SIA4_cE1fEv() personality(@__gxx_personality_v0) attributes {{{.*}}mustprogress, nothrow, optnone} {
  (void)&S<char[4]>::f;
  // LLVM: define {{.*}} @_ZN1SIcE1fEv()
  // CIR: cir.func{{.*}} @_ZN1SIcE1fEv() attributes {{{.*}}mustprogress, optnone} {
  // LLVM-NOT: [[NUW]]
  (void)&S<char>::f;
}

// LLVM: define {{.*}} @_Z1iv
void i() {
  // LLVM: declare {{.*}} @_Z1gIiEvv() [[NUW2:#[0-9]+]]
  // CIR: cir.func private @_Z1gIiEvv() attributes {"cir.target-features" = "{{[^"]*}}", nothrow}
  g<int>();
  // LLVM: declare {{.*}} @_Z1gIA2_iEvv()
  // CIR: cir.func private @_Z1gIA2_iEvv() attributes {"cir.target-features" = "{{[^"]*}}"}
  // LLVM-NOT: [[NUW]]
  g<int[2]>();

  // LLVM: declare {{.*}} @_ZN1SIiE1gEv() [[NUW2]]
  // CIR: cir.func private @_ZN1SIiE1gEv() attributes {"cir.target-features" = "{{[^"]*}}", nothrow}
  S<int>::g();
  // LLVM: declare {{.*}} @_ZN1SIA2_iE1gEv()
  // CIR: cir.func private @_ZN1SIA2_iE1gEv() attributes {"cir.target-features" = "{{[^"]*}}"}
  // LLVM-NOT: [[NUW]]
  S<int[2]>::g();

  // LLVM: declare {{.*}} @_Z1gIfEvv() [[NUW2]]
  // CIR: cir.func private @_Z1gIfEvv() attributes {"cir.target-features" = "{{[^"]*}}", nothrow}
  void (*g1)() = &g<float>;
  // LLVM: declare {{.*}} @_Z1gIdEvv()
  // CIR: cir.func private @_Z1gIdEvv() attributes {"cir.target-features" = "{{[^"]*}}"}
  // LLVM-NOT: [[NUW]]
  void (*g2)() = &g<double>;

  // LLVM: declare {{.*}} @_ZN1SIfE1gEv() [[NUW2]]
  // CIR: cir.func private @_ZN1SIfE1gEv() attributes {"cir.target-features" = "{{[^"]*}}", nothrow}
  void (*g3)() = &S<float>::g;
  // LLVM: declare {{.*}} @_ZN1SIdE1gEv()
  // CIR: cir.func private @_ZN1SIdE1gEv() attributes {"cir.target-features" = "{{[^"]*}}"}
  // LLVM-NOT: [[NUW]]
  void (*g4)() = &S<double>::g;

  // LLVM: declare {{.*}} @_Z1gIA4_cEvv() [[NUW2]]
  // CIR: cir.func private @_Z1gIA4_cEvv() attributes {"cir.target-features" = "{{[^"]*}}", nothrow}
  (void)&g<char[4]>;
  // LLVM: declare {{.*}} @_Z1gIcEvv()
  // CIR: cir.func private @_Z1gIcEvv() attributes {"cir.target-features" = "{{[^"]*}}"}
  // LLVM-NOT: [[NUW]]
  (void)&g<char>;

  // LLVM: declare {{.*}} @_ZN1SIA4_cE1gEv() [[NUW2]]
  // CIR: cir.func private @_ZN1SIA4_cE1gEv() attributes {"cir.target-features" = "{{[^"]*}}", nothrow}
  (void)&S<char[4]>::g;
  // LLVM: declare {{.*}} @_ZN1SIcE1gEv()
  // CIR: cir.func private @_ZN1SIcE1gEv() attributes {"cir.target-features" = "{{[^"]*}}"}
  // LLVM-NOT: [[NUW]]
  (void)&S<char>::g;
}

template<typename T> struct Nested {
  template<bool b, typename U> void f() noexcept(sizeof(T) == sizeof(U));
};

// LLVM: define {{.*}} @_Z1jv
void j() {
  // LLVM: declare {{.*}} @_ZN6NestedIiE1fILb1EcEEvv(
  // CIR: cir.func private @_ZN6NestedIiE1fILb1EcEEvv({{.*}}) attributes {"cir.target-features" = "{{[^"]*}}"}
  // LLVM-NOT: [[NUW]]
  Nested<int>().f<true, char>();
  // LLVM: declare {{.*}} @_ZN6NestedIlE1fILb0ElEEvv({{.*}}) [[NUW2]]
  // CIR: cir.func private @_ZN6NestedIlE1fILb0ElEEvv({{.*}}) attributes {"cir.target-features" = "{{[^"]*}}", nothrow}
  Nested<long>().f<false, long>();
}

// LLVM: attributes [[NONE]] = { {{.*}} }
// LLVM: attributes [[NUW]] = { mustprogress noinline nounwind{{.*}} }
// LLVM: attributes [[NUW2]] = { nounwind{{.*}} }



namespace PR19190 {
template <class T> struct DWFIterator { virtual void get() throw(int) = 0; };
void foo(DWFIterator<int> *foo) { foo->get(); }
}
