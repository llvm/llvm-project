// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test code generation when sycl_external attribute is used

// Function defined and not used - symbols emitted
[[clang::sycl_external]] int square(int x) { return x*x; }
// CHECK: define dso_local spir_func noundef i32 @_Z6squarei

// Function defined and used - symbols emitted
[[clang::sycl_external]] int squareUsed(int x) { return x*x; }
// CHECK: define dso_local spir_func noundef i32 @_Z10squareUsedi

// Constexpr function defined and not used - symbols emitted
[[clang::sycl_external]] constexpr int squareInlined(int x) { return x*x; }
// CHECK: define linkonce_odr spir_func noundef i32 @_Z13squareInlinedi

// Function declared but not defined or used - no symbols emitted
[[clang::sycl_external]] int declOnly();
// CHECK-NOT: define {{.*}} i32 @_Z8declOnlyv


// FIXME: Function declared and used but not defined - emit external reference
[[clang::sycl_external]] void declUsed(int y);

// Function declared with the attribute and later defined - definition emitted
[[clang::sycl_external]] int func1(int arg);
int func1(int arg) { return arg; }
// CHECK: define dso_local spir_func noundef i32 @_Z5func1i

class A {
// Unused defaulted special member functions - no symbols emitted
  [[clang::sycl_external]] A& operator=(A& a) = default;
};

class B {
  [[clang::sycl_external]] virtual void BFunc1WithAttr() { int i = 1; }
// CHECK: define linkonce_odr spir_func void @_ZN1B14BFunc1WithAttrEv
  virtual void BFunc2NoAttr() { int i = 2; }
};

class C {
// Special member function defined - definition emitted
  [[clang::sycl_external]] ~C() {}
// CHECK: define linkonce_odr spir_func void @_ZN1CD1Ev
};

// Function reachable from an unused function - definition emitted
int ret1() { return 1; }
[[clang::sycl_external]] int withAttr() { return ret1(); }
// CHECK: define dso_local spir_func noundef i32 @_Z8withAttrv
// CHECK: define dso_local spir_func noundef i32 @_Z4ret1v

template <typename T>
[[clang::sycl_external]] void tFunc1(T arg) {}
// Explicit specialization defined - symbols emitted
template<>
[[clang::sycl_external]] void tFunc1<int>(int arg) {}
// CHECK: define dso_local spir_func void @_Z6tFunc1IiEvT_

template <typename T>
[[clang::sycl_external]] void tFunc2(T arg) {}
template void tFunc2<int>(int arg);
// CHECK: define weak_odr spir_func void @_Z6tFunc2IiEvT_
template<> void tFunc2<char>(char arg) {}
// CHECK: define dso_local spir_func void @_Z6tFunc2IcEvT_
template<> [[clang::sycl_external]] void  tFunc2<long>(long arg) {}
// CHECK: define dso_local spir_func void @_Z6tFunc2IlEvT_

// Test that symbols are not emitted without the sycl_external attribute
int squareNoAttr(int x) { return x*x; }
// CHECK-NOT: define {{.*}} i32 @_Z12squareNoAttri

int main() {
  declUsed(4);
  int i = squareUsed(5);
  int j = squareNoAttr(6);
  return 0;
}
