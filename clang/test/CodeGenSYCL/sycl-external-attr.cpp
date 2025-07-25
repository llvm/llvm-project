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
[[clang::sycl_external]] int decl();

// FIXME: Function declared and used but not defined - emit external reference
[[clang::sycl_external]] void declused(int y);

// Function overload with definition - symbols emitted
[[clang::sycl_external]] int func1(int arg);
int func1(int arg) { return arg; }
// CHECK: define dso_local spir_func noundef i32 @_Z5func1i

class A {
// Defaulted special member functions - no symbols emitted
  [[clang::sycl_external]] A& operator=(A& a) = default;
};

class B {
  [[clang::sycl_external]] virtual void BFunc1WithAttr() { int i = 1; }
// CHECK: define linkonce_odr spir_func void @_ZN1B14BFunc1WithAttrEv
  virtual void BFunc2NoAttr() { int i = 2; }
};

class C {
// Special member function defined - symbols emitted
  [[clang::sycl_external]] ~C() {}
// CHECK: define linkonce_odr spir_func void @_ZN1CD1Ev
};

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

// FIXME: symbols should be emitted for the instantiation and the specialization
// tFunc2 below
template <typename T>
void tFunc2(T arg) {}
template void tFunc2<int>(int arg); // emit code for this
template<> void tFunc2<char>(char arg) {} // and this

int main() {
  declused(4);
  int i = squareUsed(5);
  return 0;
}

