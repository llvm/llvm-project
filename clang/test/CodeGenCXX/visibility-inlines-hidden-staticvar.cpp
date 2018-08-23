// RUN: %clang_cc1 -triple i386-unknown-unknown -std=c++11 -fvisibility-inlines-hidden -emit-llvm -o - %s -O2 -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -std=c++11 -emit-llvm -o - %s -O2 -disable-llvm-passes | FileCheck -check-prefixes=CHECK-NO-VIH %s

// When a function is hidden due to -fvisibility-inlines-hidden option, static local variables of the function should not be hidden by the option.

// CHECK-DAG: @_ZZ4funcvE3var = internal global i32 0
// CHECK-DAG: @_ZZ11hidden_funcvE3var = internal global i32 0
// CHECK-DAG: @_ZZ12default_funcvE3var = internal global i32 0
// CHECK-DAG: @_ZZ11inline_funcvE3var = linkonce_odr global i32 0, comdat
// CHECK-DAG: @_ZZ18inline_hidden_funcvE3var = linkonce_odr hidden global i32 0, comdat
// CHECK-DAG: @_ZZ19inline_default_funcvE3var = linkonce_odr global i32 0, comdat
// CHECK-DAG: define i32 @_Z4funcv()
// CHECK-DAG: define hidden i32 @_Z11hidden_funcv()
// CHECK-DAG: define i32 @_Z12default_funcv()
// CHECK-DAG: define linkonce_odr hidden i32 @_Z11inline_funcv()
// CHECK-DAG: define linkonce_odr hidden i32 @_Z18inline_hidden_funcv()
// CHECK-DAG: define linkonce_odr i32 @_Z19inline_default_funcv()

// CHECK-NO-VIH-DAG: @_ZZ4funcvE3var = internal global i32 0
// CHECK-NO-VIH-DAG: @_ZZ11hidden_funcvE3var = internal global i32 0
// CHECK-NO-VIH-DAG: @_ZZ12default_funcvE3var = internal global i32 0
// CHECK-NO-VIH-DAG: @_ZZ11inline_funcvE3var = linkonce_odr global i32 0, comdat
// CHECK-NO-VIH-DAG: @_ZZ18inline_hidden_funcvE3var = linkonce_odr hidden global i32 0, comdat
// CHECK-NO-VIH-DAG: @_ZZ19inline_default_funcvE3var = linkonce_odr global i32 0, comdat
// CHECK-NO-VIH-DAG: define i32 @_Z4funcv()
// CHECK-NO-VIH-DAG: define hidden i32 @_Z11hidden_funcv()
// CHECK-NO-VIH-DAG: define i32 @_Z12default_funcv()
// CHECK-NO-VIH-DAG: define linkonce_odr i32 @_Z11inline_funcv()
// CHECK-NO-VIH-DAG: define linkonce_odr hidden i32 @_Z18inline_hidden_funcv()
// CHECK-NO-VIH-DAG: define linkonce_odr i32 @_Z19inline_default_funcv()


int func(void) {
  static int var = 0;
  return var++;
}
inline int inline_func(void) {
  static int var = 0;
  return var++;
}
int __attribute__((visibility("hidden"))) hidden_func(void) {
  static int var = 0;
  return var++;
}
inline int __attribute__((visibility("hidden"))) inline_hidden_func(void) {
  static int var = 0;
  return var++;
}
int __attribute__((visibility("default"))) default_func(void) {
  static int var = 0;
  return var++;
}
inline int __attribute__((visibility("default"))) inline_default_func(void) {
  static int var = 0;
  return var++;
}
void bar(void) {
  func();
  inline_func();
  hidden_func();
  inline_hidden_func();
  default_func();
  inline_default_func();
}

