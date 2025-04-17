// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64ec-pc-windows -fms-extensions -emit-llvm -o - %s -verify | FileCheck %s

// CHECK: ;    Function Attrs: hybrid_patchable noinline nounwind optnone
// CHECK-NEXT: define dso_local i32 @func() #0 {
int __attribute__((hybrid_patchable)) func(void) {  return 1; }

// CHECK: ;    Function Attrs: hybrid_patchable noinline nounwind optnone
// CHECK-NEXT: define dso_local i32 @func2() #0 {
int __declspec(hybrid_patchable) func2(void) {  return 2; }

// CHECK: ;    Function Attrs: hybrid_patchable noinline nounwind optnone
// CHECK-NEXT: define dso_local i32 @func3() #0 {
int __declspec(hybrid_patchable) func3(void);
int func3(void) {  return 3; }

// CHECK: ;    Function Attrs: hybrid_patchable noinline nounwind optnone
// CHECK-NEXT: define dso_local i32 @func4() #0 {
[[clang::hybrid_patchable]] int func4(void);
int func4(void) {  return 3; }

// CHECK: ; Function Attrs: hybrid_patchable noinline nounwind optnone
// CHECK-NEXT: define internal void @static_func() #0 {
// expected-warning@+1 {{'hybrid_patchable' is ignored on functions without external linkage}}
static void __declspec(hybrid_patchable) static_func(void) {}

// CHECK: ;    Function Attrs: hybrid_patchable noinline nounwind optnone
// CHECK-NEXT: define linkonce_odr dso_local i32 @func5() #0 comdat {
int inline __declspec(hybrid_patchable) func5(void) {  return 4; }

void caller(void) {
  static_func();
  func5();
}
