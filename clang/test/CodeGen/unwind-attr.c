// RUN: %clang_cc1 -triple i386-unknown-unknown -fexceptions -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix NOEXC %s

int opaque();

// CHECK:       define [[INT:i.*]] @test0() {
// CHECK-NOEXC: define [[INT:i.*]] @test0() nounwind {
int test0(void) {
  return opaque();
}

// <rdar://problem/8087431>: locally infer nounwind at -O0
// CHECK:       define [[INT:i.*]] @test1() nounwind {
// CHECK-NOEXC: define [[INT:i.*]] @test1() nounwind {
int test1(void) {
  return 0;
}

// <rdar://problem/8283071>: not for weak functions
// CHECK:       define weak [[INT:i.*]] @test2() {
// CHECK-NOEXC: define weak [[INT:i.*]] @test2() nounwind {
__attribute__((weak)) int test2(void) {
  return 0;
}
