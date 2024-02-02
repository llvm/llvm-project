// RUN: %clang_cc1 -O0 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

extern void __attribute__((noinline)) bar(void);

void expect(int x) {
  if (__builtin_expect(x, 0))
    bar();
}
// CHECK: cir.func @expect
// CHECK:   cir.if {{%.*}} {
// CHECK:     cir.call @bar() : () -> ()

void expect_with_probability(int x) {
  if (__builtin_expect_with_probability(x, 1, 0.8))
    bar();
}
// CHECK: cir.func @expect_with_probability
// CHECK:   cir.if {{%.*}} {
// CHECK:     cir.call @bar() : () -> ()

void unpredictable(int x) {
  if (__builtin_unpredictable(x > 1))
    bar();
// CHECK: cir.func @unpredictable
// CHECK:   cir.if {{%.*}} {
// CHECK:     cir.call @bar() : () -> ()
}
