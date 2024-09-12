// RUN: %clang_cc1 -O0 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-O0
// RUN: %clang_cc1 -O2 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-O2

extern void __attribute__((noinline)) bar(void);

void expect(int x) {
  if (__builtin_expect(x, 0))
    bar();
}
// CIR-O0: cir.func @expect
// CIR-O0:   cir.if {{%.*}} {
// CIR-O0:     cir.call @bar() : () -> ()

// CIR-O2: cir.func @expect
// CIR-O2:   [[EXPECT:%.*]] = cir.expect({{.*}}, {{.*}}) : !s64i
// CIR-O2:   [[EXPECT_BOOL:%.*]] = cir.cast(int_to_bool, [[EXPECT]] : !s64i), !cir.bool
// CIR-O2:   cir.if [[EXPECT_BOOL]]
// CIR-O2:     cir.call @bar() : () -> ()

void expect_with_probability(int x) {
  if (__builtin_expect_with_probability(x, 1, 0.8))
    bar();
}
// CIR-O0: cir.func @expect_with_probability
// CIR-O0:   cir.if {{%.*}} {
// CIR-O0:     cir.call @bar() : () -> ()

// CIR-O2:  cir.func @expect_with_probability
// CIR-O2:    [[EXPECT:%.*]] = cir.expect({{.*}}, {{.*}}, 8.000000e-01) : !s64i
// CIR-O2:    [[EXPECT_BOOL:%.*]] = cir.cast(int_to_bool, [[EXPECT]] : !s64i), !cir.bool
// CIR-O2:    cir.if [[EXPECT_BOOL]]
// CIR-O2:      cir.call @bar() : () -> ()

void unpredictable(int x) {
  if (__builtin_unpredictable(x > 1))
    bar();
// CIR-O0: cir.func @unpredictable
// CIR-O0:   cir.if {{%.*}} {
// CIR-O0:     cir.call @bar() : () -> ()
}
