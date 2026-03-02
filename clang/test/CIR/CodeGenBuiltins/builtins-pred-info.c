// Test __builtin_expect, __builtin_expect_with_probability, and __builtin_unpredictable.
// Focus: O0 vs O2 CIR output (no cir.expect at O0), and LLVM/OGCG with -O2 -disable-llvm-passes.
// Builtin call lowering is also covered by builtin_call.cpp.
//
// RUN: %clang_cc1 -O0 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-O0
// CIR-O0-NOT: cir.expect
// RUN: %clang_cc1 -O2 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR-O2
// RUN: %clang_cc1 -O2 -disable-llvm-passes -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -O2 -disable-llvm-passes -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG

extern void __attribute__((noinline)) bar(void);

void expect(int x) {
  if (__builtin_expect(x, 0))
    bar();
}
// CIR-O0: cir.func {{.*}} @expect
// CIR-O0:   cir.if {{%.*}} {
// CIR-O0:     cir.call @bar() : () -> ()

// CIR-O2: cir.func {{.*}} @expect
// CIR-O2:   [[EXPECT:%.*]] = cir.expect({{.*}}, {{.*}}) : !s64i
// CIR-O2:   [[EXPECT_BOOL:%.*]] = cir.cast int_to_bool [[EXPECT]] : !s64i -> !cir.bool
// CIR-O2:   cir.if [[EXPECT_BOOL]]
// CIR-O2:     cir.call @bar() : () -> ()

// LLVM-LABEL: @expect
// LLVM: br i1 {{.*}}, label %[[THEN:.*]], label %[[END:.*]]
// LLVM: [[THEN]]:
// LLVM: call void @bar()

// OGCG-LABEL: @expect
// OGCG: br i1 {{.*}}, label %[[THEN:.*]], label %[[END:.*]]
// OGCG: [[THEN]]:
// OGCG: call void @bar()

void expect_with_probability(int x) {
  if (__builtin_expect_with_probability(x, 1, 0.8))
    bar();
}
// CIR-O0: cir.func {{.*}} @expect_with_probability
// CIR-O0:   cir.if {{%.*}} {
// CIR-O0:     cir.call @bar() : () -> ()

// CIR-O2:  cir.func {{.*}} @expect_with_probability
// CIR-O2:    [[EXPECT:%.*]] = cir.expect({{.*}}, {{.*}}, 8.000000e-01) : !s64i
// CIR-O2:    [[EXPECT_BOOL:%.*]] = cir.cast int_to_bool [[EXPECT]] : !s64i -> !cir.bool
// CIR-O2:    cir.if [[EXPECT_BOOL]]
// CIR-O2:      cir.call @bar() : () -> ()

// LLVM-LABEL: @expect_with_probability
// LLVM: br i1 {{.*}}, label %[[THEN:.*]], label %[[END:.*]]
// LLVM: [[THEN]]:
// LLVM: call void @bar()

// OGCG-LABEL: @expect_with_probability
// OGCG: br i1 {{.*}}, label %[[THEN:.*]], label %[[END:.*]]
// OGCG: [[THEN]]:
// OGCG: call void @bar()

void unpredictable(int x) {
  if (__builtin_unpredictable(x > 1))
    bar();
}
// CIR-O0: cir.func {{.*}} @unpredictable
// CIR-O0:   cir.if {{%.*}} {
// CIR-O0:     cir.call @bar() : () -> ()

// LLVM-LABEL: @unpredictable
// LLVM: br i1 {{.*}}, label %[[THEN:.*]], label %[[END:.*]]
// LLVM: [[THEN]]:
// LLVM: call void @bar()

// OGCG-LABEL: @unpredictable
// OGCG: br i1 {{.*}}, label %[[THEN:.*]], label %[[END:.*]]
// OGCG: [[THEN]]:
// OGCG: call void @bar()
