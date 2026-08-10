// RUN: %clang %s -target arm64-apple-darwin -emit-llvm -S -fsanitize-coverage=trace-pc-guard -mllvm -sanitizer-coverage-gated-trace-callbacks=1 -o - | FileCheck %s --check-prefixes=CHECK,GATED
// RUN: %clang %s -target arm64-apple-darwin -emit-llvm -S -fsanitize-coverage=trace-pc-guard -mllvm -sanitizer-coverage-gated-trace-callbacks=0 -o - | FileCheck %s --check-prefixes=CHECK,PLAIN
// RUN: %clang %s -target arm64-apple-darwin -emit-llvm -S -fsanitize-coverage=trace-pc-guard,trace-cmp -mllvm -sanitizer-coverage-gated-trace-callbacks=1 -o - | FileCheck %s --check-prefixes=CHECK,GATED,GATEDCMP
// RUN: %clang %s -target arm64-apple-darwin -emit-llvm -S -fsanitize-coverage=trace-pc-guard,trace-cmp -mllvm -sanitizer-coverage-gated-trace-callbacks=0 -o - | FileCheck %s --check-prefixes=CHECK,PLAIN,PLAINCMP
// RUN: not %clang %s -target arm64-apple-darwin -emit-llvm -S -fsanitize-coverage=trace-pc -mllvm -sanitizer-coverage-gated-trace-callbacks=1 -o /dev/null 2>&1 | FileCheck %s --check-prefixes=INCOMPATIBLE
// RUN: not %clang %s -target arm64-apple-darwin -emit-llvm -S -fsanitize-coverage=inline-8bit-counters -mllvm -sanitizer-coverage-gated-trace-callbacks=1 -o /dev/null 2>&1 | FileCheck %s --check-prefixes=INCOMPATIBLE
// RUN: not %clang %s -target arm64-apple-darwin -emit-llvm -S -fsanitize-coverage=inline-bool-flag -mllvm -sanitizer-coverage-gated-trace-callbacks=1 -o /dev/null 2>&1 | FileCheck %s --check-prefixes=INCOMPATIBLE

// Verify that we do not emit the __sancov_gate section for "plain" trace-pc-guard
// GATED: section "__DATA,__sancov_gate"
// PLAIN-NOT: section "__DATA,__sancov_gate"

// Produce an error for all incompatible sanitizer coverage modes.
// INCOMPATIBLE: error: 'sanitizer-coverage-gated-trace-callbacks' is only supported with trace-pc-guard or trace-cmp

int x[10];

// CHECK: define{{.*}} void @foo
void foo(int n, int m) {
  // COM: Verify that we're emitting the call to __sanitizer_cov_trace_pc_guard upon
  // COM: checking the value of __sancov_should_track.
  // GATED: [[VAL:%.*]] = load i64, {{.*}}@__sancov_should_track
  // GATED-NOT: [[VAL:%.*]] = load i64, i64* @__sancov_should_track
  // GATED-NEXT: [[CMP:%.*]] = icmp ne i64 [[VAL]], 0
  // GATED-NEXT: br i1 [[CMP]], label %[[L_TRUE:.*]], label %[[L_FALSE:.*]], !prof [[WEIGHTS:!.+]]
  // GATED: [[L_TRUE]]:
  // GATED-NEXT:   call void @__sanitizer_cov_trace_pc_guard
  // COM: Check the trace-cmp instrumentation of the if (n) branch
  // GATEDCMP: [[OPERAND:%.*]] = load i32, {{.*}}
  // GATEDCMP-NEXT: br i1 [[CMP]], label %[[L_TRUE_1:.*]], label %[[L_FALSE_1:.*]]
  // GATEDCMP: [[L_TRUE_1]]:
  // GATEDCMP-NEXT:   call void @__sanitizer_cov_trace_const_cmp4(i32 0, i32 [[OPERAND]])
  // GATED:   br i1 [[CMP]], label %[[L_TRUE_2:.*]], label %[[L_FALSE_2:.*]]
  // GATED: [[L_TRUE_2]]:
  // GATED-NEXT:   call void @__sanitizer_cov_trace_pc_guard
  // GATED: [[WEIGHTS]] = !{!"branch_weights", i32 1, i32 100000}

  // COM: With the non-gated instrumentation, we should not emit the
  // COM: __sancov_should_track global.
  // PLAIN-NOT: __sancov_should_track
  // But we should still be emitting the calls to the callback.
  // PLAIN: call void @__sanitizer_cov_trace_pc_guard
  // PLAINCMP: [[OPERAND:%.*]] = load i32, {{.*}}
  // PLAINCMP-NEXT: call void @__sanitizer_cov_trace_const_cmp4(i32 0, i32 [[OPERAND]])
  if (n) {
    x[n] = 42;
    if (m) {
      x[m] = 41;
    }
  }
}