// Check that -fsanitize-coverage-trace-args and -fsanitize-coverage-trace-ret
// reach the SanitizerCoverage pass and emit their callbacks.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 -debug-info-kind=limited \
// RUN:     -fsanitize-coverage-type=3 -fsanitize-coverage-trace-args %s -o - \
// RUN:   | FileCheck %s --check-prefix=ARGS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 -debug-info-kind=limited \
// RUN:     -fsanitize-coverage-type=3 -fsanitize-coverage-trace-ret %s -o - \
// RUN:   | FileCheck %s --check-prefix=RET
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 -debug-info-kind=limited \
// RUN:     -fsanitize-coverage-type=3 %s -o - | FileCheck %s --check-prefix=NONE

struct Foo {
  int a;
  long b;
};

void takes_struct_ptr(struct Foo *f) {}

int returns_scalar(int x) { return x + 1; }

// The struct pointer parameter is reported as the address of the 16-byte struct
// it points at, with the field layout of that struct. Nothing is spilled.
// ARGS: @__sancov_offsets_ = private unnamed_addr constant [4 x i64] [i64 0, i64 4, i64 8, i64 8]
// ARGS-LABEL: define {{.*}} @takes_struct_ptr(
// ARGS-NOT: alloca
// ARGS: %[[ADDR:.*]] = ptrtoint ptr %f to i64
// ARGS: call void @__sanitizer_cov_trace_args({{.*}}, i32 0, i32 16, i64 %[[ADDR]], ptr @__sancov_offsets_, i32 2)
// ARGS-LABEL: define {{.*}} @returns_scalar(
// ARGS: call void @__sanitizer_cov_trace_args(
// ARGS-NOT: call void @__sanitizer_cov_trace_ret(

// RET-LABEL: define {{.*}} @returns_scalar(
// RET: call void @__sanitizer_cov_trace_ret(
// RET-NOT: call void @__sanitizer_cov_trace_args(

// NONE-NOT: call void @__sanitizer_cov_trace_args(
// NONE-NOT: call void @__sanitizer_cov_trace_ret(
