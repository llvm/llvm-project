// Check that -fsanitize-coverage-trace-args and -fsanitize-coverage-trace-ret
// reach the SanitizerCoverage pass and emit their callbacks, and that debug
// information does not change what they emit.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 -debug-info-kind=limited \
// RUN:     -fsanitize-coverage-type=3 -fsanitize-coverage-trace-args %s -o - \
// RUN:   | FileCheck %s --check-prefix=ARGS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O1 \
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

// A pointer is an IR value like any other: it is reported as the address it
// holds, and the struct it points at is not read. Whether debug information
// describes that struct makes no difference. Nothing is spilled.
// ARGS-NOT: @__sancov_offsets_
// ARGS-LABEL: define {{.*}} @takes_struct_ptr(
// ARGS-NOT: alloca
// ARGS: %[[ADDR:.*]] = ptrtoint ptr %f to i64
// ARGS: call void @__sanitizer_cov_trace_args(i64 8, i64 %[[ADDR]], ptr null)
// ARGS-LABEL: define {{.*}} @returns_scalar(
// ARGS: call void @__sanitizer_cov_trace_args(i64 4,
// ARGS-NOT: call void @__sanitizer_cov_trace_ret(

// RET-LABEL: define {{.*}} @returns_scalar(
// RET: call void @__sanitizer_cov_trace_ret(i64 4,
// RET-NOT: call void @__sanitizer_cov_trace_args(

// NONE-NOT: call void @__sanitizer_cov_trace_args(
// NONE-NOT: call void @__sanitizer_cov_trace_ret(
