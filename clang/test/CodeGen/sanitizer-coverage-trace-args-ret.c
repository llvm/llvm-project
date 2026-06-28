// Test that -fsanitize-coverage=trace-args and trace-ret emit the expected callbacks.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize-coverage-trace-args -fsanitize-coverage-type=3 -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=CHECK-ARGS
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fsanitize-coverage-trace-ret -fsanitize-coverage-type=3 -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=CHECK-RET

struct Foo {
  int a;
  long b;
};

void takes_struct_ptr(struct Foo *f) {
}

int returns_scalar(int x) {
  return x + 1;
}

// CHECK-ARGS: call void @__sanitizer_cov_trace_args
// CHECK-RET: call void @__sanitizer_cov_trace_ret
