// RUN: %clangxx -fsanitize=return %gmlt %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=print_stacktrace=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-STACKTRACE
// Error message does not exact what expected
// XFAIL: target={{.*openbsd.*}}

// CHECK: missing_return.cpp:[[@LINE+1]]:5: runtime error: execution reached the end of a value-returning function without returning a value
int f() __attribute__((noinline)) {
// CHECK-STACKTRACE: #0 {{.*}}f{{.*}}missing_return.cpp:[[@LINE-1]]
}

int main(int, char **argv) {
  return f();
}
