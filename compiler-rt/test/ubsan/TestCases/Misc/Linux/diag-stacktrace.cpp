/// Fast unwinder does not work with Thumb code
// UNSUPPORTED: thumb
// UNSUPPORTED: android

// RUN: %clangxx -fsanitize=return %gmlt -O2 -fno-omit-frame-pointer -fasynchronous-unwind-tables %s -o %t
// RUN: %env_ubsan_opts=print_stacktrace=1:fast_unwind_on_fatal=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=print_stacktrace=1:fast_unwind_on_fatal=1 not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -fsanitize=return %gmlt -O2 -fno-omit-frame-pointer -fno-exceptions -fno-asynchronous-unwind-tables %s -o %t
// RUN: %env_ubsan_opts=print_stacktrace=1:fast_unwind_on_fatal=0 not %run %t 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=print_stacktrace=1:fast_unwind_on_fatal=1 not %run %t 2>&1 | FileCheck %s

// CHECK:      runtime error: execution reached the end of a value-returning function without returning a value
// CHECK-NEXT: #0 {{.*}}f() {{.*}}.cpp:[[#@LINE+1]]
__attribute__((noinline)) int f() {}

// CHECK-NEXT: #1 {{.*}}g() {{.*}}.cpp:[[#@LINE+1]]
__attribute__((noinline)) void g() { f(); }

// CHECK-NEXT: #2 {{.*}}h() {{.*}}.cpp:[[#@LINE+1]]
__attribute__((noinline)) void h() { g(); }

// CHECK-NEXT: #3 {{.*}}main {{.*}}.cpp:[[#@LINE+1]]
int main() { h(); }
