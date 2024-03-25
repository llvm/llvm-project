// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %env_asan_opts=detect_stack_use_after_return=1 not %run %t 2>&1 | FileCheck %s

// RUN: %clang_cl_asan %Od %s %Fe%t -fsanitize-address-use-after-return=always
// RUN: not %run %t 2>&1 | FileCheck %s

char *x;

void foo() {
  char stack_buffer[42];
  x = &stack_buffer[13];
}

int main() {
  foo();
  *x = 42;
// CHECK: AddressSanitizer: stack-use-after-return
// CHECK: WRITE of size 1 at {{.*}} thread T0
// CHECK-NEXT: {{#0 0x.* in main .*stack_use_after_return.cpp}}:[[@LINE-3]]
//
// CHECK: is located in stack of thread T0 at offset [[OFFSET:.*]] in frame
// CHECK-NEXT: {{#0 0x.* in foo.*stack_use_after_return.cpp}}
//
// CHECK: 'stack_buffer'{{.*}} <== Memory access at offset [[OFFSET]] is inside this variable
}
