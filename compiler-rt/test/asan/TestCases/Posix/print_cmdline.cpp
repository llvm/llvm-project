// Check that ASan can print reproducer cmdline for failed binary if desired.
//
// RUN: %clang_asan %s -o %t-exe
//
// RUN: env not %run %t-exe 2>&1 | FileCheck %s
// RUN: %env_asan_opts=print_cmdline=false not %run %t-exe 2>&1 | FileCheck %s
// RUN: %env_asan_opts=print_cmdline=true not %run %t-exe first second/third [fourth] 2>&1 | FileCheck %s --check-prefix CHECK-PRINT

volatile int ten = 10;

int main() {
  char x[10];
  // CHECK-NOT: Command:
  // CHECK-PRINT: {{Command: .*-exe first second/third \[fourth\]}}
  x[ten] = 1; // BOOM
  return  0;
}

