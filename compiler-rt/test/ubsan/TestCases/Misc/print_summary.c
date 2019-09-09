// RUN: %clang -fsanitize=undefined %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-DEFAULT
// RUN: %env_ubsan_opts=print_summary=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NO_SUMMARY

// CHECK-DEFAULT: SUMMARY: UndefinedBehaviorSanitizer: {{.*}}
// CHECK-NO_SUMMARY-NOT: SUMMARY: UndefinedBehaviorSanitizer: {{.*}}

int main(int argc, char **argv) {
  int arr[argc - 2];
  return 0;
}
