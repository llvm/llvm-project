// RUN: %clang -fsanitize=bool %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 not %run %t 2>&1 | FileCheck %s --check-prefix=SUMMARY

typedef char BOOL;
unsigned char NotABool = 123;

int main(int argc, char **argv) {
  BOOL *p = (BOOL*)&NotABool;

  // CHECK: bool.m:[[@LINE+1]]:10: runtime error: load of value 123, which is not a valid value for type 'BOOL'
  return *p;
  // SUMMARY: SUMMARY: {{.*}}Sanitizer: invalid-bool-load {{.*}}bool.m:[[@LINE-1]]
}
