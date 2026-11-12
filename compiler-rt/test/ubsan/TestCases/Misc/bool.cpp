// RUN: %clangxx -fsanitize=bool %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s
// RUN: %env_ubsan_opts=print_summary=1:report_error_type=1 not %run %t 2>&1 | FileCheck %s --check-prefix=SUMMARY

unsigned char NotABool = 123;

int main(int argc, char **argv) {
  bool *p = (bool*)&NotABool;

  // CHECK: bool.cpp:[[@LINE+1]]:10: runtime error: load of value 123, which is not a valid value for type 'bool'
  return *p;
  // SUMMARY: SUMMARY: {{.*}}Sanitizer: invalid-bool-load {{.*}}bool.cpp:[[@LINE-1]]
}
