// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=SOURCE
// RUN: %env_asan_opts=symbolize=false not %run %t 2>&1 | FileCheck %s --check-prefix=MODULE
// RUN: %env_asan_opts=print_summary=false not %run %t 2>&1 | FileCheck %s --check-prefix=MISSING

int main() {
  char *x = new char[20];
  delete[] x;
  return x[0];
  // SOURCE: ERROR: AddressSanitizer: heap-use-after-free
  // SOURCE: SUMMARY: AddressSanitizer: heap-use-after-free {{.*}}print_summary.cpp:[[@LINE-2]]{{.*}} main
  // MODULE: ERROR: AddressSanitizer: heap-use-after-free
  // MODULE: SUMMARY: AddressSanitizer: heap-use-after-free ({{.*}}+0x{{.*}})
  // MISSING: ERROR: AddressSanitizer: heap-use-after-free
  // MISSING-NOT: SUMMARY
}
