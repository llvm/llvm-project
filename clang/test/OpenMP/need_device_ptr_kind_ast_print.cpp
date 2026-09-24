// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -ast-print %s \
// RUN: | FileCheck %s

// expected-no-diagnostics

void __attribute__((noinline)) device_impl(int *xp, int *&xpref, int n) {}

#pragma omp declare variant(device_impl) \
  adjust_args(need_device_ptr(fb_nullify) : xp, xpref)
void __attribute__((noinline)) host_entry_a(int *xp, int *&xpref, int n) {}

#pragma omp declare variant(device_impl) \
  adjust_args(need_device_ptr(fb_preserve) : xp, xpref)
void __attribute__((noinline)) host_entry_b(int *xp, int *&xpref, int n) {}

// CHECK-LABEL: int main()
int main() {
  int x;
  int *xp = &x;

  host_entry_a(xp, xp, 1);
  host_entry_b(xp, xp, 1);
  return 0;
}
