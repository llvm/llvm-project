// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 %s

void __attribute__((noinline)) device_impl(int *xp, int *&xpref, int n) {}

#pragma omp declare variant(device_impl)                                       \
    adjust_args(need_device_ptr(foo) : xp, xpref)   // expected-error{{unknown modifier in 'need_device_ptr' clause (OpenMP 6.1 or later only)}} // expected-error{{expected 'match', 'adjust_args', or 'append_args' clause on 'omp declare variant' directive}}
void __attribute__((noinline)) host_entry_a(int *xp, int *&xpref, int n) {}

int main() {
  int x;
  int *xp = &x;

  host_entry_a(xp, xp, 1);
  return 0;
}
