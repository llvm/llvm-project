// RUN: %clang_cc1 -verify=omp61 -fopenmp -fopenmp-version=61 %s

void __attribute__((noinline)) device_impl(int *xp, int *&xpref, int n) {}

#pragma omp declare variant(device_impl) \
    adjust_args(need_device_ptr(foo) : xp, xpref)   // omp61-error{{invalid argument for 'need_device_ptr' kind in 'adjust_args' clause; expected 'fb_nullify' or 'fb_preserve'}} // omp61-error{{expected 'match', 'adjust_args', or 'append_args' clause on 'omp declare variant' directive}}
void __attribute__((noinline)) host_entry_a(int *xp, int *&xpref, int n) {}

#pragma omp declare variant(device_impl) \
  adjust_args(need_device_ptr(fb_nullify) : xp, xpref)
void __attribute__((noinline)) host_entry_b(int *xp, int *&xpref, int n) {}

#pragma omp declare variant(device_impl) \
  adjust_args(need_device_ptr(fb_preserve) : xp, xpref)
void __attribute__((noinline)) host_entry_c(int *xp, int *&xpref, int n) {}


int main() {
  int x;
  int *xp = &x;

  host_entry_a(xp, xp, 1);
  host_entry_b(xp, xp, 1);
  host_entry_c(xp, xp, 1);
  return 0;
}
