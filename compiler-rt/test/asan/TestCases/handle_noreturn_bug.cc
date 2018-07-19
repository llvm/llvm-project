// Regression test: __asan_handle_no_return should unpoison stack even with poison_heap=0.
// RUN: %clangxx_asan -O0 %s -o %t && \
// RUN: %env_asan_opts=poison_heap=1 %run %t && \
// RUN: %env_asan_opts=poison_heap=0 %run %t

#include <sanitizer/asan_interface.h>

int main(int argc, char **argv) {
  int x[2];
  int * volatile p = &x[0];
  __asan_handle_no_return();
  int volatile z = p[2];
}
