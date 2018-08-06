// Test for "sancov.py missing ...".

// First case: coverage from executable. main() is called on every code path.
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard %s -o %t -DFOOBAR -DMAIN
// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: cd %t-dir
// RUN: %env_asan_opts=coverage=1:coverage_dir=%t-dir %run %t
// RUN: %sancov print *.sancov > main.txt
// RUN: rm *.sancov
// RUN: count 1 < main.txt
// RUN: %env_asan_opts=coverage=1:coverage_dir=%t-dir %run %t x
// RUN: %sancov print *.sancov > foo.txt
// RUN: rm *.sancov
// RUN: count 3 < foo.txt
// RUN: %env_asan_opts=coverage=1:coverage_dir=%t-dir %run %t x x
// RUN: %sancov print *.sancov > bar.txt
// RUN: rm *.sancov
// RUN: count 4 < bar.txt
// RUN: %sancov missing %t < foo.txt > foo-missing.txt
// RUN: sort main.txt foo-missing.txt -o foo-missing-with-main.txt
// The "missing from foo" set may contain a few bogus PCs from the sanitizer
// runtime, but it must include the entire "bar" code path as a subset. Sorted
// lists can be tested for set inclusion with diff + grep.
// RUN: diff bar.txt foo-missing-with-main.txt > %t.log || true
// RUN: not grep "^<" %t.log

// Second case: coverage from DSO.
// cd %t-dir
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard %s -o %dynamiclib -DFOOBAR -shared -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard %s %dynamiclib -o %t -DMAIN
// RUN: cd ..
// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: cd %t-dir
// RUN: %env_asan_opts=coverage=1:coverage_dir=%t-dir %run %t x
// RUN: %sancov print %xdynamiclib_filename.*.sancov > foo.txt
// RUN: rm *.sancov
// RUN: count 2 < foo.txt
// RUN: %env_asan_opts=coverage=1:coverage_dir=%t-dir %run %t x x
// RUN: %sancov print %xdynamiclib_filename.*.sancov > bar.txt
// RUN: rm *.sancov
// RUN: count 3 < bar.txt
// RUN: %sancov missing %dynamiclib < foo.txt > foo-missing.txt
// RUN: diff bar.txt foo-missing.txt > %t.log || true
// RUN: not grep "^<" %t.log

// REQUIRES: x86-target-arch
// XFAIL: android

#include <stdio.h>

void foo1();
void foo2();
void bar1();
void bar2();
void bar3();

#if defined(FOOBAR)
void foo1() { fprintf(stderr, "foo1\n"); }
void foo2() { fprintf(stderr, "foo2\n"); }

void bar1() { fprintf(stderr, "bar1\n"); }
void bar2() { fprintf(stderr, "bar2\n"); }
void bar3() { fprintf(stderr, "bar3\n"); }
#endif

#if defined(MAIN)
int main(int argc, char **argv) {
  switch (argc) {
    case 1:
      break;
    case 2:
      foo1();
      foo2();
      break;
    case 3:
      bar1();
      bar2();
      bar3();
      break;
  }
}
#endif
