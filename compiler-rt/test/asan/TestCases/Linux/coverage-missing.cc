// Test for "sancov.py missing ...".

// First case: coverage from executable. main() is called on every code path.
// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t -DFOOBAR -DMAIN
// RUN: rm -rf %T/coverage-missing
// RUN: mkdir -p %T/coverage-missing
// RUN: cd %T/coverage-missing
// RUN: %env_asan_opts=coverage=1:coverage_dir=%T/coverage-missing %run %t
// RUN: %sancov print *.sancov > main.txt
// RUN: rm *.sancov
// RUN: [ $(cat main.txt | wc -l) == 1 ]
// RUN: %env_asan_opts=coverage=1:coverage_dir=%T/coverage-missing %run %t x
// RUN: %sancov print *.sancov > foo.txt
// RUN: rm *.sancov
// RUN: [ $(cat foo.txt | wc -l) == 3 ]
// RUN: %env_asan_opts=coverage=1:coverage_dir=%T/coverage-missing %run %t x x
// RUN: %sancov print *.sancov > bar.txt
// RUN: rm *.sancov
// RUN: [ $(cat bar.txt | wc -l) == 4 ]
// RUN: %sancov missing %t < foo.txt > foo-missing.txt
// RUN: sort main.txt foo-missing.txt -o foo-missing-with-main.txt
// The "missing from foo" set may contain a few bogus PCs from the sanitizer
// runtime, but it must include the entire "bar" code path as a subset. Sorted
// lists can be tested for set inclusion with diff + grep.
// RUN: ( diff bar.txt foo-missing-with-main.txt || true ) | not grep "^<"

// Second case: coverage from DSO.
// cd %T
// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %dynamiclib -DFOOBAR -shared -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func %s %dynamiclib -o %t -DMAIN
// RUN: export LIBNAME=`basename %dynamiclib`
// RUN: rm -rf %T/coverage-missing
// RUN: mkdir -p %T/coverage-missing
// RUN: cd %T/coverage-missing
// RUN: %env_asan_opts=coverage=1:coverage_dir=%T/coverage-missing %run %t x
// RUN: %sancov print $LIBNAME.*.sancov > foo.txt
// RUN: rm *.sancov
// RUN: [ $(cat foo.txt | wc -l) == 2 ]
// RUN: %env_asan_opts=coverage=1:coverage_dir=%T/coverage-missing %run %t x x
// RUN: %sancov print $LIBNAME.*.sancov > bar.txt
// RUN: rm *.sancov
// RUN: [ $(cat bar.txt | wc -l) == 3 ]
// RUN: %sancov missing %dynamiclib < foo.txt > foo-missing.txt
// RUN: ( diff bar.txt foo-missing.txt || true ) | not grep "^<"

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
