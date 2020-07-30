/// This test fails on Mac (https://bugs.llvm.org/show_bug.cgi?id=38134)
// XFAIL: darwin

// RUN: mkdir -p %t.d && cd %t.d

// RUN: %clang -E -DSHARED %s -o shared.c
// RUN: %clang --coverage -fPIC -shared shared.c -o libfunc.so
// RUN: test -f shared.gcno

/// Test the case where we exit abruptly after calling __gcov_flush, which means we don't write out the counters at exit.
// RUN: %clang -DEXIT_ABRUPTLY -DSHARED_CALL_BEFORE_FLUSH -DSHARED_CALL_AFTER_FLUSH --coverage %s -L%t.d -rpath %t.d -lfunc -o %t
// RUN: test -f gcov-shared-flush.gcno

// RUN: rm -f gcov-shared-flush.gcda shared.gcda
// RUN: %run %t
// RUN: llvm-cov gcov -t gcov-shared-flush.gcda | FileCheck %s --check-prefix=NO_WRITEOUT
// RUN: llvm-cov gcov -t shared.gcda | FileCheck %s --check-prefix=SHARED

// NO_WRITEOUT:      -: [[#%u,L:]]:#ifdef EXIT_ABRUPTLY
// NO_WRITEOUT-NEXT: 1: [[#%u,L+1]]: _exit(0);

// SHARED: 1: {{[[0-9]+}}:void foo(int n)

/// Test the case where we exit normally and we have a call to the shared library function before __gcov_flush.
// RUN: %clang -DSHARED_CALL_BEFORE_FLUSH --coverage %s -L%t.d -rpath %t.d -lfunc -o %t
// RUN: test -f gcov-shared-flush.gcno

// RUN: rm -f gcov-shared-flush.gcda shared.gcda
// RUN: %run %t
// RUN: llvm-cov gcov -t gcov-shared-flush.gcda | FileCheck %s --check-prefix=BEFORE
// RUN: llvm-cov gcov -t shared.gcda | FileCheck %s --check-prefix=SHARED_ONCE

// BEFORE:      -: {{[0-9]+}}:#ifdef SHARED_CALL_BEFORE_FLUSH
// BEFORE-NEXT: 1: {{[0-9]+}}:  foo(1);
// BEFORE:      1: {{[0-9]+}}:  __gcov_flush();
// BEFORE:      -: {{[0-9]+}}:#ifdef SHARED_CALL_AFTER_FLUSH
// BEFORE-NEXT: -: {{[0-9]+}}:  foo(1);
// BEFORE:      1: {{[0-9]+}}:  bar(5);

// SHARED_ONCE: 1: {{[0-9]+}}:void foo(int n)

// # Test the case where we exit normally and we have a call to the shared library function after __gcov_flush.
// RUN: %clang -DSHARED_CALL_AFTER_FLUSH --coverage %s -L%t.d -rpath %t.d -lfunc -o %t
// RUN: test -f gcov-shared-flush.gcno

// RUN: rm -f gcov-shared-flush.gcda shared.gcda
// RUN: %run %t
// RUN: llvm-cov gcov -t gcov-shared-flush.gcda | FileCheck %s --check-prefix=AFTER
// RUN: llvm-cov gcov -t shared.gcda > 2s.txt

// AFTER:      -: {{[0-9]+}}:#ifdef SHARED_CALL_BEFORE_FLUSH
// AFTER-NEXT: -: {{[0-9]+}}:  foo(1);
// AFTER:      1: {{[0-9]+}}:  __gcov_flush();
// AFTER:      -: {{[0-9]+}}:#ifdef SHARED_CALL_AFTER_FLUSH
// AFTER-NEXT: 1: {{[0-9]+}}:  foo(1);
// AFTER:      1: {{[0-9]+}}:  bar(5);

// # Test the case where we exit normally and we have calls to the shared library function before and after __gcov_flush.
// RUN: %clang -DSHARED_CALL_BEFORE_FLUSH -DSHARED_CALL_AFTER_FLUSH --coverage %s -L%t.d -rpath %t.d -lfunc -o %t
// RUN: test -f gcov-shared-flush.gcno

// RUN: rm -f gcov-shared-flush.gcda shared.gcda
// RUN: %run %t
// RUN: llvm-cov gcov -t gcov-shared-flush.gcda | FileCheck %s --check-prefix=BEFORE_AFTER
// RUN: llvm-cov gcov -t shared.gcda | FileCheck %s --check-prefix=SHARED_TWICE

// BEFORE_AFTER:      -: {{[0-9]+}}:#ifdef SHARED_CALL_BEFORE_FLUSH
// BEFORE_AFTER-NEXT: 1: {{[0-9]+}}:  foo(1);
// BEFORE_AFTER:      1: {{[0-9]+}}:  __gcov_flush();
// BEFORE_AFTER:      -: {{[0-9]+}}:#ifdef SHARED_CALL_AFTER_FLUSH
// BEFORE_AFTER-NEXT: 1: {{[0-9]+}}:  foo(1);
// BEFORE_AFTER:      1: {{[0-9]+}}:  bar(5);

// SHARED_TWICE: 2: {{[0-9]+}}:void foo(int n)

#ifdef SHARED
void foo(int n) {
}
#else
extern void foo(int n);
extern void __gcov_flush(void);

int bar1 = 0;
int bar2 = 1;

void bar(int n) {
  if (n % 5 == 0)
    bar1++;
  else
    bar2++;
}

int main(int argc, char *argv[]) {
#ifdef SHARED_CALL_BEFORE_FLUSH
  foo(1);
#endif

  bar(5);
  __gcov_flush();
  bar(5);

#ifdef SHARED_CALL_AFTER_FLUSH
  foo(1);
#endif

#ifdef EXIT_ABRUPTLY
  _exit(0);
#endif

  bar(5);
  return 0;
}
#endif
