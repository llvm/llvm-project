// Test that mixing instrumented and non-instrumented code doesn't lead to crash.
// Build two modules (one is instrumented, another is not) that have globals
// with same names. Check, that ASan doesn't crash with CHECK failure or
// false positive global-buffer-overflow due to sanitized library poisons
// globals from non-sanitized one.
//
// RUN: %clangxx_asan -DBUILD_INSTRUMENTED_DSO=1 -fPIC -shared -mllvm -asan-use-private-alias %s -o %dynamiclib1
// RUN: %clangxx -DBUILD_UNINSTRUMENTED_DSO=1 -fPIC -shared %s -o %dynamiclib2
// RUN: %clangxx %s -c -mllvm -asan-use-private-alias -o %t.o
// RUN: %clangxx_asan %t.o %ld_flags_rpath_exe2 %ld_flags_rpath_exe1 -o %t-EXE
// RUN: %run %t-EXE

#if defined (BUILD_INSTRUMENTED_DSO)
long h = 15;
long f = 4;
long foo(long *p) {
  return *p;
}
#elif defined (BUILD_UNINSTRUMENTED_DSO)
long foo(long *);
long h = 12;
long i = 13;
long f = 5;

int bar() {
  if (foo(&f) != 5 || foo(&h) != 12 || foo(&i) != 13)
    return 1;
  return 0;
}
#else
extern int bar();

int main() {
  return bar();
}
#endif
