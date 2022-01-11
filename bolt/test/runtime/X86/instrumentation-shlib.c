/* Checks that BOLT correctly handles instrumentation shared libraries
 * with further optimization.
 */
#include <dlfcn.h>
#include <stdio.h>

#ifdef LIB
int foo(int x) { return x + 1; }

int fib(int x) {
  if (x < 2)
    return x;
  return fib(x - 1) + fib(x - 2);
}

int bar(int x) { return x - 1; }
#endif

#ifndef LIB
int main(int argc, char **argv) {
  int (*fib_func)(int);
  char *libname;
  void *hndl;
  int val;
  if (argc < 2)
    return -1;
  /*
   * Expected library name as input to switch
   * between original and instrumented file
   */
  libname = argv[1];
  hndl = dlopen(libname, RTLD_LAZY);
  if (!hndl) {
    printf("library load failed\n");
    return -1;
  }
  fib_func = dlsym(hndl, "fib");
  if (!fib_func) {
    printf("fib_func load failed\n");
    return -1;
  }
  val = fib_func(argc);
  dlclose(hndl);
  printf("fib(%d) = %d\n", argc, val);
  return 0;
}
#endif

/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.so -Wl,-q -fpie -fPIC -shared -DLIB
RUN: %clang %cflags %s -o %t.exe -Wl,-q -ldl

RUN: llvm-bolt %t.so -instrument -instrumentation-file=%t.so.fdata \
RUN:   -o %t.so.instrumented

# Program with instrumented shared library needs to finish returning zero
RUN: %t.exe %t.so.instrumented 1 2 | FileCheck %s -check-prefix=CHECK-OUTPUT
RUN: test -f %t.so.fdata

# Test that the instrumented data makes sense
RUN: llvm-bolt %t.so -o %t.so.bolted -data %t.so.fdata \
RUN:    -reorder-blocks=cache+ -reorder-functions=hfsort+

RUN: %t.exe %t.so.bolted 1 2 | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-OUTPUT: fib(4) = 3
*/
