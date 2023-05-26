/// atexit(3) not supported in dlopen(3)ed+dlclose(3)d DSO
// XFAIL: target={{.*netbsd.*}}

// RUN: mkdir -p %t.d && cd %t.d

// RUN: echo 'void func1(int k) {}' > func1.c
// RUN: echo 'void func2(int k) {}' > func2.c
// RUN: echo 'void func3(int k) {}' > func3.c
// RUN: %clang --coverage -fPIC -shared func1.c -o func1.so -dumpdir ./
// RUN: %clang --coverage -fPIC -shared func2.c -o func2.so -dumpdir ./
// RUN: %clang --coverage -fPIC -shared func3.c -o func3.so -dumpdir ./
// RUN: %clang --coverage -fPIC -rpath %t.d %s -o %t -dumpdir ./

/// Test with two dlopened libraries.
// RUN: rm -f gcov-dlopen.gcda func1.gcda func2.gcda
// RUN: %run %t
// RUN: llvm-cov gcov -t gcov-dlopen.gcda | FileCheck %s
// RUN: llvm-cov gcov -t func1.gcda | FileCheck %s --check-prefix=FUNC1
// RUN: llvm-cov gcov -t func2.gcda | FileCheck %s --check-prefix=FUNC2

// FUNC1:     1:    1:void func1(int k) {}
// FUNC2:     1:    1:void func2(int k) {}

/// Test with three dlopened libraries.
// RUN: %clang -DUSE_LIB3 --coverage -fPIC -rpath %t.d %s -o %t -dumpdir ./
// RUN: rm -f gcov-dlopen.gcda func1.gcda func2.gcda func3.gcda
// RUN: %run %t
// RUN: llvm-cov gcov -t gcov-dlopen.gcda | FileCheck %s --check-prefix=LIB3
// RUN: llvm-cov gcov -t func1.gcda | FileCheck %s --check-prefix=FUNC1
// RUN: llvm-cov gcov -t func2.gcda | FileCheck %s --check-prefix=FUNC2
// RUN: llvm-cov gcov -t func3.gcda | FileCheck %s --check-prefix=FUNC3

// FUNC3:     1:    1:void func3(int k) {}

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  void *f1_handle = dlopen("func1.so", RTLD_LAZY | RTLD_GLOBAL);
  if (f1_handle == NULL)
    return fprintf(stderr, "unable to open 'func1.so': %s\n", dlerror());
  void (*func1)(void) = (void (*)(void))dlsym(f1_handle, "func1");
  if (func1 == NULL)
    return fprintf(stderr, "unable to lookup symbol 'func1': %s\n", dlerror());

  void *f2_handle = dlopen("func2.so", RTLD_LAZY | RTLD_GLOBAL);
  if (f2_handle == NULL)
    return fprintf(stderr, "unable to open 'func2.so': %s\n", dlerror());
  void (*func2)(void) = (void (*)(void))dlsym(f2_handle, "func2");
  if (func2 == NULL)
    return fprintf(stderr, "unable to lookup symbol 'func2': %s\n", dlerror());
  func2();

#ifdef USE_LIB3
// CHECK:          -: [[#@LINE+2]]:  void *f3_handle
// LIB3:           1: [[#@LINE+1]]:  void *f3_handle
  void *f3_handle = dlopen("func3.so", RTLD_LAZY | RTLD_GLOBAL);
  if (f3_handle == NULL)
    return fprintf(stderr, "unable to open 'func3.so': %s\n", dlerror());
  void (*func3)(void) = (void (*)(void))dlsym(f3_handle, "func3");
  if (func3 == NULL)
    return fprintf(stderr, "unable to lookup symbol 'func3': %s\n", dlerror());
  func3();
#endif

  void (*gcov_reset1)() = (void (*)())dlsym(f1_handle, "__gcov_reset");
  if (gcov_reset1 == NULL)
    return fprintf(stderr, "unable to find __gcov_reset in func1.so': %s\n", dlerror());
  void (*gcov_reset2)() = (void (*)())dlsym(f2_handle, "__gcov_reset");
  if (gcov_reset2 == NULL)
    return fprintf(stderr, "unable to find __gcov_reset in func2.so': %s\n", dlerror());
  if (gcov_reset1 == gcov_reset2)
    return fprintf(stderr, "same __gcov_reset found in func1.so and func2.so\n");

  /// Test that __gcov_dump is in the dynamic symbol table.
  void (*gcov_dump1)() = (void (*)())dlsym(f1_handle, "__gcov_dump");
  if (gcov_dump1 == NULL)
    return fprintf(stderr, "unable to find __gcov_dump in func1.so': %s\n", dlerror());

  if (dlclose(f2_handle) != 0)
    return fprintf(stderr, "unable to close 'func2.so': %s\n", dlerror());

  func1();

  return 0;
}
