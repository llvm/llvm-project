// Check that unloading a module doesn't break coverage dumping for remaining
// modules.
// RUN: %clangxx_asan -fsanitize-coverage=func -DSHARED %s -shared -o %dynamiclib1 -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func -DSHARED %s -shared -o %dynamiclib2 -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func %s %libdl -o %t
// RUN: mkdir -p %T/coverage-module-unloaded && cd %T/coverage-module-unloaded
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t %dynamiclib1 %dynamiclib2 2>&1        | FileCheck %s
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t %dynamiclib1 %dynamiclib2 foo 2>&1    | FileCheck %s
// RUN: rm -r %T/coverage-module-unloaded
//
// https://code.google.com/p/address-sanitizer/issues/detail?id=263
// XFAIL: android
// UNSUPPORTED: ios

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

#ifdef SHARED
extern "C" {
void bar() { printf("bar\n"); }
}
#else

int main(int argc, char **argv) {
  fprintf(stderr, "PID: %d\n", getpid());
  assert(argc > 2);
  void *handle1 = dlopen(argv[1], RTLD_LAZY);  // %dynamiclib1
  assert(handle1);
  void (*bar1)() = (void (*)())dlsym(handle1, "bar");
  assert(bar1);
  bar1();
  void *handle2 = dlopen(argv[2], RTLD_LAZY);  // %dynamiclib2
  assert(handle2);
  void (*bar2)() = (void (*)())dlsym(handle2, "bar");
  assert(bar2);
  bar2();

  // It matters whether the unloaded module has a higher or lower address range
  // than the remaining one. Make sure to test both cases.
  if (argc < 2)
    dlclose(bar1 < bar2 ? handle1 : handle2);
  else
    dlclose(bar1 < bar2 ? handle2 : handle1);
  return 0;
}
#endif

// CHECK: PID: [[PID:[0-9]+]]
// CHECK: [[PID]].sancov: 1 PCs written
// CHECK: coverage-module-unloaded{{.*}}1.[[PID]]
// CHECK: coverage-module-unloaded{{.*}}2.[[PID]]
// Even though we've unloaded one of the libs we still dump the coverage file
// for that lib (although the data will be inaccurate, if at all useful)
