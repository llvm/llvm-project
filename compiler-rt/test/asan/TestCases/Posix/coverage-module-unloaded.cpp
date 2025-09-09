// Check that unloading a module doesn't break coverage dumping for remaining
// modules.
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard -DSHARED %s -shared -o %dynamiclib1 -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard -DSHARED %s -shared -o %dynamiclib2 -fPIC
// RUN: %clangxx_asan -fsanitize-coverage=func,trace-pc-guard %s %libdl -o %t.exe
// RUN: mkdir -p %t.tmp/coverage-module-unloaded && cd %t.tmp/coverage-module-unloaded
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t.exe %dynamiclib1 %dynamiclib2 2>&1        | FileCheck %s
// RUN: %env_asan_opts=coverage=1:verbosity=1 %run %t.exe %dynamiclib1 %dynamiclib2 foo 2>&1    | FileCheck %s
//
// https://code.google.com/p/address-sanitizer/issues/detail?id=263
// XFAIL: android
// UNSUPPORTED: ios

#include <assert.h>
#include <dlfcn.h>
#include <stdint.h>
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
  bool lt = reinterpret_cast<uintptr_t>(bar1) < reinterpret_cast<uintptr_t>(bar2);
  if (argc < 2)
    dlclose(lt ? handle1 : handle2);
  else
    dlclose(lt ? handle2 : handle1);
  return 0;
}
#endif

// CHECK: PID: [[PID:[0-9]+]]
// CHECK-DAG: exe{{.*}}[[PID]].sancov: {{.*}}PCs written
// CHECK-DAG: dynamic{{.*}}[[PID]].sancov: {{.*}}PCs written
