// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// Check that asan_symbolize.py script works (for binaries, ASan RTL and
// shared object files.

// RUN: %clangxx_asan -O0 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O0 %s %libdl -o %t
// RUN: %env_asan_opts=symbolize=0 not %run %t 2>&1 | %asan_symbolize | FileCheck %s
// REQUIRES: stable-runtime

// UNSUPPORTED: ios

#if !defined(SHARED_LIB)
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

using std::string;

typedef void (fun_t)(int*, int);

int main(int argc, char *argv[]) {
  string path = string(argv[0]) + "-so.so";
  printf("opening %s ... \n", path.c_str());
  void *lib = dlopen(path.c_str(), RTLD_NOW);
  if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  fun_t *inc2 = (fun_t*)dlsym(lib, "inc2");
  if (!inc2) return 1;
  printf("ok\n");
  int *array = (int*)malloc(40);
  inc2(array, 1);
  inc2(array, -1);  // BOOM
  // CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
  // CHECK: READ of size 4 at 0x{{.*}}
  // CHECK: #0 {{.*}} in inc2 {{.*}}asan-symbolize-sanity-test.cc:[[@LINE+21]]
  // CHECK: #1 {{.*}} in main {{.*}}asan-symbolize-sanity-test.cc:[[@LINE-4]]
  // CHECK: allocated by thread T{{.*}} here:
  // CHECK: #{{.*}} in {{(wrap_|__interceptor_)?}}malloc
  // CHECK: #{{.*}} in main {{.*}}asan-symbolize-sanity-test.cc:[[@LINE-9]]
  return 0;
}
#else  // SHARED_LIBS
#include <stdio.h>
#include <string.h>

int pad[10];
int GLOB[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

extern "C"
void inc(int index) {
  GLOB[index]++;
}

extern "C"
void inc2(int *a, int index) {
  a[index]++;
}
#endif  // SHARED_LIBS
