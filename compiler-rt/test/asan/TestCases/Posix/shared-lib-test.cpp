// RUN: %clangxx_asan -O0 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O0 %s %libdl -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O1 %s %libdl -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O2 %s %libdl -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O3 %s %libdl -o %t && not %run %t 2>&1 | FileCheck %s

#if !defined(SHARED_LIB)
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include <string>

using std::string;

typedef void (fun_t)(int x);

int main(int argc, char *argv[]) {
  string path = string(argv[0]) + "-so.so";
  printf("opening %s ... \n", path.c_str());
  void *lib = dlopen(path.c_str(), RTLD_NOW);
  if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  fun_t *inc = (fun_t*)dlsym(lib, "inc");
  if (!inc) return 1;
  printf("ok\n");
  inc(1);
  inc(-1);  // BOOM
  // CHECK: {{.*ERROR: AddressSanitizer: global-buffer-overflow}}
  // CHECK: {{READ of size 4 at 0x.* thread T0}}
  // CHECK: {{    #0 0x.*}}
  // CHECK: {{    #1 0x.* in main .*shared-lib-test.cpp:}}[[@LINE-4]]
  return 0;
}
#else  // SHARED_LIB
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
#endif  // SHARED_LIB
