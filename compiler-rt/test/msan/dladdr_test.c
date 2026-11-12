/* RUN: %clang_msan -g %s -o %t
   RUN: %clang_msan -g %s -DBUILD_SO -fPIC -o %t-so.so -shared
   RUN: %run %t 2>&1 | FileCheck %s

   REQUIRES: glibc{{.*}}
*/

#define _GNU_SOURCE

#ifndef BUILD_SO
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef volatile long *(* get_t)();
get_t GetTls;

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);
  int i;

  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle) fprintf(stderr, "%s\n", dlerror());
  assert(handle != 0);
  GetTls = (get_t)dlsym(handle, "GetTls");
  assert(dlerror() == 0);

  Dl_info info;
  int ret = dladdr(GetTls, &info);
  assert (ret != 0);
  printf ("fname: %s\n", info.dli_fname);
  printf ("fbase: %p\n", info.dli_fbase);
  printf ("sname: %s\n", info.dli_sname);
  printf ("saddr: %p\n", info.dli_saddr);

  // CHECK: sname: GetTls

  return 0;
}
#else  // BUILD_SO
long var;
long *GetTls() {
  return &var;
}
#endif
