/* RUN: %clang_msan -g %s -o %t
   RUN: %clang_msan -g %s -DBUILD_SO -fPIC -o %t-so.so -shared
   RUN: %run %t 2>&1

   Regression test for a bug in msan/glibc integration,
   see https://sourceware.org/bugzilla/show_bug.cgi?id=16291
   and https://github.com/google/sanitizers/issues/547

   XFAIL: FreeBSD
   UNSUPPORTED: powerpc

   // Reports use-of-uninitialized-value, not analyzed
   XFAIL: netbsd

   // This is known to be broken with glibc-2.27+
   // https://bugs.llvm.org/show_bug.cgi?id=37804
   XFAIL: glibc-2.27

*/

#ifndef BUILD_SO
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef long *(* get_t)();
get_t GetTls;
void *Thread1(void *unused) {
  long uninitialized;
  long *x = GetTls();
  if (*x)
    fprintf(stderr, "bar\n");
  *x = uninitialized;
  fprintf(stderr, "stack: %p dtls: %p\n", &x, x);
  return 0;
}

void *Thread2(void *unused) {
  long *x = GetTls();
  fprintf(stderr, "stack: %p dtls: %p\n", &x, x);
  if (*x)
    fprintf(stderr, "foo\n");   // False negative here.
  return 0;
}

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);
  int i;

  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle) fprintf(stderr, "%s\n", dlerror());
  assert(handle != 0);
  GetTls = (get_t)dlsym(handle, "GetTls");
  assert(dlerror() == 0);

  pthread_t t;
  pthread_create(&t, 0, Thread1, 0);
  pthread_join(t, 0);
  pthread_create(&t, 0, Thread2, 0);
  pthread_join(t, 0);
  return 0;
}
#else  // BUILD_SO
__thread long huge_thread_local_array[1 << 17];
long *GetTls() {
  return &huge_thread_local_array[0];
}
#endif
