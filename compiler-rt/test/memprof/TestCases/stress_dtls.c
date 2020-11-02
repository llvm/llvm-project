// REQUIRES: memprof-64-bits
// Stress test dynamic TLS + dlopen + threads.
//
// RUN: %clangxx_memprof -x c -DSO_NAME=f0 %s -shared -o %t-f0.so -fPIC
// RUN: %clangxx_memprof -x c -DSO_NAME=f1 %s -shared -o %t-f1.so -fPIC
// RUN: %clangxx_memprof -x c -DSO_NAME=f2 %s -shared -o %t-f2.so -fPIC
// RUN: %clangxx_memprof %s -ldl -pthread -o %t
// RUN: %run %t 0 3
// RUN: %run %t 2 3
// RUN: %env_memprof_opts=log_path=stderr:verbosity=2 %run %t 10 2 2>&1 | FileCheck %s
// RUN: %env_memprof_opts=log_path=stderr:verbosity=2:intercept_tls_get_addr=1 %run %t 10 2 2>&1 | FileCheck %s
// RUN: %env_memprof_opts=log_path=stderr:verbosity=2:intercept_tls_get_addr=0 %run %t 10 2 2>&1 | FileCheck %s --check-prefix=CHECK0
// CHECK: __tls_get_addr
// CHECK: Creating thread 0
// CHECK: __tls_get_addr
// CHECK: Creating thread 1
// CHECK: __tls_get_addr
// CHECK: Creating thread 2
// CHECK: __tls_get_addr
// CHECK: Creating thread 3
// CHECK: __tls_get_addr
// Make sure that TLS slots don't leak
// CHECK-NOT: num_live_dtls 5
//
// CHECK0-NOT: __tls_get_addr
/*
cc=your-compiler

$cc stress_dtls.c -pthread -ldl
for((i=0;i<100;i++)); do
  $cc -fPIC -shared -DSO_NAME=f$i -o a.out-f$i.so stress_dtls.c;
done
./a.out 2 4  # <<<<<< 2 threads, 4 libs
./a.out 3 50 # <<<<<< 3 threads, 50 libs
*/
#ifndef SO_NAME
#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void **(*f_t)();

__thread int my_tls;

#define MAX_N_FUNCTIONS 1000
f_t Functions[MAX_N_FUNCTIONS];

void *PrintStuff(void *unused) {
  uintptr_t stack;
  // fprintf(stderr, "STACK: %p TLS: %p SELF: %p\n", &stack, &my_tls,
  //        (void *)pthread_self());
  int i;
  for (i = 0; i < MAX_N_FUNCTIONS; i++) {
    if (!Functions[i])
      break;
    uintptr_t dtls = (uintptr_t)Functions[i]();
    fprintf(stderr, "  dtls[%03d]: %lx\n", i, dtls);
    *(long *)dtls = 42; // check that this is writable.
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  int num_threads = 1;
  int num_libs = 1;
  if (argc >= 2)
    num_threads = atoi(argv[1]);
  if (argc >= 3)
    num_libs = atoi(argv[2]);
  assert(num_libs <= MAX_N_FUNCTIONS);

  int lib;
  for (lib = 0; lib < num_libs; lib++) {
    char buf[4096];
    snprintf(buf, sizeof(buf), "%s-f%d.so", argv[0], lib);
    void *handle = dlopen(buf, RTLD_LAZY);
    if (!handle) {
      fprintf(stderr, "%s\n", dlerror());
      exit(1);
    }
    snprintf(buf, sizeof(buf), "f%d", lib);
    Functions[lib] = (f_t)dlsym(handle, buf);
    if (!Functions[lib]) {
      fprintf(stderr, "%s\n", dlerror());
      exit(1);
    }
    fprintf(stderr, "LIB[%03d] %s: %p\n", lib, buf, Functions[lib]);
    PrintStuff(0);

    int i;
    for (i = 0; i < num_threads; i++) {
      pthread_t t;
      fprintf(stderr, "Creating thread %d\n", i);
      pthread_create(&t, 0, PrintStuff, 0);
      pthread_join(t, 0);
    }
  }
  return 0;
}
#else // SO_NAME
#ifndef DTLS_SIZE
#define DTLS_SIZE (1 << 17)
#endif
__thread void *huge_thread_local_array[DTLS_SIZE];
void **SO_NAME() {
  return &huge_thread_local_array[0];
}
#endif
