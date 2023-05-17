// RUN: %clang_dfsan %s -o %t && %run %t
//
// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out
//
// RUN: %clang_dfsan -gmlt -mllvm -dfsan-track-origins=1 -mllvm -dfsan-instrument-with-call-threshold=0 %s -o %t && \
// RUN:     %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <sanitizer/dfsan_interface.h>

#include <assert.h>
#include <pthread.h>
#include <string.h>

const int kNumThreads = 24;
int x = 0;
int __thread y, z;

static void *ThreadFn(void *a) {
  y = x;
  assert(dfsan_get_label(y) == 8);
  memcpy(&z, &y, sizeof(y));
  if ((int)a == 7)
    dfsan_print_origin_trace(&z, NULL);
  return 0;
}

int main(void) {
  dfsan_set_label(8, &x, sizeof(x));

  pthread_t t[kNumThreads];
  for (size_t i = 0; i < kNumThreads; ++i)
    pthread_create(&t[i], 0, ThreadFn, (void *)i);

  for (size_t i = 0; i < kNumThreads; ++i)
    pthread_join(t[i], 0);

  return 0;
}

// CHECK: Taint value 0x8 {{.*}} origin tracking ()
// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in ThreadFn.dfsan {{.*}}pthread.c:[[@LINE-21]]

// CHECK: Origin value: {{.*}}, Taint value was stored to memory at
// CHECK: #0 {{.*}} in ThreadFn.dfsan {{.*}}pthread.c:[[@LINE-26]]

// CHECK: Origin value: {{.*}}, Taint value was created at
// CHECK: #0 {{.*}} in main {{.*}}pthread.c:[[@LINE-20]]
