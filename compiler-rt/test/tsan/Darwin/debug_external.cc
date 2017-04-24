// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../test.h"

extern "C" {
void __tsan_on_report(void *report);
int __tsan_get_report_loc(void *report, unsigned long idx, const char **type,
                          void **addr, void **start,
                          unsigned long *size, int *tid, int *fd,
                          int *suppressable, void **trace,
                          unsigned long trace_size);
int __tsan_get_report_loc_object_type(void *report, unsigned long idx,
                                      const char **object_type);
}

void *Thread(void *arg) {
  barrier_wait(&barrier);
  *((long *)arg) = 42;
  return NULL;
}

int main() {
  barrier_init(&barrier, 2);
  void *tag = __tsan_external_register_tag("MyObject");
  long *obj = (long *)malloc(sizeof(long));
  fprintf(stderr, "obj = %p\n", obj);
  // CHECK: obj = [[ADDR:0x[0-9a-f]+]]
  __tsan_external_assign_tag(obj, tag);

  pthread_t t;
  pthread_create(&t, 0, Thread, obj);
  *obj = 41;
  barrier_wait(&barrier);
  pthread_join(t, 0);
  fprintf(stderr, "Done.\n");
  return 0;
}

void __tsan_on_report(void *report) {
  const char *type;
  void *addr;
  void *start;
  unsigned long size;
  int tid, fd, suppressable;
  void *trace[16] = {0};
  __tsan_get_report_loc(report, 0, &type, &addr, &start, &size, &tid, &fd,
                        &suppressable, trace, 16);
  fprintf(stderr, "type = %s, start = %p, size = %ld\n", type, start, size);
  // CHECK: type = heap, start = [[ADDR]], size = 8

  const char *object_type;
  __tsan_get_report_loc_object_type(report, 0, &object_type);
  fprintf(stderr, "object_type = %s\n", object_type);
  // CHECK: object_type = MyObject
}

// CHECK: Done.
// CHECK: ThreadSanitizer: reported 1 warnings
