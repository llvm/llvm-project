// RUN: %clangxx_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint64_t objs[8*3*3*2][3];

extern "C" {
void __tsan_unaligned_read2(void *addr);
void __tsan_unaligned_read4(void *addr);
void __tsan_unaligned_read8(void *addr);
void __tsan_unaligned_write2(void *addr);
void __tsan_unaligned_write4(void *addr);
void __tsan_unaligned_write8(void *addr);
}

static void access(char *p, int sz, int rw) {
  if (rw) {
    switch (sz) {
    case 0: __tsan_unaligned_write2(p); break;
    case 1: __tsan_unaligned_write4(p); break;
    case 2: __tsan_unaligned_write8(p); break;
    default: exit(1);
    }
  } else {
    switch (sz) {
    case 0: __tsan_unaligned_read2(p); break;
    case 1: __tsan_unaligned_read4(p); break;
    case 2: __tsan_unaligned_read8(p); break;
    default: exit(1);
    }
  }
}

static int accesssize(int sz) {
  switch (sz) {
  case 0: return 2;
  case 1: return 4;
  case 2: return 8;
  }
  exit(1);
}

void Test(bool main) {
  uint64_t *obj = objs[0];
  for (int off = 0; off < 8; off++) {
    for (int sz1 = 0; sz1 < 3; sz1++) {
      for (int sz2 = 0; sz2 < 3; sz2++) {
        for (int rw = 0; rw < 2; rw++) {
          char *p = (char*)obj + off;
          if (main) {
            // printf("thr=%d off=%d sz1=%d sz2=%d rw=%d p=%p\n",
            //        main, off, sz1, sz2, rw, p);
            access(p, sz1, true);
          } else {
            p += accesssize(sz1);
            // printf("thr=%d off=%d sz1=%d sz2=%d rw=%d p=%p\n",
            //        main, off, sz1, sz2, rw, p);
            access(p, sz2, rw);
          }
          obj += 3;
        }
      }
    }
  }
}

void *Thread(void *p) {
  (void)p;
  Test(false);
  return 0;
}

int main() {
  pthread_t th;
  pthread_create(&th, 0, Thread, 0);
  Test(true);
  pthread_join(th, 0);
  fprintf(stderr, "OK\n");
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: OK
