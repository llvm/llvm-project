// RUN: %clangxx_tsan %s -o %t -framework Foundation -std=c++11
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#import <libkern/OSAtomic.h>

#include <thread>

#include "../test.h"

typedef struct {
  void *next;
  long data;
} ListItem;

OSQueueHead q;

int main(int argc, const char *argv[]) {
  barrier_init(&barrier, 2);

  std::thread t1([] {
    ListItem *li = new ListItem{nullptr, 42};
    OSAtomicEnqueue(&q, li, 0);
    barrier_wait(&barrier);
  });

  std::thread t2([] {
    barrier_wait(&barrier);
    ListItem *li = (ListItem *)OSAtomicDequeue(&q, 0);
    fprintf(stderr, "data = %ld\n", li->data);
  });

  t1.join();
  t2.join();

  fprintf(stderr, "done\n");

  return 0;
}

// CHECK: data = 42
// CHECK: done
// CHECK-NOT: WARNING: ThreadSanitizer
