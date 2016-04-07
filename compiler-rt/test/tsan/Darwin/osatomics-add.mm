// RUN: %clang_tsan %s -o %t -framework Foundation -std=c++11
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#import <libkern/OSAtomic.h>

#include <thread>

volatile int64_t retainCount = 0;

long g = 0;

void dealloc() {
  g = 42;
}

void release() {
  if (OSAtomicAdd64Barrier(-1, &retainCount) == 0) {
    dealloc();
  }
}

void retain() {
  OSAtomicAdd64Barrier(1, &retainCount);
}

int main(int argc, const char * argv[]) {
  fprintf(stderr, "start\n");
  retain();
  retain();
  
  std::thread t([]{
    release();
  });

  g = 47;

  release();
  t.join();

  fprintf(stderr, "end, g = %ld\n", g);

  return 0;
}

// CHECK: start
// CHECK: end, g = 42
// CHECK-NOT: WARNING: ThreadSanitizer
