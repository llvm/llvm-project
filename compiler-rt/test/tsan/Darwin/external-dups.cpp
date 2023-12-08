// RUN: %clangxx_tsan %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#include <thread>

#import "../test.h"

void *tag;

__attribute__((no_sanitize("thread")))
void ExternalWrite(void *addr) {
  __tsan_external_write(addr, __builtin_return_address(0), tag);
}

int main(int argc, char *argv[]) {
  barrier_init(&barrier, 2);
  tag = __tsan_external_register_tag("HelloWorld");
  fprintf(stderr, "Start.\n");
  // CHECK: Start.
   
  for (int i = 0; i < 4; i++) {
    void *opaque_object = malloc(16);
    std::thread t1([opaque_object] {
      ExternalWrite(opaque_object);
      barrier_wait(&barrier);
    });
    std::thread t2([opaque_object] {
      barrier_wait(&barrier);
      ExternalWrite(opaque_object);
    });
    // CHECK: WARNING: ThreadSanitizer: race on HelloWorld
    t1.join();
    t2.join();
  }
  
  fprintf(stderr, "First phase done.\n");
  // CHECK: First phase done.

  for (int i = 0; i < 4; i++) {
    void *opaque_object = malloc(16);
    std::thread t1([opaque_object] {
      ExternalWrite(opaque_object);
      barrier_wait(&barrier);
    });
    std::thread t2([opaque_object] {
      barrier_wait(&barrier);
      ExternalWrite(opaque_object);
    });
    // CHECK: WARNING: ThreadSanitizer: race on HelloWorld
    t1.join();
    t2.join();
  }

  fprintf(stderr, "Second phase done.\n");
  // CHECK: Second phase done.
}

// CHECK: ThreadSanitizer: reported 2 warnings
