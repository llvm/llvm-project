// RUN: %clangxx_tsan %s -o %t
// RUN: %deflake %run %t 2>&1 | FileCheck %s

#include <thread>

#import "../test.h"

extern "C" {
void __tsan_write8(void *addr);
}

static void *tag = (void *)0x1;

__attribute__((no_sanitize("thread")))
void ExternalWrite(void *addr) {
  __tsan_external_write(addr, nullptr, tag);
}

__attribute__((no_sanitize("thread")))
void RegularWrite(void *addr) {
  __tsan_write8(addr);
}

int main(int argc, char *argv[]) {
  barrier_init(&barrier, 2);
  fprintf(stderr, "Start.\n");
  // CHECK: Start.
  
  {
    void *opaque_object = malloc(16);
    std::thread t1([opaque_object] {
      ExternalWrite(opaque_object);
      barrier_wait(&barrier);
    });
    std::thread t2([opaque_object] {
      barrier_wait(&barrier);
      ExternalWrite(opaque_object);
    });
    // CHECK: WARNING: ThreadSanitizer: Swift access race
    // CHECK: Modifying access of Swift variable at {{.*}} by thread {{.*}}
    // CHECK: Previous modifying access of Swift variable at {{.*}} by thread {{.*}}
    // CHECK: SUMMARY: ThreadSanitizer: Swift access race
    t1.join();
    t2.join();
  }

  fprintf(stderr, "external+external test done.\n");
  // CHECK: external+external test done.

  {
    void *opaque_object = malloc(16);
    std::thread t1([opaque_object] {
      ExternalWrite(opaque_object);
      barrier_wait(&barrier);
    });
    std::thread t2([opaque_object] {
      barrier_wait(&barrier);
      RegularWrite(opaque_object);
    });
    // CHECK: WARNING: ThreadSanitizer: Swift access race
    // CHECK: Write of size 8 at {{.*}} by thread {{.*}}
    // CHECK: Previous modifying access of Swift variable at {{.*}} by thread {{.*}}
    // CHECK: SUMMARY: ThreadSanitizer: Swift access race
    t1.join();
    t2.join();
  }
  
  fprintf(stderr, "external+regular test done.\n");
  // CHECK: external+regular test done.
  
  {
    void *opaque_object = malloc(16);
    std::thread t1([opaque_object] {
      RegularWrite(opaque_object);
      barrier_wait(&barrier);
    });
    std::thread t2([opaque_object] {
      barrier_wait(&barrier);
      ExternalWrite(opaque_object);
    });
    // CHECK: WARNING: ThreadSanitizer: Swift access race
    // CHECK: Modifying access of Swift variable at {{.*}} by thread {{.*}}
    // CHECK: Previous write of size 8 at {{.*}} by thread {{.*}}
    // CHECK: SUMMARY: ThreadSanitizer: Swift access race
    t1.join();
    t2.join();
  }
  
  fprintf(stderr, "regular+external test done.\n");
  // CHECK: regular+external test done.
}

