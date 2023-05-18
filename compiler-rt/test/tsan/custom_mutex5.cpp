// RUN: %clangxx_tsan -O1 --std=c++11 %s -o %t && %deflake %run %t 2>&1 | FileCheck %s
#include "custom_mutex.h"

#include <type_traits>

// Test that we detect the destruction of an in-use mutex when the
// thread annotations don't otherwise disable the check.

int main() {
  std::aligned_storage<sizeof(Mutex), alignof(Mutex)>::type mu1_store;
  Mutex* mu1 = reinterpret_cast<Mutex*>(&mu1_store);
  new(&mu1_store) Mutex(false, 0);
  mu1->Lock();
  mu1->~Mutex();
  mu1->Unlock();

  std::aligned_storage<sizeof(Mutex), alignof(Mutex)>::type mu2_store;
  Mutex* mu2 = reinterpret_cast<Mutex*>(&mu2_store);
  new(&mu2_store)
      Mutex(false, __tsan_mutex_not_static, __tsan_mutex_not_static);
  mu2->Lock();
  mu2->~Mutex();
  mu2->Unlock();

  fprintf(stderr, "DONE\n");
  return 0;
}

// CHECK: WARNING: ThreadSanitizer: destroy of a locked mutex
// CHECK:   main {{.*}}custom_mutex5.cpp:14
// CHECK: WARNING: ThreadSanitizer: destroy of a locked mutex
// CHECK:   main {{.*}}custom_mutex5.cpp:22
// CHECK: DONE
