//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing
// REQUIRES: has-unix-headers

// <hazard_pointer>

// A pthread key destructor that destroys (and creates) hazard pointers: this may run after libc++'s own
// per-thread record cache has been torn down, exercising the path where release/acquire bypass the cache.

#include <hazard_pointer>
#include <cassert>
#include <pthread.h>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

struct Box {
  std::hazard_pointer hp;
};

pthread_key_t key;

void destroy_box(void* p) {
  Box* box                 = static_cast<Box*>(p);
  std::hazard_pointer late = std::make_hazard_pointer(); // acquire after the cache may be gone
  assert(!late.empty());
  delete box; // release after the cache may be gone
}

int main(int, char**) {
  {
    std::hazard_pointer first = std::make_hazard_pointer();
  } // create libc++'s key before ours
  int created = pthread_key_create(&key, &destroy_box);
  assert(created == 0);
  for (int i = 0; i < 64; ++i) {
    std::thread t = support::make_test_thread([] {
      Box* box = new Box;
      box->hp  = std::make_hazard_pointer();
      int set  = pthread_setspecific(key, box);
      assert(set == 0);
      std::hazard_pointer cached[12]; // fill the cache so eviction has work to do
      for (std::hazard_pointer& h : cached)
        h = std::make_hazard_pointer();
    });
    t.join();
  }
  return 0;
}
