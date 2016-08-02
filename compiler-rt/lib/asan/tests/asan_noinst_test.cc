//===-- asan_noinst_test.cc -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This test file should be compiled w/o asan instrumentation.
//===----------------------------------------------------------------------===//

#include "asan_allocator.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_test_utils.h"
#include <sanitizer/allocator_interface.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // for memset()
#include <algorithm>
#include <vector>
#include <limits>

// ATTENTION!
// Please don't call intercepted functions (including malloc() and friends)
// in this test. The static runtime library is linked explicitly (without
// -fsanitize=address), thus the interceptors do not work correctly on OS X.

// Make sure __asan_init is called before any test case is run.
struct AsanInitCaller {
  AsanInitCaller() {
    DisableReexec();
    __asan_init();
  }
};
static AsanInitCaller asan_init_caller;

TEST(AddressSanitizer, InternalSimpleDeathTest) {
  EXPECT_DEATH(exit(1), "");
}

static void MallocStress(size_t n) {
  u32 seed = my_rand();
  BufferedStackTrace stack1;
  stack1.trace_buffer[0] = 0xa123;
  stack1.trace_buffer[1] = 0xa456;
  stack1.size = 2;

  BufferedStackTrace stack2;
  stack2.trace_buffer[0] = 0xb123;
  stack2.trace_buffer[1] = 0xb456;
  stack2.size = 2;

  BufferedStackTrace stack3;
  stack3.trace_buffer[0] = 0xc123;
  stack3.trace_buffer[1] = 0xc456;
  stack3.size = 2;

  std::vector<void *> vec;
  for (size_t i = 0; i < n; i++) {
    if ((i % 3) == 0) {
      if (vec.empty()) continue;
      size_t idx = my_rand_r(&seed) % vec.size();
      void *ptr = vec[idx];
      vec[idx] = vec.back();
      vec.pop_back();
      __asan::asan_free(ptr, &stack1, __asan::FROM_MALLOC);
    } else {
      size_t size = my_rand_r(&seed) % 1000 + 1;
      switch ((my_rand_r(&seed) % 128)) {
        case 0: size += 1024; break;
        case 1: size += 2048; break;
        case 2: size += 4096; break;
      }
      size_t alignment = 1 << (my_rand_r(&seed) % 10 + 1);
      char *ptr = (char*)__asan::asan_memalign(alignment, size,
                                               &stack2, __asan::FROM_MALLOC);
      EXPECT_EQ(size, __asan::asan_malloc_usable_size(ptr, 0, 0));
      vec.push_back(ptr);
      ptr[0] = 0;
      ptr[size-1] = 0;
      ptr[size/2] = 0;
    }
  }
  for (size_t i = 0; i < vec.size(); i++)
    __asan::asan_free(vec[i], &stack3, __asan::FROM_MALLOC);
}


TEST(AddressSanitizer, NoInstMallocTest) {
  MallocStress(ASAN_LOW_MEMORY ? 300000 : 1000000);
}

TEST(AddressSanitizer, ThreadedMallocStressTest) {
  const int kNumThreads = 4;
  const int kNumIterations = (ASAN_LOW_MEMORY) ? 10000 : 100000;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_CREATE(&t[i], 0, (void* (*)(void *x))MallocStress,
        (void*)kNumIterations);
  }
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_JOIN(t[i], 0);
  }
}

static void PrintShadow(const char *tag, uptr ptr, size_t size) {
  fprintf(stderr, "%s shadow: %lx size % 3ld: ", tag, (long)ptr, (long)size);
  uptr prev_shadow = 0;
  for (sptr i = -32; i < (sptr)size + 32; i++) {
    uptr shadow = __asan::MemToShadow(ptr + i);
    if (i == 0 || i == (sptr)size)
      fprintf(stderr, ".");
    if (shadow != prev_shadow) {
      prev_shadow = shadow;
      fprintf(stderr, "%02x", (int)*(u8*)shadow);
    }
  }
  fprintf(stderr, "\n");
}

TEST(AddressSanitizer, DISABLED_InternalPrintShadow) {
  for (size_t size = 1; size <= 513; size++) {
    char *ptr = new char[size];
    PrintShadow("m", (uptr)ptr, size);
    delete [] ptr;
    PrintShadow("f", (uptr)ptr, size);
  }
}

TEST(AddressSanitizer, QuarantineTest) {
  BufferedStackTrace stack;
  stack.trace_buffer[0] = 0x890;
  stack.size = 1;

  const int size = 1024;
  void *p = __asan::asan_malloc(size, &stack);
  __asan::asan_free(p, &stack, __asan::FROM_MALLOC);
  size_t i;
  size_t max_i = 1 << 30;
  for (i = 0; i < max_i; i++) {
    void *p1 = __asan::asan_malloc(size, &stack);
    __asan::asan_free(p1, &stack, __asan::FROM_MALLOC);
    if (p1 == p) break;
  }
  EXPECT_GE(i, 10000U);
  EXPECT_LT(i, max_i);
}

void *ThreadedQuarantineTestWorker(void *unused) {
  (void)unused;
  u32 seed = my_rand();
  BufferedStackTrace stack;
  stack.trace_buffer[0] = 0x890;
  stack.size = 1;

  for (size_t i = 0; i < 1000; i++) {
    void *p = __asan::asan_malloc(1 + (my_rand_r(&seed) % 4000), &stack);
    __asan::asan_free(p, &stack, __asan::FROM_MALLOC);
  }
  return NULL;
}

// Check that the thread local allocators are flushed when threads are
// destroyed.
TEST(AddressSanitizer, ThreadedQuarantineTest) {
  const int n_threads = 3000;
  size_t mmaped1 = __sanitizer_get_heap_size();
  for (int i = 0; i < n_threads; i++) {
    pthread_t t;
    PTHREAD_CREATE(&t, NULL, ThreadedQuarantineTestWorker, 0);
    PTHREAD_JOIN(t, 0);
    size_t mmaped2 = __sanitizer_get_heap_size();
    EXPECT_LT(mmaped2 - mmaped1, 320U * (1 << 20));
  }
}

void *ThreadedOneSizeMallocStress(void *unused) {
  (void)unused;
  BufferedStackTrace stack;
  stack.trace_buffer[0] = 0x890;
  stack.size = 1;
  const size_t kNumMallocs = 1000;
  for (int iter = 0; iter < 1000; iter++) {
    void *p[kNumMallocs];
    for (size_t i = 0; i < kNumMallocs; i++) {
      p[i] = __asan::asan_malloc(32, &stack);
    }
    for (size_t i = 0; i < kNumMallocs; i++) {
      __asan::asan_free(p[i], &stack, __asan::FROM_MALLOC);
    }
  }
  return NULL;
}

TEST(AddressSanitizer, ThreadedOneSizeMallocStressTest) {
  const int kNumThreads = 4;
  pthread_t t[kNumThreads];
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_CREATE(&t[i], 0, ThreadedOneSizeMallocStress, 0);
  }
  for (int i = 0; i < kNumThreads; i++) {
    PTHREAD_JOIN(t[i], 0);
  }
}

TEST(AddressSanitizer, ShadowRegionIsPoisonedTest) {
  using __asan::kHighMemEnd;
  // Check that __asan_region_is_poisoned works for shadow regions.
  uptr ptr = kLowShadowBeg + 200;
  EXPECT_EQ(ptr, __asan_region_is_poisoned(ptr, 100));
  ptr = kShadowGapBeg + 200;
  EXPECT_EQ(ptr, __asan_region_is_poisoned(ptr, 100));
  ptr = kHighShadowBeg + 200;
  EXPECT_EQ(ptr, __asan_region_is_poisoned(ptr, 100));
}

// Test __asan_load1 & friends.
TEST(AddressSanitizer, LoadStoreCallbacks) {
  typedef void (*CB)(uptr p);
  CB cb[2][5] = {
      {
        __asan_load1, __asan_load2, __asan_load4, __asan_load8, __asan_load16,
      }, {
        __asan_store1, __asan_store2, __asan_store4, __asan_store8,
        __asan_store16,
      }
  };

  uptr buggy_ptr;

  __asan_test_only_reported_buggy_pointer = &buggy_ptr;
  BufferedStackTrace stack;
  stack.trace_buffer[0] = 0x890;
  stack.size = 1;

  for (uptr len = 16; len <= 32; len++) {
    char *ptr = (char*) __asan::asan_malloc(len, &stack);
    uptr p = reinterpret_cast<uptr>(ptr);
    for (uptr is_write = 0; is_write <= 1; is_write++) {
      for (uptr size_log = 0; size_log <= 4; size_log++) {
        uptr size = 1 << size_log;
        CB call = cb[is_write][size_log];
        // Iterate only size-aligned offsets.
        for (uptr offset = 0; offset <= len; offset += size) {
          buggy_ptr = 0;
          call(p + offset);
          if (offset + size <= len)
            EXPECT_EQ(buggy_ptr, 0U);
          else
            EXPECT_EQ(buggy_ptr, p + offset);
        }
      }
    }
    __asan::asan_free(ptr, &stack, __asan::FROM_MALLOC);
  }
  __asan_test_only_reported_buggy_pointer = 0;
}
