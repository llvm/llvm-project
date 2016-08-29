// Tests ASAN_OPTIONS=allocator_release_to_os=1
//

// RUN: %clangxx_asan -std=c++11 %s -o %t
// RUN: %env_asan_opts=allocator_release_to_os=1 %run %t 2>&1 | FileCheck %s --check-prefix=RELEASE
// RUN: %env_asan_opts=allocator_release_to_os=0 %run %t 2>&1 | FileCheck %s --check-prefix=NO_RELEASE
//
// REQUIRES: x86_64-target-arch
#include <stdio.h>
#include <algorithm>
#include <stdint.h>
#include <assert.h>

#include <sanitizer/asan_interface.h>

void MallocReleaseStress() {
  const size_t kNumChunks = 10000;
  const size_t kAllocSize = 100;
  const size_t kNumIter = 100;
  uintptr_t *chunks[kNumChunks] = {0};

  for (size_t iter = 0; iter < kNumIter; iter++) {
    std::random_shuffle(chunks, chunks + kNumChunks);
    size_t to_replace = rand() % kNumChunks;
    for (size_t i = 0; i < kNumChunks; i++) {
      if (chunks[i])
        assert(chunks[i][0] == (uintptr_t)chunks[i]);
      if (i < to_replace) {
        delete [] chunks[i];
        chunks[i] = new uintptr_t[kAllocSize];
        chunks[i][0] = (uintptr_t)chunks[i];
      }
    }
  }
  for (auto p : chunks)
    delete[] p;
}

int main() {
  MallocReleaseStress();
  __asan_print_accumulated_stats();
}

// RELEASE: mapped:{{.*}}releases: {{[1-9]}}
// NO_RELEASE: mapped:{{.*}}releases: 0
