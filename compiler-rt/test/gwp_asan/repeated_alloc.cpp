// REQUIRES: gwp_asan
// This test ensures that normal allocation/memory access/deallocation works
// as expected and we didn't accidentally break the supporting allocator.

// RUN: %clangxx_gwp_asan %s -o %t
// RUN: %env_gwp_asan_options=MaxSimultaneousAllocations=1 %run %t
// RUN: %env_gwp_asan_options=MaxSimultaneousAllocations=2 %run %t
// RUN: %env_gwp_asan_options=MaxSimultaneousAllocations=11 %run %t
// RUN: %env_gwp_asan_options=MaxSimultaneousAllocations=12 %run %t
// RUN: %env_gwp_asan_options=MaxSimultaneousAllocations=13 %run %t

#include <cstdlib>

int main() {
  void* Pointers[16];
  for (unsigned i = 0; i < 16; ++i) {
    char *Ptr = reinterpret_cast<char*>(malloc(1 << i));
    Pointers[i] = Ptr;
    *Ptr = 0;
    Ptr[(1 << i) - 1] = 0;
  }

  for (unsigned i = 0; i < 16; ++i) {
    free(Pointers[i]);
  }

  return 0;
}
