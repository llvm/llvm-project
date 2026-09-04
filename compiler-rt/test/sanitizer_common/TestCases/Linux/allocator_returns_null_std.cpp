// Test the behavior of malloc called from std:: when the allocation size
// exceeds the sanitizer's allocator max allowed one.

// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=allocator_may_return_null=0 not %run %t 2>&1 | FileCheck %s

// UBSAN has no allocator.
// UNSUPPORTED: ubsan

// REQUIRES: x86_64-target-arch

#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char **argv) {
  // The maximum value of all supported sanitizers (search for
  // kMaxAllowedMallocSize). For ASan + LSan, ASan limit is used.
  constexpr size_t kMaxAllowedMallocSizePlusOne = (1ULL << 40) + 1;

  std::vector<char> v;
  v.resize(kMaxAllowedMallocSizePlusOne);

  fprintf(stderr, "x: %lx\n", (long)v.data());

  return 0;
}

// CHECK: #{{[0-9]+.*}}allocator_returns_null_std.cpp
// std::vector::resize uses throwing operator new[]. asan forces
// may_return_null=true on Allocate so std::get_new_handler() runs first; the
// chain-exhausted abort path emits "out-of-memory" rather than the in-place
// "allocation-size-too-big" emitted by other sanitizers.
// CHECK: {{SUMMARY: .*Sanitizer: (allocation-size-too-big|out-of-memory).*allocator_returns_null_std.cpp.*}} in main
