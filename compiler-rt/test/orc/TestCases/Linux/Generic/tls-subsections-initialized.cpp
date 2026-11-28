// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

#include <assert.h>

// Register initialized TLS emitted into a .tdata.* subsection.
thread_local int Data = 7;

int main() {
  assert(Data == 7);
  ++Data;
  assert(Data == 8);
  return 0;
}
