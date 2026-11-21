// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

#include <assert.h>

// Register zero-initialized TLS emitted into a .tbss.* subsection.
thread_local int Zero;

int main() {
  assert(Zero == 0);
  ++Zero;
  assert(Zero == 1);
  return 0;
}
