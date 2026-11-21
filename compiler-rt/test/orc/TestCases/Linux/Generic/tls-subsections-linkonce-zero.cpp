// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

#include <assert.h>

// Register zero-initialized TLS in a GNU linkonce section.
thread_local int Zero __attribute__((section(".gnu.linkonce.tb.zero")));

int main() {
  assert(Zero == 0);
  ++Zero;
  assert(Zero == 1);
  return 0;
}
