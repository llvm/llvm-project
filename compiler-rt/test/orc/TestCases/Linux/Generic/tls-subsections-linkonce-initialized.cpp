// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

#include <assert.h>

// Register initialized TLS in a GNU linkonce section.
thread_local int Data __attribute__((section(".gnu.linkonce.td.data"))) = 11;

int main() {
  assert(Data == 11);
  ++Data;
  assert(Data == 12);
  return 0;
}
