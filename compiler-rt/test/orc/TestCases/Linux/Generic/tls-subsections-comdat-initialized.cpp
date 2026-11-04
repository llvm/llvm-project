// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

#include <assert.h>

// Register initialized COMDAT TLS in an inline-function static.
inline int &inlineData() {
  static thread_local int Value = 13;
  return Value;
}

int main() {
  assert(inlineData() == 13);
  ++inlineData();
  assert(inlineData() == 14);
  return 0;
}
