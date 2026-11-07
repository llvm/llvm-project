// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

#include <assert.h>

// Register zero-initialized COMDAT TLS in an inline-function static.
inline int &inlineZero() {
  static thread_local int Value;
  return Value;
}

int main() {
  assert(inlineZero() == 0);
  ++inlineZero();
  assert(inlineZero() == 1);
  return 0;
}
