// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

#include <assert.h>

// Merge TLS subsections into existing canonical sections, preserving both
// initialized and zero-initialized values across all section forms.
thread_local int Data = 7;
thread_local int Zero;
thread_local int LinkOnceData
    __attribute__((section(".gnu.linkonce.td.data"))) = 11;
thread_local int LinkOnceZero __attribute__((section(".gnu.linkonce.tb.zero")));

inline int &inlineData() {
  static thread_local int Value = 13;
  return Value;
}

inline int &inlineZero() {
  static thread_local int Value;
  return Value;
}

thread_local int CanonicalData __attribute__((section(".tdata"))) = 17;
thread_local int CanonicalZero __attribute__((section(".tbss")));

int main() {
  assert(Data == 7);
  ++Data;
  assert(Zero == 0);
  ++Zero;
  assert(LinkOnceData == 11);
  ++LinkOnceData;
  assert(LinkOnceZero == 0);
  ++LinkOnceZero;
  assert(inlineData() == 13);
  ++inlineData();
  assert(inlineZero() == 0);
  ++inlineZero();
  assert(CanonicalData == 17);
  ++CanonicalData;
  assert(CanonicalZero == 0);
  ++CanonicalZero;
  assert(Data == 8);
  assert(Zero == 1);
  assert(LinkOnceData == 12);
  assert(LinkOnceZero == 1);
  assert(inlineData() == 14);
  assert(inlineZero() == 1);
  assert(CanonicalData == 18);
  assert(CanonicalZero == 1);
  return 0;
}
