// RUN: %clangxx -fPIC -fdata-sections -c %s -o %t.o
// RUN: %llvm_jitlink %t.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.o
// RUN: %clangxx -fPIC -fdata-sections -DWITH_CANONICAL -c %s -o %t.mixed.o
// RUN: %llvm_jitlink %t.mixed.o
// RUN: %llvm_jitlink -slab-allocate=1MB %t.mixed.o
// REQUIRES: system-linux, host-arch-compatible
// REQUIRES: host-jit-triple={{(x86_64|aarch64)-.*}}

// Register initialized, zero-initialized, COMDAT, and GNU linkonce TLS
// subsections, both with and without existing canonical TLS sections.
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

#ifdef WITH_CANONICAL
thread_local int CanonicalData __attribute__((section(".tdata"))) = 17;
thread_local int CanonicalZero __attribute__((section(".tbss")));
#endif

int main() {
  if (Data++ != 7 || Zero++ != 0 || LinkOnceData++ != 11 ||
      LinkOnceZero++ != 0 || inlineData()++ != 13 || inlineZero()++ != 0)
    return 1;
  if (Data != 8 || Zero != 1 || LinkOnceData != 12 || LinkOnceZero != 1 ||
      inlineData() != 14 || inlineZero() != 1)
    return 2;
#ifdef WITH_CANONICAL
  if (CanonicalData++ != 17 || CanonicalZero++ != 0)
    return 3;
#endif
  return 0;
}
