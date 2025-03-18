// RUN: %clangxx -c -o %t %s
// RUN: %llvm_jitlink -slab-allocate=20Mb %t
//
// REQUIRES: system-darwin && host-arch-compatible

// Test that we can throw and catch an exception through a large number of
// stack frames. The number (1022) is chosen to force emission of multiple
// unwind info second-level pages.

template <int N>
void f() { try { f<N - 1>(); } catch (...) { throw; } }

template <>
void f<0>() { throw 42; }

int main(int argc, char *argv[]) {
  try {
    f<1020>();
  } catch (int n) {
    return 42 - n;
  }
  return 1;
}

