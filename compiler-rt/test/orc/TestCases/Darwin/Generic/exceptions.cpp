// RUN: %clangxx -c -o %t %s
// RUN: %llvm_jitlink -slab-allocate=20Mb %t
//
// REQUIRES: system-darwin && host-arch-compatible

// Test that trivial throw / catch works.
int main(int argc, char *argv[]) {
  try {
    throw 42;
  } catch (int E) {
    return 42 - E;
  }
  return 1;
}
