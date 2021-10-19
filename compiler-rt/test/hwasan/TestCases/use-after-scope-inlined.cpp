// This is the ASAN test of the same name ported to HWAsan.

// Test with "-O2" only to make sure inlining (leading to use-after-scope)
// happens. "always_inline" is not enough, as Clang doesn't emit
// llvm.lifetime intrinsics at -O0.
//
// RUN: %clangxx_hwasan -mllvm -hwasan-use-after-scope -O2 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// REQUIRES: aarch64-target-arch
// REQUIRES: stable-runtime

int *arr;
__attribute__((always_inline)) void inlined(int arg) {
  int x[5];
  for (int i = 0; i < arg; i++)
    x[i] = i;
  arr = x;
}

int main(int argc, char *argv[]) {
  inlined(argc);
  return arr[argc - 1]; // BOOM
  // CHECK: ERROR: HWAddressSanitizer: tag-mismatch
  // CHECK: Cause: stack tag-mismatch
}
