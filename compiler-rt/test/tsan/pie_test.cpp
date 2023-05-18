// Check if tsan work with PIE binaries.
// RUN: %clang_tsan %s -pie -fpic -o %t && %run %t

int main(void) {
  return 0;
}
