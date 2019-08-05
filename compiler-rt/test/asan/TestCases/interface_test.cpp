// Check that user may include ASan interface header.
// RUN: %clang_asan %s -o %t && %run %t
// RUN: %clang_asan -x c %s -o %t && %run %t
// RUN: %clang %s -pie -o %t && %run %t
// RUN: %clang -x c %s -pie -o %t && %run %t
#include <sanitizer/asan_interface.h>

int main() {
  return 0;
}
