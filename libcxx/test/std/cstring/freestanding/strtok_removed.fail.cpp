// UNSUPPORTED: libcpp-has-no-cstring
// REQUIRES: freestanding
// XFAIL: *

// RUN: %clang %s -c -o /dev/null -ffreestanding --std=c++2c 2>&1 | FileCheck %s

#include <cstring> 

int main() {
  char s[] = "hello world";
  char* tok = std::strtok(s, " ");
  (void)tok;
  return 0;
}
