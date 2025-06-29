// this test confirms that std::strtok is not available in a freestanding C++ environment.
// it is expected to fail compilation when -ffreestanding is enabled.

// UNSUPPORTED: libcpp-has-no-cstring
// REQUIRES: freestanding
// XFAIL: *

// RUN: %clang %s -c -o /dev/null -ffreestanding --std=c++20 2>&1 | FileCheck %s

#include <cstring> 

int main() {
  char s[] = "hello world";
  char* tok = std::strtok(s, " ");
  (void)tok;
  return 0;
}