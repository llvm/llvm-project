// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// REQUIRES: c++2c
// XFAIL: *

// RUN: %clang %s -c -o /dev/null -ffreestanding --std=c++2c 2>&1

#include <cstring> 

int main() {
  char s[] = "hello world";
  char* tok = std::strtok(s, " ");
  (void)tok;
  return 0;
}
