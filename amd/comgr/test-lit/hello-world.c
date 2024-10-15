// RUN: clang %s -o %t
// RUN: %t | grep -e "hello world"
#include <stdio.h>

int main() {
  puts("hello world");
  return 0;
}
