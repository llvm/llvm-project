// RUN: %clang_hwasan %s -o %t && %run %t

#include <string.h>

int main() {
  char a[1];
  memset(a, 0, 0);
  memmove(a, a, 0);
  memcpy(a, a, 0);
}
