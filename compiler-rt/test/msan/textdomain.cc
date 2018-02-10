// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// UNSUPPORTED: netbsd, freebsd

#include <libintl.h>
#include <stdio.h>

int main() {
  const char *td = textdomain("abcd");
  if (td[0] == 0) {
    printf("Try read"); 
  }
  return 0;
}
