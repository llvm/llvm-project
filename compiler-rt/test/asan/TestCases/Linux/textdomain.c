// RUN: %clang_asan -O0 -g %s -o %t
// RUN: %env_asan_opts=strict_string_checks=1 %run %t

#include <stdlib.h>
#include <libintl.h>

int main() {
  textdomain(NULL);
  return 0;
}
