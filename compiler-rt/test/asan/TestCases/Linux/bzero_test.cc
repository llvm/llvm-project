// RUN: %clangxx_asan  %s -o %t
// RUN: not %run %t   2>&1 | FileCheck %s --check-prefix=A1
// RUN: %env_asan_opts=replace_intrin=0 %run %t

#include <strings.h>
int main(int argc, char **argv) {
  char a1[] = {1, 2, 3, 4, 5, 6, 7, 8};
  bzero(a1, sizeof(a1) + 1);  // BOOM
  // A1: AddressSanitizer: stack-buffer-overflow
  // A1: {{#0.*memset}}
  // A1: 'a1' <== Memory access at offset
  return 0;
}
