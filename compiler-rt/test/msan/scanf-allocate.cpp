// RUN: %clangxx_msan -O0 %s -o %t && %run %t >%t.out 2>&1
// FileCheck %s <%t.out

#include <sanitizer/msan_interface.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  char *str;
  sscanf("#string#", "%ms", &str);
  printf("str = %s\n", str);
  __msan_check_mem_is_initialized(str, strlen(str) + 1);
  // CHECK: #string#
}
