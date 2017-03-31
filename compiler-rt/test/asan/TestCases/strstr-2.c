// Test needle overflow in strstr function
// RUN: %clang_asan %s -o %t && %env_asan_opts=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strstr asan option
// Disable other interceptors because strlen may be called inside strstr
// RUN: %env_asan_opts=intercept_strstr=false:replace_str=false:intercept_strlen=false %run %t 2>&1

#include <assert.h>
#include <string.h>
#include <sanitizer/asan_interface.h>

int main(int argc, char **argv) {
  char *r = 0;
  char s1[] = "ab";
  char s2[4] = "cab";
  __asan_poison_memory_region ((char *)&s2[2], 2);
  r = strstr(s1, s2);
  // CHECK:'s2'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
  assert(r == 0);
  return 0;
}
