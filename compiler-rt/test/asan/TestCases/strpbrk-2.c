// Test stopset overflow in strpbrk function
// RUN: %clang_asan %s -o %t && %env_asan_opts=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strpbrk asan option
// RUN: %env_asan_opts=intercept_strpbrk=false %run %t 2>&1

#include <assert.h>
#include <string.h>
#include <sanitizer/asan_interface.h>

int main(int argc, char **argv) {
  char *r;
  char s1[] = "c";
  char s2[4] = "bca";
  __asan_poison_memory_region ((char *)&s2[2], 2);
  r = strpbrk(s1, s2);
  // CHECK:'s2'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
  assert(r == s1);
  return 0;
}
