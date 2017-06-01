// Test stopset overflow in strcspn function
// RUN: %clang_asan %s -o %t && %env_asan_opts=strict_string_checks=true not %run %t 2>&1 | FileCheck %s

// Test intercept_strcspn asan option
// RUN: %env_asan_opts=intercept_strspn=false %run %t 2>&1

#include <assert.h>
#include <string.h>
#include <sanitizer/asan_interface.h>

int main(int argc, char **argv) {
  size_t r;
  char s1[] = "ab";
  char s2[4] = "abc";
  __asan_poison_memory_region ((char *)&s2[2], 2);
  r = strcspn(s1, s2);
  // CHECK:'s2'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
  assert(r == 0);
  return 0;
}
