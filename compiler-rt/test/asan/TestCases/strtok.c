// RUN: %clang_asan %s -o %t

// Test overflows with strict_string_checks

// RUN: %env_asan_opts=strict_string_checks=true not %run %t test1 2>&1 | \
// RUN:    FileCheck %s --check-prefix=CHECK1
// RUN: %env_asan_opts=intercept_strtok=false %run %t test1 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t test2 2>&1 | \
// RUN:    FileCheck %s --check-prefix=CHECK2
// RUN: %env_asan_opts=intercept_strtok=false %run %t test2 2>&1
// RUN: %env_asan_opts=strict_string_checks=true not %run %t test3 2>&1 | \
// RUN:    FileCheck %s --check-prefix=CHECK3
// RUN: %env_asan_opts=intercept_strtok=false %run %t test3 2>&1
// RUN: %env_asan_opts=strict_string_checks=true %run %t test4 2>&1
// RUN: %env_asan_opts=intercept_strtok=false %run %t test4 2>&1

// Test overflows with !strict_string_checks
// RUN: %env_asan_opts=strict_string_checks=false not %run %t test5 2>&1 | \
// RUN:    FileCheck %s --check-prefix=CHECK5
// RUN: %env_asan_opts=intercept_strtok=false %run %t test5 2>&1
// RUN: %env_asan_opts=strict_string_checks=false not %run %t test6 2>&1 | \
// RUN:    FileCheck %s --check-prefix=CHECK6
// RUN: %env_asan_opts=intercept_strtok=false %run %t test6 2>&1


#include <assert.h>
#include <string.h>
#include <sanitizer/asan_interface.h>

// Check that we find overflows in the delimiters on the first call
// with strict_string_checks.
void test1() {
  char *token;
  char s[4] = "abc";
  char token_delimiter[2] = "b";
  __asan_poison_memory_region ((char *)&token_delimiter[1], 2);
  token = strtok(s, token_delimiter);
  // CHECK1: 'token_delimiter'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
}

// Check that we find overflows in the delimiters on the second call (str == NULL)
// with strict_string_checks.
void test2() {
  char *token;
  char s[4] = "abc";
  char token_delimiter[2] = "b";
  token = strtok(s, token_delimiter);
  assert(strcmp(token, "a") == 0);
  __asan_poison_memory_region ((char *)&token_delimiter[1], 2);
  token = strtok(NULL, token_delimiter);
  // CHECK2: 'token_delimiter'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
}

// Check that we find overflows in the string (only on the first call) with strict_string_checks.
void test3() {
  char *token;
  char s[4] = "abc";
  char token_delimiter[2] = "b";
  __asan_poison_memory_region ((char *)&s[3], 2);
  token = strtok(s, token_delimiter);
  // CHECK3: 's'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
}

// Check that we do not crash when strtok returns NULL with strict_string_checks.
void test4() {
  char *token;
  char s[] = "";
  char token_delimiter[] = "a";
  token = strtok(s, token_delimiter);
  assert(token == NULL);
}

// Check that we find overflows in the string (only on the first call) with !strict_string_checks.
void test5() {
  char *token;
  char s[4] = "abc";
  char token_delimiter[2] = "d";
  __asan_poison_memory_region ((char *)&s[2], 2);
  __asan_poison_memory_region ((char *)&token_delimiter[1], 2);
  token = strtok(s, token_delimiter);
  // CHECK5: 's'{{.*}} <== Memory access at offset {{[0-9]+}} partially overflows this variable
}

// Check that we find overflows in the delimiters (only on the first call) with !strict_string_checks.
void test6() {
  char *token;
  char s[4] = "abc";
  char token_delimiter[1] = {'d'};
  __asan_poison_memory_region ((char *)&token_delimiter[1], 2);
  token = strtok(s, &token_delimiter[1]);
  // CHECK6: 'token_delimiter'{{.*}} <== Memory access at offset {{[0-9]+}} overflows this variable
}

int main(int argc, char **argv) {
  if (argc != 2) return 1;
  if (!strcmp(argv[1], "test1")) test1();
  if (!strcmp(argv[1], "test2")) test2();
  if (!strcmp(argv[1], "test3")) test3();
  if (!strcmp(argv[1], "test4")) test4();
  if (!strcmp(argv[1], "test5")) test5();
  if (!strcmp(argv[1], "test6")) test6();
  return 0;
}
