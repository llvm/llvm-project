// Test ASan interception on `mbrtoc8()` and `c8rtomb()`.
//
// REQUIRES: have-uchar-c8
// RUN: %clangxx_asan -std=c++20 -O0 -g %s -o %t
// RUN: %run %t
// RUN: not %run %t mbrtoc8 2>&1 | FileCheck %s
// RUN: not %run %t c8rtomb 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <uchar.h>

static void test_mbrtoc8_uaf() {
  mbstate_t st = {};
  char8_t *p = (char8_t *)malloc(sizeof(char8_t));
  free(p);
  mbrtoc8(p, "a", 1, &st);
}

static void test_c8rtomb_uaf() {
  mbstate_t st = {};
  char *p = (char *)malloc(MB_CUR_MAX);
  free(p);
  c8rtomb(p, u8'a', &st);
}

static void test_mbrtoc8_short_alloc() {
  mbstate_t st = {};
  char *one_byte = (char *)malloc(1);
  one_byte[0] = 'a';
  char8_t c8 = 0;
  assert(mbrtoc8(&c8, one_byte, 16, &st) == 1);
  assert(c8 == u8'a');
  one_byte[0] = '\0';
  assert(mbrtoc8(&c8, one_byte, 16, &st) == 0);
  assert(c8 == u8'\0');
  free(one_byte);
}

int main(int argc, char **argv) {
  if (argc > 1) {
    if (strcmp(argv[1], "mbrtoc8") == 0) {
      test_mbrtoc8_uaf();
    } else if (strcmp(argv[1], "c8rtomb") == 0) {
      test_c8rtomb_uaf();
    }
    // CHECK: heap-use-after-free
    // CHECK: SUMMARY
    return 0;
  }

  test_mbrtoc8_short_alloc();
  return 0;
}
