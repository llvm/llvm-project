// Test ASan interception on `mbrtoc32()` and `c32rtomb()`.
//
// UNSUPPORTED: darwin, android, netbsd
// RUN: %clangxx_asan -O0 -g %s -o %t
// RUN: %run %t
// RUN: not %run %t mbrtoc32 2>&1 | FileCheck %s
// RUN: not %run %t c32rtomb 2>&1 | FileCheck %s

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <uchar.h>

static void test_mbrtoc32_uaf() {
  mbstate_t st = {};
  char32_t *p = (char32_t *)malloc(sizeof(char32_t));
  free(p);
  mbrtoc32(p, "a", 1, &st);
}

static void test_c32rtomb_uaf() {
  mbstate_t st = {};
  char *p = (char *)malloc(MB_CUR_MAX);
  free(p);
  c32rtomb(p, U'a', &st);
}

// Passing n larger than the 1-byte allocation is valid when 1 byte completes
// the character.
static void test_mbrtoc32_short_alloc() {
  mbstate_t st = {};
  char *one_byte = (char *)malloc(1);
  one_byte[0] = 'a';
  char32_t c32 = 0;
  assert(mbrtoc32(&c32, one_byte, 16, &st) == 1);
  assert(c32 == U'a');
  free(one_byte);
}

int main(int argc, char **argv) {
  if (argc > 1) {
    if (strcmp(argv[1], "mbrtoc32") == 0) {
      test_mbrtoc32_uaf();
    } else if (strcmp(argv[1], "c32rtomb") == 0) {
      test_c32rtomb_uaf();
    }
    // CHECK: heap-use-after-free
    // CHECK: SUMMARY
    return 0;
  }

  test_mbrtoc32_short_alloc();
  return 0;
}
