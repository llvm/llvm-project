// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,alpha.security.ArrayBoundV2 -verify %s
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -analyzer-checker=core,alpha.security.ArrayBoundV2 -DM32 -verify %s
// expected-no-diagnostics

#define UINT_MAX (~0u)

#ifdef M32

#define X86_ARRAY_SIZE (UINT_MAX/2 + 4)

void testIndexTooBig() {
  char arr[X86_ARRAY_SIZE];
  char *ptr = arr + UINT_MAX/2;
  ptr += 2;  // index shouldn't overflow
  *ptr = 42; // no-warning
}

#else // 64-bit tests

#define ARRAY_SIZE 0x100000000

void testIndexOverflow64() {
  char arr[ARRAY_SIZE];
  char *ptr = arr + UINT_MAX/2;
  ptr += 2;  // don't overflow 64-bit index
  *ptr = 42; // no-warning
}

#define ULONG_MAX (~0ul)
#define BIG_INDEX (ULONG_MAX/16)

void testIndexTooBig64() {
  char arr[ULONG_MAX/8-1];
  char *ptr = arr + BIG_INDEX;
  ptr += 2;  // don't overflow 64-bit index
  *ptr = 42; // no-warning
}

#endif
