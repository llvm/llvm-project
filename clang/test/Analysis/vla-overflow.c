// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -analyzer-checker=core -verify %s

typedef unsigned long size_t;
#define BIGINDEX 65536U

size_t check_VLA_overflow_sizeof(unsigned int x) {
  if (x == BIGINDEX) {
    // We expect here that size_t is a 64 bit value.
    // Size of this array should be the first to overflow.
    size_t s = sizeof(char[x][x][x][x]); // expected-warning{{Declared variable-length array (VLA) has too large size [core.VLASize]}}
    return s;
  }
  return 0;
}

void check_VLA_overflow_typedef(void) {
  unsigned int x = BIGINDEX;
  typedef char VLA[x][x][x][x]; // expected-warning{{Declared variable-length array (VLA) has too large size [core.VLASize]}}
}

void check_VLA_no_overflow(void) {
  unsigned int x = BIGINDEX;
  typedef char VLA[x][x][x][x - 1];
  typedef char VLA1[0xffffffffu];
}
