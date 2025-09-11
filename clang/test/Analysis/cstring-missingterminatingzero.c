// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.MissingTerminatingZero -verify %s

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);

size_t test_init_compound(int i) {
  char src1[6] = {1,2,3,4,5,6};
  char src2[6] = {1,2,3,0,5,6};
  switch (i) {
  case 1:
    return strlen(src1); // expected-warning{{String contains no terminating zero}}
  case 2:
    return strlen(src1 + 1); // expected-warning{{String contains no terminating zero}}
  case 3:
    return strlen(src2);
  case 4:
    return strlen(src2 + 4); // expected-warning{{String contains no terminating zero}}
  case 5:
    return strlen(src2 + 3);
  }
  src1[5] = 0;
  return strlen(src1);
}

typedef char CHAR;

size_t test_init_literal(int i) {
  CHAR src1[] = "abcdef";
  int l = strlen(src1);
  src1[6] = '.';
  src1[3] = 0;
  switch (i) {
  case 1:
    return strlen(src1);
  case 2:
    return strlen(src1 + 4); // expected-warning{{String contains no terminating zero}}
  }
  return l;
}

size_t test_init_assign(int i, char a) {
  char src[6];
  src[1] = '1';
  src[2] = '2';
  src[4] = '4';
  src[5] = '5';

  switch (i) {
  case 0:
    return strlen(src);
  case 1:
    return strlen(src + 1);
  case 2:
    return strlen(src + 2);
  case 3:
    return strlen(src + 3);
  case 4:
    return strlen(src + 4); // expected-warning{{String contains no terminating zero}}
  }
  src[5] = a;
  size_t l = strlen(src + 4);
  src[5] = 0;
  l += strlen(src + 4);
  src[5] = '5';
  return l + strlen(src + 4); // expected-warning{{String contains no terminating zero}}
}

size_t test_assign1() {
  char str1[5] = {'0','1','2','3','4'};
  char str2[5];
  str2[0] = str1[0];
  str2[1] = str1[1];
  str2[4] = str1[4];
  size_t l = strlen(str2);
  return l + strlen(str2 + 4); // expected-warning{{String contains no terminating zero}}
}

size_t test_assign2() {
  char str1[5] = {1,2,3,4,5};
  char str2[5];
  str2[0] = str1[0];
  str2[4] = str2[0];
  return strlen(str2 + 4); // expected-warning{{String contains no terminating zero}}
}

void test_ignore(char *dst) {
  char str1[5] = {1,2,3,4,5};
  strncpy(dst, str1, 5);
  strcpy(dst, str1); // expected-warning{{String contains no terminating zero}}
}
