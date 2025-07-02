// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.NotNullTerminated -verify %s

#include "Inputs/system-header-simulator.h"

extern void *malloc(size_t);

size_t test_addr_fn() {
  return strlen((char *)&malloc); // expected-warning{{Argument to string length function is the address of the function 'malloc', which is not a null-terminated string}}
}

size_t test_addr_label() {
lab:
  return strlen((char *)&&lab); // expected-warning{{Argument to string length function is the address of the label 'lab', which is not a null-terminated string}}
}

size_t test_init_compound(int i) {
  char src1[6] = {1,2,3,4,5,6};
  char src2[6] = {1,2,3,0,5,6};
  switch (i) {
  case 1:
    return strlen(src1); // expected-warning{{Terminating zero missing from string passed as 1st argument to string length function}}
  case 2:
    return strlen(src1 + 1); // expected-warning{{Terminating zero missing from string}}
  case 3:
    return strlen(src2);
  case 4:
    return strlen(src2 + 4); // expected-warning{{Terminating zero missing from string}}
  case 5:
    return strlen(src2 + 3);
  }
  src1[5] = 0;
  return strlen(src1);
}

size_t test_init_literal(int i) {
  char src1[] = "abcdef";
  int l = strlen(src1);
  src1[6] = '.';
  src1[3] = 0;
  switch (i) {
  case 1:
    return strlen(src1);
  case 2:
    return strlen(src1 + 4); // expected-warning{{Terminating zero missing from string}}
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
    return strlen(src + 4); // expected-warning{{Terminating zero missing from string}}
  }
  src[5] = a;
  size_t l = strlen(src + 4);
  src[5] = 0;
  l += strlen(src + 4);
  src[5] = '5';
  return l + strlen(src + 4); // expected-warning{{Terminating zero missing from string}}
}

size_t test_assign1() {
  char str1[5] = {'0','1','2','3','4'};
  char str2[5];
  str2[0] = str1[0];
  str2[1] = str1[1];
  str2[4] = str1[4];
  size_t l = strlen(str2);
  return l + strlen(str2 + 4); // expected-warning{{Terminating zero missing from string}}
}

size_t test_assign2() {
  char str1[5] = {1,2,3,4,5};
  char str2[5];
  str2[0] = str1[0];
  str2[4] = str2[0];
  return strlen(str2 + 4); // expected-warning{{Terminating zero missing from string}}
}

void test_ignore(char *dst) {
  char str1[5] = {1,2,3,4,5};
  strncpy(dst, str1, 5);
  strcpy(dst, str1); // expected-warning{{Terminating zero missing from string}}
}

size_t test_malloc() {
  char *buf = (char *)malloc(4);
  if (!buf)
    return 0;
  buf[3] = 'a';
  return strlen(buf);
}

extern void f_ext(char *);
char *g_buf = 0;

size_t test_escape1() {
  char buf[4] = {1,2,3,4};
  f_ext(buf);
  return strlen(buf);
}

size_t test_escape2(char *x) {
  char buf[4] = {1,2,3,4};
  g_buf = buf;
  f_ext(x);
  return strlen(buf);
}

size_t test_escape3() {
  char buf[4] = {1,2,3,4};
  f_ext(buf + 3);
  return strlen(buf);
}

void test_str_fn(int i, char *dst) {
  char buf[] = {1, 2, 3};
  switch (i) {
  case 1:
    strcpy(buf, "aa"); // expected-warning{{Terminating zero missing from string}}
    break;
  case 2:
    strcpy(dst, buf); // expected-warning{{Terminating zero missing from string}}
    break;
  case 3:
    strncpy(buf, "aa", 3);
    break;
  case 4:
    strncpy(dst, buf, 3);
    break;
  case 5:
    strcat(buf, "aa"); // expected-warning{{Terminating zero missing from string}}
    break;
  case 6:
    strcat(dst, buf); // expected-warning{{Terminating zero missing from string}}
    break;
  case 7:
    strncat(buf, "aa", 3); // expected-warning{{Terminating zero missing from string}}
    break;
  case 8:
    strncat(dst, buf, 3);
    break;
  }
}
