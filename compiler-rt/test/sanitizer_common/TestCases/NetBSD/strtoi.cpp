// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <inttypes.h>
#include <stdio.h>

void test_strtoi(const char *nptr, int base, intmax_t lo, intmax_t hi) {
  char *p;
  int status;
  intmax_t i = strtoi(nptr, &p, base, lo, hi, &status);
  printf("strtoi: conversion of '%s' to a number %s, using %jd, p=%#" PRIx8
         "\n",
         nptr, status ? "failed" : "successful", i, *p);
}

void test_strtou(const char *nptr, int base, intmax_t lo, intmax_t hi) {
  char *p;
  int status;
  uintmax_t i = strtou(nptr, &p, base, lo, hi, &status);
  printf("strtou: conversion of '%s' to a number %s, using %ju, p=%#" PRIx8
         "\n",
         nptr, status ? "failed" : "successful", i, *p);
}

int main(void) {
  printf("strtoi\n");

  test_strtoi("100", 0, 1, 100);
  test_strtoi("100", 0, 1, 10);
  test_strtoi("100xyz", 0, 1, 100);
  test_strtou("100", 0, 1, 100);
  test_strtou("100", 0, 1, 10);
  test_strtou("100xyz", 0, 1, 100);

  // CHECK: strtoi
  // CHECK: strtoi: conversion of '100' to a number successful, using 100, p=0
  // CHECK: strtoi: conversion of '100' to a number failed, using 10, p=0
  // CHECK: strtoi: conversion of '100xyz' to a number failed, using 100, p=0x78
  // CHECK: strtou: conversion of '100' to a number successful, using 100, p=0
  // CHECK: strtou: conversion of '100' to a number failed, using 10, p=0
  // CHECK: strtou: conversion of '100xyz' to a number failed, using 100, p=0x78

  return 0;
}
