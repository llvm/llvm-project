// Check that fopen64(NULL, "r") is ok.
// RUN: %clang -O2 %s -o %t && %run %t
// REQUIRES: glibc

#define _LARGEFILE64_SOURCE 1

#include <stdio.h>
const char *fn = NULL;
FILE *f;
int main() { f = fopen64(fn, "r"); }
