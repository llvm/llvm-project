// Check that fopen(NULL, "r") is ok.
// RUN: %clang -O2 %s -o %t && %run %t
#include <stdio.h>
const char *fn = NULL;
FILE *f;
int main() { f = fopen(fn, "r"); }
