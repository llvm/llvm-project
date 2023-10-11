// Check that fopen64(NULL, "r") is ok.
// `-m32` and `-D_FILE_OFFSET_BITS=64` will make fopen() call fopen64()

// REQUIRES: asan
// RUN: %clang -m32 -D_FILE_OFFSET_BITS=64 -O2 %s -o %t && %run %t
#include <stdio.h>
const char *fn = NULL;
FILE *f;
int main() { f = fopen(fn, "r"); }
