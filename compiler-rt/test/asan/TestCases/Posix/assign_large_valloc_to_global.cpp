// Make sure we don't report a leak nor hang.
// RUN: %clangxx_asan -O3 %s -o %t && %run %t
#include <stdlib.h>
#include <unistd.h>
int *p;
int main() { posix_memalign((void **)&p, 4096, 1 << 20); }
