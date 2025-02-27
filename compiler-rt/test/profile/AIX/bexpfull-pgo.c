// RUN: %clang_pgogen %s -bexpall
// RUN: %clang_pgogen %s -bexpfull

#include <string.h>
int ar[10];
int n;
int main() { memcpy(ar, ar + 1, n); };
