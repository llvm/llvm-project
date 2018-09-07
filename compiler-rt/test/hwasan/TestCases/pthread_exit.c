// Tests pthread_exit.
// RUN: %clang_hwasan %s -o %t && %run %t
// REQUIRES: stable-runtime
#include <pthread.h>
int main() { pthread_exit(NULL); }
