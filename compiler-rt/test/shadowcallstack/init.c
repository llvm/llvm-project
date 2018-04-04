// RUN: %clang_scs -D INCLUDE_RUNTIME %s -o %t
// RUN: %run %t

// RUN: %clang_scs %s -o %t
// RUN: not --crash %run %t

// Basic smoke test for the runtime

#ifdef INCLUDE_RUNTIME
#include "minimal_runtime.h"
#endif

int main(int argc, char **argv) {
  printf("In main.\n");
  return 0;
}
