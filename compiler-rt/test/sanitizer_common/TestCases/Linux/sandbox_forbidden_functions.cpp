// RUN: %clangxx %s -o %t && %run %t 2>&1 | FileCheck %s

// REQUIRES: target-x86 || target-x86_64

#include <stdio.h>
#include <stdlib.h>

// Functions that sandboxes don't like. If a sanitizer calls it, this test will
// likely fail. (There will be a false negative if the sanitizer only calls it
// during an obscure code path that is not exercised by this test.)
//
// Known false positive: TSan with high-entropy ASLR (in a non-sandboxed
//                       environment)
extern "C" int personality(unsigned long) { abort(); }

int main(int argc, char **argv) {
  printf("Hello World!\n");
  return 0;
}

// CHECK: Hello World!
