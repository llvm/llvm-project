// RUN: %clangxx -fPIC -emit-llvm -c -o %t %s
// RUN: %lli_orc_jitlink -relocation-model=pic %t | FileCheck %s

// CHECK: catch

#include <stdio.h>

int main(int argc, char *argv[]) {
  try {
    throw 0;
  } catch (int X) {
    puts("catch");
  }
  return 0;
}
