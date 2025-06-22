// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: %clangxx --coverage main.cpp -o t
// RUN: %run ./t
// RUN: llvm-cov gcov -t t-main. | FileCheck %s

//--- main.cpp
#include <stdio.h>

int main(int argc, char *argv[]) { // CHECK:      2: [[#]]:int main
  puts("");                        // CHECK-NEXT: 2: [[#]]:
#line 3
  puts(""); // line 3
  return 0; // line 4
}
// CHECK-NOT:  {{^ +[0-9]+:}}
