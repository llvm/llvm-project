// REQUIRES: clang
// RUN: not %clang %s -serialize-diagnostics %t
// RUN: clang-read-diagnostics %t 2>&1 | FileCheck %s

#include <stdio.h>

int forgot_return() {
  // CHECK: error_and_warning.c:9:1: non-void function does not return a value [category='Semantic Issue', flag=return-type]
}

int main() {
  // CHECK: error_and_warning.c:13:25: expected ';' after expression [category='Parse Issue']
  printf("Hello world!")
}
