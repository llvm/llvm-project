// REQUIRES: clang
// UNSUPPORTED: system-windows
// RUN: rm -rf %t && mkdir %t
// RUN: %clangxx %isysroot -I%flang_include %s -o %t/a.out
// RUN: %t/a.out | FileCheck %s

extern "C" {
#include "ISO_Fortran_binding.h"
}
#include <iostream>

int main() {
  std::cout << "PASS\n";
  return 0;
}

// CHECK: PASS
