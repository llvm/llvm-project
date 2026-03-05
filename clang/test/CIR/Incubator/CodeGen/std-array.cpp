// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

void t() {
  std::array<unsigned char, 9> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  (void)v.end();
}

// CHECK: ![[array:.*]] = !cir.record<struct "std::array<unsigned char, 9U>"

// CHECK: cir.call @_ZNSt5arrayIhLj9EE3endEv
