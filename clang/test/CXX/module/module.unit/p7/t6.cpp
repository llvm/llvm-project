// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %S/Inputs/CPP.cppm -I%S/Inputs -o %t/X.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %s -verify
// expected-no-diagnostics
module;
#include "Inputs/h2.h"
export module use;
import X;
void printX(CPP *cpp) {
  cpp->print();
}
