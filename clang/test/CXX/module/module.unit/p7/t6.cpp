// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %S/Inputs/CPP.cppm -I%S/Inputs -o %t/X.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %s -verify
module;
#include "Inputs/h2.h"
export module use;
import X;
void printX(CPP *cpp) {
  cpp->print(); // expected-error {{'CPP' must be defined before it is used}}
                // expected-error@-1 {{'CPP' must be defined before it is used}}
                // expected-error@-2 {{no member named 'print' in 'CPP'}}
                // expected-note@Inputs/CPP.cppm:5 {{definition here is not reachable}}
                // expected-note@Inputs/CPP.cppm:5 {{definition here is not reachable}}
}
