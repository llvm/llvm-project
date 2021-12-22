// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -x c++ -std=c++20 %S/Inputs/concept/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 -fprebuilt-module-path=%t -I%S/Inputs/concept %s -fsyntax-only -verify
// expected-no-diagnostics

module;
#include "foo.h"
export module B;
import A;
