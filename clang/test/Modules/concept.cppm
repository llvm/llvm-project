// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang -std=c++20 %S/Inputs/concept/A.cppm --precompile -o %t/A.pcm
// RUN: %clang -std=c++20 -fprebuilt-module-path=%t -I%S/Inputs/concept %s -c -Xclang -verify
// expected-no-diagnostics

module;
#include "foo.h"
export module B;
import A;
