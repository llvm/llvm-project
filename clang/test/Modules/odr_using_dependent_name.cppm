// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang -std=c++20 %S/Inputs/odr_using_dependent_name/X.cppm --precompile -o %t/X.pcm
// RUN: %clang -std=c++20 -I%S/Inputs/odr_using_dependent_name -fprebuilt-module-path=%t %s --precompile -c
// expected-no-diagnostics
module;
#include "foo.h"
export module Y;
import X;
