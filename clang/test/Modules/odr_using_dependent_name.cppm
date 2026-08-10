// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -std=c++20 %S/Inputs/odr_using_dependent_name/X.cppm -emit-module-interface -o %t/X.pcm
// RUN: %clang_cc1 -std=c++20 -I%S/Inputs/odr_using_dependent_name -fprebuilt-module-path=%t %s -fsyntax-only -verify
// expected-no-diagnostics
module;
#include "foo.h"
export module Y;
import X;
