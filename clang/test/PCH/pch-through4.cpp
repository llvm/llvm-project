// expected-no-diagnostics
// Create PCH with #pragma hdrstop processing.
// RUN: %clang_cc1 -verify -I %S -emit-pch -pch-through-hdrstop-create \
// RUN:   -fms-extensions -o %t.pch -x c++-header %s

// Create the PCH object
// RUN: %clang_cc1 -verify -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-create -fms-extensions -o %t.obj -x c++ %s

#pragma once
#include "Inputs/pch-through-macro.h"
void f(InOut(a) char *b, unsigned long a);
