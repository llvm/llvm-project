// expected-no-diagnostics
// Create PCH with #pragma hdrstop processing with no #pragma hdrstop
// RUN: %clang_cc1 -verify -I %S -emit-pch -pch-through-hdrstop-create \
// RUN:   -fms-extensions -o %t.pch -x c++-header %s

// Create the PCH object
// RUN: %clang_cc1 -verify -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-create -fms-extensions -o %t.obj -x c++ %s

// The use must still have a #pragma hdrstop
// RUN: %clang_cc1 -verify -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-use -fms-extensions -o %t.obj \
// RUN:   -x c++ %S/Inputs/pch-no-hdrstop-use.cpp

#include "Inputs/pch-through1.h"
static int bar() { return 42; }
#include "Inputs/pch-through2.h"
int pch();
