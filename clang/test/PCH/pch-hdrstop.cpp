// expected-no-diagnostics
// Create PCH with #pragma hdrstop
// RUN: %clang_cc1 -verify -I %S -emit-pch -pch-through-hdrstop-create \
// RUN:   -fms-extensions -o %t.pch -x c++-header %s

// Create PCH object with #pragma hdrstop
// RUN: %clang_cc1 -verify -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-create -fms-extensions -o %t.obj -x c++ %s

// Use PCH with #pragma hdrstop
// RUN: %clang_cc1 -verify -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-use -fms-extensions -o %t.obj \
// RUN:   -x c++ %S/Inputs/pch-hdrstop-use.cpp

// Ensure the PCH stops at the hdrstop
// RUN: %clang_cc1 -ast-dump -I %S -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-use -fms-extensions -o %t.obj \
// RUN:   -x c++ %S/Inputs/pch-hdrstop-use.cpp 2>&1 \
// RUN:   | FileCheck %S/Inputs/pch-hdrstop-use.cpp

#include "Inputs/pch-through1.h"
static int bar() { return 42; }
#include "Inputs/pch-through2.h"
int pch();
#pragma hdrstop

int pch() { return 42*42; }
int other() { return 42; }
