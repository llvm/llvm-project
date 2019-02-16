// Create PCH with #pragma hdrstop
// RUN: %clang_cc1 -I %S -emit-pch -pch-through-hdrstop-create \
// RUN:   -fms-extensions -o %t.pch -x c++-header %s

// Use PCH with no #pragma hdrstop
// RUN: not %clang_cc1 -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-hdrstop-use -fms-extensions -o %t.obj -x c++ %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-U %s

#include "Inputs/pch-through1.h"
static int bar() { return 42; }
#include "Inputs/pch-through2.h"
int pch();
//CHECK-U: hdrstop not seen while attempting to use precompiled header
