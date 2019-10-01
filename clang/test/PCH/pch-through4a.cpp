// expected-no-diagnostics
// Create PCH with a through header.
// RUN: %clang_cc1 -verify -I %S -emit-pch \
// RUN: -pch-through-header=Inputs/pch-through1.h \
// RUN:   -fms-extensions -o %t.pch -x c++-header %s

// Create the PCH object
// RUN: %clang_cc1 -verify -I %S -emit-obj -include-pch %t.pch \
// RUN:   -pch-through-header=Inputs/pch-through1.h \
// RUN:   -fms-extensions -o %t.obj -x c++ %s

#define Source(x,y)
#define InOut(size) Source(InOut, (size))
void f(InOut(a) char *b, unsigned long a);
#include "Inputs/pch-through1.h"
int other;
