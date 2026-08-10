// RUN: %clang_cc1 -verify -I %S -emit-pch \
// RUN:   -pch-through-header=Inputs/pch-through1.h -o %t.s3at1 %s

// RUN: %clang_cc1 -I %S -include-pch %t.s3at1 \
// RUN:   -pch-through-header=Inputs/pch-through1.h \
// RUN:   %S/Inputs/pch-through-use3a.cpp
//expected-no-diagnostics

#define AFOO 0
#include "Inputs/pch-through1.h"
