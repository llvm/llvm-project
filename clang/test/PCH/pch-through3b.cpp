// RUN: %clang_cc1 -I %S -emit-pch \
// RUN:   -pch-through-header=Inputs/pch-through1.h -o %t.s3bt1 %s

// RUN: %clang_cc1 -I %S -include-pch %t.s3bt1 \
// RUN:   -pch-through-header=Inputs/pch-through1.h \
// RUN:   %S/Inputs/pch-through-use3b.cpp 2>&1 | FileCheck %s

//CHECK: warning: definition of macro 'AFOO' does not match definition in
//CHECK-SAME: precompiled header
//CHECK: warning: definition of macro 'BFOO' does not match definition in
//CHECK-SAME: precompiled header

#define AFOO 0
#include "Inputs/pch-through1.h"
