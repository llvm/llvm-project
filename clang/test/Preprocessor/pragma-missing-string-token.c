// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -emit-module -x c -fmodules -I %t/Inputs -fmodule-name=aa %t/Inputs/module.modulemap -o %t/aa.pcm
// RUN: rm %t/Inputs/b.h
// RUN: not %clang_cc1 -E -fmodules -I %t/Inputs -fmodule-file=%t/aa.pcm %s -o - -fallow-pcm-with-compiler-errors 2>&1 | FileCheck %s

//--- Inputs/module.modulemap
module aa {
    header "a.h"
    header "b.h"
}

//--- Inputs/a.h
#define TEST(x) x

//--- Inputs/b.h
#define SUB "mypragma"

//--- test.c
#include "a.h"

_Pragma(SUB);
int a = TEST(SUB);

// CHECK: int a
// CHECK: 1 error generated
