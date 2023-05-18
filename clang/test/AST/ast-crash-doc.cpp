// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -emit-module -x c++ -fmodules -I %t/Inputs -fmodule-name=aa %t/Inputs/module.modulemap -o %t/aa.pcm
// RUN: rm %t/Inputs/b.h
// RUN: not %clang_cc1 -x c++ -Wdocumentation -ast-dump-all -fmodules -I %t/Inputs -fmodule-file=%t/aa.pcm %t/test.cpp | FileCheck %s

//--- Inputs/module.modulemap
module aa {
    header "a.h"
    header "b.h"
}

//--- Inputs/a.h
// empty file

//--- Inputs/b.h
/// test foo @return
int foo();


//--- test.cpp
#include "a.h"

/// test comment at the primary file

int a = foo();


// CHECK: TranslationUnitDecl
