// RUN: rm -rf %t
// RUN: split-file %s %t

//--- include/a.h
const char *p = __FILE__;
//--- include/module.modulemap
module "A" { header "a.h" }
//--- src/tu.cpp
#include "a.h"

// RUN: cd %t
// RUN: %clang_cc1 -I ./include -fmodule-name=A -fmodule-map-file=%t/include/module.modulemap %t/src/tu.cpp -E | FileCheck %s

// Make sure that headers that are referenced by module maps have __FILE__
// reflect the include path they were found with. (We make sure they cannot be
// found relative to the includer.)
// CHECK: const char *p = "./include{{/|\\\\}}a.h"
