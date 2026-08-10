// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -fmodules -fmodule-map-file=module.modulemap -fmodules-cache-path=%t -remap-file "test.cpp;%t/test.cpp"  %t/test.cpp

//--- a.h
#define FOO

//--- module.modulemap
module a {
  header "a.h"
}

//--- test.cpp
#include "a.h"

#ifndef FOO
#error foo
#endif

