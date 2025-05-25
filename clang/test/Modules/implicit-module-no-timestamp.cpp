// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: cp a1.h a.h
// RUN: %clang_cc1 -fmodules -fvalidate-ast-input-files-content -fno-pch-timestamp -fmodule-map-file=module.modulemap -fmodules-cache-path=%t test1.cpp
// RUN: cp a2.h a.h
// RUN: %clang_cc1 -fmodules -fvalidate-ast-input-files-content -fno-pch-timestamp -fmodule-map-file=module.modulemap -fmodules-cache-path=%t test2.cpp

//--- a1.h
#define FOO

//--- a2.h
#define BAR

//--- module.modulemap
module a {
  header "a.h"
}

//--- test1.cpp
#include "a.h"

#ifndef FOO
#error foo
#endif

//--- test2.cpp
#include "a.h"

#ifndef BAR
#error bar
#endif
