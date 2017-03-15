// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: clang-include-fixer -db=fuzzyYaml -input=%p/Inputs/fake_yaml_db.yaml %t.cpp --
// RUN: FileCheck %s -input-file=%t.cpp

// include-fixer will add the include, but doesn't complete the symbol.
// CHECK: #include "foobar.h"
// CHECK: fba f;

b::a::fba f;
