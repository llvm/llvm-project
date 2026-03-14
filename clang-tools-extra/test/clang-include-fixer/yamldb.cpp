// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: clang-include-fixer -db=yaml -input=%p/Inputs/fake_yaml_db.yaml %t.cpp --
// RUN: FileCheck %s -input-file=%t.cpp

// CHECK: #include "foo.h"
// CHECK: b::a::foo f;

b::a::foo f;
