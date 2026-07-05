// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: clang-include-fixer -db=fixed -input='foo= "foo.h","bar.h"' %t.cpp --
// RUN: FileCheck %s -input-file=%t.cpp

// CHECK: #include "foo.h"
// CHECK: foo f;

foo f;
