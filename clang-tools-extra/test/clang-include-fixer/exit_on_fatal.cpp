// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: not clang-include-fixer -db=fixed -input='foo= "foo.h"' %t.cpp --
// RUN: FileCheck %s -input-file=%t.cpp

// CHECK-NOT: #include
// CHECK: #include "doesnotexist.h"
// CHECK-NEXT: foo f;

#include "doesnotexist.h"
foo f;
