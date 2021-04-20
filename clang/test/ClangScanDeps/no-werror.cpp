// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/no-werror_input.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/sys-header.h %t.dir/Inputs/sys-header.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/no-werror.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | FileCheck %s

#define MACRO 201411L

#include "sys-header.h"

// CHECK: no-werror_input.cpp
// CHECK-NEXT: Inputs{{/|\\}}sys-header.h
