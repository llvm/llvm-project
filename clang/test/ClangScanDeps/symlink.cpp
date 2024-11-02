// REQUIRES: shell
// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/symlink_input.cpp
// RUN: cp %s %t.dir/symlink_input2.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: ln -s %t.dir/Inputs/header.h %t.dir/Inputs/symlink.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/symlink_cdb.json > %t.cdb
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | FileCheck %s

#include "symlink.h"
#include "header.h"

// CHECK: symlink_input.cpp
// CHECK-NEXT: Inputs{{/|\\}}symlink.h
// CHECK-NEXT: Inputs{{/|\\}}header.h

// CHECK: symlink_input2.cpp
// CHECK-NEXT: Inputs{{/|\\}}symlink.h
// CHECK-NEXT: Inputs{{/|\\}}header.h
