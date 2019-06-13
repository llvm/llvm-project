// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/regular_cdb.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header2.h %t.dir/Inputs/header2.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/regular_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1
// RUN: cat %t.dir/regular_cdb.d | FileCheck %s
// RUN: cat %t.dir/regular_cdb2.d | FileCheck --check-prefix=CHECK2 %s
// RUN: rm -rf %t.dir/regular_cdb.d %t.dir/regular_cdb2.d
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 2
// RUN: cat %t.dir/regular_cdb.d | FileCheck %s
// RUN: cat %t.dir/regular_cdb2.d | FileCheck --check-prefix=CHECK2 %s

#include "header.h"

// CHECK: regular_cdb.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
// CHECK-NOT: header2

// CHECK2: regular_cdb.cpp
// CHECK2-NEXT: Inputs{{/|\\}}header.h
// CHECK2-NEXT: Inputs{{/|\\}}header2.h
