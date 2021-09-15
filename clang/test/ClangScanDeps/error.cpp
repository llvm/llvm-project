// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/regular_cdb_input.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/regular_cdb.json > %t.cdb
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/regular_cdb_clangcl.json > %t_clangcl.cdb
//
// RUN: not clang-scan-deps -compilation-database %t.cdb -j 1 2>%t.dir/errs
// RUN: echo EOF >> %t.dir/errs
// RUN: FileCheck %s --input-file %t.dir/errs

// RUN: not clang-scan-deps -compilation-database %t_clangcl.cdb -j 1 2>%t.dir/errs_clangcl
// RUN: echo EOF >> %t.dir/errs_clangcl
// RUN: FileCheck %s --input-file %t.dir/errs_clangcl

#include "missing.h"

// CHECK: Error while scanning dependencies
// CHECK-NEXT: error: no such file or directory:
// CHECK-NEXT: error: no input files
// CHECK-NEXT: error:
// CHECK-NEXT: Error while scanning dependencies
// CHECK-NEXT: fatal error: 'missing.h' file not found
// CHECK-NEXT: Error while scanning dependencies
// CHECK-NEXT: fatal error: 'missing.h' file not found
// CHECK-NEXT: EOF
