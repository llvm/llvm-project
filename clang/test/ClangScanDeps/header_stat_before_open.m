// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/header_stat_before_open_input.m
// RUN: cp %s %t.dir/header_stat_before_open_input_clangcl.m
// RUN: mkdir %t.dir/Inputs
// RUN: cp -R %S/Inputs/frameworks %t.dir/Inputs/frameworks
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/header_stat_before_open_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | \
// RUN:   FileCheck %s

#include "Framework/Framework.h"
#include "Framework/PrivateHeader.h"

// CHECK: header_stat_before_open_input.o
// CHECK-NEXT: header_stat_before_open_input.m
// CHECK-NEXT: Inputs{{/|\\}}frameworks{{/|\\}}Framework.framework{{/|\\}}Headers{{/|\\}}Framework.h
// CHECK-NEXT: Inputs{{/|\\}}frameworks{{/|\\}}Framework.framework{{/|\\}}PrivateHeaders{{/|\\}}PrivateHeader.h

// CHECK: header_stat_before_open_input_clangcl.o
// CHECK-NEXT: header_stat_before_open_input_clangcl.m
// CHECK-NEXT: Inputs{{/|\\}}frameworks{{/|\\}}Framework.framework{{/|\\}}Headers{{/|\\}}Framework.h
// CHECK-NEXT: Inputs{{/|\\}}frameworks{{/|\\}}Framework.framework{{/|\\}}PrivateHeaders{{/|\\}}PrivateHeader.h
