// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/strip_diag_serialize_input.cpp
// RUN: cp %s %t.dir/strip_diag_serialize_input_clangcl.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/strip_diag_serialize.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 2>&1 | FileCheck %s
// CHECK-NOT: unable to open file
// CHECK: strip_diag_serialize_input.cpp

#warning "diagnostic"
