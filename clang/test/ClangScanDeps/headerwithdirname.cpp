// RUN: rm -rf %t.dir
// RUN: rm -rf %t.dir/foodir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: mkdir -p %t.dir/foodir
// RUN: cp %s %t.dir/headerwithdirname_input.cpp
// RUN: cp %s %t.dir/headerwithdirname_input_clangcl.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/foodir %t.dir/Inputs/foodir
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/headerwithdirname.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | FileCheck %s %if system-darwin %{ --check-prefixes=CHECK,CHECK-DARWIN %}

#include <foodir>

// CHECK: headerwithdirname_input{{\.o|.*\.s}}
// CHECK-DARWIN-NEXT: SDKSettings.json
// CHECK-NEXT: headerwithdirname_input.cpp
// CHECK-NEXT: Inputs{{/|\\}}foodir

// CHECK: headerwithdirname_input_clangcl.o
// CHECK-NEXT: headerwithdirname_input_clangcl.cpp
// CHECK-NEXT: Inputs{{/|\\}}foodir
