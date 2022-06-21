// REQUIRES: ondisk_cas
// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: rm -rf %t_clangcl.cdb
// RUN: rm -rf %t.module-cache
// RUN: rm -rf %t.module-cache_clangcl
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: cp %s %t.dir/modules_cdb_input2.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header2.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/module.modulemap %t.dir/Inputs/module.modulemap
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/modules_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -cas-path %t.dir/cas -format experimental-tree -compilation-database %t.cdb -j 1 -mode preprocess-dependency-directives | \
// RUN:   FileCheck %s

#include "header.h"

/// Check simple output tree output. There are 4 entries in the cdb,
/// First one is for modules_cdb_input2.cpp
// CHECK: tree llvmcas://{{[[:xdigit:]]+}}
// CHECK-SAME: modules_cdb_input2.cpp

/// Second one is for modules_cdb_input1.cpp, but it needs to load the header and the module.
// CHECK: tree llvmcas://{{[[:xdigit:]]+}}
// CHECK-SAME: modules_cdb_input.cpp

/// Third and fourth only need to load module, thus they return the same hash.
// CHECK: tree [[HASH:llvmcas://[[:xdigit:]]+]]
// CHECK-SAME: modules_cdb_input.cpp
// CHECK: tree {{.*}}[[HASH]]
// CHECK-SAME: modules_cdb_input.cpp
