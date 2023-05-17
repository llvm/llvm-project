// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/vfsoverlay_input.cpp
// RUN: cp %s %t.dir/vfsoverlay_input_clangcl.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/vfsoverlay.yaml > %t.dir/vfsoverlay.yaml
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/vfsoverlay_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -mode preprocess-dependency-directives -j 1 | FileCheck %s
// RUN: clang-scan-deps -compilation-database %t.cdb -mode preprocess                       -j 1 | FileCheck %s

#include "not_real.h"

// CHECK: vfsoverlay_input.o
// CHECK-NEXT: vfsoverlay_input.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h

// CHECK: vfsoverlay_input_clangcl.o
// CHECK-NEXT: vfsoverlay_input_clangcl.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
