// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/vfsoverlay.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/vfsoverlay.yaml > %t.dir/vfsoverlay.yaml
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/vfsoverlay_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 | \
// RUN:   FileCheck %s

#include "not_real.h"

// CHECK: clang-scan-deps dependency
// CHECK-NEXT: vfsoverlay.cpp
// CHECK-NEXT: Inputs{{/|\\}}header.h
