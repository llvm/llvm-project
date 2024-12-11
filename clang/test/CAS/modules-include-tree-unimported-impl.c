// Check that a definition for a symbol in an unimported submodule is not
// visible for codegen.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/tu.rsp -o - | FileCheck %s

// CHECK-NOT: @record = global
// CHECK: @record = external global

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -S -emit-llvm DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
module Mod {
  module A {
    header "A.h"
  }
  explicit module B {
    header "B.h"
  }
}

//--- A.h
extern int record;

//--- B.h
int record = 7;

//--- tu.c
#include "A.h"
int tu(void) {
  return record;
}
