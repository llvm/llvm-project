// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: FileCheck %s -input-file %t/deps.json -check-prefix=NO_MODULES
// NO_MODULES: "modules": []

// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid | FileCheck %s -DPREFIX=%/t
// RUN: %clang @%t/tu.rsp

// CHECK: [[PREFIX]]/tu.c llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}

// Note: this is surprising, but correct: when building the implementation files
// of a module, the first include is textual but still uses the submodule
// machinery. The second include is treated as a module import (unless in a PCH)
// but will not actually import the module only trigger visibility changes.

// CHECK: 2:1 [[PREFIX]]/Mod.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   Submodule: Mod
// CHECK: 3:1 (Module for visibility only) Mod

// CHECK: Files:
// CHECK: [[PREFIX]]/tu.c llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: [[PREFIX]]/module.modulemap
// CHECK: [[PREFIX]]/Mod.h llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: [[PREFIX]]/module.modulemap

// RUN: %deps-to-rsp %t/deps.json --tu-index 1 > %t/tu_missing_module.rsp
// RUN: %clang @%t/tu_missing_module.rsp

//--- cdb.json.template
[
{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -I DIR -fmodule-name=Mod -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
},
{
  "file": "DIR/tu_missing_module.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu_missing_module.c -I DIR -fmodule-name=NonExistent -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}
]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h
#pragma once
void top(void);

//--- tu.c
#include "Mod.h"
#include "Mod.h"
void tu(void) {
  top();
}

//--- tu_missing_module.c

