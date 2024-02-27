// REQUIRES: ondisk_cas

// This test checks that imports of transitively-loaded implementation module
// are not marked as spurious.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

//--- cdb.json.in
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -c DIR/tu.m -o DIR/tu.o -F DIR/frameworks -fmodules -fmodule-name=FW -fmodules-cache-path=DIR/module-cache"
}]

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW { umbrella header "FW.h" }
//--- frameworks/FW.framework/Headers/FW.h
#include <FW/Sub.h>
//--- frameworks/FW.framework/Headers/Sub.h

//--- module.modulemap
module Mod { header "Mod.h" }
//--- Mod.h
#include <FW/Sub.h>
//--- tu.m
#include "Mod.h"
#include <FW/Sub.h>

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full -cas-path %t/cas \
// RUN:   -module-files-dir %t/cas-outputs > %t/cas-deps.json

// RUN: %deps-to-rsp %t/cas-deps.json --module-name=FW  > %t/cas-FW.cc1.rsp
// RUN: %deps-to-rsp %t/cas-deps.json --module-name=Mod > %t/cas-Mod.cc1.rsp
// RUN: %deps-to-rsp %t/cas-deps.json --tu-index=0      > %t/cas-tu.rsp

// RUN: cat %t/cas-tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid > %t/tu-include-tree.txt
// RUN: FileCheck %s -input-file %t/tu-include-tree.txt -DPREFIX=%/t
// CHECK:      [[PREFIX]]/tu.m llvmcas://
// CHECK-NEXT: 1:1 <built-in> llvmcas://
// CHECK-NEXT: 2:1 (Module) Mod
// CHECK-NEXT: 3:1 [[PREFIX]]/frameworks/FW.framework/Headers/Sub.h llvmcas://{{.*}}
// CHECK-NEXT:   Submodule: FW

// RUN: %clang @%t/cas-FW.cc1.rsp
// RUN: %clang @%t/cas-Mod.cc1.rsp
// RUN: %clang @%t/cas-tu.rsp
