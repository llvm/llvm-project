// REQUIRES: ondisk_cas

// This test checks that we correctly handle situations where a spurious modular
// dependency (1) turns otherwise textual dependency into modular (2).
//
// (1) For example #include <Spurious/Missing.h>, where the framework Spurious
// has an umbrella header that does not include Missing.h, making it a textual
// include instead.
//
// (2) For example when compiling the implementation file of a module Mod,
// #included headers belonging to Mod are treated textually, unless some other
// module already depends on Mod in its modular form.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

//--- cdb.json.in
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -c DIR/tu.m -o DIR/tu.o -F DIR/frameworks -I DIR/include -fmodule-name=Mod -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- frameworks/Spurious.framework/Modules/module.modulemap
framework module Spurious {
  umbrella header "Spurious.h"
  module * { export * }
}
//--- frameworks/Spurious.framework/Headers/Spurious.h
#include <Mod.h>
//--- frameworks/Spurious.framework/Headers/Missing.h

//--- include/module.modulemap
module Mod { header "Mod.h" }
//--- include/Mod.h
typedef int mod_int;

//--- tu.m
#include <Spurious/Missing.h>
#include <Mod.h>
static mod_int x;

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full \
// RUN:   -module-files-dir %t/outputs > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name=Mod      > %t/Mod.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=Spurious > %t/Spurious.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index=0           > %t/tu.rsp

// RUN: %clang @%t/Mod.cc1.rsp
// RUN: %clang @%t/Spurious.cc1.rsp
// RUN: %clang @%t/tu.rsp

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full -cas-path %t/cas \
// RUN:   -module-files-dir %t/cas-outputs > %t/cas-deps.json

// RUN: %deps-to-rsp %t/cas-deps.json --module-name=Mod      > %t/cas-Mod.cc1.rsp
// RUN: %deps-to-rsp %t/cas-deps.json --module-name=Spurious > %t/cas-Spurious.cc1.rsp
// RUN: %deps-to-rsp %t/cas-deps.json --tu-index=0           > %t/cas-tu.rsp

// RUN: cat %t/cas-tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid > %t/tu-include-tree.txt
// RUN: FileCheck %s -input-file %t/tu-include-tree.txt -DPREFIX=%/t
// CHECK:      [[PREFIX]]/tu.m llvmcas://
// CHECK-NEXT: 1:1 <built-in> llvmcas://
// CHECK-NEXT: 2:1 (Spurious import) (Module) Spurious.Missing [[PREFIX]]/frameworks/Spurious.framework/Headers/Missing.h llvmcas://
// CHECK-NEXT: 3:1 (Module for visibility only) Mod

// RUN: %clang @%t/cas-Mod.cc1.rsp
// RUN: %clang @%t/cas-Spurious.cc1.rsp
// RUN: %clang @%t/cas-tu.rsp
