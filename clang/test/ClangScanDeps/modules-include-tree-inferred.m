// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: cat %t/Mod.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Mod.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Mod.casid | FileCheck %s -DPREFIX=%/t
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/tu.rsp

// CHECK: <module-includes> llvmcas://
// CHECK: 1:1 <built-in> llvmcas://
// CHECK: 2:1 [[PREFIX]]/Mod.framework/Headers/Mod.h llvmcas://
// CHECK:   Submodule: Mod
// CHECK: Module Map:
// CHECK: Mod (framework)
// CHECK: Files:
// CHECK-NOT: [[PREFIX]]/module.modulemap
// CHECK: [[PREFIX]]/Mod.framework/Headers/Mod.h llvmcas://
// CHECK-NOT: [[PREFIX]]/module.modulemap

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -F DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
framework module * {}

//--- Mod.framework/Headers/Mod.h
void pub(void);

//--- tu.m
#import <Mod/Mod.h>
void tu(void) {
  pub();
}
