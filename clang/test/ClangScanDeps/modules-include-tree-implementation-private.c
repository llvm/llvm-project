// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: FileCheck %s -input-file %t/deps.json -check-prefix=NO_MODULES
// NO_MODULES: "modules": []

// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid | FileCheck %s -DPREFIX=%/t
// RUN: %clang @%t/tu.rsp
//
// RUN: FileCheck %s -input-file=%t/tu.d -check-prefix DEPS

// DEPS: dependencies:
// DEPS-DAG: tu.m
// DEPS-DAG: Mod.h
// DEPS-DAG: Priv.h

// CHECK: [[PREFIX]]/tu.m llvmcas://
// CHECK: 1:1 <built-in> llvmcas://
// CHECK: 2:1 [[PREFIX]]/Mod.framework/Headers/Mod.h llvmcas://
// CHECK:   Submodule: Mod
// CHECK: 3:1 [[PREFIX]]/Mod.framework/PrivateHeaders/Priv.h  llvmcas://
// CHECK:   Submodule: Mod_Private
// CHECK: 4:1 (Module for visibility only) Mod
// CHECK: 5:1 (Module for visibility only) Mod_Private
// CHECK: Module Map:
// CHECK: Mod (framework)
// CHECK:   link Mod (framework)
// CHECK: Mod_Private (framework)
// CHECK:   link Mod (framework)

// CHECK: Files:
// CHECK: [[PREFIX]]/tu.m llvmcas://
// CHECK-NOT: [[PREFIX]]/module.modulemap
// CHECK: [[PREFIX]]/Mod.framework/Headers/Mod.h llvmcas://
// CHECK-NOT: [[PREFIX]]/module.modulemap
// CHECK: [[PREFIX]]/Mod.framework/PrivateHeaders/Priv.h llvmcas://
// CHECK-NOT: [[PREFIX]]/module.modulemap

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -F DIR -fmodule-name=Mod -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -MMD -MT dependencies -MF DIR/tu.d"
}]

//--- Mod.framework/Modules/module.modulemap
framework module Mod { header "Mod.h" }

//--- Mod.framework/Modules/module.private.modulemap
framework module Mod_Private { header "Priv.h" }

//--- Mod.framework/Headers/Mod.h
void pub(void);

//--- Mod.framework/PrivateHeaders/Priv.h
void priv(void);

//--- tu.m
#import <Mod/Mod.h>
#import <Mod/Priv.h>
#import <Mod/Mod.h>
#import <Mod/Priv.h>
void tu(void) {
  pub();
  priv();
}
