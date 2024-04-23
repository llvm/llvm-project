// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -format experimental-include-tree-full > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Mod_Private > %t/private.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Mod > %t/mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/private.rsp
// RUN: %clang @%t/mod.rsp
// RUN: %clang @%t/tu.rsp -dependency-dot %t/tu.dot
/// Check dependency file is generated.
// RUN: find %t/module-cache -name "*.d" | wc -l | grep 2
// RUN: FileCheck %s -input-file=%t/tu.d

// CHECK: dependencies:
// CHECK-DAG: tu.m
// CHECK-DAG: A.h

// RUN: FileCheck %s -input-file=%t/tu.dot -check-prefix DOT
// DOT: digraph "dependencies"
// DOT-DAG: [[TU:header_[0-9]+]] [ shape="box", label="{{.*}}{{/|\\}}tu.m"];
// DOT-DAG: [[HEADER:header_[0-9]+]] [ shape="box", label="{{.*}}{{/|\\}}A.h"];
// DOT-DAG: [[PCM:header_[0-9]+]] [ shape="box", label="{{.*}}{{/|\\}}Mod-{{.*}}.pcm"];
// DOT-DAG: [[TU]] -> [[HEADER]]
// DOT-DAG: [[HEADER]] -> [[PCM]]

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -F DIR -I DIR -fmodule-name=A -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -MMD -MT dependencies -MF DIR/tu.d"
}]

//--- Mod.framework/Modules/module.modulemap
framework module Mod { header "Mod.h" }

//--- Mod.framework/Modules/module.private.modulemap
framework module Mod_Private { header "Priv.h" }

//--- module.modulemap
module A {
  header "A.h"
  export *
}

//--- A.h
#include <Mod/Mod.h>

//--- Mod.framework/Headers/Mod.h
#include <Mod/Priv.h>
void pub(void);

//--- Mod.framework/PrivateHeaders/Priv.h
void priv(void);

//--- tu.m
#import "A.h"
