// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb1.json.template > %t/cdb1.json
// RUN: sed "s|DIR|%/t|g" %t/cdb2.json.template > %t/cdb2.json

// RUN: clang-scan-deps -compilation-database %t/cdb1.json -module-files-dir %t/outputs \
// RUN:   -cas-path %t/cas1 -format experimental-include-tree-full \
// RUN:   > %t/result.json
// RUN: echo "=====" >> %t/result.json
// RUN: clang-scan-deps -compilation-database %t/cdb2.json -module-files-dir %t/outputs \
// RUN:   -cas-path %t/cas2 -format experimental-include-tree-full \
// RUN:   >> %t/result.json

// RUN: cat %t/result.json | FileCheck %s -DPREFIX=%/t

// CHECK: "modules": [
// CHECK:   {
// CHECK:     "cache-key": "llvmcas://[[KEY:[[:xdigit:]]+]]"
// CHECK:     "context-hash": "[[HASH:[A-Z0-9]+]]",
// CHECK:     "name": "Mod"
// CHECK:   }
// CHECK: ]
// CHECK: "translation-units": [
// CHECK:   {
// CHECK:     "commands": [
// CHECK:       {
// CHECK:         "clang-module-deps": [
// CHECK:           {
// CHECK:             "context-hash": "[[HASH]]"
// CHECK:             "module-name": "Mod"
// CHECK:           }
// CHECK:         ],
// CHECK:         "command-line": [
// CHECK:           "-fmodule-file-cache-key"
// CHECK:           "[[PREFIX]]/outputs/[[HASH]]/Mod-[[HASH]].pcm"
// CHECK:           "llvmcas://[[KEY]]"
// CHECK:         ]

// CHECK-LABEL: =====

// CHECK: "modules": [
// CHECK:   {
// CHECK:     "cache-key": "llvmcas://[[KEY]]"
// CHECK:     "context-hash": "[[HASH]]"
// CHECK:     "name": "Mod"
// CHECK:   }
// CHECK: "translation-units": [
// CHECK:   {
// CHECK:     "commands": [
// CHECK:       {
// CHECK:         "clang-module-deps": [
// CHECK:           {
// CHECK:             "context-hash": "[[HASH]]"
// CHECK:             "module-name": "Mod"
// CHECK:           }
// CHECK:         ],
// CHECK:         "command-line": [
// CHECK:           "-fmodule-file-cache-key"
// CHECK:           "[[PREFIX]]/outputs/[[HASH]]/Mod-[[HASH]].pcm"
// CHECK:           "llvmcas://[[KEY]]"
// CHECK:         ]

//--- cdb1.json.template
[{
  "directory": "DIR",
  "command": "clang -Xclang -fcas-plugin-path -Xclang /1 -Xclang -fcas-plugin-option -Xclang a=x -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache1",
  "file": "DIR/tu.c"
}]

//--- cdb2.json.template
[{
  "directory": "DIR",
  "command": "clang -Xclang -fcas-plugin-path -Xclang /2 -Xclang -fcas-plugin-option -Xclang b=y -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache2",
  "file": "DIR/tu.c"
}]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu.c
#include "Mod.h"