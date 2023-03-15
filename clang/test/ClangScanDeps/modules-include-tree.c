// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: FileCheck %s -input-file %t/deps.json  -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT   "modules": [
// CHECK-NEXT     {
// CHECK:            "cas-include-tree-id": "[[LEFT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [
// CHECK:              {
// CHECK:                "module-name": "Top"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/Left-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[LEFT_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file-cache-key"
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/Top-{{.*}}.pcm"
// CHECK-NEXT:         "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:              "-x"
// CHECK-NEXT:         "c"
// CHECK:              "-fmodule-file=Top=[[PREFIX]]/outputs/{{.*}}/Top-{{.*}}.pcm"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=Left"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Left.h"
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK:            ]
// CHECK:            "name": "Left"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "cas-include-tree-id": "[[RIGHT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK:                "module-name": "Top"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[RIGHT_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file-cache-key
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/Top-{{.*}}.pcm"
// CHECK-NEXT:         "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:              "-x"
// CHECK-NEXT:         "c"
// CHECK:              "-fmodule-file=Top=[[PREFIX]]/outputs/{{.*}}/Top-{{.*}}.pcm"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=Right"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Right.h"
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "Right"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "cas-include-tree-id": "[[TOP_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/Top-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[TOP_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-x"
// CHECK-NEXT:         "c"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=Top"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Top.h"
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "Top"
// CHECK:          }
// CHECK:        ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK:                "cas-include-tree-id": "[[TU_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:                "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK:                    "module-name": "Left"
// CHECK:                  }
// CHECK-NEXT:             {
// CHECK:                    "module-name": "Right"
// CHECK:                  }
// CHECK-NEXT:           ]
// CHECK:                "command-line": [
// CHECK-NEXT:             "-cc1"
// CHECK:                  "-fcas-path"
// CHECK-NEXT:             "[[PREFIX]]/cas"
// CHECK:                  "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK:                  "-disable-free"
// CHECK:                  "-fcas-include-tree"
// CHECK-NEXT:             "[[TU_TREE]]"
// CHECK:                  "-fcache-compile-job"
// CHECK:                  "-fsyntax-only"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK-NEXT:             "[[PREFIX]]/outputs/{{.*}}/Left-{{.*}}.pcm"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK-NEXT:             "[[PREFIX]]/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-x"
// CHECK-NEXT:             "c"
// CHECK:                  "-fmodule-file=Left=[[PREFIX]]/outputs/{{.*}}/Left-{{.*}}.pcm"
// CHECK:                  "-fmodule-file=Right=[[PREFIX]]/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK:                  "-fmodules"
// CHECK:                  "-fno-implicit-modules"
// CHECK:                ]
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:           ]
// CHECK:                "input-file": "[[PREFIX]]/tu.c"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
module Top { header "Top.h" export *}
module Left { header "Left.h" export *}
module Right { header "Right.h" export *}

//--- Top.h
#pragma once
struct Top {
  int x;
};
void top(void);

//--- Left.h
#pragma once
#include "Top.h"
void left(void);

//--- Right.h
#pragma once
#include "Top.h"
void right(void);

//--- tu.c
#include "Left.h"
#include "Right.h"

void tu(void) {
  left();
  right();
  top();
}
