// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Extract the include-tree commands
// RUN: %deps-to-rsp %t/deps.json --module-name Top > %t/Top.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Left > %t/Left.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Right > %t/Right.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name ZAtImport > %t/ZAtImport.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name ZPragmaImport > %t/ZPragmaImport.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// Extract include-tree casids
// RUN: cat %t/Top.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Top.casid
// RUN: cat %t/Left.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Left.casid
// RUN: cat %t/Right.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/Right.casid
// RUN: cat %t/tu.rsp | sed -E 's|.*"-fcas-include-tree" "(llvmcas://[[:xdigit:]]+)".*|\1|' > %t/tu.casid

// RUN: echo "MODULE Top" > %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Top.casid >> %t/result.txt
// RUN: echo "MODULE Left" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Left.casid >> %t/result.txt
// RUN: echo "MODULE Right" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/Right.casid >> %t/result.txt
// RUN: echo "TRANSLATION UNIT" >> %t/result.txt
// RUN: clang-cas-test -cas %t/cas -print-include-tree @%t/tu.casid >> %t/result.txt
// RUN: cat %t/deps.json >> %t/result.txt

// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%/t

// CHECK-LABEL: MODULE Top
// CHECK: <module-includes> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 [[PREFIX]]/Top.h llvmcas://{{[[:xdigit:]]+}}
// CHECK: Module Map:
// CHECK: Top
// CHECK:   export *
// CHECK: Files:
// CHECK-NOT: [[PREFIX]]/module.modulemap
// CHECK: [[PREFIX]]/Top.h llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: [[PREFIX]]/module.modulemap

// CHECK-LABEL: MODULE Left
// CHECK: <module-includes> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 [[PREFIX]]/Left.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   2:1 (Module) Top
// CHECK: Module Map:
// CHECK: Left
// CHECK:   export *
// CHECK: Files:
// CHECK-NOT: [[PREFIX]]/module.modulemap llvmcas://{{[[:xdigit:]]+}}
// CHECK: [[PREFIX]]/Left.h llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: [[PREFIX]]/module.modulemap llvmcas://{{[[:xdigit:]]+}}
// CHECK: [[PREFIX]]/Top.h llvmcas://{{[[:xdigit:]]+}}

// CHECK-LABEL: MODULE Right
// CHECK: <module-includes> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 [[PREFIX]]/Right.h llvmcas://{{[[:xdigit:]]+}}
// CHECK:   2:1 (Module) Top
// CHECK: Module Map:
// CHECK: Right
// CHECK:   export *
// CHECK: Files:
// CHECK-NOT: [[PREFIX]]/module.modulemap llvmcas://{{[[:xdigit:]]+}}
// CHECK: [[PREFIX]]/Right.h llvmcas://{{[[:xdigit:]]+}}
// CHECK-NOT: [[PREFIX]]/module.modulemap llvmcas://{{[[:xdigit:]]+}}
// CHECK: [[PREFIX]]/Top.h llvmcas://{{[[:xdigit:]]+}}

// CHECK-LABEL: TRANSLATION UNIT
// CHECK: [[PREFIX]]/tu.m llvmcas://{{[[:xdigit:]]+}}
// CHECK: 1:1 <built-in> llvmcas://{{[[:xdigit:]]+}}
// CHECK: 2:1 (Module) Left
// CHECK: 3:1 (Module) Right

// Note: the modules with explicit imports are imported via parser and are not
// recorded in the include-tree; it's handled entirely by fmodule-map-file,
// fmodule-file, and fmodule-file-cache-key options.

// CHECK-NOT: Module Map
// CHECK: Files:
// CHECK-NOT: [[PREFIX]]/module.modulemap llvmcas://{{[[:xdigit:]]+}}
// CHECK: [[PREFIX]]/Left.h llvmcas://{{[[:xdigit:]]+}}
// CHECK: [[PREFIX]]/Top.h llvmcas://{{[[:xdigit:]]+}}
// CHECK: [[PREFIX]]/Right.h llvmcas://{{[[:xdigit:]]+}}

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
// CHECK-NOT: -fmodule-map-file
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
// CHECK-NOT: -fmodule-map-file
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
// CHECK-NEXT:     {
// CHECK:            "cas-include-tree-id": "[[AT_IMPORT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/ZAtImport-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[AT_IMPORT_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=ZAtImport"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/AtImport.h"
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "ZAtImport"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "cas-include-tree-id": "[[PRAGMA_IMPORT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/ZPragmaImport-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[PRAGMA_IMPORT_TREE]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodules"
// CHECK:              "-fmodule-name=ZPragmaImport"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/PragmaImport.h"
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "ZPragmaImport"
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
// CHECK-NEXT:             {
// CHECK:                    "module-name": "ZAtImport"
// CHECK:                  }
// CHECK-NEXT:             {
// CHECK:                    "module-name": "ZPragmaImport"
// CHECK:                  }
// CHECK-NEXT:           ]
// CHECK:                "command-line": [
// CHECK-NEXT:             "-cc1"
// CHECK:                  "-fcas-path"
// CHECK-NEXT:             "[[PREFIX]]/cas"
// CHECK-NOT: -fmodule-map-file
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
// CHECK:                  "-fmodule-file=Left=[[PREFIX]]/outputs/{{.*}}/Left-{{.*}}.pcm"
// CHECK:                  "-fmodule-file=Right=[[PREFIX]]/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK:                  "-fmodules"
// CHECK:                  "-fno-implicit-modules"
// CHECK:                ]
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.m"
// CHECK-NEXT:           ]
// CHECK:                "input-file": "[[PREFIX]]/tu.m"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

// Build the include-tree commands
// RUN: %clang @%t/Top.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// Ensure the pcm comes from the action cache
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/Left.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/Right.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/ZAtImport.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/ZPragmaImport.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_MISS

// Check cache hits
// RUN: %clang @%t/Top.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/Left.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/Right.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/ZAtImport.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/ZPragmaImport.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s -check-prefix=CACHE_HIT

// CACHE_MISS: compile job cache miss
// CACHE_HIT: compile job cache hit

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.m -I DIR -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache"
}]

//--- module.modulemap
module Top { header "Top.h" export *}
module Left { header "Left.h" export *}
module Right { header "Right.h" export *}
module ZAtImport { header "AtImport.h" }
module ZPragmaImport { header "PragmaImport.h" }

//--- Top.h
#pragma once
struct Top {
  int x;
};
void top(void);

//--- Left.h
#include "Top.h"
void left(void);

//--- Right.h
#include "Top.h"
void right(void);

//--- AtImport.h
void at_import(void);

//--- PragmaImport.h
void pragma_import(void);

//--- tu.m
#import "Left.h"
#import <Right.h>
@import ZAtImport;
#pragma clang module import ZPragmaImport

void tu(void) {
  top();
  left();
  right();
  at_import();
  pragma_import();
}
