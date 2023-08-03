// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json

// Scan PCH
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch.json

// Build PCH
// RUN: %deps-to-rsp %t/deps_pch.json --module-name Top > %t/Top.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --module-name Left > %t/Left.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/Top.rsp
// RUN: %clang @%t/Left.rsp
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/pch.rsp
// RUN: rm -rf %t/outputs

// Scan TU with PCH
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-include-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Build TU
// RUN: %deps-to-rsp %t/deps.json --module-name Right > %t/Right.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/Right.rsp
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/tu.rsp

// RUN: FileCheck %s -input-file %t/deps.json -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:  "modules": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": []
// CHECK:            "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]/cas"
// CHECK:              "-o"
// CHECK-NEXT:         "[[PREFIX]]/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK:              "-disable-free"
// CHECK:              "-fno-pch-timestamp"
// CHECK:              "-fcas-include-tree"
// CHECK-NEXT:         "[[RIGHT_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file=[[PREFIX]]/outputs/{{.*}}/Top-{{.*}}.pcm"
// CHECK:              "-fmodule-file-cache-key"
// CHECK-NEXT:         "[[PREFIX]]/{{.*}}/Top-{{.*}}.pcm"
// CHECK-NEXT:         "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:              "-x"
// CHECK-NEXT:         "c"
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
// CHECK-NOT: "clang-modulemap-file"
// CHECK:        ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK:                "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK:                    "module-name": "Right"
// CHECK:                  }
// CHECK-NEXT:           ]
// CHECK:                "command-line": [
// CHECK-NEXT:             "-cc1"
// CHECK:                  "-fcas-path"
// CHECK-NEXT:             "[[PREFIX]]/cas"
// CHECK-NOT: -fmodule-map-file=
// CHECK:                  "-disable-free"
// CHECK:                  "-fcas-include-tree"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-fcache-compile-job"
// CHECK:                  "-fsyntax-only"
// CHECK:                  "-fmodule-file-cache-key"
// CHECK-NEXT:             "[[PREFIX]]/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK-NEXT:             "llvmcas://{{[[:xdigit:]]+}}"
// CHECK:                  "-x"
// CHECK-NEXT:             "c"
// CHECK:                  "-fmodule-file=Right=[[PREFIX]]/outputs/{{.*}}/Right-{{.*}}.pcm"
// CHECK:                  "-fmodules"
// CHECK:                  "-fno-implicit-modules"
// CHECK:                ]
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:             "[[PREFIX]]/prefix.h.pch"
// CHECK-NEXT:           ]
// CHECK:                "input-file": "[[PREFIX]]/tu.c"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

//--- cdb_pch.json.template
[{
  "file": "DIR/prefix.h",
  "directory": "DIR",
  "command": "clang -x c-header DIR/prefix.h -o DIR/prefix.h.pch -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -include DIR/prefix.h -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
module Top { header "Top.h" export *}
module Left { header "Left.h" export *}
module Right { header "Right.h" export *}

//--- Top.h
#pragma once
struct Top { int x; };

//--- Left.h
#pragma once
#include "Top.h"
struct Left { struct Top top; };

//--- Right.h
#pragma once
#include "Top.h"
struct Right { struct Top top; };

//--- prefix.h
#include "Left.h"

//--- tu.c
#include "Right.h"

void tu(void) {
  struct Left _left;
  struct Right _right;
  struct Top _top;
}
