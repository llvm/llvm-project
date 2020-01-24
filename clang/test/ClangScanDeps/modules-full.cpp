// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: cp %s %t.dir/modules_cdb_input2.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header2.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/module.modulemap %t.dir/Inputs/module.modulemap
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/modules_cdb.json > %t.cdb
//
// RUN: echo %t.dir > %t.result
// RUN: clang-scan-deps -compilation-database %t.cdb -j 4 -full-command-line \
// RUN:   -mode preprocess-minimized-sources -format experimental-full >> %t.result
// RUN: cat %t.result | sed 's/\\/\//g' | FileCheck --check-prefixes=CHECK %s

// FIXME: Backslash issues.
// XFAIL: system-windows

#include "header.h"

// CHECK: [[PREFIX:.*]]
// CHECK-NEXT: {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[CONTEXT_HASH_H1:[A-Z0-9]+]]",
// CHECK-NEXT:           "module-name": "header2"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/Inputs/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps",
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache/[[CONTEXT_HASH_H1]]/header2-{{[A-Z0-9]+}}.pcm",
// CHECK-NEXT:         "-fmodule-map-file=[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "context-hash": "[[CONTEXT_HASH_H1]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Inputs/header.h",
// CHECK-NEXT:         "[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "header1"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/Inputs/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "context-hash": "[[CONTEXT_HASH_H2:[A-Z0-9]+]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Inputs/header.h",
// CHECK-NEXT:         "[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "header1"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/Inputs/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "context-hash": "[[CONTEXT_HASH_H1]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Inputs/header2.h",
// CHECK-NEXT:         "[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "header2"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[CONTEXT_HASH_H2]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[CONTEXT_HASH_H2]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps",
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache/[[CONTEXT_HASH_H2]]/header1-{{[A-Z0-9]+}}.pcm",
// CHECK-NEXT:         "-fmodule-map-file=[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[CONTEXT_HASH_H2]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[CONTEXT_HASH_H2]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps",
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache/[[CONTEXT_HASH_H2]]/header1-{{[A-Z0-9]+}}.pcm",
// CHECK-NEXT:         "-fmodule-map-file=[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[CONTEXT_HASH_H2]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[CONTEXT_HASH_H2]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps",
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache/[[CONTEXT_HASH_H2]]/header1-{{[A-Z0-9]+}}.pcm",
// CHECK-NEXT:         "-fmodule-map-file=[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[CONTEXT_HASH_H1]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[CONTEXT_HASH_H1]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps",
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache/[[CONTEXT_HASH_H1]]/header2-{{[A-Z0-9]+}}.pcm",
// CHECK-NEXT:         "-fmodule-map-file=[[PREFIX]]/Inputs/module.modulemap",
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache/[[CONTEXT_HASH_H1]]/header1-{{[A-Z0-9]+}}.pcm",
// CHECK-NEXT:         "-fmodule-map-file=[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input2.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input2.cpp"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
