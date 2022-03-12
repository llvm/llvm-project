// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: sed -e "s|DIR|%/t.dir|g" -e "s|FRAMEWORKS|%/S/Inputs/frameworks|g" \
// RUN:   %/S/Inputs/modules_inferred_cdb.json > %t.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -format experimental-full \
// RUN:   -generate-modules-path-args -mode preprocess-minimized-sources > %t.result
// RUN: cat %t.result | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t.dir -DSOURCEDIR=%/S --check-prefixes=CHECK

#include <Inferred/Inferred.h>

inferred a = 0;

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[SOURCEDIR]]/Inputs/frameworks/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1",
// CHECK:              "-emit-module",
// CHECK-NOT:          "-fimplicit-module-maps",
// CHECK:              "-fmodule-name=Inferred",
// CHECK:              "-fno-implicit-modules",
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_INFERRED:[A-Z0-9]+]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[SOURCEDIR]]/Inputs/frameworks/Inferred.framework/Frameworks/Sub.framework/Headers/Sub.h",
// CHECK-NEXT:         "[[SOURCEDIR]]/Inputs/frameworks/Inferred.framework/Headers/Inferred.h",
// CHECK-NEXT:         "[[SOURCEDIR]]/Inputs/frameworks/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "Inferred"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU:[A-Z0-9]+]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_INFERRED]]",
// CHECK-NEXT:           "module-name": "Inferred"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps",
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache/[[HASH_INFERRED]]/Inferred-{{[A-Z0-9]+}}.pcm"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
