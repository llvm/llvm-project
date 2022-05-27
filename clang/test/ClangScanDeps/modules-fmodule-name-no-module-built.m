// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules-fmodule-name-no-module-built.m
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header2.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/header3.h %t.dir/Inputs/header3.h
// RUN: cp %S/Inputs/module.modulemap %t.dir/Inputs/module.modulemap
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/module_fmodule_name_cdb.json > %t.cdb

// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -format experimental-full \
// RUN:   -generate-modules-path-args -mode preprocess-dependency-directives > %t.result
// RUN: cat %t.result | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t.dir --check-prefixes=CHECK %s

#import "header3.h"
#import "header.h"

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": []
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/Inputs/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_H2:[A-Z0-9]+]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Inputs/header2.h",
// CHECK-NEXT:         "[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "header2"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU:[A-Z0-9]+]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_H2]]",
// CHECK-NEXT:           "module-name": "header2"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fno-implicit-modules"
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NEXT:         "-fmodule-file=[[PREFIX]]/module-cache{{(_clangcl)?}}/[[HASH_H2]]/header2-{{[A-Z0-9]+}}.pcm"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules-fmodule-name-no-module-built.m"
// CHECK-NEXT:         "[[PREFIX]]/Inputs/header3.h"
// CHECK-NEXT:         "[[PREFIX]]/Inputs/header.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules-fmodule-name-no-module-built.m"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
