// UNSUPPORTED: powerpc64-ibm-aix
// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: cp %s %t.dir/modules_cdb_input2.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header2.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/module.modulemap %t.dir/Inputs/module.modulemap
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/modules_cdb_by_mod_name.json > %t.cdb
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/modules_cdb_clangcl_by_mod_name.json > %t_clangcl.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 -format experimental-full \
// RUN:   -mode preprocess-dependency-directives -module-name=header1 > %t.result
// RUN: cat %t.result | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t.dir --check-prefixes=CHECK %s
//
// RUN: clang-scan-deps -compilation-database %t_clangcl.cdb -j 4 -format experimental-full \
// RUN:   -mode preprocess-dependency-directives -module-name=header1 > %t_clangcl.result
// RUN: cat %t_clangcl.result | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t.dir --check-prefixes=CHECK %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_H2_DINCLUDE:[A-Z0-9]+]]",
// CHECK-NEXT:           "module-name": "header2"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/Inputs/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-name=header1"
// CHECK:              "-fno-implicit-modules"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_H1_DINCLUDE:[A-Z0-9]+]]",
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
// CHECK-NEXT:         "-cc1",
// CHECK:              "-emit-module",
// CHECK:              "-fmodule-name=header1",
// CHECK:              "-fno-implicit-modules",
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_H1:[A-Z0-9]+]]",
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
// CHECK-NEXT:         "-cc1",
// CHECK:              "-emit-module",
// CHECK:              "-fmodule-name=header2",
// CHECK:              "-fno-implicit-modules",
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_H2_DINCLUDE]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/Inputs/header2.h",
// CHECK-NEXT:         "[[PREFIX]]/Inputs/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "header2"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
