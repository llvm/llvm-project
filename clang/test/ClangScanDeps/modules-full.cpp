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
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/modules_cdb_clangcl.json > %t_clangcl.cdb
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 4 -format experimental-full \
// RUN:   -mode preprocess-dependency-directives > %t.result
// RUN: cat %t.result | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t.dir --check-prefixes=CHECK,CHECK-NO-ABS %s
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 4 -format experimental-full \
// RUN:   -generate-modules-path-args -mode preprocess-dependency-directives > %t.result
// RUN: cat %t.result | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t.dir --check-prefixes=CHECK,CHECK-ABS %s
//
// RUN: clang-scan-deps -compilation-database %t.cdb -j 4 -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t.dir/custom \
// RUN:   -mode preprocess-dependency-directives > %t.result
// RUN: cat %t.result | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t.dir --check-prefixes=CHECK,CHECK-CUSTOM %s
//
// RUN: clang-scan-deps -compilation-database %t_clangcl.cdb -j 4 -format experimental-full \
// RUN:   -mode preprocess-dependency-directives > %t_clangcl.result
// RUN: cat %t_clangcl.result | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t.dir --check-prefixes=CHECK,CHECK-NO-ABS %s

#include "header.h"

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
// CHECK-NO-ABS-NOT:   "-fmodule-file={{.*}}"
// CHECK-ABS:          "-fmodule-file=[[PREFIX]]/module-cache{{(_clangcl)?}}/[[HASH_H2_DINCLUDE]]/header2-{{[A-Z0-9]+}}.pcm"
// CHECK-CUSTOM:       "-fmodule-file=[[PREFIX]]/custom/[[HASH_H2_DINCLUDE]]/header2-{{[A-Z0-9]+}}.pcm"
// CHECK-NOT:          "-fimplicit-module-maps"
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
// CHECK-NOT:          "-fimplicit-module-maps",
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
// CHECK-NOT:          "-fimplicit-module-maps",
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
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU:[A-Z0-9]+]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_H1]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fno-implicit-modules"
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NO-ABS-NOT:   "-fmodule-file={{.*}}"
// CHECK-ABS-NEXT:     "-fmodule-file=[[PREFIX]]/module-cache{{(_clangcl)?}}/[[HASH_H1]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-CUSTOM-NEXT:  "-fmodule-file=[[PREFIX]]/custom/[[HASH_H1]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU:[A-Z0-9]+]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_H1]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fno-implicit-modules"
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NO-ABS-NOT:   "-fmodule-file={{.*}},
// CHECK-ABS-NEXT:     "-fmodule-file=[[PREFIX]]/module-cache{{(_clangcl)?}}/[[HASH_H1]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-CUSTOM-NEXT:  "-fmodule-file=[[PREFIX]]/custom/[[HASH_H1]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU:[A-Z0-9]+]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_H1]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fno-implicit-modules"
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NO-ABS-NOT:   "-fmodule-file={{.*}}"
// CHECK-ABS-NEXT:     "-fmodule-file=[[PREFIX]]/module-cache{{(_clangcl)?}}/[[HASH_H1]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-CUSTOM-NEXT:  "-fmodule-file=[[PREFIX]]/custom/[[HASH_H1]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input.cpp"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU_DINCLUDE:[A-Z0-9]+]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_H1_DINCLUDE]]",
// CHECK-NEXT:           "module-name": "header1"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fno-implicit-modules"
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NO-ABS-NOT:   "-fmodule-file={{.*}}"
// CHECK-ABS-NEXT:     "-fmodule-file=[[PREFIX]]/module-cache{{(_clangcl)?}}/[[HASH_H1_DINCLUDE]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-CUSTOM-NEXT:  "-fmodule-file=[[PREFIX]]/custom/[[HASH_H1_DINCLUDE]]/header1-{{[A-Z0-9]+}}.pcm"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules_cdb_input2.cpp"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/modules_cdb_input2.cpp"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
