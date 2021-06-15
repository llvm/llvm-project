// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/modules-pch/* %t

// Scan dependencies of the PCH:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch/cdb_pch.json > %t/cdb.json
// RUN: echo -%t > %t/result_pch.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build -mode preprocess >> %t/result_pch.json
// RUN: cat %t/result_pch.json | sed 's:\\\\\?:/:g' | FileCheck %s -check-prefix=CHECK-PCH
//
// Check we didn't build the PCH during dependency scanning.
// RUN: not cat %/t/pch.h.gch
//
// CHECK-PCH:      -[[PREFIX:.*]]
// CHECK-PCH-NEXT: {
// CHECK-PCH-NEXT:   "modules": [
// CHECK-PCH-NEXT:     {
// CHECK-PCH-NEXT:       "clang-module-deps": [],
// CHECK-PCH-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH-NEXT:         "-cc1"
// CHECK-PCH:              "-emit-module"
// CHECK-PCH:              "-fmodules"
// CHECK-PCH:              "-fmodule-name=ModCommon1"
// CHECK-PCH:              "-fno-implicit-modules"
// CHECK-PCH:            ],
// CHECK-PCH-NEXT:       "context-hash": "[[HASH_MOD_COMMON_1:.*]]",
// CHECK-PCH-NEXT:       "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/mod_common_1.h",
// CHECK-PCH-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "name": "ModCommon1"
// CHECK-PCH-NEXT:     },
// CHECK-PCH-NEXT:     {
// CHECK-PCH-NEXT:       "clang-module-deps": [],
// CHECK-PCH-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH-NEXT:         "-cc1"
// CHECK-PCH:              "-emit-module"
// CHECK-PCH:              "-fmodules"
// CHECK-PCH:              "-fmodule-name=ModCommon2"
// CHECK-PCH:              "-fno-implicit-modules"
// CHECK-PCH:            ],
// CHECK-PCH-NEXT:       "context-hash": "[[HASH_MOD_COMMON_2:.*]]",
// CHECK-PCH-NEXT:       "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/mod_common_2.h",
// CHECK-PCH-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "name": "ModCommon2"
// CHECK-PCH-NEXT:     },
// CHECK-PCH-NEXT:     {
// CHECK-PCH-NEXT:       "clang-module-deps": [
// CHECK-PCH-NEXT:         {
// CHECK-PCH-NEXT:           "context-hash": "[[HASH_MOD_COMMON_2]]",
// CHECK-PCH-NEXT:           "module-name": "ModCommon2"
// CHECK-PCH-NEXT:         }
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH-NEXT:         "-cc1"
// CHECK-PCH:              "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK-PCH:              "-emit-module"
// CHECK-PCH:              "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_COMMON_2]]/ModCommon2-{{.*}}.pcm"
// CHECK-PCH:              "-fmodules"
// CHECK-PCH:              "-fmodule-name=ModPCH"
// CHECK-PCH:              "-fno-implicit-modules"
// CHECK-PCH:            ],
// CHECK-PCH-NEXT:       "context-hash": "[[HASH_MOD_PCH:.*]]",
// CHECK-PCH-NEXT:       "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/mod_pch.h",
// CHECK-PCH-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "name": "ModPCH"
// CHECK-PCH-NEXT:     }
// CHECK-PCH-NEXT:   ],
// CHECK-PCH-NEXT:   "translation-units": [
// CHECK-PCH-NEXT:     {
// CHECK-PCH-NEXT:       "clang-context-hash": "[[HASH_PCH:.*]]",
// CHECK-PCH-NEXT:       "clang-module-deps": [
// CHECK-PCH-NEXT:         {
// CHECK-PCH-NEXT:           "context-hash": "[[HASH_MOD_COMMON_1]]",
// CHECK-PCH-NEXT:           "module-name": "ModCommon1"
// CHECK-PCH-NEXT:         },
// CHECK-PCH-NEXT:         {
// CHECK-PCH-NEXT:           "context-hash": "[[HASH_MOD_PCH]]",
// CHECK-PCH-NEXT:           "module-name": "ModPCH"
// CHECK-PCH-NEXT:         }
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH-NEXT:         "-fno-implicit-modules",
// CHECK-PCH-NEXT:         "-fno-implicit-module-maps",
// CHECK-PCH-DAG:          "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_COMMON_1]]/ModCommon1-{{.*}}.pcm",
// CHECK-PCH-DAG:          "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_COMMON_2]]/ModCommon2-{{.*}}.pcm",
// CHECK-PCH-DAG:          "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_PCH]]/ModPCH-{{.*}}.pcm",
// CHECK-PCH-NEXT:         "-fmodule-map-file=[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:         "-fmodule-map-file=[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:         "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/pch.h"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "input-file": "[[PREFIX]]/pch.h"
// CHECK-PCH-NEXT:     }
// CHECK-PCH-NEXT:   ]
// CHECK-PCH-NEXT: }

// Explicitly build the PCH:
//
// RUN: tail -n +2 %t/result_pch.json > %t/result_pch_stripped.json
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_pch_stripped.json \
// RUN:   --module-name=ModCommon1 > %t/mod_common_1.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_pch_stripped.json \
// RUN:   --module-name=ModCommon2 > %t/mod_common_2.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_pch_stripped.json \
// RUN:   --module-name=ModPCH > %t/mod_pch.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_pch_stripped.json \
// RUN:   --tu-index=0 > %t/pch.rsp
//
// RUN: %clang @%t/mod_common_1.cc1.rsp
// RUN: %clang @%t/mod_common_2.cc1.rsp
// RUN: %clang @%t/mod_pch.cc1.rsp
// RUN: %clang -x c-header %t/pch.h -fmodules -gmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -o %t/pch.h.gch @%t/pch.rsp

// Scan dependencies of the TU:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch/cdb_tu.json > %t/cdb.json
// RUN: echo -%t > %t/result_tu.json
// FIXME: Make this work with '-mode preprocess-minimized-sources'.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build -mode preprocess >> %t/result_tu.json
// RUN: cat %t/result_tu.json | sed 's:\\\\\?:/:g' | FileCheck %s -check-prefix=CHECK-TU
//
// CHECK-TU:      -[[PREFIX:.*]]
// CHECK-TU-NEXT: {
// CHECK-TU-NEXT:   "modules": [
// CHECK-TU-NEXT:     {
// CHECK-TU-NEXT:       "clang-module-deps": [],
// CHECK-TU-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU-NEXT:         "-cc1",
// CHECK-TU:              "-emit-module",
// CHECK-TU:              "-fmodule-name=ModTU",
// CHECK-TU:              "-fno-implicit-modules",
// CHECK-TU:            ],
// CHECK-TU-NEXT:       "context-hash": "[[HASH_MOD_TU:.*]]",
// CHECK-TU-NEXT:       "file-deps": [
// CHECK-TU-NEXT:         "[[PREFIX]]/mod_tu.h",
// CHECK-TU-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "name": "ModTU"
// CHECK-TU-NEXT:     }
// CHECK-TU-NEXT:   ],
// CHECK-TU-NEXT:   "translation-units": [
// CHECK-TU-NEXT:     {
// CHECK-TU-NEXT:       "clang-context-hash": "[[HASH_TU:.*]]",
// CHECK-TU-NEXT:       "clang-module-deps": [
// CHECK-TU-NEXT:         {
// CHECK-TU-NEXT:           "context-hash": "[[HASH_MOD_TU]]",
// CHECK-TU-NEXT:           "module-name": "ModTU"
// CHECK-TU-NEXT:         }
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU-NEXT:         "-fno-implicit-modules",
// CHECK-TU-NEXT:         "-fno-implicit-module-maps",
// CHECK-TU-NEXT:         "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_TU]]/ModTU-{{.*}}.pcm",
// CHECK-TU-NEXT:         "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "file-deps": [
// CHECK-TU-NEXT:         "[[PREFIX]]/tu.c",
// CHECK-TU-NEXT:         "[[PREFIX]]/pch.h.gch"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-TU-NEXT:     }
// CHECK-TU-NEXT:   ]
// CHECK-TU-NEXT: }

// Explicitly build the TU:
//
// RUN: tail -n +2 %t/result_tu.json > %t/result_tu_stripped.json
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_tu_stripped.json \
// RUN:   --module-name=ModTU > %t/mod_tu.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_tu_stripped.json \
// RUN:   --tu-index=0 > %t/tu.rsp
//
// RUN: %clang @%t/mod_tu.cc1.rsp
// RUN: %clang -fsyntax-only %t/tu.c -fmodules -gmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -include %t/pch.h -o %t/tu.o @%t/tu.rsp

// Scan dependencies of the TU that has common modules with the PCH:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch/cdb_tu_with_common.json > %t/cdb.json
// RUN: echo -%t > %t/result_tu_with_common.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build -mode preprocess >> %t/result_tu_with_common.json
// RUN: cat %t/result_tu_with_common.json | sed 's:\\\\\?:/:g' | FileCheck %s -check-prefix=CHECK-TU-WITH-COMMON
//
// CHECK-TU-WITH-COMMON:      -[[PREFIX:.*]]
// CHECK-TU-WITH-COMMON-NEXT: {
// CHECK-TU-WITH-COMMON-NEXT:   "modules": [
// CHECK-TU-WITH-COMMON-NEXT:     {
// CHECK-TU-WITH-COMMON-NEXT:       "clang-module-deps": [],
// CHECK-TU-WITH-COMMON-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-WITH-COMMON-NEXT:       "command-line": [
// CHECK-TU-WITH-COMMON-NEXT:         "-cc1",
// CHECK-TU-WITH-COMMON:              "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK-TU-WITH-COMMON:              "-emit-module",
// CHECK-TU-WITH-COMMON:              "-fmodule-file=[[PREFIX]]/build/{{.*}}/ModCommon1-{{.*}}.pcm",
// CHECK-TU-WITH-COMMON:              "-fmodule-name=ModTUWithCommon",
// CHECK-TU-WITH-COMMON:              "-fno-implicit-modules",
// CHECK-TU-WITH-COMMON:            ],
// CHECK-TU-WITH-COMMON-NEXT:       "context-hash": "[[HASH_MOD_TU_WITH_COMMON:.*]]",
// CHECK-TU-WITH-COMMON-NEXT:       "file-deps": [
// CHECK-TU-WITH-COMMON-NEXT:         "[[PREFIX]]/mod_tu_with_common.h",
// CHECK-TU-WITH-COMMON-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-TU-WITH-COMMON-NEXT:       ],
// CHECK-TU-WITH-COMMON-NEXT:       "name": "ModTUWithCommon"
// CHECK-TU-WITH-COMMON-NEXT:     }
// CHECK-TU-WITH-COMMON-NEXT:   ],
// CHECK-TU-WITH-COMMON-NEXT:   "translation-units": [
// CHECK-TU-WITH-COMMON-NEXT:     {
// CHECK-TU-WITH-COMMON-NEXT:       "clang-context-hash": "[[HASH_TU_WITH_COMMON:.*]]",
// CHECK-TU-WITH-COMMON-NEXT:       "clang-module-deps": [
// CHECK-TU-WITH-COMMON-NEXT:         {
// CHECK-TU-WITH-COMMON-NEXT:           "context-hash": "[[HASH_MOD_TU_WITH_COMMON]]",
// CHECK-TU-WITH-COMMON-NEXT:           "module-name": "ModTUWithCommon"
// CHECK-TU-WITH-COMMON-NEXT:         }
// CHECK-TU-WITH-COMMON-NEXT:       ],
// CHECK-TU-WITH-COMMON-NEXT:       "command-line": [
// CHECK-TU-WITH-COMMON-NEXT:         "-fno-implicit-modules",
// CHECK-TU-WITH-COMMON-NEXT:         "-fno-implicit-module-maps",
// FIXME: Figure out why we need `=ModCommon2` here for Clang to pick up the PCM.
// CHECK-TU-WITH-COMMON-NEXT:         "-fmodule-file=ModCommon2=[[PREFIX]]/build/{{.*}}/ModCommon2-{{.*}}.pcm",
// CHECK-TU-WITH-COMMON-NEXT:         "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK-TU-WITH-COMMON-NEXT:         "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_TU_WITH_COMMON]]/ModTUWithCommon-{{.*}}.pcm",
// CHECK-TU-WITH-COMMON-NEXT:         "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK-TU-WITH-COMMON-NEXT:       ],
// CHECK-TU-WITH-COMMON-NEXT:       "file-deps": [
// CHECK-TU-WITH-COMMON-NEXT:         "[[PREFIX]]/tu_with_common.c",
// CHECK-TU-WITH-COMMON-NEXT:         "[[PREFIX]]/pch.h.gch"
// CHECK-TU-WITH-COMMON-NEXT:       ],
// CHECK-TU-WITH-COMMON-NEXT:       "input-file": "[[PREFIX]]/tu_with_common.c"
// CHECK-TU-WITH-COMMON-NEXT:     }
// CHECK-TU-WITH-COMMON-NEXT:   ]
// CHECK-TU-WITH-COMMON-NEXT: }

// Explicitly build the TU that has common modules with the PCH:
//
// RUN: tail -n +2 %t/result_tu_with_common.json > %t/result_tu_with_common_stripped.json
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_tu_with_common_stripped.json \
// RUN:   --module-name=ModTUWithCommon > %t/mod_tu_with_common.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_tu_with_common_stripped.json \
// RUN:   --tu-index=0 > %t/tu_with_common.rsp
//
// RUN: %clang @%t/mod_tu_with_common.cc1.rsp
// RUN: %clang -fsyntax-only %t/tu_with_common.c -fmodules -gmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -include %t/pch.h -o %t/tu_with_common.o @%t/tu_with_common.rsp
