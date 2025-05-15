// Test that modifications to a common header (imported from both a PCH and a TU)
// cause rebuilds of dependent modules imported from the TU on incremental build.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module mod_common { header "mod_common.h" }
module mod_tu { header "mod_tu.h" }
module mod_tu_extra { header "mod_tu_extra.h" }

//--- mod_common.h
#define MOD_COMMON_MACRO 0
//--- mod_tu.h
#include "mod_common.h"
#if MOD_COMMON_MACRO
#include "mod_tu_extra.h"
#endif

//--- mod_tu_extra.h

//--- prefix.h
#include "mod_common.h"

//--- tu.c
#include "mod_tu.h"

// Clean: scan the PCH.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_pch_clean.json -- \
// RUN:     %clang -x c-header %t/prefix.h -o %t/prefix.h.pch -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache

// Clean: build the PCH.
// RUN: %deps-to-rsp %t/deps_pch_clean.json --module-name mod_common > %t/mod_common.rsp
// RUN: %deps-to-rsp %t/deps_pch_clean.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/mod_common.rsp
// RUN: %clang @%t/pch.rsp

// Clean: scan the TU.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_tu_clean.json -- \
// RUN:     %clang -c %t/tu.c -o %t/tu.o -include %t/prefix.h -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache
// RUN: cat %t/deps_tu_clean.json | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-TU-CLEAN -DPREFIX=%/t
// CHECK-TU-CLEAN:      {
// CHECK-TU-CLEAN-NEXT:   "modules": [
// CHECK-TU-CLEAN-NEXT:     {
// CHECK-TU-CLEAN-NEXT:       "clang-module-deps": [],
// CHECK-TU-CLEAN-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-CLEAN-NEXT:       "command-line": [
// CHECK-TU-CLEAN:            ],
// CHECK-TU-CLEAN-NEXT:       "context-hash": "{{.*}}",
// CHECK-TU-CLEAN-NEXT:       "file-deps": [
// CHECK-TU-CLEAN-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-TU-CLEAN-NEXT:         "[[PREFIX]]/mod_tu.h"
// CHECK-TU-CLEAN-NEXT:       ],
// CHECK-TU-CLEAN-NEXT:       "link-libraries": [],
// CHECK-TU-CLEAN-NEXT:       "name": "mod_tu"
// CHECK-TU-CLEAN-NEXT:     }
// CHECK-TU-CLEAN-NEXT:   ],
// CHECK-TU-CLEAN-NEXT:   "translation-units": [
// CHECK-TU-CLEAN-NEXT:     {
// CHECK-TU-CLEAN-NEXT:       "commands": [
// CHECK-TU-CLEAN-NEXT:         {
// CHECK-TU-CLEAN-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-TU-CLEAN-NEXT:           "clang-module-deps": [
// CHECK-TU-CLEAN-NEXT:             {
// CHECK-TU-CLEAN-NEXT:               "context-hash": "{{.*}}",
// CHECK-TU-CLEAN-NEXT:               "module-name": "mod_tu"
// CHECK-TU-CLEAN-NEXT:             }
// CHECK-TU-CLEAN-NEXT:           ],
// CHECK-TU-CLEAN-NEXT:           "command-line": [
// CHECK-TU-CLEAN:                ],
// CHECK-TU-CLEAN-NEXT:           "executable": "{{.*}}",
// CHECK-TU-CLEAN-NEXT:           "file-deps": [
// CHECK-TU-CLEAN-NEXT:             "[[PREFIX]]/tu.c",
// CHECK-TU-CLEAN-NEXT:             "[[PREFIX]]/prefix.h.pch"
// CHECK-TU-CLEAN-NEXT:           ],
// CHECK-TU-CLEAN-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-TU-CLEAN-NEXT:         }
// CHECK-TU-CLEAN-NEXT:       ]
// CHECK-TU-CLEAN-NEXT:     }
// CHECK-TU-CLEAN:        ]
// CHECK-TU-CLEAN:      }

// Clean: build the TU.
// RUN: %deps-to-rsp %t/deps_tu_clean.json --module-name mod_tu > %t/mod_tu.rsp
// RUN: %deps-to-rsp %t/deps_tu_clean.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/mod_tu.rsp
// RUN: %clang @%t/tu.rsp

// Incremental: modify the common module.
// RUN: sleep 1
// RUN: echo "#define MOD_COMMON_MACRO 1" > %t/mod_common.h

// Incremental: scan the PCH.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_pch_incremental.json -- \
// RUN:     %clang -x c-header %t/prefix.h -o %t/prefix.h.pch -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache

// Incremental: build the PCH.
// RUN: %deps-to-rsp %t/deps_pch_incremental.json --module-name mod_common > %t/mod_common.rsp
// RUN: %deps-to-rsp %t/deps_pch_incremental.json --tu-index 0 > %t/pch.rsp
// RUN: %clang @%t/mod_common.rsp
// RUN: %clang @%t/pch.rsp

// Incremental: scan the TU. This needs to invalidate modules imported from the
//              TU that depend on modules imported from the PCH and discover the
//              new dependency on 'mod_tu_extra'.
// RUN: clang-scan-deps -format experimental-full -o %t/deps_tu_incremental.json -- \
// RUN:     %clang -c %t/tu.c -o %t/tu.o -include %t/prefix.h -F %t \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache
// RUN: cat %t/deps_tu_incremental.json | sed 's:\\\\\?:/:g' | FileCheck %s --check-prefix=CHECK-TU-INCREMENTAL -DPREFIX=%/t
// CHECK-TU-INCREMENTAL:      {
// CHECK-TU-INCREMENTAL-NEXT:   "modules": [
// CHECK-TU-INCREMENTAL-NEXT:     {
// CHECK-TU-INCREMENTAL-NEXT:       "clang-module-deps": [
// CHECK-TU-INCREMENTAL-NEXT:         {
// CHECK-TU-INCREMENTAL-NEXT:           "context-hash": "{{.*}}",
// CHECK-TU-INCREMENTAL-NEXT:           "module-name": "mod_tu_extra"
// CHECK-TU-INCREMENTAL-NEXT:         }
// CHECK-TU-INCREMENTAL-NEXT:       ],
// CHECK-TU-INCREMENTAL-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-INCREMENTAL-NEXT:       "command-line": [
// CHECK-TU-INCREMENTAL:            ],
// CHECK-TU-INCREMENTAL-NEXT:       "context-hash": "{{.*}}",
// CHECK-TU-INCREMENTAL-NEXT:       "file-deps": [
// CHECK-TU-INCREMENTAL-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-TU-INCREMENTAL-NEXT:         "[[PREFIX]]/mod_tu.h"
// CHECK-TU-INCREMENTAL-NEXT:       ],
// CHECK-TU-INCREMENTAL-NEXT:       "link-libraries": [],
// CHECK-TU-INCREMENTAL-NEXT:       "name": "mod_tu"
// CHECK-TU-INCREMENTAL-NEXT:     },
// CHECK-TU-INCREMENTAL-NEXT:     {
// CHECK-TU-INCREMENTAL-NEXT:       "clang-module-deps": [],
// CHECK-TU-INCREMENTAL-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-INCREMENTAL-NEXT:       "command-line": [
// CHECK-TU-INCREMENTAL:            ],
// CHECK-TU-INCREMENTAL-NEXT:       "context-hash": "{{.*}}",
// CHECK-TU-INCREMENTAL-NEXT:       "file-deps": [
// CHECK-TU-INCREMENTAL-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-TU-INCREMENTAL-NEXT:         "[[PREFIX]]/mod_tu_extra.h"
// CHECK-TU-INCREMENTAL-NEXT:       ],
// CHECK-TU-INCREMENTAL-NEXT:       "link-libraries": [],
// CHECK-TU-INCREMENTAL-NEXT:       "name": "mod_tu_extra"
// CHECK-TU-INCREMENTAL-NEXT:     }
// CHECK-TU-INCREMENTAL-NEXT:   ],
// CHECK-TU-INCREMENTAL-NEXT:   "translation-units": [
// CHECK-TU-INCREMENTAL-NEXT:     {
// CHECK-TU-INCREMENTAL-NEXT:       "commands": [
// CHECK-TU-INCREMENTAL-NEXT:         {
// CHECK-TU-INCREMENTAL-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-TU-INCREMENTAL-NEXT:           "clang-module-deps": [
// CHECK-TU-INCREMENTAL-NEXT:             {
// CHECK-TU-INCREMENTAL-NEXT:               "context-hash": "{{.*}}",
// CHECK-TU-INCREMENTAL-NEXT:               "module-name": "mod_tu"
// CHECK-TU-INCREMENTAL-NEXT:             }
// CHECK-TU-INCREMENTAL-NEXT:           ],
// CHECK-TU-INCREMENTAL-NEXT:           "command-line": [
// CHECK-TU-INCREMENTAL:                ],
// CHECK-TU-INCREMENTAL-NEXT:           "executable": "{{.*}}",
// CHECK-TU-INCREMENTAL-NEXT:           "file-deps": [
// CHECK-TU-INCREMENTAL-NEXT:             "[[PREFIX]]/tu.c",
// CHECK-TU-INCREMENTAL-NEXT:             "[[PREFIX]]/prefix.h.pch"
// CHECK-TU-INCREMENTAL-NEXT:           ],
// CHECK-TU-INCREMENTAL-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-TU-INCREMENTAL-NEXT:         }
// CHECK-TU-INCREMENTAL-NEXT:       ]
// CHECK-TU-INCREMENTAL-NEXT:     }
// CHECK-TU-INCREMENTAL:        ]
// CHECK-TU-INCREMENTAL:      }

// Incremental: build the TU.
// RUN: %deps-to-rsp %t/deps_tu_incremental.json --module-name mod_tu_extra > %t/mod_tu_extra.rsp
// RUN: %deps-to-rsp %t/deps_tu_incremental.json --module-name mod_tu > %t/mod_tu.rsp
// RUN: %deps-to-rsp %t/deps_tu_incremental.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/mod_tu_extra.rsp
// RUN: %clang @%t/mod_tu.rsp
// RUN: %clang @%t/tu.rsp
