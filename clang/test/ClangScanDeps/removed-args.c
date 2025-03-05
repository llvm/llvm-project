// Some command-line arguments used for compiling translation units are not
// compatible with the semantics of modules or are likely to differ between
// identical modules discovered from different translation units. This test
// checks such arguments are removed from the command-lines: '-include',
// '-dwarf-debug-flag' and '-main-file-name'. Similarly, several arguments
// such as '-fmodules-cache-path=' are only relevant for implicit modules, and
// are removed to better-canonicalize the compilation.

// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/removed-args/* %t
// RUN: touch %t/build-session
// RUN: touch %t/tu.proftext
// RUN: llvm-profdata merge %t/tu.proftext -o %t/tu.profdata

// RUN: sed "s|DIR|%/t|g" %S/Inputs/removed-args/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
//
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK-NOT:          "-fdebug-compilation-dir="
// CHECK-NOT:          "-fcoverage-compilation-dir="
// CHECK-NOT:          "-coverage-notes-file
// CHECK-NOT:          "-coverage-data-file
// CHECK-NOT:          "-fprofile-instrument-use-path
// CHECK-NOT:          "-dwarf-debug-flags"
// CHECK-NOT:          "-main-file-name"
// CHECK-NOT:          "-include"
// CHECK-NOT:          "-fmodules-cache-path=
// CHECK-NOT:          "-fmodules-validate-once-per-build-session"
// CHECK-NOT:          "-fbuild-session-timestamp=
// CHECK-NOT:          "-fmodules-prune-interval=
// CHECK-NOT:          "-fmodules-prune-after=
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_MOD_HEADER:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/mod_header.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "ModHeader"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK-NOT:          "-fdebug-compilation-dir=
// CHECK-NOT:          "-fcoverage-compilation-dir=
// CHECK-NOT:          "-coverage-notes-file
// CHECK-NOT:          "-coverage-data-file
// CHECK-NOT:          "-fprofile-instrument-use-path
// CHECK-NOT:          "-dwarf-debug-flags"
// CHECK-NOT:          "-main-file-name"
// CHECK-NOT:          "-include"
// CHECK-NOT:          "-fmodules-cache-path=
// CHECK-NOT:          "-fmodules-validate-once-per-build-session"
// CHECK-NOT:          "-fbuild-session-timestamp=
// CHECK-NOT:          "-fmodules-prune-interval=
// CHECK-NOT:          "-fmodules-prune-after=
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_MOD_TU:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/mod_tu.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "ModTU"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-context-hash": "[[HASH_TU:.*]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD_HEADER]]",
// CHECK-NEXT:           "module-name": "ModHeader"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD_TU]]",
// CHECK-NEXT:           "module-name": "ModTU"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1",
// CHECK-NOT:          "-fmodules-cache-path=
// CHECK-NOT:          "-fmodules-validate-once-per-build-session"
// CHECK-NOT:          "-fbuild-session-timestamp=
// CHECK-NOT:          "-fbuild-session-file=
// CHECK-NOT:          "-fmodules-prune-interval=
// CHECK-NOT:          "-fmodules-prune-after=
// CHECK:            ],

// Check for removed args for PCH invocations.

// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb-pch.json.template > %t/cdb-pch.json
// RUN: clang-scan-deps -compilation-database %t/cdb-pch.json -format experimental-full > %t/result-pch.json
// RUN: cat %t/result-pch.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -check-prefix=PCH
//
// PCH-NOT:          "-fdebug-compilation-dir="
// PCH-NOT:          "-fcoverage-compilation-dir="
// PCH-NOT:          "-coverage-notes-file
// PCH-NOT:          "-coverage-data-file
// PCH-NOT:          "-fprofile-instrument-use-path
// PCH-NOT:          "-include"
// PCH-NOT:          "-fmodules-cache-path=
// PCH-NOT:          "-fmodules-validate-once-per-build-session"
// PCH-NOT:          "-fbuild-session-timestamp=
// PCH-NOT:          "-fmodules-prune-interval=
// PCH-NOT:          "-fmodules-prune-after=

//--- cdb-pch.json.template
[
  {
    "directory": "DIR",
    "command": "clang -x c-header DIR/header.h -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -fdebug-compilation-dir=DIR/debug -fcoverage-compilation-dir=DIR/coverage -ftest-coverage -fprofile-instr-use=DIR/tu.profdata -o DIR/header.h.pch -serialize-diagnostics DIR/header.h.pch.diag ",
    "file": "DIR/header.h.pch"
  }
]
