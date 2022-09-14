// Check that we can scan pch + modules with caching enabled and build the
// resulting commands.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb_pch.json.template > %t/cdb_pch.json
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// == Scan PCH
// RUN: clang-scan-deps -compilation-database %t/cdb_pch.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_pch.json

// == Check specifics of the command-line
// RUN: cat %t/deps_pch.json | FileCheck %s -DPREFIX=%/t -check-prefix=PCH

// == Build PCH
// RUN: %deps-to-rsp %t/deps_pch.json --module-name A > %t/A.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --module-name B > %t/B.rsp
// RUN: %deps-to-rsp %t/deps_pch.json --tu-index 0 > %t/pch.rsp

// RUN: %clang @%t/B.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/A.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// Ensure we load pcms from action cache
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/pch.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/pch.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-HIT

// == Scan TU, including PCH
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// == Check specifics of the command-line
// RUN: cat %t/deps.json | FileCheck %s -DPREFIX=%/t

// == Build TU, including PCH
// RUN: %deps-to-rsp %t/deps.json --module-name C > %t/C.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// RUN: %clang @%t/C.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// Ensure we load pcms from action cache
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-HIT

// PCH:      {
// PCH-NEXT:   "modules": [
// PCH-NEXT:     {
// PCH:            "casfs-root-id": "[[A_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// PCH:            "clang-module-deps": [
// PCH:              {
// PCH:                "module-name": "B"
// PCH:              }
// PCH-NEXT:       ]
// PCH:            "command-line": [
// PCH-NEXT:         "-cc1"
// PCH:              "-fcas-path"
// PCH-NEXT:         "[[PREFIX]]{{.}}cas"
// PCH:              "-faction-cache-path"
// PCH-NEXT:         "[[PREFIX]]{{.}}cache"
// PCH:              "-fcas-fs"
// PCH-NEXT:         "[[A_ROOT_ID]]"
// PCH:              "-o"
// PCH-NEXT:         "[[A_PCM:.*outputs.*A-.*\.pcm]]"
// PCH:              "-fcache-compile-job"
// PCH:              "-emit-module"
// PCH:              "-fmodule-file-cache-key=[[B_PCM:.*outputs.*B-.*\.pcm]]=[[B_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// PCH:              "-fmodule-file={{(B=)?}}[[B_PCM]]"
// PCH:            ]
// PCH:            "file-deps": [
// PCH-NEXT:         "[[PREFIX]]{{.}}A.h"
// PCH-NEXT:         "[[PREFIX]]{{.}}module.modulemap"
// PCH-NEXT:       ]
// PCH:            "name": "A"
// PCH:          }
// PCH-NEXT:     {
// PCH:            "casfs-root-id": "[[B_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// PCH:            "clang-module-deps": []
// PCH:            "command-line": [
// PCH-NEXT:         "-cc1"
// PCH:              "-fcas-path"
// PCH-NEXT:         "[[PREFIX]]{{.}}cas"
// PCH:              "-faction-cache-path"
// PCH-NEXT:         "[[PREFIX]]{{.}}cache"
// PCH:              "-fcas-fs"
// PCH-NEXT:         "[[B_ROOT_ID]]"
// PCH:              "-o"
// PCH-NEXT:         "[[B_PCM]]"
// PCH:              "-fcache-compile-job"
// PCH:              "-emit-module"
// PCH:            ]
// PCH:            "file-deps": [
// PCH-NEXT:         "[[PREFIX]]{{.}}B.h"
// PCH-NEXT:         "[[PREFIX]]{{.}}module.modulemap"
// PCH-NEXT:       ]
// PCH:            "name": "B"
// PCH:          }
// PCH-NEXT:   ]
// PCH:        "translation-units": [
// PCH:          {
// PCH:            "commands": [
// PCH:              {
// PCH:                "casfs-root-id": "[[PCH_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// PCH:                "clang-module-deps": [
// PCH:                  {
// PCH:                    "module-name": "A"
// PCH:                  }
// PCH-NEXT:           ]
// PCH:                "command-line": [
// PCH-NEXT:             "-cc1"
// PCH:                  "-fcas-path"
// PCH-NEXT:             "[[PREFIX]]{{.}}cas"
// PCH:                  "-faction-cache-path"
// PCH-NEXT:             "[[PREFIX]]{{.}}cache"
// PCH:                  "-fcas-fs"
// PCH-NEXT:             "[[PCH_ROOT_ID]]"
// PCH:                  "-fno-pch-timestamp"
// PCH:                  "-fcache-compile-job"
// PCH:                  "-emit-pch"
// PCH:                  "-fmodule-file-cache-key=[[A_PCM]]={{llvmcas://[[:xdigit:]]+}}"
// PCH:                  "-fmodule-file={{(A=)?}}[[A_PCM]]"
// PCH:                ]
// PCH:                "file-deps": [
// PCH-NEXT:             "[[PREFIX]]{{.}}prefix.h"
// PCH-NEXT:           ]
// PCH:              }

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "casfs-root-id": "[[C_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cas"
// CHECK:              "-faction-cache-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cache"
// CHECK:              "-fcas-fs"
// CHECK-NEXT:         "[[C_ROOT_ID]]"
// CHECK:              "-o"
// CHECK-NEXT:         "[[C_PCM:.*outputs.*C-.*\.pcm]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file={{(B=)?}}[[B_PCM:.*outputs.*B-.*\.pcm]]"
// CHECK:              "-fmodule-file-cache-key=[[B_PCM]]=[[B_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]{{.}}C.h"
// CHECK-NEXT:         "[[PREFIX]]{{.}}module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "C"
// CHECK:          }
// CHECK-NEXT:   ]
// CHECK:        "translation-units": [
// CHECK:          {
// CHECK:            "commands": [
// CHECK:              {
// CHECK:                "casfs-root-id": "[[TU_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:                "clang-module-deps": [
// CHECK:                  {
// CHECK:                    "module-name": "C"
// CHECK:                  }
// CHECK-NEXT:           ]
// CHECK:                "command-line": [
// CHECK-NEXT:             "-cc1"
// CHECK:                  "-fcas-path"
// CHECK-NEXT:             "[[PREFIX]]{{.}}cas"
// CHECK:                  "-faction-cache-path"
// CHECK-NEXT:             "[[PREFIX]]{{.}}cache"
// CHECK:                  "-fcas-fs"
// CHECK-NEXT:             "[[TU_ROOT_ID]]"
// CHECK:                  "-fno-pch-timestamp"
// CHECK:                  "-fcache-compile-job"
// CHECK:                  "-fmodule-file-cache-key=[[C_PCM]]={{llvmcas://[[:xdigit:]]+}}"
// CHECK:                  "-fmodule-file={{(C=)?}}[[C_PCM]]"
// CHECK:                ]
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]{{.}}tu.c"
// CHECK-NEXT:             "[[PREFIX]]{{.}}prefix.h.pch"
// CHECK-NEXT:           ]
// CHECK:              }

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

//--- cdb_pch.json.template
[
  {
    "directory" : "DIR",
    "command" : "clang_tool -x c-header DIR/prefix.h -o DIR/prefix.h.pch -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache",
    "file" : "DIR/prefix.h"
  },
]

//--- cdb.json.template
[
  {
    "directory" : "DIR",
    "command" : "clang_tool -fsyntax-only DIR/tu.c -include DIR/prefix.h -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache",
    "file" : "DIR/tu.c"
  },
]

//--- module.modulemap
module A { header "A.h" export * }
module B { header "B.h" }
module C { header "C.h" export * }

//--- A.h
#include "B.h"

//--- B.h
void B(void);

//--- C.h
#include "B.h"
void B(void);

//--- prefix.h
#include "A.h"

//--- tu.c
#include "C.h"
void tu(void) {
  B();
}
