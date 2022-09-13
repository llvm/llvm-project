// Check that we can scan modules with caching enabled and build the resulting
// commands.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Full and tree-full modes are identical here.
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-tree-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_tree.json
// RUN: diff -u %t/deps_tree.json %t/deps.json

// Disabling/re-enabling the cas should not be cached in the implicit pcms
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_no_cas.json
// RUN: cat %t/deps_no_cas.json | FileCheck %s -check-prefix=NO_CAS
// NO_CAS-NOT: fcas
// NO_CAS-NOT: faction-cache
// NO_CAS-NOT: fcache-compile-job
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps_cas_2.json
// RUN: diff -u %t/deps_cas_2.json %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Top > %t/Top.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Left > %t/Left.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name Right > %t/Right.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// Missing pcm in action cache
// RUN: not %clang @%t/Left.rsp 2> %t/error.txt
// RUN: cat %t/error.txt | FileCheck %s -check-prefix=MISSING
// MISSING: error: CAS cannot load module with key '{{.*}}' from -fmodule-file-cache-key: no such entry in action cache

// Build everything
// RUN: %clang @%t/Top.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/Left.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/Left.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: %clang @%t/Right.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/Right.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// Ensure we load pcms from action cache
// RUN: rm -rf %t/outputs
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s -check-prefix=CACHE-HIT

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

// Check specifics of the command-line
// RUN: cat %t/deps.json | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "casfs-root-id": "[[LEFT_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [
// CHECK:              {
// CHECK:                "module-name": "Top"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cas"
// CHECK:              "-faction-cache-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cache"
// CHECK:              "-fcas-fs"
// CHECK-NEXT:         "[[LEFT_ROOT_ID]]"
// CHECK:              "-o"
// CHECK-NEXT:         "[[LEFT_PCM:.*outputs.*Left-.*\.pcm]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file-cache-key=[[TOP_PCM:.*outputs.*Top-.*\.pcm]]=[[TOP_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK:              "-fmodule-file={{(Top=)?}}[[TOP_PCM]]"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]{{.}}Left.h"
// CHECK-NEXT:         "[[PREFIX]]{{.}}module.modulemap"
// CHECK-NEXT:       ]
// CHECK:            "name": "Left"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "casfs-root-id": "[[RIGHT_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": [
// CHECK:              {
// CHECK:                "module-name": "Top"
// CHECK:              }
// CHECK-NEXT:       ]
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cas"
// CHECK:              "-faction-cache-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cache"
// CHECK:              "-fcas-fs"
// CHECK-NEXT:         "[[RIGHT_ROOT_ID]]"
// CHECK:              "-o"
// CHECK-NEXT:         "[[RIGHT_PCM:.*outputs.*Right-.*\.pcm]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:              "-fmodule-file-cache-key=[[TOP_PCM]]=[[TOP_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK:              "-fmodule-file={{(Top=)?}}[[TOP_PCM]]"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]{{.}}Right.h"
// CHECK-NEXT:         "[[PREFIX]]{{.}}module.modulemap"
// CHECK:            ]
// CHECK:            "name": "Right"
// CHECK:          }
// CHECK-NEXT:     {
// CHECK:            "casfs-root-id": "[[TOP_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:            "clang-module-deps": []
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-fcas-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cas"
// CHECK:              "-faction-cache-path"
// CHECK-NEXT:         "[[PREFIX]]{{.}}cache"
// CHECK:              "-fcas-fs"
// CHECK-NEXT:         "[[TOP_ROOT_ID]]"
// CHECK:              "-o"
// CHECK-NEXT:         "[[TOP_PCM]]"
// CHECK:              "-fcache-compile-job"
// CHECK:              "-emit-module"
// CHECK:            ]
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]{{.}}Top.h"
// CHECK-NEXT:         "[[PREFIX]]{{.}}module.modulemap"
// CHECK:            ]
// CHECK:            "name": "Top"
// CHECK:          }
// CHECK-NEXT:   ]
// CHECK:        "translation-units": [
// CHECK:          {
// CHECK:            "commands": [
// CHECK:              {
// CHECK:                "casfs-root-id": "[[TU_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK:                "clang-module-deps": [
// CHECK:                  {
// CHECK:                    "module-name": "Left"
// CHECK:                  }
// CHECK:                  {
// CHECK:                    "module-name": "Right"
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
// CHECK:                  "-fcache-compile-job"
// CHECK:                  "-fmodule-file-cache-key=[[LEFT_PCM]]={{llvmcas://[[:xdigit:]]+}}"
// CHECK:                  "-fmodule-file-cache-key=[[RIGHT_PCM]]={{llvmcas://[[:xdigit:]]+}}"
// CHECK:                  "-fmodule-file={{(Left=)?}}[[LEFT_PCM]]"
// CHECK:                  "-fmodule-file={{(Right=)?}}[[RIGHT_PCM]]"
// CHECK:                ]
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]{{.}}tu.c"
// CHECK-NEXT:           ]
// CHECK:              }



//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang_tool -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache -Rcompile-job-cache",
  "file" : "DIR/tu.c"
}]

//--- module.modulemap
module Top { header "Top.h" export * }
module Left { header "Left.h" export * }
module Right { header "Right.h" export * }

//--- Top.h
#pragma once
void Top(void);

//--- Left.h
#pragma once
#include "Top.h"
void Left(void);

//--- Right.h
#pragma once
#include "Top.h"
void Right(void);

//--- tu.c
#include "Left.h"
#include "Right.h"

void tu(void) {
  Top();
  Left();
  Right();
}
