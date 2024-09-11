// Test scanning when the driver requires multiple jobs. E.g. with -save-temps
// there will be separate -E, -emit-llvm-bc, -S, and -cc1as jobs, which should
// each result in a "command" in the output.

// We use an x86_64-apple-darwin target to avoid host-dependent behaviour in
// the driver. Platforms without an integrated assembler have different commands
// REQUIRES: x86-registered-target

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -module-files-dir %t/modules \
// RUN:   -- %clang -target x86_64-apple-darwin -c %t/src0/tu.c -save-temps=obj -o %t/dst0/tu.o -I %t/include \
// RUN:     -fdepscan-prefix-map=%t/src0=^src -fdepscan-prefix-map=%t/include=^include \
// RUN:     -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   > %t/deps.0.json

// RUN: cat %t/deps.0.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -DSRC=%/t/src0 -DDST=%/t/dst0

// RUN: c-index-test core -scan-deps -working-dir %t -cas-path %t/cas -output-dir %t/modules -- \
// RUN:   %clang -target x86_64-apple-darwin -c %t/src0/tu.c -save-temps=obj -o %t/dst0/tu.o -I %t/include \
// RUN:   -fdepscan-prefix-map=%t/src0=^srcx \
// RUN:   -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   > %t/deps.txt

// RUN: cat %t/deps.txt | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -DSRC=%/t/src0 -DDST=%/t/dst0 -check-prefix=CHECK-LIBCLANG

// RUN: mkdir %t/dst0
// RUN: %deps-to-rsp %t/deps.0.json --module-name=Mod             > %t/Mod.0.rsp
// RUN: %deps-to-rsp %t/deps.0.json --tu-index 0 --tu-cmd-index 0 > %t/tu-cpp.0.rsp
// RUN: %deps-to-rsp %t/deps.0.json --tu-index 0 --tu-cmd-index 1 > %t/tu-emit-ir.0.rsp
// RUN: %deps-to-rsp %t/deps.0.json --tu-index 0 --tu-cmd-index 2 > %t/tu-emit-asm.0.rsp
// RUN: %deps-to-rsp %t/deps.0.json --tu-index 0 --tu-cmd-index 3 > %t/tu-cc1as.0.rsp
// RUN: %clang @%t/Mod.0.rsp         -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu-cpp.0.rsp      -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu-emit-ir.0.rsp  -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu-emit-asm.0.rsp -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu-cc1as.0.rsp

// RUN: clang-scan-deps -format experimental-include-tree-full -cas-path %t/cas -module-files-dir %t/modules \
// RUN:   -- %clang -target x86_64-apple-darwin -c %t/src1/tu.c -save-temps=obj -o %t/dst1/tu.o -I %t/include \
// RUN:     -fdepscan-prefix-map=%t/src1=^src -fdepscan-prefix-map=%t/include=^include \
// RUN:     -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   > %t/deps.1.json

// The dependency graph has identical structure, just the include-tree ID, dependent cache keys and prefix mappings are different.
// RUN: cat %t/deps.1.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -DSRC=%/t/src1 -DDST=%/t/dst1
// RUN: not diff %t/deps.1.json %t/deps.0.json

// RUN: mkdir %t/dst1
// RUN: %deps-to-rsp %t/deps.1.json --module-name=Mod             > %t/Mod.1.rsp
// RUN: %deps-to-rsp %t/deps.1.json --tu-index 0 --tu-cmd-index 0 > %t/tu-cpp.1.rsp
// RUN: %deps-to-rsp %t/deps.1.json --tu-index 0 --tu-cmd-index 1 > %t/tu-emit-ir.1.rsp
// RUN: %deps-to-rsp %t/deps.1.json --tu-index 0 --tu-cmd-index 2 > %t/tu-emit-asm.1.rsp
// RUN: %deps-to-rsp %t/deps.1.json --tu-index 0 --tu-cmd-index 3 > %t/tu-cc1as.1.rsp
// RUN: %clang @%t/Mod.1.rsp         -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: %clang @%t/tu-cpp.1.rsp      -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// RUN: %clang @%t/tu-emit-ir.1.rsp  -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: %clang @%t/tu-emit-asm.1.rsp -Rcompile-job-cache 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// RUN: %clang @%t/tu-cc1as.1.rsp

// RUN: diff %t/dst1/tu.i  %t/dst0/tu.i
// RUN: diff %t/dst1/tu.bc %t/dst0/tu.bc
// RUN: diff %t/dst1/tu.s  %t/dst0/tu.s
// RUN: diff %t/dst1/tu.o  %t/dst0/tu.o

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

// CHECK:      "modules": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "cache-key": "[[M_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:     "cas-include-tree-id": "[[M_INCLUDE_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:     "clang-module-deps": []
// CHECK-NEXT:     "clang-modulemap-file": "[[PREFIX]]/include/module.modulemap"
// CHECK-NEXT:     "command-line": [
// CHECK:            "-fcas-include-tree"
// CHECK-NEXT:       "[[M_INCLUDE_TREE]]"
// CHECK:          ]
// CHECK:          "name": "Mod"
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: "translation-units": [
// CHECK-NEXT:   {
// CHECK:          "commands": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "cache-key": "[[CPP_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:         "cas-include-tree-id": "[[CPP_INCLUDE_TREE:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:         "clang-context-hash": "{{.*}}"
// CHECK-NEXT:         "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "context-hash": "{{.*}}
// CHECK-NEXT:             "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "[[DST]]/tu.i"
// CHECK:                "-fcas-include-tree"
// CHECK-NEXT:           "[[CPP_INCLUDE_TREE]]"
// CHECK-NOT:            "-fcas-input-file-cache-key"
// CHECK:                "-E"
// CHECK:                "-fmodule-file-cache-key"
// CHECK-NEXT:           "[[PREFIX]]/modules/{{.*}}/Mod-{{.*}}.pcm"
// CHECK-NEXT:           "[[M_CACHE_KEY]]"
// CHECK:                "-x"
// CHECK-NEXT:           "c"
// CHECK-NOT:            "{{.*}}tu.c"
// CHECK:                "-fmodule-file={{.*}}[[PREFIX]]/modules/{{.*}}/Mod-{{.*}}.pcm"
// CHECK:              ]
// CHECK:              "file-deps": [
// CHECK-NEXT:           "[[SRC]]/tu.c"
// CHECK-NEXT:           "[[SRC]]/header.h"
// CHECK-NEXT:         ]
// CHECK:              "input-file": "[[SRC]]/tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK-NEXT:         "cache-key": "[[COMPILER_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// FIXME: This should be empty.
// CHECK-NEXT:         "cas-include-tree-id": "{{.*}}"
// CHECK-NEXT:         "clang-context-hash": "{{.*}}"
// CHECK-NEXT:         "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "context-hash": "{{.*}}
// CHECK-NEXT:             "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "[[DST]]/tu.bc"
// CHECK-NOT:            "-fcas-include-tree"
// CHECK:                "-fcas-input-file-cache-key"
// CHECK-NEXT:           "[[CPP_CACHE_KEY]]"
// CHECK:                "-emit-llvm-bc"
// CHECK:                "-fmodule-file-cache-key"
// CHECK-NEXT:           "[[PREFIX]]/modules/{{.*}}/Mod-{{.*}}.pcm"
// CHECK-NEXT:           "[[M_CACHE_KEY]]"
// CHECK:                "-x"
// CHECK-NEXT:           "c-cpp-output"
// CHECK-NOT:            "{{.*}}tu.i"
// CHECK:                "-fmodule-file={{.*}}[[PREFIX]]/modules/{{.*}}/Mod-{{.*}}.pcm"
// CHECK:              ]
// CHECK:              "input-file": "[[SRC]]/tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK-NEXT:         "cache-key": "[[BACKEND_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// FIXME: This should be empty.
// CHECK-NEXT:         "cas-include-tree-id": "{{.*}}"
// CHECK-NEXT:         "clang-context-hash": "{{.*}}"
// FIXME: This should be empty.
// CHECK-NEXT:         "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "context-hash": "{{.*}}
// CHECK-NEXT:             "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "[[DST]]/tu.s"
// CHECK:                "-fcas-input-file-cache-key"
// CHECK-NEXT:           "[[COMPILER_CACHE_KEY]]"
// CHECK:                "-S"
// CHECK:                "-x"
// CHECK-NEXT:           "ir"
// CHECK:              ]
// CHECK:              "input-file": "[[SRC]]/tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// FIXME: This should be empty.
// CHECK-NEXT:         "cas-include-tree-id": "{{.*}}"
// CHECK-NEXT:         "clang-context-hash": "{{.*}}"
// FIXME: This should be empty.
// CHECK-NEXT:         "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK:                  "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1as"
// CHECK:                "-o"
// CHECK-NEXT:           "[[DST]]/tu.o"
// FIXME: The integrated assembler should support caching too.
// CHECK:                "[[DST]]/tu.s"
// CHECK:              ]
// CHECK:              "input-file": "[[SRC]]/tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK-LIBCLANG:      modules:
// CHECK-LIBCLANG-NEXT:   module:
// CHECK-LIBCLANG-NEXT:     name: Mod
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// CHECK-LIBCLANG-NEXT:     module-map-path: [[PREFIX]]/include/module.modulemap
// CHECK-LIBCLANG-NEXT:     include-tree-id: [[M_INCLUDE_TREE:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     cache-key: [[M_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/include/module.h
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/include/module.modulemap
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -fcas-include-tree [[M_INCLUDE_TREE]]
// CHECK-LIBCLANG-NEXT: dependencies:
// CHECK-LIBCLANG-NEXT:   command 0:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// CHECK-LIBCLANG-NEXT:     include-tree-id: [[CPP_INCLUDE_TREE:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     cache-key: [[CPP_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[SRC]]/tu.c
// CHECK-LIBCLANG-NEXT:       [[SRC]]/header.h
// CHECK-LIBCLANG-NOT:             -fcas-input-file-cache-key
// CHECK-LIBCLANG-NOT:             {{.*}}tu.c
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -o [[DST]]/tu.i {{.*}} -E -fmodule-file-cache-key {{.*}} [[M_CACHE_KEY]] -x c {{.*}} -fmodule-file={{.*}}[[PREFIX]]/modules/Mod_{{.*}}.pcm
// CHECK-LIBCLANG-NEXT:   command 1:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     include-tree-id: {{.*}}
// CHECK-LIBCLANG-NEXT:     cache-key: [[COMPILER_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[SRC]]/tu.c
// CHECK-LIBCLANG-NEXT:       [[SRC]]/header.h
// CHECK-LIBCLANG-NOT:                  -fcas-include-tree
// CHECK-LIBCLANG-NOT:                  {{.*}}tu.i
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -o [[DST]]/tu.bc {{.*}} -fcas-input-file-cache-key [[CPP_CACHE_KEY]] {{.*}} -emit-llvm-bc -fmodule-file-cache-key {{.*}} [[M_CACHE_KEY]] -x c-cpp-output {{.*}} -fmodule-file={{.*}}[[PREFIX]]/modules/Mod_{{.*}}.pcm
// CHECK-LIBCLANG-NEXT:   command 2:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     include-tree-id: {{.*}}
// CHECK-LIBCLANG-NEXT:     cache-key: [[BACKEND_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[SRC]]/tu.c
// CHECK-LIBCLANG-NEXT:       [[SRC]]/header.h
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -o [[DST]]/tu.s {{.*}} -fcas-input-file-cache-key [[COMPILER_CACHE_KEY]] {{.*}} -S -x ir
// CHECK-LIBCLANG-NEXT:   command 3:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     include-tree-id: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[SRC]]/tu.c
// CHECK-LIBCLANG-NEXT:       [[SRC]]/header.h
// FIXME: The integrated assembler should support caching too.
// CHECK-LIBCLANG-NEXT:     build-args: -cc1as {{.*}} -o [[DST]]/tu.o [[DST]]/tu.s

//--- include/module.h
void bar(void);

//--- include/module.modulemap
module Mod { header "module.h" }

//--- src0/header.h
//--- src0/tu.c
#include "module.h"
#include "header.h"
#define FOO 0
void tu_save_temps(void) { bar(); }

//--- src1/header.h
//--- src1/tu.c
#include "module.h"
#include "header.h"
#define FOO 1
void tu_save_temps(void) { bar(); }
