// Test scanning when the driver requires multiple jobs. E.g. with -save-temps
// there will be separate -E, -emit-llvm-bc, -S, and -cc1as jobs, which should
// each result in a "command" in the output.

// We use an x86_64-apple-darwin target to avoid host-dependent behaviour in
// the driver. Platforms without an integrated assembler have different commands
// REQUIRES: x86-registered-target

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: mv %t/tu_define_foo_0.c %t/tu.c
// RUN: clang-scan-deps -format experimental-tree-full -cas-path %t/cas -module-files-dir %t/modules \
// RUN:   -- %clang -target x86_64-apple-darwin -c %t/tu.c -save-temps=obj -o %t/tu.o \
// RUN:     -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   > %t/deps.0.json

// RUN: cat %t/deps.0.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// RUN: CLANG_CACHE_USE_CASFS_DEPSCAN=1 c-index-test core -scan-deps -working-dir %t -cas-path %t/cas -output-dir %t/modules -- \
// RUN:   %clang -target x86_64-apple-darwin -c %t/tu.c -save-temps=obj -o %t/tu.o \
// RUN:   -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   > %t/deps.txt

// RUN: cat %t/deps.txt | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -check-prefix=CHECK-LIBCLANG

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
// RUN: mv %t/tu.i  %t/tu.0.i
// RUN: mv %t/tu.bc %t/tu.0.bc
// RUN: mv %t/tu.s  %t/tu.0.s
// RUN: mv %t/tu.o  %t/tu.0.o

// RUN: mv %t/tu_define_foo_1.c %t/tu.c
// RUN: clang-scan-deps -format experimental-tree-full -cas-path %t/cas -module-files-dir %t/modules \
// RUN:   -- %clang -target x86_64-apple-darwin -c %t/tu.c -save-temps=obj -o %t/tu.o \
// RUN:     -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   > %t/deps.1.json

// The dependency graph has identical structure, just the FS root ID and dependent cache keys are different.
// RUN: cat %t/deps.1.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// RUN: not diff %t/deps.1.json %t/deps.0.json

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
// RUN: mv %t/tu.i  %t/tu.1.i
// RUN: mv %t/tu.bc %t/tu.1.bc
// RUN: mv %t/tu.s  %t/tu.1.s
// RUN: mv %t/tu.o  %t/tu.1.o

// RUN: diff %t/tu.1.i  %t/tu.0.i
// RUN: diff %t/tu.1.bc %t/tu.0.bc
// RUN: diff %t/tu.1.s  %t/tu.0.s
// RUN: diff %t/tu.1.o  %t/tu.0.o

// CACHE-HIT: remark: compile job cache hit
// CACHE-MISS: remark: compile job cache miss

// CHECK:      "modules": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "cache-key": "[[M_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:     "casfs-root-id": "[[M_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:     "clang-module-deps": []
// CHECK-NEXT:     "clang-modulemap-file": "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:     "command-line": [
// CHECK:            "-fcas-fs"
// CHECK-NEXT:       "[[M_ROOT_ID]]"
// CHECK:          ]
// CHECK:          "name": "Mod"
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-NEXT: "translation-units": [
// CHECK-NEXT:   {
// CHECK:          "commands": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "cache-key": "[[CPP_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:         "casfs-root-id": "[[CPP_ROOT_ID:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:         "clang-context-hash": "{{.*}}"
// CHECK-NEXT:         "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "context-hash": "{{.*}}
// CHECK-NEXT:             "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-fcas-fs"
// CHECK-NEXT:           "[[CPP_ROOT_ID]]"
// CHECK:                "-o"
// CHECK-NEXT:           "[[PREFIX]]/tu.i"
// CHECK-NOT:            "-fcas-input-file-cache-key"
// CHECK:                "-E"
// CHECK:                "-fmodule-file-cache-key"
// CHECK-NEXT:           "[[PREFIX]]/modules/{{.*}}/Mod-{{.*}}.pcm"
// CHECK-NEXT:           "[[M_CACHE_KEY]]"
// CHECK:                "-x"
// CHECK-NEXT:           "c"
// CHECK:                "[[PREFIX]]/tu.c"
// CHECK:                "-fmodule-file={{.*}}[[PREFIX]]/modules/{{.*}}/Mod-{{.*}}.pcm"
// CHECK:              ]
// CHECK:              "file-deps": [
// CHECK-NEXT:           "[[PREFIX]]/tu.c"
// CHECK-NEXT:         ]
// CHECK:              "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK-NEXT:         "cache-key": "[[COMPILER_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:         "casfs-root-id": "{{.*}}"
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
// CHECK-NEXT:           "[[PREFIX]]/tu.bc"
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
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK-NEXT:         "cache-key": "[[BACKEND_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// CHECK-NEXT:         "casfs-root-id": "{{.*}}"
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
// CHECK-NEXT:           "[[PREFIX]]/tu.s"
// CHECK:                "-fcas-input-file-cache-key"
// CHECK-NEXT:           "[[COMPILER_CACHE_KEY]]"
// CHECK:                "-S"
// CHECK:                "-x"
// CHECK-NEXT:           "ir"
// CHECK:              ]
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK-NEXT:         "casfs-root-id": "{{.*}}"
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
// CHECK-NEXT:           "[[PREFIX]]/tu.o"
// FIXME: The integrated assembler should support caching too.
// CHECK:                "[[PREFIX]]/tu.s"
// CHECK:              ]
// CHECK:              "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:       }
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK-LIBCLANG:      modules:
// CHECK-LIBCLANG-NEXT:   module:
// CHECK-LIBCLANG-NEXT:     name: Mod
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// CHECK-LIBCLANG-NEXT:     module-map-path: [[PREFIX]]/module.modulemap
// CHECK-LIBCLANG-NEXT:     casfs-root-id: [[M_ROOT_ID:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     cache-key: [[M_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/module.h
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/module.modulemap
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -fcas-fs [[M_ROOT_ID]]
// CHECK-LIBCLANG-NEXT: dependencies:
// CHECK-LIBCLANG-NEXT:   command 0:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// CHECK-LIBCLANG-NEXT:     casfs-root-id: [[CPP_ROOT_ID:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     cache-key: [[CPP_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/tu.c
// CHECK-LIBCLANG-NOT:             -fcas-input-file-cache-key
// CHECK-LIBCLANG-NOT:             {{.*}}tu.c
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -o [[PREFIX]]/tu.i {{.*}} -E -fmodule-file-cache-key {{.*}} [[M_CACHE_KEY]] -x c {{.*}} -fmodule-file={{.*}}[[PREFIX]]/modules/Mod_{{.*}}.pcm
// CHECK-LIBCLANG-NEXT:   command 1:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     casfs-root-id: {{.*}}
// CHECK-LIBCLANG-NEXT:     cache-key: [[COMPILER_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/tu.c
// CHECK-LIBCLANG-NOT:                  -fcas-fs
// CHECK-LIBCLANG-NOT:                  {{.*}}tu.i
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -o [[PREFIX]]/tu.bc {{.*}} -fcas-input-file-cache-key [[CPP_CACHE_KEY]] {{.*}} -emit-llvm-bc -fmodule-file-cache-key {{.*}} [[M_CACHE_KEY]] -x c-cpp-output {{.*}} -fmodule-file={{.*}}[[PREFIX]]/modules/Mod_{{.*}}.pcm
// CHECK-LIBCLANG-NEXT:   command 2:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     casfs-root-id: {{.*}}
// CHECK-LIBCLANG-NEXT:     cache-key: [[BACKEND_CACHE_KEY:llvmcas://[[:xdigit:]]+]]
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/tu.c
// CHECK-LIBCLANG-NEXT:     build-args: -cc1 {{.*}} -o [[PREFIX]]/tu.s {{.*}} -fcas-input-file-cache-key [[COMPILER_CACHE_KEY]] {{.*}} -S -x ir
// CHECK-LIBCLANG-NEXT:   command 3:
// CHECK-LIBCLANG-NEXT:     context-hash: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     casfs-root-id: {{.*}}
// FIXME: This should be empty.
// CHECK-LIBCLANG-NEXT:     module-deps:
// CHECK-LIBCLANG-NEXT:       Mod:{{.*}}
// CHECK-LIBCLANG-NEXT:     file-deps:
// CHECK-LIBCLANG-NEXT:       [[PREFIX]]/tu.c
// FIXME: The integrated assembler should support caching too.
// CHECK-LIBCLANG-NEXT:     build-args: -cc1as {{.*}} -o [[PREFIX]]/tu.o [[PREFIX]]/tu.s

//--- module.h
void bar(void);

//--- module.modulemap
module Mod { header "module.h" }

//--- tu_define_foo_0.c
#include "module.h"
#define FOO 0
void tu_save_temps(void) { bar(); }

//--- tu_define_foo_1.c
#include "module.h"
#define FOO 1
void tu_save_temps(void) { bar(); }
