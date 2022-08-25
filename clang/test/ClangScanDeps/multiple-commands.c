// Test scanning when the driver requires multiple jobs. E.g. with -save-temps
// there will be separate -E, -emit-llvm-bc, -S, and -cc1as jobs, which should
// each result in a "command" in the output.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -module-files-dir %t/modules \
// RUN:   -j 1 -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// Build the -save-temps + -fmodules case
// RUN: %deps-to-rsp %t/deps.json --module-name=Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 1 --tu-cmd-index 0 > %t/tu-cpp.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 1 --tu-cmd-index 1 > %t/tu-emit-ir.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 1 --tu-cmd-index 2 > %t/tu-emit-asm.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 1 --tu-cmd-index 3 > %t/tu-cc1as.rsp
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/tu-cpp.rsp
// RUN: ls %t/tu_save_temps_module.i
// RUN: %clang @%t/tu-emit-ir.rsp
// RUN: ls %t/tu_save_temps_module.bc
// RUN: %clang @%t/tu-emit-asm.rsp
// RUN: ls %t/tu_save_temps_module.s
// RUN: %clang @%t/tu-cc1as.rsp
// RUN: ls %t/tu_save_temps_module.o


// CHECK:      "modules": [
// CHECK-NEXT:   {
// CHECK:          "clang-modulemap-file": "[[PREFIX]]{{.}}module.modulemap"
// CHECK:          "name": "Mod"
// CHECK:        }
// CHECK-NEXT: ]
// CHECK-NEXT: "translation-units": [
// CHECK-NEXT:   {
// CHECK:          "commands": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "clang-context-hash":
// CHECK-NEXT:         "clang-module-deps": []
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "{{.*}}tu_no_integrated_cpp{{.*}}.i"
// CHECK:                "-E"
// CHECK:              ]
// CHECK-NEXT:         "executable": "clang_tool"
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu_no_integrated_cpp.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK-NEXT:         "clang-context-hash":
// CHECK-NEXT:         "clang-module-deps": []
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "{{.*}}tu_no_integrated_cpp.o"
// CHECK:                "-emit-obj"
// CHECK:                "{{.*}}tu_no_integrated_cpp{{.*}}.i"
// CHECK:              ]
// CHECK-NEXT:         "executable": "clang_tool"
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu_no_integrated_cpp.c"
// CHECK-NEXT:       }
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   {
// CHECK-NEXT:     "commands": [
// CHECK-NEXT:       {
// CHECK:              "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK:                  "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "{{.*}}tu_save_temps_module.i"
// CHECK:                "-E"
// CHECK:                "-fmodule-file={{.*}}[[PREFIX]]{{.}}modules{{.*}}Mod-{{.*}}.pcm"
// CHECK:                "{{.*}}tu_save_temps_module.c"
// CHECK:              ]
// CHECK-NEXT:         "executable": "clang_tool"
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu_save_temps_module.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK:              "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK:                  "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "{{.*}}tu_save_temps_module.bc"
// CHECK:                "-emit-llvm-bc"
// CHECK:                "{{.*}}tu_save_temps_module.i"
// CHECK:                "-fmodule-file={{.*}}[[PREFIX]]{{.}}modules{{.*}}Mod-{{.*}}.pcm"
// CHECK:              ]
// CHECK-NEXT:         "executable": "clang_tool"
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu_save_temps_module.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK:              "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK:                  "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1"
// CHECK:                "-o"
// CHECK-NEXT:           "{{.*}}tu_save_temps_module.s"
// CHECK:                "-S"
// CHECK:                "{{.*}}tu_save_temps_module.bc"
// CHECK:              ]
// CHECK-NEXT:         "executable": "clang_tool"
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu_save_temps_module.c"
// CHECK-NEXT:       }
// CHECK-NEXT:       {
// CHECK:              "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK:                  "module-name": "Mod"
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
// CHECK-NEXT:         "command-line": [
// CHECK-NEXT:           "-cc1as"
// CHECK:                "-o"
// CHECK-NEXT:           "{{.*}}tu_save_temps_module.o"
// CHECK:                "{{.*}}tu_save_temps_module.s"
// CHECK:              ]
// CHECK-NEXT:         "executable": "clang_tool"
// CHECK:              "input-file": "[[PREFIX]]{{.}}tu_save_temps_module.c"
// CHECK-NEXT:       }
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]


//--- cdb.json.in
[
  {
    "directory": "DIR"
    "command": "clang_tool -c DIR/tu_no_integrated_cpp.c -no-integrated-cpp -o DIR/tu_no_integrated_cpp.o"
    "file": "DIR/tu_no_integrated_cpp.c"
  },
  {
    "directory": "DIR"
    "command": "clang_tool -c DIR/tu_save_temps_module.c -save-temps=obj -o DIR/tu_save_temps_module.o -fmodules -fimplicit-modules -fimplicit-module-maps"
    "file": "DIR/tu_save_temps_module.c"
  }
]

//--- plain_header.h
void foo(void);

//--- module_header.h
void bar(void);

//--- module.modulemap
module Mod { header "module_header.h" }

//--- tu_no_integrated_cpp.c
#include "plain_header.h"
void tu_no_integrated_cpp(void) { foo(); }

//--- tu_save_temps_module.c
#include "module_header.h"
void tu_save_temps(void) { bar(); }
