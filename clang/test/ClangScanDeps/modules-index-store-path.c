// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build > %t/result.json

// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK-NOT:          "-index-unit-output-path"
// CHECK:            ]

// RUN: %deps-to-rsp %t/result.json --module-name=Mod > %t/Mod.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.cc1.rsp
// RUN: %clang @%t/Mod.cc1.rsp -pedantic -Werror
// RUN: %clang @%t/tu.cc1.rsp -pedantic -Werror
// RUN: c-index-test core -print-unit %t/index | FileCheck %s -DPREFIX=%/t -check-prefix=INDEX
// INDEX-DAG: out-file: [[PREFIX]]{{/|\\}}build/{{.*(/|\\)}}Mod-{{.*}}.pcm
// INDEX-DAG: out-file: /tu.o

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -index-store-path DIR/index -index-unit-output-path /tu.o -o DIR/tu.o",
    "file": "DIR/tu.c"
  }
]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h
void mod(void);

//--- tu.c
#include "Mod.h"
void tu(void) {
  mod();
}
