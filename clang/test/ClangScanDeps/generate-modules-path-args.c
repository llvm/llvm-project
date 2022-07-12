// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_without.json.template > %t/cdb_without.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -generate-modules-path-args > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s
// RUN: clang-scan-deps -compilation-database %t/cdb_without.json \
// RUN:   -format experimental-full -generate-modules-path-args > %t/deps_without.json
// RUN: cat %t/deps_without.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t -check-prefix=WITHOUT %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-serialize-diagnostic-file"
// CHECK-NEXT:         "[[PREFIX]]{{.*}}Mod{{.*}}.diag"
// CHECK:              "-dependency-file"
// CHECK-NEXT:         "[[PREFIX]]{{.*}}Mod{{.*}}.d"
// CHECK:            ],

// WITHOUT:      {
// WITHOUT-NEXT:   "modules": [
// WITHOUT-NEXT:     {
// WITHOUT:            "command-line": [
// WITHOUT-NEXT:         "-cc1"
// WITHOUT-NOT:          "-serialize-diagnostic-file"
// WITHOUT-NOT:          "-dependency-file"
// WITHOUT:            ],

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -serialize-diagnostics DIR/tu.diag -MD -MT tu -MF DIR/tu.d",
  "file": "DIR/tu.c"
}]

//--- cdb_without.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
  "file": "DIR/tu.c"
}]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu.c
#include "Mod.h"
