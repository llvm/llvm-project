// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: sed "s|DIR|%/t|g" %t/cdb_without.json.template > %t/cdb_without.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -dependency-target foo > %t/deps_mt1.json
// RUN: cat %t/deps_mt1.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s -check-prefix=DEPS_MT1
// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-full -dependency-target foo -dependency-target bar > %t/deps_mt2.json
// RUN: cat %t/deps_mt2.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s -check-prefix=DEPS_MT2
// RUN: clang-scan-deps -compilation-database %t/cdb_without.json \
// RUN:   -format experimental-full > %t/deps_without.json
// RUN: cat %t/deps_without.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t -check-prefix=WITHOUT %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-serialize-diagnostic-file"
// CHECK-NEXT:         "[[PREFIX]]{{.*}}Mod{{.*}}.diag"
// CHECK:              "-MT"
// CHECK-NEXT:         "[[PREFIX]]{{.*}}Mod{{.*}}.pcm"
// CHECK:              "-dependency-file"
// CHECK-NEXT:         "[[PREFIX]]{{.*}}Mod{{.*}}.d"
// CHECK:            ],

// DEPS_MT1:      "-MT"
// DEPS_MT1-NEXT: "foo"

// DEPS_MT2:      "-MT"
// DEPS_MT2-NEXT: "foo"
// DEPS_MT2-NEXT: "-MT"
// DEPS_MT2-NEXT: "bar"

// WITHOUT:      {
// WITHOUT-NEXT:   "modules": [
// WITHOUT-NEXT:     {
// WITHOUT:            "command-line": [
// WITHOUT-NEXT:         "-cc1"
// WITHOUT-NOT:          "-serialize-diagnostic-file"
// WITHOUT-NOT:          "-dependency-file"
// WITHOUT-NOT:          "-MT"
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
