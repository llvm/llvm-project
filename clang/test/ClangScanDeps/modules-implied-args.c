// Check that we get canonical round-trippable command-lines, in particular
// for the options modified for modules.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -round-trip-args > %t/result.json
// RUN: cat %t/result.json  | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefixes=CHECK,NEGATIVE

// -ffast-math implies -menable-no-infs, -menable-no-nans, and -mreassociate;
// those options are modified by resetNonModularOptions.

// NEGATIVE-NOT: "-menable-no-infs"
// NEGATIVE-NOT: "-menable-no-nans"
// NEGATIVE-NOT: "-mreassociate"

// CHECK:      "modules": [
// CHECK-NEXT:   {
// CHECK:          "clang-module-deps": []
// CHECK:          "command-line": [
// CHECK:            "-ffast-math"
// CHECK:          ]
// CHECK:          "name": "Mod"
// CHECK:        }
// CHECK-NEXT: ]
// CHECK:      "translation-units": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "commands": [
// CHECK:            {
// CHECK:              "command-line": [
// CHECK:                "-ffast-math"
// CHECK:              ]

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -ffast-math"
}]

//--- module.modulemap
module Mod { header "mod.h" }
//--- mod.h
//--- tu.c
#include "mod.h"
