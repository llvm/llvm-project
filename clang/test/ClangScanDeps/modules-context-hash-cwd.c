// Test current directory pruning when computing the context hash.

// REQUIRES: shell

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb0.json.in > %t/cdb0.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb1.json.in > %t/cdb1.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb2.json.in > %t/cdb2.json
// RUN: clang-scan-deps -compilation-database %t/cdb0.json -format experimental-full > %t/result0.json
// RUN: clang-scan-deps -compilation-database %t/cdb1.json -format experimental-full > %t/result1.json
// RUN: clang-scan-deps -compilation-database %t/cdb2.json -format experimental-full -optimize-args=header-search,system-warnings,vfs,canonicalize-macros > %t/result2.json
// RUN: cat %t/result0.json %t/result1.json | FileCheck %s
// RUN: cat %t/result0.json %t/result2.json | FileCheck %s -check-prefix=SKIPOPT

//--- cdb0.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -IDIR/include/ -o DIR/tu.o",
  "file": "DIR/tu.c"
}]

//--- cdb1.json.in
[{
  "directory": "DIR/a",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -IDIR/include/ -o DIR/tu.o",
  "file": "DIR/tu.c"
}]

//--- cdb2.json.in
[{
  "directory": "DIR/a/",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -IDIR/include/ -o DIR/tu.o",
  "file": "DIR/tu.c"
}]

//--- include/module.modulemap
module mod {
  header "mod.h"
}

//--- include/mod.h

//--- tu.c
#include "mod.h"

// Check that result0 and result1 compute the same hash with optimization
// on. The only difference between result0 and result1 is the compiler's
// working directory.
// CHECK:     {
// CHECK-NEXT:  "modules": [
// CHECK-NEXT:   {
// CHECK-NEXT:     "clang-module-deps": [],
// CHECK:          "context-hash": "[[HASH:.*]]",
// CHECK:        }
// CHECK:       "translation-units": [
// CHECK:        {
// CHECK:          "commands": [
// CHECK:          {
// CHECK-NEXT:        "clang-context-hash": "{{.*}}",
// CHECK-NEXT:        "clang-module-deps": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "context-hash": "[[HASH]]",
// CHECK-NEXT:            "module-name": "mod"
// CHECK:               }
// CHECK:             ],
// CHECK:     {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "clang-module-deps": [],
// CHECK:           "context-hash": "[[HASH]]",
// CHECK:         }
// CHECK:        "translation-units": [
// CHECK:         {
// CHECK:           "commands": [
// CHECK:           {
// CHECK-NEXT:         "clang-context-hash": "{{.*}}",
// CHECK-NEXT:         "clang-module-deps": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "context-hash": "[[HASH]]",
// CHECK-NEXT:             "module-name": "mod"
// CHECK:               }
// CHECK:              ],

// Check that result0 and result2 compute different hashes because
// the working directory optmization is turned off for result2.
// SKIPOPT:      {
// SKIPOPT-NEXT:   "modules": [
// SKIPOPT-NEXT:    {
// SKIPOPT-NEXT:      "clang-module-deps": [],
// SKIPOPT:           "context-hash": "[[HASH0:.*]]",
// SKIPOPT:         }
// SKIPOPT:        "translation-units": [
// SKIPOPT:         {
// SKIPOPT:            "commands": [
// SKIPOPT:             {
// SKIPOPT-NEXT:          "clang-context-hash": "{{.*}}",
// SKIPOPT-NEXT:          "clang-module-deps": [
// SKIPOPT-NEXT:            {
// SKIPOPT-NEXT:              "context-hash": "[[HASH0]]",
// SKIPOPT-NEXT:              "module-name": "mod"
// SKIPOPT:            }
// SKIPOPT:          ],
// SKIPOPT:      {
// SKIPOPT-NEXT:   "modules": [
// SKIPOPT-NEXT:     {
// SKIPOPT-NEXT:       "clang-module-deps": [],
// SKIPOPT-NOT:        "context-hash": "[[HASH0]]",
// SKIPOPT:            "context-hash": "[[HASH2:.*]]",
// SKIPOPT:          }
// SKIPOPT:       "translation-units": [
// SKIPOPT:         {
// SKIPOPT:           "commands": [
// SKIPOPT:             {
// SKIPOPT-NEXT:          "clang-context-hash": "{{.*}}",
// SKIPOPT-NEXT:          "clang-module-deps": [
// SKIPOPT-NEXT:            {
// SKIPOPT-NOT:              "context-hash": "[[HASH0]]",
// SKIPOPT-NEXT:             "context-hash": "[[HASH2]]"
// SKIPOPT-NEXT:              "module-name": "mod"
// SKIPOPT:            }
// SKIPOPT:          ],

