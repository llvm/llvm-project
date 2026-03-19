// Most likely platform specific sed differences
// UNSUPPORTED: system-windows
// Test current directory pruning when computing the context hash.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb0.json.in > %t/cdb0.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb1.json.in > %t/cdb1.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb3.json.in > %t/cdb3.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb4.json.in > %t/cdb4.json
// RUN: sed -e "s|DIR|%/t|g" %t/cdb5.json.in > %t/cdb5.json
// RUN: clang-scan-deps -compilation-database %t/cdb0.json -format experimental-full -optimize-args=all > %t/result0.json
// RUN: clang-scan-deps -compilation-database %t/cdb1.json -format experimental-full -optimize-args=all > %t/result1.json
// It is not a typo to use cdb1.json for result2. We intend to use the same
// compilation database, but different clang-scan-deps optimize-args options.
// RUN: clang-scan-deps -compilation-database %t/cdb1.json -format experimental-full -optimize-args=header-search,system-warnings,vfs,canonicalize-macros > %t/result2.json
// RUN: clang-scan-deps -compilation-database %t/cdb3.json -format experimental-full -optimize-args=all > %t/result3.json
// RUN: clang-scan-deps -compilation-database %t/cdb4.json -format experimental-full -optimize-args=all > %t/result4.json
// RUN: clang-scan-deps -compilation-database %t/cdb5.json -format experimental-full -optimize-args=all > %t/result5.json
// RUN: cat %t/result0.json %t/result1.json | FileCheck %s
// RUN: cat %t/result0.json %t/result2.json | FileCheck %s -check-prefix=SKIPOPT
// RUN: cat %t/result3.json %t/result4.json | FileCheck %s -check-prefix=RELPATH
// RUN: cat %t/result0.json %t/result5.json | FileCheck %s

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

// cdb2 is skipped because we reuse cdb1.

//--- cdb3.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -fprebuilt-module-path=.././module -IDIR/include/ -o DIR/tu.o ",
  "file": "DIR/tu.c"
}]

//--- cdb4.json.in
[{
  "directory": "DIR/a/",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -fprebuilt-module-path=.././module -IDIR/include/ -o DIR/tu.o ",
  "file": "DIR/tu.c"
}]

//--- cdb5.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -IDIR/include/ -Xclang -working-directory=DIR/a/ -o DIR/tu.o",
  "file": "DIR/tu.c"
}]

//--- include/module.modulemap
module mod {
  header "mod.h"
}

//--- include/mod.h

//--- tu.c
#include "mod.h"

// Check that result0 and result1/result5 compute the same hash with
// optimization on. The only difference between result0 and result1/result5 is
// the compiler's working directory.
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

// Check that result3 and result4 contain different hashes because
// both have a same relative path as a command line input, and
// they are produced using different compiler working directories.
// RELPATH:      {
// RELPATH-NEXT:   "modules": [
// RELPATH-NEXT:    {
// RELPATH-NEXT:      "clang-module-deps": [],
// RELPATH:           "context-hash": "[[HASH3:.*]]",
// RELPATH:         }
// RELPATH:        "translation-units": [
// RELPATH:         {
// RELPATH:            "commands": [
// RELPATH:             {
// RELPATH-NEXT:          "clang-context-hash": "{{.*}}",
// RELPATH-NEXT:          "clang-module-deps": [
// RELPATH-NEXT:            {
// RELPATH-NEXT:              "context-hash": "[[HASH3]]",
// RELPATH-NEXT:              "module-name": "mod"
// RELPATH:            }
// RELPATH:          ],
// RELPATH:      {
// RELPATH-NEXT:   "modules": [
// RELPATH-NEXT:     {
// RELPATH-NEXT:       "clang-module-deps": [],
// RELPATH-NOT:        "context-hash": "[[HASH3]]",
// RELPATH:            "context-hash": "[[HASH4:.*]]",
// RELPATH:          }
// RELPATH:       "translation-units": [
// RELPATH:         {
// RELPATH:           "commands": [
// RELPATH:             {
// RELPATH-NEXT:          "clang-context-hash": "{{.*}}",
// RELPATH-NEXT:          "clang-module-deps": [
// RELPATH-NEXT:            {
// RELPATH-NOT:              "context-hash": "[[HASH3]]",
// RELPATH-NEXT:             "context-hash": "[[HASH4]]"
// RELPATH-NEXT:              "module-name": "mod"
// RELPATH:            }
// RELPATH:          ],

