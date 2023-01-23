// Ensure the working directory is correctly captured in cas-fs when compiling
// with caching from outside the source directory.
// FIXME: ideally we could further canonicalize the working directory when it
// is irrelevant to the compilation, but for now ensure we can compile at all.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: mkdir -p %t/B

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -action-cache-path %t/cache -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/tu.rsp

// Check specifics of the command-line
// RUN: cat %t/deps.json | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK:              "-fcas-fs-working-directory"
// CHECK-NEXT:         "[[PREFIX]]/B"
// CHECK:            ]
// CHECK:            "name": "Mod"
// CHECK:          }
// CHECK-NEXT:   ]
// CHECK:        "translation-units": [
// CHECK:          {
// CHECK:            "commands": [
// CHECK:              {
// CHECK:                "command-line": [
// CHECK:                  "-fcas-fs-working-directory"
// CHECK-NEXT:             "[[PREFIX]]/B"
// CHECK:                ]

//--- cdb.json.template
[{
  "directory" : "DIR/B",
  "command" : "clang_tool -fsyntax-only DIR/A/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/A/module-cache -Rcompile-job-cache",
  "file" : "DIR/A/tu.c"
}]

//--- A/module.modulemap
module Mod { header "Mod.h" }

//--- A/Mod.h
#pragma once
void Top(void);

//--- A/tu.c
#include "Mod.h"
