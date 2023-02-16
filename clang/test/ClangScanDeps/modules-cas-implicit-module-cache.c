// Check that the implicit modules build that the scanner performs detects
// missing cas objects.

// REQUIRES: ondisk_cas

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// Scan to populate module cache

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives

// Clear cas and re-scan

// RUN: rm -rf %t/cas

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -cas-path %t/cas -module-files-dir %t/outputs \
// RUN:   -format experimental-full -mode preprocess-dependency-directives \
// RUN:   > %t/deps.json

// Build module and TU

// RUN: %deps-to-rsp %t/deps.json --module-name Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/tu.rsp

//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang_tool -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache",
  "file" : "DIR/tu.c"
}]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h
void mod(void);

//--- tu.c
#include "Mod.h"
void tu(void) {
  mod();
}
