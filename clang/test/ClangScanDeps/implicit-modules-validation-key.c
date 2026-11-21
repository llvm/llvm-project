// Check that -fimplicit-modules-validation-key invalidates the scanner's
// implicitly built modules, but isn't passed to explicit module or translation
// unit commands, and doesn't change their context hashes.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" -e "s|KEY|a|g" %t/cdb.json.template > %t/cdb-a.json
// RUN: sed -e "s|DIR|%/t|g" -e "s|KEY|b|g" %t/cdb.json.template > %t/cdb-b.json

// RUN: clang-scan-deps -compilation-database %t/cdb-a.json -j 1 \
// RUN:   -format experimental-full > %t/a.json
// RUN: find %t/cache -name "Mod-*.pcm" -exec cp "{}" %t/mod-a.pcm ";"
// RUN: clang-scan-deps -compilation-database %t/cdb-b.json -j 1 \
// RUN:   -format experimental-full > %t/b.json
// RUN: find %t/cache -name "Mod-*.pcm" -exec cp "{}" %t/mod-b.pcm ";"
// RUN: FileCheck %s < %t/a.json
// CHECK-NOT: -fimplicit-modules-validation-key
// CHECK: "name": "Mod"
// CHECK-NOT: -fimplicit-modules-validation-key

// The scanner's own module was rebuilt with the new key.
// RUN: not diff %t/mod-a.pcm %t/mod-b.pcm

// RUN: grep '"context-hash"' %t/a.json > %t/a.hash
// RUN: grep '"context-hash"' %t/b.json > %t/b.hash
// RUN: diff %t/a.hash %t/b.hash

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -I DIR/include -fimplicit-modules-validation-key=KEY",
  "file": "DIR/tu.c"
}]

//--- include/module.modulemap
module Mod { header "mod.h" }

//--- include/mod.h
int mod;

//--- tu.c
#include "mod.h"
