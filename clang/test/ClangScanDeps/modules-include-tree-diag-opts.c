// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full -cas-path %t/cas > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name Mod > %t/Mod.rsp

// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/Mod.rsp -fmessage-length=8
// RUN: %clang @%t/Mod.rsp -fcolor-diagnostics
// RUN: %clang @%t/Mod.rsp -ferror-limit 1

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -Xclang -fcache-disable-replay -fsyntax-only DIR/tu.c -fmodules -fimplicit-modules -fimplicit-module-maps -fmodules-cache-path=DIR/module-cache"
}]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu.c
#include "Mod.h"
