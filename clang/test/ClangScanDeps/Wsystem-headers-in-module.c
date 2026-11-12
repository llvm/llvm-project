// Check that Wsystem-headers-in-module shows diagnostics in the named system
// module, but not in other system headers or modules when built with explicit
// modules.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/deps.json

// RUN: %deps-to-rsp %t/deps.json --module-name=dependent_sys_mod > %t/dependent_sys_mod.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=sys_mod > %t/sys_mod.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=other_sys_mod > %t/other_sys_mod.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// RUN: %clang @%t/dependent_sys_mod.rsp -verify
// RUN: %clang @%t/sys_mod.rsp -verify
// RUN: %clang @%t/other_sys_mod.rsp -verify
// RUN: %clang @%t/tu.rsp -verify

// CHECK-NOT: warning:
// CHECK: sys_mod.h:2:7: warning: extra ';'
// CHECK-NOT: warning:

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/mcp DIR/tu.c -isystem DIR/sys -Wextra-semi -Wsystem-headers-in-module=sys_mod",
  "file": "DIR/tu.c"
}]

//--- sys/other_sys_header.h
int x;;

//--- sys_mod.h
#include "dependent_sys_mod.h"
int y;; // expected-warning {{extra ';' outside of a function}}

//--- other_sys_mod.h
int z;;
// expected-no-diagnostics

//--- dependent_sys_mod.h
int w;;
// expected-no-diagnostics

//--- module.modulemap
module sys_mod [system] { header "sys_mod.h" }
module other_sys_mod [system] { header "other_sys_mod.h" }
module dependent_sys_mod [system] { header "dependent_sys_mod.h" }

//--- tu.c
#include "sys_mod.h"
#include "other_sys_mod.h"
#include "other_sys_header.h"
// expected-no-diagnostics
