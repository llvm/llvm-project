// Check that Wsystem-headers-in-module shows diagnostics in the named system
// module, but not in other system headers or modules.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/mcp \
// RUN:   -isystem %t/sys %t/tu.c -fsyntax-only -Wextra-semi -Wsystem-headers-in-module=sys_mod \
// RUN:   2>&1 | FileCheck %s

// CHECK-NOT: warning:
// CHECK: sys_mod.h:2:7: warning: extra ';'
// CHECK-NOT: warning:

//--- sys/other_sys_header.h
int x;;
//--- sys_mod.h
#include "dependent_sys_mod.h"
int y;;
//--- other_sys_mod.h
int z;;
//--- dependent_sys_mod.h
int w;;
//--- module.modulemap
module sys_mod [system] { header "sys_mod.h" }
module other_sys_mod [system] { header "other_sys_mod.h" }
module dependent_sys_mod [system] { header "dependent_sys_mod.h" }

//--- tu.c
#include "sys_mod.h"
#include "other_sys_mod.h"
#include "other_sys_header.h"
