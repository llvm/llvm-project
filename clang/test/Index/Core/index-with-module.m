// RUN: rm -rf %t.mcp
// RUN: c-index-test core -print-source-symbols -dump-imported-module-files -- %s -I %S/Inputs/module -fmodules -fmodules-cache-path=%t.mcp | FileCheck %s

// CHECK: [[@LINE+1]]:9 | module/C | ModA | Decl |
@import ModA;
// CHECK: [[@LINE+1]]:1 | module/C | ModA | Decl,Impl |
#include "ModA.h"

void foo() {
  // CHECK: [[@LINE+1]]:3 | function/C | ModA_func | c:@F@ModA_func | {{.*}} | Ref,Call,RelCall,RelCont | rel: 1
  ModA_func();
}

// CHECK: ==== Module ModA ====
// CHECK: 2:6 | function/C | ModA_func | c:@F@ModA_func | {{.*}} | Decl | rel: 0
// CHECK: ---- Module Inputs ----
// CHECK: user | {{.*}}ModA.h
// CHECK: user | {{.*}}module.modulemap
