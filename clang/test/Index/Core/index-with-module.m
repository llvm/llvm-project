// XFAIL: target={{.*}}-aix{{.*}}
// RUN: rm -rf %t.mcp
// RUN: c-index-test core -print-source-symbols -dump-imported-module-files -- %s -I %S/Inputs/module -fmodules -fmodules-cache-path=%t.mcp | FileCheck %s

// CHECK: [[@LINE+1]]:9 | module/C | ModA | [[ModA_USR:c:@M@ModA]] | Decl |
@import ModA;
// CHECK: [[@LINE+1]]:1 | module/C | ModA | [[ModA_USR]] | Decl,Impl |
#include "ModA.h"

@import ModA.SubModA.SubSubModA;
// CHECK: [[@LINE-1]]:9 | module/C | ModA | [[ModA_USR]] | Ref |
// CHECK: [[@LINE-2]]:14 | module/C | ModA.SubModA | c:@M@ModA@M@SubModA | Ref |
// CHECK: [[@LINE-3]]:22 | module/C | ModA.SubModA.SubSubModA | [[SubSubModA_USR:c:@M@ModA@M@SubModA@M@SubSubModA]] | Decl |
#include "SubSubModA.h" // CHECK: [[@LINE]]:1 | module/C | ModA.SubModA.SubSubModA | [[SubSubModA_USR]] | Decl,Impl |

void foo() {
  // CHECK: [[@LINE+1]]:3 | function/C | ModA_func | c:@F@ModA_func | {{.*}} | Ref,Call,RelCall,RelCont | rel: 1
  ModA_func();
}

// CHECK: ==== Module ModA ====
// CHECK-DAG: 3:9 | macro/C | MODA_MACRO | c:{{.*}}@macro@MODA_MACRO | Def |
// CHECK-DAG: 3:9 | macro/C | SUBMODA_MACRO | c:{{.*}}@macro@SUBMODA_MACRO | Def |
// CHECK-DAG: 2:6 | function/C | ModA_func | c:@F@ModA_func | {{.*}} | Decl | rel: 0
// CHECK-DAG: 2:6 | function/C | SubModA_func | c:@F@SubModA_func | {{.*}} | Decl | rel: 0
// CHECK: ---- Module Inputs ----
// CHECK: user | {{.*}}module.modulemap
// CHECK: user | {{.*}}ModA.h
