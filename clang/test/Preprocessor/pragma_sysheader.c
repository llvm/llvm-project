// RUN: %clang_cc1 -verify -pedantic %s -fsyntax-only
// RUN: %clang_cc1 -E %s | FileCheck %s
// expected-no-diagnostics
// rdar://6899937
#include "pragma_sysheader.h"


// PR9861: Verify that line markers are not messed up in -E mode.
// CHECK: # 1 "{{.*}}pragma_sysheader.h" 1
// CHECK-NEXT: # 1 "{{.*}}pragma_sysheader.h" 3
// CHECK-NEXT: typedef int x;
// CHECK-NEXT: typedef int x;
// CHECK-NEXT: # 6 "{{.*}}pragma_sysheader.c" 2
