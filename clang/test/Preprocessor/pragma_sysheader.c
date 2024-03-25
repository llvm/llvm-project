// RUN: %clang_cc1 -verify -std=c99 -Wunknown-pragmas -pedantic %s -fsyntax-only
// RUN: %clang_cc1 -verify -std=c99 -Wunknown-pragmas -pedantic %s -fsyntax-only -DGCC
// RUN: %clang_cc1 -verify -std=c99 -Wunknown-pragmas -pedantic %s -fsyntax-only -DCLANG
// RUN: %clang_cc1 -verify -std=c99 -Wunknown-pragmas -pedantic %s -fsyntax-only -fms-extensions -DMS

#include "Inputs/pragma_sysheader.h"

// RUN: %clang_cc1 -E %s | FileCheck %s
// PR9861: Verify that line markers are not messed up in -E mode.
// CHECK: # 1 "{{.*}}pragma_sysheader.h" 1
// CHECK-NEXT: # 12 "{{.*}}pragma_sysheader.h"
// CHECK: typedef int x;
// CHECK: typedef int x;
// CHECK-NEXT: # 7 "{{.*}}pragma_sysheader.c" 2
