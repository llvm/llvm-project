// Check that repeated inclusion of a modular header doesn't get translated
// into an import in textual compilation.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -E %t/tu.c -o %t/tu.ii -I %t -fmodule-map-file=%t/module.modulemap

// RUN: FileCheck --input-file=%t/tu.ii %s -DPREFIX=%t

// CHECK:      # 1 "[[PREFIX]]{{/|\\}}tu.c"
// CHECK-NEXT: # 1 "<built-in>" 1
// CHECK-NEXT: # 1 "<built-in>" 3
// CHECK-NEXT: # {{[0-9]+}} "<built-in>" 3
// CHECK-NEXT: # 1 "<command line>" 1
// CHECK-NEXT: # 1 "<built-in>" 2
// CHECK-NEXT: # 1 "[[PREFIX]]{{/|\\}}tu.c" 2
// CHECK-NEXT: # 1 "[[PREFIX]]{{/|\\}}Mod.h" 1
// CHECK-NEXT: #pragma clang module begin Mod
// CHECK-NEXT: # 2 "[[PREFIX]]{{/|\\}}tu.c" 2
// CHECK-NEXT: # 1 "[[PREFIX]]{{/|\\}}tu.c"
// CHECK-NEXT: #pragma clang module end /*Mod*/
// CHECK-NOT:  #pragma clang module import

//--- module.modulemap
module Mod { header "Mod.h" }
//--- Mod.h
#pragma once
//--- tu.c
#include "Mod.h"
#include "Mod.h"
