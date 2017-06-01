// RUN: rm -rf %t

// RUN: %clang_cc1 -x objective-c++ -std=c++11 -debug-info-kind=standalone \
// RUN:     -dwarf-ext-refs -fmodules                                   \
// RUN:     -fmodule-format=obj -fimplicit-module-maps -DMODULES \
// RUN:     -triple %itanium_abi_triple \
// RUN:     -fmodules-cache-path=%t %s -I %S/Inputs -I %t -emit-llvm -o - \
// RUN:   | FileCheck %s

#include "DebugNestedB.h"
AF af; // This type is not anchored in the module.

// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "AF",
// CHECK-SAME:           baseType: ![[AF:.*]])

// CHECK: ![[AF]] = {{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "A<F>",
// CHECK-SAME:                             elements:

