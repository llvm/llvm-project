// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-format=obj -debug-info-kind=limited -dwarf-ext-refs \
// RUN:     -fimplicit-module-maps -x c -fmodules-cache-path=%t -I %S/Inputs \
// RUN:     %s -mllvm -debug-only=pchcontainer -emit-llvm -o %t.ll \
// RUN:     2>&1 | FileCheck %s
// REQUIRES: asserts
#include "DebugSubmoduleA.h"

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "A",
// CHECK-SAME:             scope: ![[SUBMODULEA:[0-9]+]]
// CHECK: ![[SUBMODULEA]] = !DIModule(scope: ![[PARENT:[0-9]+]],
// CHECK-SAME:                        name: "DebugSubmoduleA",
// CHECK: ![[PARENT]] = !DIModule(scope: null, name: "DebugSubmodules"

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "B",
// CHECK-SAME:             scope: ![[SUBMODULEB:[0-9]+]]
// CHECK: ![[SUBMODULEB]] = !DIModule(scope: ![[PARENT]],
// CHECK-SAME:                        name: "DebugSubmoduleB",
