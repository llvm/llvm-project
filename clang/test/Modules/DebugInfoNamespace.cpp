// UNSUPPORTED: -zos, -aix
// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -std=c++11 -debug-info-kind=standalone \
// RUN:     -dwarf-ext-refs -fmodules \
// RUN:     -fmodule-format=obj -fimplicit-module-maps \
// RUN:     -triple %itanium_abi_triple -fmodules-cache-path=%t \
// RUN:     %s -I %S/Inputs/DebugInfoNamespace -I %t -emit-llvm -o - \
// RUN:     |  FileCheck %s

#include "A.h"
#include "B.h"
using namespace N;
B b;

// Verify that the forward decl of B is in module B.
//
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "B",
// CHECK-SAME:             scope: ![[N:[0-9]+]]
// CHECK: ![[N]] = !DINamespace(name: "N", scope: ![[B:[0-9]+]])
// CHECK: ![[B]] = !DIModule(scope: null, name: "B",
