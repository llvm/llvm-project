// REQUIRES: asserts

// RUN: %clang_cc1 -std=c++23 -x c++-header -emit-pch -fmodule-format=obj -I %S/Inputs \
// RUN:   -o %t.pch %S/Inputs/gmodules-deduction-guide.h \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-pch.ll
// RUN: cat %t-pch.ll | FileCheck %s

// CHECK: ![[V0:.*]] = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<A>",
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "Type0",{{.*}}, baseType: ![[V0]])
