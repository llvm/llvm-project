// Test that (the same) debug info is emitted for an Objective-C++
// module and a C++ precompiled header.

// REQUIRES: asserts, shell

// Modules:
// RUN: rm -rf %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -x objective-c++ -std=c++11 -debug-info-kind=limited -fmodules -fmodule-format=obj -fimplicit-module-maps -DMODULES -fmodules-cache-path=%t %s -I %S/Inputs -I %t -emit-llvm -o %t.ll -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s
// RUN: cat %t-mod.ll | FileCheck --check-prefix=CHECK-NEG %s
// RUN: cat %t-mod.ll | FileCheck --check-prefix=CHECK-DWO %s

// PCH:
// RUN: %clang_cc1 -triple %itanium_abi_triple -x c++ -std=c++11 -emit-pch -fmodule-format=obj -I %S/Inputs -o %t.pch %S/Inputs/DebugCXX.h -mllvm -debug-only=pchcontainer &>%t-pch.ll
// RUN: cat %t-pch.ll | FileCheck %s
// RUN: cat %t-mod.ll | FileCheck --check-prefix=CHECK-NEG %s

#ifdef MODULES
@import DebugCXX;
#endif

// CHECK: distinct !DICompileUnit(language: DW_LANG_{{.*}}C_plus_plus,
// CHECK-SAME:                    isOptimized: false,
// CHECK-SAME-NOT:                splitDebugFilename:
// CHECK-DWO:                     dwoId:
// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Enum"
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX4EnumE")
// CHECK: !DINamespace(name: "DebugCXX"
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Struct"
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX6StructE")
// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<int, DebugCXX::traits<int> >"
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIiNS_6traitsIiEEEE")
// CHECK: !DICompositeType(tag: DW_TAG_class_type,
// CHECK-SAME:             name: "Template<float, DebugCXX::traits<float> >"
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX8TemplateIfNS_6traitsIfEEEE")
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "A<void>"
// CHECK-SAME:             identifier: "_ZTSN8DebugCXX1AIJvEEE")
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "FloatInstatiation"
// no mangled name here yet.
// CHECK: !DIDerivedType(tag: DW_TAG_typedef, name: "B",
// no mangled name here yet.

// CHECK-NEG-NOT: "_ZTSN8DebugCXX8TemplateIlNS_6traitsIlEEEE"
