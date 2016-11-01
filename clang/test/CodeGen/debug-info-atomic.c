// RUN: %clang -g -c -std=c11 -S -emit-llvm -o - %s | FileCheck %s

// CHECK: !DIGlobalVariable(name: "i"{{.*}}type: !5, isLocal: false, isDefinition: true)
// CHECK: !5 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
// CHECK: !6 = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: !7)
// CHECK: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
_Atomic const int i;
