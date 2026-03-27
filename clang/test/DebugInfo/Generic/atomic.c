// RUN: %clang -g -c -std=c11 -S -emit-llvm -o - %s | FileCheck %s

// CHECK: !DIGlobalVariable(name: "i"
// CHECK-SAME: type: ![[T:.*]], isLocal: false, isDefinition: true)
// CHECK: ![[T]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[BT:.*]])
// CHECK: ![[BT]] = !DIDerivedType(tag: DW_TAG_atomic_type, baseType: ![[BTT:.*]])
// CHECK: ![[BTT]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
_Atomic const int i;
