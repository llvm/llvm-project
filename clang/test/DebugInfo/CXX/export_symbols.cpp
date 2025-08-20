// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s

// CHECK: [[SCOPE:![0-9]+]] = distinct !DICompositeType({{.*}}flags: DIFlagTypePassByValue
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, scope: [[SCOPE]]
// CHECK-SAME:                              DIFlagExportSymbols | DIFlagTypePassByValue
struct A {
 // Anonymous class exports its symbols into A
 struct {
     int y;
 };
} a;
