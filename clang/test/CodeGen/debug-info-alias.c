// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -target x86_64-unknown-linux-gnu -o - | FileCheck %s

// CHECK-DAG: [[ENTITY1:![0-9]+]] = distinct !DIGlobalVariable(name: "aliased_global"
// CHECK-DAG: [[ENTITY2:![0-9]+]] = distinct !DIGlobalVariable(name: "aliased_global_2"
// CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "__global_alias", scope: !2, entity: [[ENTITY1]]
// CHECK-DAG: [[ENTITY3:![0-9]+]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "global_alias_2", scope: !2, entity: [[ENTITY2]]
// CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "__global_alias_2_alias", scope: !2, entity: [[ENTITY3]]

int aliased_global = 1;
extern int __attribute__((alias("aliased_global"))) __global_alias;

// Recursive alias:
int aliased_global_2 = 2;
extern int __attribute__((alias("aliased_global_2"))) global_alias_2;
extern int __attribute__((alias("global_alias_2"))) __global_alias_2_alias;
