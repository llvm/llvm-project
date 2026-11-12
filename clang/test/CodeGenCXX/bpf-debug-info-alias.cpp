// RUN: %clang_cc1 -triple bpfel -emit-llvm -debug-info-kind=constructor %s -o - | FileCheck %s

// CHECK: @_Z9__cat_op0v = alias void (), ptr @e
// CHECK: @alias_var = alias i32, ptr @global_var
// CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "__cat_op0", {{.*}} entity: ![[ENTITY:[0-9]+]]
// CHECK-DAG: ![[ENTITY]] = {{.*}}!DISubprogram(name: "__cat_op0", linkageName: "_Z9__cat_op0v"
// CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "alias_var", {{.*}} entity: ![[ENTITY2:[0-9]+]]
// CHECK-DAG: ![[ENTITY2]] = {{.*}}!DIGlobalVariable(name: "global_var", {{.*}}

extern "C" void e() {}
void __attribute__((alias("e"))) __cat_op0();
void r() { __cat_op0(); }

int global_var;
extern "C" int alias_var __attribute__((alias("global_var")));
int use_alias() { return alias_var; }
