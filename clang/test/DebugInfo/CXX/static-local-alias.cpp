// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=constructor -std=c++23 %s -o - | FileCheck %s

// Aliases to static locals are emitted at file scope, outside any lexical block.

__attribute__((nodebug)) void f() { static int s; }
extern int a __attribute__((alias("_ZZ1fvE1s")));
// CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "a", {{.*}}entity: ![[A:[0-9]+]]
// CHECK-DAG: ![[A]] = !DIGlobalVariable(name: "s", scope: ![[CU:[0-9]+]], {{.*}}isDefinition: false)
// CHECK-DAG: ![[CU]] = distinct !DICompileUnit(

void g() { __attribute__((nodebug)) static int s; }
extern int b __attribute__((alias("_ZZ1gvE1s")));
// CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "b", {{.*}}entity: ![[B:[0-9]+]]
// CHECK-DAG: ![[B]] = !DIGlobalVariable(name: "s", scope: ![[G:[0-9]+]], {{.*}}isDefinition: false)
// CHECK-DAG: ![[G]] = distinct !DISubprogram(name: "g"

// The function is emitted after the alias.
constexpr const int *h() { static constexpr int s = 1; return &s; }
const int *p = h();
extern const int c __attribute__((alias("_ZZ1hvE1s")));
// CHECK-DAG: !DIImportedEntity(tag: DW_TAG_imported_declaration, name: "c", {{.*}}entity: ![[C:[0-9]+]]
// CHECK-DAG: ![[C]] = distinct !DIGlobalVariable(name: "s", scope: ![[H:[0-9]+]], {{.*}}isDefinition: true)
// CHECK-DAG: ![[H]] = distinct !DISubprogram(name: "h"
