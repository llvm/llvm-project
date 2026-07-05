// RUN: clang-check -syntax-tree-dump "%s" -- 2>&1 | FileCheck %s
int abc;
// CHECK:      TranslationUnit Detached
// CHECK-NEXT: `-SimpleDeclaration
// CHECK-NEXT:   |-'int'
// CHECK-NEXT:   |-DeclaratorList Declarators
// CHECK-NEXT:   | `-SimpleDeclarator ListElement
// CHECK-NEXT:   |   `-'abc'
// CHECK-NEXT:   `-';'
