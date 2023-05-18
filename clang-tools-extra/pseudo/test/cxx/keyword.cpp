// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
bool operator<();
// CHECK:      translation-unit~simple-declaration := decl-specifier-seq init-declarator-list ;
// CHECK-NEXT: ├─decl-specifier-seq~BOOL
// CHECK-NEXT: ├─init-declarator-list~noptr-declarator := noptr-declarator parameters-and-qualifiers
// CHECK-NEXT: │ ├─noptr-declarator~operator-function-id := OPERATOR operator-name
// CHECK-NEXT: │ │ ├─OPERATOR
// CHECK-NEXT: │ │ └─operator-name~<
// CHECK-NEXT: │ └─parameters-and-qualifiers := ( )
// CHECK-NEXT: │   ├─(
// CHECK-NEXT: │   └─)
// CHECK-NEXT: └─;
