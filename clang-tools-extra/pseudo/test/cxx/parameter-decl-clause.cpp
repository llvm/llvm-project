// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
void foo2(int, ...);
// CHECK:      translation-unit~simple-declaration := decl-specifier-seq init-declarator-list ;
// CHECK-NEXT: ├─decl-specifier-seq~VOID :=
// CHECK-NEXT: ├─init-declarator-list~noptr-declarator := noptr-declarator parameters-and-qualifiers
// CHECK-NEXT: │ ├─noptr-declarator~IDENTIFIER :=
// CHECK-NEXT: │ └─parameters-and-qualifiers := ( parameter-declaration-clause [recover=Brackets] )
// CHECK-NEXT: │   ├─( :=
// CHECK-NEXT: │   ├─parameter-declaration-clause := parameter-declaration-list , ...
// CHECK-NEXT: │   │ ├─parameter-declaration-list~INT :=
// CHECK-NEXT: │   │ ├─, :=
// CHECK-NEXT: │   │ └─... :=
// CHECK-NEXT: │   └─) :=
// CHECK-NEXT: └─; :=
