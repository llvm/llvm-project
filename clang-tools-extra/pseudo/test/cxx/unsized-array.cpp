// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s
void s(int[]);
// CHECK:      parameter-declaration-list~parameter-declaration := decl-specifier-seq abstract-declarator
// CHECK-NEXT: ├─decl-specifier-seq~INT := tok[3]
// CHECK-NEXT: └─abstract-declarator~noptr-abstract-declarator := [ ]
// CHECK-NEXT:   ├─[ := tok[4]
// CHECK-NEXT:   └─] := tok[5]
