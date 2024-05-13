// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
void s(int[]);
// CHECK:      parameter-declaration-clause~parameter-declaration := decl-specifier-seq abstract-declarator
// CHECK-NEXT: ├─decl-specifier-seq~INT := tok[3]
// CHECK-NEXT: └─abstract-declarator~noptr-abstract-declarator := [ ]
// CHECK-NEXT:   ├─[ := tok[4]
// CHECK-NEXT:   └─] := tok[5]
