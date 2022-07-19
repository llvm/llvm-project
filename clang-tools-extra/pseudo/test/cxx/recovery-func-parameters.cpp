// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
void foo(complete garbage???) {}
// CHECK:      translation-unit~function-definition := decl-specifier-seq function-declarator function-body
// CHECK-NEXT: ├─decl-specifier-seq~VOID := tok[0]
// CHECK-NEXT: ├─function-declarator~noptr-declarator := noptr-declarator parameters-and-qualifiers
// CHECK-NEXT: │ ├─noptr-declarator~IDENTIFIER := tok[1]
// CHECK-NEXT: │ └─parameters-and-qualifiers := ( parameter-declaration-clause [recover=Brackets] )
// CHECK-NEXT: │   ├─( := tok[2]
// CHECK-NEXT: │   ├─parameter-declaration-clause := <opaque>
// CHECK-NEXT: │   └─) := tok[8]
// CHECK-NEXT: └─function-body~compound-statement := { }
// CHECK-NEXT:   ├─{ := tok[9]
// CHECK-NEXT:   └─} := tok[10]
