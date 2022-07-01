// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s
auto x = { complete garbage };
// CHECK:      translation-unit~simple-declaration
// CHECK-NEXT: ├─decl-specifier-seq~AUTO := tok[0]
// CHECK-NEXT: ├─init-declarator-list~init-declarator
// CHECK-NEXT: │ ├─declarator~IDENTIFIER := tok[1]
// CHECK-NEXT: │ └─initializer~brace-or-equal-initializer
// CHECK-NEXT: │   ├─= := tok[2]
// CHECK-NEXT: │   └─initializer-clause~braced-init-list
// CHECK-NEXT: │     ├─{ := tok[3]
// CHECK-NEXT: │     ├─initializer-list := <opaque>
// CHECK-NEXT: │     └─} := tok[6]
// CHECK-NEXT: └─; := tok[7]
