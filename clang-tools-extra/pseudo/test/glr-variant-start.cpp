// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --start-symbol=statement-seq --print-forest | FileCheck %s

a + a;
// CHECK:      statement-seq~expression-statement := expression ;
// CHECK-NEXT: ├─expression~additive-expression := additive-expression + multiplicative-expression
// CHECK-NEXT: │ ├─additive-expression~IDENTIFIER :=
// CHECK-NEXT: │ ├─+ :=
// CHECK-NEXT: │ └─multiplicative-expression~IDENTIFIER :=
// CHECK-NEXT: └─; :=
