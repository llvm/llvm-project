// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s
// We loosely allow capture defaults in any position/multiple times.
auto lambda = [&, &foo, bar(x), =]{};
// CHECK:      lambda-introducer := [ capture-list ]
// CHECK-NEXT: ├─[
// CHECK-NEXT: ├─capture-list
// CHECK-NEXT: │ ├─capture-list
// CHECK-NEXT: │ │ ├─capture-list
// CHECK-NEXT: │ │ │ ├─capture-list~& := tok[4]
// CHECK-NEXT: │ │ │ ├─,
// CHECK-NEXT: │ │ │ └─capture~simple-capture
// CHECK-NEXT: │ │ │   ├─&
// CHECK-NEXT: │ │ │   └─IDENTIFIER := tok[7]
// CHECK-NEXT: │ │ ├─,
// CHECK-NEXT: │ │ └─capture~init-capture
// CHECK-NEXT: │ │   ├─IDENTIFIER := tok[9]
// CHECK-NEXT: │ │   └─initializer := ( expression-list )
// CHECK-NEXT: │ │     ├─(
// CHECK-NEXT: │ │     ├─expression-list~IDENTIFIER := tok[11]
// CHECK-NEXT: │ │     └─)
// CHECK-NEXT: │ ├─,
// CHECK-NEXT: │ └─capture~=
// CHECK-NEXT: └─]
