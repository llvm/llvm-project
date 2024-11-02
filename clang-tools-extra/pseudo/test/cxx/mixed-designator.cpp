// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
// FIXME: tighten CHECK to CHECK-NEXT once numeric literals are unambiguous.
auto x = { 1, .f = 2, [c]{3} };
// CHECK:      initializer-clause~braced-init-list
// CHECK-NEXT: ├─{ := tok[3]
// CHECK-NEXT: ├─initializer-list
// CHECK-NEXT: │ ├─initializer-list
// CHECK-NEXT: │ │ ├─initializer-list~NUMERIC_CONSTANT
// CHECK-NEXT: │ │ ├─, := tok[5]
// CHECK-NEXT: │ │ └─initializer-list-item
// CHECK-NEXT: │ │   ├─designator
// CHECK-NEXT: │ │   │ ├─. := tok[6]
// CHECK-NEXT: │ │   │ └─IDENTIFIER := tok[7]
// CHECK-NEXT: │ │   └─brace-or-equal-initializer
// CHECK-NEXT: │ │     ├─= := tok[8]
// CHECK-NEXT: │ │     └─initializer-clause~NUMERIC_CONSTANT
// CHECK-NEXT: │ ├─, := tok[10]
// CHECK-NEXT: │ └─initializer-list-item
// CHECK-NEXT: │   ├─designator
// CHECK-NEXT: │   │ ├─[ := tok[11]
// CHECK-NEXT: │   │ ├─expression~IDENTIFIER := tok[12]
// CHECK-NEXT: │   │ └─] := tok[13]
// CHECK-NEXT: │   └─brace-or-equal-initializer~braced-init-list
// CHECK-NEXT: │     ├─{ := tok[14]
// CHECK-NEXT: │     ├─initializer-list~NUMERIC_CONSTANT
// CHECK:      │     └─} := tok[16]
// CHECK-NEXT: └─} := tok[17]
