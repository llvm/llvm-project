// RUN: clang-pseudo -grammar=cxx -source=%s --start-symbol=statement-seq --print-forest | FileCheck %s

// Verify the else should belong to the nested if statement
if (true) if (true) {} else {}

// CHECK:      statement-seq~selection-statement := IF ( condition ) statement
// CHECK-NEXT: ├─IF
// CHECK-NEXT: ├─(
// CHECK-NEXT: ├─condition~TRUE
// CHECK-NEXT: ├─)
// CHECK-NEXT: └─statement~selection-statement
// CHECK-NEXT:   ├─IF
// CHECK-NEXT:   ├─(
// CHECK-NEXT:   ├─condition~TRUE
// CHECK-NEXT:   ├─)
// CHECK-NEXT:   ├─statement~compound-statement := { }
// CHECK-NEXT:   │ ├─{
// CHECK-NEXT:   │ └─}
// CHECK-NEXT:   ├─ELSE
// CHECK-NEXT:   └─statement~compound-statement := { }
// CHECK-NEXT:     ├─{
// CHECK-NEXT:     └─}
