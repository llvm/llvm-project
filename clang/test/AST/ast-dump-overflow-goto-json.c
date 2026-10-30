// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fexperimental-overflow-behavior-types -ast-dump=json -ast-dump-filter=test %s | FileCheck %s

int test(int __ob_trap a) {
  return __builtin_overflow_goto(a + 1, l);
l:
  return 0;
}

// CHECK:      "kind": "OverflowGotoExpr",
// CHECK-NEXT: "range": {
// CHECK-NEXT:  "begin": {
// CHECK-NEXT:   "offset": {{[0-9]+}},
// CHECK-NEXT:   "col": 10,
// CHECK-NEXT:   "tokLen": 23
// CHECK-NEXT:  },
// CHECK-NEXT:  "end": {
// CHECK-NEXT:   "offset": {{[0-9]+}},
// CHECK-NEXT:   "col": 42,
// CHECK-NEXT:   "tokLen": 1
// CHECK-NEXT:  }
// CHECK-NEXT: },
// CHECK-NEXT: "type": {
// CHECK-NEXT:  "qualType": "__ob_trap int"
// CHECK-NEXT: },
// CHECK-NEXT: "valueCategory": "prvalue",
// CHECK-NEXT: "name": "l",
// CHECK-NEXT: "labelDeclId": "[[LABEL:0x[0-9a-f]+]]",
// CHECK-NEXT: "inner": [
// CHECK-NEXT:  {
// CHECK-NEXT:   "id": "0x{{.*}}",
// CHECK-NEXT:   "kind": "BinaryOperator",
// CHECK:      "kind": "LabelStmt",
// CHECK:      "name": "l",
// CHECK-NEXT: "declId": "[[LABEL]]",
