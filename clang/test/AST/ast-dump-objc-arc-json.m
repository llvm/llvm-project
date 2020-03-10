// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -x objective-c -fobjc-arc -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

typedef struct {
  id f;
} S;

id TestCompoundLiteral(id a) {
  return ((S){ .f = a }).f;
}

// CHECK:  "kind": "ExprWithCleanups",
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "offset": 202,
// CHECK-NEXT:    "col": 10,
// CHECK-NEXT:    "tokLen": 1
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "offset": 218,
// CHECK-NEXT:    "col": 26,
// CHECK-NEXT:    "tokLen": 1
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "desugaredQualType": "id",
// CHECK-NEXT:   "qualType": "id",
// CHECK-NEXT:   "typeAliasDeclId": "0x{{.*}}"
// CHECK-NEXT:  },
// CHECK-NEXT:  "valueCategory": "rvalue",
// CHECK-NEXT:  "cleanupsHaveSideEffects": true,
// CHECK-NEXT:  "cleanups": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CompoundLiteralExpr"
// CHECK-NEXT:   }
// CHECK-NEXT:  ],
