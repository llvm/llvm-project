// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -ast-dump=json %s | FileCheck --check-prefix JSON %s

struct A {
  int arr[1];
};

struct B {
  const A &a = A{{0}};
};

void test() {
  B b{};
}
// CHECK: -CXXDefaultInitExpr 0x{{[^ ]*}} <{{.*}}> 'const A' lvalue has rewritten init
// CHECK-NEXT:  `-ExprWithCleanups 0x{{[^ ]*}} <{{.*}}> 'const A' lvalue
// CHECK-NEXT:    `-MaterializeTemporaryExpr 0x{{[^ ]*}} <{{.*}}> 'const A' lvalue extended by Var 0x{{[^ ]*}} 'b' 'B'
// CHECK-NEXT:      `-ImplicitCastExpr 0x{{[^ ]*}} <{{.*}}> 'const A' <NoOp>
// CHECK-NEXT:        `-CXXFunctionalCastExpr 0x{{[^ ]*}} <{{.*}}> 'A' functional cast to A <NoOp>
// CHECK-NEXT:          `-InitListExpr 0x{{[^ ]*}} <{{.*}}> 'A'
// CHECK-NEXT:            `-InitListExpr 0x{{[^ ]*}} <{{.*}}> 'int[1]'
// CHECK-NEXT:              `-IntegerLiteral 0x{{[^ ]*}} <{{.*}}> 'int' 0

// JSON:       "kind": "CXXDefaultInitExpr",
// JSON:       "type": {
// JSON-NEXT:   "qualType": "const A"
// JSON-NEXT:  },
// JSON-NEXT:  "valueCategory": "lvalue",
// JSON-NEXT:  "hasRewrittenInit": true,
// JSON-NEXT:  "inner": [
// JSON-NEXT:   {
// JSON-NEXT:    "id": "0x{{.*}}",
// JSON-NEXT:    "kind": "ExprWithCleanups",
// JSON:         "type": {
// JSON-NEXT:     "qualType": "const A"
// JSON-NEXT:    },
// JSON-NEXT:    "valueCategory": "lvalue",
// JSON-NEXT:    "inner": [
// JSON-NEXT:     {
// JSON-NEXT:      "id": "0x{{.*}}",
// JSON-NEXT:      "kind": "MaterializeTemporaryExpr",
// JSON:           "type": {
// JSON-NEXT:       "qualType": "const A"
// JSON-NEXT:      },
// JSON-NEXT:      "valueCategory": "lvalue",
// JSON-NEXT:      "extendingDecl": {
// JSON-NEXT:       "id": "0x{{.*}}",
// JSON-NEXT:       "kind": "VarDecl",
// JSON-NEXT:       "name": "b",
// JSON-NEXT:       "type": {
// JSON-NEXT:        "qualType": "B"
// JSON-NEXT:       }
// JSON-NEXT:      },
// JSON-NEXT:      "storageDuration": "automatic",
// JSON-NEXT:      "boundToLValueRef": true,
// JSON-NEXT:      "inner": [
// JSON-NEXT:       {
// JSON-NEXT:        "id": "0x{{.*}}",
// JSON-NEXT:        "kind": "ImplicitCastExpr",
// JSON:             "type": {
// JSON-NEXT:         "qualType": "const A"
// JSON-NEXT:        },
// JSON-NEXT:        "valueCategory": "prvalue",
// JSON-NEXT:        "castKind": "NoOp",
// JSON-NEXT:        "inner": [
// JSON-NEXT:         {
// JSON-NEXT:          "id": "0x{{.*}}",
// JSON-NEXT:          "kind": "CXXFunctionalCastExpr",
// JSON:               "type": {
// JSON-NEXT:           "qualType": "A"
// JSON-NEXT:          },
// JSON-NEXT:          "valueCategory": "prvalue",
// JSON-NEXT:          "castKind": "NoOp",
// JSON-NEXT:          "inner": [
// JSON-NEXT:           {
// JSON-NEXT:            "id": "0x{{.*}}",
// JSON-NEXT:            "kind": "InitListExpr",
// JSON:                 "type": {
// JSON-NEXT:             "qualType": "A"
// JSON-NEXT:            },
// JSON-NEXT:            "valueCategory": "prvalue",
// JSON-NEXT:            "inner": [
// JSON-NEXT:             {
// JSON-NEXT:              "id": "0x{{.*}}",
// JSON-NEXT:              "kind": "InitListExpr",
// JSON:                   "type": {
// JSON-NEXT:               "qualType": "int[1]"
// JSON-NEXT:              },
// JSON-NEXT:              "valueCategory": "prvalue",
// JSON-NEXT:              "inner": [
// JSON-NEXT:               {
// JSON-NEXT:                "id": "0x{{.*}}",
// JSON-NEXT:                "kind": "IntegerLiteral",
// JSON:                     "type": {
// JSON-NEXT:                 "qualType": "int"
// JSON-NEXT:                },
// JSON-NEXT:                "valueCategory": "prvalue",
// JSON-NEXT:                "value": "0"
// JSON-NEXT:               }
// JSON-NEXT:              ]
// JSON-NEXT:             }
// JSON-NEXT:            ]
