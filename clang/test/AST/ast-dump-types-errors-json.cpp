// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

using TestContainsErrors = int[sizeof(undef())];

// CHECK:  "kind": "TypeAliasDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "offset": 130,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 3,
// CHECK-NEXT:   "col": 7,
// CHECK-NEXT:   "tokLen": 18
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "offset": 124,
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "tokLen": 5
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "offset": 170,
// CHECK-NEXT:    "col": 47,
// CHECK-NEXT:    "tokLen": 1
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "TestContainsErrors",
// CHECK-NEXT:  "type": {
// CHECK-NEXT:   "qualType": "int[sizeof (<recovery-expr>(undef))]"
// CHECK-NEXT:  },
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "DependentSizedArrayType",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "int[sizeof (<recovery-expr>(undef))]"
// CHECK-NEXT:    },
// CHECK-NEXT:    "containsErrors": true,
// CHECK-NEXT:    "isDependent": true,
// CHECK-NEXT:    "isInstantiationDependent": true,
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "BuiltinType",
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "int"
// CHECK-NEXT:      }
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "UnaryExprOrTypeTraitExpr",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "offset": 155,
// CHECK-NEXT:        "col": 32,
// CHECK-NEXT:        "tokLen": 6
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "offset": 169,
// CHECK-NEXT:        "col": 46,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "unsigned long"
// CHECK-NEXT:      },
// CHECK-NEXT:      "valueCategory": "prvalue",
// CHECK-NEXT:      "name": "sizeof",
// CHECK-NEXT:      "inner": [
// CHECK-NEXT:       {
// CHECK-NEXT:        "id": "0x{{.*}}",
// CHECK-NEXT:        "kind": "ParenExpr",
// CHECK-NEXT:        "range": {
// CHECK-NEXT:         "begin": {
// CHECK-NEXT:          "offset": 161,
// CHECK-NEXT:          "col": 38,
// CHECK-NEXT:          "tokLen": 1
// CHECK-NEXT:         },
// CHECK-NEXT:         "end": {
// CHECK-NEXT:          "offset": 169,
// CHECK-NEXT:          "col": 46,
// CHECK-NEXT:          "tokLen": 1
// CHECK-NEXT:         }
// CHECK-NEXT:        },
// CHECK-NEXT:        "type": {
// CHECK-NEXT:         "qualType": "<dependent type>"
// CHECK-NEXT:        },
// CHECK-NEXT:        "valueCategory": "lvalue",
// CHECK-NEXT:        "inner": [
// CHECK-NEXT:         {
// CHECK-NEXT:          "id": "0x{{.*}}",
// CHECK-NEXT:          "kind": "RecoveryExpr",
// CHECK-NEXT:          "range": {
// CHECK-NEXT:           "begin": {
// CHECK-NEXT:            "offset": 162,
// CHECK-NEXT:            "col": 39,
// CHECK-NEXT:            "tokLen": 5
// CHECK-NEXT:           },
// CHECK-NEXT:           "end": {
// CHECK-NEXT:            "offset": 168,
// CHECK-NEXT:            "col": 45,
// CHECK-NEXT:            "tokLen": 1
// CHECK-NEXT:           }
// CHECK-NEXT:          },
// CHECK-NEXT:          "type": {
// CHECK-NEXT:           "qualType": "<dependent type>"
// CHECK-NEXT:          },
// CHECK-NEXT:          "valueCategory": "lvalue",
// CHECK-NEXT:          "inner": [
// CHECK-NEXT:           {
// CHECK-NEXT:            "id": "0x{{.*}}",
// CHECK-NEXT:            "kind": "UnresolvedLookupExpr",
// CHECK-NEXT:            "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:              "offset": 162,
// CHECK-NEXT:              "col": 39,
// CHECK-NEXT:              "tokLen": 5
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:              "offset": 162,
// CHECK-NEXT:              "col": 39,
// CHECK-NEXT:              "tokLen": 5
// CHECK-NEXT:             }
// CHECK-NEXT:            },
// CHECK-NEXT:            "type": {
// CHECK-NEXT:             "qualType": "<overloaded function type>"
// CHECK-NEXT:            },
// CHECK-NEXT:            "valueCategory": "lvalue",
// CHECK-NEXT:            "usesADL": true,
// CHECK-NEXT:            "name": "undef",
// CHECK-NEXT:            "lookups": []
// CHECK-NEXT:           }
// CHECK-NEXT:          ]
// CHECK-NEXT:         }
// CHECK-NEXT:        ]
// CHECK-NEXT:       }
// CHECK-NEXT:      ]
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }
