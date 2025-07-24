// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

using TestContainsErrors = int[sizeof(undef())];

// CHECK: {
// CHECK-NEXT:   "id": "{{.*}}",
// CHECK-NEXT:   "kind": "TypeAliasDecl",
// CHECK-NEXT:   "loc": {
// CHECK-NEXT:     "offset": {{.*}},
// CHECK-NEXT:     "file": "{{.*}}",
// CHECK-NEXT:     "line": 3,
// CHECK-NEXT:     "col": {{.*}},
// CHECK-NEXT:     "tokLen": 18
// CHECK-NEXT:   },
// CHECK-NEXT:   "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:       "offset": {{.*}},
// CHECK-NEXT:       "col": {{.*}},
// CHECK-NEXT:       "tokLen": 5
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:       "offset": {{.*}},
// CHECK-NEXT:       "col": {{.*}},
// CHECK-NEXT:       "tokLen": 1
// CHECK-NEXT:     }
// CHECK-NEXT:   },
// CHECK-NEXT:   "name": "TestContainsErrors",
// CHECK-NEXT:   "type": {
// CHECK-NEXT:     "qualType": "int[sizeof (<recovery-expr>(undef))]"
// CHECK-NEXT:   },
// CHECK-NEXT:   "typeDetails": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "id": "{{.*}}",
// CHECK-NEXT:       "kind": "DependentSizedArrayType",
// CHECK-NEXT:       "type": {
// CHECK-NEXT:         "qualType": "int[sizeof (<recovery-expr>(undef))]"
// CHECK-NEXT:       },
// CHECK-NEXT:       "containsErrors": true,
// CHECK-NEXT:       "isDependent": true,
// CHECK-NEXT:       "isInstantiationDependent": true,
// CHECK-NEXT:       "qualDetails": [
// CHECK-NEXT:         "array"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "typeDetails": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "id": "{{.*}}",
// CHECK-NEXT:           "kind": "BuiltinType",
// CHECK-NEXT:           "type": {
// CHECK-NEXT:             "qualType": "int"
// CHECK-NEXT:           },
// CHECK-NEXT:           "qualDetails": [
// CHECK-NEXT:             "signed",
// CHECK-NEXT:             "integer"
// CHECK-NEXT:           ]
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "id": "{{.*}}",
// CHECK-NEXT:           "kind": "UnaryExprOrTypeTraitExpr",
// CHECK-NEXT:           "range": {
// CHECK-NEXT:             "begin": {
// CHECK-NEXT:               "offset": {{.*}},
// CHECK-NEXT:               "col": {{.*}},
// CHECK-NEXT:               "tokLen": 6
// CHECK-NEXT:             },
// CHECK-NEXT:             "end": {
// CHECK-NEXT:               "offset": {{.*}},
// CHECK-NEXT:               "col": {{.*}},
// CHECK-NEXT:               "tokLen": 1
// CHECK-NEXT:             }
// CHECK-NEXT:           },
// CHECK-NEXT:           "type": {
// CHECK-NEXT:             "desugaredQualType": "unsigned long",
// CHECK-NEXT:             "qualType": "__size_t"
// CHECK-NEXT:           },
// CHECK-NEXT:           "valueCategory": "prvalue",
// CHECK-NEXT:           "name": "sizeof",
// CHECK-NEXT:           "inner": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "id": "{{.*}}",
// CHECK-NEXT:               "kind": "ParenExpr",
// CHECK-NEXT:               "range": {
// CHECK-NEXT:                 "begin": {
// CHECK-NEXT:                   "offset": {{.*}},
// CHECK-NEXT:                   "col": {{.*}},
// CHECK-NEXT:                   "tokLen": 1
// CHECK-NEXT:                 },
// CHECK-NEXT:                 "end": {
// CHECK-NEXT:                   "offset": {{.*}},
// CHECK-NEXT:                   "col": {{.*}},
// CHECK-NEXT:                   "tokLen": 1
// CHECK-NEXT:                 }
// CHECK-NEXT:               },
// CHECK-NEXT:               "type": {
// CHECK-NEXT:                 "qualType": "<dependent type>"
// CHECK-NEXT:               },
// CHECK-NEXT:               "valueCategory": "lvalue",
// CHECK-NEXT:               "inner": [
// CHECK-NEXT:                 {
// CHECK-NEXT:                   "id": "{{.*}}",
// CHECK-NEXT:                   "kind": "RecoveryExpr",
// CHECK-NEXT:                   "range": {
// CHECK-NEXT:                     "begin": {
// CHECK-NEXT:                       "offset": {{.*}},
// CHECK-NEXT:                       "col": {{.*}},
// CHECK-NEXT:                       "tokLen": 5
// CHECK-NEXT:                     },
// CHECK-NEXT:                     "end": {
// CHECK-NEXT:                       "offset": {{.*}},
// CHECK-NEXT:                       "col": {{.*}},
// CHECK-NEXT:                       "tokLen": 1
// CHECK-NEXT:                     }
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "type": {
// CHECK-NEXT:                     "qualType": "<dependent type>"
// CHECK-NEXT:                   },
// CHECK-NEXT:                   "valueCategory": "lvalue",
// CHECK-NEXT:                   "inner": [
// CHECK-NEXT:                     {
// CHECK-NEXT:                       "id": "{{.*}}",
// CHECK-NEXT:                       "kind": "UnresolvedLookupExpr",
// CHECK-NEXT:                       "range": {
// CHECK-NEXT:                         "begin": {
// CHECK-NEXT:                           "offset": {{.*}},
// CHECK-NEXT:                           "col": {{.*}},
// CHECK-NEXT:                           "tokLen": 5
// CHECK-NEXT:                         },
// CHECK-NEXT:                         "end": {
// CHECK-NEXT:                           "offset": {{.*}},
// CHECK-NEXT:                           "col": {{.*}},
// CHECK-NEXT:                           "tokLen": 5
// CHECK-NEXT:                         }
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "type": {
// CHECK-NEXT:                         "qualType": "<overloaded function type>"
// CHECK-NEXT:                       },
// CHECK-NEXT:                       "valueCategory": "lvalue",
// CHECK-NEXT:                       "usesADL": true,
// CHECK-NEXT:                       "name": "undef",
// CHECK-NEXT:                       "lookups": []
// CHECK-NEXT:                     }
// CHECK-NEXT:                   ]
// CHECK-NEXT:                 }
// CHECK-NEXT:               ]
// CHECK-NEXT:             }
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
