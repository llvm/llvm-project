// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/class-template.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV7MyClass.json

// CHECK:         "Name": "MyClass",
// CHECK:         "PublicMembers": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "IsStatic": false,
// CHECK-NEXT:        "Name": "MemberTemplate",
// CHECK-NEXT:        "Type": "T"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK:         "Name": "method",
// CHECK:         "Params": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "Name": "Param",
// CHECK-NEXT:        "ParamEnd": true,
// CHECK-NEXT:        "Type": {
// CHECK-NEXT:          "Name": "T",
// CHECK-NEXT:          "QualName": "T",
// CHECK-NEXT:          "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "ReturnType": {
// CHECK-NEXT:      "IsBuiltIn": false,
// CHECK-NEXT:      "IsTemplate": true,
// CHECK-NEXT:      "Name": "T",
// CHECK-NEXT:      "QualName": "T",
// CHECK-NEXT:      "USR": "0000000000000000000000000000000000000000"
// CHECK:         "Template": {
// CHECK-NEXT:      "Parameters": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Param": "typename T"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "VerticalDisplay": false
// CHECK-NEXT:    }
