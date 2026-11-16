// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --extra-arg -std=c++20 --output=%t --format=json --executor=standalone %S/../Inputs/class-requires.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV7MyClass.json

// CHECK:       "Name": "MyClass",
// CHECK-NEXT:  "Namespace": [
// CHECK-NEXT:    "GlobalNamespace"
// CHECK-NEXT:  ],
// CHECK-NEXT:  "Path": "GlobalNamespace",
// CHECK-NEXT:  "TagType": "struct",
// CHECK-NEXT:  "Template": {
// CHECK-NEXT:    "Constraints": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "Expression": "Addable<T>",
// CHECK-NEXT:        "Name": "Addable",
// CHECK-NEXT:        "QualName": "Addable",
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Parameters": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "typename T"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "VerticalDisplay": false
// CHECK-NEXT:  },
// CHECK-NEXT:  "USR": "{{[0-9A-F]*}}"
