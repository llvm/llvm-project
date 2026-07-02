// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/class-partial-specialization.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV7MyClassIPT_E.json

// CHECK:      "MangledName": "_ZTV7MyClassIPT_E",
// CHECK-NEXT: "Name": "MyClass",
// CHECK:      "Template": {
// CHECK-NEXT:   "Parameters": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "End": true,
// CHECK-NEXT:       "Param": "typename T"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "Specialization": {
// CHECK-NEXT:     "Parameters": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "Param": "T *",
// CHECK-NEXT:         "SpecParamEnd": true
// CHECK-NEXT:       }
// CHECK-NEXT:     ],
// CHECK-NEXT:     "SpecializationOf": "{{[0-9A-F]*}}",
// CHECK-NEXT:     "VerticalDisplay": false
// CHECK-NEXT:   },
// CHECK-NEXT:   "VerticalDisplay": false
// CHECK-NEXT: }
