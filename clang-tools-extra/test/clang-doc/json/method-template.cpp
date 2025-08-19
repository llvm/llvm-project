// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/_ZTV7MyClass.json

struct MyClass {
  template<class T> T methodTemplate(T param) {
  }
};

// CHECK:           "PublicFunctions": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "InfoType": "function",
// CHECK-NEXT:          "IsStatic": false,
// CHECK-NEXT:          "Location": {
// CHECK-NEXT:            "Filename": "{{.*}}method-template.cpp",
// CHECK-NEXT:            "LineNumber": 6
// CHECK-NEXT:          },
// CHECK-NEXT:          "Name": "methodTemplate",
// CHECK-NEXT:          "Namespace": [
// CHECK-NEXT:            "MyClass",
// CHECK-NEXT:            "GlobalNamespace"
// CHECK-NEXT:          ],
// CHECK-NEXT:          "Params": [
// CHECK-NEXT:            {
// CHECK-NEXT:              "End": true,
// CHECK-NEXT:              "Name": "param",
// CHECK-NEXT:              "Type": "T"
// CHECK-NEXT:            }
// CHECK-NEXT:          ],
// CHECK-NEXT:          "ReturnType": {
// CHECK-NEXT:            "IsBuiltIn": false,
// CHECK-NEXT:            "IsTemplate": true,
// CHECK-NEXT:            "Name": "T",
// CHECK-NEXT:            "QualName": "T",
// CHECK-NEXT:            "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:          },
// CHECK-NEXT:          "Template": {
// CHECK-NEXT:            "Parameters": [
// CHECK-NEXT:              "class T"
// CHECK-NEXT:            ]
// CHECK-NEXT:          },
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
