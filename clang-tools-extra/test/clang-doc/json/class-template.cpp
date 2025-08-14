// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/_ZTV7MyClass.json

template<typename T> struct MyClass {
  T MemberTemplate;
  T method(T Param); 
};

// CHECK:         "Name": "MyClass",
// CHECK:         "Name": "method",
// CHECK:         "Params": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "Name": "Param",
// CHECK-NEXT:        "Type": "T"
// CHECK-NEXT:      } 
// CHECK-NEXT:    ], 
// CHECK-NEXT:    "ReturnType": {
// CHECK-NEXT:      "IsBuiltIn": false,
// CHECK-NEXT:      "IsTemplate": true,
// CHECK-NEXT:      "Name": "T",
// CHECK-NEXT:      "QualName": "T"
// CHECK-NEXT:      "USR": "0000000000000000000000000000000000000000"
// CHECK:           "Name": "MemberTemplate",
// CHECK:           "Type": "T"
// CHECK:         "Template": {
// CHECK-NEXT:      "Parameters": [
// CHECK-NEXT:        "typename T"
// CHECK-NEXT:      ] 
