// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/_ZTV7MyClass.json

template<typename T>
concept Addable = requires(T a, T b) {
  { a + b };
};

template<typename T>
requires Addable<T>
struct MyClass;

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
// CHECK-NEXT:        "Path": "",
// CHECK-NEXT:        "QualName": "Addable",
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Parameters": [
// CHECK-NEXT:      "typename T"
// CHECK-NEXT:    ]
// CHECK-NEXT:  },
// CHECK-NEXT:  "USR": "{{[0-9A-F]*}}"
