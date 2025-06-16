// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/GlobalNamespace/index.json

template<typename T>
concept Incrementable = requires(T x) {
  ++x;
  x++;
};

template<typename T> void increment(T t) requires Incrementable<T> {
  ++t;
  t++;
}

// CHECK:       "Functions": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "IsStatic": false,
// CHECK-NEXT:      "Location": {
// CHECK-NEXT:        "Filename": "{{.*}}function-requires.cpp",
// CHECK-NEXT:        "LineNumber": 11
// CHECK-NEXT:      },
// CHECK-NEXT:      "Name": "increment",
// CHECK-NEXT:      "Params": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Name": "t",
// CHECK-NEXT:          "Type": "T"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "ReturnType": {
// CHECK-NEXT:        "IsBuiltIn": false,
// CHECK-NEXT:        "IsTemplate": false,
// CHECK-NEXT:        "Name": "void",
// CHECK-NEXT:        "QualName": "void",
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:      },
// CHECK-NEXT:      "Template": {
// CHECK-NOT:         "Constraints": [
// CHECK-NOT:           "Incrementable<T>"
// CHECK-NOT:         ],
// CHECK-NEXT:        "Parameters": [
// CHECK-NEXT:          "typename T"
// CHECK-NEXT:        ]
// CHECK-NEXT:      },
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}" 
// CHECK-NEXT:    }
