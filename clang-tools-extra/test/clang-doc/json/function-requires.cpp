// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/index.json

template<typename T>
concept Incrementable = requires(T x) {
  ++x;
  x++;
};

template<typename T> void increment(T t) requires Incrementable<T>;

template<Incrementable T> Incrementable auto incrementTwo(T t);

// CHECK:       "Functions": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "InfoType": "function",
// CHECK-NEXT:      "IsStatic": false,
// CHECK-NEXT:      "Name": "increment",
// CHECK-NEXT:      "Params": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Name": "t",
// CHECK-NEXT:          "Type": "T"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "ReturnType": {
// CHECK-NEXT:        "IsBuiltIn": true,
// CHECK-NEXT:        "IsTemplate": false,
// CHECK-NEXT:        "Name": "void",
// CHECK-NEXT:        "QualName": "void",
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:      },
// CHECK-NEXT:      "Template": {
// CHECK-NEXT:        "Constraints": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "End": true,
// CHECK-NEXT:            "Expression": "Incrementable<T>",
// CHECK-NEXT:            "Name": "Incrementable",
// CHECK-NEXT:            "Path": "",
// CHECK-NEXT:            "QualName": "Incrementable",
// CHECK-NEXT:            "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Parameters": [
// CHECK-NEXT:          "typename T"
// CHECK-NEXT:        ]
// CHECK-NEXT:      },
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}" 
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "End": true,
// CHECK-NEXT:      "InfoType": "function",
// CHECK-NEXT:      "IsStatic": false,
// CHECK-NEXT:      "Name": "incrementTwo",
// CHECK-NEXT:      "Params": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Name": "t",
// CHECK-NEXT:          "Type": "T"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK-NEXT:      "ReturnType": {
// CHECK-NEXT:        "IsBuiltIn": false,
// CHECK-NEXT:        "IsTemplate": false,
// CHECK-NEXT:        "Name": "Incrementable auto",
// CHECK-NEXT:        "QualName": "Incrementable auto",
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:      },
// CHECK-NEXT:      "Template": {
// CHECK-NEXT:        "Constraints": [
// CHECK-NEXT:          {
// CHECK-NEXT:            "End": true,
// CHECK-NEXT:            "Expression": "Incrementable<T>",
// CHECK-NEXT:            "Name": "Incrementable",
// CHECK-NEXT:            "Path": "",
// CHECK-NEXT:            "QualName": "Incrementable",
// CHECK-NEXT:            "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:          }
// CHECK-NEXT:        ],
// CHECK-NEXT:        "Parameters": [
// CHECK-NEXT:          "Incrementable T"
// CHECK-NEXT:        ]
// CHECK-NEXT:      },
// CHECK-NEXT:      "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:    }
