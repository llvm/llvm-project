// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --extra-arg -std=c++20 --output=%t --doxygen --format=json --executor=standalone %S/../Inputs/compound-constraints.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json

// CHECK:         "Name": "One",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Two",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Three",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "PreIncrementable<T>",
// CHECK-NEXT:          "Name": "PreIncrementable",
// CHECK-NEXT:          "QualName": "PreIncrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "PreDecrementable<T>",
// CHECK-NEXT:          "Name": "PreDecrementable",
// CHECK-NEXT:          "QualName": "PreDecrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
// CHECK:         "Name": "Four",
// CHECK:         "Template": {
// CHECK-NEXT:      "Constraints": [
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Incrementable<T>",
// CHECK-NEXT:          "Name": "Incrementable",
// CHECK-NEXT:          "QualName": "Incrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "Expression": "Decrementable<T>",
// CHECK-NEXT:          "Name": "Decrementable",
// CHECK-NEXT:          "QualName": "Decrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        },
// CHECK-NEXT:        {
// CHECK-NEXT:          "End": true,
// CHECK-NEXT:          "Expression": "PreIncrementable<T>",
// CHECK-NEXT:          "Name": "PreIncrementable",
// CHECK-NEXT:          "QualName": "PreIncrementable",
// CHECK-NEXT:          "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:        }
// CHECK-NEXT:      ],
