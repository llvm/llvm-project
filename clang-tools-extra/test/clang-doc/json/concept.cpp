// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/index.json

// Requires that T suports post and pre-incrementing.
template<typename T>
concept Incrementable = requires(T x) {
  ++x;
  x++;
};

// CHECK:       {
// CHECK-NEXT:    "Concepts": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "ConstraintExpression": "requires (T x) { ++x; x++; }",
// CHECK-NEXT:        "Description": {
// CHECK-NEXT:        "HasParagraphComments": true,
// CHECK-NEXT:        "ParagraphComments": [
// CHECK-NEXT:          [
// CHECK-NEXT:            {
// CHECK-NEXT:              "TextComment": " Requires that T suports post and pre-incrementing."
// CHECK:             "End": true,
// CHECK-NEXT:        "InfoType": "concept",
// CHECK-NEXT:        "IsType": true,
// CHECK-NEXT:        "Name": "Incrementable",
// CHECK-NEXT:        "Template": {
// CHECK-NEXT:          "Parameters": [
// CHECK-NEXT:            "typename T"
// CHECK-NEXT:          ]
// CHECK-NEXT:        },
// CHECK-NEXT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK:        "Name": "",
// CHECK:        "USR": "0000000000000000000000000000000000000000"
// CHECK:      }
