// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --extra-arg -std=c++20 --output=%t --format=json --executor=standalone %s

// Requires that T suports post and pre-incrementing.
template<typename T>
concept Incrementable = requires(T x) {
  ++x;
  x++;
};

// CHECK:      {
// CHECK-NOT:    "Concepts": [
// CHECK-NOT:      {
// CHECK-NOT:        "ConstraintExpression": "requires (T x) { ++x; x++; }",
// CHECK-NOT:        "Description": [
// CHECK-NOT:        {
// CHECK-NOT:          "FullComment": {
// CHECK-NOT:            "Children": [
// CHECK-NOT:              {
// CHECK-NOT:                "ParagraphComment": {
// CHECK-NOT:                  "Children": [
// CHECK-NOT:                    {
// CHECK-NOT:                      "TextComment": " Requires that T suports post and pre-incrementing."
// CHECK-NOT:        },
// CHECK-NOT:        "IsType": true,
// CHECK-NOT:        "Name": "Incrementable",
// CHECK-NOT:        "Template": {
// CHECK-NOT:          "Parameters": [
// CHECK-NOT:            "typename T"
// CHECK-NOT:          ]
// CHECK-NOT:        },
// CHECK-NOT:        "USR": "{{[0-9A-F]*}}"
// CHECK-NOT:      }
// CHECK-NOT:    ],
// CHECK:        "Name": "",
// CHECK:        "USR": "0000000000000000000000000000000000000000"
// CHECK:      }
