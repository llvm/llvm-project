// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json

/// @brief Test function for multiple return comments
///
/// @return First return value description
/// @return Second return value description
int testMultipleReturns() {
  return 42;
}

// CHECK:      "Functions": [
// CHECK-NEXT:   {
// CHECK:        "Description": {
// CHECK:          "BriefComments": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "TextComment": "Test function for multiple return comments"
// CHECK-NEXT:       }
// CHECK-NEXT:     ],
// CHECK:          "HasReturnComments": true,
// CHECK:          "ReturnComments": [
// CHECK-NEXT:       {
// CHECK-NEXT:         "TextComment": "First return value description"
// CHECK-NEXT:       },
// CHECK-NEXT:       {
// CHECK-NEXT:         "TextComment": "Second return value description"
// CHECK-NEXT:       }
// CHECK-NEXT:     ]
// CHECK:        }
