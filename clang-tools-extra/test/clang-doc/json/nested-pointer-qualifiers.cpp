// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/nested-pointer-qualifiers.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json

// CHECK:          "Functions": [
// CHECK-NEXT:       {
// CHECK:              "Name": "foo",
// CHECK:              "Params": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "Name": "ptr",
// CHECK-NEXT:             "ParamEnd": true,
// CHECK-NEXT:             "Type": {
// CHECK-NEXT:               "Name": "const int *const *",
// CHECK-NEXT:               "QualName": "const int *const *",
// CHECK-NEXT:               "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
