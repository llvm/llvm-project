// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/array-type.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json

// CHECK:          "Functions": [
// CHECK-NEXT:       {
// CHECK:              "Name": "qux",
// CHECK:              "Params": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "Name": "arr",
// CHECK-NEXT:             "ParamEnd": true,
// CHECK-NEXT:             "Type": {
// CHECK-NEXT:               "Name": "int (&)[5]",
// CHECK-NEXT:               "QualName": "int (&)[5]",
// CHECK-NEXT:               "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
