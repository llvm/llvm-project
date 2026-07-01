// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/member-function-pointer-type.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json

// CHECK:          "Functions": [
// CHECK-NEXT:       {
// CHECK:              "Name": "baz",
// CHECK:              "Params": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "Name": "fn",
// CHECK-NEXT:             "ParamEnd": true,
// CHECK-NEXT:             "Type": {
// CHECK-NEXT:               "Name": "void (Class::*)(int)",
// CHECK-NEXT:               "QualName": "void (Class::*)(int)",
// CHECK-NEXT:               "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
