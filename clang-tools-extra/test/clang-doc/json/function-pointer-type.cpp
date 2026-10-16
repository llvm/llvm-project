// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/function-pointer-type.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/index.json

// CHECK:          "Functions": [
// CHECK-NEXT:       {
// CHECK:              "Name": "bar",
// CHECK:              "Params": [
// CHECK-NEXT:           {
// CHECK-NEXT:             "Name": "fn",
// CHECK-NEXT:             "ParamEnd": true,
// CHECK-NEXT:             "Type": {
// CHECK-NEXT:               "Name": "void (*)(int)",
// CHECK-NEXT:               "QualName": "void (*)(int)",
// CHECK-NEXT:               "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         ]
