// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=json --pretty-json --output=%t --executor=standalone %S/../Inputs/index.cpp
// RUN: FileCheck %s < %t/json/index.json -check-prefix=CHECK-JSON

// CHECK-JSON:       "Index": [
// CHECK-JSON-NEXT:    {
// CHECK-JSON-NEXT:      "Name": "GlobalNamespace",
// CHECK-JSON-NEXT:      "QualName": "GlobalNamespace",
// CHECK-JSON-NEXT:      "Type": "namespace",
// CHECK-JSON-NEXT:      "USR": "0000000000000000000000000000000000000000"
// CHECK-JSON-NEXT:    },
// CHECK-JSON-NEXT:    {
// CHECK-JSON-NEXT:      "Name": "inner",
// CHECK-JSON-NEXT:      "QualName": "inner",
// CHECK-JSON-NEXT:      "Type": "namespace",
// CHECK-JSON-NEXT:      "USR": "{{([0-9A-F]{40})}}"
// CHECK-JSON-NEXT:    }
// CHECK-JSON-NEXT:  ]
