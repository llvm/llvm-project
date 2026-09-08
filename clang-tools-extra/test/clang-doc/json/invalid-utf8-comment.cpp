// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/invalid-utf8-comment.cpp 2>&1
// RUN: cat %t/json/GlobalNamespace/_ZTV1A.json | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/210675.
// clang-doc should not crash when encountering comments with invalid UTF-8 bytes.
// Instead, the invalid bytes should be replaced with valid UTF-8 replacement characters.

// CHECK:       {
// CHECK-NEXT:    "Contexts": [
// CHECK-NEXT:      {
// CHECK-NEXT:        "DocumentationFileName": "index",
// CHECK-NEXT:        "End": true,
// CHECK-NEXT:        "Name": "Global Namespace",
// CHECK-NEXT:        "QualName": "GlobalNamespace",
// CHECK-NEXT:        "RelativePath": "./",
// CHECK-NEXT:        "USR": "0000000000000000000000000000000000000000"
// CHECK-NEXT:      }
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Description": {
// CHECK-NEXT:      "HasParagraphComments": true,
// CHECK-NEXT:      "ParagraphComments": [
// CHECK-NEXT:        [
// CHECK-NEXT:          {
// CHECK:               "TextComment":
// CHECK-NEXT:          }
// CHECK-NEXT:        ]
// CHECK-NEXT:      ]
// CHECK-NEXT:    },
// CHECK-NEXT:    "DocumentationFileName": "_ZTV1A",
// CHECK-NEXT:    "HasContexts": true,
// CHECK-NEXT:    "InfoType": "record",
// CHECK-NEXT:    "IsTypedef": false,
// CHECK-NEXT:    "Location": {
// CHECK-NEXT:      "Filename": "{{.*}}invalid-utf8-comment.cpp",
// CHECK-NEXT:      "LineNumber": 2
// CHECK-NEXT:    },
// CHECK-NEXT:    "MangledName": "_ZTV1A",
// CHECK-NEXT:    "Name": "A",
// CHECK-NEXT:    "Namespace": [
// CHECK-NEXT:      "GlobalNamespace"
// CHECK-NEXT:    ],
// CHECK-NEXT:    "Path": "GlobalNamespace",
// CHECK-NEXT:    "TagType": "class",
// CHECK-NEXT:    "USR": "{{[0-9A-F]*}}"
// CHECK-NEXT:  }
