// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/nested-namespace.cpp
// RUN: FileCheck %s < %t/json/nested/index.json --check-prefix=NESTED
// RUN: FileCheck %s < %t/json/nested/inner/index.json --check-prefix=INNER

// NESTED:       "Variables": [
// NESTED-NEXT:    {
// NESTED-NEXT:      "End": true,
// NESTED-NEXT:      "InfoType": "variable",
// NESTED-NEXT:      "IsStatic": false,
// NESTED-NEXT:      "Location": {
// NESTED-NEXT:        "Filename": "{{.*}}nested-namespace.cpp",
// NESTED-NEXT:        "LineNumber": 2
// NESTED-NEXT:      },
// NESTED-NEXT:      "Name": "Global",
// NESTED-NEXT:      "Namespace": [
// NESTED-NEXT:        "nested"
// NESTED-NEXT:      ],

// INNER:       "Variables": [
// INNER-NEXT:    {
// INNER-NEXT:      "End": true,
// INNER-NEXT:      "InfoType": "variable",
// INNER-NEXT:      "IsStatic": false,
// INNER-NEXT:      "Location": {
// INNER-NEXT:        "Filename": "{{.*}}nested-namespace.cpp",
// INNER-NEXT:        "LineNumber": 4
// INNER-NEXT:      },
// INNER-NEXT:      "Name": "InnerGlobal",
// INNER-NEXT:      "Namespace": [
// INNER-NEXT:        "inner",
// INNER-NEXT:        "nested"
// INNER-NEXT:      ],
