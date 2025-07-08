// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/nested/index.json --check-prefix=NESTED
// RUN: FileCheck %s < %t/nested/inner/index.json --check-prefix=INNER

namespace nested {
  int Global;
  namespace inner {
    int InnerGlobal;
  } // namespace inner
} // namespace nested

// NESTED:       "Variables": [
// NESTED-NEXT:    {
// NESTED-NEXT:      "IsStatic": false,
// NESTED-NEXT:      "Location": {
// NESTED-NEXT:        "Filename": "{{.*}}nested-namespace.cpp",
// NESTED-NEXT:        "LineNumber": 7
// NESTED-NEXT:      },
// NESTED-NEXT:      "Name": "Global",
// NESTED-NEXT:      "Namespace": [
// NESTED-NEXT:        "nested"
// NESTED-NEXT:      ],

// INNER:       "Variables": [
// INNER-NEXT:    {
// INNER-NEXT:      "IsStatic": false,
// INNER-NEXT:      "Location": {
// INNER-NEXT:        "Filename": "{{.*}}nested-namespace.cpp",
// INNER-NEXT:        "LineNumber": 9
// INNER-NEXT:      },
// INNER-NEXT:      "Name": "InnerGlobal",
// INNER-NEXT:      "Namespace": [
// INNER-NEXT:        "inner",
// INNER-NEXT:        "nested"
// INNER-NEXT:      ],
