// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --pretty-json --output=%t --format=json --executor=standalone %S/../Inputs/multiple-namespaces.cpp
// RUN: FileCheck %s < %t/json/foo/tools/index.json --check-prefix=CHECK-FOO
// RUN: FileCheck %s < %t/json/bar/tools/index.json --check-prefix=CHECK-BAR

// CHECK-FOO: "Name": "tools"

// CHECK-BAR: "Name": "tools"
