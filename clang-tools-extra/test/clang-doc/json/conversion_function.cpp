// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --format=json --pretty-json --output=%t --executor=standalone %S/../Inputs/conversion_function.cpp
// RUN: FileCheck %s < %t/json/GlobalNamespace/_ZTV8MyStruct.json --check-prefix=CHECK-JSON

// CHECK-JSON:          "Name": "operator T",
