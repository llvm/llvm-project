// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --executor=standalone %S/../Inputs/conversion_function.cpp
// RUN: find %t/ -regex ".*/[0-9A-F]*.yaml" -exec cat {} ";" | FileCheck %s --check-prefix=CHECK-YAML

// Output correct conversion names.
// CHECK-YAML:         Name:            'operator T'
