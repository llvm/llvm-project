// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=md --executor=standalone %S/../Inputs/nested-pointer-qualifiers.cpp
// RUN: FileCheck %s --check-prefix=MD-MUSTACHE < %t/md/GlobalNamespace/index.md

// MD-MUSTACHE: *void foo(const int *const * ptr)*
