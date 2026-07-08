// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=md --executor=standalone %S/../Inputs/array-type.cpp
// RUN: FileCheck %s --check-prefix=MD < %t/GlobalNamespace/index.md
// RUN: clang-doc --output=%t --format=md_mustache --executor=standalone %S/../Inputs/array-type.cpp
// RUN: FileCheck %s --check-prefix=MD-MUSTACHE < %t/md/GlobalNamespace/index.md

// MD: *void qux(int (&)[5] arr)*
// MD-MUSTACHE: *void qux(int (&)[5] arr)*
