// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=md --executor=standalone %S/../Inputs/array-type.cpp
// RUN: FileCheck %s --check-prefix=MD < %t/md/GlobalNamespace/index.md

// MD: *void qux(int (&)[5] arr)*
