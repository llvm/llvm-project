// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=md --executor=standalone %S/../Inputs/function-pointer-type.cpp
// RUN: FileCheck %s --check-prefix=MD < %t/GlobalNamespace/index.md
// RUN: clang-doc --output=%t --format=md_mustache --executor=standalone %S/../Inputs/function-pointer-type.cpp
// RUN: FileCheck %s --check-prefix=MD-MUSTACHE < %t/md/GlobalNamespace/index.md

// MD: *void bar(void (*)(int) fn)*
// MD-MUSTACHE: *void bar(void (*)(int) fn)*
