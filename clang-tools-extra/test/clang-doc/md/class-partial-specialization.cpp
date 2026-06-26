// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=md --executor=standalone %S/../Inputs/class-partial-specialization.cpp
// RUN: FileCheck %s --check-prefix=MD < %t/GlobalNamespace/MyClass.md
// RUN: clang-doc --output=%t --format=md_mustache --executor=standalone %S/../Inputs/class-partial-specialization.cpp
// RUN: FileCheck %s --check-prefix=MD-MUSTACHE < %t/md/GlobalNamespace/_ZTV7MyClassIPT_E.md

// MD: # struct MyClass
// MD-MUSTACHE: # struct MyClass
