// RUN: not clang-rename -new-name=class -offset=133 %s 2>&1 | FileCheck %s
// CHECK: ERROR: new name is not a valid identifier in C++17.
