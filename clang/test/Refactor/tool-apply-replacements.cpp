// RUN: sed -e 's#//.*$##' %s > %t.cpp
// RUN: clang-refactor local-rename -selection=%t.cpp:7:7 -new-name=test %t.cpp -- | FileCheck %s
// RUN: clang-refactor local-rename -selection=%t.cpp:7:7-7:15 -new-name=test %t.cpp -- | FileCheck %s
// RUN: clang-refactor local-rename -i -selection=%t.cpp:7:7 -new-name=test %t.cpp --
// RUN: FileCheck -input-file=%t.cpp %s

class RenameMe {
// CHECK: class test {
};
