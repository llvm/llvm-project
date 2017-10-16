// RUN: rm -f %t.cp.cpp
// RUN: cp %s %t.cp.cpp
// RUN: clang-refactor local-rename -selection=%t.cp.cpp:9:7 -new-name=test %t.cp.cpp --
// RUN: grep -v CHECK %t.cp.cpp | FileCheck %t.cp.cpp
// RUN: cp %s %t.cp.cpp
// RUN: clang-refactor local-rename -selection=%t.cp.cpp:9:7-9:15 -new-name=test %t.cp.cpp --
// RUN: grep -v CHECK %t.cp.cpp | FileCheck %t.cp.cpp

class RenameMe {
// CHECK: class test {
};
