// RUN: clang-format %s -sort-includes -style=LLVM -disable-format | FileCheck %s

#include <b>
#include <a>
// CHECK: <a>
// CHECK-NEXT: <b>

// CHECK: int *a  ;
int *a  ;
