// RUN: rm -rf %t && mkdir -p %t && cd %t
// RUN: not --crash %clang_cc1 %s -emit-llvm -o foo.ll
// RUN: ls . | FileCheck %s --allow-empty
// CHECK-NOT: foo.ll

#pragma clang __debug crash
FOO
