// RUN: %clang_cc1 -emit-llvm -gcodeview -debug-info-kind=limited -x c %s -o - | FileCheck %s --check-prefix CHECK-C
// RUN: %clang_cc1 -emit-llvm -gcodeview -debug-info-kind=limited -x c++ %s -o - | FileCheck %s --check-prefix CHECK-CXX
//
// Test for DIFlagNonTrivial on forward declared DICompositeTypes.

struct Incomplete;
struct Incomplete (*func_ptr)(void) = 0;
// CHECK-C: !DICompositeType({{.*}}name: "Incomplete"
// CHECK-C-NOT: DIFlagNonTrivial
// CHECK-CXX: !DICompositeType({{.*}}name: "Incomplete"
// CHECK-CXX-SAME: DIFlagNonTrivial
