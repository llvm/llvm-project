// Semantic Interposition is active if
//  -fsemantic-interposition is set,
// - pic-level > 0
// - pic-is-pie is not set

// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition -pic-level 0 %s -o - | FileCheck %s -check-prefix=CHECK-NO-INTERPOSITION
// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition -pic-level 1 %s -o - | FileCheck %s -check-prefix=CHECK-INTERPOSITION
// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition -pic-level 2 %s -o - | FileCheck %s -check-prefix=CHECK-INTERPOSITION
// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition -pic-level 0 %s -o - | FileCheck %s -check-prefix=CHECK-NO-INTERPOSITION
// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition -pic-level 1 -pic-is-pie %s -o - | FileCheck %s -check-prefix=CHECK-NO-INTERPOSITION
// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition -pic-level 2 -pic-is-pie %s -o - | FileCheck %s -check-prefix=CHECK-NO-INTERPOSITION

// CHECK-NO-INTERPOSITION-NOT: "SemanticInterposition"
// CHECK-INTERPOSITION: !{{[0-9]+}} = !{i32 1, !"SemanticInterposition", i32 1}
