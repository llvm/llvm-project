// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition %s -o - | FileCheck --check-prefix=INTERPOSITION %s
// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck --check-prefix=NO %s

// INTERPOSITION: !{{[0-9]+}} = !{i32 1, !"SemanticInterposition", i32 1}
// NO-NOT: "SemanticInterposition"
