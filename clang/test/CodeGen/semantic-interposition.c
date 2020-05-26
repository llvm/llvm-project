// RUN: %clang_cc1 -emit-llvm -fsemantic-interposition %s -o - | FileCheck --check-prefix=INTERPOSITION %s
// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck --check-prefix=NO %s
/// With explicit -fno-semantic-interposition, add a module flag to inform the
/// backend that dso_local can be inferred.
// RUN: %clang_cc1 -emit-llvm -fno-semantic-interposition %s -o - | FileCheck --check-prefix=EXPLICIT_NO %s

// INTERPOSITION: !{{[0-9]+}} = !{i32 1, !"SemanticInterposition", i32 1}
// NO-NOT: "SemanticInterposition"
// EXPLICIT_NO: !{{[0-9]+}} = !{i32 1, !"SemanticInterposition", i32 0}
