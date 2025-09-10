// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -print-stats \
// RUN:    -emit-llvm -x ir /dev/null -o - 2>&1 | FileCheck %s --check-prefix=CHECK-IR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -print-stats \
// RUN:    -emit-llvm -x c /dev/null -o - 2>&1 | FileCheck %s --check-prefix=CHECK-C

// CHECK-IR: *** Source Manager Stats
// CHECK-IR: *** File Manager Stats
// CHECK-IR: *** Virtual File System Stats

// CHECK-C: *** Semantic Analysis Stats
// CHECK-C: *** Analysis Based Warnings Stats
// CHECK-C: *** AST Context Stats
// CHECK-C: *** Decl Stats
// CHECK-C: *** Stmt/Expr Stats
// CHECK-C: *** Preprocessor Stats
// CHECK-C: *** Identifier Table Stats
// CHECK-C: *** HeaderSearch Stats
// CHECK-C: *** Source Manager Stats
// CHECK-C: *** File Manager Stats
// CHECK-C: *** Virtual File System Stats
