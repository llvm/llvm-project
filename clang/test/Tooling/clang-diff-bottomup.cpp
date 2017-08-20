// RUN: %clang_cc1 -E %s > %t.src.cpp
// RUN: %clang_cc1 -E %s > %t.dst.cpp -DDEST
// RUN: clang-diff -dump-matches -s=0 %t.src.cpp %t.dst.cpp -- | FileCheck %s
//
// Test the bottom-up matching, with maxsize set to 0, so that the optimal matching will never be applied.

#ifndef DEST

void f1() { ; {{;}} }
void f2() { ;; {{;}} }

#else

// Jaccard similarity threshold is 0.5.

void f1() {
// CompoundStmt: 3 matched descendants, subtree sizes 4 and 5
// Jaccard similarity = 3 / (4 + 5 - 3) = 3 / 6 >= 0.5
// CHECK: Match FunctionDecl: f1(void (void))(1) to FunctionDecl: f1(void (void))(1)
// CHECK: Match CompoundStmt(2) to CompoundStmt(2)
// CHECK: Match CompoundStmt(4) to CompoundStmt(3)
// CHECK: Match CompoundStmt(5) to CompoundStmt(4)
// CHECK: Match NullStmt(6) to NullStmt(5)
  {{;}} ;;
}

void f2() {
// CompoundStmt: 3 matched descendants, subtree sizes 4 and 5
// Jaccard similarity = 3 / (5 + 6 - 3) = 3 / 8 < 0.5
// CHECK-NOT: Match FunctionDecl(9)
// CHECK-NOT: Match CompoundStmt(10)
// CHECK: Match CompoundStmt(11) to CompoundStmt(10)
// CHECK: Match CompoundStmt(12) to CompoundStmt(11)
// CHECK: Match NullStmt(13) to NullStmt(12)
// CHECK-NOT: Match NullStmt(13)
  {{;}} ;;;
}

#endif
