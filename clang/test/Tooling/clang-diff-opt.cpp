// RUN: %clang_cc1 -E %s > %t.src.cpp
// RUN: %clang_cc1 -E %s > %t.dst.cpp -DDEST
// RUN: clang-diff -dump-matches -s=10 %t.src.cpp %t.dst.cpp -- | FileCheck %s
//
// Test the behaviour of the matching according to the optimal tree edit
// distance, implemented with Zhang and Shasha's algorithm.
// Just for testing we use a tiny value of 10 for maxsize. Subtrees bigger than
// this size will not be processed by the optimal algorithm.

#ifndef DEST

void f1() { {;} {{;}} }

void f2() { {;} {{;;;;;}} }

void f3() { {;} {{;;;;;;}} }

#else

void f1() {
// Jaccard similarity = 3 / (5 + 4 - 3) = 3 / 6 >= 0.5
// The optimal matching algorithm should move the ; into the outer block
// CHECK: Match CompoundStmt(2) to CompoundStmt(2)
// CHECK-NOT: Match CompoundStmt(3)
// CHECK-NEXT: Match NullStmt(4) to NullStmt(3)
  ; {{;}}
}

void f2() {
  // Jaccard similarity = 7 / (10 + 10 - 7) >= 0.5
  // As none of the subtrees is bigger than 10 nodes, the optimal algorithm
  // will be run.
  // CHECK: Match NullStmt(11) to NullStmt(9)
  ;; {{;;;;;}}
}

void f3() {
  // Jaccard similarity = 8 / (11 + 11 - 8) >= 0.5
  // As the subtrees are bigger than 10 nodes, the optimal algorithm will not
  // be run.
  // CHECK: Delete NullStmt(22)
  ;; {{;;;;;;}}
}
#endif
