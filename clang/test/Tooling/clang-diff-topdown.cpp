// RUN: %clang_cc1 -E %s > %t.src.cpp
// RUN: %clang_cc1 -E %s > %t.dst.cpp -DDEST
// RUN: clang-diff -dump-matches -stop-diff-after=topdown %t.src.cpp %t.dst.cpp -- -std=c++11 | FileCheck %s
//
// Test the top-down matching of identical subtrees only.

#ifndef DEST

void f1()
{
  // Match some subtree of height greater than 2.
  // CHECK: Match CompoundStmt(3) to CompoundStmt(3)
  // CHECK: Match CompoundStmt(4) to CompoundStmt(4)
  // CHECK: Match NullStmt(5) to NullStmt(5)
  {{;}}

  // Don't match subtrees that are smaller.
  // CHECK-NOT: Match CompoundStmt(6)
  // CHECK-NOT: Match NullStmt(7)
  {;}

  // Greedy approach - use the first matching subtree when there are multiple
  // identical subtrees.
  // CHECK: Match CompoundStmt(8) to CompoundStmt(8)
  // CHECK: Match CompoundStmt(9) to CompoundStmt(9)
  // CHECK: Match NullStmt(10) to NullStmt(10)
  {{;;}}
}

int x;

namespace src {
  int x;
  int x1 = x + 1;
  int x2 = ::x + 1;
}

class A { int x = 1 + 1; void f() { int x1 = x; } };

#else


void f1() {

  {{;}}

  {;}

  {{;;}}
  // CHECK-NOT: Match {{.*}} to CompoundStmt(11)
  // CHECK-NOT: Match {{.*}} to CompoundStmt(12)
  // CHECK-NOT: Match {{.*}} to NullStmt(13)
  {{;;}}

  // CHECK-NOT: Match {{.*}} to NullStmt(14)
  ;
}

int x;

namespace dst {
  int x;
  // CHECK: Match DeclRefExpr: :x(17) to DeclRefExpr: :x(22)
  int x1 = x + 1;
  // CHECK: Match DeclRefExpr: x(21) to DeclRefExpr: x(26)
  int x2 = ::x + 1;
}

class B {
  // Only the class name changed; it is not included in the field value,
  // therefore there is no update.
  // CHECK: Match FieldDecl: :x(int)(24) to FieldDecl: :x(int)(29)
  // CHECK-NOT: Update FieldDecl: :x(int)(24)
  int x = 1+1;
  void f() {
    // CHECK: Match MemberExpr: :x(32) to MemberExpr: :x(37)
    // CHECK-NOT: Update MemberExpr: :x(32)
    int x1 = B::x;
  }

};

#endif
