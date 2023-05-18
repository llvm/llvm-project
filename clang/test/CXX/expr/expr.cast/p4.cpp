// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

struct A { int x; };
struct B { int y; };
struct C : A, B { };

// CHECK: casting_away_constness
void casting_away_constness(const B &b, const C &c, const B *bp, const C *cp) {
  // CHECK: DerivedToBase (B)
  // CHECK: DeclRefExpr {{.*}} ParmVar {{.*}} 'c'
  (void)(B&)c;
  // CHECK: BaseToDerived (B)
  // CHECK: DeclRefExpr {{.*}} ParmVar {{.*}} 'b'
  (void)(C&)b;
  // CHECK: DerivedToBase (B)
  // CHECK: DeclRefExpr {{.*}} ParmVar {{.*}} 'cp'
  (void)(B*)cp;
  // CHECK: BaseToDerived (B)
  // CHECK: DeclRefExpr {{.*}} ParmVar {{.*}} 'bp'
  (void)(C*)bp;
  // CHECK: ReturnStmt
  return;
}
