// RUN: %clang_analyze_cc1 -w -fblocks %s \
// RUN:   -analyzer-checker=debug.DumpLiveStmts \
// RUN:   2>&1 | FileCheck %s

@interface Item
// ...
@end

@interface Collection
// ...
@end

typedef void (^Blk)();

struct RAII {
  Blk blk;

public:
  RAII(Blk blk): blk(blk) {}

// CHECK: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:

  ~RAII() { blk(); }

// CHECK-NEXT: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
};

void foo(Collection *coll) {
  RAII raii(^{});
  for (Item *item in coll) {}
}
// CHECK-NEXT: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B2 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclStmt {{.*}}
// CHECK-NEXT: `-VarDecl {{.*}}  item 'Item *'
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Collection *' <LValueToRValue>
// CHECK-NEXT: `-DeclRefExpr {{.*}} 'Collection *' lvalue ParmVar {{.*}} 'coll' 'Collection *'
// CHECK-EMPTY:
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B3 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclStmt {{.*}}
// CHECK-NEXT: `-VarDecl {{.*}}  item 'Item *'
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Collection *' <LValueToRValue>
// CHECK-NEXT: `-DeclRefExpr {{.*}} 'Collection *' lvalue ParmVar {{.*}} 'coll' 'Collection *'
// CHECK-EMPTY:
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B4 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclStmt {{.*}}
// CHECK-NEXT: `-VarDecl {{.*}}  item 'Item *'
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Collection *' <LValueToRValue>
// CHECK-NEXT: `-DeclRefExpr {{.*}} 'Collection *' lvalue ParmVar {{.*}} 'coll' 'Collection *'
// CHECK-EMPTY:
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B5 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: DeclStmt {{.*}}
// CHECK-NEXT: `-VarDecl {{.*}}  item 'Item *'
// CHECK-EMPTY:
// CHECK-NEXT: CompoundStmt {{.*}}
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B0 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live statements at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:

