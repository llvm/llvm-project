// RUN: %clang_analyze_cc1 -w -fblocks %s \
// RUN:   -analyzer-checker=debug.DumpLiveExprs \
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

// CHECK: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:

  ~RAII() { blk(); }

// CHECK-NEXT: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
};

void foo(Collection *coll) {
  RAII raii(^{});
  for (Item *item in coll) {}
}
// CHECK-NEXT: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B2 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Collection *' <LValueToRValue>
// CHECK-NEXT: `-DeclRefExpr {{.*}} 'Collection *' lvalue ParmVar {{.*}} 'coll' 'Collection *'
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B3 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Collection *' <LValueToRValue>
// CHECK-NEXT: `-DeclRefExpr {{.*}} 'Collection *' lvalue ParmVar {{.*}} 'coll' 'Collection *'
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B4 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Collection *' <LValueToRValue>
// CHECK-NEXT: `-DeclRefExpr {{.*}} 'Collection *' lvalue ParmVar {{.*}} 'coll' 'Collection *'
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B5 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B0 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:
// CHECK-NEXT: [ B1 (live expressions at block exit) ]
// CHECK-EMPTY:
// CHECK-EMPTY:

