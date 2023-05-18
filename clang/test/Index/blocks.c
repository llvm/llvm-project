// RUN: c-index-test -test-load-source local -fblocks %s | FileCheck %s

typedef int int_t;
struct foo { long x; };

void test() {
  static struct foo _foo;
  __block i = 0;
  ^ int_t(struct foo *foo) { return (int_t) foo->x + i; }(&_foo);
}

// CHECK: blocks.c:6:6: FunctionDecl=test:6:6 (Definition) Extent=[6:1 - 10:2]
// CHECK: blocks.c:6:13: CompoundStmt= Extent=[6:13 - 10:2]
// CHECK: blocks.c:7:3: DeclStmt= Extent=[7:3 - 7:26]
// CHECK: blocks.c:7:21: VarDecl=_foo:7:21 (Definition) Extent=[7:3 - 7:25]
// CHECK: blocks.c:7:17: TypeRef=struct foo:4:8 Extent=[7:17 - 7:20]
// CHECK: blocks.c:8:11: VarDecl=i:8:11 (Definition) Extent=[8:3 - 8:16]
// CHECK: blocks.c:8:15: IntegerLiteral= Extent=[8:15 - 8:16]
// CHECK: blocks.c:9:3: CallExpr= Extent=[9:3 - 9:65]
// CHECK: blocks.c:9:3: BlockExpr= Extent=[9:3 - 9:58]
// CHECK: blocks.c:9:5: TypeRef=int_t:3:13 Extent=[9:5 - 9:10]
// CHECK: blocks.c:9:23: ParmDecl=foo:9:23 (Definition) Extent=[9:11 - 9:26]
// CHECK: blocks.c:9:18: TypeRef=struct foo:4:8 Extent=[9:18 - 9:21]
// CHECK: blocks.c:9:28: CompoundStmt= Extent=[9:28 - 9:58]
// CHECK: blocks.c:9:30: ReturnStmt= Extent=[9:30 - 9:55]
// CHECK: blocks.c:9:37: BinaryOperator= Extent=[9:37 - 9:55]
// CHECK: blocks.c:9:37: CStyleCastExpr= Extent=[9:37 - 9:51]
// CHECK: blocks.c:9:38: TypeRef=int_t:3:13 Extent=[9:38 - 9:43]
// CHECK: blocks.c:9:50: MemberRefExpr=x:4:19 SingleRefName=[9:50 - 9:51] RefName=[9:50 - 9:51] Extent=[9:45 - 9:51]
// CHECK: blocks.c:9:45: DeclRefExpr=foo:9:23 Extent=[9:45 - 9:48]
// CHECK: blocks.c:9:54: DeclRefExpr=i:8:11 Extent=[9:54 - 9:55]
// CHECK: blocks.c:9:59: UnaryOperator= Extent=[9:59 - 9:64]
// CHECK: blocks.c:9:60: DeclRefExpr=_foo:7:21 Extent=[9:60 - 9:64]

