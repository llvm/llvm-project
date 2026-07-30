typedef int T;
struct X { int a, b; };
void f(void *ptr) {
  T* t_ptr = (T *)ptr;
  (void)sizeof(T);
  struct X x = (struct X){1, 2};
  void *xx = ptr ? : &x;
}

int test_blocks(int x) {
  __block int y = x;
  ^{
     static int z = 0;
     y = (++z) + x;
     ^{
       ++z;
       ++y;
     }();
   }();
  return y;
}

struct Y {
  struct X array[3];
};

enum { StartIndex = 1 };

void test_members(int aval, int bval) {
  struct Y y0 = { .array[StartIndex].b = bval, .array[StartIndex].a = aval };
  __builtin_offsetof(struct Y, array[StartIndex].b);
}

// RUN: c-index-test -test-load-source all %s -fblocks | FileCheck %s
// CHECK: macro definition=__clang__
// CHECK: load-exprs.c:1:13: TypedefDecl=T:1:13 (Definition) Extent=[1:1 - 1:14]
// CHECK: load-exprs.c:2:8: StructDecl=X:2:8 (Definition) Extent=[2:1 - 2:23]
// CHECK: load-exprs.c:2:16: FieldDecl=a:2:16 (Definition) Extent=[2:12 - 2:17]
// CHECK: load-exprs.c:2:19: FieldDecl=b:2:19 (Definition) Extent=[2:12 - 2:20]
// CHECK: load-exprs.c:3:6: FunctionDecl=f:3:6 (Definition) Extent=[3:1 - 8:2]
// CHECK: load-exprs.c:3:14: ParmDecl=ptr:3:14 (Definition) Extent=[3:8 - 3:17]
// CHECK: load-exprs.c:4:6: VarDecl=t_ptr:4:6 (Definition) Extent=[4:3 - 4:22]
// CHECK: load-exprs.c:4:3: TypeRef=T:1:13 Extent=[4:3 - 4:4]
// CHECK: load-exprs.c:4:15: TypeRef=T:1:13 Extent=[4:15 - 4:16]
// CHECK: load-exprs.c:4:19: DeclRefExpr=ptr:3:14 Extent=[4:19 - 4:22]
// CHECK: load-exprs.c:5:16: TypeRef=T:1:13 Extent=[5:16 - 5:17]
// CHECK: load-exprs.c:6:12: VarDecl=x:6:12 (Definition) Extent=[6:3 - 6:32]
// CHECK: load-exprs.c:6:10: TypeRef=struct X:2:8 Extent=[6:10 - 6:11]
// CHECK: load-exprs.c:6:24: TypeRef=struct X:2:8 Extent=[6:24 - 6:25]
// CHECK: load-exprs.c:7:9: VarDecl=xx:7:9 (Definition) Extent=[7:3 - 7:24]
// CHECK: load-exprs.c:7:14: DeclRefExpr=ptr:3:14 Extent=[7:14 - 7:17]
// CHECK: load-exprs.c:7:23: DeclRefExpr=x:6:12 Extent=[7:23 - 7:24]
// CHECK: load-exprs.c:10:5: FunctionDecl=test_blocks:10:5 (Definition) Extent=[10:1 - 21:2]
// CHECK: load-exprs.c:10:21: ParmDecl=x:10:21 (Definition) Extent=[10:17 - 10:22]
// CHECK: load-exprs.c:11:15: VarDecl=y:11:15 (Definition) Extent=[11:3 - 11:20]
// CHECK: load-exprs.c:11:19: DeclRefExpr=x:10:21 Extent=[11:19 - 11:20]
// CHECK: load-exprs.c:12:3: CallExpr= Extent=[12:3 - 19:7]
// CHECK: load-exprs.c:13:17: VarDecl=z:13:17 (Definition) Extent=[13:6 - 13:22]
// CHECK: load-exprs.c:14:6: DeclRefExpr=y:11:15 Extent=[14:6 - 14:7]
// CHECK: load-exprs.c:14:13: DeclRefExpr=z:13:17 Extent=[14:13 - 14:14]
// CHECK: load-exprs.c:14:18: DeclRefExpr=x:10:21 Extent=[14:18 - 14:19]
// CHECK: load-exprs.c:15:6: CallExpr= Extent=[15:6 - 18:9]
// CHECK: load-exprs.c:16:10: DeclRefExpr=z:13:17 Extent=[16:10 - 16:11]
// CHECK: load-exprs.c:17:10: DeclRefExpr=y:11:15 Extent=[17:10 - 17:11]
// CHECK: load-exprs.c:20:10: DeclRefExpr=y:11:15 Extent=[20:10 - 20:11]
// CHECK: load-exprs.c:29:6: FunctionDecl=test_members:29:6 (Definition)
// CHECK: load-exprs.c:30:12: VarDecl=y0:30:12 (Definition) Extent=[30:3 - 30:77]
// CHECK: load-exprs.c:30:10: TypeRef=struct Y:23:8 Extent=[30:10 - 30:11]
// CHECK: load-exprs.c:30:20: MemberRef=array:24:12 Extent=[30:20 - 30:25]
// CHECK: load-exprs.c:30:26: DeclRefExpr=StartIndex:27:8 Extent=[30:26 - 30:36]
// CHECK: load-exprs.c:30:38: MemberRef=b:2:19 Extent=[30:38 - 30:39]
// CHECK: load-exprs.c:30:42: DeclRefExpr=bval:29:33 Extent=[30:42 - 30:46]
// CHECK: load-exprs.c:30:49: MemberRef=array:24:12 Extent=[30:49 - 30:54]
// CHECK: load-exprs.c:30:55: DeclRefExpr=StartIndex:27:8 Extent=[30:55 - 30:65]
// CHECK: load-exprs.c:30:67: MemberRef=a:2:16 Extent=[30:67 - 30:68]
// CHECK: load-exprs.c:30:71: DeclRefExpr=aval:29:23 Extent=[30:71 - 30:75]
// CHECK: load-exprs.c:31:29: TypeRef=struct Y:23:8 Extent=[31:29 - 31:30]
// CHECK: load-exprs.c:31:32: MemberRef=array:24:12 Extent=[31:32 - 31:37]
// CHECK: load-exprs.c:31:38: DeclRefExpr=StartIndex:27:8 Extent=[31:38 - 31:48]
// CHECK: load-exprs.c:31:50: MemberRef=b:2:19 Extent=[31:50 - 31:51]

