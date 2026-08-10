// RUN: c-index-test -write-pch %t.ast -arch x86_64 -mmacos-version-min=10.6 -fblocks -x objective-c %s
// RUN: c-index-test -test-file-scan %t.ast %s > %t 2>&1 && FileCheck --input-file=%t %s
@interface Foo 
{
}

- foo;
+ fooC;

@end

@interface Bar : Foo 
{
}

@end

@interface Foo (FooCat)
- (int) catMethodWithFloat:(float) fArg;
- (float) floatMethod;
@end

@protocol Proto
- pMethod;
@end

@protocol SubP <Proto>
- spMethod;
@end

@interface Baz : Bar <SubP>
{
    int _anIVar;
}

- (Foo *) bazMethod;

@end

enum {
  someEnum
};

int main (int argc, const char * argv[]) {
	Baz * bee;
	id a = [bee foo];
	id <SubP> c = [Foo fooC];
	id <Proto> d;
	d = c;
	[d pMethod];
	[bee catMethodWithFloat:[bee floatMethod]];
  main(someEnum, (const char **)bee);
}

#define CONCAT(X, Y) X##Y

void f() {
   int CONCAT(my,_var);
}
#undef CONCAT

// CHECK: [1:1 - 3:1] Invalid Cursor => NoDeclFound
// CHECK: [3:1 - 7:1] ObjCInterfaceDecl=Foo:3:12
// CHECK: [7:1 - 7:7] ObjCInstanceMethodDecl=foo:7:3
// CHECK: [7:7 - 8:1] ObjCInterfaceDecl=Foo:3:12
// CHECK: [8:1 - 8:8] ObjCClassMethodDecl=fooC:8:3
// CHECK: [8:8 - 10:5] ObjCInterfaceDecl=Foo:3:12
// CHECK: [10:5 - 12:1] Invalid Cursor => NoDeclFound
// CHECK: [12:1 - 12:18] ObjCInterfaceDecl=Bar:12:12
// CHECK: [12:18 - 12:21] ObjCSuperClassRef=Foo:3:12
// CHECK: [12:21 - 16:5] ObjCInterfaceDecl=Bar:12:12
// CHECK: [16:5 - 18:1] Invalid Cursor => NoDeclFound
// CHECK: [18:1 - 18:12] ObjCCategoryDecl=FooCat:18:12
// CHECK: [18:12 - 18:15] ObjCClassRef=Foo:3:12
// CHECK: [18:15 - 19:1] ObjCCategoryDecl=FooCat:18:12
// CHECK: [19:1 - 19:29] ObjCInstanceMethodDecl=catMethodWithFloat::19:9
// CHECK: [19:29 - 19:40] ParmDecl=fArg:19:36 (Definition)
// CHECK: [19:40 - 19:41] ObjCInstanceMethodDecl=catMethodWithFloat::19:9
// CHECK: [19:41 - 20:1] ObjCCategoryDecl=FooCat:18:12
// CHECK: [20:1 - 20:23] ObjCInstanceMethodDecl=floatMethod:20:11
// CHECK: [20:23 - 21:5] ObjCCategoryDecl=FooCat:18:12
// CHECK: [21:5 - 23:1] Invalid Cursor => NoDeclFound
// CHECK: [23:1 - 24:1] ObjCProtocolDecl=Proto:23:11 (Definition)
// CHECK: [24:1 - 24:11] ObjCInstanceMethodDecl=pMethod:24:3
// CHECK: [24:11 - 25:5] ObjCProtocolDecl=Proto:23:11 (Definition)
// CHECK: [25:5 - 27:1] Invalid Cursor => NoDeclFound
// CHECK: [27:1 - 27:17] ObjCProtocolDecl=SubP:27:11 (Definition)
// CHECK: [27:17 - 27:22] ObjCProtocolRef=Proto:23:1
// CHECK: [27:22 - 28:1] ObjCProtocolDecl=SubP:27:11 (Definition)
// CHECK: [28:1 - 28:12] ObjCInstanceMethodDecl=spMethod:28:3
// CHECK: [28:12 - 29:5] ObjCProtocolDecl=SubP:27:11 (Definition)
// CHECK: [29:5 - 31:1] Invalid Cursor => NoDeclFound
// CHECK: [31:1 - 31:18] ObjCInterfaceDecl=Baz:31:12
// CHECK: [31:18 - 31:21] ObjCSuperClassRef=Bar:12:12
// CHECK: [31:21 - 31:23] ObjCInterfaceDecl=Baz:31:12
// CHECK: [31:23 - 31:27] ObjCProtocolRef=SubP:27:1
// CHECK: [31:27 - 33:5] ObjCInterfaceDecl=Baz:31:12
// CHECK: [33:5 - 33:16] ObjCIvarDecl=_anIVar:33:9 (Definition)
// CHECK: [33:16 - 36:1] ObjCInterfaceDecl=Baz:31:12
// CHECK: [36:1 - 36:4] ObjCInstanceMethodDecl=bazMethod:36:1
// CHECK: [36:4 - 36:7] ObjCClassRef=Foo:3:12
// CHECK: [36:7 - 36:21] ObjCInstanceMethodDecl=bazMethod:36:1
// CHECK: [36:21 - 38:5] ObjCInterfaceDecl=Baz:31:12
// CHECK: [38:5 - 40:1] Invalid Cursor => NoDeclFound
// CHECK: [40:1 - 41:3] EnumDecl=enum (unnamed at {{.*}}):40:1 (Definition)
// CHECK: [41:3 - 41:11] EnumConstantDecl=someEnum:41:3 (Definition)
// CHECK: [41:11 - 42:2] EnumDecl=enum (unnamed at {{.*}}):40:1 (Definition)
// CHECK: [42:2 - 44:1] Invalid Cursor => NoDeclFound
// CHECK: [44:1 - 44:11] FunctionDecl=main:44:5 (Definition)
// CHECK: [44:11 - 44:19] ParmDecl=argc:44:15 (Definition)
// CHECK: [44:19 - 44:21] FunctionDecl=main:44:5 (Definition)
// CHECK: [44:21 - 44:40] ParmDecl=argv:44:34 (Definition)
// CHECK: [44:40 - 44:42] FunctionDecl=main:44:5 (Definition)
// CHECK: [44:42 - 45:2] CompoundStmt=
// CHECK: [45:2 - 45:5] ObjCClassRef=Baz:31:12
// CHECK: [45:5 - 45:11] VarDecl=bee:45:8 (Definition)
// CHECK: [45:11 - 45:12] DeclStmt=
// CHECK: [45:12 - 46:2] CompoundStmt=
// CHECK: [46:2 - 46:4] TypeRef=id:0:0
// CHECK: [46:4 - 46:9] VarDecl=a:46:5 (Definition)
// CHECK: [46:9 - 46:10] ObjCMessageExpr=foo:7:3
// CHECK: [46:10 - 46:13] DeclRefExpr=bee:45:8
// CHECK: [46:13 - 46:18] ObjCMessageExpr=foo:7:3
// CHECK: [46:18 - 46:19] DeclStmt=
// CHECK: [46:19 - 47:2] CompoundStmt=
// CHECK: [47:2 - 47:4] TypeRef=id:0:0
// CHECK: [47:4 - 47:6] VarDecl=c:47:12 (Definition)
// CHECK: [47:6 - 47:10] ObjCProtocolRef=SubP:27:1
// CHECK: [47:10 - 47:16] VarDecl=c:47:12 (Definition)
// CHECK: [47:16 - 47:17] ObjCMessageExpr=fooC:8:3
// CHECK: [47:17 - 47:20] ObjCClassRef=Foo:3:12
// CHECK: [47:20 - 47:26] ObjCMessageExpr=fooC:8:3
// CHECK: [47:26 - 47:27] DeclStmt=
// CHECK: [47:27 - 48:2] CompoundStmt=
// CHECK: [48:2 - 48:4] TypeRef=id:0:0
// CHECK: [48:4 - 48:6] VarDecl=d:48:13 (Definition)
// CHECK: [48:6 - 48:11] ObjCProtocolRef=Proto:23:1
// CHECK: [48:11 - 48:14] VarDecl=d:48:13 (Definition)
// CHECK: [48:14 - 48:15] DeclStmt=
// CHECK: [48:15 - 49:2] CompoundStmt=
// CHECK: [49:2 - 49:3] DeclRefExpr=d:48:13
// CHECK: [49:3 - 49:6] BinaryOperator=
// CHECK: [49:6 - 49:7] DeclRefExpr=c:47:12
// CHECK: [49:7 - 50:2] CompoundStmt=
// CHECK: [50:2 - 50:3] ObjCMessageExpr=pMethod:24:3
// CHECK: [50:3 - 50:4] DeclRefExpr=d:48:13
// CHECK: [50:4 - 50:13] ObjCMessageExpr=pMethod:24:3
// CHECK: [50:13 - 51:2] CompoundStmt=
// CHECK: [51:2 - 51:3] ObjCMessageExpr=catMethodWithFloat::19:9
// CHECK: [51:3 - 51:6] DeclRefExpr=bee:45:8
// CHECK: [51:6 - 51:26] ObjCMessageExpr=catMethodWithFloat::19:9
// CHECK: [51:26 - 51:27] ObjCMessageExpr=floatMethod:20:1
// CHECK: [51:27 - 51:30] DeclRefExpr=bee:45:8
// CHECK: [51:30 - 51:43] ObjCMessageExpr=floatMethod:20:1
// CHECK: [51:43 - 51:44] ObjCMessageExpr=catMethodWithFloat::19:9
// CHECK: [51:44 - 52:3] CompoundStmt=
// CHECK: [52:3 - 52:7] DeclRefExpr=main:44:5
// CHECK: [52:7 - 52:8] CallExpr=main:44:5
// CHECK: [52:8 - 52:16] DeclRefExpr=someEnum:41:3
// CHECK: [52:16 - 52:18] CallExpr=main:44:5
// CHECK: [52:18 - 52:33] CStyleCastExpr=
// CHECK: [52:33 - 52:36] DeclRefExpr=bee:45:8
// CHECK: [52:36 - 52:37] CallExpr=main:44:5
// CHECK: [52:37 - 53:2] CompoundStmt=
// CHECK: [55:9 - 55:26] macro definition=CONCAT
// CHECK: [57:1 - 57:10] FunctionDecl=f:57:6 (Definition)
// CHECK: [58:4 - 58:8] VarDecl=my_var:58:8 (Definition)
// CHECK: [58:8 - 58:15] macro expansion=CONCAT:55:9
// REQUIRES: x86-registered-target
