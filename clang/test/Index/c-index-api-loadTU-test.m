// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -emit-pch -x objective-c %s -o %t.ast
// RUN: c-index-test -test-load-tu %t.ast all | FileCheck %s

@interface Foo 
{
  __attribute__((iboutlet)) id myoutlet;
}
- (void) __attribute__((ibaction)) myMessage:(id)msg;
- foo __attribute__((deprecated));
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

// Test attribute traversal.
#define IBOutlet __attribute__((iboutlet))
#define IBOutletCollection(ClassName) __attribute__((iboutletcollection(ClassName)))
#define IBAction void)__attribute__((ibaction)

@interface TestAttributes {
  IBOutlet id anOutlet;
  IBOutletCollection(id) id anOutletCollection;
}
- (IBAction) actionMethod:(id)arg;
@end

typedef struct X0 X1;
struct X0;
struct X0  {};

@interface TestAttributes()
// <rdar://problem/9561076>
@property (retain) IBOutlet id anotherOutlet;
@end

// CHECK: c-index-api-loadTU-test.m:4:12: ObjCInterfaceDecl=Foo:4:12 Extent=[4:1 - 12:5]
// CHECK: c-index-api-loadTU-test.m:6:32: ObjCIvarDecl=myoutlet:6:32 (Definition) Extent=[6:3 - 6:40]
// CHECK: <invalid loc>:0:0: attribute(iboutlet)=
// CHECK: c-index-api-loadTU-test.m:6:29: TypeRef=id:0:0 Extent=[6:29 - 6:31]
// CHECK: c-index-api-loadTU-test.m:8:36: ObjCInstanceMethodDecl=myMessage::8:36 Extent=[8:1 - 8:54]
// CHECK: <invalid loc>:0:0: attribute(ibaction)=
// CHECK: c-index-api-loadTU-test.m:8:50: ParmDecl=msg:8:50 (Definition) Extent=[8:47 - 8:53]
// CHECK: c-index-api-loadTU-test.m:8:47: TypeRef=id:0:0 Extent=[8:47 - 8:49]
// CHECK: c-index-api-loadTU-test.m:9:3: ObjCInstanceMethodDecl=foo:9:3 (deprecated) (always deprecated: "") Extent=[9:1 - 9:35]
// CHECK: c-index-api-loadTU-test.m:10:3: ObjCClassMethodDecl=fooC:10:3 Extent=[10:1 - 10:8]
// CHECK: c-index-api-loadTU-test.m:14:12: ObjCInterfaceDecl=Bar:14:12 Extent=[14:1 - 18:5]
// CHECK: c-index-api-loadTU-test.m:14:18: ObjCSuperClassRef=Foo:4:12 Extent=[14:18 - 14:21]
// CHECK: c-index-api-loadTU-test.m:20:12: ObjCCategoryDecl=FooCat:20:12 Extent=[20:1 - 23:5]
// CHECK: c-index-api-loadTU-test.m:20:12: ObjCClassRef=Foo:4:12 Extent=[20:12 - 20:15]
// CHECK: c-index-api-loadTU-test.m:21:9: ObjCInstanceMethodDecl=catMethodWithFloat::21:9 Extent=[21:1 - 21:41]
// CHECK: c-index-api-loadTU-test.m:21:36: ParmDecl=fArg:21:36 (Definition) Extent=[21:29 - 21:40]
// CHECK: c-index-api-loadTU-test.m:22:11: ObjCInstanceMethodDecl=floatMethod:22:11 Extent=[22:1 - 22:23]
// CHECK: c-index-api-loadTU-test.m:25:11: ObjCProtocolDecl=Proto:25:11 (Definition) Extent=[25:1 - 27:5]
// CHECK: c-index-api-loadTU-test.m:26:3: ObjCInstanceMethodDecl=pMethod:26:3 Extent=[26:1 - 26:11]
// CHECK: c-index-api-loadTU-test.m:29:11: ObjCProtocolDecl=SubP:29:11 (Definition) Extent=[29:1 - 31:5]
// CHECK: c-index-api-loadTU-test.m:29:17: ObjCProtocolRef=Proto:25:11 Extent=[29:17 - 29:22]
// CHECK: c-index-api-loadTU-test.m:30:3: ObjCInstanceMethodDecl=spMethod:30:3 Extent=[30:1 - 30:12]
// CHECK: c-index-api-loadTU-test.m:33:12: ObjCInterfaceDecl=Baz:33:12 Extent=[33:1 - 40:5]
// CHECK: c-index-api-loadTU-test.m:33:18: ObjCSuperClassRef=Bar:14:12 Extent=[33:18 - 33:21]
// CHECK: c-index-api-loadTU-test.m:33:23: ObjCProtocolRef=SubP:29:11 Extent=[33:23 - 33:27]
// CHECK: c-index-api-loadTU-test.m:35:9: ObjCIvarDecl=_anIVar:35:9 (Definition) Extent=[35:5 - 35:16]
// CHECK: c-index-api-loadTU-test.m:38:11: ObjCInstanceMethodDecl=bazMethod:38:11 Extent=[38:1 - 38:21]
// CHECK: c-index-api-loadTU-test.m:38:4: ObjCClassRef=Foo:4:12 Extent=[38:4 - 38:7]
// CHECK: c-index-api-loadTU-test.m:42:1: EnumDecl=:42:1 (Definition) Extent=[42:1 - 44:2]
// CHECK: c-index-api-loadTU-test.m:43:3: EnumConstantDecl=someEnum:43:3 (Definition) Extent=[43:3 - 43:11]
// CHECK: c-index-api-loadTU-test.m:46:5: FunctionDecl=main:46:5 (Definition) Extent=[46:1 - 55:2]
// CHECK: c-index-api-loadTU-test.m:46:15: ParmDecl=argc:46:15 (Definition) Extent=[46:11 - 46:19]
// CHECK: c-index-api-loadTU-test.m:46:34: ParmDecl=argv:46:34 (Definition) Extent=[46:21 - 46:40]
// CHECK: c-index-api-loadTU-test.m:47:8: VarDecl=bee:47:8 (Definition) Extent=[47:2 - 47:11]
// CHECK: c-index-api-loadTU-test.m:47:2: ObjCClassRef=Baz:33:12 Extent=[47:2 - 47:5]
// CHECK: c-index-api-loadTU-test.m:48:5: VarDecl=a:48:5 (Definition) Extent=[48:2 - 48:18]
// CHECK: c-index-api-loadTU-test.m:48:2: TypeRef=id:0:0 Extent=[48:2 - 48:4]
// CHECK: c-index-api-loadTU-test.m:48:9: ObjCMessageExpr=foo:9:3 Extent=[48:9 - 48:18]
// CHECK: c-index-api-loadTU-test.m:48:10: DeclRefExpr=bee:47:8 Extent=[48:10 - 48:13]
// CHECK: c-index-api-loadTU-test.m:49:12: VarDecl=c:49:12 (Definition) Extent=[49:2 - 49:26]
// CHECK: c-index-api-loadTU-test.m:49:2: TypeRef=id:0:0 Extent=[49:2 - 49:4]
// CHECK: c-index-api-loadTU-test.m:49:6: ObjCProtocolRef=SubP:29:11 Extent=[49:6 - 49:10]
// CHECK: c-index-api-loadTU-test.m:49:16: UnexposedExpr=fooC:10:3 Extent=[49:16 - 49:26]
// CHECK: c-index-api-loadTU-test.m:49:16: ObjCMessageExpr=fooC:10:3 Extent=[49:16 - 49:26]
// CHECK: c-index-api-loadTU-test.m:49:17: ObjCClassRef=Foo:4:12 Extent=[49:17 - 49:20]
// CHECK: c-index-api-loadTU-test.m:50:13: VarDecl=d:50:13 (Definition) Extent=[50:2 - 50:14]
// CHECK: c-index-api-loadTU-test.m:50:2: TypeRef=id:0:0 Extent=[50:2 - 50:4]
// CHECK: c-index-api-loadTU-test.m:50:6: ObjCProtocolRef=Proto:25:11 Extent=[50:6 - 50:11]
// CHECK: c-index-api-loadTU-test.m:51:2: BinaryOperator= Extent=[51:2 - 51:7]
// CHECK: c-index-api-loadTU-test.m:51:2: DeclRefExpr=d:50:13 Extent=[51:2 - 51:3]
// CHECK: c-index-api-loadTU-test.m:51:6: UnexposedExpr=c:49:12 Extent=[51:6 - 51:7]
// CHECK: c-index-api-loadTU-test.m:51:6: DeclRefExpr=c:49:12 Extent=[51:6 - 51:7]
// CHECK: c-index-api-loadTU-test.m:52:2: ObjCMessageExpr=pMethod:26:3 Extent=[52:2 - 52:13]
// CHECK: c-index-api-loadTU-test.m:52:3: DeclRefExpr=d:50:13 Extent=[52:3 - 52:4]
// CHECK: c-index-api-loadTU-test.m:53:2: ObjCMessageExpr=catMethodWithFloat::21:9 Extent=[53:2 - 53:44]
// CHECK: c-index-api-loadTU-test.m:53:3: DeclRefExpr=bee:47:8 Extent=[53:3 - 53:6]
// CHECK: c-index-api-loadTU-test.m:53:26: ObjCMessageExpr=floatMethod:22:11 Extent=[53:26 - 53:43]
// CHECK: c-index-api-loadTU-test.m:53:27: DeclRefExpr=bee:47:8 Extent=[53:27 - 53:30]
// CHECK: c-index-api-loadTU-test.m:54:3: CallExpr=main:46:5 Extent=[54:3 - 54:37]
// CHECK: c-index-api-loadTU-test.m:54:3: UnexposedExpr=main:46:5 Extent=[54:3 - 54:7]
// CHECK: c-index-api-loadTU-test.m:54:3: DeclRefExpr=main:46:5 Extent=[54:3 - 54:7]
// CHECK: c-index-api-loadTU-test.m:54:8: DeclRefExpr=someEnum:43:3 Extent=[54:8 - 54:16]
// CHECK: c-index-api-loadTU-test.m:54:18: CStyleCastExpr= Extent=[54:18 - 54:36]
// CHECK: c-index-api-loadTU-test.m:54:33: DeclRefExpr=bee:47:8 Extent=[54:33 - 54:36]
// CHECK: c-index-api-loadTU-test.m:62:12: ObjCInterfaceDecl=TestAttributes:62:12 Extent=[62:1 - 67:5]
// CHECK: c-index-api-loadTU-test.m:63:15: ObjCIvarDecl=anOutlet:63:15 (Definition) Extent=[58:18 - 63:23]
// CHECK: <invalid loc>:0:0: attribute(iboutlet)=
// CHECK: c-index-api-loadTU-test.m:64:29: ObjCIvarDecl=anOutletCollection:64:29 (Definition) Extent=[59:39 - 64:47]
// CHECK: <invalid loc>:0:0: attribute(iboutletcollection)= [IBOutletCollection=ObjCObjectPointer]
// CHECK: c-index-api-loadTU-test.m:64:26: TypeRef=id:0:0 Extent=[64:26 - 64:28]
// CHECK: c-index-api-loadTU-test.m:66:14: ObjCInstanceMethodDecl=actionMethod::66:14 Extent=[66:1 - 66:35]
// CHECK: <invalid loc>:0:0: attribute(ibaction)=
// CHECK: c-index-api-loadTU-test.m:66:31: ParmDecl=arg:66:31 (Definition) Extent=[66:28 - 66:34]
// CHECK: c-index-api-loadTU-test.m:66:28: TypeRef=id:0:0 Extent=[66:28 - 66:30]
// CHECK: c-index-api-loadTU-test.m:69:16: StructDecl=X0:69:16 Extent=[69:9 - 69:18]
// CHECK: c-index-api-loadTU-test.m:69:19: TypedefDecl=X1:69:19 (Definition) Extent=[69:1 - 69:21]
// CHECK: c-index-api-loadTU-test.m:69:16: TypeRef=struct X0:71:8 Extent=[69:16 - 69:18]
// CHECK: c-index-api-loadTU-test.m:70:8: StructDecl=X0:70:8 Extent=[70:1 - 70:10]
// CHECK: c-index-api-loadTU-test.m:71:8: StructDecl=X0:71:8 (Definition) Extent=[71:1 - 71:14]

// CHECK: <invalid loc>:0:0: attribute(iboutlet)=
// CHECK: c-index-api-loadTU-test.m:75:29: TypeRef=id:0:0 Extent=[75:29 - 75:31]
// CHECK: c-index-api-loadTU-test.m:75:32: ObjCInstanceMethodDecl=anotherOutlet:75:32 Extent=[75:32 - 75:45]
// CHECK: c-index-api-loadTU-test.m:75:32: ObjCInstanceMethodDecl=setAnotherOutlet::75:32 Extent=[75:32 - 75:45]
// CHECK: c-index-api-loadTU-test.m:75:32: ParmDecl=anotherOutlet:75:32 (Definition) Extent=[75:32 - 75:45]

