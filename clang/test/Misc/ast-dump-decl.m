// RUN: %clang_cc1 -Wno-unused -fblocks -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

@protocol P
@end

@interface A
@end

@interface TestObjCIvarDecl : A
@end

@implementation TestObjCIvarDecl {
  int varDefault;
  @private int varPrivate;
  @protected int varProtected;
  @public int varPublic;
  @package int varPackage;
}
@end
// CHECK:      ObjCImplementationDecl{{.*}} TestObjCIvarDecl
// CHECK-NEXT:   ObjCInterface{{.*}} 'TestObjCIvarDecl'
// CHECK-NEXT:   ObjCIvarDecl{{.*}} varDefault 'int' private
// CHECK-NEXT:   ObjCIvarDecl{{.*}} varPrivate 'int' private
// CHECK-NEXT:   ObjCIvarDecl{{.*}} varProtected 'int' protected
// CHECK-NEXT:   ObjCIvarDecl{{.*}} varPublic 'int' public
// CHECK-NEXT:   ObjCIvarDecl{{.*}} varPackage 'int' package

@interface testObjCMethodDecl : A {
}
- (int) TestObjCMethodDecl: (int)i, ...;
// CHECK:      ObjCMethodDecl{{.*}} - TestObjCMethodDecl: 'int'
// CHECK-NEXT:   ParmVarDecl{{.*}} i 'int'
// CHECK-NEXT:   ...
@end

@implementation testObjCMethodDecl
- (int) TestObjCMethodDecl: (int)i, ... {
  return 0;
}
// CHECK:      ObjCMethodDecl{{.*}} - TestObjCMethodDecl: 'int'
// CHECK-NEXT:   ImplicitParamDecl{{.*}} self
// CHECK-NEXT:   ImplicitParamDecl{{.*}} _cmd
// CHECK-NEXT:   ParmVarDecl{{.*}} i 'int'
// CHECK-NEXT:   ...
// CHECK-NEXT:   CompoundStmt
@end

@protocol TestObjCProtocolDecl
- (void) foo;
@end
// CHECK:      ObjCProtocolDecl{{.*}} TestObjCProtocolDecl
// CHECK-NEXT:   ObjCMethodDecl{{.*}} foo

@interface TestObjCClass : A <P>
- (void) foo;
@end
// CHECK:      ObjCInterfaceDecl{{.*}} TestObjCClass
// CHECK-NEXT:   super ObjCInterface{{.*}} 'A'
// CHECK-NEXT:   ObjCImplementation{{.*}} 'TestObjCClass'
// CHECK-NEXT:   ObjCProtocol{{.*}} 'P'
// CHECK-NEXT:   ObjCMethodDecl{{.*}} foo

@implementation TestObjCClass : A {
  int i;
}
- (void) foo {
}
@end
// CHECK:      ObjCImplementationDecl{{.*}} TestObjCClass
// CHECK-NEXT:   super ObjCInterface{{.*}} 'A'
// CHECK-NEXT:   ObjCInterface{{.*}} 'TestObjCClass'
// CHECK-NEXT:   ObjCIvarDecl{{.*}} i
// CHECK-NEXT:   ObjCMethodDecl{{.*}} foo

@interface TestObjCClass (TestObjCCategoryDecl) <P>
- (void) bar;
@end
// CHECK:      ObjCCategoryDecl{{.*}} TestObjCCategoryDecl
// CHECK-NEXT:   ObjCInterface{{.*}} 'TestObjCClass'
// CHECK-NEXT:   ObjCCategoryImpl{{.*}} 'TestObjCClass'
// CHECK-NEXT:   ObjCProtocol{{.*}} 'P'
// CHECK-NEXT:   ObjCMethodDecl{{.*}} bar

@implementation TestObjCClass (TestObjCCategoryDecl)
- (void) bar {
}
@end
// CHECK:      ObjCCategoryImplDecl{{.*}} TestObjCClass
// CHECK-NEXT:   ObjCInterface{{.*}} 'TestObjCClass'
// CHECK-NEXT:   ObjCCategory{{.*}} 'TestObjCCategoryDecl'
// CHECK-NEXT:   ObjCMethodDecl{{.*}} bar

@compatibility_alias TestObjCCompatibleAliasDecl A;
// CHECK:      ObjCCompatibleAliasDecl{{.*}} TestObjCCompatibleAliasDecl
// CHECK-NEXT:   ObjCInterface{{.*}} 'A'

@interface TestObjCProperty: A
@property(getter=getterFoo, setter=setterFoo:) int foo;
@property int bar;
@end
// CHECK:      ObjCInterfaceDecl{{.*}} TestObjCProperty
// CHECK:        ObjCPropertyDecl{{.*}} foo 'int' assign readwrite atomic unsafe_unretained
// CHECK-NEXT:     getter ObjCMethod{{.*}} 'getterFoo'
// CHECK-NEXT:     setter ObjCMethod{{.*}} 'setterFoo:'
// CHECK-NEXT:   ObjCPropertyDecl{{.*}} bar 'int' assign readwrite atomic unsafe_unretained
// CHECK-NEXT:   ObjCMethodDecl{{.*}} getterFoo
// CHECK-NEXT:   ObjCMethodDecl{{.*}} setterFoo:
// CHECK-NEXT:     ParmVarDecl{{.*}} foo
// CHECK-NEXT:   ObjCMethodDecl{{.*}} bar
// CHECK-NEXT:   ObjCMethodDecl{{.*}} setBar:
// CHECK-NEXT:     ParmVarDecl{{.*}} bar

@implementation TestObjCProperty {
  int i;
}
@synthesize foo=i;
@synthesize bar;
@end
// CHECK:      ObjCImplementationDecl{{.*}} TestObjCProperty
// CHECK:        ObjCPropertyImplDecl{{.*}} foo synthesize
// CHECK-NEXT:     ObjCProperty{{.*}} 'foo'
// CHECK-NEXT:     ObjCIvar{{.*}} 'i' 'int'
// CHECK-NEXT:   ObjCIvarDecl{{.*}} bar 'int' synthesize private
// CHECK-NEXT:   ObjCPropertyImplDecl{{.*}} bar synthesize
// CHECK-NEXT:     ObjCProperty{{.*}} 'bar'
// CHECK-NEXT:     ObjCIvar{{.*}} 'bar' 'int'

void TestBlockDecl(int x) {
  ^(int y, ...){ x; };
}
// CHECK:      FunctionDecl{{.*}}TestBlockDecl
// CHECK:      BlockDecl
// CHECK-NEXT:   ParmVarDecl{{.*}} y 'int'
// CHECK-NEXT:   ...
// CHECK-NEXT:   capture ParmVar{{.*}} 'x' 'int'
// CHECK-NEXT:   CompoundStmt

@interface B
+ (int) foo;
@end

void f() {
  __typeof__(B.foo) Test;
}
// CHECK: VarDecl{{.*}}Test 'typeof (B.foo)':'int'
