// RUN: c-index-test -test-load-source local %s | FileCheck %s

// The method 'bar' was also being reported outside the @implementation

@interface Foo {
  id x;
}
- (id) bar;
@end

@implementation Foo
- (id) bar {
  return 0;
}
@end

@protocol Prot8380046
@end

@interface R8380046
@end

@interface R8380046 () <Prot8380046>
@end

@class NSString;

void test() {
  NSString *s = @"objc str";
}

// CHECK: local-symbols.m:5:12: ObjCInterfaceDecl=Foo:5:12 Extent=[5:1 - 9:5]
// CHECK: local-symbols.m:6:6: ObjCIvarDecl=x:6:6 (Definition) Extent=[6:3 - 6:7]
// CHECK: local-symbols.m:6:3: TypeRef=id:0:0 Extent=[6:3 - 6:5]
// CHECK: local-symbols.m:8:8: ObjCInstanceMethodDecl=bar:8:8 Extent=[8:1 - 8:12]
// CHECK: local-symbols.m:8:4: TypeRef=id:0:0 Extent=[8:4 - 8:6]
// CHECK: local-symbols.m:11:17: ObjCImplementationDecl=Foo:11:17 (Definition) Extent=[11:1 - 15:2]
// CHECK: local-symbols.m:12:8: ObjCInstanceMethodDecl=bar:12:8 (Definition) Extent=[12:1 - 14:2]
// CHECK: local-symbols.m:12:4: TypeRef=id:0:0 Extent=[12:4 - 12:6]
// CHECK: local-symbols.m:13:10: UnexposedExpr= Extent=[13:10 - 13:11]
// CHECK: local-symbols.m:13:10: IntegerLiteral= Extent=[13:10 - 13:11]
// CHECK: local-symbols.m:17:11: ObjCProtocolDecl=Prot8380046:17:11 (Definition) Extent=[17:1 - 18:5]
// CHECK: local-symbols.m:20:12: ObjCInterfaceDecl=R8380046:20:12 Extent=[20:1 - 21:5]
// CHECK: local-symbols.m:23:12: ObjCCategoryDecl=:23:12 Extent=[23:1 - 24:5]
// CHECK: local-symbols.m:23:12: ObjCClassRef=R8380046:20:12 Extent=[23:12 - 23:20]
// CHECK: local-symbols.m:23:25: ObjCProtocolRef=Prot8380046:17:11 Extent=[23:25 - 23:36]

// CHECK: local-symbols.m:29:17: ObjCStringLiteral="objc str" Extent=[29:17 - 29:28]
