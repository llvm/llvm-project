// RUN: c-index-test -test-load-source local %s | FileCheck %s

// From: <rdar://problem/7568881>
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

// From: <rdar://problem/8380046>

@protocol Prot8380046
@end

@interface R8380046
@end

@interface R8380046 () <Prot8380046>
@end

// CHECK: local-symbols.m:6:12: ObjCInterfaceDecl=Foo:6:12 Extent=[6:1 - 10:5]
// CHECK: local-symbols.m:7:6: ObjCIvarDecl=x:7:6 (Definition) Extent=[7:3 - 7:7]
// CHECK: local-symbols.m:7:3: TypeRef=id:0:0 Extent=[7:3 - 7:5]
// CHECK: local-symbols.m:9:8: ObjCInstanceMethodDecl=bar:9:8 Extent=[9:1 - 9:12]
// CHECK: local-symbols.m:9:4: TypeRef=id:0:0 Extent=[9:4 - 9:6]
// CHECK: local-symbols.m:12:17: ObjCImplementationDecl=Foo:12:17 (Definition) Extent=[12:1 - 16:2]
// CHECK: local-symbols.m:13:8: ObjCInstanceMethodDecl=bar:13:8 (Definition) Extent=[13:1 - 15:2]
// CHECK: local-symbols.m:13:4: TypeRef=id:0:0 Extent=[13:4 - 13:6]
// CHECK: local-symbols.m:14:10: UnexposedExpr= Extent=[14:10 - 14:11]
// CHECK: local-symbols.m:14:10: IntegerLiteral= Extent=[14:10 - 14:11]
// CHECK: local-symbols.m:20:11: ObjCProtocolDecl=Prot8380046:20:11 (Definition) Extent=[20:1 - 21:5]
// CHECK: local-symbols.m:23:12: ObjCInterfaceDecl=R8380046:23:12 Extent=[23:1 - 24:5]
// CHECK: local-symbols.m:26:12: ObjCCategoryDecl=:26:12 Extent=[26:1 - 27:5]
// CHECK: local-symbols.m:26:12: ObjCClassRef=R8380046:23:12 Extent=[26:12 - 26:20]
// CHECK: local-symbols.m:26:25: ObjCProtocolRef=Prot8380046:20:11 Extent=[26:25 - 26:36]

