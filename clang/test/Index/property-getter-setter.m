@interface Foo
@property (assign,readwrite,getter=b,setter=c:) id a;
@property (assign,readonly,getter=e) id d;
@property (assign,readwrite) id f;
@end

// RUN: c-index-test -test-print-type-declaration %s | FileCheck %s
// CHECK: ObjCPropertyDecl=a:2:52 [getter,assign,readwrite,setter,] (getter=b) (setter=c:) [typedeclaration=id] [typekind=ObjCId]
// CHECK: ObjCPropertyDecl=d:3:41 [readonly,getter,assign,] (getter=e) [typedeclaration=id] [typekind=ObjCId]
// CHECK: ObjCPropertyDecl=f:4:33 [assign,readwrite,] [typedeclaration=id] [typekind=ObjCId]
