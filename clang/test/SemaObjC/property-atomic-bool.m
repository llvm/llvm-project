// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -ast-dump "%s" | FileCheck %s

// CHECK: TypedefDecl {{.*}} referenced AtomicBool '_Atomic(_Bool)'
// CHECK:  AtomicType {{.*}} '_Atomic(_Bool)'
// CHECK:   BuiltinType {{.*}} '_Bool'
// CHECK: ObjCInterfaceDecl {{.*}} A0
// CHECK:  ObjCPropertyDecl {{.*}} p '_Atomic(_Bool)' {{.*}} nonatomic
// CHECK:  ObjCMethodDecl {{.*}} implicit - p '_Bool'
// CHECK:  ObjCMethodDecl {{.*}} implicit - setP: 'void'
// CHECK:   ParmVarDecl {{.*}} p '_Bool'
// CHECK: ObjCInterfaceDecl {{.*}} A1
// CHECK:  ObjCPropertyDecl {{.*}} p 'AtomicBool':'_Atomic(_Bool)' {{.*}} nonatomic
// CHECK:  ObjCMethodDecl {{.*}} implicit - p '_Bool'
// CHECK:  ObjCMethodDecl {{.*}} implicit - setP: 'void'
// CHECK:   ParmVarDecl {{.*}} p '_Bool'
// CHECK: ObjCInterfaceDecl {{.*}} A2
// CHECK:  ObjCIvarDecl {{.*}} p '_Atomic(_Bool)' protected
// CHECK:  ObjCPropertyDecl {{.*}} p '_Atomic(_Bool)'
// CHECK:  ObjCMethodDecl {{.*}} implicit - p '_Bool'
// CHECK:  ObjCMethodDecl {{.*}} implicit - setP: 'void'
// CHECK:   ParmVarDecl {{.*}} p '_Bool'
// CHECK: ObjCInterfaceDecl {{.*}} A3
// CHECK:  ObjCIvarDecl {{.*}} p 'AtomicBool':'_Atomic(_Bool)' protected
// CHECK:  ObjCPropertyDecl {{.*}} p 'AtomicBool':'_Atomic(_Bool)'
// CHECK:  ObjCMethodDecl {{.*}} implicit - p '_Bool'
// CHECK:  ObjCMethodDecl {{.*}} implicit - setP: 'void'
// CHECK:   ParmVarDecl {{.*}} p '_Bool'

typedef _Atomic(_Bool) AtomicBool;

@interface A0
@property(nonatomic) _Atomic(_Bool) p;
@end
@implementation A0
@end

@interface A1
@property(nonatomic) AtomicBool p;
@end
@implementation A1
@end

@interface A2 {
  _Atomic(_Bool) p;
}
@property _Atomic(_Bool) p;
@end

@implementation A2
@synthesize p;
@end

@interface A3 {
  AtomicBool p;
}
@property AtomicBool p;
@end

@implementation A3
@synthesize p;
@end
