// RUN: %clang_cc1 -fblocks -ast-dump %s | FileCheck %s

__attribute__((swift_attr("@actor")))
@interface View
@end

// CHECK-LABEL: InterfaceDecl {{.*}} View
// CHECK-NEXT: SwiftAttrAttr {{.*}} "@actor"

#pragma clang attribute push(__attribute__((swift_attr("@sendable"))), apply_to=objc_interface)
@interface Contact
@end
#pragma clang attribute pop

// CHECK-LABEL: InterfaceDecl {{.*}} Contact
// CHECK-NEXT: SwiftAttrAttr {{.*}} "@sendable"

#define SWIFT_SENDABLE __attribute__((swift_attr("@Sendable")))

@interface InTypeContext
- (nullable id)test:(nullable SWIFT_SENDABLE id)obj SWIFT_SENDABLE;
@end

// CHECK-LABEL: InterfaceDecl {{.*}} InTypeContext
// CHECK-NEXT: MethodDecl {{.*}} - test: 'id _Nullable':'id'
// CHECK-NEXT: ParmVarDecl {{.*}} obj 'SWIFT_SENDABLE id _Nullable':'id'
// CHECK-NEXT: SwiftAttrAttr {{.*}} "@Sendable"

@interface Generic<T: SWIFT_SENDABLE id>
@end

// CHECK-LABEL: InterfaceDecl {{.*}} Generic
// CHECK-NEXT: TypeParamDecl {{.*}} T bounded 'SWIFT_SENDABLE id':'id'

typedef SWIFT_SENDABLE Generic<id> Alias;

// CHECK-LABEL: TypedefDecl {{.*}} Alias 'Generic<id>'
// CHECK-NEXT: ObjectType {{.*}} 'Generic<id>'
// CHECK-NEXT: SwiftAttrAttr {{.*}} "@Sendable"

SWIFT_SENDABLE
typedef struct {
  void *ptr;
} SendableStruct;

// CHECK-LABEL: TypedefDecl {{.*}} SendableStruct 'struct SendableStruct'
// CHECK: SwiftAttrAttr {{.*}} "@Sendable"

@interface TestAttrPlacementInBlock1
-(void) withHandler: (void (SWIFT_SENDABLE ^)(id)) handler;
@end

// CHECK-LABEL: ObjCInterfaceDecl {{.*}} TestAttrPlacementInBlock1
// CHECK: handler 'SWIFT_SENDABLE void (^)(id)':'void (^)(id)'

@interface TestAttrPlacementInBlock2
-(void) withHandler: (void (^ SWIFT_SENDABLE)(id)) handler;
@end

// CHECK-LABEL: ObjCInterfaceDecl {{.*}} TestAttrPlacementInBlock2
// CHECK: handler 'SWIFT_SENDABLE void (^)(id)':'void (^)(id)'
