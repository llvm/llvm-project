// RUN: %clang_cc1 %s -emit-llvm -fobjc-runtime=macosx-10.6.0 -triple x86_64-apple-macosx10.6.0 -o - | FileCheck %s
// rdar://11858187

@interface I
// void objc_setProperty_nonatomic(id self, SEL _cmd, id newValue, ptrdiff_t offset);
// objc_setProperty(..., NO, NO) 
@property (nonatomic, retain) id nonatomicProperty;

// void objc_setProperty_nonatomic_copy(id self, SEL _cmd, id newValue, ptrdiff_t offset);
// objc_setProperty(..., NO, YES)
@property (nonatomic, copy) id nonatomicPropertyCopy;

// void objc_setProperty_atomic(id self, SEL _cmd, id newValue, ptrdiff_t offset);
// objc_setProperty(..., YES, NO)
@property (retain) id atomicProperty;

// void objc_setProperty_atomic_copy(id self, SEL _cmd, id newValue, ptrdiff_t offset);
// objc_setProperty(..., YES, YES)
@property (copy) id atomicPropertyCopy;
@end

@implementation I
@synthesize nonatomicProperty;
@synthesize nonatomicPropertyCopy;
@synthesize atomicProperty;
@synthesize atomicPropertyCopy;
@end

// CHECK-NOT: call void @objc_setProperty_nonatomic
// CHECK-NOT: call void @objc_setProperty_nonatomic_copy
// CHECK-NOT: call void @objc_setProperty_atomic
// CHECK-NOT: call void @objc_setProperty_atomic_copy
