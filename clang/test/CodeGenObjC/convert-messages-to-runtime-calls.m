// RUN: %clang_cc1 -fobjc-runtime=macosx-10.10.0 -emit-llvm -o - %s -fno-objc-convert-messages-to-runtime-calls -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.10.0 -emit-llvm -o - %s -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.9.0 -emit-llvm -o - %s -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.10.0 -emit-llvm -o - %s -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=ios-8.0 -emit-llvm -o - %s -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=ios-7.0 -emit-llvm -o - %s -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=MSGS
// Note: This line below is for tvos for which the driver passes through to use the ios9.0 runtime.
// RUN: %clang_cc1 -fobjc-runtime=ios-9.0 -emit-llvm -o - %s -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=watchos-2.0 -emit-llvm -o - %s -fobjc-exceptions -fexceptions | FileCheck %s --check-prefix=CALLS

#define nil (id)0

@interface NSObject
+ (id)alloc;
+ (id)allocWithZone:(void*)zone;
+ (id)alloc2;
- (id)retain;
- (void)release;
- (id)autorelease;
@end

// CHECK-LABEL: define {{.*}}void @test1
void test1(id x) {
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_alloc}}
  // CALLS: {{call.*@objc_allocWithZone}}
  // CALLS: {{call.*@objc_retain}}
  // CALLS: {{call.*@objc_release}}
  // CALLS: {{tail call.*@objc_autorelease}}
  [NSObject alloc];
  [NSObject allocWithZone:nil];
  [x retain];
  [x release];
  [x autorelease];
}

// CHECK-LABEL: define {{.*}}void @check_invoke
void check_invoke() {
  // MSGS: {{invoke.*@objc_msgSend}}
  // MSGS: {{invoke.*@objc_msgSend}}
  // CALLS: {{invoke.*@objc_alloc}}
  // CALLS: {{invoke.*@objc_allocWithZone}}
  @try {
    [NSObject alloc];
    [NSObject allocWithZone:nil];
  } @catch (...) {
  }
}

// CHECK-LABEL: define {{.*}}void @test2
void test2(void* x) {
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  [NSObject alloc2];
  [NSObject allocWithZone:(void*)-1];
  [NSObject allocWithZone:x];
}

@class A;
@interface B
+ (A*) alloc;
+ (A*) allocWithZone:(void*)zone;
- (A*) alloc;
- (A*) allocWithZone:(void*)zone;
- (A*) retain;
- (A*) autorelease;
@end

// Make sure we get a bitcast on the return type as the
// call will return i8* which we have to cast to A*
// CHECK-LABEL: define {{.*}}void @test_alloc_class_ptr
A* test_alloc_class_ptr() {
  // CALLS: {{call.*@objc_alloc}}
  // CALLS-NEXT: bitcast i8*
  // CALLS-NEXT: ret
  return [B alloc];
}

// Make sure we get a bitcast on the return type as the
// call will return i8* which we have to cast to A*
// CHECK-LABEL: define {{.*}}void @test_alloc_class_ptr
A* test_allocWithZone_class_ptr() {
  // CALLS: {{call.*@objc_allocWithZone}}
  // CALLS-NEXT: bitcast i8*
  // CALLS-NEXT: ret
  return [B allocWithZone:nil];
}

// Only call objc_alloc on a Class, not an instance
// CHECK-LABEL: define {{.*}}void @test_alloc_instance
void test_alloc_instance(A *a) {
  // CALLS: {{call.*@objc_alloc}}
  // CALLS: {{call.*@objc_allocWithZone}}
  // CALLS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  [A alloc];
  [A allocWithZone:nil];
  [a alloc];
  [a allocWithZone:nil];
}

// Make sure we get a bitcast on the return type as the
// call will return i8* which we have to cast to A*
// CHECK-LABEL: define {{.*}}void @test_retain_class_ptr
A* test_retain_class_ptr(B *b) {
  // CALLS: {{call.*@objc_retain}}
  // CALLS-NEXT: bitcast i8*
  // CALLS-NEXT: ret
  return [b retain];
}

// Make sure we get a bitcast on the return type as the
// call will return i8* which we have to cast to A*
// CHECK-LABEL: define {{.*}}void @test_autorelease_class_ptr
A* test_autorelease_class_ptr(B *b) {
  // CALLS: {{tail call.*@objc_autorelease}}
  // CALLS-NEXT: bitcast i8*
  // CALLS-NEXT: ret
  return [b autorelease];
}


@interface C
+ (id)allocWithZone:(int)intArg;
- (float) retain;
@end

// Make sure we only accept pointer types
// CHECK-LABEL: define {{.*}}void @test_allocWithZone_int
C* test_allocWithZone_int() {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  return [C allocWithZone:3];
}

// Make sure we use a message and not a call as the return type is
// not a pointer type.
// CHECK-LABEL: define {{.*}}void @test_cannot_message_return_float
float test_cannot_message_return_float(C *c) {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  return [c retain];
}

@interface TestSelf
+ (instancetype)alloc;
+ (instancetype)allocWithZone:(void*)zone;
+ (id)classMeth;
- (id)instanceMeth;
@end

@implementation TestSelf
// CHECK-LABEL: define internal i8* @"\01+[TestSelf classMeth]"(
+ (id)classMeth {
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_allocWithZone\(}}
  // CALLS: {{call.*@objc_alloc\(}}
  [self allocWithZone:nil];
  return [self alloc];
}
// CHECK-LABEL: define internal i8* @"\01-[TestSelf instanceMeth]"(
- (id)instanceMeth {
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  [self allocWithZone:nil];
  return [self alloc];
}
@end

@interface NSString : NSObject
+ (void)retain_self;
- (void)retain_super;
@end

@implementation NSString

// Make sure we can convert a message to a dynamic receiver to a call
// CHECK-LABEL: define {{.*}}void @retain_self
+ (void)retain_self {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_retain}}
  [self retain];
}

// Make sure we never convert a message to super to a call
// CHECK-LABEL: define {{.*}}void @retain_super
- (void)retain_super {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  [super retain];
}

@end

@class Ety;

// CHECK-LABEL: define {{.*}}void @testException_release
void testException_release(NSObject *a) {
  // MSGS: {{invoke.*@objc_msgSend}}
  // CALLS: invoke{{.*}}void @objc_release(i8* %
  @try {
    [a release];
  } @catch (Ety *e) {
  }
}

// CHECK-LABEL: define {{.*}}void @testException_autorelease
void testException_autorelease(NSObject *a) {
  @try {
    // MSGS: {{invoke.*@objc_msgSend}}
    // CALLS: invoke{{.*}}objc_autorelease(i8* %
    [a autorelease];
  } @catch (Ety *e) {
  }
}

// CHECK-LABEL: define {{.*}}void @testException_retain
void testException_retain(NSObject *a) {
  @try {
    // MSGS: {{invoke.*@objc_msgSend}}
    // CALLS: invoke{{.*}}@objc_retain(i8* %
    [a retain];
  } @catch (Ety *e) {
  }
}


// CHECK-LABEL: define {{.*}}void @testException_alloc(
void testException_alloc() {
  @try {
    // MSGS: {{invoke.*@objc_msgSend}}
    // CALLS: invoke{{.*}}@objc_alloc(i8* %
    [A alloc];
  } @catch (Ety *e) {
  }
}

// CHECK-LABEL: define {{.*}}void @testException_allocWithZone
void testException_allocWithZone() {
  @try {
    // MSGS: {{invoke.*@objc_msgSend}}
    // CALLS: invoke{{.*}}@objc_allocWithZone(i8* %
    [A allocWithZone:nil];
  } @catch (Ety *e) {
  }
}
