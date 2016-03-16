// RUN: %clang_cc1 -fobjc-runtime=macosx-10.10.0 -emit-llvm -o - %s -fno-objc-convert-messages-to-runtime-calls | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.10.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.9.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.10.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSGS
// Make sure we don't do calls to retain/release when using GC.
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.10.0 -emit-llvm -o - %s -fobjc-gc | FileCheck %s --check-prefix=GC
// RUN: %clang_cc1 -fobjc-runtime=ios-8.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=ios-7.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSGS
// Note: This line below is for tvos for which the driver passes through to use the ios9.0 runtime.
// RUN: %clang_cc1 -fobjc-runtime=ios-9.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=watchos-2.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS

@interface NSObject
+ (id)alloc;
+ (id)alloc2;
- (id)init;
- (id)retain;
- (void)release;
- (id)autorelease;
@end

@interface NSString : NSObject
+ (void)retain_self;
- (void)retain_super;
@end

// CHECK-LABEL: define {{.*}}void @test1
void test1(id x) {
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_alloc}}
  // CALLS: {{call.*@objc_retain}}
  // CALLS: {{call.*@objc_release}}
  // CALLS: {{call.*@objc_autorelease}}
  // GC: {{call.*@objc_alloc}}
  // GC: {{call.*@objc_msgSend}}
  // GC: {{call.*@objc_msgSend}}
  // GC: {{call.*@objc_msgSend}}
  [NSObject alloc];
  [x retain];
  [x release];
  [x autorelease];
}

// CHECK-LABEL: define {{.*}}void @test2
void test2() {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  // GC: {{call.*@objc_msgSend}}
  // Make sure alloc has the correct name and number of types.
  [NSObject alloc2];
}

@class A;
@interface B
+ (A*) alloc;
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
  // CALLS: {{call.*@objc_autorelease}}
  // CALLS-NEXT: bitcast i8*
  // CALLS-NEXT: ret
  return [b autorelease];
}

@interface C
- (float) retain;
@end

// Make sure we use a message and not a call as the return type is
// not a pointer type.
// CHECK-LABEL: define {{.*}}void @test_cannot_message_return_float
float test_cannot_message_return_float(C *c) {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  // GC: {{call.*@objc_msgSend}}
  return [c retain];
}

@implementation NSString

// Make sure we can convert a message to a dynamic receiver to a call
// CHECK-LABEL: define {{.*}}void @retain_self
+ (void)retain_self {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_retain}}
  // GC: {{call.*@objc_msgSend}}
  [self retain];
}

// Make sure we never convert a message to super to a call
// CHECK-LABEL: define {{.*}}void @retain_super
- (void)retain_super {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  // GC: {{call.*@objc_msgSend}}
  [super retain];
}

@end


