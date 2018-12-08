// RUN: %clang_cc1 -fobjc-runtime=macosx-10.10.0 -emit-llvm -o - %s -fno-objc-convert-messages-to-runtime-calls | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.10.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=macosx-10.9.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=macosx-fragile-10.10.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSGS
// RUN: %clang_cc1 -fobjc-runtime=ios-8.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=ios-7.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=MSGS
// Note: This line below is for tvos for which the driver passes through to use the ios9.0 runtime.
// RUN: %clang_cc1 -fobjc-runtime=ios-9.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS
// RUN: %clang_cc1 -fobjc-runtime=watchos-2.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=CALLS

#define nil (id)0

@interface NSObject
+ (id)alloc;
+ (id)allocWithZone:(void*)zone;
+ (id)alloc2;
@end

// CHECK-LABEL: define {{.*}}void @test1
void test1(id x) {
  // MSGS: {{call.*@objc_msgSend}}
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_alloc}}
  // CALLS: {{call.*@objc_allocWithZone}}
  [NSObject alloc];
  [NSObject allocWithZone:nil];
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
+ (A*)allocWithZone:(void*)zone;
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


@interface C
+ (id)allocWithZone:(int)intArg;
@end

// Make sure we only accept pointer types
// CHECK-LABEL: define {{.*}}void @test_allocWithZone_int
C* test_allocWithZone_int() {
  // MSGS: {{call.*@objc_msgSend}}
  // CALLS: {{call.*@objc_msgSend}}
  return [C allocWithZone:3];
}

