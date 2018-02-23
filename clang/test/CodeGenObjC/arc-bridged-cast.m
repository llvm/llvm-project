// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -O2 -disable-llvm-passes -o - %s | FileCheck %s

typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

@interface NSString
@end

CFTypeRef CFCreateSomething(void);
CFStringRef CFCreateString(void);
CFTypeRef CFGetSomething(void);
CFStringRef CFGetString(void);

id CreateSomething(void);
NSString *CreateNSString(void);

// CHECK-LABEL: define void @bridge_transfer_from_cf
void bridge_transfer_from_cf(int *i) {
  // CHECK: store i32 7
  *i = 7;
  // CHECK: call i8* @CFCreateSomething()
  id obj1 = (__bridge_transfer id)CFCreateSomething();
  // CHECK-NOT: retain
  // CHECK: store i32 11
  *i = 11;
  // CHECK: call i8* @CFCreateSomething()
  // CHECK-NOT: retain
  // CHECK: store i32 13
  (void)(__bridge_transfer id)CFCreateSomething(), *i = 13;
  // CHECK: call void @objc_release
  // CHECK: store i32 17
  *i = 17;
  // CHECK: call void @objc_release
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @bridge_from_cf
void bridge_from_cf(int *i) {
  // CHECK: store i32 7
  *i = 7;
  // CHECK: call i8* @CFCreateSomething()
  id obj1 = (__bridge id)CFCreateSomething();
  // CHECK: objc_retainAutoreleasedReturnValue
  // CHECK: store i32 11
  *i = 11;
  // CHECK: call i8* @CFCreateSomething()
  // CHECK-NOT: release
  // CHECK: store i32 13
  (void)(__bridge id)CFCreateSomething(), *i = 13;
  // CHECK: store i32 17
  *i = 17;
  // CHECK: call void @objc_release
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @bridge_retained_of_cf
void bridge_retained_of_cf(int *i) {
  *i = 7;
  // CHECK: call i8* @CreateSomething()
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething();
  // CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
  // CHECK: store i32 11
  *i = 11;
  // CHECK: call i8* @CreateSomething()
  (__bridge_retained CFTypeRef)CreateSomething(), *i = 13;
  // CHECK-NEXT: call i8* @objc_retainAutoreleasedReturnValue
  // CHECK: store i32 13
  // CHECK: store i32 17
  *i = 17;
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define void @bridge_of_cf
void bridge_of_cf(int *i) {
  // CHECK: store i32 7
  *i = 7;
  // CHECK: call void @llvm.lifetime.start
  // CHECK-NEXT: call i8* @CreateSomething()
  CFTypeRef cf1 = (__bridge CFTypeRef)CreateSomething();
  // CHECK-NOT: retain
  // CHECK: store i32 11
  *i = 11;
  // CHECK: call i8* @CreateSomething
  (__bridge CFTypeRef)CreateSomething(), *i = 13;
  // CHECK: store i32 13
  // CHECK-NOT: release
  // CHECK: store i32 17
  *i = 17;
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.lifetime.end
  // CHECK-NEXT: ret void
}

