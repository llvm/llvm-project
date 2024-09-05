// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fobjc-runtime=gnustep-2.2 -fobjc-dispatch-method=non-legacy -emit-llvm -o - %s | FileCheck %s

// Pass and return for type size <= 8 bytes.
struct S1 {
  int a[2];
};

// Pass and return hfa <= 8 bytes
struct F1 {
  float a[2];
};

// Pass and return for type size > 16 bytes.
struct S2 {
  int a[5];
};

// Pass and return aggregate (of size < 16 bytes) with non-trivial destructor.
// Sret and inreg: Returned in x0
struct S3 {
  int a[3];
  ~S3();
};
S3::~S3() {
}


@interface MsgTest { id isa; } @end
@implementation MsgTest 
- (S1) smallS1 {
  S1 x;
  x.a[0] = 0;
  x.a[1] = 1;
  return x;

}
- (F1) smallF1 {
  F1 x;
  x.a[0] = 0.2f;
  x.a[1] = 0.5f;
  return x;
}
- (S2) stretS2 {
  S2 x;
  for (int i = 0; i < 5; i++) {
    x.a[i] = i;
  }
  return x;
}
- (S3) stretInRegS3 {
  S3 x;
  for (int i = 0; i < 3; i++) {
    x.a[i] = i;
  }
  return x;
}
+ (S3) msgTestStretInRegS3 {
  S3 x;
  for (int i = 0; i < 3; i++) {
    x.a[i] = i;
  }
  return x;
}
@end

void test0(MsgTest *t) {
    // CHECK: call {{.*}} @objc_msgSend
    S1 ret = [t smallS1];
    // CHECK: call {{.*}} @objc_msgSend
    F1 ret2 = [t smallF1];
    // CHECK: call {{.*}} @objc_msgSend_stret
    S2 ret3 = [t stretS2];
    // CHECK: call {{.*}} @objc_msgSend_stret2
    S3 ret4 = [t stretInRegS3];
    // CHECK: call {{.*}} @objc_msgSend_stret2
    S3 ret5 = [MsgTest msgTestStretInRegS3];
}
