// RUN: %clang_cc1 -triple arm64-apple-darwin    -S -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -S -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-darwin    -fobjc-arc -S -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,ARC
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -S -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,ARC

struct stret { int x[100]; };
struct stret one = {{1}};

@interface Test
+(struct stret) method;
+(struct stret) methodConsuming:(id __attribute__((ns_consumed)))consumed;
@end

void foo(id o, id p) {
  [o method];
  // CHECK: @llvm.lifetime.start
  // CHECK: call void bitcast {{.*}} @objc_msgSend
  // CHECK: @llvm.lifetime.end
  // CHECK-NOT: call void @llvm.memset

  [o methodConsuming:p];
  // ARC: [[T0:%.*]] = icmp eq i8*
  // ARC: br i1 [[T0]]

  // CHECK: @llvm.lifetime.start
  // CHECK: call void bitcast {{.*}} @objc_msgSend
  // CHECK: @llvm.lifetime.end
  // ARC: br label

  // ARC: call void @llvm.objc.release
  // ARC: br label

  // CHECK-NOT: call void @llvm.memset
}
