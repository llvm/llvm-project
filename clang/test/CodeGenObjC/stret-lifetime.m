// RUN: %clang_cc1 -triple arm64-apple-darwin    -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-darwin    -fobjc-arc -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,ARC
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -emit-llvm -o - -O2 -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,ARC

struct stret { int x[100]; };
struct stret one = {{1}};

@interface Test
+(struct stret) method;
+(struct stret) methodConsuming:(id __attribute__((ns_consumed)))consumed;
@end

void foo(id o, id p) {
  [o method];
  // CHECK: @llvm.lifetime.start
  // CHECK: call void @objc_msgSend
  // CHECK: @llvm.lifetime.end
  // CHECK-NOT: call void @llvm.memset

  [o methodConsuming:p];
  // ARC: [[T0:%.*]] = icmp eq ptr
  // ARC: br i1 [[T0]]

  // CHECK: @llvm.lifetime.start
  // CHECK: call void @objc_msgSend
  // CHECK: @llvm.lifetime.end
  // ARC: br label

  // ARC: call void @llvm.objc.release
  // ARC: br label

  // CHECK-NOT: call void @llvm.memset
}
