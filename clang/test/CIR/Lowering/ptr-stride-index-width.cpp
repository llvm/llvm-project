// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

const unsigned char *f(const unsigned char *p, int n) {
  const int k = -n;
  return p + k;
}

// CHECK-LABEL: define {{.*}} ptr @_Z1fPKhi(
// CHECK: [[N:%.*]] = load i32, ptr {{.*}}, align 4
// CHECK: [[NEG:%.*]] = sub nsw i32 0, [[N]]
// CHECK: store i32 [[NEG]], ptr {{.*}}, align 4
// CHECK: [[EXT:%.*]] = sext i32 [[N]] to i64
// CHECK: [[WIDENED:%.*]] = sub i64 0, [[EXT]]
// CHECK: getelementptr i8, ptr {{.*}}, i64 [[WIDENED]]
