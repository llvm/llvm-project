// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK-LABEL: define noundef i1 {{.*}}fn{{.*}}(i1 noundef %x)
// CHECK: [[X:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[Y:%.*]] = zext i1 {{%.*}} to i32
// CHECK-NEXT: store i32 [[Y]], ptr [[X]], align 4
// CHECK-NEXT: [[Z:%.*]] = load i32, ptr [[X]], align 4
// CHECK-NEXT: [[L:%.*]] = trunc i32 [[Z]] to i1
// CHECK-NEXT: ret i1 [[L]]
bool fn(bool x) {
  return x;
}
