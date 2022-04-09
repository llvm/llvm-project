// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

struct Bar {
 float f1;
 float f2;
 unsigned u;
};

struct Bar foo(__builtin_va_list ap) {
  return __builtin_va_arg(ap, struct Bar);
// CHECK: [[FPOP:%.*]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* {{.*}}, i32 0, i32 1
// CHECK: [[FPO:%.*]] = load i32, i32* [[FPOP]]
// CHECK: [[FPVEC:%.*]] = getelementptr i8, i8* {{.*}}, i32 [[FPO]]
// CHECK: bitcast i8* [[FPVEC]] to <2 x float>*
}
