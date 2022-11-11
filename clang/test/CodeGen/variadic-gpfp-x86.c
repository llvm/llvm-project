// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

struct Bar {
 float f1;
 float f2;
 unsigned u;
};

struct Bar foo(__builtin_va_list ap) {
  return __builtin_va_arg(ap, struct Bar);
// CHECK: [[FPOP:%.*]] = getelementptr inbounds %struct.__va_list_tag, ptr {{.*}}, i32 0, i32 1
// CHECK: [[FPO:%.*]] = load i32, ptr [[FPOP]]
// CHECK: [[FPVEC:%.*]] = getelementptr i8, ptr {{.*}}, i32 [[FPO]]
}
