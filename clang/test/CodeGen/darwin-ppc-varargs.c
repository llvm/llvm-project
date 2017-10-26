// RUN: %clang_cc1 -triple powerpc-apple-macosx10.5.0 -target-feature +altivec -Os -emit-llvm -o - %s | FileCheck %s

int f(__builtin_va_list args) { return __builtin_va_arg(args, int); }

// CHECK: @f(i8* {{.*}}[[PARAM:%[a-zA-Z0-9]+]])
// CHECK: [[BITCAST:%[0-9]+]] = bitcast i8* [[PARAM]] to i32*
// CHECK: [[VALUE:%[0-9]+]] = load i32, i32* [[BITCAST]], align 4
// CHECK: ret i32 [[VALUE]]

void h(vector int);
int g(__builtin_va_list args) {
  int i = __builtin_va_arg(args, int);
  h(__builtin_va_arg(args, vector int));
  int j = __builtin_va_arg(args, int);
  return i + j;
}

// CHECK: @g(i8* {{.*}}[[PARAM:%[a-zA-Z0-9]+]])
// CHECK: [[NEXT:%[-_.a-zA-Z0-9]+]] = getelementptr inbounds i8, i8* [[PARAM]], i32 4
// CHECK: [[BITCAST:%[0-9]+]] = bitcast i8* [[PARAM]] to i32*
// CHECK: [[LOAD:%[0-9]+]] = load i32, i32* [[BITCAST]], align 4
// CHECK: [[PTRTOINT:%[0-9]+]] = ptrtoint i8* [[NEXT]] to i32
// CHECK: [[ADD:%[0-9]+]] = add i32 [[PTRTOINT]], 15
// CHECK: [[AND:%[0-9]+]] = and i32 [[ADD]], -16
// CHECK: [[INTTOPTR:%[0-9]+]] = inttoptr i32 [[AND]] to <4 x i32>*
// CHECK: [[ARG:%[0-9]]] = load <4 x i32>, <4 x i32>* [[INTTOPTR]], align 16
// CHECK: call void @h(<4 x i32> [[ARG]]

