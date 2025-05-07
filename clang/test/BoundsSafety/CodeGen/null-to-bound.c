

// RUN: %clang_cc1 -O0 -triple arm64 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -triple arm64 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int main() {
    int *__bidi_indexable ptrBound = 0;
    int *__indexable ptrArray = 0;

    return 0;
}

// CHECK: [[BOUND_STRUCT:%.*]] = type { ptr, ptr, ptr }
// CHECK: [[ARRAY_STRUCT:%.*]] = type { ptr, ptr }
// CHECK: [[PTR_BOUND:%.*]] = alloca [[BOUND_STRUCT]], align 8
// CHECK: [[PTR_ARRAY:%.*]] = alloca [[ARRAY_STRUCT]], align 8
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[PTR_BOUND]], i8 0, i64 24, i1 false)
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[PTR_ARRAY]], i8 0, i64 16, i1 false)
