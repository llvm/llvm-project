
// RUN: %clang_cc1 -O3  -fbounds-safety -triple arm64-apple-darwin -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int consume(char* __bidi_indexable d) __attribute__((optnone)) {
    return d[0];
}

// CHECK-LABEL: consume
// CHECK:  [[ADDR:%[-_.[:alnum:]]+]] = getelementptr inbounds nuw %"__bounds_safety::wide_ptr.bidi_indexable", ptr [[WIDE_PTR:%[-_.[:alnum:]]+]], i32 0, i32 0
// CHECK:  [[PTR:%[-_.[:alnum:]]+]] = load ptr, ptr [[ADDR]], align 8
// CHECK:  [[PTR_INDEXED:%[-_.[:alnum:]]+]] = getelementptr i8, ptr [[PTR]], i64 0
// CHECK:  [[UB_ADDR:%[-_.[:alnum:]]+]] = getelementptr inbounds nuw %"__bounds_safety::wide_ptr.bidi_indexable", ptr [[WIDE_PTR]], i32 0, i32 1
// CHECK:  [[UB:%[-_.[:alnum:]]+]] = load ptr, ptr [[UB_ADDR]], align 8
// CHECK:  [[LB_ADDR:%[-_.[:alnum:]]+]] = getelementptr inbounds nuw %"__bounds_safety::wide_ptr.bidi_indexable", ptr [[WIDE_PTR]], i32 0, i32 2
// CHECK:  [[LB:%[-_.[:alnum:]]+]] = load ptr, ptr [[LB_ADDR]], align 8
// CHECK:  [[CMP1:%[-_.[:alnum:]]+]] = icmp ult ptr [[PTR_INDEXED]], [[UB]]
// CHECK:  br i1 [[CMP1]], label %[[CONT1:[-_.[:alnum:]]+]], label %[[TRAP1:[-_.[:alnum:]]+]]

// CHECK: [[TRAP1]]:
// CHECK:  call void @llvm.ubsantrap(i8 25)

// CHECK: [[CONT1]]:
// CHECK:  [[CMP2:%[-_.[:alnum:]]+]] = icmp uge ptr [[PTR_INDEXED]], [[LB]]
// CHECK:  br i1 [[CMP2]], label %[[CONT2:[-_.[:alnum:]]+]], label %[[TRAP2:[-_.[:alnum:]]+]]

// CHECK: [[TRAP2]]:
// CHECK:  call void @llvm.ubsantrap(i8 25)

int main(void) {
    char *data = "ab";
    return consume(data);
}

// CHECK-LABEL: main
// CHECK-NOT: ubsantrap

