// RUN: %clang -O0 --target=powerpcle-unknown-unknown -mcpu=e500 -mhard-float -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -O0 --target=powerpc-unknown-unknown -mcpu=e500 -mhard-float -S -emit-llvm %s -o - | FileCheck %s

#include <stdarg.h>
volatile double f;
void testfunc(int c, ...) {
    va_list vargs;
    va_start(vargs, c);
    f = va_arg(vargs, double);
    va_end(vargs);
}

int main() {
    testfunc(0, 25.5);
}

// CHECK: %struct.__va_list_tag = type { i8, i8, i16, ptr, ptr }

// CHECK: define dso_local void @testfunc(i32 noundef %[[NONVAR_ARG:[.-z]+]], ...) #{{[0-9]+}} {
// CHECK:   %[[NONVAR_ARG_ADDR:[.-z]+]] = alloca i32, align 4
// CHECK:   %[[VARGS:[.-z]+]] = alloca [1 x %struct.__va_list_tag], align 4
// CHECK:   store i32 %[[NONVAR_ARG]], ptr %[[NONVAR_ARG_ADDR]], align 4
// CHECK:   %[[ARRAYDECAY:[.-z]+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VARGS]], i32 0, i32 0
// CHECK:   call void @llvm.va_start.p0(ptr %[[ARRAYDECAY]])
// CHECK:   %[[ARRAYDECAY1:[.-z]+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VARGS]], i32 0, i32 0
// CHECK:   %[[GPR:[.-z]+]] = getelementptr inbounds {{(nuw )?}}%struct.__va_list_tag, ptr %[[ARRAYDECAY1]], i32 0, i32 0
// CHECK:   %[[NUM_USED_REGS:[.-z]+]] = load i8, ptr %[[GPR]], align 4
// CHECK:   %[[UR_PLUS_1:[0-9]+]] = add i8 %[[NUM_USED_REGS]], 1
// CHECK:   %[[UR_ALIGN:[0-9]+]] = and i8 %[[UR_PLUS_1]], -2
// CHECK:   %[[COND:[.-z]+]] = icmp ult i8 %[[UR_ALIGN]], 8
// CHECK:   br i1 %[[COND]], label %[[using_regs:[.-z]+]], label %[[using_overflow:[.-z]+]]

// CHECK: [[using_regs]]:{{.*}}
// CHECK:  %[[GPRS_PTR_PTR:[0-9]+]]  = getelementptr inbounds {{(nuw )?}}%struct.__va_list_tag, ptr %[[ARRAYDECAY1]], i32 0, i32 4
// CHECK:  %[[GPRS_PTR:[0-9]+]] = load ptr, ptr %[[GPRS_PTR_PTR]], align 4
// CHECK:  %[[UR_ALIGN_X4:[0-9]+]]  = mul i8 %[[UR_ALIGN]], 4
// CHECK:  %[[ARG_PTR:[0-9]+]] = getelementptr inbounds i8, ptr %[[GPRS_PTR]], i8 %[[UR_ALIGN_X4]]
// CHECK:  %[[UR_NEW:[0-9]+]]  = add i8 %[[UR_ALIGN]], 2
// CHECK:  store i8 %[[UR_NEW]], ptr %[[GPR]], align 4
// CHECK:  br label %[[cont:[.-z]+]]

// don't care about 'using overflow' code, it does not depend on ABI

// CHECK: [[cont]]:{{.*}}
// CHECK:   %[[VAARG_ADDR:[.-z]+]] = phi ptr [ %[[ARG_PTR]], %[[using_regs]] ], [ %{{[.-z]+}}, %[[using_overflow]] ]
// CHECK:   [[ARG_VAL:[0-9]+]] = load double, ptr %[[VAARG_ADDR]], align 4
// CHECK:   store volatile double %[[ARG_VAL]], ptr @f, align 8
// CHECK:   %[[ARRAYDECAY2:[.-z]+]] = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %[[VARGS]], i32 0, i32 0
// CHECK:   call void @llvm.va_end.p0(ptr %[[ARRAYDECAY2]])
// CHECK:   ret void
// CHECK: }

// CHECK: define dso_local i32 @main() #0 {
// CHECK:   call void (i32, ...) @testfunc(i32 noundef 0, double noundef 2.550000e+01)
// CHECK: }
