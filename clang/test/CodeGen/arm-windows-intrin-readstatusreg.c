// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_ReadStatusReg --implicit-check-not=@__prefetch

// _ReadStatusReg reads APSR (0) or SPSR (1). __prefetch is a pld.

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

int test_ReadStatusReg_apsr(void) { return _ReadStatusReg(0); }
// CHECK-LABEL: @test_ReadStatusReg_apsr(
// CHECK: call i32 @llvm.read_register.i32(metadata [[APSR:![0-9]+]])

int test_ReadStatusReg_spsr(void) { return _ReadStatusReg(1); }
// CHECK-LABEL: @test_ReadStatusReg_spsr(
// CHECK: call i32 @llvm.read_register.i32(metadata [[SPSR:![0-9]+]])

void test_prefetch(const void *p) { __prefetch(p); }
// CHECK-LABEL: @test_prefetch(
// CHECK: call void @llvm.prefetch.p0(ptr %{{.*}}, i32 0, i32 3, i32 1)

// CHECK-DAG: [[APSR]] = !{!"apsr"}
// CHECK-DAG: [[SPSR]] = !{!"spsr"}
