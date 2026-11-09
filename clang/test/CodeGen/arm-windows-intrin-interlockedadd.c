// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=19.00 \
// RUN:         -triple thumbv7--windows -emit-llvm %s -o - \
// RUN:         | FileCheck %s --implicit-check-not=@_InterlockedAdd

// _InterlockedAdd and its variants return the new value.

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;
#include <intrin.h>

long test_InterlockedAdd(long volatile *p, long v) { return _InterlockedAdd(p, v); }
// CHECK-LABEL: @test_InterlockedAdd(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[V:%.*]] seq_cst, align 4
// CHECK: [[NEW:%.*]] = add i32 [[OLD]], [[V]]
// CHECK: ret i32 [[NEW]]

long test_InterlockedAdd_acq(long volatile *p, long v) { return _InterlockedAdd_acq(p, v); }
// CHECK-LABEL: @test_InterlockedAdd_acq(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[V:%.*]] acquire, align 4
// CHECK: add i32 [[OLD]], [[V]]

long test_InterlockedAdd_nf(long volatile *p, long v) { return _InterlockedAdd_nf(p, v); }
// CHECK-LABEL: @test_InterlockedAdd_nf(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[V:%.*]] monotonic, align 4
// CHECK: add i32 [[OLD]], [[V]]

long test_InterlockedAdd_rel(long volatile *p, long v) { return _InterlockedAdd_rel(p, v); }
// CHECK-LABEL: @test_InterlockedAdd_rel(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i32 [[V:%.*]] release, align 4
// CHECK: add i32 [[OLD]], [[V]]

__int64 test_InterlockedAdd64(__int64 volatile *p, __int64 v) { return _InterlockedAdd64(p, v); }
// CHECK-LABEL: @test_InterlockedAdd64(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[V:%.*]] seq_cst, align 8
// CHECK: [[NEW:%.*]] = add i64 [[OLD]], [[V]]
// CHECK: ret i64 [[NEW]]

__int64 test_InterlockedAdd64_acq(__int64 volatile *p, __int64 v) { return _InterlockedAdd64_acq(p, v); }
// CHECK-LABEL: @test_InterlockedAdd64_acq(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[V:%.*]] acquire, align 8
// CHECK: add i64 [[OLD]], [[V]]

__int64 test_InterlockedAdd64_nf(__int64 volatile *p, __int64 v) { return _InterlockedAdd64_nf(p, v); }
// CHECK-LABEL: @test_InterlockedAdd64_nf(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[V:%.*]] monotonic, align 8
// CHECK: add i64 [[OLD]], [[V]]

__int64 test_InterlockedAdd64_rel(__int64 volatile *p, __int64 v) { return _InterlockedAdd64_rel(p, v); }
// CHECK-LABEL: @test_InterlockedAdd64_rel(
// CHECK: [[OLD:%.*]] = atomicrmw add ptr %{{.*}}, i64 [[V:%.*]] release, align 8
// CHECK: add i64 [[OLD]], [[V]]
