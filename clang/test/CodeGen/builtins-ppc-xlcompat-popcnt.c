// REQUIRES: powerpc-registered-target.
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern unsigned int ui;
extern unsigned long long ull;

// CHECK-LABEL: @test_builtin_ppc_poppar4(
// CHECK:         [[TMP0:%.*]] = load i32, i32* @ui, align 4
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* @ui, align 4
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ctpop.i32(i32 [[TMP1]])
// CHECK-NEXT:    [[TMP3:%.*]] = and i32 [[TMP2]], 1
// CHECK-NEXT:    ret i32 [[TMP3]]
//
int test_builtin_ppc_poppar4() {
 return __builtin_ppc_poppar4(ui);
}

// CHECK-LABEL: @test_builtin_ppc_poppar8(
// CHECK:         [[TMP0:%.*]] = load i64, i64* @ull, align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load i64, i64* @ull, align 8
// CHECK-NEXT:    [[TMP2:%.*]] = call i64 @llvm.ctpop.i64(i64 [[TMP1]])
// CHECK-NEXT:    [[TMP3:%.*]] = and i64 [[TMP2]], 1
// CHECK-NEXT:    [[CAST:%.*]] = trunc i64 [[TMP3]] to i32
// CHECK-NEXT:    ret i32 [[CAST]]
//
int test_builtin_ppc_poppar8() {
 return __builtin_ppc_poppar8(ull);
}
