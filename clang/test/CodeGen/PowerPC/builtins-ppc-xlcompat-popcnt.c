// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o - -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm %s -o - -target-cpu pwr7 | FileCheck %s

extern unsigned int ui;
extern unsigned long long ull;

// CHECK-LABEL: @test_builtin_ppc_poppar4(
// CHECK:         [[TMP2:%.*]] = call i32 @llvm.ctpop.i32(i32 {{.*}})
// CHECK-NEXT:    [[TMP3:%.*]] = and i32 [[TMP2]], 1
// CHECK-NEXT:    ret i32 [[TMP3]]
//
int test_builtin_ppc_poppar4() {
 return __builtin_ppc_poppar4(ui);
}

// CHECK-LABEL: @test_builtin_ppc_poppar8(
// CHECK:         [[TMP2:%.*]] = call i64 @llvm.ctpop.i64(i64 {{.*}})
// CHECK-NEXT:    [[TMP3:%.*]] = and i64 [[TMP2]], 1
// CHECK-NEXT:    [[CAST:%.*]] = trunc i64 [[TMP3]] to i32
// CHECK-NEXT:    ret i32 [[CAST]]
//
int test_builtin_ppc_poppar8() {
 return __builtin_ppc_poppar8(ull);
}

// CHECK-LABEL: @testcntlz4(
// CHECK:         [[TMP:%.*]] = call i32 @llvm.ctlz.i32(i32 {{%.*}}, i1 false)
// CHECK-NEXT:    ret i32 [[TMP]]
//
unsigned int testcntlz4(unsigned int value) {
  return __cntlz4(value);
}

// CHECK-LABEL: @testcntlz8(
// CHECK:         [[TMP:%.*]] = call i64 @llvm.ctlz.i64(i64 {{%.*}}, i1 false)
// CHECK-NEXT:    [[CAST:%.*]] = trunc i64 [[TMP]] to i32
// CHECK-NEXT:    ret i32 [[CAST]]
//
unsigned int testcntlz8(unsigned long long value) {
  return __cntlz8(value);
}

// CHECK-LABEL: @testcnttz4(
// CHECK:         [[TMP:%.*]] = call i32 @llvm.cttz.i32(i32 {{%.*}}, i1 false)
// CHECK-NEXT:    ret i32 [[TMP]]
//
unsigned int testcnttz4(unsigned int value) {
  return __cnttz4(value);
}

// CHECK-LABEL: @testcnttz8(
// CHECK:         [[TMP:%.*]] = call i64 @llvm.cttz.i64(i64 {{%.*}}, i1 false)
// CHECK-NEXT:    [[CAST:%.*]] = trunc i64 [[TMP]] to i32
// CHECK-NEXT:    ret i32 [[CAST]]
//
unsigned int testcnttz8(unsigned long long value) {
  return __cnttz8(value);
}

// CHECK-LABEL: @testpopcnt4(
// CHECK:         [[TMP:%.*]] = call i32 @llvm.ctpop.i32(i32 {{%.*}})
// CHECK-NEXT:    ret i32 [[TMP]]
//
int testpopcnt4(unsigned int value) {
  return __popcnt4(value);
}

// CHECK-LABEL: @testpopcnt8(
// CHECK:         [[TMP:%.*]] = call i64 @llvm.ctpop.i64(i64 {{%.*}})
// CHECK-NEXT:    [[CAST:%.*]] = trunc i64 [[TMP]] to i32
// CHECK-NEXT:    ret i32 [[CAST]]
//
int testpopcnt8(unsigned long long value) {
  return __popcnt8(value);
}
