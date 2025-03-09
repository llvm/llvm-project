// Test CodeGen for nuw/nsw builtins.

// RUN: %clang_cc1 -triple "x86_64-unknown-unknown" -emit-llvm -x c %s -o - | FileCheck %s

//------------------------------------------------------------------------------
// int
//------------------------------------------------------------------------------
int test_sadd_nuw(int x, int y) { return __builtin_sadd_nuw(x, y); }
// CHECK-LABEL: @test_sadd_nuw
// CHECK: [[RV:%.+]] = add nuw i32
// CHECK: ret i32 [[RV]]

int test_sadd_nsw(int x, int y) { return __builtin_sadd_nsw(x, y); }
// CHECK-LABEL: @test_sadd_nsw
// CHECK: [[RV:%.+]] = add nsw i32
// CHECK: ret i32 [[RV]]

int test_sadd_nuw_nsw(int x, int y) { return __builtin_sadd_nuw_nsw(x, y); }
// CHECK-LABEL: @test_sadd_nuw_nsw
// CHECK: [[RV:%.+]] = add nuw nsw i32
// CHECK: ret i32 [[RV]]

//------------------------------------------------------------------------------
// long int
//------------------------------------------------------------------------------
long int test_saddl_nuw(long int x, long int y) { return __builtin_saddl_nuw(x, y); }
// CHECK-LABEL: @test_saddl_nuw
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

long int test_saddl_nsw(long int x, long int y) { return __builtin_saddl_nsw(x, y); }
// CHECK-LABEL: @test_saddl_nsw
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

long int test_saddl_nuw_nsw(long int x, long int y) { return __builtin_saddl_nuw_nsw(x, y); }
// CHECK-LABEL: @test_saddl_nuw_nsw
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]

//------------------------------------------------------------------------------
// long int
//------------------------------------------------------------------------------
long long int test_saddll_nuw(long long int x, long long int y) { return __builtin_saddll_nuw(x, y); }
// CHECK-LABEL: @test_saddll_nuw
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

long long int test_saddll_nsw(long long int x, long long int y) { return __builtin_saddll_nsw(x, y); }
// CHECK-LABEL: @test_saddll_nsw
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

long long int test_saddll_nuw_nsw(long long int x, long long int y) { return __builtin_saddll_nuw_nsw(x, y); }
// CHECK-LABEL: @test_saddll_nuw_nsw
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]

//------------------------------------------------------------------------------
// unsigned int
//------------------------------------------------------------------------------
unsigned int test_uadd_nuw(unsigned int x, unsigned int y) { return __builtin_uadd_nuw(x, y); }
// CHECK-LABEL: @test_uadd_nuw
// CHECK: [[RV:%.+]] = add nuw i32
// CHECK: ret i32 [[RV]]

unsigned int test_uadd_nsw(unsigned int x, unsigned int y) { return __builtin_uadd_nsw(x, y); }
// CHECK-LABEL: @test_uadd_nsw
// CHECK: [[RV:%.+]] = add nsw i32
// CHECK: ret i32 [[RV]]

unsigned int test_uadd_nuw_nsw(unsigned int x, unsigned int y) { return __builtin_uadd_nuw_nsw(x, y); }
// CHECK-LABEL: @test_uadd_nuw_nsw
// CHECK: [[RV:%.+]] = add nuw nsw i32
// CHECK: ret i32 [[RV]]

//------------------------------------------------------------------------------
// unsigned long int
//------------------------------------------------------------------------------
unsigned long int test_uaddl_nuw(unsigned long int x, unsigned long int y) { return __builtin_uaddl_nuw(x, y); }
// CHECK-LABEL: @test_uaddl_nuw
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

unsigned long int test_uaddl_nsw(unsigned long int x, unsigned long int y) { return __builtin_uaddl_nsw(x, y); }
// CHECK-LABEL: @test_uaddl_nsw
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

unsigned long int test_uaddl_nuw_nsw(unsigned long int x, unsigned long int y) { return __builtin_uaddl_nuw_nsw(x, y); }
// CHECK-LABEL: @test_uaddl_nuw_nsw
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]

//------------------------------------------------------------------------------
// unsigned long long int
//------------------------------------------------------------------------------
unsigned long long int test_uaddll_nuw(unsigned long long int x, unsigned long long int y) { return __builtin_uaddll_nuw(x, y); }
// CHECK-LABEL: @test_uaddll_nuw
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

unsigned long long int test_uaddll_nsw(unsigned long long int x, unsigned long long int y) { return __builtin_uaddll_nsw(x, y); }
// CHECK-LABEL: @test_uaddll_nsw
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

unsigned long long int test_uaddll_nuw_nsw(unsigned long long int x, unsigned long long int y) { return __builtin_uaddll_nuw_nsw(x, y); }
// CHECK-LABEL: @test_uaddll_nuw_nsw
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]
