// Test CodeGen for nuw/nsw builtins.

// RUN: %clang_cc1 -triple "x86_64-unknown-unknown" -emit-llvm -x c %s -o - | FileCheck %s

//------------------------------------------------------------------------------
// int
//------------------------------------------------------------------------------
int test_add_nuw(int x, int y) { return __builtin_add_nuw(x, y); }
// CHECK-LABEL: @test_add_nuw
// CHECK: [[RV:%.+]] = add nuw i32
// CHECK: ret i32 [[RV]]

int test_add_nsw(int x, int y) { return __builtin_add_nsw(x, y); }
// CHECK-LABEL: @test_add_nsw
// CHECK: [[RV:%.+]] = add nsw i32
// CHECK: ret i32 [[RV]]

int test_add_nuw_nsw(int x, int y) { return __builtin_add_nuw_nsw(x, y); }
// CHECK-LABEL: @test_add_nuw_nsw
// CHECK: [[RV:%.+]] = add nuw nsw i32
// CHECK: ret i32 [[RV]]

//------------------------------------------------------------------------------
// long int
//------------------------------------------------------------------------------
long int test_add_nuw_l(long int x, long int y) { return __builtin_add_nuw(x, y); }
// CHECK-LABEL: @test_add_nuw_l
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

long int test_add_nsw_l(long int x, long int y) { return __builtin_add_nsw(x, y); }
// CHECK-LABEL: @test_add_nsw_l
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

long int test_add_nuw_nsw_l(long int x, long int y) { return __builtin_add_nuw_nsw(x, y); }
// CHECK-LABEL: @test_add_nuw_nsw_l
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]

//------------------------------------------------------------------------------
// long int
//------------------------------------------------------------------------------
long long int test_add_nuw_ll(long long int x, long long int y) { return __builtin_add_nuw(x, y); }
// CHECK-LABEL: @test_add_nuw_ll
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

long long int test_add_nsw_ll(long long int x, long long int y) { return __builtin_add_nsw(x, y); }
// CHECK-LABEL: @test_add_nsw_ll
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

long long int test_add_nuw_nsw_ll(long long int x, long long int y) { return __builtin_add_nuw_nsw(x, y); }
// CHECK-LABEL: @test_add_nuw_nsw_ll
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]

//------------------------------------------------------------------------------
// unsigned int
//------------------------------------------------------------------------------
unsigned int test_add_nuw_u(unsigned int x, unsigned int y) { return __builtin_add_nuw(x, y); }
// CHECK-LABEL: @test_add_nuw_u
// CHECK: [[RV:%.+]] = add nuw i32
// CHECK: ret i32 [[RV]]

unsigned int test_add_nsw_u(unsigned int x, unsigned int y) { return __builtin_add_nsw(x, y); }
// CHECK-LABEL: @test_add_nsw_u
// CHECK: [[RV:%.+]] = add nsw i32
// CHECK: ret i32 [[RV]]

unsigned int test_add_nuw_nsw_u(unsigned int x, unsigned int y) { return __builtin_add_nuw_nsw(x, y); }
// CHECK-LABEL: @test_add_nuw_nsw_u
// CHECK: [[RV:%.+]] = add nuw nsw i32
// CHECK: ret i32 [[RV]]

//------------------------------------------------------------------------------
// unsigned long int
//------------------------------------------------------------------------------
unsigned long int test_add_nuw_ul(unsigned long int x, unsigned long int y) { return __builtin_add_nuw(x, y); }
// CHECK-LABEL: @test_add_nuw_ul
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

unsigned long int test_add_nsw_ul(unsigned long int x, unsigned long int y) { return __builtin_add_nsw(x, y); }
// CHECK-LABEL: @test_add_nsw_ul
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

unsigned long int test_add_nuw_nsw_ul(unsigned long int x, unsigned long int y) { return __builtin_add_nuw_nsw(x, y); }
// CHECK-LABEL: @test_add_nuw_nsw_ul
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]

//------------------------------------------------------------------------------
// unsigned long long int
//------------------------------------------------------------------------------
unsigned long long int test_add_nuw_ull(unsigned long long int x, unsigned long long int y) { return __builtin_add_nuw(x, y); }
// CHECK-LABEL: @test_add_nuw_ull
// CHECK: [[RV:%.+]] = add nuw i64
// CHECK: ret i64 [[RV]]

unsigned long long int test_add_nsw_ull(unsigned long long int x, unsigned long long int y) { return __builtin_add_nsw(x, y); }
// CHECK-LABEL: @test_add_nsw_ull
// CHECK: [[RV:%.+]] = add nsw i64
// CHECK: ret i64 [[RV]]

unsigned long long int test_add_nuw_nsw_ull(unsigned long long int x, unsigned long long int y) { return __builtin_add_nuw_nsw(x, y); }
// CHECK-LABEL: @test_add_nuw_nsw_ull
// CHECK: [[RV:%.+]] = add nuw nsw i64
// CHECK: ret i64 [[RV]]
