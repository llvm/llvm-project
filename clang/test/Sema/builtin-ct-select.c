// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// Test integer types
int test_int(int cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_int
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

long test_long(int cond, long a, long b) {
  // CHECK-LABEL: define {{.*}} @test_long
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i64 @llvm.ct.select.i64(i1 [[COND]], i64 %{{.*}}, i64 %{{.*}})
  // CHECK: ret i64 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

short test_short(int cond, short a, short b) {
  // CHECK-LABEL: define {{.*}} @test_short
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i16 @llvm.ct.select.i16(i1 [[COND]], i16 %{{.*}}, i16 %{{.*}})
  // CHECK: ret i16 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

unsigned char test_uchar(int cond, unsigned char a, unsigned char b) {
  // CHECK-LABEL: define {{.*}} @test_uchar
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i8 @llvm.ct.select.i8(i1 [[COND]], i8 %{{.*}}, i8 %{{.*}})
  // CHECK: ret i8 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

long long test_longlong(int cond, long long a, long long b) {
  // CHECK-LABEL: define {{.*}} @test_longlong
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i64 @llvm.ct.select.i64(i1 [[COND]], i64 %{{.*}}, i64 %{{.*}})
  // CHECK: ret i64 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test floating point types
float test_float(int cond, float a, float b) {
  // CHECK-LABEL: define {{.*}} @test_float
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call float @llvm.ct.select.f32(i1 [[COND]], float %{{.*}}, float %{{.*}})
  // CHECK: ret float [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

double test_double(int cond, double a, double b) {
  // CHECK-LABEL: define {{.*}} @test_double
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call double @llvm.ct.select.f64(i1 [[COND]], double %{{.*}}, double %{{.*}})
  // CHECK: ret double [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test pointer types
int *test_pointer(int cond, int *a, int *b) {
  // CHECK-LABEL: define {{.*}} @test_pointer
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call ptr @llvm.ct.select.p0(i1 [[COND]], ptr %{{.*}}, ptr %{{.*}})
  // CHECK: ret ptr [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test with different condition types
int test_char_cond(char cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_char_cond
  // CHECK: [[COND:%.*]] = icmp ne i8 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

int test_long_cond(long cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_long_cond
  // CHECK: [[COND:%.*]] = icmp ne i64 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test with boolean condition
int test_bool_cond(_Bool cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_bool_cond
  // CHECK: [[COND:%.*]] = trunc i8 %{{.*}} to i1
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test with constants
int test_constant_cond(void) {
  // CHECK-LABEL: define {{.*}} @test_constant_cond
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 true, i32 42, i32 24)
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(1, 42, 24);
}

int test_zero_cond(void) {
  // CHECK-LABEL: define {{.*}} @test_zero_cond
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 false, i32 42, i32 24)
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(0, 42, 24);
}

// Test type promotion
int test_promotion(int cond, short a, short b) {
  // CHECK-LABEL: define {{.*}} @test_promotion
  // CHECK-DAG: [[A_EXT:%.*]] = sext i16 %{{.*}} to i32
  // CHECK-DAG: [[B_EXT:%.*]] = sext i16 %{{.*}} to i32
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 [[A_EXT]], i32 [[B_EXT]])
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, (int)a, (int)b);
}

// Test mixed signedness
unsigned int test_mixed_signedness(int cond, int a, unsigned int b) {
  // CHECK-LABEL: define {{.*}} @test_mixed_signedness
  // CHECK-DAG: [[A_EXT:%.*]] = sext i32 %{{.*}} to i64
  // CHECK-DAG: [[B_EXT:%.*]] = zext i32 %{{.*}} to i64
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i64 @llvm.ct.select.i64(i1 [[COND]], i64 [[A_EXT]], i64 [[B_EXT]])
  // CHECK: [[RESULT_TRUNC:%.*]] = trunc i64 [[RESULT]] to i32
  // CHECK: ret i32 [[RESULT_TRUNC]]
  return __builtin_ct_select(cond, (long)a, (long)b);
}

// Test complex expression
int test_complex_expr_alt(int x, int y) {
  // CHECK-LABEL: define {{.*}} @test_complex_expr_alt
  // CHECK-DAG: [[CMP:%.*]] = icmp sgt i32 %{{.*}}, 0
  // CHECK-DAG: [[ADD:%.*]] = add nsw i32 %{{.*}}, %{{.*}}
  // CHECK-DAG: [[SUB:%.*]] = sub nsw i32 %{{.*}}, %{{.*}}
  // Separate the final sequence to ensure proper ordering
  // CHECK-NEXT: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[CMP]], i32 [[ADD]], i32 [[SUB]])
  // CHECK-NEXT: ret i32 [[RESULT]]
  return __builtin_ct_select(x > 0, x + y, x - y);
}

// Test nested calls
int test_nested_structured(int cond1, int cond2, int a, int b, int c) {
  // CHECK-LABEL: define {{.*}} @test_nested_structured
  // Phase 1: Conditions (order doesn't matter)
  // CHECK-DAG: [[COND1:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[COND2:%.*]] = icmp ne i32 %{{.*}}, 0
  
  // Phase 2: Inner select (must happen before outer)
  // CHECK: [[INNER:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND2]], i32 %{{.*}}, i32 %{{.*}})
  
  // Phase 3: Outer select (must use inner result)
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND1]], i32 [[INNER]], i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond1, __builtin_ct_select(cond2, a, b), c);
}

// Test with function calls
int helper(int x) { return x * 2; }
int test_function_calls(int cond, int x, int y) {
  // CHECK-LABEL: define {{.*}} @test_function_calls
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[CALL1:%.*]] = call i32 @helper(i32 noundef %{{.*}})
  // CHECK-DAG: [[CALL2:%.*]] = call i32 @helper(i32 noundef %{{.*}})
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 [[CALL1]], i32 [[CALL2]])
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(cond, helper(x), helper(y));
}

// Test using ct_select as condition for another ct_select
int test_intrinsic_condition(int cond1, int cond2, int a, int b, int c, int d) {
  // CHECK-LABEL: define {{.*}} @test_intrinsic_condition
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[INNER_COND:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK-DAG: [[FINAL_COND:%.*]] = icmp ne i32 [[INNER_COND]], 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[FINAL_COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(__builtin_ct_select(cond1, cond2, a), b, c);
}

// Test using comparison result of ct_select as condition
int test_comparison_condition(int cond, int a, int b, int c, int d) {
  // CHECK-LABEL: define {{.*}} @test_comparison_condition
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[FIRST_SELECT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: [[CMP:%.*]] = icmp sgt i32 [[FIRST_SELECT]], %{{.*}}
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[CMP]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(__builtin_ct_select(cond, a, b) > c, d, a);
}

// Test using ct_select result in arithmetic as condition
int test_arithmetic_condition(int cond, int a, int b, int c, int d) {
  // CHECK-LABEL: define {{.*}} @test_arithmetic_condition
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[FIRST_SELECT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: [[ADD:%.*]] = add nsw i32 [[FIRST_SELECT]], %{{.*}}
  // CHECK: [[FINAL_COND:%.*]] = icmp ne i32 [[ADD]], 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[FINAL_COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(__builtin_ct_select(cond, a, b) + c, d, a);
}

// Test chained ct_select as conditions
int test_chained_conditions(int cond1, int cond2, int cond3, int a, int b, int c, int d, int e) {
  // CHECK-LABEL: define {{.*}} @test_chained_conditions
  // CHECK: [[COND1:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[FIRST:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND1]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK-DAG: [[COND2:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[SECOND:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND2]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK-DAG: [[FINAL_COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[FINAL_COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  int first_select = __builtin_ct_select(cond1, a, b);
  int second_select = __builtin_ct_select(cond2, first_select, c);
  return __builtin_ct_select(second_select, d, e);
}

// Test using ct_select with pointer condition
//int test_pointer_condition(int *ptr1, int *ptr2, int a, int b, int c) {
  // NO-CHECK-LABEL: define {{.*}} @test_pointer_condition
  // NO-CHECK: [[PTR_COND:%.*]] = icmp ne ptr %{{.*}}, null
  // NO-CHECK: [[PTR_SELECT:%.*]] = call ptr @llvm.ct.select.p0(i1 [[PTR_COND]], ptr %{{.*}}, ptr %{{.*}})
  // NO-CHECK: [[FINAL_COND:%.*]] = icmp ne ptr [[PTR_SELECT]], null
  // NO-CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[FINAL_COND]], i32 %{{.*}}, i32 %{{.*}})
  // NO-CHECK: ret i32 [[RESULT]]
//  return __builtin_ct_select(__builtin_ct_select(ptr1, ptr1, ptr2), a, b);
//}


// Test using ct_select result in logical operations as condition
int test_logical_condition(int cond1, int cond2, int a, int b, int c, int d) {
  // CHECK-LABEL: define {{.*}} @test_logical_condition
  // CHECK-DAG: [[COND1:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[COND2:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[FIRST_SELECT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND1]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK-DAG: [[SELECT_BOOL:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(__builtin_ct_select(cond1, a, b) && cond2, c, d);
}

// Test multiple levels of ct_select as conditions
int test_deep_condition_nesting(int cond1, int cond2, int cond3, int a, int b, int c, int d, int e, int f) {
  // CHECK-LABEL: define {{.*}} @test_deep_condition_nesting
  // CHECK-DAG: [[COND1:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[COND2:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[INNER1:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND2]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK-DAG: [[INNER1_COND:%.*]] = icmp ne i32 [[INNER1]], 0
  // CHECK-DAG: [[INNER2:%.*]] = call i32 @llvm.ct.select.i32(i1 [[INNER1_COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK-DAG: [[OUTER:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND1]], i32 [[INNER2]], i32 %{{.*}})
  // CHECK-DAG: [[FINAL_COND:%.*]] = icmp ne i32 [[OUTER]], 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[FINAL_COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(__builtin_ct_select(cond1, __builtin_ct_select(__builtin_ct_select(cond2, a, b), c, d), e), f, a);
}

// Test ct_select with complex condition expressions
int test_complex_condition_expr(int x, int y, int z, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_complex_condition_expr
  // CHECK: [[CMP1:%.*]] = icmp sgt i32 %{{.*}}, %{{.*}}
  // CHECK: [[SELECT1:%.*]] = call i32 @llvm.ct.select.i32(i1 [[CMP1]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: [[CMP2:%.*]] = icmp slt i32 [[SELECT1]], %{{.*}}
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[CMP2]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  return __builtin_ct_select(__builtin_ct_select(x > y, x, y) < z, a, b);
}

// Test vector types - 128-bit vectors
typedef int __attribute__((vector_size(16))) int4;
typedef float __attribute__((vector_size(16))) float4;
typedef short __attribute__((vector_size(16))) short8;
typedef char __attribute__((vector_size(16))) char16;

int4 test_vector_int4(int cond, int4 a, int4 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_int4
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.ct.select.v4i32(i1 [[COND]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: ret <4 x i32> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

float4 test_vector_float4(int cond, float4 a, float4 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_float4
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[COND]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: ret <4 x float> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

short8 test_vector_short8(int cond, short8 a, short8 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_short8
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <8 x i16> @llvm.ct.select.v8i16(i1 [[COND]], <8 x i16> %{{.*}}, <8 x i16> %{{.*}})
  // CHECK: ret <8 x i16> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

char16 test_vector_char16(int cond, char16 a, char16 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_char16
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <16 x i8> @llvm.ct.select.v16i8(i1 [[COND]], <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK: ret <16 x i8> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test 256-bit vectors
typedef int __attribute__((vector_size(32))) int8;
typedef float __attribute__((vector_size(32))) float8;
typedef double __attribute__((vector_size(32))) double4;

int8 test_vector_int8(int cond, int8 a, int8 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_int8
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call <8 x i32> @llvm.ct.select.v8i32(i1 [[COND]], <8 x i32> %{{.*}}, <8 x i32> %{{.*}})
  return __builtin_ct_select(cond, a, b);
}

float8 test_vector_float8(int cond, float8 a, float8 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_float8
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call <8 x float> @llvm.ct.select.v8f32(i1 [[COND]], <8 x float> %{{.*}}, <8 x float> %{{.*}})
  return __builtin_ct_select(cond, a, b);
}

double4 test_vector_double4(int cond, double4 a, double4 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_double4
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call <4 x double> @llvm.ct.select.v4f64(i1 [[COND]], <4 x double> %{{.*}}, <4 x double> %{{.*}})
  return __builtin_ct_select(cond, a, b);
}

// Test 512-bit vectors
typedef int __attribute__((vector_size(64))) int16;
typedef float __attribute__((vector_size(64))) float16;

int16 test_vector_int16(int cond, int16 a, int16 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_int16
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <16 x i32> @llvm.ct.select.v16i32(i1 [[COND]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}})
  return __builtin_ct_select(cond, a, b);
}

float16 test_vector_float16(int cond, float16 a, float16 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_float16
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <16 x float> @llvm.ct.select.v16f32(i1 [[COND]], <16 x float> %{{.*}}, <16 x float> %{{.*}})
  return __builtin_ct_select(cond, a, b);
}

// Test vector operations with different condition types
int4 test_vector_char_cond(char cond, int4 a, int4 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_char_cond
  // CHECK: [[COND:%.*]] = icmp ne i8 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.ct.select.v4i32(i1 [[COND]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: ret <4 x i32> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

float4 test_vector_long_cond(long cond, float4 a, float4 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_long_cond
  // CHECK: [[COND:%.*]] = icmp ne i64 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[COND]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: ret <4 x float> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test vector constants
int4 test_vector_constant_cond(void) {
  // CHECK-LABEL: define {{.*}} @test_vector_constant_cond
  // CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.ct.select.v4i32(i1 true, <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: ret <4 x i32> [[RESULT]]
  int4 a = {1, 2, 3, 4};
  int4 b = {5, 6, 7, 8};
  return __builtin_ct_select(1, a, b);
}

float4 test_vector_zero_cond(void) {
  // CHECK-LABEL: define {{.*}} @test_vector_zero_cond
  // CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 false, <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: ret <4 x float> [[RESULT]]
  float4 a = {1.0f, 2.0f, 3.0f, 4.0f};
  float4 b = {5.0f, 6.0f, 7.0f, 8.0f};
  return __builtin_ct_select(0, a, b);
}

// Test nested vector selections
int4 test_vector_nested(int cond1, int cond2, int4 a, int4 b, int4 c) {
  // CHECK-LABEL: define {{.*}} @test_vector_nested
  // CHECK-DAG: [[COND1:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[COND2:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[INNER:%.*]] = call <4 x i32> @llvm.ct.select.v4i32(i1 [[COND2]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.ct.select.v4i32(i1 [[COND1]], <4 x i32> [[INNER]], <4 x i32> %{{.*}})
  // CHECK: ret <4 x i32> [[RESULT]]
  return __builtin_ct_select(cond1, __builtin_ct_select(cond2, a, b), c);
}

// Test vector selection with complex expressions
float4 test_vector_complex_expr(int x, int y, float4 a, float4 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_complex_expr
  // CHECK: [[CMP:%.*]] = icmp sgt i32 %{{.*}}, %{{.*}}
  // CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[CMP]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: ret <4 x float> [[RESULT]]
  return __builtin_ct_select(x > y, a, b);
}

// Test vector with different element sizes
typedef long long __attribute__((vector_size(16))) long2;
typedef double __attribute__((vector_size(16))) double2;

long2 test_vector_long2(int cond, long2 a, long2 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_long2
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <2 x i64> @llvm.ct.select.v2i64(i1 [[COND]], <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK: ret <2 x i64> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

double2 test_vector_double2(int cond, double2 a, double2 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_double2
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <2 x double> @llvm.ct.select.v2f64(i1 [[COND]], <2 x double> %{{.*}}, <2 x double> %{{.*}})
  // CHECK: ret <2 x double> [[RESULT]]
  return __builtin_ct_select(cond, a, b);
}

// Test mixed vector operations
int4 test_vector_from_scalar_condition(int4 vec_cond, int4 a, int4 b) {
  // CHECK-LABEL: define {{.*}} @test_vector_from_scalar_condition
  // Extract first element and use as condition
  int scalar_cond = vec_cond[0];
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <4 x i32> @llvm.ct.select.v4i32(i1 [[COND]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}})
  // CHECK: ret <4 x i32> [[RESULT]]
  return __builtin_ct_select(scalar_cond, a, b);
}

// Test vector chaining
float4 test_vector_chaining(int cond1, int cond2, int cond3, float4 a, float4 b, float4 c, float4 d) {
  // CHECK-LABEL: define {{.*}} @test_vector_chaining
  // CHECK-DAG: [[COND1:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[COND2:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[COND3:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[FIRST:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[COND1]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-DAG: [[SECOND:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[COND2]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK-DAG: [[RESULT:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[COND3]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: ret <4 x float> [[RESULT]]
  float4 first = __builtin_ct_select(cond1, a, b);
  float4 second = __builtin_ct_select(cond2, first, c);
  return __builtin_ct_select(cond3, second, d);
}

// Test special floating point values - NaN
float test_nan_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_nan_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call float @llvm.ct.select.f32(i1 [[COND]], float  %{{.*}}, float 1.000000e+00)
  // CHECK: ret float [[RESULT]]
  float nan_val = __builtin_nanf("");
  return __builtin_ct_select(cond, nan_val, 1.0f);
}

double test_nan_double_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_nan_double_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call double @llvm.ct.select.f64(i1 [[COND]], double %{{.*}}, double 2.000000e+00)
  // CHECK: ret double [[RESULT]]
  double nan_val = __builtin_nan("");
  return __builtin_ct_select(cond, nan_val, 2.0);
}

// Test infinity values
float test_infinity_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_infinity_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call float @llvm.ct.select.f32(i1 [[COND]], float %{{.*}}, float %{{.*}})
  // CHECK: ret float [[RESULT]]
  float pos_inf = __builtin_inff();
  float neg_inf = -__builtin_inff();
  return __builtin_ct_select(cond, pos_inf, neg_inf);
}

double test_infinity_double_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_infinity_double_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call double @llvm.ct.select.f64(i1 [[COND]], double %{{.*}}, double %{{.*}})
  // CHECK: ret double [[RESULT]]
  double pos_inf = __builtin_inf();
  double neg_inf = -__builtin_inf();
  return __builtin_ct_select(cond, pos_inf, neg_inf);
}

// Test subnormal/denormal values
float test_subnormal_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_subnormal_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call float @llvm.ct.select.f32(i1 [[COND]], float %{{.*}}, float %{{.*}})
  // CHECK: ret float [[RESULT]]
  // Very small subnormal values
  float subnormal1 = 1e-40f;
  float subnormal2 = 1e-45f;
  return __builtin_ct_select(cond, subnormal1, subnormal2);
}

// Test integer overflow boundaries
int test_integer_overflow_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_integer_overflow_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  int max_int = __INT_MAX__;
  int min_int = (-__INT_MAX__ - 1);
  return __builtin_ct_select(cond, max_int, min_int);
}

long long test_longlong_overflow_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_longlong_overflow_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i64 @llvm.ct.select.i64(i1 [[COND]], i64 %{{.*}}, i64 %{{.*}})
  // CHECK: ret i64 [[RESULT]]
  long long max_ll = __LONG_LONG_MAX__;
  long long min_ll = (-__LONG_LONG_MAX__ - 1);
  return __builtin_ct_select(cond, max_ll, min_ll);
}

// Test unsigned overflow boundaries
unsigned int test_unsigned_overflow_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_unsigned_overflow_operands
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  unsigned int max_uint = 4294967295;
  unsigned int min_uint = 0;
  return __builtin_ct_select(cond, max_uint, min_uint);
}

// Test null pointer dereference avoidance
int* test_null_pointer_operands(int cond, int* valid_ptr) {
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call ptr @llvm.ct.select.p0(i1 [[COND]], ptr %{{.*}}, ptr %{{.*}})
  // CHECK: ret ptr [[RESULT]]
  int* null_ptr = (int*)0;
  return __builtin_ct_select(cond, null_ptr, valid_ptr);
}

// Test volatile operations
volatile int global_volatile = 42;
int test_volatile_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_volatile_operands
  // CHECK-DAG: [[VOLATILE_LOAD:%.*]] = load volatile i32, ptr {{.*}}
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 100)
  // CHECK: ret i32 [[RESULT]]
  volatile int vol_val = global_volatile;
  return __builtin_ct_select(cond, vol_val, 100);
}

// Test uninitialized variable behavior (should still work with ct_select)
int test_uninitialized_operands(int cond, int initialized) {
  // CHECK-LABEL: define {{.*}} @test_uninitialized_operands
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  int uninitialized; // Intentionally uninitialized
  return __builtin_ct_select(cond, uninitialized, initialized);
}

// Test zero division avoidance patterns
int test_division_by_zero_avoidance(int cond, int dividend, int divisor) {
  // CHECK-LABEL: define {{.*}} @test_division_by_zero_avoidance
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[DIV_RESULT:%.*]] = sdiv i32 %{{.*}}, %{{.*}}
  // CHECK-DAG: [[SAFE_DIVISOR:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 1)
  // First get a safe divisor (never zero)
  int safe_divisor = __builtin_ct_select(divisor != 0, divisor, 1);
  // Then perform division with guaranteed non-zero divisor
  return dividend / safe_divisor;
}

// Test array bounds checking patterns
int test_array_bounds_protection(int cond, int index, int* array) {
  // CHECK-LABEL: define {{.*}} @test_array_bounds_protection
  // CHECK-DAG: [[SAFE_INDEX:%.*]] = call i32 @llvm.ct.select.i32(i1 {{.*}}, i32 %{{.*}}, i32 0)
  // Use ct_select to ensure safe array indexing
  int safe_index = __builtin_ct_select(index >= 0 && index < 10, index, 0);
  return array[safe_index];
}

// Test bit manipulation edge cases
unsigned int test_bit_manipulation_edge_cases(int cond, unsigned int value) {
  // CHECK-LABEL: define {{.*}} @test_bit_manipulation_edge_cases
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[SHIFT_LEFT:%.*]] = shl i32 %{{.*}}, 31
  // CHECK-DAG: [[SHIFT_RIGHT:%.*]] = lshr i32 %{{.*}}, 31
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  // Test extreme bit shifts that could cause undefined behavior
  unsigned int left_shift = value << 31;   // Could overflow
  unsigned int right_shift = value >> 31;  // Extract sign bit
  return __builtin_ct_select(cond, left_shift, right_shift);
}

// Test signed integer wraparound
int test_signed_wraparound(int cond, int a, int b) {
  // CHECK-LABEL: define {{.*}} @test_signed_wraparound
  // CHECK-DAG: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK-DAG: [[ADD:%.*]] = add nsw i32 %{{.*}}, %{{.*}}
  // CHECK-DAG: [[SUB:%.*]] = sub nsw i32 %{{.*}}, %{{.*}}
  // CHECK-DAG: [[RESULT:%.*]] = call i32 @llvm.ct.select.i32(i1 [[COND]], i32 %{{.*}}, i32 %{{.*}})
  // CHECK: ret i32 [[RESULT]]
  int sum = a + b;      // Could overflow
  int diff = a - b;     // Could underflow
  return __builtin_ct_select(cond, sum, diff);
}

// Test vector NaN handling
float4 test_vector_nan_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_vector_nan_operands
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[COND]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: ret <4 x float> [[RESULT]]
  float nan_val = __builtin_nanf("");
  float4 nan_vec = {nan_val, nan_val, nan_val, nan_val};
  float4 normal_vec = {1.0f, 2.0f, 3.0f, 4.0f};
  return __builtin_ct_select(cond, nan_vec, normal_vec);
}

// Test vector infinity handling
float4 test_vector_infinity_operands(int cond) {
  // CHECK-LABEL: define {{.*}} @test_vector_infinity_operands
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call <4 x float> @llvm.ct.select.v4f32(i1 [[COND]], <4 x float> %{{.*}}, <4 x float> %{{.*}})
  // CHECK: ret <4 x float> [[RESULT]]
  float pos_inf = __builtin_inff();
  float neg_inf = -__builtin_inff();
  float4 inf_vec = {pos_inf, neg_inf, pos_inf, neg_inf};
  float4 zero_vec = {0.0f, 0.0f, 0.0f, 0.0f};
  return __builtin_ct_select(cond, inf_vec, zero_vec);
}

// Test mixed special values
double test_mixed_special_values(int cond) {
  // CHECK-LABEL: define {{.*}} @test_mixed_special_values
  // CHECK: [[COND:%.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: [[RESULT:%.*]] = call double @llvm.ct.select.f64(i1 [[COND]], double %{{.*}}, double %{{.*}})
  // CHECK: ret double [[RESULT]]
  double nan_val = __builtin_nan("");
  double inf_val = __builtin_inf();
  return __builtin_ct_select(cond, nan_val, inf_val);
}

// Test constant-time memory access pattern
int test_constant_time_memory_access(int secret_index, int* data_array) {
  // CHECK-LABEL: define {{.*}} @test_constant_time_memory_access
  // This pattern ensures constant-time memory access regardless of secret_index value
  int result = 0;
  // Use ct_select to accumulate values without revealing the secret index
  for (int i = 0; i < 8; i++) {
    int is_target = (i == secret_index);
    int current_value = data_array[i];
    int selected_value = __builtin_ct_select(is_target, current_value, 0);
    result += selected_value;
  }
  return result;
}

// Test timing-attack resistant comparison
int test_timing_resistant_comparison(const char* secret, const char* guess) {
  // CHECK-LABEL: define {{.*}} @test_timing_resistant_comparison
  // Constant-time string comparison using ct_select
  int match = 1;
  for (int i = 0; i < 32; i++) {
    int chars_equal = (secret[i] == guess[i]);
    int both_null = (secret[i] == 0) && (guess[i] == 0);
    int still_matching = __builtin_ct_select(chars_equal || both_null, match, 0);
    match = __builtin_ct_select(both_null, match, still_matching);
  }
  return match;
}
