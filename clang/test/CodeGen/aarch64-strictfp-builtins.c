// RUN: %clang_cc1 %s -emit-llvm -ffp-exception-behavior=maytrap -fexperimental-strict-floating-point -o - -triple arm64-none-linux-gnu | FileCheck %s

// Test that the constrained intrinsics are picking up the exception
// metadata from the AST instead of the global default from the command line.

#pragma float_control(except, on)

int printf(const char *, ...);

// CHECK-LABEL: @p(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[STR_ADDR:%.*]] = alloca i8*, align 8
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i8* [[STR:%.*]], i8** [[STR_ADDR]], align 8
// CHECK-NEXT:    store i32 [[X:%.*]], i32* [[X_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i8*, i8** [[STR_ADDR]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* [[X_ADDR]], align 4
// CHECK-NEXT:    [[CALL:%.*]] = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0), i8* [[TMP0]], i32 [[TMP1]])  [[ATTR4:#.*]]
// CHECK-NEXT:    ret void
//
void p(char *str, int x) {
  printf("%s: %d\n", str, x);
}

#define P(n,args) p(#n #args, __builtin_##n args)

// CHECK-LABEL: @test_long_double_isinf(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LD_ADDR:%.*]] = alloca fp128, align 16
// CHECK-NEXT:    store fp128 [[D:%.*]], fp128* [[LD_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load fp128, fp128* [[LD_ADDR]], align 16
// CHECK-NEXT:    [[BITCAST:%.*]] = bitcast fp128 [[TMP0]] to i128
// CHECK-NEXT:    [[SHL1:%.*]] = shl i128 [[BITCAST]], 1
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq i128 [[SHL1]], -10384593717069655257060992658440192
// CHECK-NEXT:    [[RES:%.*]] = zext i1 [[CMP]] to i32
// CHECK-NEXT:    call void @p(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.[[#STRID:1]], i64 0, i64 0), i32 [[RES]]) [[ATTR4]]
// CHECK-NEXT:    ret void
//
void test_long_double_isinf(long double ld) {
  P(isinf, (ld));

  return;
}

// CHECK-LABEL: @test_long_double_isfinite(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LD_ADDR:%.*]] = alloca fp128, align 16
// CHECK-NEXT:    store fp128 [[D:%.*]], fp128* [[LD_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load fp128, fp128* [[LD_ADDR]], align 16
// CHECK-NEXT:    [[BITCAST:%.*]] = bitcast fp128 [[TMP0]] to i128
// CHECK-NEXT:    [[SHL1:%.*]] = shl i128 [[BITCAST]], 1
// CHECK-NEXT:    [[CMP:%.*]] = icmp ult i128 [[SHL1]], -10384593717069655257060992658440192
// CHECK-NEXT:    [[RES:%.*]] = zext i1 [[CMP]] to i32
// CHECK-NEXT:    call void @p(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.[[#STRID:STRID+1]], i64 0, i64 0), i32 [[RES]]) [[ATTR4]]
// CHECK-NEXT:    ret void
//
void test_long_double_isfinite(long double ld) {
  P(isfinite, (ld));

  return;
}

// CHECK-LABEL: @test_long_double_isnan(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[LD_ADDR:%.*]] = alloca fp128, align 16
// CHECK-NEXT:    store fp128 [[D:%.*]], fp128* [[LD_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load fp128, fp128* [[LD_ADDR]], align 16
// CHECK-NEXT:    [[BITCAST:%.*]] = bitcast fp128 [[TMP0]] to i128
// CHECK-NEXT:    [[ABS:%.*]] = and i128 [[BITCAST]], 170141183460469231731687303715884105727
// CHECK-NEXT:    [[TMP1:%.*]] = sub i128 170135991163610696904058773219554885632, [[ABS]]
// CHECK-NEXT:    [[ISNAN:%.*]] = lshr i128 [[TMP1]], 127
// CHECK-NEXT:    [[RES:%.*]] = trunc i128 [[ISNAN]] to i32
// CHECK-NEXT:    call void @p(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.[[#STRID:STRID+1]], i64 0, i64 0), i32 [[RES]])
// CHECK-NEXT:    ret void
//
void test_long_double_isnan(long double ld) {
  P(isnan, (ld));

  return;
}
