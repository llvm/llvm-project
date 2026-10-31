// RUN: %clang_cc1 -triple xtensa -O0 -emit-llvm %s -o - | FileCheck %s

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int int64_t;

// Test scalar arguments

// CHECK-LABEL: define dso_local zeroext i1 @f_scalar_i1(
// CHECK-SAME: i1 noundef zeroext [[X:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    [[STOREDV:%.*]] = zext i1 [[X]] to i8
// CHECK-NEXT:    store i8 [[STOREDV]], ptr [[X_ADDR]], align 1
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[X_ADDR]], align 1
// CHECK-NEXT:    [[LOADEDV:%.*]] = icmp ne i8 [[TMP0]], 0
// CHECK-NEXT:    ret i1 [[LOADEDV]]
//
_Bool f_scalar_i1(_Bool x) { return x; }

// CHECK-LABEL: define dso_local signext i8 @f_scalar_i8(
// CHECK-SAME: i8 noundef signext [[X:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca i8, align 1
// CHECK-NEXT:    store i8 [[X]], ptr [[X_ADDR]], align 1
// CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[X_ADDR]], align 1
// CHECK-NEXT:    ret i8 [[TMP0]]
//
int8_t f_scalar_i8(int8_t x) { return x; }

// CHECK-LABEL: define dso_local signext i16 @f_scalar_i16(
// CHECK-SAME: i16 noundef signext [[X:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca i16, align 2
// CHECK-NEXT:    store i16 [[X]], ptr [[X_ADDR]], align 2
// CHECK-NEXT:    [[TMP0:%.*]] = load i16, ptr [[X_ADDR]], align 2
// CHECK-NEXT:    ret i16 [[TMP0]]
//
int16_t f_scalar_i16(int16_t x) { return x; }

// CHECK-LABEL: define dso_local i32 @f_scalar_i32(
// CHECK-SAME: i32 noundef [[X:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[X]], ptr [[X_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[X_ADDR]], align 4
// CHECK-NEXT:    ret i32 [[TMP0]]
//
int32_t f_scalar_i32(int32_t x) { return x; }

// CHECK-LABEL: define dso_local i64 @f_scalar_i64(
// CHECK-SAME: i64 noundef [[X:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca i64, align 8
// CHECK-NEXT:    store i64 [[X]], ptr [[X_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i64, ptr [[X_ADDR]], align 8
// CHECK-NEXT:    ret i64 [[TMP0]]
//
int64_t f_scalar_i64(int64_t x) { return x; }

// CHECK-LABEL: define dso_local float @f_scalar_float(
// CHECK-SAME: i32 noundef [[X_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[X:%.*]] = alloca float, align 4
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca float, align 4
// CHECK-NEXT:    store i32 [[X_COERCE]], ptr [[X]], align 4
// CHECK-NEXT:    [[X1:%.*]] = load float, ptr [[X]], align 4
// CHECK-NEXT:    store float [[X1]], ptr [[X_ADDR]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load float, ptr [[X_ADDR]], align 4
// CHECK-NEXT:    ret float [[TMP0]]
//
float f_scalar_float(float x) { return x; }

// CHECK-LABEL: define dso_local i64 @f_scalar_double(
// CHECK-SAME: i64 noundef [[X_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[X:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[X_ADDR:%.*]] = alloca double, align 8
// CHECK-NEXT:    store i64 [[X_COERCE]], ptr [[X]], align 8
// CHECK-NEXT:    [[X1:%.*]] = load double, ptr [[X]], align 8
// CHECK-NEXT:    store double [[X1]], ptr [[X_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load double, ptr [[X_ADDR]], align 8
// CHECK-NEXT:    store double [[TMP0]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    [[TMP1:%.*]] = load i64, ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret i64 [[TMP1]]
//
double f_scalar_double(double x) { return x; }

// Test aggregate arguments

struct S16 { int a[4]; } __attribute__ ((aligned (16)));

// CHECK-LABEL: define dso_local void @callee_struct_a16b_1(
// CHECK-SAME: i128 [[A_COERCE:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[A:%.*]] = alloca [[STRUCT_S16:%.*]], align 16
// CHECK-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw [[STRUCT_S16]], ptr [[A]], i32 0, i32 0
// CHECK-NEXT:    store i128 [[A_COERCE]], ptr [[COERCE_DIVE]], align 16
// CHECK-NEXT:    ret void
//
void callee_struct_a16b_1(struct S16 a) {}


// CHECK-LABEL: define dso_local void @callee_struct_a16b_2(
// CHECK-SAME: i128 [[A_COERCE:%.*]], i32 noundef [[B:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[A:%.*]] = alloca [[STRUCT_S16:%.*]], align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[COERCE_DIVE:%.*]] = getelementptr inbounds nuw [[STRUCT_S16]], ptr [[A]], i32 0, i32 0
// CHECK-NEXT:    store i128 [[A_COERCE]], ptr [[COERCE_DIVE]], align 16
// CHECK-NEXT:    store i32 [[B]], ptr [[B_ADDR]], align 4
// CHECK-NEXT:    ret void
//
void callee_struct_a16b_2(struct S16 a, int b) {}


// CHECK-LABEL: define dso_local void @callee_struct_a16b_3(
// CHECK-SAME: i32 noundef [[A:%.*]], ptr noundef byval([[STRUCT_S16:%.*]]) align 16 [[B:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i32 [[A]], ptr [[A_ADDR]], align 4
// CHECK-NEXT:    ret void
//
void callee_struct_a16b_3(int a, struct S16 b) {}

// Test variable arguments
int f_va_callee(int, ...);

// CHECK-LABEL: define dso_local void @f_va_caller(
// CHECK-SAME: ) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[DOTCOMPOUNDLITERAL:%.*]] = alloca [[STRUCT_S16:%.*]], align 16
// CHECK-NEXT:    [[COERCE:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[BYVAL_TEMP:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[A:%.*]] = getelementptr inbounds nuw [[STRUCT_S16]], ptr [[DOTCOMPOUNDLITERAL]], i32 0, i32 0
// CHECK-NEXT:    store i32 6, ptr [[A]], align 4
// CHECK-NEXT:    [[ARRAYINIT_ELEMENT:%.*]] = getelementptr inbounds i32, ptr [[A]], i32 1
// CHECK-NEXT:    store i32 7, ptr [[ARRAYINIT_ELEMENT]], align 4
// CHECK-NEXT:    [[ARRAYINIT_ELEMENT1:%.*]] = getelementptr inbounds i32, ptr [[A]], i32 2
// CHECK-NEXT:    store i32 8, ptr [[ARRAYINIT_ELEMENT1]], align 4
// CHECK-NEXT:    [[ARRAYINIT_ELEMENT2:%.*]] = getelementptr inbounds i32, ptr [[A]], i32 3
// CHECK-NEXT:    store i32 9, ptr [[ARRAYINIT_ELEMENT2]], align 4
// CHECK-NEXT:    store double 4.000000e+00, ptr [[COERCE]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load i64, ptr [[COERCE]], align 8
// CHECK-NEXT:    store double 5.000000e+00, ptr [[BYVAL_TEMP]], align 8
// CHECK-NEXT:    [[CALL:%.*]] = call i32 (i32, ...) @f_va_callee(i32 noundef 1, i32 noundef 2, i64 noundef 3, i64 noundef [[TMP0]], ptr noundef byval(double) align 8 [[BYVAL_TEMP]], ptr noundef byval([[STRUCT_S16]]) align 16 [[DOTCOMPOUNDLITERAL]])
// CHECK-NEXT:    ret void
//
void f_va_caller(void) {
  f_va_callee(1, 2, 3LL, 4.0f, 5.0, (struct S16){6, 7, 8, 9});
}

// CHECK-LABEL: define dso_local i32 @f_va_1(
// CHECK-SAME: i32 noundef [[FMT_COERCE:%.*]], ...) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[FMT:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[FMT_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[VA:%.*]] = alloca [[STRUCT___VA_LIST_TAG:%.*]], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[COERCE_VAL_IP:%.*]] = inttoptr i32 [[FMT_COERCE]] to ptr
// CHECK-NEXT:    store ptr [[COERCE_VAL_IP]], ptr [[FMT]], align 4
// CHECK-NEXT:    [[FMT1:%.*]] = load ptr, ptr [[FMT]], align 4
// CHECK-NEXT:    store ptr [[FMT1]], ptr [[FMT_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.va_start.p0(ptr [[VA]])
// CHECK-NEXT:    [[__VA_STK:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[__VA_STK]], align 4
// CHECK-NEXT:    [[__VA_REG:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 1
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[__VA_REG]], align 4
// CHECK-NEXT:    [[__VA_NDX:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 2
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = lshr i32 [[TMP2]], 2
// CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], 1
// CHECK-NEXT:    [[TMP5:%.*]] = shl i32 [[TMP4]], 2
// CHECK-NEXT:    store i32 [[TMP5]], ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[COND:%.*]] = icmp ule i32 [[TMP4]], 6
// CHECK-NEXT:    br i1 [[COND]], label %[[USING_REGSAVEAREA:.*]], label %[[USING_OVERFLOW:.*]]
// CHECK:       [[USING_REGSAVEAREA]]:
// CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 [[TMP3]]
// CHECK-NEXT:    br label %[[CONT:.*]]
// CHECK:       [[USING_OVERFLOW]]:
// CHECK-NEXT:    [[COND_OVERFLOW:%.*]] = icmp ule i32 [[TMP3]], 6
// CHECK-NEXT:    [[TMP7:%.*]] = sub i32 8, [[TMP3]]
// CHECK-NEXT:    [[TMP8:%.*]] = select i1 [[COND_OVERFLOW]], i32 [[TMP7]], i32 0
// CHECK-NEXT:    [[TMP9:%.*]] = add i32 [[TMP3]], [[TMP8]]
// CHECK-NEXT:    [[TMP10:%.*]] = add i32 [[TMP4]], [[TMP8]]
// CHECK-NEXT:    [[TMP11:%.*]] = shl i32 [[TMP10]], 2
// CHECK-NEXT:    store i32 [[TMP11]], ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds i32, ptr [[TMP0]], i32 [[TMP9]]
// CHECK-NEXT:    br label %[[CONT]]
// CHECK:       [[CONT]]:
// CHECK-NEXT:    [[TMP13:%.*]] = phi ptr [ [[TMP6]], %[[USING_REGSAVEAREA]] ], [ [[TMP12]], %[[USING_OVERFLOW]] ]
// CHECK-NEXT:    [[TMP14:%.*]] = load i32, ptr [[TMP13]], align 4
// CHECK-NEXT:    store i32 [[TMP14]], ptr [[V]], align 4
// CHECK-NEXT:    call void @llvm.va_end.p0(ptr [[VA]])
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, ptr [[V]], align 4
// CHECK-NEXT:    ret i32 [[TMP15]]
//
int f_va_1(char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  int v = __builtin_va_arg(va, int);
  __builtin_va_end(va);

  return v;
}

// CHECK-LABEL: define dso_local i64 @f_va_2(
// CHECK-SAME: i32 noundef [[FMT_COERCE:%.*]], ...) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[FMT:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[FMT_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[VA:%.*]] = alloca [[STRUCT___VA_LIST_TAG:%.*]], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[COERCE_VAL_IP:%.*]] = inttoptr i32 [[FMT_COERCE]] to ptr
// CHECK-NEXT:    store ptr [[COERCE_VAL_IP]], ptr [[FMT]], align 4
// CHECK-NEXT:    [[FMT1:%.*]] = load ptr, ptr [[FMT]], align 4
// CHECK-NEXT:    store ptr [[FMT1]], ptr [[FMT_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.va_start.p0(ptr [[VA]])
// CHECK-NEXT:    [[__VA_STK:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[__VA_STK]], align 4
// CHECK-NEXT:    [[__VA_REG:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 1
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[__VA_REG]], align 4
// CHECK-NEXT:    [[__VA_NDX:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 2
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = lshr i32 [[TMP2]], 2
// CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], 1
// CHECK-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], -2
// CHECK-NEXT:    [[TMP6:%.*]] = add i32 [[TMP5]], 2
// CHECK-NEXT:    [[TMP7:%.*]] = shl i32 [[TMP6]], 2
// CHECK-NEXT:    store i32 [[TMP7]], ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[COND:%.*]] = icmp ule i32 [[TMP6]], 6
// CHECK-NEXT:    br i1 [[COND]], label %[[USING_REGSAVEAREA:.*]], label %[[USING_OVERFLOW:.*]]
// CHECK:       [[USING_REGSAVEAREA]]:
// CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 [[TMP5]]
// CHECK-NEXT:    br label %[[CONT:.*]]
// CHECK:       [[USING_OVERFLOW]]:
// CHECK-NEXT:    [[COND_OVERFLOW:%.*]] = icmp ule i32 [[TMP5]], 6
// CHECK-NEXT:    [[TMP9:%.*]] = sub i32 8, [[TMP5]]
// CHECK-NEXT:    [[TMP10:%.*]] = select i1 [[COND_OVERFLOW]], i32 [[TMP9]], i32 0
// CHECK-NEXT:    [[TMP11:%.*]] = add i32 [[TMP5]], [[TMP10]]
// CHECK-NEXT:    [[TMP12:%.*]] = add i32 [[TMP6]], [[TMP10]]
// CHECK-NEXT:    [[TMP13:%.*]] = shl i32 [[TMP12]], 2
// CHECK-NEXT:    store i32 [[TMP13]], ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds i32, ptr [[TMP0]], i32 [[TMP11]]
// CHECK-NEXT:    br label %[[CONT]]
// CHECK:       [[CONT]]:
// CHECK-NEXT:    [[TMP15:%.*]] = phi ptr [ [[TMP8]], %[[USING_REGSAVEAREA]] ], [ [[TMP14]], %[[USING_OVERFLOW]] ]
// CHECK-NEXT:    [[TMP16:%.*]] = load double, ptr [[TMP15]], align 4
// CHECK-NEXT:    store double [[TMP16]], ptr [[V]], align 8
// CHECK-NEXT:    call void @llvm.va_end.p0(ptr [[VA]])
// CHECK-NEXT:    [[TMP17:%.*]] = load double, ptr [[V]], align 8
// CHECK-NEXT:    store double [[TMP17]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    [[TMP18:%.*]] = load i64, ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret i64 [[TMP18]]
//
double f_va_2(char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  double v = __builtin_va_arg(va, double);
  __builtin_va_end(va);

  return v;
}

// CHECK-LABEL: define dso_local i64 @f_va_3(
// CHECK-SAME: i32 noundef [[FMT_COERCE:%.*]], ...) #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[RETVAL:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[FMT:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[FMT_ADDR:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[VA:%.*]] = alloca [[STRUCT___VA_LIST_TAG:%.*]], align 4
// CHECK-NEXT:    [[V:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[W:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[X:%.*]] = alloca double, align 8
// CHECK-NEXT:    [[COERCE_VAL_IP:%.*]] = inttoptr i32 [[FMT_COERCE]] to ptr
// CHECK-NEXT:    store ptr [[COERCE_VAL_IP]], ptr [[FMT]], align 4
// CHECK-NEXT:    [[FMT1:%.*]] = load ptr, ptr [[FMT]], align 4
// CHECK-NEXT:    store ptr [[FMT1]], ptr [[FMT_ADDR]], align 4
// CHECK-NEXT:    call void @llvm.va_start.p0(ptr [[VA]])
// CHECK-NEXT:    [[__VA_STK:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 0
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[__VA_STK]], align 4
// CHECK-NEXT:    [[__VA_REG:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 1
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[__VA_REG]], align 4
// CHECK-NEXT:    [[__VA_NDX:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 2
// CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[TMP3:%.*]] = lshr i32 [[TMP2]], 2
// CHECK-NEXT:    [[TMP4:%.*]] = add i32 [[TMP3]], 1
// CHECK-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], -2
// CHECK-NEXT:    [[TMP6:%.*]] = add i32 [[TMP5]], 2
// CHECK-NEXT:    [[TMP7:%.*]] = shl i32 [[TMP6]], 2
// CHECK-NEXT:    store i32 [[TMP7]], ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[COND:%.*]] = icmp ule i32 [[TMP6]], 6
// CHECK-NEXT:    br i1 [[COND]], label %[[USING_REGSAVEAREA:.*]], label %[[USING_OVERFLOW:.*]]
// CHECK:       [[USING_REGSAVEAREA]]:
// CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 [[TMP5]]
// CHECK-NEXT:    br label %[[CONT:.*]]
// CHECK:       [[USING_OVERFLOW]]:
// CHECK-NEXT:    [[COND_OVERFLOW:%.*]] = icmp ule i32 [[TMP5]], 6
// CHECK-NEXT:    [[TMP9:%.*]] = sub i32 8, [[TMP5]]
// CHECK-NEXT:    [[TMP10:%.*]] = select i1 [[COND_OVERFLOW]], i32 [[TMP9]], i32 0
// CHECK-NEXT:    [[TMP11:%.*]] = add i32 [[TMP5]], [[TMP10]]
// CHECK-NEXT:    [[TMP12:%.*]] = add i32 [[TMP6]], [[TMP10]]
// CHECK-NEXT:    [[TMP13:%.*]] = shl i32 [[TMP12]], 2
// CHECK-NEXT:    store i32 [[TMP13]], ptr [[__VA_NDX]], align 4
// CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds i32, ptr [[TMP0]], i32 [[TMP11]]
// CHECK-NEXT:    br label %[[CONT]]
// CHECK:       [[CONT]]:
// CHECK-NEXT:    [[TMP15:%.*]] = phi ptr [ [[TMP8]], %[[USING_REGSAVEAREA]] ], [ [[TMP14]], %[[USING_OVERFLOW]] ]
// CHECK-NEXT:    [[TMP16:%.*]] = load double, ptr [[TMP15]], align 4
// CHECK-NEXT:    store double [[TMP16]], ptr [[V]], align 8
// CHECK-NEXT:    [[__VA_STK2:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 0
// CHECK-NEXT:    [[TMP17:%.*]] = load ptr, ptr [[__VA_STK2]], align 4
// CHECK-NEXT:    [[__VA_REG3:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 1
// CHECK-NEXT:    [[TMP18:%.*]] = load ptr, ptr [[__VA_REG3]], align 4
// CHECK-NEXT:    [[__VA_NDX4:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 2
// CHECK-NEXT:    [[TMP19:%.*]] = load i32, ptr [[__VA_NDX4]], align 4
// CHECK-NEXT:    [[TMP20:%.*]] = lshr i32 [[TMP19]], 2
// CHECK-NEXT:    [[TMP21:%.*]] = add i32 [[TMP20]], 1
// CHECK-NEXT:    [[TMP22:%.*]] = shl i32 [[TMP21]], 2
// CHECK-NEXT:    store i32 [[TMP22]], ptr [[__VA_NDX4]], align 4
// CHECK-NEXT:    [[COND5:%.*]] = icmp ule i32 [[TMP21]], 6
// CHECK-NEXT:    br i1 [[COND5]], label %[[USING_REGSAVEAREA6:.*]], label %[[USING_OVERFLOW7:.*]]
// CHECK:       [[USING_REGSAVEAREA6]]:
// CHECK-NEXT:    [[TMP23:%.*]] = getelementptr inbounds i32, ptr [[TMP18]], i32 [[TMP20]]
// CHECK-NEXT:    br label %[[CONT9:.*]]
// CHECK:       [[USING_OVERFLOW7]]:
// CHECK-NEXT:    [[COND_OVERFLOW8:%.*]] = icmp ule i32 [[TMP20]], 6
// CHECK-NEXT:    [[TMP24:%.*]] = sub i32 8, [[TMP20]]
// CHECK-NEXT:    [[TMP25:%.*]] = select i1 [[COND_OVERFLOW8]], i32 [[TMP24]], i32 0
// CHECK-NEXT:    [[TMP26:%.*]] = add i32 [[TMP20]], [[TMP25]]
// CHECK-NEXT:    [[TMP27:%.*]] = add i32 [[TMP21]], [[TMP25]]
// CHECK-NEXT:    [[TMP28:%.*]] = shl i32 [[TMP27]], 2
// CHECK-NEXT:    store i32 [[TMP28]], ptr [[__VA_NDX4]], align 4
// CHECK-NEXT:    [[TMP29:%.*]] = getelementptr inbounds i32, ptr [[TMP17]], i32 [[TMP26]]
// CHECK-NEXT:    br label %[[CONT9]]
// CHECK:       [[CONT9]]:
// CHECK-NEXT:    [[TMP30:%.*]] = phi ptr [ [[TMP23]], %[[USING_REGSAVEAREA6]] ], [ [[TMP29]], %[[USING_OVERFLOW7]] ]
// CHECK-NEXT:    [[TMP31:%.*]] = load i32, ptr [[TMP30]], align 4
// CHECK-NEXT:    store i32 [[TMP31]], ptr [[W]], align 4
// CHECK-NEXT:    [[__VA_STK10:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 0
// CHECK-NEXT:    [[TMP32:%.*]] = load ptr, ptr [[__VA_STK10]], align 4
// CHECK-NEXT:    [[__VA_REG11:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 1
// CHECK-NEXT:    [[TMP33:%.*]] = load ptr, ptr [[__VA_REG11]], align 4
// CHECK-NEXT:    [[__VA_NDX12:%.*]] = getelementptr inbounds nuw [[STRUCT___VA_LIST_TAG]], ptr [[VA]], i32 0, i32 2
// CHECK-NEXT:    [[TMP34:%.*]] = load i32, ptr [[__VA_NDX12]], align 4
// CHECK-NEXT:    [[TMP35:%.*]] = lshr i32 [[TMP34]], 2
// CHECK-NEXT:    [[TMP36:%.*]] = add i32 [[TMP35]], 1
// CHECK-NEXT:    [[TMP37:%.*]] = and i32 [[TMP36]], -2
// CHECK-NEXT:    [[TMP38:%.*]] = add i32 [[TMP37]], 2
// CHECK-NEXT:    [[TMP39:%.*]] = shl i32 [[TMP38]], 2
// CHECK-NEXT:    store i32 [[TMP39]], ptr [[__VA_NDX12]], align 4
// CHECK-NEXT:    [[COND13:%.*]] = icmp ule i32 [[TMP38]], 6
// CHECK-NEXT:    br i1 [[COND13]], label %[[USING_REGSAVEAREA14:.*]], label %[[USING_OVERFLOW15:.*]]
// CHECK:       [[USING_REGSAVEAREA14]]:
// CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i32, ptr [[TMP33]], i32 [[TMP37]]
// CHECK-NEXT:    br label %[[CONT17:.*]]
// CHECK:       [[USING_OVERFLOW15]]:
// CHECK-NEXT:    [[COND_OVERFLOW16:%.*]] = icmp ule i32 [[TMP37]], 6
// CHECK-NEXT:    [[TMP41:%.*]] = sub i32 8, [[TMP37]]
// CHECK-NEXT:    [[TMP42:%.*]] = select i1 [[COND_OVERFLOW16]], i32 [[TMP41]], i32 0
// CHECK-NEXT:    [[TMP43:%.*]] = add i32 [[TMP37]], [[TMP42]]
// CHECK-NEXT:    [[TMP44:%.*]] = add i32 [[TMP38]], [[TMP42]]
// CHECK-NEXT:    [[TMP45:%.*]] = shl i32 [[TMP44]], 2
// CHECK-NEXT:    store i32 [[TMP45]], ptr [[__VA_NDX12]], align 4
// CHECK-NEXT:    [[TMP46:%.*]] = getelementptr inbounds i32, ptr [[TMP32]], i32 [[TMP43]]
// CHECK-NEXT:    br label %[[CONT17]]
// CHECK:       [[CONT17]]:
// CHECK-NEXT:    [[TMP47:%.*]] = phi ptr [ [[TMP40]], %[[USING_REGSAVEAREA14]] ], [ [[TMP46]], %[[USING_OVERFLOW15]] ]
// CHECK-NEXT:    [[TMP48:%.*]] = load double, ptr [[TMP47]], align 4
// CHECK-NEXT:    store double [[TMP48]], ptr [[X]], align 8
// CHECK-NEXT:    call void @llvm.va_end.p0(ptr [[VA]])
// CHECK-NEXT:    [[TMP49:%.*]] = load double, ptr [[V]], align 8
// CHECK-NEXT:    [[TMP50:%.*]] = load double, ptr [[X]], align 8
// CHECK-NEXT:    [[ADD:%.*]] = fadd double [[TMP49]], [[TMP50]]
// CHECK-NEXT:    store double [[ADD]], ptr [[RETVAL]], align 8
// CHECK-NEXT:    [[TMP51:%.*]] = load i64, ptr [[RETVAL]], align 8
// CHECK-NEXT:    ret i64 [[TMP51]]
//
double f_va_3(char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  double v = __builtin_va_arg(va, double);
  int w = __builtin_va_arg(va, int);
  double x = __builtin_va_arg(va, double);
  __builtin_va_end(va);

  return v + x;
}

#define	__malloc_like	__attribute__((__malloc__))

char *bufalloc () __malloc_like ;//__result_use_check;
extern void* malloc (unsigned size);

// CHECK: define dso_local noalias ptr @bufalloc() #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[BUF:%.*]] = alloca ptr, align 4
// CHECK-NEXT:    [[CALL:%.*]] = call ptr @malloc(i32 noundef 1024)
// CHECK-NEXT:    store ptr [[CALL]], ptr [[BUF]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[BUF]], align 4
// CHECK-NEXT:    ret ptr [[TMP0]]
//
char *bufalloc ()
{
  char* buf = malloc(1024);

  return buf;
}
