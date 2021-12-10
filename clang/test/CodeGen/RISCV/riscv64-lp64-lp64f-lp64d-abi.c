// RUN: %clang_cc1 -triple riscv64 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-abi lp64f -emit-llvm %s -o - \
// RUN:     | FileCheck %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +d -target-feature +f -target-abi lp64d -emit-llvm %s -o - \
// RUN:     | FileCheck %s

// This file contains test cases that will have the same output for the lp64,
// lp64f, and lp64d ABIs.

#include <stddef.h>
#include <stdint.h>

// CHECK-LABEL: define{{.*}} void @f_void()
void f_void(void) {}

// Scalar arguments and return values smaller than the word size are extended
// according to the sign of their type, up to 32 bits

// CHECK-LABEL: define{{.*}} zeroext i1 @f_scalar_0(i1 zeroext %x)
_Bool f_scalar_0(_Bool x) { return x; }

// CHECK-LABEL: define{{.*}} signext i8 @f_scalar_1(i8 signext %x)
int8_t f_scalar_1(int8_t x) { return x; }

// CHECK-LABEL: define{{.*}} zeroext i8 @f_scalar_2(i8 zeroext %x)
uint8_t f_scalar_2(uint8_t x) { return x; }

// CHECK-LABEL: define{{.*}} signext i32 @f_scalar_3(i32 signext %x)
uint32_t f_scalar_3(int32_t x) { return x; }

// CHECK-LABEL: define{{.*}} i64 @f_scalar_4(i64 %x)
int64_t f_scalar_4(int64_t x) { return x; }

// CHECK-LABEL: define{{.*}} float @f_fp_scalar_1(float %x)
float f_fp_scalar_1(float x) { return x; }

// CHECK-LABEL: define{{.*}} double @f_fp_scalar_2(double %x)
double f_fp_scalar_2(double x) { return x; }

// CHECK-LABEL: define{{.*}} fp128 @f_fp_scalar_3(fp128 %x)
long double f_fp_scalar_3(long double x) { return x; }

// Empty structs or unions are ignored.

struct empty_s {};

// CHECK-LABEL: define{{.*}} void @f_agg_empty_struct()
struct empty_s f_agg_empty_struct(struct empty_s x) {
  return x;
}

union empty_u {};

// CHECK-LABEL: define{{.*}} void @f_agg_empty_union()
union empty_u f_agg_empty_union(union empty_u x) {
  return x;
}

// Aggregates <= 2*xlen may be passed in registers, so will be coerced to
// integer arguments. The rules for return are the same.

struct tiny {
  uint16_t a, b, c, d;
};

// CHECK-LABEL: define{{.*}} void @f_agg_tiny(i64 %x.coerce)
void f_agg_tiny(struct tiny x) {
  x.a += x.b;
  x.c += x.d;
}

// CHECK-LABEL: define{{.*}} i64 @f_agg_tiny_ret()
struct tiny f_agg_tiny_ret() {
  return (struct tiny){1, 2, 3, 4};
}

typedef uint16_t v4i16 __attribute__((vector_size(8)));
typedef int64_t v1i64 __attribute__((vector_size(8)));

// CHECK-LABEL: define{{.*}} void @f_vec_tiny_v4i16(i64 %x.coerce)
void f_vec_tiny_v4i16(v4i16 x) {
  x[0] = x[1];
  x[2] = x[3];
}

// CHECK-LABEL: define{{.*}} i64 @f_vec_tiny_v4i16_ret()
v4i16 f_vec_tiny_v4i16_ret() {
  return (v4i16){1, 2, 3, 4};
}

// CHECK-LABEL: define{{.*}} void @f_vec_tiny_v1i64(i64 %x.coerce)
void f_vec_tiny_v1i64(v1i64 x) {
  x[0] = 114;
}

// CHECK-LABEL: define{{.*}} i64 @f_vec_tiny_v1i64_ret()
v1i64 f_vec_tiny_v1i64_ret() {
  return (v1i64){1};
}

struct small {
  int64_t a, *b;
};

// CHECK-LABEL: define{{.*}} void @f_agg_small([2 x i64] %x.coerce)
void f_agg_small(struct small x) {
  x.a += *x.b;
  x.b = &x.a;
}

// CHECK-LABEL: define{{.*}} [2 x i64] @f_agg_small_ret()
struct small f_agg_small_ret() {
  return (struct small){1, 0};
}

typedef uint16_t v8i16 __attribute__((vector_size(16)));
typedef __int128_t v1i128 __attribute__((vector_size(16)));

// CHECK-LABEL: define{{.*}} void @f_vec_small_v8i16(i128 %x.coerce)
void f_vec_small_v8i16(v8i16 x) {
  x[0] = x[7];
}

// CHECK-LABEL: define{{.*}} i128 @f_vec_small_v8i16_ret()
v8i16 f_vec_small_v8i16_ret() {
  return (v8i16){1, 2, 3, 4, 5, 6, 7, 8};
}

// CHECK-LABEL: define{{.*}} void @f_vec_small_v1i128(i128 %x.coerce)
void f_vec_small_v1i128(v1i128 x) {
  x[0] = 114;
}

// CHECK-LABEL: define{{.*}} i128 @f_vec_small_v1i128_ret()
v1i128 f_vec_small_v1i128_ret() {
  return (v1i128){1};
}

// Aggregates of 2*xlen size and 2*xlen alignment should be coerced to a
// single 2*xlen-sized argument, to ensure that alignment can be maintained if
// passed on the stack.

struct small_aligned {
  __int128_t a;
};

// CHECK-LABEL: define{{.*}} void @f_agg_small_aligned(i128 %x.coerce)
void f_agg_small_aligned(struct small_aligned x) {
  x.a += x.a;
}

// CHECK-LABEL: define{{.*}} i128 @f_agg_small_aligned_ret(i128 %x.coerce)
struct small_aligned f_agg_small_aligned_ret(struct small_aligned x) {
  return (struct small_aligned){10};
}

// Aggregates greater > 2*xlen will be passed and returned indirectly
struct large {
  int64_t a, b, c, d;
};

// CHECK-LABEL: define{{.*}} void @f_agg_large(%struct.large* %x)
void f_agg_large(struct large x) {
  x.a = x.b + x.c + x.d;
}

// The address where the struct should be written to will be the first
// argument
// CHECK-LABEL: define{{.*}} void @f_agg_large_ret(%struct.large* noalias sret(%struct.large) align 8 %agg.result, i32 signext %i, i8 signext %j)
struct large f_agg_large_ret(int32_t i, int8_t j) {
  return (struct large){1, 2, 3, 4};
}

typedef unsigned char v32i8 __attribute__((vector_size(32)));

// CHECK-LABEL: define{{.*}} void @f_vec_large_v32i8(<32 x i8>* %0)
void f_vec_large_v32i8(v32i8 x) {
  x[0] = x[7];
}

// CHECK-LABEL: define{{.*}} void @f_vec_large_v32i8_ret(<32 x i8>* noalias sret(<32 x i8>) align 32 %agg.result)
v32i8 f_vec_large_v32i8_ret() {
  return (v32i8){1, 2, 3, 4, 5, 6, 7, 8};
}

// Scalars passed on the stack should not have signext/zeroext attributes
// (they are anyext).

// CHECK-LABEL: define{{.*}} signext i32 @f_scalar_stack_1(i64 %a.coerce, [2 x i64] %b.coerce, i128 %c.coerce, %struct.large* %d, i8 zeroext %e, i8 signext %f, i8 %g, i8 %h)
int f_scalar_stack_1(struct tiny a, struct small b, struct small_aligned c,
                     struct large d, uint8_t e, int8_t f, uint8_t g, int8_t h) {
  return g + h;
}

// CHECK-LABEL: define{{.*}} signext i32 @f_scalar_stack_2(i32 signext %a, i128 %b, i64 %c, fp128 %d, <32 x i8>* %0, i8 zeroext %f, i8 %g, i8 %h)
int f_scalar_stack_2(int32_t a, __int128_t b, int64_t c, long double d, v32i8 e,
                     uint8_t f, int8_t g, uint8_t h) {
  return g + h;
}

// Ensure that scalars passed on the stack are still determined correctly in
// the presence of large return values that consume a register due to the need
// to pass a pointer.

// CHECK-LABEL: define{{.*}} void @f_scalar_stack_3(%struct.large* noalias sret(%struct.large) align 8 %agg.result, i32 signext %a, i128 %b, fp128 %c, <32 x i8>* %0, i8 zeroext %e, i8 %f, i8 %g)
struct large f_scalar_stack_3(uint32_t a, __int128_t b, long double c, v32i8 d,
                              uint8_t e, int8_t f, uint8_t g) {
  return (struct large){a, e, f, g};
}

// Ensure that ABI lowering happens as expected for vararg calls.
// Specifically, ensure that signext is emitted for varargs that will be
// passed in registers but not on the stack. Ensure this takes into account
// the use of "aligned" register pairs for varargs with 2*xlen alignment.

int f_va_callee(int, ...);

// CHECK-LABEL: define{{.*}} void @f_va_caller()
void f_va_caller() {
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i64 3, double 4.000000e+00, double 5.000000e+00, i64 {{%.*}}, [2 x i64] {{%.*}}, i128 {{%.*}}, %struct.large* {{%.*}})
  f_va_callee(1, 2, 3LL, 4.0f, 5.0, (struct tiny){6, 7, 8, 9},
              (struct small){10, NULL}, (struct small_aligned){11},
              (struct large){12, 13, 14, 15});
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, fp128 0xL00000000000000004001400000000000, i32 signext 6, i32 signext 7, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, 5.0L, 6, 7, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i128 {{%.*}}, i32 signext 6, i32 signext 7, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, (struct small_aligned){5}, 6, 7, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, [2 x i64] {{%.*}}, i32 signext 6, i32 signext 7, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, (struct small){5, NULL}, 6, 7, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i32 signext 5, fp128 0xL00000000000000004001800000000000, i32 7, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, 5, 6.0L, 7, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i32 signext 5, i128 {{%.*}}, i32 7, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, 5, (struct small_aligned){6}, 7, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i32 signext 5, [2 x i64] {{%.*}}, i32 signext 7, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, 5, (struct small){6, NULL}, 7, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i32 signext 5, i32 signext 6, fp128 0xL00000000000000004001C00000000000, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, 5, 6, 7.0L, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i32 signext 5, i32 signext 6, i128 {{%.*}}, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, 5, 6, (struct small_aligned){7}, 8, 9);
  // CHECK: call signext i32 (i32, ...) @f_va_callee(i32 signext 1, i32 signext 2, i32 signext 3, i32 signext 4, i32 signext 5, i32 signext 6, [2 x i64] {{.*}}, i32 8, i32 9)
  f_va_callee(1, 2, 3, 4, 5, 6, (struct small){7, NULL}, 8, 9);
}

// CHECK-LABEL: define{{.*}} signext i32 @f_va_1(i8* %fmt, ...) {{.*}} {
// CHECK:   [[FMT_ADDR:%.*]] = alloca i8*, align 8
// CHECK:   [[VA:%.*]] = alloca i8*, align 8
// CHECK:   [[V:%.*]] = alloca i32, align 4
// CHECK:   store i8* %fmt, i8** [[FMT_ADDR]], align 8
// CHECK:   [[VA1:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_start(i8* [[VA1]])
// CHECK:   [[ARGP_CUR:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK:   [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR]], i64 8
// CHECK:   store i8* [[ARGP_NEXT]], i8** [[VA]], align 8
// CHECK:   [[TMP0:%.*]] = bitcast i8* [[ARGP_CUR]] to i32*
// CHECK:   [[TMP1:%.*]] = load i32, i32* [[TMP0]], align 8
// CHECK:   store i32 [[TMP1]], i32* [[V]], align 4
// CHECK:   [[VA2:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK:   call void @llvm.va_end(i8* [[VA2]])
// CHECK:   [[TMP2:%.*]] = load i32, i32* [[V]], align 4
// CHECK:   ret i32 [[TMP2]]
// CHECK: }
int f_va_1(char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  int v = __builtin_va_arg(va, int);
  __builtin_va_end(va);

  return v;
}

// An "aligned" register pair (where the first register is even-numbered) is
// used to pass varargs with 2x xlen alignment and 2x xlen size. Ensure the
// correct offsets are used.

// CHECK-LABEL: @f_va_2(
// CHECK:         [[FMT_ADDR:%.*]] = alloca i8*, align 8
// CHECK-NEXT:    [[VA:%.*]] = alloca i8*, align 8
// CHECK-NEXT:    [[V:%.*]] = alloca fp128, align 16
// CHECK-NEXT:    store i8* [[FMT:%.*]], i8** [[FMT_ADDR]], align 8
// CHECK-NEXT:    [[VA1:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:    call void @llvm.va_start(i8* [[VA1]])
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = ptrtoint i8* [[ARGP_CUR]] to i64
// CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[TMP0]], 15
// CHECK-NEXT:    [[TMP2:%.*]] = and i64 [[TMP1]], -16
// CHECK-NEXT:    [[ARGP_CUR_ALIGNED:%.*]] = inttoptr i64 [[TMP2]] to i8*
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR_ALIGNED]], i64 16
// CHECK-NEXT:    store i8* [[ARGP_NEXT]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast i8* [[ARGP_CUR_ALIGNED]] to fp128*
// CHECK-NEXT:    [[TMP4:%.*]] = load fp128, fp128* [[TMP3]], align 16
// CHECK-NEXT:    store fp128 [[TMP4]], fp128* [[V]], align 16
// CHECK-NEXT:    [[VA2:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:    call void @llvm.va_end(i8* [[VA2]])
// CHECK-NEXT:    [[TMP5:%.*]] = load fp128, fp128* [[V]], align 16
// CHECK-NEXT:    ret fp128 [[TMP5]]
long double f_va_2(char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  long double v = __builtin_va_arg(va, long double);
  __builtin_va_end(va);

  return v;
}

// Two "aligned" register pairs.

// CHECK-LABEL: @f_va_3(
// CHECK:         [[FMT_ADDR:%.*]] = alloca i8*, align 8
// CHECK-NEXT:    [[VA:%.*]] = alloca i8*, align 8
// CHECK-NEXT:    [[V:%.*]] = alloca fp128, align 16
// CHECK-NEXT:    [[W:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[X:%.*]] = alloca fp128, align 16
// CHECK-NEXT:    store i8* [[FMT:%.*]], i8** [[FMT_ADDR]], align 8
// CHECK-NEXT:    [[VA1:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:    call void @llvm.va_start(i8* [[VA1]])
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = ptrtoint i8* [[ARGP_CUR]] to i64
// CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[TMP0]], 15
// CHECK-NEXT:    [[TMP2:%.*]] = and i64 [[TMP1]], -16
// CHECK-NEXT:    [[ARGP_CUR_ALIGNED:%.*]] = inttoptr i64 [[TMP2]] to i8*
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR_ALIGNED]], i64 16
// CHECK-NEXT:    store i8* [[ARGP_NEXT]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast i8* [[ARGP_CUR_ALIGNED]] to fp128*
// CHECK-NEXT:    [[TMP4:%.*]] = load fp128, fp128* [[TMP3]], align 16
// CHECK-NEXT:    store fp128 [[TMP4]], fp128* [[V]], align 16
// CHECK-NEXT:    [[ARGP_CUR2:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[ARGP_NEXT3:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR2]], i64 8
// CHECK-NEXT:    store i8* [[ARGP_NEXT3]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast i8* [[ARGP_CUR2]] to i32*
// CHECK-NEXT:    [[TMP6:%.*]] = load i32, i32* [[TMP5]], align 8
// CHECK-NEXT:    store i32 [[TMP6]], i32* [[W]], align 4
// CHECK-NEXT:    [[ARGP_CUR4:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint i8* [[ARGP_CUR4]] to i64
// CHECK-NEXT:    [[TMP8:%.*]] = add i64 [[TMP7]], 15
// CHECK-NEXT:    [[TMP9:%.*]] = and i64 [[TMP8]], -16
// CHECK-NEXT:    [[ARGP_CUR4_ALIGNED:%.*]] = inttoptr i64 [[TMP9]] to i8*
// CHECK-NEXT:    [[ARGP_NEXT5:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR4_ALIGNED]], i64 16
// CHECK-NEXT:    store i8* [[ARGP_NEXT5]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP10:%.*]] = bitcast i8* [[ARGP_CUR4_ALIGNED]] to fp128*
// CHECK-NEXT:    [[TMP11:%.*]] = load fp128, fp128* [[TMP10]], align 16
// CHECK-NEXT:    store fp128 [[TMP11]], fp128* [[X]], align 16
// CHECK-NEXT:    [[VA6:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:    call void @llvm.va_end(i8* [[VA6]])
// CHECK-NEXT:    [[TMP12:%.*]] = load fp128, fp128* [[V]], align 16
// CHECK-NEXT:    [[TMP13:%.*]] = load fp128, fp128* [[X]], align 16
// CHECK-NEXT:    [[ADD:%.*]] = fadd fp128 [[TMP12]], [[TMP13]]
// CHECK-NEXT:    ret fp128 [[ADD]]
long double f_va_3(char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  long double v = __builtin_va_arg(va, long double);
  int w = __builtin_va_arg(va, int);
  long double x = __builtin_va_arg(va, long double);
  __builtin_va_end(va);

  return v + x;
}

// CHECK-LABEL: @f_va_4(
// CHECK:         [[FMT_ADDR:%.*]] = alloca i8*, align 8
// CHECK-NEXT:    [[VA:%.*]] = alloca i8*, align 8
// CHECK-NEXT:    [[V:%.*]] = alloca i32, align 4
// CHECK-NEXT:    [[TS:%.*]] = alloca [[STRUCT_TINY:%.*]], align 2
// CHECK-NEXT:    [[SS:%.*]] = alloca [[STRUCT_SMALL:%.*]], align 8
// CHECK-NEXT:    [[LS:%.*]] = alloca [[STRUCT_LARGE:%.*]], align 8
// CHECK-NEXT:    [[RET:%.*]] = alloca i32, align 4
// CHECK-NEXT:    store i8* [[FMT:%.*]], i8** [[FMT_ADDR]], align 8
// CHECK-NEXT:    [[VA1:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:    call void @llvm.va_start(i8* [[VA1]])
// CHECK-NEXT:    [[ARGP_CUR:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[ARGP_NEXT:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR]], i64 8
// CHECK-NEXT:    store i8* [[ARGP_NEXT]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast i8* [[ARGP_CUR]] to i32*
// CHECK-NEXT:    [[TMP1:%.*]] = load i32, i32* [[TMP0]], align 8
// CHECK-NEXT:    store i32 [[TMP1]], i32* [[V]], align 4
// CHECK-NEXT:    [[ARGP_CUR2:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[ARGP_NEXT3:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR2]], i64 8
// CHECK-NEXT:    store i8* [[ARGP_NEXT3]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP2:%.*]] = bitcast i8* [[ARGP_CUR2]] to %struct.tiny*
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast %struct.tiny* [[TS]] to i8*
// CHECK-NEXT:    [[TMP4:%.*]] = bitcast %struct.tiny* [[TMP2]] to i8*
// CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 2 [[TMP3]], i8* align 8 [[TMP4]], i64 8, i1 false)
// CHECK-NEXT:    [[ARGP_CUR4:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[ARGP_NEXT5:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR4]], i64 16
// CHECK-NEXT:    store i8* [[ARGP_NEXT5]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast i8* [[ARGP_CUR4]] to %struct.small*
// CHECK-NEXT:    [[TMP6:%.*]] = bitcast %struct.small* [[SS]] to i8*
// CHECK-NEXT:    [[TMP7:%.*]] = bitcast %struct.small* [[TMP5]] to i8*
// CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP6]], i8* align 8 [[TMP7]], i64 16, i1 false)
// CHECK-NEXT:    [[ARGP_CUR6:%.*]] = load i8*, i8** [[VA]], align 8
// CHECK-NEXT:    [[ARGP_NEXT7:%.*]] = getelementptr inbounds i8, i8* [[ARGP_CUR6]], i64 8
// CHECK-NEXT:    store i8* [[ARGP_NEXT7]], i8** [[VA]], align 8
// CHECK-NEXT:    [[TMP8:%.*]] = bitcast i8* [[ARGP_CUR6]] to %struct.large**
// CHECK-NEXT:    [[TMP9:%.*]] = load %struct.large*, %struct.large** [[TMP8]], align 8
// CHECK-NEXT:    [[TMP10:%.*]] = bitcast %struct.large* [[LS]] to i8*
// CHECK-NEXT:    [[TMP11:%.*]] = bitcast %struct.large* [[TMP9]] to i8*
// CHECK-NEXT:    call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TMP10]], i8* align 8 [[TMP11]], i64 32, i1 false)
// CHECK-NEXT:    [[VA8:%.*]] = bitcast i8** [[VA]] to i8*
// CHECK-NEXT:    call void @llvm.va_end(i8* [[VA8]])
// CHECK-NEXT:    [[A:%.*]] = getelementptr inbounds [[STRUCT_TINY]], %struct.tiny* [[TS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP12:%.*]] = load i16, i16* [[A]], align 2
// CHECK-NEXT:    [[CONV:%.*]] = zext i16 [[TMP12]] to i64
// CHECK-NEXT:    [[A9:%.*]] = getelementptr inbounds [[STRUCT_SMALL]], %struct.small* [[SS]], i32 0, i32 0
// CHECK-NEXT:    [[TMP13:%.*]] = load i64, i64* [[A9]], align 8
// CHECK-NEXT:    [[ADD:%.*]] = add nsw i64 [[CONV]], [[TMP13]]
// CHECK-NEXT:    [[C:%.*]] = getelementptr inbounds [[STRUCT_LARGE]], %struct.large* [[LS]], i32 0, i32 2
// CHECK-NEXT:    [[TMP14:%.*]] = load i64, i64* [[C]], align 8
// CHECK-NEXT:    [[ADD10:%.*]] = add nsw i64 [[ADD]], [[TMP14]]
// CHECK-NEXT:    [[CONV11:%.*]] = trunc i64 [[ADD10]] to i32
// CHECK-NEXT:    store i32 [[CONV11]], i32* [[RET]], align 4
// CHECK-NEXT:    [[TMP15:%.*]] = load i32, i32* [[RET]], align 4
// CHECK-NEXT:    ret i32 [[TMP15]]
int f_va_4(char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  int v = __builtin_va_arg(va, int);
  struct tiny ts = __builtin_va_arg(va, struct tiny);
  struct small ss = __builtin_va_arg(va, struct small);
  struct large ls = __builtin_va_arg(va, struct large);
  __builtin_va_end(va);

  int ret = ts.a + ss.a + ls.c;

  return ret;
}
