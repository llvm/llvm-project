// RUN: %clang_cc1 -triple s390x-ibm-zos \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-feature +vector \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefixes=CHECK,CHECKI128 %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu z13 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefixes=CHECK,CHECKI128 %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch11 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefixes=CHECK,CHECKI128 %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu z14 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefixes=CHECK,CHECKI128 %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch12 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefixes=CHECK,CHECKI128 %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu z15 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefixes=CHECK,CHECKI128 %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch13 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck --check-prefixes=CHECK,CHECKI128 %s

// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch11 \
// RUN:   -DTEST_VEC -fzvector -emit-llvm -no-enable-noundef-analysis \
// RUN:   -o - %s | FileCheck --check-prefixes=CHECK,CHECKVEC,CHECKI128 %s

// Scalar types

signed char pass_schar(signed char arg) { return arg; }
// CHECK-LABEL: define signext i8 @pass_schar(i8 signext %{{.*}})

unsigned char pass_uchar(unsigned char arg) { return arg; }
// CHECK-LABEL: define zeroext i8 @pass_uchar(i8 zeroext %{{.*}})

short pass_short(short arg) { return arg; }
// CHECK-LABEL: define signext i16 @pass_short(i16 signext %{{.*}})

int pass_int(int arg) { return arg; }
// CHECK-LABEL: define signext i32 @pass_int(i32 signext %{{.*}})

long pass_long(long arg) { return arg; }
// CHECK-LABEL: define signext i64 @pass_long(i64 signext %{{.*}})

long long pass_longlong(long long arg) { return arg; }
// CHECK-LABEL: define i64 @pass_longlong(i64 %{{.*}})

#ifdef __VX__
__int128 pass_int128(__int128 arg) { return arg; }
// CHECKI128-LABEL: define i128 @pass_int128(i128 %{{.*}})
#endif

float pass_float(float arg) { return arg; }
// CHECK-LABEL: define float @pass_float(float %{{.*}})

double pass_double(double arg) { return arg; }
// CHECK-LABEL: define double @pass_double(double %{{.*}})

long double pass_longdouble(long double arg) { return arg; }
// CHECK-LABEL: define fp128 @pass_longdouble(fp128 %{{.*}})

enum Color { Red, Blue };
enum Color pass_enum(enum Color arg) { return arg; }
// CHECK-LABEL: define zeroext i32 @pass_enum(i32 zeroext %{{.*}})

#ifdef TEST_VEC
vector unsigned int pass_vector(vector unsigned int arg) { return arg; };
// CHECKVEC-LABEL: define <4 x i32> @pass_vector(<4 x i32> %{{.*}})

struct SingleVec { vector unsigned int v; };
struct SingleVec pass_SingleVec_agg(struct SingleVec arg) { return arg; };
// CHECKVEC-LABEL: define inreg [2 x i64] @pass_SingleVec_agg([2 x i64] %{{.*}})
#endif

// Complex types

_Complex float pass_complex_float(_Complex float arg) { return arg; }
// CHECK-LABEL: define { float, float } @pass_complex_float({ float, float } %{{.*}})

_Complex double pass_complex_double(_Complex double arg) { return arg; }
// CHECK-LABEL: define { double, double } @pass_complex_double({ double, double } %{{.*}})

_Complex long double pass_complex_longdouble(_Complex long double arg) { return arg; }
// CHECK-LABEL: define { fp128, fp128 } @pass_complex_longdouble({ fp128, fp128 } %{{.*}})

// Verify that the following are complex-like types
struct complexlike_float { float re, im; };
struct complexlike_float pass_complexlike_float(struct complexlike_float arg) { return arg; }
// CHECK-LABEL: define %struct.complexlike_float @pass_complexlike_float({ float, float } %{{.*}})

struct complexlike_double { double re, im; };
struct complexlike_double pass_complexlike_double(struct complexlike_double arg) { return arg; }
// CHECK-LABEL: define %struct.complexlike_double @pass_complexlike_double({ double, double } %{{.*}})

struct complexlike_longdouble { long double re, im; };
struct complexlike_longdouble pass_complexlike_longdouble(struct complexlike_longdouble arg) { return arg; }
// CHECK-LABEL: define %struct.complexlike_longdouble @pass_complexlike_longdouble({ fp128, fp128 } %{{.*}})

// Aggregate types

struct agg_1byte { char a[1]; };
struct agg_1byte pass_agg_1byte(struct agg_1byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_1byte(i64 %{{.*}})

struct agg_2byte { char a[2]; };
struct agg_2byte pass_agg_2byte(struct agg_2byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_2byte(i64 %{{.*}})

struct agg_3byte { char a[3]; };
struct agg_3byte pass_agg_3byte(struct agg_3byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_3byte(i64 %{{.*}})

struct agg_4byte { char a[4]; };
struct agg_4byte pass_agg_4byte(struct agg_4byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_4byte(i64 %{{.*}})

struct agg_5byte { char a[5]; };
struct agg_5byte pass_agg_5byte(struct agg_5byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_5byte(i64 %{{.*}})

struct agg_6byte { char a[6]; };
struct agg_6byte pass_agg_6byte(struct agg_6byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_6byte(i64 %{{.*}})

struct agg_7byte { char a[7]; };
struct agg_7byte pass_agg_7byte(struct agg_7byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_7byte(i64 %{{.*}})

struct agg_8byte { char a[8]; };
struct agg_8byte pass_agg_8byte(struct agg_8byte arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_8byte(i64 %{{.*}})

struct agg_9byte { char a[9]; };
struct agg_9byte pass_agg_9byte(struct agg_9byte arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @pass_agg_9byte([2 x i64] %{{.*}})

struct agg_16byte { char a[16]; };
struct agg_16byte pass_agg_16byte(struct agg_16byte arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @pass_agg_16byte([2 x i64] %{{.*}})

struct agg_24byte { char a[24]; };
struct agg_24byte pass_agg_24byte(struct agg_24byte arg) { return arg; }
// CHECK-LABEL: define inreg [3 x i64] @pass_agg_24byte([3 x i64] %{{.*}})

struct agg_25byte { char a[25]; };
struct agg_25byte pass_agg_25byte(struct agg_25byte arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_25byte(ptr dead_on_unwind noalias writable sret{{.*}} align 1 %{{.*}}, [4 x i64] %{{.*}})

// Check that a float-like aggregate type is really passed as aggregate
struct agg_float { float a; };
struct agg_float pass_agg_float(struct agg_float arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_float(i64 %{{.*}})

// Verify that the following are *not* float-like aggregate types

struct agg_nofloat2 { float a; int b; };
struct agg_nofloat2 pass_agg_nofloat2(struct agg_nofloat2 arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_nofloat2(i64 %{{.*}})

struct agg_nofloat3 { float a; int : 0; };
struct agg_nofloat3 pass_agg_nofloat3(struct agg_nofloat3 arg) { return arg; }
// CHECK-LABEL: define inreg [1 x i64] @pass_agg_nofloat3(i64 %{{.*}})

// Accessing variable argument lists

// z/OS has two different implementations for variable argument handling.
// Functions ending with _e test the extended variant of vararg functions
// (__builtin_va_start, __builtin_va_arg, __builtin_va_end). The type of
// va_list is __builtin_va_list.
// Functions ending with _s test the standard variant of vararg functions
// (__builtin_zos_va_start, __builtin_va_arg, __builtin_zos_va_end). The type of
// va_list is __builtin_va_list.

int dofmt_e(const char *fmt, ...) {
  __builtin_va_list va;

  __builtin_va_start(va, fmt);
  int v = __builtin_va_arg(va, int);
  __builtin_va_end(va);

  return v;
}
// CHECK-LABEL: define signext i32 @dofmt_e(ptr %{{.*}}, ...)
// CHECK: [[FMT_ADDR:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: [[VA:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: [[V:%[._a-z0-9]+]] = alloca i32, align 4
// CHECK: store ptr %{{.*}}, ptr [[FMT_ADDR]], align 8
// CHECK: call void @llvm.va_start.p0(ptr [[VA]])
// CHECK: [[ARGP_CURR:%[._a-z0-9]+]] = load ptr, ptr [[VA]], align 8
// CHECK: [[ARGP_NEXT:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_CURR]], i64 8
// CHECK: store ptr [[ARGP_NEXT]], ptr [[VA]], align 8
// CHECK: [[V_ADDR:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_CURR]], i64 4
// CHECK: [[VAL:%[._a-z0-9]+]] = load i32, ptr [[V_ADDR]], align 4
// CHECK: store i32 [[VAL]], ptr [[V]], align 4
// CHECK: call void @llvm.va_end.p0(ptr [[VA]])
// CHECK: [[VAL2:%[._a-z0-9]+]] = load i32, ptr [[V]], align 4
// CHECK: ret i32 [[VAL2]]

int dofmt_s(const char *fmt, ...) {
  __builtin_zos_va_list va;

  __builtin_zos_va_start(va, fmt);
  int v = __builtin_va_arg(va, int);
  __builtin_zos_va_end(va);

  return v;
}
// CHECK-LABEL: define signext i32 @dofmt_s(ptr %{{.*}}, ...)
// CHECK: [[FMT_ADDR:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: [[VA:%[._a-z0-9]+]] = alloca [2 x ptr], align 8
// CHECK: [[V:%[._a-z0-9]+]] = alloca i32, align 4
// CHECK: store ptr %{{.*}}, ptr [[FMT_ADDR]], align 8
// CHECK: [[DECAY1:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VA]], i64 0, i64 0
// CHECK: [[VALIST_CURR1:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[DECAY1]], i64 0, i64 0
// CHECK: store ptr null, ptr [[VALIST_CURR1]], align 8
// CHECK: [[VALIST_NEXT1:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[DECAY1]], i64 0, i64 1
// CHECK: call void @llvm.va_start.p0(ptr [[VALIST_NEXT1]])
// CHECK: [[DECAY2:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VA]], i64 0, i64 0
// CHECK: [[VALIST_CURR2:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[DECAY2]], i64 0, i64 0
// CHECK: [[VALIST_NEXT2:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[DECAY2]], i64 0, i64 1
// CHECK: [[ARGP_NEXT:%[._a-z0-9]+]] = load ptr, ptr [[VALIST_NEXT2]], align 8
// CHECK: %0 = getelementptr inbounds i8, ptr [[ARGP_NEXT]], i32 7
// CHECK: [[ARGP_NEXT_ALIGNED:%[._a-z0-9]+]] = call ptr @llvm.ptrmask.p0.i64(ptr %0, i64 -8)
// CHECK: store ptr [[ARGP_NEXT_ALIGNED]], ptr [[VALIST_CURR2]], align 8
// CHECK: [[ARGP_NEXT_NEXT:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_NEXT_ALIGNED]], i64 4
// CHECK: store ptr [[ARGP_NEXT_NEXT]], ptr [[VALIST_NEXT2]], align 8
// CHECK: [[V_ADDR:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_NEXT_ALIGNED]], i64 4
// CHECK: [[VAL:%[._a-z0-9]+]] = load i32, ptr [[V_ADDR]], align 4
// CHECK: store i32 [[VAL]], ptr [[V]], align 4
// CHECK: [[DECAY3:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VA]], i64 0, i64 0
// CHECK: [[VALIST_CURR3:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[DECAY3]], i64 0, i64 0
// CHECK: store ptr null, ptr [[VALIST_CURR3]], align 8
// CHECK: [[VALIST_NEXT3:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[DECAY3]], i64 0, i64 1
// CHECK: call void @llvm.va_end.p0(ptr [[VALIST_NEXT3]])
// CHECK: [[VAL2:%[._a-z0-9]+]] = load i32, ptr [[V]], align 4
// CHECK: ret i32 [[VAL2]]

int va_int_e(__builtin_va_list l) { return __builtin_va_arg(l, int); }
// CHECK-LABEL: define signext i32 @va_int_e(ptr %{{.*}})
// CHECK: [[L_ADDR:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: store ptr %{{.*}}, ptr [[L_ADDR]], align 8
// CHECK: [[ARGP_CURR:%[._a-z0-9]+]] = load ptr, ptr [[L_ADDR]], align 8
// CHECK: [[ARGP_NEXT:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_CURR]], i64 8
// CHECK: store ptr [[ARGP_NEXT]], ptr [[L_ADDR]], align 8
// CHECK: [[V_ADDR:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_CURR]], i64 4
// CHECK: [[VAL:%[._a-z0-9]+]] = load i32, ptr [[V_ADDR]], align 4
// CHECK: ret i32 [[VAL]]

int va_int_s(__builtin_zos_va_list l) { return __builtin_va_arg(l, int); }
// CHECK-LABEL: define signext i32 @va_int_s(ptr %{{.*}})
// CHECK: [[L_ADDR:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: store ptr %{{.*}}, ptr [[L_ADDR]], align 8
// CHECK: [[VALIST:%[._a-z0-9]+]] = load ptr, ptr [[L_ADDR]], align 8
// CHECK: [[VALIST_CURR:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VALIST]], i64 0, i64 0
// CHECK: [[VALIST_NEXT:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VALIST]], i64 0, i64 1
// CHECK: [[ARGP_NEXT:%[._a-z0-9]+]] = load ptr, ptr [[VALIST_NEXT]], align 8
// CHECK: %1 = getelementptr inbounds i8, ptr %arg.next, i32 7
// CHECK: [[ARGP_NEXT_ALIGNED]] = call ptr @llvm.ptrmask.p0.i64(ptr %1, i64 -8)
// CHECK: store ptr [[ARGP_NEXT_ALIGNED]], ptr [[VALIST_CURR]], align 8
// CHECK: [[ARGP_NEXT_NEXT:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_NEXT_ALIGNED]], i64 4
// CHECK: store ptr [[ARGP_NEXT_NEXT]], ptr [[VALIST_NEXT]], align 8
// CHECK: [[V_ADDR:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_NEXT_ALIGNED]], i64 4
// CHECK: [[VAL:%[._a-z0-9]+]] = load i32, ptr [[V_ADDR]], align 4
// CHECK: ret i32 [[VAL]]

long va_long_e(__builtin_va_list l) { return __builtin_va_arg(l, long); }
// CHECK-LABEL: define signext i64 @va_long_e(ptr %{{.*}})
// CHECK: [[L_ADDR:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: store ptr %{{.*}}, ptr [[L_ADDR]], align 8
// CHECK: [[ARGP_CURR:%[._a-z0-9]+]] = load ptr, ptr [[L_ADDR]], align 8
// CHECK: [[ARGP_NEXT:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_CURR]], i64 8
// CHECK: store ptr [[ARGP_NEXT]], ptr [[L_ADDR]], align 8
// CHECK: [[VAL:%[._a-z0-9]+]] = load i64, ptr [[ARGP_CURR]], align 8
// CHECK: ret i64 [[VAL]]

long va_long_s(__builtin_zos_va_list l) { return __builtin_va_arg(l, long); }
// CHECK-LABEL: define signext i64 @va_long_s(ptr %{{.*}})
// CHECK: [[L_ADDR:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: store ptr %{{.*}}, ptr [[L_ADDR]], align 8
// CHECK: [[VALIST:%[._a-z0-9]+]] = load ptr, ptr [[L_ADDR]], align 8
// CHECK: [[VALIST_CURR:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VALIST]], i64 0, i64 0
// CHECK: [[VALIST_NEXT:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VALIST]], i64 0, i64 1
// CHECK: [[ARGP_NEXT:%[._a-z0-9]+]] = load ptr, ptr [[VALIST_NEXT]], align 8
// CHECK: %1 = getelementptr inbounds i8, ptr %arg.next, i32 7
// CHECK: %arg.next.aligned = call ptr @llvm.ptrmask.p0.i64(ptr %1, i64 -8)
// CHECK: store ptr [[ARGP_NEXT_ALIGNED]], ptr [[VALIST_CURR]], align 8
// CHECK: [[ARGP_NEXT_NEXT:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_NEXT_ALIGNED]], i64 8
// CHECK: store ptr [[ARGP_NEXT_NEXT]], ptr [[VALIST_NEXT]], align 8
// CHECK: [[VAL:%[._a-z0-9]+]] = load i64, ptr [[ARGP_NEXT_ALIGNED]], align 8
// CHECK: ret i64 [[VAL]]

struct agg_threedouble {
  double a, b, c;
};
struct agg_threedouble va_3double_s(__builtin_zos_va_list l) {
  return __builtin_va_arg(l, struct agg_threedouble);
}
// CHECK-LABEL: define inreg [3 x i64] @va_3double_s(ptr %{{.*}})
// CHECK: [[RETVAL:%[._a-z0-9]+]] = alloca %struct.agg_threedouble, align 8
// CHECK: [[L_ADDR:%[._a-z0-9]+]] = alloca ptr, align 8
// CHECK: store ptr %{{.*}}, ptr [[L_ADDR]], align 8
// CHECK: [[VALIST:%[._a-z0-9]+]] = load ptr, ptr [[L_ADDR]], align 8
// CHECK: [[VALIST_CURR:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VALIST]], i64 0, i64 0
// CHECK: [[VALIST_NEXT:%[._a-z0-9]+]] = getelementptr inbounds [2 x ptr], ptr [[VALIST]], i64 0, i64 1
// CHECK: [[ARGP_NEXT:%[._a-z0-9]+]] = load ptr, ptr [[VALIST_NEXT]], align 8
// CHECK: %1 = getelementptr inbounds i8, ptr %arg.next, i32 7
// CHECK: [[ARGP_NEXT_ALIGNED]] = call ptr @llvm.ptrmask.p0.i64(ptr %1, i64 -8)
// CHECK: store ptr [[ARGP_NEXT_ALIGNED]], ptr [[VALIST_CURR]], align 8
// CHECK: [[ARGP_NEXT_NEXT:%[._a-z0-9]+]] = getelementptr inbounds i8, ptr [[ARGP_NEXT_ALIGNED]], i64 24
// CHECK: store ptr [[ARGP_NEXT_NEXT]], ptr [[VALIST_NEXT]], align 8
// CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[ARGP_NEXT_ALIGNED]], i64 24, i1 false)
// CHECK: [[RETVAL2:%[._a-z0-9]+]] = load [3 x i64], ptr [[RETVAL]], align 8
// CHECK: ret [3 x i64] [[RETVAL2]]
