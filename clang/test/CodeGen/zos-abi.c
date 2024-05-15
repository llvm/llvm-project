// RUN: %clang_cc1 -triple s390x-ibm-zos \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-feature +vector \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu z13 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch11 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu z14 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch12 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu z15 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch13 \
// RUN:   -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple s390x-ibm-zos -target-cpu arch11 \
// RUN:   -DTEST_VEC -fzvector -emit-llvm -no-enable-noundef-analysis \
// RUN:   -o - %s | FileCheck --check-prefixes=CHECKVEC %s

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
// CHECK-LABEL: define i64 @pass_long(i64 %{{.*}})

long long pass_longlong(long long arg) { return arg; }
// CHECK-LABEL: define i64 @pass_longlong(i64 %{{.*}})

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

// Structures with extra padding are not considered complex types.
struct complexlike_float_padded1 {
  float x __attribute__((aligned(8)));
  float y __attribute__((aligned(8)));
};
struct complexlike_float_padded1 pass_complexlike_float_padded1(struct complexlike_float_padded1 arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @pass_complexlike_float_padded1([2 x i64] %{{.*}})
struct complexlike_float_padded2 {
  float x;
  float y;
} __attribute__((aligned(16)));
struct complexlike_float_padded2 pass_complexlike_float_padded2(struct complexlike_float_padded2 arg) { return arg; }
// CHECK-LABEL: define inreg [2 x i64] @pass_complexlike_float_padded2([2 x i64] %{{.*}})

typedef double align32_double __attribute__((aligned(32)));
struct complexlike_double_padded {
  align32_double x;
  double y;
};
struct complexlike_double_padded pass_complexlike_double_padded(struct complexlike_double_padded arg) { return arg; }
// CHECK-LABEL: define void @pass_complexlike_double_padded(ptr {{.*}} sret(%struct.complexlike_double_padded) align 32 %{{.*}}, [4 x i64] %{{.*}})

// Unnamed types

int pass_unnamed_int(int) { return 0; }
// CHECK-LABEL: define signext i32 @pass_unnamed_int(i32 signext %{{.*}})

signed char pass_unnamed_schar(signed char) { return '0'; }
// CHECK-LABEL: define signext i8 @pass_unnamed_schar(i8 signext %{{.*}})

long double pass_unnamed_longdouble(long double) { return 0; }
// CHECK-LABEL: define fp128 @pass_unnamed_longdouble(fp128 %{{.*}})

// Aggregate types

struct agg_1byte { char a[1]; };
struct agg_1byte pass_agg_1byte(struct agg_1byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_1byte(i64 %{{.*}})

struct agg_2byte { char a[2]; };
struct agg_2byte pass_agg_2byte(struct agg_2byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_2byte(i64 %{{.*}})

struct agg_3byte { char a[3]; };
struct agg_3byte pass_agg_3byte(struct agg_3byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_3byte(i64 %{{.*}})

struct agg_4byte { char a[4]; };
struct agg_4byte pass_agg_4byte(struct agg_4byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_4byte(i64 %{{.*}})

struct agg_5byte { char a[5]; };
struct agg_5byte pass_agg_5byte(struct agg_5byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_5byte(i64 %{{.*}})

struct agg_6byte { char a[6]; };
struct agg_6byte pass_agg_6byte(struct agg_6byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_6byte(i64 %{{.*}})

struct agg_7byte { char a[7]; };
struct agg_7byte pass_agg_7byte(struct agg_7byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_7byte(i64 %{{.*}})

struct agg_8byte { char a[8]; };
struct agg_8byte pass_agg_8byte(struct agg_8byte arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_8byte(i64 %{{.*}})

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
// CHECK-LABEL: define inreg i64 @pass_agg_float(i64 %{{.*}})

// Verify that the following are *not* float-like aggregate types

struct agg_nofloat2 { float a; int b; };
struct agg_nofloat2 pass_agg_nofloat2(struct agg_nofloat2 arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_nofloat2(i64 %{{.*}})

struct agg_nofloat3 { float a; int : 0; };
struct agg_nofloat3 pass_agg_nofloat3(struct agg_nofloat3 arg) { return arg; }
// CHECK-LABEL: define inreg i64 @pass_agg_nofloat3(i64 %{{.*}})
