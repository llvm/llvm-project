// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-feature +vector \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu z13 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu arch11 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu z14 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu arch12 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu z15 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu arch13 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu arch13 \
// RUN:   -emit-llvm -o - %s -mfloat-abi soft | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SOFT-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu arch14 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HARD-FLOAT
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple s390x-linux-gnu -target-cpu arch14 \
// RUN:   -emit-llvm -o - %s -mfloat-abi soft | FileCheck %s \
// RUN:   --check-prefixes=CHECK,SOFT-FLOAT

// Scalar types

char pass_char(char arg) { return arg; }
// CHECK-LABEL: define{{.*}} signext i8 @pass_char(i8 signext %{{.*}})

short pass_short(short arg) { return arg; }
// CHECK-LABEL: define{{.*}} signext i16 @pass_short(i16 signext %{{.*}})

int pass_int(int arg) { return arg; }
// CHECK-LABEL: define{{.*}} signext i32 @pass_int(i32 signext %{{.*}})

long pass_long(long arg) { return arg; }
// CHECK-LABEL: define{{.*}} i64 @pass_long(i64 %{{.*}})

long long pass_longlong(long long arg) { return arg; }
// CHECK-LABEL: define{{.*}} i64 @pass_longlong(i64 %{{.*}})

__int128 pass_int128(__int128 arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_int128(i128* noalias sret(i128) align 16 %{{.*}}, i128* %0)

float pass_float(float arg) { return arg; }
// CHECK-LABEL: define{{.*}} float @pass_float(float %{{.*}})

double pass_double(double arg) { return arg; }
// CHECK-LABEL: define{{.*}} double @pass_double(double %{{.*}})

long double pass_longdouble(long double arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_longdouble(fp128* noalias sret(fp128) align 8 %{{.*}}, fp128* %0)


// Complex types

_Complex char pass_complex_char(_Complex char arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_char({ i8, i8 }* noalias sret({ i8, i8 }) align 1 %{{.*}}, { i8, i8 }* %{{.*}}arg)

_Complex short pass_complex_short(_Complex short arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_short({ i16, i16 }* noalias sret({ i16, i16 }) align 2 %{{.*}}, { i16, i16 }* %{{.*}}arg)

_Complex int pass_complex_int(_Complex int arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_int({ i32, i32 }* noalias sret({ i32, i32 }) align 4 %{{.*}}, { i32, i32 }* %{{.*}}arg)

_Complex long pass_complex_long(_Complex long arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_long({ i64, i64 }* noalias sret({ i64, i64 }) align 8 %{{.*}}, { i64, i64 }* %{{.*}}arg)

_Complex long long pass_complex_longlong(_Complex long long arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_longlong({ i64, i64 }* noalias sret({ i64, i64 }) align 8 %{{.*}}, { i64, i64 }* %{{.*}}arg)

_Complex float pass_complex_float(_Complex float arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_float({ float, float }* noalias sret({ float, float }) align 4 %{{.*}}, { float, float }* %{{.*}}arg)

_Complex double pass_complex_double(_Complex double arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_double({ double, double }* noalias sret({ double, double }) align 8 %{{.*}}, { double, double }* %{{.*}}arg)

_Complex long double pass_complex_longdouble(_Complex long double arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_complex_longdouble({ fp128, fp128 }* noalias sret({ fp128, fp128 }) align 8 %{{.*}}, { fp128, fp128 }* %{{.*}}arg)


// Aggregate types

struct agg_1byte { char a[1]; };
struct agg_1byte pass_agg_1byte(struct agg_1byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_1byte(%struct.agg_1byte* noalias sret(%struct.agg_1byte) align 1 %{{.*}}, i8 %{{.*}})

struct agg_2byte { char a[2]; };
struct agg_2byte pass_agg_2byte(struct agg_2byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_2byte(%struct.agg_2byte* noalias sret(%struct.agg_2byte) align 1 %{{.*}}, i16 %{{.*}})

struct agg_3byte { char a[3]; };
struct agg_3byte pass_agg_3byte(struct agg_3byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_3byte(%struct.agg_3byte* noalias sret(%struct.agg_3byte) align 1 %{{.*}}, %struct.agg_3byte* %{{.*}})

struct agg_4byte { char a[4]; };
struct agg_4byte pass_agg_4byte(struct agg_4byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_4byte(%struct.agg_4byte* noalias sret(%struct.agg_4byte) align 1 %{{.*}}, i32 %{{.*}})

struct agg_5byte { char a[5]; };
struct agg_5byte pass_agg_5byte(struct agg_5byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_5byte(%struct.agg_5byte* noalias sret(%struct.agg_5byte) align 1 %{{.*}}, %struct.agg_5byte* %{{.*}})

struct agg_6byte { char a[6]; };
struct agg_6byte pass_agg_6byte(struct agg_6byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_6byte(%struct.agg_6byte* noalias sret(%struct.agg_6byte) align 1 %{{.*}}, %struct.agg_6byte* %{{.*}})

struct agg_7byte { char a[7]; };
struct agg_7byte pass_agg_7byte(struct agg_7byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_7byte(%struct.agg_7byte* noalias sret(%struct.agg_7byte) align 1 %{{.*}}, %struct.agg_7byte* %{{.*}})

struct agg_8byte { char a[8]; };
struct agg_8byte pass_agg_8byte(struct agg_8byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_8byte(%struct.agg_8byte* noalias sret(%struct.agg_8byte) align 1 %{{.*}}, i64 %{{.*}})

struct agg_16byte { char a[16]; };
struct agg_16byte pass_agg_16byte(struct agg_16byte arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_16byte(%struct.agg_16byte* noalias sret(%struct.agg_16byte) align 1 %{{.*}}, %struct.agg_16byte* %{{.*}})


// Float-like aggregate types

struct agg_float { float a; };
struct agg_float pass_agg_float(struct agg_float arg) { return arg; }
// HARD-FLOAT-LABEL: define{{.*}} void @pass_agg_float(%struct.agg_float* noalias sret(%struct.agg_float) align 4 %{{.*}}, float %{{.*}})
// SOFT-FLOAT-LABEL: define{{.*}} void @pass_agg_float(%struct.agg_float* noalias sret(%struct.agg_float) align 4 %{{.*}}, i32 %{{.*}})

struct agg_double { double a; };
struct agg_double pass_agg_double(struct agg_double arg) { return arg; }
// HARD-FLOAT-LABEL: define{{.*}} void @pass_agg_double(%struct.agg_double* noalias sret(%struct.agg_double) align 8 %{{.*}}, double %{{.*}})
// SOFT-FLOAT-LABEL: define{{.*}} void @pass_agg_double(%struct.agg_double* noalias sret(%struct.agg_double) align 8 %{{.*}}, i64 %{{.*}})

struct agg_longdouble { long double a; };
struct agg_longdouble pass_agg_longdouble(struct agg_longdouble arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_longdouble(%struct.agg_longdouble* noalias sret(%struct.agg_longdouble) align 8 %{{.*}}, %struct.agg_longdouble* %{{.*}})

struct agg_float_a8 { float a __attribute__((aligned (8))); };
struct agg_float_a8 pass_agg_float_a8(struct agg_float_a8 arg) { return arg; }
// HARD-FLOAT-LABEL: define{{.*}} void @pass_agg_float_a8(%struct.agg_float_a8* noalias sret(%struct.agg_float_a8) align 8 %{{.*}}, double %{{.*}})
// SOFT-FLOAT-LABEL: define{{.*}} void @pass_agg_float_a8(%struct.agg_float_a8* noalias sret(%struct.agg_float_a8) align 8 %{{.*}}, i64 %{{.*}})

struct agg_float_a16 { float a __attribute__((aligned (16))); };
struct agg_float_a16 pass_agg_float_a16(struct agg_float_a16 arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_float_a16(%struct.agg_float_a16* noalias sret(%struct.agg_float_a16) align 16 %{{.*}}, %struct.agg_float_a16* %{{.*}})


// Verify that the following are *not* float-like aggregate types

struct agg_nofloat1 { float a; float b; };
struct agg_nofloat1 pass_agg_nofloat1(struct agg_nofloat1 arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_nofloat1(%struct.agg_nofloat1* noalias sret(%struct.agg_nofloat1) align 4 %{{.*}}, i64 %{{.*}})

struct agg_nofloat2 { float a; int b; };
struct agg_nofloat2 pass_agg_nofloat2(struct agg_nofloat2 arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_nofloat2(%struct.agg_nofloat2* noalias sret(%struct.agg_nofloat2) align 4 %{{.*}}, i64 %{{.*}})

struct agg_nofloat3 { float a; int : 0; };
struct agg_nofloat3 pass_agg_nofloat3(struct agg_nofloat3 arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_agg_nofloat3(%struct.agg_nofloat3* noalias sret(%struct.agg_nofloat3) align 4 %{{.*}}, i32 %{{.*}})


// Union types likewise are *not* float-like aggregate types

union union_float { float a; };
union union_float pass_union_float(union union_float arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_union_float(%union.union_float* noalias sret(%union.union_float) align 4 %{{.*}}, i32 %{{.*}})

union union_double { double a; };
union union_double pass_union_double(union union_double arg) { return arg; }
// CHECK-LABEL: define{{.*}} void @pass_union_double(%union.union_double* noalias sret(%union.union_double) align 8 %{{.*}}, i64 %{{.*}})


// Accessing variable argument lists

int va_int(__builtin_va_list l) { return __builtin_va_arg(l, int); }
// CHECK-LABEL: define{{.*}} signext i32 @va_int(%struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 20
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to i32*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 4
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to i32*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi i32* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[RET:%[^ ]+]] = load i32, i32* [[VA_ARG_ADDR]]
// CHECK: ret i32 [[RET]]

long va_long(__builtin_va_list l) { return __builtin_va_arg(l, long); }
// CHECK-LABEL: define{{.*}} i64 @va_long(%struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to i64*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to i64*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi i64* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[RET:%[^ ]+]] = load i64, i64* [[VA_ARG_ADDR]]
// CHECK: ret i64 [[RET]]

long long va_longlong(__builtin_va_list l) { return __builtin_va_arg(l, long long); }
// CHECK-LABEL: define{{.*}} i64 @va_longlong(%struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to i64*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to i64*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi i64* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[RET:%[^ ]+]] = load i64, i64* [[VA_ARG_ADDR]]
// CHECK: ret i64 [[RET]]

double va_double(__builtin_va_list l) { return __builtin_va_arg(l, double); }
// CHECK-LABEL: define{{.*}} double @va_double(%struct.__va_list_tag* %{{.*}})
// HARD-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 1
// SOFT-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// HARD-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 4
// SOFT-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// HARD-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 128
// SOFT-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to double*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to double*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi double* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[RET:%[^ ]+]] = load double, double* [[VA_ARG_ADDR]]
// CHECK: ret double [[RET]]

long double va_longdouble(__builtin_va_list l) { return __builtin_va_arg(l, long double); }
// CHECK-LABEL: define{{.*}} void @va_longdouble(fp128* noalias sret(fp128) align 8 %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to fp128**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to fp128**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi fp128** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load fp128*, fp128** [[VA_ARG_ADDR]]
// CHECK: [[RET:%[^ ]+]] = load fp128, fp128* [[INDIRECT_ARG]]
// CHECK: store fp128 [[RET]], fp128* %{{.*}}
// CHECK: ret void

_Complex char va_complex_char(__builtin_va_list l) { return __builtin_va_arg(l, _Complex char); }
// CHECK-LABEL: define{{.*}} void @va_complex_char({ i8, i8 }* noalias sret({ i8, i8 }) align 1 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to { i8, i8 }**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to { i8, i8 }**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi { i8, i8 }** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load { i8, i8 }*, { i8, i8 }** [[VA_ARG_ADDR]]
// CHECK: ret void

struct agg_1byte va_agg_1byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_1byte); }
// CHECK-LABEL: define{{.*}} void @va_agg_1byte(%struct.agg_1byte* noalias sret(%struct.agg_1byte) align 1 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 23
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_1byte*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 7
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_1byte*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_1byte* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_2byte va_agg_2byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_2byte); }
// CHECK-LABEL: define{{.*}} void @va_agg_2byte(%struct.agg_2byte* noalias sret(%struct.agg_2byte) align 1 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 22
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_2byte*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 6
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_2byte*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_2byte* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_3byte va_agg_3byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_3byte); }
// CHECK-LABEL: define{{.*}} void @va_agg_3byte(%struct.agg_3byte* noalias sret(%struct.agg_3byte) align 1 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_3byte**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_3byte**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_3byte** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load %struct.agg_3byte*, %struct.agg_3byte** [[VA_ARG_ADDR]]
// CHECK: ret void

struct agg_4byte va_agg_4byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_4byte); }
// CHECK-LABEL: define{{.*}} void @va_agg_4byte(%struct.agg_4byte* noalias sret(%struct.agg_4byte) align 1 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 20
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_4byte*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 4
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_4byte*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_4byte* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_8byte va_agg_8byte(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_8byte); }
// CHECK-LABEL: define{{.*}} void @va_agg_8byte(%struct.agg_8byte* noalias sret(%struct.agg_8byte) align 1 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_8byte*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_8byte*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_8byte* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_float va_agg_float(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_float); }
// CHECK-LABEL: define{{.*}} void @va_agg_float(%struct.agg_float* noalias sret(%struct.agg_float) align 4 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// HARD-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 1
// SOFT-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// HARD-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 4
// SOFT-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// HARD-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 128
// SOFT-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 20
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_float*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 4
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_float*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_float* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_double va_agg_double(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_double); }
// CHECK-LABEL: define{{.*}} void @va_agg_double(%struct.agg_double* noalias sret(%struct.agg_double) align 8 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// HARD-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 1
// SOFT-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// HARD-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 4
// SOFT-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// HARD-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 128
// SOFT-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_double*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_double*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_double* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_longdouble va_agg_longdouble(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_longdouble); }
// CHECK-LABEL: define{{.*}} void @va_agg_longdouble(%struct.agg_longdouble* noalias sret(%struct.agg_longdouble) align 8 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_longdouble**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_longdouble**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_longdouble** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load %struct.agg_longdouble*, %struct.agg_longdouble** [[VA_ARG_ADDR]]
// CHECK: ret void

struct agg_float_a8 va_agg_float_a8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_float_a8); }
// CHECK-LABEL: define{{.*}} void @va_agg_float_a8(%struct.agg_float_a8* noalias sret(%struct.agg_float_a8) align 8 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// HARD-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 1
// SOFT-FLOAT: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// HARD-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 4
// SOFT-FLOAT: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// HARD-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 128
// SOFT-FLOAT: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_float_a8*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_float_a8*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_float_a8* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_float_a16 va_agg_float_a16(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_float_a16); }
// CHECK-LABEL: define{{.*}} void @va_agg_float_a16(%struct.agg_float_a16* noalias sret(%struct.agg_float_a16) align 16 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_float_a16**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_float_a16**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_float_a16** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load %struct.agg_float_a16*, %struct.agg_float_a16** [[VA_ARG_ADDR]]
// CHECK: ret void

struct agg_nofloat1 va_agg_nofloat1(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_nofloat1); }
// CHECK-LABEL: define{{.*}} void @va_agg_nofloat1(%struct.agg_nofloat1* noalias sret(%struct.agg_nofloat1) align 4 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_nofloat1*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_nofloat1*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_nofloat1* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_nofloat2 va_agg_nofloat2(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_nofloat2); }
// CHECK-LABEL: define{{.*}} void @va_agg_nofloat2(%struct.agg_nofloat2* noalias sret(%struct.agg_nofloat2) align 4 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_nofloat2*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_nofloat2*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_nofloat2* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

struct agg_nofloat3 va_agg_nofloat3(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_nofloat3); }
// CHECK-LABEL: define{{.*}} void @va_agg_nofloat3(%struct.agg_nofloat3* noalias sret(%struct.agg_nofloat3) align 4 %{{.*}}, %struct.__va_list_tag* %{{.*}}
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 20
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_nofloat3*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 4
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_nofloat3*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_nofloat3* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void

