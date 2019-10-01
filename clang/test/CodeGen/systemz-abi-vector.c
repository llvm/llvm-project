// RUN: %clang_cc1 -triple s390x-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-feature +vector \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-VECTOR %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z13 \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-VECTOR %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu arch11 \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-VECTOR %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z14 \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-VECTOR %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu arch12 \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-VECTOR %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu z15 \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-VECTOR %s
// RUN: %clang_cc1 -triple s390x-linux-gnu -target-cpu arch13 \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-VECTOR %s

// Vector types

typedef __attribute__((vector_size(1))) char v1i8;

typedef __attribute__((vector_size(2))) char v2i8;
typedef __attribute__((vector_size(2))) short v1i16;

typedef __attribute__((vector_size(4))) char v4i8;
typedef __attribute__((vector_size(4))) short v2i16;
typedef __attribute__((vector_size(4))) int v1i32;
typedef __attribute__((vector_size(4))) float v1f32;

typedef __attribute__((vector_size(8))) char v8i8;
typedef __attribute__((vector_size(8))) short v4i16;
typedef __attribute__((vector_size(8))) int v2i32;
typedef __attribute__((vector_size(8))) long long v1i64;
typedef __attribute__((vector_size(8))) float v2f32;
typedef __attribute__((vector_size(8))) double v1f64;

typedef __attribute__((vector_size(16))) char v16i8;
typedef __attribute__((vector_size(16))) short v8i16;
typedef __attribute__((vector_size(16))) int v4i32;
typedef __attribute__((vector_size(16))) long long v2i64;
typedef __attribute__((vector_size(16))) __int128 v1i128;
typedef __attribute__((vector_size(16))) float v4f32;
typedef __attribute__((vector_size(16))) double v2f64;
typedef __attribute__((vector_size(16))) long double v1f128;

typedef __attribute__((vector_size(32))) char v32i8;

unsigned int align = __alignof__ (v16i8);
// CHECK: @align = global i32 16
// CHECK-VECTOR: @align = global i32 8

v1i8 pass_v1i8(v1i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i8(<1 x i8>* noalias sret %{{.*}}, <1 x i8>* %0)
// CHECK-VECTOR-LABEL: define <1 x i8> @pass_v1i8(<1 x i8> %{{.*}})

v2i8 pass_v2i8(v2i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i8(<2 x i8>* noalias sret %{{.*}}, <2 x i8>* %0)
// CHECK-VECTOR-LABEL: define <2 x i8> @pass_v2i8(<2 x i8> %{{.*}})

v4i8 pass_v4i8(v4i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4i8(<4 x i8>* noalias sret %{{.*}}, <4 x i8>* %0)
// CHECK-VECTOR-LABEL: define <4 x i8> @pass_v4i8(<4 x i8> %{{.*}})

v8i8 pass_v8i8(v8i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v8i8(<8 x i8>* noalias sret %{{.*}}, <8 x i8>* %0)
// CHECK-VECTOR-LABEL: define <8 x i8> @pass_v8i8(<8 x i8> %{{.*}})

v16i8 pass_v16i8(v16i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v16i8(<16 x i8>* noalias sret %{{.*}}, <16 x i8>* %0)
// CHECK-VECTOR-LABEL: define <16 x i8> @pass_v16i8(<16 x i8> %{{.*}})

v32i8 pass_v32i8(v32i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_v32i8(<32 x i8>* noalias sret %{{.*}}, <32 x i8>* %0)
// CHECK-VECTOR-LABEL: define void @pass_v32i8(<32 x i8>* noalias sret %{{.*}}, <32 x i8>* %0)

v1i16 pass_v1i16(v1i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i16(<1 x i16>* noalias sret %{{.*}}, <1 x i16>* %0)
// CHECK-VECTOR-LABEL: define <1 x i16> @pass_v1i16(<1 x i16> %{{.*}})

v2i16 pass_v2i16(v2i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i16(<2 x i16>* noalias sret %{{.*}}, <2 x i16>* %0)
// CHECK-VECTOR-LABEL: define <2 x i16> @pass_v2i16(<2 x i16> %{{.*}})

v4i16 pass_v4i16(v4i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4i16(<4 x i16>* noalias sret %{{.*}}, <4 x i16>* %0)
// CHECK-VECTOR-LABEL: define <4 x i16> @pass_v4i16(<4 x i16> %{{.*}})

v8i16 pass_v8i16(v8i16 arg) { return arg; }
// CHECK-LABEL: define void @pass_v8i16(<8 x i16>* noalias sret %{{.*}}, <8 x i16>* %0)
// CHECK-VECTOR-LABEL: define <8 x i16> @pass_v8i16(<8 x i16> %{{.*}})

v1i32 pass_v1i32(v1i32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i32(<1 x i32>* noalias sret %{{.*}}, <1 x i32>* %0)
// CHECK-VECTOR-LABEL: define <1 x i32> @pass_v1i32(<1 x i32> %{{.*}})

v2i32 pass_v2i32(v2i32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i32(<2 x i32>* noalias sret %{{.*}}, <2 x i32>* %0)
// CHECK-VECTOR-LABEL: define <2 x i32> @pass_v2i32(<2 x i32> %{{.*}})

v4i32 pass_v4i32(v4i32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4i32(<4 x i32>* noalias sret %{{.*}}, <4 x i32>* %0)
// CHECK-VECTOR-LABEL: define <4 x i32> @pass_v4i32(<4 x i32> %{{.*}})

v1i64 pass_v1i64(v1i64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i64(<1 x i64>* noalias sret %{{.*}}, <1 x i64>* %0)
// CHECK-VECTOR-LABEL: define <1 x i64> @pass_v1i64(<1 x i64> %{{.*}})

v2i64 pass_v2i64(v2i64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2i64(<2 x i64>* noalias sret %{{.*}}, <2 x i64>* %0)
// CHECK-VECTOR-LABEL: define <2 x i64> @pass_v2i64(<2 x i64> %{{.*}})

v1i128 pass_v1i128(v1i128 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1i128(<1 x i128>* noalias sret %{{.*}}, <1 x i128>* %0)
// CHECK-VECTOR-LABEL: define <1 x i128> @pass_v1i128(<1 x i128> %{{.*}})

v1f32 pass_v1f32(v1f32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1f32(<1 x float>* noalias sret %{{.*}}, <1 x float>* %0)
// CHECK-VECTOR-LABEL: define <1 x float> @pass_v1f32(<1 x float> %{{.*}})

v2f32 pass_v2f32(v2f32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2f32(<2 x float>* noalias sret %{{.*}}, <2 x float>* %0)
// CHECK-VECTOR-LABEL: define <2 x float> @pass_v2f32(<2 x float> %{{.*}})

v4f32 pass_v4f32(v4f32 arg) { return arg; }
// CHECK-LABEL: define void @pass_v4f32(<4 x float>* noalias sret %{{.*}}, <4 x float>* %0)
// CHECK-VECTOR-LABEL: define <4 x float> @pass_v4f32(<4 x float> %{{.*}})

v1f64 pass_v1f64(v1f64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1f64(<1 x double>* noalias sret %{{.*}}, <1 x double>* %0)
// CHECK-VECTOR-LABEL: define <1 x double> @pass_v1f64(<1 x double> %{{.*}})

v2f64 pass_v2f64(v2f64 arg) { return arg; }
// CHECK-LABEL: define void @pass_v2f64(<2 x double>* noalias sret %{{.*}}, <2 x double>* %0)
// CHECK-VECTOR-LABEL: define <2 x double> @pass_v2f64(<2 x double> %{{.*}})

v1f128 pass_v1f128(v1f128 arg) { return arg; }
// CHECK-LABEL: define void @pass_v1f128(<1 x fp128>* noalias sret %{{.*}}, <1 x fp128>* %0)
// CHECK-VECTOR-LABEL: define <1 x fp128> @pass_v1f128(<1 x fp128> %{{.*}})


// Vector-like aggregate types

struct agg_v1i8 { v1i8 a; };
struct agg_v1i8 pass_agg_v1i8(struct agg_v1i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_v1i8(%struct.agg_v1i8* noalias sret %{{.*}}, i8 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_v1i8(%struct.agg_v1i8* noalias sret %{{.*}}, <1 x i8> %{{.*}})

struct agg_v2i8 { v2i8 a; };
struct agg_v2i8 pass_agg_v2i8(struct agg_v2i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_v2i8(%struct.agg_v2i8* noalias sret %{{.*}}, i16 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_v2i8(%struct.agg_v2i8* noalias sret %{{.*}}, <2 x i8> %{{.*}})

struct agg_v4i8 { v4i8 a; };
struct agg_v4i8 pass_agg_v4i8(struct agg_v4i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_v4i8(%struct.agg_v4i8* noalias sret %{{.*}}, i32 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_v4i8(%struct.agg_v4i8* noalias sret %{{.*}}, <4 x i8> %{{.*}})

struct agg_v8i8 { v8i8 a; };
struct agg_v8i8 pass_agg_v8i8(struct agg_v8i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_v8i8(%struct.agg_v8i8* noalias sret %{{.*}}, i64 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_v8i8(%struct.agg_v8i8* noalias sret %{{.*}}, <8 x i8> %{{.*}})

struct agg_v16i8 { v16i8 a; };
struct agg_v16i8 pass_agg_v16i8(struct agg_v16i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_v16i8(%struct.agg_v16i8* noalias sret %{{.*}}, %struct.agg_v16i8* %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_v16i8(%struct.agg_v16i8* noalias sret %{{.*}}, <16 x i8> %{{.*}})

struct agg_v32i8 { v32i8 a; };
struct agg_v32i8 pass_agg_v32i8(struct agg_v32i8 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_v32i8(%struct.agg_v32i8* noalias sret %{{.*}}, %struct.agg_v32i8* %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_v32i8(%struct.agg_v32i8* noalias sret %{{.*}}, %struct.agg_v32i8* %{{.*}})


// Verify that the following are *not* vector-like aggregate types

struct agg_novector1 { v4i8 a; v4i8 b; };
struct agg_novector1 pass_agg_novector1(struct agg_novector1 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_novector1(%struct.agg_novector1* noalias sret %{{.*}}, i64 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_novector1(%struct.agg_novector1* noalias sret %{{.*}}, i64 %{{.*}})

struct agg_novector2 { v4i8 a; float b; };
struct agg_novector2 pass_agg_novector2(struct agg_novector2 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_novector2(%struct.agg_novector2* noalias sret %{{.*}}, i64 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_novector2(%struct.agg_novector2* noalias sret %{{.*}}, i64 %{{.*}})

struct agg_novector3 { v4i8 a; int : 0; };
struct agg_novector3 pass_agg_novector3(struct agg_novector3 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_novector3(%struct.agg_novector3* noalias sret %{{.*}}, i32 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_novector3(%struct.agg_novector3* noalias sret %{{.*}}, i32 %{{.*}})

struct agg_novector4 { v4i8 a __attribute__((aligned (8))); };
struct agg_novector4 pass_agg_novector4(struct agg_novector4 arg) { return arg; }
// CHECK-LABEL: define void @pass_agg_novector4(%struct.agg_novector4* noalias sret %{{.*}}, i64 %{{.*}})
// CHECK-VECTOR-LABEL: define void @pass_agg_novector4(%struct.agg_novector4* noalias sret %{{.*}}, i64 %{{.*}})


// Accessing variable argument lists

v1i8 va_v1i8(__builtin_va_list l) { return __builtin_va_arg(l, v1i8); }
// CHECK-LABEL: define void @va_v1i8(<1 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to <1 x i8>**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to <1 x i8>**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi <1 x i8>** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load <1 x i8>*, <1 x i8>** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define <1 x i8> @va_v1i8(%struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to <1 x i8>*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[RET:%[^ ]+]] = load <1 x i8>, <1 x i8>* [[MEM_ADDR]]
// CHECK-VECTOR: ret <1 x i8> [[RET]]

v2i8 va_v2i8(__builtin_va_list l) { return __builtin_va_arg(l, v2i8); }
// CHECK-LABEL: define void @va_v2i8(<2 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to <2 x i8>**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to <2 x i8>**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi <2 x i8>** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load <2 x i8>*, <2 x i8>** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define <2 x i8> @va_v2i8(%struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to <2 x i8>*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[RET:%[^ ]+]] = load <2 x i8>, <2 x i8>* [[MEM_ADDR]]
// CHECK-VECTOR: ret <2 x i8> [[RET]]

v4i8 va_v4i8(__builtin_va_list l) { return __builtin_va_arg(l, v4i8); }
// CHECK-LABEL: define void @va_v4i8(<4 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to <4 x i8>**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to <4 x i8>**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi <4 x i8>** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load <4 x i8>*, <4 x i8>** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define <4 x i8> @va_v4i8(%struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to <4 x i8>*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[RET:%[^ ]+]] = load <4 x i8>, <4 x i8>* [[MEM_ADDR]]
// CHECK-VECTOR: ret <4 x i8> [[RET]]

v8i8 va_v8i8(__builtin_va_list l) { return __builtin_va_arg(l, v8i8); }
// CHECK-LABEL: define void @va_v8i8(<8 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to <8 x i8>**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to <8 x i8>**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi <8 x i8>** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load <8 x i8>*, <8 x i8>** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define <8 x i8> @va_v8i8(%struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to <8 x i8>*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[RET:%[^ ]+]] = load <8 x i8>, <8 x i8>* [[MEM_ADDR]]
// CHECK-VECTOR: ret <8 x i8> [[RET]]

v16i8 va_v16i8(__builtin_va_list l) { return __builtin_va_arg(l, v16i8); }
// CHECK-LABEL: define void @va_v16i8(<16 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to <16 x i8>**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to <16 x i8>**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi <16 x i8>** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load <16 x i8>*, <16 x i8>** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define <16 x i8> @va_v16i8(%struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to <16 x i8>*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 16
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[RET:%[^ ]+]] = load <16 x i8>, <16 x i8>* [[MEM_ADDR]]
// CHECK-VECTOR: ret <16 x i8> [[RET]]

v32i8 va_v32i8(__builtin_va_list l) { return __builtin_va_arg(l, v32i8); }
// CHECK-LABEL: define void @va_v32i8(<32 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to <32 x i8>**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to <32 x i8>**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi <32 x i8>** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load <32 x i8>*, <32 x i8>** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define void @va_v32i8(<32 x i8>* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK-VECTOR: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK-VECTOR: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK-VECTOR: br i1 [[FITS_IN_REGS]],
// CHECK-VECTOR: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK-VECTOR: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK-VECTOR: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK-VECTOR: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK-VECTOR: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK-VECTOR: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to <32 x i8>**
// CHECK-VECTOR: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK-VECTOR: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to <32 x i8>**
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[VA_ARG_ADDR:%[^ ]+]] = phi <32 x i8>** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK-VECTOR: [[INDIRECT_ARG:%[^ ]+]] = load <32 x i8>*, <32 x i8>** [[VA_ARG_ADDR]]
// CHECK-VECTOR: ret void

struct agg_v1i8 va_agg_v1i8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_v1i8); }
// CHECK-LABEL: define void @va_agg_v1i8(%struct.agg_v1i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 23
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_v1i8*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 7
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_v1i8*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_v1i8* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define void @va_agg_v1i8(%struct.agg_v1i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to %struct.agg_v1i8*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: ret void

struct agg_v2i8 va_agg_v2i8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_v2i8); }
// CHECK-LABEL: define void @va_agg_v2i8(%struct.agg_v2i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 22
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_v2i8*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 6
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_v2i8*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_v2i8* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define void @va_agg_v2i8(%struct.agg_v2i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to %struct.agg_v2i8*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: ret void

struct agg_v4i8 va_agg_v4i8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_v4i8); }
// CHECK-LABEL: define void @va_agg_v4i8(%struct.agg_v4i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 20
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_v4i8*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 4
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_v4i8*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_v4i8* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define void @va_agg_v4i8(%struct.agg_v4i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to %struct.agg_v4i8*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: ret void

struct agg_v8i8 va_agg_v8i8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_v8i8); }
// CHECK-LABEL: define void @va_agg_v8i8(%struct.agg_v8i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_v8i8*
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_v8i8*
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_v8i8* [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define void @va_agg_v8i8(%struct.agg_v8i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to %struct.agg_v8i8*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: ret void

struct agg_v16i8 va_agg_v16i8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_v16i8); }
// CHECK-LABEL: define void @va_agg_v16i8(%struct.agg_v16i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_v16i8**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_v16i8**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_v16i8** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load %struct.agg_v16i8*, %struct.agg_v16i8** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define void @va_agg_v16i8(%struct.agg_v16i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[OVERFLOW_ARG_AREA]] to %struct.agg_v16i8*
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA1:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 16
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA1]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: ret void

struct agg_v32i8 va_agg_v32i8(__builtin_va_list l) { return __builtin_va_arg(l, struct agg_v32i8); }
// CHECK-LABEL: define void @va_agg_v32i8(%struct.agg_v32i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK: br i1 [[FITS_IN_REGS]],
// CHECK: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_v32i8**
// CHECK: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_v32i8**
// CHECK: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_v32i8** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK: [[INDIRECT_ARG:%[^ ]+]] = load %struct.agg_v32i8*, %struct.agg_v32i8** [[VA_ARG_ADDR]]
// CHECK: ret void
// CHECK-VECTOR-LABEL: define void @va_agg_v32i8(%struct.agg_v32i8* noalias sret %{{.*}}, %struct.__va_list_tag* %{{.*}})
// CHECK-VECTOR: [[REG_COUNT_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 0
// CHECK-VECTOR: [[REG_COUNT:%[^ ]+]] = load i64, i64* [[REG_COUNT_PTR]]
// CHECK-VECTOR: [[FITS_IN_REGS:%[^ ]+]] = icmp ult i64 [[REG_COUNT]], 5
// CHECK-VECTOR: br i1 [[FITS_IN_REGS]],
// CHECK-VECTOR: [[SCALED_REG_COUNT:%[^ ]+]] = mul i64 [[REG_COUNT]], 8
// CHECK-VECTOR: [[REG_OFFSET:%[^ ]+]] = add i64 [[SCALED_REG_COUNT]], 16
// CHECK-VECTOR: [[REG_SAVE_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 3
// CHECK-VECTOR: [[REG_SAVE_AREA:%[^ ]+]] = load i8*, i8** [[REG_SAVE_AREA_PTR:[^ ]+]]
// CHECK-VECTOR: [[RAW_REG_ADDR:%[^ ]+]] = getelementptr i8, i8* [[REG_SAVE_AREA]], i64 [[REG_OFFSET]]
// CHECK-VECTOR: [[REG_ADDR:%[^ ]+]] = bitcast i8* [[RAW_REG_ADDR]] to %struct.agg_v32i8**
// CHECK-VECTOR: [[REG_COUNT1:%[^ ]+]] = add i64 [[REG_COUNT]], 1
// CHECK-VECTOR: store i64 [[REG_COUNT1]], i64* [[REG_COUNT_PTR]]
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA_PTR:%[^ ]+]] = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %{{.*}}, i32 0, i32 2
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA:%[^ ]+]] = load i8*, i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[RAW_MEM_ADDR:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 0
// CHECK-VECTOR: [[MEM_ADDR:%[^ ]+]] = bitcast i8* [[RAW_MEM_ADDR]] to %struct.agg_v32i8**
// CHECK-VECTOR: [[OVERFLOW_ARG_AREA2:%[^ ]+]] = getelementptr i8, i8* [[OVERFLOW_ARG_AREA]], i64 8
// CHECK-VECTOR: store i8* [[OVERFLOW_ARG_AREA2]], i8** [[OVERFLOW_ARG_AREA_PTR]]
// CHECK-VECTOR: [[VA_ARG_ADDR:%[^ ]+]] = phi %struct.agg_v32i8** [ [[REG_ADDR]], %{{.*}} ], [ [[MEM_ADDR]], %{{.*}} ]
// CHECK-VECTOR: [[INDIRECT_ARG:%[^ ]+]] = load %struct.agg_v32i8*, %struct.agg_v32i8** [[VA_ARG_ADDR]]
// CHECK-VECTOR: ret void
