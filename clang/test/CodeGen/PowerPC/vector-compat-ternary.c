// RUN: %clang_cc1 -no-opaque-pointers -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=mixed -triple powerpc-unknown-unknown -S -emit-llvm %s -o - | FileCheck %s
// RUN: not %clang_cc1 -no-opaque-pointers -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=gcc -triple powerpc-unknown-unknown -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: %clang_cc1 -no-opaque-pointers -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=xl -triple powerpc-unknown-unknown -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -Xclang -no-opaque-pointers -mcpu=pwr8 -faltivec-src-compat=xl --target=powerpc-unknown-unknown -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -Xclang -no-opaque-pointers -mcpu=pwr9 -faltivec-src-compat=xl --target=powerpc-unknown-unknown -S -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: @ui8(
// CHECK:         [[A_ADDR:%.*]] = alloca <16 x i8>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <16 x i8>, align 16
// CHECK-NEXT:    store <16 x i8> [[A:%.*]], <16 x i8>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <16 x i8> [[B:%.*]], <16 x i8>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <16 x i8>, <16 x i8>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpequb.p(i32 2, <16 x i8> [[TMP0]], <16 x i8> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(16 * sizeof(char)))) char' (vector of 16 'char' values) where arithmetic or pointer type is required
int ui8(vector unsigned char a, vector unsigned char b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @si8(
// CHECK:         [[A_ADDR:%.*]] = alloca <16 x i8>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <16 x i8>, align 16
// CHECK-NEXT:    store <16 x i8> [[A:%.*]], <16 x i8>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <16 x i8> [[B:%.*]], <16 x i8>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <16 x i8>, <16 x i8>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpequb.p(i32 2, <16 x i8> [[TMP0]], <16 x i8> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(16 * sizeof(char)))) char' (vector of 16 'char' values) where arithmetic or pointer type is required
int si8(vector signed char a, vector signed char b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @ui16(
// CHECK:         [[A_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    store <8 x i16> [[A:%.*]], <8 x i16>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <8 x i16> [[B:%.*]], <8 x i16>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i16>, <8 x i16>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i16>, <8 x i16>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpequh.p(i32 2, <8 x i16> [[TMP0]], <8 x i16> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(8 * sizeof(short)))) short' (vector of 8 'short' values) where arithmetic or pointer type is required
int ui16(vector unsigned short a, vector unsigned short b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @si16(
// CHECK:         [[A_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    store <8 x i16> [[A:%.*]], <8 x i16>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <8 x i16> [[B:%.*]], <8 x i16>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i16>, <8 x i16>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i16>, <8 x i16>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpequh.p(i32 2, <8 x i16> [[TMP0]], <8 x i16> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(8 * sizeof(short)))) short' (vector of 8 'short' values) where arithmetic or pointer type is required
int si16(vector signed short a, vector signed short b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @ui32(
// CHECK:         [[A_ADDR:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    store <4 x i32> [[A:%.*]], <4 x i32>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <4 x i32> [[B:%.*]], <4 x i32>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x i32>, <4 x i32>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, <4 x i32>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpequw.p(i32 2, <4 x i32> [[TMP0]], <4 x i32> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(4 * sizeof(long)))) long' (vector of 4 'long' values) where arithmetic or pointer type is required
int ui32(vector unsigned int a, vector unsigned int b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @si32(
// CHECK:         [[A_ADDR:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    store <4 x i32> [[A:%.*]], <4 x i32>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <4 x i32> [[B:%.*]], <4 x i32>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x i32>, <4 x i32>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, <4 x i32>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpequw.p(i32 2, <4 x i32> [[TMP0]], <4 x i32> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(4 * sizeof(long)))) long' (vector of 4 'long' values) where arithmetic or pointer type is required
int si32(vector signed int a, vector signed int b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @si64(
// CHECK:         [[A_ADDR:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    store <2 x i64> [[A:%.*]], <2 x i64>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <2 x i64> [[B:%.*]], <2 x i64>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i64>, <2 x i64>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i64>, <2 x i64>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpequd.p(i32 2, <2 x i64> [[TMP0]], <2 x i64> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(2 * sizeof(long long)))) long long' (vector of 2 'long long' values) where arithmetic or pointer type is required
int si64(vector long long a, vector long long b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @f32(
// CHECK:         [[A_ADDR:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <4 x float>, align 16
// CHECK-NEXT:    store <4 x float> [[A:%.*]], <4 x float>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <4 x float> [[B:%.*]], <4 x float>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x float>, <4 x float>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x float>, <4 x float>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.altivec.vcmpeqfp.p(i32 2, <4 x float> [[TMP0]], <4 x float> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(4 * sizeof(long)))) long' (vector of 4 'long' values) where arithmetic or pointer type is required
int f32(vector float a, vector float b) {
  return a == b ? 3 : 7;
}

// CHECK-LABEL: @f64(
// CHECK:         [[A_ADDR:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <2 x double>, align 16
// CHECK-NEXT:    store <2 x double> [[A:%.*]], <2 x double>* [[A_ADDR]], align 16
// CHECK-NEXT:    store <2 x double> [[B:%.*]], <2 x double>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x double>, <2 x double>* [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x double>, <2 x double>* [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP2:%.*]] = call i32 @llvm.ppc.vsx.xvcmpeqdp.p(i32 2, <2 x double> [[TMP0]], <2 x double> [[TMP1]])
// CHECK-NEXT:    [[TOBOOL:%.*]] = icmp ne i32 [[TMP2]], 0
// CHECK-NEXT:    [[TMP3:%.*]] = zext i1 [[TOBOOL]] to i64
// CHECK-NEXT:    [[COND:%.*]] = select i1 [[TOBOOL]], i32 3, i32 7
// CHECK-NEXT:    ret i32 [[COND]]
//
// ERROR: error: used type '__attribute__((__vector_size__(2 * sizeof(long long)))) long long' (vector of 2 'long long' values) where arithmetic or pointer type is required
int f64(vector double a, vector double b) {
  return a == b ? 3 : 7;
}
