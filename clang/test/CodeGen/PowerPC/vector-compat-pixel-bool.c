// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=mixed -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=gcc -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=xl -triple powerpc-unknown-unknown -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: %clang -mcpu=pwr8 -faltivec-src-compat=gcc --target=powerpc-unknown-unknown -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang -mcpu=pwr9 -faltivec-src-compat=gcc --target=powerpc-unknown-unknown -S -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: @bi8(
// CHECK:         [[A_ADDR:%.*]] = alloca <16 x i8>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <16 x i8>, align 16
// CHECK-NEXT:    store <16 x i8> [[A:%.*]], ptr [[A_ADDR]], align 16
// CHECK-NEXT:    store <16 x i8> [[B:%.*]], ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <16 x i8>, ptr [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <16 x i8>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq <16 x i8> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT:%.*]] = sext <16 x i1> [[CMP]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[SEXT]]
//
// ERROR: returning 'int' from a function with incompatible result type
vector unsigned char bi8(vector bool char a, vector bool char b) {
  return a == b;
}

// CHECK-LABEL: @bi16(
// CHECK:         [[A_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    store <8 x i16> [[A:%.*]], ptr [[A_ADDR]], align 16
// CHECK-NEXT:    store <8 x i16> [[B:%.*]], ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i16>, ptr [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i16>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq <8 x i16> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT:%.*]] = sext <8 x i1> [[CMP]] to <8 x i16>
// CHECK-NEXT:    ret <8 x i16> [[SEXT]]
//
// ERROR: returning 'int' from a function with incompatible result type
vector bool short bi16(vector bool short a, vector bool short b) {
  return a == b;
}

// CHECK-LABEL: @bi32(
// CHECK:         [[A_ADDR:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <4 x i32>, align 16
// CHECK-NEXT:    store <4 x i32> [[A:%.*]], ptr [[A_ADDR]], align 16
// CHECK-NEXT:    store <4 x i32> [[B:%.*]], ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x i32>, ptr [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq <4 x i32> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT:%.*]] = sext <4 x i1> [[CMP]] to <4 x i32>
// CHECK-NEXT:    ret <4 x i32> [[SEXT]]
//
// ERROR: returning 'int' from a function with incompatible result type
vector bool int bi32(vector bool int a, vector bool int b) {
  return a == b;
}

// CHECK-LABEL: @bi64(
// CHECK:         [[A_ADDR:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <2 x i64>, align 16
// CHECK-NEXT:    store <2 x i64> [[A:%.*]], ptr [[A_ADDR]], align 16
// CHECK-NEXT:    store <2 x i64> [[B:%.*]], ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x i64>, ptr [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <2 x i64>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq <2 x i64> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT:%.*]] = sext <2 x i1> [[CMP]] to <2 x i64>
// CHECK-NEXT:    ret <2 x i64> [[SEXT]]
//
// ERROR: returning 'int' from a function with incompatible result type
vector long long bi64(vector bool long long a, vector bool long long b) {
  return a == b;
}

// CHECK-LABEL: @VecPixel(
// CHECK:         [[A_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    [[B_ADDR:%.*]] = alloca <8 x i16>, align 16
// CHECK-NEXT:    store <8 x i16> [[A:%.*]], ptr [[A_ADDR]], align 16
// CHECK-NEXT:    store <8 x i16> [[B:%.*]], ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[TMP0:%.*]] = load <8 x i16>, ptr [[A_ADDR]], align 16
// CHECK-NEXT:    [[TMP1:%.*]] = load <8 x i16>, ptr [[B_ADDR]], align 16
// CHECK-NEXT:    [[CMP:%.*]] = icmp eq <8 x i16> [[TMP0]], [[TMP1]]
// CHECK-NEXT:    [[SEXT:%.*]] = sext <8 x i1> [[CMP]] to <8 x i16>
// CHECK-NEXT:    ret <8 x i16> [[SEXT]]
//
// ERROR: returning 'int' from a function with incompatible result type
vector pixel VecPixel(vector pixel a, vector pixel b) {
  return a == b;
}
