// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -flax-vector-conversions=none -O2 -target-feature +altivec -target-feature +power8-vector \
// RUN: -triple powerpc64-unknown-linux -emit-llvm %s -o - | FileCheck %s
//
// RUN: %clang_cc1 -flax-vector-conversions=none -O2 -target-feature +altivec -target-feature +power8-vector \
// RUN: -triple powerpc64le-unknown-linux -emit-llvm %s -o - | FileCheck %s 

// RUN: %clang_cc1 -flax-vector-conversions=none -O2 -target-feature +altivec -target-feature +power8-vector \
// RUN: -triple powerpc64-unknown-aix -emit-llvm %s -o - | FileCheck %s
#include <altivec.h>

// CHECK-LABEL: @test_vaddeuqm_c(
// CHECK:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[B:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[C:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <1 x i128> @llvm.ppc.altivec.vaddeuqm(<1 x i128> [[TMP0]], <1 x i128> [[TMP1]], <1 x i128> [[TMP2]])
// CHECK-NEXT:    [[TMP4:%.*]] = bitcast <1 x i128> [[TMP3]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP4]]
vector unsigned char test_vaddeuqm_c(vector unsigned char a, vector unsigned char b,
                                vector unsigned char c) {
  return __builtin_altivec_vaddeuqm_c(a, b, c);
}

// CHECK-LABEL: @test_vaddcuq_c(
// CHECK:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[B:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <1 x i128> @llvm.ppc.altivec.vaddcuq(<1 x i128> [[TMP0]], <1 x i128> [[TMP1]])
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <1 x i128> [[TMP2]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP3]]
vector unsigned char test_vaddcuq_c(vector unsigned char a, vector unsigned char b) {
  return __builtin_altivec_vaddcuq_c(a, b);
}

// CHECK-LABEL: @test_vaddecuq_c(
// CHECK:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[B:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[C:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <1 x i128> @llvm.ppc.altivec.vaddecuq(<1 x i128> [[TMP0]], <1 x i128> [[TMP1]], <1 x i128> [[TMP2]])
// CHECK-NEXT:    [[TMP4:%.*]] = bitcast <1 x i128> [[TMP3]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP4]]
vector unsigned char test_vaddecuq_c(vector unsigned char a, vector unsigned char b,
                                vector unsigned char c) {
  return __builtin_altivec_vaddecuq_c(a, b, c);
}

// CHECK-LABEL: @test_vsubeuqm_c(
// CHECK:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[B:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[C:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <1 x i128> @llvm.ppc.altivec.vsubeuqm(<1 x i128> [[TMP0]], <1 x i128> [[TMP1]], <1 x i128> [[TMP2]])
// CHECK-NEXT:    [[TMP4:%.*]] = bitcast <1 x i128> [[TMP3]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP4]]
vector unsigned char test_vsubeuqm_c(vector unsigned char a, vector unsigned char b,
                                vector unsigned char c) {
  return __builtin_altivec_vsubeuqm_c(a, b, c);
}

// CHECK-LABEL: @test_vsubcuq_c(
// CHECK:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[B:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP2:%.*]] = tail call <1 x i128> @llvm.ppc.altivec.vsubcuq(<1 x i128> [[TMP0]], <1 x i128> [[TMP1]])
// CHECK-NEXT:    [[TMP3:%.*]] = bitcast <1 x i128> [[TMP2]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP3]]
vector unsigned char test_vsubcuq_c(vector unsigned char a, vector unsigned char b) {
  return __builtin_altivec_vsubcuq_c(a, b);
}

// CHECK-LABEL: @test_vsubecuq_c(
// CHECK:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP1:%.*]] = bitcast <16 x i8> [[B:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP2:%.*]] = bitcast <16 x i8> [[C:%.*]] to <1 x i128>
// CHECK-NEXT:    [[TMP3:%.*]] = tail call <1 x i128> @llvm.ppc.altivec.vsubecuq(<1 x i128> [[TMP0]], <1 x i128> [[TMP1]], <1 x i128> [[TMP2]])
// CHECK-NEXT:    [[TMP4:%.*]] = bitcast <1 x i128> [[TMP3]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP4]]
vector unsigned char test_vsubecuq_c(vector unsigned char a, vector unsigned char b,
                                vector unsigned char c) {
  return __builtin_altivec_vsubecuq_c(a, b, c);
}
