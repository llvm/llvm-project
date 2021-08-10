// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpcle-unknown-unknown \
// RUN:   -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:    -emit-llvm %s -o -  -target-cpu pwr8 | FileCheck %s

// All of these cipher builtins are only for Power 8 and up.

// CHECK-LABEL: @testvcipher(
// CHECK:         [[TMP4:%.*]] = call <2 x i64> @llvm.ppc.altivec.crypto.vcipher
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP5]]
//
vector unsigned char testvcipher(vector unsigned char state_array, vector unsigned char round_key) {
  return __vcipher(state_array, round_key);
}

// CHECK-LABEL: @testvcipherlast(
// CHECK:         [[TMP4:%.*]] = call <2 x i64> @llvm.ppc.altivec.crypto.vcipherlast
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP5]]
//
vector unsigned char testvcipherlast(vector unsigned char state_array, vector unsigned char round_key) {
  return __vcipherlast(state_array, round_key);
}

// CHECK-LABEL: @testvncipher(
// CHECK:         [[TMP4:%.*]] = call <2 x i64> @llvm.ppc.altivec.crypto.vncipher
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP5]]
//
vector unsigned char testvncipher(vector unsigned char state_array, vector unsigned char round_key) {
  return __vncipher(state_array, round_key);
}

// CHECK-LABEL: @testvncipherlast(
// CHECK:         [[TMP4:%.*]] = call <2 x i64> @llvm.ppc.altivec.crypto.vncipherlast
// CHECK-NEXT:    [[TMP5:%.*]] = bitcast <2 x i64> [[TMP4]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP5]]
//
vector unsigned char testvncipherlast(vector unsigned char state_array, vector unsigned char round_key) {
  return __vncipherlast(state_array, round_key);
}

// CHECK-LABEL: @testvpermxor(
// CHECK:         [[TMP3:%.*]] = call <16 x i8> @llvm.ppc.altivec.crypto.vpermxor
// CHECK-NEXT:    ret <16 x i8> [[TMP3]]
//
vector unsigned char testvpermxor(vector unsigned char a, vector unsigned char b, vector unsigned char mask) {
  return __vpermxor(a, b, mask);
}

// CHECK-LABEL: @testvpmsumb(
// CHECK:         [[TMP2:%.*]] = call <16 x i8> @llvm.ppc.altivec.crypto.vpmsumb
// CHECK-NEXT:    ret <16 x i8> [[TMP2]]
//
vector unsigned char testvpmsumb(vector unsigned char a, vector unsigned char b) {
  return __vpmsumb(a, b);
}

// CHECK-LABEL: @testvpmsumd(
// CHECK:         [[TMP2:%.*]] = call <2 x i64> @llvm.ppc.altivec.crypto.vpmsumd
// CHECK-NEXT:    ret <2 x i64> [[TMP2]]
//
vector unsigned long long testvpmsumd(vector unsigned long long a, vector unsigned long long b) {
  return __vpmsumd(a, b);
}

// CHECK-LABEL: @testvpmsumh(
// CHECK:         [[TMP2:%.*]] = call <8 x i16> @llvm.ppc.altivec.crypto.vpmsumh
// CHECK-NEXT:    ret <8 x i16> [[TMP2]]
//
vector unsigned short testvpmsumh(vector unsigned short a, vector unsigned short b) {
  return __vpmsumh(a, b);
}

// CHECK-LABEL: @testvpmsumw(
// CHECK:         [[TMP2:%.*]] = call <4 x i32> @llvm.ppc.altivec.crypto.vpmsumw
// CHECK-NEXT:    ret <4 x i32> [[TMP2]]
//
vector unsigned int testvpmsumw(vector unsigned int a, vector unsigned int b) {
  return __vpmsumw(a, b);
}
