// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -target powerpc64-gnu-linux -mcpu=pwr8 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-BE
// RUN: %clang -S -emit-llvm -target powerpc64le-gnu-linux -mcpu=pwr8 -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,CHECK-LE

#include <mmintrin.h>

unsigned long long int ull1, ull2;
__m64 m1, m2, res;

void __attribute__((noinline))
test_packs() {
  res = _mm_packs_pu16((__m64)ull1, (__m64)ull2);
  res = _mm_packs_pi16((__m64)ull1, (__m64)ull2);
  res = _mm_packs_pi32((__m64)ull1, (__m64)ull2);
}

// CHECK-LABEL: @test_packs

// CHECK: define available_externally i64 @_mm_packs_pu16(i64 [[REG1:[0-9a-zA-Z_%.]+]], i64 [[REG2:[0-9a-zA-Z_%.]+]])
// CHECK: store i64 [[REG1]], i64* [[REG3:[0-9a-zA-Z_%.]+]], align 8
// CHECK-NEXT: store i64 [[REG2]], i64* [[REG4:[0-9a-zA-Z_%.]+]], align 8
// CHECK-LE: load i64, i64* [[REG3]], align 8
// CHECK: load i64, i64* [[REG4]], align 8
// CHECK-BE: load i64, i64* [[REG3]], align 8
// CHECK: [[REG5:[0-9a-zA-Z_%.]+]] = call <8 x i16> @vec_cmplt
// CHECK-NEXT: store <8 x i16> [[REG5]], <8 x i16>* [[REG6:[0-9a-zA-Z_%.]+]], align 16
// CHECK-NEXT: [[REG7:[0-9a-zA-Z_%.]+]] = load <8 x i16>, <8 x i16>* [[REG8:[0-9a-zA-Z_%.]+]], align 16
// CHECK-NEXT: [[REG9:[0-9a-zA-Z_%.]+]] = load <8 x i16>, <8 x i16>* [[REG8]], align 16
// CHECK-NEXT: [[REG10:[0-9a-zA-Z_%.]+]] = call <16 x i8> @vec_packs(unsigned short vector[8], unsigned short vector[8])(<8 x i16> [[REG7]], <8 x i16> [[REG9]])
// CHECK-NEXT: store <16 x i8> [[REG10]], <16 x i8>* [[REG11:[0-9a-zA-Z_%.]+]], align 16
// CHECK-NEXT: [[REG12:[0-9a-zA-Z_%.]+]] = load <8 x i16>, <8 x i16>* [[REG6]], align 16
// CHECK-NEXT: [[REG13:[0-9a-zA-Z_%.]+]] = load <8 x i16>, <8 x i16>* [[REG6]], align 16
// CHECK-NEXT: [[REG14:[0-9a-zA-Z_%.]+]] = call <16 x i8> @vec_pack(bool vector[8], bool vector[8])(<8 x i16> [[REG12]], <8 x i16> [[REG13]])
// CHECK-NEXT: store <16 x i8> [[REG14]], <16 x i8>* [[REG15:[0-9a-zA-Z_%.]+]], align 16
// CHECK-NEXT: [[REG16:[0-9a-zA-Z_%.]+]] = load <16 x i8>, <16 x i8>* [[REG11]], align 16
// CHECK-NEXT: [[REG17:[0-9a-zA-Z_%.]+]] = load <16 x i8>, <16 x i8>* [[REG15]], align 16
// CHECK-NEXT: call <16 x i8> @vec_sel(unsigned char vector[16], unsigned char vector[16], bool vector[16])(<16 x i8> [[REG16]], <16 x i8> zeroinitializer, <16 x i8> [[REG17]])

// CHECK: define available_externally i64 @_mm_packs_pi16(i64 [[REG18:[0-9a-zA-Z_%.]+]], i64 [[REG19:[0-9a-zA-Z_%.]+]])
// CHECK: store i64 [[REG18]], i64* [[REG20:[0-9a-zA-Z_%.]+]], align 8
// CHECK-NEXT: store i64 [[REG19]], i64* [[REG21:[0-9a-zA-Z_%.]+]], align 8
// CHECK-LE: load i64, i64* [[REG20]], align 8
// CHECK: load i64, i64* [[REG21]], align 8
// CHECK-BE: load i64, i64* [[REG20]], align 8
// CHECK: [[REG22:[0-9a-zA-Z_%.]+]] = load <8 x i16>, <8 x i16>* [[REG23:[0-9a-zA-Z_%.]+]], align 16
// CHECK-NEXT: [[REG24:[0-9a-zA-Z_%.]+]] = load <8 x i16>, <8 x i16>* [[REG23]], align 16
// CHECK-NEXT: call <16 x i8> @vec_packs(short vector[8], short vector[8])(<8 x i16> [[REG22]], <8 x i16> [[REG24]])

// CHECK: define available_externally i64 @_mm_packs_pi32(i64 [[REG25:[0-9a-zA-Z_%.]+]], i64 [[REG26:[0-9a-zA-Z_%.]+]])
// CHECK: store i64 [[REG25]], i64* [[REG27:[0-9a-zA-Z_%.]+]], align 8
// CHECK-NEXT: store i64 [[REG26]], i64* [[REG28:[0-9a-zA-Z_%.]+]], align 8
// CHECK-LE: load i64, i64* [[REG27]], align 8
// CHECK: load i64, i64* [[REG28]], align 8
// CHECK-BE: load i64, i64* [[REG27]], align 8
// CHECK: [[REG29:[0-9a-zA-Z_%.]+]] = load <4 x i32>, <4 x i32>* [[REG30:[0-9a-zA-Z_%.]+]], align 16
// CHECK-NEXT: [[REG31:[0-9a-zA-Z_%.]+]] = load <4 x i32>, <4 x i32>* [[REG30]], align 16
// CHECK-NEXT: call <8 x i16> @vec_packs(int vector[4], int vector[4])(<4 x i32> [[REG29]], <4 x i32> [[REG31]])
