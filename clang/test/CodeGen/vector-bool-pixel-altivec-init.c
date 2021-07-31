// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=mixed -triple powerpc-unknown-unknown -S \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=MIXED
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=mixed -triple powerpc64le-unknown-unknown -S \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=MIXED
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=xl -triple powerpc-unknown-unknown -S \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=XL
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=xl -triple powerpc64le-unknown-unknown -S \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=XL
// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=gcc -triple powerpc-unknown-unknown -S \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC
// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=gcc -triple powerpc64le-unknown-unknown -S \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC
// RUN: %clang -mcpu=pwr8 -faltivec-src-compat=mixed --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=MIXED
// RUN: %clang -mcpu=pwr9 -faltivec-src-compat=mixed --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=MIXED
// RUN: %clang -mcpu=pwr8 -faltivec-src-compat=xl --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=XL
// RUN: %clang -mcpu=pwr9 -faltivec-src-compat=xl --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=XL
// RUN: not %clang -mcpu=pwr8 -faltivec-src-compat=gcc --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC
// RUN: not %clang -mcpu=pwr9 -faltivec-src-compat=gcc --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC

// Vector bool type
vector bool char vbi8_1;
vector bool char vbi8_2;

vector bool short vbi16_1;
vector bool short vbi16_2;

vector bool int vbi32_1;
vector bool int vbi32_2;

vector bool long long vbi64_1;
vector bool long long vbi64_2;

// Vector pixel type
vector pixel p1;

////////////////////////////////////////////////////////////////////////////////
void test_vector_bool_pixel_init() {
  // vector bool char initialization
  vbi8_1 = (vector bool char)('a');
  // MIXED: <i8 97, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  // XL: <i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97>
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned char' (vector of 16 'unsigned char' values) and integer type 'unsigned char' of different size
  char c = 'c';
  vbi8_2 = (vector bool char)(c);
  // MIXED: [[INS:%.*]] = insertelement <16 x i8>
  // MIXED: store <16 x i8> [[INS:%.*]]
  // XL: [[INS_ELT:%.*]] = insertelement <16 x i8>
  // XL: [[SHUFF:%.*]] = shufflevector <16 x i8> [[INS_ELT]], <16 x i8> poison, <16 x i32> zeroinitializer
  // XL: store <16 x i8> [[SHUFF]]
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned char' (vector of 16 'unsigned char' values) and integer type 'unsigned char' of different size

  // vector bool short initialization
  vbi16_1 = (vector bool short)(5);
  // MIXED: <i16 5, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  // XL: <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned short' (vector of 8 'unsigned short' values) and integer type 'unsigned short' of different size
  short si16 = 55;
  vbi16_2 = (vector bool short)(si16);
  // MIXED: [[INS:%.*]] = insertelement <8 x i16>
  // MIXED: store <8 x i16> [[INS:%.*]]
  // XL: [[INS_ELT:%.*]] = insertelement <8 x i16>
  // XL: [[SHUFF:%.*]] = shufflevector <8 x i16> [[INS_ELT]], <8 x i16> poison, <8 x i32> zeroinitializer
  // XL: store <8 x i16> [[SHUFF]]
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned short' (vector of 8 'unsigned short' values) and integer type 'unsigned short' of different size

  // vector bool int initialization
  vbi32_1 = (vector bool int)(9);
  // MIXED: <i32 9, i32 0, i32 0, i32 0>
  // XL: <i32 9, i32 9, i32 9, i32 9>
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned int' (vector of 4 'unsigned int' values) and integer type 'unsigned int' of different size
  int si32 = 99;
  vbi32_2 = (vector bool int)(si32);
  // MIXED: [[INS:%.*]] = insertelement <4 x i32>
  // MIXED: store <4 x i32> [[INS:%.*]]
  // XL: [[INS_ELT:%.*]] = insertelement <4 x i32>
  // XL: [[SHUFF:%.*]] = shufflevector <4 x i32> [[INS_ELT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // XL: store <4 x i32> [[SHUFF]]
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned int' (vector of 4 'unsigned int' values) and integer type 'unsigned int' of different size

  // vector bool long long initialization
  vbi64_1 = (vector bool long long)(13);
  // MIXED: <i64 13, i64 0>
  // XL: <i64 13, i64 13>
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned long long' (vector of 2 'unsigned long long' values) and integer type 'unsigned long long' of different size
  long long si64 = 1313;
  vbi64_2 = (vector bool long long)(si64);
  // MIXED: [[INS:%.*]] = insertelement <2 x i64>
  // MIXED: store <2 x i64> [[INS:%.*]]
  // XL: [[INS_ELT:%.*]] = insertelement <2 x i64>
  // XL: [[SHUFF:%.*]] = shufflevector <2 x i64> [[INS_ELT]], <2 x i64> poison, <2 x i32> zeroinitializer
  // XL: store <2 x i64> [[SHUFF]]
  // GCC: error: invalid conversion between vector type '__vector __bool unsigned long long' (vector of 2 'unsigned long long' values) and integer type 'unsigned long long' of different size

  // vector pixel initialization
  p1 = (vector pixel)(1);
  // MIXED: <i16 1, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  // XL: <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
  // GCC: error: invalid conversion between vector type '__vector __pixel ' (vector of 8 'unsigned short' values) and integer type 'unsigned short' of different size
}
