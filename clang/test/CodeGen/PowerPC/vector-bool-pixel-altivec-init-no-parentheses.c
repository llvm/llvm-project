// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=mixed -triple powerpc-unknown-unknown \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=MIXED-ERR
// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=mixed -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=MIXED-ERR
// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=gcc -triple powerpc-unknown-unknown \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC-ERR
// RUN: not %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=gcc -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC-ERR
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=xl -triple powerpc-unknown-unknown \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=XL
// RUN: %clang_cc1 -target-feature +altivec -target-feature +vsx \
// RUN:   -faltivec-src-compat=xl -triple powerpc64le-unknown-unknown \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=XL
// RUN: not %clang -mcpu=pwr8 -faltivec-src-compat=mixed --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=MIXED-ERR
// RUN: not %clang -mcpu=pwr9 -faltivec-src-compat=mixed --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=MIXED-ERR
// RUN: not %clang -mcpu=pwr8 -faltivec-src-compat=gcc --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC-ERR
// RUN: not %clang -mcpu=pwr9 -faltivec-src-compat=gcc --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=GCC-ERR
// RUN: %clang -mcpu=pwr8 -faltivec-src-compat=xl --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=XL
// RUN: %clang -mcpu=pwr9 -faltivec-src-compat=xl --target=powerpc-unknown-unknown \
// RUN:   -S -emit-llvm %s -o - | FileCheck %s --check-prefix=XL

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
void test_vector_bool_pixel_init_no_parentheses(void) {
  // vector bool char initialization
  vbi8_1 = (vector bool char)'a';
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned char'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned char' (vector of 16 'unsigned char' values) and integer type 'int' of different size
  // XL: <i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97, i8 97>
  char c = 'c';
  vbi8_2 = (vector bool char)c;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned char'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned char' (vector of 16 'unsigned char' values) and integer type 'char' of different size
  // XL: [[INS_ELT:%.*]] = insertelement <16 x i8>
  // XL: [[SHUFF:%.*]] = shufflevector <16 x i8> [[INS_ELT]], <16 x i8> poison, <16 x i32> zeroinitializer
  // XL: store <16 x i8> [[SHUFF]]

  // vector bool short initialization
  vbi16_1 = (vector bool short)5;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned short'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned short' (vector of 8 'unsigned short' values) and integer type 'int' of different size
  // XL: <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  short si16 = 55;
  vbi16_2 = (vector bool short)si16;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned short'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned short' (vector of 8 'unsigned short' values) and integer type 'short' of different size
  // XL: [[INS_ELT:%.*]] = insertelement <8 x i16>
  // XL: [[SHUFF:%.*]] = shufflevector <8 x i16> [[INS_ELT]], <8 x i16> poison, <8 x i32> zeroinitializer
  // XL: store <8 x i16> [[SHUFF]]

  // vector bool int initialization
  vbi32_1 = (vector bool int)9;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned int'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned int' (vector of 4 'unsigned int' values) and integer type 'int' of different size
  // XL: <i32 9, i32 9, i32 9, i32 9>
  int si32 = 99;
  vbi32_2 = (vector bool int)si32;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned int'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned int' (vector of 4 'unsigned int' values) and integer type 'int' of different size
  // XL: [[INS_ELT:%.*]] = insertelement <4 x i32>
  // XL: [[SHUFF:%.*]] = shufflevector <4 x i32> [[INS_ELT]], <4 x i32> poison, <4 x i32> zeroinitializer
  // XL: store <4 x i32> [[SHUFF]]

  // vector bool long long initialization
  vbi64_1 = (vector bool long long)13;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned long long'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned long long' (vector of 2 'unsigned long long' values) and integer type 'int' of different size
  // XL: <i64 13, i64 13>
  long long si64 = 1313;
  vbi64_2 = (vector bool long long)si64;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __bool unsigned long long'
  // GCC-ERR: error: invalid conversion between vector type '__vector __bool unsigned long long' (vector of 2 'unsigned long long' values) and integer type 'long long' of different size
  // XL: [[INS_ELT:%.*]] = insertelement <2 x i64>
  // XL: [[SHUFF:%.*]] = shufflevector <2 x i64> [[INS_ELT]], <2 x i64> poison, <2 x i32> zeroinitializer
  // XL: store <2 x i64> [[SHUFF]]

  // vector pixel initialization
  p1 = (vector pixel)1;
  // MIXED-ERR: error: invalid conversion between vector type '__vector __pixel '
  // GCC-ERR: error: invalid conversion between vector type '__vector __pixel ' (vector of 8 'unsigned short' values) and integer type 'int' of different size
  // XL: <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>
}
