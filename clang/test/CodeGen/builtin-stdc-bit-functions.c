// RUN: %clang_cc1 -ffreestanding %s -emit-llvm -o - | FileCheck %s
// RUN: %if clang-target-64-bits %{ %clang_cc1 -ffreestanding %s -emit-llvm -o - | FileCheck %s --check-prefix=INT128 %}

// CHECK-LABEL: test_leading_zeros
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_leading_zeros(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_leading_zeros(uc);
  r = __builtin_stdc_leading_zeros(us);
  r = __builtin_stdc_leading_zeros(ui);
  r = __builtin_stdc_leading_zeros(ull);
}

// CHECK-LABEL: test_leading_ones
// CHECK: xor i8 %{{.*}}, -1
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: xor i16 %{{.*}}, -1
// CHECK: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// CHECK: xor i32 %{{.*}}, -1
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// CHECK: xor i64 %{{.*}}, -1
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_leading_ones(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_leading_ones(uc);
  r = __builtin_stdc_leading_ones(us);
  r = __builtin_stdc_leading_ones(ui);
  r = __builtin_stdc_leading_ones(ull);
}

// CHECK-LABEL: test_trailing_zeros
// CHECK: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// CHECK: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_trailing_zeros(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_trailing_zeros(uc);
  r = __builtin_stdc_trailing_zeros(us);
  r = __builtin_stdc_trailing_zeros(ui);
  r = __builtin_stdc_trailing_zeros(ull);
}

// CHECK-LABEL: test_trailing_ones
// CHECK: xor i8 %{{.*}}, -1
// CHECK: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// CHECK: xor i16 %{{.*}}, -1
// CHECK: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// CHECK: xor i32 %{{.*}}, -1
// CHECK: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// CHECK: xor i64 %{{.*}}, -1
// CHECK: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_trailing_ones(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_trailing_ones(uc);
  r = __builtin_stdc_trailing_ones(us);
  r = __builtin_stdc_trailing_ones(ui);
  r = __builtin_stdc_trailing_ones(ull);
}

// CHECK-LABEL: test_first_leading_zero
// CHECK: xor i8 %{{.*}}, -1
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: xor i32 %{{.*}}, -1
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_first_leading_zero(unsigned char uc, unsigned int ui) {
  volatile int r;
  r = __builtin_stdc_first_leading_zero(uc);
  r = __builtin_stdc_first_leading_zero(ui);
}

// CHECK-LABEL: test_first_leading_one
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_first_leading_one(unsigned char uc, unsigned int ui) {
  volatile int r;
  r = __builtin_stdc_first_leading_one(uc);
  r = __builtin_stdc_first_leading_one(ui);
}

// CHECK-LABEL: test_first_trailing_zero
// CHECK: xor i8 %{{.*}}, -1
// CHECK: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// CHECK: xor i32 %{{.*}}, -1
// CHECK: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
void test_first_trailing_zero(unsigned char uc, unsigned int ui) {
  volatile int r;
  r = __builtin_stdc_first_trailing_zero(uc);
  r = __builtin_stdc_first_trailing_zero(ui);
}

// CHECK-LABEL: test_first_trailing_one
// CHECK: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
void test_first_trailing_one(unsigned char uc, unsigned int ui) {
  volatile int r;
  r = __builtin_stdc_first_trailing_one(uc);
  r = __builtin_stdc_first_trailing_one(ui);
}

// CHECK-LABEL: test_count_zeros
// CHECK: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// CHECK: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// CHECK: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// CHECK: call i64 @llvm.ctpop.i64(i64 %{{.*}})
void test_count_zeros(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_count_zeros(uc);
  r = __builtin_stdc_count_zeros(us);
  r = __builtin_stdc_count_zeros(ui);
  r = __builtin_stdc_count_zeros(ull);
}

// CHECK-LABEL: test_count_ones
// CHECK: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// CHECK: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// CHECK: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// CHECK: call i64 @llvm.ctpop.i64(i64 %{{.*}})
void test_count_ones(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_count_ones(uc);
  r = __builtin_stdc_count_ones(us);
  r = __builtin_stdc_count_ones(ui);
  r = __builtin_stdc_count_ones(ull);
}

// CHECK-LABEL: test_has_single_bit
// CHECK: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// CHECK: call i32 @llvm.ctpop.i32(i32 %{{.*}})
void test_has_single_bit(unsigned char uc, unsigned int ui) {
  volatile int r;
  r = __builtin_stdc_has_single_bit(uc);
  r = __builtin_stdc_has_single_bit(ui);
}

// CHECK-LABEL: test_bit_width
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_bit_width(unsigned char uc, unsigned int ui) {
  volatile int r;
  r = __builtin_stdc_bit_width(uc);
  r = __builtin_stdc_bit_width(ui);
}

// CHECK-LABEL: test_bit_floor
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_bit_floor(unsigned char uc, unsigned int ui) {
  volatile unsigned char rc;
  volatile unsigned int ri;
  rc = __builtin_stdc_bit_floor(uc);
  ri = __builtin_stdc_bit_floor(ui);
}

// CHECK-LABEL: test_bit_ceil
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_bit_ceil(unsigned char uc, unsigned int ui) {
  volatile unsigned char rc;
  volatile unsigned int ri;
  rc = __builtin_stdc_bit_ceil(uc);
  ri = __builtin_stdc_bit_ceil(ui);
}

// Test with _BitInt types
// CHECK-LABEL: test_bitint
// CHECK: call i37 @llvm.ctlz.i37(i37 %{{.*}}, i1 false)
// CHECK: call i37 @llvm.cttz.i37(i37 %{{.*}}, i1 false)
// CHECK: call i37 @llvm.ctpop.i37(i37 %{{.*}})
void test_bitint(unsigned _BitInt(37) bi) {
  volatile int r;
  r = __builtin_stdc_leading_zeros(bi);
  r = __builtin_stdc_trailing_zeros(bi);
  r = __builtin_stdc_count_ones(bi);
}

// Additional _BitInt coverage
// CHECK-LABEL: test_bitint_floor_ceil
// CHECK: call i9 @llvm.ctlz.i9(i9 %{{.*}}, i1 false)
// CHECK: call i9 @llvm.ctlz.i9(i9 %{{.*}}, i1 false)
void test_bitint_floor_ceil(unsigned _BitInt(9) bi9) {
  volatile unsigned _BitInt(9) rb;
  rb = __builtin_stdc_bit_floor(bi9);
  rb = __builtin_stdc_bit_ceil(bi9);
}

// CHECK-LABEL: test_bitint_first_and_count
// CHECK: call i9 @llvm.ctlz.i9(i9 %{{.*}}, i1 false)
// CHECK: call i9 @llvm.cttz.i9(i9 %{{.*}}, i1 false)
// CHECK: call i9 @llvm.ctpop.i9(i9 %{{.*}})
void test_bitint_first_and_count(unsigned _BitInt(9) bi9) {
  volatile int r;
  r = __builtin_stdc_first_leading_zero(bi9);
  r = __builtin_stdc_first_trailing_zero(bi9);
  r = __builtin_stdc_count_zeros(bi9);
}

// CHECK-LABEL: test_bit_ceil_all_ones
// CHECK: store volatile i32 -1, ptr %r
void test_bit_ceil_all_ones(void) {
  volatile unsigned int r = __builtin_stdc_bit_ceil(0xFFFFFFFFU);
}

// CHECK-LABEL: test_bit_ceil_all_ones_bitint
// CHECK: store volatile i32 131071, ptr %r
void test_bit_ceil_all_ones_bitint(void) {
  volatile unsigned _BitInt(17) r =
      __builtin_stdc_bit_ceil((unsigned _BitInt(17))(-1));
}

// CHECK-LABEL: test_bit_floor_all_ones_bitint
// CHECK: store volatile i32 65536, ptr %r
void test_bit_floor_all_ones_bitint(void) {
  volatile unsigned _BitInt(17) r =
      __builtin_stdc_bit_floor((unsigned _BitInt(17))(-1));
}

// CHECK-LABEL: test_first_trailing_zero_all_ones_bitint
// CHECK: store volatile i32 0, ptr %r
void test_first_trailing_zero_all_ones_bitint(void) {
  volatile unsigned _BitInt(17) r =
      __builtin_stdc_first_trailing_zero((unsigned _BitInt(17))(-1));
}

#ifdef __SIZEOF_INT128__
// INT128-LABEL: test_int128
// INT128: call i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 false)
// INT128: call i128 @llvm.cttz.i128(i128 %{{.*}}, i1 false)
// INT128: call i128 @llvm.ctpop.i128(i128 %{{.*}})
void test_int128(unsigned __int128 u128) {
  volatile int r;
  r = __builtin_stdc_leading_zeros(u128);
  r = __builtin_stdc_trailing_zeros(u128);
  r = __builtin_stdc_count_ones(u128);
}
#endif
