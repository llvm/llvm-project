// RUN: %clang_cc1 -ffreestanding -std=c23 %s -emit-llvm -o - | FileCheck %s
// RUN: %if clang-target-64-bits %{ %clang_cc1 -ffreestanding -std=c23 %s -emit-llvm -o - | FileCheck %s --check-prefix=INT128 %}
// RUN: %clang_cc1 -std=c23 -isystem %S/Inputs -DTEST_LIB_SPELLINGS %s -emit-llvm -o - | FileCheck %s --check-prefix=LIB

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
// CHECK: xor i16 %{{.*}}, -1
// CHECK: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// CHECK: xor i32 %{{.*}}, -1
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// CHECK: xor i64 %{{.*}}, -1
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_first_leading_zero(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_first_leading_zero(uc);
  r = __builtin_stdc_first_leading_zero(us);
  r = __builtin_stdc_first_leading_zero(ui);
  r = __builtin_stdc_first_leading_zero(ull);
}

// CHECK-LABEL: test_first_leading_one
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_first_leading_one(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_first_leading_one(uc);
  r = __builtin_stdc_first_leading_one(us);
  r = __builtin_stdc_first_leading_one(ui);
  r = __builtin_stdc_first_leading_one(ull);
}

// CHECK-LABEL: test_first_trailing_zero
// CHECK: xor i8 %{{.*}}, -1
// CHECK: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// CHECK: xor i16 %{{.*}}, -1
// CHECK: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// CHECK: xor i32 %{{.*}}, -1
// CHECK: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// CHECK: xor i64 %{{.*}}, -1
// CHECK: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_first_trailing_zero(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_first_trailing_zero(uc);
  r = __builtin_stdc_first_trailing_zero(us);
  r = __builtin_stdc_first_trailing_zero(ui);
  r = __builtin_stdc_first_trailing_zero(ull);
}

// CHECK-LABEL: test_first_trailing_one
// CHECK: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// CHECK: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_first_trailing_one(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_first_trailing_one(uc);
  r = __builtin_stdc_first_trailing_one(us);
  r = __builtin_stdc_first_trailing_one(ui);
  r = __builtin_stdc_first_trailing_one(ull);
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
// CHECK: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// CHECK: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// CHECK: call i64 @llvm.ctpop.i64(i64 %{{.*}})
void test_has_single_bit(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_has_single_bit(uc);
  r = __builtin_stdc_has_single_bit(us);
  r = __builtin_stdc_has_single_bit(ui);
  r = __builtin_stdc_has_single_bit(ull);
}

// CHECK-LABEL: test_bit_width
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_bit_width(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile int r;
  r = __builtin_stdc_bit_width(uc);
  r = __builtin_stdc_bit_width(us);
  r = __builtin_stdc_bit_width(ui);
  r = __builtin_stdc_bit_width(ull);
}

// CHECK-LABEL: test_bit_floor_uc
// CHECK: [[LZ:%.*]] = call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 true)
// CHECK: [[SHIFT:%.*]] = sub i8 7, [[LZ]]
// CHECK: [[SHIFTED:%.*]] = shl i8 1, [[SHIFT]]
// CHECK: [[ISZERO:%.*]] = icmp eq i8 %{{.*}}, 0
// CHECK: select i1 [[ISZERO]], i8 0, i8 [[SHIFTED]]
unsigned char test_bit_floor_uc(unsigned char uc) { return __builtin_stdc_bit_floor(uc); }

// CHECK-LABEL: test_bit_floor_us
// CHECK: [[LZ:%.*]] = call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 true)
// CHECK: [[SHIFT:%.*]] = sub i16 15, [[LZ]]
// CHECK: [[SHIFTED:%.*]] = shl i16 1, [[SHIFT]]
// CHECK: [[ISZERO:%.*]] = icmp eq i16 %{{.*}}, 0
// CHECK: select i1 [[ISZERO]], i16 0, i16 [[SHIFTED]]
unsigned short test_bit_floor_us(unsigned short us) { return __builtin_stdc_bit_floor(us); }

// CHECK-LABEL: test_bit_floor_ui
// CHECK: [[LZ:%.*]] = call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// CHECK: [[SHIFT:%.*]] = sub i32 31, [[LZ]]
// CHECK: [[SHIFTED:%.*]] = shl i32 1, [[SHIFT]]
// CHECK: [[ISZERO:%.*]] = icmp eq i32 %{{.*}}, 0
// CHECK: select i1 [[ISZERO]], i32 0, i32 [[SHIFTED]]
unsigned int test_bit_floor_ui(unsigned int ui) { return __builtin_stdc_bit_floor(ui); }

// CHECK-LABEL: test_bit_floor_ull
// CHECK: [[LZ:%.*]] = call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
// CHECK: [[SHIFT:%.*]] = sub i64 63, [[LZ]]
// CHECK: [[SHIFTED:%.*]] = shl i64 1, [[SHIFT]]
// CHECK: [[ISZERO:%.*]] = icmp eq i64 %{{.*}}, 0
// CHECK: select i1 [[ISZERO]], i64 0, i64 [[SHIFTED]]
unsigned long long test_bit_floor_ull(unsigned long long ull) { return __builtin_stdc_bit_floor(ull); }

// CHECK-LABEL: test_bit_ceil_uc
// CHECK: icmp ule i8 %{{.*}}, 1
// CHECK: br i1 %{{.*}}, label %bitceil.merge, label %bitceil.calc
// CHECK: bitceil.calc:
// CHECK: sub i8 %{{.*}}, 1
// CHECK: [[LZ:%.*]] = call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: [[SHIFT:%.*]] = sub i8 7, [[LZ]]
// CHECK: shl i8 2, [[SHIFT]]
// CHECK: bitceil.merge:
// CHECK: phi i8 [ 1, %{{.*}} ], [ %{{.*}}, %bitceil.calc ]
unsigned char test_bit_ceil_uc(unsigned char uc) { return __builtin_stdc_bit_ceil(uc); }

// CHECK-LABEL: test_bit_ceil_us
// CHECK: icmp ule i16 %{{.*}}, 1
// CHECK: br i1 %{{.*}}, label %bitceil.merge, label %bitceil.calc
// CHECK: bitceil.calc:
// CHECK: sub i16 %{{.*}}, 1
// CHECK: [[LZ:%.*]] = call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// CHECK: [[SHIFT:%.*]] = sub i16 15, [[LZ]]
// CHECK: shl i16 2, [[SHIFT]]
// CHECK: bitceil.merge:
// CHECK: phi i16 [ 1, %{{.*}} ], [ %{{.*}}, %bitceil.calc ]
unsigned short test_bit_ceil_us(unsigned short us) { return __builtin_stdc_bit_ceil(us); }

// CHECK-LABEL: test_bit_ceil_ui
// CHECK: icmp ule i32 %{{.*}}, 1
// CHECK: br i1 %{{.*}}, label %bitceil.merge, label %bitceil.calc
// CHECK: bitceil.calc:
// CHECK: sub i32 %{{.*}}, 1
// CHECK: [[LZ:%.*]] = call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// CHECK: [[SHIFT:%.*]] = sub i32 31, [[LZ]]
// CHECK: shl i32 2, [[SHIFT]]
// CHECK: bitceil.merge:
// CHECK: phi i32 [ 1, %{{.*}} ], [ %{{.*}}, %bitceil.calc ]
unsigned int test_bit_ceil_ui(unsigned int ui) { return __builtin_stdc_bit_ceil(ui); }

// CHECK-LABEL: test_bit_ceil_ull
// CHECK: icmp ule i64 %{{.*}}, 1
// CHECK: br i1 %{{.*}}, label %bitceil.merge, label %bitceil.calc
// CHECK: bitceil.calc:
// CHECK: sub i64 %{{.*}}, 1
// CHECK: [[LZ:%.*]] = call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// CHECK: [[SHIFT:%.*]] = sub i64 63, [[LZ]]
// CHECK: shl i64 2, [[SHIFT]]
// CHECK: bitceil.merge:
// CHECK: phi i64 [ 1, %{{.*}} ], [ %{{.*}}, %bitceil.calc ]
unsigned long long test_bit_ceil_ull(unsigned long long ull) { return __builtin_stdc_bit_ceil(ull); }

// CHECK-LABEL: test_bit_floor
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 true)
// CHECK: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 true)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
void test_bit_floor(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile unsigned char rc;
  volatile unsigned short rs;
  volatile unsigned int ri;
  volatile unsigned long long rll;
  rc = __builtin_stdc_bit_floor(uc);
  rs = __builtin_stdc_bit_floor(us);
  ri = __builtin_stdc_bit_floor(ui);
  rll = __builtin_stdc_bit_floor(ull);
}

// CHECK-LABEL: test_bit_ceil
// CHECK: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// CHECK: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// CHECK: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_bit_ceil(unsigned char uc, unsigned short us, unsigned int ui, unsigned long long ull) {
  volatile unsigned char rc;
  volatile unsigned short rs;
  volatile unsigned int ri;
  volatile unsigned long long rll;
  rc = __builtin_stdc_bit_ceil(uc);
  rs = __builtin_stdc_bit_ceil(us);
  ri = __builtin_stdc_bit_ceil(ui);
  rll = __builtin_stdc_bit_ceil(ull);
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
// CHECK: call i9 @llvm.ctlz.i9(i9 %{{.*}}, i1 true)
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

// INT128-LABEL: test_int128_leading_trailing_ones
// INT128: xor i128 %{{.*}}, -1
// INT128: call i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 false)
// INT128: xor i128 %{{.*}}, -1
// INT128: call i128 @llvm.cttz.i128(i128 %{{.*}}, i1 false)
void test_int128_leading_trailing_ones(unsigned __int128 u128) {
  volatile int r;
  r = __builtin_stdc_leading_ones(u128);
  r = __builtin_stdc_trailing_ones(u128);
}

// INT128-LABEL: test_int128_first
// INT128: xor i128 %{{.*}}, -1
// INT128: call i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 false)
// INT128: call i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 false)
// INT128: xor i128 %{{.*}}, -1
// INT128: call i128 @llvm.cttz.i128(i128 %{{.*}}, i1 false)
// INT128: call i128 @llvm.cttz.i128(i128 %{{.*}}, i1 false)
void test_int128_first(unsigned __int128 u128) {
  volatile int r;
  r = __builtin_stdc_first_leading_zero(u128);
  r = __builtin_stdc_first_leading_one(u128);
  r = __builtin_stdc_first_trailing_zero(u128);
  r = __builtin_stdc_first_trailing_one(u128);
}

// INT128-LABEL: test_int128_count_has_width
// INT128: call i128 @llvm.ctpop.i128(i128 %{{.*}})
// INT128: call i128 @llvm.ctpop.i128(i128 %{{.*}})
// INT128: call i128 @llvm.ctpop.i128(i128 %{{.*}})
// INT128: call i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 false)
void test_int128_count_has_width(unsigned __int128 u128) {
  volatile int r;
  r = __builtin_stdc_count_zeros(u128);
  r = __builtin_stdc_count_ones(u128);
  r = __builtin_stdc_has_single_bit(u128);
  r = __builtin_stdc_bit_width(u128);
}

// INT128-LABEL: test_int128_floor_ceil
// INT128: call i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 true)
// INT128: call i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 false)
void test_int128_floor_ceil(unsigned __int128 u128) {
  volatile unsigned __int128 r;
  r = __builtin_stdc_bit_floor(u128);
  r = __builtin_stdc_bit_ceil(u128);
}
#endif

// CHECK-LABEL: test_ulong
// CHECK: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: xor {{i32|i64}} %{{.*}}, -1
// CHECK: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: xor {{i32|i64}} %{{.*}}, -1
// CHECK: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: xor {{i32|i64}} %{{.*}}, -1
// CHECK: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: xor {{i32|i64}} %{{.*}}, -1
// CHECK: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: call {{i32|i64}} @llvm.ctpop.{{i32|i64}}({{i32|i64}} %{{.*}})
// CHECK: call {{i32|i64}} @llvm.ctpop.{{i32|i64}}({{i32|i64}} %{{.*}})
// CHECK: call {{i32|i64}} @llvm.ctpop.{{i32|i64}}({{i32|i64}} %{{.*}})
// CHECK: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// CHECK: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 true)
// CHECK: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
void test_ulong(unsigned long ul) {
  volatile unsigned r;
  volatile unsigned long rl;
  volatile _Bool rb;
  r  = __builtin_stdc_leading_zeros(ul);
  r  = __builtin_stdc_leading_ones(ul);
  r  = __builtin_stdc_trailing_zeros(ul);
  r  = __builtin_stdc_trailing_ones(ul);
  r  = __builtin_stdc_first_leading_zero(ul);
  r  = __builtin_stdc_first_leading_one(ul);
  r  = __builtin_stdc_first_trailing_zero(ul);
  r  = __builtin_stdc_first_trailing_one(ul);
  r  = __builtin_stdc_count_zeros(ul);
  r  = __builtin_stdc_count_ones(ul);
  rb = __builtin_stdc_has_single_bit(ul);
  r  = __builtin_stdc_bit_width(ul);
  rl = __builtin_stdc_bit_floor(ul);
  rl = __builtin_stdc_bit_ceil(ul);
}

#ifdef TEST_LIB_SPELLINGS
#include <stdbit.h>

// LIB-LABEL: test_lib_leading_zeros
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_lib_leading_zeros(unsigned ui) {
  volatile unsigned r = stdc_leading_zeros(ui);
}

// LIB-LABEL: test_lib_leading_ones
// LIB: xor i32 %{{.*}}, -1
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_lib_leading_ones(unsigned ui) {
  volatile unsigned r = stdc_leading_ones(ui);
}

// LIB-LABEL: test_lib_trailing_zeros
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
void test_lib_trailing_zeros(unsigned ui) {
  volatile unsigned r = stdc_trailing_zeros(ui);
}

// LIB-LABEL: test_lib_trailing_ones
// LIB: xor i32 %{{.*}}, -1
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
void test_lib_trailing_ones(unsigned ui) {
  volatile unsigned r = stdc_trailing_ones(ui);
}

// LIB-LABEL: test_lib_first_leading_zero
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_lib_first_leading_zero(unsigned ui) {
  volatile unsigned r = stdc_first_leading_zero(ui);
}

// LIB-LABEL: test_lib_first_leading_one
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_lib_first_leading_one(unsigned ui) {
  volatile unsigned r = stdc_first_leading_one(ui);
}

// LIB-LABEL: test_lib_first_trailing_zero
// LIB: xor i32 %{{.*}}, -1
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
void test_lib_first_trailing_zero(unsigned ui) {
  volatile unsigned r = stdc_first_trailing_zero(ui);
}

// LIB-LABEL: test_lib_first_trailing_one
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
void test_lib_first_trailing_one(unsigned ui) {
  volatile unsigned r = stdc_first_trailing_one(ui);
}

// LIB-LABEL: test_lib_count_zeros
// LIB: call i32 @llvm.ctpop.i32(i32 %{{.*}})
void test_lib_count_zeros(unsigned ui) {
  volatile unsigned r = stdc_count_zeros(ui);
}

// LIB-LABEL: test_lib_count_ones
// LIB: call i32 @llvm.ctpop.i32(i32 %{{.*}})
void test_lib_count_ones(unsigned ui) {
  volatile unsigned r = stdc_count_ones(ui);
}

// LIB-LABEL: test_lib_has_single_bit
// LIB: call i32 @llvm.ctpop.i32(i32 %{{.*}})
void test_lib_has_single_bit(unsigned ui) {
  volatile _Bool r = stdc_has_single_bit(ui);
}

// LIB-LABEL: test_lib_bit_width
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_lib_bit_width(unsigned ui) {
  volatile unsigned r = stdc_bit_width(ui);
}

// LIB-LABEL: test_lib_bit_floor
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
void test_lib_bit_floor(unsigned ui) {
  volatile unsigned r = stdc_bit_floor(ui);
}

// LIB-LABEL: test_lib_bit_ceil
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
void test_lib_bit_ceil(unsigned ui) {
  volatile unsigned r = stdc_bit_ceil(ui);
}

// LIB-LABEL: test_lib_typed_leading_zeros
// LIB: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LIB: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LIB: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_leading_zeros(unsigned char uc, unsigned short us,
                                  unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_leading_zeros_uc(uc);
  r = stdc_leading_zeros_us(us);
  r = stdc_leading_zeros_ui(ui);
  r = stdc_leading_zeros_ull(ull);
}

// LIB-LABEL: test_lib_typed_leading_ones
// LIB: xor i8 %{{.*}}, -1
// LIB: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LIB: xor i16 %{{.*}}, -1
// LIB: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LIB: xor i32 %{{.*}}, -1
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LIB: xor i64 %{{.*}}, -1
// LIB: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_leading_ones(unsigned char uc, unsigned short us,
                                 unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_leading_ones_uc(uc);
  r = stdc_leading_ones_us(us);
  r = stdc_leading_ones_ui(ui);
  r = stdc_leading_ones_ull(ull);
}

// LIB-LABEL: test_lib_typed_trailing_zeros
// LIB: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LIB: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LIB: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_trailing_zeros(unsigned char uc, unsigned short us,
                                   unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_trailing_zeros_uc(uc);
  r = stdc_trailing_zeros_us(us);
  r = stdc_trailing_zeros_ui(ui);
  r = stdc_trailing_zeros_ull(ull);
}

// LIB-LABEL: test_lib_typed_trailing_ones
// LIB: xor i8 %{{.*}}, -1
// LIB: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LIB: xor i16 %{{.*}}, -1
// LIB: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LIB: xor i32 %{{.*}}, -1
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LIB: xor i64 %{{.*}}, -1
// LIB: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_trailing_ones(unsigned char uc, unsigned short us,
                                  unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_trailing_ones_uc(uc);
  r = stdc_trailing_ones_us(us);
  r = stdc_trailing_ones_ui(ui);
  r = stdc_trailing_ones_ull(ull);
}

// LIB-LABEL: test_lib_typed_first_leading_zero
// LIB: xor i8 %{{.*}}, -1
// LIB: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LIB: xor i16 %{{.*}}, -1
// LIB: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LIB: xor i32 %{{.*}}, -1
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LIB: xor i64 %{{.*}}, -1
// LIB: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_first_leading_zero(unsigned char uc, unsigned short us,
                                       unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_first_leading_zero_uc(uc);
  r = stdc_first_leading_zero_us(us);
  r = stdc_first_leading_zero_ui(ui);
  r = stdc_first_leading_zero_ull(ull);
}

// LIB-LABEL: test_lib_typed_first_leading_one
// LIB: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LIB: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LIB: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_first_leading_one(unsigned char uc, unsigned short us,
                                      unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_first_leading_one_uc(uc);
  r = stdc_first_leading_one_us(us);
  r = stdc_first_leading_one_ui(ui);
  r = stdc_first_leading_one_ull(ull);
}

// LIB-LABEL: test_lib_typed_first_trailing_zero
// LIB: xor i8 %{{.*}}, -1
// LIB: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LIB: xor i16 %{{.*}}, -1
// LIB: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LIB: xor i32 %{{.*}}, -1
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LIB: xor i64 %{{.*}}, -1
// LIB: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_first_trailing_zero(unsigned char uc, unsigned short us,
                                        unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_first_trailing_zero_uc(uc);
  r = stdc_first_trailing_zero_us(us);
  r = stdc_first_trailing_zero_ui(ui);
  r = stdc_first_trailing_zero_ull(ull);
}

// LIB-LABEL: test_lib_typed_first_trailing_one
// LIB: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LIB: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LIB: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LIB: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_first_trailing_one(unsigned char uc, unsigned short us,
                                       unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_first_trailing_one_uc(uc);
  r = stdc_first_trailing_one_us(us);
  r = stdc_first_trailing_one_ui(ui);
  r = stdc_first_trailing_one_ull(ull);
}

// LIB-LABEL: test_lib_typed_count_zeros
// LIB: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// LIB: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LIB: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LIB: call i64 @llvm.ctpop.i64(i64 %{{.*}})
void test_lib_typed_count_zeros(unsigned char uc, unsigned short us,
                                unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_count_zeros_uc(uc);
  r = stdc_count_zeros_us(us);
  r = stdc_count_zeros_ui(ui);
  r = stdc_count_zeros_ull(ull);
}

// LIB-LABEL: test_lib_typed_count_ones
// LIB: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// LIB: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LIB: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LIB: call i64 @llvm.ctpop.i64(i64 %{{.*}})
void test_lib_typed_count_ones(unsigned char uc, unsigned short us,
                               unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_count_ones_uc(uc);
  r = stdc_count_ones_us(us);
  r = stdc_count_ones_ui(ui);
  r = stdc_count_ones_ull(ull);
}

// LIB-LABEL: test_lib_typed_has_single_bit
// LIB: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// LIB: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LIB: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LIB: call i64 @llvm.ctpop.i64(i64 %{{.*}})
void test_lib_typed_has_single_bit(unsigned char uc, unsigned short us,
                                   unsigned int ui, unsigned long long ull) {
  volatile _Bool r;
  r = stdc_has_single_bit_uc(uc);
  r = stdc_has_single_bit_us(us);
  r = stdc_has_single_bit_ui(ui);
  r = stdc_has_single_bit_ull(ull);
}

// LIB-LABEL: test_lib_typed_bit_width
// LIB: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LIB: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LIB: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_bit_width(unsigned char uc, unsigned short us,
                              unsigned int ui, unsigned long long ull) {
  volatile unsigned r;
  r = stdc_bit_width_uc(uc);
  r = stdc_bit_width_us(us);
  r = stdc_bit_width_ui(ui);
  r = stdc_bit_width_ull(ull);
}

// LIB-LABEL: test_lib_typed_bit_floor
// LIB: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 true)
// LIB: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 true)
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// LIB: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
void test_lib_typed_bit_floor(unsigned char uc, unsigned short us,
                              unsigned int ui, unsigned long long ull) {
  volatile unsigned char rc = stdc_bit_floor_uc(uc);
  volatile unsigned short rs = stdc_bit_floor_us(us);
  volatile unsigned int ri = stdc_bit_floor_ui(ui);
  volatile unsigned long long rll = stdc_bit_floor_ull(ull);
}

// LIB-LABEL: test_lib_typed_bit_ceil
// LIB: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LIB: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LIB: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LIB: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
void test_lib_typed_bit_ceil(unsigned char uc, unsigned short us,
                             unsigned int ui, unsigned long long ull) {
  volatile unsigned char rc = stdc_bit_ceil_uc(uc);
  volatile unsigned short rs = stdc_bit_ceil_us(us);
  volatile unsigned int ri = stdc_bit_ceil_ui(ui);
  volatile unsigned long long rll = stdc_bit_ceil_ull(ull);
}

// LIB-LABEL: test_lib_typed_ul
// LIB: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: xor {{i32|i64}} %{{.*}}, -1
// LIB: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: xor {{i32|i64}} %{{.*}}, -1
// LIB: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: xor {{i32|i64}} %{{.*}}, -1
// LIB: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: xor {{i32|i64}} %{{.*}}, -1
// LIB: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: call {{i32|i64}} @llvm.cttz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: call {{i32|i64}} @llvm.ctpop.{{i32|i64}}({{i32|i64}} %{{.*}})
// LIB: call {{i32|i64}} @llvm.ctpop.{{i32|i64}}({{i32|i64}} %{{.*}})
// LIB: call {{i32|i64}} @llvm.ctpop.{{i32|i64}}({{i32|i64}} %{{.*}})
// LIB: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
// LIB: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 true)
// LIB: call {{i32|i64}} @llvm.ctlz.{{i32|i64}}({{i32|i64}} %{{.*}}, i1 false)
void test_lib_typed_ul(unsigned long ul) {
  volatile unsigned r;
  volatile unsigned long rl;
  volatile _Bool rb;
  r  = stdc_leading_zeros_ul(ul);
  r  = stdc_leading_ones_ul(ul);
  r  = stdc_trailing_zeros_ul(ul);
  r  = stdc_trailing_ones_ul(ul);
  r  = stdc_first_leading_zero_ul(ul);
  r  = stdc_first_leading_one_ul(ul);
  r  = stdc_first_trailing_zero_ul(ul);
  r  = stdc_first_trailing_one_ul(ul);
  r  = stdc_count_zeros_ul(ul);
  r  = stdc_count_ones_ul(ul);
  rb = stdc_has_single_bit_ul(ul);
  r  = stdc_bit_width_ul(ul);
  rl = stdc_bit_floor_ul(ul);
  rl = stdc_bit_ceil_ul(ul);
}
#endif
