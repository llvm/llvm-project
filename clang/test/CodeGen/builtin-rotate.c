// RUN: %clang_cc1 -ffreestanding %s -emit-llvm -o - | FileCheck %s
// RUN: %if clang-target-64-bits %{ %clang_cc1 -ffreestanding %s -emit-llvm -o - | FileCheck %s --check-prefix=INT128 %}

#include<stdint.h>

unsigned char rotl8(unsigned char x, unsigned char y) {
// CHECK-LABEL: rotl8
// CHECK: [[F:%.*]] = call i8 @llvm.fshl.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK-NEXT: ret i8 [[F]]

  return __builtin_rotateleft8(x, y);
}

short rotl16(short x, short y) {
// CHECK-LABEL: rotl16
// CHECK: [[F:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK-NEXT: ret i16 [[F]]

  return __builtin_rotateleft16(x, y);
}

int rotl32(int x, unsigned int y) {
// CHECK-LABEL: rotl32
// CHECK: [[F:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-NEXT: ret i32 [[F]]

  return __builtin_rotateleft32(x, y);
}

unsigned long long rotl64(unsigned long long x, long long y) {
// CHECK-LABEL: rotl64
// CHECK: [[F:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-NEXT: ret i64 [[F]]

  return __builtin_rotateleft64(x, y);
}

char rotr8(char x, char y) {
// CHECK-LABEL: rotr8
// CHECK: [[F:%.*]] = call i8 @llvm.fshr.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
// CHECK-NEXT: ret i8 [[F]]

  return __builtin_rotateright8(x, y);
}

unsigned short rotr16(unsigned short x, unsigned short y) {
// CHECK-LABEL: rotr16
// CHECK: [[F:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
// CHECK-NEXT: ret i16 [[F]]

  return __builtin_rotateright16(x, y);
}

unsigned int rotr32(unsigned int x, int y) {
// CHECK-LABEL: rotr32
// CHECK: [[F:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
// CHECK-NEXT: ret i32 [[F]]

  return __builtin_rotateright32(x, y);
}

long long rotr64(long long x, unsigned long long y) {
// CHECK-LABEL: rotr64
// CHECK: [[F:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
// CHECK-NEXT: ret i64 [[F]]

  return __builtin_rotateright64(x, y);
}

// CHECK-LABEL: test_builtin_stdc_rotate_left
// CHECK:  call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 3)
// CHECK:  call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 5)
// CHECK:  call i32 @llvm.fshl.i32(i32 %{{.*}}, i32 %{{.*}}, i32 8)
// CHECK:  call i64 @llvm.fshl.i64(i64 %{{.*}}, i64 %{{.*}}, i64 8)
// CHECK:  call i64 @llvm.fshl.i64(i64 %{{.*}}, i64 %{{.*}}, i64 16)
// CHECK:  call i128 @llvm.fshl.i128(i128 %{{.*}}, i128 %{{.*}}, i128 32)
// CHECK:  call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 7)
// CHECK:  call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 11)
// CHECK:  call i32 @llvm.fshl.i32(i32 %{{.*}}, i32 %{{.*}}, i32 29)
// CHECK:  call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 0)
// CHECK:  call i32 @llvm.fshl.i32(i32 %{{.*}}, i32 %{{.*}}, i32 27)
// CHECK:  call i32 @llvm.fshl.i32(i32 42, i32 42, i32 %{{.*}})
// CHECK:  call i9 @llvm.fshl.i9(i9 %{{.*}}, i9 %{{.*}}, i9 1)
// CHECK:  call i37 @llvm.fshl.i37(i37 %{{.*}}, i37 %{{.*}}, i37 36)
// CHECK:  call i9 @llvm.fshl.i9(i9 %{{.*}}, i9 %{{.*}}, i9 8)
// CHECK:  call i37 @llvm.fshl.i37(i37 %{{.*}}, i37 %{{.*}}, i37 32)
// CHECK:  call i10 @llvm.fshl.i10(i10 %{{.*}}, i10 %{{.*}}, i10 3)
// CHECK:  call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 8)
// CHECK:  call i24 @llvm.fshl.i24(i24 %{{.*}}, i24 %{{.*}}, i24 12)
// CHECK:  call i48 @llvm.fshl.i48(i48 %{{.*}}, i48 %{{.*}}, i48 24)
// CHECK:  call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 0)
// CHECK:  call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 0)
// CHECK:  call i32 @llvm.fshl.i32(i32 %{{.*}}, i32 %{{.*}}, i32 16)
// CHECK:  call i32 @llvm.fshl.i32(i32 %{{.*}}, i32 %{{.*}}, i32 31)
// CHECK:  call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 0)
// CHECK:  call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 0)
void test_builtin_stdc_rotate_left(uint8_t u8, uint16_t u16,
                                   uint32_t u32, uint64_t u64,
                                   uint64_t u64_2, unsigned _BitInt(128) u128,
                                   unsigned _BitInt(9) u9, unsigned _BitInt(37) u37,
                                   unsigned _BitInt(10) u10, unsigned _BitInt(16) u16_bit,
                                   unsigned _BitInt(24) u24, unsigned _BitInt(48) u48) {

  volatile uint8_t result_u8;
  volatile uint16_t result_u16;
  volatile uint32_t result_u32;
  volatile uint64_t result_u64;
  volatile uint64_t result_u64_2;
  volatile unsigned _BitInt(128) result_u128;
  volatile unsigned _BitInt(9) result_u9;
  volatile unsigned _BitInt(37) result_u37;
  volatile unsigned _BitInt(10) result_u10;
  volatile unsigned _BitInt(16) result_u16_bit;
  volatile unsigned _BitInt(24) result_u24;
  volatile unsigned _BitInt(48) result_u48;

  result_u8 = __builtin_stdc_rotate_left(u8, 3);
  result_u16 = __builtin_stdc_rotate_left(u16, 5);
  result_u32 = __builtin_stdc_rotate_left(u32, 8);
  result_u64 = __builtin_stdc_rotate_left(u64, 8);
  result_u64_2 = __builtin_stdc_rotate_left(u64_2, 16);
  result_u128 = __builtin_stdc_rotate_left(u128, 32);

  result_u8 = __builtin_stdc_rotate_left(u8, -1);
  result_u16 = __builtin_stdc_rotate_left(u16, -5);
  result_u32 = __builtin_stdc_rotate_left(u32, -3);
  result_u8 = __builtin_stdc_rotate_left(u8, -65536);
  result_u32 = __builtin_stdc_rotate_left(u32, (int64_t)-4294967333LL);

  int var = 3;
  result_u32 = __builtin_stdc_rotate_left(42U, var);

  result_u9 = __builtin_stdc_rotate_left(u9, 1);
  result_u37 = __builtin_stdc_rotate_left(u37, 36);
  result_u9 = __builtin_stdc_rotate_left(u9, -1);
  result_u37 = __builtin_stdc_rotate_left(u37, -5);

  result_u8 = __builtin_stdc_rotate_left((uint8_t)0xAB, 1000000);
  result_u32 = __builtin_stdc_rotate_left(0x12345678U, 4294967295U);
  result_u8 = __builtin_stdc_rotate_left((uint8_t)0xAB, -1000000);
  result_u8 = __builtin_stdc_rotate_left((uint8_t)0x01, 2147483647);

  result_u8 = __builtin_stdc_rotate_left((uint8_t)0xAB, 7);
  result_u8 = __builtin_stdc_rotate_left((uint8_t)0xAB, 8);
  result_u8 = __builtin_stdc_rotate_left((uint8_t)0xAB, 9);

  result_u8 = __builtin_stdc_rotate_left((uint8_t)0xFF, 1073741824);
  result_u32 = __builtin_stdc_rotate_left(0U, 2147483647);
  result_u8 = __builtin_stdc_rotate_left((uint8_t)0x01, 1000000007);

  result_u37 = __builtin_stdc_rotate_left((unsigned _BitInt(37))0x1000000000ULL, 1000000000);

  result_u10 = __builtin_stdc_rotate_left(u10, 3);
  result_u16_bit = __builtin_stdc_rotate_left(u16_bit, 8);
  result_u24 = __builtin_stdc_rotate_left(u24, 12);
  result_u48 = __builtin_stdc_rotate_left(u48, 24);

  result_u10 = __builtin_stdc_rotate_left((unsigned _BitInt(10))0x3FF, -1);
  result_u16_bit = __builtin_stdc_rotate_left((unsigned _BitInt(16))0xFFFF, -5);
  result_u24 = __builtin_stdc_rotate_left((unsigned _BitInt(24))0xABCDEF, 1000000);
  result_u48 = __builtin_stdc_rotate_left((unsigned _BitInt(48))0x123456789ABC, -2147483648);

  uint8_t x = 0x42;
  uint32_t z = 0x12345678;
  result_u8 = __builtin_stdc_rotate_right(__builtin_stdc_rotate_left(x, 1000000), -1000000);
  result_u32 = __builtin_stdc_rotate_left(__builtin_stdc_rotate_left(z, 50000), 4294967295U);

  uint8_t temp = (uint8_t)x ^ __builtin_stdc_rotate_right((uint8_t)x, 1073741824);
  result_u8 = __builtin_stdc_rotate_left(temp, 0x12345678);
}

// CHECK-LABEL: test_builtin_stdc_rotate_right
// CHECK:  call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 3)
// CHECK:  call i16 @llvm.fshr.i16(i16 %{{.*}}, i16 %{{.*}}, i16 5)
// CHECK:  call i32 @llvm.fshr.i32(i32 %{{.*}}, i32 %{{.*}}, i32 8)
// CHECK:  call i64 @llvm.fshr.i64(i64 %{{.*}}, i64 %{{.*}}, i64 8)
// CHECK:  call i64 @llvm.fshr.i64(i64 %{{.*}}, i64 %{{.*}}, i64 16)
// CHECK:  call i128 @llvm.fshr.i128(i128 %{{.*}}, i128 %{{.*}}, i128 32)
// CHECK:  call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 7)
// CHECK:  call i16 @llvm.fshr.i16(i16 %{{.*}}, i16 %{{.*}}, i16 13)
// CHECK:  call i64 @llvm.fshr.i64(i64 %{{.*}}, i64 %{{.*}}, i64 48)
// CHECK:  call i9 @llvm.fshr.i9(i9 %{{.*}}, i9 %{{.*}}, i9 1)
// CHECK:  call i9 @llvm.fshr.i9(i9 %{{.*}}, i9 %{{.*}}, i9 8)
// CHECK:  call i12 @llvm.fshr.i12(i12 %{{.*}}, i12 %{{.*}}, i12 6)
// CHECK:  call i20 @llvm.fshr.i20(i20 %{{.*}}, i20 %{{.*}}, i20 10)
// CHECK:  call i32 @llvm.fshr.i32(i32 %{{.*}}, i32 %{{.*}}, i32 16)
// CHECK:  call i16 @llvm.fshr.i16(i16 %{{.*}}, i16 %{{.*}}, i16 15)
// CHECK:  call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 1)
void test_builtin_stdc_rotate_right(uint8_t u8, uint16_t u16,
                                    uint32_t u32, uint64_t u64,
                                    uint64_t u64_2, unsigned _BitInt(128) u128,
                                    unsigned _BitInt(9) u9, unsigned _BitInt(12) u12,
                                    unsigned _BitInt(20) u20, unsigned _BitInt(32) u32_bit) {

  volatile uint8_t result_u8;
  volatile uint16_t result_u16;
  volatile uint32_t result_u32;
  volatile uint64_t result_u64;
  volatile uint64_t result_u64_2;
  volatile unsigned _BitInt(128) result_u128;
  volatile unsigned _BitInt(9) result_u9;
  volatile unsigned _BitInt(12) result_u12;
  volatile unsigned _BitInt(20) result_u20;
  volatile unsigned _BitInt(32) result_u32_bit;

  result_u8 = __builtin_stdc_rotate_right(u8, 3);
  result_u16 = __builtin_stdc_rotate_right(u16, 5);
  result_u32 = __builtin_stdc_rotate_right(u32, 8);
  result_u64 = __builtin_stdc_rotate_right(u64, 8);
  result_u64_2 = __builtin_stdc_rotate_right(u64_2, 16);
  result_u128 = __builtin_stdc_rotate_right(u128, 32);

  result_u8 = __builtin_stdc_rotate_right(u8, -1);
  result_u16 = __builtin_stdc_rotate_right(u16, -3);
  result_u64_2 = __builtin_stdc_rotate_right(u64_2, -16);

  result_u9 = __builtin_stdc_rotate_right(u9, 1);
  result_u9 = __builtin_stdc_rotate_right(u9, -1);

  result_u16 = __builtin_stdc_rotate_right((uint16_t)0x1234, 2147483647);
  result_u16 = __builtin_stdc_rotate_right((uint16_t)0x1234, -2147483647);
  result_u8 = __builtin_stdc_rotate_right((uint8_t)0x80, -2147483648);

  result_u16 = __builtin_stdc_rotate_right((uint16_t)0xFFFF, -1073741824);
  result_u64_2 = __builtin_stdc_rotate_right(0ULL, -2147483648);
  result_u8 = __builtin_stdc_rotate_right((uint8_t)0x80, -1000000007);

  result_u12 = __builtin_stdc_rotate_right(u12, 6);
  result_u20 = __builtin_stdc_rotate_right(u20, 10);
  result_u32_bit = __builtin_stdc_rotate_right(u32_bit, 16);

  result_u12 = __builtin_stdc_rotate_right((unsigned _BitInt(12))0xFFF, -3);
  result_u20 = __builtin_stdc_rotate_right((unsigned _BitInt(20))0x12345, 1000000);
  result_u32_bit = __builtin_stdc_rotate_right((unsigned _BitInt(32))0xABCDEF01, -2147483647);

  uint16_t y = 0x1234;
  result_u16 = __builtin_stdc_rotate_left(__builtin_stdc_rotate_right(y, 2147483647), -2147483647);
}

// Test _BitInt types with various bit widths
// CHECK-LABEL: test_bitint_extremes
// CHECK:  call i3 @llvm.fshl.i3(i3 %{{.*}}, i3 %{{.*}}, i3 %{{.*}})
// CHECK:  call i23 @llvm.fshl.i23(i23 1193046, i23 1193046, i23 %{{.*}})
// CHECK:  call i37 @llvm.fshl.i37(i37 %{{.*}}, i37 %{{.*}}, i37 %{{.*}})
// CHECK:  call i67 @llvm.fshl.i67(i67 81985529216486895, i67 81985529216486895, i67 %{{.*}})
// CHECK:  call i127 @llvm.fshl.i127(i127 1, i127 1, i127 %{{.*}})
// CHECK:  call i3 @llvm.fshr.i3(i3 %{{.*}}, i3 %{{.*}}, i3 %{{.*}})
// CHECK:  call i23 @llvm.fshr.i23(i23 1193046, i23 1193046, i23 %{{.*}})
// CHECK:  call i37 @llvm.fshr.i37(i37 %{{.*}}, i37 %{{.*}}, i37 %{{.*}})
// CHECK:  call i67 @llvm.fshr.i67(i67 1311768467463790320, i67 1311768467463790320, i67 %{{.*}})
// CHECK:  call i127 @llvm.fshr.i127(i127 1, i127 1, i127 %{{.*}})
void test_bitint_extremes(unsigned _BitInt(3) u3, unsigned _BitInt(37) u37, int shift) {
  volatile unsigned _BitInt(3) result_u3;
  volatile unsigned _BitInt(23) result_u23;
  volatile unsigned _BitInt(37) result_u37;
  volatile unsigned _BitInt(67) result_u67;
  volatile unsigned _BitInt(127) result_u127;

  result_u3 = __builtin_stdc_rotate_left(u3, shift);
  result_u23 = __builtin_stdc_rotate_left((unsigned _BitInt(23))0x123456, shift);
  result_u37 = __builtin_stdc_rotate_left(u37, shift);
  result_u67 = __builtin_stdc_rotate_left((unsigned _BitInt(67))0x123456789ABCDEFULL, shift);
  result_u127 = __builtin_stdc_rotate_left((unsigned _BitInt(127))1, shift);

  result_u3 = __builtin_stdc_rotate_right(u3, shift);
  result_u23 = __builtin_stdc_rotate_right((unsigned _BitInt(23))0x123456, shift);
  result_u37 = __builtin_stdc_rotate_right(u37, shift);
  result_u67 = __builtin_stdc_rotate_right((unsigned _BitInt(67))0x123456789ABCDEF0ULL, shift);
  result_u127 = __builtin_stdc_rotate_right((unsigned _BitInt(127))1, shift);
}

// CHECK-LABEL: test_wider_shift_amount
// CHECK: call i8 @llvm.fshl.i8(i8 %{{.*}}, i8 %{{.*}}, i8 7)
// CHECK: call i8 @llvm.fshr.i8(i8 %{{.*}}, i8 %{{.*}}, i8 0)
// CHECK: call i16 @llvm.fshl.i16(i16 %{{.*}}, i16 %{{.*}}, i16 11)
// CHECK: call i16 @llvm.fshr.i16(i16 %{{.*}}, i16 %{{.*}}, i16 12)
// CHECK: call i32 @llvm.fshl.i32(i32 %{{.*}}, i32 %{{.*}}, i32 0)
// CHECK: call i32 @llvm.fshr.i32(i32 %{{.*}}, i32 %{{.*}}, i32 0)
// CHECK: call i9 @llvm.fshl.i9(i9 %{{.*}}, i9 %{{.*}}, i9 8)
// CHECK: call i9 @llvm.fshr.i9(i9 %{{.*}}, i9 %{{.*}}, i9 8)
void test_wider_shift_amount(uint8_t u8, uint16_t u16, uint32_t u32, unsigned _BitInt(9) u9) {
  volatile uint8_t result_u8;
  volatile uint16_t result_u16;
  volatile uint32_t result_u32;
  volatile unsigned _BitInt(9) result_u9;

  result_u8 = __builtin_stdc_rotate_left(u8, (int64_t)-1);
  result_u8 = __builtin_stdc_rotate_right(u8, (int64_t)-1000);

  result_u16 = __builtin_stdc_rotate_left(u16, (int64_t)-5);
  result_u16 = __builtin_stdc_rotate_right(u16, (int64_t)-100);

  result_u32 = __builtin_stdc_rotate_left(u32, (int64_t)-2147483648);
  result_u32 = __builtin_stdc_rotate_right(u32, (int64_t)-1073741824);

  result_u9 = __builtin_stdc_rotate_left(u9, (int64_t)-1);
  result_u9 = __builtin_stdc_rotate_right(u9, (int64_t)-1000000);

  result_u8 = __builtin_stdc_rotate_left((uint8_t)0xFF, (int64_t)-2147483648);
  result_u16 = __builtin_stdc_rotate_right((uint16_t)0x1234, (int64_t)-1073741824);
  result_u32 = __builtin_stdc_rotate_left(0x12345678U, (int64_t)-4294967296);

  result_u9 = __builtin_stdc_rotate_left((unsigned _BitInt(9))0x1FF, (int64_t)-2147483647);
}

#ifdef __SIZEOF_INT128__
// INT128-LABEL: test_int128_rotate
// INT128:  call i128 @llvm.fshl.i128(i128 %{{.*}}, i128 %{{.*}}, i128 32)
// INT128:  call i128 @llvm.fshr.i128(i128 %{{.*}}, i128 %{{.*}}, i128 32)
void test_int128_rotate(unsigned __int128 u128) {
  volatile unsigned __int128 result_u128;
  result_u128 = __builtin_stdc_rotate_left(u128, 32);
  result_u128 = __builtin_stdc_rotate_right(u128, 32);
}
#endif
