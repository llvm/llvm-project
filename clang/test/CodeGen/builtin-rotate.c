// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

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
// CHECK-DAG: call i8 @llvm.fshl.i8
// CHECK-DAG: call i16 @llvm.fshl.i16
// CHECK-DAG: call i32 @llvm.fshl.i32
// CHECK-DAG: call i{{32|64}} @llvm.fshl.i{{32|64}}
// CHECK-DAG: call i64 @llvm.fshl.i64
// CHECK-DAG: call i128 @llvm.fshl.i128
// CHECK-DAG: call i9 @llvm.fshl.i9
// CHECK-DAG: call i37 @llvm.fshl.i37
// CHECK-DAG: call i10 @llvm.fshl.i10
// CHECK-DAG: call i24 @llvm.fshl.i24
// CHECK-DAG: call i48 @llvm.fshl.i48
void test_builtin_stdc_rotate_left(unsigned char uc, unsigned short us,
                                   unsigned int ui, unsigned long ul,
                                   unsigned long long ull, unsigned __int128 ui128,
                                   unsigned _BitInt(9) val9, unsigned _BitInt(37) val37,
                                   unsigned _BitInt(10) val10, unsigned _BitInt(16) val16,
                                   unsigned _BitInt(24) val24, unsigned _BitInt(48) val48) {

  volatile unsigned char result_uc;
  volatile unsigned short result_us;
  volatile unsigned int result_ui;
  volatile unsigned long result_ul;
  volatile unsigned long long result_ull;
  volatile unsigned __int128 result_ui128;
  volatile unsigned _BitInt(9) result_9;
  volatile unsigned _BitInt(37) result_37;
  volatile unsigned _BitInt(10) result_10;
  volatile unsigned _BitInt(16) result_16;
  volatile unsigned _BitInt(24) result_24;
  volatile unsigned _BitInt(48) result_48;

  result_uc = __builtin_stdc_rotate_left(uc, 3);
  result_us = __builtin_stdc_rotate_left(us, 5);
  result_ui = __builtin_stdc_rotate_left(ui, 8);
  result_ul = __builtin_stdc_rotate_left(ul, 8);
  result_ull = __builtin_stdc_rotate_left(ull, 16);
  result_ui128 = __builtin_stdc_rotate_left(ui128, 32);

  result_uc = __builtin_stdc_rotate_left(uc, -1);
  result_us = __builtin_stdc_rotate_left(us, -5);
  result_ui = __builtin_stdc_rotate_left(ui, -3);
  result_uc = __builtin_stdc_rotate_left(uc, -65536);
  result_ui = __builtin_stdc_rotate_left(ui, -4294967333LL);

  int var = 3;
  result_ui = __builtin_stdc_rotate_left(42U, var);

  result_9 = __builtin_stdc_rotate_left(val9, 1);
  result_37 = __builtin_stdc_rotate_left(val37, 36);
  result_9 = __builtin_stdc_rotate_left(val9, -1);
  result_37 = __builtin_stdc_rotate_left(val37, -5);

  result_uc = __builtin_stdc_rotate_left((unsigned char)0xAB, 1000000);
  result_ui = __builtin_stdc_rotate_left(0x12345678U, 4294967295U);
  result_uc = __builtin_stdc_rotate_left((unsigned char)0xAB, -1000000);
  result_uc = __builtin_stdc_rotate_left((unsigned char)0x01, 2147483647);

  result_uc = __builtin_stdc_rotate_left((unsigned char)0xAB, 7);
  result_uc = __builtin_stdc_rotate_left((unsigned char)0xAB, 8);
  result_uc = __builtin_stdc_rotate_left((unsigned char)0xAB, 9);

  result_uc = __builtin_stdc_rotate_left((unsigned char)0xFF, 1073741824);
  result_ui = __builtin_stdc_rotate_left(0U, 2147483647);
  result_uc = __builtin_stdc_rotate_left((unsigned char)0x01, 1000000007);

  result_37 = __builtin_stdc_rotate_left((unsigned _BitInt(37))0x1000000000ULL, 1000000000);

  result_10 = __builtin_stdc_rotate_left(val10, 3);
  result_16 = __builtin_stdc_rotate_left(val16, 8);
  result_24 = __builtin_stdc_rotate_left(val24, 12);
  result_48 = __builtin_stdc_rotate_left(val48, 24);

  result_10 = __builtin_stdc_rotate_left((unsigned _BitInt(10))0x3FF, -1);
  result_16 = __builtin_stdc_rotate_left((unsigned _BitInt(16))0xFFFF, -5);
  result_24 = __builtin_stdc_rotate_left((unsigned _BitInt(24))0xABCDEF, 1000000);
  result_48 = __builtin_stdc_rotate_left((unsigned _BitInt(48))0x123456789ABC, -2147483648);

  unsigned char x = 0x42;
  unsigned int z = 0x12345678;
  result_uc = __builtin_stdc_rotate_right(__builtin_stdc_rotate_left(x, 1000000), -1000000);
  result_ui = __builtin_stdc_rotate_left(__builtin_stdc_rotate_left(z, 50000), 4294967295U);

  unsigned char temp = (unsigned char)x ^ __builtin_stdc_rotate_right((unsigned char)x, 1073741824);
  result_uc = __builtin_stdc_rotate_left(temp, 0x12345678);
}

// CHECK-LABEL: test_builtin_stdc_rotate_right
// CHECK-DAG: call i8 @llvm.fshr.i8
// CHECK-DAG: call i16 @llvm.fshr.i16
// CHECK-DAG: call i32 @llvm.fshr.i32
// CHECK-DAG: call i{{32|64}} @llvm.fshr.i{{32|64}}
// CHECK-DAG: call i64 @llvm.fshr.i64
// CHECK-DAG: call i128 @llvm.fshr.i128
// CHECK-DAG: call i9 @llvm.fshr.i9
// CHECK-DAG: call i12 @llvm.fshr.i12
// CHECK-DAG: call i20 @llvm.fshr.i20
// CHECK-DAG: call i32 @llvm.fshr.i32
void test_builtin_stdc_rotate_right(unsigned char uc, unsigned short us,
                                    unsigned int ui, unsigned long ul,
                                    unsigned long long ull, unsigned __int128 ui128,
                                    unsigned _BitInt(9) val9, unsigned _BitInt(12) val12,
                                    unsigned _BitInt(20) val20, unsigned _BitInt(32) val32) {

  volatile unsigned char result_uc;
  volatile unsigned short result_us;
  volatile unsigned int result_ui;
  volatile unsigned long result_ul;
  volatile unsigned long long result_ull;
  volatile unsigned __int128 result_ui128;
  volatile unsigned _BitInt(9) result_9;
  volatile unsigned _BitInt(12) result_12;
  volatile unsigned _BitInt(20) result_20;
  volatile unsigned _BitInt(32) result_32;

  result_uc = __builtin_stdc_rotate_right(uc, 3);
  result_us = __builtin_stdc_rotate_right(us, 5);
  result_ui = __builtin_stdc_rotate_right(ui, 8);
  result_ul = __builtin_stdc_rotate_right(ul, 8);
  result_ull = __builtin_stdc_rotate_right(ull, 16);
  result_ui128 = __builtin_stdc_rotate_right(ui128, 32);

  result_uc = __builtin_stdc_rotate_right(uc, -1);
  result_us = __builtin_stdc_rotate_right(us, -3);
  result_ull = __builtin_stdc_rotate_right(ull, -16);

  result_9 = __builtin_stdc_rotate_right(val9, 1);
  result_9 = __builtin_stdc_rotate_right(val9, -1);

  result_us = __builtin_stdc_rotate_right((unsigned short)0x1234, 2147483647);
  result_us = __builtin_stdc_rotate_right((unsigned short)0x1234, -2147483647);
  result_uc = __builtin_stdc_rotate_right((unsigned char)0x80, -2147483648);

  result_us = __builtin_stdc_rotate_right((unsigned short)0xFFFF, -1073741824);
  result_ull = __builtin_stdc_rotate_right(0ULL, -2147483648);
  result_uc = __builtin_stdc_rotate_right((unsigned char)0x80, -1000000007);

  result_12 = __builtin_stdc_rotate_right(val12, 6);
  result_20 = __builtin_stdc_rotate_right(val20, 10);
  result_32 = __builtin_stdc_rotate_right(val32, 16);

  result_12 = __builtin_stdc_rotate_right((unsigned _BitInt(12))0xFFF, -3);
  result_20 = __builtin_stdc_rotate_right((unsigned _BitInt(20))0x12345, 1000000);
  result_32 = __builtin_stdc_rotate_right((unsigned _BitInt(32))0xABCDEF01, -2147483647);

  unsigned short y = 0x1234;
  result_us = __builtin_stdc_rotate_left(__builtin_stdc_rotate_right(y, 2147483647), -2147483647);
}

// Test _BitInt types with various bit widths
// CHECK-LABEL: test_bitint_extremes
// CHECK: call i3 @llvm.fshl.i3
// CHECK: call i23 @llvm.fshl.i23
// CHECK: call i37 @llvm.fshl.i37
// CHECK: call i67 @llvm.fshl.i67
// CHECK: call i127 @llvm.fshl.i127
// CHECK: call i3 @llvm.fshr.i3
// CHECK: call i23 @llvm.fshr.i23
// CHECK: call i37 @llvm.fshr.i37
// CHECK: call i67 @llvm.fshr.i67
// CHECK: call i127 @llvm.fshr.i127
void test_bitint_extremes(unsigned _BitInt(3) val3, unsigned _BitInt(37) val37, int shift) {
  volatile unsigned _BitInt(3) result_3;
  volatile unsigned _BitInt(23) result_23;
  volatile unsigned _BitInt(37) result_37;
  volatile unsigned _BitInt(67) result_67;
  volatile unsigned _BitInt(127) result_127;

  result_3 = __builtin_stdc_rotate_left(val3, shift);
  result_23 = __builtin_stdc_rotate_left((unsigned _BitInt(23))0x123456, shift);
  result_37 = __builtin_stdc_rotate_left(val37, shift);
  result_67 = __builtin_stdc_rotate_left((unsigned _BitInt(67))0x123456789ABCDEFULL, shift);
  result_127 = __builtin_stdc_rotate_left((unsigned _BitInt(127))1, shift);

  result_3 = __builtin_stdc_rotate_right(val3, shift);
  result_23 = __builtin_stdc_rotate_right((unsigned _BitInt(23))0x123456, shift);
  result_37 = __builtin_stdc_rotate_right(val37, shift);
  result_67 = __builtin_stdc_rotate_right((unsigned _BitInt(67))0x123456789ABCDEF0ULL, shift);
  result_127 = __builtin_stdc_rotate_right((unsigned _BitInt(127))1, shift);
}
