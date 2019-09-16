// RUN: %clang_cc1 -triple avr-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Check that the parameter types match. This verifies pr43309.
// RUN: %clang_cc1 -triple avr-unknown-unknown -Wconversion -verify %s
// expected-no-diagnostics

unsigned char bitrev8(unsigned char data) {
    return __builtin_bitreverse8(data);
}

// CHECK: define zeroext i8 @bitrev8
// CHECK: i8 @llvm.bitreverse.i8(i8

unsigned int bitrev16(unsigned int data) {
    return __builtin_bitreverse16(data);
}

// CHECK: define i16 @bitrev16
// CHECK: i16 @llvm.bitreverse.i16(i16

unsigned long bitrev32(unsigned long data) {
    return __builtin_bitreverse32(data);
}
// CHECK: define i32 @bitrev32
// CHECK: i32 @llvm.bitreverse.i32(i32

unsigned long long bitrev64(unsigned long long data) {
    return __builtin_bitreverse64(data);
}

// CHECK: define i64 @bitrev64
// CHECK: i64 @llvm.bitreverse.i64(i64

unsigned char rotleft8(unsigned char x, unsigned char y) {
    return __builtin_rotateleft8(x, y);
}

// CHECK: define zeroext i8 @rotleft8
// CHECK: i8 @llvm.fshl.i8(i8

unsigned int rotleft16(unsigned int x, unsigned int y) {
    return __builtin_rotateleft16(x, y);
}

// CHECK: define i16 @rotleft16
// CHECK: i16 @llvm.fshl.i16(i16

unsigned long rotleft32(unsigned long x, unsigned long y) {
    return __builtin_rotateleft32(x, y);
}
// CHECK: define i32 @rotleft32
// CHECK: i32 @llvm.fshl.i32(i32

unsigned long long rotleft64(unsigned long long x, unsigned long long y) {
    return __builtin_rotateleft64(x, y);
}

// CHECK: define i64 @rotleft64
// CHECK: i64 @llvm.fshl.i64(i64

unsigned char rotright8(unsigned char x, unsigned char y) {
    return __builtin_rotateright8(x, y);
}

// CHECK: define zeroext i8 @rotright8
// CHECK: i8 @llvm.fshr.i8(i8

unsigned int rotright16(unsigned int x, unsigned int y) {
    return __builtin_rotateright16(x, y);
}

// CHECK: define i16 @rotright16
// CHECK: i16 @llvm.fshr.i16(i16

unsigned long rotright32(unsigned long x, unsigned long y) {
    return __builtin_rotateright32(x, y);
}
// CHECK: define i32 @rotright32
// CHECK: i32 @llvm.fshr.i32(i32

unsigned long long rotright64(unsigned long long x, unsigned long long y) {
    return __builtin_rotateright64(x, y);
}

// CHECK: define i64 @rotright64
// CHECK: i64 @llvm.fshr.i64(i64

unsigned int byteswap16(unsigned int x) {
    return __builtin_bswap16(x);
}

// CHECK: define i16 @byteswap16
// CHECK: i16 @llvm.bswap.i16(i16

unsigned long byteswap32(unsigned long x) {
    return __builtin_bswap32(x);
}
// CHECK: define i32 @byteswap32
// CHECK: i32 @llvm.bswap.i32(i32

unsigned long long byteswap64(unsigned long long x) {
    return __builtin_bswap64(x);
}

// CHECK: define i64 @byteswap64
// CHECK: i64 @llvm.bswap.i64(i64
