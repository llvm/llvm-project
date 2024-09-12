// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void f() {
// CIR-LABEL: @f
// LLVM-LABEL: @f
  unsigned int v[4];
  unsigned int h = __builtin_rotateleft32(v[0], 1);
// CIR: %[[CONST:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[CAST:.*]] = cir.cast(integral, %[[CONST]] : !s32i), !u32i
// CIR: cir.rotate left {{.*}}, %[[CAST]] -> !u32i

// LLVM: %[[SRC:.*]] = load i32, ptr
// LLVM: call i32 @llvm.fshl.i32(i32 %[[SRC]], i32 %[[SRC]], i32 1)
}

unsigned char rotl8(unsigned char x, unsigned char y) {
// CIR-LABEL: rotl8
// CIR: cir.rotate left {{.*}}, {{.*}} -> !u8i

// LLVM-LABEL: rotl8
// LLVM: [[F:%.*]] = call i8 @llvm.fshl.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
  return __builtin_rotateleft8(x, y);
}

short rotl16(short x, short y) {
// CIR-LABEL: rotl16
// CIR: cir.rotate left {{.*}}, {{.*}} -> !u16i

// LLVM-LABEL: rotl16
// LLVM: [[F:%.*]] = call i16 @llvm.fshl.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
  return __builtin_rotateleft16(x, y);
}

int rotl32(int x, unsigned int y) {
// CIR-LABEL: rotl32
// CIR: cir.rotate left {{.*}}, {{.*}} -> !u32i

// LLVM-LABEL: rotl32
// LLVM: [[F:%.*]] = call i32 @llvm.fshl.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
  return __builtin_rotateleft32(x, y);
}

unsigned long long rotl64(unsigned long long x, long long y) {
// CIR-LABEL: rotl64
// CIR: cir.rotate left {{.*}}, {{.*}} -> !u64i

// LLVM-LABEL: rotl64
// LLVM: [[F:%.*]] = call i64 @llvm.fshl.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
  return __builtin_rotateleft64(x, y);
}

char rotr8(char x, char y) {
// CIR-LABEL: rotr8
// CIR: cir.rotate right {{.*}}, {{.*}} -> !u8i

// LLVM-LABEL: rotr8
// LLVM: [[F:%.*]] = call i8 @llvm.fshr.i8(i8 [[X:%.*]], i8 [[X]], i8 [[Y:%.*]])
  return __builtin_rotateright8(x, y);
}

unsigned short rotr16(unsigned short x, unsigned short y) {
// CIR-LABEL: rotr16
// CIR: cir.rotate right {{.*}}, {{.*}} -> !u16i

// LLVM-LABEL: rotr16
// LLVM: [[F:%.*]] = call i16 @llvm.fshr.i16(i16 [[X:%.*]], i16 [[X]], i16 [[Y:%.*]])
  return __builtin_rotateright16(x, y);
}

unsigned int rotr32(unsigned int x, int y) {
// CIR-LABEL: rotr32
// CIR: cir.rotate right {{.*}}, {{.*}} -> !u32i

// LLVM-LABEL: rotr32
// LLVM: [[F:%.*]] = call i32 @llvm.fshr.i32(i32 [[X:%.*]], i32 [[X]], i32 [[Y:%.*]])
  return __builtin_rotateright32(x, y);
}

long long rotr64(long long x, unsigned long long y) {
// CIR-LABEL: rotr64
// CIR: cir.rotate right {{.*}}, {{.*}} -> !u64i

// LLVM-LABEL: rotr64
// LLVM: [[F:%.*]] = call i64 @llvm.fshr.i64(i64 [[X:%.*]], i64 [[X]], i64 [[Y:%.*]])
  return __builtin_rotateright64(x, y);
}