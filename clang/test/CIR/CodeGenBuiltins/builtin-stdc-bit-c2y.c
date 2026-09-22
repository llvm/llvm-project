// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

typedef __SIZE_TYPE__ size_t;

unsigned char stdc_rotate_left_uc(unsigned char, unsigned);
unsigned long long stdc_rotate_right_ull(unsigned long long, unsigned);
unsigned char stdc_memreverse8u8(unsigned char);
unsigned stdc_memreverse8u32(unsigned);
void stdc_memreverse8(size_t, unsigned char *);

unsigned char test_stdc_rotate_left_uc(unsigned char x, unsigned amount) {
  return stdc_rotate_left_uc(x, amount);
}

// CIR-LABEL: test_stdc_rotate_left_uc
// CIR: cir.cast integral {{.*}} : !u32i -> !u8i
// CIR: cir.rotate left {{.*}} : !u8i

// LLVM-LABEL: test_stdc_rotate_left_uc
// LLVM: call i8 @llvm.fshl.i8(

unsigned long long test_stdc_rotate_right_ull(unsigned long long x,
                                              unsigned amount) {
  return stdc_rotate_right_ull(x, amount);
}

// CIR-LABEL: test_stdc_rotate_right_ull
// CIR: cir.cast integral {{.*}} : !u32i -> !u64i
// CIR: cir.rotate right {{.*}} : !u64i

// LLVM-LABEL: test_stdc_rotate_right_ull
// LLVM: call i64 @llvm.fshr.i64(

unsigned char test_stdc_memreverse8u8(unsigned char x) {
  return stdc_memreverse8u8(x);
}

// CIR-LABEL: test_stdc_memreverse8u8
// CIR-NOT: cir.byte_swap
// CIR-NOT: cir.call
// CIR: cir.return

// LLVM-LABEL: test_stdc_memreverse8u8
// LLVM-NOT: @llvm.bswap
// LLVM-NOT: call
// LLVM: ret i8

unsigned test_stdc_memreverse8u32(unsigned x) {
  return stdc_memreverse8u32(x);
}

// CIR-LABEL: test_stdc_memreverse8u32
// CIR: cir.byte_swap {{.*}} : !u32i

// LLVM-LABEL: test_stdc_memreverse8u32
// LLVM: call i32 @llvm.bswap.i32(

void test_stdc_memreverse8_zero(unsigned char *p) {
  stdc_memreverse8(0, p);
}

// CIR-LABEL: test_stdc_memreverse8_zero
// CIR-NOT: cir.byte_swap
// CIR-NOT: cir.call
// CIR: cir.return

// LLVM-LABEL: test_stdc_memreverse8_zero
// LLVM-NOT: @llvm.bswap
// LLVM-NOT: call
// LLVM: ret void

void test_stdc_memreverse8_u32(unsigned char *p) {
  stdc_memreverse8(4, p);
}

// CIR-LABEL: test_stdc_memreverse8_u32
// CIR: cir.byte_swap {{.*}} : !u32i
// CIR: cir.store

// LLVM-LABEL: test_stdc_memreverse8_u32
// LLVM: call i32 @llvm.bswap.i32(
// LLVM: store i32

void test_builtin_stdc_memreverse8_u64(unsigned char *p) {
  __builtin_stdc_memreverse8(8, p);
}

// CIR-LABEL: test_builtin_stdc_memreverse8_u64
// CIR: cir.byte_swap {{.*}} : !u64i
// CIR: cir.store

// LLVM-LABEL: test_builtin_stdc_memreverse8_u64
// LLVM: call i64 @llvm.bswap.i64(
// LLVM: store i64

void test_builtin_stdc_memreverse8_size3(unsigned char *p) {
  __builtin_stdc_memreverse8(3, p);
}

// CIR-LABEL: test_builtin_stdc_memreverse8_size3
// CIR: cir.call @stdc_memreverse8

// LLVM-LABEL: test_builtin_stdc_memreverse8_size3
// LLVM: call void @stdc_memreverse8(

void test_builtin_stdc_memreverse8_dynamic(size_t n, unsigned char *p) {
  __builtin_stdc_memreverse8(n, p);
}

// CIR-LABEL: test_builtin_stdc_memreverse8_dynamic
// CIR: cir.call @stdc_memreverse8

// LLVM-LABEL: test_builtin_stdc_memreverse8_dynamic
// LLVM: call void @stdc_memreverse8(
