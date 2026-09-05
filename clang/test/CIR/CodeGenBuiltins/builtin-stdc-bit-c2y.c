// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -emit-llvm %s -o - | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c2y -fclangir -emit-cir -verify -DNYI_MEMREVERSE8 %s -o -

typedef __SIZE_TYPE__ size_t;

unsigned char stdc_rotate_left_uc(unsigned char, unsigned);
unsigned long long stdc_rotate_right_ull(unsigned long long, unsigned);
unsigned char stdc_memreverse8u8(unsigned char);
unsigned stdc_memreverse8u32(unsigned);
void stdc_memreverse8(size_t, unsigned char *);

#ifndef NYI_MEMREVERSE8
unsigned char test_stdc_rotate_left_uc(unsigned char x, unsigned amount) {
  return stdc_rotate_left_uc(x, amount);
}

// CIR-LABEL: test_stdc_rotate_left_uc
// CIR: cir.cast integral {{.*}} : !u32i -> !u8i
// CIR: cir.rotate left {{.*}} : !u8i

// LLVM-LABEL: test_stdc_rotate_left_uc
// LLVM: call i8 @llvm.fshl.i8(

// OGCG-LABEL: test_stdc_rotate_left_uc
// OGCG: call i8 @llvm.fshl.i8(

unsigned long long test_stdc_rotate_right_ull(unsigned long long x,
                                              unsigned amount) {
  return stdc_rotate_right_ull(x, amount);
}

// CIR-LABEL: test_stdc_rotate_right_ull
// CIR: cir.cast integral {{.*}} : !u32i -> !u64i
// CIR: cir.rotate right {{.*}} : !u64i

// LLVM-LABEL: test_stdc_rotate_right_ull
// LLVM: call i64 @llvm.fshr.i64(

// OGCG-LABEL: test_stdc_rotate_right_ull
// OGCG: call i64 @llvm.fshr.i64(

unsigned char test_stdc_memreverse8u8(unsigned char x) {
  return stdc_memreverse8u8(x);
}

// CIR-LABEL: test_stdc_memreverse8u8
// CIR-NOT: cir.byte_swap

// LLVM-LABEL: test_stdc_memreverse8u8
// LLVM-NOT: @llvm.bswap

// OGCG-LABEL: test_stdc_memreverse8u8
// OGCG-NOT: @llvm.bswap

unsigned test_stdc_memreverse8u32(unsigned x) {
  return stdc_memreverse8u32(x);
}

// CIR-LABEL: test_stdc_memreverse8u32
// CIR: cir.byte_swap {{.*}} : !u32i

// LLVM-LABEL: test_stdc_memreverse8u32
// LLVM: call i32 @llvm.bswap.i32(

// OGCG-LABEL: test_stdc_memreverse8u32
// OGCG: call i32 @llvm.bswap.i32(
#else
void test_stdc_memreverse8(unsigned char *p) {
  stdc_memreverse8(4, p); // expected-error {{ClangIR code gen Not Yet Implemented: unimplemented X86 builtin call: stdc_memreverse8}}
}
#endif
