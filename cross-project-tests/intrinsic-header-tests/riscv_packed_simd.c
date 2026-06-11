// REQUIRES: riscv-registered-target
// expected-no-diagnostics

// RUN: %clang %s -O2 -S -o - --target=riscv32 \
// RUN:   -menable-experimental-extensions -march=rv32i_p0p21 \
// RUN:   -Werror -Wextra -Xclang -verify \
// RUN:   | FileCheck %s --check-prefixes=CHECK,RV32
// RUN: %clang %s -O2 -S -o - --target=riscv64 \
// RUN:   -menable-experimental-extensions -march=rv64i_p0p21 \
// RUN:   -Werror -Wextra -Xclang -verify \
// RUN:   | FileCheck %s --check-prefixes=CHECK,RV64

#include <riscv_packed_simd.h>

// CHECK-LABEL: test_pmv_s_u8x4:
// CHECK:       pmv.bs
uint8x4_t test_pmv_s_u8x4(uint8_t x) { return __riscv_pmv_s_u8x4(x); }

// CHECK-LABEL: test_pmv_s_i8x4:
// CHECK:       pmv.bs
int8x4_t test_pmv_s_i8x4(int8_t x) { return __riscv_pmv_s_i8x4(x); }

// CHECK-LABEL: test_pmv_s_u16x2:
// CHECK:       pmv.hs
uint16x2_t test_pmv_s_u16x2(uint16_t x) { return __riscv_pmv_s_u16x2(x); }

// CHECK-LABEL: test_pmv_s_i16x2:
// CHECK:       pmv.hs
int16x2_t test_pmv_s_i16x2(int16_t x) { return __riscv_pmv_s_i16x2(x); }

// TODO: On RV64, the 32-bit packed constant splat emits `lui`+`addi` instead
// of `pli.b`/`pli.h` or `plui.h`.
// CHECK-LABEL: test_pmv_s_u8x4_imm:
// RV32:        pli.b
// RV64:        lui
int8x4_t test_pmv_s_u8x4_imm(void) { return __riscv_pmv_s_u8x4(5); }

// CHECK-LABEL: test_pmv_s_i8x4_imm:
// RV32:        pli.b
// RV64:        lui
int8x4_t test_pmv_s_i8x4_imm(void) { return __riscv_pmv_s_i8x4(-3); }

// CHECK-LABEL: test_pmv_s_u16x2_imm:
// RV32:        pli.h
// RV64:        lui
uint16x2_t test_pmv_s_u16x2_imm(void) { return __riscv_pmv_s_u16x2(42); }

// CHECK-LABEL: test_pmv_s_i16x2_imm:
// RV32:        pli.h
// RV64:        lui
int16x2_t test_pmv_s_i16x2_imm(void) { return __riscv_pmv_s_i16x2(-5); }

// CHECK-LABEL: test_pmv_s_u16x2_imm_hi:
// RV32:        plui.h
// RV64:        lui
uint16x2_t test_pmv_s_u16x2_imm_hi(void) { return __riscv_pmv_s_u16x2(0x3600); }

// CHECK-LABEL: test_pmv_s_i16x2_imm_hi:
// RV32:        plui.h
// RV64:        lui
int16x2_t test_pmv_s_i16x2_imm_hi(void) { return __riscv_pmv_s_i16x2(0x3600); }

// CHECK-LABEL: test_pmv_s_u8x8:
// RV32:        pmv.dbs
// RV64:        pmv.bs
uint8x8_t test_pmv_s_u8x8(uint8_t x) { return __riscv_pmv_s_u8x8(x); }

// CHECK-LABEL: test_pmv_s_i8x8:
// RV32:        pmv.dbs
// RV64:        pmv.bs
int8x8_t test_pmv_s_i8x8(int8_t x) { return __riscv_pmv_s_i8x8(x); }

// CHECK-LABEL: test_pmv_s_u16x4:
// RV32:        pmv.dhs
// RV64:        pmv.hs
uint16x4_t test_pmv_s_u16x4(uint16_t x) { return __riscv_pmv_s_u16x4(x); }

// CHECK-LABEL: test_pmv_s_i16x4:
// RV32:        pmv.dhs
// RV64:        pmv.hs
int16x4_t test_pmv_s_i16x4(int16_t x) { return __riscv_pmv_s_i16x4(x); }

// TODO: On RV32, the 32x2 variable splat emits a plain `mv` instead of
// `padd.dws` with rs1_p=x0.
// CHECK-LABEL: test_pmv_s_u32x2:
// RV32:        mv{{[[:space:]]}}
// RV64:        pmv.ws
uint32x2_t test_pmv_s_u32x2(uint32_t x) { return __riscv_pmv_s_u32x2(x); }

// CHECK-LABEL: test_pmv_s_i32x2:
// RV32:        mv{{[[:space:]]}}
// RV64:        pmv.ws
int32x2_t test_pmv_s_i32x2(int32_t x) { return __riscv_pmv_s_i32x2(x); }

// CHECK-LABEL: test_pmv_s_u8x8_imm:
// RV32:        pli.db
// RV64:        pli.b
uint8x8_t test_pmv_s_u8x8_imm(void) { return __riscv_pmv_s_u8x8(5); }

// CHECK-LABEL: test_pmv_s_i8x8_imm:
// RV32:        pli.db
// RV64:        pli.b
int8x8_t test_pmv_s_i8x8_imm(void) { return __riscv_pmv_s_i8x8(-3); }

// CHECK-LABEL: test_pmv_s_u16x4_imm:
// RV32:        pli.dh
// RV64:        pli.h
uint16x4_t test_pmv_s_u16x4_imm(void) { return __riscv_pmv_s_u16x4(42); }

// CHECK-LABEL: test_pmv_s_i16x4_imm:
// RV32:        pli.dh
// RV64:        pli.h
int16x4_t test_pmv_s_i16x4_imm(void) { return __riscv_pmv_s_i16x4(-5); }

// CHECK-LABEL: test_pmv_s_u16x4_imm_hi:
// RV32:        plui.dh
// RV64:        plui.h
uint16x4_t test_pmv_s_u16x4_imm_hi(void) { return __riscv_pmv_s_u16x4(0x3600); }

// CHECK-LABEL: test_pmv_s_i16x4_imm_hi:
// RV32:        plui.dh
// RV64:        plui.h
int16x4_t test_pmv_s_i16x4_imm_hi(void) { return __riscv_pmv_s_i16x4(0x3600); }

// Note: Constants that fit `addi`'s 12-bit immediate fold to 2x `li`.
// Larger constants follow `lui`+`addi`+`mv`; see `_imm_big` below.
// CHECK-LABEL: test_pmv_s_u32x2_imm:
// RV32-COUNT-2: li{{[[:space:]]}}
// RV64:         pli.w
uint32x2_t test_pmv_s_u32x2_imm(void) { return __riscv_pmv_s_u32x2(42); }

// CHECK-LABEL: test_pmv_s_i32x2_imm:
// RV32-COUNT-2: li{{[[:space:]]}}
// RV64:         pli.w
int32x2_t test_pmv_s_i32x2_imm(void) { return __riscv_pmv_s_i32x2(-5); }

// CHECK-LABEL: test_pmv_s_u32x2_imm_big:
// RV32:        lui
// RV32-NEXT:   addi
// RV32-NEXT:   mv{{[[:space:]]}}
// RV32-NEXT:   ret
uint32x2_t test_pmv_s_u32x2_imm_big(void) {
  return __riscv_pmv_s_u32x2(0x12345);
}

// CHECK-LABEL: test_pmv_s_i32x2_imm_big:
// RV32:        lui
// RV32-NEXT:   addi
// RV32-NEXT:   mv{{[[:space:]]}}
// RV32-NEXT:   ret
int32x2_t test_pmv_s_i32x2_imm_big(void) {
  return __riscv_pmv_s_i32x2(0x12345);
}

// CHECK-LABEL: test_padd_i8x4:
// CHECK:       padd.b
int8x4_t test_padd_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_padd_i8x4(a, b);
}

// CHECK-LABEL: test_padd_u8x4:
// CHECK:       padd.b
uint8x4_t test_padd_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_padd_u8x4(a, b);
}

// CHECK-LABEL: test_padd_i16x2:
// CHECK:       padd.h
int16x2_t test_padd_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_padd_i16x2(a, b);
}

// CHECK-LABEL: test_padd_u16x2:
// CHECK:       padd.h
uint16x2_t test_padd_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_padd_u16x2(a, b);
}

// CHECK-LABEL: test_psub_i8x4:
// CHECK:       psub.b
int8x4_t test_psub_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_psub_i8x4(a, b);
}

// CHECK-LABEL: test_psub_u8x4:
// CHECK:       psub.b
uint8x4_t test_psub_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_psub_u8x4(a, b);
}

// CHECK-LABEL: test_psub_i16x2:
// CHECK:       psub.h
int16x2_t test_psub_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_psub_i16x2(a, b);
}

// CHECK-LABEL: test_psub_u16x2:
// CHECK:       psub.h
uint16x2_t test_psub_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_psub_u16x2(a, b);
}

// CHECK-LABEL: test_pneg_i8x4:
// CHECK:       pneg.b
int8x4_t test_pneg_i8x4(int8x4_t a) { return __riscv_pneg_i8x4(a); }

// CHECK-LABEL: test_pneg_i16x2:
// CHECK:       pneg.h
int16x2_t test_pneg_i16x2(int16x2_t a) { return __riscv_pneg_i16x2(a); }

// CHECK-LABEL: test_padd_i8x8:
// RV32:        padd.db
// RV64:        padd.b
int8x8_t test_padd_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_padd_i8x8(a, b);
}

// CHECK-LABEL: test_padd_u8x8:
// RV32:        padd.db
// RV64:        padd.b
uint8x8_t test_padd_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_padd_u8x8(a, b);
}

// CHECK-LABEL: test_padd_i16x4:
// RV32:        padd.dh
// RV64:        padd.h
int16x4_t test_padd_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_padd_i16x4(a, b);
}

// CHECK-LABEL: test_padd_u16x4:
// RV32:        padd.dh
// RV64:        padd.h
uint16x4_t test_padd_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_padd_u16x4(a, b);
}

// CHECK-LABEL: test_padd_i32x2:
// RV32:        padd.dw
// RV64:        padd.w
int32x2_t test_padd_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_padd_i32x2(a, b);
}

// CHECK-LABEL: test_padd_u32x2:
// RV32:        padd.dw
// RV64:        padd.w
uint32x2_t test_padd_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_padd_u32x2(a, b);
}

// CHECK-LABEL: test_psub_i8x8:
// RV32:        psub.db
// RV64:        psub.b
int8x8_t test_psub_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_psub_i8x8(a, b);
}

// CHECK-LABEL: test_psub_u8x8:
// RV32:        psub.db
// RV64:        psub.b
uint8x8_t test_psub_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_psub_u8x8(a, b);
}

// CHECK-LABEL: test_psub_i16x4:
// RV32:        psub.dh
// RV64:        psub.h
int16x4_t test_psub_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_psub_i16x4(a, b);
}

// CHECK-LABEL: test_psub_u16x4:
// RV32:        psub.dh
// RV64:        psub.h
uint16x4_t test_psub_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_psub_u16x4(a, b);
}

// CHECK-LABEL: test_psub_i32x2:
// RV32:        psub.dw
// RV64:        psub.w
int32x2_t test_psub_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_psub_i32x2(a, b);
}

// CHECK-LABEL: test_psub_u32x2:
// RV32:        psub.dw
// RV64:        psub.w
uint32x2_t test_psub_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_psub_u32x2(a, b);
}

// CHECK-LABEL: test_pneg_i8x8:
// RV32:        pneg.db
// RV64:        pneg.b
int8x8_t test_pneg_i8x8(int8x8_t a) { return __riscv_pneg_i8x8(a); }

// CHECK-LABEL: test_pneg_i16x4:
// RV32:        pneg.dh
// RV64:        pneg.h
int16x4_t test_pneg_i16x4(int16x4_t a) { return __riscv_pneg_i16x4(a); }

// CHECK-LABEL: test_pneg_i32x2:
// RV32:        pneg.dw
// RV64:        pneg.w
int32x2_t test_pneg_i32x2(int32x2_t a) { return __riscv_pneg_i32x2(a); }

// CHECK-LABEL: test_padd_s_u8x4:
// CHECK:       padd.bs
uint8x4_t test_padd_s_u8x4(uint8x4_t a, uint8_t b) {
  return __riscv_padd_s_u8x4(a, b);
}

// CHECK-LABEL: test_padd_s_i8x4:
// CHECK:       padd.bs
int8x4_t test_padd_s_i8x4(int8x4_t a, int8_t b) {
  return __riscv_padd_s_i8x4(a, b);
}

// CHECK-LABEL: test_padd_s_u16x2:
// CHECK:       padd.hs
uint16x2_t test_padd_s_u16x2(uint16x2_t a, uint16_t b) {
  return __riscv_padd_s_u16x2(a, b);
}

// CHECK-LABEL: test_padd_s_i16x2:
// CHECK:       padd.hs
int16x2_t test_padd_s_i16x2(int16x2_t a, int16_t b) {
  return __riscv_padd_s_i16x2(a, b);
}

// CHECK-LABEL: test_padd_s_u8x8:
// RV32:        padd.dbs
// RV64:        padd.bs
uint8x8_t test_padd_s_u8x8(uint8x8_t a, uint8_t b) {
  return __riscv_padd_s_u8x8(a, b);
}

// CHECK-LABEL: test_padd_s_i8x8:
// RV32:        padd.dbs
// RV64:        padd.bs
int8x8_t test_padd_s_i8x8(int8x8_t a, int8_t b) {
  return __riscv_padd_s_i8x8(a, b);
}

// CHECK-LABEL: test_padd_s_u16x4:
// RV32:        padd.dhs
// RV64:        padd.hs
uint16x4_t test_padd_s_u16x4(uint16x4_t a, uint16_t b) {
  return __riscv_padd_s_u16x4(a, b);
}

// CHECK-LABEL: test_padd_s_i16x4:
// RV32:        padd.dhs
// RV64:        padd.hs
int16x4_t test_padd_s_i16x4(int16x4_t a, int16_t b) {
  return __riscv_padd_s_i16x4(a, b);
}

// CHECK-LABEL: test_padd_s_u32x2:
// RV32:        padd.dws
// RV64:        padd.ws
uint32x2_t test_padd_s_u32x2(uint32x2_t a, uint32_t b) {
  return __riscv_padd_s_u32x2(a, b);
}

// CHECK-LABEL: test_padd_s_i32x2:
// RV32:        padd.dws
// RV64:        padd.ws
int32x2_t test_padd_s_i32x2(int32x2_t a, int32_t b) {
  return __riscv_padd_s_i32x2(a, b);
}

// CHECK-LABEL: test_psadd_i8x4:
// CHECK:       psadd.b
int8x4_t test_psadd_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_psadd_i8x4(a, b);
}

// CHECK-LABEL: test_psadd_i16x2:
// CHECK:       psadd.h
int16x2_t test_psadd_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_psadd_i16x2(a, b);
}

// CHECK-LABEL: test_psaddu_u8x4:
// CHECK:       psaddu.b
uint8x4_t test_psaddu_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_psaddu_u8x4(a, b);
}

// CHECK-LABEL: test_psaddu_u16x2:
// CHECK:       psaddu.h
uint16x2_t test_psaddu_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_psaddu_u16x2(a, b);
}

// CHECK-LABEL: test_pssub_i8x4:
// CHECK:       pssub.b
int8x4_t test_pssub_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_pssub_i8x4(a, b);
}

// CHECK-LABEL: test_pssub_i16x2:
// CHECK:       pssub.h
int16x2_t test_pssub_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_pssub_i16x2(a, b);
}

// CHECK-LABEL: test_pssubu_u8x4:
// CHECK:       pssubu.b
uint8x4_t test_pssubu_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_pssubu_u8x4(a, b);
}

// CHECK-LABEL: test_pssubu_u16x2:
// CHECK:       pssubu.h
uint16x2_t test_pssubu_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_pssubu_u16x2(a, b);
}

// CHECK-LABEL: test_psadd_i8x8:
// RV32:        psadd.db
// RV64:        psadd.b
int8x8_t test_psadd_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_psadd_i8x8(a, b);
}

// CHECK-LABEL: test_psadd_i16x4:
// RV32:        psadd.dh
// RV64:        psadd.h
int16x4_t test_psadd_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_psadd_i16x4(a, b);
}

// CHECK-LABEL: test_psadd_i32x2:
// RV32:        psadd.dw
// RV64:        psadd.w
int32x2_t test_psadd_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_psadd_i32x2(a, b);
}

// CHECK-LABEL: test_psaddu_u8x8:
// RV32:        psaddu.db
// RV64:        psaddu.b
uint8x8_t test_psaddu_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_psaddu_u8x8(a, b);
}

// CHECK-LABEL: test_psaddu_u16x4:
// RV32:        psaddu.dh
// RV64:        psaddu.h
uint16x4_t test_psaddu_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_psaddu_u16x4(a, b);
}

// CHECK-LABEL: test_psaddu_u32x2:
// RV32:        psaddu.dw
// RV64:        psaddu.w
uint32x2_t test_psaddu_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_psaddu_u32x2(a, b);
}

// CHECK-LABEL: test_pssub_i8x8:
// RV32:        pssub.db
// RV64:        pssub.b
int8x8_t test_pssub_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_pssub_i8x8(a, b);
}

// CHECK-LABEL: test_pssub_i16x4:
// RV32:        pssub.dh
// RV64:        pssub.h
int16x4_t test_pssub_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_pssub_i16x4(a, b);
}

// CHECK-LABEL: test_pssub_i32x2:
// RV32:        pssub.dw
// RV64:        pssub.w
int32x2_t test_pssub_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_pssub_i32x2(a, b);
}

// CHECK-LABEL: test_pssubu_u8x8:
// RV32:        pssubu.db
// RV64:        pssubu.b
uint8x8_t test_pssubu_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_pssubu_u8x8(a, b);
}

// CHECK-LABEL: test_pssubu_u16x4:
// RV32:        pssubu.dh
// RV64:        pssubu.h
uint16x4_t test_pssubu_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_pssubu_u16x4(a, b);
}

// CHECK-LABEL: test_pssubu_u32x2:
// RV32:        pssubu.dw
// RV64:        pssubu.w
uint32x2_t test_pssubu_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_pssubu_u32x2(a, b);
}

// CHECK-LABEL: test_psh1add_i16x2:
// CHECK:       psh1add.h
int16x2_t test_psh1add_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_psh1add_i16x2(a, b);
}

// CHECK-LABEL: test_psh1add_u16x2:
// CHECK:       psh1add.h
uint16x2_t test_psh1add_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_psh1add_u16x2(a, b);
}

// CHECK-LABEL: test_pssh1sadd_i16x2:
// CHECK:       pssh1sadd.h
int16x2_t test_pssh1sadd_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_pssh1sadd_i16x2(a, b);
}

// CHECK-LABEL: test_psh1add_i16x4:
// RV32:        psh1add.dh
// RV64:        psh1add.h
int16x4_t test_psh1add_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_psh1add_i16x4(a, b);
}

// CHECK-LABEL: test_psh1add_u16x4:
// RV32:        psh1add.dh
// RV64:        psh1add.h
uint16x4_t test_psh1add_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_psh1add_u16x4(a, b);
}

// CHECK-LABEL: test_psh1add_i32x2:
// RV32:        psh1add.dw
// RV64:        psh1add.w
int32x2_t test_psh1add_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_psh1add_i32x2(a, b);
}

// CHECK-LABEL: test_psh1add_u32x2:
// RV32:        psh1add.dw
// RV64:        psh1add.w
uint32x2_t test_psh1add_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_psh1add_u32x2(a, b);
}

// CHECK-LABEL: test_pssh1sadd_i16x4:
// RV32:        pssh1sadd.dh
// RV64:        pssh1sadd.h
int16x4_t test_pssh1sadd_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_pssh1sadd_i16x4(a, b);
}

// CHECK-LABEL: test_pssh1sadd_i32x2:
// RV32:        pssh1sadd.dw
// RV64:        pssh1sadd.w
int32x2_t test_pssh1sadd_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_pssh1sadd_i32x2(a, b);
}

// CHECK-LABEL: test_pmin_i8x4:
// CHECK:       pmin.b
int8x4_t test_pmin_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_pmin_i8x4(a, b);
}

// CHECK-LABEL: test_pmin_i16x2:
// CHECK:       pmin.h
int16x2_t test_pmin_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_pmin_i16x2(a, b);
}

// CHECK-LABEL: test_pminu_u8x4:
// CHECK:       pminu.b
uint8x4_t test_pminu_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_pminu_u8x4(a, b);
}

// CHECK-LABEL: test_pminu_u16x2:
// CHECK:       pminu.h
uint16x2_t test_pminu_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_pminu_u16x2(a, b);
}

// CHECK-LABEL: test_pmax_i8x4:
// CHECK:       pmax.b
int8x4_t test_pmax_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_pmax_i8x4(a, b);
}

// CHECK-LABEL: test_pmax_i16x2:
// CHECK:       pmax.h
int16x2_t test_pmax_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_pmax_i16x2(a, b);
}

// CHECK-LABEL: test_pmaxu_u8x4:
// CHECK:       pmaxu.b
uint8x4_t test_pmaxu_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_pmaxu_u8x4(a, b);
}

// CHECK-LABEL: test_pmaxu_u16x2:
// CHECK:       pmaxu.h
uint16x2_t test_pmaxu_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_pmaxu_u16x2(a, b);
}

// CHECK-LABEL: test_pmin_i8x8:
// RV32:        pmin.db
// RV64:        pmin.b
int8x8_t test_pmin_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_pmin_i8x8(a, b);
}

// CHECK-LABEL: test_pmin_i16x4:
// RV32:        pmin.dh
// RV64:        pmin.h
int16x4_t test_pmin_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_pmin_i16x4(a, b);
}

// CHECK-LABEL: test_pmin_i32x2:
// RV32:        pmin.dw
// RV64:        pmin.w
int32x2_t test_pmin_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_pmin_i32x2(a, b);
}

// CHECK-LABEL: test_pminu_u8x8:
// RV32:        pminu.db
// RV64:        pminu.b
uint8x8_t test_pminu_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_pminu_u8x8(a, b);
}

// CHECK-LABEL: test_pminu_u16x4:
// RV32:        pminu.dh
// RV64:        pminu.h
uint16x4_t test_pminu_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_pminu_u16x4(a, b);
}

// CHECK-LABEL: test_pminu_u32x2:
// RV32:        pminu.dw
// RV64:        pminu.w
uint32x2_t test_pminu_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_pminu_u32x2(a, b);
}

// CHECK-LABEL: test_pmax_i8x8:
// RV32:        pmax.db
// RV64:        pmax.b
int8x8_t test_pmax_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_pmax_i8x8(a, b);
}

// CHECK-LABEL: test_pmax_i16x4:
// RV32:        pmax.dh
// RV64:        pmax.h
int16x4_t test_pmax_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_pmax_i16x4(a, b);
}

// CHECK-LABEL: test_pmax_i32x2:
// RV32:        pmax.dw
// RV64:        pmax.w
int32x2_t test_pmax_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_pmax_i32x2(a, b);
}

// CHECK-LABEL: test_pmaxu_u8x8:
// RV32:        pmaxu.db
// RV64:        pmaxu.b
uint8x8_t test_pmaxu_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_pmaxu_u8x8(a, b);
}

// CHECK-LABEL: test_pmaxu_u16x4:
// RV32:        pmaxu.dh
// RV64:        pmaxu.h
uint16x4_t test_pmaxu_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_pmaxu_u16x4(a, b);
}

// CHECK-LABEL: test_pmaxu_u32x2:
// RV32:        pmaxu.dw
// RV64:        pmaxu.w
uint32x2_t test_pmaxu_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_pmaxu_u32x2(a, b);
}

// CHECK-LABEL: test_psll_s_u8x4:
// CHECK:       psll.bs
uint8x4_t test_psll_s_u8x4(uint8x4_t a, unsigned n) {
  return __riscv_psll_s_u8x4(a, n);
}

// CHECK-LABEL: test_psll_s_i8x4:
// CHECK:       psll.bs
int8x4_t test_psll_s_i8x4(int8x4_t a, unsigned n) {
  return __riscv_psll_s_i8x4(a, n);
}

// CHECK-LABEL: test_psll_s_u16x2:
// CHECK:       psll.hs
uint16x2_t test_psll_s_u16x2(uint16x2_t a, unsigned n) {
  return __riscv_psll_s_u16x2(a, n);
}

// CHECK-LABEL: test_psll_s_i16x2:
// CHECK:       psll.hs
int16x2_t test_psll_s_i16x2(int16x2_t a, unsigned n) {
  return __riscv_psll_s_i16x2(a, n);
}

// CHECK-LABEL: test_psrl_s_u8x4:
// CHECK:       psrl.bs
uint8x4_t test_psrl_s_u8x4(uint8x4_t a, unsigned n) {
  return __riscv_psrl_s_u8x4(a, n);
}

// CHECK-LABEL: test_psrl_s_u16x2:
// CHECK:       psrl.hs
uint16x2_t test_psrl_s_u16x2(uint16x2_t a, unsigned n) {
  return __riscv_psrl_s_u16x2(a, n);
}

// CHECK-LABEL: test_psra_s_i8x4:
// CHECK:       psra.bs
int8x4_t test_psra_s_i8x4(int8x4_t a, unsigned n) {
  return __riscv_psra_s_i8x4(a, n);
}

// CHECK-LABEL: test_psra_s_i16x2:
// CHECK:       psra.hs
int16x2_t test_psra_s_i16x2(int16x2_t a, unsigned n) {
  return __riscv_psra_s_i16x2(a, n);
}

// CHECK-LABEL: test_psll_s_u8x4_imm:
// CHECK:       pslli.b{{[[:space:]]+}}{{.*}}, 2
uint8x4_t test_psll_s_u8x4_imm(uint8x4_t a) {
  return __riscv_psll_s_u8x4(a, 2);
}

// CHECK-LABEL: test_psll_s_i8x4_imm:
// CHECK:       pslli.b{{[[:space:]]+}}{{.*}}, 3
int8x4_t test_psll_s_i8x4_imm(int8x4_t a) { return __riscv_psll_s_i8x4(a, 3); }

// CHECK-LABEL: test_psll_s_u16x2_imm:
// CHECK:       pslli.h{{[[:space:]]+}}{{.*}}, 5
uint16x2_t test_psll_s_u16x2_imm(uint16x2_t a) {
  return __riscv_psll_s_u16x2(a, 5);
}

// CHECK-LABEL: test_psll_s_i16x2_imm:
// CHECK:       pslli.h{{[[:space:]]+}}{{.*}}, 7
int16x2_t test_psll_s_i16x2_imm(int16x2_t a) {
  return __riscv_psll_s_i16x2(a, 7);
}

// CHECK-LABEL: test_psrl_s_u8x4_imm:
// CHECK:       psrli.b{{[[:space:]]+}}{{.*}}, 2
uint8x4_t test_psrl_s_u8x4_imm(uint8x4_t a) {
  return __riscv_psrl_s_u8x4(a, 2);
}

// CHECK-LABEL: test_psrl_s_u16x2_imm:
// CHECK:       psrli.h{{[[:space:]]+}}{{.*}}, 3
uint16x2_t test_psrl_s_u16x2_imm(uint16x2_t a) {
  return __riscv_psrl_s_u16x2(a, 3);
}

// CHECK-LABEL: test_psra_s_i8x4_imm:
// CHECK:       psrai.b{{[[:space:]]+}}{{.*}}, 4
int8x4_t test_psra_s_i8x4_imm(int8x4_t a) { return __riscv_psra_s_i8x4(a, 4); }

// CHECK-LABEL: test_psra_s_i16x2_imm:
// CHECK:       psrai.h{{[[:space:]]+}}{{.*}}, 5
int16x2_t test_psra_s_i16x2_imm(int16x2_t a) {
  return __riscv_psra_s_i16x2(a, 5);
}

// CHECK-LABEL: test_psll_s_u8x8:
// RV32:        psll.dbs
// RV64:        psll.bs
uint8x8_t test_psll_s_u8x8(uint8x8_t a, unsigned n) {
  return __riscv_psll_s_u8x8(a, n);
}

// CHECK-LABEL: test_psll_s_i8x8:
// RV32:        psll.dbs
// RV64:        psll.bs
int8x8_t test_psll_s_i8x8(int8x8_t a, unsigned n) {
  return __riscv_psll_s_i8x8(a, n);
}

// CHECK-LABEL: test_psll_s_u16x4:
// RV32:        psll.dhs
// RV64:        psll.hs
uint16x4_t test_psll_s_u16x4(uint16x4_t a, unsigned n) {
  return __riscv_psll_s_u16x4(a, n);
}

// CHECK-LABEL: test_psll_s_i16x4:
// RV32:        psll.dhs
// RV64:        psll.hs
int16x4_t test_psll_s_i16x4(int16x4_t a, unsigned n) {
  return __riscv_psll_s_i16x4(a, n);
}

// CHECK-LABEL: test_psll_s_u32x2:
// RV32:        psll.dws
// RV64:        psll.ws
uint32x2_t test_psll_s_u32x2(uint32x2_t a, unsigned n) {
  return __riscv_psll_s_u32x2(a, n);
}

// CHECK-LABEL: test_psll_s_i32x2:
// RV32:        psll.dws
// RV64:        psll.ws
int32x2_t test_psll_s_i32x2(int32x2_t a, unsigned n) {
  return __riscv_psll_s_i32x2(a, n);
}

// CHECK-LABEL: test_psrl_s_u8x8:
// RV32:        psrl.dbs
// RV64:        psrl.bs
uint8x8_t test_psrl_s_u8x8(uint8x8_t a, unsigned n) {
  return __riscv_psrl_s_u8x8(a, n);
}

// CHECK-LABEL: test_psrl_s_u16x4:
// RV32:        psrl.dhs
// RV64:        psrl.hs
uint16x4_t test_psrl_s_u16x4(uint16x4_t a, unsigned n) {
  return __riscv_psrl_s_u16x4(a, n);
}

// CHECK-LABEL: test_psrl_s_u32x2:
// RV32:        psrl.dws
// RV64:        psrl.ws
uint32x2_t test_psrl_s_u32x2(uint32x2_t a, unsigned n) {
  return __riscv_psrl_s_u32x2(a, n);
}

// CHECK-LABEL: test_psra_s_i8x8:
// RV32:        psra.dbs
// RV64:        psra.bs
int8x8_t test_psra_s_i8x8(int8x8_t a, unsigned n) {
  return __riscv_psra_s_i8x8(a, n);
}

// CHECK-LABEL: test_psra_s_i16x4:
// RV32:        psra.dhs
// RV64:        psra.hs
int16x4_t test_psra_s_i16x4(int16x4_t a, unsigned n) {
  return __riscv_psra_s_i16x4(a, n);
}

// CHECK-LABEL: test_psra_s_i32x2:
// RV32:        psra.dws
// RV64:        psra.ws
int32x2_t test_psra_s_i32x2(int32x2_t a, unsigned n) {
  return __riscv_psra_s_i32x2(a, n);
}

// CHECK-LABEL: test_psll_s_u8x8_imm:
// RV32:        pslli.db{{[[:space:]]+}}{{.*}}, 2
// RV64:        pslli.b{{[[:space:]]+}}{{.*}}, 2
uint8x8_t test_psll_s_u8x8_imm(uint8x8_t a) {
  return __riscv_psll_s_u8x8(a, 2);
}

// CHECK-LABEL: test_psll_s_i8x8_imm:
// RV32:        pslli.db{{[[:space:]]+}}{{.*}}, 3
// RV64:        pslli.b{{[[:space:]]+}}{{.*}}, 3
int8x8_t test_psll_s_i8x8_imm(int8x8_t a) { return __riscv_psll_s_i8x8(a, 3); }

// CHECK-LABEL: test_psll_s_u16x4_imm:
// RV32:        pslli.dh{{[[:space:]]+}}{{.*}}, 4
// RV64:        pslli.h{{[[:space:]]+}}{{.*}}, 4
uint16x4_t test_psll_s_u16x4_imm(uint16x4_t a) {
  return __riscv_psll_s_u16x4(a, 4);
}

// CHECK-LABEL: test_psll_s_i16x4_imm:
// RV32:        pslli.dh{{[[:space:]]+}}{{.*}}, 5
// RV64:        pslli.h{{[[:space:]]+}}{{.*}}, 5
int16x4_t test_psll_s_i16x4_imm(int16x4_t a) {
  return __riscv_psll_s_i16x4(a, 5);
}

// CHECK-LABEL: test_psll_s_u32x2_imm:
// RV32:        pslli.dw{{[[:space:]]+}}{{.*}}, 7
// RV64:        pslli.w{{[[:space:]]+}}{{.*}}, 7
uint32x2_t test_psll_s_u32x2_imm(uint32x2_t a) {
  return __riscv_psll_s_u32x2(a, 7);
}

// CHECK-LABEL: test_psll_s_i32x2_imm:
// RV32:        pslli.dw{{[[:space:]]+}}{{.*}}, 9
// RV64:        pslli.w{{[[:space:]]+}}{{.*}}, 9
int32x2_t test_psll_s_i32x2_imm(int32x2_t a) {
  return __riscv_psll_s_i32x2(a, 9);
}

// CHECK-LABEL: test_psrl_s_u8x8_imm:
// RV32:        psrli.db{{[[:space:]]+}}{{.*}}, 2
// RV64:        psrli.b{{[[:space:]]+}}{{.*}}, 2
uint8x8_t test_psrl_s_u8x8_imm(uint8x8_t a) {
  return __riscv_psrl_s_u8x8(a, 2);
}

// CHECK-LABEL: test_psrl_s_u16x4_imm:
// RV32:        psrli.dh{{[[:space:]]+}}{{.*}}, 3
// RV64:        psrli.h{{[[:space:]]+}}{{.*}}, 3
uint16x4_t test_psrl_s_u16x4_imm(uint16x4_t a) {
  return __riscv_psrl_s_u16x4(a, 3);
}

// CHECK-LABEL: test_psrl_s_u32x2_imm:
// RV32:        psrli.dw{{[[:space:]]+}}{{.*}}, 5
// RV64:        psrli.w{{[[:space:]]+}}{{.*}}, 5
uint32x2_t test_psrl_s_u32x2_imm(uint32x2_t a) {
  return __riscv_psrl_s_u32x2(a, 5);
}

// CHECK-LABEL: test_psra_s_i8x8_imm:
// RV32:        psrai.db{{[[:space:]]+}}{{.*}}, 4
// RV64:        psrai.b{{[[:space:]]+}}{{.*}}, 4
int8x8_t test_psra_s_i8x8_imm(int8x8_t a) { return __riscv_psra_s_i8x8(a, 4); }

// CHECK-LABEL: test_psra_s_i16x4_imm:
// RV32:        psrai.dh{{[[:space:]]+}}{{.*}}, 5
// RV64:        psrai.h{{[[:space:]]+}}{{.*}}, 5
int16x4_t test_psra_s_i16x4_imm(int16x4_t a) {
  return __riscv_psra_s_i16x4(a, 5);
}

// CHECK-LABEL: test_psra_s_i32x2_imm:
// RV32:        psrai.dw{{[[:space:]]+}}{{.*}}, 11
// RV64:        psrai.w{{[[:space:]]+}}{{.*}}, 11
int32x2_t test_psra_s_i32x2_imm(int32x2_t a) {
  return __riscv_psra_s_i32x2(a, 11);
}

// CHECK-LABEL: test_pand_i8x4:
// CHECK:       and{{[[:space:]]}}
int8x4_t test_pand_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_pand_i8x4(a, b);
}

// CHECK-LABEL: test_pand_u8x4:
// CHECK:       and{{[[:space:]]}}
uint8x4_t test_pand_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_pand_u8x4(a, b);
}

// CHECK-LABEL: test_pand_i16x2:
// CHECK:       and{{[[:space:]]}}
int16x2_t test_pand_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_pand_i16x2(a, b);
}

// CHECK-LABEL: test_pand_u16x2:
// CHECK:       and{{[[:space:]]}}
uint16x2_t test_pand_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_pand_u16x2(a, b);
}

// CHECK-LABEL: test_por_i8x4:
// CHECK:       or{{[[:space:]]}}
int8x4_t test_por_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_por_i8x4(a, b);
}

// CHECK-LABEL: test_por_u8x4:
// CHECK:       or{{[[:space:]]}}
uint8x4_t test_por_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_por_u8x4(a, b);
}

// CHECK-LABEL: test_por_i16x2:
// CHECK:       or{{[[:space:]]}}
int16x2_t test_por_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_por_i16x2(a, b);
}

// CHECK-LABEL: test_por_u16x2:
// CHECK:       or{{[[:space:]]}}
uint16x2_t test_por_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_por_u16x2(a, b);
}

// CHECK-LABEL: test_pxor_i8x4:
// CHECK:       xor{{[[:space:]]}}
int8x4_t test_pxor_i8x4(int8x4_t a, int8x4_t b) {
  return __riscv_pxor_i8x4(a, b);
}

// CHECK-LABEL: test_pxor_u8x4:
// CHECK:       xor{{[[:space:]]}}
uint8x4_t test_pxor_u8x4(uint8x4_t a, uint8x4_t b) {
  return __riscv_pxor_u8x4(a, b);
}

// CHECK-LABEL: test_pxor_i16x2:
// CHECK:       xor{{[[:space:]]}}
int16x2_t test_pxor_i16x2(int16x2_t a, int16x2_t b) {
  return __riscv_pxor_i16x2(a, b);
}

// CHECK-LABEL: test_pxor_u16x2:
// CHECK:       xor{{[[:space:]]}}
uint16x2_t test_pxor_u16x2(uint16x2_t a, uint16x2_t b) {
  return __riscv_pxor_u16x2(a, b);
}

// CHECK-LABEL: test_pnot_i8x4:
// CHECK:       not{{[[:space:]]}}
int8x4_t test_pnot_i8x4(int8x4_t a) { return __riscv_pnot_i8x4(a); }

// CHECK-LABEL: test_pnot_u8x4:
// CHECK:       not{{[[:space:]]}}
uint8x4_t test_pnot_u8x4(uint8x4_t a) { return __riscv_pnot_u8x4(a); }

// CHECK-LABEL: test_pnot_i16x2:
// CHECK:       not{{[[:space:]]}}
int16x2_t test_pnot_i16x2(int16x2_t a) { return __riscv_pnot_i16x2(a); }

// CHECK-LABEL: test_pnot_u16x2:
// CHECK:       not{{[[:space:]]}}
uint16x2_t test_pnot_u16x2(uint16x2_t a) { return __riscv_pnot_u16x2(a); }

// CHECK-LABEL: test_pand_i8x8:
// RV32-COUNT-2: and{{[[:space:]]}}
// RV64:         and{{[[:space:]]}}
int8x8_t test_pand_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_pand_i8x8(a, b);
}

// CHECK-LABEL: test_pand_u8x8:
// RV32-COUNT-2: and{{[[:space:]]}}
// RV64:         and{{[[:space:]]}}
uint8x8_t test_pand_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_pand_u8x8(a, b);
}

// CHECK-LABEL: test_pand_i16x4:
// RV32-COUNT-2: and{{[[:space:]]}}
// RV64:         and{{[[:space:]]}}
int16x4_t test_pand_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_pand_i16x4(a, b);
}

// CHECK-LABEL: test_pand_u16x4:
// RV32-COUNT-2: and{{[[:space:]]}}
// RV64:         and{{[[:space:]]}}
uint16x4_t test_pand_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_pand_u16x4(a, b);
}

// CHECK-LABEL: test_pand_i32x2:
// RV32-COUNT-2: and{{[[:space:]]}}
// RV64:         and{{[[:space:]]}}
int32x2_t test_pand_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_pand_i32x2(a, b);
}

// CHECK-LABEL: test_pand_u32x2:
// RV32-COUNT-2: and{{[[:space:]]}}
// RV64:         and{{[[:space:]]}}
uint32x2_t test_pand_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_pand_u32x2(a, b);
}

// CHECK-LABEL: test_por_i8x8:
// RV32-COUNT-2: or{{[[:space:]]}}
// RV64:         or{{[[:space:]]}}
int8x8_t test_por_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_por_i8x8(a, b);
}

// CHECK-LABEL: test_por_u8x8:
// RV32-COUNT-2: or{{[[:space:]]}}
// RV64:         or{{[[:space:]]}}
uint8x8_t test_por_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_por_u8x8(a, b);
}

// CHECK-LABEL: test_por_i16x4:
// RV32-COUNT-2: or{{[[:space:]]}}
// RV64:         or{{[[:space:]]}}
int16x4_t test_por_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_por_i16x4(a, b);
}

// CHECK-LABEL: test_por_u16x4:
// RV32-COUNT-2: or{{[[:space:]]}}
// RV64:         or{{[[:space:]]}}
uint16x4_t test_por_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_por_u16x4(a, b);
}

// CHECK-LABEL: test_por_i32x2:
// RV32-COUNT-2: or{{[[:space:]]}}
// RV64:         or{{[[:space:]]}}
int32x2_t test_por_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_por_i32x2(a, b);
}

// CHECK-LABEL: test_por_u32x2:
// RV32-COUNT-2: or{{[[:space:]]}}
// RV64:         or{{[[:space:]]}}
uint32x2_t test_por_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_por_u32x2(a, b);
}

// CHECK-LABEL: test_pxor_i8x8:
// RV32-COUNT-2: xor{{[[:space:]]}}
// RV64:         xor{{[[:space:]]}}
int8x8_t test_pxor_i8x8(int8x8_t a, int8x8_t b) {
  return __riscv_pxor_i8x8(a, b);
}

// CHECK-LABEL: test_pxor_u8x8:
// RV32-COUNT-2: xor{{[[:space:]]}}
// RV64:         xor{{[[:space:]]}}
uint8x8_t test_pxor_u8x8(uint8x8_t a, uint8x8_t b) {
  return __riscv_pxor_u8x8(a, b);
}

// CHECK-LABEL: test_pxor_i16x4:
// RV32-COUNT-2: xor{{[[:space:]]}}
// RV64:         xor{{[[:space:]]}}
int16x4_t test_pxor_i16x4(int16x4_t a, int16x4_t b) {
  return __riscv_pxor_i16x4(a, b);
}

// CHECK-LABEL: test_pxor_u16x4:
// RV32-COUNT-2: xor{{[[:space:]]}}
// RV64:         xor{{[[:space:]]}}
uint16x4_t test_pxor_u16x4(uint16x4_t a, uint16x4_t b) {
  return __riscv_pxor_u16x4(a, b);
}

// CHECK-LABEL: test_pxor_i32x2:
// RV32-COUNT-2: xor{{[[:space:]]}}
// RV64:         xor{{[[:space:]]}}
int32x2_t test_pxor_i32x2(int32x2_t a, int32x2_t b) {
  return __riscv_pxor_i32x2(a, b);
}

// CHECK-LABEL: test_pxor_u32x2:
// RV32-COUNT-2: xor{{[[:space:]]}}
// RV64:         xor{{[[:space:]]}}
uint32x2_t test_pxor_u32x2(uint32x2_t a, uint32x2_t b) {
  return __riscv_pxor_u32x2(a, b);
}

// CHECK-LABEL: test_pnot_i8x8:
// RV32-COUNT-2: not{{[[:space:]]}}
// RV64:         not{{[[:space:]]}}
int8x8_t test_pnot_i8x8(int8x8_t a) { return __riscv_pnot_i8x8(a); }

// CHECK-LABEL: test_pnot_u8x8:
// RV32-COUNT-2: not{{[[:space:]]}}
// RV64:         not{{[[:space:]]}}
uint8x8_t test_pnot_u8x8(uint8x8_t a) { return __riscv_pnot_u8x8(a); }

// CHECK-LABEL: test_pnot_i16x4:
// RV32-COUNT-2: not{{[[:space:]]}}
// RV64:         not{{[[:space:]]}}
int16x4_t test_pnot_i16x4(int16x4_t a) { return __riscv_pnot_i16x4(a); }

// CHECK-LABEL: test_pnot_u16x4:
// RV32-COUNT-2: not{{[[:space:]]}}
// RV64:         not{{[[:space:]]}}
uint16x4_t test_pnot_u16x4(uint16x4_t a) { return __riscv_pnot_u16x4(a); }

// CHECK-LABEL: test_pnot_i32x2:
// RV32-COUNT-2: not{{[[:space:]]}}
// RV64:         not{{[[:space:]]}}
int32x2_t test_pnot_i32x2(int32x2_t a) { return __riscv_pnot_i32x2(a); }

// CHECK-LABEL: test_pnot_u32x2:
// RV32-COUNT-2: not{{[[:space:]]}}
// RV64:         not{{[[:space:]]}}
uint32x2_t test_pnot_u32x2(uint32x2_t a) { return __riscv_pnot_u32x2(a); }
