// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-CLANG %s
// RUN: %clang -S -O1 -fenable-ripple -emit-llvm %s -o - | FileCheck --check-prefix=CHECK-RIPPLE %s
#include <ripple.h>

typedef signed char sc;
typedef unsigned char uc;
typedef signed short ss;
typedef unsigned short us;
typedef signed int si;
typedef unsigned int ui;
typedef signed long sl;
typedef unsigned long ul;
typedef signed long long sll;
typedef unsigned long long ull;

#define gen_check_ripple_sat_1d(N, OP, SHORTTYPE, LONGTYPE)                    \
  void check_##OP##_sat_##SHORTTYPE(const LONGTYPE x1[N],                      \
                                    const LONGTYPE x2[N], LONGTYPE y[N]) {     \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int v0 = ripple_id(BS, 0);                                                 \
    y[v0] = ripple_##OP##_sat(x1[v0], x2[v0]);                                 \
  }

// {{{ add

gen_check_ripple_sat_1d(128, add, i8, int8_t);
// CHECK-CLANG: @check_add_sat_i8
// CHECK-RIPPLE: @check_add_sat_i8
// CHECK-CLANG: @llvm.sadd.sat.i8
// CHECK-RIPPLE: @llvm.sadd.sat.v128i8
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(64, add, i16, int16_t);
// CHECK-CLANG: @check_add_sat_i16
// CHECK-RIPPLE: @check_add_sat_i16
// CHECK-CLANG: @llvm.sadd.sat.i16
// CHECK-RIPPLE: @llvm.sadd.sat.v64i16
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(32, add, i32, int32_t);
// CHECK-CLANG: @check_add_sat_i32
// CHECK-RIPPLE: @check_add_sat_i32
// CHECK-CLANG: @llvm.sadd.sat.i32
// CHECK-RIPPLE: @llvm.sadd.sat.v32i32
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(16, add, i64, int64_t);
// CHECK-CLANG: @check_add_sat_i64
// CHECK-RIPPLE: @check_add_sat_i64
// CHECK-CLANG: @llvm.sadd.sat.i64
// CHECK-RIPPLE: @llvm.sadd.sat.v16i64
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(128, add, u8, uint8_t);
// CHECK-CLANG: @check_add_sat_u8
// CHECK-RIPPLE: @check_add_sat_u8
// CHECK-CLANG: @llvm.uadd.sat.i8
// CHECK-RIPPLE: @llvm.uadd.sat.v128i8
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(64, add, u16, uint16_t);
// CHECK-CLANG: @check_add_sat_u16
// CHECK-RIPPLE: @check_add_sat_u16
// CHECK-CLANG: @llvm.uadd.sat.i16
// CHECK-RIPPLE: @llvm.uadd.sat.v64i16
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(32, add, u32, uint32_t);
// CHECK-CLANG: @check_add_sat_u32
// CHECK-RIPPLE: @check_add_sat_u32
// CHECK-CLANG: @llvm.uadd.sat.i32
// CHECK-RIPPLE: @llvm.uadd.sat.v32i32
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(16, add, u64, uint64_t);
// CHECK-CLANG: @check_add_sat_u64
// CHECK-RIPPLE: @check_add_sat_u64
// CHECK-CLANG: @llvm.uadd.sat.i64
// CHECK-RIPPLE: @llvm.uadd.sat.v16i64
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret

gen_check_ripple_sat_1d(128, add, c, char);
gen_check_ripple_sat_1d(128, add, sc, sc);
gen_check_ripple_sat_1d(128, add, uc, uc);
gen_check_ripple_sat_1d(128, add, ss, ss);
gen_check_ripple_sat_1d(128, add, us, us);
gen_check_ripple_sat_1d(128, add, si, si);
gen_check_ripple_sat_1d(128, add, ui, ui);
gen_check_ripple_sat_1d(128, add, sl, sl);
gen_check_ripple_sat_1d(128, add, ul, ul);
gen_check_ripple_sat_1d(128, add, sll, sll);
gen_check_ripple_sat_1d(128, add, ull, ull);

// }}}


// {{{ sub

gen_check_ripple_sat_1d(128, sub, i8, int8_t);
// CHECK-CLANG: @check_sub_sat_i8
// CHECK-RIPPLE: @check_sub_sat_i8
// CHECK-CLANG: @llvm.ssub.sat.i8
// CHECK-RIPPLE: @llvm.ssub.sat.v128i8
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(64, sub, i16, int16_t);
// CHECK-CLANG: @check_sub_sat_i16
// CHECK-RIPPLE: @check_sub_sat_i16
// CHECK-CLANG: @llvm.ssub.sat.i16
// CHECK-RIPPLE: @llvm.ssub.sat.v64i16
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(32, sub, i32, int32_t);
// CHECK-CLANG: @check_sub_sat_i32
// CHECK-RIPPLE: @check_sub_sat_i32
// CHECK-CLANG: @llvm.ssub.sat.i32
// CHECK-RIPPLE: @llvm.ssub.sat.v32i32
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(16, sub, i64, int64_t);
// CHECK-CLANG: @check_sub_sat_i64
// CHECK-RIPPLE: @check_sub_sat_i64
// CHECK-CLANG: @llvm.ssub.sat.i64
// CHECK-RIPPLE: @llvm.ssub.sat.v16i64
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(128, sub, u8, uint8_t);
// CHECK-CLANG: @check_sub_sat_u8
// CHECK-RIPPLE: @check_sub_sat_u8
// CHECK-CLANG: @llvm.usub.sat.i8
// CHECK-RIPPLE: @llvm.usub.sat.v128i8
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(64, sub, u16, uint16_t);
// CHECK-CLANG: @check_sub_sat_u16
// CHECK-RIPPLE: @check_sub_sat_u16
// CHECK-CLANG: @llvm.usub.sat.i16
// CHECK-RIPPLE: @llvm.usub.sat.v64i16
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(32, sub, u32, uint32_t);
// CHECK-CLANG: @check_sub_sat_u32
// CHECK-RIPPLE: @check_sub_sat_u32
// CHECK-CLANG: @llvm.usub.sat.i32
// CHECK-RIPPLE: @llvm.usub.sat.v32i32
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret
gen_check_ripple_sat_1d(16, sub, u64, uint64_t);
// CHECK-CLANG: @check_sub_sat_u64
// CHECK-RIPPLE: @check_sub_sat_u64
// CHECK-CLANG: @llvm.usub.sat.i64
// CHECK-RIPPLE: @llvm.usub.sat.v16i64
// CHECK-CLANG: ret
// CHECK-RIPPLE: ret

gen_check_ripple_sat_1d(128, sub, c, char);
gen_check_ripple_sat_1d(128, sub, sc, sc);
gen_check_ripple_sat_1d(128, sub, uc, uc);
gen_check_ripple_sat_1d(128, sub, ss, ss);
gen_check_ripple_sat_1d(128, sub, us, us);
gen_check_ripple_sat_1d(128, sub, si, si);
gen_check_ripple_sat_1d(128, sub, ui, ui);
gen_check_ripple_sat_1d(128, sub, sl, sl);
gen_check_ripple_sat_1d(128, sub, ul, ul);
gen_check_ripple_sat_1d(128, sub, sll, sll);
gen_check_ripple_sat_1d(128, sub, ull, ull);

// }}}

int main(void) {
  // CHECK-CLANG: @main
  // CHECK-RIPPLE: @main
  return 0;
}
