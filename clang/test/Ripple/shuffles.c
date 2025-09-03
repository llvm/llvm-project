// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

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

size_t get_src_id(size_t id, size_t n) {
  return (id % 2) ? (id - 1) : (id + 1);
}

#define gen_shuffle_test(N, LONGTYPE, SHORTTYPE)                               \
  void check_shuffle_##SHORTTYPE(LONGTYPE in[N], LONGTYPE out[N]) {            \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int id = ripple_id(BS, 0);                                                 \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE shuf_tmp = ripple_shuffle(tmp, get_src_id);                       \
    out[id] = shuf_tmp;                                                        \
  }

gen_shuffle_test(128, uint8_t, u8);
// CHECK: @check_shuffle_u8
// CHECK: llvm.ripple.shuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(128, int8_t, i8);
// CHECK: @check_shuffle_i8
// CHECK: llvm.ripple.shuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(64, uint16_t, u16);
// CHECK: @check_shuffle_u16
// CHECK: llvm.ripple.shuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(64, int16_t, i16);
// CHECK: @check_shuffle_i16
// CHECK: llvm.ripple.shuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(32, uint32_t, u32);
// CHECK: @check_shuffle_u32
// CHECK: llvm.ripple.shuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(32, int32_t, i32);
// CHECK: @check_shuffle_i32
// CHECK: llvm.ripple.shuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(16, uint64_t, u64);
// CHECK: @check_shuffle_u64
// CHECK: llvm.ripple.shuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(16, int64_t, i64);
// CHECK: @check_shuffle_i64
// CHECK: llvm.ripple.shuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

void check_shuffle_ptr(char* in[32], char* out[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  int id = ripple_id(BS, 0);
  char* tmp = in[id];
  char* shuf_tmp = ripple_shuffle_p(tmp, get_src_id);
  out[id] = shuf_tmp;
}
// CHECK: @check_shuffle_ptr
// CHECK: llvm.ripple.shuffle.p0(ptr {{.*}}, ptr {{.*}}, i1 false, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_test(128, char, c);
gen_shuffle_test(128, sc, sc);
gen_shuffle_test(128, uc, uc);
gen_shuffle_test(128, ss, ss);
gen_shuffle_test(128, us, us);
gen_shuffle_test(128, si, si);
gen_shuffle_test(128, ui, ui);
gen_shuffle_test(128, sl, sl);
gen_shuffle_test(128, ul, ul);
gen_shuffle_test(128, sll, sll);
gen_shuffle_test(128, ull, ull);

#define gen_shuffle_pair_test(N, LONGTYPE, SHORTTYPE)                          \
  void check_shuffle_pair_##SHORTTYPE(LONGTYPE in[N], LONGTYPE in2[N],         \
                                     LONGTYPE out[N]) {                        \
    ripple_block_t BS = ripple_set_block_shape(0, N);                          \
    int id = ripple_id(BS, 0);                                                 \
    LONGTYPE tmp = in[id];                                                     \
    LONGTYPE tmp2 = in2[id];                                                   \
    LONGTYPE shuf_tmp = ripple_shuffle_pair(tmp, tmp2, get_src_id);            \
    out[id] = shuf_tmp;                                                        \
  }

gen_shuffle_pair_test(128, uint8_t, u8);
// CHECK: @check_shuffle_pair_u8
// CHECK: llvm.ripple.shuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(128, int8_t, i8);
// CHECK: @check_shuffle_pair_i8
// CHECK: llvm.ripple.shuffle.i8(i8 %{{[0-9]+}}, i8 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(64, uint16_t, u16);
// CHECK: @check_shuffle_pair_u16
// CHECK: llvm.ripple.shuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(64, int16_t, i16);
// CHECK: @check_shuffle_pair_i16
// CHECK: llvm.ripple.shuffle.i16(i16 %{{[0-9]+}}, i16 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(32, uint32_t, u32);
// CHECK: @check_shuffle_pair_u32
// CHECK: llvm.ripple.shuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(32, int32_t, i32);
// CHECK: @check_shuffle_pair_i32
// CHECK: llvm.ripple.shuffle.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(16, uint64_t, u64);
// CHECK: @check_shuffle_pair_u64
// CHECK: llvm.ripple.shuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(16, int64_t, i64);
// CHECK: @check_shuffle_pair_i64
// CHECK: llvm.ripple.shuffle.i64(i64 %{{[0-9]+}}, i64 %{{[0-9]+}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

void check_shuffle_pair_ptr(char* in[16], char* in2[16],
                                   char* out[16]) {
  ripple_block_t BS = ripple_set_block_shape(0, 16);
  int id = ripple_id(BS, 0);
  char* tmp = in[id];
  char* tmp2 = in2[id];
  char* shuf_tmp = ripple_shuffle_pair_p(tmp, tmp2, get_src_id);
  out[id] = shuf_tmp;
}
// CHECK: @check_shuffle_pair_ptr
// CHECK: llvm.ripple.shuffle.p0(ptr %{{.*}}, ptr %{{.*}}, i1 true, ptr nonnull @get_src_id)
// CHECK: ret

gen_shuffle_pair_test(128, char, c);
gen_shuffle_pair_test(128, sc, sc);
gen_shuffle_pair_test(128, uc, uc);
gen_shuffle_pair_test(128, ss, ss);
gen_shuffle_pair_test(128, us, us);
gen_shuffle_pair_test(128, si, si);
gen_shuffle_pair_test(128, ui, ui);
gen_shuffle_pair_test(128, sl, sl);
gen_shuffle_pair_test(128, ul, ul);
gen_shuffle_pair_test(128, sll, sll);
gen_shuffle_pair_test(128, ull, ull);
