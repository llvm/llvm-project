// REQUIRES: hexagon-registered-target
// RUN: %clang++ -g -S -fenable-ripple --target=hexagon -mhvx -mv79 -emit-llvm -mllvm -ripple-disable-link %s -o - 2>&1 | FileCheck %s

#include <ripple.h>
#include <ripple/HVX_Narrow_Shift.h>

#define HVX_32_i32 32
#define HVX_64_i16 64

extern "C" {
// i32 to i16
void test_narrow_shift_i32toi16(size_t Length, int32_t *Out, int32_t *Val_even,
                                int32_t *Val_odd, uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_i32toi16(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_i32toi16
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_i32toi16

void test_narrow_shift_sat_i32toi16(size_t Length, int32_t *Out,
                                    int32_t *Val_even, int32_t *Val_odd,
                                    uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i32toi16(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i32toi16
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_sat_i32toi16

void test_narrow_shift_rnd_sat_i32toi16(size_t Length, int32_t *Out,
                                        int32_t *Val_even, int32_t *Val_odd,
                                        uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i32toi16(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i32toi16
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_rnd_sat_i32toi16

void test_narrow_shift_i32toi16_noshuff(size_t Length, int32_t *Out,
                                        int32_t *Val_even, int32_t *Val_odd,
                                        uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_i32toi16_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_i32toi16_noshuff
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_i32toi16_noshuff

void test_narrow_shift_sat_i32toi16_noshuff(size_t Length, int32_t *Out,
                                            int32_t *Val_even, int32_t *Val_odd,
                                            uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i32toi16_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i32toi16_noshuff
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_sat_i32toi16_noshuff

void test_narrow_shift_rnd_sat_i32toi16_noshuff(size_t Length, int32_t *Out,
                                                int32_t *Val_even,
                                                int32_t *Val_odd,
                                                uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i32toi16_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i32toi16_noshuff
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_rnd_sat_i32toi16_noshuff

// i32 to u16
void test_narrow_shift_sat_i32tou16(size_t Length, uint32_t *Out,
                                    int32_t *Val_even, int32_t *Val_odd,
                                    uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i32tou16(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i32tou16
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_sat_i32tou16

void test_narrow_shift_rnd_sat_i32tou16(size_t Length, uint32_t *Out,
                                        int32_t *Val_even, int32_t *Val_odd,
                                        uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i32tou16(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i32tou16
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_rnd_sat_i32tou16

void test_narrow_shift_sat_i32tou16_noshuff(size_t Length, uint32_t *Out,
                                            int32_t *Val_even, int32_t *Val_odd,
                                            uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i32tou16_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i32tou16_noshuff
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_sat_i32tou16_noshuff

void test_narrow_shift_rnd_sat_i32tou16_noshuff(size_t Length, uint32_t *Out,
                                                int32_t *Val_even,
                                                int32_t *Val_odd,
                                                uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i32tou16_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i32tou16_noshuff
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_rnd_sat_i32tou16_noshuff

// u32 to u16
void test_narrow_shift_sat_u32tou16(size_t Length, uint32_t *Out,
                                    uint32_t *Val_even, uint32_t *Val_odd,
                                    uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_u32tou16(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_u32tou16
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_sat_u32tou16

void test_narrow_shift_rnd_sat_u32tou16(size_t Length, uint32_t *Out,
                                        uint32_t *Val_even, uint32_t *Val_odd,
                                        uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_u32tou16(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_u32tou16
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_rnd_sat_u32tou16

void test_narrow_shift_sat_u32tou16_noshuff(size_t Length, uint32_t *Out,
                                            uint32_t *Val_even,
                                            uint32_t *Val_odd, uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_u32tou16_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_u32tou16_noshuff
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_sat_u32tou16_noshuff

void test_narrow_shift_rnd_sat_u32tou16_noshuff(size_t Length, uint32_t *Out,
                                                uint32_t *Val_even,
                                                uint32_t *Val_odd,
                                                uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_u32tou16_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_u32tou16_noshuff
// CHECK: call <32 x i32> @ripple_pure_hvx_narsh_rnd_sat_u32tou16_noshuff

// i16 to i8
void test_narrow_shift_sat_i16toi8(size_t Length, int16_t *Out,
                                   int16_t *Val_even, int16_t *Val_odd,
                                   uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i16toi8(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i16toi8
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_sat_i16toi8

void test_narrow_shift_rnd_sat_i16toi8(size_t Length, int16_t *Out,
                                       int16_t *Val_even, int16_t *Val_odd,
                                       uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i16toi8(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i16toi8
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_rnd_sat_i16toi8

void test_narrow_shift_sat_i16toi8_noshuff(size_t Length, int16_t *Out,
                                           int16_t *Val_even,
                                           int16_t *Val_odd, uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i16toi8_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i16toi8_noshuff
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_sat_i16toi8_noshuff

void test_narrow_shift_rnd_sat_i16toi8_noshuff(size_t Length, int16_t *Out,
                                               int16_t *Val_even,
                                               int16_t *Val_odd,
                                               uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i16toi8_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i16toi8_noshuff
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_rnd_sat_i16toi8_noshuff

// i16 to u8
void test_narrow_shift_sat_i16tou8(size_t Length, uint16_t *Out,
                                   int16_t *Val_even, int16_t *Val_odd,
                                   uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i16tou8(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i16tou8
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_sat_i16tou8

void test_narrow_shift_rnd_sat_i16tou8(size_t Length, uint16_t *Out,
                                       int16_t *Val_even, int16_t *Val_odd,
                                       uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i16tou8(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i16tou8
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_rnd_sat_i16tou8

void test_narrow_shift_sat_i16tou8_noshuff(size_t Length, uint16_t *Out,
                                           int16_t *Val_even,
                                           int16_t *Val_odd, uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_i16tou8_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_i16tou8_noshuff
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_sat_i16tou8_noshuff

void test_narrow_shift_rnd_sat_i16tou8_noshuff(size_t Length, uint16_t *Out,
                                               int16_t *Val_even,
                                               int16_t *Val_odd,
                                               uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_i16tou8_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_i16tou8_noshuff
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_rnd_sat_i16tou8_noshuff

// u16 to u8
void test_narrow_shift_sat_u16tou8(size_t Length, uint16_t *Out,
                                   uint16_t *Val_even, uint16_t *Val_odd,
                                   uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_u16tou8(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_u16tou8
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_sat_u16tou8

void test_narrow_shift_rnd_sat_u16tou8(size_t Length, uint16_t *Out,
                                       uint16_t *Val_even, uint16_t *Val_odd,
                                       uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_u16tou8(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_u16tou8
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_rnd_sat_u16tou8

void test_narrow_shift_sat_u16tou8_noshuff(size_t Length, uint16_t *Out,
                                           uint16_t *Val_even,
                                           uint16_t *Val_odd, uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_sat_u16tou8_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_sat_u16tou8_noshuff
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_sat_u16tou8_noshuff

void test_narrow_shift_rnd_sat_u16tou8_noshuff(size_t Length, uint16_t *Out,
                                               uint16_t *Val_even,
                                               uint16_t *Val_odd,
                                               uint32_t Shift) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  ripple_parallel_full(BS, 0);
  for (size_t i = 0; i < Length; i++) {
    Out[i] = hvx_narsh_rnd_sat_u16tou8_noshuff(Val_even[i], Val_odd[i], Shift);
  }
}
// CHECK: @test_narrow_shift_rnd_sat_u16tou8_noshuff
// CHECK: call <64 x i16> @ripple_pure_hvx_narsh_rnd_sat_u16tou8_noshuff
}
