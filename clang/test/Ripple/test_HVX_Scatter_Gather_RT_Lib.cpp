// REQUIRES: hexagon-registered-target
// RUN: %clang++ -g -S -fenable-ripple --target=hexagon -mhvx -mv79 -emit-llvm %s -o - -mllvm -ripple-disable-link 2>&1 | FileCheck %s

#include <ripple.h>
#include <ripple_math.h>
#include <ripple/HVX_Scatter_Gather.h>

#define HVX_128_i8 128
#define HVX_64_i16 64
#define HVX_32_i32 32

extern "C" {
void Ripple_gather_u32(size_t length, uint32_t *destination, uint32_t *source,
                       int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_u32
// CHECK: call void @ripple_hvx_gather_u32
// CHECK: call void @ripple_mask_hvx_gather_u32

void Ripple_gather_i32(size_t length, int32_t *destination, int32_t *source,
                       int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_i32
// CHECK: call void @ripple_hvx_gather_i32
// CHECK: call void @ripple_mask_hvx_gather_i32

void Ripple_gather_i64(size_t length, int64_t *destination, int64_t *source,
                       int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_i64
// CHECK: call void @ripple_hvx_gather_i64
// CHECK: call void @ripple_mask_hvx_gather_i64

void Ripple_gather_i16(size_t length, int16_t *destination, int16_t *source,
                       int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_i16
// CHECK: call void @ripple_hvx_gather_i16
// CHECK: call void @ripple_mask_hvx_gather_i16

void Ripple_gather_i8(size_t length, int8_t *destination, int8_t *source,
                      int8_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_128_i8);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_i8
// CHECK: call void @ripple_hvx_gather_i8
// CHECK: call void @ripple_mask_hvx_gather_i8

void Ripple_gather_i16_16(size_t length, int16_t *destination, int16_t *source,
                          int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_i16_16
// CHECK: call void @ripple_hvx_gather_i16
// CHECK: call void @ripple_mask_hvx_gather_i16

void Ripple_gather_i8_16(size_t length, int8_t *destination, int8_t *source,
                         int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_128_i8);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_i8_16
// CHECK: call void @ripple_hvx_gather_i8
// CHECK: call void @ripple_mask_hvx_gather_i8

void Ripple_gather_f32(size_t length, float *destination, float *source,
                       int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_f32
// CHECK: call void @ripple_hvx_gather_f32
// CHECK: call void @ripple_mask_hvx_gather_f32

void Ripple_gather_f16(size_t length, _Float16 *destination, _Float16 *source,
                       int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_f16
// CHECK: call void @ripple_hvx_gather_f16
// CHECK: call void @ripple_mask_hvx_gather_f16

void Ripple_gather_f64(size_t length, double *destination, double *source,
                       int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
  if (i + v < length) {
    hvx_gather(destination + i, source, indexes[i + v], length);
  }
}
// CHECK: @Ripple_gather_f64
// CHECK: call void @ripple_hvx_gather_f64
// CHECK: call void @ripple_mask_hvx_gather_f64

void Ripple_scatter_i64(size_t length, int64_t *destination, int64_t *source,
                        int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_i64
// CHECK: call void @ripple_hvx_scatter_i64
// CHECK: call void @ripple_mask_hvx_scatter_i64

void Ripple_scatter_i32(size_t length, int32_t *destination, int32_t *source,
                        int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_i32
// CHECK: call void @ripple_hvx_scatter_i32
// CHECK: call void @ripple_mask_hvx_scatter_i32

void Ripple_scatter_i16(size_t length, int16_t *destination, int16_t *source,
                        int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_i16
// CHECK: call void @ripple_hvx_scatter_i16
// CHECK: call void @ripple_mask_hvx_scatter_i16

void Ripple_scatter_i8(size_t length, int8_t *destination, int8_t *source,
                       int8_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_128_i8);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_i8
// CHECK: call void @ripple_hvx_scatter_i8
// CHECK: call void @ripple_mask_hvx_scatter_i8

void Ripple_scatter_i16_16(size_t length, int16_t *destination, int16_t *source,
                           int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_i16_16
// CHECK: call void @ripple_hvx_scatter_i16
// CHECK: call void @ripple_mask_hvx_scatter_i16

void Ripple_scatter_i8_16(size_t length, int8_t *destination, int8_t *source,
                          int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_128_i8);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_i8_16
// CHECK: call void @ripple_hvx_scatter_i8
// CHECK: call void @ripple_mask_hvx_scatter_i8

void Ripple_scatter_u64(size_t length, uint64_t *destination, uint64_t *source,
                        int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_u64
// CHECK: call void @ripple_hvx_scatter_u64
// CHECK: call void @ripple_mask_hvx_scatter_u64

void Ripple_scatter_u32(size_t length, uint32_t *destination, uint32_t *source,
                        int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_u32
// CHECK: call void @ripple_hvx_scatter_u32
// CHECK: call void @ripple_mask_hvx_scatter_u32

void Ripple_scatter_u16(size_t length, uint16_t *destination, uint16_t *source,
                        int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_u16
// CHECK: call void @ripple_hvx_scatter_u16
// CHECK: call void @ripple_mask_hvx_scatter_u16

void Ripple_scatter_u8(size_t length, uint8_t *destination, uint8_t *source,
                       uint8_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_128_i8);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_u8
// CHECK: call void @ripple_hvx_scatter_u8
// CHECK: call void @ripple_mask_hvx_scatter_u8

void Ripple_scatter_u16_16(size_t length, uint16_t *destination,
                           uint16_t *source, int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_u16
// CHECK: call void @ripple_hvx_scatter_u16
// CHECK: call void @ripple_mask_hvx_scatter_u16

void Ripple_scatter_u8_16(size_t length, uint8_t *destination, uint8_t *source,
                          int16_t *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_128_i8);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_u8_16
// CHECK: call void @ripple_hvx_scatter_u8
// CHECK: call void @ripple_mask_hvx_scatter_u8

void Ripple_scatter_f32(size_t length, float *destination, float *source,
                        int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_f32
// CHECK: call void @ripple_hvx_scatter_f32
// CHECK: call void @ripple_mask_hvx_scatter_f32

void Ripple_scatter_f16(size_t length, _Float16 *destination, _Float16 *source,
                        int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_64_i16);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_f16
// CHECK: call void @ripple_hvx_scatter_f16
// CHECK: call void @ripple_mask_hvx_scatter_f16

void Ripple_scatter_f64(size_t length, double *destination, double *source,
                        int *indexes) {
  ripple_block_t BS = ripple_set_block_shape(0, HVX_32_i32);
  int nv = ripple_get_block_size(BS, 0);
  int v = ripple_id(BS, 0);
  int i;

  for (i = 0; i + nv <= length; i += nv) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
  if (i + v < length) {
    hvx_scatter(destination, indexes[i + v], source[i + v], length);
  }
}
// CHECK: @Ripple_scatter_f64
// CHECK: call void @ripple_hvx_scatter_f64
// CHECK: call void @ripple_mask_hvx_scatter_f64
}
