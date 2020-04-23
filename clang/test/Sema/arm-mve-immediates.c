// RUN: %clang_cc1 -triple thumbv8.1m.main-none-none-eabi -fallow-half-arguments-and-returns -target-feature +mve.fp -verify -fsyntax-only %s

#include <arm_mve.h>

void test_load_offsets(uint32x4_t addr32, uint64x2_t addr64)
{
  // Offsets that should be a multiple of 8 times 0,1,...,127
  vldrdq_gather_base_s64(addr64, 0);
  vldrdq_gather_base_s64(addr64, 8);
  vldrdq_gather_base_s64(addr64, 2*8);
  vldrdq_gather_base_s64(addr64, 125*8);
  vldrdq_gather_base_s64(addr64, 126*8);
  vldrdq_gather_base_s64(addr64, 127*8);
  vldrdq_gather_base_s64(addr64, -125*8);
  vldrdq_gather_base_s64(addr64, -126*8);
  vldrdq_gather_base_s64(addr64, -127*8);
  vldrdq_gather_base_s64(addr64, 128*8); // expected-error {{argument value 1024 is outside the valid range [-1016, 1016]}}
  vldrdq_gather_base_s64(addr64, -128*8); // expected-error {{argument value -1024 is outside the valid range [-1016, 1016]}}
  vldrdq_gather_base_s64(addr64, 4); // expected-error {{argument should be a multiple of 8}}
  vldrdq_gather_base_s64(addr64, 1); // expected-error {{argument should be a multiple of 8}}

  // Offsets that should be a multiple of 4 times 0,1,...,127
  vldrwq_gather_base_s32(addr32, 0);
  vldrwq_gather_base_s32(addr32, 4);
  vldrwq_gather_base_s32(addr32, 2*4);
  vldrwq_gather_base_s32(addr32, 125*4);
  vldrwq_gather_base_s32(addr32, 126*4);
  vldrwq_gather_base_s32(addr32, 127*4);
  vldrwq_gather_base_s32(addr32, -125*4);
  vldrwq_gather_base_s32(addr32, -126*4);
  vldrwq_gather_base_s32(addr32, -127*4);
  vldrwq_gather_base_s32(addr32, 128*4); // expected-error {{argument value 512 is outside the valid range [-508, 508]}}
  vldrwq_gather_base_s32(addr32, -128*4); // expected-error {{argument value -512 is outside the valid range [-508, 508]}}
  vldrwq_gather_base_s32(addr32, 2); // expected-error {{argument should be a multiple of 4}}
  vldrwq_gather_base_s32(addr32, 1); // expected-error {{argument should be a multiple of 4}}

  // Show that the polymorphic store intrinsics get the right set of
  // error checks after overload resolution. These ones expand to the
  // 8-byte granular versions...
  vstrdq_scatter_base(addr64, 0, addr64);
  vstrdq_scatter_base(addr64, 8, addr64);
  vstrdq_scatter_base(addr64, 2*8, addr64);
  vstrdq_scatter_base(addr64, 125*8, addr64);
  vstrdq_scatter_base(addr64, 126*8, addr64);
  vstrdq_scatter_base(addr64, 127*8, addr64);
  vstrdq_scatter_base(addr64, -125*8, addr64);
  vstrdq_scatter_base(addr64, -126*8, addr64);
  vstrdq_scatter_base(addr64, -127*8, addr64);
  vstrdq_scatter_base(addr64, 128*8, addr64); // expected-error {{argument value 1024 is outside the valid range [-1016, 1016]}}
  vstrdq_scatter_base(addr64, -128*8, addr64); // expected-error {{argument value -1024 is outside the valid range [-1016, 1016]}}
  vstrdq_scatter_base(addr64, 4, addr64); // expected-error {{argument should be a multiple of 8}}
  vstrdq_scatter_base(addr64, 1, addr64); // expected-error {{argument should be a multiple of 8}}

  /// ... and these ones to the 4-byte.
  vstrwq_scatter_base(addr32, 0, addr32);
  vstrwq_scatter_base(addr32, 4, addr32);
  vstrwq_scatter_base(addr32, 2*4, addr32);
  vstrwq_scatter_base(addr32, 125*4, addr32);
  vstrwq_scatter_base(addr32, 126*4, addr32);
  vstrwq_scatter_base(addr32, 127*4, addr32);
  vstrwq_scatter_base(addr32, -125*4, addr32);
  vstrwq_scatter_base(addr32, -126*4, addr32);
  vstrwq_scatter_base(addr32, -127*4, addr32);
  vstrwq_scatter_base(addr32, 128*4, addr32); // expected-error {{argument value 512 is outside the valid range [-508, 508]}}
  vstrwq_scatter_base(addr32, -128*4, addr32); // expected-error {{argument value -512 is outside the valid range [-508, 508]}}
  vstrwq_scatter_base(addr32, 2, addr32); // expected-error {{argument should be a multiple of 4}}
  vstrwq_scatter_base(addr32, 1, addr32); // expected-error {{argument should be a multiple of 4}}
}

void test_lane_indices(uint8x16_t v16, uint16x8_t v8,
                       uint32x4_t v4, uint64x2_t v2)
{
  vgetq_lane_u8(v16, -1); // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vgetq_lane_u8(v16, 0);
  vgetq_lane_u8(v16, 15);
  vgetq_lane_u8(v16, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}

  vgetq_lane_u16(v8, -1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  vgetq_lane_u16(v8, 0);
  vgetq_lane_u16(v8, 7);
  vgetq_lane_u16(v8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  vgetq_lane_u32(v4, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vgetq_lane_u32(v4, 0);
  vgetq_lane_u32(v4, 3);
  vgetq_lane_u32(v4, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  vgetq_lane_u64(v2, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vgetq_lane_u64(v2, 0);
  vgetq_lane_u64(v2, 1);
  vgetq_lane_u64(v2, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  vsetq_lane_u8(23, v16, -1); // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vsetq_lane_u8(23, v16, 0);
  vsetq_lane_u8(23, v16, 15);
  vsetq_lane_u8(23, v16, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}

  vsetq_lane_u16(23, v8, -1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  vsetq_lane_u16(23, v8, 0);
  vsetq_lane_u16(23, v8, 7);
  vsetq_lane_u16(23, v8, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  vsetq_lane_u32(23, v4, -1); // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  vsetq_lane_u32(23, v4, 0);
  vsetq_lane_u32(23, v4, 3);
  vsetq_lane_u32(23, v4, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  vsetq_lane_u64(23, v2, -1); // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  vsetq_lane_u64(23, v2, 0);
  vsetq_lane_u64(23, v2, 1);
  vsetq_lane_u64(23, v2, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
}

void test_immediate_shifts(uint8x16_t vb, uint16x8_t vh, uint32x4_t vw)
{
  vshlq_n(vb, 0);
  vshlq_n(vb, 7);
  vshlq_n(vh, 0);
  vshlq_n(vh, 15);
  vshlq_n(vw, 0);
  vshlq_n(vw, 31);

  vshlq_n(vb, -1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  vshlq_n(vb, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  vshlq_n(vh, -1); // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vshlq_n(vh, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vshlq_n(vw, -1); // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  vshlq_n(vw, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  vqshlq_n(vb, 0);
  vqshlq_n(vb, 7);
  vqshlq_n(vh, 0);
  vqshlq_n(vh, 15);
  vqshlq_n(vw, 0);
  vqshlq_n(vw, 31);

  vqshlq_n(vb, -1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  vqshlq_n(vb, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  vqshlq_n(vh, -1); // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vqshlq_n(vh, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vqshlq_n(vw, -1); // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  vqshlq_n(vw, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  vsliq(vb, vb, 0);
  vsliq(vb, vb, 7);
  vsliq(vh, vh, 0);
  vsliq(vh, vh, 15);
  vsliq(vw, vw, 0);
  vsliq(vw, vw, 31);

  vsliq(vb, vb, -1); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  vsliq(vb, vb, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  vsliq(vh, vh, -1); // expected-error {{argument value -1 is outside the valid range [0, 15]}}
  vsliq(vh, vh, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  vsliq(vw, vw, -1); // expected-error {{argument value -1 is outside the valid range [0, 31]}}
  vsliq(vw, vw, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}

  vshllbq(vb, 1);
  vshllbq(vb, 8);
  vshllbq(vh, 1);
  vshllbq(vh, 16);

  vshllbq(vb, 0); // expected-error {{argument value 0 is outside the valid range [1, 8]}}
  vshllbq(vb, 9); // expected-error {{argument value 9 is outside the valid range [1, 8]}}
  vshllbq(vh, 0); // expected-error {{argument value 0 is outside the valid range [1, 16]}}
  vshllbq(vh, 17); // expected-error {{argument value 17 is outside the valid range [1, 16]}}

  vshrq(vb, 1);
  vshrq(vb, 8);
  vshrq(vh, 1);
  vshrq(vh, 16);
  vshrq(vw, 1);
  vshrq(vw, 32);

  vshrq(vb, 0); // expected-error {{argument value 0 is outside the valid range [1, 8]}}
  vshrq(vb, 9); // expected-error {{argument value 9 is outside the valid range [1, 8]}}
  vshrq(vh, 0); // expected-error {{argument value 0 is outside the valid range [1, 16]}}
  vshrq(vh, 17); // expected-error {{argument value 17 is outside the valid range [1, 16]}}
  vshrq(vw, 0); // expected-error {{argument value 0 is outside the valid range [1, 32]}}
  vshrq(vw, 33); // expected-error {{argument value 33 is outside the valid range [1, 32]}}

  vshrntq(vb, vh, 1);
  vshrntq(vb, vh, 8);
  vshrntq(vh, vw, 1);
  vshrntq(vh, vw, 16);

  vshrntq(vb, vh, 0); // expected-error {{argument value 0 is outside the valid range [1, 8]}}
  vshrntq(vb, vh, 9); // expected-error {{argument value 9 is outside the valid range [1, 8]}}
  vshrntq(vh, vw, 0); // expected-error {{argument value 0 is outside the valid range [1, 16]}}
  vshrntq(vh, vw, 17); // expected-error {{argument value 17 is outside the valid range [1, 16]}}

  vsriq(vb, vb, 1);
  vsriq(vb, vb, 8);
  vsriq(vh, vh, 1);
  vsriq(vh, vh, 16);
  vsriq(vw, vw, 1);
  vsriq(vw, vw, 32);

  vsriq(vb, vb, 0); // expected-error {{argument value 0 is outside the valid range [1, 8]}}
  vsriq(vb, vb, 9); // expected-error {{argument value 9 is outside the valid range [1, 8]}}
  vsriq(vh, vh, 0); // expected-error {{argument value 0 is outside the valid range [1, 16]}}
  vsriq(vh, vh, 17); // expected-error {{argument value 17 is outside the valid range [1, 16]}}
  vsriq(vw, vw, 0); // expected-error {{argument value 0 is outside the valid range [1, 32]}}
  vsriq(vw, vw, 33); // expected-error {{argument value 33 is outside the valid range [1, 32]}}
}

void test_simd_bic_orr(int16x8_t h, int32x4_t w)
{
    h = vbicq(h, 0x0000);
    h = vbicq(h, 0x0001);
    h = vbicq(h, 0x00FF);
    h = vbicq(h, 0x0100);
    h = vbicq(h, 0x0101); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}
    h = vbicq(h, 0x01FF); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}
    h = vbicq(h, 0xFF00);

    w = vbicq(w, 0x00000000);
    w = vbicq(w, 0x00000001);
    w = vbicq(w, 0x000000FF);
    w = vbicq(w, 0x00000100);
    w = vbicq(w, 0x0000FF00);
    w = vbicq(w, 0x00010000);
    w = vbicq(w, 0x00FF0000);
    w = vbicq(w, 0x01000000);
    w = vbicq(w, 0xFF000000);
    w = vbicq(w, 0x01000001); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}
    w = vbicq(w, 0x01FFFFFF); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}

    h = vorrq(h, 0x0000);
    h = vorrq(h, 0x0001);
    h = vorrq(h, 0x00FF);
    h = vorrq(h, 0x0100);
    h = vorrq(h, 0x0101); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}
    h = vorrq(h, 0x01FF); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}
    h = vorrq(h, 0xFF00);

    w = vorrq(w, 0x00000000);
    w = vorrq(w, 0x00000001);
    w = vorrq(w, 0x000000FF);
    w = vorrq(w, 0x00000100);
    w = vorrq(w, 0x0000FF00);
    w = vorrq(w, 0x00010000);
    w = vorrq(w, 0x00FF0000);
    w = vorrq(w, 0x01000000);
    w = vorrq(w, 0xFF000000);
    w = vorrq(w, 0x01000001); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}
    w = vorrq(w, 0x01FFFFFF); // expected-error-re {{argument should be an 8-bit value shifted by a multiple of 8 bits{{$}}}}
}

void test_simd_vmvn(void)
{
    uint16x8_t h;
    h = vmvnq_n_u16(0x0000);
    h = vmvnq_n_u16(0x0001);
    h = vmvnq_n_u16(0x00FF);
    h = vmvnq_n_u16(0x0100);
    h = vmvnq_n_u16(0x0101); // expected-error {{argument should be an 8-bit value shifted by a multiple of 8 bits, or in the form 0x??FF}}
    h = vmvnq_n_u16(0x01FF);
    h = vmvnq_n_u16(0xFF00);

    uint32x4_t w;
    w = vmvnq_n_u32(0x00000000);
    w = vmvnq_n_u32(0x00000001);
    w = vmvnq_n_u32(0x000000FF);
    w = vmvnq_n_u32(0x00000100);
    w = vmvnq_n_u32(0x0000FF00);
    w = vmvnq_n_u32(0x00010000);
    w = vmvnq_n_u32(0x00FF0000);
    w = vmvnq_n_u32(0x01000000);
    w = vmvnq_n_u32(0xFF000000);
    w = vmvnq_n_u32(0x01000001); // expected-error {{argument should be an 8-bit value shifted by a multiple of 8 bits, or in the form 0x??FF}}
    w = vmvnq_n_u32(0x01FFFFFF); // expected-error {{argument should be an 8-bit value shifted by a multiple of 8 bits, or in the form 0x??FF}}
    w = vmvnq_n_u32(0x0001FFFF); // expected-error {{argument should be an 8-bit value shifted by a multiple of 8 bits, or in the form 0x??FF}}
    w = vmvnq_n_u32(0x000001FF);
}

void test_vidup(void)
{
    vidupq_n_u16(0x12345678, 1);
    vidupq_n_u16(0x12345678, 2);
    vidupq_n_u16(0x12345678, 4);
    vidupq_n_u16(0x12345678, 8);

    vidupq_n_u16(0x12345678, 0); // expected-error {{argument value 0 is outside the valid range [1, 8]}}
    vidupq_n_u16(0x12345678, 16); // expected-error {{argument value 16 is outside the valid range [1, 8]}}
    vidupq_n_u16(0x12345678, -1); // expected-error {{argument value -1 is outside the valid range [1, 8]}}
    vidupq_n_u16(0x12345678, -2); // expected-error {{argument value -2 is outside the valid range [1, 8]}}
    vidupq_n_u16(0x12345678, -4); // expected-error {{argument value -4 is outside the valid range [1, 8]}}
    vidupq_n_u16(0x12345678, -8); // expected-error {{argument value -8 is outside the valid range [1, 8]}}
    vidupq_n_u16(0x12345678, 3); // expected-error {{argument should be a power of 2}}
    vidupq_n_u16(0x12345678, 7); // expected-error {{argument should be a power of 2}}
}

void test_vcvtq(void)
{
    uint16x8_t vec_u16;
    float16x8_t vec_f16;
    vcvtq_n_f16_u16(vec_u16, 0); // expected-error {{argument value 0 is outside the valid range [1, 16]}}
    vcvtq_n_f16_u16(vec_u16, 1);
    vcvtq_n_f16_u16(vec_u16, 16);
    vcvtq_n_f16_u16(vec_u16, 17); // expected-error {{argument value 17 is outside the valid range [1, 16]}}

    int32x4_t vec_s32;
    float32x4_t vec_f32;
    vcvtq_n_s32_f32(vec_s32, -1); // expected-error {{argument value -1 is outside the valid range [1, 32]}}
    vcvtq_n_s32_f32(vec_s32, 1);
    vcvtq_n_s32_f32(vec_s32, 32);
    vcvtq_n_s32_f32(vec_s32, 33); // expected-error {{argument value 33 is outside the valid range [1, 32]}}
}
