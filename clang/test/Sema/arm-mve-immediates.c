// RUN: %clang_cc1 -triple thumbv8.1m.main-arm-none-eabi -fallow-half-arguments-and-returns -target-feature +mve.fp -verify -fsyntax-only %s

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
  vldrdq_gather_base_s64(addr64, -8); // expected-error {{argument value -8 is outside the valid range [0, 1016]}}
  vldrdq_gather_base_s64(addr64, 128*8); // expected-error {{argument value 1024 is outside the valid range [0, 1016]}}
  vldrdq_gather_base_s64(addr64, 4); // expected-error {{argument should be a multiple of 8}}
  vldrdq_gather_base_s64(addr64, 1); // expected-error {{argument should be a multiple of 8}}

  // Offsets that should be a multiple of 4 times 0,1,...,127
  vldrwq_gather_base_s32(addr32, 0);
  vldrwq_gather_base_s32(addr32, 4);
  vldrwq_gather_base_s32(addr32, 2*4);
  vldrwq_gather_base_s32(addr32, 125*4);
  vldrwq_gather_base_s32(addr32, 126*4);
  vldrwq_gather_base_s32(addr32, 127*4);
  vldrwq_gather_base_s32(addr32, -4); // expected-error {{argument value -4 is outside the valid range [0, 508]}}
  vldrwq_gather_base_s32(addr32, 128*4); // expected-error {{argument value 512 is outside the valid range [0, 508]}}
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
  vstrdq_scatter_base(addr64, -8, addr64); // expected-error {{argument value -8 is outside the valid range [0, 1016]}}
  vstrdq_scatter_base(addr64, 128*8, addr64); // expected-error {{argument value 1024 is outside the valid range [0, 1016]}}
  vstrdq_scatter_base(addr64, 4, addr64); // expected-error {{argument should be a multiple of 8}}
  vstrdq_scatter_base(addr64, 1, addr64); // expected-error {{argument should be a multiple of 8}}

  /// ... and these ones to the 4-byte.
  vstrwq_scatter_base(addr32, 0, addr32);
  vstrwq_scatter_base(addr32, 4, addr32);
  vstrwq_scatter_base(addr32, 2*4, addr32);
  vstrwq_scatter_base(addr32, 125*4, addr32);
  vstrwq_scatter_base(addr32, 126*4, addr32);
  vstrwq_scatter_base(addr32, 127*4, addr32);
  vstrwq_scatter_base(addr32, -4, addr32); // expected-error {{argument value -4 is outside the valid range [0, 508]}}
  vstrwq_scatter_base(addr32, 128*4, addr32); // expected-error {{argument value 512 is outside the valid range [0, 508]}}
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
