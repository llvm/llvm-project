// RUN: %clang_cc1 -triple armv8.2a-arm-none-eabi -target-feature +neon -target-feature +bf16 -mfloat-abi hard \
// RUN: -disable-O0-optnone -S -emit-llvm -o - %s \
// RUN: | opt -S -instcombine \
// RUN: | FileCheck %s

// REQUIRES: arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: @test_vreinterpret_bf16_s8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i8> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_s8(int8x8_t a)      { return vreinterpret_bf16_s8(a);    }
// CHECK-LABEL: @test_vreinterpret_bf16_s16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_s16(int16x4_t a)    { return vreinterpret_bf16_s16(a);   }
// CHECK-LABEL: @test_vreinterpret_bf16_s32(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[A:%.*]]
//
bfloat16x4_t test_vreinterpret_bf16_s32(int32x2_t a)    { return vreinterpret_bf16_s32(a);   }
// CHECK-LABEL: @test_vreinterpret_bf16_f32(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x float> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_f32(float32x2_t a)  { return vreinterpret_bf16_f32(a);   }
// CHECK-LABEL: @test_vreinterpret_bf16_u8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i8> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_u8(uint8x8_t a)     { return vreinterpret_bf16_u8(a);    }
// CHECK-LABEL: @test_vreinterpret_bf16_u16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_u16(uint16x4_t a)   { return vreinterpret_bf16_u16(a);   }
// CHECK-LABEL: @test_vreinterpret_bf16_u32(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x i32> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[A:%.*]]
//
bfloat16x4_t test_vreinterpret_bf16_u32(uint32x2_t a)   { return vreinterpret_bf16_u32(a);   }
// CHECK-LABEL: @test_vreinterpret_bf16_p8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i8> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_p8(poly8x8_t a)     { return vreinterpret_bf16_p8(a);    }
// CHECK-LABEL: @test_vreinterpret_bf16_p16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x i16> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_p16(poly16x4_t a)   { return vreinterpret_bf16_p16(a);   }
// CHECK-LABEL: @test_vreinterpret_bf16_u64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <1 x i64> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_u64(uint64x1_t a)   { return vreinterpret_bf16_u64(a);   }
// CHECK-LABEL: @test_vreinterpret_bf16_s64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <1 x i64> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_s64(int64x1_t a)    { return vreinterpret_bf16_s64(a);   }
// CHECK-LABEL: @test_vreinterpretq_bf16_s8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_s8(int8x16_t a)    { return vreinterpretq_bf16_s8(a);   }
// CHECK-LABEL: @test_vreinterpretq_bf16_s16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i16> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_s16(int16x8_t a)   { return vreinterpretq_bf16_s16(a);  }
// CHECK-LABEL: @test_vreinterpretq_bf16_s32(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x i32> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[A:%.*]]
//
bfloat16x8_t test_vreinterpretq_bf16_s32(int32x4_t a)   { return vreinterpretq_bf16_s32(a);  }
// CHECK-LABEL: @test_vreinterpretq_bf16_f32(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x float> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_f32(float32x4_t a) { return vreinterpretq_bf16_f32(a);  }
// CHECK-LABEL: @test_vreinterpretq_bf16_u8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_u8(uint8x16_t a)   { return vreinterpretq_bf16_u8(a);   }
// CHECK-LABEL: @test_vreinterpretq_bf16_u16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i16> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_u16(uint16x8_t a)  { return vreinterpretq_bf16_u16(a);  }
// CHECK-LABEL: @test_vreinterpretq_bf16_u32(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x i32> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[A:%.*]]
//
bfloat16x8_t test_vreinterpretq_bf16_u32(uint32x4_t a)  { return vreinterpretq_bf16_u32(a);  }
// CHECK-LABEL: @test_vreinterpretq_bf16_p8(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <16 x i8> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_p8(poly8x16_t a)   { return vreinterpretq_bf16_p8(a);   }
// CHECK-LABEL: @test_vreinterpretq_bf16_p16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x i16> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_p16(poly16x8_t a)  { return vreinterpretq_bf16_p16(a);  }
// CHECK-LABEL: @test_vreinterpretq_bf16_u64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x i64> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_u64(uint64x2_t a)  { return vreinterpretq_bf16_u64(a);  }
// CHECK-LABEL: @test_vreinterpretq_bf16_s64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x i64> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_s64(int64x2_t a)   { return vreinterpretq_bf16_s64(a);  }
// CHECK-LABEL: @test_vreinterpret_bf16_p64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <1 x i64> [[A:%.*]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[TMP0]]
//
bfloat16x4_t test_vreinterpret_bf16_p64(poly64x1_t a)   { return vreinterpret_bf16_p64(a);   }
// CHECK-LABEL: @test_vreinterpretq_bf16_p64(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <2 x i64> [[A:%.*]] to <8 x bfloat>
// CHECK-NEXT:    ret <8 x bfloat> [[TMP0]]
//
bfloat16x8_t test_vreinterpretq_bf16_p64(poly64x2_t a)  { return vreinterpretq_bf16_p64(a);  }

// TODO: poly128_t not implemented on aarch32
// CHCK-LABEL: @test_vreinterpretq_bf16_p128(
// CHCK-NEXT:  entry:
// CHCK-NEXT:    [[TMP0:%.*]] = bitcast i128 [[A:%.*]] to <4 x i32>
// CHCK-NEXT:    ret <4 x i32> [[TMP0]]
//
//bfloat16x8_t test_vreinterpretq_bf16_p128(poly128_t a)  { return vreinterpretq_bf16_p128(a); }

// CHECK-LABEL: @test_vreinterpret_s8_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <8 x i8>
// CHECK-NEXT:    ret <8 x i8> [[TMP0]]
//
int8x8_t    test_vreinterpret_s8_bf16(bfloat16x4_t a)    { return vreinterpret_s8_bf16(a);    }
// CHECK-LABEL: @test_vreinterpret_s16_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <4 x i16>
// CHECK-NEXT:    ret <4 x i16> [[TMP0]]
//
int16x4_t   test_vreinterpret_s16_bf16(bfloat16x4_t a)   { return vreinterpret_s16_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_s32_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <2 x i32>
// CHECK-NEXT:    ret <2 x i32> [[A:%.*]]
//
int32x2_t   test_vreinterpret_s32_bf16(bfloat16x4_t a)   { return vreinterpret_s32_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_f32_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <2 x float>
// CHECK-NEXT:    ret <2 x float> [[TMP0]]
//
float32x2_t test_vreinterpret_f32_bf16(bfloat16x4_t a)   { return vreinterpret_f32_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_u8_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <8 x i8>
// CHECK-NEXT:    ret <8 x i8> [[TMP0]]
//
uint8x8_t   test_vreinterpret_u8_bf16(bfloat16x4_t a)    { return vreinterpret_u8_bf16(a);    }
// CHECK-LABEL: @test_vreinterpret_u16_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <4 x i16>
// CHECK-NEXT:    ret <4 x i16> [[TMP0]]
//
uint16x4_t  test_vreinterpret_u16_bf16(bfloat16x4_t a)   { return vreinterpret_u16_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_u32_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <2 x i32>
// CHECK-NEXT:    ret <2 x i32> [[A:%.*]]
//
uint32x2_t  test_vreinterpret_u32_bf16(bfloat16x4_t a)   { return vreinterpret_u32_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_p8_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <8 x i8>
// CHECK-NEXT:    ret <8 x i8> [[TMP0]]
//
poly8x8_t   test_vreinterpret_p8_bf16(bfloat16x4_t a)    { return vreinterpret_p8_bf16(a);    }
// CHECK-LABEL: @test_vreinterpret_p16_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <4 x i16>
// CHECK-NEXT:    ret <4 x i16> [[TMP0]]
//
poly16x4_t  test_vreinterpret_p16_bf16(bfloat16x4_t a)   { return vreinterpret_p16_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_u64_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <1 x i64>
// CHECK-NEXT:    ret <1 x i64> [[TMP0]]
//
uint64x1_t  test_vreinterpret_u64_bf16(bfloat16x4_t a)   { return vreinterpret_u64_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_s64_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <1 x i64>
// CHECK-NEXT:    ret <1 x i64> [[TMP0]]
//
int64x1_t   test_vreinterpret_s64_bf16(bfloat16x4_t a)   { return vreinterpret_s64_bf16(a);   }
// CHECK-LABEL: @test_vreinterpret_p64_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <4 x bfloat> [[A:%.*]] to <1 x i64>
// CHECK-NEXT:    ret <1 x i64> [[TMP0]]
//
poly64x1_t  test_vreinterpret_p64_bf16(bfloat16x4_t a)   { return vreinterpret_p64_bf16(a);   }
// CHECK-LABEL: @test_vreinterpretq_s8_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP0]]
//
int8x16_t   test_vreinterpretq_s8_bf16(bfloat16x8_t a)   { return vreinterpretq_s8_bf16(a);   }
// CHECK-LABEL: @test_vreinterpretq_s16_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <8 x i16>
// CHECK-NEXT:    ret <8 x i16> [[TMP0]]
//
int16x8_t   test_vreinterpretq_s16_bf16(bfloat16x8_t a)  { return vreinterpretq_s16_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_s32_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <4 x i32>
// CHECK-NEXT:    ret <4 x i32> [[A:%.*]]
//
int32x4_t   test_vreinterpretq_s32_bf16(bfloat16x8_t a)  { return vreinterpretq_s32_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_f32_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <4 x float>
// CHECK-NEXT:    ret <4 x float> [[TMP0]]
//
float32x4_t test_vreinterpretq_f32_bf16(bfloat16x8_t a)  { return vreinterpretq_f32_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_u8_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP0]]
//
uint8x16_t  test_vreinterpretq_u8_bf16(bfloat16x8_t a)   { return vreinterpretq_u8_bf16(a);   }
// CHECK-LABEL: @test_vreinterpretq_u16_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <8 x i16>
// CHECK-NEXT:    ret <8 x i16> [[TMP0]]
//
uint16x8_t  test_vreinterpretq_u16_bf16(bfloat16x8_t a)  { return vreinterpretq_u16_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_u32_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <4 x i32>
// CHECK-NEXT:    ret <4 x i32> [[A:%.*]]
//
uint32x4_t  test_vreinterpretq_u32_bf16(bfloat16x8_t a)  { return vreinterpretq_u32_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_p8_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <16 x i8>
// CHECK-NEXT:    ret <16 x i8> [[TMP0]]
//
poly8x16_t  test_vreinterpretq_p8_bf16(bfloat16x8_t a)   { return vreinterpretq_p8_bf16(a);   }
// CHECK-LABEL: @test_vreinterpretq_p16_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <8 x i16>
// CHECK-NEXT:    ret <8 x i16> [[TMP0]]
//
poly16x8_t  test_vreinterpretq_p16_bf16(bfloat16x8_t a)  { return vreinterpretq_p16_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_u64_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <2 x i64>
// CHECK-NEXT:    ret <2 x i64> [[TMP0]]
//
uint64x2_t  test_vreinterpretq_u64_bf16(bfloat16x8_t a)  { return vreinterpretq_u64_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_s64_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <2 x i64>
// CHECK-NEXT:    ret <2 x i64> [[TMP0]]
//
int64x2_t   test_vreinterpretq_s64_bf16(bfloat16x8_t a)  { return vreinterpretq_s64_bf16(a);  }
// CHECK-LABEL: @test_vreinterpretq_p64_bf16(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = bitcast <8 x bfloat> [[A:%.*]] to <2 x i64>
// CHECK-NEXT:    ret <2 x i64> [[TMP0]]
//
poly64x2_t  test_vreinterpretq_p64_bf16(bfloat16x8_t a)  { return vreinterpretq_p64_bf16(a);  }

// TODO: poly128_t not implemented on aarch32
// CHCK-LABEL: @test_vreinterpretq_p128_bf16(
// CHCK-NEXT:  entry:
// CHCK-NEXT:    [[TMP0:%.*]] = bitcast <4 x i32> [[A:%.*]] to i128
// CHCK-NEXT:    ret i128 [[TMP0]]
//
//poly128_t   test_vreinterpretq_p128_bf16(bfloat16x8_t a) { return vreinterpretq_p128_bf16(a); }
