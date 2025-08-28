// REQUIRES: hexagon-registered-target
// RUN: %clang --target=hexagon-unknown-elf -mv79 -mhvx=v79 %s -O2 -fenable-ripple -S -emit-llvm -o - | FileCheck %s
// RUN: %clang --target=hexagon-unknown-elf -mv79 -mhvx=v79 -x c %s -O2 -fenable-ripple -S -emit-llvm -o - -D__hexagon__=1 | FileCheck %s

#include <ripple.h>
#include <ripple_hvx.h>

#define values8 1, 2, 3, 4, 5, 6, 7, 8
#define values16 values8, values8
#define values32 values16, values16
#define values64 values32, values32
#define values128 values64, values64
#define values256 values128, values128
#define values512 values256, values256

#ifdef __cplusplus
extern "C" {
#endif

#define testCastToFrom(VecType, NumVal)                                        \
  void testFor##VecType(char *li) {                                            \
    VecType V = {values##NumVal};                                              \
    *(VecType *)li = hvx_cast_from_i32(hvx_cast_to_i32(V), V[0]);              \
  }

// CHECK: testForv128i8
// CHECK: store <128 x i8>
testCastToFrom(v128i8, 128);
// CHECK: testForv128u8
// CHECK: store <128 x i8>
testCastToFrom(v128u8, 128);

// CHECK: testForv64i16
// CHECK: store <64 x i16>
testCastToFrom(v64i16, 64);
// CHECK: testForv64u16
// CHECK: store <64 x i16>
testCastToFrom(v64u16, 64);

// CHECK: testForv32i32
// CHECK: store <32 x i32>
testCastToFrom(v32i32, 32);
// CHECK: testForv32u32
// CHECK: store <32 x i32>
testCastToFrom(v32u32, 32);

// CHECK: testForv32f32
// CHECK: store <32 x float>
testCastToFrom(v32f32, 32);

// CHECK: testForv16f64
// CHECK: store <16 x double>
testCastToFrom(v16f64, 16);

// CHECK: testForv64f16
// CHECK: store <64 x half>
testCastToFrom(v64f16, 64);

#define testHVXToRipple(ScalTy, VecType, NumVal)                               \
  ScalTy testHVXToRipple##VecType(char *li) {                                  \
    VecType V = {values##NumVal};                                              \
    ripple_block_t BS = ripple_set_block_shape(0, 1);                          \
    return hvx_to_ripple_##VecType(BS, *((VecType *)li));                      \
  }

// CHECK: testHVXToRipplev128i8
// CHECK: load <128 x i8>
testHVXToRipple(int8_t, v128i8, 128);
// CHECK: testHVXToRipplev128u8
// CHECK: load <128 x i8>
testHVXToRipple(uint8_t, v128u8, 128);
// CHECK: testHVXToRipplev256i8
// CHECK: load <256 x i8>
testHVXToRipple(int8_t, v256i8, 256);
// CHECK: testHVXToRipplev256u8
// CHECK: load <256 x i8>
testHVXToRipple(uint8_t, v256u8, 256);
// CHECK: testHVXToRipplev512i8
// CHECK: load <512 x i8>
testHVXToRipple(int8_t, v512i8, 512);
// CHECK: testHVXToRipplev512u8
// CHECK: load <512 x i8>
testHVXToRipple(uint8_t, v512u8, 512);

// CHECK: testHVXToRipplev64i16
// CHECK: load <64 x i16>
testHVXToRipple(int16_t, v64i16, 64);
// CHECK: testHVXToRipplev64u16
// CHECK: load <64 x i16>
testHVXToRipple(uint16_t, v64u16, 64);
// CHECK: testHVXToRipplev64f16
// CHECK: load <64 x half>
testHVXToRipple(_Float16, v64f16, 64);
// CHECK: testHVXToRipplev128i16
// CHECK: load <128 x i16>
testHVXToRipple(int16_t, v128i16, 128);
// CHECK: testHVXToRipplev128u16
// CHECK: load <128 x i16>
testHVXToRipple(uint16_t, v128u16, 128);
// CHECK: testHVXToRipplev128f16
// CHECK: load <128 x half>
testHVXToRipple(_Float16, v128f16, 128);
// CHECK: testHVXToRipplev256i16
// CHECK: load <256 x i16>
testHVXToRipple(int16_t, v256i16, 256);
// CHECK: testHVXToRipplev256u16
// CHECK: load <256 x i16>
testHVXToRipple(uint16_t, v256u16, 256);
// CHECK: testHVXToRipplev256f16
// CHECK: load <256 x half>
testHVXToRipple(_Float16, v256f16, 256);

// CHECK: testHVXToRipplev32i32
// CHECK: load <32 x i32>
testHVXToRipple(int32_t, v32i32, 32);
// CHECK: testHVXToRipplev32u32
// CHECK: load <32 x i32>
testHVXToRipple(uint32_t, v32u32, 32);
// CHECK: testHVXToRipplev32f32
// CHECK: load <32 x float>
testHVXToRipple(float, v32f32, 32);
// CHECK: testHVXToRipplev64i32
// CHECK: load <64 x i32>
testHVXToRipple(int32_t, v64i32, 64);
// CHECK: testHVXToRipplev64u32
// CHECK: load <64 x i32>
testHVXToRipple(uint32_t, v64u32, 64);
// CHECK: testHVXToRipplev64f32
// CHECK: load <64 x float>
testHVXToRipple(float, v64f32, 64);
// CHECK: testHVXToRipplev128i32
// CHECK: load <128 x i32>
testHVXToRipple(int32_t, v128i32, 128);
// CHECK: testHVXToRipplev128u32
// CHECK: load <128 x i32>
testHVXToRipple(uint32_t, v128u32, 128);
// CHECK: testHVXToRipplev128f32
// CHECK: load <128 x float>
testHVXToRipple(float, v128f32, 128);

// CHECK: testHVXToRipplev16i64
// CHECK: load <16 x i64>
testHVXToRipple(int64_t, v16i64, 16);
// CHECK: testHVXToRipplev16u64
// CHECK: load <16 x i64>
testHVXToRipple(uint64_t, v16u64, 16);
// CHECK: testHVXToRipplev16f64
// CHECK: load <16 x double>
testHVXToRipple(double, v16f64, 16);
// CHECK: testHVXToRipplev32i64
// CHECK: load <32 x i64>
testHVXToRipple(int64_t, v32i64, 32);
// CHECK: testHVXToRipplev32u64
// CHECK: load <32 x i64>
testHVXToRipple(uint64_t, v32u64, 32);
// CHECK: testHVXToRipplev32f64
// CHECK: load <32 x double>
testHVXToRipple(double, v32f64, 32);
// CHECK: testHVXToRipplev64i64
// CHECK: load <64 x i64>
testHVXToRipple(int64_t, v64i64, 64);
// CHECK: testHVXToRipplev64u64
// CHECK: load <64 x i64>
testHVXToRipple(uint64_t, v64u64, 64);
// CHECK: testHVXToRipplev64f64
// CHECK: load <64 x double>
testHVXToRipple(double, v64f64, 64);

#ifdef __cplusplus
#define testRippleToHVX(ScalTy, VecType, NumVal)                               \
  v##NumVal##VecType testRippleToHVXv##NumVal##VecType(char *li) {             \
    ripple_block_t BS = ripple_set_block_shape(0, NumVal);                     \
    ScalTy V = 42 + ripple_id(BS, 0);                                          \
    return ripple_to_hvx(BS, NumVal, VecType, V);                              \
  }                                                                            \
  v##NumVal##VecType testRippleToHVX2dv##NumVal##VecType(char *li) {           \
    ripple_block_t BS = ripple_set_block_shape(0, 2, NumVal / 2);              \
    ScalTy V = 42 + ripple_id(BS, 0) + ripple_id(BS, 1);                       \
    return ripple_to_hvx_2d(BS, NumVal, VecType, V);                           \
  }
#else // ! __cplusplus
#define testRippleToHVX(ScalTy, VecType, NumVal)                               \
  v##NumVal##VecType testRippleToHVXv##NumVal##VecType(char *li) {             \
    ripple_block_t BS = ripple_set_block_shape(0, NumVal);                     \
    ScalTy V = 42 + ripple_id(BS, 0);                                          \
    return ripple_to_hvx(BS, NumVal, VecType, V);                              \
  }                                                                            \
  v##NumVal##VecType testRippleToHVX2dv##NumVal##VecType(char *li) {           \
    ripple_block_t BS = ripple_set_block_shape(0, 2, NumVal / 2);              \
    ScalTy V = 42 + ripple_id(BS, 0) + ripple_id(BS, 1);                       \
    return ripple_to_hvx_2d(BS, NumVal, VecType, V);                           \
  }
#endif // __cplusplus

// CHECK: testRippleToHVXv128i8
// CHECK: ret <128 x i8>
// CHECK: ret <128 x i8>
testRippleToHVX(int8_t, i8, 128);
// CHECK: testRippleToHVXv128u8
// CHECK: ret <128 x i8>
// CHECK: ret <128 x i8>
testRippleToHVX(uint8_t, u8, 128);
// CHECK: testRippleToHVXv256i8
// CHECK: ret <256 x i8>
// CHECK: ret <256 x i8>
testRippleToHVX(int8_t, i8, 256);
// CHECK: testRippleToHVXv256u8
// CHECK: ret <256 x i8>
// CHECK: ret <256 x i8>
testRippleToHVX(uint8_t, u8, 256);

// CHECK: testRippleToHVXv64i16
// CHECK: ret <64 x i16>
// CHECK: ret <64 x i16>
testRippleToHVX(int16_t, i16, 64);
// CHECK: testRippleToHVXv64u16
// CHECK: ret <64 x i16>
// CHECK: ret <64 x i16>
testRippleToHVX(uint16_t, u16, 64);
// CHECK: testRippleToHVXv128i16
// CHECK: ret <128 x i16>
// CHECK: ret <128 x i16>
testRippleToHVX(int16_t, i16, 128);
// CHECK: testRippleToHVXv128u16
// CHECK: ret <128 x i16>
// CHECK: ret <128 x i16>
testRippleToHVX(uint16_t, u16, 128);

// CHECK: testRippleToHVXv64f16
// CHECK: ret <64 x half>
// CHECK: ret <64 x half>
testRippleToHVX(_Float16, f16, 64);
// CHECK: testRippleToHVXv128f16
// CHECK: ret <128 x half>
// CHECK: ret <128 x half>
testRippleToHVX(_Float16, f16, 128);

// CHECK: testRippleToHVXv32i32
// CHECK: ret <32 x i32>
// CHECK: ret <32 x i32>
testRippleToHVX(int32_t, i32, 32);
// CHECK: testRippleToHVXv32u32
// CHECK: ret <32 x i32>
// CHECK: ret <32 x i32>
testRippleToHVX(uint32_t, u32, 32);
// CHECK: testRippleToHVXv64i32
// CHECK: ret <64 x i32>
// CHECK: ret <64 x i32>
testRippleToHVX(int32_t, i32, 64);
// CHECK: testRippleToHVXv64u32
// CHECK: ret <64 x i32>
// CHECK: ret <64 x i32>
testRippleToHVX(uint32_t, u32, 64);

// CHECK: testRippleToHVXv32f32
// CHECK: ret <32 x float>
// CHECK: ret <32 x float>
testRippleToHVX(float, f32, 32);
// CHECK: testRippleToHVXv64f32
// CHECK: ret <64 x float>
// CHECK: ret <64 x float>
testRippleToHVX(float, f32, 64);

// CHECK: testRippleToHVXv16i64
// CHECK: ret <16 x i64>
// CHECK: ret <16 x i64>
testRippleToHVX(int64_t, i64, 16);
// CHECK: testRippleToHVXv16u64
// CHECK: ret <16 x i64>
// CHECK: ret <16 x i64>
testRippleToHVX(uint64_t, u64, 16);
// CHECK: testRippleToHVXv32i64
// CHECK: ret <32 x i64>
// CHECK: ret <32 x i64>
testRippleToHVX(int64_t, i64, 32);
// CHECK: testRippleToHVXv32u64
// CHECK: ret <32 x i64>
// CHECK: ret <32 x i64>
testRippleToHVX(uint64_t, u64, 32);

// CHECK: testRippleToHVXv16f64
// CHECK: ret <16 x double>
// CHECK: ret <16 x double>
testRippleToHVX(double, f64, 16);
// CHECK: testRippleToHVXv32f64
// CHECK: ret <32 x double>
// CHECK: ret <32 x double>
testRippleToHVX(double, f64, 32);

#ifdef __cplusplus
} // extern "C"
#endif
