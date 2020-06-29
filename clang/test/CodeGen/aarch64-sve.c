// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:  -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s -check-prefix=CHECK-DEBUG
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:  -emit-llvm -o - %s 2>&1 | FileCheck %s -check-prefix=CHECK

// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVInt8_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVInt16_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVInt32_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVInt64_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVUint8_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVUint16_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVUint32_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVUint64_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVFloat16_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVFloat32_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVFloat64_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVBFloat16_t'
// CHECK-DEBUG: cannot yet generate debug info for SVE type '__SVBool_t'

// CHECK: @ptr = global <vscale x 16 x i8>* null, align 8
// CHECK: %s8 = alloca <vscale x 16 x i8>, align 16
// CHECK: %s16 = alloca <vscale x 8 x i16>, align 16
// CHECK: %s32 = alloca <vscale x 4 x i32>, align 16
// CHECK: %s64 = alloca <vscale x 2 x i64>, align 16
// CHECK: %u8 = alloca <vscale x 16 x i8>, align 16
// CHECK: %u16 = alloca <vscale x 8 x i16>, align 16
// CHECK: %u32 = alloca <vscale x 4 x i32>, align 16
// CHECK: %u64 = alloca <vscale x 2 x i64>, align 16
// CHECK: %f16 = alloca <vscale x 8 x half>, align 16
// CHECK: %f32 = alloca <vscale x 4 x float>, align 16
// CHECK: %f64 = alloca <vscale x 2 x double>, align 16
// CHECK: %bf16 = alloca <vscale x 8 x bfloat>, align 16
// CHECK: %b8 = alloca <vscale x 16 x i1>, align 2

__SVInt8_t *ptr;

void test_locals(void) {
  __SVInt8_t s8;
  __SVInt16_t s16;
  __SVInt32_t s32;
  __SVInt64_t s64;

  __SVUint8_t u8;
  __SVUint16_t u16;
  __SVUint32_t u32;
  __SVUint64_t u64;

  __SVFloat16_t f16;
  __SVFloat32_t f32;
  __SVFloat64_t f64;

  __SVBFloat16_t bf16;

  __SVBool_t b8;
}
