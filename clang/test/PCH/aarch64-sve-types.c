// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -include-pch %t \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

__SVInt8_t *s8;
__SVInt16_t *s16;
__SVInt32_t *s32;
__SVInt64_t *s64;

__SVUint8_t *u8;
__SVUint16_t *u16;
__SVUint32_t *u32;
__SVUint64_t *u64;

__SVFloat16_t *f16;
__SVFloat32_t *f32;
__SVFloat64_t *f64;

__SVBFloat16_t *bf16;

__SVBool_t *b8;
