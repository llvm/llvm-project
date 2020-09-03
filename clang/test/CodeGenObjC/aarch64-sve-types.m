// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +sve,+bf16 2>&1 | FileCheck %s

// CHECK: error: cannot yet @encode type __SVInt8_t
const char s8[] = @encode(__SVInt8_t);
// CHECK: error: cannot yet @encode type __SVInt16_t
const char s16[] = @encode(__SVInt16_t);
// CHECK: error: cannot yet @encode type __SVInt32_t
const char s32[] = @encode(__SVInt32_t);
// CHECK: error: cannot yet @encode type __SVInt64_t
const char s64[] = @encode(__SVInt64_t);

// CHECK: error: cannot yet @encode type __SVUint8_t
const char u8[] = @encode(__SVUint8_t);
// CHECK: error: cannot yet @encode type __SVUint16_t
const char u16[] = @encode(__SVUint16_t);
// CHECK: error: cannot yet @encode type __SVUint32_t
const char u32[] = @encode(__SVUint32_t);
// CHECK: error: cannot yet @encode type __SVUint64_t
const char u64[] = @encode(__SVUint64_t);

// CHECK: error: cannot yet @encode type __SVFloat16_t
const char f16[] = @encode(__SVFloat16_t);
// CHECK: error: cannot yet @encode type __SVFloat32_t
const char f32[] = @encode(__SVFloat32_t);
// CHECK: error: cannot yet @encode type __SVFloat64_t
const char f64[] = @encode(__SVFloat64_t);

// CHECK: error: cannot yet @encode type __SVBFloat16_t
const char bf16[] = @encode(__SVBFloat16_t);

// CHECK: error: cannot yet @encode type __SVBool_t
const char b8[] = @encode(__SVBool_t);
