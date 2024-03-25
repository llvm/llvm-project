// RUN: %clang_cc1 -fclang-abi-compat=latest -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -fclang-abi-compat=latest -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +sve,+bf16 | FileCheck %s

namespace std { class type_info; };

auto &s8 = typeid(__SVInt8_t);
auto &s16 = typeid(__SVInt16_t);
auto &s32 = typeid(__SVInt32_t);
auto &s64 = typeid(__SVInt64_t);

auto &u8 = typeid(__SVUint8_t);
auto &u16 = typeid(__SVUint16_t);
auto &u32 = typeid(__SVUint32_t);
auto &u64 = typeid(__SVUint64_t);

auto &f16 = typeid(__SVFloat16_t);
auto &f32 = typeid(__SVFloat32_t);
auto &f64 = typeid(__SVFloat64_t);

auto &bf16 = typeid(__SVBfloat16_t);

auto &b8 = typeid(__SVBool_t);
auto &c8 = typeid(__SVCount_t);

// CHECK-DAG: @_ZTSu10__SVInt8_t = {{.*}} c"u10__SVInt8_t\00"
// CHECK-DAG: @_ZTIu10__SVInt8_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu10__SVInt8_t

// CHECK-DAG: @_ZTSu11__SVInt16_t = {{.*}} c"u11__SVInt16_t\00"
// CHECK-DAG: @_ZTIu11__SVInt16_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu11__SVInt16_t

// CHECK-DAG: @_ZTSu11__SVInt32_t = {{.*}} c"u11__SVInt32_t\00"
// CHECK-DAG: @_ZTIu11__SVInt32_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu11__SVInt32_t

// CHECK-DAG: @_ZTSu11__SVInt64_t = {{.*}} c"u11__SVInt64_t\00"
// CHECK-DAG: @_ZTIu11__SVInt64_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu11__SVInt64_t

// CHECK-DAG: @_ZTSu11__SVUint8_t = {{.*}} c"u11__SVUint8_t\00"
// CHECK-DAG: @_ZTIu11__SVUint8_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu11__SVUint8_t

// CHECK-DAG: @_ZTSu12__SVUint16_t = {{.*}} c"u12__SVUint16_t\00"
// CHECK-DAG: @_ZTIu12__SVUint16_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu12__SVUint16_t

// CHECK-DAG: @_ZTSu12__SVUint32_t = {{.*}} c"u12__SVUint32_t\00"
// CHECK-DAG: @_ZTIu12__SVUint32_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu12__SVUint32_t

// CHECK-DAG: @_ZTSu12__SVUint64_t = {{.*}} c"u12__SVUint64_t\00"
// CHECK-DAG: @_ZTIu12__SVUint64_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu12__SVUint64_t

// CHECK-DAG: @_ZTSu13__SVFloat16_t = {{.*}} c"u13__SVFloat16_t\00"
// CHECK-DAG: @_ZTIu13__SVFloat16_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu13__SVFloat16_t

// CHECK-DAG: @_ZTSu13__SVFloat32_t = {{.*}} c"u13__SVFloat32_t\00"
// CHECK-DAG: @_ZTIu13__SVFloat32_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu13__SVFloat32_t

// CHECK-DAG: @_ZTSu13__SVFloat64_t = {{.*}} c"u13__SVFloat64_t\00"
// CHECK-DAG: @_ZTIu13__SVFloat64_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu13__SVFloat64_t
//
// CHECK-DAG: @_ZTSu14__SVBfloat16_t = {{.*}} c"u14__SVBfloat16_t\00"
// CHECK-DAG: @_ZTIu14__SVBfloat16_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu14__SVBfloat16_t

// CHECK-DAG: @_ZTSu10__SVBool_t = {{.*}} c"u10__SVBool_t\00"
// CHECK-DAG: @_ZTIu10__SVBool_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu10__SVBool_t

// CHECK-DAG: @_ZTSu11__SVCount_t = {{.*}} c"u11__SVCount_t\00"
// CHECK-DAG: @_ZTIu11__SVCount_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu11__SVCount_t
