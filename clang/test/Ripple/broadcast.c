// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -S -Wpedantic -O2 -fenable-ripple -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: %clang -x c++ -S -Wpedantic -O2 -fenable-ripple -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

#ifdef __cplusplus
extern "C" {
#endif

// CHECK-LABEL: define{{.*}}void @checkScalarBcast
// CHECK-SAME: i32 {{.*}} %[[I32:[0-9a-z]+]],
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Insert:[a-zA-Z0-9_.]+]] = insertelement <32 x i32> poison, i32 %[[I32]], i64 0
// CHECK: %[[Bcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i32> %[[Insert]], <32 x i32> poison, <32 x i32> zeroinitializer
// CHECK: store <32 x i32> %[[Bcast]], ptr %[[PTR]]
void checkScalarBcast(int32_t in, int32_t *tmp) {
  ripple_block_t BS = ripple_set_block_shape(0, 16, 2);
  int32_t x = ripple_broadcast(BS, 0b11, in);
  tmp[ripple_id(BS, 0) + ripple_id(BS, 1) * ripple_get_block_size(BS, 0)] = x;
}

// CHECK-LABEL: define{{.*}}void @checkCstBcast
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: store <32 x i32> splat (i32 42), ptr %[[PTR]]
void checkCstBcast(int32_t *tmp) {
  ripple_block_t BS = ripple_set_block_shape(0, 16, 2);
  int32_t x = ripple_broadcast(BS, 0b11, (int32_t)42);
  tmp[ripple_id(BS, 0) + ripple_id(BS, 1) * ripple_get_block_size(BS, 0)] = x;
}

// Pointer bcast tests
#define pointer_bcast_test(T) \
void pointer_##T(int size, T *a) { \
  ripple_block_t BS = ripple_set_block_shape(0, 32, 4); \
  size_t BlockX = ripple_id(BS, 0), BlockY = ripple_id(BS, 1); \
  a[BlockX + ripple_get_block_size(BS, 0) * BlockY] = *(ripple_broadcast_ptr(BS, 0x2, a + BlockX)); \
}

// CHECK-LABEL: pointer_float
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load float, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x float> poison, float [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x float> [[Splat]], <128 x float> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x float> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(float)

// CHECK-LABEL: pointer_double
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load double, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x double> poison, double [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x double> [[Splat]], <128 x double> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x double> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(double)

// CHECK-LABEL: pointer_int8_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i8, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i8> poison, i8 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i8> [[Splat]], <128 x i8> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i8> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(int8_t)

// CHECK-LABEL: pointer_int16_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i16, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i16> poison, i16 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i16> [[Splat]], <128 x i16> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i16> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(int16_t)

// CHECK-LABEL: pointer_int32_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i32, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i32> poison, i32 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i32> [[Splat]], <128 x i32> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i32> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(int32_t)

// CHECK-LABEL: pointer_int64_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i64, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i64> poison, i64 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i64> [[Splat]], <128 x i64> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i64> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(int64_t)

// CHECK-LABEL: pointer_uint8_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i8, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i8> poison, i8 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i8> [[Splat]], <128 x i8> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i8> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(uint8_t)

// CHECK-LABEL: pointer_uint16_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i16, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i16> poison, i16 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i16> [[Splat]], <128 x i16> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i16> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(uint16_t)

// CHECK-LABEL: pointer_uint32_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i32, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i32> poison, i32 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i32> [[Splat]], <128 x i32> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i32> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(uint32_t)

// CHECK-LABEL: pointer_uint64_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: [[Load:%.*]] = load i64, ptr %[[PTR]]
// CHECK: [[Splat:%.*]] = insertelement <128 x i64> poison, i64 [[Load]], i64 0
// CHECK: [[Bcast:%.*]] = shufflevector <128 x i64> [[Splat]], <128 x i64> poison, <128 x i32> zeroinitializer
// CHECK: store <128 x i64> [[Bcast]], ptr %[[PTR]]
pointer_bcast_test(uint64_t)

// Scalar bcast tests
#define bcast_scalar_test(T) \
void scalar_##T(int size, T *a) { \
  ripple_block_t BS = ripple_set_block_shape(0, 32, 4); \
  size_t BlockX = ripple_id(BS, 0), BlockY = ripple_id(BS, 1); \
  a[BlockX + BlockY * ripple_get_block_size(BS, 0)] = ripple_broadcast(BS, 0x2, a[BlockX]); \
}

// CHECK-LABEL: scalar_float
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x float>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x float> %[[Val]]
bcast_scalar_test(float)

// CHECK-LABEL: scalar_double
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x double>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x double> %[[Val]]
bcast_scalar_test(double)

// CHECK-LABEL: scalar_int8_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i8>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i8> %[[Val]]
bcast_scalar_test(int8_t)

// CHECK-LABEL: scalar_int16_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i16>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i16> %[[Val]]
bcast_scalar_test(int16_t)

// CHECK-LABEL: scalar_int32_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i32>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i32> %[[Val]]
bcast_scalar_test(int32_t)

// CHECK-LABEL: scalar_int64_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i64>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i64> %[[Val]]
bcast_scalar_test(int64_t)

// CHECK-LABEL: scalar_uint8_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i8>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i8> %[[Val]]
bcast_scalar_test(uint8_t)

// CHECK-LABEL: scalar_uint16_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i16>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i16> %[[Val]]
bcast_scalar_test(uint16_t)

// CHECK-LABEL: scalar_uint32_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i32>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i32> %[[Val]]
bcast_scalar_test(uint32_t)

// CHECK-LABEL: scalar_uint64_t
// CHECK-SAME: ptr {{.*}} %[[PTR:[0-9a-z]+]]
// CHECK: %[[Val:[a-zA-Z0-9_.]+]] = load <32 x i64>, ptr %[[PTR]]
// CHECK: %[[ValBcast:[a-zA-Z0-9_.]+]] = shufflevector <32 x i64> %[[Val]]
bcast_scalar_test(uint64_t)

#ifdef __cplusplus
} // extern "C"
#endif
