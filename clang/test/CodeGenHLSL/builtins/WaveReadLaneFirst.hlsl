// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -fnative-half-type -fnative-int16-type -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -fnative-half-type -fnative-int16-type -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// CHECK-LABEL: test_int
int test_int(int expr) {
  // CHECK-SPIRV: %[[#entry_tok0:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.readlane.first.i32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.first.i32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.readlane.first.i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.readlane.first.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint
uint test_uint(uint expr) {
  // CHECK-SPIRV: %[[#entry_tok0:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.readlane.first.i32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.first.i32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_int64_t
int64_t test_int64_t(int64_t expr) {
  // CHECK-SPIRV: %[[#entry_tok1:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.readlane.first.i64([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok1]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.first.i64([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.readlane.first.i64([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.readlane.first.i64([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t
uint64_t test_uint64_t(uint64_t expr) {
  // CHECK-SPIRV: %[[#entry_tok1:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.readlane.first.i64([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok1]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.first.i64([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-LABEL: test_int16
int16_t test_int16(int16_t expr) {
  // CHECK-SPIRV: %[[#entry_tok2:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.readlane.first.i16([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok2]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.first.i16([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.readlane.first.i16([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.readlane.first.i16([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint16
uint16_t test_uint16(uint16_t expr) {
  // CHECK-SPIRV: %[[#entry_tok2:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.readlane.first.i16([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok2]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.first.i16([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}
#endif

// CHECK-LABEL: test_bool
bool test_bool(bool expr) {
  // CHECK-SPIRV: %[[#entry_tok3:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call i1 @llvm.spv.wave.readlane.first.i1(i1 %{{[a-zA-Z0-9]+}}) [ "convergencectrl"(token %[[#entry_tok3]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call i1 @llvm.dx.wave.readlane.first.i1(i1 %{{[a-zA-Z0-9]+}})
  // CHECK: ret i1 %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_half
half test_half(half expr) {
  // CHECK-SPIRV: %[[#entry_tok4:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.spv.wave.readlane.first.f16([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok4]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.dx.wave.readlane.first.f16([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_double
double test_double(double expr) {
  // CHECK-SPIRV: %[[#entry_tok5:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.spv.wave.readlane.first.f64([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok5]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.dx.wave.readlane.first.f64([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_floatv4
float4 test_floatv4(float4 expr) {
  // CHECK-SPIRV: %[[#entry_tok6:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.spv.wave.readlane.first.v4f32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok6]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.dx.wave.readlane.first.v4f32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_float2x2
float2x2 test_float2x2(float2x2 expr) {
  // CHECK-SPIRV: %[[#entry_tok7:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.spv.wave.readlane.first.v4f32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok7]]) ]
  // CHECK-DXIL: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.dx.wave.readlane.first.v4f32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
