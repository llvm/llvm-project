// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -fnative-half-type -fnative-int16-type -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DTARGET=dx
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -fnative-half-type -fnative-int16-type -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s -DTARGET=spv

// CHECK-LABEL: test_int
int test_int(int expr) {
  // CHECK: %[[#entry_tok0:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.i32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK: declare [[TY]] @llvm.[[TARGET]].wave.readlane.first.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint
uint test_uint(uint expr) {
  // CHECK: %[[#entry_tok0:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.i32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_int64_t
int64_t test_int64_t(int64_t expr) {
  // CHECK: %[[#entry_tok1:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.i64([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok1]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK: declare [[TY]] @llvm.[[TARGET]].wave.readlane.first.i64([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t
uint64_t test_uint64_t(uint64_t expr) {
  // CHECK: %[[#entry_tok1:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.i64([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok1]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-LABEL: test_int16
int16_t test_int16(int16_t expr) {
  // CHECK: %[[#entry_tok2:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.i16([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok2]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK: declare [[TY]] @llvm.[[TARGET]].wave.readlane.first.i16([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint16
uint16_t test_uint16(uint16_t expr) {
  // CHECK: %[[#entry_tok2:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.i16([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok2]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}
#endif

// CHECK-LABEL: test_bool
bool test_bool(bool expr) {
  // CHECK: %[[#entry_tok3:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call i1 @llvm.[[TARGET]].wave.readlane.first.i1(i1 %{{[a-zA-Z0-9]+}}) [ "convergencectrl"(token %[[#entry_tok3]]) ]
  // CHECK: ret i1 %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_half
half test_half(half expr) {
  // CHECK: %[[#entry_tok4:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.f16([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok4]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_double
double test_double(double expr) {
  // CHECK: %[[#entry_tok5:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.f64([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok5]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_floatv4
float4 test_floatv4(float4 expr) {
  // CHECK: %[[#entry_tok6:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.v4f32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok6]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_floatv5
vector<float, 5> test_floatv5(vector<float, 5> expr) {
  // CHECK: %[[#entry_tok7:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.v5f32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok7]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_float2x3
float2x3 test_float2x3(float2x3 expr) {
  // CHECK: %[[#entry_tok8:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.v6f32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok8]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK-LABEL: test_float3x4
float3x4 test_float3x4(float3x4 expr) {
  // CHECK: %[[#entry_tok9:]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.[[TARGET]].wave.readlane.first.v12f32([[TY]] %[[#]]) [ "convergencectrl"(token %[[#entry_tok9]]) ]
  // CHECK: ret [[TY]] %[[RET]]
  return WaveReadLaneFirst(expr);
}

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
