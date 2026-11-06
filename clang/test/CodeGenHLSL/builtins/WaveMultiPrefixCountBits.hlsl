// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header \
// RUN:   -triple dxil-pc-shadermodel6.5-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | \
// RUN:   opt -S -strip-convergence-intrinsics -dxil-op-lower | \
// RUN:   FileCheck %s --check-prefix=DXIL

// CHECK-LABEL: test_bool
// DXIL-LABEL: test_bool
uint test_bool(bool value, uint4 mask) {
  // CHECK: %[[TOKEN:.*]] = call token @llvm.experimental.convergence.entry()
  // CHECK: %[[VALUE:.*]] = icmp ne i32 %{{.*}}, 0
  // CHECK: %[[MASK:.*]] = load <4 x i32>
  // CHECK: %[[M0:.*]] = extractelement <4 x i32> %[[MASK]], i64 0
  // CHECK: %[[M1:.*]] = extractelement <4 x i32> %[[MASK]], i64 1
  // CHECK: %[[M2:.*]] = extractelement <4 x i32> %[[MASK]], i64 2
  // CHECK: %[[M3:.*]] = extractelement <4 x i32> %[[MASK]], i64 3
  // CHECK: %[[RET:.*]] = call i32 @llvm.dx.wave.multi.prefix.bit.count(i1 %[[VALUE]], i32 %[[M0]], i32 %[[M1]], i32 %[[M2]], i32 %[[M3]]) [ "convergencectrl"(token %[[TOKEN]]) ]
  // CHECK: ret i32 %[[RET]]
  // DXIL: call i32 @dx.op.waveMultiPrefixBitCount(i32 167, i1 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return WaveMultiPrefixCountBits(value, mask);
}

// CHECK: declare i32 @llvm.dx.wave.multi.prefix.bit.count(i1, i32, i32, i32, i32) #[[ATTR:[0-9]+]]

// CHECK-LABEL: test_true
// DXIL-LABEL: test_true
uint test_true() {
  // CHECK: call i32 @llvm.dx.wave.multi.prefix.bit.count(i1 true, i32 1, i32 2, i32 4, i32 8)
  // DXIL: call i32 @dx.op.waveMultiPrefixBitCount(i32 167, i1 true, i32 1, i32 2, i32 4, i32 8)
  return WaveMultiPrefixCountBits(true, uint4(1, 2, 4, 8));
}

// CHECK-LABEL: test_false
// DXIL-LABEL: test_false
uint test_false() {
  // CHECK: call i32 @llvm.dx.wave.multi.prefix.bit.count(i1 false, i32 16, i32 32, i32 64, i32 128)
  // DXIL: call i32 @dx.op.waveMultiPrefixBitCount(i32 167, i1 false, i32 16, i32 32, i32 64, i32 128)
  return __builtin_hlsl_wave_multi_prefix_count_bits(
      false, uint4(16, 32, 64, 128));
}

// CHECK: attributes #[[ATTR]] = { convergent {{.*}}memory(none) }
