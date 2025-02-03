// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -O1 -o - | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: D3DCOLORtoUBYTE4
int4 test_D3DCOLORtoUBYTE4(float4 p1) {
  // CHECK: %[[SCALED:.*]] = fmul [[FMFLAGS:.*]][[FLOAT_TYPE:<4 x float>]] %{{.*}}, splat (float 0x406FE01000000000)
  // CHECK: %[[CONVERTED:.*]] = fptoui [[FLOAT_TYPE]] %[[SCALED]] to [[INT_TYPE:<4 x i32>]]
  // CHECK: %[[SHUFFLED:.*]] = shufflevector [[INT_TYPE]] %[[CONVERTED]], [[INT_TYPE]] poison, <4 x i32> <i32 2, i32 1, i32 0, i32 3>
  // CHECK: ret [[INT_TYPE]] %[[SHUFFLED]]
  return D3DCOLORtoUBYTE4(p1);
}
