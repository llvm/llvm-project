// RUN: fir-opt --target-rewrite %s | FileCheck %s

// CHECK-LABEL: func.func @libm_preserves_fmfs
func.func @libm_preserves_fmfs(%arg0 : complex<f32>) -> complex<f32> {
  %0 = fir.call @csin(%arg0) fastmath<contract> : (complex<f32>) -> complex<f32>
  // CHECK: fir.call @csin(%{{.+}}) fastmath<contract>
  return %0 : complex<f32>
}
