// RUN: fir-opt --target-rewrite="target=nvptx64-nvidia-cuda skip-host-abi-conversion" %s | FileCheck %s

// NVPTX does not implement complex argument or return marshalling. Host-only
// functions in an OpenMP device module are removed after lowering, so their
// signatures must be left unchanged instead of reaching the NVPTX ABI.
module attributes {omp.is_target_device = true} {
  // CHECK-LABEL: func.func @host_complex(
  // CHECK-SAME: complex<f64>) -> complex<f64>
  func.func @host_complex(%arg: complex<f64>) -> complex<f64> {
    // CHECK: fir.call @csqrt({{.*}}) : (complex<f64>) -> complex<f64>
    %result = fir.call @csqrt(%arg) : (complex<f64>) -> complex<f64>
    // CHECK: fir.address_of(@csqrt) : (complex<f64>) -> complex<f64>
    %addr = fir.address_of(@csqrt) : (complex<f64>) -> complex<f64>
    // CHECK: fir.call %{{.*}}({{.*}}) : (complex<f64>) -> complex<f64>
    %indirect = fir.call %addr(%result) : (complex<f64>) -> complex<f64>
    return %indirect : complex<f64>
  }

  // CHECK-LABEL: func.func private @csqrt(complex<f64>) -> complex<f64>
  func.func private @csqrt(complex<f64>) -> complex<f64>
}
