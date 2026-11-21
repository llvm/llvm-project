// RUN: fir-opt --omp-mark-declare-target --target-rewrite="target=amdgcn-amd-amdhsa skip-host-abi-conversion" %s | FileCheck %s

// Host-only signatures, calls, and function addresses must not undergo device
// ABI rewriting. Calls to device-visible functions must still agree with their
// rewritten signatures, including calls made from host-only functions.
module attributes {omp.is_target_device = true} {
  // CHECK-LABEL: func.func @host_only(
  // CHECK-SAME: complex<f128>
  // CHECK-SAME: -> complex<f128>
  func.func @host_only(%arg: complex<f128>) -> complex<f128> {
    // CHECK: fir.call @host_callee({{.*}}) : (complex<f128>) -> complex<f128>
    %result = fir.call @host_callee(%arg) : (complex<f128>) -> complex<f128>
    // CHECK: fir.address_of(@host_callee) : (complex<f128>) -> complex<f128>
    %addr = fir.address_of(@host_callee) : (complex<f128>) -> complex<f128>
    // CHECK: fir.call %{{.*}}({{.*}}) : (complex<f128>) -> complex<f128>
    %indirect = fir.call %addr(%result) : (complex<f128>) -> complex<f128>
    return %indirect : complex<f128>
  }

  // CHECK-LABEL: func.func private @host_callee(complex<f128>) -> complex<f128>
  func.func private @host_callee(complex<f128>) -> complex<f128>

  // CHECK-LABEL: func.func @explicit_host(
  // CHECK-SAME: complex<f128>
  // CHECK-SAME: -> complex<f128>
  func.func @explicit_host(%arg: complex<f128>) -> complex<f128> attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    return %arg : complex<f128>
  }

  // CHECK-LABEL: func.func @host_calls_device(
  func.func @host_calls_device(%arg: complex<f64>) {
    // CHECK: fir.call @device_callee({{.*}}, {{.*}}) : (f64, f64) -> tuple<f64, f64>
    %result = fir.call @device_callee(%arg) : (complex<f64>) -> complex<f64>
    // CHECK: fir.address_of(@device_callee) : (f64, f64) -> tuple<f64, f64>
    %addr = fir.address_of(@device_callee) : (complex<f64>) -> complex<f64>
    return
  }

  // CHECK-LABEL: func.func @device_callee(
  // CHECK-SAME: f64, {{.*}}: f64) -> tuple<f64, f64>
  func.func @device_callee(%arg: complex<f64>) -> complex<f64> attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    return %arg : complex<f64>
  }

  // CHECK-LABEL: func.func @device_indirect(
  func.func @device_indirect(%addr: () -> (), %arg: complex<f64>) -> complex<f64> attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>} {
    %callee = fir.convert %addr : (() -> ()) -> ((complex<f64>) -> complex<f64>)
    // CHECK: fir.call %{{.*}}({{.*}}, {{.*}}) : (f64, f64) -> tuple<f64, f64>
    %result = fir.call %callee(%arg) : (complex<f64>) -> complex<f64>
    return %result : complex<f64>
  }

  // CHECK-LABEL: func.func @host_with_target(
  func.func @host_with_target() {
    omp.target kernel_type(generic) {
      %addr = fir.undefined () -> ()
      %arg = fir.undefined complex<f64>
      %callee = fir.convert %addr : (() -> ()) -> ((complex<f64>) -> complex<f64>)
      // CHECK: fir.call %{{.*}}({{.*}}, {{.*}}) : (f64, f64) -> tuple<f64, f64>
      %result = fir.call %callee(%arg) : (complex<f64>) -> complex<f64>
      omp.terminator
    }
    return
  }

  omp.declare_reduction @host_reduce : i32 init {
  ^bb0(%value: i32):
    %addr = fir.undefined () -> ()
    %quad = fir.undefined complex<f128>
    %callee = fir.convert %addr : (() -> ()) -> ((complex<f128>) -> f128)
    // CHECK: fir.call %{{.*}}({{.*}}) : (complex<f128>) -> f128
    %abs = fir.call %callee(%quad) : (complex<f128>) -> f128
    omp.yield(%value : i32)
  } combiner {
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    omp.yield(%sum : i32)
  }

  func.func @host_uses_reduction(%value: !fir.ref<i32>) {
    omp.parallel reduction(@host_reduce %value -> %arg : !fir.ref<i32>) {
      omp.terminator
    }
    return
  }

  func.func @host_target_in_reduction(%value: !fir.ref<i32>) {
    %map = omp.map.info var_ptr(%value : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) name("value") -> !fir.ref<i32>
    omp.target kernel_type(generic) in_reduction(@host_reduce %value : !fir.ref<i32>) map_entries(%map -> %arg : !fir.ref<i32>) {
      omp.terminator
    }
    return
  }

  omp.declare_reduction @device_reduce : i32 init {
  ^bb0(%value: i32):
    %addr = fir.undefined () -> ()
    %complex = fir.undefined complex<f64>
    %callee = fir.convert %addr : (() -> ()) -> ((complex<f64>) -> f64)
    // CHECK: fir.call %{{.*}}({{.*}}, {{.*}}) : (f64, f64) -> f64
    %result = fir.call %callee(%complex) : (complex<f64>) -> f64
    omp.yield(%value : i32)
  } combiner {
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    omp.yield(%sum : i32)
  }

  func.func @device_uses_reduction(%value: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to>} {
    omp.parallel reduction(@device_reduce %value -> %arg : !fir.ref<i32>) {
      omp.terminator
    }
    return
  }

  func.func @private_callee(%arg: complex<f64>) -> complex<f64> attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} {
    return %arg : complex<f64>
  }

  omp.private {type = private} @target_private_recipe : !fir.ref<i32> init {
  ^bb0(%orig: !fir.ref<i32>, %priv: !fir.ref<i32>):
    %addr = fir.address_of(@private_callee) : (complex<f64>) -> complex<f64>
    %arg = fir.undefined complex<f64>
    // CHECK: fir.call %{{.*}}({{.*}}, {{.*}}) : (f64, f64) -> tuple<f64, f64>
    %result = fir.call %addr(%arg) : (complex<f64>) -> complex<f64>
    omp.yield(%priv : !fir.ref<i32>)
  }

  func.func @host_uses_target_private(%value: !fir.ref<i32>) {
    // The recipe is referenced directly by omp.target's private clause.
    omp.target kernel_type(generic) private(@target_private_recipe %value -> %arg : !fir.ref<i32>) {
      omp.terminator
    }
    return
  }

  omp.declare_reduction @target_reduce : i32 init {
  ^bb0(%value: i32):
    %addr = fir.undefined () -> ()
    %complex = fir.undefined complex<f64>
    %callee = fir.convert %addr : (() -> ()) -> ((complex<f64>) -> f64)
    // CHECK: fir.call %{{.*}}({{.*}}, {{.*}}) : (f64, f64) -> f64
    %result = fir.call %callee(%complex) : (complex<f64>) -> f64
    omp.yield(%value : i32)
  } combiner {
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    omp.yield(%sum : i32)
  }

  func.func @host_target_uses_reduction(%value: !fir.ref<i32>) {
    omp.target kernel_type(generic) {
      %target_value = fir.undefined !fir.ref<i32>
      omp.parallel reduction(@target_reduce %target_value -> %arg : !fir.ref<i32>) {
        omp.terminator
      }
      omp.terminator
    }
    return
  }

  func.func @host_associated(%arg: complex<f128>, %host: !fir.ref<i32> {fir.host_assoc}) -> complex<f128> {
    // CHECK-LABEL: func.func @host_associated(
    // CHECK-SAME: complex<f128>
    // CHECK-SAME: fir.host_assoc
    return %arg : complex<f128>
  }
}
