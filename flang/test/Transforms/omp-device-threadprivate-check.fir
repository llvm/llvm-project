// RUN: fir-opt --omp-device-threadprivate-check -split-input-file -verify-diagnostics %s | FileCheck %s

// Accesses to threadprivate variables in device code are diagnosed.

module attributes {omp.is_gpu = true, omp.is_target_device = true} {
  // expected-note @below {{declared here}}
  fir.global common @c_(dense<0> : vector<4xi8>) {alignment = 4 : i64} : !fir.array<4xi8>
  // expected-note @below {{declared here}}
  fir.global internal @_QFsave_varEy : i32

  func.func @common_member(%arg0: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>} {
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@c_) : !fir.ref<!fir.array<4xi8>>
    %1 = omp.threadprivate %0 : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<i32>
    %4 = fir.load %arg0 : !fir.ref<i32>
    // expected-error @below {{THREADPRIVATE common block '/c/' cannot be accessed in target device code}}
    fir.store %4 to %3 : !fir.ref<i32>
    return
  }

  func.func @save_var(%arg0: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = nohost, capture_clause = to, implicit = true>} {
    %0 = fir.address_of(@_QFsave_varEy) : !fir.ref<i32>
    %1 = omp.threadprivate %0 : !fir.ref<i32> -> !fir.ref<i32>
    // expected-error @below {{THREADPRIVATE variable 'y' cannot be accessed in target device code}}
    %2 = fir.load %1 : !fir.ref<i32>
    fir.store %2 to %arg0 : !fir.ref<i32>
    return
  }
}

// -----

// Threadprivate operations of variables that are not accessed in device code
// are removed. Host code is not modified.

module attributes {omp.is_gpu = true, omp.is_target_device = true} {
  fir.global common @c_(dense<0> : vector<4xi8>) {alignment = 4 : i64} : !fir.array<4xi8>
  fir.global @_QMmEmv : i32

  // CHECK-LABEL: func.func @unused
  // CHECK-NOT:     omp.threadprivate
  // CHECK-NOT:     fir.coordinate_of
  // CHECK-NOT:     fir.declare
  // CHECK:         return
  func.func @unused() attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>} {
    %c0 = arith.constant 0 : index
    %0 = fir.address_of(@c_) : !fir.ref<!fir.array<4xi8>>
    %1 = omp.threadprivate %0 : !fir.ref<!fir.array<4xi8>> -> !fir.ref<!fir.array<4xi8>>
    %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<i32>
    %4 = fir.declare %3 storage(%1[0]) uniq_name("_QFunusedEx") : (!fir.ref<i32>, !fir.ref<!fir.array<4xi8>>) -> !fir.ref<i32>
    %5 = fir.address_of(@_QMmEmv) : !fir.ref<i32>
    %6 = omp.threadprivate %5 : !fir.ref<i32> -> !fir.ref<i32>
    return
  }

  // CHECK-LABEL: func.func @host_only
  // CHECK:         omp.threadprivate
  func.func @host_only(%arg0: !fir.ref<i32>) {
    %0 = fir.address_of(@_QMmEmv) : !fir.ref<i32>
    %1 = omp.threadprivate %0 : !fir.ref<i32> -> !fir.ref<i32>
    %2 = fir.load %1 : !fir.ref<i32>
    fir.store %2 to %arg0 : !fir.ref<i32>
    return
  }

  // CHECK-LABEL: func.func @device_type_host
  // CHECK:         omp.threadprivate
  func.func @device_type_host(%arg0: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = host, capture_clause = to>} {
    %0 = fir.address_of(@_QMmEmv) : !fir.ref<i32>
    %1 = omp.threadprivate %0 : !fir.ref<i32> -> !fir.ref<i32>
    %2 = fir.load %1 : !fir.ref<i32>
    fir.store %2 to %arg0 : !fir.ref<i32>
    return
  }
}

// -----

// Nothing is done when compiling for the host.

module attributes {omp.is_target_device = false} {
  fir.global @_QMmEmv : i32

  // CHECK-LABEL: func.func @host_module
  // CHECK:         omp.threadprivate
  func.func @host_module(%arg0: !fir.ref<i32>) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>} {
    %0 = fir.address_of(@_QMmEmv) : !fir.ref<i32>
    %1 = omp.threadprivate %0 : !fir.ref<i32> -> !fir.ref<i32>
    %2 = fir.load %1 : !fir.ref<i32>
    fir.store %2 to %arg0 : !fir.ref<i32>
    return
  }
}
