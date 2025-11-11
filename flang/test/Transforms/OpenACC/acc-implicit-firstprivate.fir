// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" -split-input-file | FileCheck %s

// Test implicit firstprivate behavior for various scalar types in parallel and serial constructs.
// Scalars in parallel/serial constructs should be implicitly firstprivate according to OpenACC spec.

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_i32 : !fir.ref<i32> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<i32>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca i32
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<i32>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<i32>, %[[DST:.*]]: !fir.ref<i32>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<i32>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<i32>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_i32_scalar_in_parallel
func.func @test_i32_scalar_in_parallel() {
  %scalar = fir.alloca i32 {bindc_name = "i32_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<i32>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i32_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_i32 -> %[[FIRSTPRIV]] : !fir.ref<i32>)

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_i64 : !fir.ref<i64> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<i64>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca i64
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<i64>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<i64>, %[[DST:.*]]: !fir.ref<i64>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<i64>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<i64>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_i64_scalar_in_parallel
func.func @test_i64_scalar_in_parallel() {
  %scalar = fir.alloca i64 {bindc_name = "i64_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<i64>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<i64>) -> !fir.ref<i64> {implicit = true, name = "i64_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_i64 -> %[[FIRSTPRIV]] : !fir.ref<i64>)

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_f32 : !fir.ref<f32> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<f32>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca f32
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<f32>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<f32>, %[[DST:.*]]: !fir.ref<f32>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<f32>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<f32>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_f32_scalar_in_parallel
func.func @test_f32_scalar_in_parallel() {
  %scalar = fir.alloca f32 {bindc_name = "f32_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<f32>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<f32>) -> !fir.ref<f32> {implicit = true, name = "f32_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_f32 -> %[[FIRSTPRIV]] : !fir.ref<f32>)

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_f64 : !fir.ref<f64> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<f64>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca f64
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<f64>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<f64>, %[[DST:.*]]: !fir.ref<f64>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<f64>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<f64>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_f64_scalar_in_parallel
func.func @test_f64_scalar_in_parallel() {
  %scalar = fir.alloca f64 {bindc_name = "f64_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<f64>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<f64>) -> !fir.ref<f64> {implicit = true, name = "f64_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_f64 -> %[[FIRSTPRIV]] : !fir.ref<f64>)

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_l32 : !fir.ref<!fir.logical<4>> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca !fir.logical<4>
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<!fir.logical<4>>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<!fir.logical<4>>, %[[DST:.*]]: !fir.ref<!fir.logical<4>>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<!fir.logical<4>>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<!fir.logical<4>>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_logical_scalar_in_parallel
func.func @test_logical_scalar_in_parallel() {
  %scalar = fir.alloca !fir.logical<4> {bindc_name = "logical_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<!fir.logical<4>>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>> {implicit = true, name = "logical_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_l32 -> %[[FIRSTPRIV]] : !fir.ref<!fir.logical<4>>)

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_z32 : !fir.ref<complex<f32>> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<complex<f32>>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca complex<f32>
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<complex<f32>>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<complex<f32>>, %[[DST:.*]]: !fir.ref<complex<f32>>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<complex<f32>>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<complex<f32>>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_complex_scalar_in_parallel
func.func @test_complex_scalar_in_parallel() {
  %scalar = fir.alloca complex<f32> {bindc_name = "complex_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<complex<f32>>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<complex<f32>>) -> !fir.ref<complex<f32>> {implicit = true, name = "complex_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_z32 -> %[[FIRSTPRIV]] : !fir.ref<complex<f32>>)

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_z64 : !fir.ref<complex<f64>> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<complex<f64>>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca complex<f64>
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<complex<f64>>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<complex<f64>>, %[[DST:.*]]: !fir.ref<complex<f64>>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<complex<f64>>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<complex<f64>>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_complex8_scalar_in_parallel
func.func @test_complex8_scalar_in_parallel() {
  %scalar = fir.alloca complex<f64> {bindc_name = "complex8_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<complex<f64>>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<complex<f64>>) -> !fir.ref<complex<f64>> {implicit = true, name = "complex8_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_z64 -> %[[FIRSTPRIV]] : !fir.ref<complex<f64>>)

// -----

// Test with serial construct

// CHECK-LABEL: func.func @test_i32_scalar_in_serial
func.func @test_i32_scalar_in_serial() {
  %scalar = fir.alloca i32 {bindc_name = "serial_i32_var"}
  acc.serial {
    %load = fir.load %scalar : !fir.ref<i32>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "serial_i32_var"}
// CHECK: acc.serial firstprivate(@firstprivatization_ref_i32 -> %[[FIRSTPRIV]] : !fir.ref<i32>)

// -----

// Test with serial construct and f64

// CHECK-LABEL: func.func @test_f64_scalar_in_serial
func.func @test_f64_scalar_in_serial() {
  %scalar = fir.alloca f64 {bindc_name = "serial_f64_var"}
  acc.serial {
    %load = fir.load %scalar : !fir.ref<f64>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<f64>) -> !fir.ref<f64> {implicit = true, name = "serial_f64_var"}
// CHECK: acc.serial firstprivate(@firstprivatization_ref_f64 -> %[[FIRSTPRIV]] : !fir.ref<f64>)

// -----

// Test i8 and i16 scalar types

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_i8 : !fir.ref<i8> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<i8>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca i8
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<i8>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<i8>, %[[DST:.*]]: !fir.ref<i8>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<i8>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<i8>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_i8_scalar_in_parallel
func.func @test_i8_scalar_in_parallel() {
  %scalar = fir.alloca i8 {bindc_name = "i8_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<i8>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<i8>) -> !fir.ref<i8> {implicit = true, name = "i8_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_i8 -> %[[FIRSTPRIV]] : !fir.ref<i8>)

// -----

// CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_ref_i16 : !fir.ref<i16> init {
// CHECK:       ^bb0(%{{.*}}: !fir.ref<i16>):
// CHECK:         %[[ALLOC:.*]] = fir.alloca i16
// CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]]
// CHECK:         acc.yield %[[DECL]]#0 : !fir.ref<i16>
// CHECK:       } copy {
// CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<i16>, %[[DST:.*]]: !fir.ref<i16>):
// CHECK:         %[[LOADED:.*]] = fir.load %[[SRC]] : !fir.ref<i16>
// CHECK:         fir.store %[[LOADED]] to %[[DST]] : !fir.ref<i16>
// CHECK:         acc.terminator
// CHECK:       }

// CHECK-LABEL: func.func @test_i16_scalar_in_parallel
func.func @test_i16_scalar_in_parallel() {
  %scalar = fir.alloca i16 {bindc_name = "i16_var"}
  acc.parallel {
    %load = fir.load %scalar : !fir.ref<i16>
    acc.yield
  }
  return
}

// CHECK: %[[FIRSTPRIV:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<i16>) -> !fir.ref<i16> {implicit = true, name = "i16_var"}
// CHECK: acc.parallel firstprivate(@firstprivatization_ref_i16 -> %[[FIRSTPRIV]] : !fir.ref<i16>)

