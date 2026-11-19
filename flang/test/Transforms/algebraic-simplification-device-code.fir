// RUN: fir-opt --flang-algebraic-simplification %s | FileCheck %s

// A complex power with a negative constant exponent stays a power in OpenACC
// and CUDA Fortran device code, and in the functions that device code
// references. The simplification applies everywhere else, and to the other
// powers in device code.

acc.routine @acc_routine_0 func(@acc_routine) seq

// CHECK-LABEL: func.func @host(
// CHECK: %[[MUL:.*]] = complex.mul
// CHECK: complex.div %{{.*}}, %[[MUL]]
// CHECK-NOT: complex.powi
func.func @host(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// The power before the region is host code.
// CHECK-LABEL: func.func @acc_region(
// CHECK: complex.div
// CHECK: acc.serial
// CHECK-NOT: complex.div
// CHECK: complex.powi %{{.*}}, %c-2_i32 fastmath<contract> : complex<f64>, i32
// CHECK-NOT: complex.div
// CHECK: return
func.func @acc_region(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  acc.serial {
    %2 = fir.load %b : !fir.ref<complex<f64>>
    %3 = complex.powi %2, %c fastmath<contract> : complex<f64>, i32
    fir.store %3 to %a : !fir.ref<complex<f64>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @acc_region_i64_exponent(
// CHECK: complex.powi %{{.*}}, %c-3_i64 : complex<f32>, i64
// CHECK-NOT: complex.div
// CHECK: return
func.func @acc_region_i64_exponent(%a: !fir.ref<complex<f32>>, %b: !fir.ref<complex<f32>>) {
  %c = arith.constant -3 : i64
  acc.parallel {
    %0 = fir.load %b : !fir.ref<complex<f32>>
    %1 = complex.powi %0, %c : complex<f32>, i64
    fir.store %1 to %a : !fir.ref<complex<f32>>
    acc.yield
  }
  return
}

// The other powers are still simplified in device code.
// CHECK-LABEL: func.func @acc_region_other_powers(
// CHECK: acc.serial
// CHECK-NOT: powi
// CHECK: complex.mul
// CHECK: arith.divf
// CHECK-NOT: powi
// CHECK: return
func.func @acc_region_other_powers(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>, %r: !fir.ref<f64>) {
  %c2 = arith.constant 2 : i32
  %c0 = arith.constant 0 : i32
  %cm2 = arith.constant -2 : i32
  acc.serial {
    %0 = fir.load %b : !fir.ref<complex<f64>>
    %1 = complex.powi %0, %c2 : complex<f64>, i32
    fir.store %1 to %a : !fir.ref<complex<f64>>
    %2 = complex.powi %0, %c0 : complex<f64>, i32
    fir.store %2 to %a : !fir.ref<complex<f64>>
    %3 = fir.load %r : !fir.ref<f64>
    %4 = math.fpowi %3, %cm2 : f64, i32
    fir.store %4 to %r : !fir.ref<f64>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @acc_routine(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @acc_routine(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>} {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  fir.call @called_from_routine(%a, %b) : (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> ()
  return
}

// CHECK-LABEL: func.func @called_from_routine(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @called_from_routine(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: func.func @cuf_device(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @cuf_device(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// A host_device procedure is compiled for the device as well.
// CHECK-LABEL: func.func @cuf_host_device(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @cuf_host_device(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// A host_device procedure that has a device copy is host code. The copy is
// device code.
// CHECK-LABEL: func.func @cuf_copied(
// CHECK: complex.div
// CHECK-NOT: complex.powi
// CHECK: return
func.func @cuf_copied(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) attributes {cuf.proc_attr = #cuf.cuda_proc<host_device>} {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: func.func @cuf_copied.device(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @cuf_copied.device(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) attributes {cuf.device_copy_of = @cuf_copied, cuf.proc_attr = #cuf.cuda_proc<device>} {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: func.func @cuf_kernel(
// CHECK: cuf.kernel
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @cuf_kernel(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%iv : index) = (%c1 : index) to (%c1 : index) step (%c1 : index) {
    %0 = fir.load %b : !fir.ref<complex<f64>>
    %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
    fir.store %1 to %a : !fir.ref<complex<f64>>
    "fir.end"() : () -> ()
  }
  return
}

// The exponent only becomes a constant during the simplification.
// CHECK-LABEL: func.func @acc_region_folded_exponent(
// CHECK: acc.serial
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @acc_region_folded_exponent(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i8
  acc.serial {
    %e = fir.convert %c : (i8) -> i32
    %0 = fir.load %b : !fir.ref<complex<f64>>
    %1 = complex.powi %0, %e fastmath<contract> : complex<f64>, i32
    fir.store %1 to %a : !fir.ref<complex<f64>>
    acc.yield
  }
  return
}

// Functions without a directive that device code calls.
// CHECK-LABEL: func.func @called_from_region(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @called_from_region(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  fir.call @called_from_callee(%a, %b) : (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> ()
  return
}

// CHECK-LABEL: func.func @called_from_callee(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @called_from_callee(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: func.func @caller(
// CHECK: acc.serial
// CHECK: fir.call @called_from_region
func.func @caller(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  acc.serial {
    fir.call @called_from_region(%a, %b) : (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> ()
    acc.yield
  }
  return
}

// A function that device code references without calling it directly.
// CHECK-LABEL: func.func @referenced_from_region(
// CHECK: complex.powi
// CHECK-NOT: complex.div
// CHECK: return
func.func @referenced_from_region(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

func.func private @apply((!fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> (), !fir.ref<complex<f64>>, !fir.ref<complex<f64>>)

// CHECK-LABEL: func.func @referencing_caller(
func.func @referencing_caller(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  acc.serial {
    %f = fir.address_of(@referenced_from_region) : (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> ()
    fir.call @apply(%f, %a, %b) : ((!fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> (), !fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> ()
    acc.yield
  }
  return
}

// A function that only host code calls.
// CHECK-LABEL: func.func @called_from_host(
// CHECK: complex.div
// CHECK-NOT: complex.powi
// CHECK: return
func.func @called_from_host(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  %c = arith.constant -2 : i32
  %0 = fir.load %b : !fir.ref<complex<f64>>
  %1 = complex.powi %0, %c fastmath<contract> : complex<f64>, i32
  fir.store %1 to %a : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: func.func @host_caller(
func.func @host_caller(%a: !fir.ref<complex<f64>>, %b: !fir.ref<complex<f64>>) {
  fir.call @called_from_host(%a, %b) : (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>) -> ()
  return
}
