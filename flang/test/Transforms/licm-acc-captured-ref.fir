// RUN: fir-opt %s --flang-licm | FileCheck %s

// CHECK-LABEL: func.func @captured_descriptor
// CHECK: acc.compute_region
// CHECK: fir.load
// CHECK: scf.for
// CHECK-NOT: fir.load
// CHECK: acc.yield
func.func @captured_descriptor(%x: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %n: index) {
  acc.kernel_environment {
    acc.compute_region ins(%ref = %x, %bound = %n) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %bound step %c1 {
        %descriptor = fir.load %ref : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// CHECK-LABEL: func.func @captured_optional_descriptor
// CHECK: acc.compute_region
// CHECK-NOT: fir.load
// CHECK: scf.for
// CHECK: fir.load
// CHECK: acc.yield
func.func @captured_optional_descriptor(%x: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.optional}, %n: index) {
  acc.kernel_environment {
    acc.compute_region ins(%ref = %x, %bound = %n) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %bound step %c1 {
        %descriptor = fir.load %ref : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// Capturing a reference must preserve the ability to prove that loading a
// non-optional scalar is safe even when the loop might execute zero times.
// CHECK-LABEL: func.func @captured_scalar
// CHECK: acc.compute_region
// CHECK: fir.load
// CHECK: scf.for
// CHECK-NOT: fir.load
// CHECK: acc.yield
func.func @captured_scalar(%x: !fir.ref<i32>, %n: index) {
  acc.kernel_environment {
    acc.compute_region ins(%ref = %x, %bound = %n) : (!fir.ref<i32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant 0 : i32
      %result = scf.for %i = %c0 to %bound step %c1 iter_args(%previous = %zero) -> i32 {
        %value = fir.load %ref : !fir.ref<i32>
        scf.yield %value : i32
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// CHECK-LABEL: func.func @captured_optional
// CHECK: acc.compute_region
// CHECK-NOT: fir.load
// CHECK: scf.for
// CHECK: fir.load
// CHECK: acc.yield
// Capture forwarding must retain OPTIONAL, not treat every input as present.
func.func @captured_optional(%x: !fir.ref<i32> {fir.optional}, %n: index) {
  acc.kernel_environment {
    acc.compute_region ins(%ref = %x, %bound = %n) : (!fir.ref<i32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant 0 : i32
      %result = scf.for %i = %c0 to %bound step %c1 iter_args(%previous = %zero) -> i32 {
        %value = fir.load %ref : !fir.ref<i32>
        scf.yield %value : i32
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// CHECK-LABEL: func.func @captured_element
// CHECK: acc.compute_region
// CHECK-NOT: fir.load
// CHECK: scf.for
// CHECK: fir.load
// CHECK: acc.yield
// A captured scalar reference can designate an array element whose subscript
// is not known to be valid. Its type alone does not justify speculation.
func.func @captured_element(%x: !fir.ref<!fir.array<10xi32>>, %index: index, %n: index) {
  %element = fir.coordinate_of %x, %index : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  acc.kernel_environment {
    acc.compute_region ins(%ref = %element, %bound = %n) : (!fir.ref<i32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant 0 : i32
      %result = scf.for %i = %c0 to %bound step %c1 iter_args(%previous = %zero) -> i32 {
        %value = fir.load %ref : !fir.ref<i32>
        scf.yield %value : i32
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// CHECK-LABEL: func.func @captured_modified
// CHECK: acc.compute_region
// CHECK-NOT: fir.load
// CHECK: scf.for
// CHECK: fir.load
// CHECK: acc.yield
// Speculative-read safety does not replace the loop's mod/ref check.
func.func @captured_modified(%x: !fir.ref<i32>, %n: index) {
  acc.kernel_environment {
    acc.compute_region ins(%ref = %x, %bound = %n) : (!fir.ref<i32>, index) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %zero = arith.constant 0 : i32
      %result = scf.for %i = %c0 to %bound step %c1 iter_args(%previous = %zero) -> i32 {
        %value = fir.load %ref : !fir.ref<i32>
        fir.store %zero to %ref : !fir.ref<i32>
        scf.yield %value : i32
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}
