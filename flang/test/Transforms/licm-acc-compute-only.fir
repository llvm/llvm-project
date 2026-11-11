// RUN: fir-opt %s --flang-licm="only-inside=acc.compute_region" | FileCheck %s
// RUN: fir-opt %s --flang-licm | FileCheck %s --check-prefix=GLOBAL

// Filtering must retain host loads even in a loop enclosing a compute region,
// and find device loops beneath host conditionals. The default is unchanged.
// CHECK-LABEL: func.func @mixed
// CHECK: scf.for
// CHECK: fir.load
// CHECK: scf.if
// CHECK: acc.compute_region
// CHECK: fir.load
// CHECK: scf.for
// CHECK-NOT: fir.load
// CHECK: acc.yield
// CHECK: } else {
// CHECK: scf.for
// CHECK: fir.load
// GLOBAL-LABEL: func.func @mixed
// GLOBAL: fir.load
// GLOBAL: scf.for
func.func @mixed(%x: !fir.ref<i32>, %n: index, %condition: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %h = %c0 to %n step %c1 {
    %host = fir.load %x : !fir.ref<i32>
    scf.if %condition {
      acc.kernel_environment {
        acc.compute_region ins(%ref = %x, %bound = %n) : (!fir.ref<i32>, index) {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          scf.for %i = %zero to %bound step %one {
            %device = fir.load %ref : !fir.ref<i32>
          }
          acc.yield
        } <{origin = "acc.kernels"}>
      }
    } else {
      scf.for %i = %c0 to %n step %c1 {
        %fallback = fir.load %x : !fir.ref<i32>
      }
    }
  }
  return
}
