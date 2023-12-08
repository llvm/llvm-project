// Test that the HLFIR pipeline does not call MLIR canonicalizer with block
// merging enabled (moving fir.shape to block argument would cause failures
// when translating the FIR to LLVM).
// RUN: %flang_fc1 %s -flang-experimental-hlfir -emit-llvm -O2 -o - | FileCheck %s

func.func @no_shape_merge(%cdt: i1, %from: !fir.ref<!fir.array<?xf64>>, %to : !fir.ref<f64>) {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c5 = arith.constant 5 : index
  %shape1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %shape2 = fir.shape %c20 : (index) -> !fir.shape<1>
  cf.cond_br %cdt, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %coor1 = fir.array_coor %from(%shape1) %c5 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %load1 = fir.load %coor1 : !fir.ref<f64>
  fir.store %load1 to %to : !fir.ref<f64>
  cf.br ^bb3
^bb2:  // pred: ^bb0
  %coor2 = fir.array_coor %from(%shape2) %c5 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %load2 = fir.load %coor2 : !fir.ref<f64>
  fir.store %load2 to %to : !fir.ref<f64>
  cf.br ^bb3
^bb3:  // pred: ^bb1, ^bb2
  return
}

// Note: block merging happens in the output below, but after FIR codegen.

// CHECK-LABEL:  define void @no_shape_merge(
// CHECK:  %[[GEP:.*]] = getelementptr double, ptr %{{.*}}
// CHECK:  %[[LOAD:.*]] = load double, ptr %[[GEP]]
// CHECK:  store double %[[LOAD]], ptr %{{.*}}
// CHECK:  ret void
