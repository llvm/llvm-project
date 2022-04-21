// RUN: tco %s | FileCheck %s

// CHECK-LABEL: define void @x([10 x float]* %0)
func.func @x(%arr : !fir.ref<!fir.array<10xf32>>) {
  %1 = arith.constant 0 : index
  %2 = arith.constant 9 : index
  %stepvar = arith.constant 1 : index
  // CHECK: alloca [10 x float], i64 1
  %a = fir.alloca !fir.array<10xf32>
  fir.do_loop %iv = %1 to %2 step %stepvar unordered {
    %3 = fir.coordinate_of %arr, %iv : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
  // CHECK: %[[reg10:.*]] = load float, float*
    %4 = fir.load %3 : !fir.ref<f32>
  // CHECK: %[[reg11:.*]] = getelementptr [10 x float], [10 x float]*
    %5 = fir.coordinate_of %a, %iv : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
  // CHECK: store float %[[reg10]], float* %[[reg11]]
    fir.store %4 to %5 : !fir.ref<f32>
  }
  %6 = fir.embox %a : (!fir.ref<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>>
  fir.call @y(%6) : (!fir.box<!fir.array<10xf32>>) -> ()
  return
}

func.func private @y(!fir.box<!fir.array<10xf32>>) -> ()
