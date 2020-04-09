func @x(%arr : !fir.ref<!fir.array<10xf32>>) {
  %1 = constant 0 : index
  %2 = constant 9 : index
  %a = fir.alloca !fir.array<10xf32>
  fir.loop %iv = %1 to %2 unordered {
    %3 = fir.coordinate_of %arr, %iv : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
    %4 = fir.load %3 : !fir.ref<f32>
    %5 = fir.coordinate_of %a, %iv : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
    fir.store %4 to %5 : !fir.ref<f32>
  } { arrayset }
  %6 = fir.embox %a : (!fir.ref<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>>
  fir.call @y(%6) : (!fir.box<!fir.array<10xf32>>) -> ()
  return
}

func @y(!fir.box<!fir.array<10xf32>>) -> ()
