#x0 = (d0, d1) -> (d1, d0)

func @f(%arg : !fir.ref<!fir.array<?xf64>>) {
  %1 = fir.embox %arg [#x0] : (!fir.ref<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>, >
  return
}
