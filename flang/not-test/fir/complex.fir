func @foo(%a : !fir.complex<4>, %b : !fir.complex<4>, %c : !fir.complex<4>, %d : !fir.complex<4>, %e : !fir.complex<4>) -> !fir.complex<4> {
  %1 = fir.addc %a, %b : !fir.complex<4>
  %2 = fir.mulc %1, %c : !fir.complex<4>
  %3 = fir.subc %2, %d : !fir.complex<4>
  %4 = fir.divc %3, %e : !fir.complex<4>
  return %4 : !fir.complex<4>
}

func @f2(%a : !fir.complex<4>) -> f32 {
  %0 = constant 0 : i32
  %1 = fir.extract_value %a, %0 : (!fir.complex<4>, i32) -> f32
  return %1 : f32
}

func @f3(%a : !fir.complex<4>) -> !fir.complex<4> {
  %0 = constant 1 : i32
  %1 = fir.extract_value %a, %0 : (!fir.complex<4>, i32) -> f32
  %2 = constant 0.0 : f32
  %3 = fir.subf %2, %1 : f32
  %4 = fir.insert_value %a, %3, %0 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  return %4 : !fir.complex<4>
}
