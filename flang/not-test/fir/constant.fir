func @x() -> !fir.array<3x!fir.char<1>> {
  %1 = fir.constant "xyz" : !fir.array<3x!fir.char<1>>
  return %1 : !fir.array<3x!fir.char<1>>
}

func @y() -> !fir.real<10> {
  %1 = fir.constant "42.4" : !fir.real<10>
  return %1 : !fir.real<10>
}

func @z() -> !fir.logical<1> {
  %1 = fir.constant "true" : !fir.logical<1>
  return %1 : !fir.logical<1>
}

func @z2() -> !fir.ref<!fir.char<1>> {
  %1 = fir.constant "abc" : !fir.ref<!fir.char<1>>
  return %1 : !fir.ref<!fir.char<1>>
}
