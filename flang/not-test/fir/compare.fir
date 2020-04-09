func @cmp(%a : !fir.real<10>, %b : !fir.real<10>) -> i1 {
  %1 = "fir.cmpf"(%a, %b) {predicate = 1} : (!fir.real<10>, !fir.real<10>) -> i1
  return %1 : i1
}

func @cmp2(%a : !fir.real<16>, %b : !fir.real<16>) -> i1 {
  %1 = fir.cmpf "ult", %a, %b : !fir.real<16>
  return %1 : i1
}

func @cmp3(%a : !fir.complex<4>, %b : !fir.complex<4>) -> i1 {
  %1 = fir.cmpc "ueq", %a, %b : !fir.complex<4>
  return %1 : i1
}

func @neg1(%a : !fir.real<8>) -> !fir.real<8> {
  %1 = "fir.negf"(%a) : (!fir.real<8>) -> !fir.real<8>
  return %1 : !fir.real<8>
}

func @neg2(%a : !fir.real<8>) -> !fir.real<8> {
  %1 = fir.negf %a : !fir.real<8>
  return %1 : !fir.real<8>
}

func @neg3(%a : !fir.complex<8>) -> !fir.complex<8> {
  %1 = fir.negc %a : !fir.complex<8>
  return %1 : !fir.complex<8>
}
