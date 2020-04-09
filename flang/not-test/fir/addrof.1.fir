fir.global @var_x : !fir.int<4> {}

func @getAddressOfX() -> !fir.ref<!fir.int<4>> {
  %1 = fir.address_of(@var_x) : !fir.ref<!fir.int<4>>
  return %1 : !fir.ref<!fir.int<4>>
}

