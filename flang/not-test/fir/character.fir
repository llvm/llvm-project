
fir.global @name constant : !fir.char<1> {
  constant "Your name"
  //constant 1
}

func @get_name() -> !fir.boxchar<1> {
  %j1 = fir.address_of (@name_constant) : !fir.ref<!fir.char<1>>
  %j2 = constant 9 : i64
  %j3 = fir.emboxchar %j1, %j2 : (!fir.ref<!fir.char<1>>, i64) -> !fir.boxchar<1>
  return %j3 : !fir.boxchar<1>
}
