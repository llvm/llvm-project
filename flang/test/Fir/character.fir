// RUN: tco -emit-fir %s | tco | FileCheck %s
// UNSUPPORTED: !fir

// CHECK-LABEL: define
func @get_name() -> !fir.boxchar<1> {
  %j1 = fir.address_of (@name_constant) : !fir.ref<!fir.char<1>>
  %j2 = constant 9 : i64
  %j3 = fir.emboxchar %j1, %j2 : (!fir.ref<!fir.char<1>>, i64) -> !fir.boxchar<1>
  return %j3 : !fir.boxchar<1>
}
fir.global @name constant : !fir.array<9x!fir.char<1>> {
  %str = fir.string_lit "Your name"(9) : !fir.char<1>
  //constant 1
  fir.has_value %str : !fir.array<9x!fir.char<1>>
}


