// dynamically sized type
func @_QQSIZEOF_a(%p1 : i64, %p2 : i64) -> i64 {
  %c1 = constant 1 : i64  // sizeof CHARACTER(1)
  %1 = muli %p1, %c1 : i64
  %c2 = constant 2 : i64  // sizeof CHARACTER(2)
  %2 = muli %p2, %c2 : i64
  %3 = addi %1, %2 : i64
  return %3 : i64
}

func @_QQOFFSETOF_a_f1(%p1 : i64, %p2 : i64) -> i32 {
  %0 = constant 0 : i64
  %1 = fir.convert %0 : (i64) -> i32
  return %1 : i32
}

func @_QQOFFSETOF_a_f2(%p1 : i64, %p2 : i64) -> i32 {
  %c1 = constant 1 : i64
  %1 = muli %p1, %c1 : i64
  %2 = fir.convert %1 : (i64) -> i32
  return %2 : i32
}

// get 'a%f2(x)'
func @f(%a : !fir.box<!fir.type<a(p1:i64,p2:i64){f1:!fir.array<?x!fir.char<1>>, f2:!fir.array<?x!fir.char<2>>}>>, %x : i32, %q1 : i64, %q2 : i64) -> !fir.char<2> {
  %p1p = fir.len_param_index p1, !fir.type<a>
  %p2p = fir.len_param_index p2, !fir.type<a>
  %p1c = fir.coordinate_of %a, %p1p : (!fir.box<!fir.type<a>>, !fir.len) -> !fir.ref<i64>
  %p2c = fir.coordinate_of %a, %p2p : (!fir.box<!fir.type<a>>, !fir.len) -> !fir.ref<i64>
  %p1 = fir.load %p1c : !fir.ref<i64>
  %p2 = fir.load %p2c : !fir.ref<i64>
  %1 = fir.field_index f2, !fir.type<a>(%p1, %p2) : i64, i64
  %2 = fir.coordinate_of %a, %1, %x : (!fir.box<!fir.type<a>>, !fir.field, i32) -> !fir.ref<!fir.char<2>>
  %3 = fir.load %2 : !fir.ref<!fir.char<2>>
  return %3 : !fir.char<2>
}
