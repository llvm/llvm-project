func @f_tuple(%a : tuple<i32, i64>) -> i64 {
  %0 = constant 1 : i32
  %1 = fir.extract_value %a, %0 : (tuple<i32, i64>, i32) -> i64
  return %1 : i64
}

func @f_record(%a : !fir.type<rec1{fa:i32, fb:i64}>) -> i64 {
  %0 = fir.field_index fb, !fir.type<rec1>
  %1 = fir.extract_value %a, %0 : (!fir.type<rec1>, !fir.field) -> i64
  return %1 : i64
}
