func @f1(%a : i32, %b : i32) -> i32 {
  %1 = addi %a, %b : i32
  %2 = addi %b, %a : i32
  %3 = muli %1, %2 : i32
  return %3 : i32
}

func @f2(%a : !fir.ref<i32>) -> i32 {
  %1 = fir.load %a : !fir.ref<i32>
  %2 = fir.load %a : !fir.ref<i32>
  %3 = addi %1, %2 : i32
  %4 = fir.load %a : !fir.ref<i32>
  %5 = addi %3, %4 : i32
  %6 = fir.load %a : !fir.ref<i32>
  %7 = addi %5, %6 : i32
  %8 = fir.load %a : !fir.ref<i32>
  %9 = addi %7, %8 : i32
  %10 = fir.load %a : !fir.ref<i32>
  %11 = addi %10, %9 : i32
  return %11 : i32
}
