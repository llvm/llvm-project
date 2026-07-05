! RUN: %python %S/test_folding.py %s %flang_fc1
module m
  real(4), parameter :: x20_4 = erfc_scaled(20._4)
  logical, parameter :: t20_4 = x20_4 == 0.02817435003817081451416015625_4
  real(8), parameter :: x20_8 = erfc_scaled(20._8)
  logical, parameter :: t20_8 = x20_8 == 0.0281743487410513193669459042212110944092273712158203125_8
end
