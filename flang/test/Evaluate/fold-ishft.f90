! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of ISHFT
module m1
  logical :: test_ishft_lsb = all(ishft(1, [-32, -31, -1, 0, 1, 2, 31, 32]) == [0, 0, 0, 1, 2, 4, int(z'80000000'), 0])
  logical :: test_ishft_msb = all(ishft(ishft(1,31), [-32, -31, -1, 0, 1, 2, 31, 32]) == [0, 1, int(z'40000000'), int(z'80000000'), 0, 0, 0, 0])
end module
