! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of DPROD()
module m
  logical, parameter :: test_kind = kind(dprod(2., 3.)) == kind(0.d0)
  logical, parameter :: test_ss = dprod(2., 3.) == 6.d0
  logical, parameter :: test_sv = all(dprod(2., [3.,4.]) == [6.d0,8.d0])
  logical, parameter :: test_vv = all(dprod([2.,3.], [4.,5.]) == [8.d0,15.0d0])
end
