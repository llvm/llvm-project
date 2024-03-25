! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of NORM2(), F'2023 16.9.153
module m
  ! Examples from the standard
  logical, parameter :: test_ex1 = norm2([3.,4.]) == 5.
  real, parameter :: ex2(2,2) = reshape([1.,3.,2.,4.],[2,2])
  real, parameter :: ex2_norm2_1(2) = norm2(ex2, dim=1)
  real, parameter :: ex2_1(2) = [3.162277698516845703125,4.472136020660400390625]
  logical, parameter :: test_ex2_1 = all(ex2_norm2_1 == ex2_1)
  real, parameter :: ex2_norm2_2(2) = norm2(ex2, dim=2)
  real, parameter :: ex2_2(2) = [2.2360680103302001953125,5.]
  logical, parameter :: test_ex2_2 = all(ex2_norm2_2 == ex2_2)
  !  0  3  6  9
  !  1  4  7 10
  !  2  5  8 11
  integer, parameter :: dp = kind(0.d0)
  real(dp), parameter :: a(3,4) = &
    reshape([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], shape(a))
  real(dp), parameter :: nAll = norm2(a)
  real(dp), parameter :: check_nAll = 11._dp * sqrt(sum((a/11._dp)**2))
  logical, parameter :: test_all = nAll == check_nAll
  real(dp), parameter :: norms1(4) = norm2(a, dim=1)
  real(dp), parameter :: check_norms1(4) = [ &
    2.236067977499789805051477742381393909454345703125_8, &
    7.07106781186547550532850436866283416748046875_8, &
    1.2206555615733702069292121450416743755340576171875e1_8, &
    1.7378147196982769884243680280633270740509033203125e1_8 ]
  logical, parameter :: test_norms1 = all(norms1 == check_norms1)
  real(dp), parameter :: norms2(3) = norm2(a, dim=2)
  real(dp), parameter :: check_norms2(3) = [ &
    1.1224972160321822656214862945489585399627685546875e1_8, &
    1.28840987267251261272349438513629138469696044921875e1_8, &
    1.4628738838327791427218471653759479522705078125e1_8 ]
  logical, parameter :: test_norms2 = all(norms2 == check_norms2)
  logical, parameter :: test_normZ = norm2([0.,0.,0.]) == 0.
end
