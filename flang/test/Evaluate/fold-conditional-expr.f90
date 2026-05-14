! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of conditional expressions (Fortran 2023)
module m
  ! Basic scalar folding: constant condition selects the chosen branch.
  logical, parameter :: test_true_int  = (.true.  ? 1 : 2) == 1
  logical, parameter :: test_false_int = (.false. ? 1 : 2) == 2
  logical, parameter :: test_true_real  = (.true.  ? 1.0 : 2.0) == 1.0
  logical, parameter :: test_false_real = (.false. ? 1.0 : 2.0) == 2.0
  logical, parameter :: test_true_logical  = (.true.  ? .true.  : .false.)
  logical, parameter :: test_false_logical = (.false. ? .false. : .true.)

  ! Multi-branch: right-skewed tree folds correctly.
  ! (.true. ? 10 : .false. ? 20 : 30) == 10
  logical, parameter :: test_multi_first  = (.true.  ? 10 : .false. ? 20 : 30) == 10
  ! (.false. ? 10 : .true.  ? 20 : 30) == 20
  logical, parameter :: test_multi_second = (.false. ? 10 : .true.  ? 20 : 30) == 20
  ! (.false. ? 10 : .false. ? 20 : 30) == 30
  logical, parameter :: test_multi_third  = (.false. ? 10 : .false. ? 20 : 30) == 30

  ! Named constant expressions in branches are folded.
  integer, parameter :: x = 5
  logical, parameter :: test_branch_fold = (.true. ? x + 1 : x + 2) == 6

  ! Named constant as condition.
  logical, parameter :: cond = .true.
  logical, parameter :: test_named_cond = (cond ? 42 : 0) == 42

  ! Character: constant condition selects the branch value.
  logical, parameter :: test_char = (.true. ? 'yes' : 'no') == 'yes'

  ! Non-constant branch: only the selected branch need be constant (F2023 10.1.12).
  integer :: non_const = 99
  logical, parameter :: test_true_const_else_nonconstant  = (.true.  ? 10 : non_const) == 10
  logical, parameter :: test_false_const_then_nonconstant = (.false. ? non_const : 10) == 10

  ! Named constant condition with a non-constant branch.
  logical, parameter :: flag = .true.
  logical, parameter :: test_named_cond_nonconstant = (flag ? 1 : non_const) == 1
  logical, parameter :: flag_false = .false.
  logical, parameter :: test_named_cond_false_nonconstant = (flag_false ? non_const : 1) == 1

end module
