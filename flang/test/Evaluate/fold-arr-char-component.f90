! RUN: %python %S/test_folding.py %s %flang_fc1
! Ensure that array-valued component references have lengths
! (see https://github.com/llvm/llvm-project/issues/123362)
module m
  type cdt
    character(7) :: a = "ibm704", b = "cdc6600"
  end type
  type(cdt), parameter :: arr(2) = cdt()
  integer, parameter :: check(*) = scan(arr%a, arr%b)
  logical, parameter :: test1 = all(check == 5) ! the '0'
end
