! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
subroutine test_storage_size(n)
  interface
    function return_char(l)
      integer :: l
      character(l) :: return_char
    end function
  end interface
  integer n
  !CHECK: PRINT *, storage_size(return_char(n))
  print*, storage_size(return_char(n))
  !CHECK: PRINT *, sizeof(return_char(n))
  print*, sizeof(return_char(n))
end subroutine

module pdts
  type t(l)
    integer, len :: l
    character(l) :: c
  end type
contains
  function return_pdt(n)
    type(t(n)) :: return_pdt
  end function
  subroutine test(k)
    ! NOTE: flang design for length parametrized derived type
    ! is to use allocatables for the automatic components. Hence,
    ! their size is independent from the length parameters and is
    ! a compile time constant.
    !CHECK: PRINT *, 192_4
    print *, storage_size(return_pdt(k))
  end subroutine
end module

subroutine test_assumed_rank(x)
  real :: x(..)
  !CHECK: PRINT *, sizeof(x)
  print *, sizeof(x)
end subroutine
