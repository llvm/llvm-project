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
