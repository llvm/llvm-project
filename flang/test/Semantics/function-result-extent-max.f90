!RUN: %flang_fc1 -cpp -DFILE1 -DFILE2 %s | FileCheck %s --allow-empty
!RUN: %flang_fc1 -cpp -DFILE1 %s | FileCheck %s --allow-empty
!RUN: %flang_fc1 -cpp -DFILE2 %s | FileCheck %s --allow-empty

!CHECK-NOT: error
!CHECK-NOT: warning

#ifdef FILE1
module m
    implicit none
    private
    public :: f1
  interface f1
    pure module function f1(n) result(res)
      integer, intent(in) :: n
      integer :: res(max(n, 0))
    end function f1
  end interface f1
end module m
#endif

#ifdef FILE2
submodule (m) sm
  implicit none
contains
  pure module function f1(n) result(res)
      integer, intent(in) :: n
      integer :: res(max(n, 0))
      res(:) = 0
    end function f1
end submodule sm
#endif
