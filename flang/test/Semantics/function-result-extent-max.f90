!RUN: %flang_fc1 -cpp -DFILE1 -DFILE2 %s | FileCheck %s --allow-empty
!RUN: %flang_fc1 -cpp -DFILE1 %s | FileCheck %s --allow-empty && %flang_fc1 -cpp -DFILE2 %s | FileCheck %s --allow-empty

!CHECK-NOT: error
!CHECK-NOT: warning

#ifdef FILE1
module function_with_max_result_extent_module
    implicit none
    public :: function_with_max_result_extent
  interface function_with_max_result_extent
    pure module function function_with_max_result_extent(n) result(res)
      integer, intent(in) :: n
      integer :: res(max(n, 0))
    end function function_with_max_result_extent
  end interface function_with_max_result_extent
end module function_with_max_result_extent_module
#endif

#ifdef FILE2
submodule (function_with_max_result_extent_module) function_with_max_result_extent_submodule
  implicit none
contains
  pure module function function_with_max_result_extent(n) result(res)
      integer, intent(in) :: n
      integer :: res(max(n, 0))
      res(:) = 0
    end function function_with_max_result_extent
end submodule function_with_max_result_extent_submodule
#endif
