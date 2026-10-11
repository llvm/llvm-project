 ! RUN: %check_flang_tidy %s bugprone-mismatched-intent %t
module test
  type :: child_t
     integer :: i
  end type child_t

  type :: parent_t
     type(child_t) :: child
  end type parent_t

  interface
     subroutine mutate_member(parent, child)
       import parent_t
       import child_t
       type(parent_t), intent(in) :: parent
       type(child_t), intent(inout) :: child
    end subroutine mutate_member
 end interface

contains
  subroutine check(parent)
    type(parent_t), intent(inout) :: parent
    call mutate_member(parent, parent%child)
    ! CHECK-MESSAGES: :[[@LINE-1]]:5: warning: mismatched intent between class 'parent' and its member 'child'
  end subroutine check
end module test
