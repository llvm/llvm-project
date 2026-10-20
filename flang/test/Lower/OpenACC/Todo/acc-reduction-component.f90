! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenACC reduction with component reference not yet supported

module m_reduction_component
  type :: t
    real :: a(20)
  end type
contains
  subroutine component_array_section_reduction(x, nn)
    integer, intent(in) :: nn
    type(t), intent(inout) :: x
    integer :: i, k

    !$acc parallel loop reduction(+:x%a(1:16))
    do i = 1, nn
      do k = 1, 16
        x%a(k) = x%a(k) + 1.0
      end do
    end do
  end subroutine component_array_section_reduction
end module m_reduction_component
