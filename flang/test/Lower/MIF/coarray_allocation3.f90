! RUN: not %flang_fc1 -emit-hlfir -fcoarray %s -o - 2>&1 | FileCheck %s

!CHECK: not yet implemented: Coarray with an allocatable direct component and/or requiring finalization.

module m_test
    implicit none

    type :: test_type
        integer :: id
        real, allocatable :: arr(:)
    contains
        final :: finalize_func
    end type test_type

contains

    subroutine finalize_func(this)
        type(test_type), intent(inout) :: this
        if (allocated(this%arr)) deallocate(this%arr)
    end subroutine finalize_func

end module m_test

program test_final_coarray
    use m_test
    implicit none
    type(test_type), allocatable :: A[:]

end program test_final_coarray
