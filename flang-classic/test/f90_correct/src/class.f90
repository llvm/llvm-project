! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module type_def
    implicit none
    type, abstract :: t
    contains
      procedure, nopass :: recursub
      procedure(assign_operator), deferred :: copy
    end type t

    abstract interface
        subroutine assign_operator(left, right)
            import :: t
            class(t), intent(in) :: right
            class(t), intent(inout) :: left
        end subroutine assign_operator
    end interface

    type, extends(t) :: tt
        integer :: m
    contains
      procedure :: copy => copy_sub
    end type tt

contains
recursive subroutine recursub(array)
    class(t), dimension(:) :: array
    class(t), allocatable :: tmp
    integer :: j

    j = size(array)
    allocate(tmp, source = array(1))
    if(size(array) < 2) return

    call array(1)%copy(array(j))
    call array(j)%copy(tmp)

    call recursub(array(2:j-1))
end subroutine recursub

subroutine copy_sub(left, right)
    class(tt), intent(inout) :: left
    class(t), intent(in) :: right

    select type (right)
        type is (tt)
            select type (left)
                type is (tt)
                    left = right
            end select
    end select
end subroutine copy_sub

end module type_def

program test
    use type_def
    implicit none
    type(tt), dimension(20) :: array
    integer, dimension(20) :: init_value
    data init_value / 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, &
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20 /

    array%m = init_value
    call recursub(array)

    if(any(array%m .ne. init_value(20:1:-1))) STOP 1
    print *, 'PASS'
end program test
