!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!  Module defining a user generic which resolves to a function returning
!  an array whose size is the SIZE of an array argument.
!
module square
interface square
    function square_s(x)		!"scalar" version
    integer, intent(in) :: x
    integer :: square_s
    end function square_s
    function square_v(x)		!"vector" version
    integer, dimension(:), intent(in) :: x
    integer, dimension(size(x)) :: square_v
    end function square_v
endinterface
integer result(10)
common/result/result
endmodule square

function square_s(x)		!"scalar" version
    integer, intent(in) :: x
    integer :: square_s
    square_s = x * x
end function square_s

function square_v(x)		!"vector" version
    integer, dimension(:), intent(in) :: x
    integer, dimension(size(x)) :: square_v
    square_v = x * x
end function square_v

subroutine test_v(x)
    use square
    implicit none
    integer, dimension(:), intent(in) :: x
    integer, dimension(size(x)) :: gam1
    gam1=square(x)
!    print *, gam1
    result(1:3) = gam1(1:3)
end subroutine test_v

subroutine test_s(x)
    use square
    implicit none
    integer :: x, gam1
    result(4) = square(x)
end subroutine test_s

program test
    interface
	subroutine test_v(x)
	integer, dimension(:), intent(in) :: x
	endsubroutine
    endinterface
    integer, dimension(3) :: inp = (/2,-3,4/)
    integer, dimension(4) :: expect = (/4,9,16,25/)
    integer result(10)
    common/result/result
    call test_v(inp)
    call test_s(-5)
!    print *, result(1:3), result(4)
    call check(result, expect, 4)
end
