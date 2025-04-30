
!** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!** See https://llvm.org/LICENSE.txt for license information.
!** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! testing if contiguous arrays passed as argument are really contiguous

program main
 integer, parameter :: m=6,n=4,h=2
 integer :: i,j

 integer(kind=4), allocatable :: big_array(:, :)
 integer(kind=4) :: expected(n-h, m-h)
 integer(kind=4) :: res(n-h, m-h)
 allocate(big_array(n, m))
 do i=1,n
  do j=1,m
    big_array(i,j) = i
  enddo
 enddo
 expected = big_array(1:n-h,1:m-h)
 call pass_contiguous_array(big_array(1:n-h,1:m-h), m, n, h, res)
 call check(res,expected,(n-h)*(m-h));

contains
  subroutine pass_contiguous_array(arr, m, n, h, res)
    use iso_c_binding
    implicit none
    integer(kind=4), target, contiguous, intent(in) :: arr(:,:)
    integer(kind=4),  target, intent(inout) :: res(n-h,m-h)
    integer, intent(in) :: m, n, h
    integer :: err

    interface
      function pass_contiguous_array_c(data, m, n,res) result(error_code) BIND(c)
        import c_int, c_float, c_double, c_ptr
        integer(c_int), VALUE, intent(in) :: m
        integer(c_int), VALUE, intent(in) :: n
        type(c_ptr),    VALUE, intent(in) :: data
        type(c_ptr),    VALUE, intent(in) :: res
        integer(c_int)                    :: error_code
      end function pass_contiguous_array_c
    end interface

    err = pass_contiguous_array_c(c_loc(arr), m-h, n-h,c_loc(res))
  end subroutine pass_contiguous_array
end program
