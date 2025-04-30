! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module cmpmd
    implicit none
    contains

    subroutine cmp (mm, nn, retval)
        implicit none
        real(kind=16), dimension(:,:), intent(in) :: mm, nn
        logical, dimension(20), intent(inout) :: retval
        real(kind=16), dimension(ubound(mm, 1), ubound(mm, 2)) :: su, mu
        integer :: i, j, k

        do i = lbound(mm, 1), ubound(mm, 1)
           do j = lbound(mm, 2), ubound(mm, 2)
              su(i, j) = abs(mm(i, j) - nn(i, j))
              mu(i, j) = nn(i, j) * 1.0e-33_16
              k = j + (i - 1) * ubound(mm, 2)
              retval(k) = (su(i, j) .lt. mu(i, j))
           end do
        end do
    end subroutine cmp

end module


! Program to test the quad type's precision in MATMUL intrinsic.
program matmultest
    implicit none
    real(kind=16), dimension(4, 5) :: d
    real(kind=16), dimension(4, 5) :: a 
    real(kind=16), dimension(5, 4) :: b 
    real(kind=16), dimension(5, 5) :: f     
    integer :: i, j, k

    ! set value in matrix a, b, f 
    do j = 1, 5
       do k = 1, 5
          f(j, k) = j * k * 0.954655468782234682_16
       end do
       do i = 1, 4
          a(i, j) = i * 0.123456789123456789123456789123456789_16
          b(j, i) = j * 0.987654321987654321987654321987654321987654321_16
       end do
    end do

    d = matmul(a, f)
    
    call test(d, matmul(f, b))

end program matmultest


subroutine test (m1, m2)
    use cmpmd
    implicit none
    integer :: i
    real(kind=16), intent(in) :: m1(4, 5), m2(5, 4)
    real(kind=16) :: stdm1(4, 5), stdm2(5, 4)
    logical, dimension(40) :: result, expect
    logical, dimension(20) :: result1, result2
    integer, dimension (1:2) :: order1 = (/ 2, 1 /)

    ! Output Data
    real(kind=16) :: array1(20) = (/ 1.76788048342504699468738846468738849_16, 3.53576096685009398937477692937477699_16, &
                                 5.30364145027514098406216539406216548_16, 7.07152193370018797874955385874955397_16, &
                                 8.83940241712523497343694232343694093_16, 3.53576096685009398937477692937477699_16, &
                                 7.07152193370018797874955385874955397_16, 10.6072829005502819681243307881243310_16, &
                                 14.1430438674003759574991077174991079_16, 17.6788048342504699468738846468738819_16, &
                                 5.30364145027514098406216539406216548_16, 10.6072829005502819681243307881243310_16, &
                                 15.9109243508254229521864961821864964_16, 21.2145658011005639362486615762486619_16, &
                                 26.5182072513757049203108269703108243_16, 7.07152193370018797874955385874955397_16, &
                                 14.1430438674003759574991077174991079_16, 21.2145658011005639362486615762486619_16, &
                                 28.2860877348007519149982154349982159_16, 35.3576096685009398937477692937477637_16 /)
    real(kind=16) :: array2(20) = (/ 51.8578279863558360305906867405906833_16, 51.8578279863558360305906867405906833_16, &
                                 51.8578279863558360305906867405906833_16, 51.8578279863558360305906867405906833_16, &
                                 103.715655972711672061181373481181367_16, 103.715655972711672061181373481181367_16, &
                                 103.715655972711672061181373481181367_16, 103.715655972711672061181373481181367_16, &
                                 155.573483959067508091772060221772050_16, 155.573483959067508091772060221772050_16, &
                                 155.573483959067508091772060221772050_16, 155.573483959067508091772060221772050_16, &
                                 207.431311945423344122362746962362733_16, 207.431311945423344122362746962362733_16, &
                                 207.431311945423344122362746962362733_16, 207.431311945423344122362746962362733_16, &
                                 259.289139931779180152953433702953416_16, 259.289139931779180152953433702953416_16, &
                                 259.289139931779180152953433702953416_16, 259.289139931779180152953433702953416_16 /)
     
    stdm1 = reshape(array1, (/ 4, 5 /), order = order1)
    stdm2 = reshape(array2, (/ 5, 4 /), order = order1)
    
    result1 = .false.
    result2 = .false.    
    result = .false.
    expect = .true.

    call cmp(m1, stdm1, result1)
    call cmp(m2, stdm2, result2)
      
    do i = 1, 40
       if (i .le. 20) then
          result(i) = result1(i)
       else
          result(i) = result2(i - 20)
       end if
    end do

    call check(result, expect, 40)    

end subroutine test

