! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


! Program to test for boundary value in MATMUL intrinsic.
program matmultest
    use ieee_arithmetic

    implicit none
    real(kind=16) :: zero, nan, pmax, mmax, &
                     mzero, nostd, t, su, mu
    real(kind=16), dimension(3, 3) :: mat_a
    real(kind=16), dimension(3, 4) :: mat_b, mat_ab
    real(kind=16), dimension(9) :: a 
    real(kind=16), dimension(12) :: b, mat_res
    logical, dimension(12) :: result, expect

    zero = +0._16
    nan = 0._16 / zero
    pmax = +1._16 / zero
    mmax = -1._16 / zero
    mzero = -0._16
    nostd = 2.3e-33_16

    a = (/ nan, 1.3_16, pmax, mmax, 4._16, &
            zero, mzero, 9.9e-10_16, nostd /)

    b = (/ zero, nan, 5.678e-10_16, nostd, &
            mzero, 1.3_16, pmax, mmax, &
            33.5_16, mzero, mmax, 3.1415926_16 /)

    t = 1.28700000000000000000000298999999996e-9_16
    
    mat_a = reshape(a, (/ 3, 3 /))
    mat_b = reshape(b, (/ 3, 4 /))
    
    mat_ab = matmul(mat_a, mat_b)
    mat_res = reshape(mat_ab, (/ 12 /))
    
    su = abs(mat_res(5) - t)
    mu = t * 1.0e-33_16

    result = .false.
    expect = .true.
    
    result(1) = ieee_is_nan(mat_res(1))
    result(2) = ieee_is_nan(mat_res(2))
    result(3) = ieee_is_nan(mat_res(3))
    result(4) = ieee_is_nan(mat_res(4))
    result(5) = su .lt. mu
    result(6) = .not. ieee_is_finite(mat_res(6))
    result(7) = ieee_is_nan(mat_res(7))
    result(8) = ieee_is_nan(mat_res(8))
    result(9) = ieee_is_nan(mat_res(9))
    result(10) = ieee_is_nan(mat_res(10))
    result(11) = .not. ieee_is_finite(mat_res(11))
    result(12) = ieee_is_nan(mat_res(12))
   
    call check(result, expect, 12)

end program matmultest                         
