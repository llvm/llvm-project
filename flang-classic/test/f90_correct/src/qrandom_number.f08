! Part of the LLVM Project, under the Apache License v2.0
! See https://llvm.org/LICENSE.txt for license informatio
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test RANDOM_NUMBER intrinsic with quad-precision arguments

program test
  real(16) :: num(10), r16, num222(2, 2, 2), num66(6,6), num2(2:6,2:6), &
  num7(5,5,5,5,5,5,5)
  real(8) :: r8
  integer(4) :: seed(34), myseed(36), res(4), rres(4), s
  res = 0
  rres = 1
  data myseed /25423,1,2,477,888888,3,5,4,78,93,2,566,532,552,7788,9555, &
  4455,555,22233333,444444,55555555,88,88888888,55,5214,6987,546,63312, &
  3123,46563,84654,456641,46546321,1313115,45634165,13153/
  call random_seed(put = myseed)
  call random_number(num(1:5))
  call random_seed(get = myseed)
  call random_number(num(6:10))
  if( all (num66(1:6,1:3) .EQ. num66(1:6,4:6))) res(1) = 1
  call random_number(num222)

  call random_seed(size = s)
  call random_seed(get = seed)
  call random_number(num(1:5))
  call random_seed(put = seed)
  call random_number(num(6:10))
  if(all (num(1:5) .EQ. num(6:10)))  res(2) = 1
  
  call random_seed(get = seed)
  call random_number(num66(1:6,1:3))
  call random_seed(put = seed)
  call random_number(num66(1:6,4:6))
  if(all (num66(1:6,1:3) .EQ. num66(1:6,4:6))) res(3) =1
 
  call random_seed(get = seed)
  call random_number(r16)
  call random_seed(put = seed)
  call random_number(r8)
  if( (r16 - r8) .LT. 0.000000000000001) res(4) = 1

  call random_seed()
  call random_number(num2)
  call random_number(num7)

  call check(res, rres, 4)
end
