! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! do concurrent variable declaration errors

subroutine s1
  !implicit none
  integer, parameter :: k  = 8
  !{error "PGF90-S-1219-Unimplemented feature: boz feature."}
  real(kind = k) :: tmpa = z'abcd abcd abcd abcd abcd'
end

subroutine s2
  !implicit none
  integer, parameter :: k  = 8
  !{error "PGF90-S-1219-Unimplemented feature: boz feature."}
  real(kind = k) :: tmpa = o'2234567123456712345672' 
end

program test
  call s1
  call s2
end
