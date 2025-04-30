! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! do concurrent locality lists with intrinsic and keyword identifiers

module d
  integer :: a(5) = 1
end module d

subroutine ex ! explicit declarations
  use d
  integer if, index
  do concurrent (i=1:5) local(if,index) shared(a) default(none)
    if = a(i)
    index = if
    a(i) = index + 1
  end do
end

subroutine im ! implicit declarations
  use d
  do concurrent (i=1:5) local(if,index) shared(a) default(none)
    if = a(i)
    index = if
    a(i) = index + 1
  end do
end

  use d
  call ex
  call im
  if (any(a .ne. 3)) print*, 'FAIL'
  if (all(a .eq. 3)) print*, 'PASS'
end
