! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! do concurrent locality list

integer, parameter :: N = 6
integer, target    :: a(N,N,N) ! operand via p
integer            :: r(N,N)   ! result, unspecified locality
integer            :: s(N,N)   ! shared locality
integer            :: t(N,N)   ! local locality
integer, pointer   :: p(:,:,:) ! local_init locality

p => a

do concurrent (integer(kind=1)::i=N:1:-1)
  do j = 1,N
    a(i,j,:) = 2*(i+j)
    s(i,j)   = -i-j
  enddo
enddo

do concurrent (integer(2)::i=1:N,j=1:N,i.ne.j) local(t) local_init(p) shared(s)
  do k=1,N
    do concurrent (m=1:N)
      t(k,m) = p(k,m,k)
    enddo
  enddo
  r(i,j) = t(i,j) + s(i,j)
enddo

! print*, r !    0    3    4    5    6    7
            !    3    0    5    6    7    8
            !    4    5    0    7    8    9
            !    5    6    7    0    9   10
            !    6    7    8    9    0   11
            !    7    8    9   10   11    0  -->  sums to 210

if (sum(r).ne.210) print*, 'FAIL'
if (sum(r).eq.210) print*, 'PASS'

end
