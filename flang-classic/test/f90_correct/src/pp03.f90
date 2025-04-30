!*** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!*** See https://llvm.org/LICENSE.txt for license information.
!*** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test recursive calls with pointer arguments
!
module factorize
 type factors
  integer :: x
  type(factors),pointer:: left,right
 end type
contains
 recursive subroutine factor(p,n)
  type(factors),pointer:: p
  integer n
  integer i,j
  allocate(p)
  p%x = n
  nullify(p%left)
  nullify(p%right)
  do i = int(dsqrt(dble(n))),2,-1
   j = n/i
   if( j*i .eq. n )then
    call factor( p%left, j )
    call factor( p%right, i )
    return
   endif
  enddo
 end subroutine 
 recursive subroutine fill( p, x, i )
  type(factors) :: p
  integer x(:),i
  if( associated(p%left) .and. associated(p%right) )then
   call fill( p%left, x, i )
   call fill( p%right, x, i )
  else if( associated(p%left) .or. associated(p%right) )then
   print *,'fill: error at ',p%x
   print *,associated(p%left), ' = associated(p%left)'
   print *,associated(p%right), ' = associated(p%right)'
  else
   i = i + 1
   x(i) = p%x
  endif
 end subroutine
 recursive function count( p ) result(res)
  type(factors) :: p
  integer :: res
  if( associated(p%left) .and. associated(p%right) )then
   res = count( p%left )
   res = res + count( p%right )
  else if( associated(p%left) .or. associated(p%right) )then
   print *,'count: error at ',p%x
   print *,associated(p%left), ' = associated(p%left)'
   print *,associated(p%right), ' = associated(p%right)'
   res = 0
  else
   res = 1
  endif
 end function
end module
 use factorize
 type(factors),pointer:: f
 integer,allocatable::x(:)
 integer values(5), nn
 data values / 77, 100, 31, 128, 362880 /
 integer result(10), expect(10)
 data expect / 77, 2, 100, 4, 31, 1, 128, 7, 362880, 13 /
 do i = 1,5
  call factor( f, values(i) )
  n = count( f )
  allocate (x(n))
  nn = 0
  call fill( f, x, nn )
! print *,x
  nn = product(x)
  if( nn .ne. values(i) )then
   print *,nn,' is result, should be ',values(i)
  endif
! print *,nn,n
  result(2*i-1) = nn
  result(2*i) = n
  deallocate (x)     !4/16/2000 - can't allocate if already allocated.
 enddo
 call check( result, expect, 10 )
end
