! @@name:	SIMD.8f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
module work
 
integer :: P(1000)
real    :: A(1000)

contains
function do_work(arr) result(pri)
  implicit none
  real, dimension(*) :: arr

  real :: pri
  integer :: i, j

  !$omp simd private(j) lastprivate(pri)
  do i = 1, 999
    j = P(i)
 
    pri = 0.5
    if (mod(j-1, 2) == 0) then
      pri = A(j+1) + arr(i)
    endif
    A(j) = pri * 1.5
    pri = pri + A(j)
  end do

end function do_work

end module work
 
program simd_8f
  use work
  implicit none
  real :: pri, arr(1000)
  integer :: i

  do i = 1, 1000
     P(i)   = i
     A(i)   = (i-1) * 1.5
     arr(i) = (i-1) * 1.8
  end do
  pri = do_work(arr)
  if (pri == 8237.25) then
    print 2, "passed", pri
  else
    print 2, "failed", pri
  endif
2 format(a, ": result pri = ", f7.2, " (8237.25)")

end program
