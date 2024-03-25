! RUN: %flang_fc1 -fopenacc %s

! Check common block resolution.
! Check that symbol are correctly resolved in device, host and self clause.

subroutine sub(a)
 implicit none
 real :: a(10)
 real :: b(10), c(10), d
 common/foo/ b, d, c
 integer :: i, n

 !$acc declare present(/foo/)
 !$acc parallel loop gang vector
  do i = 1, n
	  b(i) = a(i) + c(i) * d
  end do
end subroutine

program test_resolve04
  real :: a(10), b(10)
  common /foo/ b, c

!$acc data create(/foo/)
!$acc update device(/foo/)
!$acc update host(/foo/)
!$acc update self(/foo/)
!$acc end data

!$acc data copy(/foo/)
!$acc end data

end

