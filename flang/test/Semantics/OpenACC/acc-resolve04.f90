! RUN: %flang_fc1 -fopenacc %s

! Check that symbol are correctly resolved in device, host and self clause.

program test_resolve04
  real :: a(10), b(10)
  common /foo/ b, c

!$acc data create(/foo/)
!$acc update device(/foo/)
!$acc update host(/foo/)
!$acc update self(/foo/)
!$acc end data

end

