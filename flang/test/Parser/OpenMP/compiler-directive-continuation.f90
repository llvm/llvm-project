! RUN: %flang_fc1 -fopenmp -E %s 2>&1 | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s 


! Test in mixed way, i.e., combination of Fortran free source form 
! and free source form with conditional compilation sentinel.
! CHECK-LABEL: subroutine mixed_form1()
! CHECK-OMP: i = 1 +100+ 1000+ 10 + 1 +1000000000 + 1000000
! CHECK: i = 1 + 10 + 10000 + 1000000
subroutine mixed_form1()
   i = 1 &
  !$+100&
  !$&+ 1000&
   &+ 10 + 1&
  !$& +100000&
   &0000 + 1000000
end subroutine	


! Testing continuation lines in only Fortran Free form Source
! CHECK-LABEL: subroutine mixed_form2()
! CHECK-OMP: i = 1 +10 +100 + 1000 + 10000
! CHECK: i = 1 +10 +100 + 1000 + 10000
subroutine mixed_form2()
   i = 1 &
   +10 &
   &+100
   & + 1000 &
   + 10000
end subroutine


! Testing continuation line in only free source form conditional compilation sentinel.
! CHECK-LABEL: subroutine mixed_form3()
! CHECK-OMP: i=0
! CHECK-OMP: i = 1 +10 +100+1000
subroutine mixed_form3()
   !$ i=0
   !$ i = 1 &
   !$ & +10 &
   !$&+100&
   !$ +1000 
end subroutine

