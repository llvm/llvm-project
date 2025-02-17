! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! OpenMP 5.2 5.1.1 Variables Referenced in a Construct
! DO CONCURRENT indices have predetermined private DSA.
!
! As DO CONCURRENT indices are defined in the construct itself, and OpenMP
! directives may not appear in it, they are already private.
! Check that index symbols are not modified.

!DEF: /private_iv (Subroutine)Subprogram
subroutine private_iv
   !DEF: /private_iv/i ObjectEntity INTEGER(4)
   integer i
   !$omp parallel default(private)
   !$omp single
   !DEF: /private_iv/OtherConstruct1/OtherConstruct1/Forall1/i ObjectEntity INTEGER(4)
   do concurrent(i=1:2)
   end do
   !$omp end single
   !$omp end parallel
end subroutine
