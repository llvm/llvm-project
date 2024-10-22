! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! OpenMP 5.1.1
! DO Concurrent indices are private

!DEF: /private_iv (Subroutine)Subprogram
subroutine private_iv
   !DEF: /private_iv/i ObjectEntity INTEGER(4)
   integer i
   !$omp parallel default(private)
   !$omp single
   !DEF: /private_iv/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
   do concurrent(i=1:2)
   end do
   !$omp end single
   !$omp end parallel
end subroutine
