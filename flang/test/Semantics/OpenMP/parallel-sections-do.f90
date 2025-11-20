! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! Check that loop iteration variables are private and predetermined, even when
! nested inside parallel/sections constructs.

!DEF: /test1 (Subroutine) Subprogram
subroutine test1
  !DEF: /test1/i ObjectEntity INTEGER(4)
  integer i

  !$omp parallel default(none)
    !$omp sections
      !$omp section
        !DEF: /test1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
        do i = 1, 10
        end do
    !$omp end sections
  !$omp end parallel
end subroutine
