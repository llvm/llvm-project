!RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fdebug-dump-symbols %s 2>&1 | FileCheck %s --check-prefix=IMPLICIT
!RUN: %flang_fc1 -fopenmp -fopenmp-version=60 -fdebug-dump-symbols %s 2>&1 | FileCheck %s --check-prefix=NOIMPLICIT


!IMPLICIT: k2 (OmpLinear, OmpPreDetermined): {{.*}}
!NOIMPLICIT: k2 (OmpLastPrivate, OmpPreDetermined): {{.*}}
subroutine implicit_linear
  integer :: k1, k2

  !$omp simd
  do k2=1,10
  do k1=1,10
  end do
  end do
end subroutine
