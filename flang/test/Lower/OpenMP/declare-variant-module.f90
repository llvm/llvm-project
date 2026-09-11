! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! The base and its variant are sibling module procedures, so the variant is
! accessible at every reference to the base (here, in a sibling procedure of the
! same module).

module m
contains
  subroutine base
    !$omp declare variant (base:vsub) match (construct={parallel})
  end subroutine base

  subroutine vsub
  end subroutine vsub

  subroutine caller
    call base()
    !$omp parallel
    call base()
    !$omp end parallel
  end subroutine caller
end module m

! CHECK-LABEL: func.func @_QMmPcaller
! CHECK: fir.call @_QMmPbase(){{.*}}: () -> ()
! CHECK: omp.parallel
! CHECK: fir.call @_QMmPvsub(){{.*}}: () -> ()
