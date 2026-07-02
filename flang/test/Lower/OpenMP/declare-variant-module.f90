! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

module m
contains
  subroutine base
    !$omp declare variant (base:vsub) match (construct={parallel})
  contains
    subroutine vsub
    end subroutine
  end subroutine base

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
! CHECK: fir.call @_QMmFbasePvsub(){{.*}}: () -> ()
