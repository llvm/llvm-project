!RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 %s -o - | FileCheck %s

module m
!$omp requires atomic_default_mem_order(acq_rel)

contains

!CHECK: %[[V:[0-9]+]]:2 = hlfir.declare {{.*}} {uniq_name = "_QMmFf00Ev"}
!CHECK: %[[X:[0-9]+]]:2 = hlfir.declare {{.*}} {uniq_name = "_QMmFf00Ex"}
!CHECK: omp.atomic.read %[[V]]#0 = %[[X]]#0 memory_order(acquire)
!CHECK: omp.atomic.write %[[X]]#0 = %{{[0-9]+}} memory_order(release)

subroutine f00(x, v)
  integer :: x, v
  !$omp atomic read
    v = x

  !$omp atomic write
    x = v
end

end module
