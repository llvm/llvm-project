! A host variable given VOLATILE inside an internal procedure has the attribute
! on the host-associated symbol, not on the ultimate one. The reduction must
! still be treated as volatile: named @add_reduction_volatile_i32, with no
! atomic region.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

! CHECK:       omp.declare_reduction @add_reduction_volatile_i32 : i32
! CHECK-NOT:     llvm.atomicrmw
! CHECK-LABEL: func.func {{.*}}inner
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_volatile_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32, volatile>)

subroutine host(n)
  integer :: n, i
  integer :: hv
  hv = 0
  call inner()
contains
  subroutine inner()
    volatile :: hv
    !$omp parallel do reduction(+:hv)
    do i = 1, n
      hv = hv + 1
    end do
  end subroutine
end subroutine
