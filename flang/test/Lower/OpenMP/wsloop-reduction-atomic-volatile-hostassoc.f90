! A host variable given VOLATILE inside an internal procedure has the attribute
! on the host-associated symbol, not on the ultimate one. The reduction must
! still be treated as volatile: named @add_reduction_volatile_i32, with no
! atomic region.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %s | FileCheck %s --implicit-check-not=llvm.atomicrmw

! CHECK:       omp.declare_reduction @add_reduction_volatile_i32 : i32
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

! Inside an enclosing parallel, the construct symbols link straight to the
! ultimate symbol and skip the VOLATILE host-association link.
! CHECK-LABEL: func.func {{.*}}inner2
! CHECK:         omp.wsloop {{.*}}reduction(@add_reduction_volatile_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32, volatile>)

subroutine host2(n)
  integer :: n, i, hv
  hv = 0
  call inner2()
contains
  subroutine inner2()
    volatile :: hv
    !$omp parallel
    !$omp parallel do reduction(+:hv)
    do i = 1, n
      hv = hv + 1
    end do
    !$omp end parallel
  end subroutine
end subroutine
