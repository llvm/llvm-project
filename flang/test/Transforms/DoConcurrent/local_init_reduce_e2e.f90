! Tests end-to-end conversion of LOCAL_INIT + REDUCE together from Fortran
! source through the full pipeline.
!
! This is the most important combination test: the inlined copy for LOCAL_INIT
! must appear before any reduction-related code in the omp.loop_nest body.
! We do not place the copy_region on the omp.private op because the LLVM IR
! translation (copyFirstPrivateVars in OpenMPToLLVMIRTranslation.cpp) fires
! omp.private's copy_region once per thread before the worksharing loop, not
! once per iteration.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

subroutine test_local_init_reduce(n)
  implicit none
  integer, intent(in) :: n
  real :: tmp, total
  integer :: i

  tmp = 1.0
  total = 0.0

  do concurrent (i = 1:n) local_init(tmp) reduce(+:total)
    tmp = tmp + real(i)
    total = total + tmp
  end do
end subroutine

! Verify: omp.private for LOCAL_INIT (no copy_region)
! CHECK:      omp.private {type = private} @[[PRIVATIZER:.*tmp.*\.omp]] : f32

! Verify: omp.wsloop has both private and reduction clauses
! CHECK:      omp.wsloop private(@[[PRIVATIZER]] %[[ORIG:.*]] -> %[[PRIV:.*]] : !fir.ref<f32>) reduction(@{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<f32>)

! Verify: per-iteration copy inlined at top of loop_nest body, before user code
! CHECK:        omp.loop_nest
! CHECK-NEXT:     %[[VAL:.*]] = fir.load %[[ORIG]] : !fir.ref<f32>
! CHECK-NEXT:     hlfir.assign %[[VAL]] to %[[PRIV]] : f32, !fir.ref<f32>
